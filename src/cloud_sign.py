"""Nonblocking cloud worker and lossless segment capture, separate from preview queue."""
from collections import deque
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import numpy as np
from src.pose_sequence import PoseSequence, MAX_SECONDS, MAX_FRAMES, VERSION


class SegmentRecorder:
    def __init__(self, max_seconds=MAX_SECONDS):
        self.max_seconds = max_seconds
        self.lock = threading.Lock()
        self.active = False
        self.frames = []
        self.error = None
        self.started = 0

    def start(self):
        with self.lock:
            self.frames, self.error = [], None
            self.started = time.monotonic()
            self.active = True

    def cancel(self):
        with self.lock:
            self.active, self.frames, self.error = False, [], None

    def add(self, points, mask, timestamp, image_size, world=None, world_mask=None):
        with self.lock:
            if not self.active:
                return
            if time.monotonic() - self.started > self.max_seconds or len(self.frames) >= MAX_FRAMES:
                self.active, self.frames, self.error = False, [], "too_long"
                return
            if self.frames and (timestamp <= self.frames[-1][0]
                                or tuple(image_size) != self.frames[-1][3]
                                or (world is None) != (self.frames[-1][4] is None)):
                self.active, self.frames, self.error = False, [], "camera_discontinuity"
                return
            self.frames.append((timestamp, points.copy(), mask.copy(), tuple(image_size),
                                None if world is None else (world.copy(), world_mask.copy())))

    def finish(self):
        with self.lock:
            self.active = False
            frames, self.frames = self.frames, []
            error, self.error = self.error, None
            elapsed = time.monotonic() - self.started
        if error or elapsed > self.max_seconds:
            raise ValueError(error or "too_long")
        if len(frames) < 2:
            raise ValueError("too_short")
        timestamps = np.array([f[0] for f in frames])
        ended = time.time()
        world = ((None, None) if frames[0][4] is None
                 else ([f[4][0] for f in frames], [f[4][1] for f in frames]))
        return PoseSequence(timestamps - timestamps[0], [f[1] for f in frames],
                            [f[2] for f in frames], frames[0][3],
                            world=world[0], world_mask=world[1]).validate(), ended


class CloudClient:
    def __init__(self, url=None, token=None, timeout=30):
        self.url = (url or os.environ.get("SIGN_SERVER_URL", "http://127.0.0.1:8765")).rstrip("/")
        self.token = token if token is not None else os.environ.get("SIGN_SERVER_TOKEN", "")
        self.timeout = timeout
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
            raise ValueError("Server URL must be HTTP(S) without embedded credentials")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Remote servers require HTTPS")

    def translate(self, payload):
        deadline = time.monotonic() + self.timeout
        body = json.dumps(payload, allow_nan=False).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Translation deadline exceeded")
            request = urllib.request.Request(self.url + "/v1/sign/translate", body, headers)
            try:
                with urllib.request.urlopen(request, timeout=remaining) as response:
                    raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError("Response too large")
                result = json.loads(raw)
                validate_response(result, payload)
                return result
            except urllib.error.HTTPError:
                # Authentication, invalid data and queue refusal are not connection retries.
                raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt:
                    raise
        raise TimeoutError("Translation failed")


def validate_response(result, payload):
    if not isinstance(result, dict) or any(result.get(k) != payload[k]
                                          for k in ("session_id", "segment_id", "version")):
        raise ValueError("Mismatched translation result")
    if result.get("status") not in ("ok", "no_sign", "uncertain", "unavailable"):
        raise ValueError("Unknown translation status")
    text = result.get("text")
    if not isinstance(text, str) or len(text) > 200:
        raise ValueError("Invalid translation text")
    if type(result.get("auto_speech")) is not bool or type(result.get("mock")) is not bool:
        raise ValueError("Invalid speech policy")
    if (result["status"] != "ok" and (text or result["auto_speech"])
            or result["mock"] and result["auto_speech"]
            or result["status"] == "ok" and not text.strip()):
        raise ValueError("Unsafe translation response")
    timings = result.get("timings_ms", {})
    if not isinstance(timings, dict) or any(
            not isinstance(k, str) or type(v) not in (int, float)
            or not np.isfinite(v) or v < 0 for k, v in timings.items()):
        raise ValueError("Invalid timing measurements")
    return result


class CloudWorker(threading.Thread):
    def __init__(self, client=None, stats=None):
        super().__init__(name="cloud-sign", daemon=True)
        self.client = client or CloudClient()
        self.stats = stats
        self.session_id = uuid.uuid4().hex
        self.jobs, self.results = queue.Queue(maxsize=1), queue.Queue()
        self.lock = threading.Lock()
        self.pending = None
        self.deadline = 0
        self.seen = deque(maxlen=256)
        self._quit = threading.Event()

    @property
    def busy(self):
        with self.lock:
            return self.pending is not None

    def submit(self, sequence, origin_t):
        with self.lock:
            if self.pending is not None:
                return False
            segment_id = uuid.uuid4().hex
            pack_start = time.perf_counter()
            payload = {**sequence.to_dict(), "session_id": self.session_id, "segment_id": segment_id}
            if self.stats:
                self.stats.add("pose_pack", (time.perf_counter() - pack_start) * 1000)
            self.pending = segment_id
            self.deadline = time.monotonic() + self.client.timeout
            self.jobs.put_nowait((payload, origin_t, self.deadline))
            return True

    def cancel(self):
        # Leave pending set until the request completes: never stack cancelled requests.
        with self.lock:
            if self.pending:
                self.seen.append(self.pending)

    def stop(self):
        self.cancel()
        self._quit.set()

    def run(self):
        while not self._quit.is_set():
            try:
                payload, origin_t, deadline = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            start = time.perf_counter()
            try:
                result = self.client.translate(payload)
            except Exception:
                result = {"status": "unavailable", "text": "", "auto_speech": False,
                          "segment_id": payload["segment_id"], "mock": False}
            if self.stats:
                self.stats.add("cloud_roundtrip", (time.perf_counter() - start) * 1000)
                for name, value in result.get("timings_ms", {}).items():
                    if isinstance(value, (int, float)) and np.isfinite(value) and value >= 0:
                        self.stats.add("cloud_server_" + name, value)
            self.results.put((payload["segment_id"], result, origin_t, deadline))

    def poll(self):
        try:
            segment_id, result, origin_t, deadline = self.results.get_nowait()
        except queue.Empty:
            return None
        with self.lock:
            stale = (self._quit.is_set() or segment_id != self.pending
                     or segment_id in self.seen or time.monotonic() > deadline)
            if segment_id == self.pending:
                self.pending = None
            self.seen.append(segment_id)
        if stale:
            return {"status": "expired", "text": "", "auto_speech": False}, origin_t
        return result, origin_t
