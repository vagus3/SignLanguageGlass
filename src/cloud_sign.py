"""Nonblocking cloud worker and lossless segment capture, separate from preview queue."""
from collections import deque
import gzip
import json
import os
import queue
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import numpy as np
from src import clock
from src.ctc_runtime import GLOSS_VERSION
from src.pose_sequence import PoseSequence, MAX_SECONDS, MAX_FRAMES, VERSION


class SegmentRecorder:
    def __init__(self, max_seconds=MAX_SECONDS):
        self.max_seconds = max_seconds
        self.lock = threading.Lock()
        self.active = False
        self.frames = []
        self.error = None
        self.started = 0
        # Recent frames while idle, so automatic starts can include the onset.
        self.recent = deque(maxlen=64)

    def start(self, pre_roll=0.0):
        with self.lock:
            latest = self.recent[-1][0] if self.recent else None
            self.frames = ([f for f in self.recent if f[0] >= latest - pre_roll]
                           if pre_roll > 0 and latest is not None else [])
            self.error = None
            span = self.frames[-1][0] - self.frames[0][0] if self.frames else 0.0
            self.started = clock.now() - span
            self.active = True
            self.recent.clear()

    def cancel(self):
        with self.lock:
            self.active, self.frames, self.error = False, [], None

    def add(self, points, mask, timestamp, image_size, world=None, world_mask=None):
        with self.lock:
            if not self.active:
                if self.recent and (timestamp <= self.recent[-1][0]
                                    or tuple(image_size) != self.recent[-1][3]
                                    or (world is None) != (self.recent[-1][4] is None)):
                    self.recent.clear()
                self.recent.append((timestamp, points.copy(), mask.copy(), tuple(image_size),
                                    None if world is None else (world.copy(), world_mask.copy())))
                return
            if clock.now() - self.started > self.max_seconds or len(self.frames) >= MAX_FRAMES:
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
            elapsed = clock.now() - self.started
        if error or elapsed > self.max_seconds:
            raise ValueError(error or "too_long")
        if len(frames) < 2:
            raise ValueError("too_short")
        timestamps = np.array([f[0] for f in frames])
        ended = clock.now()  # origin_t for latency: elapsed clock
        world = ((None, None) if frames[0][4] is None
                 else ([f[4][0] for f in frames], [f[4][1] for f in frames]))
        return PoseSequence(timestamps - timestamps[0], [f[1] for f in frames],
                            [f[2] for f in frames], frames[0][3],
                            world=world[0], world_mask=world[1]).validate(), ended


def parse_rounding(text):
    """"" -> no rounding; "4" -> 2D only; "4,3" -> 2D and 3D decimal places."""
    if not text.strip():
        return None, None
    parts = [int(v) for v in text.split(",")]
    if len(parts) > 2 or any(not 1 <= v <= 9 for v in parts):
        raise ValueError("SIGN_SERVER_ROUND must be '<2D places>[,<3D places>]', 1..9")
    return parts[0], parts[1] if len(parts) == 2 else None


class CloudClient:
    """HTTPS client for /v1/sign/translate (path A: the server recognizes the pose).

    Bodies are compact JSON with the shortest exact float32 decimals, gzip level 1
    by default: a 139-frame 2D segment is about 57 KB instead of 262 KB, at a few ms
    of CPU. SIGN_SERVER_GZIP=0 sends plain JSON to a server that predates gzip.
    SIGN_SERVER_ROUND="4" (2D) or "4,3" (2D, 3D) also rounds points; that is lossy
    and stays off until tools/evaluate_rounding.py shows recognition is unchanged.
    """

    path = "/v1/sign/translate"
    # Below this a gzip header costs more than it saves (gloss requests are ~200 B).
    GZIP_MIN_BYTES = 1024

    def __init__(self, url=None, token=None, timeout=30, compress=None, rounding=None):
        self.url = (url or os.environ.get("SIGN_SERVER_URL", "http://127.0.0.1:8765")).rstrip("/")
        self.token = token if token is not None else os.environ.get("SIGN_SERVER_TOKEN", "")
        self.timeout = timeout
        self.compress = (os.environ.get("SIGN_SERVER_GZIP", "1") != "0"
                         if compress is None else bool(compress))
        self.decimals, self.world_decimals = parse_rounding(
            os.environ.get("SIGN_SERVER_ROUND", "") if rounding is None else rounding)
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
            raise ValueError("Server URL must be HTTP(S) without embedded credentials")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Remote servers require HTTPS")

    def payload(self, sequence, session_id, segment_id, stats=None):
        """Request body for one finished segment; runs on the worker thread."""
        start = clock.now()
        payload = {**sequence.to_dict(self.decimals, self.world_decimals),
                   "session_id": session_id, "segment_id": segment_id}
        if stats:
            stats.add("pose_pack", (clock.now() - start) * 1000)
        return payload

    def encode(self, payload):
        """(body bytes, headers). Encoded once, so a retry resends identical bytes."""
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        if self.compress and len(body) >= self.GZIP_MIN_BYTES:
            body = gzip.compress(body, compresslevel=1, mtime=0)
            headers["Content-Encoding"] = "gzip"
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        return body, headers

    def translate(self, payload):
        deadline = clock.now() + self.timeout
        body, headers = self.encode(payload)
        for attempt in range(2):
            remaining = deadline - clock.now()
            if remaining <= 0:
                raise TimeoutError("Translation deadline exceeded")
            request = urllib.request.Request(self.url + self.path, body, headers)
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


class GlossClient(CloudClient):
    """Path B (REVIEW_AND_PLAN 3.3): recognize on the device, send only the glosses.

    recognizer is a src.ctc_runtime.OnnxRecognizer; the server must be started with
    the same model's sidecar (--device-model), whose model_id this client sends.
    """

    path = "/v1/sign/compose"

    def __init__(self, recognizer, url=None, token=None, timeout=30, compress=None):
        super().__init__(url, token, timeout, compress)
        self.recognizer = recognizer
        self.model_id = (recognizer.metadata or {}).get("model_id")
        if not self.model_id:
            raise ValueError("Device recognizer has no model_id; export it with "
                             "tools/export_ctc_onnx.py")

    def payload(self, sequence, session_id, segment_id, stats=None):
        start = clock.now()
        glosses, score = self.recognizer.predict(sequence)
        if stats:
            stats.add("device_recognize", (clock.now() - start) * 1000)
        return {"version": GLOSS_VERSION, "session_id": session_id, "segment_id": segment_id,
                "glosses": list(glosses), "score": float(score), "recognizer": self.model_id}


STATUSES = ("ok", "fallback", "no_sign", "uncertain", "unavailable")
FALLBACK_SOURCES = ("reviewed_rules", "words")
# Long enough for a fallback preview of several reviewed sentences; HUD pages it.
MAX_TEXT = 1000


def validate_response(result, payload):
    """Reject results that do not belong to this request or could speak unsafely.

    "fallback" carries preview text (reviewed sentences or the word list) when
    sentence generation failed; it must never be marked for automatic speech.
    """
    if not isinstance(result, dict) or any(result.get(k) != payload[k]
                                          for k in ("session_id", "segment_id", "version")):
        raise ValueError("Mismatched translation result")
    status = result.get("status")
    if status not in STATUSES:
        raise ValueError("Unknown translation status")
    text = result.get("text")
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise ValueError("Invalid translation text")
    if type(result.get("auto_speech")) is not bool or type(result.get("mock")) is not bool:
        raise ValueError("Invalid speech policy")
    has_text = status in ("ok", "fallback")
    if (not has_text and (text or result["auto_speech"])
            or result["mock"] and result["auto_speech"]
            or has_text and not text.strip()
            or status == "fallback" and (result["auto_speech"]
                                         or result.get("source") not in FALLBACK_SOURCES)):
        raise ValueError("Unsafe translation response")
    timings = result.get("timings_ms", {})
    if not isinstance(timings, dict) or any(
            not isinstance(k, str) or type(v) not in (int, float)
            or not np.isfinite(v) or v < 0 for k, v in timings.items()):
        raise ValueError("Invalid timing measurements")
    return result


def describe(error):
    """Short reason for the operator's terminal; never shown on the HUD or spoken."""
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}"
    if isinstance(error, urllib.error.URLError):
        return f"연결 실패 ({error.reason})"
    if isinstance(error, TimeoutError):
        return "시간 초과"
    return type(error).__name__


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
            self.pending = segment_id
            self.deadline = clock.now() + self.client.timeout
            # The body is built on the worker thread: packing (path A) or device
            # recognition (path B) must not stall the camera/HUD loop.
            self.jobs.put_nowait((sequence, segment_id, origin_t, self.deadline))
            return True

    def _payload(self, sequence, segment_id):
        build = getattr(self.client, "payload", None)
        if build is None:   # clients that only implement translate(payload)
            return {**sequence.to_dict(), "session_id": self.session_id,
                    "segment_id": segment_id}
        return build(sequence, self.session_id, segment_id, self.stats)

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
                sequence, segment_id, origin_t, deadline = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            start = clock.now()
            try:
                payload = self._payload(sequence, segment_id)
                start = clock.now()
                result = self.client.translate(payload)
            except Exception as error:
                result = {"status": "unavailable", "text": "", "auto_speech": False,
                          "segment_id": segment_id, "mock": False, "error": describe(error)}
            if self.stats:
                self.stats.add("cloud_roundtrip", (clock.now() - start) * 1000)
                for name, value in result.get("timings_ms", {}).items():
                    if isinstance(value, (int, float)) and np.isfinite(value) and value >= 0:
                        self.stats.add("cloud_server_" + name, value)
            self.results.put((segment_id, result, origin_t, deadline))

    def poll(self):
        try:
            segment_id, result, origin_t, deadline = self.results.get_nowait()
        except queue.Empty:
            return None
        with self.lock:
            stale = (self._quit.is_set() or segment_id != self.pending
                     or segment_id in self.seen or clock.now() > deadline)
            if segment_id == self.pending:
                self.pending = None
            self.seen.append(segment_id)
        if stale:
            return {"status": "expired", "text": "", "auto_speech": False}, origin_t
        return result, origin_t
