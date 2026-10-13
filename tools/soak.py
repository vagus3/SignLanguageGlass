#!/usr/bin/env python
"""Unattended soak of the cloud pipeline: real HTTP, real threads, injected faults.

python tools/soak.py --minutes 15 --out reports/soak.json
python tools/soak.py --minutes 15 --url https://server.example   # a real server; cancels only

Synthetic 30 fps poses (sign above the webcam rest line, then lower the hands)
drive the same objects main.py uses: SegmentRecorder, ActivitySegmenter,
CloudWorker over HTTP, CloudController and HudContent; a recording TTS stands in
for the speaker. With the built-in server it injects every failure the demo must
survive: slow answers past the client deadline, generation errors (fallback
previews), a server outage and cancels while a request is in flight. It fails if
anything is spoken that must not be (cancelled, expired, unapproved, twice), if a
result is lost, or if memory or threads keep growing. Camera, microphone, STT,
real TTS audio and heat need the on-device soak in TEST_CHECKLIST.md.
"""
import argparse
import json
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import clock
from src.cloud_control import CloudController
from src.cloud_sign import CloudClient, CloudWorker, SegmentRecorder
from src.hud import HudContent, HudRenderer
from src.latency import LatencyTracker
from src.segmenter import ActivitySegmenter
from src.translation import Glossary

FPS = 30
SIZE = (640, 640)
_HAND = np.zeros((21, 2))
_HAND[:, 1] = -np.linspace(0, .12, 21)
_HAND[9] = [0, -.09]


def pose(t, cycle):
    """Sign for cycle[0] s above the rest line, then hands lowered for cycle[1] s."""
    signing = t % sum(cycle) < cycle[0]
    points, mask = np.zeros((49, 2), np.float32), np.zeros(49, bool)
    x = .4 + .1 * np.sin(2 * np.pi * 1.5 * t)
    points[:21], mask[:21] = _HAND + [x, .6 if signing else .95], True
    points[43], points[44], mask[43:45] = [.35, .4], [.65, .4], True
    return np.clip(points, 0, 1), mask


class SoakRecognizer:
    labels = ["<blank>", "안녕하세요"]
    metadata = {"model_id": "soak"}

    def __init__(self, glossary):
        self.calibration = {"enabled": True, "threshold": 0.5, "negative_seconds": 600,
                            "reviewer": "soak", "mapping_hash": glossary.policy_hash(),
                            "language_model": FaultyTranslator.model}

    def predict(self, sequence):
        return ["안녕하세요"], 0.99


class FaultyTranslator:
    model = "soak-translator"

    def __init__(self, slow_delay):
        self.mode, self.slow_delay = "normal", slow_delay

    def translate(self, glosses, meanings):
        mode = self.mode
        if mode == "slow":
            time.sleep(self.slow_delay)
        elif mode == "error":
            raise TimeoutError("injected")
        else:
            time.sleep(0.03)
        return "안녕하세요."


def start_server(slow_delay):
    """The real FastAPI app over HTTP on a free local port, with switchable faults."""
    import uvicorn
    from fastapi import HTTPException
    from src.sign_server import TranslationService, create_app

    class Service(TranslationService):
        down = False

        def process(self, request, received_at=None):
            if self.down:
                raise HTTPException(503, "injected outage")
            return super().process(request, received_at)

    glossary = Glossary({"안녕하세요": "안녕하세요"})
    translator = FaultyTranslator(slow_delay)
    service = Service(SoakRecognizer(glossary), translator, glossary)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(service, token="soak-token"),
                                           host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True, name="soak-server").start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("soak server did not start")
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", service, translator, server


class AuditQueue(queue.Queue):
    """CloudWorker.results that remembers what poll() took, so stale items are traceable."""

    def __init__(self):
        super().__init__()
        self.taken = []

    def get_nowait(self):
        item = super().get_nowait()
        segment_id, result, origin_t, deadline = item
        self.taken.append({"segment": segment_id, "status": result.get("status"),
                           "auto_speech": result.get("auto_speech"), "mock": result.get("mock"),
                           "late": clock.now() > deadline, "origin_t": origin_t})
        return item


class AuditedWorker(CloudWorker):
    def __init__(self, client, stats):
        super().__init__(client, stats=stats)
        self.results = AuditQueue()
        self.submitted, self.cancelled = {}, set()

    def submit(self, sequence, origin_t):
        accepted = super().submit(sequence, origin_t)
        if accepted:
            self.submitted[origin_t] = self.pending
        return accepted

    def cancel(self):
        if self.pending:
            self.cancelled.add(self.pending)
        super().cancel()


class RecordingTts:
    def __init__(self):
        self.spoken = []

    def say(self, text, origin_t=0.0):
        self.spoken.append((text, origin_t))


def rss_mb():
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(__import__("os").getpid())],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def audit(worker, tts, service_results):
    """Every spoken sentence must be a fresh, approved, uncancelled result, spoken once."""
    taken = {t["origin_t"]: t for t in worker.results.taken}
    problems, seen = [], set()
    for text, origin_t in tts.spoken:
        segment = worker.submitted.get(origin_t)
        record = taken.get(origin_t)
        if segment is None or record is None:
            problems.append(f"speech without a submitted segment: {text!r}")
            continue
        if segment in seen:
            problems.append(f"{segment} spoken twice")
        seen.add(segment)
        if segment in worker.cancelled:
            problems.append(f"{segment} spoken after cancel")
        if record["late"]:
            problems.append(f"{segment} spoken after its deadline")
        if record["status"] != "ok" or not record["auto_speech"] or record["mock"]:
            problems.append(f"{segment} spoken without approval ({record['status']})")
    return problems


def run(seconds, *, url=None, token=None, cycle=(3.0, 2.0), client_timeout=2.0,
        slow_delay=3.0, outage=(0.4, 0.5), slow_every=7, error_every=13, cancel_every=11,
        out=lambda *a: None):
    local = url is None
    service = translator = server = None
    if local:
        url, service, translator, server = start_server(slow_delay)
        token = "soak-token"
    stats, tts = LatencyTracker(), RecordingTts()
    worker = AuditedWorker(CloudClient(url, token=token, timeout=client_timeout), stats)
    recorder, segmenter = SegmentRecorder(), ActivitySegmenter(viewpoint="webcam")
    content = HudContent(HudRenderer())
    controller = CloudController(worker, recorder, content, tts=tts, stats=stats, out=out)
    worker.start()
    threads, memory = [threading.active_count()], [rss_mb()]
    start = clock.now()
    frame, submitted, next_sample = 0, 0, 10.0
    try:
        while True:
            now = clock.now()
            t = now - start
            if t >= seconds:
                break
            if local:
                service.down = outage[0] * seconds <= t < outage[1] * seconds
            points, mask = pose(t, cycle)
            recorder.add(points, mask, now, SIZE)
            event = segmenter.update(points, mask, now, recorder.active, SIZE)
            if event:
                before = len(worker.submitted)
                if local and event == "stop":
                    k = before + 1
                    translator.mode = ("slow" if k % slow_every == 0 else
                                       "error" if k % error_every == 0 else "normal")
                controller.command("auto_" + event)
                if len(worker.submitted) > before:
                    submitted += 1
                    if submitted % cancel_every == 0:
                        controller.command("cancel")
            controller.tick()
            content.frame()
            if t >= next_sample:
                threads.append(threading.active_count())
                memory.append(rss_mb())
                next_sample += 10.0
            frame += 1
            time.sleep(max(0.0, start + frame / FPS - clock.now()))
        # Let the last request finish, then take its result.
        end = clock.now() + client_timeout + 1.0
        while worker.busy and clock.now() < end:
            controller.tick()
            time.sleep(0.02)
        controller.tick()
    finally:
        worker.stop()
        worker.join(2)
        if server is not None:
            server.should_exit = True
    threads.append(threading.active_count())
    memory.append(rss_mb())
    statuses = {}
    for t in worker.results.taken:
        key = "expired" if t["late"] or t["segment"] in worker.cancelled else t["status"]
        statuses[key] = statuses.get(key, 0) + 1
    problems = audit(worker, tts, statuses)
    lost = len(worker.submitted) - len(worker.results.taken)
    if lost:
        problems.append(f"{lost} submitted segments never returned a result")
    known = [m for m in memory if m is not None]
    growth = (known[-1] - known[min(1, len(known) - 1)]) if len(known) > 1 else None
    if growth is not None and growth > 50:
        problems.append(f"memory grew {growth:.0f} MB")
    if threads[-1] > threads[0] + 2:
        problems.append(f"threads grew {threads[0]} -> {threads[-1]}")
    summary = stats.summary()
    return {"seconds": seconds, "frames": frame, "fps": frame / seconds,
            "segments": len(worker.submitted), "results": statuses,
            "spoken": len(tts.spoken), "cancelled": len(worker.cancelled),
            "injected": None if not local else {"slow_every": slow_every,
                                                "error_every": error_every,
                                                "cancel_every": cancel_every,
                                                "outage": outage},
            "rss_mb": {"samples": memory, "growth_after_warmup": growth},
            "threads": threads,
            "latency_ms": {k: {"n": v[0], "p50": v[1], "p95": v[2]} for k, v in summary.items()},
            "problems": problems, "passed": not problems}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--url", help="real server (fault injection off; cancels still injected)")
    p.add_argument("--token", help="real server token (default SIGN_SERVER_TOKEN)")
    p.add_argument("--out")
    args = p.parse_args()
    if args.out and Path(args.out).exists():
        p.error("Output exists; choose a new path")
    report = run(args.minutes * 60, url=args.url, token=args.token)
    print(json.dumps({k: v for k, v in report.items() if k != "latency_ms"},
                     ensure_ascii=False, indent=2))
    for name, v in report["latency_ms"].items():
        print(f"  {name:<26} n={v['n']:<5} p50={v['p50']:.1f}ms p95={v['p95']:.1f}ms")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
