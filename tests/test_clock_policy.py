"""Elapsed-time code ignores system clock corrections; Unix time is only for data and logs."""
import _path  # noqa: F401
import ast
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
offenders = []
for path in [ROOT / "main.py", ROOT / "check_env.py", ROOT / "config.py",
             *sorted((ROOT / "src").glob("*.py")), *sorted((ROOT / "tools").glob("*.py"))]:
    if path.name == "clock.py":
        continue
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "time" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"):
            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
assert not offenders, ("Use clock.now() for elapsed time and clock.unix_time() for data: "
                       + ", ".join(offenders))

from src import clock
import src.camera as camera
from src.audio_gate import AudioGate
from src.cloud_sign import SegmentRecorder
from src.latency import LatencyTracker
from src.predictor import PhraseBuilder

real_time = time.time
offset = [0.0]
with patch("time.time", lambda: real_time() + offset[0]):
    # TTS echo gate: the tail must end on schedule, not an hour later.
    gate = AudioGate(tail_sec=0.05)
    with gate.output():
        pass
    offset[0] = -3600.0          # NTP pulls the system clock back an hour
    time.sleep(0.08)
    assert gate.is_open(), "A clock correction kept the microphone muted"

    offset[0] = 0.0
    stats = LatencyTracker()
    t0 = clock.now()
    offset[0] = -3600.0
    time.sleep(0.01)
    stats.since("x", t0)
    assert 5 <= stats.summary()["x"][1] < 1000, stats.summary()

    offset[0] = 0.0
    phrase = PhraseBuilder(window_sec=0.2, flush_sec=0.1)
    assert phrase.feed("감사합니다") is None
    offset[0] = -3600.0
    time.sleep(0.15)
    assert phrase.poll() == "감사합니다", "A clock correction stalled the sentence latch"

    # Capture timestamps come from clock.now(), so the segment stays in order.
    offset[0] = 0.0
    recorder = SegmentRecorder()
    recorder.start()
    points, mask = np.full((49, 2), .5, np.float32), np.ones(49, bool)
    for i in range(6):
        if i == 3:
            offset[0] = -3600.0
        recorder.add(points, mask, clock.now(), (640, 480))
        time.sleep(0.005)
    sequence, origin_t = recorder.finish()
    assert len(sequence.timestamps) == 6 and 0 <= clock.now() - origin_t < 1

    # Camera reconnect: after a failed reopen the next attempt follows reopen_sec.
    offset[0] = 0.0
    calls = []

    class FakeCap:
        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    def fake_open(verbose=True, camera_index=None):
        calls.append(1)
        return FakeCap() if len(calls) >= 3 else None

    with patch.object(camera, "open_camera", fake_open):
        reader = camera.CameraReader(reopen_sec=0.1)
        assert not reader.read()[0] and len(calls) == 2
        offset[0] = -3600.0
        end, ok = clock.now() + 1.0, False
        while not ok and clock.now() < end:
            ok = reader.read()[0]
        assert ok, "A clock correction postponed the camera reconnect"
print("[OK] no wall-clock elapsed time; gate, latency, latch, recorder, camera survive a clock jump")
