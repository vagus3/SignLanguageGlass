"""Laptop profiler: camera format request order, delivered fps, stage timings and advice."""
import _path  # noqa: F401
import sys
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import config as C
from src.camera import fourcc_name
from tools.profile_pipeline import advice, grab_camera, measure, read_video

assert fourcc_name(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG" and fourcc_name(0) == "?"

calls = []


class FakeCapture:
    """Records property order; delivers frames at about 50 per second."""

    def __init__(self, index, backend=None):
        self.props = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        calls.append(prop)
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        import time
        time.sleep(0.02)
        return True, np.zeros((72, 128, 3), np.uint8)

    def release(self):
        pass


saved_mac, C.IS_MAC = C.IS_MAC, False            # the FOURCC request is skipped on macOS
try:
    with mock.patch("cv2.VideoCapture", FakeCapture):
        frames, fps, name = grab_camera(0, 0.3, "MJPG")
        assert name == "MJPG" and 30 < fps < 60 and frames[0].shape == (72, 128, 3), (name, fps)
        assert calls.index(cv2.CAP_PROP_FOURCC) < calls.index(cv2.CAP_PROP_FRAME_WIDTH), \
            "DirectShow only accepts the format before the size"
        calls.clear()
        _, _, name = grab_camera(0, 0.1, None)
        assert cv2.CAP_PROP_FOURCC not in calls and name == "?"
    assert C.CAM_FOURCC == "MJPG", "the comparison run restores the configured format"
finally:
    C.IS_MAC = saved_mac

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for i in range(12):
        image = np.full((180, 320, 3), 30 + i * 5, np.uint8)
        cv2.circle(image, (160, 90), 30, (200, 180, 160), -1)
        writer.write(image)
    writer.release()
    clip = read_video(path, 10)
    assert len(clip) == 10 and clip[0].shape == (C.CAM_H, C.CAM_W, 3), "resized to camera size"
    stages = measure(clip, warmup=2)
assert set(stages) == {"convert", "holistic", "face_detection", "video_overlay", "hud"}
assert all(0 <= v["p50"] <= v["p95"] for v in stages.values())

fps, tips = advice({"stages": stages, "camera_fps": 9.5})
assert fps > 0 and any("카메라가 초당 10장" in t for t in tips), tips
slow = {k: {"p50": 100.0, "p95": 100.0} for k in stages}
fps, tips = advice({"stages": slow})
assert fps < 15 and any("CPU" in t for t in tips) and any("VIDEO_OVERLAY_FPS" in t for t in tips)
print("[OK] profiler: MJPG requested before size, delivered fps, stage timings, advice")
sys.exit(0)
