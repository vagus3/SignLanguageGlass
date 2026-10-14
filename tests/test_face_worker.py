"""Face-only capture avoids Holistic, emits all boxes and releases failures."""
import _path  # noqa: F401
import queue
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import cv2  # noqa: F401 — load native extensions before patch.dict restores sys.modules

from main import VisionThread, FaceThread


def detection(x):
    return NS(score=[0.9], location_data=NS(relative_bounding_box=
              NS(xmin=x, ymin=0.2, width=0.2, height=0.4)))


state = {}


class Camera:
    def __init__(self, **kwargs):
        pass

    def ok(self):
        return True

    def read(self):
        state["worker"].stop()
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        state["released"] = True


class Detector:
    def __init__(self, **kwargs):
        if state.get("fail_init"):
            raise RuntimeError("detector initialization failed")

    def process(self, frame):
        if state.get("fail_process"):
            raise RuntimeError("detector processing failed")
        return NS(detections=[detection(0.1), detection(0.6)])

    def close(self):
        state["closed"] = True


fake_mp = NS(solutions=NS(face_detection=NS(FaceDetection=Detector)))
for constructor in (lambda q: VisionThread(q, enable_sign=False), lambda q: FaceThread(q, 1)):
    for failure in (None, "fail_init", "fail_process"):
        state.clear()
        if failure:
            state[failure] = True
        out = queue.Queue(maxsize=1)
        worker = constructor(out)
        state["worker"] = worker
        with patch("src.camera.CameraReader", Camera), \
                patch.dict("sys.modules", {"mediapipe": fake_mp}), \
                patch("src.landmarks.make_holistic", side_effect=AssertionError("Holistic unnecessary")):
            worker.run()
        kind, payload = out.get_nowait()
        assert state["released"]
        if failure:
            assert kind == "error", (kind, payload)
        else:
            assert len(payload["face_boxes"]) == 2
            assert payload["captured_at"] > 0 and payload["size"] == (640, 480)
        if failure != "fail_init":
            assert state["closed"]

print("[OK] Face-only workers: no Holistic, multiple detections, timestamps and error cleanup")
