"""Hand-only preview coordinates and independent detector integration."""
import _path  # noqa: F401
import queue
from types import SimpleNamespace as NS
from unittest.mock import patch

import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2

import main
from src import landmarks as L


hand = landmark_pb2.NormalizedLandmarkList()
for i in range(21):
    hand.landmark.add(x=0.2 + (i % 5) * 0.02, y=0.4 + (i // 5) * 0.03, z=0)
hands = NS(multi_hand_landmarks=[hand])
frame = np.zeros((240, 320, 3), np.uint8)
L.draw_hand_preview(frame, NS(multi_hand_landmarks=None))
assert not frame.any(), "No detections must not leave stale joints"
L.draw_hand_preview(frame, hands)
assert frame[96, 64].any(), "Wrist must match the original image coordinates"
assert not frame[:70].any(), "Hand preview must not draw face or body landmarks"
assert cv2.flip(frame, 1)[96, 255].any(), "Image and joints must mirror together"

out = queue.Queue()
worker = main.VisionThread(out, enable_sign=False, track_face=False, preview_hands=True)
empty = NS(pose_landmarks=None, face_landmarks=None,
           left_hand_landmarks=None, right_hand_landmarks=None)


class Camera:
    def ok(self):
        return True

    def read(self):
        worker.stop()
        return True, np.zeros((240, 320, 3), np.uint8)

    def release(self):
        self.closed = True


class Detector:
    def __init__(self, result):
        self.result = result
        self.closed = False

    def process(self, rgb):
        assert not rgb.any(), "Detection must use the image before overlays"
        return self.result

    def close(self):
        self.closed = True


camera = Camera()
holistic, detector = Detector(empty), Detector(hands)
with patch("src.camera.CameraReader", return_value=camera), \
        patch.object(L, "make_holistic", return_value=holistic) as make_holistic, \
        patch.object(L, "make_preview_hands", return_value=detector):
    worker.run()
kind, payload = out.get_nowait()
assert kind == "frame" and payload["hand_count"] == 1
assert payload["frame"][96, 64].any(), "Hands must display even when Holistic finds none"
make_holistic.assert_not_called()  # --no-sign uses only the independent hand preview.
assert camera.closed and not holistic.closed and detector.closed
print("[OK] hand-only preview, coordinates, mirroring, independent detection and cleanup")
