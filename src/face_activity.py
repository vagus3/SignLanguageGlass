"""Multi-face lip measurements, in the original (unmirrored) camera coordinates."""
import math

import config as C


def mesh_observations(result, width, height):
    boxes, openings = [], []
    for face in getattr(result, "multi_face_landmarks", None) or []:
        points = face.landmark
        if len(points) < 455:
            continue
        xy = [(p.x * width, p.y * height) for p in points]
        if not all(math.isfinite(v) for p in xy for v in p):
            continue
        x1, x2 = max(0, min(p[0] for p in xy)), min(width - 1, max(p[0] for p in xy))
        y1, y2 = max(0, min(p[1] for p in xy)), min(height - 1, max(p[1] for p in xy))
        # Cheek distance normalizes scale; Euclidean lip distance tolerates head roll.
        scale = math.dist(xy[234], xy[454])
        if x2 <= x1 or y2 <= y1 or scale < 12:
            continue
        boxes.append((x1, y1, x2, y2))
        openings.append(math.dist(xy[13], xy[14]) / scale)
    return boxes, openings


class MouthDetector:
    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=C.SPEAKER_MAX_FACES, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, rgb):
        import cv2
        h, w = rgb.shape[:2]
        if w > 640:
            rgb = cv2.resize(rgb, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
        return self.mesh.process(rgb)

    def close(self):
        self.mesh.close()
