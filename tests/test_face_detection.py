"""별도 전방 카메라의 얼굴 중심 좌표 변환 검사."""
import _path  # noqa: F401
from types import SimpleNamespace

from src.landmarks import detection_face_center_px


def det(x, y, w, h, score):
    box = SimpleNamespace(xmin=x, ymin=y, width=w, height=h)
    location = SimpleNamespace(relative_bounding_box=box)
    return SimpleNamespace(location_data=location, score=[score])


assert detection_face_center_px(SimpleNamespace(detections=[]), 1280, 720) is None

# 첫 항목이 아니라 신뢰도가 가장 높은 얼굴을 선택해야 합니다.
res = SimpleNamespace(detections=[
    det(0.1, 0.1, 0.2, 0.2, 0.5),
    det(0.4, 0.2, 0.2, 0.4, 0.9),
])
x, y = detection_face_center_px(res, 1000, 500)
assert abs(x - 500) < 1e-6 and abs(y - 200) < 1e-6, (x, y)

# 카메라 경계를 벗어난 검출 상자는 안전하게 클램프합니다.
x, y = detection_face_center_px(
    SimpleNamespace(detections=[det(-1.0, 2.0, 0.1, 0.1, 1.0)]), 100, 50)
assert (x, y) == (0.0, 49.0), (x, y)
print("  [OK] 별도 얼굴 카메라 좌표 선택/클램프")
