"""한/두 카메라 및 고정 자막 실행 계획 검사."""
import _path  # noqa: F401
import config as C
from main import resolve_camera_plan

assert resolve_camera_plan(False, False, None, None) == (True, False, C.CAM_INDEX)
assert resolve_camera_plan(True, True, None, None) == (False, False, C.CAM_INDEX)
assert resolve_camera_plan(True, False, 0, 1) == (False, True, 0)
assert resolve_camera_plan(False, False, 0, 1) == (True, True, 0)

try:
    resolve_camera_plan(False, False, 0, 0)
    raise AssertionError("같은 장치를 두 번 여는 계획이 통과했습니다")
except ValueError:
    pass
print("  [OK] 카메라 실행 계획/장치 충돌 차단")
