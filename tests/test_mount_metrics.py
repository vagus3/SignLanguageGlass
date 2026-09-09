"""장착 시점 통과 기준 계산 검사."""
import _path  # noqa: F401
from tools.check_mount import make_report

r = make_report(100, 90, 60, 80, 71, 10.0)
assert r["any_hand_rate"] == 0.9 and r["hand_gate_pass"]
assert r["face_rate"] == 0.8875 and not r["face_gate_pass"]
assert r["sign_fps"] == 10.0 and r["face_fps"] == 8.0

z = make_report(0, 0, 0, 0, 0, 1.0)
assert z["any_hand_rate"] == 0.0 and z["face_rate"] == 0.0
print("  [OK] 장착 검출률/90% 게이트 계산")
