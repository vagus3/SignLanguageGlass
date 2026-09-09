"""어깨가 잘린 착용형 시점에서도 손 특징이 살아 있는지 검사."""
import _path  # noqa: F401
import numpy as np

from fake_res import Res
from src.landmarks import extract, N_POSE, N_HAND


hand = np.zeros((21, 3), np.float32)
for i in range(21):
    hand[i] = [0.30 + i * 0.008, 0.40 + (i % 5) * 0.012, i * 0.001]
hand[9] = [0.39, 0.48, 0.01]  # 손바닥 스케일이 0이 되지 않게 명시

a = extract(Res(lh=hand))
start = N_POSE * 3
left = a[start:start + N_HAND * 3].reshape(N_HAND, 3)
assert a[-3] == 1.0, "어깨가 없는데 왼손 검출 플래그가 꺼졌습니다"
assert np.any(left != 0), "어깨가 없다는 이유로 손 특징 전체가 사라졌습니다"

# 손 전체가 평행 이동해도 로컬 손 모양(1~20번)은 같고 0번 위치만 변합니다.
moved = hand.copy()
moved[:, 0] += 0.15
moved[:, 1] -= 0.10
b = extract(Res(lh=moved))[start:start + N_HAND * 3].reshape(N_HAND, 3)
np.testing.assert_allclose(left[1:], b[1:], atol=1e-5)
assert not np.allclose(left[0], b[0])
print("  [OK] 어깨 없는 하향 시점 손 특징 + 이동 불변 손 모양")
