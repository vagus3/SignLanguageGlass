"""ego_relative_v1: 카메라 설정(화면 안 크기·위치)이 달라도 손 위치·속도 특징이 거의 같다."""
import _path  # noqa
import numpy as np
from src.pose_sequence import FEATURE_BLOCKS, FEATURE_SETS, JOINTS, PoseSequence

rng = np.random.default_rng(0)
T = 24
t = np.arange(T) / 30.0
base = np.zeros((T, JOINTS, 2), np.float32)
mask = np.zeros((T, JOINTS), bool)
for start in (0, 21):
    shape = rng.uniform(-0.04, 0.04, (21, 2)) + [0.0, -0.05]
    shape[9] = [0.0, -0.1]                                  # 손바닥 길이 0.1
    wrist = np.stack([0.4 + 0.1 * np.sin(t * 3), 0.5 + 0.05 * t], -1)
    base[:, start:start + 21] = wrist[:, None] + shape[None]
    mask[:, start:start + 21] = True


def make(scale, shift):
    c = np.where(mask[..., None], (base - 0.5) * scale + 0.5 + shift, 0).astype(np.float32)
    return PoseSequence(t, c, mask, (1280, 720)).validate()


relative = FEATURE_SETS["ego_relative_v1"][0]
plain = FEATURE_SETS["2d"][0]
a, b = make(1.0, 0.0), make(0.6, 0.08)       # 카메라가 멀어져 손이 작아지고 옮겨진 경우
cols = np.concatenate([FEATURE_BLOCKS["position"], FEATURE_BLOCKS["velocity"]])
gap_rel = np.abs(relative(a)[:, cols] - relative(b)[:, cols]).mean()
gap_plain = np.abs(plain(a)[:, cols] - plain(b)[:, cols]).mean()
assert relative(a).shape == plain(a).shape and FEATURE_SETS["ego_relative_v1"][1] == 331
assert gap_rel < gap_plain * 0.2, (gap_rel, gap_plain)
# 손 모양·보임·시간 열은 그대로
keep = np.concatenate([FEATURE_BLOCKS[k] for k in ("hand_shape", "mask", "timing")])
assert np.allclose(relative(a)[:, keep], plain(a)[:, keep])
# 보이지 않는 손은 0
assert not np.isnan(relative(a)).any()
print(f"[OK] ego_relative_v1: 시점이 바뀐 차이 {gap_plain:.3f} -> {gap_rel:.3f}")
