"""수집 FPS가 달라도 저장 시퀀스가 정확히 같은 시간폭인지 확인."""
import _path  # noqa: F401
import sys
import numpy as np

from src.landmarks import FEATURE_DIM, N_FLAGS, resample_timed


def sample(fps):
    ts = np.arange(0.0, 1.0 + 1.0 / fps, 1.0 / fps)
    x = np.zeros((len(ts), FEATURE_DIM), np.float32)
    x[:, 0] = ts                         # 시간에 선형인 가짜 좌표
    x[:, -N_FLAGS:] = (ts[:, None] >= 0.5).astype(np.float32)
    return resample_timed(x, ts, 30, start=0.0, end=1.0)


a10, a30 = sample(10), sample(30)
assert a10.shape == (30, FEATURE_DIM)
assert np.allclose(a10[:, 0], a30[:, 0], atol=1e-6)
assert set(np.unique(a10[:, -N_FLAGS:])).issubset({0.0, 1.0})

print("  10fps와 30fps 수집 결과의 시간축 일치")
print("  검출 플래그 0/1 유지")
sys.exit(0)
