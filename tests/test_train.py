import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys
from pathlib import Path
import numpy as np
from src.train import augment, N_FLAGS
from src.landmarks import FEATURE_DIM

rng = np.random.default_rng(0)
n, T, D = 6, 30, FEATURE_DIM
X = rng.normal(0, 0.5, (n, T, D)).astype(np.float32)
X[:, :, 45:108] = 0.0                                  # 왼손 미검출
X[:, :, -3:] = np.array([0.0, 1.0, 1.0], np.float32)   # 플래그

Xa, ya = augment(X, np.arange(n) % 3, 2, rng)
print(f"  증강 결과 shape {Xa.shape}  (원본 {X.shape} + 2배)")

uniq = np.unique(Xa[:, :, -N_FLAGS:])
print(f"  플래그에 나타난 값들      : {uniq}")
assert set(uniq.tolist()) <= {0.0, 1.0}, "플래그가 0/1 이 아닙니다"

print(f"  미검출 좌표 최대 절댓값   : {np.abs(Xa[:, :, 45:108]).max()}")
assert np.abs(Xa[:, :, 45:108]).max() == 0.0, "0 패딩이 잡음으로 오염됐습니다"

moved = np.abs(Xa[n:, :, :45] - np.tile(X[:, :, :45], (2, 1, 1))).max()
print(f"  좌표 구간은 실제로 증강됨 : 최대 변화 {moved:.3f}")
assert moved > 0.01
print(f"  라벨 길이 일치            : {len(ya)} == {len(Xa)}")
assert len(ya) == len(Xa)

# ---- class_weight 키 매핑 (sklearn 없이 'balanced' 공식 직접 재현) ----
labels = ["a", "b", "c", "d"]
ytr = np.array([0, 0, 1, 1, 3, 3])          # 2번 클래스가 통째로 빠진 상황
present = np.unique(ytr)
w = len(ytr) / (len(present) * np.bincount(ytr)[present])
old = {i: float(v) for i, v in enumerate(w)}
new = {int(k): float(v) for k, v in zip(present, w)}
print(f"\n  학습셋에 있는 클래스 {present.tolist()}  ('c'(2번) 없음)")
print(f"  수정 전 키 : {sorted(old)}   <- 3번 가중치가 2번에 붙고, keras 는 3번에서 KeyError")
print(f"  수정 후 키 : {sorted(new)}   <- 라벨 번호와 일치")
assert sorted(new) == [0, 1, 3]
print("\n  [OK] 전부 통과")
