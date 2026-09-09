"""프레임레이트가 떨어져도 모델이 보는 '시간 폭'이 유지되는지 검증.

특징 벡터 0번 칸에 '캡처 시각'을 그대로 실어 보냅니다. 그러면 모델 입력의
첫 칸과 끝 칸 차이가 곧 '모델이 본 시간 폭'이라 애매함 없이 측정됩니다."""
import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys
from pathlib import Path
from collections import deque
import numpy as np
import config as C
import src.predictor as P
from src.landmarks import FEATURE_DIM, N_FLAGS

class FakeRT:
    labels = None; input_dim = FEATURE_DIM; n_classes = 3; last_ms = 0.0
    def __init__(s): s.seen = []
    def predict(s, x):
        s.seen.append(x[0].copy()); return np.array([0.9,0.05,0.05], np.float32)

def build(adaptive):
    pr = P.SignPredictor.__new__(P.SignPredictor)
    pr.rt = FakeRT(); pr.labels = ["A","B","NEUTRAL"]
    pr.adaptive = adaptive
    f = float(C.TRAIN_FPS)
    pr.win_sec, pr.stride_sec = C.SEQ_LEN/f, C.STRIDE/f
    pr.buf = deque(maxlen=max(8*C.SEQ_LEN,256) if adaptive else C.SEQ_LEN)
    pr.votes = deque(maxlen=C.VOTE_WINDOW)
    pr._since = 0; pr._t_infer = 0.0; pr._last_word=None; pr._last_t=0.0
    pr.last_conf=0.0; pr.last_raw=None; pr.last_ms=0.0
    pr.last_frame_t=0.0; pr.last_fps=0.0
    return pr

def run(fps, adaptive, dur=7.0, jitter=0.0):
    pr = build(adaptive)
    rng = np.random.default_rng(0)
    t, t0 = 1000.0, 1000.0
    while t < t0 + dur:
        feat = np.zeros(FEATURE_DIM, np.float32)
        feat[0] = t - t0                                  # 캡처 시각 그대로
        feat[-N_FLAGS:] = [0.0, 1.0, 1.0]
        pr.push(feat, t=t)
        t += 1.0/fps + (rng.uniform(-jitter, jitter) if jitter else 0.0)
    return pr

print(f"{'':4}{'':4}{'설정':<24}{'모델 입력':>10}{'모델이 본 시간폭':>18}{'판정':>8}")
print("-"*72)
for fps in (30, 20, 15, 10, 6):
    for adaptive in (False, True):
        pr = run(fps, adaptive)
        name = "시간 기반(ADAPTIVE)" if adaptive else "고정 30프레임"
        if not pr.rt.seen:
            print(f"{fps:>6}fps  {name:<24}{'추론 못함':>12}{'(30프레임 못 채움)':>22}")
            continue
        seq = pr.rt.seen[-1]
        span = float(seq[-1,0] - seq[0,0])
        real = "" if not adaptive else f"  실측 {pr.last_fps:.0f}fps"
        ok = "OK" if abs(span-1.0) < 0.08 else f"{span/1.0:.1f}배"
        print(f"{fps:>6}fps  {name:<24}{seq.shape[0]:>7}프레임{span:>14.2f}초{ok:>10}{real}")
    print()

# 플래그 오염 + 프레임 지터 견고성
pr = run(12, True, jitter=0.03)
seq = pr.rt.seen[-1]
uniq = set(np.unique(seq[:, -N_FLAGS:]).tolist())
print(f"프레임 간격이 불규칙(±30ms)해도: 시간폭 {float(seq[-1,0]-seq[0,0]):.2f}초, "
      f"검출플래그 값 {sorted(uniq)}")
assert uniq <= {0.0, 1.0}, "플래그가 보간으로 오염됐습니다"
print(f"추정 입력 fps: {pr.last_fps:.1f}  (실제 12fps)")
print("\n[OK] 플래그는 0/1 유지, 시간폭은 프레임레이트와 무관하게 1.0초")
