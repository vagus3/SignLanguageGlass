"""evaluate.py 를 합성 데이터 + 가짜 런타임으로 전 구간 실행."""
import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys, json, tempfile
from pathlib import Path
import numpy as np
import config as C
from src.landmarks import FEATURE_DIM

# --- 합성 데이터: 촬영자 3명 x 5어휘 x 8시퀀스 ---
tmp = Path(tempfile.mkdtemp())
C.DATA_DIR = tmp / "raw"
C.LABELS_PATH = tmp / "labels.json"
VOCAB = ["NEUTRAL", "안녕하세요", "감사합니다", "병원", "얼마"]
C.VOCAB = VOCAB
from src.dataset_contract import current_contract
rng = np.random.default_rng(0)
C.DATA_DIR.mkdir(parents=True)
(C.DATA_DIR / "dataset_meta.json").write_text(
    json.dumps(current_contract(), ensure_ascii=False), encoding="utf-8")
for li, lab in enumerate(VOCAB):
    d = C.DATA_DIR / lab; d.mkdir(parents=True)
    for signer in ("kim", "lee", "park"):
        for n in range(8):
            a = (rng.normal(li, 0.4, (C.SEQ_LEN, FEATURE_DIM))).astype(np.float32)
            np.save(d / f"{signer}_{n:03d}.npy", a)
C.LABELS_PATH.write_text(json.dumps(VOCAB, ensure_ascii=False), encoding="utf-8")

# 하나는 일부러 shape 을 틀리게 (경고 경로 확인)
np.save(C.DATA_DIR / "병원" / "kim_099.npy", np.zeros((10, FEATURE_DIM), np.float32))

# --- 가짜 런타임: 클래스 li 의 평균이 li 이므로 대충 맞히도록 ---
import src.runtime as R
class FakeRT:
    backend = "onnx"; input_dim = FEATURE_DIM; n_classes = len(VOCAB)
    spec = {"seq_len": C.SEQ_LEN, "input_dim": FEATURE_DIM}
    def __init__(s, path, labels=None): s.labels = labels; s.last_ms = 0.0
    def predict(s, x):
        m = float(x.mean())
        d = -np.abs(np.arange(len(VOCAB)) - m) * 3.0
        s.last_ms = 2.0 + np.random.rand()
        e = np.exp(d - d.max()); return (e / e.sum()).astype(np.float32)
R.SignRuntime = FakeRT

model = tmp / "sign_lstm.onnx"; model.write_bytes(b"fake")
C.MODEL_PATH = model

import tools.evaluate as E
E.C = C
for split in ("group", "random"):
    print("\n" + "#"*66)
    print(f"#  --split {split}")
    print("#"*66)
    sys.argv = ["evaluate.py", str(model), "--split", split,
                "--csv", str(tmp / f"eval_{split}.csv")]
    E.main()

print("\n" + "#"*66)
print("#  --holdout park  (특정 촬영자를 통째로 테스트셋으로)")
print("#"*66)
sys.argv = ["evaluate.py", str(model), "--holdout", "park"]
E.main()

for f in tmp.glob("eval_*.csv"):
    print(f"\nCSV 확인 {f.name}: {len(f.read_text(encoding='utf-8-sig').splitlines())}행")
