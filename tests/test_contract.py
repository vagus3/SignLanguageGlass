import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys, json, types
from pathlib import Path
import src.runtime as R
import src.predictor as P
from src.landmarks import FEATURE_DIM

class FakeRT:
    def __init__(self, path, labels=None, input_dim=246, n_classes=15):
        self.labels, self.input_dim, self.n_classes = labels, input_dim, n_classes

def build(labels, input_dim, n_classes, tmp):
    model = Path(tmp) / "m.onnx"; model.write_bytes(b"x")
    lp = Path(tmp) / "labels.json"
    if labels is not None:
        lp.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    R.SignRuntime = lambda p, labels=None: FakeRT(p, labels, input_dim, n_classes)
    return P.SignPredictor(model_path=model, labels_path=lp)

import tempfile
V = ["NEUTRAL"] + [f"w{i}" for i in range(14)]      # 15개

cases = [
    ("정상 (246차원 / 라벨 15 / 클래스 15)", V, 246, 15, None),
    ("차원 불일치 (모델 101 = features_v2 로 학습)", V, 101, 15, "특징 차원 불일치"),
    ("라벨 수 불일치 (VOCAB 늘리고 재학습 안 함)", V + ["새단어"], 246, 15, "라벨 개수 불일치"),
    ("labels.json 없음 (경고만, 숫자 인덱스)", None, 246, 15, None),
    ("torch 백엔드처럼 크기를 모를 때 (검사 생략)", V, None, None, None),
]
bad = 0
for name, labels, dim, ncls, want_err in cases:
    try:
        pr = build(labels, dim, ncls, tempfile.mkdtemp())
        got = None
        extra = f"labels {len(pr.labels)}개"
    except RuntimeError as e:
        got = str(e).split("\n")[0]
        extra = ""
    if want_err is None:
        ok = got is None
    else:
        ok = got is not None and want_err in got
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    print(f"        -> {got or ('통과, ' + extra)}")
print(f"\n실패 {bad}")
sys.exit(1 if bad else 0)
