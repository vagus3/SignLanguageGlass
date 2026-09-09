"""PyTorch -> TorchScript/ONNX -> runtime 경로의 선택적 관통 테스트."""
import _path  # noqa: F401
import sys
import tempfile
from pathlib import Path

try:
    import torch
    import onnx  # noqa: F401
except ImportError as e:
    print(f"학습 선택 의존성 없음: {e.name}")
    sys.exit(77)

import numpy as np
import config as C
from src.landmarks import FEATURE_DIM
from src.train import build_model, export_models, fit_model
from src.runtime import SignRuntime

torch.manual_seed(0)
rng = np.random.default_rng(0)
X = rng.normal(size=(8, C.SEQ_LEN, FEATURE_DIM)).astype(np.float32)
y = np.array([0, 1] * 4, np.int64)
model = build_model(2)
before = next(model.parameters()).detach().clone()
fit_model(model, X[:6], y[:6], X[6:], y[6:], {0: 1.0, 1: 1.0},
          epochs=1, batch=2, seed=0)
after = next(model.parameters()).detach()
assert not torch.equal(before, after), "학습 후 파라미터가 바뀌지 않았습니다"

with tempfile.TemporaryDirectory() as td:
    pt = Path(td) / "sign_lstm.pt"
    onnx_path = Path(td) / "sign_lstm.onnx"
    err = export_models(model, pt, onnx_path)
    assert pt.exists() and onnx_path.exists() and err <= 1e-4
    rt = SignRuntime(onnx_path, labels=["a", "b"], warmup=False)
    p = rt.predict(X[:1])
    assert rt.output_kind == "logits"
    assert p.shape == (2,) and np.isfinite(p).all()
    assert np.isclose(p.sum(), 1.0, atol=1e-5)

print("  1 epoch 학습 + TorchScript/ONNX + runtime 확률 변환 확인")
sys.exit(0)
