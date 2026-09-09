"""명시된 모델 출력 형식에 따라 확률 변환이 정확한지 검증."""
import _path  # noqa: F401
import sys
import numpy as np

from src.runtime import SignRuntime, softmax


class FakeSession:
    def __init__(self, out):
        self.out = np.asarray([out], np.float32)

    def run(self, *_args, **_kwargs):
        return [self.out]


def runtime(out, kind):
    rt = SignRuntime.__new__(SignRuntime)
    rt.backend = "onnx"
    rt._sess = FakeSession(out)
    rt._iname = "input"
    rt.output_kind = kind
    rt._warned_auto_output = False
    rt.last_ms = 0.0
    return rt


raw = np.array([0.2, 0.3, 0.5], np.float32)
got = runtime(raw, "logits").predict(np.zeros((1, 2, 3), np.float32))
assert np.allclose(got, softmax(raw)), (got, softmax(raw))
assert not np.allclose(got, raw), "합이 1인 양수 로짓을 확률로 오인했습니다"

prob = runtime(raw, "probabilities").predict(np.zeros((1, 2, 3), np.float32))
assert np.allclose(prob, raw)

try:
    runtime([np.nan, 0, 1], "logits").predict(np.zeros((1, 2, 3), np.float32))
    raise AssertionError("NaN 출력을 허용했습니다")
except RuntimeError:
    pass

print("  양수 로짓 softmax / 확률 유지 / NaN 거부 확인")
sys.exit(0)
