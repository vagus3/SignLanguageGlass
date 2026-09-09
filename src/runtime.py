# -*- coding: utf-8 -*-
"""
통합 추론 런타임

학습은 무슨 프레임워크로 하든, 안경에서 돌아가는 건 이거 하나입니다.

    rt = SignRuntime("models/sign.onnx")
    prob = rt.predict(x)        # x: (1, seq, dim) -> (classes,)

우선순위
    1) onnxruntime   50MB, 시작 빠름, CPU 최적화. 기본값.
    2) torch         .pt 를 아직 변환 안 했을 때의 임시 경로
    3) keras         .h5 를 아직 변환 안 했을 때의 임시 경로

2) 3) 은 개발 중 편의용입니다. 실제 시연은 반드시 ONNX 로 하세요.
TF 와 torch 를 동시에 올리면 RAM 1GB 에 시작만 수 초 걸립니다.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def softmax(z):
    z = np.asarray(z, np.float32)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class SignRuntime:
    def __init__(self, path, labels=None, warmup=True):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} 가 없습니다.")

        self.backend = None
        self._sess = self._model = self._iname = None
        # 모델이 스스로 밝히는 입출력 크기. 알 수 없으면 None(torch 등).
        # 호출부가 "전처리 차원 / 라벨 개수"를 대조하는 데 씁니다.
        self.input_dim = None
        self.n_classes = None
        self.spec = self._load_spec()
        self.labels = labels or self.spec.get("labels")
        self.output_kind = self.spec.get("output_kind", "auto")
        self._warned_auto_output = False

        suf = self.path.suffix.lower()
        if suf == ".onnx":
            self._init_onnx()
        elif suf in (".pt", ".pth"):
            self._init_torch()
        elif suf in (".h5", ".keras", ".hdf5"):
            self._init_keras()
        else:
            raise ValueError(f"지원하지 않는 확장자: {suf}")

        self.last_ms = 0.0
        if warmup:
            self._warmup()

    # ------------------------------------------------------------------
    def _load_spec(self):
        """모델 옆의 *_preproc.json 을 읽습니다. 전처리 사양의 단일 출처."""
        for cand in (self.path.with_name(self.path.stem + "_preproc.json"),
                     self.path.parent / "preproc.json"):
            if cand.exists():
                try:
                    spec = json.loads(cand.read_text(encoding="utf-8"))
                    print(f"[runtime] 전처리 사양 로드: {cand.name}")
                    return spec
                except Exception as e:
                    print(f"[runtime] {cand.name} 파싱 실패: {e}")
        print("[runtime] 전처리 사양 파일이 없습니다. 차원 검증을 못 합니다.")
        return {}

    def _init_onnx(self):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2      # 비전 스레드가 CPU 를 써야 하므로 제한
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(self.path), so,
                                          providers=["CPUExecutionProvider"])
        self._iname = self._sess.get_inputs()[0].name
        self.backend = "onnx"
        shp = self._sess.get_inputs()[0].shape
        oshp = self._sess.get_outputs()[0].shape
        self.input_dim = shp[-1] if isinstance(shp[-1], int) else None
        self.n_classes = oshp[-1] if isinstance(oshp[-1], int) else None
        meta = self._sess.get_modelmeta().custom_metadata_map
        if self.output_kind == "auto" and meta.get("output_kind"):
            self.output_kind = meta["output_kind"]
        print(f"[runtime] onnxruntime  입력 {shp}  출력 {oshp}")
        self._check_dim(self.input_dim)

    def _init_torch(self):
        import torch
        try:
            self._model = torch.jit.load(str(self.path), map_location="cpu").eval()
        except Exception:
            obj = torch.load(str(self.path), map_location="cpu",
                             weights_only=False)
            if not hasattr(obj, "eval"):
                raise RuntimeError(
                    "state_dict 만 있는 파일은 직접 로드할 수 없습니다.\n"
                    "  tools/to_onnx.py 로 ONNX 변환 후 쓰세요.")
            self._model = obj.eval()
        torch.set_num_threads(2)
        self.backend = "torch"
        print("[runtime] torch (개발용 임시 경로). 시연 전 ONNX 변환 권장.")

    def _init_keras(self):
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        self._model = tf.keras.models.load_model(str(self.path), compile=False)
        self.backend = "keras"
        self.input_dim = self._model.input_shape[-1]
        self.n_classes = self._model.output_shape[-1]
        print("[runtime] keras (개발용 임시 경로). 시연 전 ONNX 변환 권장.")
        self._check_dim(self.input_dim)

    def _check_dim(self, model_dim):
        """
        전처리가 만드는 차원과 모델이 기대하는 차원을 대조합니다.
        안 맞으면 조용히 이상한 결과가 나오므로 시끄럽게 죽는 게 낫습니다.
        """
        want = self.spec.get("input_dim")
        if want and model_dim and int(want) != int(model_dim):
            raise RuntimeError(
                f"차원 불일치: 전처리 사양 {want} vs 모델 {model_dim}\n"
                f"  학습 때와 다른 특징 추출을 쓰고 있습니다. 이대로 돌리면\n"
                f"  에러 없이 성능만 무너집니다. preproc.json 을 확인하세요.")

    def _warmup(self):
        # spec 파일이 없어도 모델이 입력 차원을 알고 있으면 워밍업할 수 있습니다.
        # 첫 추론은 메모리 할당·커널 선택 때문에 10배까지 느립니다. 그걸 첫
        # 수어 동작이 아니라 시작 시점에 치릅니다.
        d = self.spec.get("input_dim") or self.input_dim
        s = self.spec.get("seq_len", 30)
        if not d:
            return
        try:
            self.predict(np.zeros((1, int(s), int(d)), np.float32))
            print(f"[runtime] 워밍업 완료 ({self.last_ms:.1f}ms)")
        except Exception as e:
            print(f"[runtime] 워밍업 실패: {e}")

    # ------------------------------------------------------------------
    def predict(self, x):
        """x: (1, seq, dim) float32 -> (classes,) 확률."""
        x = np.asarray(x, np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        t0 = time.perf_counter()

        if self.backend == "onnx":
            out = self._sess.run(None, {self._iname: x})[0]
        elif self.backend == "torch":
            import torch
            with torch.no_grad():
                out = self._model(torch.from_numpy(x))
            out = (out[0] if isinstance(out, (tuple, list)) else out).numpy()
        else:
            out = self._model(x, training=False).numpy()

        self.last_ms = (time.perf_counter() - t0) * 1000
        out = np.asarray(out, np.float32)
        if out.ndim > 1:
            out = out[0]
        if out.ndim != 1 or out.size == 0 or not np.isfinite(out).all():
            raise RuntimeError(f"모델 출력이 유효하지 않습니다: shape={out.shape}")

        # 양수 로짓 [0.2, 0.3, 0.5]는 합이 1이라도 확률이 아닙니다. 새 모델은
        # sidecar/ONNX metadata에 출력 의미를 기록하고, 그것을 최우선으로 씁니다.
        if self.output_kind == "logits":
            out = softmax(out)
        elif self.output_kind == "probabilities":
            if out.min() < 0 or float(out.sum()) <= 0:
                raise RuntimeError("확률 모델이 음수 또는 합계 0인 출력을 반환했습니다")
            out = out / out.sum()
        elif self.output_kind == "auto":
            if not self._warned_auto_output:
                print("[runtime] 출력 형식 메타데이터 없음 -> 값으로 확률/로짓을 추정합니다")
                self._warned_auto_output = True
            if out.min() < 0 or abs(float(out.sum()) - 1.0) > 1e-2:
                out = softmax(out)
        else:
            raise RuntimeError(f"지원하지 않는 output_kind: {self.output_kind}")
        return out

    def top(self, x, k=3):
        p = self.predict(x)
        idx = np.argsort(-p)[:k]
        return [(self.labels[i] if self.labels else int(i), float(p[i]))
                for i in idx]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seq", type=int, default=30)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--bench", type=int, default=50)
    a = ap.parse_args()

    rt = SignRuntime(a.model)
    dim = a.dim or rt.spec.get("input_dim")
    if not dim:
        raise SystemExit("--dim 을 지정하세요.")

    x = np.random.randn(1, a.seq, int(dim)).astype(np.float32)
    p = rt.predict(x)
    print(f"\n출력 {p.shape}  합계 {p.sum():.4f}  argmax {p.argmax()}")

    ts = []
    for _ in range(a.bench):
        rt.predict(x)
        ts.append(rt.last_ms)
    ts = np.array(ts)
    print(f"\n추론 {a.bench}회  중앙값 {np.median(ts):.2f}ms  "
          f"p95 {np.percentile(ts, 95):.2f}ms")
    print(f"  -> 5프레임(약 170ms)마다 추론하므로 여유가 충분합니다."
          if np.median(ts) < 30 else
          f"  -> 느립니다. STRIDE 를 늘리거나 모델을 줄이세요.")
