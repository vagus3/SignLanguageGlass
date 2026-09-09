# -*- coding: utf-8 -*-
"""
모델 -> ONNX 변환 + 출력 일치 검증

    python tools/to_onnx.py model.h5                    # Keras
    python tools/to_onnx.py model.pt                    # TorchScript / nn.Module
    python tools/to_onnx.py w.pth --arch mymodel:SignLSTM --in-dim 101 --classes 15

★ 변환보다 '검증'이 본체입니다.
  ONNX 변환은 조용히 틀릴 수 있습니다. LSTM 초기 상태 처리나 동적 축이
  어긋나면 shape 은 맞는데 값만 달라집니다. 그러면 학습 정확도는 그대로인데
  실행하면 이상하게 나오는, 원인 찾기 제일 어려운 버그가 됩니다.
  그래서 원본과 ONNX 에 같은 난수를 넣고 최대 오차를 반드시 확인합니다.
"""
import sys
import json
import argparse
import importlib
from pathlib import Path

import numpy as np

TOL = 1e-4


def report(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.shape != b.shape:
        print(f"  [실패] shape 불일치 {a.shape} vs {b.shape}")
        return False
    err = np.abs(a - b).max()
    same_arg = int(a.argmax()) == int(b.argmax())
    print(f"  최대 절대오차 : {err:.3e}   (허용 {TOL:.0e})")
    print(f"  argmax 일치   : {same_arg}")
    if err < TOL and same_arg:
        print("  [통과] 원본과 ONNX 출력이 일치합니다.")
        return True
    print("  [실패] 값이 다릅니다. opset 을 바꾸거나 동적 축 설정을 확인하세요.")
    return False


# ---------------------------------------------------------------- Keras
def from_keras(src, dst, seq_len, opset):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    try:
        import tf2onnx
    except ImportError:
        raise SystemExit("pip install tf2onnx")

    m = tf.keras.models.load_model(str(src), compile=False)
    ishape = list(m.input_shape)
    ishape[0] = 1
    for i, v in enumerate(ishape):
        if v is None:
            ishape[i] = seq_len
    print(f"  입력 shape: {ishape}")

    spec = (tf.TensorSpec([None] + ishape[1:], tf.float32, name="input"),)
    tf2onnx.convert.from_keras(m, input_signature=spec, opset=opset,
                               output_path=str(dst))

    x = np.random.randn(*ishape).astype(np.float32)
    return x, m.predict(x, verbose=0)


# ---------------------------------------------------------------- PyTorch
def load_torch_model(src, arch, in_dim, classes):
    import torch
    import torch.nn as nn

    try:
        return torch.jit.load(str(src), map_location="cpu").eval()
    except Exception:
        pass

    obj = torch.load(str(src), map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()

    if not arch:
        raise SystemExit(
            "가중치(state_dict)만 있는 파일입니다. 모델 클래스가 필요합니다.\n"
            "  --arch 모듈경로:클래스명  (예: --arch src.models:SignLSTM)\n"
            "생성자 인자가 필요하면 --in-dim / --classes 로 넘기세요.")

    mod, cls = arch.split(":")
    sys.path.insert(0, str(Path.cwd()))
    Klass = getattr(importlib.import_module(mod), cls)
    kw = {}
    if in_dim:
        kw["input_dim"] = in_dim
    if classes:
        kw["num_classes"] = classes
    try:
        model = Klass(**kw)
    except TypeError:
        model = Klass()          # 인자 이름이 다르면 기본 생성자로

    sd = obj
    for key in ("state_dict", "model_state_dict", "model", "net"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            sd = obj[key]
            break
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [경고] 못 채운 파라미터: {list(missing)[:6]}")
    if unexpected:
        print(f"  [경고] 남는 파라미터: {list(unexpected)[:6]}")
    return model.eval()


def from_torch(src, dst, arch, in_dim, classes, seq_len, opset):
    import torch
    model = load_torch_model(src, arch, in_dim, classes)

    if not in_dim:
        raise SystemExit("--in-dim 을 지정하세요 "
                         "(tools/inspect_model.py 가 알려준 입력 차원).")
    x = torch.randn(1, seq_len, in_dim)
    print(f"  입력 shape: {tuple(x.shape)}")

    with torch.no_grad():
        ref = model(x)
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref

    torch.onnx.export(
        model, x, str(dst),
        input_names=["input"], output_names=["output"],
        # 배치와 시퀀스 길이를 동적으로. 슬라이딩 윈도우 길이를 바꿔도
        # 재변환하지 않아도 됩니다.
        dynamic_axes={"input": {0: "batch", 1: "seq"},
                      "output": {0: "batch"}},
        opset_version=opset,
    )
    return x.numpy(), ref.numpy()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--arch", help="state_dict 전용. 예: src.models:SignLSTM")
    ap.add_argument("--in-dim", type=int, default=None)
    ap.add_argument("--classes", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    src = Path(a.src)
    if not src.exists():
        raise SystemExit(f"경로가 없습니다: {src}")
    dst = Path(a.out) if a.out else src.with_suffix(".onnx")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.inspect_model import detect
    kind = detect(src)
    print(f"판별: {kind}  ->  {dst}")

    if kind in ("keras", "savedmodel"):
        x, ref = from_keras(src, dst, a.seq_len, a.opset)
    elif kind == "torch":
        x, ref = from_torch(src, dst, a.arch, a.in_dim, a.classes,
                            a.seq_len, a.opset)
    elif kind == "onnx":
        raise SystemExit("이미 ONNX 입니다.")
    else:
        raise SystemExit(f"{kind} 는 아직 지원하지 않습니다.")

    print("\n── 출력 일치 검증 ──────────────────────────────")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    ok = report(ref, got)

    meta = dst.with_name(dst.stem + "_preproc.json")
    if not meta.exists():
        meta.write_text(json.dumps({
            "note": "학습과 추론이 같은 전처리를 쓰도록 강제하는 사양 파일입니다.",
            "onnx": dst.name,
            "seq_len": a.seq_len,
            "input_dim": a.in_dim or (x.shape[-1] if hasattr(x, "shape") else None),
            "feature": "features_v2(101) 이면 hand_canonical+body_relative",
            "normalize": "TODO: 학습 때 쓴 정규화를 여기에 적으세요",
            "target_fps": 30,
            "mirror": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n전처리 사양 템플릿 생성: {meta}")
        print("  반드시 실제 학습 전처리에 맞게 채우세요. 이게 비면 나중에")
        print("  '학습은 잘 되는데 실시간은 이상함' 을 며칠 동안 찾게 됩니다.")

    print(f"\n완료: {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
