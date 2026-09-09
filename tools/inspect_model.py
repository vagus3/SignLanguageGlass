# -*- coding: utf-8 -*-
"""
모델 점검기 (프레임워크 자동 판별)

    python tools/inspect_model.py <모델경로>

지원 형식
    .pt .pth .bin .ckpt   PyTorch (state_dict / nn.Module / TorchScript)
    .h5 .keras .hdf5      Keras
    (폴더)/saved_model.pb TensorFlow SavedModel
    .onnx                 ONNX
    .tflite               TensorFlow Lite

확장자를 믿지 않고 매직바이트로도 확인합니다.
배포된 모델은 확장자가 엉뚱하게 붙어 있는 경우가 종종 있습니다.

알려주는 것 중 제일 중요한 두 가지
    입력 차원   추론 전처리가 정확히 이 shape 를 만들어야 합니다
    출력 클래스 수백~수천이면 마지막 층만 갈아끼워 전이학습하세요
"""
import argparse
from pathlib import Path


# ---------------------------------------------------------------- 판별
def detect(p: Path):
    if p.is_dir():
        return "savedmodel" if (p / "saved_model.pb").exists() else "dir"

    try:
        magic = p.open("rb").read(16)
    except OSError:
        magic = b""
    ext = p.suffix.lower()

    if magic.startswith(b"\x89HDF\r\n\x1a\n"):
        return "keras"                        # HDF5
    if magic.startswith(b"PK"):
        # zip 컨테이너: torch 신형 저장본 또는 .keras(v3)
        return "keras" if ext == ".keras" else "torch"
    if ext == ".onnx":
        return "onnx"
    if ext == ".tflite" or magic[4:8] == b"TFL3":
        return "tflite"
    if ext in (".h5", ".hdf5", ".keras"):
        return "keras"
    return "torch"                            # 구형 pickle 저장본 포함


def head(title):
    print(f"\n── {title} " + "─" * max(0, 48 - len(title)))


def summary(in_dim, out_dim):
    print("\n" + "=" * 62)
    print("  요약")
    print("=" * 62)
    print(f"  입력 차원   : {in_dim}   <- 전처리가 이 shape 를 만들어야 합니다")
    print(f"  출력 클래스 : {out_dim}")

    if isinstance(in_dim, int):
        for dim, name in ((84, "손 2개(21x2점, xy)"),
                          (98, "손 2개 + 상체 7점"),
                          (101, "features_v2 공통 부분집합"),
                          (246, "signglass v1 특징"),
                          (411, "OpenPose 전체(137점 x3)"),
                          (1662, "MediaPipe 전체")):
            if in_dim == dim:
                print(f"    {dim} == {name}")
        if in_dim % 3 == 0:
            print(f"    3D 또는 (x,y,conf) 라면 점 {in_dim // 3}개")
        if in_dim % 2 == 0:
            print(f"    2D 라면 점 {in_dim // 2}개")

    if isinstance(out_dim, int) and out_dim > 100:
        print(f"\n  ※ 클래스가 {out_dim}개입니다. 시연 어휘는 15개면 되므로")
        print("     마지막 층만 15개짜리로 갈아끼우고 나머지는 얼려서")
        print("     전이학습하세요. 처음부터 학습할 필요가 없습니다.")

    print("\n  다음:  python tools/to_onnx.py <이 모델>")
    print("         변환해두면 실행 환경은 onnxruntime 하나로 끝납니다.")


# ---------------------------------------------------------------- ONNX
def show_onnx(p):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    head("입출력")
    in_dim = out_dim = None
    for i in sess.get_inputs():
        print(f"  입력  {i.name:<20} {i.shape}  {i.type}")
        if i.shape and isinstance(i.shape[-1], int):
            in_dim = i.shape[-1]
    for o in sess.get_outputs():
        print(f"  출력  {o.name:<20} {o.shape}  {o.type}")
        if o.shape and isinstance(o.shape[-1], int):
            out_dim = o.shape[-1]

    try:
        import onnx
        m = onnx.load(str(p))
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        head("연산자")
        print("  " + ", ".join(f"{k}x{v}" for k, v in
                               sorted(ops.items(), key=lambda x: -x[1])[:14]))
        print(f"\n  opset: {[i.version for i in m.opset_import]}")
    except Exception:
        pass
    summary(in_dim, out_dim)


# ---------------------------------------------------------------- Keras / TF
def show_keras(p):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    try:
        m = tf.keras.models.load_model(str(p), compile=False)
    except Exception as e:
        print(f"로드 실패: {e}")
        print("커스텀 레이어가 있으면 custom_objects 를 넘겨야 합니다.")
        return
    head("구조")
    m.summary()
    ish, osh = getattr(m, "input_shape", None), getattr(m, "output_shape", None)
    print(f"\n  input_shape  = {ish}")
    print(f"  output_shape = {osh}")
    summary(ish[-1] if ish else None, osh[-1] if osh else None)


def show_savedmodel(p):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    m = tf.saved_model.load(str(p))
    head("서명(signature)")
    for name, fn in m.signatures.items():
        print(f"  [{name}]")
        for k, v in fn.structured_input_signature[1].items():
            print(f"    입력  {k}: {v.shape} {v.dtype.name}")
        for k, v in fn.structured_outputs.items():
            print(f"    출력  {k}: {v.shape} {v.dtype.name}")
    print("\n  SavedModel 은 바로 변환됩니다:")
    print(f"    python -m tf2onnx.convert --saved-model {p} --output model.onnx")


def show_tflite(p):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(p))
    it.allocate_tensors()
    head("입출력")
    ind = it.get_input_details()
    outd = it.get_output_details()
    for d in ind:
        print(f"  입력  {d['name']:<24} {list(d['shape'])} {d['dtype'].__name__}")
    for d in outd:
        print(f"  출력  {d['name']:<24} {list(d['shape'])} {d['dtype'].__name__}")
    summary(int(ind[0]["shape"][-1]) if ind else None,
            int(outd[0]["shape"][-1]) if outd else None)


# ---------------------------------------------------------------- PyTorch
def infer_io_torch(sd):
    in_dim = out_dim = None
    hidden = layers = 0
    kind = None
    all_keys = " ".join(sd.keys()).lower()
    for k, v in sd.items():
        shp = tuple(getattr(v, "shape", ()))
        if not shp:
            continue
        if k.endswith("weight_ih_l0") and len(shp) == 2:
            gates = 4 if "lstm" in all_keys else 3
            in_dim = in_dim or shp[1]
            hidden = shp[0] // gates
            kind = "LSTM" if gates == 4 else "GRU/RNN"
        if "weight_ih_l" in k:
            try:
                layers = max(layers,
                             int(k.split("weight_ih_l")[1].split("_")[0]) + 1)
            except (ValueError, IndexError):
                pass
        if in_dim is None and len(shp) == 3 and k.endswith("weight"):
            in_dim, kind = shp[1], kind or "Conv1d/TCN"
        if "self_attn" in k:
            kind = kind or "Transformer"
    lin = [tuple(v.shape) for k, v in sd.items()
           if k.endswith("weight") and len(getattr(v, "shape", ())) == 2]
    if lin:
        out_dim = lin[-1][0]
        if in_dim is None:
            in_dim, kind = lin[0][1], kind or "MLP"
    return kind, in_dim, out_dim, hidden, layers


def show_torch(p):
    import torch
    import torch.nn as nn

    try:
        m = torch.jit.load(str(p), map_location="cpu")
        print("\n형식: TorchScript")
        print("  -> 구조 정의 없이 바로 추론·변환 가능. 이식성이 가장 좋습니다.")
        print(m)
        return
    except Exception:
        pass

    obj = torch.load(str(p), map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        print("\n형식: nn.Module 통째 저장")
        print("  -> 클래스 정의가 import 가능해야 로드됩니다. 이식성이 나쁘니")
        print("     state_dict 나 TorchScript 로 다시 저장해 두세요.")
        print(obj)
        sd = obj.state_dict()
    elif isinstance(obj, dict):
        sd = obj
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                print(f"\n형식: 체크포인트 dict (가중치는 '{key}' 안)")
                for mk in ("epoch", "acc", "accuracy", "best_acc", "classes",
                           "labels", "vocab", "config", "args", "input_dim"):
                    if mk in obj:
                        print(f"    {mk} = {str(obj[mk])[:180]}")
                break
        else:
            print("\n형식: state_dict (가중치만)")
    else:
        raise SystemExit(f"알 수 없는 형식: {type(obj)}")

    head(f"레이어 ({len(sd)}개 텐서)")
    total = 0
    for k, v in sd.items():
        if hasattr(v, "shape"):
            print(f"  {k:<44} {tuple(v.shape)}")
            total += v.numel()
    print(f"\n  총 파라미터: {total:,}")

    kind, in_dim, out_dim, hid, lay = infer_io_torch(sd)
    print(f"\n  구조 추정: {kind or '판별 실패'}"
          + (f"  hidden={hid} layers={lay}" if hid else ""))
    summary(in_dim, out_dim)


# ---------------------------------------------------------------- main
DISPATCH = {"onnx": show_onnx, "keras": show_keras, "savedmodel": show_savedmodel,
            "tflite": show_tflite, "torch": show_torch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--as", dest="force", choices=list(DISPATCH),
                    help="자동 판별이 틀렸을 때 강제 지정")
    a = ap.parse_args()

    p = Path(a.path)
    if not p.exists():
        raise SystemExit(f"경로가 없습니다: {p}")

    kind = a.force or detect(p)
    size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            if p.is_dir() else p.stat().st_size)
    print("=" * 62)
    print(f"  {p.name}   ({size / 1e6:.1f} MB)   판별: {kind}")
    print("=" * 62)

    if kind == "dir":
        raise SystemExit("폴더인데 saved_model.pb 가 없습니다. 안쪽 파일을 지정하세요.")
    try:
        DISPATCH[kind](p)
    except ImportError as e:
        raise SystemExit(f"필요한 패키지가 없습니다: {e}\n"
                         f"  pip install -r requirements-train.txt")


if __name__ == "__main__":
    main()
