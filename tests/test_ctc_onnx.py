"""CTC ONNX export matches PyTorch on any length, keeps calibration, and runs without torch."""
import _path  # noqa: F401
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    import torch
except ImportError:
    sys.exit(77)
from src.continuous_sign import CtcRecognizer, Recognizer, save_checkpoint
from src.ctc_runtime import OnnxRecognizer
from src.pose_sequence import PoseSequence
from tools.export_ctc_onnx import export

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(0)
rng = np.random.default_rng(3)
labels = ["<blank>"] + [f"G{i}" for i in range(9)]
calibration = {"enabled": True, "threshold": 0.4, "negative_seconds": 600, "reviewer": "fixture",
               "mapping_hash": "abc", "language_model": "fixture-model"}


def sequence(n, world=False):
    return PoseSequence(np.arange(n) / 30, rng.uniform(.1, .9, (n, 49, 2)), np.ones((n, 49), bool),
                        (1280, 720), world=rng.normal(0, .3, (n, 49, 3)) if world else None,
                        world_mask=np.ones((n, 49), bool) if world else None)


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for feature_set, world in (("2d", False), ("3d", True)):
        checkpoint = td / f"{feature_set}.pt"
        save_checkpoint(checkpoint, CtcRecognizer(len(labels), hidden=32, feature_set=feature_set),
                        labels, calibration=calibration, metadata={"split_seed": 7})
        spec = export(checkpoint, td / f"{feature_set}.onnx")
        assert spec["verified"]["max_abs_diff"] < 1e-4 and 900 in spec["verified"]["lengths"]
        device, server = OnnxRecognizer(td / f"{feature_set}.onnx"), Recognizer(checkpoint)
        assert device.calibration == calibration and device.metadata["split_seed"] == 7
        assert device.feature_set == feature_set and device.labels == labels
        for n in (2, 45, 300):
            s = sequence(n, world)
            (a, sa), (b, sb) = device.predict(s), server.predict(s)
            assert a == b and abs(sa - sb) < 1e-5, (feature_set, n)
        empty = sequence(5, world)
        empty.mask[:] = False
        assert device.predict(empty) == ([], 0.0)

    try:
        OnnxRecognizer(td / "3d.onnx").predict(sequence(5))
        raise AssertionError("A 3D model accepted a 2D sequence")
    except ValueError:
        pass
    try:
        export(td / "2d.pt", td / "2d.onnx")
        raise AssertionError("Export overwrote an existing model")
    except ValueError:
        pass
    sidecar = td / "2d.json"
    original = json.loads(sidecar.read_text(encoding="utf-8"))
    for change in ({"feature_dim": 12}, {"format": "other"}, {"labels": ["G0"] + labels[1:]},
                   {"calibration": {**calibration, "negative_seconds": 10}}):
        sidecar.write_text(json.dumps({**original, **change}), encoding="utf-8")
        try:
            OnnxRecognizer(td / "2d.onnx")
            raise AssertionError(f"Accepted a tampered sidecar: {list(change)}")
        except ValueError:
            pass
    sidecar.write_text(json.dumps(original), encoding="utf-8")

    # The device environment has onnxruntime but not torch.
    probe = (f"import sys; sys.modules['torch'] = None; sys.path.insert(0, {str(ROOT)!r})\n"
             "import numpy as np\n"
             "from src.ctc_runtime import OnnxRecognizer\n"
             "from src.pose_sequence import PoseSequence\n"
             f"r = OnnxRecognizer({str(td / '2d.onnx')!r})\n"
             "s = PoseSequence(np.arange(9) / 30, np.full((9, 49, 2), .5), np.ones((9, 49), bool), (640, 480))\n"
             "print(r.predict(s))\n")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
print("[OK] CTC ONNX: equal to PyTorch on 2..900 frames (2D/3D), calibration kept, "
      "tamper checks, torch-free device runtime")
