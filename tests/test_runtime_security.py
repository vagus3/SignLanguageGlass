"""The runtime never unpickles a .pt module; conversion does so only when told to."""
import _path  # noqa: F401
import sys
import tempfile
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:
    sys.exit(77)
from src.runtime import SignRuntime
from tools.to_onnx import load_torch_model


class Marker:
    """Unpickling this writes a file: proof that loading ran code from the model file."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    marker = td / "executed"
    hostile = td / "hostile.pt"
    torch.save({"model": nn.Linear(3, 2), "payload": Marker(str(marker))}, hostile)
    try:
        SignRuntime(hostile, warmup=False)
        raise AssertionError("Runtime opened a pickled module")
    except RuntimeError as error:
        assert "TorchScript" in str(error)
    assert not marker.exists(), "Runtime executed code from a model file"
    try:
        load_torch_model(hostile, None, 3, 2)
        raise AssertionError("Conversion unpickled without --trust-pickle")
    except SystemExit as error:
        assert "--trust-pickle" in str(error)
    assert not marker.exists(), "Conversion executed code without --trust-pickle"

    scripted = td / "scripted.pt"
    torch.jit.trace(nn.Linear(3, 2).eval(), torch.zeros(1, 3)).save(str(scripted))
    rt = SignRuntime(scripted, warmup=False)
    assert rt.backend == "torch"

    # A state_dict still converts: weights_only=True covers tensors and dicts.
    weights = td / "weights.pt"
    torch.save(nn.Linear(3, 2).state_dict(), weights)
    model = load_torch_model(weights, "torch.nn:Identity", None, None)
    assert isinstance(model, nn.Module)

    # The explicit opt-in still works for a file the user made themselves.
    module = td / "module.pt"
    torch.save(nn.Linear(3, 2), module)
    assert isinstance(load_torch_model(module, None, 3, 2, trust_pickle=True), nn.Linear)
    # The marker does detect execution: the trusted path really runs the payload.
    try:
        load_torch_model(hostile, None, 3, 2, trust_pickle=True)
    except SystemExit:
        pass  # a dict, not a module; the payload already ran while unpickling
    assert marker.exists()
print("[OK] runtime: pickled modules refused without running code, TorchScript loads, "
      "conversion unpickles only with --trust-pickle")
