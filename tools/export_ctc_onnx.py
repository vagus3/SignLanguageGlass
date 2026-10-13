#!/usr/bin/env python
"""Export a CTC checkpoint to ONNX for torch-free inference (src/ctc_runtime.py).

python tools/export_ctc_onnx.py models/continuous/ctc.pt --out models/continuous/ctc.onnx

The exported graph takes one unpadded sequence (1, frames, features) of any
length, which is what a device recognizes after each segment. It is checked
against the PyTorch model on several lengths, including the 900-frame maximum;
the files are only written when log-probabilities and greedy decoding agree.
Whether to recognize on the device is decided by REVIEW_AND_PLAN 3.3 on the
target hardware, not by this export.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuous_sign import Recognizer
from src.ctc_runtime import ONNX_FORMAT, OnnxRecognizer, greedy_decode
from src.pose_sequence import FEATURE_SETS, MAX_FRAMES, VERSION, VERSION_3D, needs_world

CHECK_LENGTHS = (2, 37, 150, MAX_FRAMES)


class _Unpadded(nn.Module):
    """One sequence without padding: the LSTM sees exactly what packing would give it."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        hidden = self.model.encoder(features)
        hidden, _ = self.model.temporal(hidden)
        return self.model.output(hidden).log_softmax(-1)


def export(checkpoint, out, *, atol=1e-4, seed=0):
    out = Path(out)
    if out.exists() or out.with_suffix(".json").exists():
        raise ValueError("Output exists; choose a new path")
    recognizer = Recognizer(checkpoint)
    model = recognizer.model.eval()
    dim = FEATURE_SETS[recognizer.feature_set][1]
    spec = {"format": ONNX_FORMAT, "feature_set": recognizer.feature_set, "feature_dim": dim,
            "version": VERSION_3D if needs_world(recognizer.feature_set) else VERSION,
            "labels": recognizer.labels, "calibration": recognizer.calibration,
            "metadata": recognizer.metadata, "source_checkpoint": Path(checkpoint).name}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix="onnx-stage-") as td:
        staged = Path(td) / out.name
        # export() restores the wrapper's mode recursively afterwards; a new module is in
        # training mode, which would switch the LSTM dropout of the checked model back on.
        torch.onnx.export(_Unpadded(model).eval(), torch.zeros(1, 16, dim), str(staged),
                          input_names=["features"], output_names=["log_probs"],
                          dynamic_axes={"features": {1: "frames"}, "log_probs": {1: "frames"}},
                          opset_version=17)
        staged.with_suffix(".json").write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        exported = OnnxRecognizer(staged)
        rng = np.random.default_rng(seed)
        worst = 0.0
        for frames in CHECK_LENGTHS:
            features = rng.normal(0, 1, (frames, dim)).astype(np.float32)
            with torch.inference_mode():
                expected = model(torch.from_numpy(features)[None],
                                 torch.tensor([frames]))[:, 0].numpy()
            got = exported.log_probs(features)
            worst = max(worst, float(np.abs(got - expected).max()))
            if worst > atol:
                raise ValueError(f"ONNX differs from PyTorch by {worst:.2e} at {frames} frames")
            if (greedy_decode(np.exp(got), recognizer.labels)[0]
                    != greedy_decode(np.exp(expected), recognizer.labels)[0]):
                raise ValueError(f"ONNX decodes differently at {frames} frames")
        spec["verified"] = {"lengths": list(CHECK_LENGTHS), "max_abs_diff": worst, "atol": atol}
        staged.with_suffix(".json").write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        staged.replace(out)
        staged.with_suffix(".json").replace(out.with_suffix(".json"))
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    try:
        spec = export(args.checkpoint, args.out)
    except ValueError as error:
        p.error(str(error))
    print(f"{args.out}: {spec['feature_set']} features, {len(spec['labels']) - 1} glosses, "
          f"max |ONNX - PyTorch| {spec['verified']['max_abs_diff']:.1e} over "
          f"{spec['verified']['lengths']} frames")


if __name__ == "__main__":
    main()
