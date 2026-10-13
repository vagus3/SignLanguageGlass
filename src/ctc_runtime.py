"""Torch-free CTC decoding and onnxruntime inference, for recognition on the device.

The device environment (requirements.txt) has onnxruntime but not torch. Export a
checkpoint with tools/export_ctc_onnx.py; the sidecar JSON carries the labels,
feature set, pose contract, calibration and metadata, so OnnxRecognizer makes the
same decisions as the PyTorch Recognizer in src/continuous_sign.py.
"""
import json
from pathlib import Path
import numpy as np
from src.pose_sequence import FEATURE_SETS, VERSION, VERSION_3D, needs_world

ONNX_FORMAT = "ctc-onnx-v1"
# What a device sends after recognizing a segment itself (POST /v1/sign/compose).
GLOSS_VERSION = "gloss-sequence-v1"


def greedy_decode(probabilities, labels):
    """Collapse repeats separated by CTC blank; score is not calibrated probability."""
    a = np.asarray(probabilities)
    path = a.argmax(-1)
    tokens, scores, previous = [], [], 0
    for i, token in enumerate(path):
        token = int(token)
        if token != 0 and token != previous:
            tokens.append(labels[token])
            scores.append(float(a[i, token]))
        previous = token
    return tokens, min(scores) if scores else 0.0


def check_contract(feature_set, version, feature_dim, labels):
    if (feature_set not in FEATURE_SETS
            or version != (VERSION_3D if needs_world(feature_set) else VERSION)
            or feature_dim != FEATURE_SETS[feature_set][1]):
        raise ValueError("Checkpoint pose contract mismatch")
    if not labels or labels[0] != "<blank>" or len(set(labels)) != len(labels):
        raise ValueError("Invalid checkpoint vocabulary")


def check_calibration(calibration):
    if calibration.get("enabled"):
        threshold = calibration.get("threshold")
        if (not isinstance(threshold, (int, float)) or not np.isfinite(threshold)
                or not 0 <= threshold <= 1 or calibration.get("negative_seconds", 0) < 600
                or not calibration.get("reviewer")):
            raise ValueError("Invalid automatic speech calibration")


class OnnxRecognizer:
    """Same interface as continuous_sign.Recognizer: predict(), labels, calibration."""

    def __init__(self, path, threads=2):
        import onnxruntime as ort
        path = Path(path)
        spec = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if spec.get("format") != ONNX_FORMAT:
            raise ValueError("Not a ctc-onnx-v1 export")
        self.feature_set = spec["feature_set"]
        self.labels = spec["labels"]
        check_contract(self.feature_set, spec["version"], spec["feature_dim"], self.labels)
        self.calibration = spec.get("calibration") or {}
        check_calibration(self.calibration)
        self.metadata = spec.get("metadata", {})
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads      # leave cores for the camera thread
        self.session = ort.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])

    def log_probs(self, features):
        """(T, F) features -> (T, classes) log-probabilities."""
        x = np.asarray(features, np.float32)[None]
        return self.session.run(None, {"features": x})[0][0]

    def predict(self, sequence):
        sequence.validate()
        if not sequence.mask[:, :42].any():
            return [], 0.0
        features = FEATURE_SETS[self.feature_set][0](sequence)
        return greedy_decode(np.exp(self.log_probs(features)), self.labels)
