"""Resampled word batches train, reload and preserve the new feature contract."""
import _path  # noqa: F401
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
try:
    import torch
except ImportError:
    sys.exit(77)
from src.pose_sequence import PoseSequence, VERSION
from src.mediapipe_features import FEATURE_SET
from src.continuous_sign import collate, Recognizer, greedy_decode
from src.train_continuous import prepare_training, train_one

torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    ts = np.arange(41) / 40
    p = np.zeros((41, 49, 2), np.float32)
    p[:, :21] = [.4, .6] + np.c_[np.linspace(0, .02, 21), -np.linspace(0, .1, 21)]
    p[:, 21:42] = [.6, .6] + np.c_[np.linspace(0, .02, 21), -np.linspace(0, .1, 21)]
    p[:, 42:] = [[.5, .35], [.35, .35], [.65, .35], [.3, .5], [.7, .5], [.4, .6], [.6, .6]]
    sequence = PoseSequence(ts, p, np.ones((41, 49), bool), (1280, 720))
    clips = []
    for i in range(6):
        sequence.save(root / f"{i}.npz")
        clips.append({"id": str(i), "file": f"{i}.npz", "signer_id": f"s{i//2}",
                      "session_id": "one", "group_id": str(i), "glosses": ["A"],
                      "continuous": False})
    manifest = {"version": VERSION, "extractor": {"name": "mediapipe-holistic"}, "clips": clips}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    skipped, splits, labels = prepare_training(path, feature_sets=[FEATURE_SET], task="words")
    x, lengths, _, _ = collate(splits["train"], labels, FEATURE_SET)
    assert x.shape[1:] == (31, 331) and lengths.tolist() == [31] * len(splits["train"])
    report = train_one(splits, labels, FEATURE_SET, root / "model", epochs=1,
                       task="words", extractor=manifest["extractor"])
    loaded = Recognizer(root / "model/ctc.pt")
    assert loaded.feature_set == FEATURE_SET and loaded.metadata["task"] == "words"
    assert not loaded.metadata["continuous"] and not loaded.metadata["synthetic_sentences"]
    assert report["validation"]["wer"] is not None
    one, one_lengths, _, _ = collate([splits["train"][0]], labels, FEATURE_SET)
    with torch.inference_mode():
        expected = greedy_decode(loaded.model(one, one_lengths)[:, 0].exp().numpy(), labels)
    assert loaded.predict(sequence) == expected, "Training batch and reloaded inference must agree"
    impossible = {**splits["train"][0], "glosses": ["A"] * 40}
    try:
        collate([impossible], labels, FEATURE_SET)
        raise AssertionError("Impossible resampled CTC target passed")
    except ValueError as error:
        assert "preprocessing" in str(error)
    manifest["extractor"]["name"] = "openpose-json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        prepare_training(path, feature_sets=[FEATURE_SET], task="words")
        raise AssertionError("OpenPose input passed the MediaPipe training guard")
    except ValueError as error:
        assert "MediaPipe" in str(error)
print("[OK] MediaPipe word CTC: resampled lengths, training, validation, reload and source guard")
