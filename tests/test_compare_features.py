"""Feature blocks partition the 3D features; comparisons share data, splits and settings."""
import _path  # noqa
import sys
try:
    import torch
except ImportError:
    sys.exit(77)
import contextlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np
from src.pose_sequence import (PoseSequence, VERSION, VERSION_3D, FEATURE_BLOCKS, FEATURE_SETS,
                                FEATURE_DIM, FEATURE_DIM_3D, features, features_3d, needs_world)
from src.continuous_sign import Recognizer
from tools.compare_features import compare

torch.set_num_threads(1)
columns = np.sort(np.concatenate(list(FEATURE_BLOCKS.values())))
assert np.array_equal(columns, np.arange(FEATURE_DIM_3D)), "Blocks must cover each column once"
assert FEATURE_SETS["2d"][1] == FEATURE_DIM and FEATURE_SETS["3d"][1] == FEATURE_DIM_3D
assert FEATURE_SETS["2d_no_velocity"][1] == FEATURE_DIM - 98
assert needs_world("3d_orientation") and not needs_world("2d_no_position")

rng = np.random.default_rng(1)


def clip(n, world=True):
    return PoseSequence(np.arange(n) / 30, rng.uniform(.1, .8, (n, 49, 2)), np.ones((n, 49), bool),
                        (640, 480), world=rng.normal(0, 1, (n, 49, 3)) if world else None,
                        world_mask=np.ones((n, 49), bool) if world else None)


s = clip(12)
assert np.array_equal(FEATURE_SETS["2d"][0](s), features(s)), "2d preset is the original order"
assert np.array_equal(FEATURE_SETS["3d"][0](s), features_3d(s))


def dataset(root, world):
    root.mkdir()
    clips = []
    for i in range(6):
        clip(16 + i, world).save(root / f"{i}.npz")
        clips.append({"id": str(i), "file": f"{i}.npz", "signer_id": f"s{i}", "session_id": "d",
                      "group_id": str(i), "glosses": ["A", "B"], "continuous": True})
    clips[-1].update(glosses=[], negative=True)
    (root / "manifest.json").write_text(json.dumps(
        {"version": VERSION_3D if world else VERSION, "clips": clips}))
    return root / "manifest.json"


with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
    td = Path(td)
    manifest = dataset(td / "data3d", True)
    result = compare(manifest, td / "run", ["2d", "3d_orientation"], epochs=1, batch=2)
    assert [r["feature_set"] for r in result["results"]] == ["2d", "3d_orientation"]
    assert not result["test_evaluated"] and result["split_sizes"]["test"] >= 1
    loaded = [Recognizer(td / "run" / r["checkpoint"]) for r in result["results"]]
    assert loaded[0].feature_set == "2d" and loaded[1].feature_set == "3d_orientation"
    assert loaded[0].metadata["splits"] == loaded[1].metadata["splits"] == result["splits"]
    assert all(r["val_wer"] is not None and r["feature_dim"] > 0 for r in result["results"])
    table = (td / "run" / "comparison.md").read_text(encoding="utf-8")
    assert "| 3d_orientation |" in table and "test는 평가하지 않음" in table
    again = compare(manifest, td / "again", ["2d"], epochs=1, batch=2)
    assert again["splits"] == result["splits"], "Same seed must reproduce the split"
    manifest_2d = dataset(td / "data2d", False)
    for source, names in [(manifest_2d, ["2d", "3d"]), (manifest, ["2d", "2d"]),
                          (manifest, ["nope"]), (manifest, [])]:
        try:
            compare(source, td / "bad", names, epochs=1)
            raise AssertionError(f"Invalid comparison accepted: {names}")
        except ValueError:
            pass
    assert not (td / "bad").exists(), "Rejected comparisons must not leave output"
    try:
        compare(manifest, td / "run", ["2d"], epochs=1)
        raise AssertionError("Existing comparison overwritten")
    except ValueError:
        pass
print("[OK] feature blocks, shared splits, per-config checkpoints and validation-only comparison")
