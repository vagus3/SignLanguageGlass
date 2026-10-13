"""Extractor comparison finds renumbered frames, swapped hand labels and plain agreement."""
import _path  # noqa: F401
import json
import tempfile
from pathlib import Path
import numpy as np
from src.pose_sequence import PoseSequence, VERSION
from tools.compare_extractors import compare, table

rng = np.random.default_rng(2)


def walk(n):
    start = rng.uniform(0.3, 0.7, (1, 49, 2))
    return np.clip(start + np.cumsum(rng.normal(0, 0.006, (n, 49, 2)), axis=0), 0.05, 0.95)


def noisy(coords):
    return np.clip(coords + rng.normal(0, 0.001, coords.shape), 0, 1)


def seq(coords, size, timestamps=None):
    t = np.arange(len(coords)) / 30 if timestamps is None else timestamps
    return PoseSequence(t, coords, np.ones((len(coords), 49), bool), size)


def dataset(root, clips):
    root.mkdir()
    rows = []
    for i, (identity, s) in enumerate(clips.items()):
        s.save(root / f"{i}.npz")
        rows.append({"id": identity, "file": f"{i}.npz"})
    (root / "manifest.json").write_text(json.dumps({"version": VERSION, "clips": rows}))
    return root / "manifest.json"


same = walk(120)
video = walk(125)
# Like AI Hub WORD1515_F: the JSON drops video frames 62-66 and renumbers the rest.
json_frames = np.delete(video, range(62, 67), axis=0)
# Like WORD1515_L: the same drop, but the frame numbers (timestamps) keep the gap.
kept_numbers = np.delete(np.arange(125), range(62, 67)) / 30
labels = walk(120)
swapped = noisy(labels)
swapped[:, :21], swapped[:, 21:42] = swapped[:, 21:42].copy(), swapped[:, :21].copy()

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    first = dataset(td / "openpose", {"same": seq(same, (1920, 1080)),
                                      "renumbered": seq(json_frames, (1920, 1080)),
                                      "swapped": seq(labels, (1920, 1080)),
                                      "gap_kept": seq(json_frames, (1920, 1080), kept_numbers),
                                      "extra": seq(same, (1920, 1080))})
    second = dataset(td / "mediapipe", {"same": seq(noisy(same), (1280, 720)),
                                        "renumbered": seq(noisy(video), (1280, 720)),
                                        "swapped": seq(swapped, (1280, 720)),
                                        "gap_kept": seq(noisy(video), (1280, 720))})
    report = compare(first, second)
    clips = {c["id"]: c for c in report["clips"]}
    assert report["only_in_first"] == ["extra"] and report["only_in_second"] == []

    c = clips["same"]
    assert not c["misaligned"] and not c["hands_swapped"] and not c["aspect_mismatch"]
    assert c["hands_px"] < 5 and c["body_px"] < 5, "Pixels use the first dataset's 1920 width"

    c = clips["renumbered"]
    assert c["frames"] == [120, 125] and c["misaligned"]
    offsets = {w["start"]: w["offset"] for w in c["windows"]}
    assert offsets[0] == 0 and offsets[30] == 0 and offsets[60] == 5 and offsets[90] == 5, offsets
    # A clip-wide median hides it (62 of 120 frames still line up); the windows do not.
    shifted = [w for w in c["windows"] if w["offset"]]
    assert all(w["median_px_at_0"] > 10 > 5 > w["median_px"] for w in shifted), shifted
    assert c["hands_px_aligned"] < 5, "Aligning the windows removes the error"

    c = clips["gap_kept"]
    assert not c["misaligned"] and c["hands_px"] < 5, "A gap the timestamps keep is not an error"
    assert clips["swapped"]["hands_swapped"] and not clips["same"]["hands_swapped"]
    assert report["summary"]["misaligned"] == ["renumbered"]
    assert report["summary"]["hands_swapped"] == ["swapped"]
    text = table(report)
    assert "swapped hands" in text and "offset [5] from frame 60" in text

    lonely = dataset(td / "lonely", {"other": seq(same, (1280, 720))})
    try:
        compare(first, lonely)
        raise AssertionError("Compared manifests with no shared clip")
    except ValueError:
        pass
print("[OK] extractor comparison: timestamp pairing, renumbered-frame offsets, kept gaps, "
      "swapped hand labels")
