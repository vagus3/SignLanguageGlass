"""Segmentation evaluation: replay like main.py, score edges, cuts, splits and false starts."""
import _path  # noqa: F401
import json
import tempfile
from pathlib import Path
import numpy as np
from src.pose_sequence import PoseSequence
from tools.evaluate_segments import errors, evaluate, grid_search, load, replay, score_clip
from src.segmenter import ActivitySegmenter

FPS = 30
hand = np.zeros((21, 2))
hand[:, 1] = -np.linspace(0, .12, 21)
hand[9] = [0, -.09]


def clip(script, seconds):
    """script(t) -> (x, y) of the left hand; shoulders visible (webcam rest line y=0.7)."""
    n = int(seconds * FPS)
    points, mask = np.zeros((n, 49, 2), np.float32), np.zeros((n, 49), bool)
    for i in range(n):
        x, y = script(i / FPS)
        points[i, :21], mask[i, :21] = hand + [x, y], True
        points[i, 43], points[i, 44], mask[i, 43:45] = [.35, .4], [.65, .4], True
    return PoseSequence(np.arange(n) / FPS, np.clip(points, 0, 1), mask, (640, 640))


def signing(t):
    return .4 + .1 * np.sin(2 * np.pi * 1.5 * t), .6


def lowered(t):
    return .4 + .1 * np.sin(2 * np.pi * 1.5 * t), .95      # typing below the rest line


def two_sentences(t):
    return signing(t) if 2 <= t < 5 or 7 <= t < 10 else (.4, .95)


def pause_inside(t):                                        # hands held still for 2 s mid-sentence
    if 2 <= t < 4 or 6 <= t < 8:
        return signing(t)
    return (.45, .6) if 4 <= t < 6 else (.4, .95)


conversation = clip(two_sentences, 12)
segments = replay(conversation, ActivitySegmenter(viewpoint="webcam"))
assert len(segments) == 2 and all(kind == "stop" for *_, kind in segments), segments
rows, false = score_clip(segments, [(2, 5), (7, 10)])
assert all(r["detected"] and not r["split"] for r in rows) and not false
# Pre-roll keeps the onset: segments start at most ~0.1 s after the true start.
assert all(r["start_delay"] < 0.15 for r in rows), rows
assert all(0 < r["end_delay"] < 1.5 for r in rows), rows      # lowering the hands ends it

report = evaluate([("talk", conversation, [(2, 5), (7, 10)]),
                   ("typing", clip(lowered, 20), []),
                   ("waving", clip(signing, 6), [])])
assert report["intervals"] == 2 and report["detected"] == 2 and report["split"] == 0
assert report["false_starts"] == 1, "waving above the rest line is a false start"
assert abs(report["false_per_10min"] - 600 / 38) < 0.5
assert report["start_delay"]["p50"] < report["end_delay"]["p50"]

# A 2 s still pause inside one sentence splits it with the default still_hold of 1.5 s.
paused = [("pause", clip(pause_inside, 10), [(2, 8)])]
default = evaluate(paused)
assert default["split"] == 1 or default["cut_end"] == 1, default
tuned = grid_search(paused, grid={"still_hold": (1.5, 2.5)})
assert tuned["params"] == {"still_hold": 2.5} and tuned["split"] == 0 and tuned["cut_end"] == 0
assert errors(tuned) < errors(default)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    conversation.save(td / "a.npz")
    clip(lowered, 5).save(td / "b.npz")
    rows = [{"id": "a", "file": "a.npz", "sign_intervals": [[2, 5], [7, 10]], "split": "val"},
            {"id": "b", "file": "b.npz", "negative": True, "split": "test"},
            {"id": "c", "file": "a.npz", "split": "val"}]           # unannotated: skipped
    (td / "manifest.json").write_text(json.dumps({"clips": rows}), encoding="utf-8")
    assert [c[0] for c in load(td / "manifest.json")] == ["a", "b"]
    assert [c[0] for c in load(td / "manifest.json", "val")] == ["a"]
    rows[0]["sign_intervals"] = [[3, 2]]
    (td / "manifest.json").write_text(json.dumps({"clips": rows}), encoding="utf-8")
    try:
        load(td / "manifest.json")
        raise AssertionError("A reversed interval was accepted")
    except ValueError:
        pass
print("[OK] segment evaluation: main-loop replay with pre-roll, edges, splits, "
      "false starts per 10 min, split-restricted grid search")
