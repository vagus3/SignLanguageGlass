"""AI Hub word bundles: only the listed words are read from the zips, with sign intervals."""
import _path  # noqa: F401
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from src.continuous_sign import load_dataset
from tools.prepare_aihub_words import prepare

rng = np.random.default_rng(4)


def frame(empty=False):
    if empty:
        return {"people": []}
    part = lambda n: np.c_[rng.uniform(100, 900, (n, 2)), np.ones(n)].ravel().tolist()
    return {"people": {"pose_keypoints_2d": part(25), "hand_left_keypoints_2d": part(21),
                       "hand_right_keypoints_2d": part(21)}}


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    clips = {"NIA_SL_WORD0001_REAL03_F": [[1.0, 2.0]], "NIA_SL_WORD0001_REAL03_L": [[1.1, 2.1]],
             "NIA_SL_WORD0002_REAL03_F": [[0.5, 1.5]], "NIA_SL_WORD0003_REAL03_F": [],
             "NIA_SL_WORD0009_REAL03_F": [[0.2, 0.9]]}
    with zipfile.ZipFile(td / "morpheme.zip", "w") as z:
        for clip, spans in clips.items():
            z.writestr(f"morpheme/03/{clip}_morpheme.json", json.dumps(
                {"metaData": {"name": clip + ".mp4"},
                 "data": [{"start": a, "end": b, "attributes": [{"name": "x"}]}
                          for a, b in spans]}))
    with zipfile.ZipFile(td / "keypoint.zip", "w") as z:
        for clip in clips:
            for i in range(6):
                z.writestr(f"03/{clip}/{clip}_{i:012d}_keypoints.json",
                           json.dumps(frame(empty=clip.endswith("0002_REAL03_F") and i == 2)))
    words = {"병원": ["WORD0001"], "아프다": ["WORD0002"], "머리": ["WORD0003"],
             "목": ["WORD0500"]}
    report, skipped, missing = prepare(td / "morpheme.zip", td / "keypoint.zip", words,
                                       td / "out")
    rows = {r["id"]: r for r in report["clips"]}
    assert set(rows) == {"NIA_SL_WORD0001_REAL03_F", "NIA_SL_WORD0001_REAL03_L",
                         "NIA_SL_WORD0002_REAL03_F"}, "unlisted words are never read"
    row = rows["NIA_SL_WORD0001_REAL03_L"]
    assert row["glosses"] == ["병원"] and row["signer_id"] == "REAL03" and row["view"] == "L"
    assert row["group_id"] == "WORD0001_REAL03" and row["sign_intervals"] == [[1.1, 2.1]]
    assert row["frames"] == 6 and row["continuous"] is False
    assert [s["id"] for s in skipped] == ["NIA_SL_WORD0003_REAL03_F"], "no interval: skipped"
    assert missing == ["WORD0003", "WORD0500"]
    loaded, _ = load_dataset(td / "out" / "manifest.json")
    gap = next(r for r in loaded if r["id"].endswith("0002_REAL03_F"))["sequence"]
    assert not gap.mask[2].any() and gap.mask[1].any(), "a frame without a person stays empty"
    try:
        prepare(td / "morpheme.zip", td / "keypoint.zip", {"a": ["WORD0001"], "b": ["WORD0001"]},
                td / "other")
        raise AssertionError("one word number under two labels must be rejected")
    except ValueError:
        pass
print("[OK] AI Hub words: listed words only, intervals, signer/view/group, empty frames, skips")
