"""Team glasses recordings mixed with the synthetic AI Hub view for fine-tuning."""
import _path  # noqa
import json
import tempfile
from pathlib import Path
import numpy as np
from src.pose_sequence import JOINT_NAMES, VERSION, PoseSequence
from src.team_data import MixedComposer, TeamComposer, load_team

rng = np.random.default_rng(0)


def clip(n=40):
    t = np.arange(n) / 30
    p = np.full((n, 49, 2), 0.5, np.float32) + rng.normal(0, .01, (n, 49, 2)).astype(np.float32)
    m = np.zeros((n, 49), bool)
    m[:, :42] = True
    p[~m] = 0
    return PoseSequence(t, p, m, (1280, 720))


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    clips = []
    for signer in ("a", "b"):
        for glosses in (["119"], ["부르다"], [], ["없는말"]):
            name = f"{signer}_{'_'.join(glosses) or 'none'}.npz"
            clip().save(root / name)
            clips.append({"id": name, "file": name, "signer_id": signer, "glosses": glosses})
    (root / "manifest.json").write_text(json.dumps(
        {"version": VERSION, "joint_names": JOINT_NAMES,
         "extractor": {"name": "mediapipe-hands"}, "clips": clips}, ensure_ascii=False))
    vocab = {"words": [{"gloss": "119"}, {"gloss": "부르다"}]}
    rows = load_team([root / "manifest.json"], vocab)
    assert len(rows) == 6, "words outside the vocabulary are left out"
    (root / "other.json").write_text(json.dumps({"extractor": {"name": "openpose-json"},
                                                 "clips": []}))
    try:
        load_team([root / "other.json"])
        raise AssertionError("non-glasses data was accepted")
    except ValueError:
        pass

phrases = [{"glosses": ["119", "부르다"], "text": "119를 불러 주세요."},
           {"glosses": ["119"], "text": "119요."}]
team = TeamComposer(rows, phrases, phrase_rate=0.5)
seen = {}
for _ in range(300):
    item = team.sample(rng)
    seen[tuple(item["glosses"])] = item["text"]
    s = item["sequence"]
    assert (s.coordinates[~s.mask] == 0).all() and len(s.timestamps) >= 2
assert seen[("119", "부르다")] == "119를 불러 주세요.", "same-person words joined into a phrase"
assert seen[("119",)] == "119요." and seen[("부르다",)] == "부르다." and seen[()] == "(수어 아님)"
assert all(team.sample(rng, "b") for _ in range(5))


class Fake:
    signers = ["REAL01"]

    def sample(self, rng, signer=None):
        return {"glosses": ["x"], "text": "x", "sequence": None, "from": "aihub"}


mixed = MixedComposer(team, Fake(), team_rate=0.3)
share = np.mean([mixed.sample(rng).get("from") != "aihub" for _ in range(2000)])
assert 0.25 < share < 0.35, share
assert mixed.sample(rng, "a")["sequence"] is not None, "a team signer samples team data"
print("[OK] team data: vocabulary filter, glasses-only, targets, joined phrases, mixing")
