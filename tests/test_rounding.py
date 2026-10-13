"""Exact float32 encoding is lossless; rounding is opt-in, measured and never on timestamps."""
import _path  # noqa: F401
import gzip
import json
import numpy as np
from src.cloud_sign import CloudClient, parse_rounding
from src.pose_sequence import PoseSequence, _exact32
from tools.evaluate_rounding import evaluate, table

rng = np.random.default_rng(11)
values = np.concatenate([rng.uniform(0, 1, 50000), rng.uniform(0, 1e-6, 500), rng.normal(0, 4, 50000),
                         [0, -0.0, 1, 1e-38, 3.4e38, -1e4, 1e4]]).astype(np.float32)
back = np.asarray(json.loads(json.dumps(_exact32(values).tolist())), np.float32)
assert np.array_equal(back.view(np.uint32), values.view(np.uint32)), "not bit-exact"

n = 60
s3 = PoseSequence(np.arange(n) / 30 + 1e-7, rng.uniform(.1, .9, (n, 49, 2)), np.ones((n, 49), bool),
                  (1280, 720), world=rng.normal(0, .4, (n, 49, 3)), world_mask=np.ones((n, 49), bool))
exact = s3.to_dict()
again = PoseSequence.from_dict(json.loads(json.dumps(exact)))
assert np.array_equal(again.coordinates, s3.coordinates) and np.array_equal(again.world, s3.world)
assert np.array_equal(again.timestamps, s3.timestamps)
old = json.dumps({**exact, "coordinates": s3.coordinates.tolist(), "world": s3.world.tolist()},
                 separators=(",", ":")).encode()
new = json.dumps(exact, separators=(",", ":")).encode()
assert len(gzip.compress(new, 1)) < 0.75 * len(gzip.compress(old, 1)), "exact encoding should be smaller"

rounded = s3.to_dict(4, 3)
assert rounded["timestamps"] == exact["timestamps"], "timestamps are never rounded"
assert all(abs(v * 1e4 - round(v * 1e4)) < 1e-6 for v in np.ravel(rounded["coordinates"])[:500])
assert all(abs(v * 1e3 - round(v * 1e3)) < 1e-6 for v in np.ravel(rounded["world"])[:500])
assert s3.to_dict(4)["world"] == exact["world"], "3D stays exact unless asked"

assert parse_rounding("") == (None, None) and parse_rounding("4") == (4, None)
assert parse_rounding("4,3") == (4, 3)
for bad in ("0", "4,3,2", "x", "10"):
    try:
        parse_rounding(bad)
        raise AssertionError(bad)
    except ValueError:
        pass
plain, lossy = CloudClient(rounding=""), CloudClient(rounding="4,3")
assert plain.payload(s3, "s", "a")["coordinates"] == exact["coordinates"]
assert lossy.payload(s3, "s", "a")["coordinates"] == rounded["coordinates"]


class ScoreFromPoints:
    """Score = mean x of the first hand, so rounding can move it across the threshold."""
    feature_set, calibration = "2d", {"threshold": 0.5}

    def predict(self, sequence):
        return ["A"], float(sequence.coordinates[:, :21, 0].mean())


s2 = PoseSequence(np.arange(n) / 30, np.full((n, 49, 2), .49996), np.ones((n, 49), bool),
                  (1280, 720))
report = evaluate([("near", s2), ("other", PoseSequence(np.arange(n) / 30,
                   rng.uniform(.1, .9, (n, 49, 2)), np.ones((n, 49), bool), (1280, 720)))],
                  two_d=(4, 3), recognizer=ScoreFromPoints())
settings = report["settings"]
assert list(settings) == ["exact", "2d=4", "2d=3"], list(settings)
assert settings["exact"]["feature_max"] == 0 and settings["exact"]["threshold_flips"] == 0
assert abs(settings["exact"]["gzip_vs_plain"] - 1) < 1e-9
assert settings["2d=4"]["threshold_flips"] == 1, "0.49996 rounds to 0.5000"
assert settings["2d=3"]["gzip_kb_mean"] < settings["exact"]["gzip_kb_mean"]
assert "threshold flips" in table(report)
print("[OK] rounding: exact float32 encoding bit-exact and smaller, opt-in 2D/3D rounding, "
      "timestamps kept, evaluation reports size, drift and decision flips")
