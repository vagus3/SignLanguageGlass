"""MediaPipe preprocessing invariance, timing, loss handling and runtime dispatch."""
import _path  # noqa: F401
import numpy as np
from src.ctc_runtime import OnnxRecognizer, check_contract
from src.mediapipe_features import FEATURE_SET, PoseSmoother, prepare, require_mediapipe
from src.pose_sequence import PoseSequence, FEATURE_SETS, VERSION, features
from src.segmenter import ActivitySegmenter


def sequence(fps=30):
    ts = np.arange(fps + 1) / fps
    p = np.zeros((len(ts), 49, 2), np.float32)
    for start, x in ((0, .42), (21, .58)):
        p[:, start:start + 21, 0] = x + np.linspace(0, .03, 21)
        p[:, start:start + 21, 1] = .6 - np.linspace(0, .12, 21)
        p[:, start:start + 21, 0] += ts[:, None] * .03
    p[:, 42:] = [[.5, .35], [.35, .35], [.65, .35],
                 [.3, .5], [.7, .5], [.4, .6], [.6, .6]]
    return PoseSequence(ts, p, np.ones(p.shape[:2], bool), (1280, 720)).validate()


extract = FEATURE_SETS[FEATURE_SET][0]
s = sequence()
baseline = extract(s)
assert baseline.shape == (31, 331) and baseline.dtype == np.float32
assert np.isfinite(baseline).all()
check_contract(FEATURE_SET, VERSION, 331, ["<blank>", "A"])
assert not np.allclose(baseline[:, 84:182], features(s)[:, 84:182])
for scale, shift in ((1.0, [.08, -.05]), (.65, [.1, .1])):
    moved = PoseSequence(s.timestamps, (s.coordinates - .5) * scale + .5 + shift,
                         s.mask.copy(), s.image_size)
    assert np.allclose(extract(moved), baseline, atol=2e-4), "Camera placement must normalize"
assert np.allclose(extract(sequence(10)), baseline, atol=2e-4), "Same trajectory, same 30Hz input"

# Do not fill missing joints with stale observations, or bridge capture outages.
missing = sequence()
missing.mask[10:15, :21] = False
x = extract(missing)
assert not x[10:15, :42].any() and not x[10:15, 182:203].any()
assert not x[15, 231:273].any(), "Reacquisition must not invent a velocity"
gapped = sequence()
indices = np.r_[0:4, 15:31]
gapped = PoseSequence(gapped.timestamps[indices], gapped.coordinates[indices],
                      gapped.mask[indices], gapped.image_size)
resampled = prepare(gapped)
assert not resampled.mask[4:15].any(), "Long gaps must remain absent"
assert np.isfinite(extract(gapped)).all()
no_shoulders = sequence()
no_shoulders.mask[:, 43:45] = False
x = extract(no_shoulders)
assert not x[:, 84:182].any() and not x[:, 231:329].any()
assert x[:, :84].any(), "Hand shape must survive shoulders leaving the frame"
short = PoseSequence([0, .01], s.coordinates[:2], s.mask[:2], s.image_size)
assert extract(short).shape == (2, 331)

smoother = PoseSmoother()
smoother.update(s.coordinates[0], s.mask[0], 0, s.image_size)
lost = s.mask[0].copy()
lost[:21] = False
points, mask = smoother.update(s.coordinates[1], lost, .033, s.image_size)
assert not points[:21].any() and not mask[:21].any()
points, _ = smoother.update(s.coordinates[-1], s.mask[-1], 1, s.image_size)
assert np.allclose(points, s.coordinates[-1]), "Long gaps reset stale filter state"
spike_filter = PoseSmoother()
spike_filter.update(s.coordinates[0], s.mask[0], 0, s.image_size)
spiked = s.coordinates[0].copy()
spiked[8, 0] += .3
points, mask = spike_filter.update(spiked, s.mask[0], 1 / 30, s.image_size)
assert not mask[8] and not points[8].any(), "An isolated large jump must be marked missing"
assert s.mask[0].all(), "Filtering must not mutate the raw recording's mask"

# Regression: +/-3.2px alternating detection noise previously started recording.
segmenter, recording, events = ActivitySegmenter(), False, []
for i in range(120):
    p, m = s.coordinates[0].copy(), s.mask[0].copy()
    p[:21, 0] += .0025 * (-1 if i % 2 else 1)
    event = segmenter.update(p, m, i / 30, recording, s.image_size)
    if event:
        events.append(event)
        recording = event == "start"
assert not events, events

for name in (None, "openpose-json", "sen-xy-json"):
    try:
        require_mediapipe({"extractor": {"name": name}})
        raise AssertionError(f"Accepted non-MediaPipe source: {name}")
    except ValueError:
        pass
require_mediapipe({"extractor": {"name": "mediapipe-holistic"}})

# Both training's FEATURE_SETS and the torch-free ONNX runtime select this exact pipeline.
seen = []
recognizer = OnnxRecognizer.__new__(OnnxRecognizer)
recognizer.feature_set, recognizer.labels = FEATURE_SET, ["<blank>", "A"]
def log_probs(array):
    seen.append(array.copy())
    return np.tile(np.log([.99, .01]), (len(array), 1))
recognizer.log_probs = log_probs
assert recognizer.predict(s) == ([], 0.0)
assert np.array_equal(seen[0], baseline)
print("[OK] MediaPipe body normalization, 30Hz, gaps, masks, jitter and ONNX dispatch")
