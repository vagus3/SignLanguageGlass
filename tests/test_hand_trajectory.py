"""Wrist motion survives shape normalization; camera transforms and loss are safe."""
import _path  # noqa: F401
import numpy as np
from src.hand_trajectory import extract, FEATURE_DIM
from src.pose_sequence import PoseSequence


def sample(moving=False, fps=30):
    ts = np.arange(fps + 1) / fps
    p = np.zeros((len(ts), 49, 2), np.float32)
    for start, x in ((0, .4), (21, .6)):
        p[:, start:start + 21] = [x, .6] + np.c_[np.linspace(0, .03, 21),
                                                               -np.linspace(0, .1, 21)]
    if moving:
        p[:, :21, 0] += ts[:, None] * .06
    p[:, 42:] = [[.5, .35], [.35, .35], [.65, .35], [.3, .5], [.7, .5], [.4, .6], [.6, .6]]
    return PoseSequence(ts, p, np.ones(p.shape[:2], bool), (1280, 720))


still, moving = extract(sample()), extract(sample(True))
assert moving.shape == (30, FEATURE_DIM) and moving.dtype == np.float32
assert np.allclose(still[:, :462], moving[:, :462], atol=1e-5)
assert np.allclose(still[:, 462:466], 0)
assert np.isclose(moving[-1, 462], .2, atol=1e-6), "0.06 movement / 0.30 shoulder width"
assert np.all(moving[:, 466:] == 1)
assert np.allclose(extract(sample(True, 10)), moving, atol=1e-5)
for scale, shift in ((1, [.03, -.03]), (.7, [.05, .05])):
    s = sample(True)
    s.coordinates = (s.coordinates - .5) * scale + .5 + shift
    assert np.allclose(extract(s), moving, atol=1e-5)
# A camera pan moves shoulders and hands together; it must not become wrist motion.
s = sample()
s.coordinates[:, :, 0] += s.timestamps[:, None] * .03
assert np.allclose(extract(s)[:, 462:466], 0, atol=1e-6)
s = sample(True)
s.mask[:, 43] = False
assert not extract(s)[:, 462:].any(), "Missing body reference is not a stationary valid hand"
s = sample(True)
s.mask[8:20, :21] = False
x = extract(s)
missing = x[:, 466] == 0
assert missing.any() and not x[missing, 462:464].any()
assert np.isin(x[:, 466:], [0, 1]).all() and np.isfinite(x).all()
s = sample(True)
indices = np.r_[0:4, 18:31]
gapped = PoseSequence(s.timestamps[indices], s.coordinates[indices], s.mask[indices], s.image_size)
x = extract(gapped)
assert not x[5:15, 462:].any(), "Do not interpolate a wrist across a capture outage"
print("[OK] 468-D wrist trajectory: motion, scale, translation, camera pan, FPS and missingness")
