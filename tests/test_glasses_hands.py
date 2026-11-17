"""Wearer-view hand tracking keeps left/right stable when the per-frame label flips."""
import _path  # noqa
import numpy as np
from src.glasses_hands import HandTracker


def frame(left=None, right=None):
    p, m = np.zeros((49, 2), np.float32), np.zeros(49, bool)
    for start, x, y in ((0, left, 0.5), (21, right, 0.7)):   # crossing hands pass at different heights
        if x is not None:
            p[start:start + 21] = [x, y]
            m[start:start + 21] = True
    return p, m


def side(points, mask):
    return (round(float(points[0, 0]), 2) if mask[0] else None,
            round(float(points[21, 0]), 2) if mask[21] else None)


t = HandTracker()
assert side(*t.update(*frame(0.3, 0.7))) == (0.3, 0.7)
# The detector swaps its labels for a frame: tracking by position undoes it.
p, m = frame(0.31, 0.72)
p[:21], p[21:42] = p[21:42].copy(), p[:21].copy()          # labels swapped by the detector
assert side(*t.update(p, m)) == (0.31, 0.72)
# Only the right hand is seen, but labelled left: it stays the right hand.
p, m = frame(None, 0.73)
p[:21], p[21:42], m[:21], m[21:42] = p[21:42].copy(), p[:21].copy(), m[21:42].copy(), m[:21].copy()
assert side(*t.update(p, m)) == (None, 0.73)
# Hands crossing slowly keep their identity.
t.reset()
xs = np.linspace(0.3, 0.7, 12)
for a, b in zip(xs, xs[::-1]):
    out = side(*t.update(*frame(a, b)))
assert out == (0.7, 0.3), out
# First appearance with labels opposite to the positions: positions win.
t.reset()
assert side(*t.update(*frame(0.8, 0.2))) == (0.2, 0.8)
print("[OK] glasses hand tracking: label flips, one hand, crossing, first appearance")
