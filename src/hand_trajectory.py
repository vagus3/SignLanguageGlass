"""One experimental change to planar hand shape: retain wrist displacement.

First 462 columns are the existing hand descriptor applied to image XY with
explicit z=0. Six extra columns are left/right wrist dx,dy and two validity
flags. This is a 2D representation and does not recover measured depth.
"""
import numpy as np
from src.hand_shape3d import hand_descriptor, resample_descriptors

FEATURE_VERSION = "hand-shape-xy-wrist-v1"
SHAPE_DIM, MOTION_DIM = 462, 6
FEATURE_DIM = SHAPE_DIM + MOTION_DIM


def extract(sequence, length=30):
    """Return (length,468), with displacement in shoulder-width units.

    Track each wrist relative to the shoulder center, then subtract its first
    valid position. Camera translation and a constant uniform scale cancel.
    Missing shoulders/wrists remain missing; no held wrist or invented depth.
    """
    s = sequence.validate()
    p = np.zeros((len(s.timestamps), 49, 3), np.float32)
    p[..., :2] = s.coordinates * s.image_size
    shape = np.stack([np.r_[hand_descriptor(frame[:21], mask[:21]),
                            hand_descriptor(frame[21:42], mask[21:42])]
                      for frame, mask in zip(p, s.mask)])
    shape = resample_descriptors(shape, length=length, frame_positions=s.timestamps)
    center = (p[:, 43, :2] + p[:, 44, :2]) / 2
    widths = np.linalg.norm(p[:, 43, :2] - p[:, 44, :2], axis=1)
    body = s.mask[:, 43] & s.mask[:, 44] & (widths > 1e-6)
    scale = float(np.median(widths[body])) if body.any() else 1.0
    motion = np.zeros((len(p), MOTION_DIM), np.float32)
    for hand, wrist in enumerate((0, 21)):
        valid = body & s.mask[:, wrist]
        if valid.any():
            relative = (p[:, wrist, :2] - center) / scale
            first = np.flatnonzero(valid)[0]
            motion[valid, hand * 2:hand * 2 + 2] = relative[valid] - relative[first]
            motion[:, 4 + hand] = valid
    grid = np.linspace(s.timestamps[0], s.timestamps[-1], length)
    hi = np.searchsorted(s.timestamps, grid).clip(1, len(p) - 1)
    lo = hi - 1
    gap = s.timestamps[hi] - s.timestamps[lo]
    w = ((grid - s.timestamps[lo]) / gap).clip(0, 1)
    near = np.where(w < .5, lo, hi)
    out = motion[lo] * (1 - w[:, None]) + motion[hi] * w[:, None]
    for hand in (0, 1):
        columns = [hand * 2, hand * 2 + 1, hand + 4]
        changed = motion[lo, hand + 4] != motion[hi, hand + 4]
        out[np.ix_(changed, columns)] = motion[near[changed]][:, columns]
        out[:, hand + 4] = motion[near, hand + 4]
        exact = (np.isclose(grid, s.timestamps[lo], atol=1e-8, rtol=0)
                 | np.isclose(grid, s.timestamps[hi], atol=1e-8, rtol=0))
        missing = ((gap > .25) & ~exact) | (out[:, hand + 4] == 0)
        out[np.ix_(missing, columns)] = 0
    return np.c_[shape, out].astype(np.float32)
