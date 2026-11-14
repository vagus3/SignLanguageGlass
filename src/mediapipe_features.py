"""Versioned MediaPipe preprocessing shared by CTC training and inference.

Raw pose recordings stay untouched. Only models explicitly trained with this
feature set receive 30 Hz, filtered, shoulder-relative features.
"""
import numpy as np

FEATURE_SET = "mediapipe_body_v1"
FPS = 30.0
MAX_GAP = 0.2
PIPELINE = {"feature_set": FEATURE_SET, "fps": FPS, "max_gap_seconds": MAX_GAP,
            "position": "shoulder-center-and-width", "filter": "adaptive-lowpass-v1"}


def require_mediapipe(manifest):
    """Do not silently train this pipeline on OpenPose or unknown JSON points."""
    extractor = manifest.get("extractor")
    if not isinstance(extractor, dict) or extractor.get("name") != "mediapipe-holistic":
        raise ValueError(
            "mediapipe_body_v1 requires MediaPipe recordings or video re-extraction. "
            "Use tools/extract_mediapipe.py on the original videos; "
            "OpenPose/SYN JSON coordinates cannot be converted into MediaPipe detections.")


class PoseSmoother:
    """Mask-aware, scale-adaptive low pass; never hold a lost hand as visible.

    Signed derivatives are filtered before their norm, so alternating detector
    jitter does not look like sustained fast motion. Gaps reset filter state.
    """
    def __init__(self):
        self.previous = None

    def update(self, coordinates, mask, timestamp, image_size):
        p = np.asarray(coordinates, np.float64).copy()
        m = np.asarray(mask, bool).copy()
        aspect = image_size[1] / image_size[0]
        p[:, 1] *= aspect
        t = float(timestamp)
        width = np.linalg.norm(p[43] - p[44])
        scale = np.full(len(p), max(width * 0.25, 1e-5)
                        if m[43] and m[44] else 0.05)
        if not (m[43] and m[44]):
            for start in (0, 21):
                palm = np.linalg.norm(p[start + 9] - p[start])
                if m[start] and m[start + 9] and palm > 1e-5:
                    scale[start:start + 21] = palm
        filtered, derivative = p.copy(), np.zeros_like(p)
        if self.previous is not None:
            before, old_raw, old_filtered, old_m, old_derivative, old_size = self.previous
            dt = t - before
            if 0 < dt <= MAX_GAP and tuple(image_size) == old_size:
                valid = m & old_m
                jump = np.linalg.norm(p - old_raw, axis=1) / scale
                for start in (0, 21):
                    common = valid[start:start + 21]
                    hand_jump = jump[start:start + 21]
                    # An isolated multi-palm jump in an otherwise still hand is
                    # missing data. Do not clamp it into a fabricated fingertip.
                    if common.sum() >= 8 and np.median(hand_jump[common]) < 0.25:
                        bad = common & (hand_jump > 3.0) & (dt <= 0.1)
                        m[start:start + 21][bad] = False
                valid = m & old_m
                alpha_d = 2 * np.pi * dt / (1 + 2 * np.pi * dt)
                derivative = (alpha_d * (p - old_raw) / dt / scale[:, None]
                              + (1 - alpha_d) * old_derivative)
                derivative[~valid] = 0
                cutoff = 1.5 + 0.5 * np.linalg.norm(derivative, axis=1)
                alpha = (2 * np.pi * cutoff * dt / (1 + 2 * np.pi * cutoff * dt))[:, None]
                filtered[valid] = (alpha * p + (1 - alpha) * old_filtered)[valid]
        filtered[~m] = 0
        self.previous = (t, p, filtered.copy(), m.copy(), derivative, tuple(image_size))
        filtered[:, 1] /= aspect
        return filtered.astype(np.float32), m.copy()


def prepare(sequence):
    """Resample without bridging missing joints or long capture gaps, then filter."""
    from src.pose_sequence import PoseSequence
    s = sequence.validate()
    ts = s.timestamps - s.timestamps[0]
    grid = np.arange(int(np.floor(ts[-1] * FPS + 1e-7)) + 1) / FPS
    if len(grid) < 2:
        grid = np.array([0.0, ts[-1]])
    hi = np.searchsorted(ts, grid, side="left").clip(1, len(ts) - 1)
    lo = hi - 1
    weight = ((grid - ts[lo]) / (ts[hi] - ts[lo])).clip(0, 1)
    p = (1 - weight[:, None, None]) * s.coordinates[lo] + weight[:, None, None] * s.coordinates[hi]
    m = s.mask[lo] & s.mask[hi] & ((ts[hi] - ts[lo]) <= MAX_GAP + 1e-7)[:, None]
    # Exact source samples remain valid even at an edge of a long gap.
    for indices in (lo, hi):
        exact = np.isclose(grid, ts[indices], atol=1e-7, rtol=0)
        p[exact], m[exact] = s.coordinates[indices[exact]], s.mask[indices[exact]]
    p[~m] = 0
    smoother = PoseSmoother()
    for i, t in enumerate(grid):
        p[i], m[i] = smoother.update(p[i], m[i], t, s.image_size)
    return PoseSequence(grid, p, m, s.image_size).validate()


def features(sequence):
    """331 columns, new semantics; legacy 2d checkpoints must keep legacy features."""
    s = prepare(sequence)
    p = s.coordinates.copy()
    p[:, :, 1] *= s.image_size[1] / s.image_size[0]
    local = []
    for start in (0, 21):
        hand, valid = p[:, start:start + 21], s.mask[:, start:start + 21]
        scale = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)
        good = valid[:, 0] & valid[:, 9] & (scale > 1e-6)
        shape = (hand - hand[:, :1]) / np.maximum(scale[:, None, None], 1e-6)
        shape[~(valid & good[:, None])] = 0
        local.append(np.clip(shape, -10, 10).reshape(len(p), -1))
    center = (p[:, 43] + p[:, 44]) / 2
    width = np.linalg.norm(p[:, 43] - p[:, 44], axis=1)
    placed = s.mask[:, 43] & s.mask[:, 44] & (width > 1e-6)
    valid = s.mask & placed[:, None]
    position = (p - center[:, None]) / np.maximum(width[:, None, None], 1e-6)
    position[~valid] = 0
    dt = np.r_[0, np.diff(s.timestamps)]
    velocity = np.zeros_like(position)
    pair = valid[1:] & valid[:-1]
    velocity[1:] = np.where(pair[..., None], np.diff(position, axis=0)
                           / np.maximum(dt[1:, None, None], 1e-3), 0)
    return np.concatenate([*local, np.clip(position, -10, 10).reshape(len(p), -1),
                           s.mask.astype(np.float32),
                           np.clip(velocity, -10, 10).reshape(len(p), -1),
                           dt[:, None], s.timestamps[:, None]], axis=1).astype(np.float32)
