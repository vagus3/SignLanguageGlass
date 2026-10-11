"""Versioned, image-based 2D pose sequences shared by AI Hub and Holistic.

Coordinates are x/width, y/height in the original unmirrored image. The
image size preserves aspect ratio when computing distances. Faces never enter
this contract: an egocentric camera may see the interlocutor's face.
"""
from dataclasses import dataclass
import numpy as np
from src.features_v2 import BODY_NAMES, MP_POSE, OP_POSE

VERSION = "pose2d-sequence-v1"
JOINT_NAMES = ([f"left_hand_{i}" for i in range(21)]
               + [f"right_hand_{i}" for i in range(21)] + BODY_NAMES)
JOINTS = len(JOINT_NAMES)
FEATURE_DIM = 84 + 98 + 49 + 98 + 2
MAX_FRAMES = 900
MAX_SECONDS = 15.0


@dataclass
class PoseSequence:
    timestamps: np.ndarray
    coordinates: np.ndarray
    mask: np.ndarray
    image_size: tuple
    version: str = VERSION

    def validate(self):
        if self.version != VERSION:
            raise ValueError("Unsupported pose version")
        t = np.asarray(self.timestamps, dtype=np.float64)
        p = np.asarray(self.coordinates, dtype=np.float32)
        raw_mask = np.asarray(self.mask)
        if t.ndim != 1 or not 2 <= len(t) <= MAX_FRAMES:
            raise ValueError("A sequence needs 2..900 timestamps")
        if p.shape != (len(t), JOINTS, 2) or raw_mask.shape != (len(t), JOINTS):
            raise ValueError("Expected coordinates (T,49,2), mask (T,49)")
        if not np.isfinite(t).all() or not np.isfinite(p).all():
            raise ValueError("Non-finite pose values")
        if np.any(np.diff(t) <= 0) or t[-1] - t[0] > MAX_SECONDS + 1e-6:
            raise ValueError("Timestamps must increase over at most 15 seconds")
        if not np.isin(raw_mask, [0, 1]).all():
            raise ValueError("Mask must contain boolean/0/1 values")
        if len(self.image_size) != 2 or any(
                not isinstance(v, (int, np.integer)) or isinstance(v, bool)
                or not 1 <= v <= 16384 for v in self.image_size):
            raise ValueError("image_size must be [width,height] positive integers")
        m = raw_mask.astype(bool)
        if np.any((p[m] < 0) | (p[m] > 1)):
            raise ValueError("Valid coordinates must be normalized to [0,1]")
        self.timestamps, self.coordinates, self.mask = t, np.where(m[..., None], p, 0), m
        self.image_size = tuple(int(v) for v in self.image_size)
        return self

    def to_dict(self):
        self.validate()
        return {"version": self.version, "timestamps": self.timestamps.tolist(),
                "coordinates": self.coordinates.tolist(), "mask": self.mask.tolist(),
                "image_size": list(self.image_size)}

    @classmethod
    def from_dict(cls, data):
        return cls(data["timestamps"], data["coordinates"], data["mask"],
                   data["image_size"], data["version"]).validate()

    def save(self, path):
        self.validate()
        np.savez_compressed(path, timestamps=self.timestamps, coordinates=self.coordinates,
                            mask=self.mask, image_size=self.image_size, version=self.version)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(d["timestamps"], d["coordinates"], d["mask"],
                       tuple(d["image_size"].tolist()), str(d["version"])).validate()


def _clean_frame(points, mask, input_is_mirrored=False):
    points = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(points).all(axis=1)
    mask &= ((points >= 0) & (points <= 1)).all(axis=1)
    if input_is_mirrored:
        points = points.copy()
        points[:, 0] = 1 - points[:, 0]
    return np.where(mask[:, None], points, 0).astype(np.float32), mask


def from_mediapipe(results, *, input_is_mirrored=False):
    """Holistic's anatomical left/right; mirrored image coordinates are undone."""
    points, mask = np.zeros((JOINTS, 2), np.float32), np.zeros(JOINTS, bool)
    for start, name in [(0, "left_hand_landmarks"), (21, "right_hand_landmarks")]:
        hand = getattr(results, name, None)
        if hand is not None and len(hand.landmark) == 21:
            points[start:start+21] = [[p.x, p.y] for p in hand.landmark]
            mask[start:start+21] = True
    pose = getattr(results, "pose_landmarks", None)
    if pose is not None and len(pose.landmark) > max(MP_POSE.values()):
        for i, name in enumerate(BODY_NAMES[1:], 43):
            p = pose.landmark[MP_POSE[name]]
            points[i], mask[i] = [p.x, p.y], getattr(p, "visibility", 0) > 0.5
        points[42] = (points[43] + points[44]) / 2
        mask[42] = mask[43] and mask[44]
    return _clean_frame(points, mask, input_is_mirrored)


def from_openpose(person, image_size, *, input_is_mirrored=False, conf_min=0.2):
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError("An explicit image size is required")
    points, mask = np.zeros((JOINTS, 2), np.float32), np.zeros(JOINTS, bool)
    for start, name in [(0, "hand_left_keypoints_2d"), (21, "hand_right_keypoints_2d")]:
        flat = person.get(name, [])
        if len(flat) == 63:
            a = np.asarray(flat, np.float32).reshape(21, 3)
            points[start:start+21], mask[start:start+21] = a[:, :2] / [w, h], a[:, 2] >= conf_min
    flat = person.get("pose_keypoints_2d", [])
    if len(flat) == 75:
        a = np.asarray(flat, np.float32).reshape(25, 3)
        for i, name in enumerate(BODY_NAMES[1:], 43):
            p = a[OP_POSE[name]]
            points[i], mask[i] = p[:2] / [w, h], p[2] >= conf_min
        points[42], mask[42] = (points[43] + points[44]) / 2, mask[43] and mask[44]
    return _clean_frame(points, mask, input_is_mirrored)


def features(sequence):
    """Preserve orientation, trajectories, missingness and irregular frame timing."""
    s = sequence.validate()
    p = s.coordinates.copy()
    p[:, :, 1] *= s.image_size[1] / s.image_size[0]
    local = []
    for start in [0, 21]:
        hand, valid = p[:, start:start+21], s.mask[:, start:start+21]
        scale = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)
        good = valid[:, 0] & valid[:, 9] & (scale > 1e-6)
        x = (hand - hand[:, :1]) / np.maximum(scale[:, None, None], 1e-6)
        x[~(valid & good[:, None])] = 0
        local.append(np.clip(x, -10, 10).reshape(len(p), -1))
    dt = np.r_[0, np.diff(s.timestamps)]
    velocity = np.zeros_like(p)
    pair_valid = s.mask[1:] & s.mask[:-1] & (dt[1:, None] <= 0.25)
    velocity[1:] = np.where(pair_valid[..., None],
                            np.diff(p, axis=0) / np.maximum(dt[1:, None, None], 1e-3), 0)
    elapsed = s.timestamps - s.timestamps[0]
    return np.concatenate([*local, p.reshape(len(p), -1), s.mask.astype(np.float32),
                           np.clip(velocity, -10, 10).reshape(len(p), -1),
                           dt[:, None], elapsed[:, None]], axis=1).astype(np.float32)
