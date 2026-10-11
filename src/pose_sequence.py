"""Versioned, image-based 2D pose sequences shared by AI Hub and Holistic.

Coordinates are x/width, y/height in the original unmirrored image. The
image size preserves aspect ratio when computing distances. Faces never enter
this contract: an egocentric camera may see the interlocutor's face.

v2 adds optional 3D `world` points in a camera-aligned frame (x right, y down,
z away from the camera). Units and origin may differ between hands and body;
features_3d never subtracts a hand point from a body point.
"""
from dataclasses import dataclass
import numpy as np
from src.features_v2 import BODY_NAMES, MP_POSE, OP_POSE

VERSION = "pose2d-sequence-v1"
VERSION_3D = "pose3d-sequence-v2"
VERSIONS = (VERSION, VERSION_3D)
JOINT_NAMES = ([f"left_hand_{i}" for i in range(21)]
               + [f"right_hand_{i}" for i in range(21)] + BODY_NAMES)
JOINTS = len(JOINT_NAMES)
FEATURE_DIM = 84 + 98 + 49 + 98 + 2
# Per hand: palm normal, finger direction, wrist position, wrist velocity, 2 flags.
WORLD_DIM = 2 * (3 + 3 + 3 + 3 + 2)
FEATURE_DIM_3D = FEATURE_DIM + WORLD_DIM
MAX_FRAMES = 900
MAX_SECONDS = 15.0


@dataclass
class PoseSequence:
    timestamps: np.ndarray
    coordinates: np.ndarray
    mask: np.ndarray
    image_size: tuple
    version: str = VERSION
    world: np.ndarray = None
    world_mask: np.ndarray = None

    def validate(self):
        if self.world is not None and self.version == VERSION:
            self.version = VERSION_3D
        if self.version not in VERSIONS:
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
        if self.version == VERSION_3D:
            if self.world is None or self.world_mask is None:
                raise ValueError("3D sequences need world and world_mask")
            w = np.asarray(self.world, dtype=np.float32)
            raw_world_mask = np.asarray(self.world_mask)
            if w.shape != (len(t), JOINTS, 3) or raw_world_mask.shape != (len(t), JOINTS):
                raise ValueError("Expected world (T,49,3), world_mask (T,49)")
            if not np.isfinite(w).all() or np.abs(w).max(initial=0) > 1e4:
                raise ValueError("World coordinates must be finite and bounded")
            if not np.isin(raw_world_mask, [0, 1]).all():
                raise ValueError("World mask must contain boolean/0/1 values")
            wm = raw_world_mask.astype(bool)
            self.world, self.world_mask = np.where(wm[..., None], w, 0), wm
        elif self.world_mask is not None:
            raise ValueError("world_mask without world")
        return self

    def to_dict(self):
        self.validate()
        data = {"version": self.version, "timestamps": self.timestamps.tolist(),
                "coordinates": self.coordinates.tolist(), "mask": self.mask.tolist(),
                "image_size": list(self.image_size)}
        if self.version == VERSION_3D:
            data.update(world=self.world.tolist(), world_mask=self.world_mask.tolist())
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(data["timestamps"], data["coordinates"], data["mask"],
                   data["image_size"], data["version"],
                   data.get("world"), data.get("world_mask")).validate()

    def save(self, path):
        self.validate()
        extra = ({"world": self.world, "world_mask": self.world_mask}
                 if self.version == VERSION_3D else {})
        np.savez_compressed(path, timestamps=self.timestamps, coordinates=self.coordinates,
                            mask=self.mask, image_size=self.image_size, version=self.version,
                            **extra)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            world = (d["world"], d["world_mask"]) if "world" in d.files else (None, None)
            return cls(d["timestamps"], d["coordinates"], d["mask"],
                       tuple(d["image_size"].tolist()), str(d["version"]), *world).validate()


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


def _clean_world(points, mask, input_is_mirrored):
    mask = np.asarray(mask, bool) & np.isfinite(points).all(axis=1)
    if input_is_mirrored:
        points = points * [-1, 1, 1]
    return np.where(mask[:, None], points, 0).astype(np.float32), mask


def world_from_mediapipe(results, image_size, *, input_is_mirrored=False):
    """Approximate 3D for orientation and depth, not a metric hand reconstruction.

    Hands: Holistic image landmarks as (x*w, y*h, z*w); MediaPipe puts z on
    roughly the x scale, so palm orientation is approximate. Body:
    pose_world_landmarks (meters, hip origin), so depth is relative to the body.
    """
    w, h = image_size
    points, mask = np.zeros((JOINTS, 3), np.float32), np.zeros(JOINTS, bool)
    for start, name in [(0, "left_hand_landmarks"), (21, "right_hand_landmarks")]:
        hand = getattr(results, name, None)
        if hand is not None and len(hand.landmark) == 21:
            points[start:start+21] = [[p.x * w, p.y * h, p.z * w] for p in hand.landmark]
            mask[start:start+21] = True
    pose = getattr(results, "pose_world_landmarks", None)
    if pose is not None and len(pose.landmark) > max(MP_POSE.values()):
        for i, name in enumerate(BODY_NAMES[1:], 43):
            p = pose.landmark[MP_POSE[name]]
            points[i], mask[i] = [p.x, p.y, p.z], getattr(p, "visibility", 0) > 0.5
        points[42], mask[42] = (points[43] + points[44]) / 2, mask[43] and mask[44]
    return _clean_world(points, mask, input_is_mirrored)


def world_from_openpose(person, *, input_is_mirrored=False, world_to_camera=None):
    """Read 3D points; optionally apply an explicit rigid world-to-camera matrix.

    Without a matrix the caller must supply camera-frame points (x right, y
    down, z away). Dataset adapters must verify that convention, not infer it
    from the presence of OpenPose fields. Units are preserved.
    """
    points, mask = np.zeros((JOINTS, 3), np.float32), np.zeros(JOINTS, bool)
    for start, name in [(0, "hand_left_keypoints_3d"), (21, "hand_right_keypoints_3d")]:
        flat = person.get(name, [])
        if len(flat) == 84:
            a = np.asarray(flat, np.float32).reshape(21, 4)
            points[start:start+21], mask[start:start+21] = a[:, :3], a[:, 3] > 0
    flat = person.get("pose_keypoints_3d", [])
    if len(flat) == 100:
        a = np.asarray(flat, np.float32).reshape(25, 4)
        for i, name in enumerate(BODY_NAMES[1:], 43):
            p = a[OP_POSE[name]]
            points[i], mask[i] = p[:3], p[3] > 0
        points[42], mask[42] = (points[43] + points[44]) / 2, mask[43] and mask[44]
    if world_to_camera is not None:
        matrix = np.asarray(world_to_camera, dtype=np.float64)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            raise ValueError("world_to_camera must be a finite 3x4 [R|t] matrix")
        rotation = matrix[:, :3]
        if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
            raise ValueError("world_to_camera rotation must be orthonormal with determinant +1")
        points = points @ rotation.T + matrix[:, 3]
    return _clean_world(points, mask, input_is_mirrored)


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


def _unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-9), norm[..., 0] > 1e-9


def world_features(sequence):
    """Palm orientation from hand joints only; wrist position from body joints only."""
    s = sequence.validate()
    if s.version != VERSION_3D:
        raise ValueError("3D features need a pose3d sequence with world coordinates")
    w, m = s.world.astype(np.float64), s.world_mask
    center = (w[:, 43] + w[:, 44]) / 2
    width = np.linalg.norm(w[:, 43] - w[:, 44], axis=1)
    body_ok = m[:, 43] & m[:, 44] & (width > 1e-6)
    dt = np.r_[0, np.diff(s.timestamps)]
    out = []
    # The same cross product faces opposite palm sides on mirrored hands.
    for start, wrist, sign in [(0, 48, 1.0), (21, 47, -1.0)]:
        hand, valid = w[:, start:start+21], m[:, start:start+21]
        normal, normal_ok = _unit(sign * np.cross(hand[:, 5] - hand[:, 0],
                                                  hand[:, 17] - hand[:, 0]))
        direction, direction_ok = _unit(hand[:, 9] - hand[:, 0])
        oriented = valid[:, [0, 5, 9, 17]].all(axis=1) & normal_ok & direction_ok
        placed = body_ok & m[:, wrist]
        position = (w[:, wrist] - center) / np.maximum(width, 1e-6)[:, None]
        velocity = np.zeros_like(position)
        pair = placed[1:] & placed[:-1] & (dt[1:] <= 0.25)
        velocity[1:] = np.where(pair[:, None], np.diff(position, axis=0)
                                / np.maximum(dt[1:, None], 1e-3), 0)
        out += [np.where(oriented[:, None], normal, 0), np.where(oriented[:, None], direction, 0),
                np.clip(np.where(placed[:, None], position, 0), -10, 10),
                np.clip(velocity, -10, 10), oriented[:, None], placed[:, None]]
    return np.concatenate(out, axis=1).astype(np.float32)


def features_3d(sequence):
    return np.concatenate([features(sequence), world_features(sequence)], axis=1)


# Column groups of features_3d, so experiments can remove one cue at a time.
FEATURE_BLOCKS = {"hand_shape": np.r_[0:84], "position": np.r_[84:182],
                  "mask": np.r_[182:231], "velocity": np.r_[231:329], "timing": np.r_[329:331]}
WORLD_BLOCKS = {"palm_orientation": [0, 1, 2, 3, 4, 5, 12], "wrist_3d": [6, 7, 8, 9, 10, 11, 13]}
for _name, _columns in WORLD_BLOCKS.items():
    FEATURE_BLOCKS[_name] = np.array([FEATURE_DIM + 14 * hand + c
                                      for hand in (0, 1) for c in _columns])
_2D = ["hand_shape", "position", "mask", "velocity", "timing"]
FEATURE_PRESETS = {
    "2d": _2D, "3d": _2D + list(WORLD_BLOCKS),
    "2d_no_hand_shape": [b for b in _2D if b != "hand_shape"],
    "2d_no_position": [b for b in _2D if b != "position"],
    "2d_no_velocity": [b for b in _2D if b != "velocity"],
    "3d_orientation": _2D + ["palm_orientation"], "3d_wrist": _2D + ["wrist_3d"],
}


def needs_world(feature_set):
    return any(block in WORLD_BLOCKS for block in FEATURE_PRESETS[feature_set])


def _selector(blocks):
    columns = np.sort(np.concatenate([FEATURE_BLOCKS[b] for b in blocks]))
    extract = features_3d if any(b in WORLD_BLOCKS for b in blocks) else features
    return (lambda sequence: extract(sequence)[:, columns]), len(columns)


FEATURE_SETS = {name: _selector(blocks) for name, blocks in FEATURE_PRESETS.items()}
