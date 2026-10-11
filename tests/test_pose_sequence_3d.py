"""3D palm orientation, body-relative wrist depth, v1/v2 storage and checkpoint feature sets."""
import _path  # noqa
import sys
try:
    import torch
except ImportError:
    sys.exit(77)
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from src.pose_sequence import (PoseSequence, VERSION, VERSION_3D, FEATURE_DIM, FEATURE_DIM_3D,
                                WORLD_DIM, features_3d, world_features, world_from_openpose)
from src.features_v2 import OP_POSE, BODY_NAMES
from src.continuous_sign import CtcRecognizer, Recognizer, save_checkpoint
from src.cloud_sign import SegmentRecorder

torch.set_num_threads(1)
rng = np.random.default_rng(3)
# Left hand, palm toward the camera (-z), fingers up (-y), thumb toward image right.
left = rng.normal(0, .01, (21, 3))
left[0], left[5], left[9], left[17] = [0, 0, 0], [.03, -.08, 0], [0, -.09, 0], [-.03, -.07, 0]
right = left * [-1, 1, 1]  # Mirror image: the right hand in the same pose.
frame = np.zeros((49, 3))
frame[:21], frame[21:42] = left + [.1, .1, 2], right + [-.1, .1, 2]
frame[43], frame[44] = [-.2, 0, 2], [.2, 0, 2]          # r/l shoulder
frame[47], frame[48] = [-.1, .1, 1.8], [.1, .1, 1.9]     # r/l wrist (body model)
frame[42] = (frame[43] + frame[44]) / 2


def sequence(world, n=4, world_mask=None):
    w = np.stack([world + [.001 * i, 0, 0] for i in range(n)])
    m = np.ones((n, 49), bool) if world_mask is None else world_mask
    return PoseSequence(np.arange(n) / 30, rng.uniform(.1, .8, (n, 49, 2)),
                        np.ones((n, 49), bool), (640, 480), world=w, world_mask=m)


s = sequence(frame)
assert s.validate().version == VERSION_3D
x = world_features(s)
assert x.shape == (4, WORLD_DIM) and features_3d(s).shape == (4, FEATURE_DIM_3D)
left_normal, right_normal = x[0, 0:3], x[0, 14:17]
assert left_normal[2] < -.99 and right_normal[2] < -.99, "Both palms face the camera"
assert np.allclose(x[0, 3:6], [0, -1, 0], atol=1e-6), "Fingers point up"
assert np.allclose(x[0, 6:9], [.25, .25, -.25]), "Left wrist is in front of the shoulders"
assert np.allclose(x[0, 12:14], 1) and np.allclose(x[0, 26:28], 1)

# Same pose further away and larger: rotation-free features must not change.
assert np.allclose(world_features(sequence(frame * 2.5 + [.3, -.1, 1])), x, atol=1e-5)
angle = np.deg2rad(40)
R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
turned = world_features(sequence(frame @ R.T))
assert np.allclose(turned[0, 0:3], R @ left_normal, atol=1e-5), "Normal rotates with the hand"
assert np.allclose(turned[0, 6:9], R @ x[0, 6:9], atol=1e-5)

hidden = np.ones((4, 49), bool)
hidden[:, 5] = False      # left index MCP
hidden[:, 44] = False     # left shoulder
y = world_features(sequence(frame, world_mask=hidden))
assert np.allclose(y[:, :3], 0) and np.allclose(y[:, 12], 0), "Missing MCP invalidates orientation"
assert np.allclose(y[:, 6:12], 0) and np.allclose(y[:, 13], 0), "Missing shoulder invalidates depth"
assert np.allclose(y[:, 14:17], x[:, 14:17]), "Other hand orientation is unaffected"

plain = PoseSequence(np.arange(4) / 30, s.coordinates, s.mask, (640, 480)).validate()
assert plain.version == VERSION
try:
    world_features(plain)
    raise AssertionError("2D sequence produced 3D features")
except ValueError:
    pass
with tempfile.TemporaryDirectory() as td:
    for item in (s, plain):
        path = Path(td) / f"{item.version}.npz"
        item.save(path)
        loaded = PoseSequence.load(path)
        assert loaded.version == item.version
        assert item.world is None or np.allclose(loaded.world, item.world)
assert np.allclose(PoseSequence.from_dict(json.loads(json.dumps(s.to_dict()))).world, s.world)
for bad in ({"version": VERSION_3D}, {"world_mask": s.world_mask},
            {"world": s.world[:, :, :2], "world_mask": s.world_mask},
            {"world": np.full_like(s.world, np.nan), "world_mask": s.world_mask}):
    try:
        PoseSequence(s.timestamps, s.coordinates, s.mask, s.image_size, **bad).validate()
        raise AssertionError(f"Bad 3D input passed: {list(bad)}")
    except ValueError:
        pass

# AI Hub OpenPose 3D: body joints map through OP_POSE, mirroring flips x.
pose3d = np.zeros((25, 4))
for name in BODY_NAMES[1:]:
    pose3d[OP_POSE[name]] = [*frame[BODY_NAMES.index(name) + 42], 1]
person = {"pose_keypoints_3d": pose3d.ravel().tolist(),
          "hand_left_keypoints_3d": np.c_[frame[:21], np.ones(21)].ravel().tolist(),
          "hand_right_keypoints_3d": np.c_[frame[21:42], np.ones(21)].ravel().tolist()}
world, world_mask = world_from_openpose(person)
assert world_mask.all() and np.allclose(world, frame, atol=1e-6)
assert np.allclose(world_from_openpose(person, input_is_mirrored=True)[0], frame * [-1, 1, 1],
                   atol=1e-6)
assert not world_from_openpose({})[1].any()

# Transform common-world coordinates before mirror correction; preserve masks.
transform = np.c_[R, [.3, -.2, .5]]
converted, converted_mask = world_from_openpose(person, world_to_camera=transform)
expected = frame @ R.T + transform[:, 3]
assert converted_mask.all() and np.allclose(converted, expected, atol=1e-6)
mirrored, _ = world_from_openpose(person, world_to_camera=transform, input_is_mirrored=True)
assert np.allclose(mirrored, expected * [-1, 1, 1], atol=1e-6)
empty, empty_mask = world_from_openpose({}, world_to_camera=transform)
assert not empty_mask.any() and not empty.any(), "Translation must not populate missing joints"
for bad in (np.eye(4), np.zeros((3, 4)), np.c_[np.diag([-1, 1, 1]), np.zeros(3)],
            np.full((3, 4), np.nan)):
    try:
        world_from_openpose(person, world_to_camera=bad)
        raise AssertionError("Invalid extrinsic matrix accepted")
    except ValueError:
        pass

# Checkpoints remember their feature set; a 3D model refuses 2D input.
with tempfile.TemporaryDirectory() as td:
    model = CtcRecognizer(3, hidden=8, feature_set="3d")
    save_checkpoint(Path(td) / "ctc3d.pt", model, ["<blank>", "A", "B"])
    recognizer = Recognizer(Path(td) / "ctc3d.pt")
    assert recognizer.feature_set == "3d"
    recognizer.predict(s)
    try:
        recognizer.predict(plain)
        raise AssertionError("3D checkpoint accepted 2D input")
    except ValueError:
        pass
    legacy = torch.load(Path(td) / "ctc3d.pt", weights_only=True)
    legacy_model = CtcRecognizer(3, hidden=8)
    legacy.update(version=VERSION, feature_dim=FEATURE_DIM, state_dict=legacy_model.state_dict())
    del legacy["feature_set"]
    torch.save(legacy, Path(td) / "legacy.pt")
    assert Recognizer(Path(td) / "legacy.pt").feature_set == "2d"
    legacy.update(version=VERSION_3D)
    torch.save(legacy, Path(td) / "mismatch.pt")
    try:
        Recognizer(Path(td) / "mismatch.pt")
        raise AssertionError("Mismatched checkpoint contract loaded")
    except ValueError:
        pass

recorder = SegmentRecorder()
recorder.start()
for i in range(3):
    recorder.add(s.coordinates[0], s.mask[0], time.monotonic(), (640, 480),
                 s.world[0], s.world_mask[0])
recorded, _ = recorder.finish()
assert recorded.version == VERSION_3D and recorded.world.shape == (3, 49, 3)
recorder.start()
recorder.add(s.coordinates[0], s.mask[0], time.monotonic(), (640, 480), s.world[0], s.world_mask[0])
recorder.add(s.coordinates[0], s.mask[0], time.monotonic(), (640, 480))
assert recorder.error == "camera_discontinuity", "Mixed 2D/3D frames must not be combined"

try:
    from src.sign_server import TranslationRequest, TranslationService, MockRecognizer, \
        MockTranslator, MAX_BODY_BYTES
except ImportError:
    print("[OK] 3D orientation/depth features, storage, checkpoints (server skipped)")
    sys.exit(0)
payload = {**s.to_dict(), "session_id": "a", "segment_id": "b"}
request = TranslationRequest(**payload)
assert request.sequence().version == VERSION_3D
result = TranslationService(MockRecognizer(), MockTranslator(), {"안녕하세요": "안녕하세요"},
                            mock=True).process(request)
assert result["version"] == VERSION_3D, "Server must echo the 3D contract version"
try:
    TranslationRequest(**{**payload, "version": VERSION})
    raise AssertionError("v1 label with world coordinates passed")
except ValueError:
    pass
n = 900
longest = PoseSequence(np.arange(n) / 60, rng.uniform(0, 1, (n, 49, 2)), np.ones((n, 49), bool),
                       (1920, 1080), world=rng.uniform(-1000, 1000, (n, 49, 3)),
                       world_mask=np.ones((n, 49), bool))
size = len(json.dumps({**longest.to_dict(), "session_id": "a" * 128, "segment_id": "b" * 128}))
assert size < MAX_BODY_BYTES, f"Longest 3D segment ({size} bytes) exceeds the server limit"
print("[OK] 3D orientation/depth features, storage, checkpoints, recorder and server contract")
