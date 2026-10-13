"""Anatomical correspondence, aspect ratio, timing and mask contract tests."""
import _path  # noqa
from types import SimpleNamespace as NS
import tempfile
from pathlib import Path
import numpy as np
from src.pose_sequence import (PoseSequence, features, from_openpose, from_mediapipe,
                                FEATURE_DIM, VERSION)
from src.joints import MP_POSE, OP_POSE, BODY_NAMES

rng = np.random.default_rng(8)
hands = [rng.uniform(.1, .8, (21, 2)) for _ in range(2)]
body = rng.uniform(.1, .8, (7, 2))
pose = [NS(x=0, y=0, visibility=0) for _ in range(33)]
op = np.zeros((25, 3))
for i, name in enumerate(BODY_NAMES[1:], 1):
    x, y = body[i]
    pose[MP_POSE[name]] = NS(x=x, y=y, visibility=1)
    op[OP_POSE[name]] = [x*640, y*480, 1]
mp = NS(left_hand_landmarks=NS(landmark=[NS(x=x, y=y) for x, y in hands[0]]),
        right_hand_landmarks=NS(landmark=[NS(x=x, y=y) for x, y in hands[1]]),
        pose_landmarks=NS(landmark=pose))
person = {"pose_keypoints_2d": op.ravel().tolist()}
for side, hand in zip(["left", "right"], hands):
    person[f"hand_{side}_keypoints_2d"] = np.c_[hand*[640,480], np.ones(21)].ravel().tolist()
a, am = from_mediapipe(mp)
b, bm = from_openpose(person, (640,480))
assert np.allclose(a,b) and np.array_equal(am,bm)
mirrored = NS(**vars(mp))
mirrored.left_hand_landmarks = NS(landmark=[NS(x=1-x,y=y) for x,y in hands[0]])
mirrored.right_hand_landmarks = NS(landmark=[NS(x=1-x,y=y) for x,y in hands[1]])
mirrored.pose_landmarks = NS(landmark=[NS(x=1-p.x,y=p.y,visibility=p.visibility) for p in pose])
assert np.allclose(from_mediapipe(mirrored,input_is_mirrored=True)[0], a)

s = PoseSequence([0,.1,.4], np.stack([a,a+.01,a+.02]), np.stack([am,am,am]), (640,480))
x = features(s)
assert x.shape == (3, FEATURE_DIM) and np.isfinite(x).all()
assert np.allclose(x[2,231:329],0), "A long capture gap must not invent velocity"
s.mask[1,0] = False
assert np.allclose(features(s)[1,:42],0), "Missing wrist must invalidate local hand shape"
with tempfile.TemporaryDirectory() as td:
    path = Path(td)/"clip.npz"
    s.save(path)
    assert np.array_equal(PoseSequence.load(path).mask,s.mask)
    assert PoseSequence.load(path).version == VERSION
for change in ({"timestamps":[0,0,.4]}, {"version":"other"}, {"mask":np.full((3,49),.5)},
               {"coordinates":np.full((3,49,2),np.nan)}, {"image_size":(0,480)}):
    data = dict(timestamps=s.timestamps,coordinates=s.coordinates,mask=s.mask,image_size=s.image_size)
    data.update(change)
    try:
        PoseSequence(**data).validate()
        raise AssertionError(f"Bad input passed: {change.keys()}")
    except ValueError:
        pass
print("[OK] shared 2D adapters, mirroring, missingness, time gaps and versioned storage")
