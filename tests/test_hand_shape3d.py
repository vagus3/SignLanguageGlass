"""시점 불변성, 결측 처리, MP/AI Hub 좌표 계약 검증."""
import _path  # noqa: F401
from types import SimpleNamespace as NS
import numpy as np
from src.hand_shape3d import (hand_descriptor, from_openpose, from_mediapipe_world,
                             resample_descriptors, MASK_INDICES)

rng = np.random.default_rng(8)
p = rng.normal(size=(21, 3))
rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
if np.linalg.det(rotation) < 0:
    rotation[:, 0] *= -1
q = (p @ rotation) * 2.7 + [10, -2, 4]
np.testing.assert_allclose(hand_descriptor(p), hand_descriptor(q), atol=2e-6)

bad = p.copy(); bad[7] = np.nan
masked = hand_descriptor(bad)
assert np.isfinite(masked).all() and masked[210 + 7] == 0
bad[0] = np.nan
assert not hand_descriptor(bad).any()
person = {'hand_left_keypoints_3d': np.c_[p, np.ones(21)].ravel().tolist(),
          'hand_right_keypoints_3d': np.zeros(84).tolist()}
op = from_openpose(person)
world = NS(landmark=[NS(x=x,y=y,z=z) for x,y,z in p])
mp = NS(multi_hand_world_landmarks=[world], multi_handedness=[
    NS(classification=[NS(label='Right',score=0.95)])])
np.testing.assert_array_equal(op, from_mediapipe_world(mp, input_is_mirrored=False))
seq = resample_descriptors([op, np.zeros(462)], 30)
assert set(np.unique(seq[:, MASK_INDICES])) <= {0,1}
assert np.isfinite(seq).all()
print('3D 회전/이동/크기 불변 + 결측 + MediaPipe world adapter 확인')
