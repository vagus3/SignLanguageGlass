import numpy as np

class _P:
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, x, y, z, v=1.0):
        self.x, self.y, self.z, self.visibility = float(x), float(y), float(z), float(v)

class _LM:
    def __init__(self, arr):
        self.landmark = [_P(*row) for row in arr]

class Res:
    def __init__(self, pose=None, lh=None, rh=None, face=None):
        self.pose_landmarks = _LM(pose) if pose is not None else None
        self.left_hand_landmarks = _LM(lh) if lh is not None else None
        self.right_hand_landmarks = _LM(rh) if rh is not None else None
        self.face_landmarks = _LM(face) if face is not None else None

def make(seed, with_pose=True, with_lh=True, with_rh=True, with_face=True,
         n_pose=33, n_face=468):
    r = np.random.default_rng(seed)
    return Res(
        pose=r.random((n_pose, 3)) if with_pose else None,
        lh=r.random((21, 3)) if with_lh else None,
        rh=r.random((21, 3)) if with_rh else None,
        face=r.random((n_face, 3)) if with_face else None,
    )
