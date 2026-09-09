"""AI Hub 3D와 MediaPipe Hands world 좌표가 공유하는 손 모양 특징.

한 손: 관절 쌍 거리 210개 / 손바닥 길이 + 관절 유효 마스크 21개.
양손: 462차원. 평행 이동, 3D 강체 회전, 균일 확대/축소에 불변.
손바닥 방향, 몸 기준 위치, 양손 간 거리, 좌우 반사 구분은 사라집니다.
완전한 수어 특징이 아닌 사전학습용 손 모양 분기입니다.
"""
import numpy as np

FEATURE_VERSION = "hand-shape3d-v1"
PAIR_I, PAIR_J = np.triu_indices(21, 1)
HAND_DIM = 210 + 21
FEATURE_DIM = HAND_DIM * 2
MASK_INDICES = np.r_[210:231, 441:462]


def hand_descriptor(points, valid=None):
    if points is None:
        return np.zeros(HAND_DIM, np.float32)
    p = np.asarray(points, np.float64)
    if p.shape != (21, 3):
        raise ValueError(f"3D 손 좌표는 (21,3)이어야 합니다: {p.shape}")
    mask = np.isfinite(p).all(axis=1)
    if valid is not None:
        v = np.asarray(valid, bool)
        if v.shape != (21,):
            raise ValueError("관절 마스크는 (21,)이어야 합니다")
        mask &= v
    # 0=손목, 9=중지 MCP. 정규화 기준점이 없으면 손 전체 무효.
    scale = np.linalg.norm(p[9] - p[0]) if mask[0] and mask[9] else 0.0
    if scale <= 1e-8:
        return np.zeros(HAND_DIM, np.float32)
    clean = np.where(mask[:, None], p, 0.0)
    distances = np.linalg.norm(clean[PAIR_I] - clean[PAIR_J], axis=1) / scale
    distances[~(mask[PAIR_I] & mask[PAIR_J])] = 0
    return np.r_[distances, mask].astype(np.float32)


def raw_openpose_hands(person):
    """실제 *_keypoints_3d 필드만 읽음. 2D confidence를 z로 쓰지 않음."""
    hands = []
    for side in ("left", "right"):
        values = person.get(f"hand_{side}_keypoints_3d", [])
        if len(values) != 84:
            raise ValueError(f"{side}: 21개의 (x,y,z,c) 좌표가 필요합니다")
        hands.append(np.asarray(values, np.float64).reshape(21, 4))
    return np.stack(hands)


def from_raw_hands(hands):
    hands = np.asarray(hands)
    if hands.shape != (2, 21, 4):
        raise ValueError(f"양손 좌표는 (2,21,4)이어야 합니다: {hands.shape}")
    return np.concatenate([hand_descriptor(h[:, :3], h[:, 3] > 0) for h in hands])


def from_openpose(person):
    return from_raw_hands(raw_openpose_hands(person))


def from_mediapipe_world(results, *, input_is_mirrored):
    """mp.solutions.hands.Hands의 world 결과용. Holistic 결과와 구별합니다.

    MediaPipe handedness는 거울 입력을 가정하므로 원본 영상이면 좌우 교환.
    정규화 영상 좌표의 x/y/z를 world 좌표 대신 넣어서는 안 됩니다.
    """
    worlds = getattr(results, "multi_hand_world_landmarks", None) or []
    labels = getattr(results, "multi_handedness", None) or []
    chosen = {}
    for world, label in zip(worlds, labels):
        classification = label.classification[0]
        side = classification.label.lower()
        if side not in ("left", "right"):
            continue
        if not input_is_mirrored:
            side = "right" if side == "left" else "left"
        score = float(classification.score)
        points = np.array([[p.x, p.y, p.z] for p in world.landmark])
        if side not in chosen or score > chosen[side][0]:
            chosen[side] = (score, points)
    return np.concatenate([hand_descriptor(chosen[s][1] if s in chosen else None)
                           for s in ("left", "right")])


def resample_descriptors(seq, length=30, frame_positions=None):
    seq = np.asarray(seq, np.float32)
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM or len(seq) < 1 or length < 2:
        raise ValueError("비어 있지 않은 (frames,462) 시퀀스와 length>=2가 필요합니다")
    positions = (np.arange(len(seq), dtype=float) if frame_positions is None
                 else np.asarray(frame_positions, dtype=float))
    if positions.shape != (len(seq),) or not np.isfinite(positions).all() or np.any(np.diff(positions) <= 0):
        raise ValueError("프레임 위치는 시퀀스 길이와 같고 엄격히 증가해야 합니다")
    if len(seq) == 1:
        return np.repeat(seq, length, axis=0)
    grid = np.linspace(positions[0], positions[-1], length)
    hi = np.searchsorted(positions, grid, side='right').clip(1, len(seq)-1)
    lo = hi - 1
    weight = (grid - positions[lo]) / (positions[hi] - positions[lo])
    out = seq[lo] * (1 - weight[:, None]) + seq[hi] * weight[:, None]
    near = np.where(weight < 0.5, lo, hi)
    # 결측을 가로질러 가짜 관절 모양을 보간하지 않음.
    changed = (seq[lo][:, MASK_INDICES] != seq[hi][:, MASK_INDICES]).any(axis=1)
    out[changed] = seq[near[changed]]
    out[:, MASK_INDICES] = seq[near][:, MASK_INDICES]
    return out.astype(np.float32)
