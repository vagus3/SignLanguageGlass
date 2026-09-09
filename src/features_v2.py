# -*- coding: utf-8 -*-
"""
공통 부분집합 2D 특징 (features v2, 레거시 실험 경로)

보유 AI Hub 샘플에는 3D 키포인트도 있습니다. 3D 손 모양 전이 경로는
hand_shape3d.py 및 tools/prepare_aihub3d.py를 사용하세요.

■ 풀려는 문제
    학습 데이터 : AI Hub 제공 키포인트 = OpenPose 형식
    추론 시점   : 안경 카메라 + MediaPipe
    두 토폴로지가 다릅니다. 그대로 섞으면 에러 없이 성능만 무너집니다.

■ 해법
    양쪽에 '똑같은 의미로' 존재하는 점만 씁니다.

        손 21x2점   OpenPose 와 MediaPipe 가 순서까지 완전히 동일
                    (0=손목, 1-4 엄지, 5-8 검지, 9-12 중지, 13-16 약지, 17-20 새끼)
        상체 7점    어깨/팔꿈치/손목 좌우 + 목(어깨 중점)
                    MediaPipe 33점과 OpenPose BODY_25 에서 각각 유도

    얼굴은 버립니다(70 vs 468, 매핑 불가). 비수지신호를 잃지만,
    호환성을 얻는 대가로는 쌉니다. 나중에 일인칭 데이터를 직접 모을 때
    MediaPipe 얼굴 특징을 별도 채널로 추가하면 됩니다.

■ 정규화 (시점 견고성의 핵심)
    A. 손 로컬  : 손목 원점, 손목->중지MCP 거리를 1로, 그 벡터를 +Y 로 회전
                  -> 2D 평면 내 이동·균일 스케일·회전만 제거합니다.
                  정면→일인칭의 3D 시점 회전에는 불변이 아닙니다.
    B. 몸 상대  : 어깨 중점 원점, 어깨 너비를 1로
                  -> 손의 '위치/궤적'. 시점이 바뀌면 달라지는 부분입니다.

    두 구간의 전이 성능은 착용 데이터에서 비교해야 합니다. 배열 슬라이스만으로
    가중치가 분리되는 것은 아니며, 분리 인코더를 학습해야 선택적으로 고정할 수 있습니다.

■ 차원
    손 로컬   21 x 2 x 2손 = 84
    몸 상대   7 x 2         = 14
    플래그                  = 3   (왼손/오른손/몸 유효)
    합계                    = 101
"""
import numpy as np

# ---- 손: 두 형식이 동일한 21점 --------------------------------------
N_HAND = 21
WRIST, MIDDLE_MCP = 0, 9

# ---- 상체 공통 7점의 의미 순서 --------------------------------------
BODY_NAMES = ["neck", "r_shoulder", "l_shoulder",
              "r_elbow", "l_elbow", "r_wrist", "l_wrist"]

# MediaPipe Pose(33) 인덱스
MP_POSE = {"r_shoulder": 12, "l_shoulder": 11,
           "r_elbow": 14, "l_elbow": 13,
           "r_wrist": 16, "l_wrist": 15}       # neck 은 어깨 중점으로 유도

# OpenPose BODY_25 인덱스
OP_POSE = {"neck": 1, "r_shoulder": 2, "l_shoulder": 5,
           "r_elbow": 3, "l_elbow": 6,
           "r_wrist": 4, "l_wrist": 7}

DIM_HAND = N_HAND * 2 * 2       # 84
DIM_BODY = len(BODY_NAMES) * 2  # 14
FEATURE_DIM_V2 = DIM_HAND + DIM_BODY + 3   # 101
_EPS = 1e-6


# ================================================================ 정규화
def canon_hand(pts):
    """
    손 21x2 -> 손목 원점 + 크기 1 + 회전 정렬.

    손목->중지MCP 벡터를 +Y 축에 맞춰 회전시키면, 손목이 어느 방향으로
    영상 평면 안에서 회전한 경우 정렬할 수 있습니다. 화면 밖 방향으로 회전할 때
    생기는 원근·가림·투영 변화는 이 함수로 제거되지 않습니다.
    """
    if pts is None:
        return np.zeros((N_HAND, 2), np.float32), 0.0

    p = np.asarray(pts, np.float32)[:, :2] - np.asarray(pts, np.float32)[WRIST, :2]
    v = p[MIDDLE_MCP]
    scale = float(np.linalg.norm(v))
    if scale < _EPS:
        return np.zeros((N_HAND, 2), np.float32), 0.0
    p = p / scale

    # v 를 +Y(0,1) 로 보내는 회전
    c, s = v[1] / scale, v[0] / scale        # cos, sin
    R = np.array([[c, -s], [s, c]], np.float32)
    return (p @ R.T).astype(np.float32), 1.0


def norm_body(body, valid):
    """상체 7점 -> 어깨 중점 원점, 어깨 너비 1."""
    if not valid:
        return np.zeros((len(BODY_NAMES), 2), np.float32), 0.0
    b = np.asarray(body, np.float32)
    origin = (b[1] + b[2]) / 2.0                       # 어깨 중점
    scale = float(np.linalg.norm(b[1] - b[2]))
    if scale < _EPS:
        return np.zeros((len(BODY_NAMES), 2), np.float32), 0.0
    return ((b - origin) / scale).astype(np.float32), 1.0


def assemble(lh, rh, body, body_valid):
    """정규화된 조각들을 101차원 벡터로."""
    ch_l, f_l = canon_hand(lh)
    ch_r, f_r = canon_hand(rh)
    nb, f_b = norm_body(body, body_valid)
    vec = np.concatenate([ch_l.ravel(), ch_r.ravel(), nb.ravel(),
                          np.array([f_l, f_r, f_b], np.float32)])
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# 특징 구간 인덱스. 선택적 동결에는 별도 모델 분기가 필요합니다.
SLICE_HAND = slice(0, DIM_HAND)
SLICE_BODY = slice(DIM_HAND, DIM_HAND + DIM_BODY)


# ================================================ 입력 A: MediaPipe
def from_mediapipe(results):
    """Holistic 결과 -> 101차원. 추론 경로에서 사용."""
    def hand(lms):
        if lms is None:
            return None
        return np.array([[p.x, p.y] for p in lms.landmark], np.float32)

    lh, rh = hand(results.left_hand_landmarks), hand(results.right_hand_landmarks)

    body, valid = np.zeros((len(BODY_NAMES), 2), np.float32), False
    pl = results.pose_landmarks
    if pl is not None and len(pl.landmark) > 24:
        lm = pl.landmark
        pt = {k: np.array([lm[i].x, lm[i].y], np.float32)
              for k, i in MP_POSE.items()}
        pt["neck"] = (pt["r_shoulder"] + pt["l_shoulder"]) / 2.0   # 목 유도
        body = np.stack([pt[n] for n in BODY_NAMES])
        # 어깨가 둘 다 최소한의 신뢰도로 잡혔을 때만 유효로 봅니다.
        # 안경 하향 카메라에서는 자기 어깨가 안 보이는 경우가 많습니다.
        vis = min(lm[MP_POSE["r_shoulder"]].visibility,
                  lm[MP_POSE["l_shoulder"]].visibility)
        valid = vis > 0.5
    return assemble(lh, rh, body, valid)


# ================================================ 입력 B: AI Hub JSON
def _triplets(flat):
    """OpenPose 의 [x,y,conf, x,y,conf, ...] -> (N,3)."""
    if not flat:
        return None
    a = np.asarray(flat, np.float32)
    if a.size % 3:
        return None
    return a.reshape(-1, 3)


def from_openpose(person, conf_min=0.2):
    """
    AI Hub 키포인트 JSON 의 person 노드 -> 101차원. 학습 경로에서 사용.

    person 예:
      {"pose_keypoints_2d":[...75], "hand_left_keypoints_2d":[...63],
       "hand_right_keypoints_2d":[...63], "face_keypoints_2d":[...210]}

    좌표가 절대 픽셀이어도 상관없습니다. 정규화가 원점·스케일을 없애므로
    MediaPipe 의 0~1 좌표와 같은 공간으로 떨어집니다.
    """
    def hand(key):
        t = _triplets(person.get(key))
        if t is None or t.shape[0] < N_HAND:
            return None
        if np.median(t[:, 2]) < conf_min:      # 신뢰도 낮으면 없는 손 취급
            return None
        return t[:N_HAND, :2]

    lh, rh = hand("hand_left_keypoints_2d"), hand("hand_right_keypoints_2d")

    body, valid = np.zeros((len(BODY_NAMES), 2), np.float32), False
    pose = _triplets(person.get("pose_keypoints_2d"))
    if pose is not None and pose.shape[0] >= 8:
        idx = [OP_POSE[n] for n in BODY_NAMES]
        if max(idx) < pose.shape[0]:
            sel = pose[idx]
            body = sel[:, :2]
            valid = bool(min(sel[1, 2], sel[2, 2]) > conf_min)   # 양 어깨 신뢰도
    return assemble(lh, rh, body, valid)


def person_from_frame(node):
    """키포인트 JSON 한 프레임에서 person dict 를 꺼냅니다(구조 편차 흡수)."""
    if isinstance(node, dict):
        if "people" in node:
            ppl = node["people"]
            if isinstance(ppl, list) and ppl:
                return ppl[0]
            if isinstance(ppl, dict):
                return ppl
        if any(k.endswith("_keypoints_2d") for k in node):
            return node
    return None


# ================================================ 시퀀스 리샘플
def resample(seq, target_len=30):
    """
    가변 길이 시퀀스 -> 고정 길이. 선형 보간으로 균등 리샘플합니다.

    ★ 프레임 수를 그냥 잘라내면 안 됩니다.
      수어는 속도 개인차가 큰데, 앞 30프레임만 자르면 어떤 사람은 동작이
      끝나 있고 어떤 사람은 시작도 안 한 상태가 됩니다.
      전체를 시간축으로 균등하게 늘이거나 줄여야 합니다.
    """
    a = np.asarray(seq, np.float32)
    n = a.shape[0]
    if n == 0:
        return np.zeros((target_len, FEATURE_DIM_V2), np.float32)
    if n == target_len:
        return a
    src = np.linspace(0, n - 1, target_len)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo).astype(np.float32)[:, None]
    return ((1 - w) * a[lo] + w * a[hi]).astype(np.float32)


if __name__ == "__main__":
    # 두 경로가 같은 벡터를 만드는지 자가 검증
    rng = np.random.default_rng(0)
    fake_hand = rng.random((21, 3)).astype(np.float32) * 100
    fake_hand[:, 2] = 0.9
    fake_pose = rng.random((25, 3)).astype(np.float32) * 100
    fake_pose[:, 2] = 0.9

    person = {
        "hand_left_keypoints_2d": fake_hand.ravel().tolist(),
        "hand_right_keypoints_2d": fake_hand.ravel().tolist(),
        "pose_keypoints_2d": fake_pose.ravel().tolist(),
    }
    v = from_openpose(person)
    print(f"OpenPose 경로 : {v.shape}  (기대 {FEATURE_DIM_V2})")
    print(f"  손 구간 노름 : {np.linalg.norm(v[SLICE_HAND]):.3f}")
    print(f"  몸 구간 노름 : {np.linalg.norm(v[SLICE_BODY]):.3f}")
    print(f"  플래그       : {v[-3:]}")

    # 스케일/평행이동 불변성 확인: 좌표를 2배 키우고 옮겨도 같아야 함
    p2 = {k: (np.asarray(t, np.float32).reshape(-1, 3)
              * np.array([2, 2, 1]) + np.array([300, 500, 0])).ravel().tolist()
          for k, t in person.items()}
    v2 = from_openpose(p2)
    print(f"\n2배 확대 + 평행이동 후 최대 오차: {np.abs(v - v2).max():.6f}")
    print("  -> 0 에 가까워야 정규화가 제대로 된 것입니다.")

    print(f"\n리샘플 47프레임 -> {resample(rng.random((47, FEATURE_DIM_V2))).shape}")
