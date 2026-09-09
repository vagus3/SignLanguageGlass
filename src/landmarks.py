# -*- coding: utf-8 -*-
"""
Holistic 랜드마크 추출 + 정규화

이 프로젝트에서 가장 중요한 모듈입니다.
흔한 튜토리얼은 1662차원(얼굴 468점 전체)을 그대로 넣는데, 그러면
 - 얼굴 좌표가 전체 차원의 84%를 차지해 손 정보가 묻히고
 - 원본 좌표를 쓰기 때문에 "카메라 앞 어디에 서 있었는지"를 학습해버립니다.

여기서는 246차원으로 줄입니다. 몸은 어깨 기준, 손은 손목/손바닥 기준으로
정규화하므로 착용형 하향 카메라에서 어깨가 잘려도 손 특징을 유지합니다.
"""
import numpy as np

# --- pose: 상체만. 0=코, 11~24=어깨/팔/손목/골반 -------------------
POSE_KEEP = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# --- face: 비수지신호(의문/부정)를 담는 눈썹·눈·입만 ----------------
# 한국수어에서 의문문과 부정은 손이 아니라 표정으로 표현됩니다.
FACE_KEEP = [
    70, 63, 105, 66, 107,          # 왼쪽 눈썹
    336, 296, 334, 293, 300,       # 오른쪽 눈썹
    33, 133, 159, 145,             # 왼쪽 눈
    362, 263, 386, 374,            # 오른쪽 눈
    61, 291, 13, 14, 78, 308,      # 입
]
FACE_EYE_L, FACE_EYE_R = 33, 263   # 정규화 기준 (양쪽 눈꼬리)

N_POSE, N_HAND, N_FACE = len(POSE_KEEP), 21, len(FACE_KEEP)
# 벡터 맨 뒤 3칸은 [왼손, 오른손, 얼굴] 검출 여부(0.0/1.0)입니다.
# 좌표가 아니라 '있었나 없었나'를 뜻하므로 보간·증강 대상에서 빼야 합니다.
N_FLAGS = 3
FEATURE_DIM = N_POSE * 3 + N_HAND * 3 * 2 + N_FACE * 3 + N_FLAGS   # 45+126+72+3 = 246
FEATURE_VERSION = "wearable-v3"

L_SH, R_SH = 11, 12

# KEEP 리스트로 이미 추린 배열 안에서의 위치.
# 랜드마크를 통째로 ndarray 로 옮긴 뒤 고르면 얼굴 468점을 매 프레임 파이썬
# 루프로 훑게 됩니다(30fps 면 초당 4만 번). 필요한 점만 뽑아서 옮기고,
# 정규화 기준점도 그 안에서 찾습니다.
POSE_I_L_SH = POSE_KEEP.index(L_SH)
POSE_I_R_SH = POSE_KEEP.index(R_SH)
FACE_I_EYE_L = FACE_KEEP.index(FACE_EYE_L)
FACE_I_EYE_R = FACE_KEEP.index(FACE_EYE_R)

_EPS = 1e-6


def make_holistic(model_complexity=0):
    """Holistic 인스턴스 생성. mediapipe 버전 문제를 여기서 한 번에 잡습니다."""
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "이 mediapipe 버전에는 solutions 모듈이 없습니다.\n"
            "  pip install 'mediapipe==0.10.21'\n"
            "(0.10.31 부터 레거시 솔루션이 삭제되었습니다)"
        )
    return mp.solutions.holistic.Holistic(
        model_complexity=model_complexity,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _to_array(landmark_list, keep=None):
    """
    protobuf 랜드마크 -> (N,3) ndarray. 없거나 점이 모자라면 None.

    keep 을 주면 그 점만 옮깁니다. 점이 모자라 IndexError 가 나면 None 을
    돌려주므로, 호출부에서 따로 길이를 검사할 필요가 없습니다.
    """
    if landmark_list is None:
        return None
    lm = landmark_list.landmark
    idx = keep if keep is not None else range(len(lm))
    try:
        return np.array([[lm[i].x, lm[i].y, lm[i].z] for i in idx], dtype=np.float32)
    except IndexError:
        return None


def _normalize_hand(hand):
    """손목 위치 + 손목 기준 손 모양으로 바꿉니다.

    0번 슬롯에는 카메라 중심 기준 손목 x/y와 손바닥 크기를 넣고, 1~20번은
    손목을 원점·손목~중지 MCP(9번)를 길이 1로 둡니다. 어깨가 없어도 동작하며
    손 위치와 카메라 거리 정보도 완전히 버리지 않습니다.
    """
    wrist = hand[0].copy()
    scale = float(np.linalg.norm(hand[9, :2] - wrist[:2]))
    scale = scale if scale > _EPS else 1.0
    out = (hand - wrist) / scale
    out[0] = (wrist[0] - 0.5, wrist[1] - 0.5, scale)
    return out.astype(np.float32)


def extract(results):
    """
    Holistic 결과 -> (246,) float32 특징 벡터

    정규화 방식
      body       : 어깨 중점을 원점, 어깨 너비를 1로
      hands      : 손목을 원점, 손바닥 길이를 1로. 0번 슬롯은 손목 위치/크기
      face       : 얼굴 중심을 원점, 눈 사이 거리를 1로 -> 표정만 남김
                   (얼굴을 어깨 스케일로 나누면 표정 변화가 뭉개집니다)
    """
    pose = _to_array(results.pose_landmarks, POSE_KEEP)
    out_pose = np.zeros((N_POSE, 3), np.float32)
    out_lh = np.zeros((N_HAND, 3), np.float32)
    out_rh = np.zeros((N_HAND, 3), np.float32)
    out_face = np.zeros((N_FACE, 3), np.float32)
    flags = np.zeros(3, np.float32)   # [lh, rh, face] 존재 여부

    if pose is not None:
        l_sh, r_sh = pose[POSE_I_L_SH], pose[POSE_I_R_SH]
        origin = (l_sh + r_sh) / 2.0
        scale = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
        scale = scale if scale > _EPS else 1.0

        out_pose = (pose - origin) / scale

    # 손은 pose 블록 밖에서 처리해야 합니다. 안경의 하향 시점에서는 어깨가
    # 잘리는 경우가 많고, 그때 손까지 0으로 만들면 인식기가 아무것도 못 봅니다.
    lh = _to_array(results.left_hand_landmarks)
    if lh is not None:
        out_lh = _normalize_hand(lh)
        flags[0] = 1.0
    rh = _to_array(results.right_hand_landmarks)
    if rh is not None:
        out_rh = _normalize_hand(rh)
        flags[1] = 1.0

    f = _to_array(results.face_landmarks, FACE_KEEP)
    if f is not None:
        f_origin = f.mean(axis=0)
        f_scale = float(np.linalg.norm(
            f[FACE_I_EYE_L][:2] - f[FACE_I_EYE_R][:2]))
        f_scale = f_scale if f_scale > _EPS else 1.0
        out_face = (f - f_origin) / f_scale
        flags[2] = 1.0

    vec = np.concatenate([
        out_pose.ravel(), out_lh.ravel(), out_rh.ravel(), out_face.ravel(), flags
    ]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def resample_timed(features, timestamps, target_len, start=None, end=None):
    """실제 캡처 시각을 기준으로 특징 시퀀스를 고정 길이로 리샘플합니다.

    카메라 설정이 30fps여도 Holistic 처리량이 15fps라면 30개 프레임을 모으는
    데 2초가 걸립니다. 그 데이터를 1초 동작이라고 저장하면 학습과 실시간
    추론의 시간축이 달라집니다. 수집기는 이 함수로 정확히 같은 시간 구간을
    ``target_len``개로 바꿉니다.

    마지막 ``N_FLAGS``개 검출 플래그는 연속값이 아니므로 보간하지 않고 가장
    가까운 프레임의 0/1 값을 유지합니다.
    """
    x = np.asarray(features, np.float32)
    ts = np.asarray(timestamps, np.float64)
    if x.ndim != 2 or len(x) != len(ts):
        raise ValueError("features는 (frames, dim), timestamps는 같은 길이여야 합니다")
    if len(x) < 2:
        raise ValueError("시간 리샘플에는 프레임이 2개 이상 필요합니다")
    if target_len < 2:
        raise ValueError("target_len은 2 이상이어야 합니다")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps는 엄격히 증가해야 합니다")

    t0 = ts[0] if start is None else float(start)
    t1 = ts[-1] if end is None else float(end)
    if t1 <= t0:
        raise ValueError("end는 start보다 커야 합니다")

    grid = np.linspace(t0, t1, target_len)
    hi = np.searchsorted(ts, grid, side="left").clip(1, len(ts) - 1)
    lo = hi - 1
    w = ((grid - ts[lo]) / np.maximum(ts[hi] - ts[lo], _EPS)).clip(0.0, 1.0)
    out = ((1.0 - w[:, None]).astype(np.float32) * x[lo]
           + w[:, None].astype(np.float32) * x[hi])

    near = np.where(w < 0.5, lo, hi)
    out[:, -N_FLAGS:] = x[near][:, -N_FLAGS:]
    return out.astype(np.float32)


def hands_visible(results):
    """수집기에서 '손이 안 잡히는 구간'을 걸러내기 위한 헬퍼."""
    return (results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None)


def face_center_px(results, w, h):
    """
    Phase 2 말풍선 위치용. 얼굴 중심 픽셀 좌표. 못 찾으면 None.

    468점을 두 번(x 한 번, y 한 번) 훑으면 30fps 에서 초당 2만8천 번
    속성 접근입니다. 한 번에 옮겨서 평균만 냅니다.
    """
    f = results.face_landmarks
    if f is None:
        return None
    lm = f.landmark
    n = len(lm)
    if n == 0:
        return None
    sx = sy = 0.0
    for p in lm:                       # numpy 로 옮기는 비용이 평균값보다 큽니다
        sx += p.x
        sy += p.y
    return (sx / n * w, sy / n * h)


def detection_face_center_px(results, w, h):
    """MediaPipe FaceDetection 결과에서 가장 확실한 얼굴 중심을 반환합니다."""
    detections = getattr(results, "detections", None) or []
    if not detections:
        return None
    det = max(detections,
              key=lambda d: float(d.score[0]) if getattr(d, "score", None) else 0.0)
    box = det.location_data.relative_bounding_box
    x = (float(box.xmin) + float(box.width) * 0.5) * w
    y = (float(box.ymin) + float(box.height) * 0.5) * h
    return (max(0.0, min(x, float(max(w - 1, 0)))),
            max(0.0, min(y, float(max(h - 1, 0)))))


def draw_debug(image, results):
    """수집/디버그용 오버레이."""
    import mediapipe as mp
    mpd, mph = mp.solutions.drawing_utils, mp.solutions.holistic
    mpd.draw_landmarks(image, results.pose_landmarks, mph.POSE_CONNECTIONS)
    mpd.draw_landmarks(image, results.left_hand_landmarks, mph.HAND_CONNECTIONS)
    mpd.draw_landmarks(image, results.right_hand_landmarks, mph.HAND_CONNECTIONS)
    return image
