"""안경 시점용 입력 특징: 손 위치·속도를 카메라 설정에 덜 의존하게 바꿉니다 (ego_relative_v1).

기본 특징(src.pose_sequence.features)은 손목의 화면 절대 위치와 화면 비율 속도를 그대로 넣습니다.
같은 동작도 카메라 기울기·시야각·팔 뻗은 거리에 따라 숫자가 달라집니다. 여기서는 손마다
- 위치: (좌표 - 이 구간의 손목 중앙 위치) / 이 구간의 손바닥 길이 중앙값
- 속도: 속도 / 이 구간의 손바닥 길이 중앙값
으로 바꿉니다. 손 모양(84), 보임 표시(49), 시간(2)은 그대로이고 열 배치도 같아서 331차원입니다.
구간 단위로 계산하므로 수어 한 구간을 자른 뒤 인식하는 흐름(QwenSignClient)에서 그대로 씁니다.
"""
import numpy as np


FEATURE_SET = "ego_relative_v1"
FALLBACK_SCALE = 0.1


def features_relative(sequence):
    from src.pose_sequence import FEATURE_BLOCKS, JOINTS, features   # 순환 참조를 피해 늦게 불러옴
    s = sequence.validate()
    out = features(sequence).copy()
    p = s.coordinates.copy()
    p[:, :, 1] *= s.image_size[1] / s.image_size[0]
    position = np.zeros((len(p), JOINTS, 2), np.float32)
    velocity = out[:, FEATURE_BLOCKS["velocity"]].reshape(len(p), JOINTS, 2)
    new_velocity = np.zeros_like(velocity)
    for start in (0, 21):
        hand, valid = p[:, start:start + 21], s.mask[:, start:start + 21]
        wrist = valid[:, 0]
        palm_ok = wrist & valid[:, 9]
        if not wrist.any():
            continue
        palm = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)[palm_ok]
        scale = float(np.median(palm)) if len(palm) and np.median(palm) > 1e-6 else FALLBACK_SCALE
        ref = np.median(hand[wrist, 0], axis=0)
        rel = np.where(valid[..., None], (hand - ref) / scale, 0)
        position[:, start:start + 21] = np.clip(rel, -10, 10)
        new_velocity[:, start:start + 21] = np.clip(velocity[:, start:start + 21] / scale, -10, 10)
    out[:, FEATURE_BLOCKS["position"]] = position.reshape(len(p), -1)
    out[:, FEATURE_BLOCKS["velocity"]] = new_velocity.reshape(len(p), -1)
    return out.astype(np.float32)
