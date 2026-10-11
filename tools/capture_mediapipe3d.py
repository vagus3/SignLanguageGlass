#!/usr/bin/env python3
"""웹캠 MediaPipe Hands world landmark를 AI Hub 공통 462차원으로 수집한다.

스페이스바를 누르면 최근 30개 유효 프레임을 한 시퀀스로 저장하고, q로 끝낸다.
파일명은 ``{signer}__{session}_{번호}.npy``이며 shape은 ``(30, 462)``이다.
"""
import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hand_shape3d import FEATURE_DIM, from_mediapipe_world


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('label', help='수어 라벨/WORD ID')
    ap.add_argument('--signer', required=True, help='촬영자 ID')
    ap.add_argument('--session', default=time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--out', type=Path, default=Path('data/mediapipe3d'))
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--mirrored', action='store_true',
                    help='입력을 좌우 반전해 화면과 특징 모두 거울 기준으로 처리')
    args = ap.parse_args()

    import mediapipe as mp
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f'카메라 {args.camera}를 열 수 없습니다')
    folder = args.out / args.label
    folder.mkdir(parents=True, exist_ok=True)
    existing = list(folder.glob(f'{args.signer}__{args.session}_*.npy'))
    number = len(existing) + 1
    window = deque(maxlen=30)

    try:
        with mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2,
                model_complexity=1, min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('카메라 프레임 읽기 실패')
                if args.mirrored:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                feature = from_mediapipe_world(results,
                                               input_is_mirrored=args.mirrored)
                if feature.shape != (FEATURE_DIM,):
                    raise RuntimeError(f'특징 shape 오류: {feature.shape}')
                # 손이 하나라도 검출된 프레임만 수집한다.
                if np.any(feature):
                    window.append(feature)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS
                ) if results.multi_hand_landmarks else None
                cv2.putText(frame, f'{args.label}  frames {len(window)}/30',
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if len(window) == 30 else (0, 180, 255), 2)
                cv2.putText(frame, 'SPACE save | q quit', (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('MediaPipe 3D capture', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord(' ') and len(window) == 30:
                    path = folder / f'{args.signer}__{args.session}_{number:04d}.npy'
                    np.save(path, np.stack(window).astype(np.float32))
                    print(f'저장: {path}')
                    number += 1
                    window.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
