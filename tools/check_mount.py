#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""실제 장착 카메라의 손/얼굴 검출률을 측정합니다.

테스트 동안 실제 시연할 수어를 반복하고 상대방은 대화 거리에 서세요.

    python tools/check_mount.py --duration 30 --preview
    python tools/check_mount.py --sign-camera 0 --face-camera 1 --duration 30
    python tools/check_mount.py --face-camera 1 --json reports/mount.json
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import landmarks as L


def make_report(sign_frames, any_hand, both_hands, face_frames, faces, elapsed):
    def rate(n, d):
        return float(n / d) if d else 0.0

    hand_rate = rate(any_hand, sign_frames)
    face_rate = rate(faces, face_frames)
    return {
        "duration_sec": round(float(elapsed), 3),
        "sign_frames": int(sign_frames),
        "sign_fps": round(rate(sign_frames, elapsed), 2),
        "any_hand_rate": round(hand_rate, 4),
        "both_hands_rate": round(rate(both_hands, sign_frames), 4),
        "face_frames": int(face_frames),
        "face_fps": round(rate(face_frames, elapsed), 2),
        "face_rate": round(face_rate, 4),
        "hand_gate_pass": hand_rate >= 0.90,
        "face_gate_pass": face_rate >= 0.90,
    }


def main():
    ap = argparse.ArgumentParser(description="장착 카메라 손/얼굴 검출률 측정")
    ap.add_argument("--sign-camera", type=int, default=None, metavar="INDEX")
    ap.add_argument("--face-camera", type=int, default=None, metavar="INDEX",
                    help="생략하면 수어 카메라 한 대에서 얼굴도 측정")
    ap.add_argument("--duration", type=float, default=30.0, help="측정 초 (기본 30)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--json", default=None, metavar="PATH", help="결과 저장 경로")
    a = ap.parse_args()
    if a.duration <= 0:
        raise SystemExit("--duration은 0보다 커야 합니다")

    sign_index = C.CAM_INDEX if a.sign_camera is None else a.sign_camera
    if a.face_camera is not None and a.face_camera == sign_index:
        raise SystemExit("수어/얼굴 카메라 번호가 같습니다. 한 대 구성은 --face-camera를 빼세요.")

    import cv2
    import mediapipe as mp
    from src.camera import open_camera

    sign_cap = open_camera(camera_index=sign_index)
    if sign_cap is None:
        raise SystemExit(f"수어 카메라 {sign_index}를 열 수 없습니다")
    face_cap = None
    if a.face_camera is not None:
        face_cap = open_camera(camera_index=a.face_camera)
        if face_cap is None:
            sign_cap.release()
            raise SystemExit(f"얼굴 카메라 {a.face_camera}를 열 수 없습니다")

    holistic = L.make_holistic(C.MODEL_COMPLEXITY)
    detector = (mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) if face_cap else None)
    sign_frames = any_hand = both_hands = face_frames = faces = 0
    t0 = time.perf_counter()
    print(f"{a.duration:g}초 동안 실제 시연 수어를 반복하세요. "
          "상대방은 대화 거리에 서 있어야 합니다.")
    try:
        while time.perf_counter() - t0 < a.duration:
            ok, sign_frame = sign_cap.read()
            if not ok:
                continue
            sh, sw = sign_frame.shape[:2]
            rgb = cv2.cvtColor(sign_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            left = res.left_hand_landmarks is not None
            right = res.right_hand_landmarks is not None
            sign_frames += 1
            any_hand += int(left or right)
            both_hands += int(left and right)

            face_frame = sign_frame
            if face_cap is None:
                face = L.face_center_px(res, sw, sh)
            else:
                ok_face, face_frame = face_cap.read()
                if not ok_face:
                    continue
                fh, fw = face_frame.shape[:2]
                frgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                frgb.flags.writeable = False
                face = L.detection_face_center_px(detector.process(frgb), fw, fh)
            face_frames += 1
            faces += int(face is not None)

            if a.preview:
                L.draw_debug(sign_frame, res)
                cv2.putText(sign_frame, f"hand {any_hand/sign_frames:.0%}", (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("mount: sign camera", sign_frame)
                if face_cap is not None:
                    if face:
                        cv2.drawMarker(face_frame, (int(face[0]), int(face[1])),
                                       (0, 255, 255), cv2.MARKER_CROSS, 32, 2)
                    cv2.imshow("mount: face camera", face_frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        elapsed = time.perf_counter() - t0
        sign_cap.release()
        if face_cap is not None:
            face_cap.release()
        holistic.close()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()

    report = make_report(sign_frames, any_hand, both_hands,
                         face_frames, faces, elapsed)
    print("\n=== 장착 시점 검출률 ===")
    print(f"손 1개 이상 : {report['any_hand_rate']:.1%}  "
          f"({'PASS' if report['hand_gate_pass'] else 'FAIL'}; 기준 90%)")
    print(f"두 손       : {report['both_hands_rate']:.1%}  (참고값)")
    print(f"상대 얼굴   : {report['face_rate']:.1%}  "
          f"({'PASS' if report['face_gate_pass'] else 'FAIL'}; 기준 90%)")
    print(f"처리량      : 손 {report['sign_fps']:.1f}fps / 얼굴 {report['face_fps']:.1f}fps")

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
