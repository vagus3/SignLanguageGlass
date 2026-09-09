# -*- coding: utf-8 -*-
"""
HUD 시차(Parallax) 보정 도구

    python -m src.calibrate --display serial

■ 왜 필요한가
    카메라는 콧대 위, 눈은 그 뒤 아래에 있습니다. 두 위치가 다르므로
    "카메라가 본 얼굴 위치"와 "눈이 본 얼굴 위치"가 어긋납니다.
    보정 없이 말풍선을 띄우면 상대방 얼굴에서 한참 벗어난 곳에 뜹니다.

    또 이 값은 안경을 다시 조립할 때마다 바뀌는 '측정치'이므로
    코드(config.py)가 아니라 데이터(models/hud_calib.json)로 저장합니다.

■ 사용법
    1. 안경을 쓰고 상대방(또는 거울 속 자신)을 정면으로 봅니다
    2. 화면에 십자 표식이 뜹니다
    3. 표식이 얼굴 중심에 겹쳐 보일 때까지 방향키로 밀고, [ ] 로 배율 조정
    4. s 를 눌러 저장

    ※ preview 백엔드로도 감은 잡히지만, 진짜 보정은 실제 광학계를 쓴 뒤
      눈으로 보면서 해야 합니다.

키:
    ← → ↑ ↓   오프셋 1px 이동   (macOS/Windows 에서는 안 먹을 수 있습니다)
    w a x d   오프셋 5px 이동   (위/왼쪽/아래/오른쪽 — s 가 아니라 x 입니다)
    [ ]       가로 배율 -/+        - =       세로 배율 -/+
    r         초기화
    s         저장            q   저장 없이 종료
"""
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import landmarks as L
from src.camera import open_camera
import src.hud as hud_mod
from src.one_euro import OneEuroPoint


def main():
    ap = argparse.ArgumentParser(description="HUD 광학 시차 보정")
    ap.add_argument("--display", default=None,
                    choices=["preview", "serial", "pi", "null"],
                    help="반드시 실제 장치에서는 serial 또는 pi를 지정하세요")
    a = ap.parse_args()

    cap = open_camera()
    if cap is None:
        print("카메라를 열 수 없습니다.")
        return

    # hud.load_calib() 는 모듈의 _CALIB 객체 자체를 교체합니다. 따라서
    # ``from src.hud import _CALIB`` 로 가져오면 교체 전 객체를 계속 수정하게
    # 되고, 키를 눌러도 실제 렌더링에는 보정이 반영되지 않습니다.
    hud_mod.load_calib()
    off = list(hud_mod._CALIB["offset"])
    scl = list(hud_mod._CALIB["scale"])

    holistic = L.make_holistic(C.MODEL_COMPLEXITY)
    hud = hud_mod.HudRenderer()
    from src.oled_bridge import make_display
    display = make_display(a.display)
    smooth = OneEuroPoint(C.FILTER_MIN_CUTOFF, C.FILTER_BETA)

    print(__doc__)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            face = L.face_center_px(res, w, h)
            if face:
                face = smooth(face[0], face[1], time.time())
            else:
                smooth.reset()

            # 현재 보정값을 실제 OLED와 프리뷰에 즉시 반영합니다.
            hud_mod._CALIB["offset"], hud_mod._CALIB["scale"] = off, scl
            img = hud.calibration_marker(face_xy=face, cam_size=(w, h))
            display.show(img)

            arr = np.array(img, dtype=np.uint8) * 255
            big = cv2.resize(arr, (C.OLED_W * 6, C.OLED_H * 6),
                             interpolation=cv2.INTER_NEAREST)
            big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
            cv2.putText(big, f"off {off}  scale {scl[0]:.2f},{scl[1]:.2f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("calibrate (s=save, q=quit)", big)

            cam = cv2.flip(frame, 1) if C.MIRROR_PREVIEW else frame
            if face:
                fx = int(w - face[0]) if C.MIRROR_PREVIEW else int(face[0])
                cv2.drawMarker(cam, (fx, int(face[1])), (0, 255, 255),
                               cv2.MARKER_CROSS, 40, 2)
            cv2.imshow("camera", cam)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print("저장하지 않고 종료")
                break
            elif k == ord("s"):
                hud_mod.save_calib(off, scl)
                print(f"저장 완료 -> {C.CALIB_PATH}")
                break
            elif k == ord("a"):
                off[0] -= 5
            elif k == ord("d"):
                off[0] += 5
            elif k == ord("w"):
                off[1] -= 5
            elif k == ord("x"):
                off[1] += 5
            elif k == 81:                 # ←
                off[0] -= 1
            elif k == 83:                 # →
                off[0] += 1
            elif k == 82:                 # ↑
                off[1] -= 1
            elif k == 84:                 # ↓
                off[1] += 1
            elif k == ord("["):
                scl[0] = round(max(0.1, scl[0] - 0.05), 2)
            elif k == ord("]"):
                scl[0] = round(scl[0] + 0.05, 2)
            elif k == ord("-"):
                scl[1] = round(max(0.1, scl[1] - 0.05), 2)
            elif k == ord("="):
                scl[1] = round(scl[1] + 0.05, 2)
            elif k == ord("r"):
                off, scl = [0, 0], [1.0, 1.0]
    finally:
        display.close()
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
