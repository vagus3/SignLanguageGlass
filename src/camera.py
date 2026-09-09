# -*- coding: utf-8 -*-
"""
카메라 공용 모듈

collect.py 와 main.py 가 서로 다른 카메라 설정을 쓰면, 수집할 때와 실행할 때
영상 특성이 달라져서 모델 정확도가 떨어집니다. 반드시 같은 경로로 여세요.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def open_camera(verbose=True, camera_index=None):
    """설정대로 카메라를 엽니다. 실패하면 None."""
    import cv2

    index = C.CAM_INDEX if camera_index is None else int(camera_index)
    cap = cv2.VideoCapture(index, C.camera_backend())
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
    cap.set(cv2.CAP_PROP_FPS, C.CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 지연 누적 방지

    if C.CAM_MANUAL_EXPOSURE:
        _apply_manual_exposure(cap, cv2, verbose)

    if verbose:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[camera {index}] {w}x{h}")
    return cap


def _apply_manual_exposure(cap, cv2, verbose):
    """
    백엔드마다 값 규약이 다릅니다.
        DSHOW(Windows) : AUTO_EXPOSURE 0.25=수동 / 0.75=자동, EXPOSURE 는 log2 초
        V4L2(Linux)    : AUTO_EXPOSURE 1=수동 / 3=자동
        AVFoundation(macOS) : OpenCV 노출 제어를 대부분 무시합니다
    """
    if C.IS_MAC:
        if verbose:
            print("[camera] macOS 는 OpenCV 노출 제어가 먹지 않습니다.\n"
                  "         모션 블러가 보이면 조명을 밝게 하거나, 수집·시연을\n"
                  "         Windows 쪽에서 하세요. (같은 조건으로 통일하는 게 중요)")
        return

    auto_off = 0.25 if C.IS_WIN else 1
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_off)
    cap.set(cv2.CAP_PROP_EXPOSURE, C.CAM_EXPOSURE)
    if C.CAM_GAIN is not None:
        cap.set(cv2.CAP_PROP_GAIN, C.CAM_GAIN)

    got = cap.get(cv2.CAP_PROP_EXPOSURE)
    if verbose:
        if abs(got - C.CAM_EXPOSURE) < 1.5:
            print(f"[camera] 수동 노출 고정 {got}")
        else:
            print(f"[camera] 수동 노출 미지원 (요청 {C.CAM_EXPOSURE}, 실제 {got}).\n"
                  f"         UVC 수동 노출을 지원하는 카메라로 바꾸면 인식률이 오릅니다.")


class CameraReader:
    """
    끊김 복구가 들어간 래퍼.
    USB 카메라는 케이블이 흔들리면 실제로 잘 빠집니다. 그때 read() 가 계속
    False 를 뱉으면서 루프가 CPU 100% 로 도는 걸 막습니다.

    ★ 호출부(main.py 의 비전 루프)는 read() 가 False 를 주면 곧바로
      continue 합니다. 그래서 '실패할 때 반드시 쉬는 것'이 이 클래스의 책임입니다.
      카메라가 아예 사라져서 재연결까지 실패한 상태에서도 마찬가지입니다.
    """

    def __init__(self, max_retry=5, reopen_sec=1.0, camera_index=None):
        self.camera_index = camera_index
        self.cap = self._open(verbose=True)
        self.max_retry = max_retry
        self.reopen_sec = reopen_sec      # 재연결 실패 후 다음 시도까지의 간격
        self._fails = 0
        self._next_open = 0.0

    def ok(self):
        return self.cap is not None

    def read(self):
        if self.cap is None:
            # 카메라가 없는 상태. 여기서 그냥 돌아가면 호출부가 continue 로
            # 되돌아와 코어 하나를 100% 태웁니다. 쉬면서 주기적으로만 재시도합니다.
            time.sleep(0.05)
            if time.time() >= self._next_open:
                self._reopen()
            return False, None

        got, frame = self.cap.read()
        if got:
            self._fails = 0
            return True, frame

        self._fails += 1
        time.sleep(0.05)                      # 스핀 방지
        if self._fails >= self.max_retry:
            print("[camera] 프레임 유실 -> 재연결 시도")
            self.release()
            self._reopen()
            self._fails = 0
        return False, None

    def _reopen(self):
        self._next_open = time.time() + self.reopen_sec
        self.cap = self._open(verbose=False)
        if self.cap is not None:
            print("[camera] 재연결 성공")

    def _open(self, verbose):
        # 기본 카메라 경로는 기존 호출 서명을 유지해 테스트/외부 코드와도 호환합니다.
        if getattr(self, "camera_index", None) is None:
            return open_camera(verbose=verbose)
        return open_camera(verbose=verbose, camera_index=self.camera_index)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
