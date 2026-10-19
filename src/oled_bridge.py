# -*- coding: utf-8 -*-
"""OLED 미리보기와 이전 물리 디스플레이 출력 경로.

현재 기본 출력은 노트북의 미리보기 창입니다. serial/pi 백엔드는 예전
하드웨어 실험을 위해 남겨 두었습니다.
"""
import sys
import queue
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.hud import pack_ssd1306

MAGIC = b"\xa5\x5a"     # 프레임 동기화 헤더. 없으면 한 바이트만 밀려도 계속 깨집니다


class NullDisplay:
    has_window = False

    def show(self, img):
        pass

    def close(self):
        pass


class PreviewDisplay:
    """실물 OLED 없이 레이아웃·가독성·좌표 매핑을 전부 검증할 수 있습니다."""

    has_window = True
    dual = True     # 노트북에서는 내 결과와 상대의 말을 두 영역으로 함께 보여줍니다

    def __init__(self, scale=6):
        self.scale = scale
        self._last = None

    def show(self, img):
        import cv2
        import numpy as np
        # 대부분의 프레임은 직전과 같은 화면입니다. 창은 마지막 그림을 유지하므로
        # 확대·imshow 를 건너뜁니다. 키 입력은 메인 루프의 waitKey 가 계속 받습니다.
        frame = (img.size, img.tobytes())
        if frame == self._last:
            return
        self._last = frame
        arr = np.array(img, dtype=np.uint8) * 255
        h, w = arr.shape[:2]
        big = cv2.resize(arr, (w * self.scale, h * self.scale),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imshow("HUD (OLED simulation)", big)

    def close(self):
        import cv2
        try:
            cv2.destroyWindow("HUD (OLED simulation)")
        except Exception:
            pass


class SerialDisplay:
    """

    has_window = False
    ★ 시리얼 쓰기는 반드시 별도 스레드에서.
      MCU 가 PC 보다 느리면 OS 송신 버퍼가 가득 차고, ser.write() 가
      메인 스레드를 블로킹합니다. 그러면 OLED 가 느린 것으로 끝나지 않고
      영상 루프 전체가 같이 멈춥니다.
    """

    def __init__(self, port=None, baud=None):
        import serial
        port = port or C.SERIAL_PORT or self.autodetect()
        if port is None:
            raise RuntimeError("시리얼 포트를 찾지 못했습니다. config.SERIAL_PORT 를 직접 지정하세요.")
        self.ser = serial.Serial(port, baud or C.SERIAL_BAUD, timeout=0.1,
                                 write_timeout=0.5)
        print(f"[display] serial {port} @ {baud or C.SERIAL_BAUD}")
        self._last = None
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, name="oled", daemon=True)
        self._thread.start()

    def _writer(self):
        while not self._stop.is_set():
            try:
                buf = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.ser.write(MAGIC + buf)
            except Exception as e:
                print(f"[display] 전송 실패: {e}")

    @staticmethod
    def autodetect():
        """Windows=COMx, macOS=/dev/cu.usbmodem*|usbserial*"""
        from serial.tools import list_ports
        cands = []
        for p in list_ports.comports():
            d = (p.device or "") + " " + (p.description or "")
            if any(k in d for k in ("usbmodem", "usbserial", "wchusb",
                                    "Arduino", "Pico", "CH340", "COM")):
                cands.append(p.device)
        return cands[0] if cands else None

    def show(self, img):
        buf = pack_ssd1306(img)
        if buf == self._last:       # 변화 없으면 전송 생략 (대역폭·CPU 절약)
            return
        self._last = buf
        try:                        # 큐를 쌓지 않고 항상 최신 프레임만
            self._q.put_nowait(buf)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(buf)
            except queue.Empty:
                pass

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass


class PiDisplay:
    """이전 Raspberry Pi GPIO 출력 경로."""

    has_window = False

    def __init__(self, i2c=False, rotate=0):
        from luma.oled.device import ssd1306
        if i2c:
            from luma.core.interface.serial import i2c as _iface
            iface = _iface(port=1, address=0x3C)
        else:
            from luma.core.interface.serial import spi as _iface
            iface = _iface(device=0, port=0, gpio_DC=24, gpio_RST=25)
        self.dev = ssd1306(iface, width=C.OLED_W, height=C.OLED_H, rotate=rotate)
        self._last = None
        print(f"[display] pi {'i2c' if i2c else 'spi'} "
              f"{C.OLED_W}x{C.OLED_H}")

    def show(self, img):
        # luma 가 PIL 이미지를 그대로 받습니다. pack_ssd1306 이 필요 없습니다.
        # 변화가 없으면 전송을 건너뜁니다(Pi CPU 를 비전에 양보).
        buf = img.tobytes()
        if buf == self._last:
            return
        self._last = buf
        self.dev.display(img.convert("1"))

    def close(self):
        try:
            self.dev.cleanup()
        except Exception:
            pass


def make_display(backend=None):
    backend = (backend or C.DISPLAY_BACKEND).lower()
    if backend == "serial":
        try:
            return SerialDisplay()
        except Exception as e:
            print(f"[display] serial 실패({e}) -> preview 로 대체")
            return PreviewDisplay()
    if backend == "pi":
        try:
            return PiDisplay()
        except Exception as e:
            print(f"[display] pi 실패({e}) -> preview 로 대체")
            return PreviewDisplay()
    if backend == "preview":
        return PreviewDisplay()
    return NullDisplay()
