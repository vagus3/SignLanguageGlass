# -*- coding: utf-8 -*-
"""
디스플레이 백엔드

PC 에는 I2C/SPI 포트가 없습니다. 명세서의 "Python에서 I2C 제어 코드"는
라즈베리파이 기준이고, PC 직결 구조에서는 반드시 브리지 MCU 가 필요합니다.

    PC ──USB(시리얼)──▶ 아두이노/Pico ──SPI──▶ SSD1306/1309 OLED

I2C 가 아니라 SPI 를 쓰는 이유: 128x64 는 프레임당 1KB 인데
I2C 400kHz 로는 파이썬 오버헤드까지 합쳐 15~25fps 가 한계입니다.
SPI 8MHz 면 여유롭습니다.

백엔드 4종 (config.DISPLAY_BACKEND 로 선택)
    preview : PC 창에 6배 확대   -> 하드웨어 0원, Step 1~2 전체를 여기서 검증
    serial  : PC + 브리지 MCU 경유 실제 OLED
    pi      : 라즈베리파이 GPIO 직결 (브리지 MCU 불필요)
    null    : 출력 없음

■ 라즈베리파이를 쓰면 브리지가 사라집니다
    위 구조가 필요한 이유는 "PC 에 SPI 포트가 없어서" 하나뿐입니다.
    라즈베리파이는 GPIO 에 SPI/I2C 가 있으므로 OLED 를 직접 물리면 됩니다.

        Pi ──SPI(GPIO)──▶ SSD1306/1309 OLED

    아두이노도, 시리얼 프로토콜도, 펌웨어도 필요 없어집니다.
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

    def __init__(self, scale=6):
        self.scale = scale

    def show(self, img):
        import cv2
        import numpy as np
        arr = np.array(img, dtype=np.uint8) * 255
        big = cv2.resize(arr, (C.OLED_W * self.scale, C.OLED_H * self.scale),
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
    """

    has_window = False
    라즈베리파이 GPIO 직결 (luma.oled).

        pip install luma.oled
        sudo raspi-config  ->  Interface Options  ->  SPI 활성화

    배선 (SPI)
        OLED          Pi 40핀
        GND           GND      (6번)
        VCC           3V3      (1번)
        SCL/D0(CLK)   GPIO11 SCLK (23번)
        SDA/D1(MOSI)  GPIO10 MOSI (19번)
        RES           GPIO25      (22번)
        DC            GPIO24      (18번)
        CS            GPIO8  CE0  (24번)

    ※ I2C 모듈을 쓸 거면 i2c=True 로. 다만 128x64 는 프레임당 1KB 라
      I2C 400kHz 로는 15~25fps 가 한계입니다. SPI 를 권합니다.
    """

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
