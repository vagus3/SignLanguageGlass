# -*- coding: utf-8 -*-
"""
Phase 2. 공간 AR 말풍선 렌더러

여기서 해결하는 문제 3가지

1) 좌표 변환
   카메라 1280x720 -> OLED 128x64 는 단순 비율 축소가 아닙니다.
   디스플레이가 실제로 덮는 시야는 카메라 시야(110도)의 일부(약 20~30도)뿐이라,
   카메라 화면 중앙 영역만 잘라서 매핑해야 합니다. (config.HUD_FOV_X/Y)

2) 시차(Parallax)
   카메라는 콧대 위, 눈은 그 뒤 아래에 있습니다. 그래서 카메라가 본 얼굴 위치와
   눈이 본 얼굴 위치가 다릅니다. 1회성 오프셋+스케일 보정으로 잡습니다.
   완전한 해법은 호모그래피지만 데모에는 어파인으로 충분합니다.

3) 화면 밖 화자
   얼굴이 디스플레이 커버 영역 밖이면 말풍선을 붙일 자리가 없습니다.
   이때는 방향 화살표로 "저쪽에서 말하고 있음"만 표시합니다.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# ---------------------------------------------------------------- 캘리브레이션
_CALIB = {"offset": list(C.HUD_CALIB_OFFSET), "scale": list(C.HUD_CALIB_SCALE)}


def load_calib():
    """
    시차 보정값을 파일에서 읽습니다. (src/calibrate.py 가 저장)
    config.py 를 직접 고치지 않는 이유: 보정값은 안경을 다시 조립할 때마다
    바뀌는 '측정치'라서, 코드가 아니라 데이터로 다뤄야 합니다.
    """
    global _CALIB
    if C.CALIB_PATH.exists():
        try:
            d = json.loads(C.CALIB_PATH.read_text(encoding="utf-8"))
            _CALIB = {"offset": d.get("offset", [0, 0]),
                      "scale": d.get("scale", [1.0, 1.0])}
            print(f"[hud] 캘리브레이션 로드 {_CALIB}")
        except Exception as e:
            print(f"[hud] 캘리브레이션 읽기 실패({e}) -> 기본값 사용")
    return _CALIB


def save_calib(offset, scale):
    C.CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.CALIB_PATH.write_text(
        json.dumps({"offset": list(offset), "scale": list(scale)}, indent=2),
        encoding="utf-8")
    load_calib()


# ---------------------------------------------------------------- 좌표 변환
def camera_to_hud(cx, cy, cam_w, cam_h):
    """
    카메라 픽셀 -> HUD 픽셀.
    반환: (x, y, inside)  inside=False 면 디스플레이 커버 영역 밖.
    """
    u, v = cx / max(cam_w, 1), cy / max(cam_h, 1)
    x0, x1 = C.HUD_FOV_X
    y0, y1 = C.HUD_FOV_Y
    inside = (x0 <= u <= x1) and (y0 <= v <= y1)

    nx = (u - x0) / max(x1 - x0, 1e-6)
    ny = (v - y0) / max(y1 - y0, 1e-6)

    sx, sy = _CALIB["scale"]
    ox, oy = _CALIB["offset"]
    x = int(nx * C.OLED_W * sx + ox)
    y = int(ny * C.OLED_H * sy + oy)
    return x, y, inside


# ---------------------------------------------------------------- 렌더러
class HudRenderer:
    def __init__(self):
        from PIL import ImageFont
        self.font = self._load_font(ImageFont)

    # 갈무리가 없을 때 대신 쓸 시스템 한글 폰트. 픽셀 폰트가 아니라 11px 에서
    # 뭉개지지만, 적어도 글자를 읽을 수는 있습니다. PIL 기본 폰트는 한글
    # 글리프가 아예 없어서 네모(두부)만 나옵니다.
    SYSTEM_KO = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",   # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",           # macOS
        "C:/Windows/Fonts/malgun.ttf",                          # Windows
        "C:/Windows/Fonts/gulim.ttc",                           # Windows
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",      # Linux / Pi
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    @classmethod
    def _load_font(cls, ImageFont):
        if C.HUD_FONT.exists():
            try:
                return ImageFont.truetype(str(C.HUD_FONT), C.HUD_FONT_SIZE)
            except Exception as e:
                print(f"[hud] {C.HUD_FONT.name} 을 읽지 못했습니다: {e}")

        for cand in cls.SYSTEM_KO:
            if Path(cand).exists():
                try:
                    f = ImageFont.truetype(cand, C.HUD_FONT_SIZE)
                    print(f"[hud] {C.HUD_FONT.name} 없음 -> 시스템 한글 폰트로 대체: "
                          f"{Path(cand).name}\n"
                          f"      읽히기는 하지만 11px 에서 뭉개집니다. 실물 OLED 용으로는"
                          f" 갈무리(Galmuri11.ttf) 같은\n"
                          f"      저해상도 픽셀 폰트를 assets/fonts/ 에 넣으세요.")
                    return f
                except Exception:
                    continue

        print(f"[hud] 한글 폰트를 하나도 못 찾았습니다 -> 기본 폰트 (한글이 네모로 나옵니다).\n"
              f"      갈무리(Galmuri11.ttf) 를 assets/fonts/ 에 넣으세요.")
        return ImageFont.load_default()

    # ---------------------------------------------------------------
    def text_w(self, s):
        """문자열의 실제 픽셀 폭. 폰트를 못 읽었으면 대략값."""
        try:
            return self.font.getlength(s)
        except Exception:
            return len(s) * C.HUD_FONT_SIZE

    def _fits(self, s, max_w):
        return self.text_w(s) <= max_w and len(s) <= C.HUD_MAX_CHARS_PER_LINE

    def _cut(self, s, max_w):
        """한 줄에 넣을 수 있는 최대 글자 수(최소 1)."""
        k = 1
        while k < len(s) and self._fits(s[:k + 1], max_w):
            k += 1
        return k

    def wrap(self, text, max_w=None, max_lines=None):
        """
        폭 기준 줄바꿈. 넘치면 뒤쪽을 남깁니다.
        자막은 '방금 한 말'이 중요하므로 앞이 아니라 뒤를 살립니다.

        글자 수로 자르지 않는 이유:
          한글은 약 11px, 숫자·영문은 약 6px 라 폭이 두 배 차이 납니다.
          "9글자" 로 고정하면 "3만 5천원입니다" 같은 혼합 문장에서 절반이
          빈 채로 줄이 넘어가고, 반대로 큰 폰트에서는 말풍선을 뚫고 나갑니다.
          실제 폰트 메트릭으로 재고, 가능하면 어절(공백) 경계에서 끊습니다.
          HUD_MAX_CHARS_PER_LINE 은 이제 안전 상한으로만 씁니다.
        """
        max_w = max_w or (C.OLED_W - 10)     # 테두리 1 + 안쪽 여백 4, 좌우
        max_lines = max_lines or C.HUD_MAX_LINES

        words = text.replace("\n", " ").split()
        if not words:
            return [""]

        lines, cur = [], ""
        for w in words:
            cand = f"{cur} {w}" if cur else w
            if self._fits(cand, max_w):
                cur = cand
                continue
            if cur:
                lines.append(cur)
                cur = ""
            while not self._fits(w, max_w):   # 한 줄보다 긴 어절은 글자로 쪼갬
                k = self._cut(w, max_w)
                lines.append(w[:k])
                w = w[k:]
            cur = w
        if cur:
            lines.append(cur)

        if len(lines) > max_lines:
            lines = lines[-max_lines:]
            lines[0] = "..." + lines[0][1:]   # 픽셀 폰트에 '…' 글리프가 없을 수 있음
        return lines

    def render(self, text, face_xy=None, cam_size=None, partial=False):
        """
        1비트 PIL 이미지(128x64) 반환.
        face_xy 가 있으면 얼굴 옆에 말풍선, 없으면 하단 고정 자막.
        """
        from PIL import Image, ImageDraw

        img = Image.new("1", (C.OLED_W, C.OLED_H), 0)
        d = ImageDraw.Draw(img)
        if not text:
            return img

        lines = self.wrap(text)
        lh = C.HUD_FONT_SIZE + 2
        bw = int(max(self.text_w(ln) for ln in lines)) + 8
        bh = lh * len(lines) + 6
        bw, bh = min(bw, C.OLED_W - 2), min(bh, C.OLED_H - 2)

        anchor = None
        if face_xy and cam_size:
            hx, hy, inside = camera_to_hud(face_xy[0], face_xy[1], *cam_size)
            if inside:
                anchor = (hx, hy)
            else:
                self._arrow(d, hx, hy)

        if anchor:
            bx = anchor[0] + 8                      # 얼굴 오른쪽에
            if bx + bw > C.OLED_W:
                bx = anchor[0] - 8 - bw             # 자리 없으면 왼쪽으로
            by = anchor[1] - bh // 2
            bx = max(1, min(bx, C.OLED_W - bw - 1))
            by = max(1, min(by, C.OLED_H - bh - 1))
            d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=3,
                                fill=0, outline=1)
            # 얼굴이 말풍선 왼쪽에 있으면 꼬리도 왼쪽으로 나가야 합니다
            self._tail(d, bx, bw, by + bh // 2, face_left=(anchor[0] < bx))
        else:
            bx, by = 1, C.OLED_H - bh - 1
            bw = C.OLED_W - 3
            d.rectangle([bx, by, bx + bw, by + bh], fill=0, outline=1)

        for i, ln in enumerate(lines):
            d.text((bx + 4, by + 3 + i * lh), ln, font=self.font, fill=1)
        if partial:
            d.point([(bx + bw - 2, by + 2)], fill=1)   # 진행 중 표시
        return img

    def calibration_marker(self, face_xy=None, cam_size=None):
        """현재 시차 보정을 적용한 광학 정렬용 십자 표식을 렌더링합니다."""
        from PIL import Image, ImageDraw

        img = Image.new("1", (C.OLED_W, C.OLED_H), 0)
        d = ImageDraw.Draw(img)
        if face_xy and cam_size:
            x, y, inside = camera_to_hud(face_xy[0], face_xy[1], *cam_size)
            if not inside:
                self._arrow(d, x, y)
            x = max(0, min(x, C.OLED_W - 1))
            y = max(0, min(y, C.OLED_H - 1))
        else:
            x, y = C.OLED_W // 2, C.OLED_H // 2

        arm = 7
        d.line((max(0, x - arm), y, min(C.OLED_W - 1, x + arm), y), fill=1)
        d.line((x, max(0, y - arm), x, min(C.OLED_H - 1, y + arm)), fill=1)
        d.rectangle((max(0, x - 2), max(0, y - 2),
                     min(C.OLED_W - 1, x + 2), min(C.OLED_H - 1, y + 2)),
                    outline=1)
        return img

    # ---------------------------------------------------------------
    @staticmethod
    def _tail(d, bx, bw, by, face_left):
        if face_left:
            d.polygon([(bx, by - 3), (bx, by + 3), (bx - 5, by)], fill=1)
        else:
            x = bx + bw
            d.polygon([(x, by - 3), (x, by + 3), (x + 5, by)], fill=1)

    @staticmethod
    def _arrow(d, hx, hy):
        """화면 밖 화자 방향 표시."""
        if hx < 0:
            d.polygon([(2, 32), (10, 27), (10, 37)], fill=1)
        elif hx >= C.OLED_W:
            w = C.OLED_W
            d.polygon([(w - 3, 32), (w - 11, 27), (w - 11, 37)], fill=1)
        elif hy < 0:
            d.polygon([(C.OLED_W // 2, 2),
                       (C.OLED_W // 2 - 5, 10),
                       (C.OLED_W // 2 + 5, 10)], fill=1)
        else:
            h = C.OLED_H
            d.polygon([(C.OLED_W // 2, h - 3),
                       (C.OLED_W // 2 - 5, h - 11),
                       (C.OLED_W // 2 + 5, h - 11)], fill=1)


# ------------------------------------------------------- SSD1306/1309 패킹
def pack_ssd1306(img):
    """
    PIL 1비트 이미지 -> SSD1306/1309 프레임버퍼 1024바이트

    이 칩의 버퍼는 페이지 구조입니다.
      - 세로 64px 를 8px 씩 8페이지로 나눔
      - 1바이트 = 세로 8픽셀 (LSB 가 위쪽)
      - 순서: page0 의 col0..127, page1 의 col0..127, ...
    이 형식을 안 맞추면 화면이 줄무늬로 깨집니다.

    픽셀을 파이썬 루프로 돌면 프레임당 8192회 접근이라 30fps 에서 CPU 를
    꽤 먹습니다. np.packbits 로 벡터화하면 5배 이상 빠릅니다(결과는 동일).
    """
    import numpy as np

    a = np.array(img, dtype=np.uint8)          # (H, W), 값 0/1
    h, w = a.shape
    a = a.reshape(h // 8, 8, w)                # (page, bit, x)
    return np.packbits(a, axis=1, bitorder="little").reshape(-1).tobytes()


if __name__ == "__main__":
    # 폰트/레이아웃 확인용 미리보기
    import cv2
    import numpy as np

    r = HudRenderer()
    im = r.render("병원이 어디예요?", face_xy=(500, 300), cam_size=(1280, 720))
    arr = (np.array(im, dtype=np.uint8) * 255)
    cv2.imshow("HUD preview x6", cv2.resize(arr, (C.OLED_W * 6, C.OLED_H * 6),
                                            interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
