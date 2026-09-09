import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys
from pathlib import Path
import config as C
from src.hud import HudRenderer, pack_ssd1306

r = HudRenderer()
MAXW = C.OLED_W - 10
tests = ["병원이 어디예요?", "3만 5천원입니다", "안녕하세요 반갑습니다 오늘 날씨가 좋네요",
         "아주아주아주아주아주아주아주아주긴어절하나", "", "   ", "네"]
bad = 0
for t in tests:
    lines = r.wrap(t)
    ws = [r.text_w(l) for l in lines]
    over = [l for l, w in zip(lines, ws) if w > MAXW]
    bad += len(over) + (len(lines) > C.HUD_MAX_LINES)
    print(f"  {t!r}\n      -> {lines}   폭 {[round(w) for w in ws]} / 상한 {MAXW}")
    if over: print(f"      !! 넘침: {over}")

print("\n  렌더 + SSD1306 패킹 검사")
for t in tests:
    for face in (None, (500, 300), (50, 300), (1200, 300),
                 (640, 20), (640, 700)):
        img = r.render(t, face_xy=face, cam_size=(1280, 720), partial=True)
        buf = pack_ssd1306(img)
        assert len(buf) == 1024, len(buf)
        assert img.size == (C.OLED_W, C.OLED_H)
print("      OK (모든 조합에서 128x64 / 1024바이트)")

print("\n  캘리브레이션 십자 표식 검사")
for face in (None, (640, 360), (0, 0), (1279, 719)):
    marker = r.calibration_marker(face, (1280, 720))
    assert marker.size == (C.OLED_W, C.OLED_H)
    assert len(pack_ssd1306(marker)) == 1024
    assert marker.getbbox() is not None
print("      OK (얼굴 미검출/화면 안팎 모두 표시)")
print(f"\n실패 {bad}")
sys.exit(1 if bad else 0)
