import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys
from pathlib import Path
from PIL import ImageFont
import config as C
from src.hud import HudRenderer

FONT = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
r = HudRenderer()
r.font = ImageFont.truetype(FONT, C.HUD_FONT_SIZE)   # 실제 한글 폰트 11px
MAXW = C.OLED_W - 10
print(f"  실측: 한글 1자 {r.text_w('가'):.0f}px, 숫자 1자 {r.text_w('1'):.0f}px, 상한 {MAXW}px\n")

def old_wrap(text, per_line=9, max_lines=3):
    text = text.replace("\n", " ").strip()
    lines = [text[i:i+per_line] for i in range(0, len(text), per_line)] or [""]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]; lines[0] = "…" + lines[0][1:]
    return lines

for t in ["병원이 어디예요?", "3만 5천원입니다", "네 알겠습니다",
          "다시 한 번 말씀해 주시겠어요?", "이름이 뭐예요?"]:
    new = r.wrap(t); old = old_wrap(t)
    nw = [round(r.text_w(l)) for l in new]; ow = [round(r.text_w(l)) for l in old]
    over = [l for l, w in zip(new, nw) if w > MAXW]
    print(f"  {t!r}")
    print(f"     수정 전 (9자 고정) : {old}  폭 {ow}")
    print(f"     수정 후 (폭 기준)  : {new}  폭 {nw}")
    assert not over, f"넘침 {over}"
    assert len(new) <= C.HUD_MAX_LINES
print("\n  [OK] 모든 줄이 상한 안에 들어옵니다")
