"""HUD render cache: same pixels as a fresh render, isolated copies, bounded size."""
import _path  # noqa: F401
from unittest import mock
import config as C
from src import hud
from src.hud import HudRenderer
from src.oled_bridge import PreviewDisplay

r = HudRenderer()
cases = [("", None, None, False), ("수어 입력 대기", None, None, False),
         ("어디가 아프세요? 병원에 같이 가 드릴까요?", (640, 300), (1280, 720), False),
         ("어디가 아프세요?", (640, 300), (1280, 720), True),
         ("저쪽에서 말하는 중", (20, 30), (1280, 720), False)]       # outside: arrow
for text, face, cam, partial in cases:
    first = r.render(text, face_xy=face, cam_size=cam, partial=partial)
    again = r.render(text, face_xy=face, cam_size=cam, partial=partial)
    fresh = HudRenderer().render(text, face_xy=face, cam_size=cam, partial=partial)
    assert first.tobytes() == again.tobytes() == fresh.tobytes(), text
# Long enough to need several pages with any fallback font (CI has NanumGothic, not Galmuri).
pages = r.paginate(" ".join(["병원이 어디예요? 조금만 천천히 말해 주세요. 다시 한 번 말씀해 주시겠어요?"] * 3))
assert len(pages) > 1, (r.font.getname(), pages)
for i, page in enumerate(pages):
    assert (r.render_lines(page, i, len(pages)).tobytes()
            == HudRenderer().render_lines(page, i, len(pages)).tobytes())
assert (r.render_lines(pages[0], 0, len(pages)).tobytes()
        != r.render_lines(pages[0], 1, len(pages)).tobytes()), "page dots must differ"

# A caller drawing on the returned image must not change later frames.
img = r.render("수어 입력 대기")
img.paste(1, (0, 0, C.OLED_W, C.OLED_H))
assert r.render("수어 입력 대기").tobytes() == HudRenderer().render("수어 입력 대기").tobytes()

# The key uses calibrated HUD coordinates, so a new parallax calibration redraws.
before = r.render("말풍선", face_xy=(640, 300), cam_size=(1280, 720)).tobytes()
with mock.patch.dict(hud._CALIB, {"offset": [15, 5], "scale": [1.0, 1.0]}):
    moved = r.render("말풍선", face_xy=(640, 300), cam_size=(1280, 720)).tobytes()
assert moved != before

# Wrapped lines are cached as immutable tuples; callers get their own lists.
lines = r.wrap("아주 긴 문장을 여러 줄로 나누어 보여 주는 경우입니다 끝까지 보여야 합니다")
lines.append("오염")
assert "오염" not in r.wrap("아주 긴 문장을 여러 줄로 나누어 보여 주는 경우입니다 끝까지 보여야 합니다")

for i in range(HudRenderer.CACHE_SIZE * 2):
    r.render(f"자막 {i}")
assert len(r._image_cache) <= HudRenderer.CACHE_SIZE
assert len(r._line_cache) <= HudRenderer.CACHE_SIZE

# The preview window redraws only when the picture changes.
shown = []
with mock.patch("cv2.imshow", lambda name, arr: shown.append(arr.shape)):
    display = PreviewDisplay(scale=2)
    for text in ("가", "가", "가", "나", "나", "가"):
        display.show(r.render(text))
assert len(shown) == 3, shown
print("[OK] HUD cache: identical pixels, isolated copies, calibration-aware keys, "
      "bounded size, preview skips unchanged frames")
