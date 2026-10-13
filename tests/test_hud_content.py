"""HUD content: newest of partner speech and my result, pages instead of cuts, laptop dual view."""
import _path  # noqa: F401
import numpy as np
import config as C
from src.hud import HudContent, HudRenderer

r = HudRenderer()
t = [100.0]
content = HudContent(r, hold_sec=4.0, own_hold_sec=4.0, page_sec=2.0, now=lambda: t[0])


def same(a, b):
    return np.array_equal(np.array(a), np.array(b))


content.status("수어 입력 대기")
assert same(content.frame(), r.render("수어 입력 대기"))

# The newest of the two wins; partner captions follow the face, mine stay fixed.
content.partner("안녕하세요 접수하러 오셨어요?", partial=True)
t[0] += 0.5
content.own("병원이 어디예요?")
assert same(content.frame(), r.render_lines(r.paginate("병원이 어디예요?")[0]))
t[0] += 0.5
content.partner("저쪽으로 가세요")
face = {"face_xy": (500, 300), "cam_size": (1280, 720)}
assert same(content.frame(**face), r.render("저쪽으로 가세요", **face))
t[0] += 4.1
assert same(content.frame(), r.render("수어 입력 대기")), "Both expired: back to the status line"

# My long result is paged, never cut, and held until every page was shown.
long = ("조금만 천천히 말해 주세요 그리고 병원 접수는 몇 층에서 하는지 "
        "다시 한 번 알려 주시겠어요 정말 감사합니다")
pages = r.paginate(long)
assert len(pages) >= 2 and all(len(p) <= C.HUD_MAX_LINES for p in pages)
assert "".join(line for p in pages for line in p).replace(" ", "") == long.replace(" ", "")
assert not any(line.startswith("...") for p in pages for line in p)
assert r.wrap(long)[0].startswith("..."), "Partner captions still keep only the latest lines"
assert not same(r.render_lines(pages[0], 0, len(pages)), r.render_lines(pages[0])), \
    "Multi-page captions show a page marker"
t[0] = 200.0
content.own(long)
assert same(content.frame(), r.render_lines(pages[0], 0, len(pages)))
t[0] = 202.0
assert same(content.frame(), r.render_lines(pages[1], 1, len(pages)))
held = max(4.0, len(pages) * 2.0)
t[0] = 200.0 + held - 0.01
assert not same(content.frame(), r.render("수어 입력 대기"))
t[0] = 200.0 + held + 0.01
assert same(content.frame(), r.render("수어 입력 대기"))

# Laptop preview: my result above, the partner's speech below, at the same time.
t[0] = 300.0
content.own("병원이 어디예요?")
content.partner("2층으로 가세요")
img = content.dual()
assert img.size == (C.OLED_W, C.OLED_H * 2 + 2)
top = img.crop((0, 0, C.OLED_W, C.OLED_H))
bottom = img.crop((0, C.OLED_H + 2, C.OLED_W, C.OLED_H * 2 + 2))
assert same(top, r.render_lines(r.paginate("병원이 어디예요?")[0]))
assert same(bottom, r.render("2층으로 가세요"))
assert all(img.getpixel((x, C.OLED_H)) for x in range(C.OLED_W)), "Divider between regions"
assert same(content.frame(), r.render("2층으로 가세요")), "One-panel HUD shows only the newest"
t[0] = 305.0
assert same(content.dual().crop((0, 0, C.OLED_W, C.OLED_H)), r.render("수어 입력 대기"))
print("[OK] HUD content: newest wins, lossless paging and hold, laptop two-region preview")
