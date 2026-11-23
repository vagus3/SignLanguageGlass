"""말풍선 유리 효과: 뒤 영상이 흐리게 비치고, 원본 프레임은 건드리지 않습니다."""
import _path  # noqa: F401

import numpy as np

from src import video_overlay
from src.caption_state import CaptionEvent, CaptionState
from src.face_tracker import FaceTrack
from src.video_overlay import VideoOverlay

face = FaceTrack(1, (180, 180, 320, 370), 100)
caption = CaptionState()
caption.update(CaptionEvent("final", "안녕하세요", 1, 100), face)


def bubble_inside(frame):
    """말풍선 안쪽(글자가 없는 오른쪽 아래 구석) 픽셀."""
    rects = []
    overlay = VideoOverlay()
    original = overlay._bubble
    overlay._bubble = lambda *a: rects.append(a[3:7]) or original(*a)
    before = frame.copy()
    out = overlay.render(frame, caption, face, 100, tracks=[face], selected_id=1)
    assert np.array_equal(frame, before), "원본 프레임은 바뀌면 안 됩니다"
    bx, by, bw, bh = (int(v) for v in rects[0])
    return out[by + bh - 20:by + bh - 8, bx + bw - 60:bx + bw - 20]


# 뒤가 밝은 곳과 어두운 곳에서 말풍선 안쪽 색이 달라야 (=비쳐야) 합니다.
bright = bubble_inside(np.full((480, 800, 3), 230, np.uint8)).mean()
dark = bubble_inside(np.zeros((480, 800, 3), np.uint8)).mean()
assert bright - dark > 80, (bright, dark)
# 그래도 어두운 색이 섞여 흰 글자가 읽히게: 흰 배경보다 확실히 어둡습니다.
assert bright < 200, bright

# GLASS_TINT 1.0 이면 예전처럼 뒤를 완전히 가립니다.
saved = video_overlay.GLASS_TINT
try:
    video_overlay.GLASS_TINT = 1.0
    a = bubble_inside(np.full((480, 800, 3), 230, np.uint8)).mean()
    b = bubble_inside(np.zeros((480, 800, 3), np.uint8)).mean()
    assert abs(a - b) < 1, (a, b)
finally:
    video_overlay.GLASS_TINT = saved

# 화면 가장자리에 걸친 말풍선도 문제없이 그립니다.
for w, h in ((160, 120), (320, 240), (1280, 720)):
    assert VideoOverlay().render(np.zeros((h, w, 3), np.uint8), caption, None, 100).shape == (h, w, 3)
print("[OK] 유리 말풍선: 뒤가 비치고, 원본 보존, 불투명 전환 가능")
