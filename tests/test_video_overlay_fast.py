"""Cached caption drawing: same pixels as a fresh draw, moves with the face, never mutates frames."""
import _path  # noqa: F401
import numpy as np

from src.caption_state import CaptionEvent, CaptionState
from src.face_tracker import FaceTrack
from src.video_overlay import CACHE_SIZE, VideoOverlay

rng = np.random.default_rng(3)
frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
untouched = frame.copy()
text = "내일 오후 세 시에 병원에서 만나요. 접수비는 12,500원입니다."


def caption_for(face, partial=False):
    state = CaptionState()
    state.update(CaptionEvent("partial" if partial else "final", text, 1, 100), face)
    return state


face = FaceTrack(1, (280, 250, 520, 540), 100)
other = FaceTrack(2, (800, 220, 980, 500), 100)
shared = VideoOverlay()
for mirror in (False, True):
    for partial in (False, True):
        caption = caption_for(face, partial)
        first = shared.render(frame, caption, face, 100, mirror=mirror, tracks=[face, other],
                              selected_id=1)
        cached = shared.render(frame, caption, face, 100, mirror=mirror, tracks=[face, other],
                               selected_id=1)
        fresh = VideoOverlay().render(frame, caption, face, 100, mirror=mirror,
                                      tracks=[face, other], selected_id=1)
        assert np.array_equal(first, cached) and np.array_equal(first, fresh), (mirror, partial)
        assert first.shape == frame.shape and first.dtype == np.uint8
assert np.array_equal(frame, untouched), "the capture frame is never drawn on"

# A face moving by whole pixels moves the cached bubble with it, pixel for pixel.
flat = np.full((720, 1280, 3), 40, np.uint8)
caption = caption_for(face)
base = VideoOverlay().render(flat, caption, face, 100, tracks=[], selected_id=None)
moved_face = FaceTrack(1, (280 + 30, 250 + 20, 520 + 30, 540 + 20), 100)
moved = shared.render(flat, caption_for(moved_face), moved_face, 100, tracks=[], selected_id=None)
changed = np.argwhere((base != 40).any(axis=2) & (np.arange(720)[:, None] > 60))
(y0, x0), (y1, x1) = changed.min(axis=0), changed.max(axis=0)
assert np.array_equal(base[y0:y1 + 1, x0:x1 + 1], moved[y0 + 20:y1 + 21, x0 + 30:x1 + 31])

# No face: the readable fixed caption at the bottom; tiny frames still return their own size.
fixed = VideoOverlay().render(flat, caption_for(None), None, 100)
assert (fixed[-120:] != 40).any() and not (fixed[200:400, 300:900] != 40).any()
for w, h in ((160, 120), (320, 240), (640, 480)):
    assert VideoOverlay().render(np.zeros((h, w, 3), np.uint8), caption, None, 100).shape == (h, w, 3)

# The sprite cache stays bounded however many different sentences pass through.
bounded = VideoOverlay()
for i in range(CACHE_SIZE * 3):
    bounded.render(flat, caption_for(face) if i % 2 else CaptionState(), face, 100,
                   tracks=[FaceTrack(i, (100 + i, 300, 300 + i, 500), 100)], selected_id=i)
assert len(bounded.sprites) <= CACHE_SIZE and len(bounded.layouts) <= CACHE_SIZE
print("[OK] video overlay: cached sprites equal fresh drawing, follow the face, bounded, "
      "capture frames untouched")
