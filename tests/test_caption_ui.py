"""Real Korean drawing: bounds, mirroring, full transcript and scrolling."""
import _path  # noqa: F401
from pathlib import Path
import os

import numpy as np
from PIL import Image

from src.caption_state import CaptionEvent, CaptionState
from src.face_tracker import FaceTrack
from src.caption_history import CaptionHistory
from src.text_layout import korean_font, wrap_text
from src.transcript_view import TranscriptView
from src.video_overlay import VideoOverlay

font = korean_font(20)
text = "내일 오후 세 시에 병원에서 만나요. 접수비는 12,500원입니다."
lines = wrap_text(text, font, 170)
assert "".join(lines) == text
assert all(font.getlength(line) <= 170 for line in lines)

overlay = VideoOverlay()
face = FaceTrack(1, (180, 180, 320, 370), 100)
caption = CaptionState()
caption.update(CaptionEvent("final", text, 1, 100), face)
frame = np.full((480, 800, 3), (65, 45, 30), dtype=np.uint8)
before = frame.copy()
normal = overlay.render(frame, caption, face, 100, tracks=[face], selected_id=1)
assert np.array_equal(frame, before), "Overlay must not modify capture frames"
# The text box flips to the opposite side of the mirrored face.
assert not np.array_equal(normal[220:260, 350:550], frame[220:260, 350:550])
mirrored = overlay.render(frame, caption, face, 100, mirror=True, tracks=[face], selected_id=1)
assert not np.array_equal(mirrored[220:260, 250:450], frame[220:260, 250:450])
assert normal.shape == mirrored.shape == frame.shape
for w, h in ((160, 120), (320, 240), (640, 480), (1280, 720)):
    image = overlay.render(np.zeros((h, w, 3), np.uint8), caption, None, 100)
    assert image.shape == (h, w, 3)

history = CaptionHistory(limit=200)
for i in range(1, 20):
    history.append(CaptionEvent("final", f"{i}번째 문장. " + text * 3, i, 100 + i), 1)
view = TranscriptView()
latest = view.render(history)
view.scroll(12)
older = view.render(history)
assert not np.array_equal(latest, older)
offset = view.offset
history.append(CaptionEvent("final", "새 문장: " + text, 20, 120), None)
updated = view.render(history)
assert view.offset > offset, "New results must not pull the reader to the latest page"
assert np.array_equal(older[88:-40], updated[88:-40]), "The visible reading position stays fixed"
view.scroll(100000)
view.render(history)
assert any(line == "발화 1 · 대상 1" for line, _ in view._rows), "Oldest retained text remains reachable"
view.latest()
assert view.offset == 0
image = view.render(history)
assert image.shape == (640, 640, 3)
assert all(not line.startswith("…") for line, _ in view._rows), "History never truncates text"

# Optional review artifacts, kept outside the repository by the caller.
if os.environ.get("CAPTION_PREVIEW_DIR"):
    destination = Path(os.environ["CAPTION_PREVIEW_DIR"])
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normal[:, :, ::-1]).save(destination / "bubble.png")
    Image.fromarray(mirrored[:, :, ::-1]).save(destination / "mirrored.png")
    Image.fromarray(image[:, :, ::-1]).save(destination / "transcript.png")
print("[OK] Korean video captions, mirrored placement, bounded sizes and lossless transcript UI")
