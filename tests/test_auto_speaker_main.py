"""Main keeps A's late final and B's new partial in separate bubbles and records."""
import _path  # noqa: F401
import contextlib
import io
import sys
from unittest.mock import patch

import cv2
import numpy as np

import main
from src.caption_state import CaptionEvent
from src.video_overlay import VideoOverlay
from src.transcript_view import TranscriptView


now = [100.0]
state, callbacks = {}, {}
boxes = [(80, 140, 180, 300), (430, 140, 530, 300)]


class Vision:
    def __init__(self, out, **kwargs):
        assert kwargs["track_mouths"]
        self.out = out
        state["vision"] = self

    def emit(self, visible=None):
        self.out.put(("frame", {"frame": np.zeros((480, 640, 3), np.uint8),
                               "face": None, "face_boxes": boxes if visible is None else visible,
                               "mouth_openings": [.02, .02], "captured_at": now[0],
                               "size": (640, 480)}))

    def start(self):
        self.emit()

    def stop(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class Stt:
    def __init__(self, out, **kwargs):
        assert kwargs["caption_events"] and kwargs["speakers"] is not None
        state["out"] = out

    def start(self):
        pass

    def stop(self):
        pass

    def join(self, timeout=None):
        pass


class Overlay(VideoOverlay):
    def render(self, frame, caption, track, at, **kwargs):
        assert kwargs["speaker_mode"]
        state["captions"] = {c.track_id: c for c in kwargs["captions"]}
        state["manual"] = kwargs["manual"]
        return super().render(frame, caption, track, at, **kwargs)


class Transcript(TranscriptView):
    def render(self, history):
        state["records"] = list(history.records)
        return super().render(history)


def emit(kind, text, uid, speaker):
    state["out"].put((kind, CaptionEvent(kind, text, uid, now[0], 99, 100,
                                         speaker_id=speaker, speaker_resolved=True)))


step = [0]
def wait_key(delay):
    current = step[0]
    step[0] += 1
    if current == 0:
        emit("partial", "A의 말", 1, 1)
    elif current == 1:
        assert state["captions"][1].text == "A의 말"
        emit("partial", "B의 말", 2, 2)
    elif current == 2:
        assert set(state["captions"]) == {1, 2}
        emit("final", "A의 마지막 문장", 1, 1)
    elif current == 3:
        assert state["captions"][2].text == "B의 말"
        assert state["records"][0].track_id == 1
        # The image is mirrored: x=480 on the camera becomes x=159 on screen.
        callbacks["Signglass - captions"](cv2.EVENT_LBUTTONDOWN, 159, 200, 0, None)
        emit("final", "B의 마지막 문장", 2, 2)
    elif current == 4:
        assert state["manual"]
        assert [r.track_id for r in state["records"]] == [1, 2]
        assert state["captions"][1].text == "A의 마지막 문장"
        emit("partial", "화자 불확실", 3, None)
    elif current == 5:
        assert set(state["captions"]) == {1, 2, None}
        assert state["captions"][None].anchor(None, now[0]) is None
        callbacks["Signglass - captions"](cv2.EVENT_RBUTTONDOWN, 0, 0, 0, None)
        now[0] += .6
        state["vision"].emit([])
        return -1
    elif current == 6:
        assert not state["manual"]
        assert state["captions"][1].anchor(None, now[0]).track_id == 1
        assert state["captions"][2].anchor(None, now[0]).track_id == 2
        return ord("q")
    else:
        raise AssertionError("Interaction did not finish")
    now[0] += .1
    state["vision"].emit()
    return -1


with patch.object(sys, "argv", ["main.py", "--no-sign", "--no-tts", "--display", "null",
                                "--auto-speaker", "--transcript"]), \
        patch("src.clock.now", lambda: now[0]), \
        patch.object(main, "VisionThread", Vision), \
        patch("src.stt_worker.SttWorker", Stt), \
        patch("src.video_overlay.VideoOverlay", Overlay), \
        patch("src.transcript_view.TranscriptView", Transcript), \
        patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.destroyAllWindows"), \
        patch("cv2.waitKey", wait_key), \
        patch("cv2.setMouseCallback", lambda name, fn: callbacks.update({name: fn})), \
        contextlib.redirect_stdout(io.StringIO()):
    main.main()
print("[OK] automatic-speaker main: separate bubbles, late final, manual lock, unknown and occlusion")
