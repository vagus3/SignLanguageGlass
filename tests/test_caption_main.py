"""Main app: mirrored click -> stable binding -> transcript -> face loss.

Camera/STT queues and GUI input are simulated; real caption state and Korean
renderers run. No webcam, microphone or native window is opened.
"""
import _path  # noqa: F401
import contextlib
import io
import sys
from unittest.mock import patch

import cv2
import numpy as np

import main
import config as C
from src.caption_state import CaptionEvent
from src.video_overlay import VideoOverlay
from src.transcript_view import TranscriptView

t = [100.0]
callbacks, state, rendered = {}, {}, {}
long_text = "내일 오후 세 시에 병원에서 만나요. " * 12
boxes = [(100, 140, 200, 300), (420, 140, 520, 300)]


class Vision:
    def __init__(self, out, **kwargs):
        self.out = out
        state["vision"] = self

    def emit(self, detections=None):
        self.out.put(("frame", {
            "frame": np.zeros((480, 640, 3), np.uint8), "face": None,
            "face_boxes": boxes if detections is None else detections,
            "captured_at": t[0], "size": (640, 480), "word": None,
        }))

    def start(self):
        self.emit()

    def stop(self):
        state["vision_stopped"] = True

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class Stt:
    def __init__(self, out, **kwargs):
        assert kwargs["caption_events"]
        state["stt_out"] = out

    def start(self):
        pass

    def stop(self):
        state["stt_stopped"] = True

    def join(self, timeout=None):
        pass


class Overlay(VideoOverlay):
    def render(self, frame, caption, track, now, **kwargs):
        rendered.update(track_id=track.track_id if track else None,
                        caption_id=caption.track_id, partial=caption.partial,
                        manual=kwargs["manual"], text=caption.text)
        return super().render(frame, caption, track, now, **kwargs)


class Transcript(TranscriptView):
    def render(self, history):
        state["records"] = list(history.records)
        return super().render(history)


def emit(kind, text, uid):
    state["stt_out"].put((kind, CaptionEvent(kind, text, uid, t[0])))


phase = [0]


def wait_key(delay):
    step = phase[0]
    phase[0] += 1
    mouse = callbacks["Signglass - captions"]
    key = -1
    if step == 0:
        assert rendered["track_id"] is None, "Multiple faces require selection"
        # Original x=150 becomes displayed x=489 in the mirrored image.
        mouse(cv2.EVENT_LBUTTONDOWN, 489, 200, 0, None)
    elif step == 1:
        assert rendered["manual"] and rendered["track_id"] == 1
        emit("partial", "내일", 1)
    elif step == 2:
        assert rendered["caption_id"] == 1 and rendered["partial"]
        mouse(cv2.EVENT_LBUTTONDOWN, 169, 200, 0, None)
    elif step == 3:
        assert rendered["track_id"] == 2 and rendered["caption_id"] is None
        emit("final", long_text, 1)
        key = ord("h")
    elif step == 4:
        assert state["records"][0].text == long_text.strip()
        assert state["records"][0].track_id is None
        emit("partial", "두 번째", 2)
        emit("final", "두 번째 문장입니다", 2)
    elif step == 5:
        assert len(state["records"]) == 2 and state["records"][1].track_id == 2
        emit("final", "중복 결과", 2)
        emit("cancel", "", 1)
    elif step == 6:
        assert len(state["records"]) == 2
        assert rendered["text"] == "두 번째 문장입니다"
        # Advance past the face-loss grace period.
        t[0] += 1.0
        state["vision"].emit([])
        return -1
    elif step == 7:
        assert rendered["track_id"] is None and rendered["manual"]
        emit("partial", "얼굴 없는 발화", 3)
        mouse(cv2.EVENT_RBUTTONDOWN, 0, 0, 0, None)
        t[0] += 0.1
        state["vision"].emit([boxes[0]])
        return -1
    elif step == 8:
        assert not rendered["manual"] and rendered["track_id"] is not None
        assert rendered["caption_id"] is None, "A future face cannot own an older event"
        emit("final", "얼굴 없는 발화", 3)
    elif step == 9:
        assert len(state["records"]) == 3
        assert state["records"][2].track_id is None
        return ord("q")
    else:
        raise AssertionError("Main did not complete the scripted interaction")
    t[0] += 0.1
    state["vision"].emit()
    return key


output = io.StringIO()
with patch.object(sys, "argv", ["main.py", "--no-sign", "--no-tts", "--display", "null",
                                "--video-captions"]), \
        patch.object(C, "MIRROR_PREVIEW", True), \
        patch("src.clock.now", lambda: t[0]), \
        patch.object(main, "VisionThread", Vision), \
        patch("src.stt_worker.SttWorker", Stt), \
        patch("src.video_overlay.VideoOverlay", Overlay), \
        patch("src.transcript_view.TranscriptView", Transcript), \
        patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.destroyWindow"), \
        patch("cv2.destroyAllWindows"), patch("cv2.waitKey", wait_key), \
        patch("cv2.setMouseCallback", lambda name, cb: callbacks.update({name: cb})), \
        contextlib.redirect_stdout(output):
    main.main()
assert state["vision_stopped"] and state["stt_stopped"]
assert "[음성] " + long_text.strip() in output.getvalue()


class BadVision(Vision):
    def start(self):
        self.out.put(("error", "카메라를 열 수 없습니다"))


class AvailableStt(Stt):
    def start(self):
        emit("final", "카메라 없이도 음성 자막은 유지합니다", 1)


def quit_after_fallback(delay):
    assert rendered["track_id"] is None
    assert rendered["text"] == "카메라 없이도 음성 자막은 유지합니다"
    assert len(state["records"]) == 1
    return ord("q")


with patch.object(sys, "argv", ["main.py", "--no-sign", "--no-tts", "--display", "null",
                                "--video-captions", "--transcript"]), \
        patch("src.clock.now", lambda: t[0]), \
        patch.object(main, "VisionThread", BadVision), \
        patch("src.stt_worker.SttWorker", AvailableStt), \
        patch("src.video_overlay.VideoOverlay", Overlay), \
        patch("src.transcript_view.TranscriptView", Transcript), \
        patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.destroyWindow"), \
        patch("cv2.destroyAllWindows"), patch("cv2.waitKey", quit_after_fallback), \
        patch("cv2.setMouseCallback"), contextlib.redirect_stdout(io.StringIO()):
    main.main()
print("[OK] Main mirrored selection, utterance binding, loss, transcript and cleanup")
