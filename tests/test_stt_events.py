"""Real STT worker event IDs and cancellation across a simulated TTS gate."""
import _path  # noqa: F401
import queue
import threading
from types import SimpleNamespace as NS
from unittest.mock import patch

import config as C
from src.stt_worker import SttWorker
from src.caption_state import CaptionEvent

ready = threading.Event()
state = {}


class Stream:
    def __init__(self, callback, **kwargs):
        self.callback = callback

    def __enter__(self):
        state["callback"] = self.callback
        ready.set()
        return self

    def __exit__(self, *exc):
        state["closed"] = True


class Decoder:
    n = 0

    def feed(self, pcm):
        self.n += 1
        return [("partial", "안녕")] if self.n == 1 else [("final", "안녕하세요")]

    def reset(self):
        self.n = 0


gate = NS(open=True)
gate.is_open = lambda: gate.open
out = queue.Queue()
worker = SttWorker(out, gate=gate, caption_events=True)
worker._decoder = Decoder


def block():
    state["callback"](bytes(C.AUDIO_BLOCK * 2), C.AUDIO_BLOCK, None, None)


def result(kind):
    actual, event = out.get(timeout=5)
    assert actual == kind and isinstance(event, CaptionEvent), (actual, event)
    return event


with patch.dict("sys.modules", {"sounddevice": NS(RawInputStream=Stream)}):
    worker.start()
    try:
        assert ready.wait(5)
        block()
        first = result("partial")
        gate.open = False
        block()
        cancelled = result("cancel")
        assert first.utterance_id == cancelled.utterance_id
        gate.open = True
        block()
        second = result("partial")
        block()
        final = result("final")
        assert second.utterance_id == final.utterance_id > first.utterance_id
        assert first.observed_at <= cancelled.observed_at <= final.observed_at
    finally:
        worker.stop()
        worker.join(2)
assert not worker.is_alive() and state["closed"]
print("[OK] STT metadata: stable IDs, TTS cancellation, immediate resume and stream cleanup")
