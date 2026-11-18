"""Capture-clock audio blocks and sample-boundary routing for alternating speakers."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioBlock:
    pcm: bytes
    start: float
    end: float
    serial: int
    discontinuity: bool = False

    @classmethod
    def capture(cls, pcm, rate, timing, now, serial, discontinuity=False):
        duration = len(pcm) / (2 * rate)
        start = now - duration
        # PortAudio and perf_counter have different epochs. Convert via currentTime.
        try:
            age = float(timing.currentTime) - float(timing.inputBufferAdcTime)
            if math.isfinite(age) and 0 <= age <= 5:
                start = now - age
        except (AttributeError, TypeError, ValueError):
            pass
        return cls(pcm, start, start + duration, serial, discontinuity)


class SpeakerAudioRouter:
    def __init__(self, decoder, publish, speakers, rate=16000, vad=None, only_on_screen=False):
        self.decoder, self.publish, self.speakers, self.rate = decoder, publish, speakers, rate
        # True: 입이 움직인 얼굴이 없는 소리는 받아쓰지 않습니다 (화면 밖 사람, TV 소리).
        self.only_on_screen = only_on_screen
        if vad is None:
            import webrtcvad
            vad = webrtcvad.Vad(2)
        self.vad = vad
        self.owner = self.start = None
        self.end = 0.0
        self.published = False

    def prepare(self, block):
        """Observe VAD immediately, before holding PCM for visual confirmation."""
        size = self.rate // 50 * 2  # 20ms, not an entire 250ms input block.
        slices = []
        for offset in range(0, len(block.pcm), size):
            pcm = block.pcm[offset:offset + size]
            start = block.start + offset / (2 * self.rate)
            end = start + len(pcm) / (2 * self.rate)
            # Only VAD receives padding; the decoder receives each real sample once.
            padded = pcm + bytes(size - len(pcm))
            speech = self.vad.is_speech(padded, self.rate)
            self.speakers.observe_audio(end, speech)
            slices.append((pcm, start, end, speech))
        return slices

    def _emit(self, kind, text):
        self.publish(kind, text, audio_start=self.start, audio_end=self.end,
                     speaker_id=self.owner, speaker_resolved=True)
        self.published = kind == "partial"

    def finish(self):
        if self.start is not None:
            for kind, text in self.decoder.flush():
                if text:
                    self._emit(kind, text)
            if self.published:
                self._emit("cancel", "")
        self.decoder.reset()
        self.owner = self.start = None
        self.published = False

    def cancel(self):
        if self.published:
            self._emit("cancel", "")
        self.decoder.reset()
        self.owner = self.start = None
        self.published = False

    def feed(self, slices):
        for pcm, start, end, speech in slices:
            if speech:
                owner = self.speakers.resolve((start + end) / 2)
                if self.only_on_screen and owner is None:
                    # 말하는 도중 입이 잠깐 안 잡힌 것은 이어서 받고, 새로 시작하는 소리만 거릅니다.
                    if self.start is None:
                        continue
                    owner = self.owner
                if self.start is not None and owner != self.owner:
                    self.finish()  # A is finalized before ANY of B's samples enter STT.
                if self.start is None:
                    self.owner, self.start = owner, start
            if self.start is None:
                continue  # Do not carry unowned pre-speech PCM into the next turn.
            self.end = end
            for kind, text in self.decoder.feed(pcm):
                # Suppress silence hallucinations without a VAD speech segment.
                if self.start is not None:
                    self._emit(kind, text)
                if kind == "final":
                    self.owner = self.start = None
                    self.published = False
