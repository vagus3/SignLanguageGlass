"""Buffered PCM routes by capture time and finalizes A before decoding B."""
import _path  # noqa: F401
import queue
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from src.active_speaker import SpeakerTimeline
from src.caption_state import CaptionBook
from src.face_tracker import FaceTrack, FaceTracker
from src.speaker_audio import AudioBlock, SpeakerAudioRouter
from src.stt_worker import SttWorker, VoskDecoder
from src import clock


class Decoder:
    def __init__(self):
        self.samples = []
        self.segments = []

    def feed(self, pcm):
        self.samples.extend(np.frombuffer(pcm, np.int16).tolist())
        return [("partial", "A" if self.samples[0] == 100 else "B")]

    def flush(self):
        if not self.samples:
            return []
        self.segments.append(self.samples.copy())
        return [("final", "A입니다" if self.samples[0] == 100 else "B입니다")]

    def reset(self):
        self.samples = []


class Vad:
    def is_speech(self, pcm, rate):
        return any(pcm)


class AudioTests(unittest.TestCase):
    def test_vosk_vad_recognizes_router_20ms_speech(self):
        class SpeechVad:
            def is_speech(self, pcm, rate):
                self.length = len(pcm)
                return True
        vad = SpeechVad()
        self.assertTrue(VoskDecoder._has_speech(vad, bytes(640), 16000))
        self.assertEqual(vad.length, 640)

    def test_vosk_vad_preserves_normal_30ms_frames(self):
        class SpeechVad:
            lengths = []
            def is_speech(self, pcm, rate):
                self.lengths.append(len(pcm))
                return len(self.lengths) == 2
        vad = SpeechVad()
        self.assertTrue(VoskDecoder._has_speech(vad, bytes(8000), 16000))
        self.assertEqual(vad.lengths, [960, 960])

    def test_vosk_continuous_20ms_speech_does_not_trigger_silence_endpoint(self):
        class Recognizer:
            def AcceptWaveform(self, pcm):
                return False
            def PartialResult(self):
                return '{"partial": "말"}'
            def FinalResult(self):
                return '{"text": "말"}'
        decoder = VoskDecoder.__new__(VoskDecoder)
        decoder.rate, decoder.rec = 16000, Recognizer()
        decoder.vad = NS(is_speech=lambda pcm, rate: True)
        decoder.last_partial, decoder.silence_ms = '', 0
        with patch('config.STT_ENDPOINT_MS', 100):
            events = [event for _ in range(10) for event in decoder.feed(bytes(640))]
        self.assertEqual(events, [('partial', '말')])
        self.assertEqual(decoder.silence_ms, 0)

    def test_unowned_non_speech_is_not_carried_into_next_turn(self):
        speakers = SpeakerTimeline()
        speakers.select(FaceTrack(1, (0, 0, 100, 100), 10), 10, True)
        speakers.select(FaceTrack(2, (0, 0, 100, 100), 10.02), 10.02, True)
        decoder = Decoder()
        router = SpeakerAudioRouter(decoder, lambda *args, **kwargs: None, speakers, vad=Vad())
        router.feed([(bytes([100, 0]) * 320, 10, 10.02, False),
                     (bytes([200, 0]) * 320, 10.02, 10.04, True)])
        router.finish()
        self.assertEqual(decoder.segments, [[200] * 320])

    def test_silence_after_speech_reaches_decoder_for_endpoint(self):
        speakers = SpeakerTimeline()
        speakers.select(FaceTrack(1, (0, 0, 100, 100), 10), 10, True)
        decoder = Decoder()
        router = SpeakerAudioRouter(decoder, lambda *args, **kwargs: None, speakers, vad=Vad())
        router.feed([(bytes([100, 0]) * 320, 10, 10.02, True),
                     (bytes(640), 10.02, 10.04, False)])
        router.finish()
        self.assertEqual(decoder.segments, [[100] * 320 + [0] * 320])

    def test_new_arrivals_keep_distinct_ids_audio_and_bubbles(self):
        tracker, speakers = FaceTracker(), SpeakerTimeline()
        a_box, b_box = (0, 0, 100, 100), (200, 0, 300, 100)
        def frame(i, boxes, openings):
            at = 10 + i / 30
            tracker.update(boxes, (640, 480), at, openings)
            speakers.observe(tracker.tracks(at), tracker.mouths, at)
            return at
        # A acquires focus before B enters.
        for i in range(30):
            frame(i, [a_box], [.02 if i % 2 else .06])
        a_id = speakers.focus(10 + 29 / 30)
        self.assertIsNotNone(a_id)
        # Detector order changes when B enters; neither IDs nor focus may swap.
        for i in range(30, 60):
            frame(i, [b_box, a_box], [.02, .02 if i % 2 else .06])
        b_id = next(t.track_id for t in tracker.tracks(10 + 59 / 30)
                    if t.track_id != a_id)
        self.assertEqual(speakers.focus(10 + 59 / 30), a_id)
        self.assertNotEqual(a_id, b_id)
        # A becomes quiet and B speaks, then both remain visible and quiet.
        for i in range(60, 90):
            frame(i, [a_box, b_box], [.02, .02 if i % 2 else .06])
        for i in range(90, 105):
            frame(i, [a_box, b_box], [.02, .02])
        self.assertEqual(speakers.focus(10 + 104 / 30), b_id)
        self.assertEqual(speakers.resolve(11.5), a_id)
        self.assertEqual(speakers.resolve(12.7), b_id)
        worker = SttWorker(queue.Queue(), caption_events=True)
        decoder = Decoder()
        router = SpeakerAudioRouter(decoder, worker._publish, speakers, vad=Vad())
        router.feed([(np.full(320, 100, np.int16).tobytes(), 11.5, 11.52, True),
                     (np.full(320, 200, np.int16).tobytes(), 12.7, 12.72, True)])
        router.finish()
        book, finals = CaptionBook(), []
        while not worker.out.empty():
            _, event = worker.out.get_nowait()
            book.update(event, speakers.face_at(event.speaker_id, event.audio_end))
            if event.kind == 'final':
                finals.append(event)
        self.assertEqual([e.speaker_id for e in finals], [a_id, b_id])
        self.assertEqual([e.text for e in finals], ['A입니다', 'B입니다'])
        self.assertEqual(decoder.segments, [[100] * 320, [200] * 320])
        self.assertEqual({s.track_id for s in book.visible_states(finals[-1].observed_at)},
                         {a_id, b_id})
        # B leaves long enough to expire. C entering the same position gets a new ID.
        for i in range(105, 126):
            frame(i, [a_box], [.02])
        self.assertIsNone(speakers.focus(10 + 125 / 30))
        at = frame(126, [a_box, b_box], [.02, .02])
        c_id = next(t.track_id for t in tracker.tracks(at) if t.track_id != a_id)
        self.assertNotIn(c_id, (a_id, b_id))
        self.assertIsNone(speakers.focus(at))
        for i in range(127, 151):
            at = frame(i, [a_box, b_box], [.02, .02 if i % 2 else .06])
        self.assertEqual(speakers.focus(at), c_id)
        # C cannot move the existing B caption anchor to its new ID.
        b_state = next(s for s in book.states.values() if s.track_id == b_id)
        old_anchor = b_state.held
        c_track = next(t for t in tracker.tracks(at) if t.track_id == c_id)
        self.assertEqual(b_state.anchor(c_track, finals[-1].observed_at).track_id, b_id)
        self.assertEqual(b_state.held, old_anchor)

    def test_adc_time_is_converted_to_application_clock(self):
        timing = NS(currentTime=500.5, inputBufferAdcTime=500.2)
        block = AudioBlock.capture(bytes(8000), 16000, timing, 100.5, 1)
        self.assertAlmostEqual(block.start, 100.2)
        self.assertAlmostEqual(block.end, 100.45)
        fallback = AudioBlock.capture(bytes(8000), 16000, None, 100.5, 2)
        self.assertEqual((fallback.start, fallback.end), (100.25, 100.5))

    def test_turn_boundary_within_one_audio_block_splits_samples_and_ids(self):
        speakers = SpeakerTimeline()
        a, b = FaceTrack(1, (0, 0, 100, 100), 10), FaceTrack(2, (200, 0, 300, 100), 10)
        speakers.select(a, 10, manual=True)
        speakers.select(b, 10.1, manual=True)
        worker = SttWorker(queue.Queue(), caption_events=True)
        decoder = Decoder()
        router = SpeakerAudioRouter(decoder, worker._publish, speakers, vad=Vad())
        pcm = np.array([100] * 1600 + [200] * 1600, np.int16).tobytes()
        prepared = router.prepare(AudioBlock(pcm, 10, 10.2, 1))
        self.assertEqual(decoder.samples, [], "Preparation must not decode buffered audio")
        router.feed(prepared)
        router.finish()
        events = []
        while not worker.out.empty():
            events.append(worker.out.get()[1])
        finals = [e for e in events if e.kind == "final"]
        self.assertEqual([e.text for e in finals], ["A입니다", "B입니다"])
        self.assertEqual([e.speaker_id for e in finals], [1, 2])
        self.assertLess(finals[0].utterance_id, finals[1].utterance_id)
        self.assertAlmostEqual(finals[0].audio_end, 10.1)
        self.assertAlmostEqual(finals[1].audio_start, 10.1)
        self.assertEqual(decoder.segments, [[100] * 1600, [200] * 1600])
        self.assertGreater(finals[0].observed_at, finals[0].audio_end)
        book = CaptionBook()
        for event in events:
            book.update(event, a if event.speaker_id == 1 else b)
        self.assertEqual({s.track_id for s in book.visible_states(finals[-1].observed_at)}, {1, 2})

    def test_short_silence_does_not_switch_to_listener_moving_in_silence(self):
        speakers = SpeakerTimeline()
        speakers.select(FaceTrack(1, (0, 0, 100, 100), 10), 10, True)
        worker = SttWorker(queue.Queue(), caption_events=True)
        router = SpeakerAudioRouter(Decoder(), worker._publish, speakers, vad=Vad())
        router.feed([(bytes([100, 0]) * 320, 10, 10.02, True)])
        speakers.select(FaceTrack(2, (200, 0, 300, 100), 10.02), 10.02, True)
        router.feed([(bytes(640), 10.02, 10.04, False)])
        self.assertEqual(router.owner, 1)
        router.cancel()
        events = []
        while not worker.out.empty():
            events.append(worker.out.get()[1])
        self.assertEqual(events[-1].kind, "cancel")
        self.assertEqual(events[-1].speaker_id, 1)
        self.assertEqual(len({e.utterance_id for e in events}), 1)
        self.assertIsNone(router.start)

    def test_worker_tts_gate_clears_buffered_audio_and_cancels_owned_partial(self):
        ready, callback = threading.Event(), {}
        class Stream:
            def __init__(self, callback, **kwargs):
                self.callback = callback
            def __enter__(self):
                callback["send"] = self.callback
                ready.set()
                return self
            def __exit__(self, *args):
                pass
        class RecordingDecoder(Decoder):
            def __init__(self):
                super().__init__()
                self.received = []
            def feed(self, pcm):
                self.received.extend(np.frombuffer(pcm, np.int16).tolist())
                return super().feed(pcm)
        gate = NS(open=True)
        gate.is_open = lambda: gate.open
        decoder = RecordingDecoder()
        worker = SttWorker(queue.Queue(), gate=gate, caption_events=True,
                           speakers=SpeakerTimeline())
        worker._decoder = lambda: decoder
        with patch.dict("sys.modules", {"sounddevice": NS(RawInputStream=Stream)}), \
                patch("webrtcvad.Vad", return_value=Vad()):
            worker.start()
            try:
                self.assertTrue(ready.wait(3))
                callback["send"](np.full(4000, 100, np.int16).tobytes(), 4000, None, None)
                # The .65s visual-confirmation buffer has not reached the decoder.
                gate.open = False
                callback["send"](bytes(8000), 4000, None, None)
                threading.Event().wait(.1)
                self.assertEqual(decoder.received, [])
                gate.open = True
                callback["send"](np.full(4000, 200, np.int16).tobytes(), 4000, None, None)
                kind, partial = worker.out.get(timeout=3)
                self.assertEqual(kind, "partial")
                self.assertTrue(partial.speaker_resolved)
                gate.open = False
                callback["send"](bytes(8000), 4000, None, None)
                while True:
                    kind, event = worker.out.get(timeout=3)
                    if kind == "cancel":
                        break
                self.assertEqual(partial.utterance_id, event.utterance_id)
                self.assertEqual(set(decoder.received), {200})
            finally:
                worker.stop()
                worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_worker_audio_loss_cancels_instead_of_joining_sentences(self):
        class Stream:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        worker = SttWorker(queue.Queue(), caption_events=True, speakers=SpeakerTimeline())
        decoder = Decoder()
        worker._decoder = lambda: decoder
        at = clock.now() - 2
        worker._audio.put(AudioBlock(np.full(320, 100, np.int16).tobytes(), at, at + .02, 1))
        # Serial 2 was dropped before the worker could decode it.
        worker._audio.put(AudioBlock(np.full(320, 200, np.int16).tobytes(), at + .04, at + .06, 3))
        with patch.dict("sys.modules", {"sounddevice": NS(RawInputStream=Stream)}), \
                patch("webrtcvad.Vad", return_value=Vad()):
            worker.start()
            try:
                events = [worker.out.get(timeout=3)[1] for _ in range(3)]
                self.assertEqual([e.kind for e in events], ["partial", "cancel", "partial"])
                self.assertEqual(events[0].utterance_id, events[1].utterance_id)
                self.assertGreater(events[2].utterance_id, events[1].utterance_id)
                self.assertEqual(decoder.samples, [200] * 320)
            finally:
                worker.stop()
                worker.join(2)
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
