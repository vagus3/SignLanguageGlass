"""FIFO speech, Korean voice selection, engine errors and COM thread ownership."""
import _path  # noqa: F401
import sys
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from src.audio_gate import AudioGate
from src.tts_player import TtsPlayer, is_korean_voice


class Engine:
    def __init__(self, voices, outcome="ok"):
        self.voices, self.outcome = voices, outcome
        self.callbacks, self.spoken, self.properties = {}, [], {}

    def getProperty(self, name):
        return self.voices

    def setProperty(self, name, value):
        self.properties[name] = value

    def connect(self, topic, callback):
        self.callbacks[topic] = callback
        return topic

    def disconnect(self, token):
        del self.callbacks[token]

    def say(self, text):
        self.spoken.append(text)

    def runAndWait(self):
        if self.outcome == "error":
            self.callbacks["error"](None, RuntimeError("device unavailable"))
        elif self.outcome != "silent":
            self.callbacks["finished-utterance"](None, self.outcome == "ok")

    def stop(self):
        pass


class TtsWorkerTests(unittest.TestCase):
    def test_language_metadata_and_names_without_false_positives(self):
        for voice in (NS(id="voice", name="voice", languages=[b"\x05ko-kr"]),
                      NS(id="TTS_MS_KO-KR_HEAMI_11.0", name="Heami"),
                      NS(id="voice", name="Yuna")):
            self.assertTrue(is_korean_voice(voice))
        for voice in (NS(id="voice", name="Kyoko", languages=[b"ja"]),
                      NS(id="voice", name="Slovak", languages=["sk"]),
                      NS(id="voice", name="English", languages=["en-US"])):
            self.assertFalse(is_korean_voice(voice))

    def test_fifo_does_not_drop_sentences_already_queued(self):
        player = TtsPlayer(preload=False)
        spoken = []
        player._speak_fallback = spoken.append
        for text in ["첫 문장", "두 번째 문장", "세 번째 문장"]:
            player.say(text)
        player.start()
        try:
            events = [player.events.get(timeout=2) for _ in range(3)]
            self.assertTrue(all(event["ok"] for event in events))
            self.assertEqual(spoken, ["첫 문장", "두 번째 문장", "세 번째 문장"])
        finally:
            player.stop()
            player.join(timeout=2)
        self.assertFalse(player.is_alive())

    def play_engine(self, engine, count=1):
        com_calls = []
        com = NS(CoInitialize=lambda: com_calls.append(("init", threading.get_ident())),
                 CoUninitialize=lambda: com_calls.append(("close", threading.get_ident())))
        gate = AudioGate(tail_sec=0)
        player = TtsPlayer(preload=False, gate=gate)
        with patch.object(sys, "platform", "win32"), \
                patch.dict(sys.modules, {"pyttsx3": NS(init=lambda: engine), "pythoncom": com}):
            for i in range(count):
                player.say(f"테스트 {i}")
            player.start()
            try:
                events = [player.events.get(timeout=2) for _ in range(count)]
            finally:
                player.stop()
                player.join(timeout=2)
        self.assertTrue(gate.is_open(), "Failure must always release the microphone gate")
        self.assertFalse(player.is_alive())
        self.assertEqual([name for name, _ in com_calls], ["init", "close"])
        self.assertEqual(com_calls[0][1], com_calls[1][1])
        self.assertNotEqual(com_calls[0][1], threading.get_ident())
        return events

    def test_korean_voice_selected_and_worker_reused(self):
        engine = Engine([NS(id="ja", name="Kyoko"), NS(id="ko", name="Heami")])
        events = self.play_engine(engine, count=2)
        self.assertTrue(all(e["ok"] for e in events))
        self.assertEqual(engine.properties["voice"], "ko")
        self.assertEqual(engine.spoken, ["테스트 0", "테스트 1"])
        self.assertFalse(engine.callbacks)

    def test_missing_korean_voice_is_an_explicit_error(self):
        engine = Engine([NS(id="en", name="English")])
        result = self.play_engine(engine)[0]
        self.assertFalse(result["ok"])
        self.assertIn("한국어", result["error"])
        self.assertEqual(engine.spoken, [])

    def test_async_error_cancellation_and_silent_engine_are_not_success(self):
        for outcome in ["error", "cancel", "silent"]:
            with self.subTest(outcome=outcome):
                events = self.play_engine(Engine([NS(id="ko", name="Heami")], outcome), count=2)
                self.assertTrue(all(not event["ok"] for event in events))

    def test_failed_wav_playback_uses_os_fallback(self):
        player = TtsPlayer(preload=False)
        player._lookup = lambda text: ([1, 2], 16000)
        player._speak_fallback = Mock()
        with patch.dict(sys.modules, {"sounddevice": NS(play=Mock(side_effect=OSError("device")))}):
            player._play("안녕하세요")
        player._speak_fallback.assert_called_once_with("안녕하세요")


if __name__ == "__main__":
    unittest.main()
