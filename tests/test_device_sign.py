"""CTC word reading reaches TTS without a server; rejected segments stay silent."""
import _path  # noqa: F401
import contextlib
import io
import sys
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import main
from src import clock
from src.device_sign import DeviceSignClient
from src.translation import Glossary
from src.tts_player import TtsPlayer


def recognizer(glosses=None, score=.95, labels=None):
    return NS(labels=labels or ["<blank>", "병원", "검사_0029"], feature_set="2d",
              predict=lambda sequence: (["병원", "검사_0029"] if glosses is None else glosses, score))


class DeviceSignTests(unittest.TestCase):
    def test_words_reach_real_worker_once_without_network_or_mock_sentence(self):
        played = []
        received = []
        deadline = time.monotonic() + 10

        class Recorder:
            active = False
            error = None

            def start(self, pre_roll=0.0):
                self.active = True

            def finish(self):
                self.active = False
                return "captured-sequence", clock.now()

            def cancel(self):
                self.active = False

        class Vision:
            def __init__(self, out, **kwargs):
                self.commands = kwargs["segment_events"]
                assert kwargs["segment_recorder"] is not None
                assert not kwargs["enable_sign"]  # Use CTC, not the 246-feature LSTM.

            def start(self):
                self.commands.put("start")
                self.commands.put("stop")

            def stop(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        model = recognizer()
        def predict(sequence):
            received.append(sequence)
            return ["병원", "검사_0029"], .95
        model.predict = predict

        def wait_key(delay):
            if played:
                return ord("q")
            if time.monotonic() > deadline:
                raise AssertionError("The recognized words never reached TTS")
            time.sleep(.005)
            return -1

        argv = ["main.py", "--sign-backend", "device", "--onnx", "model.onnx",
                "--no-stt", "--fixed-caption", "--display", "null"]
        with patch.object(sys, "argv", argv), \
                patch("src.ctc_runtime.OnnxRecognizer", return_value=model), \
                patch("src.cloud_sign.SegmentRecorder", Recorder), \
                patch.object(main, "VisionThread", Vision), \
                patch("src.predictor.SignPredictor") as old_predictor, \
                patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network")) as network, \
                patch.object(TtsPlayer, "_preload"), \
                patch.object(TtsPlayer, "_play", lambda self, text, *args: played.append(text)), \
                patch("cv2.imshow"), patch("cv2.destroyAllWindows"), \
                patch("cv2.waitKey", wait_key), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main.main()
        self.assertEqual(received, ["captured-sequence"])
        self.assertEqual(played, ["병원 검사"])
        self.assertIn("[수어] 병원 검사", output.getvalue())
        network.assert_not_called()
        old_predictor.assert_not_called()

    def test_empty_uncertain_and_invalid_predictions_never_speak(self):
        for glosses, score, status in [
            ([], 0, "no_sign"), (["병원"], .2, "uncertain"),
            (["병원"], float("nan"), "uncertain"), (["병원"], 1.1, "uncertain"),
            (["unknown"], .99, "uncertain"),
        ]:
            with self.subTest(glosses=glosses, score=score):
                client = DeviceSignClient(recognizer(glosses, score))
                result = client.translate(client.payload(None, "session", "segment"))
                self.assertEqual(result["status"], status)
                self.assertEqual(result["text"], "")
                self.assertFalse(result["auto_speech"])

    def test_explicit_mapping_and_repeated_words_keep_meaning_and_order(self):
        model = recognizer(["G1", "G0", "G1"], labels=["<blank>", "G0", "G1"])
        with self.assertRaisesRegex(ValueError, "--sign-mapping"):
            DeviceSignClient(model)
        client = DeviceSignClient(model, Glossary({"G0": "없다", "G1": "통증"}))
        result = client.translate(client.payload(None, "session", "segment"))
        self.assertEqual(result["text"], "통증 없다 통증")
        self.assertTrue(result["auto_speech"])
        self.assertFalse(result["mock"])

    def test_invalid_cli_stops_before_opening_camera(self):
        for argv in (["--sign-backend", "device"],
                     ["--sign-mapping", "mapping.json"],
                     ["--sign-backend", "device", "--onnx", "model.onnx", "--sign-model", "old.onnx"]):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["main.py", *argv]), \
                    patch.object(main, "VisionThread") as vision, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main.main()
                self.assertEqual(result.exception.code, 2)
                vision.assert_not_called()


if __name__ == "__main__":
    unittest.main()
