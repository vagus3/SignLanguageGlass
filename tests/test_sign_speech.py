"""Confirmed signs survive dropped previews and reach the real TTS worker once."""
import _path  # noqa: F401
import contextlib
import io
import queue
import sys
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import main
from src import landmarks as L
from src.tts_player import TtsPlayer


class SignSpeechTests(unittest.TestCase):
    def test_model_failure_is_fatal_before_camera_or_audio_start(self):
        for failure in (FileNotFoundError("missing model"), RuntimeError("wrong features")):
            with self.subTest(failure=failure), \
                    patch.object(sys, "argv", ["main.py", "--no-stt"]), \
                    patch("src.predictor.SignPredictor", side_effect=failure), \
                    patch("main.VisionThread.start") as vision, \
                    patch("src.tts_player.TtsPlayer.start") as tts, \
                    contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as result:
                    main.main()
                self.assertEqual(result.exception.code, 2)
                self.assertIn(str(failure), error.getvalue())
                self.assertIn("--test-tts", error.getvalue())
                vision.assert_not_called()
                tts.assert_not_called()

    def test_tts_only_does_not_load_sign_model_or_open_camera(self):
        with patch.object(sys, "argv", ["main.py", "--test-tts", "안녕하세요"]), \
                patch("src.tts_player.test_output", return_value=True) as output, \
                patch("src.predictor.SignPredictor") as predictor, \
                patch("src.camera.CameraReader") as camera:
            with self.assertRaises(SystemExit) as result:
                main.main()
            self.assertEqual(result.exception.code, 0)
            output.assert_called_once_with("안녕하세요")
            predictor.assert_not_called()
            camera.assert_not_called()

    def test_slow_gui_keeps_confirmed_words_and_delivers_sentences_once(self):
        now, played, workers = [100.0], [], []
        frame = np.zeros((120, 160, 3), np.uint8)
        result = NS(left_hand_landmarks=None, right_hand_landmarks=None,
                    pose_landmarks=None, face_landmarks=None, multi_hand_landmarks=None)
        words = iter(["이름", "무엇", "감사합니다", None, None])
        predictor = NS(labels=["이름", "무엇", "감사합니다"], last_ms=1,
                       last_frame_t=100.0, last_fps=30, last_conf=.95, last_raw="감사합니다",
                       push=lambda *args, **kwargs: next(words))

        class Camera:
            count = 0
            closed = False

            def ok(self):
                return True

            def read(self):
                self.count += 1
                if self.count == 5:
                    workers[0].stop()
                return True, frame.copy()

            def release(self):
                self.closed = True

        camera = Camera()
        holistic = NS(process=lambda rgb: result, close=lambda: None)
        real_vision = main.VisionThread

        class SlowGuiVision(real_vision):
            def start(self):
                workers.append(self)
                self.run()  # GUI has not consumed a single preview yet.
                self.assert_latest_preview()

            def assert_latest_preview(self):
                assert self.out.qsize() == 1
                assert self.word_events.qsize() == 3

            def join(self, timeout=None):
                pass

        def wait_key(delay):
            now[0] += .25
            time.sleep(.002)  # Let the actual TTS worker consume its queue.
            if len(played) == 2:
                return ord("q")
            if now[0] > 130:
                raise AssertionError(f"Confirmed words never reached TTS: {played}")
            return -1

        with patch.object(sys, "argv", ["main.py", "--no-stt", "--fixed-caption",
                                        "--debug", "--display", "null"]), \
                patch("src.clock.now", lambda: now[0]), \
                patch("src.predictor.SignPredictor", return_value=predictor), \
                patch.object(main, "VisionThread", SlowGuiVision), \
                patch("src.camera.CameraReader", return_value=camera), \
                patch.object(L, "make_holistic", return_value=holistic), \
                patch.object(L, "make_preview_hands", return_value=holistic), \
                patch.object(L, "extract", return_value=np.zeros(L.FEATURE_DIM)), \
                patch.object(TtsPlayer, "_preload"), \
                patch.object(TtsPlayer, "_play", lambda self, text, *args: played.append(text)), \
                patch("cv2.imshow"), patch("cv2.destroyAllWindows"), \
                patch("cv2.waitKey", wait_key), contextlib.redirect_stdout(io.StringIO()) as log:
            main.main()
        self.assertEqual(played, ["이름이 뭐예요?", "감사합니다"])
        self.assertEqual(log.getvalue().count("[수어]"), 3)
        self.assertIn("[수어 상태]", log.getvalue())
        self.assertTrue(camera.closed)


if __name__ == "__main__":
    unittest.main()
