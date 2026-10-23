"""Exposure rejection, driver clamping and CLI overrides without real hardware."""
import _path  # noqa: F401
import contextlib
import io
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import config as C
import main
from src.camera import _apply_manual_exposure, CameraReader


class ExposureTests(unittest.TestCase):
    def setUp(self):
        settings = patch.multiple(C, IS_WIN=True, IS_MAC=False,
                                  CAM_MANUAL_EXPOSURE=True, CAM_EXPOSURE=-8, CAM_GAIN=0)
        settings.start()
        self.addCleanup(settings.stop)

    def apply(self, rejected=None, exposure=-8, auto=0, gain=0):
        cap = Mock()
        cap.set.side_effect = lambda prop, value: prop != rejected
        values = {cv2.CAP_PROP_EXPOSURE: exposure, cv2.CAP_PROP_AUTO_EXPOSURE: auto,
                  cv2.CAP_PROP_GAIN: gain}
        cap.get.side_effect = values.__getitem__
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _apply_manual_exposure(cap, cv2, verbose=True)
        return cap, output.getvalue()

    def test_manual_mode_precedes_exposure_and_gain(self):
        cap, output = self.apply()
        self.assertEqual([call.args for call in cap.set.call_args_list],
                         [(cv2.CAP_PROP_AUTO_EXPOSURE, 0),
                          (cv2.CAP_PROP_EXPOSURE, -8), (cv2.CAP_PROP_GAIN, 0)])
        self.assertIn("수동 노출 적용 (", output)
        self.assertIn("게인 적용 (", output)

    def test_rejected_set_is_not_reported_as_success(self):
        for prop in (cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_EXPOSURE):
            with self.subTest(prop=prop):
                _, output = self.apply(rejected=prop)
                self.assertIn("수동 노출 적용 확인 실패", output)
        _, output = self.apply(rejected=cv2.CAP_PROP_GAIN)
        self.assertIn("게인 적용 확인 실패", output)

    def test_clamped_exposure_and_remaining_auto_are_detected(self):
        for values in ({"exposure": -7}, {"auto": 1}):
            with self.subTest(values=values):
                _, output = self.apply(**values)
                self.assertIn("수동 노출 적용 확인 실패", output)
        _, output = self.apply(gain=100)
        self.assertIn("게인 적용 확인 실패", output)

    def test_unset_gain_is_left_alone(self):
        with patch.object(C, "CAM_GAIN", None):
            cap, _ = self.apply()
        self.assertNotIn(cv2.CAP_PROP_GAIN, [c.args[0] for c in cap.set.call_args_list])

    def test_v4l2_uses_its_manual_mode(self):
        with patch.object(C, "IS_WIN", False):
            cap, output = self.apply(auto=1)
        self.assertEqual(cap.set.call_args_list[0].args, (cv2.CAP_PROP_AUTO_EXPOSURE, 1))
        self.assertIn("수동 노출 적용 (", output)

    def test_reopen_reapplies_settings(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: -8 if prop == cv2.CAP_PROP_EXPOSURE else 0
        with patch("cv2.VideoCapture", return_value=cap):
            reader = CameraReader(camera_index=2)
            reader.release()
            cap.reset_mock()
            reader._reopen()
            cap.set.assert_any_call(cv2.CAP_PROP_EXPOSURE, -8)
            cap.set.assert_any_call(cv2.CAP_PROP_GAIN, 0)
            reader.release()

    def test_cli_applies_before_workers_start(self):
        class BeforeWorkers(Exception):
            pass

        with patch.object(C, "CAM_MANUAL_EXPOSURE", False), \
                patch.object(sys, "argv", ["main.py", "--exposure", "-9", "--gain", "0"]), \
                patch("src.hud.load_calib", side_effect=BeforeWorkers), \
                patch("main.VisionThread.start") as start:
            with self.assertRaises(BeforeWorkers):
                main.main()
            self.assertEqual(C.CAM_EXPOSURE, -9)
            self.assertEqual(C.CAM_GAIN, 0)
            self.assertTrue(C.CAM_MANUAL_EXPOSURE)
            start.assert_not_called()

    def test_cli_rejects_nonfinite_values(self):
        for flag, value in (("--exposure", "nan"), ("--gain", "inf")):
            with self.subTest(flag=flag), \
                    patch.object(sys, "argv", ["main.py", flag, value]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
