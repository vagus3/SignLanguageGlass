"""Spacing preserves recognized characters and caption attribution."""
import _path  # noqa: F401
import queue
import unittest
from unittest.mock import Mock, patch

from src.korean_spacing import KoreanSpacing
from src.stt_worker import SttWorker


class SpacingTests(unittest.TestCase):
    def test_real_korean_spacing(self):
        formatter = KoreanSpacing()
        self.assertEqual(formatter.format("지금머리가너무아파요"), "지금 머리가 너무 아파요")
        self.assertIsNotNone(formatter._kiwi)
        self.assertEqual(formatter.format("안녕하세요저는학생입니다"), "안녕하세요 저는 학생입니다")

    def test_reject_spelling_changes_and_survive_failure(self):
        formatter = KoreanSpacing()
        formatter._prepared = True
        formatter._kiwi = Mock()
        formatter._kiwi.space.return_value = "오늘 날씨가 좋아요"
        self.assertEqual(formatter.format("오늘날씨가조아요"), "오늘날씨가조아요")
        formatter._kiwi.space.side_effect = RuntimeError("spacing failure")
        with self.assertLogs("src.korean_spacing", level="WARNING"):
            self.assertEqual(formatter.format("머리가아파요"), "머리가아파요")
        self.assertEqual(formatter.format(""), "")
        self.assertEqual(formatter.format("ABC 123"), "ABC 123")

    def test_missing_package_preserves_original(self):
        formatter = KoreanSpacing()
        with patch.dict("sys.modules", {"kiwipiepy": None}), \
                self.assertLogs("src.korean_spacing", level="WARNING"):
            self.assertEqual(formatter.format("머리가아파요"), "머리가아파요")

    def test_partial_final_and_cancel_keep_ownership(self):
        worker = SttWorker(queue.Queue(), caption_events=True)
        attribution = dict(audio_start=10., audio_end=11., speaker_id=7,
                           speaker_resolved=True)
        for kind in ("partial", "final"):
            worker._publish(kind, "지금머리가너무아파요", **attribution)
        events = [worker.out.get_nowait()[1] for _ in range(2)]
        self.assertEqual(events[0].utterance_id, events[1].utterance_id)
        for event in events:
            self.assertEqual(event.text, "지금 머리가 너무 아파요")
            self.assertEqual((event.speaker_id, event.audio_start, event.audio_end),
                             (7, 10., 11.))
        worker._publish("cancel", "")
        self.assertEqual(worker.out.get_nowait()[1].text, "")


if __name__ == "__main__":
    unittest.main()
