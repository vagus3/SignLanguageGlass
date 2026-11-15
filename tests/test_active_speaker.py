"""Alternating mouths, silence, laughter/overlap, occlusion and explicit ownership."""
import _path  # noqa: F401
import unittest
import queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from src.active_speaker import SpeakerTimeline
from src.caption_history import CaptionHistory
from src.caption_state import CaptionBook, CaptionEvent
from src.face_activity import mesh_observations
from src.face_tracker import FaceTrack, FaceTracker
from src.video_overlay import VideoOverlay


BOXES = [(80, 140, 180, 300), (430, 140, 530, 300)]


def tracks(t):
    return [FaceTrack(i + 1, box, t) for i, box in enumerate(BOXES)]


def observe(timeline, i, a, b, **kwargs):
    t = 10 + i * .05
    timeline.observe(tracks(t), {1: a, 2: b}, t, **kwargs)
    return t


class SpeakerTests(unittest.TestCase):
    def test_intermittent_mouth_motion_acquires_and_holds_one_face(self):
        timeline = SpeakerTimeline()
        for i in range(90):
            t = 10 + i / 30
            phase = i % 9
            opening = .02 + (.01 * (phase % 2) if phase < 5 else 0)
            timeline.observe([FaceTrack(1, BOXES[0], t)], {1: opening}, t)
        self.assertEqual(timeline._at(t, settled=False), 1)
        for j in range(1, 13):
            now = t + j / 30
            timeline.observe([FaceTrack(1, BOXES[0], now)], {1: .02}, now)
        self.assertEqual(timeline._at(now, settled=False), 1)
        timeline.observe([FaceTrack(1, BOXES[0], now + 1)], {1: .02}, now + 1)
        self.assertEqual(timeline.focus(now + 1), 1)
        # A visible but silent face remains selected beyond the old timeout.
        for j in range(1, 61):
            at = now + 1 + j / 30
            timeline.observe([FaceTrack(1, BOXES[0], at)], {1: .02}, at)
        self.assertEqual(timeline.focus(at), 1)
        timeline.observe([], {}, at + .05)
        self.assertIsNone(timeline.focus(at + .05))
        # Returning faces receive new IDs and must qualify again.
        timeline.observe([FaceTrack(3, BOXES[0], at + .1)], {3: .02}, at + .1)
        self.assertIsNone(timeline.focus(at + .1))

    def test_single_open_close_is_not_repeated_speech(self):
        timeline = SpeakerTimeline()
        for i in range(30):
            t = 10 + i / 30
            timeline.observe([FaceTrack(1, BOXES[0], t)],
                             {1: .15 if 5 <= i <= 8 else .02}, t)
        self.assertIsNone(timeline.resolve(10.25))

    def test_alternating_speakers_are_bound_by_capture_time(self):
        timeline = SpeakerTimeline()
        for i in range(21):
            observe(timeline, i, .02 if i % 2 else .06, .02)
        for i in range(21, 45):
            observe(timeline, i, .02, .02 if i % 2 else .06)
        self.assertEqual(timeline.resolve(10.5), 1)
        self.assertEqual(timeline.resolve(11.7), 2)
        # Delayed A results still resolve to A after B is selected.
        self.assertEqual(timeline.face_at(1, 10.8).track_id, 1)
        timeline.observe_audio(12.2, True)
        self.assertEqual(timeline.focus(12.2), 2)
        self.assertIsNone(timeline.focus(13))

    def test_static_open_mouth_and_brief_listener_movement_do_not_select(self):
        timeline = SpeakerTimeline()
        for i in range(25):
            observe(timeline, i, .15, .02 + (.06 if i == 10 else 0))
        for t in (10.2, 10.5, 10.7, 11):
            self.assertIsNone(timeline.resolve(t))

    def test_short_pause_holds_speaker_but_similar_motion_is_unknown(self):
        timeline = SpeakerTimeline()
        for i in range(16):
            observe(timeline, i, .02 if i % 2 else .06, .02)
        for i in range(16, 23):
            observe(timeline, i, .02, .02)
        self.assertEqual(timeline.resolve(10.8), 1)
        for i in range(23, 42):
            value = .02 if i % 2 else .06
            observe(timeline, i, value, value)
        self.assertIsNone(timeline.resolve(11.5))
        self.assertIsNone(timeline.resolve(12.5), "Stale faces cannot own new audio")

    def test_resuming_speech_cannot_relabel_an_ambiguous_interval(self):
        timeline = SpeakerTimeline()
        for i in range(20):
            observe(timeline, i, .02 if i % 2 else .06, .02)
        for i in range(20, 30):
            value = .02 if i % 2 else .06
            observe(timeline, i, value, value)
        for i in range(30, 50):
            observe(timeline, i, .02 if i % 2 else .06, .02)
        self.assertIsNone(timeline.resolve(11.3))
        self.assertEqual(timeline.resolve(12), 1)

    def test_listener_laugh_does_not_steal_an_ongoing_turn(self):
        timeline = SpeakerTimeline()
        for i in range(40):
            observe(timeline, i, .02 if i % 2 else .06, .15 if i == 20 else .02)
        for t in np.arange(10.6, 11.6, .05):
            self.assertNotEqual(timeline.resolve(t), 2)

    def test_manual_lock_ignores_other_mouth_and_loss_does_not_switch(self):
        timeline = SpeakerTimeline()
        for i in range(20):
            observe(timeline, i, .02, .02 if i % 2 else .06, manual=True, selected_id=1)
        self.assertEqual(timeline.resolve(10.6), 1)
        timeline.observe([tracks(11)[1]], {2: .06}, 11, manual=True, selected_id=1)
        self.assertIsNone(timeline.resolve(11))
        timeline.select(None, 11.05, manual=False)
        for i in range(22, 40):
            observe(timeline, i, .02, .02 if i % 2 else .06)
        self.assertEqual(timeline.resolve(11.5), 2)

    def test_motion_stays_with_geometric_id_when_detection_order_changes(self):
        tracker = FaceTracker()
        tracker.update(BOXES, (640, 480), 10, [.01, .05])
        tracker.update(list(reversed(BOXES)), (640, 480), 10.05, [.06, .02])
        self.assertEqual(tracker.mouths, {2: .06, 1: .02})

    def test_lips_are_scale_normalized_and_invalid_meshes_are_ignored(self):
        points = [NS(x=.5, y=.5) for _ in range(468)]
        for index, xy in {234: (.3, .5), 454: (.7, .5), 13: (.5, .5),
                          14: (.5, .54), 10: (.5, .2), 152: (.5, .8)}.items():
            points[index] = NS(x=xy[0], y=xy[1])
        result = NS(multi_face_landmarks=[NS(landmark=points)])
        boxes, ratios = mesh_observations(result, 640, 480)
        large_boxes, large_ratios = mesh_observations(result, 1280, 960)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(ratios[0], large_ratios[0])
        np.testing.assert_allclose(np.array(boxes) * 2, large_boxes)
        points[13].x = float("nan")
        self.assertEqual(mesh_observations(result, 640, 480), ([], []))

    def test_both_camera_workers_emit_mouths_from_the_same_face_observation(self):
        from main import FaceThread, VisionThread
        points = [NS(x=.5, y=.5) for _ in range(468)]
        for index, xy in {234: (.3, .5), 454: (.7, .5), 13: (.5, .5),
                          14: (.5, .54), 10: (.5, .2), 152: (.5, .8)}.items():
            points[index] = NS(x=xy[0], y=xy[1])
        result = NS(multi_face_landmarks=[NS(landmark=points)])
        expected = mesh_observations(result, 640, 480)
        for kind in ("vision", "face"):
            out = queue.Queue()
            worker = (VisionThread(out, enable_sign=False, track_mouths=True) if kind == "vision"
                      else FaceThread(out, 1, track_mouths=True))
            def read():
                worker.stop()
                return True, np.zeros((480, 640, 3), np.uint8)
            camera = NS(ok=lambda: True, read=read, release=Mock())
            detector = NS(process=lambda rgb: result, close=Mock())
            with patch("src.camera.CameraReader", return_value=camera), \
                    patch("src.face_activity.MouthDetector", return_value=detector):
                worker.run()
            _, payload = out.get_nowait()
            self.assertEqual((payload["face_boxes"], payload["mouth_openings"]), expected)
            camera.release.assert_called_once()
            detector.close.assert_called_once()

    def test_late_final_stays_with_a_while_b_partial_and_bubble_survive(self):
        book = CaptionBook()
        a, b = tracks(10)
        event = lambda kind, text, uid, who, at: CaptionEvent(
            kind, text, uid, at, audio_start=9, audio_end=10,
            speaker_id=who, speaker_resolved=True)
        book.update(event("partial", "첫 번째", 1, 1, 10), a)
        book.update(event("partial", "두 번째", 2, 2, 10.1), b)
        state_a = book.update(event("final", "첫 번째 문장입니다", 1, 1, 10.2), b)
        self.assertEqual(state_a.track_id, 1)
        self.assertEqual(book.latest.text, "두 번째")
        self.assertEqual({s.track_id for s in book.visible_states(10.3)}, {1, 2})
        self.assertEqual(state_a.anchor(b, 10.3).track_id, 1)
        # Explicitly uncertain audio must never acquire a later visible face.
        unknown = book.update(event("partial", "겹친 말", 3, None, 10.3), a)
        book.update(event("final", "겹친 말입니다", 3, None, 10.4), b)
        self.assertIsNone(unknown.track_id)
        self.assertIsNone(unknown.anchor(b, 10.5))
        history = CaptionHistory()
        history.append(event("final", "B", 2, 2, 10.5), 2)
        history.append(event("final", "A", 1, 1, 10.6), 1)
        self.assertEqual([r.track_id for r in history.records], [1, 2])
        self.assertFalse(history.append(event("final", "duplicate", 1, 1, 11), 1))

    def test_two_bubbles_render_without_overlap_and_without_moving_ownership(self):
        book = CaptionBook()
        for face in tracks(10):
            book.update(CaptionEvent("final", "각 사람의 마지막 문장", face.track_id, 10,
                                     speaker_id=face.track_id, speaker_resolved=True), face)
        overlay, rectangles = VideoOverlay(), []
        original = overlay._bubble
        def record(lines, size, partial, bx, by, bw, bh, tail):
            rectangles.append((bx, by, bx + bw, by + bh))
            return original(lines, size, partial, bx, by, bw, bh, tail)
        overlay._bubble = record
        frame = np.zeros((480, 640, 3), np.uint8)
        rendered = overlay.render(frame, book.latest, tracks(10)[1], 10.2,
                                  tracks=tracks(10.2), captions=book.visible_states(10.2),
                                  speaker_mode=True, selected_id=2, mirror=True)
        self.assertEqual(len(rectangles), 2)
        a, b = rectangles
        self.assertEqual(max(0, min(a[2], b[2]) - max(a[0], b[0])) *
                         max(0, min(a[3], b[3]) - max(a[1], b[1])), 0)
        self.assertTrue(rendered.any())
        self.assertFalse(frame.any())

    def test_final_without_prior_partial_can_arrive_after_a_newer_speaker(self):
        book = CaptionBook(limit=2)
        a, b = tracks(10)
        make = lambda kind, uid, who: CaptionEvent(kind, str(uid), uid, 10,
                                                  speaker_id=who, speaker_resolved=True)
        book.update(make("partial", 2, 2), b)
        self.assertIsNotNone(book.update(make("final", 1, 1), a))
        self.assertEqual(book.latest.utterance_id, 2)
        self.assertEqual({s.track_id for s in book.visible_states(10)}, {1, 2})
        self.assertIsNone(book.update(make("partial", 2, 1), a), "Ownership cannot change")
        book.update(make("partial", 3, 2), b)
        self.assertIsNone(book.update(make("final", 1, 1), a), "An evicted ID cannot reappear")


class ContinuityTests(unittest.TestCase):
    """Brief detection misses, movement, entries, exits and overlap (PR3 plan table)."""

    def speak(self, timeline, tracker, i, boxes, mouths):
        t = 10 + i * .05
        tracker.update(boxes, (640, 480), t, mouths)
        timeline.observe(tracker.tracks(t), tracker.mouths, t)
        return t

    def test_brief_miss_keeps_id_selection_and_audio_owner(self):
        logs = []
        tracker, timeline = FaceTracker(log=logs.append), SpeakerTimeline(log=logs.append)
        for i in range(12):
            self.speak(timeline, tracker, i, BOXES[:1], [.02 if i % 2 else .06])
        self.assertEqual(timeline.resolve(10.4), 1)
        for i in (12, 13):                      # two frames without a detection
            self.speak(timeline, tracker, i, [], [])
        for i in range(14, 20):
            self.speak(timeline, tracker, i, BOXES[:1], [.02 if i % 2 else .06])
        self.assertEqual(timeline.resolve(10.62), 1, "audio during the miss stays with A")
        self.assertEqual([t.track_id for t in tracker.tracks(11)], [1])
        self.assertEqual(timeline.resolve(10.78), 1)
        self.assertTrue(any("잠시 놓침" in line for line in logs))
        self.assertTrue(any("다시 잡힘" in line for line in logs))

    def test_gap_difference_is_not_speech(self):
        timeline = SpeakerTimeline()
        tracker = FaceTracker()
        for i in range(6):
            self.speak(timeline, tracker, i, BOXES[:1], [.02])
        self.speak(timeline, tracker, 6, [], [])
        self.speak(timeline, tracker, 9, BOXES[:1], [.09])   # mouth changed during the gap
        self.speak(timeline, tracker, 10, BOXES[:1], [.09])
        self.assertIsNone(timeline.resolve(10.5))

    def test_moving_face_keeps_id_but_far_jump_does_not(self):
        tracker = FaceTracker()
        tracker.update([(100, 100, 200, 200)], (640, 480), 10)
        moved = tracker.update([(160, 110, 260, 210)], (640, 480), 10.05)  # little overlap
        self.assertEqual(moved.track_id, 1)
        tracker.update([(500, 100, 600, 200)], (640, 480), 10.1)
        far = [t for t in tracker.tracks(10.1) if t.seen_at == 10.1]
        self.assertEqual(len(far), 1)
        self.assertNotEqual(far[0].track_id, 1, "a jump across the image is another face")

    def test_entry_keeps_speaker_and_new_speaker_needs_confirmation(self):
        tracker, timeline = FaceTracker(), SpeakerTimeline()
        for i in range(12):
            self.speak(timeline, tracker, i, BOXES[:1], [.02 if i % 2 else .06])
        for i in range(12, 16):                 # B enters silently
            self.speak(timeline, tracker, i, BOXES, [.02 if i % 2 else .06, .02])
        self.assertEqual({t.track_id for t in tracker.tracks(10.8)}, {1, 2})
        for i in range(16, 40):                 # A stops, B speaks
            self.speak(timeline, tracker, i, BOXES, [.02, .02 if i % 2 else .06])
        self.assertEqual(timeline.resolve(10.75), 1, "B entering did not take A's turn")
        self.assertEqual(timeline.resolve(11.6), 2)

    def test_exit_then_new_person_gets_new_id_and_no_caption_inheritance(self):
        tracker, timeline = FaceTracker(), SpeakerTimeline()
        book = CaptionBook()
        for i in range(12):
            t = self.speak(timeline, tracker, i, BOXES[:1], [.02 if i % 2 else .06])
        book.update(CaptionEvent("final", "A의 말", 1, t, speaker_id=1, speaker_resolved=True),
                    tracker.current(t))
        for i in range(12, 24):                 # A gone for 0.6 s > lost_sec
            self.speak(timeline, tracker, i, [], [])
        t = self.speak(timeline, tracker, 24, BOXES[:1], [.02])
        self.assertNotEqual(tracker.current(t).track_id, 1)
        state = book.states[1]
        self.assertEqual(state.track_id, 1)
        anchor = state.anchor(tracker.current(t), t)
        self.assertTrue(anchor is None or anchor.track_id == 1,
                        "A's caption never moves to the newcomer")

    def test_overlap_is_uncertain_but_current_speaker_is_remembered(self):
        tracker, timeline = FaceTracker(), SpeakerTimeline()
        for i in range(14):
            self.speak(timeline, tracker, i, BOXES, [.02 if i % 2 else .06, .02])
        for i in range(14, 24):
            v = .02 if i % 2 else .06
            self.speak(timeline, tracker, i, BOXES, [v, v])
        self.assertIsNone(timeline.resolve(11.0), "overlapping speech is uncertain")
        self.speak(timeline, tracker, 24, BOXES, [.06, .02])
        self.speak(timeline, tracker, 25, BOXES, [.02, .02])
        self.speak(timeline, tracker, 26, BOXES, [.06, .02])
        self.speak(timeline, tracker, 27, BOXES, [.02, .02])
        self.assertEqual(timeline._active, 1, "A resumes without a fresh confirmation")


if __name__ == "__main__":
    unittest.main()
