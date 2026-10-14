"""Utterance boundaries, cancellation and face loss without camera hardware."""
import _path  # noqa: F401

from src.caption_state import CaptionEvent, CaptionState
from src.face_tracker import FaceTrack, FaceTracker
from src.caption_history import CaptionHistory

face = FaceTrack(1, (100, 50, 160, 130), 10.0)
state = CaptionState()
state.update(CaptionEvent("partial", "내일", 1, 10.0), face)
state.update(CaptionEvent("partial", "내일 세 시", 1, 10.1), face)
assert state.text == "내일 세 시" and state.anchor(face, 10.1) == face
state.update(CaptionEvent("final", "내일 세 시에 만나요", 1, 10.2), face)
state.update(CaptionEvent("partial", "내일", 1, 10.3), face)
assert state.text == "내일 세 시에 만나요" and not state.partial
assert state.visible(13.19) and not state.visible(13.21)

state.update(CaptionEvent("partial", "다음 문장", 2, 11.0),
             FaceTrack(1, face.box, 11.0))
state.update(CaptionEvent("cancel", "", 2, 11.1), None)
state.update(CaptionEvent("partial", "늦게 도착한 결과", 2, 11.2), face)
assert not state.visible(11.2), "A cancelled utterance cannot be resurrected"

state.update(CaptionEvent("partial", "안녕하세요", 3, 12.0),
             FaceTrack(1, face.box, 12.0))
assert state.anchor(None, 12.1) is None
state.update(CaptionEvent("final", "안녕하세요", 3, 12.2),
             FaceTrack(2, face.box, 12.2))
assert state.anchor(FaceTrack(2, face.box, 12.3), 12.3) is None
state.update(CaptionEvent("cancel", "", 2, 12.4), None)
assert state.text == "안녕하세요", "Old cancellation must not erase a new utterance"

tracker = FaceTracker()
track = tracker.update([face.box], (640, 480), 20.0)
assert tracker.update([], (640, 480), 20.2).track_id == track.track_id
assert tracker.current(20.5) is None
next_track = tracker.update([face.box], (640, 480), 20.6)
assert next_track.track_id != track.track_id
assert tracker.update([face.box, (400, 50, 460, 130)], (640, 480), 20.7)

# Two faces require selection; ordering the detector output differently cannot
# exchange their IDs. Losing a manual target must not select the remaining one.
tracker = FaceTracker()
left, right = (100, 100, 180, 200), (400, 100, 480, 200)
assert tracker.update([left, right], (640, 480), 30) is None
assert not tracker.select_at(300, 150, 30)
assert tracker.select_at(140, 150, 30)
selected = tracker.current(30).track_id
assert tracker.update([right, left], (640, 480), 30.1).track_id == selected
assert tracker.update([right], (640, 480), 30.7) is None
assert tracker.manual and tracker.selected_id == selected
assert tracker.select_at(440, 150, 30.7)
assert tracker.current(30.7).track_id != selected
tracker.automatic(30.7)
assert not tracker.manual and tracker.current(30.7)

# Overlapping detections make geometric identity ambiguous: never silently
# preserve a manually selected ID through a split or crossing.
tracker = FaceTracker()
tracker.update([left], (640, 480), 40)
assert tracker.select_at(140, 150, 40)
old_id = tracker.selected_id
assert tracker.update([left, (110, 100, 190, 200)], (640, 480), 40.1) is None
assert all(t.track_id != old_id for t in tracker.tracks(40.1))
assert not tracker.select_at(145, 150, 40.1), "Overlapping click is ambiguous"

# A result queued before a face appeared cannot be attributed to that face.
tracker = FaceTracker()
tracker.update([], (640, 480), 45)
new_face = tracker.update([left], (640, 480), 45.2)
assert tracker.at(45.1) is None
assert tracker.at(45.3).track_id == new_face.track_id
assert tracker.at(46) is None
tracker = FaceTracker()
tracker.update([left, right], (640, 480), 47)
assert tracker.select_at(140, 150, 47.2)
tracker.update([left, right], (640, 480), 47.1)  # frame queued before the click
assert tracker.at(47.15) is None
assert tracker.at(47.25).track_id == tracker.selected_id

# Only finals are recorded, exactly once, with full text and a bounded count.
history = CaptionHistory(limit=2)
long = "생략하지 않고 전체 문장을 보관해야 합니다. " * 20
assert not history.append(CaptionEvent("partial", long, 1, 50), 1)
assert history.append(CaptionEvent("final", long, 1, 50.1), 1)
assert history.records[0].text == long.strip()
assert not history.append(CaptionEvent("final", "중복", 1, 50.2), 1)
assert history.append(CaptionEvent("final", "두 번째", 2, 51), None)
assert history.append(CaptionEvent("final", "세 번째", 3, 52), 2)
assert [r.utterance_id for r in history.records] == [2, 3]

print("[OK] Caption lifecycle, cancellation, face selection/loss/ambiguity, bounded history")
