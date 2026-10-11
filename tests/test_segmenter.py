"""Automatic segmentation: webcam rest line, glasses hands-only, jitter, limits, manual control."""
import _path  # noqa
import numpy as np
from src.cloud_sign import SegmentRecorder
from src.segmenter import ActivitySegmenter

FPS = 30
hand = np.zeros((21, 2))
hand[:, 1] = -np.linspace(0, .12, 21)
hand[9] = [0, -.09]  # wrist -> middle MCP = one palm length


def frame(x, visible=True, y=.6, shoulders=False):
    """Left hand; shoulders only when a frontal webcam sees them (rest line y=0.7)."""
    points, mask = np.zeros((49, 2), np.float32), np.zeros(49, bool)
    if visible:
        points[:21], mask[:21] = hand + [x, y], True
    if shoulders:
        points[43], points[44], mask[43:45] = [.35, .4], [.65, .4], True
    return points, mask


def moving(t):
    return .4 + .1 * np.sin(2 * np.pi * 1.5 * t)


def simulate(script, seconds, *, manual=(), busy=False, scene=None, motion=None, **options):
    """script(t) -> frame() arguments; main-loop behaviour is applied on the next frame."""
    scene = scene or {}
    segmenter, recorder = ActivitySegmenter(**options), SegmentRecorder()
    events, segments, queued = [], [], None
    for i in range(int(seconds * FPS)):
        t = i / FPS
        if queued == "start" and not busy:
            recorder.start(pre_roll=.3)
        elif queued == "stop":
            segments.append(recorder.finish()[0])
        elif queued == "discard":
            recorder.cancel()
        queued = None
        for when, action in manual:
            if abs(when - t) < 1e-9:
                if action == "start":
                    recorder.start()
                else:
                    segments.append(recorder.finish()[0])
        points, mask = frame(*script(t), **scene)
        if motion is not None:
            motion(points, mask, t)
        recorder.add(points, mask, t, (640, 640))
        event = segmenter.update(points, mask, t, recorder.active, (640, 640))
        if event:
            events.append((round(t, 3), event))
            queued = event
    simulate.last = segmenter
    return events, segments


# Webcam: typing below the rest line never starts, even while moving fast.
webcam = {"shoulders": True}
events, _ = simulate(lambda t: (moving(t), True, .85), 4, scene=webcam)
assert events == [] and abs(simulate.last.rest_line - .7) < 1e-6, events
# Raise the hands and sign 1..3 s, then lower them to the keyboard (still moving).
events, segments = simulate(lambda t: (moving(t), True, .6 if 1 <= t < 3 else .85), 6,
                            scene=webcam)
assert [e for _, e in events] == ["start", "stop"], events
assert 1.2 <= events[0][0] <= 1.6 and 3.7 <= events[1][0] <= 4.0, events
# Without shoulders the webcam mode falls back to the hands-only rule.
events, _ = simulate(lambda t: (moving(t), True, .85), 2)
assert [e for _, e in events] == ["start"] and simulate.last.rest_line is None, events
# The glasses viewpoint ignores the rest line even when shoulders are visible.
events, _ = simulate(lambda t: (moving(t), True, .85), 2, scene=webcam, viewpoint="glasses")
assert [e for _, e in events] == ["start"], events
try:
    ActivitySegmenter(viewpoint="phone")
    raise AssertionError("Unknown viewpoint accepted")
except ValueError:
    pass

# Still -> sign 1..3s -> hands stay visible but still.
events, segments = simulate(lambda t: (moving(t) if 1 <= t < 3 else moving(3), True), 7)
assert [e for _, e in events] == ["start", "stop"], events
assert 1.2 <= events[0][0] <= 1.6 and 4.3 <= events[1][0] <= 5.0, events
assert segments[0].timestamps[-1] >= events[1][0] - events[0][0] + .25, "Pre-roll keeps the onset"

# Hands leave the view after signing.
events, _ = simulate(lambda t: (moving(t), t < 3) if t >= 1 else (.4, False), 6)
assert [e for _, e in events] == ["start", "stop"] and 3.7 <= events[1][0] <= 4.0, events

# A brief twitch never starts; a very short burst is discarded rather than translated.
events, _ = simulate(lambda t: (moving(min(max(t, 1), 1.15)), True), 4)
assert events == [], events
events, segments = simulate(lambda t: (moving(t), t < 1.5) if t >= 1 else (.4, True), 4,
                            min_seconds=.5)
assert [e for _, e in events] == ["start", "discard"] and not segments, events

# Continuous signing is split before the 15 s pose limit; every segment stays valid.
events, segments = simulate(lambda t: (moving(t), True), 32)
starts = [t for t, e in events if e == "start"]
stops = [t for t, e in events if e == "stop"]
assert len(stops) >= 2 and all(b - a <= 14.1 for a, b in zip(starts, stops)), events
assert all(s.timestamps[-1] <= 15 for s in segments)

# A manual (SPACE) segment is never stopped automatically.
events, segments = simulate(lambda t: (moving(min(max(t, 1), 3)), True), 9,
                            manual=[(.5, "start"), (8.0, "stop")])
assert events == [] and len(segments) == 1 and segments[0].timestamps[-1] > 7, events
# Manually stopping an automatic segment is followed, not fought.
events, segments = simulate(lambda t: (moving(min(t, 2)), True), 4, manual=[(2.0, "stop")])
assert [e for _, e in events] == ["start"] and len(segments) == 1, events

# If the main loop refuses to start (translation busy), retry only after the timeout.
events, _ = simulate(lambda t: (moving(t), True), 3, busy=True)
assert all(e == "start" for _, e in events)
assert all(b - a > 1.0 for (a, _), (b, _) in zip(events, events[1:])), events
# Finger articulation must start and sustain capture even with fixed wrists.
def articulate(points, mask, t):
    if 1 <= t < 4:
        points[[4, 8, 12, 16, 20], 0] += .06 * np.sin(12 * t)


for scene in ({}, webcam):
    events, segments = simulate(lambda t: (.4, True), 7, scene=scene, motion=articulate)
    assert [e for _, e in events] == ["start", "stop"], events
    assert events[0][0] < 2 and events[1][0] > 5, events
    assert len(segments) == 1

# A hand rotating about a stationary wrist also counts as activity.
def rotate(points, mask, t):
    angle = 1.2 * np.sin(10 * t)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    points[:21] = (points[:21] - points[0]) @ rotation.T + points[0]


events, _ = simulate(lambda t: (.4, True), 4, motion=rotate)
assert [e for _, e in events] == ["start"], events

# A missing joint's arbitrary coordinates must never create motion.
def missing_tip(points, mask, t):
    mask[8] = False
    points[8, 0] += .3 * np.sin(12 * t)


events, _ = simulate(lambda t: (.4, True), 4, motion=missing_tip)
assert events == [], events

import cv2  # noqa: E402
from main import draw_segment_overlay  # noqa: E402
segmenter = ActivitySegmenter()
segmenter.update(*frame(.4, shoulders=True), 0, False, (640, 640))
preview = np.zeros((640, 640, 3), np.uint8)
draw_segment_overlay(cv2, preview, SegmentRecorder(), segmenter)
assert preview[int(.7 * 640), 320].any() and not preview[int(.5 * 640), 320].any()
print("[OK] webcam rest line, hands-only fallback, jitter, 15 s split, manual coexistence")
