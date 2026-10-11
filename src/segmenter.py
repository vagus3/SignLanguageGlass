"""Rule-based automatic sentence segmentation for the cloud backend.

A signing hand must keep moving to start a segment, which ends when the hands
are gone or stay still. Speeds are in palm lengths per second.

- glasses: shoulders are often cut off, so hands are used alone and "gone"
  means out of view.
- webcam: a frontal laptop camera sees the shoulders. Only hands above a rest
  line (shoulder height + rest_ratio shoulder widths) count, so typing on the
  keyboard is ignored and lowering the hands ends the sentence. Shoulder width
  is a steadier scale than the palm, which foreshortens as it turns.
  Frames without both shoulders fall back to the glasses rule.

Not learned; thresholds need checking on recordings from the same camera.
"""
import numpy as np
from src.pose_sequence import MAX_SECONDS

VIEWPOINTS = ("webcam", "glasses")
PALM_PER_SHOULDER = 0.25  # wrist-to-middle-knuckle length / shoulder width, adults


class ActivitySegmenter:
    def __init__(self, *, viewpoint="webcam", rest_ratio=1.0, start_speed=1.5,
                 still_speed=0.4, start_hold=0.3, end_hold=0.8,
                 still_hold=1.5, min_seconds=0.5, max_seconds=MAX_SECONDS - 1.0,
                 pending_timeout=1.0, smoothing=0.3):
        if viewpoint not in VIEWPOINTS:
            raise ValueError(f"viewpoint must be one of {VIEWPOINTS}")
        self.viewpoint, self.rest_ratio = viewpoint, rest_ratio
        self.rest_line = None  # Normalized image y of the rest line, for the preview.
        self.start_speed, self.still_speed = start_speed, still_speed
        self.start_hold, self.end_hold, self.still_hold = start_hold, end_hold, still_hold
        self.min_seconds, self.max_seconds = min_seconds, max_seconds
        self.pending_timeout, self.smoothing = pending_timeout, smoothing
        self.previous, self.speed = None, 0.0
        self.owned, self.pending, self.started = False, None, 0.0
        self.moving_since = self.idle_since = self.still_since = None

    def _motion(self, points, mask, t, image_size):
        p = np.asarray(points, np.float64).copy()
        aspect = image_size[1] / image_size[0] if image_size is not None else 1.0
        p[:, 1] *= aspect
        m = np.asarray(mask, bool)
        wrists = p[[0, 21]]
        palms = np.linalg.norm(p[[9, 30]] - wrists, axis=1)
        valid = m[[0, 21]] & m[[9, 30]] & (palms > 1e-6)
        width = np.linalg.norm(p[43] - p[44])
        self.rest_line = None
        if self.viewpoint == "webcam" and m[43] and m[44] and width > 1e-6:
            rest = (p[43, 1] + p[44, 1]) / 2 + self.rest_ratio * width
            self.rest_line = rest / aspect
            valid &= wrists[:, 1] < rest
            palms = np.full(2, PALM_PER_SHOULDER * width)
        raw = 0.0
        if self.previous is not None:
            before, old_wrists, old_valid = self.previous
            dt, both = t - before, valid & old_valid
            if 0 < dt <= 0.25 and both.any():
                raw = float(np.max(np.linalg.norm(wrists - old_wrists, axis=1)[both]
                                   / palms[both]) / dt)
        self.previous = (t, wrists, valid)
        self.speed = self.speed + self.smoothing * (raw - self.speed) if valid.any() else 0.0
        return bool(valid.any()), self.speed

    def _emit(self, event, t):
        self.pending = (event, t)
        self.moving_since = self.idle_since = self.still_since = None
        return event

    def update(self, points, mask, t, recording, image_size=None):
        """Return None, "start", "stop" or "discard" (too short to translate)."""
        present, speed = self._motion(points, mask, t, image_size)
        if self.pending is not None:
            # The main loop applies events; wait until it has, or it refused.
            event, since = self.pending
            if recording == (event == "start"):
                self.pending, self.owned, self.started = None, event == "start", since
            elif t - since > self.pending_timeout:
                self.pending, self.owned = None, False
            else:
                return None
        if self.owned and not recording:
            self.owned = False  # Stopped by hand or cancelled by the recorder.
        if recording and not self.owned:
            self.moving_since = None  # A manual segment: never stop it automatically.
            return None
        if not self.owned:
            if present and speed >= self.start_speed:
                self.moving_since = t if self.moving_since is None else self.moving_since
                if t - self.moving_since >= self.start_hold:
                    return self._emit("start", t)
            else:
                self.moving_since = None
            return None
        if t - self.started >= self.max_seconds:
            return self._emit("stop", t)
        if present:
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = t
        if not present or speed >= self.still_speed:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = t
        gone = self.idle_since is not None and t - self.idle_since >= self.end_hold
        still = self.still_since is not None and t - self.still_since >= self.still_hold
        if gone or still:
            ended = self.idle_since if gone else self.still_since
            return self._emit("stop" if ended - self.started >= self.min_seconds else "discard", t)
        return None
