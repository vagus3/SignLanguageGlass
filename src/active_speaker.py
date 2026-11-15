"""Timestamped visual speaker evidence shared by the GUI and STT worker.

Decisions remain provisional for confirm_sec. Audio is held briefly so a newly
confirmed speaker can own the beginning of their turn, not the result-arrival time.
This is a turn-taking heuristic, not audio separation or biometric identification.
"""
import math
import threading
from collections import deque


class SpeakerTimeline:
    def __init__(self, confirm_sec=.25, hold_sec=.4, motion_threshold=.08,
                 dominance=1.8, history_sec=12.0, max_age=.45):
        self.confirm_sec, self.hold_sec = confirm_sec, hold_sec
        self.motion_threshold, self.dominance = motion_threshold, dominance
        self.history_sec, self.max_age = history_sec, max_age
        self._lock = threading.RLock()
        self._samples = {}
        self._motion_onsets, self._changed_at = {}, {}
        self._history = deque(maxlen=2000)  # (capture time, speaker ID, visible tracks, manual)
        self._candidate = None
        self._candidate_since = self._last_motion = float("-inf")
        self._candidate_start = float("-inf")
        self._decision_floor = float("-inf")
        self._active = self._focus = None
        self._last_voice = self._committed = float("-inf")
        self.latest = float("-inf")

    def observe_audio(self, end_at, speech):
        if speech:
            with self._lock:
                self._last_voice = max(self._last_voice, end_at)

    def mute(self):
        with self._lock:
            self._last_voice = float("-inf")
            self._focus = None
            self._active = self._candidate = None
            self._samples.clear()
            self._motion_onsets.clear()
            self._changed_at.clear()

    def _revise(self, start, ident):
        self._history = deque(
            ((t, ident if t >= start and t > self._committed and not locked
              and (ident is None or ident in faces) else who, faces, locked)
             for t, who, faces, locked in self._history), maxlen=2000)

    def observe(self, tracks, mouths, at, manual=False, selected_id=None):
        with self._lock:
            if at <= self.latest:
                return
            visible = {t.track_id: t for t in tracks if at - t.seen_at < 1e-6}
            self.latest = at
            scores = {}
            for ident in list(self._samples):
                if ident not in visible:
                    del self._samples[ident]
                    self._motion_onsets.pop(ident, None)
                    self._changed_at.pop(ident, None)
            for ident, value in mouths.items():
                if ident not in visible or not math.isfinite(value):
                    continue
                samples = self._samples.setdefault(ident, deque(maxlen=40))
                if samples and at - samples[-1][0] > self.max_age:
                    samples.clear()
                if samples and abs(value - samples[-1][1]) >= .001:
                    if at - self._changed_at.get(ident, float("-inf")) > .15:
                        self._motion_onsets[ident] = samples[-1][0]
                    self._changed_at[ident] = at
                samples.append((at, value))
                while samples and at - samples[0][0] > .4:
                    samples.popleft()
                changes = [(b[0], abs(b[1] - a[1])) for a, b in zip(samples, list(samples)[1:])]
                moving = [(t, d) for t, d in changes if d >= .001]
                # One yawn/open mouth followed by a static pose is not sustained speech.
                recent = [(t, d) for t, d in changes if at - t <= .12]
                if len(moving) >= 2 and at - moving[-1][0] <= .08:
                    scores[ident] = sum(d for _, d in recent) / max(min(at - samples[0][0], .12), .01)
            if manual:
                decision = selected_id if selected_id in visible else None
                self._active, self._candidate = decision, None
            else:
                ranked = sorted(((v, k) for k, v in scores.items()
                                 if v >= self.motion_threshold), reverse=True)
                ambiguous = len(ranked) > 1 and ranked[0][0] < ranked[1][0] * self.dominance
                winner = ranked[0][1] if ranked and not ambiguous else None
                if ambiguous:
                    self._active = self._candidate = None
                    self._decision_floor = at
                    decision = None
                elif winner is None:
                    self._candidate = None
                    decision = (self._active if self._active in visible
                                and at - self._last_motion <= self.hold_sec else None)
                    if decision is None:
                        self._active = None
                        self._decision_floor = at
                elif winner == self._active:
                    decision = winner
                    self._last_motion = at
                    self._candidate = None
                else:
                    if winner != self._candidate:
                        self._candidate, self._candidate_since = winner, at
                        self._candidate_start = max(self._motion_onsets.get(winner, at),
                                                    at - .12, self._decision_floor + 1e-6)
                        self._revise(self._candidate_start, None)
                    decision = None  # Do not assign the possible new turn to the old face.
                    if at - self._candidate_since >= self.confirm_sec:
                        self._active, self._last_motion = winner, at
                        decision = winner
                        self._revise(self._candidate_start, winner)
                        self._candidate = None
            self._history.append((at, decision, visible, manual))
            while self._history and at - self._history[0][0] > self.history_sec:
                self._history.popleft()

    def select(self, track, at, manual):
        """A click is an explicit timestamped boundary, never a retrospective rebind."""
        with self._lock:
            self._candidate = self._active = None
            self._samples.clear()
            self._motion_onsets.clear()
            self._changed_at.clear()
            self.latest = max(self.latest, at)
            self._decision_floor = self.latest
            faces = {track.track_id: track} if track else {}
            ident = track.track_id if manual and track else None
            self._history.append((self.latest, ident, faces, manual))

    def invalidate(self, at):
        self.select(None, at, False)

    def _at(self, at, settled):
        for t, ident, faces, manual in reversed(self._history):
            if t <= at:
                if at - t > self.max_age:
                    return None
                if settled and not manual and self.latest < at + self.confirm_sec:
                    return None
                return ident
        return None

    def resolve(self, at):
        with self._lock:
            ident = self._at(at, settled=True)
            self._committed = max(self._committed, at)
            return ident

    def focus(self, now):
        with self._lock:
            if now - self._last_voice <= self.hold_sec:
                self._focus = self._at(now, settled=False)
            else:
                self._focus = None
            return self._focus

    def face_at(self, ident, at):
        with self._lock:
            for t, _, faces, _ in reversed(self._history):
                if t <= at and ident in faces and at - t <= self.max_age:
                    return faces[ident]
        return None
