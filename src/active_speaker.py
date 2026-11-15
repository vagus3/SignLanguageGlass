"""Timestamped visual speaker evidence shared by the GUI and STT worker.

Decisions remain provisional for confirm_sec. Audio is held briefly so a newly
confirmed speaker can own the beginning of their turn, not the result-arrival time.
This is a turn-taking heuristic, not audio separation or biometric identification.

Face states: "seen" (detected in this frame), "briefly lost" (the tracker still keeps
it, seen within max_age) and "left" (dropped by the tracker). A briefly lost face keeps
its mouth history and the speaker choice; only a face that left releases them. Mouth
changes across a detection gap are never counted as speech. When faces move alike the
audio is uncertain, but the current speaker is remembered.
"""
import math
import threading
from collections import deque


class SpeakerTimeline:
    def __init__(self, confirm_sec=.15, hold_sec=.65, motion_threshold=.08,
                 dominance=1.8, history_sec=12.0, max_age=.45, gap_sec=.15, log=None):
        self.confirm_sec, self.hold_sec = confirm_sec, hold_sec
        self.motion_threshold, self.dominance = motion_threshold, dominance
        self.history_sec, self.max_age = history_sec, max_age
        self.gap_sec = gap_sec      # a longer gap between mouth samples breaks the motion chain
        self.log = log              # optional callable(str), called on state changes only
        self._lost = set()          # active/candidate faces currently briefly lost
        self._uncertain = False
        self._lock = threading.RLock()
        self._samples = {}
        self._motion_onsets, self._changed_at = {}, {}
        self._history = deque(maxlen=2000)  # (capture time, speaker ID, present tracks, manual)
        self._candidate = None
        self._candidate_since = self._last_motion = float("-inf")
        self._candidate_start = float("-inf")
        self._candidate_seen = float("-inf")
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

    def _note(self, at, message):
        if self.log is not None:
            self.log(f"[화자 {at:.2f}] {message}")

    def _set_active(self, ident, at, reason):
        if ident != self._active:
            self._note(at, f"선택 {self._active} -> {ident} ({reason})")
        self._active = ident

    def _revise(self, start, ident):
        self._history = deque(
            ((t, ident if t >= start and t > self._committed and not locked
              and (ident is None or ident in faces) else who, faces, locked)
             for t, who, faces, locked in self._history), maxlen=2000)

    def observe(self, tracks, mouths, at, manual=False, selected_id=None):
        with self._lock:
            if at <= self.latest:
                return
            # seen: detected now. present: still kept by the tracker (includes briefly lost).
            seen = {t.track_id: t for t in tracks if at - t.seen_at < 1e-6}
            present = {t.track_id: t for t in tracks if 0 <= at - t.seen_at <= self.max_age}
            self.latest = at
            scores = {}
            for ident in (self._active, self._candidate):
                if ident is None:
                    continue
                if ident in present and ident not in seen and ident not in self._lost:
                    self._lost.add(ident)
                    self._note(at, f"얼굴 {ident} 잠시 놓침 (선택·기록 유지)")
                elif ident in seen and ident in self._lost:
                    self._lost.discard(ident)
                    self._note(at, f"얼굴 {ident} 다시 잡힘")
            self._lost &= set(present)
            for ident in list(self._samples):
                if ident not in present:
                    del self._samples[ident]
                    self._motion_onsets.pop(ident, None)
                    self._changed_at.pop(ident, None)
            for ident, value in mouths.items():
                if ident not in seen or not math.isfinite(value):
                    continue
                samples = self._samples.setdefault(ident, deque(maxlen=40))
                if samples and at - samples[-1][0] > self.gap_sec:
                    samples.clear()     # a change across a detection gap is not speech
                if samples and abs(value - samples[-1][1]) >= .001:
                    if at - self._changed_at.get(ident, float("-inf")) > .15:
                        self._motion_onsets[ident] = samples[-1][0]
                    self._changed_at[ident] = at
                samples.append((at, value))
                while samples and at - samples[0][0] > .4:
                    samples.popleft()
                changes = [(b[0], abs(b[1] - a[1])) for a, b in zip(samples, list(samples)[1:])]
                # One yawn/open mouth followed by a static pose is not sustained speech.
                recent = [(t, d) for t, d in changes if at - t <= .24]
                repeated = [(t, d) for t, d in recent if d >= .001]
                if len(repeated) >= 3 and at - repeated[-1][0] <= .12:
                    scores[ident] = sum(d for _, d in recent) / max(min(at - samples[0][0], .24), .01)
            ambiguous = False
            if manual:
                decision = selected_id if selected_id in present else None
                self._active, self._candidate = decision, None
            else:
                ranked = sorted(((v, k) for k, v in scores.items()
                                 if v >= self.motion_threshold), reverse=True)
                ambiguous = len(ranked) > 1 and ranked[0][0] < ranked[1][0] * self.dominance
                winner = ranked[0][1] if ranked and not ambiguous else None
                if ambiguous:
                    # This audio is uncertain; the current speaker is remembered, not cleared.
                    if not self._uncertain:
                        self._note(at, f"여러 얼굴이 비슷하게 움직임: 불확실 "
                                       f"(선택 {self._active} 기억)")
                    self._candidate = None
                    self._decision_floor = at
                    decision = None
                elif winner is None:
                    pending = (self._candidate in present
                               and at - self._candidate_seen <= .12)
                    if not pending:
                        self._candidate = None
                    # Silence does not revoke an established visible track.
                    decision = (self._active if not pending and self._active in present
                                else None)
                    if decision is None and not pending:
                        if self._active is not None:
                            self._set_active(None, at, "얼굴이 화면을 떠남")
                        self._decision_floor = at
                elif winner == self._active:
                    decision = winner
                    self._last_motion = at
                    self._candidate = None
                else:
                    self._candidate_seen = at
                    if winner != self._candidate:
                        self._candidate, self._candidate_since = winner, at
                        self._candidate_start = max(self._motion_onsets.get(winner, at),
                                                    at - .12, self._decision_floor + 1e-6)
                        self._revise(self._candidate_start, None)
                    decision = None  # Do not assign the possible new turn to the old face.
                    if at - self._candidate_since >= self.confirm_sec:
                        self._set_active(winner, at,
                                         f"입 움직임 {self.confirm_sec:.2f}초 우세")
                        self._last_motion = at
                        decision = winner
                        self._revise(self._candidate_start, winner)
                        self._candidate = None
            if self._uncertain and not ambiguous:
                self._note(at, "불확실 구간 끝")
            self._uncertain = ambiguous
            self._history.append((at, decision, present, manual))
            while self._history and at - self._history[0][0] > self.history_sec:
                self._history.popleft()

    def select(self, track, at, manual):
        """A click is an explicit timestamped boundary, never a retrospective rebind."""
        with self._lock:
            self._candidate = None
            self._set_active(None, max(self.latest, at), "클릭/초기화")
            self._lost.clear()
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
            self._focus = self._at(now, settled=False)
            return self._focus

    def face_at(self, ident, at):
        with self._lock:
            for t, _, faces, _ in reversed(self._history):
                if t <= at and ident in faces and at - t <= self.max_age:
                    return faces[ident]
        return None
