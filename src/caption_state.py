"""Utterance lifetime and face binding shared by video and OLED rendering."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CaptionEvent:
    kind: str
    text: str
    utterance_id: int
    observed_at: float  # recognition result time, NOT acoustic speech onset
    audio_start: float | None = None
    audio_end: float | None = None
    speaker_id: int | None = None
    speaker_resolved: bool = False  # True + None means explicitly uncertain, never late-bind.


class CaptionState:
    def __init__(self, hold_sec=3.0, partial_timeout=8.0, anchor_max_age=0.45):
        self.hold_sec = hold_sec
        self.partial_timeout = partial_timeout
        self.anchor_max_age = anchor_max_age
        self.utterance_id = None
        self.text = ""
        self.partial = False
        self.track_id = None
        self.held = None        # last box of the bound face, kept while it is hidden
        self.released = None    # utterance whose partner was changed: stays unbound
        self.updated_at = float("-inf")
        self.expires_at = float("-inf")

    def update(self, event, track):
        if self.utterance_id is not None and event.utterance_id < self.utterance_id:
            return False
        if event.utterance_id == self.utterance_id and not self.partial:
            return False  # cancelled/finalized IDs are terminal
        if (event.utterance_id == self.utterance_id and event.speaker_resolved
                and event.speaker_id != self.track_id):
            return False  # A producer must start a new utterance at a speaker boundary.
        if event.kind == "cancel":
            self.text = ""
            self.partial = False
            self.track_id = None
            self.held = None
            self.utterance_id = event.utterance_id
            self.expires_at = event.observed_at
            return True
        if event.kind not in ("partial", "final") or not event.text.strip():
            return False
        # Resolved audio already owns an ID; a new frame only updates its position.
        if event.speaker_resolved:
            fresh = track is not None and track.track_id == event.speaker_id
        else:
            fresh = track and 0 <= event.observed_at - track.seen_at <= self.anchor_max_age
        if event.utterance_id != self.utterance_id:
            self.track_id = (event.speaker_id if event.speaker_resolved else
                             track.track_id if fresh else None)
            self.held = track if fresh else None
            self.utterance_id = event.utterance_id
        elif (not event.speaker_resolved and self.held is None and fresh
              and self.released != event.utterance_id):
            # The speaker came into view mid-utterance: bind once, never re-bind later.
            self.track_id, self.held = track.track_id, track
        self.text = event.text.strip()
        self.partial = event.kind == "partial"
        self.updated_at = event.observed_at
        duration = (self.partial_timeout if self.partial else
                    max(self.hold_sec, min(10.0, len(self.text) / 8.0)))
        self.expires_at = event.observed_at + duration
        return True

    def release_anchor(self):
        """Changing the selected partner affects subsequent utterances only."""
        self.track_id = None
        self.held = None
        self.released = self.utterance_id

    def visible(self, now):
        return bool(self.text) and now < self.expires_at

    def anchor(self, track, now):
        """The bound face, or where it was last seen while it is hidden.

        Signing hands and turned heads hide a face for a moment; the bubble stays
        where that speaker was instead of dropping to a bottom caption. It never
        jumps to another face: a track with a different ID is ignored.
        """
        if not self.visible(now):
            return None
        if track is not None and track.track_id == self.track_id:
            self.held = track
        return self.held


class CaptionBook:
    """Utterance ownership survives focus changes and delayed finals.

    Keep the latest visible utterance per face plus one unassigned caption. Older
    finals update their own state, never replace a newer speaker's partial text.
    """
    def __init__(self, hold_sec=3.0, partial_timeout=8.0, anchor_max_age=.45, limit=32):
        self.settings = hold_sec, partial_timeout, anchor_max_age
        self.limit = max(1, limit)
        self.states = {}
        self.evicted_through = -1
        self.empty = CaptionState(*self.settings)

    @property
    def latest(self):
        return self.states[max(self.states)] if self.states else self.empty

    def update(self, event, track):
        state = self.states.get(event.utterance_id)
        if state is None:
            if event.utterance_id <= self.evicted_through:
                return None  # Already evicted/expired; cannot resurrect a delayed event.
            state = self.states[event.utterance_id] = CaptionState(*self.settings)
        if not state.update(event, track):
            return None
        while len(self.states) > self.limit:
            self.evicted_through = min(self.states)
            del self.states[self.evicted_through]
        return state

    def visible_states(self, now):
        latest = {}
        for uid in sorted(self.states):
            state = self.states[uid]
            if state.visible(now):
                latest[state.track_id] = state
        return list(latest.values())

    def cancel_partials(self, now):
        for state in self.states.values():
            if state.partial:
                state.text, state.partial, state.expires_at = "", False, now
