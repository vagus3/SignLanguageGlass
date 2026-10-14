"""Utterance lifetime and face binding shared by video and OLED rendering."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CaptionEvent:
    kind: str
    text: str
    utterance_id: int
    observed_at: float  # recognition result time, NOT acoustic speech onset


class CaptionState:
    def __init__(self, hold_sec=3.0, partial_timeout=8.0, anchor_max_age=0.45):
        self.hold_sec = hold_sec
        self.partial_timeout = partial_timeout
        self.anchor_max_age = anchor_max_age
        self.utterance_id = None
        self.text = ""
        self.partial = False
        self.track_id = None
        self.updated_at = float("-inf")
        self.expires_at = float("-inf")

    def update(self, event, track):
        if self.utterance_id is not None and event.utterance_id < self.utterance_id:
            return False
        if event.utterance_id == self.utterance_id and not self.partial:
            return False  # cancelled/finalized IDs are terminal
        if event.kind == "cancel":
            self.text = ""
            self.partial = False
            self.track_id = None
            self.utterance_id = event.utterance_id
            self.expires_at = event.observed_at
            return True
        if event.kind not in ("partial", "final") or not event.text.strip():
            return False
        if event.utterance_id != self.utterance_id:
            # A delayed STT event must not bind to a much newer camera frame.
            fresh = track and 0 <= event.observed_at - track.seen_at <= self.anchor_max_age
            self.track_id = track.track_id if fresh else None
            self.utterance_id = event.utterance_id
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

    def visible(self, now):
        return bool(self.text) and now < self.expires_at

    def anchor(self, track, now):
        if not self.visible(now):
            return None
        if track is None or track.track_id != self.track_id:
            # Once lost, this utterance must never jump to another face.
            self.track_id = None
            return None
        return track
