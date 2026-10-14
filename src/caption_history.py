"""Bounded in-memory transcript of accepted final STT results."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CaptionRecord:
    utterance_id: int
    text: str
    track_id: int | None
    observed_at: float


class CaptionHistory:
    def __init__(self, limit=200):
        self.records = deque(maxlen=max(1, limit))
        self.revision = 0
        self._last_id = -1

    def append(self, event, track_id):
        if (event.kind != "final" or not event.text.strip() or
                event.utterance_id <= self._last_id):
            return False
        self.records.append(CaptionRecord(event.utterance_id, event.text.strip(),
                                          track_id, event.observed_at))
        self._last_id = event.utterance_id
        self.revision += 1
        return True
