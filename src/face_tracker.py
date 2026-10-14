"""Conservative geometric tracks and explicit conversation-partner selection.

Track IDs are temporary spatial associations, not biometric identities or
active-speaker recognition. Ambiguous matches deliberately lose the anchor.
"""
from dataclasses import dataclass
from collections import deque

from src.one_euro import OneEuroPoint


@dataclass(frozen=True)
class FaceTrack:
    track_id: int
    box: tuple
    seen_at: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def _overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return intersection / max(area(a) + area(b) - intersection, 1)


class FaceTracker:
    def __init__(self, lost_sec=0.45):
        self.lost_sec = lost_sec
        self.serial = 0
        self.size = None
        self._tracks = {}
        self._raw = {}
        self._filters = {}
        self.selected_id = None
        self.manual = False
        self._timeline = deque(maxlen=128)

    def clear(self):
        self._tracks.clear()
        self._raw.clear()
        self._filters.clear()
        self._timeline.clear()
        # In manual mode a lost selection stays unbound until another click.
        # Automatically switching people would attribute speech incorrectly.
        if not self.manual:
            self.selected_id = None

    def _remove(self, track_id):
        self._tracks.pop(track_id, None)
        self._raw.pop(track_id, None)
        self._filters.pop(track_id, None)

    def tracks(self, now):
        for track_id, track in list(self._tracks.items()):
            if now - track.seen_at > self.lost_sec:
                self._remove(track_id)
        return list(self._tracks.values())

    def current(self, now):
        self.tracks(now)
        return self._tracks.get(self.selected_id)

    def _remember(self, now):
        # Capture timestamps can predate a click processed on the main thread.
        # Do not retroactively apply that click to older recognition results.
        timestamp = max(now, self._timeline[-1][0]) if self._timeline else now
        self._timeline.append((timestamp, self.current(now)))

    def at(self, observed_at):
        """Selected face at result time; never attach old results to future faces."""
        for timestamp, track in reversed(self._timeline):
            if timestamp <= observed_at:
                if track and 0 <= observed_at - track.seen_at <= self.lost_sec:
                    return track
                return None
        return None

    def select_at(self, x, y, now):
        """Select a uniquely hit face in ORIGINAL camera pixels."""
        matches = [t for t in self.tracks(now)
                   if t.box[0] <= x <= t.box[2] and t.box[1] <= y <= t.box[3]]
        if len(matches) != 1:
            return False
        self.manual = True
        self.selected_id = matches[0].track_id
        self._remember(now)
        return True

    def automatic(self, now):
        self.manual = False
        tracks = self.tracks(now)
        self.selected_id = tracks[0].track_id if len(tracks) == 1 else None
        self._remember(now)

    def update(self, boxes, size, now):
        if size != self.size:
            self.clear()
            self.size = size
        previous = self.tracks(now)
        if not boxes:
            self._remember(now)
            return self.current(now)
        candidates = [[t.track_id for t in previous
                       if _overlap(self._raw[t.track_id], box) >= 0.25]
                      for box in boxes]
        counts = {t.track_id: sum(t.track_id in ids for ids in candidates)
                  for t in previous}
        ambiguous = {track_id for ids in candidates for track_id in ids
                     if len(ids) != 1 or counts[track_id] != 1}
        for track_id in ambiguous:
            self._remove(track_id)
        for box, ids in zip(boxes, candidates):
            if len(ids) == 1 and ids[0] not in ambiguous:
                track_id = ids[0]
            else:
                self.serial += 1
                track_id = self.serial
                self._filters[track_id] = OneEuroPoint(1.0, 0.02)
            self._raw[track_id] = box
            x1, y1, x2, y2 = box
            cx, cy = self._filters[track_id]((x1 + x2) / 2, (y1 + y2) / 2, now)
            hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
            self._tracks[track_id] = FaceTrack(
                track_id, (cx - hw, cy - hh, cx + hw, cy + hh), now)
        # Keep a still-tracked partner, otherwise auto-select only a lone face.
        if not self.manual and self.selected_id not in self._tracks:
            tracks = self.tracks(now)
            self.selected_id = tracks[0].track_id if len(tracks) == 1 else None
        self._remember(now)
        return self.current(now)
