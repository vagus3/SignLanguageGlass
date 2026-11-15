"""Conservative geometric tracks and explicit conversation-partner selection.

Track IDs are temporary spatial associations, not biometric identities or
active-speaker recognition. Ambiguous matches deliberately lose the anchor.

A face missing from a few detections keeps its ID for lost_sec ("briefly lost"); a
face found again within that time rejoins it. Matching uses box overlap, centre
distance (in face widths, growing with the time since last seen) and size change, and
is one-to-one. When two pairings are nearly as good, neither is trusted: a new ID is
safer than handing one person's conversation to another.
"""
import math
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


def _cost(previous, box, dt, lost_sec):
    """Lower is better; None when the box cannot be the same face."""
    pw = max(previous[2] - previous[0], 1.0)
    width = max(box[2] - box[0], 1.0)
    size = abs(math.log(width / pw))
    pc = ((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2)
    bc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    distance = math.hypot(bc[0] - pc[0], bc[1] - pc[1]) / pw
    overlap = _overlap(previous, box)
    reach = 0.6 + 2.0 * min(max(dt, 0.0), lost_sec)      # face widths
    if overlap < 0.25 and (distance > reach or size > math.log(1.5)):
        return None
    return distance + size - overlap


AMBIGUOUS_MARGIN = 0.3     # a second pairing this close to the best one is not trusted


def _overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return intersection / max(area(a) + area(b) - intersection, 1)


class FaceTracker:
    def __init__(self, lost_sec=0.45, log=None):
        self.lost_sec = lost_sec
        self.log = log            # optional callable(str): ID created / rejoined / ended
        self.serial = 0
        self.size = None
        self._tracks = {}
        self._raw = {}
        self._filters = {}
        self.selected_id = None
        self.manual = False
        self.mouths = {}
        self._timeline = deque(maxlen=128)

    def clear(self):
        self._tracks.clear()
        self._raw.clear()
        self._filters.clear()
        self._timeline.clear()
        self.mouths.clear()
        # In manual mode a lost selection stays unbound until another click.
        # Automatically switching people would attribute speech incorrectly.
        if not self.manual:
            self.selected_id = None

    def _remove(self, track_id):
        self._tracks.pop(track_id, None)
        self._raw.pop(track_id, None)
        self._filters.pop(track_id, None)

    def _note(self, now, message):
        if self.log is not None:
            self.log(f"[얼굴 {now:.2f}] {message}")

    def tracks(self, now):
        for track_id, track in list(self._tracks.items()):
            if now - track.seen_at > self.lost_sec:
                self._note(now, f"ID {track_id} 이탈 확정 ({self.lost_sec:.2f}초 동안 안 보임)")
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

    def update(self, boxes, size, now, mouth_openings=()):
        if size != self.size:
            self.clear()
            self.size = size
        previous = self.tracks(now)
        self.mouths = {}
        if not boxes:
            self._remember(now)
            return self.current(now)
        pairs = {}       # (box index, track ID) -> cost
        for index, box in enumerate(boxes):
            for track in previous:
                cost = _cost(self._raw[track.track_id], box, now - track.seen_at, self.lost_sec)
                if cost is not None:
                    pairs[index, track.track_id] = cost
        ambiguous_boxes, ambiguous_tracks = set(), set()
        for index in range(len(boxes)):
            costs = sorted(c for (i, _), c in pairs.items() if i == index)
            if len(costs) > 1 and costs[1] - costs[0] < AMBIGUOUS_MARGIN:
                ambiguous_boxes.add(index)
        for track in previous:
            costs = sorted(c for (_, t), c in pairs.items() if t == track.track_id)
            if len(costs) > 1 and costs[1] - costs[0] < AMBIGUOUS_MARGIN:
                ambiguous_tracks.add(track.track_id)
        # Overlapping split/merge (two boxes on one face, or one box over two faces): the
        # original conservative rule, kept regardless of costs.
        overlapping = [(i, t.track_id) for i, box in enumerate(boxes) for t in previous
                       if _overlap(self._raw[t.track_id], box) >= 0.25]
        for i, track_id in overlapping:
            if sum(j == i for j, _ in overlapping) > 1:
                ambiguous_boxes.add(i)
            if sum(t == track_id for _, t in overlapping) > 1:
                ambiguous_tracks.add(track_id)
                ambiguous_boxes.update(j for j, t in overlapping if t == track_id)
        ambiguous_tracks |= {t for (i, t) in pairs if i in ambiguous_boxes}
        for track_id in ambiguous_tracks:
            self._note(now, f"ID {track_id} 대응이 애매해 연결을 끊음")
            self._remove(track_id)
        assigned, used = {}, set()
        for (index, track_id), _ in sorted(pairs.items(), key=lambda kv: kv[1]):
            if (index in assigned or track_id in used or index in ambiguous_boxes
                    or track_id in ambiguous_tracks):
                continue
            assigned[index] = track_id
            used.add(track_id)
        seen_before = {t.track_id: t.seen_at for t in previous}
        candidates = [[assigned[i]] if i in assigned else [] for i in range(len(boxes))]
        ambiguous = set()
        for index, (box, ids) in enumerate(zip(boxes, candidates)):
            if len(ids) == 1 and ids[0] not in ambiguous:
                track_id = ids[0]
                gap = now - seen_before[track_id]
                if gap > 0.1:
                    self._note(now, f"ID {track_id} 다시 연결 ({gap:.2f}초 만)")
            else:
                self.serial += 1
                track_id = self.serial
                self._filters[track_id] = OneEuroPoint(1.0, 0.02)
                self._note(now, f"ID {track_id} 새로 생김")
            self._raw[track_id] = box
            x1, y1, x2, y2 = box
            cx, cy = self._filters[track_id]((x1 + x2) / 2, (y1 + y2) / 2, now)
            hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
            self._tracks[track_id] = FaceTrack(
                track_id, (cx - hw, cy - hh, cx + hw, cy + hh), now)
            if index < len(mouth_openings):
                self.mouths[track_id] = mouth_openings[index]
        # Keep a still-tracked partner, otherwise auto-select only a lone face.
        if not self.manual and self.selected_id not in self._tracks:
            tracks = self.tracks(now)
            self.selected_id = tracks[0].track_id if len(tracks) == 1 else None
        self._remember(now)
        return self.current(now)
