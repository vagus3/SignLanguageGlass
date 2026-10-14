"""Scrollable transcript window; only final results enter the history."""
from PIL import Image, ImageDraw
import numpy as np

from src.text_layout import korean_font, wrap_text


class TranscriptView:
    def __init__(self, width=640, height=640):
        self.width, self.height = width, height
        self.offset = 0  # wrapped lines above the newest page
        self.max_offset = 0
        self._revision = -1
        self._rows = []
        self._image = None
        self._key = None
        self._record_rows = {}
        self._last_id = -1

    def scroll(self, lines):
        # The exact upper bound is discovered lazily while scrolling backwards.
        self.offset = max(0, self.offset + lines)

    def latest(self):
        self.offset = 0

    def _layout_record(self, record):
        if record not in self._record_rows:
            target = (f"대상 {record.track_id}" if record.track_id is not None
                      else "대상 미지정")
            rows = [(f"발화 {record.utterance_id} · {target}", True)]
            rows.extend((line, False) for line in
                        wrap_text(record.text, korean_font(20), self.width - 40))
            rows.append(("", False))
            self._record_rows[record] = rows
        return self._record_rows[record]

    def render(self, history):
        font = korean_font(20)
        if self._revision != history.revision:
            if self.offset:
                self.offset += sum(len(self._layout_record(record)) for record in history.records
                                   if record.utterance_id > self._last_id)
            live = set(history.records)
            self._record_rows = {record: rows for record, rows in self._record_rows.items()
                                 if record in live}
            self._last_id = history.records[-1].utterance_id if history.records else -1
            self._revision = history.revision
        key = (history.revision, self.offset)
        if self._key == key:
            return self._image.copy()
        page_lines = (self.height - 120) // 28
        groups, count = [], 0
        for record in reversed(history.records):
            rows = self._layout_record(record)
            groups.append(rows)
            count += len(rows)
            if count >= page_lines + self.offset:
                break
        self._rows = [row for group in reversed(groups) for row in group]
        self.max_offset = max(0, len(self._rows) - page_lines)
        self.offset = min(self.offset, self.max_offset)
        key = (history.revision, self.offset)
        image = Image.new("RGB", (self.width, self.height), (20, 26, 36))
        draw = ImageDraw.Draw(image)
        draw.text((20, 16), f"대화 기록 · 최근 {len(history.records)}개", font=font,
                  fill=(230, 238, 245))
        draw.text((20, 48), "휠: 이전/다음  |  L: 최신  |  H: 닫기", font=korean_font(16),
                  fill=(160, 175, 190))
        end = len(self._rows) - self.offset
        rows = self._rows[max(0, end - page_lines):end]
        if not rows:
            rows = [("확정된 음성이 여기에 표시됩니다.", True)]
        for index, (line, label) in enumerate(rows):
            draw.text((20, 88 + index * 28), line, font=font,
                      fill=(106, 206, 255) if label else (255, 255, 255))
        draw.text((20, self.height - 28), "앱을 종료하면 기록이 지워집니다", font=korean_font(16),
                  fill=(160, 175, 190))
        self._image = np.asarray(image)[:, :, ::-1].copy()
        self._key = key
        return self._image.copy()
