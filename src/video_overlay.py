"""Korean captions drawn onto the video, independently of OLED coordinates."""
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw

from src.text_layout import korean_font, wrap_text


class VideoOverlay:
    def __init__(self):
        self.layouts = OrderedDict()
        self.side = "right"
        self.track_id = None

    def _font(self, size):
        return korean_font(size)

    def _lines(self, text, font, width, size, max_lines):
        key = (text, width, size, max_lines)
        if key not in self.layouts:
            lines = wrap_text(text, font, width)
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
                lines[0] = "…" + lines[0]
                while font.getlength(lines[0]) > width and len(lines[0]) > 1:
                    lines[0] = "…" + lines[0][2:]
            self.layouts[key] = lines
            if len(self.layouts) > 64:
                self.layouts.popitem(last=False)
        self.layouts.move_to_end(key)
        return self.layouts[key]

    def render(self, frame, caption, track, now, mirror=False, tracks=(), manual=False,
               selected_id=None):
        # Always copy: debug drawing must not modify the capture worker's frame.
        output = frame[:, ::-1].copy() if mirror else frame.copy()
        h, w = output.shape[:2]
        image = Image.fromarray(output[:, :, ::-1])
        draw = ImageDraw.Draw(image)
        for candidate in tracks:
            x1, y1, x2, y2 = candidate.box
            if mirror:
                x1, x2 = w - x2, w - x1
            selected = candidate.track_id == selected_id
            color = (106, 206, 255) if selected else (155, 165, 180)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2 if selected else 1)
            draw.text((x1 + 4, max(62, y1 - 24)), f"대상 {candidate.track_id}",
                      font=korean_font(max(12, min(18, w // 32))), fill=color)
        ui_size = max(12, min(18, w // 32))
        draw.rectangle((0, 0, w, ui_size * 2 + 18), fill=(20, 26, 36))
        target = f"대상 {track.track_id}" if track else "미지정"
        mode = "수동" if manual else "자동"
        draw.text((10, 5), f"대화 상대: {target} · {mode}", font=korean_font(ui_size),
                  fill=(230, 238, 245))
        draw.text((10, ui_size + 10), "클릭: 선택 · 우클릭: 자동 · H: 기록",
                  font=korean_font(ui_size), fill=(160, 175, 190))
        if not caption.visible(now):
            return np.asarray(image)[:, :, ::-1].copy()
        size = max(12, min(28, w // 32, h // 12))
        font = self._font(size)
        anchor = caption.anchor(track, now)
        box = anchor.box if anchor else None
        if box and mirror:
            box = (w - box[2], box[1], w - box[0], box[3])
        width = min(w - 24, max(160, int(w * 0.36)))
        if box:
            space = int(max(box[0] - 24, w - box[2] - 24))
            if space >= max(120, size * 6 + 24):
                width = min(width, space)
        if width < 24 or h < 160:
            return np.asarray(image)[:, :, ::-1].copy()
        line_h = size + 8
        max_lines = max(1, min(3, (h - 90) // line_h - 1))
        lines = self._lines(caption.text, font, width - 24, size, max_lines)
        bw, bh = width, line_h * (len(lines) + 1) + 20
        tail = None
        if box:
            x1, y1, x2, y2 = box
            if self.track_id != anchor.track_id:
                self.side, self.track_id = "right", anchor.track_id
            right, left = w - x2 - 24, x1 - 24
            available = right if self.side == "right" else left
            other = left if self.side == "right" else right
            if available < bw and other >= bw:
                self.side = "left" if self.side == "right" else "right"
                available = other
            if available >= bw:
                bx = x2 + 16 if self.side == "right" else x1 - 16 - bw
                by = max(62, min(y1, h - bh - 8))
                ty = max(by + 10, min((y1 + y2) / 2, by + bh - 10))
                edge = bx if self.side == "right" else bx + bw
                tip = edge - 12 if self.side == "right" else edge + 12
                tail = [(edge, ty - 7), (tip, ty), (edge, ty + 7)]
            else:
                box = None  # too little room: readable fixed caption
        if not box:
            bx, by = (w - bw) / 2, max(4, h - bh - 16)
        color = (106, 206, 255) if caption.partial else (230, 238, 245)
        if tail:
            draw.polygon(tail, fill=color)
        draw.rounded_rectangle((bx, by, bx + bw, by + bh), radius=12,
                               fill=(20, 26, 36), outline=color, width=2)
        status = "인식 중 ···" if caption.partial else "인식 완료"
        draw.text((bx + 12, by + 8), status, font=font, fill=color)
        for index, line in enumerate(lines):
            draw.text((bx + 12, by + 8 + line_h * (index + 1)), line,
                      font=font, fill=(255, 255, 255))
        return np.asarray(image)[:, :, ::-1].copy()
