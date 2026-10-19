"""Korean captions drawn onto the video, independently of OLED coordinates.

Only the small pieces are drawn with PIL (header bar, track labels, the caption
bubble) and cached; the camera frame itself stays a numpy array. Converting a
1280x720 frame to PIL and back on every redraw cost about 13 ms per frame.
"""
from collections import OrderedDict
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.text_layout import korean_font, wrap_text

HEADER_BG = (20, 26, 36)          # RGB, as PIL draws it
SELECTED = (106, 206, 255)
CANDIDATE = (155, 165, 180)
CACHE_SIZE = 64


def _bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


def _blend(region, rgb, alpha):
    """region (h, w, 3) BGR uint8 in place: colour or sprite blended by alpha (h, w) 0..255."""
    if alpha.max() == 0:
        return
    a = alpha.astype(np.float32)[..., None] / 255.0
    region[:] = np.rint(region * (1.0 - a) + rgb * a).astype(np.uint8)


class VideoOverlay:
    def __init__(self):
        self.layouts = OrderedDict()
        self.sprites = OrderedDict()
        self.side = "right"
        self.track_id = None

    def _font(self, size):
        return korean_font(size)

    def _cached(self, key, build):
        if key in self.sprites:
            self.sprites.move_to_end(key)
        else:
            self.sprites[key] = build()
            if len(self.sprites) > CACHE_SIZE:
                self.sprites.popitem(last=False)
        return self.sprites[key]

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
            if len(self.layouts) > CACHE_SIZE:
                self.layouts.popitem(last=False)
        self.layouts.move_to_end(key)
        return self.layouts[key]

    # ---------------------------------------------------------- small pieces
    def _text_mask(self, text, size):
        """Glyph coverage of text drawn at the origin, and its offset (PIL 'la' anchor)."""
        def build():
            font = korean_font(size)
            left, top, right, bottom = font.getbbox(text)
            mask = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
            ImageDraw.Draw(mask).text((-left, -top), text, font=font, fill=255)
            return np.asarray(mask), left, top
        return self._cached(("text", text, size), build)

    def _put_text(self, output, xy, text, size, rgb):
        mask, left, top = self._text_mask(text, size)
        x, y = int(math.floor(xy[0])) + left, int(math.floor(xy[1])) + top
        h, w = output.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + mask.shape[1]), min(h, y + mask.shape[0])
        if x0 < x1 and y0 < y1:
            _blend(output[y0:y1, x0:x1], np.array(_bgr(rgb), np.float32),
                   mask[y0 - y:y1 - y, x0 - x:x1 - x])

    def _header(self, w, size, target, mode):
        def build():
            height = size * 2 + 19          # PIL's rectangle (0, 0, w, size*2+18) is inclusive
            image = Image.new("RGB", (w, height), HEADER_BG)
            draw = ImageDraw.Draw(image)
            font = korean_font(size)
            draw.text((10, 5), f"대화 상대: {target} · {mode}", font=font, fill=(230, 238, 245))
            draw.text((10, size + 10), "클릭: 선택 · 우클릭: 자동 · H: 기록", font=font,
                      fill=(160, 175, 190))
            return np.asarray(image)[:, :, ::-1].copy()
        return self._cached(("header", w, size, target, mode), build)

    def _bubble(self, lines, size, partial, bx, by, bw, bh, tail):
        """RGBA sprite of the bubble (tail, box, status and lines) and its integer origin."""
        xs = [bx, bx + bw] + ([p[0] for p in tail] if tail else [])
        ox, oy = int(math.floor(min(xs))), int(math.floor(by))
        # Same fractional offsets rasterize the same way, so the sprite can be reused
        # while the face (and so the bubble) moves by whole pixels.
        rel_tail = tuple((px - ox, py - oy) for px, py in tail) if tail else None
        key = ("bubble", tuple(lines), size, partial, bx - ox, by - oy, bw, bh, rel_tail)

        def build():
            width = int(math.ceil(max(xs) - ox)) + 2
            image = Image.new("RGBA", (width, int(math.ceil(bh)) + 2), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            font = korean_font(size)
            line_h = size + 8
            color = SELECTED if partial else (230, 238, 245)
            x, y = bx - ox, by - oy
            if rel_tail:
                draw.polygon(list(rel_tail), fill=color)
            draw.rounded_rectangle((x, y, x + bw, y + bh), radius=12, fill=HEADER_BG,
                                   outline=color, width=2)
            draw.text((x + 12, y + 8), "인식 중 ···" if partial else "인식 완료", font=font,
                      fill=color)
            for index, line in enumerate(lines):
                draw.text((x + 12, y + 8 + line_h * (index + 1)), line, font=font,
                          fill=(255, 255, 255))
            rgba = np.asarray(image)
            return rgba[:, :, 2::-1].astype(np.float32), rgba[:, :, 3].copy()
        bgr, alpha = self._cached(key, build)
        return bgr, alpha, ox, oy

    # ---------------------------------------------------------------- render
    def render(self, frame, caption, track, now, mirror=False, tracks=(), manual=False,
               selected_id=None):
        # Always a new array: debug drawing must not modify the capture worker's frame.
        output = cv2.flip(frame, 1) if mirror else frame.copy()
        h, w = output.shape[:2]
        ui_size = max(12, min(18, w // 32))
        for candidate in tracks:
            x1, y1, x2, y2 = candidate.box
            if mirror:
                x1, x2 = w - x2, w - x1
            selected = candidate.track_id == selected_id
            color = SELECTED if selected else CANDIDATE
            corners = [(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))]
            if selected:            # PIL width=2 draws the second line inside the first
                a, b, c, d = corners[0]
                corners.append((a + 1, b + 1, c - 1, d - 1))
            for a, b, c, d in corners:
                cv2.rectangle(output, (a, b), (c, d), _bgr(color), 1)
            self._put_text(output, (x1 + 4, max(62, y1 - 24)), f"대상 {candidate.track_id}",
                           ui_size, color)
        target = f"대상 {track.track_id}" if track else "미지정"
        header = self._header(w, ui_size, target, "수동" if manual else "자동")
        rows = min(h, header.shape[0])
        output[:rows] = header[:rows]
        if not caption.visible(now):
            return output
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
            return output
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
        bgr, alpha, ox, oy = self._bubble(lines, size, caption.partial, bx, by, bw, bh, tail)
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(w, ox + alpha.shape[1]), min(h, oy + alpha.shape[0])
        if x0 < x1 and y0 < y1:
            _blend(output[y0:y1, x0:x1], bgr[y0 - oy:y1 - oy, x0 - ox:x1 - ox],
                   alpha[y0 - oy:y1 - oy, x0 - ox:x1 - ox])
        return output
