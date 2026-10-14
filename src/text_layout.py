"""Shared Korean font loading and lossless pixel-width wrapping."""
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

import config as C
from src.hud import HudRenderer


@lru_cache(maxsize=16)
def korean_font(size):
    for path in [str(C.HUD_FONT), *HudRenderer.SYSTEM_KO]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    raise RuntimeError("말풍선에 사용할 한글 폰트를 찾을 수 없습니다")


def wrap_text(text, font, width):
    """Wrap every character, retaining spaces and explicit newlines."""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        remaining = paragraph
        while remaining:
            if font.getlength(remaining) <= width:
                lines.append(remaining)
                break
            # Measure a logarithmic number of prefixes rather than reshaping
            # every character for every historical sentence on each update.
            low, high = 1, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if font.getlength(remaining[:middle]) <= width:
                    low = middle
                else:
                    high = middle - 1
            cut = low
            # Prefer a nearby word boundary, keeping spaces in the transcript.
            boundary = remaining.rfind(" ", 0, cut)
            if boundary >= cut // 2:
                cut = boundary + 1
            lines.append(remaining[:cut])
            remaining = remaining[cut:]
    return lines
