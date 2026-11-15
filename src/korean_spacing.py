"""Offline spacing for captions; never change recognized non-whitespace text."""
import logging
import re
from functools import lru_cache


class KoreanSpacing:
    def __init__(self):
        self._kiwi = None
        self._prepared = False

    def prepare(self):
        if self._prepared:
            return
        self._prepared = True
        try:
            from kiwipiepy import Kiwi
            self._kiwi = Kiwi(num_workers=1)
            # Load dictionaries before microphone capture, not on the first turn.
            self._kiwi.space("안녕하세요")
        except Exception:
            self._kiwi = None
            logging.getLogger(__name__).warning(
                "한국어 띄어쓰기 보정 초기화 실패: 원문 자막을 사용합니다", exc_info=True)

    @lru_cache(maxsize=128)
    def format(self, text):
        if not text or not re.search(r"[가-힣]", text):
            return text
        self.prepare()
        if self._kiwi is None:
            return text
        try:
            spaced = self._kiwi.space(text, reset_whitespace=True)
            # Spacing models can also normalize spelling. Keep STT words intact.
            if re.sub(r"\s", "", spaced) == re.sub(r"\s", "", text):
                return spaced
        except Exception:
            logging.getLogger(__name__).warning(
                "한국어 띄어쓰기 보정 실패: 원문 자막을 사용합니다", exc_info=True)
        return text
