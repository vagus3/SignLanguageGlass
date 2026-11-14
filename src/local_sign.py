"""서버 없이 이 기기에서 수어 구간을 인식하고 문장으로 바꿉니다.

CloudWorker 가 쓰는 클라이언트와 같은 모양(payload, translate, timeout)이라
CloudClient 자리에 그대로 들어갑니다. 네트워크는 쓰지 않습니다.

1. OnnxRecognizer 로 구간을 인식합니다 (src/ctc_runtime.py).
2. 상황 문장 목록(assets/phrases/*.json)이 있으면 decode_with_phrases 로 보정합니다.
   상대가 방금 한 말(heard)은 그 말에 대한 대답일 법한 문장에 가산점을 줍니다.
3. 목록에 있는 단어열이면 그 문장을, 아니면 인식된 단어를 그대로 보여 줍니다.

문맥으로 고른 문장은 추정입니다. "(추정)" 을 붙여 보여 주고, 자동으로 말하지 않습니다.
"""
import json
import re
from pathlib import Path

import numpy as np

from src import clock
from src.ctc_runtime import GLOSS_VERSION, decode_with_phrases
from src.pose_sequence import FEATURE_SETS

GUESS = "(추정) "
_VARIANT = re.compile(r"_\d+$")      # 같은 이름의 다른 수어를 구분하는 번호 (검사_0029)


def load_phrases(path, labels):
    """[{"glosses", "text", "after"}]; 모델이 모르는 단어가 든 문장은 빼고 몇 개인지 돌려줍니다."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("문장 목록은 JSON 배열이어야 합니다")
    known, dropped = [], 0
    for entry in entries:
        glosses, text = entry.get("glosses"), entry.get("text")
        if (not isinstance(glosses, list) or not glosses or not isinstance(text, str)
                or not text.strip()):
            raise ValueError(f"문장 항목에는 glosses 와 text 가 필요합니다: {entry!r}")
        if all(g in labels for g in glosses):
            known.append({"glosses": list(glosses), "text": text.strip(),
                          "after": list(entry.get("after") or ())})
        else:
            dropped += 1
    return known, dropped


def plain_words(glosses):
    return " ".join(_VARIANT.sub("", g) for g in glosses)


class LocalSignClient:
    timeout = 10

    def __init__(self, recognizer, phrases=(), *, bonus=1.0, heard_seconds=20.0):
        self.recognizer, self.phrases = recognizer, list(phrases)
        self.bonus, self.heard_seconds = bonus, heard_seconds
        self.texts = {tuple(p["glosses"]): p["text"] for p in self.phrases}
        self._heard = ("", float("-inf"))

    def hear(self, text, at=None):
        """상대가 방금 한 말(음성 인식 확정 결과). 잠시 동안만 문맥으로 씁니다."""
        self._heard = (text or "", clock.now() if at is None else at)

    def heard(self, now=None):
        text, at = self._heard
        now = clock.now() if now is None else now
        return text if now - at <= self.heard_seconds else ""

    def recognize(self, sequence):
        """(단어열, 점수, 출처). 출처: "greedy" = 본 그대로, "phrase" = 문맥으로 고름."""
        sequence.validate()
        if not sequence.mask[:, :42].any():
            return [], 0.0, "greedy"
        features = FEATURE_SETS[self.recognizer.feature_set][0](sequence)
        return decode_with_phrases(self.recognizer.log_probs(features), self.recognizer.labels,
                                   self.phrases, bonus=self.bonus, heard=self.heard())

    def payload(self, sequence, session_id, segment_id, stats=None):
        start = clock.now()
        glosses, score, source = self.recognize(sequence)
        if stats:
            stats.add("device_recognize", (clock.now() - start) * 1000)
        return {"version": GLOSS_VERSION, "session_id": session_id, "segment_id": segment_id,
                "glosses": list(glosses), "score": float(score), "source": source}

    def translate(self, payload):
        glosses = payload["glosses"]
        result = {key: payload[key] for key in ("version", "session_id", "segment_id")}
        result.update(glosses=glosses, text="", source="", auto_speech=False, mock=False,
                      timings_ms={})
        threshold = self.recognizer.calibration.get("threshold")
        if not glosses:
            result["status"] = "no_sign"
        elif (payload["source"] == "greedy" and threshold is not None
              and payload["score"] < threshold):
            result["status"] = "uncertain"
        else:
            text = self.texts.get(tuple(glosses))
            guessed = payload["source"] == "phrase"
            result.update(status="ok", source="phrase" if text else "words",
                          text=(GUESS if guessed else "") + (text or plain_words(glosses)))
        return result
