"""Recognize a CTC segment and speak its words locally, without a translation server."""
import math
import re

import config as C
from src import clock
from src.ctc_runtime import GLOSS_VERSION
from src.translation import Glossary


class DeviceSignClient:
    """CloudWorker's client interface, with all processing on this device.

    This explicit word-reading mode does not claim sentence translation or cloud
    speech calibration. The ordinary cloud response policy remains unchanged.
    """

    timeout = 30.0

    def __init__(self, recognizer, mapping=None, threshold=None):
        self.recognizer = recognizer
        self.threshold = float(C.CONF_THRESHOLD if threshold is None else threshold)
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("수어 신뢰도 기준은 0~1 사이여야 합니다")
        if mapping is not None:
            glossary = mapping if isinstance(mapping, Glossary) else Glossary.load(mapping)
            self.meanings = glossary.meanings
        else:
            # AI Hub word labels include homonym IDs (검사_0029). IDs are not speech.
            # Opaque gloss IDs need an explicit mapping; never guess their meaning.
            self.meanings = {}
            for label in recognizer.labels[1:]:
                word = re.sub(r"_\d+$", "", label).strip()
                if re.fullmatch(r"[가-힣]+(?: [가-힣]+)*", word):
                    self.meanings[label] = word
        missing = [g for g in recognizer.labels[1:] if g not in self.meanings]
        if missing:
            raise ValueError("한국어 뜻이 없는 수어 라벨: " + ", ".join(missing[:5])
                             + ". --sign-mapping 으로 뜻 매핑 JSON을 지정하세요")

    def payload(self, sequence, session_id, segment_id, stats=None):
        start = clock.now()
        glosses, score = self.recognizer.predict(sequence)
        if stats:
            stats.add("device_recognize", (clock.now() - start) * 1000)
        return {"version": GLOSS_VERSION, "session_id": session_id, "segment_id": segment_id,
                "glosses": list(glosses), "score": float(score)}

    def translate(self, payload):
        glosses, score = payload["glosses"], payload["score"]
        result = {k: payload[k] for k in ("version", "session_id", "segment_id")}
        result.update(glosses=glosses, score=score, text="", source="device_words",
                      status="no_sign", auto_speech=False, mock=False)
        if not glosses:
            return result
        if (not math.isfinite(score) or not self.threshold <= score <= 1
                or any(g not in self.meanings for g in glosses)):
            result["status"] = "uncertain"
            return result
        result.update(status="ok", text=" ".join(self.meanings[g] for g in glosses),
                      auto_speech=True)
        return result
