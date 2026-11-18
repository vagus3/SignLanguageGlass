"""서버 없이 이 기기에서 수어 구간을 인식하고 문장으로 바꿉니다.

CloudWorker 가 쓰는 클라이언트와 같은 모양(payload, translate, timeout)이라
CloudClient 자리에 그대로 들어갑니다. 네트워크는 쓰지 않습니다.

1. OnnxRecognizer 로 구간을 인식합니다 (src/ctc_runtime.py).
2. 상황 문장 목록(assets/phrases/*.json)이 있으면 decode_with_phrases 로 보정합니다.
   상대가 방금 한 말(heard)은 그 말에 대한 대답일 법한 문장에 가산점을 줍니다.
3. 목록에 있는 단어열이면 그 문장을, 아니면 인식된 단어를 그대로 보여 줍니다.

문맥으로 고른 문장도 화면에 그대로 보여 줍니다. 점수가 speak_min 이상이면 소리로 말합니다.
"""
import json
import re
from pathlib import Path

import numpy as np

from src import clock
from src.ctc_runtime import GLOSS_VERSION, decode_with_phrases
from src.pose_sequence import FEATURE_SETS

_VARIANT = re.compile(r"_\d+$")      # 같은 이름의 다른 수어를 구분하는 번호 (검사_0029)


def load_phrases(path, labels):
    """[{"glosses", "text", "after"}]; 모델이 모르는 단어가 든 문장은 빼고 몇 개인지 돌려줍니다.

    path 는 파일, 폴더(안의 *.json 전부), 또는 쉼표로 이은 여러 개입니다. 같은 단어열이
    여러 목록에 있으면 먼저 나온 것을 씁니다."""
    files = []
    for part in str(path).split(","):
        files += sorted(Path(part).glob("*.json")) if Path(part).is_dir() else [Path(part)]
    entries = []
    for file in files:
        loaded = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"{file}: 문장 목록은 JSON 배열이어야 합니다")
        entries += loaded
    known, dropped, seen = [], 0, set()
    for entry in entries:
        glosses, text = entry.get("glosses"), entry.get("text")
        if (not isinstance(glosses, list) or not glosses or not isinstance(text, str)
                or not text.strip()):
            raise ValueError(f"문장 항목에는 glosses 와 text 가 필요합니다: {entry!r}")
        if tuple(glosses) in seen:
            continue
        seen.add(tuple(glosses))
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

    def __init__(self, recognizer, phrases=(), *, bonus=1.0, heard_seconds=20.0, save_dir=None,
                 speak_min=0.0):
        """save_dir: 인식한 구간의 좌표와 결과를 남길 폴더 (실제 카메라 문제를 살펴볼 때)."""
        self.save_dir = Path(save_dir) if save_dir else None
        self.speak_min = speak_min      # 이 점수 이상이면 소리로 말함 (1 보다 크면 말하지 않음)
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
        if self.save_dir:
            try:        # 진단용 저장이 인식을 막으면 안 됩니다
                self.save_dir.mkdir(parents=True, exist_ok=True)
                name = f"{session_id}_{segment_id:04d}" if isinstance(segment_id, int) \
                    else f"{session_id}_{segment_id}"
                sequence.save(self.save_dir / f"{name}.npz")
                (self.save_dir / f"{name}.json").write_text(json.dumps(
                    {"glosses": list(glosses), "score": float(score), "source": source,
                     "frames": len(sequence.timestamps),
                     "seconds": float(sequence.timestamps[-1] - sequence.timestamps[0])},
                    ensure_ascii=False), encoding="utf-8")
            except (OSError, ValueError) as error:
                print(f"[수어] 구간 저장 실패: {error}")
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
            spoken = text or plain_words(glosses)
            result.update(status="ok", source="phrase" if text else "words",
                          text=spoken, speech=spoken,
                          auto_speech=bool(payload["score"] >= self.speak_min))
        return result
