"""서버 없이 이 기기에서 수어 구간을 인식하고 문장으로 바꿉니다.

CloudWorker 가 쓰는 클라이언트와 같은 모양(payload, translate, timeout)이라
CloudClient 자리에 그대로 들어갑니다. 네트워크는 쓰지 않습니다.

1. OnnxRecognizer 로 구간을 인식합니다 (src/ctc_runtime.py).
2. CTC가 읽은 단어열에 정확히 맞는 등록 문장을 먼저 찾습니다.
3. 나머지는 상대 발화를 참고해 문장으로 만들고, 실패하면 인식 단어를 읽습니다.
"""
import json
import re
from pathlib import Path

import numpy as np

from src import clock
from src.ctc_runtime import GLOSS_VERSION, greedy_decode
from src.pose_sequence import FEATURE_SETS

_VARIANT = re.compile(r"_\d+$")      # 같은 이름의 다른 수어를 구분하는 번호 (검사_0029)
_NEUTRAL_PARTICLES = {"은", "는", "이", "가", "을", "를", "요"}
_STATEMENT_ENDINGS = {"다", "아요", "어요", "습니다", "ᆸ니다", "ᆫ다", "는다"}


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

    def __init__(self, recognizer, phrases=(), *, heard_seconds=20.0, save_dir=None, composer=None):
        """save_dir: 인식한 구간의 좌표와 결과를 남길 폴더 (실제 카메라 문제를 살펴볼 때)."""
        self.save_dir = Path(save_dir) if save_dir else None
        self.recognizer = recognizer
        self.heard_seconds = heard_seconds
        phrases = list(phrases)
        self.texts = {tuple(p["glosses"]): p["text"] for p in phrases}
        self._heard = ("", float("-inf"))
        self.composer = composer
        if composer is not None:
            from src.translation import Glossary, learn_stems
            meanings = {g: plain_words([g]) for g in recognizer.labels[1:]}
            kinds = {g: "negation" for g, word in meanings.items()
                     if word in ("없다", "아니다", "못하다")}
            kinds.update({g: "place" for g, word in meanings.items() if word == "병원"})
            self.glossary = Glossary(meanings, kinds=kinds, stems=learn_stems(meanings, phrases))
            self._kiwi = None
            self.timeout = 30
            composer.max_seconds = min(getattr(composer, "max_seconds", None) or 15.0, 15.0)

    def hear(self, text, at=None):
        """상대가 방금 한 말(음성 인식 확정 결과). 잠시 동안만 문맥으로 씁니다."""
        self._heard = (text or "", clock.now() if at is None else at)

    def heard(self, now=None):
        text, at = self._heard
        now = clock.now() if now is None else now
        return text if now - at <= self.heard_seconds else ""

    def recognize(self, sequence):
        """Decode the visual model output before consulting the conversation."""
        sequence.validate()
        if not sequence.mask[:, :42].any():
            return [], 0.0
        features = FEATURE_SETS[self.recognizer.feature_set][0](sequence)
        return greedy_decode(np.exp(self.recognizer.log_probs(features)), self.recognizer.labels)

    def payload(self, sequence, session_id, segment_id, stats=None):
        start = clock.now()
        context = self.heard()
        glosses, score = self.recognize(sequence)
        if glosses:
            print(f"[수어 인식] {plain_words(glosses)} (CTC 점수 {score:.3f})", flush=True)
        if stats:
            stats.add("device_recognize", (clock.now() - start) * 1000)
        if self.save_dir:
            try:        # 진단용 저장이 인식을 막으면 안 됩니다
                self.save_dir.mkdir(parents=True, exist_ok=True)
                name = f"{session_id}_{segment_id:04d}" if isinstance(segment_id, int) \
                    else f"{session_id}_{segment_id}"
                sequence.save(self.save_dir / f"{name}.npz")
                (self.save_dir / f"{name}.json").write_text(json.dumps(
                    {"glosses": list(glosses), "score": float(score), "source": "greedy",
                     "frames": len(sequence.timestamps),
                     "seconds": float(sequence.timestamps[-1] - sequence.timestamps[0])},
                    ensure_ascii=False), encoding="utf-8")
            except (OSError, ValueError) as error:
                print(f"[수어] 구간 저장 실패: {error}")
        return {"version": GLOSS_VERSION, "session_id": session_id, "segment_id": segment_id,
                "glosses": list(glosses), "score": float(score), "source": "greedy", "context": context}

    def word_result(self, payload):
        """A valid local recognition result, also usable when composition times out."""
        glosses = payload["glosses"]
        words = plain_words(glosses)
        result = {key: payload[key] for key in ("version", "session_id", "segment_id")}
        result.update(glosses=glosses, recognized_words=words, recognition_score=payload["score"],
                      status="ok" if words else "no_sign", text=words, speech=words,
                      source="words" if words else "", auto_speech=bool(words), mock=False,
                      timings_ms={})
        return result

    def _content(self, text):
        if self._kiwi is None:
            from kiwipiepy import Kiwi
            self._kiwi = Kiwi(num_workers=1)
        # Ignore only ordinary grammatical insertions. Direction, tense and
        # sentence mood (e.g. 에서, 세요 and ?) need evidence in the input.
        parts = []
        for token in self._kiwi.tokenize(text):
            form, tag = token.form, token.tag
            if (tag.startswith("J") and form in _NEUTRAL_PARTICLES
                    or tag == "EF" and form in _STATEMENT_ENDINGS
                    or tag in ("SF", "SP") and form in (".", ",")):
                continue
            parts.append(form)
        return "".join(parts)

    def translate(self, payload):
        result = self.word_result(payload)
        threshold = self.recognizer.calibration.get("threshold")
        glosses = result["glosses"]
        if not glosses or (threshold is not None and payload["score"] < threshold):
            return result
        text = self.texts.get(tuple(glosses))
        if text:
            result.update(text=text, speech=text, source="phrase")
        elif self.composer is not None:
            from src.translation import compose
            start = clock.now()
            made = compose(glosses, self.glossary, self.composer,
                           context=payload.get("context") or None)
            issues = list(made.issues)
            if made.status == "ok":
                try:
                    expected = self._content(result["recognized_words"])
                    if expected and self._content(made.text) == expected:
                        result.update(text=made.text, speech=made.text, source=made.source)
                    else:
                        issues.append("changed_words")
                except Exception as error:
                    issues.append(f"word_check:{type(error).__name__}")
            result["issues"] = issues
            result["timings_ms"]["language"] = (clock.now() - start) * 1000
            print(f"[문장 생성: {result['source']}] {result['text']}"
                  + (f" ({', '.join(issues)})" if issues else ""), flush=True)
        return result
