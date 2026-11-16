"""안경 시점 수어 -> Qwen 미세조정 모델 -> 문장. 서버 없이 이 기기에서 돌아갑니다.

CloudWorker 가 쓰는 클라이언트(payload, translate, timeout)와 같은 모양이라 LocalSignClient
자리에 그대로 들어갑니다.

1. 상황 문장 목록(+단어 하나, 수어 아님) 중에서 Qwen 이 가장 그럴듯한 답을 고릅니다 (src.train_qwen.rank).
   자유롭게 문장을 쓰지 않으므로 하지 않은 말을 지어내지 않습니다.
2. 같은 모델의 단어 판정층이 본 단어열과 비교합니다. 둘이 다르면 점수를 절반으로 낮추고
   화면에 "(추정)" 을 붙입니다.
3. 점수가 speak_min 이상이면 소리로 말합니다.

composer(문장 모델, 예: src.translation.LocalTranslator 의 Qwen2.5-3B)를 주면, 목록 문장이
맞지 않을 때 단어 판정층이 본 단어와 상대가 한 말로 새 문장을 만듭니다
(src.translation.compose: 숫자·부정·사람이 바뀌면 거르고, 실패하면 소리 없는 미리보기).
"""
import json
from pathlib import Path

import numpy as np

from src import clock
from src.ctc_runtime import GLOSS_VERSION
from src.local_sign import GUESS, plain_words

ROOT = Path(__file__).resolve().parents[1]


REVIEWER = "팀 작성 (수어 전문가 검수 전)"


def glasses_glossary(vocabulary, phrases):
    """안경 단어의 뜻(단어 그대로)과 상황 문장. 문장 모델 결과의 보조 검사에 씁니다."""
    from src.translation import Glossary
    meanings = {w["gloss"]: w["gloss"] for w in vocabulary["words"]}
    kinds = {g: "negation" for g in ("못하다", "없다") if g in meanings}
    kinds.update({g: "person" for g in ("엄마", "남편", "아내", "친구") if g in meanings})
    sentences = [{"glosses": p["glosses"], "text": p["text"], "reviewed_by": REVIEWER}
                 for p in phrases if all(g in meanings for g in p["glosses"])]
    return Glossary(meanings, kinds=kinds, sentences=sentences, strict_additions=True)


class QwenSignClient:
    timeout = 20

    def __init__(self, path, *, device=None, phrases=None, vocabulary=None, speak_min=0.5,
                 heard_seconds=20.0, save_dir=None, composer=None):
        import torch
        from src.train_qwen import candidate_texts, load
        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model, self.tokenizer, self.spec = load(path, device=device, attention="sdpa",
                                                     merge=True)
        self.device = device
        phrases = phrases or json.loads((ROOT / "assets/glasses/phrases.json").read_text(
            encoding="utf-8"))
        vocabulary = vocabulary or json.loads((ROOT / "assets/glasses/vocabulary.json").read_text(
            encoding="utf-8"))
        words = [w["gloss"] for w in vocabulary["words"] if w["gloss"] in self.model.glosses]
        self.options = candidate_texts(self.model, phrases, words)
        self.phrase_texts = {p["text"] for p in phrases}
        self.composer = composer
        self.glossary = glasses_glossary(vocabulary, phrases) if composer is not None else None
        self.speak_min, self.heard_seconds = speak_min, heard_seconds
        self.save_dir = Path(save_dir) if save_dir else None
        self._heard = ("", float("-inf"))

    def hear(self, text, at=None):
        self._heard = (text or "", clock.now() if at is None else at)

    def heard(self, now=None):
        text, at = self._heard
        now = clock.now() if now is None else now
        return text if now - at <= self.heard_seconds else ""

    def recognize(self, sequence):
        """(글, 단어열, 점수, 단어 판정과 일치 여부, 후보 상위 3개)."""
        from src.train_qwen import rank
        sequence.validate()
        if not sequence.mask[:, :42].any():
            return "", [], 0.0, True, []
        posterior, words, seen = rank(self.model, self.tokenizer, sequence, self.options,
                                      self.device, heard=self.heard() or None, with_score=True)
        best = max(posterior, key=posterior.get)
        glosses = self.options[best]
        agree = words == glosses
        score = posterior[best] * (1.0 if agree else 0.5)
        top = sorted(posterior.items(), key=lambda kv: -kv[1])[:3]
        if self.composer is not None and words and not agree:
            # 목록 문장이 본 단어와 다르면, 본 단어로 문장 모델이 새로 만듭니다.
            from src.translation import compose
            made = compose(words, self.glossary, self.composer, context=self.heard() or None)
            self.last_composition = made
            if made.status == "ok":
                return made.text, words, seen, True, top
            return made.text, words, 0.0, False, top
        self.last_composition = None
        return best, glosses, score, agree, top

    def payload(self, sequence, session_id, segment_id, stats=None):
        start = clock.now()
        text, glosses, score, agree, top = self.recognize(sequence)
        if stats:
            stats.add("device_recognize", (clock.now() - start) * 1000)
        if self.save_dir:
            try:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                name = f"{session_id}_{segment_id}"
                sequence.save(self.save_dir / f"{name}.npz")
                (self.save_dir / f"{name}.json").write_text(json.dumps(
                    {"text": text, "glosses": glosses, "score": score, "agree": agree,
                     "top": top}, ensure_ascii=False), encoding="utf-8")
            except (OSError, ValueError) as error:
                print(f"[수어] 구간 저장 실패: {error}")
        made = getattr(self, "last_composition", None)
        source = ("composed" if made is not None and made.status == "ok" else
                  made.source if made is not None else "phrase" if agree else "guess")
        return {"version": GLOSS_VERSION, "session_id": session_id, "segment_id": segment_id,
                "glosses": list(glosses), "score": float(score), "source": source, "text": text}

    def translate(self, payload):
        glosses = payload["glosses"]
        result = {key: payload[key] for key in ("version", "session_id", "segment_id")}
        result.update(glosses=glosses, text="", source="", auto_speech=False, mock=False,
                      timings_ms={})
        if not glosses:
            result["status"] = "no_sign"
            return result
        if payload["source"] in ("reviewed_rules", "words"):
            # 문장 모델이 실패했거나 검사에 걸림: 소리 없는 미리보기 (앱이 "다시 입력"을 안내)
            result.update(status="fallback", source=payload["source"], text=payload["text"])
            return result
        if payload["source"] == "composed":
            result.update(status="ok", source="composed", text=payload["text"],
                          speech=payload["text"],
                          auto_speech=bool(payload["score"] >= self.speak_min))
            return result
        spoken = payload["text"] if payload["text"] in self.phrase_texts else plain_words(glosses)
        guessed = payload["source"] == "guess"
        result.update(status="ok", source="phrase" if spoken in self.phrase_texts else "words",
                      text=(GUESS if guessed else "") + spoken, speech=spoken,
                      auto_speech=bool(payload["score"] >= self.speak_min))
        return result
