"""서버 없이 이 기기에서 수어 구간을 인식하고 문장으로 바꿉니다.

CloudWorker 가 쓰는 클라이언트와 같은 모양(payload, translate, timeout)이라
CloudClient 자리에 그대로 들어갑니다. 네트워크는 쓰지 않습니다.

1. OnnxRecognizer 로 구간을 인식합니다 (src/ctc_runtime.py).
2. CTC가 읽은 단어열에 정확히 맞는 등록 문장을 먼저 찾습니다.
3. 나머지는 (인식 점수가 낮아도) 상대 발화를 참고해 문장 모델로 문장을 만들어 소리로 냅니다.
   뜻을 바꾼 문장(단어 빠짐, 없던 부정·질문·숫자·명사 추가)이면 인식 단어를 그대로 읽습니다.
"""
import json
import re
from pathlib import Path

import numpy as np

from src import clock
from src.ctc_runtime import GLOSS_VERSION, greedy_decode
from src.pose_sequence import FEATURE_SETS

_VARIANT = re.compile(r"_\d+$")      # 같은 이름의 다른 수어를 구분하는 번호 (검사_0029)
_DIGITS = re.compile(r"\d+")
# 문장 모델이 만든 문장이 인식 단어의 뜻을 지켰는지 볼 때 쓰는 표. 등록 문장과 문장 짝(사람이 쓴
# 410개)에서 뜻은 같고 말만 다른 경우를 모았습니다. 이 표로 사람이 쓴 문장 중 6개만 걸립니다.
_SAME_MEANING = {
    "노인": ("어르신",), "골절": ("뼈", "부러지"), "가능": ("수있", "되"), "사라지다": ("없어",),
    "실종": ("없어", "사라지"), "전화걸다": ("전화",), "곳": ("어디",), "틀리다": ("아니",),
    "맞다": ("네",), "핸드폰": ("휴대폰",), "화상": ("데",), "피해자": ("피해",),
    "도움받다": ("도와", "도움"), "통증": ("아프", "아파"), "밥": ("식사", "먹"),
    "잃어버리다": ("잃어",), "원하다": ("싶",), "부탁": ("주세요", "주시"), "나쁘다": ("안좋",),
    "자다": ("잠",), "때문": ("아서", "어서", "해서", "니까", "라서"), "혹시": ("?",),
    "약": ("약국",), "경찰": ("경찰서",)}
# 문장에 부정(안·못·않·없·아니)이 들어가도 되는 단어. 이 밖에는 부정이 생기면 뜻이 뒤집힌 것입니다.
_NEGATIVE_WORDS = {"없다", "아니다", "못하다", "틀리다", "사라지다", "실종", "잃어버리다", "싫다",
                   "나쁘다"}
# 묻는 말(?)이 되어도 되는 단어. 표정 없이 손만 보므로, 이런 단어가 없으면 묻는 말로 바꾸지 않습니다.
_QUESTION_WORDS = ("혹시", "얼마", "어디", "무엇", "뭐", "언제", "누구", "왜", "몇", "어떻", "가능", "곳")
# 문장을 자연스럽게 하려고 넣어도 되는 일반 명사. 그 밖의 새 명사(증상, 장소, 사람)는 지어낸 말입니다.
_GENERIC_NOUNS = {"사람", "것", "수", "때", "저", "제", "분", "좀", "지금", "여기", "저기", "일", "네"}
_CONTENT_TAGS = ("NNG", "NNP", "NR", "SN", "VV", "VA", "VX", "VCN", "XR", "XSV", "XSA", "MAG")


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
        self._kiwi = None
        self.composer = composer
        if composer is not None:
            from src.translation import Glossary, learn_stems
            meanings = {g: plain_words([g]) for g in recognizer.labels[1:]}
            kinds = {g: "negation" for g, word in meanings.items()
                     if word in ("없다", "아니다", "못하다")}
            kinds.update({g: "place" for g, word in meanings.items() if word == "병원"})
            self.glossary = Glossary(meanings, kinds=kinds, stems=learn_stems(meanings, phrases))
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

    def _tokens(self, text):
        if self._kiwi is None:
            from kiwipiepy import Kiwi
            self._kiwi = Kiwi(num_workers=1)
        return [(t.form, t.tag.split("-")[0]) for t in self._kiwi.tokenize(text)]

    def faithful(self, sentence, words, context=""):
        """문장이 인식 단어(words, 변형 번호 없이)의 뜻을 지켰는지. 문제 목록을 돌려줍니다(비면 통과).

        조사·어미·이음말과 '가야 해요' 같은 동사는 붙여도 됩니다. 막는 것: 인식 단어가 빠짐,
        없던 부정(안 괜찮아요), 없던 질문(아파요?), 숫자 변화, 입력과 상대 말에 없던 명사
        (치료가 필요해요)."""
        tokens = self._tokens(sentence)
        flat = sentence.replace(" ", "")
        lemmas = "".join(form for form, tag in tokens if tag in _CONTENT_TAGS)
        morphemes = "".join(form for form, _ in tokens)      # 아파서 -> 아프어서

        def forms(word):
            stem = word[:-1] if word.endswith("다") and len(word) >= 2 else word
            found = {word, stem, *_SAME_MEANING.get(word, ())}
            if stem.endswith("하") and len(stem) >= 3:        # 피곤하다 -> 피곤
                found.add(stem[:-1])
            return found

        def negative(parts):
            return any(tag == "VCN" or form == "못하" or (tag == "MAG" and form in ("안", "못"))
                       or (tag == "VX" and form in ("않", "말")) or (tag == "VA" and form == "없")
                       for form, tag in parts)

        issues = [f"dropped:{w}" for w in words
                  if not any(f in flat or f in lemmas or f in morphemes for f in forms(w))]
        if negative(tokens) and not set(words) & _NEGATIVE_WORDS:
            issues.append("negation_added")
        if "?" in sentence and not any(q in w for w in words for q in _QUESTION_WORDS):
            issues.append("question")
        if _DIGITS.findall(sentence) != _DIGITS.findall(" ".join(words)):
            issues.append("number")
        allowed = "".join(f for w in words for f in forms(w)) + (context or "")
        issues += [f"added:{form}" for form, tag in tokens
                   if tag in ("NNG", "NNP") and form not in _GENERIC_NOUNS and form not in allowed]
        return issues

    def translate(self, payload):
        result = self.word_result(payload)
        glosses = result["glosses"]
        if not glosses:
            return result
        # 인식 점수가 낮아도 문장을 만들어 말합니다 (낮은 점수는 로그에만 표시).
        threshold = self.recognizer.calibration.get("threshold")
        result["low_score"] = bool(threshold is not None and payload["score"] < threshold)
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
                    issues += self.faithful(made.text, result["recognized_words"].split(),
                                            payload.get("context") or "")
                except Exception as error:
                    issues.append(f"word_check:{type(error).__name__}")
                if not issues:
                    result.update(text=made.text, speech=made.text, source=made.source)
                else:
                    result["rejected_text"] = made.text
            result["issues"] = issues
            result["timings_ms"]["language"] = (clock.now() - start) * 1000
            note = issues + (["인식 점수 낮음"] if result["low_score"] else [])
            print(f"[문장 생성: {result['source']}] {result['text']}"
                  + (f" (버린 문장: {result['rejected_text']})" if "rejected_text" in result else "")
                  + (f" ({', '.join(note)})" if note else ""), flush=True)
        return result
