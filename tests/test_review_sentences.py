"""Semantic review sheet: export what would be spoken, import verdicts, reuse in calibration."""
import _path  # noqa: F401
import json
import tempfile
from pathlib import Path
import config as C
from src.translation import Glossary, MockTranslator, compose
from tools.evaluate_continuous import choose_threshold, review_verdict, summarize
from tools.review_sentences import (apply_review, default_sequences, draft_glossary,
                                    export_rows, read_sheet, write_sheet)


class Echo:
    """Deterministic stand-in for the language model; counts calls."""
    model = "fixture-llm"

    def __init__(self):
        self.calls = 0

    def translate(self, glosses, meanings):
        self.calls += 1
        return " ".join(meanings[g] for g in glosses) + "요."


glossary = draft_glossary()
assert "NEUTRAL" not in glossary.meanings and len(glossary.meanings) == len(C.VOCAB) - 1
sequences = default_sequences(glossary, extra=[["이름", "무엇"], ["병원", "없는단어"]])
assert sequences[:len(C.PHRASE_RULES)] == [list(p) for p, _ in C.PHRASE_RULES]
assert sequences.count(["이름", "무엇"]) == 1, "duplicates are removed"
llm = Echo()
rows = export_rows(glossary, sequences, llm)
assert rows[0]["rule_text"] == C.PHRASE_RULES[0][1] and rows[0]["source"] == "language_model"
assert rows[0]["model"] == "fixture-llm" and rows[0]["text"] == "이름 무엇요."
unknown = rows[-1]
assert unknown["source"] == "uncertain" and "unknown:없는단어" in unknown["issues"]

with tempfile.TemporaryDirectory() as td:
    sheet = Path(td) / "review_sheet.csv"
    write_sheet(rows, sheet)
    assert sheet.read_bytes().startswith(b"\xef\xbb\xbf"), "Excel needs the UTF-8 BOM"
    filled = read_sheet(sheet)
    assert [json.loads(r["glosses"]) for r in filled] == sequences
filled[0].update(meaning_preserved="Y", reviewer="검수자A")                     # approve
filled[1].update(corrected_text="병원이 어디인가요?", reviewer="검수자B")        # correct
filled[2].update(meaning_preserved="아니요", reviewer="검수자A", note="어색함")  # reject
# filled[3]: left empty -> skipped; the unknown-gloss row is also left empty.
updated, review, counts = apply_review(glossary, filled[:-1])
assert counts == {"approved": 1, "corrected": 1, "rejected": 1, "removed": 0,
                  "skipped": len(filled) - 4}, counts
assert updated.reviewed(["이름", "무엇"]) == "이름 무엇요."
assert updated.reviewed(["어디", "병원"]) == "병원이 어디인가요?"
assert updated.reviewed(list(C.PHRASE_RULES[2][0])) is None
assert review["reviewer"] == "검수자A, 검수자B"
assert [s["meaning_preserved"] for s in review["sequences"]] == [True, True, False]
assert updated.policy_hash() != glossary.policy_hash()
# Reviewed sentences can be synthesized ahead, so translation memory speaks from the cache.
from src.build_tts_cache import phrases
assert "병원이 어디인가요?" in phrases(updated) and "병원이 어디인가요?" not in phrases()
assert Glossary(**{"meanings": updated.meanings}).policy_hash() == glossary.policy_hash()
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "glossary.json"
    path.write_text(json.dumps(updated.to_dict(), ensure_ascii=False), encoding="utf-8")
    assert Glossary.load(path).policy_hash() == updated.policy_hash()

# Reviewed sentences become translation memory: no model call for those glosses.
calls = llm.calls
redo = export_rows(updated, default_sequences(updated), llm)
assert redo[0]["source"] == "reviewed" and redo[1]["text"] == "병원이 어디인가요?"
assert llm.calls == calls + len(redo) - 2
assert compose(["이름", "무엇"], updated, MockTranslator()).source == "reviewed"

# Rejecting a reviewed sentence removes it; unsafe approvals are refused.
redo[1].update(meaning_preserved="N", reviewer="검수자C")
removed, _, counts = apply_review(updated, [redo[1]])
assert counts["removed"] == 1 and removed.reviewed(["어디", "병원"]) is None
for row, why in ((dict(filled[0], reviewer=""), "reviewer"),
                 (dict(filled[0], model=MockTranslator.model), "mock"),
                 (dict(unknown, meaning_preserved="Y", reviewer="A"), "glossary"),
                 (dict(filled[0], meaning_preserved="maybe"), "Y 또는 N")):
    try:
        apply_review(glossary, [row])
        raise AssertionError(why)
    except ValueError as error:
        assert why in str(error), (why, error)

# Calibration reuses gloss-level verdicts only where the recognition was right.
text = "이름 무엇요."
right = {"id": "a", "reference": ["이름", "무엇"], "prediction": ["이름", "무엇"],
         "score": .9, "llm_text": text}
wrong = {**right, "id": "b", "reference": ["이름", "얼마"]}
assert review_verdict(right, review) is True
assert review_verdict(wrong, review) is None, "approving the sentence is not approving the signing"
assert review_verdict({**right, "llm_text": "다른 문장"}, review) is None
rejected = {**right, "prediction": list(C.PHRASE_RULES[2][0]),
            "reference": list(C.PHRASE_RULES[2][0]), "llm_text": filled[2]["text"]}
assert review_verdict(rejected, review) is False
per_clip = {**review, "clips": {"a": {"prediction": right["prediction"], "text": text,
                                      "meaning_preserved": False}}}
assert review_verdict(right, per_clip) is False, "a clip-level verdict wins"
negative = {"id": "n", "reference": [], "prediction": [], "score": 0, "negative": True}
assert choose_threshold([right, negative], 600, review)["enabled"]
assert not choose_threshold([right, wrong, negative], 600, review)["enabled"]
summary = summarize({"clips": [right, wrong]}, [], {}, review)
assert summary["meaning_reviewed_clips"] == 1 and summary["unreviewed_output_clips"] == 1
# Tomorrow's reviewers save from Excel: Korean Excel's plain "CSV" is CP949, notes can span lines.
import io
import os
import subprocess
import sys
import urllib.error
from unittest.mock import patch
from src.translation import VllmTranslator
from tools.review_sentences import check_language_model, generation_failures

with tempfile.TemporaryDirectory() as td:
    sheet = Path(td) / "sheet.csv"
    noted = [dict(r) for r in rows]
    noted[0].update(meaning_preserved="Y", reviewer="검수자A", note="어색함\n다음 줄, 쉼표 포함")
    write_sheet(noted, sheet)
    utf8 = read_sheet(sheet)
    excel = Path(td) / "excel.csv"
    excel.write_bytes(sheet.read_bytes().decode("utf-8-sig").encode("cp949"))
    assert read_sheet(excel) == utf8 and utf8[0]["note"] == "어색함\n다음 줄, 쉼표 포함"
    unicode_text = Path(td) / "unicode.csv"
    unicode_text.write_bytes(sheet.read_bytes().decode("utf-8-sig").encode("utf-16"))
    try:
        read_sheet(unicode_text)
        raise AssertionError("A UTF-16 sheet was read as text")
    except ValueError as error:
        assert "CSV UTF-8" in str(error)
    header = Path(td) / "header.csv"
    header.write_text("번호,문장\n1,가\n", encoding="utf-8")
    try:
        read_sheet(header)
        raise AssertionError("A sheet without the exported columns was accepted")
    except ValueError as error:
        assert "빠진 열" in str(error)


class Served:
    url, model = "http://llm.test/v1", "demo-model"

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def served_models(self):
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


check_language_model(Served(["demo-model", "other"]))
for behaviour, expected in ((urllib.error.URLError("refused"), "연결할 수 없습니다"),
                            (TimeoutError(), "연결할 수 없습니다"),
                            (urllib.error.HTTPError("u", 401, "x", {}, None), "HTTP 401"),
                            (["other"], "모델이 없습니다")):
    try:
        check_language_model(Served(behaviour))
        raise AssertionError(expected)
    except ValueError as error:
        assert expected in str(error), (expected, error)
with patch("urllib.request.urlopen",
           return_value=io.BytesIO(b'{"data":[{"id":"Qwen/Qwen2.5-7B-Instruct"},{"x":1}]}')):
    assert VllmTranslator(url="http://llm.test/v1").served_models() == ["Qwen/Qwen2.5-7B-Instruct"]
broken = [dict(rows[0], issues="generation:URLError"), dict(rows[1], issues="number")]
assert generation_failures(broken) == [rows[0]["id"]], "only generation errors block the sheet"

# End to end: with the language model off, no sheet is written and the reason is in Korean.
with tempfile.TemporaryDirectory() as td:
    draft = subprocess.run([sys.executable, "tools/review_sentences.py", "draft", "--out",
                            str(Path(td) / "g.json")], capture_output=True, text=True)
    assert draft.returncode == 0, draft.stderr
    done = subprocess.run([sys.executable, "tools/review_sentences.py", "export", "--glossary",
                           str(Path(td) / "g.json"), "--out", str(Path(td) / "sheet.csv")],
                          capture_output=True, text=True,
                          env={**os.environ, "VLLM_BASE_URL": "http://127.0.0.1:9/v1"})
    assert done.returncode != 0 and "연결할 수 없습니다" in done.stderr, done.stderr
    assert not (Path(td) / "sheet.csv").exists()
print("[OK] review sheet: export/import, translation memory, unsafe approvals refused, "
      "gloss-level verdicts reused only for correct recognition")
