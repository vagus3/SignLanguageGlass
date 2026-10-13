"""Sentence composition: reviewed sentences first, checked model output, preview-only fallback."""
import _path  # noqa: F401
import json
import tempfile
from pathlib import Path
from src.translation import Glossary, compose, mapping_hash

meanings = {"NAME": "이름", "WHAT": "무엇", "HOSPITAL": "병원", "WHERE": "어디", "NOT": "아니다",
            "GO": "가다", "THREE": "3", "KIM": "김철수", "HELLO": "안녕하세요"}
kinds = {"HOSPITAL": "place", "NOT": "negation", "THREE": "number", "KIM": "person"}
sentences = [{"glosses": ["NAME", "WHAT"], "text": "이름이 뭐예요?", "reviewed_by": "팀원A"},
             {"glosses": ["HOSPITAL", "WHERE"], "text": "병원이 어디예요?", "reviewed_by": "팀원A"}]
glossary = Glossary(meanings, kinds=kinds, sentences=sentences)


class Fixed:
    model = "fixture"

    def __init__(self, text):
        self.text, self.calls = text, 0

    def translate(self, glosses, meanings):
        self.calls += 1
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


# Plain mappings keep the hash that existing speech calibrations were made with.
assert Glossary.from_mapping({"A": "가"}).policy_hash() == mapping_hash({"A": "가"})
assert glossary.policy_hash() != mapping_hash(meanings), "Reviewed sentences are part of the policy"
edited = [dict(sentences[0], text="이름이 무엇입니까?"), sentences[1]]
assert Glossary(meanings, kinds=kinds, sentences=edited).policy_hash() != glossary.policy_hash()

assert compose(["NAME", "UNKNOWN"], glossary, Fixed("x")).status == "uncertain"
assert compose([], glossary, Fixed("x")).status == "uncertain"

model = Fixed(RuntimeError("must not be called"))
c = compose(["NAME", "WHAT"], glossary, model)
assert (c.status, c.text, c.source, model.calls) == ("ok", "이름이 뭐예요?", "reviewed", 0)

c = compose(["HOSPITAL", "GO"], glossary, Fixed("병원에 가요."))
assert (c.status, c.text, c.source) == ("ok", "병원에 가요.", "language_model")
assert compose(["GO", "NOT"], glossary, Fixed("가지 않아요.")).status == "ok"
assert compose(["HELLO", "GO"], glossary, Fixed("안녕하세요, 가요.")).status == "ok", \
    "'안녕' is not a negation and no negation gloss was signed"
for glosses, text, issue in ((["HOSPITAL", "GO"], "거기에 가요.", "place:HOSPITAL"),
                             (["GO", "NOT"], "가요.", "negation"),
                             (["THREE", "GO"], "4번 가요.", "number"),
                             (["KIM", "GO"], "그 사람이 가요.", "person:KIM")):
    c = compose(glosses, glossary, Fixed(text))
    assert c.status == "fallback" and issue in c.issues, (glosses, c)

c = compose(["NAME", "WHAT", "HOSPITAL", "WHERE"], glossary, Fixed(TimeoutError()))
assert (c.status, c.source, c.text) == ("fallback", "reviewed_rules", "이름이 뭐예요? 병원이 어디예요?")
assert c.issues == ("generation:TimeoutError",)
c = compose(["NAME", "GO"], glossary, Fixed(TimeoutError()))
assert (c.status, c.source, c.text) == ("fallback", "words", "이름 가다"), "Partial cover is not enough"
c = compose(["GO"], glossary, Fixed("  "))
assert c.status == "fallback" and c.issues == ("empty",)

with tempfile.TemporaryDirectory() as td:
    plain = Path(td) / "mapping.json"
    plain.write_text(json.dumps({"A": "가"}), encoding="utf-8")
    assert Glossary.load(plain).meanings == {"A": "가"}
    full = Path(td) / "glossary.json"
    full.write_text(json.dumps({
        "glosses": {"HOSPITAL": {"ko": "병원", "kind": "place", "aihub": ["WORD1503"]},
                    "WHERE": {"ko": "어디"}},
        "sentences": [{"glosses": ["HOSPITAL", "WHERE"], "text": "병원이 어디예요?",
                       "reviewed_by": "팀원A", "reviewed_on": "2026-10-12"}]},
        ensure_ascii=False), encoding="utf-8")
    loaded = Glossary.load(full)
    assert loaded.kinds == {"HOSPITAL": "place"} and loaded.aihub == {"HOSPITAL": ["WORD1503"]}
    assert loaded.reviewed(["HOSPITAL", "WHERE"]) == "병원이 어디예요?"
    for broken in ([1, 2], {"glosses": {"A": "가"}}):
        full.write_text(json.dumps(broken), encoding="utf-8")
        try:
            Glossary.load(full)
            raise AssertionError(f"Accepted {broken}")
        except ValueError:
            pass

one = {"glosses": ["A"], "text": "가요", "reviewed_by": "r"}
for bad in (dict(meanings={"A": ""}), dict(meanings={"A": "가"}, kinds={"A": "color"}),
            dict(meanings={"A": "가"}, sentences=[dict(one, glosses=["B"])]),
            dict(meanings={"A": "가"}, sentences=[dict(one, reviewed_by="")]),
            dict(meanings={"A": "가"}, sentences=[one, one])):
    try:
        Glossary(bad.pop("meanings"), **bad)
        raise AssertionError("Accepted an invalid glossary")
    except ValueError:
        pass
print("[OK] glossary policy hash, reviewed-first composition, auxiliary checks, fallback previews")
