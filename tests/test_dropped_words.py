"""A generated sentence that leaves out an input word is rejected (stems learned from pairs)."""
import _path  # noqa
import json
from pathlib import Path

from src.qwen_sign import glasses_glossary
from src.translation import Glossary, compose, learn_stems

ROOT = Path(__file__).resolve().parents[1]


class Fixed:
    def __init__(self, text):
        self.text = text

    def translate(self, glosses, mapping, context=None):
        return self.text


# 1. learn_stems: longest prefix present in >= floor of the pair texts; conjugation-safe.
meanings = {"아프다": "아프다", "병원": "병원", "부르다": "부르다", "의사": "의사"}
pairs = [{"glosses": ["아프다", "병원"], "text": "아파요. 병원에 가야 해요."},
         {"glosses": ["아프다", "의사"], "text": "아파서 의사가 필요해요."},
         {"glosses": ["아프다", "병원", "의사"], "text": "아파요. 병원 의사를 만나고 싶어요."},
         {"glosses": ["부르다", "의사"], "text": "의사를 불러 주세요."},
         {"glosses": ["부르다", "병원"], "text": "병원을 불러 주세요."},
         {"glosses": ["부르다", "병원", "의사"], "text": "병원 의사를 불러 주세요."}]
stems = learn_stems(meanings, pairs)
assert stems["병원"] == "병원" and stems["아프다"] == "아", stems
assert "부르다" not in stems, "an irregular form with no common prefix gets no check"

# 2. The check itself, with a plain glossary.
g = Glossary(meanings, stems=stems)
assert g.check("아파요. 의사가 필요해요.", ["아프다", "병원", "의사"]) == ["dropped:병원"]
assert g.check("아파요. 병원에 가서 의사를 만나고 싶어요.", ["아프다", "병원", "의사"]) == []
assert Glossary(meanings).check("아파요.", ["아프다", "병원"]) == [], "off unless stems are given"

# 3. In compose(): a dropped word falls back to a silent preview, not speech.
made = compose(["아프다", "병원", "의사"], g, Fixed("아파요. 의사가 필요해요."))
assert made.status == "fallback" and "dropped:병원" in made.issues, made

# 4. The app glossary learns stems from the reviewed pairs and keeps every pair passing.
vocab = json.loads((ROOT / "assets/glasses/vocabulary.json").read_text(encoding="utf-8"))
phrases = json.loads((ROOT / "assets/glasses/phrases.json").read_text(encoding="utf-8"))
app = glasses_glossary(vocab, phrases)
assert app.stems["병원"] == "병원" and len(app.stems) > 20
draft = json.loads((ROOT / "assets/glasses/compose_pairs_draft.json").read_text(encoding="utf-8"))
bad = [p for p in draft["pairs"] if any(i.startswith("dropped") for i in app.check(p["text"], p["glosses"]))]
assert not bad, f"reviewed pairs must pass their own check: {bad[:3]}"
assert glasses_glossary(vocab, phrases, check_dropped=False).stems == {}
print("[OK] dropped words: learned stems, rejected in compose, reviewed pairs all pass")
