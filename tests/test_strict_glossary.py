"""Glasses vocabulary: a generated sentence that adds another listed word is rejected."""
import _path  # noqa
from src.translation import Glossary, compose


class Fixed:
    accepts_context = True

    def __init__(self, text):
        self.text, self.seen = text, None

    def translate(self, glosses, mapping, context=None):
        self.seen = context
        return self.text


meanings = {w: w for w in ("엄마", "119", "부르다", "쓰러지다", "도와주다", "약", "못하다", "걷다")}
strict = Glossary(meanings, kinds={"엄마": "person", "못하다": "negation"},
                  sentences=[{"glosses": ["엄마", "부르다"], "text": "엄마를 불러 주세요.",
                              "reviewed_by": "test"}], strict_additions=True)
# An added event ('쓰러졌어요') is caught; the plain sentence passes.
made = compose(["엄마", "119", "부르다"], strict, Fixed("엄마가 쓰러졌어요. 119를 불러 주세요."))
assert made.status == "fallback" and "added:쓰러지다" in made.issues, made
translator = Fixed("엄마와 119를 불러 주세요.")
made = compose(["엄마", "119", "부르다"], strict, translator, context="무슨 일이세요?")
assert made.status == "ok" and translator.seen == "무슨 일이세요?", "context reaches the model"
# A listed sentence is used as is, without the model.
assert compose(["엄마", "부르다"], strict, Fixed("x")).source == "reviewed"
# Negation must survive; a dropped person is caught as before.
assert compose(["걷다", "못하다"], strict, Fixed("걸어요.")).status == "fallback"
assert compose(["엄마", "119", "부르다"], strict, Fixed("119를 불러 주세요.")).status == "fallback"
# Without strict_additions the old behaviour is unchanged.
loose = Glossary(meanings)
assert compose(["119", "부르다"], loose, Fixed("쓰러졌어요. 119를 불러 주세요.")).status == "ok"
print("[OK] strict glossary: added words rejected, context passed, reviewed and negation kept")
