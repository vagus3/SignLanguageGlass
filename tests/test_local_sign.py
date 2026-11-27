"""On-device recognition and sentence choice without a server."""
import _path  # noqa
import json
import tempfile
from pathlib import Path
import numpy as np
from src.cloud_sign import validate_response
from src.local_sign import LocalSignClient, load_phrases, plain_words
from src.pose_sequence import PoseSequence

labels = ["<blank>", "머리", "목", "아프다", "검사_0029"]


class Recognizer:
    feature_set, calibration = "2d", {}

    def __init__(self, rows):
        p = np.array(rows, float)
        self.output, self.labels = np.log(p / p.sum(-1, keepdims=True)), labels

    def log_probs(self, features):
        return self.output


def sequence(hands=True):
    mask = np.ones((5, 49), bool)
    mask[:, :42] = hands
    return PoseSequence(np.arange(5) / 30, np.full((5, 49, 2), .5), mask, (640, 480))


with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "phrases.json"
    path.write_text(json.dumps([
        {"glosses": ["머리", "아프다"], "text": "머리가 아파요.", "after": ["어디"]},
        {"glosses": ["목", "아프다"], "text": "목이 아파요."},
        {"glosses": ["없는말"], "text": "모르는 단어"}], ensure_ascii=False), encoding="utf-8")
    phrases, dropped = load_phrases(path, labels)
    assert len(phrases) == 2 and dropped == 1
    path.write_text('[{"glosses": [], "text": "x"}]', encoding="utf-8")
    try:
        load_phrases(path, labels)
        raise AssertionError("An empty phrase was accepted")
    except ValueError:
        pass
assert plain_words(["검사_0029", "아프다"]) == "검사 아프다"

unsure = [[.05, .40, .50, .05, .001], [.9, .03, .03, .04, .001], [.05, .02, .03, .90, .001],
          [.05, .02, .03, .90, .001], [.9, .03, .03, .04, .001]]
client = LocalSignClient(Recognizer(unsure), phrases, bonus=0.2)


def run(client, seq=None):
    payload = client.payload(seq or sequence(), "session", "segment")
    return validate_response(client.translate(payload), payload)


# Seen as-is and in the list: the reviewed sentence, not marked as a guess.
result = run(client)
assert (result["status"], result["text"], result["source"]) == ("ok", "목이 아파요.", "phrase")
# The default speaks even when the score is below 0.5; an explicit threshold still applies.
assert result["auto_speech"] is True and result["speech"] == "목이 아파요."
assert not run(LocalSignClient(Recognizer(unsure), phrases, bonus=0.2, speak_min=0.5))["auto_speech"]
assert run(LocalSignClient(Recognizer(unsure), phrases, bonus=0.2, speak_min=0.4))["auto_speech"]
# What the other person just said tips a near-tie; the result is shown without a prefix.
client.hear("어디 아프세요?", at=100.0)
assert client.heard(now=110.0) and not client.heard(now=121.0), "context expires"
client.hear("어디 아프세요?")
result = run(client)
assert result["text"] == "머리가 아파요." and result["glosses"] == ["머리", "아프다"]
assert result["speech"] == "머리가 아파요." and result["auto_speech"]
# Not in the list: the recognized words, without variant numbers.
clear = [[.01, .01, .01, .01, .96], [.9, .03, .03, .04, .001], [.01, .01, .01, .96, .01],
         [.9, .03, .03, .04, .001], [.9, .03, .03, .04, .001]]
result = run(LocalSignClient(Recognizer(clear), phrases))
assert (result["text"], result["source"]) == ("검사 아프다", "words")
assert result["auto_speech"] and result["speech"] == "검사 아프다", "clear signs are spoken"
assert not run(LocalSignClient(Recognizer(clear), phrases, speak_min=1.01))["auto_speech"], "off"
# No hands, or nothing recognized: no text at all.
assert run(client, sequence(hands=False))["status"] == "no_sign"
silent = LocalSignClient(Recognizer([[.97, .01, .01, .01, .001]] * 5), phrases)
assert run(silent) ["status"] == "no_sign"
# A calibrated threshold rejects weak as-seen results.
weak = Recognizer(unsure)
weak.calibration = {"threshold": 0.9}
result = run(LocalSignClient(weak, phrases, bonus=0))
assert result["status"] == "uncertain" and result["text"] == "" and not result["auto_speech"]

print("[OK] Local sign client: phrase list, context from speech, plain display, "
      "words fallback, no-sign and threshold")

# --save-segments: each recognized segment and its result are kept for diagnosis.
import tempfile
with tempfile.TemporaryDirectory() as td:
    keeper = LocalSignClient(Recognizer(clear), save_dir=td)
    keeper.payload(sequence(), "s", 3)
    from pathlib import Path as _P
    saved = sorted(p.name for p in _P(td).iterdir())
    assert saved == ["s_0003.json", "s_0003.npz"], saved
print("[OK] segments saved for diagnosis")


# CTC results outside the phrase list reach the existing contextual composer.
class Composer:
    accepts_context = True

    def __init__(self, answer="검사가 아파요."):
        self.answer, self.calls = answer, []

    def translate(self, glosses, meanings, context=None):
        self.calls.append((glosses, meanings, context))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


composer = Composer()
composed = LocalSignClient(Recognizer(clear), phrases, composer=composer)
composed.hear("검사는 어떠세요?")
payload = composed.payload(sequence(), "s", "c")
composed.hear("나중에 한 말")
result = validate_response(composed.translate(payload), payload)
assert result["status"] == "ok" and result["source"] == "language_model"
assert result["speech"] == "검사가 아파요." and result["auto_speech"]
assert composer.calls[0][0] == ["검사_0029", "아프다"]
assert composer.calls[0][1]["검사_0029"] == "검사"
assert composer.calls[0][2] == "검사는 어떠세요?", "context belongs to the segment"
assert result["timings_ms"]["language"] >= 0

for answer in (TimeoutError(), "검사 123 아파요."):
    failed = run(LocalSignClient(Recognizer(clear), phrases, composer=Composer(answer)))
    assert failed["status"] == "fallback" and failed["source"] == "words"
    assert failed["text"] == "검사 아프다" and not failed["auto_speech"]

unused = Composer(RuntimeError("reviewed phrases should bypass generation"))
assert run(LocalSignClient(Recognizer(unsure), phrases, composer=unused))["source"] == "phrase"
assert unused.calls == []
assert not run(LocalSignClient(Recognizer(clear), composer=Composer(), speak_min=1.01))["auto_speech"]
print("[OK] CTC composer: context snapshot, meanings, speech threshold, checked fallback")
