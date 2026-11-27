"""On-device recognition and sentence choice without a server."""
import _path  # noqa
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.cloud_sign import CloudWorker, validate_response
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
client = LocalSignClient(Recognizer(unsure), phrases)


def run(client, seq=None):
    payload = client.payload(seq or sequence(), "session", "segment")
    return validate_response(client.translate(payload), payload)


# Seen as-is and in the list: the reviewed sentence, not marked as a guess.
result = run(client)
assert (result["status"], result["text"], result["source"]) == ("ok", "목이 아파요.", "phrase")
# The result retains the CTC words separately from the spoken sentence.
assert result["auto_speech"] is True and result["speech"] == "목이 아파요."
assert result["recognized_words"] == "목 아프다"
# A nearby phrase suggested by the conversation must not replace the visual result.
client.hear("어디 아프세요?", at=100.0)
assert client.heard(now=110.0) and not client.heard(now=121.0), "context expires"
client.hear("어디 아프세요?")
result = run(client)
assert result["text"] == "목이 아파요." and result["glosses"] == ["목", "아프다"]
assert result["speech"] == "목이 아파요." and result["auto_speech"]
# Not in the list: the recognized words, without variant numbers.
clear = [[.01, .01, .01, .01, .96], [.9, .03, .03, .04, .001], [.01, .01, .01, .96, .01],
         [.9, .03, .03, .04, .001], [.9, .03, .03, .04, .001]]
result = run(LocalSignClient(Recognizer(clear), phrases))
assert (result["text"], result["source"]) == ("검사 아프다", "words")
assert result["auto_speech"] and result["speech"] == "검사 아프다", "clear signs are spoken"
# No hands, or nothing recognized: no text at all.
result = run(client, sequence(hands=False))
assert result["status"] == "no_sign" and not result["auto_speech"]
silent = LocalSignClient(Recognizer([[.97, .01, .01, .01, .001]] * 5), phrases)
assert run(silent) ["status"] == "no_sign"
# Low-confidence detections are still spoken: the matching phrase, or the words without a composer.
weak = Recognizer(unsure)
weak.calibration = {"threshold": 0.9}
result = run(LocalSignClient(weak, phrases))
assert result["source"] == "phrase" and result["speech"] == "목이 아파요." and result["auto_speech"]
assert result["low_score"]
result = run(LocalSignClient(weak))
assert result["source"] == "words" and result["speech"] == "목 아프다" and result["auto_speech"]

print("[OK] Local sign client: phrase list, context from speech, plain display, "
      "words fallback, no-sign and threshold")

# --save-segments: each recognized segment and its result are kept for diagnosis.
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

for answer in (TimeoutError(), "검사 123 아파요.", "검사가 아프지 않아요.",
               "검사가 아파요. 치료가 필요해요.", "검사예요.", "검사가 아파요?"):
    composer.answer = answer
    failed = validate_response(composed.translate(payload), payload)
    assert failed["source"] == "words" and failed["issues"], (answer, failed)
    assert failed["text"] == failed["speech"] == "검사 아프다" and failed["auto_speech"]

# Natural sentences with connectives, helper verbs or nouns from the other person's question
# are spoken; meaning changes above (number, negation, question, new nouns, dropped words) are not.
for answer in ("검사하고 나서 아파요.", "검사가 아파서 힘들어요."):
    composer.answer = answer
    spoken = validate_response(composed.translate(payload), payload)
    assert spoken["source"] == "language_model" and spoken["speech"] == answer, (answer, spoken)
# A noun is allowed only when the other person said it.
composer.answer = "머리 검사가 아파요."
assert validate_response(composed.translate(payload), payload)["issues"] == ["added:머리"]
composed.hear("머리 검사 받으셨어요?")
asked = composed.payload(sequence(), "s", "d")
spoken = validate_response(composed.translate(asked), asked)
assert spoken["source"] == "language_model" and spoken["speech"] == "머리 검사가 아파요."

travel = Recognizer(clear)
travel.labels = ["<blank>", "머리", "목", "오다", "병원"]
travel_composer = Composer()
travel_client = LocalSignClient(travel, composer=travel_composer)
for answer in ("병원에 와요.", "병원에 오세요."):
    travel_composer.answer = answer
    result = run(travel_client)
    assert result["speech"] == answer and result["source"] == "language_model"
for answer in ("병원에 오세요?", "병원에 안 와요.", "응급실에 와요."):
    travel_composer.answer = answer
    result = run(travel_client)
    assert result["speech"] == result["recognized_words"] == "병원 오다"
    assert result["source"] == "words" and result["auto_speech"] and result["issues"], result

unused = Composer(RuntimeError("reviewed phrases should bypass generation"))
assert run(LocalSignClient(Recognizer(unsure), phrases, composer=unused))["source"] == "phrase"
assert unused.calls == []
# Uncertain recognition also becomes a sentence instead of bare words.
low = Composer("목이 아파요.")
result = run(LocalSignClient(weak, composer=low))
assert result["source"] == "language_model" and result["speech"] == "목이 아파요."
assert result["low_score"] and result["auto_speech"] and low.calls
print("[OK] CTC composer: context snapshot, faithful sentence, spoken fallback")


class SlowComposer(Composer):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def translate(self, glosses, meanings, context=None):
        self.entered.set()
        self.release.wait(5)
        return super().translate(glosses, meanings, context)


slow = SlowComposer()
worker = CloudWorker(LocalSignClient(Recognizer(clear), composer=slow))
worker.start()
try:
    assert worker.submit(sequence(), 2.0)
    assert slow.entered.wait(2)
    with patch("src.cloud_sign.clock.now", return_value=worker.deadline + 1):
        result, origin = worker.poll()
    assert result["auto_speech"] and result["speech"] == "검사 아프다" and origin == 2.0
    assert not worker.busy
    slow.release.set()
    late = worker.results.get(timeout=3)
    worker.results.put(late)
    assert worker.poll() is None, "A late sentence must not be spoken a second time"
finally:
    slow.release.set()
    worker.stop()
    worker.join(timeout=3)
print("[OK] CTC worker speaks recognized words at the deadline and drops late composition")
