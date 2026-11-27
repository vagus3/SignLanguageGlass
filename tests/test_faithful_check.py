"""로컬 CTC 문장 검사: 사람이 쓴 문장은 거의 통과하고, 뜻을 바꾼 문장은 막습니다."""
import _path  # noqa: F401
import json

from src.local_sign import LocalSignClient, plain_words


class Recognizer:
    feature_set, calibration, labels = "2d", {}, ["<blank>"]


client = LocalSignClient(Recognizer())
ROOT = _path.ROOT

# 1. 사람이 쓴 등록 문장·문장 짝 (같은 뜻 표가 망가지면 여기서 걸립니다).
flagged, total = [], 0
for name in ("assets/phrases/hospital.json", "assets/phrases/emergency.json",
             "assets/glasses/phrases.json", "assets/glasses/compose_pairs_draft.json"):
    loaded = json.loads((ROOT / name).read_text(encoding="utf-8"))
    for pair in loaded["pairs"] if isinstance(loaded, dict) else loaded:
        total += 1
        issues = client.faithful(pair["text"], plain_words(pair["glosses"]).split(),
                                 pair.get("context") or "")
        if issues:
            flagged.append((pair["glosses"], pair["text"], issues))
assert total >= 400, total
assert len(flagged) <= total * 0.02, flagged

# 2. 뜻을 바꾼 문장은 막습니다.
words = ["머리", "아프다"]
cases = {"머리가 아파요.": [], "머리가 아파서 힘들어요.": [], "머리가 너무 아파요.": [],
         "머리가 안 아파요.": ["negation_added"], "머리가 아파요?": ["question"],
         "머리가 아파요. 열이 나요.": ["added:열"], "머리예요.": ["dropped:아프다"],
         "머리가 3번 아파요.": ["number"]}
for sentence, want in cases.items():
    assert client.faithful(sentence, words) == want, (sentence, client.faithful(sentence, words))
assert client.faithful("약국이 어디예요?", ["약", "곳"]) == []
assert client.faithful("위험해요. 위가 아파요.", ["위", "아프다"]) == ["added:위험"]
# 상대가 말한 명사는 써도 됩니다.
assert client.faithful("약이 없어요.", ["없다"], "약 드셨어요?") == []
assert client.faithful("약이 없어요.", ["없다"]) == ["added:약"]
print(f"[OK] 로컬 문장 검사: 사람이 쓴 문장 {total}개 중 {len(flagged)}개만 걸림, 뜻 변화는 막음")
