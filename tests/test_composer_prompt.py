"""문장 모델 입력 형식: 미세조정 모델은 학습 때 형식, 기본 모델은 통역 형식. 학습과 실행이 같은 입력."""
import _path  # noqa: F401
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src import train_composer
from src.translation import PLAIN_SYSTEM, LocalTranslator

mapping = {"아프다": "아프다", "병원": "병원", "도와주다": "도와주다"}
examples = [{"glosses": ["아프다"], "text": "아파요."},
            {"glosses": ["병원"], "text": "병원이요."}]


class Tokenizer:
    """채팅 형식을 글자로 이어 붙이기만 하는 가짜 토크나이저."""
    def apply_chat_template(self, messages, **kwargs):
        return [f"{m['role']}:{m['content']}" for m in messages]

    def __call__(self, text):
        return type("Ids", (), {"input_ids": [text]})()


def fake(style):
    """모델을 불러오지 않고 LocalTranslator 의 형식 선택만 재현합니다."""
    t = object.__new__(LocalTranslator)
    t.examples, t.prompt_style, t.tokenizer = list(examples), style, Tokenizer()
    return t


# 1. 미세조정 모델 형식: 시스템 지시 없음, '문장:' 예시, '상대가 방금 한 말:'
tuned = fake("tuned")
messages = tuned.messages(["아프다", "병원"], mapping, "어디가 불편하세요?")
assert [m["role"] for m in messages] == ["user"], messages
text = messages[0]["content"]
assert text.startswith("착용자가 한국수어로 말한 단어열을"), text
assert "문장: 아파요." in text and "상대가 방금 한 말: 어디가 불편하세요?" in text, text
assert text.endswith("단어: 아프다 병원\n문장:"), text
assert "통역:" not in text

# 2. 기본 모델 형식: 시스템 지시 + '통역:' 예시 + '대화 참고:'
plain = fake("plain")
messages = plain.messages(["아프다", "병원"], mapping, "어디가 불편하세요?")
assert messages[0] == {"role": "system", "content": PLAIN_SYSTEM}, messages
assert "대화 참고: 어디가 불편하세요?" in messages[1]["content"]
assert messages[1]["content"].endswith("단어: 아프다 병원\n통역:")

# 3. 폴더에 composer.json 이 있으면 tuned, 기본 모델 이름이면 plain 을 고릅니다.
with tempfile.TemporaryDirectory() as td:
    Path(td, "composer.json").write_text(json.dumps({"base": "base-model"}), encoding="utf-8")

    class Loaded:
        def to(self, *_):
            return self

        def eval(self):
            return self

    class Peft:
        @staticmethod
        def from_pretrained(llm, adapter):
            return type("M", (), {"merge_and_unload": lambda self: Loaded()})()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=Loaded()), \
            patch("peft.PeftModel", Peft):
        assert LocalTranslator(td, device="cpu").prompt_style == "tuned"
        assert LocalTranslator("base-model", device="cpu").prompt_style == "plain"
        assert LocalTranslator("base-model", device="cpu", prompt_style="tuned").prompt_style == "tuned"
        try:
            LocalTranslator("base-model", device="cpu", prompt_style="other")
            raise AssertionError("잘못된 형식은 거부해야 합니다")
        except ValueError:
            pass

# 4. 학습 입력은 실행 때와 같은 messages() 로 만듭니다 (형식이 어긋나지 않게).
prompt, answer = train_composer.encode(tuned, {"glosses": ["아프다"], "text": "아파요.",
                                               "context": None}, mapping)
assert prompt == tuned.tokenizer.apply_chat_template(tuned.messages(["아프다"], mapping, None))
assert answer == ["아파요.<|im_end|>"]
print("[OK] 문장 모델 입력 형식: 미세조정 모델은 학습 형식, 기본 모델은 통역 형식, 학습과 실행 일치")
