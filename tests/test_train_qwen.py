"""Pose -> Qwen fine-tuning plumbing with a tiny randomly initialised Qwen2 (no download).

Needs transformers + peft and the Qwen tokenizer in the local Hugging Face cache; skips
(exit 77) otherwise.
"""
import _path  # noqa
import sys
import numpy as np
import torch
try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
except Exception as error:      # noqa: BLE001
    print(f"Qwen 토크나이저/peft 없음: {type(error).__name__}")
    sys.exit(77)
from src.pose_sequence import PoseSequence
from src.train_qwen import PoseQwen, batch_inputs, candidate_texts, evaluate, generate, rank, train

torch.manual_seed(0)
torch.set_num_threads(1)
config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=256, tie_word_embeddings=True)
llm = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(r=4, lora_alpha=8,
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
model = PoseQwen(llm, ["가", "나"], hidden=32)
model.phrase_texts = {"가요."}


def item(word, rng):
    """Two signs that differ in where the left hand moves; '' = no sign."""
    t = np.arange(30) / 30
    p = np.full((30, 49, 2), 0.5, np.float32) + rng.normal(0, 0.002, (30, 49, 2)).astype(np.float32)
    mask = np.zeros((30, 49), bool)
    if word:
        mask[:, :21] = True
        p[:, :21] += np.outer(np.sin(t * np.pi), {"가": (0.2, 0), "나": (0, 0.2)}[word])[:, None]
    p[~mask] = 0
    return {"glosses": [word] if word else [], "text": {"가": "가요.", "나": "나.", "": "(수어 아님)"}[word],
            "sequence": PoseSequence(t, p, mask, (1280, 720))}


class Toy:
    def sample(self, rng, signer=None):
        return item(["가", "나", ""][int(rng.integers(3))], rng)


rng = np.random.default_rng(0)
items = [item("가", rng), item("나", rng), item("", rng)]
embeds, mask, labels, ctc, lengths = batch_inputs(model, tokenizer, items, "cpu")
assert embeds.shape[:2] == mask.shape == labels.shape and ctc.shape[1] == 3
# Only the answer is learned: prompt and pose positions are ignored by the loss.
target = tokenizer(" 가요." + tokenizer.eos_token).input_ids
assert labels[0][labels[0] != -100].tolist() == target
# Pose tokens: 30 frames pooled by 4 -> 8 positions between the prompt and the answer.
assert int(mask[0].sum()) - len(target) == len(tokenizer("수어를 한국어 문장으로 옮기세요.\n수어:").input_ids) \
    + 8 + len(tokenizer("\n문장:").input_ids)
# Generation pads on the left so every row continues from its last real position.
_, gen_mask, _, _, _ = batch_inputs(model, tokenizer, items, "cpu", with_target=False)
assert gen_mask[:, -1].all() and not gen_mask[:, 0].all() or gen_mask.all()

frozen = {n: p.clone() for n, p in model.llm.named_parameters() if not p.requires_grad}
lines = []
history, best = train(model, tokenizer, Toy(), steps=100, batch=6, device="cpu",
                      val_items=items, eval_every=100, log=lines.append,
                      lr_lora=3e-3, lr_pose=3e-3)
losses = [float(l.split("loss=")[1].split()[0]) for l in lines if l.startswith("step=")]
assert losses[-1] < losses[0] - 0.2, losses   # a random tiny base learns slowly; the trend is what matters
assert history and history[-1]["step"] == 100 and best is not None
assert all(torch.equal(p, frozen[n]) for n, p in model.llm.named_parameters()
           if not p.requires_grad), "the base model stays frozen; only LoRA learns"
outputs = generate(model, tokenizer, items, "cpu")
assert all(isinstance(text, str) and 0 <= score <= 1 for text, score, _ in outputs)
report, rows = evaluate(model, tokenizer, items, "cpu")
assert set(report) >= {"exact", "phrase_exact", "no_sign_correct", "ctc_exact"} and len(rows) == 3
# Choosing among listed answers: the cached ranking equals scoring each full sequence.
options = candidate_texts(model, [{"glosses": ["가"], "text": "가요."}], ["가", "나"])
assert options == {"가요.": ["가"], "나.": ["나"], "(수어 아님)": []}
posterior, words = rank(model, tokenizer, items[0]["sequence"], options, "cpu", shortlist=0)
texts = list(options)
with torch.no_grad():
    e, m, l, _, _ = batch_inputs(model, tokenizer, [{**items[0], "text": t} for t in texts], "cpu")
    logits = model.llm(inputs_embeds=e, attention_mask=m).logits[:, :-1]
    target = l[:, 1:]
    ok = target != -100
    total = (logits.log_softmax(-1).gather(-1, target.clamp(min=0)[..., None])[..., 0] * ok).sum(1)
assert np.allclose([posterior[t] for t in texts], total.softmax(0).numpy(), atol=1e-5)
assert abs(sum(posterior.values()) - 1) < 1e-5 and isinstance(words, list)
short, _ = rank(model, tokenizer, items[0]["sequence"], options, "cpu", shortlist=2)
assert len(short) == 2, "only the word head's best candidates reach Qwen"
print("[OK] Qwen fine-tuning: pose tokens inside the prompt, answer-only loss, frozen base, "
      "loss falls on a toy task")
