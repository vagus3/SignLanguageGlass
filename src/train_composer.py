"""문장 모델(Qwen2.5-3B-Instruct)을 '단어열(+상대의 말) -> 문장' 짝으로 LoRA 미세조정합니다.

python -m src.train_composer --out models/composer_3b_v1

- 자료: assets/glasses/compose_pairs_draft.json (팀 검수 전 초안) + 상황 문장 목록.
- 입력 형식은 실행할 때(src.translation.LocalTranslator.prompt)와 똑같이 만듭니다. 예시 문장은
  자기 자신을 뺀 짝에서 고릅니다. 정답 문장 부분만 손실에 넣습니다.
- Qwen 본체는 bf16 으로 얼려 두고 LoRA 만 fp32 로 학습합니다 (16GB 맥에서 돌도록).
- 짝의 10%는 학습에서 빼 두고, 미세조정 전후로 같은 짝을 만들게 해서 비교합니다
  (정답과 똑같은 비율, 안전 검사 통과 비율).

결과 폴더: adapter/ (LoRA), composer.json (기반 모델, 학습 설정, 비교 결과)
"""
import argparse
import json
import random
from pathlib import Path

import torch

from src import clock

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen2.5-3B-Instruct"


def load_pairs():
    draft = json.loads((ROOT / "assets/glasses/compose_pairs_draft.json").read_text(encoding="utf-8"))
    phrases = json.loads((ROOT / "assets/glasses/phrases.json").read_text(encoding="utf-8"))
    pairs, seen = [], set()
    for entry in draft["pairs"] + [{"glosses": p["glosses"], "text": p["text"]} for p in phrases]:
        key = (tuple(entry["glosses"]), entry.get("context"))
        if key not in seen:
            seen.add(key)
            pairs.append(entry)
    return pairs, phrases


def encode(translator, pair, mapping, with_answer=True):
    """(입력 글자 번호, 정답 글자 번호) — LocalTranslator 가 실행 때 만드는 것과 같은 입력."""
    tok = translator.tokenizer
    messages = [{"role": "user", "content": translator.prompt(pair["glosses"], mapping,
                                                               pair.get("context"))}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    answer = tok(pair["text"] + "<|im_end|>").input_ids if with_answer else []
    return prompt, answer


def judge(translator, pairs, glossary):
    """같은 짝을 만들게 해서 (정답과 같음, 안전 검사 통과) 비율과 예시를 돌려줍니다."""
    from src.translation import compose
    same = passed = 0
    rows = []
    for pair in pairs:
        start = clock.now()
        try:
            text = translator.translate(pair["glosses"], glossary.meanings, pair.get("context"))
            issues = glossary.check(text, pair["glosses"])
        except ValueError as error:
            text, issues = "", [str(error)]
        same += text == pair["text"]
        passed += not issues
        rows.append({"glosses": pair["glosses"], "context": pair.get("context"), "want": pair["text"],
                     "got": text, "issues": issues, "seconds": round(clock.now() - start, 2)})
    n = max(len(pairs), 1)
    return {"exact": same / n, "passed": passed / n, "count": len(pairs)}, rows


def main():
    from peft import LoraConfig, get_peft_model
    from src.qwen_sign import glasses_glossary
    from src.translation import LocalTranslator
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default=BASE)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        p.error("결과 폴더가 이미 있습니다")
    pairs, phrases = load_pairs()
    random.Random(args.seed).shuffle(pairs)
    held = pairs[:max(1, int(len(pairs) * args.holdout))]
    train_pairs = pairs[len(held):]
    vocabulary = json.loads((ROOT / "assets/glasses/vocabulary.json").read_text(encoding="utf-8"))
    # 평가는 덧붙인 단어까지 세는 엄격한 검사로 (앱은 덧붙임을 허용하지만 비교 기준은 유지).
    glossary = glasses_glossary(vocabulary, phrases, strict_additions=True)
    # 예시 문장은 학습 짝에서만 고릅니다 (확인용 짝의 답을 보여 주지 않음).
    translator = LocalTranslator(args.base, examples=train_pairs)
    translator.llm.to(torch.bfloat16)
    before, before_rows = judge(translator, held, glossary)
    print(f"미세조정 전 (확인용 {len(held)}개): 정답과 같음 {before['exact']:.0%}, "
          f"안전 검사 통과 {before['passed']:.0%}", flush=True)

    llm = get_peft_model(translator.llm, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for name, param in llm.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    llm.gradient_checkpointing_enable()
    llm.enable_input_require_grads()
    trainable = [q for q in llm.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * -(-len(train_pairs) // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps,
                                                   pct_start=0.1)
    pad = translator.tokenizer.pad_token_id
    started, step = clock.now(), 0
    for epoch in range(args.epochs):
        order = train_pairs[:]
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order), args.batch):
            llm.train()
            optimizer.zero_grad()
            chunk = order[start:start + args.batch]
            total = 0.0
            for pair in chunk:
                # 자기 자신이 예시로 나오면 답을 베끼므로, 이 짝을 뺀 예시로 입력을 만듭니다.
                translator.examples = [e for e in train_pairs if e is not pair]
                prompt, answer = encode(translator, pair, glossary.meanings)
                ids = torch.tensor([prompt + answer], device=translator.device)
                # 16GB 맥: 한 개씩, 정답 위치의 글자 확률만 계산합니다 (전체 어휘 15만 개 x 길이를
                # 한꺼번에 만들면 메모리를 넘습니다).
                logits = llm(input_ids=ids, logits_to_keep=len(answer) + 1).logits[0, :-1]
                target = torch.tensor(answer, device=translator.device)
                loss = torch.nn.functional.cross_entropy(logits.float(), target) / len(chunk)
                loss.backward()
                total += float(loss)
            loss = torch.tensor(total)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            if step % 10 == 0 or step == steps:
                print(f"step={step}/{steps} loss={float(loss):.3f} "
                      f"{(clock.now() - started) / step:.1f}s/step", flush=True)
    translator.examples = train_pairs
    llm.eval()
    translator.llm = llm
    after, after_rows = judge(translator, held, glossary)
    print(f"미세조정 후 (확인용 {len(held)}개): 정답과 같음 {after['exact']:.0%}, "
          f"안전 검사 통과 {after['passed']:.0%}", flush=True)
    out.mkdir(parents=True)
    llm.save_pretrained(out / "adapter")
    (out / "composer.json").write_text(json.dumps(
        {"base": args.base, "pairs": "assets/glasses/compose_pairs_draft.json (팀 검수 전 초안)",
         "train": len(train_pairs), "held_out": len(held), "epochs": args.epochs,
         "lr": args.lr, "rank": args.rank, "seed": args.seed,
         "before": before, "after": after,
         "held_out_rows": [{"before": b, "after": a} for b, a in zip(before_rows, after_rows)]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
