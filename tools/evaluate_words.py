#!/usr/bin/env python
"""단어 모델을 학습에 안 쓴 수어자로 평가하고, 상황 문장 보정의 득실을 잽니다.

python tools/evaluate_words.py models/words_v1/ctc.pt data/processed/aihub_words/*/manifest.json \
    --signers REAL15,REAL16 --phrases data/processed/hospital_phrases.json

세 가지를 봅니다 (보정 세기 --bonus 값마다):
- 상황 문장: 목록에 있는 문장을 수어했을 때 문장 전체를 맞힌 비율 (보정의 이득)
- 그 밖의 조합: 목록에 없는 말을 했을 때 맞힌 비율과, 맞던 것을 목록 문장으로
  바꿔 버린 비율 (보정의 피해)
- 수어 아님: 준비·복귀 동작에서 무언가 출력한 비율

--signers 에는 그 모델의 학습에 쓰지 않은 수어자만 넣으세요. 아니면 숫자가 부풀려집니다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuous_sign import Recognizer
from src.ctc_runtime import decode_with_phrases
from src.pose_sequence import FEATURE_SETS
from src.train_words import Composer, join, load_words, rest_rows, sign_piece


def log_probs(recognizer, sequence):
    x = torch.from_numpy(FEATURE_SETS[recognizer.feature_set][0](sequence))[None]
    with torch.inference_mode():
        return recognizer.model(x, torch.tensor([x.shape[1]]))[:, 0].numpy()


def phrase_examples(rows, phrases):
    """수어자·카메라 방향마다 상황 문장을 한 번씩 이어 붙입니다 (변형 없음)."""
    examples = []
    composer = Composer(rows)
    for key in composer.keys:
        by_label = {}
        for row in composer.groups[key]:
            by_label.setdefault(row["label"], row)
        for phrase in phrases:
            if all(g in by_label for g in phrase["glosses"]):
                pieces = [sign_piece(by_label[g], None) for g in phrase["glosses"]]
                examples.append({"glosses": list(phrase["glosses"]), "sequence": join(
                    pieces, by_label[phrase["glosses"][0]]["sequence"].image_size)})
    return examples


def other_examples(rows, phrases, count, seed=0):
    """목록에 없는 1~3단어 조합 (변형 없음)."""
    known = {tuple(p["glosses"]) for p in phrases}
    rng = np.random.default_rng(seed)
    composer = Composer(rows)
    examples = []
    while len(examples) < count:
        group = composer.groups[composer.keys[int(rng.integers(len(composer.keys)))]]
        chosen = [group[int(i)] for i in rng.choice(len(group), int(rng.integers(1, 4)),
                                                    replace=False)]
        glosses = [r["label"] for r in chosen]
        if tuple(glosses) not in known:
            examples.append({"glosses": glosses, "sequence": join(
                [sign_piece(r, None) for r in chosen], chosen[0]["sequence"].image_size)})
    return examples


def evaluate(recognizer, rows, phrases, bonuses, others=400):
    sets = {"phrases": phrase_examples(rows, phrases),
            "others": other_examples(rows, phrases, others), "rests": rest_rows(rows)}
    outputs = {name: [(e["glosses"], log_probs(recognizer, e["sequence"])) for e in examples]
               for name, examples in sets.items()}
    report = {"counts": {name: len(examples) for name, examples in sets.items()}, "bonus": {}}
    for bonus in [None, *bonuses]:
        entry = {}
        for name, items in outputs.items():
            exact, replaced_correct = [], 0
            for reference, output in items:
                plain = decode_with_phrases(output, recognizer.labels, [])[0]
                glosses, _, source = (plain, 0, "greedy") if bonus is None else \
                    decode_with_phrases(output, recognizer.labels, phrases, bonus=bonus)
                exact.append(glosses == reference)
                replaced_correct += plain == reference and glosses != reference
            if name == "rests":
                entry["false_output_rate"] = 1 - float(np.mean(exact)) if exact else None
            else:
                entry[f"{name}_exact"] = float(np.mean(exact)) if exact else None
                if name == "others":
                    entry["others_broken"] = replaced_correct / len(items) if items else None
        report["bonus"]["off" if bonus is None else str(bonus)] = entry
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("manifests", nargs="+")
    p.add_argument("--signers", required=True, help="학습에 쓰지 않은 수어자 (쉼표로 구분)")
    p.add_argument("--phrases", required=True)
    p.add_argument("--bonus", default="0,2,4,8", help="비교할 보정 세기들")
    p.add_argument("--json", help="결과를 저장할 파일")
    args = p.parse_args()
    recognizer = Recognizer(args.checkpoint)
    signers = set(args.signers.split(","))
    trained = set(recognizer.metadata.get("train_signers", [])) & signers
    if trained:
        print(f"[경고] {sorted(trained)} 는 이 모델의 학습에 쓰였습니다. 결과가 부풀려집니다.")
    rows = [r for r in load_words(args.manifests)[0]
            if r["signer_id"] in signers and r["label"] in recognizer.labels]
    if not rows:
        p.error("해당 수어자의 클립이 없습니다")
    phrases = [entry for entry in json.loads(Path(args.phrases).read_text(encoding="utf-8"))
               if all(g in recognizer.labels for g in entry["glosses"])]
    report = evaluate(recognizer, rows, phrases, [float(b) for b in args.bonus.split(",")])
    counts = report["counts"]
    print(f"상황 문장 {counts['phrases']}개, 그 밖의 조합 {counts['others']}개, "
          f"수어 아님 {counts['rests']}개")
    print(f"{'보정':>6} {'상황 문장':>9} {'그 밖':>7} {'맞던 것 망침':>10} {'수어 아님 오출력':>12}")
    for bonus, e in report["bonus"].items():
        print(f"{bonus:>6} {e['phrases_exact']:>10.1%} {e['others_exact']:>8.1%} "
              f"{e['others_broken']:>12.1%} {e['false_output_rate']:>14.1%}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
