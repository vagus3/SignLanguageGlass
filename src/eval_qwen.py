"""학습한 안경 시점 Qwen 모델을, 학습에 쓰지 않은 수어자로 확인합니다.

python -m src.eval_qwen models/glasses_qwen_v2 --signers REAL17,REAL18

- 수어자의 3D 관절(data/raw/aihub103_3d/word/<수어자>/)을 안경 시점으로 바꿔서(src.ego_synth)
  모델에 넣고 정답과 비교합니다. 새 수어자는 tools 의 pick_3d 로 먼저 뽑아 두어야 하고,
  단어 구간은 data/processed/word3d_intervals_val.json 에 들어 있어야 합니다.
- 두 가지를 잽니다:
  1) 섞은 예시(단어 1~3개 조합, 상황 문장, 수어 아님) — 학습 때 확인용과 같은 방식
  2) 단어 하나씩: 클립마다 시점을 바꿔 여러 번, 단어별 정확도 (약한 단어를 찾는 용도)
- 모델의 단어 목록에 없는 단어는 뺍니다 (예전 모델 30개도 같은 기준으로 잴 수 있음).
- 합성한 안경 시점 기준이며 실제 안경 카메라 성능이 아닙니다.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src import clock
from src.ego_synth import EgoComposer, load_clips, piece, random_view, render
from src.train_qwen import ROOT, evaluate, fixed_items, load, sentence_manifests


def word_items(composer, signers, reps, seed, kind="WORD"):
    """단어 하나짜리 예시: 클립마다 시점을 reps 번 바꿔서. kind: WORD(단어 클립) 또는 SEN(문장 안에서 잘라낸 단어)."""
    rng = np.random.default_rng(seed)
    items = []
    for signer in signers:
        for gloss, entries in sorted(composer.singles[signer].items()):
            for row, i in entries:
                if kind not in row["id"]:
                    continue
                for _ in range(reps):
                    part = composer.cut(row, i, rng)
                    if part is None:
                        continue
                    items.append({"glosses": [gloss], "text": composer.target([gloss]),
                                  "sequence": render([part], random_view(rng), rng),
                                  "signer": signer})
    return items


def sentence_items(rows, signers, phrases, reps, seed):
    """문장 클립 통째로: 클립 전체(앞뒤 0.2초)에서 정답은 문장의 단어열 (모두 모델 단어일 때만)."""
    rng = np.random.default_rng(seed)
    text = {tuple(p["glosses"]): p["text"] for p in phrases}
    items = []
    for row in rows:
        if "SEN" not in row["id"] or row["signer"] not in signers:
            continue
        if len(row["usable"]) != len(row["glosses"]):
            continue
        part = piece(row, max(row["intervals"][0][0] - 0.2, 0.0), row["intervals"][-1][1] + 0.2)
        if part is None:
            continue
        for _ in range(reps):
            items.append({"glosses": row["glosses"],
                          "text": text.get(tuple(row["glosses"])) or " ".join(row["glosses"]) + ".",
                          "sequence": render([part], random_view(rng), rng), "signer": row["signer"]})
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--signers", required=True, help="확인할 수어자 (예: REAL17,REAL18)")
    p.add_argument("--count", type=int, default=200, help="섞은 예시 개수")
    p.add_argument("--reps", type=int, default=3, help="단어 하나씩: 클립당 시점 바꿔 재는 횟수")
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", help="결과를 저장할 json")
    args = p.parse_args()
    signers = [s for s in args.signers.split(",") if s]
    model, tokenizer, spec = load(args.model, args.device)
    vocabulary = json.loads((ROOT / "assets/glasses/vocabulary.json").read_text(encoding="utf-8"))
    vocabulary = {**vocabulary,
                  "words": [w for w in vocabulary["words"] if w["gloss"] in spec["glosses"]]}
    phrases = json.loads((ROOT / "assets/glasses/phrases.json").read_text(encoding="utf-8"))
    intervals = {}
    for name in ("word3d_intervals.json", "word3d_intervals_val.json"):
        path = ROOT / "data/processed" / name
        if path.exists():
            intervals.update(json.loads(path.read_text()))
    rows = [r for r in load_clips(ROOT / "data/raw/aihub103_3d",
                                  json.loads((ROOT / "data/processed/word3d_ids.json").read_text()),
                                  intervals, sentence_manifests(), vocabulary) if r["signer"] in signers]
    composer = EgoComposer(rows, phrases)
    if not composer.signers:
        p.error(f"{signers} 의 3D 클립이 없습니다 (pick_3d 와 구간 파일을 확인하세요)")
    print(f"모델 {args.model} (단어 {len(spec['glosses'])}개) / 확인 수어자 {composer.signers}, "
          f"클립 {len(rows)}개", flush=True)

    start = clock.now()
    mixed, _ = evaluate(model, tokenizer, fixed_items(composer, composer.signers, args.count,
                                                      args.seed), args.device)
    print("섞은 예시:", {k: (None if v[0] is None else round(v[0], 3), v[1])
                        for k, v in mixed.items()}, flush=True)

    items = word_items(composer, composer.signers, args.reps, args.seed)
    report, rows_out = evaluate(model, tokenizer, items, args.device)
    per = defaultdict(lambda: [0, 0, 0])
    per_signer = defaultdict(lambda: [0, 0])
    for item, row in zip(items, rows_out):
        gloss = item["glosses"][0]
        per[gloss][0] += row["output"] == row["reference"]
        per[gloss][1] += row["ctc"] == item["glosses"]
        per[gloss][2] += 1
        per_signer[item["signer"]][0] += row["output"] == row["reference"]
        per_signer[item["signer"]][1] += 1
    total = sum(v[2] for v in per.values())
    exact = sum(v[0] for v in per.values()) / total
    ctc = sum(v[1] for v in per.values()) / total
    print(f"단어 하나씩 ({total}개): Qwen 정답 {exact:.1%}, 단어 판정층 정답 {ctc:.1%}")
    for signer, (ok, n) in sorted(per_signer.items()):
        print(f"  {signer}: {ok / n:.1%} ({n}개)")
    worst = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][2])
    print("약한 단어 12개:", ", ".join(f"{g} {v[0] / v[2]:.0%}" for g, v in worst[:12]))
    cut = word_items(composer, composer.signers, args.reps, args.seed + 1, kind="SEN")
    inside = None
    if cut:
        _, cut_rows = evaluate(model, tokenizer, cut, args.device)
        inside = {"count": len(cut),
                  "exact": float(np.mean([r["output"] == r["reference"] for r in cut_rows])),
                  "ctc": float(np.mean([r["ctc"] == g["glosses"] for r, g in zip(cut_rows, cut)]))}
        print(f"문장 안에서 잘라낸 단어 ({len(cut)}개): Qwen 정답 {inside['exact']:.1%}, "
              f"단어 판정층 정답 {inside['ctc']:.1%}", flush=True)
    sents = sentence_items(rows, composer.signers, phrases, 1, args.seed)
    sentence = None
    if sents:
        rep, srows = evaluate(model, tokenizer, sents, args.device)
        sentence = {"count": len(sents), "exact": float(np.mean([r["output"] == r["reference"] for r in srows])),
                    "ctc": float(np.mean([r["ctc"] == r["glosses"] for r in srows]))}
        print(f"문장 클립 통째로 ({len(sents)}개): Qwen 정답 {sentence['exact']:.1%}, "
              f"단어 판정층 정답 {sentence['ctc']:.1%}")
    print(f"걸린 시간 {clock.now() - start:.0f}초")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "signers": composer.signers, "mixed": mixed,
             "word_exact": exact, "word_ctc": ctc, "sentence": sentence, "inside_sentence": inside,
             "per_word": {g: {"exact": v[0] / v[2], "ctc": v[1] / v[2], "n": v[2]}
                          for g, v in per.items()},
             "per_signer": {s: v[0] / v[1] for s, v in per_signer.items()}},
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
