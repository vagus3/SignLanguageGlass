#!/usr/bin/env python
"""Train several feature configurations on one manifest and compare validation results.

python tools/compare_features.py data/processed/continuous3d/manifest.json \
  --out reports/features --features 2d 2d_no_velocity 3d_orientation 3d

Every configuration uses the same rows, splits, vocabulary, seed and settings.
The test split is never evaluated here: choose on validation, then run
evaluate_continuous.py once on test with the chosen checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pose_sequence import FEATURE_PRESETS, FEATURE_SETS
from src.continuous_sign import Recognizer
from src.train_continuous import prepare_training, train_one
from tools.evaluate_continuous import run_evaluation, summarize


def compare(manifest, out, feature_sets, *, epochs=30, batch=8, seed=0, device="cpu"):
    out = Path(out)
    if out.exists():
        raise ValueError("Output exists; choose a new directory")
    if not feature_sets or len(set(feature_sets)) != len(feature_sets):
        raise ValueError("Give one or more distinct feature sets")
    unknown = set(feature_sets) - set(FEATURE_SETS)
    if unknown:
        raise ValueError(f"Unknown feature sets: {sorted(unknown)}")
    skipped, splits, labels = prepare_training(manifest, seed, feature_sets)
    out.mkdir(parents=True)
    results = []
    for name in feature_sets:
        begin = time.perf_counter()
        report = train_one(splits, labels, name, out / name, epochs=epochs, batch=batch,
                           seed=seed, device=device, skipped=skipped)
        seconds = time.perf_counter() - begin
        evaluation = run_evaluation(Recognizer(out / name / "ctc.pt"), splits["val"])
        summary = summarize(evaluation, splits["val"], {})
        results.append({
            "feature_set": name, "blocks": FEATURE_PRESETS[name],
            "feature_dim": FEATURE_SETS[name][1], "val_wer": evaluation["wer"],
            "val_errors": evaluation["errors"], "val_reference_words": evaluation["reference_words"],
            "output_rate": summary["recognition_coverage"],
            "false_outputs_on_negatives": summary["false_recognition_outputs"],
            "false_outputs_per_10min": summary["false_outputs_per_10min"],
            "final_train_loss": report["history"][-1], "train_seconds": seconds,
            "checkpoint": str(Path(name) / "ctc.pt")})
    comparison = {
        "manifest": str(manifest), "seed": seed, "epochs": epochs, "batch": batch,
        "vocabulary": labels[1:], "split_sizes": {s: len(c) for s, c in splits.items()},
        "splits": {s: [r["id"] for r in c] for s, c in splits.items()},
        "results": results, "test_evaluated": False,
        "note": ("One seed per configuration; with little data, confirm differences "
                 "with other --seed values before drawing conclusions.")}
    (out / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    (out / "comparison.md").write_text(markdown(comparison), encoding="utf-8")
    return comparison


def markdown(comparison):
    def number(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"
    lines = [f"# 특징 구성 비교 (validation, seed {comparison['seed']}, "
             f"{comparison['epochs']} epochs)", "",
             "| 구성 | 차원 | val WER | 출력률 | 일상동작 오출력 | 오출력/10분 | 학습 손실 | 학습 초 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in comparison["results"]:
        lines.append(f"| {r['feature_set']} | {r['feature_dim']} | {number(r['val_wer'])} | "
                     f"{number(r['output_rate'])} | {r['false_outputs_on_negatives']} | "
                     f"{number(r['false_outputs_per_10min'], 2)} | "
                     f"{number(r['final_train_loss'], 2)} | {r['train_seconds']:.1f} |")
    sizes = comparison["split_sizes"]
    lines += ["", f"분할: train {sizes['train']} / val {sizes['val']} / test {sizes['test']} "
              "(test는 평가하지 않음).", "", comparison["note"], ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--features", nargs="+", default=["2d", "3d"], choices=sorted(FEATURE_SETS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1:
        p.error("epochs and batch must be positive")
    try:
        comparison = compare(args.manifest, args.out, args.features, epochs=args.epochs,
                             batch=args.batch, seed=args.seed, device=args.device)
    except ValueError as error:
        p.error(str(error))
    print(markdown(comparison))


if __name__ == "__main__":
    main()
