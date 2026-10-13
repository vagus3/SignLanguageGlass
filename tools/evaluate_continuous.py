#!/usr/bin/env python
"""Held-out CTC evaluation, optional sentence composition and validation-only calibration.

--compare-llm runs the same composition as the server (reviewed sentence, language
model, fallback), so llm_text is what the server would speak; fallback previews
are recorded but never count as spoken output.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuous_sign import (Recognizer, load_dataset, split_dataset,
                                 save_checkpoint, edit_distance)
from src.translation import Glossary, compose, mapping_hash  # noqa: F401  (mapping_hash: API)


def negative_duration(rows):
    """Only annotated, nonoverlapping real capture intervals count toward 10 minutes."""
    intervals = {}
    total = 0.0
    for row in rows:
        if not row.get("negative") or row["glosses"]:
            continue
        start = row.get("capture_start")
        if not isinstance(start, (int, float)) or not np.isfinite(start):
            raise ValueError("Negative calibration clips need a real capture_start timestamp")
        duration = float(row["sequence"].timestamps[-1] - row["sequence"].timestamps[0])
        intervals.setdefault((row["signer_id"], row["session_id"]), []).append((start, start+duration))
        total += duration
    for items in intervals.values():
        items.sort()
        if any(b[0] < a[1] for a, b in zip(items, items[1:])):
            raise ValueError("Overlapping negative capture intervals cannot inflate duration")
    return total


def review_verdict(clip, review):
    """True/False when a person judged exactly what this clip would speak, else None.

    Two forms of review count (tools/review_sentences.py writes the second):
    - clips: a reviewer watched this clip and judged its prediction and text.
      That verdict wins, including a rejection.
    - sequences: a reviewer judged "these glosses -> this sentence". A rejection
      always counts. An approval says nothing about whether the signer meant those
      glosses, so it only counts when the prediction equals the reference glosses.
    Either way the reviewed text must be the text the system actually produced.
    """
    text = clip.get("llm_text")
    if not text or not review.get("reviewer"):
        return None
    entry = review.get("clips", {}).get(clip["id"])
    if entry is not None:
        if (entry.get("prediction") != clip["prediction"] or entry.get("text") != text
                or type(entry.get("meaning_preserved")) is not bool):
            return None
        return entry["meaning_preserved"]
    verdicts = [s["meaning_preserved"] for s in review.get("sequences", [])
                if s.get("glosses") == clip["prediction"] and s.get("text") == text
                and type(s.get("meaning_preserved")) is bool]
    if not verdicts:
        return None
    if not all(verdicts):
        return False
    return True if clip["prediction"] == clip["reference"] else None


def review_accepts(clip, review):
    return review_verdict(clip, review) is True


def choose_threshold(results, duration, review):
    """Maximize positive output coverage subject to <=2 false outputs per 600 seconds."""
    positives = [r for r in results if r["reference"]]
    negatives = [r for r in results if r.get("negative") and not r["reference"]]
    if duration < 600 or not positives or not review.get("reviewer"):
        return {"enabled": False, "reason": "Need >=600s negatives, positive clips and semantic review"}
    candidates = sorted({r["score"] for r in results if r["prediction"]} | {0.0, 1.000001})
    choices = []
    for threshold in candidates:
        accepted = [r for r in positives if r["prediction"] and r["score"] >= threshold]
        false = sum(bool(r["prediction"]) and r["score"] >= threshold for r in negatives)
        if false * 600 / duration > 2:
            continue
        # The review must assess actual LLM output from these predictions, not references alone.
        if not all(review_accepts(r, review) for r in accepted):
            continue
        if accepted:
            choices.append((len(accepted), -threshold, threshold, false))
    if not choices:
        return {"enabled": False, "reason": "No threshold passes false-output and meaning checks"}
    _, _, threshold, false = max(choices)
    return {"enabled": True, "threshold": threshold, "negative_seconds": duration,
            "false_outputs": false, "false_outputs_per_10min": false * 600 / duration,
            "coverage": sum(bool(r["prediction"]) and r["score"] >= threshold
                            for r in positives) / len(positives),
            "reviewer": review["reviewer"]}


def rules_text(glosses, mapping):
    import config as C
    words = [mapping[g] for g in glosses]
    output, i = [], 0
    rules = sorted(C.PHRASE_RULES, key=lambda r: -len(r[0]))
    while i < len(words):
        for pattern, sentence in rules:
            if tuple(words[i:i+len(pattern)]) == pattern:
                output.append(sentence)
                i += len(pattern)
                break
        else:
            output.append(words[i])
            i += 1
    return " ".join(output)


def run_evaluation(recognizer, rows, mapping=None, translator=None):
    """mapping: a Glossary or a plain {gloss: meaning} dict."""
    glossary = Glossary.from_mapping(mapping) if mapping else None
    results, errors, count = [], 0, 0
    for row in rows:
        prediction, score = recognizer.predict(row["sequence"])
        reference = row["glosses"]
        errors += edit_distance(reference, prediction)
        count += len(reference)
        entry = {"id": row["id"], "reference": reference, "prediction": prediction,
                 "score": score, "negative": row.get("negative", False),
                 "reference_text": row.get("text", ""), "llm_text": None}
        if prediction and glossary and not glossary.unknown(prediction):
            entry["rule_text"] = rules_text(prediction, glossary.meanings)
            if translator:
                composition = compose(prediction, glossary, translator)
                entry["composition"] = {"status": composition.status,
                                        "source": composition.source,
                                        "issues": list(composition.issues)}
                if composition.status == "ok":
                    entry["llm_text"] = composition.text
                else:
                    entry["llm_error"] = "fallback_preview_not_spoken"
                    entry["fallback_text"] = composition.text
        results.append(entry)
    return {"wer": errors / count if count else None, "reference_words": count,
            "errors": errors, "clips": results, "semantic_accuracy": "requires human review"}


def summarize(report, rows, calibration, review=None):
    threshold = calibration.get("threshold", 0)
    positives = [r for r in report["clips"] if r["reference"]]
    emitted = [r for r in positives if r["prediction"] and r["score"] >= threshold]
    negatives = [r for r in report["clips"] if r.get("negative") and not r["reference"]]
    false = sum(bool(r["prediction"]) and r["score"] >= threshold for r in negatives)
    try:
        duration = negative_duration(rows)
        duration_error = None
    except ValueError as error:
        duration, duration_error = None, str(error)
    reviewed, meaning_errors = 0, 0
    review = review or {}
    for result in emitted:
        verdict = review_verdict(result, review)
        if verdict is not None:
            reviewed += 1
            meaning_errors += not verdict
    return {"recognition_coverage": len(emitted) / len(positives) if positives else None,
            "recognition_abstention_rate": 1-len(emitted)/len(positives) if positives else None,
            "threshold": calibration.get("threshold"),
            "automatic_speech_calibrated": calibration.get("enabled", False),
            "false_recognition_outputs": false, "verified_negative_seconds": duration,
            "false_outputs_per_10min": false*600/duration if duration else None,
            "negative_duration_error": duration_error, "meaning_reviewed_clips": reviewed,
            "meaning_errors": meaning_errors, "unreviewed_output_clips": len(emitted)-reviewed,
            "note": "Coverage counts recognizer outputs; actual speech also requires valid LLM output"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--mapping", help="glossary.json (meanings, kinds, reviewed sentences) or mapping.json")
    p.add_argument("--compare-llm", action="store_true")
    p.add_argument("--calibrate-out")
    p.add_argument("--semantic-review", help="review.json: reviewer, clips keyed by ID with "
                   "prediction/text/meaning_preserved and/or gloss-level sequences "
                   "(tools/review_sentences.py import)")
    args = p.parse_args()
    if args.calibrate_out and (args.split != "val" or not args.semantic_review
                              or not args.compare_llm or not args.mapping):
        p.error("Calibration requires --split val --compare-llm --mapping --semantic-review")
    paths = [Path(args.out)] + ([Path(args.calibrate_out)] if args.calibrate_out else [])
    if any(path.exists() for path in paths):
        p.error("Output exists; choose new paths")
    recognizer = Recognizer(args.checkpoint)
    rows, skipped = load_dataset(args.manifest)
    # Recheck signer and motion independence; never calibrate on test.
    splits = split_dataset(rows, recognizer.metadata.get("split_seed", 0))
    expected = recognizer.metadata.get("splits")
    actual = {s: [r["id"] for r in clips] for s, clips in splits.items()}
    if expected != actual:
        p.error("Dataset/splits differ from training checkpoint")
    try:
        mapping = Glossary.load(args.mapping) if args.mapping else None
    except ValueError as error:
        p.error(str(error))
    translator = None
    if args.compare_llm:
        if not mapping:
            p.error("--mapping required for language comparison")
        from src.translation import VllmTranslator
        translator = VllmTranslator()
    report = run_evaluation(recognizer, splits[args.split], mapping, translator)
    report.update(split=args.split, skipped=skipped, mock=False)
    review = (json.loads(Path(args.semantic_review).read_text(encoding="utf-8"))
              if args.semantic_review else {})
    report["summary"] = summarize(report, splits[args.split], recognizer.calibration, review)
    if args.calibrate_out:
        duration = negative_duration(splits["val"])
        calibration = choose_threshold(report["clips"], duration, review)
        # Same hash the server compares: meanings, kinds and reviewed sentences.
        calibration.update(mapping_hash=mapping.policy_hash(), language_model=translator.model)
        report["calibration"] = calibration
        report["summary"] = summarize(report, splits["val"], calibration, review)
        destination = Path(args.calibrate_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(destination, recognizer.model, recognizer.labels,
                        calibration=calibration, metadata=recognizer.metadata)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WER={report['wer']}; semantic evaluation requires reviewed actual outputs")


if __name__ == "__main__":
    main()
