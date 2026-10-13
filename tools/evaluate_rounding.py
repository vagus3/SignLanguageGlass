#!/usr/bin/env python
"""Does rounding transmitted points change recognition? (REVIEW_AND_PLAN P1-5)

python tools/evaluate_rounding.py data/processed/continuous/manifest.json \
    --checkpoint models/continuous/ctc.pt --out reports/rounding.json

For every clip and setting it reports the body size (compact JSON and gzip level 1,
as CloudClient sends it) and how far the model features move. With --checkpoint it
also compares the recognized glosses, the score and threshold decisions with the
unrounded input. "exact" is the default lossless encoding (shortest float32
decimals); "2d=N" rounds image coordinates and "3d=N" world coordinates in meters.
Enable SIGN_SERVER_ROUND only for a setting that changes no decision on held-out
real data.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pose_sequence import FEATURE_SETS, PoseSequence, VERSION_3D


def settings(two_d, three_d, has_world):
    out = [("exact", None, None)]
    out += [(f"2d={d}", d, None) for d in two_d]
    if has_world:
        out += [(f"2d={d},3d={w}", d, w) for d in two_d for w in three_d]
    return out


def body_sizes(sequence, decimals, world_decimals):
    body = json.dumps(sequence.to_dict(decimals, world_decimals), allow_nan=False,
                      separators=(",", ":")).encode()
    return len(body), len(gzip.compress(body, compresslevel=1, mtime=0))


def evaluate(clips, *, feature_set="2d", two_d=(4, 3), three_d=(4, 3), recognizer=None):
    has_world = all(s.version == VERSION_3D for _, s in clips)
    if recognizer is not None:
        feature_set = recognizer.feature_set
    extract = FEATURE_SETS[feature_set][0]
    threshold = (getattr(recognizer, "calibration", {}) or {}).get("threshold")
    plain = {cid: body_sizes(s, None, None) for cid, s in clips}
    base = {cid: recognizer.predict(s) for cid, s in clips} if recognizer else {}
    report = {"clips": len(clips), "feature_set": feature_set, "threshold": threshold,
              "settings": {}}
    for name, decimals, world_decimals in settings(two_d, three_d, has_world):
        rows = []
        for cid, sequence in clips:
            moved = PoseSequence.from_dict(json.loads(json.dumps(
                sequence.to_dict(decimals, world_decimals))))
            drift = np.abs(extract(moved) - extract(sequence))
            row = {"json": body_sizes(sequence, decimals, world_decimals)[0],
                   "gzip": body_sizes(sequence, decimals, world_decimals)[1],
                   "feature_max": float(drift.max()), "feature_mean": float(drift.mean())}
            if recognizer is not None:
                glosses, score = recognizer.predict(moved)
                ref_glosses, ref_score = base[cid]
                row.update(same_glosses=glosses == ref_glosses,
                           score_delta=abs(score - ref_score),
                           threshold_flip=(threshold is not None
                                           and (score >= threshold) != (ref_score >= threshold)))
            rows.append(row)
        summary = {
            "json_kb_mean": float(np.mean([r["json"] for r in rows]) / 1024),
            "gzip_kb_mean": float(np.mean([r["gzip"] for r in rows]) / 1024),
            "gzip_vs_plain": float(np.mean([r["gzip"] / plain[cid][1]
                                            for r, (cid, _) in zip(rows, clips)])),
            "feature_max": max(r["feature_max"] for r in rows),
            "feature_mean": float(np.mean([r["feature_mean"] for r in rows]))}
        if recognizer is not None:
            summary.update(changed_glosses=sum(not r["same_glosses"] for r in rows),
                           max_score_delta=max(r["score_delta"] for r in rows),
                           threshold_flips=sum(r["threshold_flip"] for r in rows))
        report["settings"][name] = summary
    return report


def table(report):
    lines = [f"{'setting':<12} {'json KB':>8} {'gzip KB':>8} {'gzip/exact':>10} "
             f"{'feat max':>9} {'feat mean':>10}  recognition"]
    exact = report["settings"]["exact"]["gzip_kb_mean"]
    for name, s in report["settings"].items():
        recog = ("" if "changed_glosses" not in s else
                 f"{s['changed_glosses']} changed, {s['threshold_flips']} threshold flips, "
                 f"max score delta {s['max_score_delta']:.2e}")
        lines.append(f"{name:<12} {s['json_kb_mean']:>8.1f} {s['gzip_kb_mean']:>8.1f} "
                     f"{s['gzip_kb_mean'] / exact:>10.2f} {s['feature_max']:>9.2e} "
                     f"{s['feature_mean']:>10.2e}  {recog}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--checkpoint", help="compare recognition (PyTorch checkpoint)")
    p.add_argument("--features", choices=sorted(FEATURE_SETS), default="2d",
                   help="feature set for drift without a checkpoint")
    p.add_argument("--2d", dest="two_d", type=int, nargs="+", default=[4, 3])
    p.add_argument("--3d", dest="three_d", type=int, nargs="+", default=[4, 3])
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    args = p.parse_args()
    if args.out and Path(args.out).exists():
        p.error("Output exists; choose a new path")
    path = Path(args.manifest)
    rows = json.loads(path.read_text(encoding="utf-8"))["clips"][:args.limit]
    clips = [(r["id"], PoseSequence.load(path.parent / r["file"])) for r in rows]
    recognizer = None
    if args.checkpoint:
        from src.continuous_sign import Recognizer
        recognizer = Recognizer(args.checkpoint)
    report = evaluate(clips, feature_set=args.features, two_d=args.two_d,
                      three_d=args.three_d, recognizer=recognizer)
    print(table(report))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
