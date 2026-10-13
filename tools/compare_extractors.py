#!/usr/bin/env python
"""Compare two pose datasets of the same clips, e.g. OpenPose JSON vs MediaPipe video.

python tools/compare_extractors.py data/processed/op/manifest.json \\
  data/processed/mp/manifest.json --out reports/extractors.json

Frames are paired by timestamp, because training uses the timestamps: a
numbering gap that the timestamps keep is fine, frames renumbered after a drop
are not. Per clip: frame counts, detection rates, median joint distance in
pixels of the first dataset's image, whether the hand labels agree or look
swapped, and the frame offset that best aligns each 30-frame window. It
compares coordinates only; whether re-extraction helps recognition still needs
training on the same split.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pose_sequence import PoseSequence

LEFT, RIGHT = np.arange(0, 21), np.arange(21, 42)
HANDS = np.r_[LEFT, RIGHT]
BODY = np.arange(43, 49)            # shoulders, elbows, wrists; the neck is derived
WINDOW, MAX_OFFSET, MIN_PAIRS = 30, 8, 20
# A window counts as misaligned when another offset cuts the distance this much.
SHIFT_GAIN, SWAP_GAIN = 0.7, 0.5


def _distances(pa, ma, pb, mb, joints_a, joints_b, i, j):
    """Distances for A frames i vs B frames j where both joints are valid."""
    i, j = np.asarray(i), np.asarray(j)
    keep = (j >= 0) & (j < len(pb))
    i, j = i[keep], j[keep]
    if not len(i):
        return np.empty(0)
    valid = ma[i][:, joints_a] & mb[j][:, joints_b]
    d = np.linalg.norm(pa[i][:, joints_a] - pb[j][:, joints_b], axis=-1)
    return d[valid]


def _median(values):
    return float(np.median(values)) if len(values) else None


def _rates(mask):
    return {"left_hand": float(mask[:, 0].mean()), "right_hand": float(mask[:, 21].mean()),
            "shoulders": float((mask[:, 43] & mask[:, 44]).mean())}


def compare_clip(a, b):
    """a, b: PoseSequence of the same clip. Pixels use a's image size."""
    scale = np.asarray(a.image_size, np.float32)
    pa, pb = a.coordinates * scale, b.coordinates * scale
    ma, mb = a.mask, b.mask
    frames = np.arange(len(pa))
    # The B frame each A frame should match, from the timestamps.
    step = float(np.median(np.diff(b.timestamps)))
    base = np.rint((a.timestamps - a.timestamps[0]) / step).astype(int)
    windows = []
    for lo in range(0, len(pa), WINDOW):
        span = frames[lo:lo + WINDOW]
        scores = {}
        for d in range(-MAX_OFFSET, MAX_OFFSET + 1):
            dist = _distances(pa, ma, pb, mb, HANDS, HANDS, span, base[span] + d)
            if len(dist) >= MIN_PAIRS:
                scores[d] = float(np.median(dist))
        if 0 not in scores:
            continue
        best = min(scores, key=scores.get)
        if scores[best] >= SHIFT_GAIN * scores[0]:
            best = 0                     # Not clearly better: keep the given alignment.
        windows.append({"start": int(lo), "end": int(span[-1]) + 1, "offset": int(best),
                        "median_px": scores[best], "median_px_at_0": scores[0]})
    aligned = np.concatenate([_distances(pa, ma, pb, mb, HANDS, HANDS,
                                         frames[w["start"]:w["end"]],
                                         base[w["start"]:w["end"]] + w["offset"])
                              for w in windows]) if windows else np.empty(0)
    direct = _distances(pa, ma, pb, mb, HANDS, HANDS, frames, base)
    swapped = _distances(pa, ma, pb, mb, HANDS, np.r_[RIGHT, LEFT], frames, base)
    direct_median, swapped_median = _median(direct), _median(swapped)
    aspect_a = a.image_size[0] / a.image_size[1]
    aspect_b = b.image_size[0] / b.image_size[1]
    return {
        "frames": [len(pa), len(pb)], "image_size": [list(a.image_size), list(b.image_size)],
        "aspect_mismatch": abs(aspect_a - aspect_b) > 0.01 * aspect_a,
        "detection": [_rates(ma), _rates(mb)],
        "hands_px": direct_median, "hands_px_aligned": _median(aligned),
        "body_px": _median(_distances(pa, ma, pb, mb, BODY, BODY, frames, base)),
        "hands_swapped": bool(direct_median is not None and swapped_median is not None
                              and swapped_median < SWAP_GAIN * direct_median),
        "windows": windows,
        "misaligned": any(w["offset"] != 0 for w in windows),
    }


def compare(first, second):
    """Compare the clips both manifests share, matched by clip id."""
    first, second = Path(first), Path(second)
    ma = json.loads(first.read_text(encoding="utf-8"))
    mb = json.loads(second.read_text(encoding="utf-8"))
    rows_a = {r["id"]: r for r in ma["clips"]}
    rows_b = {r["id"]: r for r in mb["clips"]}
    shared = [k for k in rows_a if k in rows_b]
    if not shared:
        raise ValueError("The manifests share no clip ids")
    clips = []
    for identity in shared:
        a = PoseSequence.load(first.parent / rows_a[identity]["file"])
        b = PoseSequence.load(second.parent / rows_b[identity]["file"])
        clips.append({"id": identity, **compare_clip(a, b)})

    def pooled(key):
        values = [c[key] for c in clips if c[key] is not None]
        return float(np.median(values)) if values else None

    return {
        "first": str(first), "second": str(second),
        "extractors": [ma.get("extractor"), mb.get("extractor")],
        "pixels": "first dataset image size",
        "clips": clips,
        "only_in_first": [k for k in rows_a if k not in rows_b],
        "only_in_second": [k for k in rows_b if k not in rows_a],
        "summary": {
            "clips": len(clips),
            "median_hands_px": pooled("hands_px"),
            "median_hands_px_aligned": pooled("hands_px_aligned"),
            "median_body_px": pooled("body_px"),
            "misaligned": [c["id"] for c in clips if c["misaligned"]],
            "hands_swapped": [c["id"] for c in clips if c["hands_swapped"]],
            "aspect_mismatch": [c["id"] for c in clips if c["aspect_mismatch"]],
        },
    }


def table(report):
    def px(v):
        return "-" if v is None else f"{v:.1f}"
    lines = [f"{'clip':28} {'frames':>9} {'hands px':>9} {'aligned':>8} {'body px':>8}  flags"]
    for c in report["clips"]:
        shifts = sorted({w["offset"] for w in c["windows"] if w["offset"]})
        flags = (["swapped hands"] if c["hands_swapped"] else []) + (
            [f"offset {shifts} from frame "
             f"{min(w['start'] for w in c['windows'] if w['offset'])}"] if shifts else [])
        lines.append(f"{c['id'][:28]:28} {c['frames'][0]:>4}/{c['frames'][1]:<4} "
                     f"{px(c['hands_px']):>9} {px(c['hands_px_aligned']):>8} "
                     f"{px(c['body_px']):>8}  {', '.join(flags)}")
    s = report["summary"]
    lines.append(f"median hands {px(s['median_hands_px'])}px (aligned {px(s['median_hands_px_aligned'])}px), "
                 f"body {px(s['median_body_px'])}px over {s['clips']} clips; "
                 f"misaligned {len(s['misaligned'])}, swapped {len(s['hands_swapped'])}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("first", help="reference manifest (pixels use its image size)")
    p.add_argument("second")
    p.add_argument("--out", help="write the full report as JSON")
    args = p.parse_args()
    try:
        report = compare(args.first, args.second)
    except ValueError as error:
        p.error(str(error))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table(report))


if __name__ == "__main__":
    main()
