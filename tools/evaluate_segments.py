#!/usr/bin/env python
"""How well does automatic segmentation find sentences? (src/segmenter.py)

python tools/evaluate_segments.py data/processed/continuous/manifest.json --viewpoint webcam
python tools/evaluate_segments.py MANIFEST --split val --grid --out reports/segments.json

Each clip is replayed frame by frame through ActivitySegmenter, applying its
events the way main.py does (an automatic start keeps PRE_ROLL seconds before it).
Rows need "sign_intervals": [[start, end], ...] in seconds from the clip start
(the morpheme time annotations, or hand-marked team recordings); rows with
"negative": true and no intervals measure false starts during everyday motion.
  detected        true intervals overlapped by a segment
  start/end delay segment edge minus true edge (positive = late), p50/p95
  cut_start/end   segments that begin after / end before the true edge by > TOLERANCE
  split           true intervals broken into several segments
  false_per_10min segments overlapping no true interval
--grid searches thresholds on one split only; report final numbers on another.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pose_sequence import PoseSequence
from src.segmenter import VIEWPOINTS, ActivitySegmenter

PRE_ROLL = 0.3      # main.py: CloudController starts automatic segments with this pre-roll
TOLERANCE = 0.2     # seconds a segment may miss a true edge before it counts as cut
GRID = {"start_speed": (1.0, 1.5, 2.0), "still_speed": (0.3, 0.4, 0.6),
        "end_hold": (0.6, 0.8, 1.2), "still_hold": (1.0, 1.5, 2.0)}


def replay(sequence, segmenter, pre_roll=PRE_ROLL):
    """Segments [(start, end, closed_by)] as the main loop would record them."""
    t0 = float(sequence.timestamps[0])
    segments, recording, start = [], False, None
    for t, points, mask in zip(sequence.timestamps, sequence.coordinates, sequence.mask):
        t = float(t) - t0
        event = segmenter.update(points, mask, t, recording, sequence.image_size)
        if event == "start" and not recording:
            recording, start = True, max(0.0, t - pre_roll)
        elif event in ("stop", "discard") and recording:
            recording = False
            if event == "stop":
                segments.append((start, t, "stop"))
    if recording:
        segments.append((start, float(sequence.timestamps[-1]) - t0, "clip_end"))
    return segments


def score_clip(segments, intervals, tolerance=TOLERANCE):
    rows, used = [], set()
    for true_start, true_end in intervals:
        hits = [i for i, (s, e, _) in enumerate(segments) if s < true_end and e > true_start]
        used.update(hits)
        if not hits:
            rows.append({"detected": False})
            continue
        start, end = segments[hits[0]][0], segments[hits[-1]][1]
        rows.append({"detected": True, "start_delay": start - true_start,
                     "end_delay": end - true_end, "split": len(hits) > 1,
                     "cut_start": start > true_start + tolerance,
                     "cut_end": end < true_end - tolerance})
    false = [segments[i] for i in range(len(segments)) if i not in used]
    return rows, false


def evaluate(clips, viewpoint="webcam", **params):
    """clips: [(id, PoseSequence, intervals)]."""
    rows, false, seconds, per_clip = [], 0, 0.0, []
    for clip_id, sequence, intervals in clips:
        segmenter = ActivitySegmenter(viewpoint=viewpoint, **params)
        segments = replay(sequence, segmenter)
        scored, extra = score_clip(segments, intervals)
        rows += scored
        false += len(extra)
        seconds += float(sequence.timestamps[-1] - sequence.timestamps[0])
        per_clip.append({"id": clip_id, "segments": segments, "intervals": intervals,
                         "false": len(extra)})
    hits = [r for r in rows if r["detected"]]

    def pct(key):
        values = [r[key] for r in hits]
        return ({"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
                if values else None)
    return {"intervals": len(rows), "detected": len(hits),
            "start_delay": pct("start_delay"), "end_delay": pct("end_delay"),
            "cut_start": sum(r["cut_start"] for r in hits),
            "cut_end": sum(r["cut_end"] for r in hits),
            "split": sum(r["split"] for r in hits), "false_starts": false,
            "seconds": seconds, "false_per_10min": false * 600 / seconds if seconds else None,
            "params": params, "clips": per_clip}


def errors(report):
    """Grid objective: every broken sentence, then false starts."""
    missed = report["intervals"] - report["detected"]
    return (missed + report["cut_start"] + report["cut_end"] + report["split"],
            report["false_per_10min"] or 0.0)


def grid_search(clips, viewpoint="webcam", grid=GRID):
    best = None
    names = sorted(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        report = evaluate(clips, viewpoint, **dict(zip(names, values)))
        if best is None or errors(report) < errors(best):
            best = report
    return best


def load(manifest, split=None):
    path = Path(manifest)
    rows = json.loads(path.read_text(encoding="utf-8"))["clips"]
    if split:
        if any("split" not in r for r in rows):
            raise ValueError("--split needs a split on every row")
        rows = [r for r in rows if r["split"] == split]
    clips = []
    for row in rows:
        intervals = row.get("sign_intervals")
        if intervals is None and not row.get("negative"):
            continue        # no boundary annotation: nothing to score against
        intervals = [tuple(map(float, i)) for i in intervals or []]
        if any(not 0 <= s < e for s, e in intervals):
            raise ValueError(f"{row['id']}: sign_intervals must be increasing [start, end]")
        clips.append((row["id"], PoseSequence.load(path.parent / row["file"]), intervals))
    if not clips:
        raise ValueError("No rows with sign_intervals or negative=true")
    return clips


def summary_line(r):
    def edge(v):
        return f"{v['p50']:+.2f}/{v['p95']:+.2f}s" if v else "-"
    return (f"detected {r['detected']}/{r['intervals']}  start {edge(r['start_delay'])}  "
            f"end {edge(r['end_delay'])}  cut start/end {r['cut_start']}/{r['cut_end']}  "
            f"split {r['split']}  false {r['false_starts']} "
            f"({(r['false_per_10min'] or 0):.1f}/10min over {r['seconds']:.0f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--viewpoint", choices=VIEWPOINTS, default="webcam")
    p.add_argument("--split", help="only rows with this split (required with --grid)")
    p.add_argument("--grid", action="store_true", help="search thresholds (tuning split only)")
    p.add_argument("--out")
    args = p.parse_args()
    if args.grid and not args.split:
        p.error("--grid needs --split: tune on one split, report on another")
    if args.out and Path(args.out).exists():
        p.error("Output exists; choose a new path")
    try:
        clips = load(args.manifest, args.split)
    except (OSError, ValueError) as error:
        p.error(str(error))
    report = (grid_search(clips, args.viewpoint) if args.grid
              else evaluate(clips, args.viewpoint))
    print(summary_line(report))
    if args.grid:
        print("best params:", report["params"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
