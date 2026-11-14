#!/usr/bin/env python
"""SEN keypoint JSON + matching morpheme JSON -> per-clip training data.

Without --image-size, cache pixel coordinates (sen-raw-xy-v1). With a verified
--image-size WIDTH HEIGHT, emit pose2d-sequence-v1 for the CTC trainer.
Use --from-cache to normalize a previously packed dataset without reopening JSON.
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_sources import source_entry
from src.joints import BODY_NAMES, OP_POSE, person_from_frame
from src.pose_sequence import JOINTS, MAX_FRAMES, MAX_SECONDS, PoseSequence, VERSION
from tools.prepare_continuous import write_dataset
from tools.prepare_sen_colab import CLIP, clip_folders

RAW_VERSION = "sen-raw-xy-v1"
FRAME = re.compile(r"_(\d+)_keypoints\.json$")
MASK_POLICY = "finite, nonzero xy; third value ignored"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def image_size(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer))
            or not 1 <= v <= 16384 for v in value)):
        raise ValueError("image_size must be actual [width,height] integers in 1..16384")
    return tuple(int(v) for v in value)


def positive_fps(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError("fps must be a finite positive number")
    return float(value)


def read_morphemes(path, identity):
    """Join by full clip ID and order by time; never substitute a REAL/SEN summary."""
    node = read_json(path)
    if not isinstance(node, dict) or not isinstance(node.get("metaData"), dict):
        raise ValueError("missing morpheme metaData")
    meta = node["metaData"]
    if Path(str(meta.get("name", ""))).stem != identity:
        raise ValueError("morpheme metaData.name differs from keypoint clip ID")
    duration = meta.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("invalid morpheme duration")
    if not isinstance(node.get("data"), list) or not node["data"]:
        raise ValueError("empty morpheme data")
    segments = []
    for item in node["data"]:
        if not isinstance(item, dict):
            raise ValueError("morpheme segment must be an object")
        start, end = item.get("start"), item.get("end")
        try:
            if isinstance(start, bool) or isinstance(end, bool):
                raise ValueError()
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            raise ValueError("invalid morpheme start/end") from None
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration + .05:
            raise ValueError("morpheme time outside clip duration")
        attributes = item.get("attributes")
        if not isinstance(attributes, list) or len(attributes) != 1 or not isinstance(attributes[0], dict):
            raise ValueError("one unambiguous attributes.name is required per segment")
        name = attributes[0].get("name")
        if not isinstance(name, str) or not name.strip() or name.strip() == "<blank>":
            raise ValueError("empty or reserved morpheme label")
        segments.append({"start": start, "end": end, "gloss": name.strip(),
                         "attributes": attributes[0]})
    segments.sort(key=lambda s: (s["start"], s["end"]))
    if any(a["end"] > b["start"] + .001 for a, b in zip(segments, segments[1:])):
        raise ValueError("overlapping morphemes have no unambiguous CTC ordering")
    return segments, float(duration)


def official_annotations(root, *, view="F", size=None, fps=30):
    """Index folders and labels only: no expensive recursive frame-file scan."""
    root = Path(root).resolve()
    if view not in (*"FDLRU", "all"):
        raise ValueError("view must be F/D/L/R/U or all")
    size = image_size(size) if size is not None else None
    fps = positive_fps(fps)
    folders = clip_folders(root)
    selected = [p for p in folders if view == "all" or CLIP.fullmatch(p.name)[4] == view]
    if not selected:
        raise ValueError("No matching SEN clip folders")
    # Prune keypoint clip folders: do not enumerate 300,000 frame files just
    # to find 2,500 label files. Also works when root itself is the SEN folder.
    label_paths = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not CLIP.fullmatch(d))
        label_paths.extend(Path(directory) / f for f in files if f.endswith("_morpheme.json"))
    labels = {}
    for path in sorted(label_paths):
        identity = path.name.removesuffix("_morpheme.json")
        labels.setdefault(identity, []).append(path)
    rows, excluded = [], []
    seen = set()
    for folder in selected:
        identity = folder.name
        if identity in seen:
            raise ValueError(f"Duplicate keypoint clip ID: {identity}")
        seen.add(identity)
        matches = labels.get(identity, [])
        try:
            if len(matches) != 1:
                raise ValueError("missing exact morpheme JSON" if not matches else "duplicate morpheme JSON")
            segments, duration = read_morphemes(matches[0], identity)
        except (ValueError, OSError) as error:
            excluded.append({"id": identity, "reason": str(error)})
            continue
        match = CLIP.fullmatch(identity)
        rows.append({"id": identity, "keypoints_dir": folder.relative_to(root).as_posix(),
                     "sentence_id": match[2], "source_token": match[3],
                     "signer_id": f"unverified-source-token:{match[3]}",
                     "session_id": f"aihub-import:{match[3]}", "group_id": match[1],
                     "continuous": True, "glosses": [s["gloss"] for s in segments],
                     "gloss_segments": segments, "text": None, "source_duration_seconds": duration,
                     "label_file": matches[0].relative_to(root).as_posix()})
    if not rows:
        raise ValueError(f"No clips with exact valid morpheme labels; first exclusions: {excluded[:3]}")
    return {"image_size": list(size) if size else None, "fps": fps, "clips": rows,
            "label_alignment_verified": True, "signer_identity_verified": False,
            "label_matching": "exact full clip ID + morpheme metaData.name; ordered by start/end",
            "index_audit": {"clip_folders": len(folders), "selected_view": view,
                            "selected_clips": len(selected), "labelled_clips": len(rows),
                            "excluded": excluded}}


def read_xy(person):
    """Read actual 2D x/y, retaining missing hands/body as masked zeros."""
    xy, mask = np.zeros((JOINTS, 2), np.float32), np.zeros(JOINTS, bool)
    for key, count, offset in [("hand_left_keypoints_2d", 21, 0),
                               ("hand_right_keypoints_2d", 21, 21),
                               ("pose_keypoints_2d", 25, None)]:
        values = person.get(key, [])
        if values is None or (isinstance(values, list) and not values):
            continue
        try:
            a = np.asarray(values, np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: numeric triples required") from None
        if a.shape not in ((count * 3,), (count, 3)):
            raise ValueError(f"{key}: expected {count} (x,y,third) triples; got {a.shape}")
        p = a.reshape(count, 3)[:, :2]
        valid = np.isfinite(p).all(axis=1) & (p != 0).any(axis=1)
        valid &= (np.abs(p) <= np.finfo(np.float32).max).all(axis=1)
        p = np.where(valid[:, None], p, 0).astype(np.float32)
        if offset is not None:
            xy[offset:offset+count], mask[offset:offset+count] = p, valid
        else:
            for i, name in enumerate(BODY_NAMES[1:], 43):
                xy[i], mask[i] = p[OP_POSE[name]], valid[OP_POSE[name]]
            xy[42] = xy[43] / 2 + xy[44] / 2
            mask[42] = mask[43] and mask[44]
    return np.where(mask[:, None], xy, 0), mask


@dataclass
class RawXY:
    timestamps: np.ndarray
    xy: np.ndarray
    mask: np.ndarray

    def validate(self):
        t, xy, m = np.asarray(self.timestamps), np.asarray(self.xy), np.asarray(self.mask)
        if t.ndim != 1 or not 2 <= len(t) <= MAX_FRAMES or xy.shape != (len(t), JOINTS, 2) or m.shape != (len(t), JOINTS):
            raise ValueError("expected 2..900 frames, xy (T,49,2), mask (T,49)")
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or t[-1] - t[0] > MAX_SECONDS + 1e-6:
            raise ValueError("timestamps must increase over at most 15 seconds")
        if not np.isfinite(xy).all() or not np.isin(m, [0, 1]).all():
            raise ValueError("invalid cached coordinates or mask")
        if not m[:, :42].any():
            raise ValueError("no valid hand coordinates")
        self.timestamps = t.astype(np.float64)
        self.mask = m.astype(bool)
        self.xy = np.where(self.mask[..., None], xy, 0).astype(np.float32)
        return self

    def save(self, path):
        self.validate()
        np.savez_compressed(path, timestamps=self.timestamps, xy_pixels=self.xy,
                            mask=self.mask, version=RAW_VERSION)


def load_json_clip(root, row, identity, fps):
    root = Path(root).resolve()
    folder = (root / row["keypoints_dir"]).resolve()
    folder.relative_to(root)
    if not folder.is_dir():
        raise ValueError(f"keypoint folder missing: {folder}")
    frames = []
    for path in folder.glob("*_keypoints.json"):
        match = FRAME.search(path.name)
        if not match or path.name[:match.start()] != identity:
            raise ValueError(f"frame filename differs from clip ID: {path.name}")
        frames.append((int(match[1]), path))
    frames.sort(key=lambda item: item[0])
    positions = np.asarray([n for n, _ in frames], dtype=np.int64)
    if len(frames) < 2 or np.any(np.diff(positions) <= 0):
        raise ValueError("at least two unique increasing frame numbers required")
    # Sequential filenames alone cannot reveal a truncated tail. Compare with
    # the matched label's video duration before accepting a training example.
    span_seconds = (int(positions[-1]) - int(positions[0]) + 1) / fps
    duration = row.get("source_duration_seconds")
    if duration is not None and abs(span_seconds - duration) > 1 / fps + .001:
        raise ValueError(f"keypoint time span {span_seconds:.3f}s differs from morpheme "
                         f"duration {duration:.3f}s; expected about {round(duration * fps)} "
                         f"frames, observed span {positions[-1] - positions[0] + 1}")
    segments = row.get("gloss_segments", [])
    if segments and max(s["end"] for s in segments) > span_seconds + 1 / fps + .001:
        raise ValueError("morpheme labels extend beyond available keypoint frames")
    coordinates, masks = [], []
    for _, path in frames:
        try:
            person = person_from_frame(read_json(path))
            if not isinstance(person, dict):
                raise ValueError("no person object")
            xy, mask = read_xy(person)
        except (ValueError, OSError) as error:
            raise ValueError(f"{path.name}: {error}") from error
        coordinates.append(xy)
        masks.append(mask)
    raw = RawXY((positions - positions[0]) / fps, np.asarray(coordinates), np.asarray(masks)).validate()
    gaps = [{"start": int(a+1), "end": int(b-1)} for a, b in zip(positions, positions[1:]) if b-a > 1]
    return raw, {"source_first_frame": int(positions[0]), "source_last_frame": int(positions[-1]),
                 "fps": fps, "quality": {"missing_frames": int(np.sum(np.diff(positions)-1)),
                                          "missing_frame_ranges": gaps,
                                          "frame_span_seconds": span_seconds,
                                          "valid_hand_point_fraction": float(raw.mask[:, :42].mean())}}


def normalized(raw, size, *, mirrored=False):
    size = image_size(size)
    raw.validate()
    points = raw.xy / np.asarray(size, np.float32)
    bounded = ((points >= 0) & (points <= 1)).all(axis=-1)
    mask = raw.mask & bounded
    if not mask[:, :42].any():
        raise ValueError("no valid hands within image_size; check actual width/height")
    if mirrored:
        points[..., 0] = 1 - points[..., 0]
    sequence = PoseSequence(raw.timestamps, np.where(mask[..., None], points, 0), mask, size).validate()
    return sequence, {"out_of_bounds_points": int((raw.mask & ~bounded).sum()),
                      "valid_hand_point_fraction": float(mask[:, :42].mean())}


def dataset_extra(source, raw=False):
    return {"extractor": {"name": "sen-xy-json", "mask_policy": MASK_POLICY},
            "sources": [source_entry("aihub-sign-video")],
            "label_alignment_verified": source.get("label_alignment_verified", False),
            "signer_identity_verified": source.get("signer_identity_verified", False),
            "label_matching": source.get("label_matching", "unverified annotation input"),
            "index_audit": source.get("index_audit", {}),
            "normalization_pending": raw,
            "training_scope": "raw cache; normalize with verified image_size" if raw else
                              "CTC input; verify signer identity before independent evaluation"}


def convert_annotations(root, annotations, out, *, raw=False, limit=0, skip_invalid=False, progress=None):
    source = dict(annotations) if isinstance(annotations, dict) else read_json(annotations)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be 0 (all) or a positive integer")
    rows = source["clips"][:limit] if limit else source["clips"]
    # Validate shared metadata before opening thousands of JSON files.
    for row in rows:
        if row.get("glosses") == []:
            raise ValueError(f"{row.get('id')}: SEN glosses must be a nonempty ordered list")
        positive_fps(row.get("fps", source.get("fps")))
        if not raw:
            image_size(row.get("image_size", source.get("image_size")))

    def load_clip(row, identity):
        fps = positive_fps(row.get("fps", source.get("fps")))
        sequence, fields = load_json_clip(root, row, identity, fps)
        if not raw:
            size = image_size(row.get("image_size", source.get("image_size")))
            sequence, quality = normalized(sequence, size, mirrored=row.get("input_is_mirrored", False))
            fields["quality"].update(quality)
            fields["image_size"] = list(size)
        return sequence, fields

    return write_dataset(rows, out, load_clip, version=RAW_VERSION if raw else VERSION,
                         drop=("keypoints_dir",), extra=dataset_extra(source, raw),
                         skip_invalid=skip_invalid, progress=progress)


def normalize_cache(manifest_path, out, size, *, limit=0, skip_invalid=False, progress=None):
    size = image_size(size)
    path = Path(manifest_path)
    if path.is_dir():
        path = path / "manifest.json"
    source = read_json(path)
    if source.get("version") != RAW_VERSION:
        raise ValueError(f"expected raw cache version {RAW_VERSION}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be 0 (all) or a positive integer")

    def load_clip(row, identity):
        cached = (path.parent / row["file"]).resolve()
        cached.relative_to(path.parent.resolve())
        with np.load(cached, allow_pickle=False) as data:
            if str(data["version"]) != RAW_VERSION:
                raise ValueError("cached clip version differs from manifest")
            raw = RawXY(data["timestamps"], data["xy_pixels"], data["mask"]).validate()
        sequence, quality = normalized(raw, size, mirrored=row.get("input_is_mirrored", False))
        return sequence, {"image_size": list(size), "quality": {**row.get("quality", {}), **quality}}

    rows = source["clips"][:limit] if limit else source["clips"]
    return write_dataset(rows, out, load_clip, version=VERSION,
                         extra={**dataset_extra(source),
                                "source_conversion_exclusions": source.get("excluded_from_conversion", [])},
                         skip_invalid=skip_invalid, progress=progress)


def package_colab(dataset, out):
    """Bundle code and already converted clips; Colab need not read frame JSON."""
    dataset, out = Path(dataset).resolve(), Path(out)
    manifest = read_json(dataset / "manifest.json")
    if manifest.get("version") != VERSION or manifest.get("normalization_pending", True):
        raise ValueError("Colab bundle requires normalized pose2d training data")
    if out.exists():
        raise ValueError("Bundle output exists; use a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=out.parent, prefix="sen-bundle-") as td:
        stage = Path(td) / "result"
        stage.mkdir()
        with zipfile.ZipFile(stage / "colab_code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("src", "tools"):
                for path in sorted((project / folder).glob("*.py")):
                    archive.write(path, path.relative_to(project).as_posix())
            archive.write(project / "config.py", "config.py")
        with zipfile.ZipFile(stage / "sen_training_data.zip", "w", zipfile.ZIP_STORED) as archive:
            archive.write(dataset / "manifest.json", "manifest.json")
            for row in manifest["clips"]:
                path = (dataset / row["file"]).resolve()
                path.relative_to(dataset)
                archive.write(path, row["file"])
        stage.rename(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="SEN root, or cached dataset with --from-cache")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--view", choices=[*"FDLRU", "all"], default="F")
    parser.add_argument("--limit", type=int, default=0, help="0: all clips (default)")
    parser.add_argument("--skip-invalid", action="store_true", help="record bad clips and continue")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--colab-out", type=Path, help="also create code/data ZIPs in a new directory")
    args = parser.parse_args()
    if args.colab_out and not args.from_cache and args.image_size is None:
        parser.error("--colab-out requires verified --image-size WIDTH HEIGHT")
    if args.out.exists():
        parser.error("Output exists; use a new --out directory")
    if args.colab_out and args.colab_out.exists():
        parser.error("Bundle output exists; use a new --colab-out directory")

    def progress(index, identity, accepted):
        if index == 1 or index % 25 == 0 or not accepted:
            print(f"[{index}] {'OK' if accepted else 'REJECTED'} {identity}", flush=True)

    if args.from_cache:
        if args.image_size is None:
            parser.error("--from-cache requires verified --image-size WIDTH HEIGHT")
        result = normalize_cache(args.root, args.out, args.image_size, limit=args.limit,
                                 skip_invalid=args.skip_invalid, progress=progress)
    else:
        source = official_annotations(args.root, view=args.view, size=args.image_size, fps=args.fps)
        print(f"Exact labelled clips: {len(source['clips'])}; label exclusions: "
              f"{len(source['index_audit']['excluded'])}", flush=True)
        result = convert_annotations(args.root, source, args.out, raw=args.image_size is None,
                                     limit=args.limit, skip_invalid=args.skip_invalid, progress=progress)
    print(json.dumps({"version": result["version"], **result["conversion_summary"],
                      "normalization_pending": result["normalization_pending"],
                      "manifest": str(args.out / "manifest.json")}, ensure_ascii=False, indent=2))
    if args.colab_out:
        print("Colab ZIPs:", package_colab(args.out, args.colab_out))


if __name__ == "__main__":
    main()
