#!/usr/bin/env python
"""Re-extract pose sequences from source videos with the live MediaPipe setup.

python tools/extract_mediapipe.py annotations.json --out data/processed/mp-subset --limit 20
python tools/extract_mediapipe.py annotations.json --out data/processed/mp-3d --with-3d --workers 4

annotations.json is the prepare_continuous format with a "video" path per clip
(relative to the file); clips without one are skipped. Keeping the same ids and
splits as the OpenPose-JSON dataset makes the two comparable
(tools/compare_extractors.py) and trainable with the same split.

Each clip gets a fresh Holistic so tracking never carries over between clips.
Frames are resized to the live camera width (config.CAM_W) keeping the aspect
ratio. Timestamps are frame index / fps, using the annotation fps if given and
the container fps otherwise. --with-3d stores world_from_mediapipe points
(image-scale hands, pose_world body), the same as the live --pose-3d path.
"""
import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.pose_sequence import (PoseSequence, VERSION, VERSION_3D, from_mediapipe,
                                world_from_mediapipe)
from src.data_sources import SOURCES, source_entry
from tools.prepare_continuous import write_dataset


def extract_clip(video, *, width, fps=None, with_world=False, input_is_mirrored=False):
    """Run a fresh Holistic over every decodable frame of one video."""
    import cv2
    from src.landmarks import make_holistic

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    rate = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(rate) or rate <= 0:
        cap.release()
        raise ValueError(f"{video}: unknown fps; give fps in the annotations")
    holistic = make_holistic(C.MODEL_COMPLEXITY)
    coordinates, masks, worlds, world_masks, size = [], [], [], [], None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if size is None:
                h, w = frame.shape[:2]
                size = (int(width), max(1, round(h * width / w)))
                shrink = width < w
            if frame.shape[1::-1] != size:
                frame = cv2.resize(frame, size,
                                   interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            points, mask = from_mediapipe(res, input_is_mirrored=input_is_mirrored)
            coordinates.append(points)
            masks.append(mask)
            if with_world:
                world, world_mask = world_from_mediapipe(res, size,
                                                         input_is_mirrored=input_is_mirrored)
                worlds.append(world)
                world_masks.append(world_mask)
    finally:
        cap.release()
        holistic.close()
    if len(coordinates) < 2:
        raise ValueError(f"{video}: fewer than two decodable frames")
    m = np.asarray(masks)
    return {"coordinates": np.asarray(coordinates, np.float32), "mask": m,
            "image_size": size, "fps": rate, "fps_source": "annotation" if fps else "container",
            "world": np.asarray(worlds, np.float32) if with_world else None,
            "world_mask": np.asarray(world_masks) if with_world else None,
            "detection": {"left_hand": float(m[:, 0].mean()), "right_hand": float(m[:, 21].mean()),
                          "both_hands": float((m[:, 0] & m[:, 21]).mean()),
                          "shoulders": float((m[:, 43] & m[:, 44]).mean())}}


def _run(job):
    video, options = job
    return extract_clip(video, **options)


def prepare(annotation_path, out, *, with_world=False, limit=None, only=None, workers=1,
            width=None, source_id="aihub-sign-video"):
    annotation_path = Path(annotation_path)
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    clips = [row for row in source["clips"] if row.get("video")]
    if only:
        wanted = set(only)
        clips = [row for row in clips if row.get("id") in wanted]
    if limit is not None:
        clips = clips[:limit]
    if not clips:
        raise ValueError("No clips with a video path were selected")
    width = int(width or C.CAM_W)
    jobs = [(annotation_path.parent / row["video"],
             {"width": width, "fps": row.get("fps", source.get("fps")), "with_world": with_world,
              "input_is_mirrored": row.get("input_is_mirrored", False)}) for row in clips]
    # spawn, not Linux's default fork: a forked child inherits MediaPipe/TFLite worker threads'
    # locked allocator state and crashes ("malloc(): unaligned tcache chunk"). macOS already spawns.
    pool = (ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
            if workers > 1 else None)
    try:
        # Results arrive in clip order; workers stay ahead of the writer.
        results = iter(pool.map(_run, jobs) if pool else map(_run, jobs))

        def load_clip(row, identity):
            try:
                r = next(results)
            except ValueError as error:
                raise ValueError(f"{identity}: {error}") from error
            timestamps = np.arange(len(r["coordinates"])) / r["fps"]
            sequence = PoseSequence(timestamps, r["coordinates"], r["mask"], r["image_size"],
                                    world=r["world"], world_mask=r["world_mask"]).validate()
            fields = {"image_size": list(r["image_size"]), "fps": r["fps"],
                      "fps_source": r["fps_source"], "detection": r["detection"]}
            if with_world:
                fields["world_source"] = "mediapipe: image-scale hands, pose_world body"
            return sequence, fields

        import mediapipe as mp
        return write_dataset(
            clips, out, load_clip, version=VERSION_3D if with_world else VERSION,
            drop=("video", "keypoints_dir", "world_to_camera", "coordinate_frame_3d"),
            extra={"extractor": {"name": "mediapipe-holistic", "mediapipe": mp.__version__,
                                 "model_complexity": C.MODEL_COMPLEXITY, "width": width,
                                 "fresh_tracker_per_clip": True},
                   "sources": [source_entry(source_id)]})
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("annotations")
    p.add_argument("--out", required=True)
    p.add_argument("--with-3d", action="store_true")
    p.add_argument("--limit", type=int, help="first N clips that have a video")
    p.add_argument("--only", nargs="+", metavar="ID", help="extract only these clip ids")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--width", type=int, default=C.CAM_W, help="resize width (live camera width)")
    p.add_argument("--source", choices=sorted(SOURCES), default="aihub-sign-video",
                   help="data source recorded in the manifest and trained models")
    args = p.parse_args()
    if args.workers < 1 or args.width < 16 or (args.limit is not None and args.limit < 1):
        p.error("workers, width and limit must be positive")
    try:
        report = prepare(args.annotations, args.out, with_world=args.with_3d, limit=args.limit,
                         only=args.only, workers=args.workers, width=args.width,
                         source_id=args.source)
    except ValueError as error:
        p.error(str(error))
    rates = {k: np.mean([c["detection"][k] for c in report["clips"]])
             for k in ("left_hand", "right_hand", "both_hands", "shoulders")}
    print(f"{len(report['clips'])} clips -> {args.out}; mean detection "
          + ", ".join(f"{k} {v:.0%}" for k, v in rates.items()))


if __name__ == "__main__":
    main()
