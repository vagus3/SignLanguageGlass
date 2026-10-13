#!/usr/bin/env python
"""Convert annotated AI Hub/OpenPose clips into versioned pose sequences.

Input JSON: {"clips": [{"id":..., "keypoints_dir":..., "image_size":[w,h],
"fps":30, "signer_id":..., "session_id":..., "group_id":...,
"glosses":["label",...], "text":"Korean reference", "split":"train"}]}
glosses may be null, which makes the clip ineligible for CTC training.
--with-3d requires coordinate_frame_3d="camera" or "world" at the top level
or per clip. "world" also requires a 3x4 world_to_camera=[R|t] matrix,
with p_camera = R @ p_world + t. Camera axes: x right, y down, z away.
Raw camparam matrices are not interpreted without a verified convention.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_sources import SOURCES, source_entry
from src.joints import person_from_frame
from src.pose_sequence import (PoseSequence, VERSION, VERSION_3D, JOINT_NAMES, from_openpose,
                                world_from_openpose)


LIMITATIONS = ["2D viewpoint mismatch needs wearable evaluation",
               "Interlocutor face excluded; signer expression unavailable"]


def write_dataset(clips, out, load_clip, *, version, drop=(), extra=None):
    """Validate clip metadata, convert each clip and publish the dataset atomically.

    load_clip(row, identity) -> (PoseSequence, extra row fields). Shared by the
    OpenPose JSON conversion and video re-extraction so both keep the same ids,
    required fields and manifest layout.
    """
    out = Path(out)
    if out.exists():
        raise ValueError("Output exists; use a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix="pose-stage-") as td:
        stage = Path(td) / "result"
        stage.mkdir()
        rows, missing = [], []
        seen = set()
        for index, row in enumerate(clips):
            identity = row.get("id")
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError("Each clip needs a unique nonempty id")
            seen.add(identity)
            sequence, fields = load_clip(row, identity)
            filename = f"clip_{index:06d}.npz"
            sequence.save(stage / filename)
            for key in ("signer_id", "session_id", "group_id"):
                if not row.get(key):
                    raise ValueError(f"{identity}: missing {key}")
            rows.append({k: v for k, v in {**row, **fields, "file": filename,
                                         "frames": len(sequence.timestamps),
                                         "duration_seconds": float(sequence.timestamps[-1])}.items()
                         if k not in drop})
            if row.get("glosses") is None:
                missing.append({"id": identity, "reason": "ordered glosses unavailable"})
        if not rows:
            raise ValueError("No clips supplied")
        report = {"version": version, "joint_names": JOINT_NAMES, **(extra or {}), "clips": rows,
                  "excluded_from_ctc": missing, "deployment_ready": False,
                  "limitations": list(LIMITATIONS)}
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        stage.rename(out)
    return report


def prepare(annotation_path, out, with_world=False, source_id="aihub-sign-video"):
    annotation_path = Path(annotation_path)
    source = json.loads(annotation_path.read_text(encoding="utf-8"))

    def load_clip(row, identity):
        folder = annotation_path.parent / row["keypoints_dir"]
        size = row.get("image_size", source.get("image_size"))
        fps = row.get("fps", source.get("fps"))
        if size is None or fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{identity}: explicit image_size and fps required")
        frame_3d = row.get("coordinate_frame_3d", source.get("coordinate_frame_3d"))
        transform = row.get("world_to_camera", source.get("world_to_camera"))
        if with_world:
            if frame_3d not in ("camera", "world"):
                raise ValueError(f"{identity}: --with-3d requires coordinate_frame_3d "
                                 "('camera' or 'world'); do not assume AI Hub is camera-aligned")
            if frame_3d == "world" and transform is None:
                raise ValueError(f"{identity}: world coordinates require world_to_camera [R|t]")
            if frame_3d == "camera" and transform is not None:
                raise ValueError(f"{identity}: camera coordinates must not have world_to_camera")
        frames = []
        for p in folder.glob("*_keypoints.json"):
            match = re.search(r"_(\d+)_keypoints\.json$", p.name)
            if not match:
                raise ValueError(f"Invalid frame filename: {p}")
            frames.append((int(match[1]), p))
        frames.sort()
        if len(frames) < 2:
            raise ValueError(f"{identity}: fewer than two frames")
        coordinates, masks, worlds, world_masks = [], [], [], []
        for number, p in frames:
            person = person_from_frame(json.loads(p.read_text(encoding="utf-8")))
            if person is None:
                raise ValueError(f"{identity}: no person in frame {number}")
            points, mask = from_openpose(person, size,
                                         input_is_mirrored=row.get("input_is_mirrored", False))
            coordinates.append(points)
            masks.append(mask)
            if with_world:
                world, world_mask = world_from_openpose(
                    person, input_is_mirrored=row.get("input_is_mirrored", False),
                    world_to_camera=transform)
                worlds.append(world)
                world_masks.append(world_mask)
        positions = np.array([n for n, _ in frames])
        timestamps = (positions - positions[0]) / float(fps)
        if with_world and not np.any(world_masks):
            raise ValueError(f"{identity}: no *_keypoints_3d values for --with-3d")
        sequence = PoseSequence(timestamps, coordinates, masks, tuple(size),
                                world=worlds if with_world else None,
                                world_mask=world_masks if with_world else None).validate()
        provenance = ({"coordinate_frame_3d": "camera",
                       "source_coordinate_frame_3d": frame_3d,
                       "source_world_to_camera": transform} if with_world else {})
        return sequence, provenance

    return write_dataset(source["clips"], out, load_clip,
                         version=VERSION_3D if with_world else VERSION,
                         drop=("keypoints_dir", "world_to_camera"),
                         extra={"extractor": {"name": "openpose-json"},
                                "sources": [source_entry(source_id)]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations")
    parser.add_argument("--out", required=True)
    parser.add_argument("--with-3d", action="store_true",
                        help="also store *_keypoints_3d as pose3d-sequence-v2")
    parser.add_argument("--source", choices=sorted(SOURCES), default="aihub-sign-video",
                        help="data source recorded in the manifest and trained models")
    args = parser.parse_args()
    result = prepare(args.annotations, args.out, args.with_3d, args.source)
    print(f"{len(result['clips'])} clips; {len(result['excluded_from_ctc'])} without glosses")
