"""Exact SYN labels, real frame timing, raw-cache parity and corrupt clip handling."""
import _path  # noqa: F401
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from src.pose_sequence import PoseSequence, features, FEATURE_DIM, VERSION
from tools.prepare_sen import (RAW_VERSION, official_annotations, convert_annotations,
                               normalize_cache, read_morphemes, package_colab)
from tools.prepare_continuous import write_dataset


def fails(action, message):
    try:
        action()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected failure: {message}")


def label(root, identity, segments=None):
    path = root / "SEN/morpheme/05" / f"{identity}_morpheme.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if segments is None:
        # Deliberately reverse time order and preserve a repeated target token.
        segments = [{"start": .15, "end": .2, "attributes": [{"name": "B"}]},
                    {"start": 0, "end": .02, "attributes": [{"name": "A", "attribute": ["question"]}]},
                    {"start": .05, "end": .1, "attributes": [{"name": "A"}]}]
    path.write_text(json.dumps({"metaData": {"name": identity + ".mp4", "duration": 10/30},
                                "data": segments}), encoding="utf-8-sig")
    return path


def clip(root, identity, positions=(1, 2, 3, 10), missing=False):
    folder = root / "SEN/keypoint/05" / identity
    folder.mkdir(parents=True)
    for index in positions:
        hand = np.c_[np.arange(21) + 100 + index, np.arange(21) + 200,
                     np.full(21, 1.7)]
        hand[0] = [0, 0, 9]
        hand[1, 2] = 0  # Never treat the third SYN value as a confidence filter.
        hand[2, 0] = float("nan")  # Must be masked, without contaminating features.
        person = {"hand_left_keypoints_2d": hand.ravel().tolist(),
                  "hand_right_keypoints_2d": [] if missing else hand.tolist()}
        if not missing:
            person["pose_keypoints_2d"] = np.tile([150, 180, 2.5], (25, 1)).ravel().tolist()
        (folder / f"{identity}_{index}_keypoints.json").write_text(
            json.dumps({"people": person}), encoding="utf-8-sig")
    return folder


with tempfile.TemporaryDirectory() as td:
    base = Path(td)
    raw = base / "source"
    identity = "NIA_SL_SEN0009_SYN05_F"
    folder = clip(raw, identity)
    label_path = label(raw, identity)
    # Same SEN, other source/view: a label is never borrowed by SEN number.
    unlabelled = "NIA_SL_SEN0009_SYN06_F"
    clip(raw, unlabelled)
    clip(raw, "NIA_SL_SEN0009_SYN05_L")
    label(raw, "NIA_SL_SEN0009_SYN05_L")
    source = official_annotations(raw, size=[640, 480])
    assert source["label_alignment_verified"] and not source["signer_identity_verified"]
    assert source["index_audit"]["labelled_clips"] == 1
    assert source["index_audit"]["excluded"][0]["id"] == unlabelled
    row = source["clips"][0]
    assert row["glosses"] == ["A", "A", "B"]
    assert row["gloss_segments"][0]["attributes"]["attribute"] == ["question"]
    assert row["group_id"] == "NIA_SL_SEN0009_SYN05"
    all_views = official_annotations(raw, view="all", size=[640, 480])
    assert {r["group_id"] for r in all_views["clips"]} == {row["group_id"]}

    direct = convert_annotations(raw, source, base / "direct")
    assert direct["version"] == VERSION and not direct["normalization_pending"]
    assert direct["conversion_summary"] == {"selected": 1, "converted": 1, "rejected": 0, "frames": 4}
    sequence = PoseSequence.load(base / "direct" / direct["clips"][0]["file"])
    assert np.allclose(sequence.timestamps, [0, 1/30, 2/30, 9/30])
    assert np.isclose(sequence.coordinates[0, 1, 0], 102/640)
    assert not sequence.mask[:, 0].any() and not sequence.mask[:, 2].any()
    assert sequence.mask[:, 1].all()
    assert direct["clips"][0]["quality"]["missing_frames"] == 6
    assert direct["clips"][0]["quality"]["missing_frame_ranges"] == [{"start": 4, "end": 9}]
    x = features(sequence)
    assert x.shape == (4, FEATURE_DIM) and np.isfinite(x).all()

    # Pixel cache works without resolution and later reproduces direct conversion.
    source["image_size"] = None
    packed = convert_annotations(raw, source, base / "packed", raw=True)
    assert packed["version"] == RAW_VERSION and packed["normalization_pending"]
    restored = normalize_cache(base / "packed", base / "restored", [640, 480])
    cached = PoseSequence.load(base / "restored" / restored["clips"][0]["file"])
    assert np.array_equal(sequence.coordinates, cached.coordinates)
    assert np.array_equal(sequence.mask, cached.mask)
    assert np.array_equal(sequence.timestamps, cached.timestamps)
    assert restored["label_alignment_verified"]
    # Shared writer must retain background/blank and text-only training semantics.
    shared = write_dataset([{**row, "id": "blank", "glosses": [], "negative": True},
                            {**row, "id": "text-only", "glosses": None}],
                           base / "shared", lambda row, identity: (sequence, {}), version=VERSION)
    assert len(shared["clips"]) == 2 and len(shared["excluded_from_ctc"]) == 1
    bundle = package_colab(base / "direct", base / "bundle")
    import zipfile
    with zipfile.ZipFile(bundle / "sen_training_data.zip") as z:
        assert set(z.namelist()) == {"manifest.json", direct["clips"][0]["file"]}
        assert z.testzip() is None
    with zipfile.ZipFile(bundle / "colab_code.zip") as z:
        assert "tools/prepare_sen.py" in z.namelist() and "src/train_continuous.py" in z.namelist()
    fails(lambda: package_colab(base / "packed", base / "raw-bundle"), "normalized")
    # Execute the notebook's actual preparation cells for both supported inputs.
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "notebooks/aihub_sen_colab.ipynb").read_text(encoding="utf-8"))
    for prepared in (True, False):
        work = base / f"notebook-{prepared}"
        work.mkdir()
        namespace = {"json": json, "Path": Path, "shutil": shutil, "WORK": work,
                     "USE_PREPARED_DATA": prepared, "IMAGE_SIZE": [640, 480], "FPS": 30,
                     "MAX_CLIPS": 1, "DATA_MANIFEST": base / "direct/manifest.json",
                     "DATA_ROOT": raw}
        for index in (8, 10):
            exec(compile("".join(notebook["cells"][index]["source"]), f"notebook-{index}", "exec"), namespace)
        ready = json.loads((work / "processed/manifest.json").read_text(encoding="utf-8"))
        assert len(ready["clips"]) == 1 and ready["label_alignment_verified"]
        PoseSequence.load(work / "processed" / ready["clips"][0]["file"])
    fails(lambda: convert_annotations(raw, source, base / "no-size"), "image_size")
    assert not (base / "no-size").exists()
    for size in ([True, 480], [640.0, 480], [0, 480], [640, float("nan")]):
        fails(lambda: normalize_cache(base / "packed", base / "bad-size", size), "image_size")
    fails(lambda: normalize_cache(base / "direct", base / "not-raw", [640, 480]), "raw cache")
    fails(lambda: convert_annotations(raw, source, base / "packed", raw=True), "Output exists")

    source["image_size"] = [640, 480]
    # A genuinely absent hand or body is a mask, not a malformed clip.
    partial = "NIA_SL_SEN0010_SYN05_F"
    clip(raw, partial, missing=True)
    label(raw, partial)
    partial_source = official_annotations(raw, size=[640, 480])
    result = convert_annotations(raw, partial_source, base / "partial")
    partial_row = next(r for r in result["clips"] if r["id"] == partial)
    s = PoseSequence.load(base / "partial" / partial_row["file"])
    assert s.mask[:, :21].any() and not s.mask[:, 21:].any()

    # One corrupt clip aborts atomically by default; opt-in skipping leaves an audit.
    broken = "NIA_SL_SEN0011_SYN05_F"
    bad_folder = clip(raw, broken)
    label(raw, broken)
    next(bad_folder.glob("*.json")).write_text("{broken", encoding="utf-8")
    broken_source = official_annotations(raw, size=[640, 480])
    fails(lambda: convert_annotations(raw, broken_source, base / "strict"), broken)
    assert not (base / "strict").exists()
    result = convert_annotations(raw, broken_source, base / "skip", skip_invalid=True)
    assert result["conversion_summary"]["converted"] == 2
    assert result["conversion_summary"]["rejected"] == 1
    assert result["excluded_from_conversion"][0]["id"] == broken
    assert "keypoints.json" in result["excluded_from_conversion"][0]["reason"]

    # Sequential but truncated clips must not silently receive the full label.
    truncated = "NIA_SL_SEN0012_SYN05_F"
    clip(raw, truncated, positions=(0, 1, 2, 3))
    label(raw, truncated)
    truncated_source = official_annotations(raw, size=[640, 480])
    result = convert_annotations(raw, truncated_source, base / "truncated", skip_invalid=True)
    rejected = next(r for r in result["excluded_from_conversion"] if r["id"] == truncated)
    assert "time span" in rejected["reason"] and "duration" in rejected["reason"]

    # Numeric ordering, duplicated frame numbers and path containment.
    duplicate = folder / f"{identity}_01_keypoints.json"
    duplicate.write_bytes(next(folder.glob("*_1_keypoints.json")).read_bytes())
    fails(lambda: convert_annotations(raw, source, base / "duplicate"), "unique increasing")
    duplicate.unlink()
    source["clips"][0]["keypoints_dir"] = "../outside"
    fails(lambda: convert_annotations(raw, source, base / "escape"), identity)
    source["clips"][0]["keypoints_dir"] = folder.relative_to(raw).as_posix()
    # CTC needs an extra blank frame between adjacent equal labels.
    source["clips"][0]["glosses"] = ["A"] * 3
    fails(lambda: convert_annotations(raw, source, base / "short-target"), "too few frames for CTC")
    source["clips"][0]["glosses"] = []
    fails(lambda: convert_annotations(raw, source, base / "empty-target"), "nonempty ordered list")

    # Exact metadata identity and temporal ambiguity are rejected at indexing.
    node = json.loads(label_path.read_text(encoding="utf-8-sig"))
    node["metaData"]["name"] = "NIA_SL_SEN0009_REAL05_F.mp4"
    label_path.write_text(json.dumps(node), encoding="utf-8")
    fails(lambda: read_morphemes(label_path, identity), "differs")
    label(label_path.parents[3], identity)  # Restore the exact identity.
    node["metaData"]["name"] = identity + ".mp4"
    node["data"][0]["start"] = .01
    label_path.write_text(json.dumps(node), encoding="utf-8")
    fails(lambda: read_morphemes(label_path, identity), "overlapping")

print("SEN exact labels, temporal order, masks, cache parity, CTC and atomic rejection: PASS")
