"""Video re-extraction keeps clip ids, resizes like the live camera and writes a trainable manifest."""
import _path  # noqa: F401
import json
import sys
import tempfile
from pathlib import Path
import numpy as np

try:
    import cv2
    import torch  # noqa: F401  (load_dataset)
except ImportError:
    sys.exit(77)
from src.continuous_sign import load_dataset
from src.pose_sequence import PoseSequence, VERSION, VERSION_3D
from tools.extract_mediapipe import prepare

SAMPLE = Path(__file__).resolve().parents[1] / (
    "data/raw/New_sample/원천데이터/REAL/WORD/01/NIA_SL_WORD1501_REAL01_F.mp4")


def video(path, frames, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    assert writer.isOpened(), "OpenCV cannot write mp4v test videos"
    rng = np.random.default_rng(frames)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (90, 160, 3), dtype=np.uint8))
    writer.release()


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        video(td / "a.mp4", 8, 30)
        video(td / "b.mp4", 10, 25)
        clips = [{"id": "a", "video": "a.mp4", "fps": 15, "signer_id": "s1", "session_id": "d",
                  "group_id": "a", "glosses": ["A", "B"], "continuous": True,
                  "image_size": [1920, 1080], "keypoints_dir": "unused"},
                 {"id": "b", "video": "b.mp4", "signer_id": "s2", "session_id": "d",
                  "group_id": "b", "glosses": None},
                 {"id": "c", "keypoints_dir": "only-json", "signer_id": "s3", "session_id": "d",
                  "group_id": "c", "glosses": ["A"]}]
        annotations = td / "annotations.json"
        annotations.write_text(json.dumps({"clips": clips}), encoding="utf-8")

        report = prepare(annotations, td / "mp", width=128)
        rows = {r["id"]: r for r in report["clips"]}
        assert report["version"] == VERSION and list(rows) == ["a", "b"], "Clips without video skip"
        assert report["extractor"]["name"] == "mediapipe-holistic"
        assert rows["a"]["image_size"] == [128, 72], "Resize keeps the aspect ratio"
        assert (rows["a"]["fps"], rows["a"]["fps_source"]) == (15.0, "annotation")
        assert (rows["b"]["fps"], rows["b"]["fps_source"]) == (25.0, "container")
        assert rows["a"]["frames"] == 8 and rows["b"]["frames"] == 10
        assert "video" not in rows["a"] and "keypoints_dir" not in rows["a"]
        assert set(rows["a"]["detection"]) == {"left_hand", "right_hand", "both_hands", "shoulders"}
        assert [m["id"] for m in report["excluded_from_ctc"]] == ["b"]
        a = PoseSequence.load(td / "mp" / rows["a"]["file"])
        assert a.coordinates.shape == (8, 49, 2) and a.image_size == (128, 72)
        assert np.allclose(a.timestamps, np.arange(8) / 15)
        trainable, skipped = load_dataset(td / "mp" / "manifest.json")
        assert [r["id"] for r in trainable] == ["a"] and len(skipped) == 1

        three_d = prepare(annotations, td / "mp3d", width=128, with_world=True, workers=2)
        assert three_d["version"] == VERSION_3D
        b3 = PoseSequence.load(td / "mp3d" / three_d["clips"][1]["file"])
        b2 = PoseSequence.load(td / "mp" / rows["b"]["file"])
        assert b3.world.shape == (10, 49, 3)
        assert np.array_equal(b3.coordinates, b2.coordinates), "Workers must not change results"

        for kwargs, why in (({"out": td / "mp"}, "existing output"),
                            ({"out": td / "none", "only": ["zzz"]}, "no selected clip")):
            try:
                prepare(annotations, width=128, **kwargs)
                raise AssertionError(f"Accepted {why}")
            except ValueError:
                pass
        clips[0]["video"] = "missing.mp4"
        annotations.write_text(json.dumps({"clips": clips}), encoding="utf-8")
        try:
            prepare(annotations, td / "broken", width=128)
            raise AssertionError("Accepted an unreadable video")
        except ValueError as error:
            assert str(error).startswith("a:"), error
        assert not (td / "broken").exists(), "A failed run must not leave a dataset"

        if SAMPLE.exists():
            clips = [{"id": "aihub", "video": str(SAMPLE), "signer_id": "REAL01",
                      "session_id": "sample", "group_id": "WORD1501", "glosses": ["WORD1501"]}]
            annotations.write_text(json.dumps({"clips": clips}), encoding="utf-8")
            real = prepare(annotations, td / "real")["clips"][0]
            assert real["frames"] == 139 and real["detection"]["both_hands"] > 0.9, real
    print("[OK] video re-extraction: ids, resize, fps source, 3D, workers, failures"
          + ("" if SAMPLE.exists() else " (AI Hub sample absent: real-video check skipped)"))


if __name__ == "__main__":
    main()
