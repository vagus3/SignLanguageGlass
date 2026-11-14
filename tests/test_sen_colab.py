"""SEN 라벨 대안 보존과 SYN x/y 실험 변환의 입력 계약을 검증합니다."""
import _path  # noqa: F401
import ast
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from tools.prepare_sen_colab import read_labels, export_index, prepare_xy
from src.pose_sequence import PoseSequence, VERSION


def docx(path, records):
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    root = ET.Element(f"{{{namespace}}}document")
    table = ET.SubElement(root, f"{{{namespace}}}tbl")
    for identity, text in records:
        row = ET.SubElement(table, f"{{{namespace}}}tr")
        for value in (identity, text):
            cell = ET.SubElement(row, f"{{{namespace}}}tc")
            p = ET.SubElement(cell, f"{{{namespace}}}p")
            r = ET.SubElement(p, f"{{{namespace}}}r")
            ET.SubElement(r, f"{{{namespace}}}t").text = value
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", ET.tostring(root))


with tempfile.TemporaryDirectory() as td:
    base = Path(td)
    labels = base / "labels.docx"
    docx(labels, [("SEN number", "morphemes"), ("SEN0005", "HERE"),
                  ("SEN0009", "DRIVE + SLOW / DRIVE + GENTLY")])
    parsed = read_labels(labels)
    assert parsed["SEN0009"] == [["DRIVE", "SLOW"], ["DRIVE", "GENTLY"]]
    assert parsed["SEN0005"] == [["HERE"]]
    conflict = base / "conflict.docx"
    docx(conflict, [("SEN0005", "HERE"), ("SEN0005", "THERE")])
    try:
        read_labels(conflict)
        raise AssertionError("Conflicting duplicate SEN should fail")
    except ValueError:
        pass

    raw = base / "raw"
    names = ["NIA_SL_SEN0005_SYN05_F", "NIA_SL_SEN0005_SYN05_L",
             "NIA_SL_SEN0009_SYN05_F", "NIA_SL_SEN0999_SYN05_F"]
    for name in names:
        (raw / "SEN" / "keypoint" / "05" / name).mkdir(parents=True)
    folder = raw / "SEN" / "keypoint" / "05" / names[0]
    hand = np.c_[np.arange(21) + 100, np.full(21, 100), np.full(21, 1.7)]
    hand[0] = [0, 0, 7]
    hand[1, 2] = 0  # Third value is deliberately NOT used as confidence.
    pose = np.c_[np.arange(25) + 100, np.full(25, 200), np.full(25, 2.5)]
    person = {"hand_left_keypoints_2d": hand.ravel().tolist(),
              "hand_right_keypoints_2d": hand.ravel().tolist(),
              "pose_keypoints_2d": pose.ravel().tolist()}
    for index in (0, 2):
        (folder / f"{folder.name}_{index:012d}_keypoints.json").write_text(
            json.dumps({"people": person}), encoding="utf-8")
    index_dir = base / "index"
    audit = export_index(raw, labels, index_dir)
    assert audit["selected_clips"] == 1 and audit["clip_folders"] == 4
    assert {r["reason"] for r in audit["excluded"]} == {"missing", "ambiguous"}
    assert not audit["label_alignment_verified"] and not audit["signer_identity_verified"]
    annotation = index_dir / "annotations.json"
    source = json.loads(annotation.read_text(encoding="utf-8"))
    assert source["image_size"] is None
    assert source["clips"][0]["group_id"] == "NIA_SL_SEN0005_SYN05"
    try:
        prepare_xy(raw, annotation, base / "must-not-exist")
        raise AssertionError("Missing image size must prevent conversion")
    except ValueError:
        assert not (base / "must-not-exist").exists()
    source["image_size"] = [640, 480]  # Synthetic test fixture dimensions.
    annotation.write_text(json.dumps(source), encoding="utf-8")
    result = prepare_xy(raw, annotation, base / "processed")
    assert result["version"] == VERSION and not result["label_alignment_verified"]
    sequence = PoseSequence.load(base / "processed" / result["clips"][0]["file"])
    assert sequence.coordinates.shape == (2, 49, 2)
    assert np.isclose(sequence.timestamps[-1], 2 / 30)
    assert not sequence.mask[0, 0] and sequence.mask[0, 1]
    assert np.allclose(sequence.coordinates[0, 1], [101 / 640, 100 / 480])
    assert result["sources"][0]["id"] == "aihub-sign-video"
    assert result["clips"][0]["glosses"] == ["HERE"]
    try:
        prepare_xy(raw, annotation, base / "processed")
        raise AssertionError("Existing output must not be overwritten")
    except ValueError:
        pass

notebook = Path(__file__).resolve().parents[1] / "notebooks" / "aihub_sen_colab.ipynb"
data = json.loads(notebook.read_text(encoding="utf-8"))
assert data["nbformat"] == 4
for index, cell in enumerate(data["cells"]):
    if cell["cell_type"] == "code":
        assert cell["execution_count"] is None and cell["outputs"] == []
        ast.parse("".join(cell["source"]), filename=f"colab-cell-{index}")
print("SEN alternatives, exclusions, metadata guard, XY mask, frame timing and notebook syntax: PASS")
