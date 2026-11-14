#!/usr/bin/env python
"""SEN DOCX 라벨 내보내기와 Colab용 2D 실험 데이터 변환.

REAL 형태소 요약 문서를 SYN에 SEN 번호로 대응하므로 정답 대응은 미검증입니다.
여러 후보가 있는 SEN은 제외합니다. 이 출력은 학습 실행 확인용입니다.
클립별 원본 morpheme JSON이 있으면 tools/prepare_sen.py를 사용합니다.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLIP = re.compile(r"^(NIA_SL_(SEN\d+)_(SYN\d+|REAL\d+))_([FDLRU])$")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def read_labels(path):
    """DOCX의 SEN | morpheme 표를 읽고 대안 후보를 각각 보존합니다."""
    with zipfile.ZipFile(path) as document:
        root = ET.fromstring(document.read("word/document.xml"))
    labels = {}
    for row in root.findall(".//w:tr", NS):
        cells = ["".join(t.text or "" for t in cell.findall(".//w:t", NS)).strip()
                 for cell in row.findall("w:tc", NS)]
        if len(cells) < 2 or not re.fullmatch(r"SEN\d+", cells[0]):
            continue
        variants = []
        for alternative in cells[1].split("/"):
            glosses = [g.strip() for g in alternative.split("+")]
            if not glosses or any(not g for g in glosses):
                raise ValueError(f"빈 형태소 라벨: {cells[0]}")
            if glosses not in variants:
                variants.append(glosses)
        if cells[0] in labels and labels[cells[0]] != variants:
            raise ValueError(f"중복 SEN의 라벨이 다릅니다: {cells[0]}")
        labels[cells[0]] = variants
    if not labels:
        raise ValueError("SEN 번호와 형태소 단어열이 있는 표를 찾지 못했습니다")
    return labels


def clip_folders(root):
    """프레임 파일을 모두 열거하지 않고 클립 폴더를 찾습니다."""
    pending, found = [Path(root)], []
    while pending:
        directory = pending.pop()
        if CLIP.fullmatch(directory.name):
            found.append(directory)
            continue
        with os.scandir(directory) as entries:
            pending.extend(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False))
    return sorted(found)


def export_index(root, docx, out, view="F"):
    root, out = Path(root).resolve(), Path(out)
    labels = read_labels(docx)
    folders = clip_folders(root)
    if not folders:
        raise ValueError("SEN 클립 폴더를 찾지 못했습니다")
    rows, excluded = [], []
    for folder in folders:
        match = CLIP.fullmatch(folder.name)
        if match[4] != view:
            continue
        variants = labels.get(match[2], [])
        if len(variants) != 1:
            excluded.append({"id": folder.name, "sentence_id": match[2],
                             "reason": "ambiguous" if variants else "missing", "variants": variants})
            continue
        rows.append({
            "id": folder.name, "keypoints_dir": folder.relative_to(root).as_posix(),
            "sentence_id": match[2], "source_token": match[3],
            "signer_id": f"unverified-source-token:{match[3]}",
            "session_id": f"aihub-import:{match[3]}", "group_id": match[1],
            "continuous": True, "glosses": variants[0], "text": None,
        })
    if not rows:
        raise ValueError("선택한 시점에 단일 라벨을 가진 클립이 없습니다")
    annotations = {"image_size": None, "fps": 30, "clips": rows}
    report = {
        "clip_folders": len(folders), "selected_view": view,
        "selected_clips": len(rows), "excluded": excluded,
        "sentence_ids_in_document": len(labels),
        "source_tokens": sorted({CLIP.fullmatch(p.name)[3] for p in folders}),
        "label_alignment_verified": False, "signer_identity_verified": False,
        "label_source": Path(docx).name,
        "label_matching": "SEN ID lookup; REAL morpheme summary applied to SYN requires verification",
        "fps_source": "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103 (30fps keypoints)",
        "missing_metadata": ["image_size"], "deployment_ready": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, content in [("sen_labels.json", labels), ("annotations.json", annotations),
                          ("data_audit.json", report)]:
        (out / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_xy(root, annotations, out, *, limit=20, skip_invalid=False, progress=None):
    """실험용 마스크: 유한하고 (0,0)이 아닌 x/y만 사용. 세 번째 값은 사용하지 않음.

    SYN의 세 번째 값 의미를 confidence로 가정하지 않습니다. 이 마스크는 가림이나
    검출 신뢰도를 나타내지 않으며, 실사용·정식 평가는 검증된 원본 라벨/좌표 규격이 필요합니다.
    """
    from tools.prepare_sen import convert_annotations
    return convert_annotations(root, annotations, out, limit=limit,
                               skip_invalid=skip_invalid, progress=progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("docx", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--view", choices=list("FDLRU"), default="F")
    args = parser.parse_args()
    report = export_index(args.root, args.docx, args.out, args.view)
    print(json.dumps({k: v for k, v in report.items() if k != "excluded"}, ensure_ascii=False, indent=2))
    print(f"제외 클립: {len(report['excluded'])}; 상세: {args.out / 'data_audit.json'}")


if __name__ == "__main__":
    main()
