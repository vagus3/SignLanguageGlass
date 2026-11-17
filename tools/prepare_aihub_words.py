#!/usr/bin/env python
"""AI Hub '수어 영상' 단어 묶음에서 고른 단어만 꺼내 좌표 시퀀스로 바꿉니다.

python tools/prepare_aihub_words.py \
    --morpheme .../01_real_word_morpheme.zip --keypoints .../01_real_word_keypoint.zip \
    --words data/processed/hospital_words.json --out data/processed/aihub_words/REAL01

압축을 풀지 않고 필요한 클립만 읽습니다 (묶음 하나가 11GB, 프레임 JSON 약 200만 개).
관절 묶음 하나에는 수어자 한 명이 들어 있으므로 묶음마다 --out 을 따로 줍니다.

--words 는 {"라벨": ["WORD1496", ...]} 입니다. 같은 이름에 단어 번호가 여럿이면
(동음이의어·변형) 번호마다 라벨을 나눠 적으세요. 형태소 파일의 이름만으로는 뜻을
구분할 수 없습니다.

결과 manifest 의 각 클립은 영상 전체(준비·복귀 동작 포함)이고, 수어 구간은
sign_intervals(초)에 있습니다. continuous=false 이므로 연속 수어 학습에는 쓰이지 않습니다.

문장 묶음은 --words 대신 --sentences 에 ["SEN0301", ...] 목록을 줍니다. 클립마다 형태소
파일의 단어열이 glosses 가 되고, sign_intervals 는 단어마다 하나씩입니다 (continuous=true).
"""
import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_sources import source_entry
from src.joints import person_from_frame
from src.pose_sequence import PoseSequence, VERSION, from_openpose
from tools.prepare_continuous import write_dataset

CLIP = re.compile(r"(NIA_SL_((?:WORD|SEN)\d+)_(REAL\d+)_([A-Z]))")
FRAME = re.compile(r"_(\d+)_keypoints\.json$")
IMAGE_SIZE, FPS = (1920, 1080), 30


def read_intervals(morpheme_zip, wanted):
    """{clip id: [[start, end], ...]} and {clip id: [gloss, ...]} for the wanted numbers."""
    intervals, names = {}, {}
    with zipfile.ZipFile(morpheme_zip) as z:
        for name in z.namelist():
            match = CLIP.search(name)
            if not match or match[2] not in wanted or not name.endswith("_morpheme.json"):
                continue
            data = sorted(json.loads(z.read(name)).get("data", []),
                          key=lambda s: float(s["start"]))
            intervals[match[1]] = [[float(s["start"]), float(s["end"])] for s in data]
            names[match[1]] = [(s.get("attributes") or [{}])[0].get("name") for s in data]
    return intervals, names


def frame_index(keypoint_zip, wanted):
    """{clip id: [(frame number, member name), ...]} sorted by frame."""
    clips = defaultdict(list)
    for name in keypoint_zip.namelist():
        frame = FRAME.search(name)
        match = CLIP.search(name) if frame else None
        if match and match[2] in wanted:
            clips[match[1]].append((int(frame[1]), name))
    return {clip: sorted(frames) for clip, frames in clips.items()}


def prepare(morpheme_zip, keypoint_zip, words, out, session_id=None):
    """words: {"라벨": [단어 번호]} 이면 단어 묶음, [문장 번호] 목록이면 문장 묶음."""
    sentences = isinstance(words, list)
    session_id = session_id or ("aihub-sentence-train" if sentences else "aihub-word-train")
    label_of = {}
    for label, numbers in ({None: words} if sentences else words).items():
        for number in numbers:
            if number in label_of:
                raise ValueError(f"{number} is listed twice")
            label_of[number] = label
    intervals, names = read_intervals(morpheme_zip, label_of)
    skipped, rows = [], []
    with zipfile.ZipFile(keypoint_zip) as z:
        frames = frame_index(z, label_of)
        for clip in sorted(frames):
            _, number, signer, view = CLIP.search(clip).groups()
            if not intervals.get(clip):
                skipped.append({"id": clip, "reason": "수어 구간 주석 없음"})
                continue
            if sentences and not all(names[clip]):
                skipped.append({"id": clip, "reason": "이름 없는 수어 구간"})
                continue
            rows.append({"id": clip, "signer_id": signer, "session_id": session_id,
                         "group_id": f"{number}_{signer}",
                         "glosses": names[clip] if sentences else [label_of[number]],
                         "aihub_word": number, "view": view, "continuous": sentences,
                         "sign_intervals": intervals[clip]})

        def load_clip(row, identity):
            coordinates, masks, numbers = [], [], []
            for number, member in frames[identity]:
                person = person_from_frame(json.loads(z.read(member)))
                if person is None:      # 사람이 안 잡힌 프레임은 빈 프레임으로 둡니다
                    person = {}
                points, mask = from_openpose(person, IMAGE_SIZE)
                coordinates.append(points)
                masks.append(mask)
                numbers.append(number)
            positions = np.array(numbers)
            return PoseSequence((positions - positions[0]) / float(FPS), coordinates, masks,
                                IMAGE_SIZE).validate(), {"image_size": list(IMAGE_SIZE),
                                                         "fps": FPS}

        usable = []
        for row in rows:
            if len(frames[row["id"]]) < 2:
                skipped.append({"id": row["id"], "reason": "프레임 2개 미만"})
            else:
                usable.append(row)
        report = write_dataset(usable, out, load_clip, version=VERSION,
                               extra={"extractor": {"name": "openpose-json"},
                                      "sources": [source_entry("aihub-sign-video")],
                                      "skipped": skipped})
    missing = sorted(set(label_of) - {r["aihub_word"] for r in usable})
    return report, skipped, missing


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--morpheme", required=True, help="단어 형태소 zip")
    p.add_argument("--keypoints", required=True, help="단어 관절 zip (수어자 한 명)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--words", help='{"라벨": ["WORD0001", ...]} JSON')
    group.add_argument("--sentences", help='["SEN0301", ...] JSON (문장 묶음)')
    p.add_argument("--out", required=True, help="새로 만들 결과 폴더")
    args = p.parse_args()
    words = json.loads(Path(args.words or args.sentences).read_text(encoding="utf-8"))
    if isinstance(words, list) != bool(args.sentences):
        p.error("--words 는 {라벨: [번호]}, --sentences 는 [번호] 목록이어야 합니다")
    try:
        report, skipped, missing = prepare(args.morpheme, args.keypoints, words, args.out)
    except ValueError as error:
        p.error(str(error))
    signers = sorted({r["signer_id"] for r in report["clips"]})
    print(f"{len(report['clips'])}클립, 수어자 {signers}, 라벨 "
          f"{len({g for r in report['clips'] for g in r['glosses']})}개 -> {args.out}")
    if skipped:
        print(f"건너뜀 {len(skipped)}클립: {skipped[:3]}")
    if missing:
        print(f"이 묶음에 없는 단어 번호: {missing}")


if __name__ == "__main__":
    main()
