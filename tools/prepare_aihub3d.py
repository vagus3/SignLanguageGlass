#!/usr/bin/env python
"""보유 AI Hub WORD 샘플 → 3D 손 모양 학습 입력 + 원본 동작 그룹 감사.

python tools/prepare_aihub3d.py data/raw/New_sample --out data/processed/aihub3d

원시 3D 좌표도 함께 보관해 추후 방향/동작 분기를 만들 수 있게 합니다.
동일 동작의 다른 시점은 항상 동일 group_id. 무작위 시점 분할 금지.
단어 뜻 매핑이 없으면 WORD ID 자체를 라벨로 사용합니다.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features_v2 import person_from_frame
from src.hand_shape3d import (FEATURE_DIM, FEATURE_VERSION, from_raw_hands,
                             raw_openpose_hands, resample_descriptors)

CLIP = re.compile(r"^(NIA_SL_(WORD\d+)_([^_]+))_([A-Z])$")
FRAME = re.compile(r"_(\d+)_keypoints\.json$")


def load_clip(folder):
    paths = sorted(folder.glob('*_keypoints.json'))
    indices, raw, camera = [], [], None
    for path in paths:
        match = FRAME.search(path.name)
        if not match:
            raise ValueError(f"프레임 번호 형식 오류: {path}")
        node = json.loads(path.read_text(encoding='utf-8'))
        person = person_from_frame(node)
        if person is None:
            raise ValueError(f"person 없음: {path}")
        indices.append(int(match[1]))
        raw.append(raw_openpose_hands(person))
        if camera is None:
            camera = node.get('camparam')
    if not raw or np.any(np.diff(indices) <= 0):
        raise ValueError(f"비어 있거나 프레임 번호가 중복된 클립: {folder}")
    raw = np.stack(raw)
    if not np.isfinite(raw).all():
        raise ValueError(f"NaN/Inf 3D 좌표: {folder}")
    digest = hashlib.sha256(np.asarray(indices, '<i8').tobytes()
                            + raw.astype('<f8').tobytes()).hexdigest()
    return np.asarray(indices), raw, camera, digest


def prepare(root, out):
    root, out = Path(root), Path(out)
    # 기존 변환 결과를 덮지 않음. 재실행은 새로운 --out 경로로.
    if out.exists():
        raise ValueError(f"출력 경로가 이미 있습니다: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='aihub-stage-', dir=out.parent) as td:
        staging = Path(td) / 'result'
        report = _prepare(root, staging)
        staging.rename(out)
    return report


def _prepare(root, out):
    folders = sorted({p.parent for p in root.rglob('*_keypoints.json')})
    groups = defaultdict(list)
    for folder in folders:
        m = CLIP.fullmatch(folder.name)
        if not m:
            raise ValueError(f"지원하지 않는 클립 이름: {folder.name}")
        groups[m[1]].append((m[2], m[3], m[4], folder))
    if not groups:
        raise ValueError("WORD 키포인트 클립을 찾지 못했습니다")

    out.mkdir(parents=True)
    rows, all_features = [], []
    label_groups = defaultdict(set)
    total_frames = valid_points = points = duplicates = 0
    incomplete_clips = []
    for group_id, entries in sorted(groups.items()):
        seen = {}
        loaded = [(entry, load_clip(entry[3])) for entry in entries]
        # 누락 프레임이 있는 뷰는 완전한 뷰의 부분집합인지도 검사합니다.
        loaded.sort(key=lambda x: (-len(x[1][0]), x[0][2] != 'F', x[0][2]))
        for (label, subject_token, view, folder), (indices, raw, camera, digest) in loaded:
            total_frames += len(raw)
            valid_points += int((raw[..., 3] > 0).sum())
            points += raw[..., 3].size
            missing = sorted(set(range(int(indices[0]), int(indices[-1])+1)) - set(indices.tolist()))
            if missing:
                incomplete_clips.append({'clip': folder.name, 'missing_frame_indices': missing})
            duplicate = None
            for previous, prev_indices, prev_raw in seen.values():
                loc = np.searchsorted(prev_indices, indices)
                if (np.all(loc < len(prev_indices))
                        and np.array_equal(prev_indices[loc], indices)
                        and np.array_equal(prev_raw[loc], raw)):
                    duplicate = previous
                    break
            if duplicate is not None:
                duplicate['views'].append(view)
                duplicate['source_paths'].append(str(folder.relative_to(root)))
                duplicate['camera_parameters'][view] = camera
                duplicate['frames_by_view'][view] = len(indices)
                duplicates += 1
                continue
            seq = np.stack([from_raw_hands(frame) for frame in raw])
            features = resample_descriptors(seq, frame_positions=indices)
            name = folder.name + '.npz'
            np.savez_compressed(out / name, raw_hands=raw, frame_indices=indices,
                                shape_features=features)
            row = {'file': name, 'label_id': label, 'group_id': group_id,
                   'subject_token': subject_token, 'views': [view],
                   'source_paths': [str(folder.relative_to(root))],
                   'frames': len(raw), 'raw_sha256': digest,
                   'frames_by_view': {view: len(indices)},
                   'camera_parameters': {view: camera}}
            seen[digest] = (row, indices, raw)
            rows.append(row)
            all_features.append(features)
            label_groups[label].add(group_id)
        print(f"{group_id}: {len(entries)} views -> {len(seen)} unique 3D", flush=True)

    labels = sorted(label_groups)
    np.savez_compressed(out / 'dataset.npz', X=np.stack(all_features),
                        y=np.array([labels.index(r['label_id']) for r in rows]),
                        groups=np.array([r['group_id'] for r in rows]),
                        labels=np.array(labels))
    counts = {k: len(v) for k, v in sorted(label_groups.items())}
    report = {
        'feature_version': FEATURE_VERSION, 'feature_dim': FEATURE_DIM,
        'sequence_length': 30, 'source_clips': len(folders),
        'source_frames': total_frames, 'labels': labels,
        'subject_tokens': sorted({r['subject_token'] for r in rows}),
        'unique_motion_groups': len(groups), 'unique_3d_sequences': len(rows),
        'duplicate_3d_clips_removed': duplicates,
        'incomplete_clips': incomplete_clips,
        'valid_3d_point_fraction': valid_points / points,
        'groups_per_label': counts,
        'can_make_disjoint_train_test_with_all_classes': min(counts.values()) >= 2,
        'deployment_ready': False,
        'limitations': [
            'WORD IDs have not been mapped to Korean vocabulary.',
            'Confidence flags are annotations, not measured live detection accuracy.',
            'Shape descriptors discard palm orientation, wrist trajectory and body location.',
            '30-frame clip resampling is not the runtime 1-second sliding-window contract.',
            'One motion group must not cross training, validation or test splits.',
        ],
        'clips': rows,
    }
    (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'clips'},
                     ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    prepare(args.root, args.out)
