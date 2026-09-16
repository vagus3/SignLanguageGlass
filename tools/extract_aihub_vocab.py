#!/usr/bin/env python3
"""AI Hub 수어 형태소 ZIP에서 WORD ID → 한국어 어휘 사전을 추출한다."""
import argparse
import json
from pathlib import Path
import re
from zipfile import ZipFile


WORD = re.compile(r'NIA_SL_(WORD\d+)_([^_]+)_([FLRUD])_morpheme\.json$')


def extract(zip_path):
    selected = {}
    with ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match = WORD.search(name)
            if not match:
                continue
            word_id, subject, view = match.groups()
            rank = (view != 'F', subject, name)
            if word_id not in selected or rank < selected[word_id][0]:
                selected[word_id] = (rank, name)

        vocab = {}
        conflicts = {}
        for word_id, (_, name) in sorted(selected.items()):
            node = json.loads(archive.read(name))
            values = {
                attr.get('name', '').strip()
                for item in node.get('data', [])
                for attr in item.get('attributes', [])
                if attr.get('name', '').strip()
            }
            if len(values) != 1:
                conflicts[word_id] = sorted(values)
                continue
            vocab[word_id] = values.pop()
    if conflicts:
        preview = dict(list(conflicts.items())[:10])
        raise ValueError(f'어휘가 없거나 여러 개인 WORD ID: {preview}')
    return vocab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('zip_path', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    vocab = extract(args.zip_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
    print(f'저장: {args.out} ({len(vocab):,}개 어휘)')


if __name__ == '__main__':
    main()
