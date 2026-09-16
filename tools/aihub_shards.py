#!/usr/bin/env python3
"""AI Hub 라벨 파일을 filekey 단위로 내려받아 3D 특징 shard로 변환한다.

API Key는 명령행 인자로 받지 않고 ``AIHUB_API_KEY`` 환경변수에서만 읽는다.
각 filekey는 독립 작업 디렉터리에서 처리하므로 중단 후 재실행할 수 있다.

예:
    export AIHUB_API_KEY='...'
    python tools/aihub_shards.py --dataset-key 123 \
      --file-keys 456,457 --work-dir /mnt/aihub-work \
      --out-dir /mnt/aihub-features
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_aihub3d import prepare


def parse_file_keys(value):
    keys = []
    for item in value.split(','):
        item = item.strip()
        if not item.isdigit():
            raise argparse.ArgumentTypeError(f"filekey는 숫자여야 합니다: {item!r}")
        if item not in keys:
            keys.append(item)
    if not keys:
        raise argparse.ArgumentTypeError("filekey가 하나 이상 필요합니다")
    return keys


def validate_output(path):
    dataset = path / 'dataset.npz'
    manifest = path / 'manifest.json'
    if not dataset.is_file() or not manifest.is_file():
        raise RuntimeError(f"변환 결과가 불완전합니다: {path}")
    with np.load(dataset, allow_pickle=False) as data:
        if data['X'].ndim != 3 or data['X'].shape[1:] != (30, 462):
            raise RuntimeError(f"예상하지 못한 특징 shape: {data['X'].shape}")
        count = int(data['X'].shape[0])
    report = json.loads(manifest.read_text(encoding='utf-8'))
    if report.get('feature_dim') != 462 or report.get('unique_3d_sequences') != count:
        raise RuntimeError("dataset.npz와 manifest.json의 계약이 일치하지 않습니다")
    return count


def extract_downloaded_zips(source):
    """aihubshell이 병합만 한 ZIP을 경로 이탈 없이 작업 폴더에 해제한다."""
    archives = sorted(source.rglob('*.zip'))
    if not archives:
        return
    extracted = source / '_extracted'
    marker = extracted / '.complete'
    if marker.is_file():
        return
    if extracted.exists():
        raise RuntimeError(f"불완전한 압축 해제 폴더가 있습니다: {extracted}")

    extracted.mkdir(parents=True)
    root = extracted.resolve()
    for archive_path in archives:
        with ZipFile(archive_path) as archive:
            members = archive.infolist()
            required = sum(item.file_size for item in members)
            free = shutil.disk_usage(extracted).free
            if free < required * 1.1:
                raise RuntimeError(
                    f"압축 해제 공간 부족: 필요 약 {required / 2**30:.1f}GiB, "
                    f"여유 {free / 2**30:.1f}GiB")
            for item in members:
                target = (extracted / item.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"ZIP 경로 이탈 항목: {item.filename}")
            print(f"압축 해제: {archive_path.name} ({len(members):,} entries)", flush=True)
            archive.extractall(extracted)
    marker.write_text('ok\n', encoding='utf-8')


def run_shard(*, shell, dataset_key, file_key, api_key, work_dir, out_dir,
              delete_source=False):
    source = work_dir / f"filekey-{file_key}"
    target = out_dir / f"filekey-{file_key}"
    done = target / '.complete'
    if done.is_file():
        return 'skipped', validate_output(target)
    if target.exists():
        raise RuntimeError(f"불완전한 기존 출력이 있습니다. 확인 후 이동하세요: {target}")

    source.mkdir(parents=True, exist_ok=True)
    archives = list(source.rglob('*.zip'))
    if archives:
        print(f"[resume] 기존 병합 ZIP {len(archives)}개 사용", flush=True)
    else:
        command = [str(shell), '-mode', 'd', '-datasetkey', str(dataset_key),
                   '-filekey', file_key, '-aihubapikey', api_key]
        subprocess.run(command, cwd=source, check=True)
    extract_downloaded_zips(source)
    if not any(source.rglob('*_keypoints.json')):
        raise RuntimeError(
            f"filekey {file_key}에서 *_keypoints.json을 찾지 못했습니다. "
            "원천 영상이 아니라 3D 라벨링 파일key인지 확인하세요.")

    out_dir.mkdir(parents=True, exist_ok=True)
    prepare(source, target)
    count = validate_output(target)
    done.write_text(json.dumps({'dataset_key': str(dataset_key),
                                'file_key': file_key,
                                'sequences': count}) + '\n', encoding='utf-8')
    if delete_source:
        resolved = source.resolve()
        if resolved.parent != work_dir.resolve() or not resolved.name.startswith('filekey-'):
            raise RuntimeError(f"삭제 안전 검사 실패: {resolved}")
        shutil.rmtree(resolved)
    return 'converted', count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-key', required=True)
    ap.add_argument('--file-keys', required=True, type=parse_file_keys,
                    help='라벨링 파일key. 쉼표로 여러 개 지정')
    ap.add_argument('--work-dir', type=Path, required=True,
                    help='다운로드·압축해제용 임시 디스크')
    ap.add_argument('--out-dir', type=Path, required=True,
                    help='검증된 특징 shard 저장 위치')
    ap.add_argument('--aihubshell', type=Path,
                    default=Path(shutil.which('aihubshell') or 'aihubshell'))
    ap.add_argument('--delete-source', action='store_true',
                    help='변환 검증 후 해당 filekey 원본 작업 폴더 삭제')
    args = ap.parse_args()

    api_key = os.environ.get('AIHUB_API_KEY')
    if not api_key:
        raise SystemExit('AIHUB_API_KEY 환경변수를 설정하세요. 키를 명령행에 쓰지 마세요.')
    shell = args.aihubshell.expanduser()
    if not shell.is_file() or not os.access(shell, os.X_OK):
        raise SystemExit(f"실행 가능한 aihubshell이 없습니다: {shell}")
    if args.work_dir.resolve() == args.out_dir.resolve():
        raise SystemExit('--work-dir와 --out-dir는 서로 달라야 합니다')

    for key in args.file_keys:
        state, count = run_shard(
            shell=shell, dataset_key=args.dataset_key, file_key=key,
            api_key=api_key, work_dir=args.work_dir, out_dir=args.out_dir,
            delete_source=args.delete_source)
        print(f"[{state}] filekey={key}, sequences={count}", flush=True)


if __name__ == '__main__':
    main()
