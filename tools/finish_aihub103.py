#!/usr/bin/env python3
"""Finish approved REAL/WORD downloads sequentially; keep validation separate."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.aihub_shards import run_shard
from tools.extract_aihub_vocab import extract


def main():
    root = Path(__file__).resolve().parents[1]
    key = (Path.home() / '.config/signglass/aihub-api-key').read_text().strip()
    if not key:
        raise RuntimeError('AI Hub API Key file is empty')
    shell = Path.home() / '.local/bin/aihubshell'
    raw = root / 'data/raw/aihub103/shards'
    processed = root / 'data/processed/aihub103'

    for file_key in ['39600'] + [str(k) for k in range(39602, 39617)]:
        print(f'[start] training filekey={file_key}', flush=True)
        state, count = run_shard(
            shell=shell, dataset_key='103', file_key=file_key, api_key=key,
            work_dir=raw, out_dir=processed / 'shards', delete_source=True)
        print(f'[{state}] training filekey={file_key}, sequences={count}', flush=True)

    vocab_path = processed / 'validation/word_vocab.json'
    if not vocab_path.is_file():
        source = raw / 'filekey-39478'
        source.mkdir(parents=True, exist_ok=True)
        archives = list(source.rglob('*.zip'))
        if not archives:
            subprocess.run([str(shell), '-mode', 'd', '-datasetkey', '103',
                            '-filekey', '39478', '-aihubapikey', key],
                           cwd=source, check=True)
            archives = list(source.rglob('*.zip'))
        if len(archives) != 1:
            raise RuntimeError('Expected exactly one validation morpheme ZIP')
        vocab = extract(archives[0])
        if not vocab:
            raise RuntimeError('Validation vocabulary is empty')
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + '\n')
        if json.loads(vocab_path.read_text()) != vocab:
            raise RuntimeError('Validation vocabulary verification failed')
        shutil.rmtree(source)
        print(f'[converted] validation vocabulary={len(vocab)}; source deleted', flush=True)

    state, count = run_shard(
        shell=shell, dataset_key='103', file_key='39479', api_key=key,
        work_dir=raw, out_dir=processed / 'validation/shards', delete_source=True)
    print(f'[{state}] validation filekey=39479, sequences={count}', flush=True)
    print('[all-complete] AIHub 103 REAL/WORD training and validation', flush=True)


if __name__ == '__main__':
    main()
