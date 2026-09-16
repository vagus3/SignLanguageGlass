"""분할 다운로드 도구의 키 파싱과 출력 계약 검증."""
import _path  # noqa: F401
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from tools.aihub_shards import extract_downloaded_zips, parse_file_keys, validate_output

assert parse_file_keys('12, 13,12') == ['12', '13']
try:
    parse_file_keys('12,bad')
    raise AssertionError('잘못된 filekey를 허용했습니다')
except argparse.ArgumentTypeError:
    pass

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    np.savez_compressed(root / 'dataset.npz', X=np.zeros((2, 30, 462), np.float32))
    (root / 'manifest.json').write_text(json.dumps({
        'feature_dim': 462, 'unique_3d_sequences': 2
    }))
    assert validate_output(root) == 2

with tempfile.TemporaryDirectory() as td:
    from zipfile import ZipFile
    root = Path(td)
    with ZipFile(root / 'labels.zip', 'w') as archive:
        archive.writestr('NIA_SL_WORD0001_REAL01_F/x_keypoints.json', '{}')
    extract_downloaded_zips(root)
    assert (root / '_extracted/NIA_SL_WORD0001_REAL01_F/x_keypoints.json').is_file()
    assert (root / '_extracted/.complete').is_file()

print('AI Hub filekey 파싱 + shard 출력 계약 확인')
