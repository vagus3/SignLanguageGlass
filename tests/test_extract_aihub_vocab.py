"""형태소 ZIP에서 시점 중복 없이 어휘를 추출하는지 검증."""
import _path  # noqa: F401
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

from tools.extract_aihub_vocab import extract

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / 'morpheme.zip'
    with ZipFile(path, 'w') as archive:
        for word, value in [('WORD0002', '학교'), ('WORD0001', '고민')]:
            node = {'data': [{'attributes': [{'name': value}]}]}
            for view in ('D', 'F', 'L'):
                archive.writestr(
                    f'morpheme/01/NIA_SL_{word}_REAL01_{view}_morpheme.json',
                    json.dumps(node))
    assert extract(path) == {'WORD0001': '고민', 'WORD0002': '학교'}

print('AI Hub WORD 어휘 ZIP 직접 추출 확인')
