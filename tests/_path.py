# -*- coding: utf-8 -*-
"""
저장소 루트를 sys.path 에 올립니다.

각 테스트 맨 위에서 `import _path` 한 줄이면 됩니다.
(테스트를 직접 실행하면 sys.path[0] 이 tests/ 라서 이 모듈이 잡힙니다)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
