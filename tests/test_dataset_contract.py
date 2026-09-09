"""구형/혼합 전처리 데이터가 조용히 학습되는 경로를 차단합니다."""
import _path  # noqa: F401
import json
import tempfile
from pathlib import Path

import numpy as np

from src.dataset_contract import (current_contract, ensure_dataset_contract,
                                  capture_prefix, signer_from_stem,
                                  session_from_stem)

root = Path(tempfile.mkdtemp()) / "raw"
got = ensure_dataset_contract(root, create=True)
assert got == current_contract()

stem = capture_prefix("kim", "day-2") + "_003"
assert signer_from_stem(stem) == "kim"
assert session_from_stem(stem) == "day-2"
assert signer_from_stem("legacy_user_003") == "legacy_user"

# 메타데이터 없는 기존 배열은 현재 버전이라고 추측해 덮어쓰면 안 됩니다.
old = Path(tempfile.mkdtemp()) / "raw"
(old / "word").mkdir(parents=True)
np.save(old / "word" / "me_000.npy", np.zeros((2, 2), np.float32))
try:
    ensure_dataset_contract(old, create=True)
    raise AssertionError("메타데이터 없는 기존 데이터가 통과했습니다")
except RuntimeError:
    pass

# 차원이 같아도 특징 버전이 다르면 실패해야 합니다.
bad = current_contract()
bad["feature_version"] = "legacy-v2"
(root / "dataset_meta.json").write_text(json.dumps(bad), encoding="utf-8")
try:
    ensure_dataset_contract(root)
    raise AssertionError("다른 특징 버전이 통과했습니다")
except RuntimeError:
    pass
print("  [OK] 데이터 계약/촬영자·세션 식별")
