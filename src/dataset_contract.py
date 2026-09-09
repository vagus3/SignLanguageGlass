# -*- coding: utf-8 -*-
"""직접 수집 데이터의 전처리 계약과 촬영자/세션 파일명 규칙."""
import json
import re
from pathlib import Path

import config as C
from src.landmarks import FEATURE_DIM, FEATURE_VERSION

META_NAME = "dataset_meta.json"
_SAFE = re.compile(r"[^0-9A-Za-z가-힣.-]+")


def current_contract():
    return {
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "seq_len": C.SEQ_LEN,
        "target_fps": C.TRAIN_FPS,
    }


def ensure_dataset_contract(root=None, create=False):
    """계약이 현재 코드와 같은지 검사하고, 빈 수집 폴더라면 생성합니다."""
    root = Path(root or C.DATA_DIR)
    path = root / META_NAME
    expected = current_contract()
    if not path.exists():
        has_samples = root.exists() and any(root.rglob("*.npy"))
        if has_samples:
            raise RuntimeError(
                f"{path}: 기존 .npy의 전처리 버전을 확인할 수 없습니다.\n"
                "  기존 파일을 별도 폴더로 옮기고 src.collect로 새로 수집하세요.")
        if not create:
            raise RuntimeError(
                f"{path}: 계약 파일과 직접 수집 데이터가 없습니다.\n"
                "  먼저 python -m src.collect를 실행하세요.")
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return expected

    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"{path} 파싱 실패: {e}") from e
    bad = {k: (got.get(k), v) for k, v in expected.items() if got.get(k) != v}
    if bad:
        detail = ", ".join(f"{k}={old!r}(필요 {new!r})"
                           for k, (old, new) in bad.items())
        raise RuntimeError(
            f"데이터 전처리 계약 불일치: {detail}\n"
            "  차원이 같아도 의미가 다를 수 있습니다. 현재 코드로 다시 수집하세요.")
    return got


def _safe_id(value, fallback):
    value = _SAFE.sub("-", str(value).strip()).strip("-.")
    return value or fallback


def capture_prefix(signer=None, session=None):
    """새 형식: signer__session. '__'로 촬영자와 세션을 명확히 분리합니다."""
    return (f"{_safe_id(signer or C.SIGNER_ID, 'unknown')}__"
            f"{_safe_id(session or C.SESSION_ID, 'session')}")


def signer_from_stem(stem):
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return prefix.split("__", 1)[0] if "__" in prefix else prefix


def session_from_stem(stem):
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return prefix.split("__", 1)[1] if "__" in prefix else "legacy"
