"""Where training data came from, carried from manifests into trained models.

The AI Hub usage policy asks that results built on its data state that they are
results of the National Information Society Agency (NIA) project, including
derived works such as trained models, and forbids passing the data on without
approval. Manifests record their sources, training copies them into checkpoint
metadata, and the ONNX sidecar and the server's /health carry them on.

Datasets converted before sources were recorded can be marked afterwards:
python -m src.data_sources data/processed/.../manifest.json --source aihub-sign-video
"""
import argparse
import json
from pathlib import Path

SOURCES = {
    "aihub-sign-video": {
        "id": "aihub-sign-video",
        "name": "AI 허브 수어 영상",
        "provider": "한국지능정보사회진흥원(NIA), 구축: 테스트웍스 외",
        "url": "https://aihub.or.kr/aidata/7965",
        "attribution": "이 프로젝트는 한국지능정보사회진흥원의 사업결과물인 AI 허브 "
                       "'수어 영상' 데이터를 활용했습니다(https://aihub.or.kr/aidata/7965).",
        "redistribution": False,
    },
    "team": {
        "id": "team",
        "name": "팀 자체 수집",
        "provider": "signglass 팀 (참여자 동의 범위 안에서만 사용)",
        "url": "",
        "attribution": "",
        "redistribution": False,
    },
}


def source_entry(source_id):
    if source_id not in SOURCES:
        raise ValueError(f"Unknown data source {source_id!r}; one of {sorted(SOURCES)}")
    return dict(SOURCES[source_id])


def manifest_sources(manifest):
    """Sources recorded in a manifest dict; manifests from before this field report none."""
    sources = manifest.get("sources") or []
    if not isinstance(sources, list) or any(not isinstance(s, dict) or "id" not in s
                                            for s in sources):
        raise ValueError("Manifest 'sources' must be a list of source records")
    return sources


def attribution(sources):
    """Distinct attribution sentences, in order, for the sources that require one."""
    lines = []
    for source in sources:
        text = source.get("attribution")
        if text and text not in lines:
            lines.append(text)
    return lines


def set_sources(manifest_path, source_ids):
    """Record the sources of an existing manifest, replacing what it had."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["sources"] = [source_entry(s) for s in dict.fromkeys(source_ids)]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return manifest["sources"]


def main():
    p = argparse.ArgumentParser(description="기존 데이터셋 manifest 에 데이터 출처를 기록합니다.")
    p.add_argument("manifests", nargs="+")
    p.add_argument("--source", action="append", choices=sorted(SOURCES), required=True,
                   help="여러 출처면 반복 (예: --source aihub-sign-video --source team)")
    args = p.parse_args()
    for manifest in args.manifests:
        try:
            sources = set_sources(manifest, args.source)
        except (OSError, ValueError) as error:
            p.error(f"{manifest}: {error}")
        print(f"{manifest}: 출처 {', '.join(s['name'] for s in sources)}")


if __name__ == "__main__":
    main()
