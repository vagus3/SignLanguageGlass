# -*- coding: utf-8 -*-
"""
TTS 캐시 생성기 (개발 중 1회만 실행)

    python -m src.build_tts_cache

핵심 아이디어
    수어 어휘가 10~15개로 '고정'되어 있으므로, 발화할 문장 전체를 미리
    wav 로 뽑아둘 수 있습니다. 그러면 실행 시점에는 파일 재생만 하므로

        실행 시 합성 대기와 네트워크 의존성 제거 (장치/재생 지연은 남음)

    실시간 TTS 호출은 300~800ms 가 걸리고 네트워크가 끊기면 시연이 죽습니다.
    이 트릭 하나로 Phase 1 의 지연 예산 대부분이 사라집니다.
"""
import sys
import asyncio
import hashlib
import argparse
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# 캐시 키 버전. v1 = 원문 그대로의 md5 앞 10자리, v2 = 정규화한 문장의 md5.
KEY_VERSION = "v2"

# 읽으면 같은 소리가 나도록 검수한 문장 끝 변형만 합칩니다. '?' 는 억양과
# 뜻이 달라 합치지 않습니다("네." 대답 / "네?" 되물음).
_SAME_ENDINGS = ".!~…"
_FULL_WIDTH = str.maketrans({"？": "?", "！": "!", "．": ".", "。": ".", "～": "~"})


def normalize_tts_text(text):
    """캐시 키용 정규화. 공백 변형과 검수된 문장 끝 부호 변형만 통합합니다.

    숫자·소수점·부호·쉼표 등 문장 안의 문자는 바꾸지 않습니다.
    """
    s = unicodedata.normalize("NFC", text).translate(_FULL_WIDTH)
    s = " ".join(s.split())
    s = re.sub(r"\s+([?!.,~…])(?=\s|$)", r"\1", s)   # "뭐예요 ?" -> "뭐예요?"
    return s.rstrip(_SAME_ENDINGS).rstrip() or s


def cache_key(text):
    digest = hashlib.md5(normalize_tts_text(text).encode("utf-8")).hexdigest()[:16]
    return f"{KEY_VERSION}_{digest}"


def cache_path(text):
    return C.TTS_CACHE / f"{cache_key(text)}.wav"


def legacy_cache_path(text):
    """v1 파일 이름(원문 md5). 이미 만든 캐시를 다시 합성하지 않고 쓰기 위한 호환용."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return C.TTS_CACHE / f"{h}.wav"


def _read_manifest():
    path = C.TTS_CACHE / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def legacy_aliases(available_stems):
    """{정규화한 문장: 기존 파일 stem}. manifest 원문과 config 문장에서 찾습니다."""
    known = [(Path(name).stem, entry.get("text"))
             for name, entry in _read_manifest().items() if isinstance(entry, dict)]
    known += [(legacy_cache_path(text).stem, text) for text in phrases()]
    aliases = {}
    for stem, text in known:
        if stem in available_stems and isinstance(text, str):
            aliases.setdefault(normalize_tts_text(text), stem)
    return aliases


def cached_file(text, aliases=None):
    """이 문장의 캐시 wav 경로. 재생과 같은 순서(v2 키 -> 같은 정규화의 v1 -> v1 원문)로
    찾고 없으면 None. aliases 는 legacy_aliases() 결과를 여러 문장에 재사용할 때 넘깁니다."""
    stems = {p.stem for p in C.TTS_CACHE.glob("*.wav")}
    if aliases is None:
        aliases = legacy_aliases(stems)
    for stem in (cache_key(text), aliases.get(normalize_tts_text(text)),
                 legacy_cache_path(text).stem):
        if stem in stems:
            return C.TTS_CACHE / f"{stem}.wav"
    return None


def migrate_legacy(texts, manifest):
    """v1 파일을 v2 이름으로 복사합니다(재합성 없음). 복사한 문장 목록을 돌려줍니다."""
    copied = []
    for text in texts:
        new, old = cache_path(text), legacy_cache_path(text)
        if new.exists() or not old.exists():
            continue
        shutil.copy2(old, new)
        entry = dict(manifest.get(old.name) or {})
        entry.update(text=text, normalized=normalize_tts_text(text), key_version=KEY_VERSION)
        manifest[new.name] = entry
        copied.append(text)
    return copied


def phrases(glossary=None):
    """캐싱할 문장 = 어휘 전체 + 규칙 기반 완성 문장 (+ 용어집의 검수 문장).

    glossary(경로 또는 Glossary)를 주면 의미 검수를 통과한 문장도 넣습니다. 검수 문장은
    언어 모델을 거치지 않으므로, 미리 합성해 두면 바로 캐시 음성으로 나갑니다.
    """
    out = [w for w in C.VOCAB if w != "NEUTRAL"]
    out += [s for _, s in C.PHRASE_RULES]
    out += ["잠시만요", "네, 알겠습니다"]     # 시연용 여유분
    if glossary is not None:
        from src.translation import Glossary
        if not isinstance(glossary, Glossary):
            glossary = Glossary.load(glossary)
        out += [entry["text"] for entry in glossary.sentences.values()]
    return sorted(set(out))


async def _synth(text, dst):
    import edge_tts
    import soundfile as sf
    import numpy as np

    mp3 = dst.with_suffix(".mp3")
    await edge_tts.Communicate(text, C.TTS_VOICE).save(str(mp3))

    # soundfile(libsndfile 1.2+) 은 mp3 를 읽을 수 있습니다. 추가 의존성 불필요.
    data, sr = sf.read(mp3, dtype="float32", always_2d=True)
    data = data.mean(axis=1)                       # 모노
    if sr != C.AUDIO_RATE:                         # 선형 리샘플로 충분
        n = int(len(data) * C.AUDIO_RATE / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n),
                         np.arange(len(data)), data).astype("float32")
    sf.write(dst, data, C.AUDIO_RATE, subtype="PCM_16")
    mp3.unlink(missing_ok=True)


def synth_macos(text, dst, voice="Yuna"):
    """설치된 macOS 한국어 음성으로 16kHz PCM 캐시 생성. 재생하지 않습니다."""
    import soundfile as sf
    import numpy as np

    if sys.platform != "darwin":
        raise RuntimeError("--engine macos는 macOS에서만 사용할 수 있습니다")
    with tempfile.TemporaryDirectory(prefix="signglass-tts-") as td:
        source = Path(td) / "speech.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(source), text],
                       check=True, capture_output=True, timeout=30)
        data, sr = sf.read(source, dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if not len(data) or not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-5:
            raise RuntimeError("TTS가 비어 있거나 무음인 파일을 생성했습니다")
        if sr != C.AUDIO_RATE:
            from scipy.signal import resample_poly
            from math import gcd
            divisor = gcd(sr, C.AUDIO_RATE)
            data = resample_poly(data, C.AUDIO_RATE // divisor, sr // divisor)
        sf.write(dst, data, C.AUDIO_RATE, subtype="PCM_16")


async def main(engine="edge", voice="Yuna", glossary=None):
    C.TTS_CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = C.TTS_CACHE / "manifest.json"
    manifest = _read_manifest()
    wanted = phrases(glossary)
    copied = migrate_legacy(wanted, manifest)
    if copied:
        print(f"기존 v1 캐시 {len(copied)}개를 {KEY_VERSION} 이름으로 복사했습니다 (재합성 없음).")
    # 정규화하면 같은 문장은 한 번만 합성합니다.
    todo = list({cache_key(p): p for p in wanted if not cache_path(p).exists()}.values())
    if not todo:
        if copied:
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print("캐시가 이미 최신입니다.")
        return
    print(f"{len(todo)}개 생성 (engine={engine}, voice={voice if engine == 'macos' else C.TTS_VOICE})")
    failed = []
    for i, text in enumerate(todo, 1):
        dst = cache_path(text)
        try:
            if engine == "macos":
                await asyncio.to_thread(synth_macos, text, dst, voice)
            else:
                await _synth(text, dst)
            manifest[dst.name] = {"text": text, "normalized": normalize_tts_text(text),
                                  "key_version": KEY_VERSION, "engine": engine,
                                  "voice": voice if engine == "macos" else C.TTS_VOICE,
                                  "sample_rate": C.AUDIO_RATE}
            print(f"  [{i}/{len(todo)}] {text}  ->  {dst.name}")
        except Exception as e:
            failed.append(text)
            print(f"  [실패] {text}: {e}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n성공 {len(todo) - len(failed)} / 실패 {len(failed)}: {C.TTS_CACHE}")
    print("키는 정규화한 문장의 md5(v2)입니다. 공백·문장 끝 마침표 차이는 같은 파일을 쓰고, "
          "config.py 문구를 바꾸면 다시 돌리세요.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="한국어 TTS WAV 캐시 생성")
    ap.add_argument("--engine", choices=["edge", "macos"], default="edge")
    ap.add_argument("--voice", default="Yuna", help="macOS 로컬 음성 이름")
    ap.add_argument("--glossary", help="glossary.json 의 검수 문장도 합성 (의미 검수 후)")
    args = ap.parse_args()
    asyncio.run(main(args.engine, args.voice, args.glossary))
