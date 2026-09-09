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
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def cache_path(text):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return C.TTS_CACHE / f"{h}.wav"


def phrases():
    """캐싱할 문장 = 어휘 전체 + 규칙 기반 완성 문장."""
    out = [w for w in C.VOCAB if w != "NEUTRAL"]
    out += [s for _, s in C.PHRASE_RULES]
    out += ["잠시만요", "네, 알겠습니다"]     # 시연용 여유분
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


async def main(engine="edge", voice="Yuna"):
    C.TTS_CACHE.mkdir(parents=True, exist_ok=True)
    todo = [p for p in phrases() if not cache_path(p).exists()]
    if not todo:
        print("캐시가 이미 최신입니다.")
        return
    print(f"{len(todo)}개 생성 (engine={engine}, voice={voice if engine == 'macos' else C.TTS_VOICE})")
    manifest_path = C.TTS_CACHE / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failed = []
    for i, text in enumerate(todo, 1):
        dst = cache_path(text)
        try:
            if engine == "macos":
                await asyncio.to_thread(synth_macos, text, dst, voice)
            else:
                await _synth(text, dst)
            manifest[dst.name] = {"text": text, "engine": engine,
                                  "voice": voice if engine == "macos" else C.TTS_VOICE,
                                  "sample_rate": C.AUDIO_RATE}
            print(f"  [{i}/{len(todo)}] {text}  ->  {dst.name}")
        except Exception as e:
            failed.append(text)
            print(f"  [실패] {text}: {e}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n성공 {len(todo) - len(failed)} / 실패 {len(failed)}: {C.TTS_CACHE}")
    print("인덱스는 텍스트 md5 로 계산되므로 config.py 의 문구를 바꾸면 다시 돌리세요.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="한국어 TTS WAV 캐시 생성")
    ap.add_argument("--engine", choices=["edge", "macos"], default="edge")
    ap.add_argument("--voice", default="Yuna", help="macOS 로컬 음성 이름")
    args = ap.parse_args()
    asyncio.run(main(args.engine, args.voice))
