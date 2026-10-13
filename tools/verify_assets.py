#!/usr/bin/env python
"""마이크/스피커 없이 모델·폰트·캐시 및 합성음성→STT 연결을 검사.

python tools/verify_assets.py --out reports/assets.json
python tools/verify_assets.py --stt vosk          # 대체 엔진으로 검사
설정된 STT 엔진(config.STT_BACKEND)으로 검사합니다. 합성음성 검사는 실제 사람의
STT 성능 평가가 아닙니다(tools/compare_stt.py, TEST_CHECKLIST 2번).
"""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.build_tts_cache import cached_file, phrases
from src.hud import HudRenderer
from src.stt_worker import BACKENDS, open_decoder

PACKAGES = ['torch', 'onnx', 'onnxruntime', 'scikit-learn', 'sherpa-onnx', 'vosk', 'mediapipe']
STT_PHRASES = ['안녕하세요', '병원이 어디예요?', '감사합니다']


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main(out, backend=None):
    from tools.compare_stt import cer, read_pcm, transcribe
    backend = backend or C.STT_BACKEND
    report = {'packages': {name: installed(name) for name in PACKAGES},
              'wav_count': 0, 'synthetic_stt_only': True,
              'human_speech_accuracy_evaluated': False, 'examples': []}
    for text in phrases():
        path = cached_file(text)
        if path is None:
            raise RuntimeError(f'TTS 캐시 없음: {text} (python -m src.build_tts_cache)')
        data, rate = sf.read(path)
        if (rate != C.AUDIO_RATE or data.ndim != 1 or not len(data)
                or not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-5):
            raise RuntimeError(f'유효하지 않은 TTS 캐시: {text}')
        report['wav_count'] += 1
    hud = HudRenderer()
    report['font'] = hud.font.getname()
    if not C.HUD_FONT.exists() or not hud.render('안녕하세요').getbbox():
        raise RuntimeError('폰트/렌더링 검사 실패')
    decoder = open_decoder(backend)
    report['stt_backend'] = backend
    report['stt_model'] = (str(decoder.files['encoder'].parent) if hasattr(decoder, 'files')
                           else str(C.VOSK_MODEL))
    for text in STT_PHRASES:
        recognized = transcribe(decoder, read_pcm(cached_file(text)))[0]
        errors, chars = cer(text, recognized)
        report['examples'].append({'expected': text, 'recognized': recognized,
                                   'cer': errors / max(chars, 1)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('reports/assets.json'))
    ap.add_argument('--stt', choices=BACKENDS, default=None,
                    help='검사할 STT 엔진 (기본: config.STT_BACKEND)')
    args = ap.parse_args()
    main(args.out, args.stt)
