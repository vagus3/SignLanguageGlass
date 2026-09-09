#!/usr/bin/env python
"""마이크/스피커 없이 모델·폰트·캐시 및 합성음성→STT 연결을 검사.

python tools/verify_assets.py --out reports/assets.json
합성음성 검사는 실제 사람의 STT 성능 평가가 아닙니다.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.build_tts_cache import phrases, cache_path
from src.hud import HudRenderer


def main(out):
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    report = {'packages': {name: version(name) for name in
                           ['torch','onnx','onnxruntime','scikit-learn','vosk','mediapipe']},
              'wav_count': 0, 'synthetic_stt_only': True,
              'human_speech_accuracy_evaluated': False, 'examples': []}
    for text in phrases():
        data, rate = sf.read(cache_path(text))
        if (rate != C.AUDIO_RATE or data.ndim != 1 or not len(data)
                or not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-5):
            raise RuntimeError(f'유효하지 않은 TTS 캐시: {text}')
        report['wav_count'] += 1
    hud = HudRenderer()
    report['font'] = hud.font.getname()
    if not C.HUD_FONT.exists() or not hud.render('안녕하세요').getbbox():
        raise RuntimeError('폰트/렌더링 검사 실패')
    model = Model(str(C.VOSK_MODEL))
    report['vosk_model_loaded'] = True
    for text in ['안녕하세요', '병원이 어디예요?', '감사합니다']:
        data, rate = sf.read(cache_path(text), dtype='int16')
        pcm = data.tobytes() + bytes(rate * 2)
        rec = KaldiRecognizer(model, rate)
        parts = []
        for i in range(0, len(pcm), C.AUDIO_BLOCK * 2):
            if rec.AcceptWaveform(pcm[i:i + C.AUDIO_BLOCK * 2]):
                parts.append(json.loads(rec.Result()).get('text', ''))
        parts.append(json.loads(rec.FinalResult()).get('text', ''))
        report['examples'].append({'expected': text, 'recognized': ' '.join(filter(None, parts))})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('reports/assets.json'))
    main(ap.parse_args().out)
