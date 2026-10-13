"""Asset check uses the configured STT engine; skipped where models, font or TTS cache are absent."""
import _path  # noqa: F401
import json
import sys
import tempfile
from pathlib import Path
import config as C
from src.build_tts_cache import cached_file, phrases
from src.stt_worker import transducer_files

try:
    import sherpa_onnx  # noqa: F401
    transducer_files(C.SHERPA_MODEL)
    assets = C.HUD_FONT.exists() and all(cached_file(t) for t in phrases())
except (ImportError, OSError, ValueError):
    assets = False
if not assets:
    print("sherpa 모델·폰트·TTS 캐시가 없어 자산 점검을 건너뜁니다")
    sys.exit(77)
from tools.verify_assets import main

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "assets.json"
    report = main(out, "sherpa")
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(report))
assert report["stt_backend"] == "sherpa" and report["wav_count"] == len(phrases())
assert report["packages"]["sherpa-onnx"] and "stt_model" in report
# A decoding fault (wrong rate, broken model) would garble all three; the default model gets
# these synthetic phrases right.
assert sum(e["cer"] == 0 for e in report["examples"]) >= 2, report["examples"]
print("[OK] verify_assets: configured STT engine, TTS cache, font and package report")
