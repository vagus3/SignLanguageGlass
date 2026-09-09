import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf
import config as C

tmp = Path(tempfile.mkdtemp()); C.TTS_CACHE = tmp
import importlib
import src.build_tts_cache as B; importlib.reload(B)
import src.tts_player as T; importlib.reload(T)

# 일부러 서로 다른 샘플레이트로 캐시를 만듭니다
made = {}
for text, sr in (("얼마예요?", 16000), ("감사합니다", 24000), ("안녕하세요", 16000)):
    dst = B.cache_path(text)
    n = sr // 2
    sf.write(dst, np.sin(np.linspace(0, 400, n)).astype("float32"), sr, subtype="PCM_16")
    made[text] = sr

played = []
class FakeSd:
    @staticmethod
    def play(data, sr, blocking=True): played.append((len(data), sr))
sys.modules["sounddevice"] = FakeSd

p = T.TtsPlayer(preload=True)
for text in made:
    p._play(text)

print("  캐시에 넣은 레이트 :", made)
print("  실제 재생된 레이트 :", {t: sr for t, (n, sr) in zip(made, played)})
for text, (n, sr) in zip(made, played):
    assert sr == made[text], f"{text}: {sr} != {made[text]}"
print("\n  [OK] 파일마다 자기 샘플레이트로 재생됩니다")
print("  (수정 전이었다면 세 개 모두 마지막에 읽은 파일의 레이트로 재생)")
