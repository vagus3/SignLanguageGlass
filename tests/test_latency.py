"""지연 하네스 + TTS 경로 통합 확인 (오디오 장치 없이)."""
import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys, time, types
import numpy as np
from src.latency import LatencyTracker, NullTracker

stats = LatencyTracker()

# 구간 측정 API
for i in range(50):
    with stats.span("holistic"):
        time.sleep(0.002 + (0.01 if i == 7 else 0))     # 한 번 튀는 프레임
    stats.add("infer", 3.0 + np.random.rand())
    stats.add("sign_total", 8.0 + np.random.rand()*2)

# TtsPlayer 를 실제로 돌려 tts_start / end_to_end 가 찍히는지
played = []
class FakeSd:
    @staticmethod
    def play(data, sr, blocking=True):
        played.append(sr); time.sleep(0.01)
sys.modules["sounddevice"] = FakeSd

import src.tts_player as T
t = T.TtsPlayer(preload=False, stats=stats)
t._cache["dummy"] = (np.zeros(100, np.float32), 16000)
t.start()

import src.build_tts_cache as B
for i in range(5):
    word_t = time.time()
    time.sleep(0.30)                       # PhraseBuilder 조합 대기 흉내
    stats.since("phrase_latch", word_t)
    key = B.cache_path(f"문장{i}").stem
    t._cache[key] = (np.zeros(100, np.float32), 16000)
    t.say(f"문장{i}", origin_t=word_t)
    time.sleep(0.12)
t.stop(); t.join(timeout=2)

print(stats.report("지연 측정 (모의 실행)"))
s = stats.summary()
for k in ("holistic", "infer", "sign_total", "phrase_latch", "tts_start", "end_to_end"):
    assert k in s, f"{k} 미기록"
e2e = s["end_to_end"][1]; latch = s["phrase_latch"][1]
print(f"검증: end_to_end({e2e:.0f}ms) > phrase_latch({latch:.0f}ms) 이어야 함 -> {e2e > latch}")
assert e2e > latch

csv = stats.save_csv("/tmp/lat_test.csv")
print(f"CSV 저장 확인: {csv}")
assert csv and csv.exists()

n = NullTracker()
with n.span("x"): pass
n.add("y", 1); n.since("z", 0)
assert n.report() == "" and n.save_csv("/tmp/nope.csv") is None
print("NullTracker(--stats 없을 때) 무해 동작 확인")
