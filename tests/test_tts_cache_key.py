"""TTS cache keys merge spacing and reviewed sentence endings only; existing v1 caches still play."""
import _path  # noqa: F401
import importlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import config as C

C.TTS_CACHE = Path(tempfile.mkdtemp())
import src.build_tts_cache as B
importlib.reload(B)
import src.tts_player as T
importlib.reload(T)

n = B.normalize_tts_text
same = [("안녕하세요", "안녕하세요."), ("안녕하세요", " 안녕하세요 !"), ("안녕하세요", "안녕하세요..."),
        ("병원이 어디예요?", "병원이 어디예요 ?"), ("얼마예요?", "얼마예요？"), ("네?", "네?!"),
        ("네, 알겠습니다", "네 , 알겠습니다"), ("잠시만요", "잠시만요~")]
for a, b in same:
    assert n(a) == n(b) and B.cache_path(a) == B.cache_path(b), (a, b)
# Question intonation, numbers, decimals, signs and inner punctuation keep meaning.
different = [("네", "네?"), ("병원이 어디예요", "병원이 어디예요?"), ("3.5층이에요", "35층이에요"),
             ("영하 -3도", "영하 3도"), ("1,000원", "1000원"), ("3 .5", "3.5"),
             ("네, 알겠습니다", "네 알겠습니다")]
for a, b in different:
    assert n(a) != n(b), (a, b)
assert n("...") == "..." and n("?") == "?", "Punctuation-only text must not become empty"
assert B.cache_path("안녕하세요").name.startswith(B.KEY_VERSION + "_")


def tone(path, sr):
    sf.write(path, np.sin(np.linspace(0, 400, sr // 4)).astype("float32"), sr, subtype="PCM_16")


tone(B.cache_path("안녕하세요"), 16000)                    # v2 key
tone(B.cache_path("네"), 8000)                             # must not answer "네?"
legacy_manifest = B.legacy_cache_path("조금만 천천히 말해 주세요.")
tone(legacy_manifest, 22050)                               # v1 file listed in manifest.json
(C.TTS_CACHE / "manifest.json").write_text(json.dumps(
    {legacy_manifest.name: {"text": "조금만 천천히 말해 주세요.", "engine": "macos"}}),
    encoding="utf-8")
tone(B.legacy_cache_path("이름이 뭐예요?"), 24000)          # v1 file of a config phrase, no manifest

played, spoken = [], []


class FakeSd:
    @staticmethod
    def play(data, sr, blocking=True):
        played.append(sr)


sys.modules["sounddevice"] = FakeSd
player = T.TtsPlayer(preload=True)
player._speak_fallback = spoken.append
for text in ["안녕하세요.", "조금만 천천히 말해 주세요", "이름이 뭐예요 ?", "네?"]:
    player._play(text)
assert played == [16000, 22050, 24000], played
assert spoken == ["네?"], "A question must not reuse the statement recording"

# The build tool copies v1 files to v2 names instead of synthesizing them again.
manifest = json.loads((C.TTS_CACHE / "manifest.json").read_text(encoding="utf-8"))
copied = B.migrate_legacy(["조금만 천천히 말해 주세요.", "안녕하세요", "없는 문장"], manifest)
assert copied == ["조금만 천천히 말해 주세요."]
new = B.cache_path("조금만 천천히 말해 주세요.")
assert new.exists() and legacy_manifest.exists(), "Copy, so older checkouts still find v1"
assert manifest[new.name]["key_version"] == B.KEY_VERSION
assert manifest[new.name]["engine"] == "macos", "Keep how the audio was made"
assert B.migrate_legacy(["조금만 천천히 말해 주세요."], manifest) == []
print("[OK] normalized v2 keys, meaning-preserving variants, v1 cache compatibility, migration")
