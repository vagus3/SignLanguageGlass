"""STT decoders behind one worker contract, gate handling, model discovery and CER tooling."""
import _path  # noqa: F401
import json
import queue
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
import config as C
from src import stt_worker
from src.stt_worker import SttWorker, open_decoder, transducer_files
from tools.compare_stt import cer, load_refs, normalize

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    try:
        transducer_files(td)
        raise AssertionError("A model without tokens.txt was accepted")
    except FileNotFoundError:
        pass
    for name in ("tokens.txt", "encoder-x.onnx", "encoder-x.int8.onnx", "decoder-x.onnx",
                 "joiner-x.int8.onnx"):
        (td / name).write_bytes(b"")
    files = transducer_files(td)
    assert files["encoder"].name == "encoder-x.int8.onnx", "int8 parts are preferred"
    assert files["decoder"].name == "decoder-x.onnx" and files["joiner"].name == "joiner-x.int8.onnx"
    (td / "joiner-y.int8.onnx").write_bytes(b"")
    try:
        transducer_files(td)
        raise AssertionError("Two joiners were silently resolved")
    except ValueError:
        pass
    refs = td / "trans.txt"
    refs.write_text("0.wav 안녕하세요 반갑습니다.\n\n1.wav 네\n", encoding="utf-8")
    assert load_refs(refs) == {"0.wav": "안녕하세요 반갑습니다.", "1.wav": "네"}
    (td / "refs.json").write_text(json.dumps({"a.wav": "가"}), encoding="utf-8")
    assert load_refs(td / "refs.json") == {"a.wav": "가"}
try:
    open_decoder("whisper")
    raise AssertionError("Unknown backend accepted")
except ValueError:
    pass
assert normalize("병원이 어디예요?") == "병원이어디예요"
assert cer("병원이 어디예요?", "병원이어디예요") == (0, 7)
assert cer("안녕하세요", "한편 계세요")[0] == 3


class FakeDecoder:
    """Turns each block into events; records resets so the gate behaviour is visible."""
    name = "fake"

    def __init__(self):
        self.fed, self.resets = 0, 0

    def feed(self, pcm):
        self.fed += 1
        return [("partial", f"말{self.fed}")] if self.fed % 2 else [("final", f"문장{self.fed}")]

    def reset(self):
        self.resets += 1

    def flush(self):
        return []


class Gate:
    def __init__(self):
        self.open = True

    def is_open(self):
        return self.open


def fake_sounddevice(blocks, gate, closed_at):
    """RawInputStream that plays `blocks` into the callback, closing the gate midway."""
    class RawInputStream:
        def __init__(self, callback, **kwargs):
            self.callback = callback

        def __enter__(self):
            def play():
                for i in range(blocks):
                    gate.open = i not in closed_at
                    self.callback(bytes(C.AUDIO_BLOCK * 2), C.AUDIO_BLOCK, None, None)
                    time.sleep(0.02)
                gate.open = True
            threading.Thread(target=play, daemon=True).start()
            return self

        def __exit__(self, *exc):
            return False
    return types.SimpleNamespace(RawInputStream=RawInputStream)


gate, decoder, out = Gate(), FakeDecoder(), queue.Queue()
sys.modules["sounddevice"] = fake_sounddevice(6, gate, closed_at={2, 3})
worker = SttWorker(out, gate=gate, backend="sherpa")
worker._decoder = lambda: decoder
worker.start()
time.sleep(0.5)
worker.stop()
worker.join(2)
events = []
while not out.empty():
    events.append(out.get())
# Blocks captured while TTS played never reach the decoder (callback drops them).
assert decoder.fed == 4, decoder.fed
assert events[0] == ("partial", "말1") and ("final", "문장2") in events, events
assert not worker.is_alive()

# A missing sherpa model is reported with the way back to vosk, not a crash.
out = queue.Queue()
missing = SttWorker(out, backend="sherpa")
with tempfile.TemporaryDirectory() as td:
    C.SHERPA_MODEL = Path(td) / "absent"
    missing.run()
kind, message = out.get_nowait()
assert kind == "error" and "--stt vosk" in message, message
del sys.modules["sounddevice"]

# The vosk fallback decodes too (the refactor moved its loop into VoskDecoder).
from src.build_tts_cache import cached_file
from src.stt_worker import VoskDecoder
from tools.compare_stt import read_pcm, transcribe
try:
    import vosk  # noqa: F401
    vosk_ready = C.VOSK_MODEL.exists() and cached_file("이름이 뭐예요?") is not None
except ImportError:
    vosk_ready = False
if vosk_ready:
    vosk_decoder = VoskDecoder()
    text, seconds, first_ms, final_ms = transcribe(vosk_decoder, read_pcm(cached_file("이름이 뭐예요?")))
    assert cer("이름이 뭐예요?", text)[0] <= 2, text          # vosk heard "이 이름이 뭐 예요"
    assert final_ms is not None, "vosk endpointing (VAD silence) must finalize before the flush"
    assert isinstance(SttWorker(queue.Queue(), backend="vosk")._decoder(), VoskDecoder)
else:
    print("vosk 모델 또는 TTS 캐시가 없어 vosk 인식 확인은 건너뜁니다")

# Real recognition when the default model is installed (README "STT 모델").
C.SHERPA_MODEL = C.MODEL_DIR / "sherpa-ko-174m-chunk32"
try:
    import sherpa_onnx  # noqa: F401
    transducer_files(C.SHERPA_MODEL)
except (ImportError, OSError, ValueError):
    print("[OK] STT contract checked; sherpa model not installed, recognition skipped")
    sys.exit(77)
wav = cached_file("병원이 어디예요?")
if wav is None:
    print("[OK] STT contract checked; TTS cache missing, recognition skipped")
    sys.exit(77)
text, seconds, first_ms, final_ms = transcribe(stt_worker.SherpaDecoder(), read_pcm(wav))
assert normalize(text) == "병원이어디예요", text
assert first_ms is not None and seconds < 5
print("[OK] STT: one contract for sherpa/vosk, gate drops TTS echo, int8 model discovery, "
      "missing-model hint, CER tooling, real sherpa recognition")
