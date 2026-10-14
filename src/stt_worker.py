# -*- coding: utf-8 -*-
"""
Phase 2. 실시간 한국어 STT 스레드

두 엔진을 같은 계약(partial/final/error 큐, AudioGate)으로 씁니다.
    sherpa  sherpa-onnx 스트리밍 Zipformer. 오프라인, 부분결과, ONNX 라서
            노트북과 휴대폰(Android/iOS)에서 같은 모델을 쓸 수 있습니다.
    vosk    예전 기본값(82MB, Apache 2.0). 대체 경로로 남겨 둡니다.
    whisper 계열은 정확하지만 배치형이라 발화가 끝나야 결과가 나와 쓰지 않습니다.

모델 준비 (README "STT 모델")
    sherpa: models/sherpa-ko/ 에 encoder/decoder/joiner(.onnx)와 tokens.txt
    vosk:   https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip
            압축 해제 후 models/vosk-ko/
    두 엔진 비교:  python tools/compare_stt.py --tts-cache
    단독 테스트:   python -m src.stt_worker [--backend sherpa|vosk]
"""
import sys
import json
import queue
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import clock
from src.caption_state import CaptionEvent

BACKENDS = ("sherpa", "vosk")


class VoskDecoder:
    """vosk + webrtcvad 엔드포인팅. feed() 는 int16 PCM 한 블록을 받아 이벤트를 돌려줍니다."""

    name = "vosk"

    def __init__(self, model_dir=None, rate=None, use_vad=True):
        from vosk import Model, KaldiRecognizer, SetLogLevel
        model_dir = Path(model_dir or C.VOSK_MODEL)
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} 없음. vosk-model-small-ko-0.22 를 받아 "
                                    f"models/vosk-ko/ 로 옮기세요")
        self.rate = rate or C.AUDIO_RATE
        SetLogLevel(-1)
        self.rec = KaldiRecognizer(Model(str(model_dir)), self.rate)
        self.rec.SetWords(False)
        self.vad = None
        if use_vad:
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(C.VAD_AGGRESSIVENESS)
            except ImportError:
                pass
        self.last_partial, self.silence_ms = "", 0

    def feed(self, pcm):
        # ★ 무음도 반드시 vosk 에 넣어야 합니다. VAD 로 무음을 걸러서 안 넣으면
        #   인식기가 "발화가 끝났다"를 영영 알 수 없어 final 결과가 안 나옵니다.
        #   VAD 는 '버리는 용도'가 아니라 '종료 판단 용도'로만 씁니다.
        events = []
        speech = self._has_speech(self.vad, pcm, self.rate) if self.vad else True
        self.silence_ms = 0 if speech else self.silence_ms + len(pcm) // 2 * 1000 / self.rate
        if self.rec.AcceptWaveform(pcm):
            text = json.loads(self.rec.Result()).get("text", "").strip()
            if text:
                events.append(("final", text))
            self.last_partial, self.silence_ms = "", 0
            return events
        partial = json.loads(self.rec.PartialResult()).get("partial", "").strip()
        if partial and partial != self.last_partial:
            self.last_partial = partial
            events.append(("partial", partial))
        # 무음이 충분히 이어지면 강제로 확정 (엔드포인팅)
        if self.last_partial and self.silence_ms >= C.STT_ENDPOINT_MS:
            events += self.flush()
        return events

    def flush(self):
        text = json.loads(self.rec.FinalResult()).get("text", "").strip()
        self.last_partial, self.silence_ms = "", 0
        return [("final", text)] if text else []

    def reset(self):
        self.rec.Reset()
        self.last_partial, self.silence_ms = "", 0

    @staticmethod
    def _has_speech(vad, pcm, rate=16000, frame_ms=30):
        """30ms 프레임 단위로 발화가 하나라도 있으면 통과."""
        n = int(rate * frame_ms / 1000) * 2   # int16 = 2 bytes
        for i in range(0, len(pcm) - n + 1, n):
            try:
                if vad.is_speech(pcm[i:i + n], rate):
                    return True
            except Exception:
                return True
        return False


def transducer_files(model_dir):
    """encoder/decoder/joiner/tokens 경로. 같은 부품의 int8 판이 있으면 그쪽을 씁니다."""
    model_dir = Path(model_dir)
    if not (model_dir / "tokens.txt").exists():
        raise FileNotFoundError(f"{model_dir}/tokens.txt 없음. README 'STT 모델'을 보세요")
    files = {"tokens": model_dir / "tokens.txt"}
    for part in ("encoder", "decoder", "joiner"):
        found = sorted(model_dir.glob(f"{part}*.onnx"))
        int8 = [p for p in found if p.name.endswith(".int8.onnx")]
        if not found:
            raise FileNotFoundError(f"{model_dir} 에 {part}*.onnx 가 없습니다")
        if len(int8 or found) > 1:
            raise ValueError(f"{model_dir} 에 {part} 모델이 여러 개입니다: "
                             f"{[p.name for p in int8 or found]}")
        files[part] = (int8 or found)[0]
    return files


class SherpaDecoder:
    """sherpa-onnx 스트리밍 transducer. 엔드포인트 판단은 디코더가 직접 합니다."""

    name = "sherpa"

    def __init__(self, model_dir=None, rate=None, threads=2):
        import sherpa_onnx
        f = transducer_files(model_dir or C.SHERPA_MODEL)
        self.rate = rate or C.AUDIO_RATE
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(f["tokens"]), encoder=str(f["encoder"]), decoder=str(f["decoder"]),
            joiner=str(f["joiner"]), num_threads=threads, sample_rate=self.rate,
            feature_dim=80, decoding_method="greedy_search", enable_endpoint_detection=True,
            # rule1: 아무 말 없이 2.4초, rule2: 말한 뒤 무음이 STT_ENDPOINT_MS, rule3: 20초
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=C.STT_ENDPOINT_MS / 1000,
            rule3_min_utterance_length=20.0)
        self.files = f
        self.stream = self.rec.create_stream()
        self.last = ""

    def feed(self, pcm):
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self.stream.accept_waveform(self.rate, samples)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        text = self.rec.get_result(self.stream).strip()
        if self.rec.is_endpoint(self.stream):
            self.rec.reset(self.stream)
            self.last = ""
            return [("final", text)] if text else []
        if text and text != self.last:
            self.last = text
            return [("partial", text)]
        return []

    def flush(self):
        """입력 끝(파일 비교용): 남은 프레임을 마저 디코딩해 확정합니다."""
        self.stream.accept_waveform(self.rate, np.zeros(int(self.rate * 0.6), np.float32))
        self.stream.input_finished()
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        text = self.rec.get_result(self.stream).strip()
        self.reset()
        return [("final", text)] if text else []

    def reset(self):
        self.stream = self.rec.create_stream()
        self.last = ""


def open_decoder(backend=None, model_dir=None):
    backend = backend or C.STT_BACKEND
    if backend not in BACKENDS:
        raise ValueError(f"STT backend must be one of {BACKENDS}")
    return (SherpaDecoder if backend == "sherpa" else VoskDecoder)(model_dir)


class SttWorker(threading.Thread):
    """
    결과를 out_queue 에 넣습니다.
        ("partial", "안녕하세")   말하는 중 (계속 덮어쓰기)
        ("final",   "안녕하세요") 발화 확정
        ("error",   "...")        패키지·모델 없음 등

    caption_events=True이면 partial/final/cancel의 두 번째 값은 CaptionEvent.
    오류와 기본 호출 모드는 기존 문자열 계약을 유지합니다. observed_at은
    인식 결과 생성 시각이며 음성 시작/종료 시각이나 화자 식별 결과가 아닙니다.
    """

    daemon = True

    def __init__(self, out_queue=None, device=None, use_vad=True, gate=None, backend=None,
                 caption_events=False):
        super().__init__(name="stt")
        self.out = out_queue or queue.Queue()
        self.device = device
        self.use_vad = use_vad
        self.backend = backend or C.STT_BACKEND
        self.gate = gate            # AudioGate: TTS 재생 중 입력 차단
        self.caption_events = caption_events
        self._utterance_id = 0
        self._utterance_open = False
        self._gate_interrupted = threading.Event()
        # ★ 이름을 _stop 으로 두면 안 됩니다.
        # threading.Thread 에는 _stop() 이라는 내부 메서드가 있고, join() 이
        # 스레드 종료를 확인한 뒤 그걸 호출합니다. Event 로 덮어쓰면
        # join() 이 'Event object is not callable' 로 죽습니다.
        self._quit = threading.Event()
        # AUDIO_BLOCK=4000이면 한 블록이 250ms입니다. 32개 큐는 장애 순간에
        # 최대 8초 지연을 만들 수 있으므로 짧게 유지하고 최신 입력을 살립니다.
        self._audio = queue.Queue(maxsize=4)

    def stop(self):
        self._quit.set()

    def _fallback_hint(self):
        return " (--stt vosk 로 예전 엔진을 쓸 수 있습니다)" if self.backend == "sherpa" else ""

    def _decoder(self):
        if self.backend == "vosk":
            return VoskDecoder(use_vad=self.use_vad)
        return open_decoder(self.backend)

    def _publish(self, kind, text):
        if not self.caption_events:
            self.out.put((kind, text))
            return
        if not self._utterance_open:
            self._utterance_id += 1
            self._utterance_open = True
        self.out.put((kind, CaptionEvent(kind, text, self._utterance_id, clock.now())))
        if kind in ("final", "cancel"):
            self._utterance_open = False

    # ------------------------------------------------------------------
    def run(self):
        try:
            import sounddevice as sd
            decoder = self._decoder()
        except ImportError as e:
            self.out.put(("error", f"패키지 없음: {e}" + self._fallback_hint()))
            return
        except (OSError, ValueError) as e:
            self.out.put(("error", str(e) + self._fallback_hint()))
            return

        def cb(indata, frames, time_info, status):
            # TTS 중 받은 자기 소리를 애초에 큐에 넣지 않습니다. 재생이 끝난 뒤
            # 밀린 블록이 STT로 흘러가 에코 자막이 되는 경로도 차단합니다.
            if self.gate is not None and not self.gate.is_open():
                self._gate_interrupted.set()
                return
            try:
                self._audio.put_nowait(bytes(indata))
            except queue.Full:
                # 오래된 250ms를 버리고 현재 블록을 보존합니다. 음성 일부가
                # 빠지더라도 수 초 늦은 자막보다 실시간 대화에서 덜 해롭습니다.
                try:
                    self._audio.get_nowait()
                    self._audio.put_nowait(bytes(indata))
                except queue.Empty:
                    pass

        try:
            with sd.RawInputStream(samplerate=C.AUDIO_RATE, blocksize=C.AUDIO_BLOCK,
                                   dtype="int16", channels=1,
                                   device=self.device, callback=cb):
                while not self._quit.is_set():
                    # Check even without incoming audio: the callback drops audio
                    # while TTS plays, so waiting for a block leaves stale partials.
                    if (self._gate_interrupted.is_set() or
                            (self.gate is not None and not self.gate.is_open())):
                        self._gate_interrupted.clear()
                        decoder.reset()
                        while not self._audio.empty():
                            try:
                                self._audio.get_nowait()
                            except queue.Empty:
                                break
                        if self.caption_events and self._utterance_open:
                            self._publish("cancel", "")
                        self._quit.wait(0.02)
                        continue
                    try:
                        data = self._audio.get(timeout=0.3)
                    except queue.Empty:
                        continue
                    # TTS 가 스피커로 나가는 동안은 내 목소리를 받아쓰지 않습니다.
                    # 인식기도 리셋해서 반쯤 들어간 자기 음성을 버립니다.
                    if self.gate is not None and not self.gate.is_open():
                        decoder.reset()
                        self._gate_interrupted.set()
                        continue
                    for kind, text in decoder.feed(data):
                        self._publish(kind, text)
        except Exception as e:
            self.out.put(("error", str(e)))

    # 예전 호출부 호환 (tools 등)
    _has_speech = staticmethod(VoskDecoder._has_speech)


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default=None)
    q = queue.Queue()
    w = SttWorker(q, backend=ap.parse_args().backend)
    w.start()
    print(f"[{w.backend}] 말해보세요. Ctrl+C 로 종료.")
    try:
        while True:
            kind, text = q.get()
            if kind == "error":
                print("ERROR:", text)
                break
            if kind == "partial":
                print(f"\r  ... {text}", end="", flush=True)
            else:
                print(f"\r  >>> {text}" + " " * 20)
            time.sleep(0.001)
    except KeyboardInterrupt:
        w.stop()
