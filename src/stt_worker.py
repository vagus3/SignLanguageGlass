# -*- coding: utf-8 -*-
"""
Phase 2. 실시간 한국어 STT 스레드 (vosk)

vosk 를 고른 이유
    - 완전 오프라인, 82MB, Apache 2.0 (비용 0원)
    - 스트리밍 + 부분결과(partial) 지원 -> 말하는 도중에 자막이 갱신됨
    - whisper 계열은 정확하지만 배치형이라 발화가 끝나야 결과가 나옴

모델 준비
    https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip
    압축 해제 후 models/vosk-ko/ 에 배치

    단독 테스트:  python -m src.stt_worker
"""
import sys
import json
import queue
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


class SttWorker(threading.Thread):
    """
    결과를 out_queue 에 넣습니다.
        ("partial", "안녕하세")   말하는 중 (계속 덮어쓰기)
        ("final",   "안녕하세요") 발화 확정
    """

    daemon = True

    def __init__(self, out_queue=None, device=None, use_vad=True, gate=None):
        super().__init__(name="stt")
        self.out = out_queue or queue.Queue()
        self.device = device
        self.use_vad = use_vad
        self.gate = gate            # AudioGate: TTS 재생 중 입력 차단
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

    # ------------------------------------------------------------------
    def run(self):
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer, SetLogLevel
        except ImportError as e:
            self.out.put(("error", f"패키지 없음: {e}"))
            return

        if not C.VOSK_MODEL.exists():
            self.out.put(("error",
                          f"{C.VOSK_MODEL} 없음. vosk-model-small-ko-0.22 를 받아 "
                          f"models/vosk-ko/ 로 옮기세요"))
            return

        SetLogLevel(-1)
        rec = KaldiRecognizer(Model(str(C.VOSK_MODEL)), C.AUDIO_RATE)
        rec.SetWords(False)

        vad = None
        if self.use_vad:
            try:
                import webrtcvad
                vad = webrtcvad.Vad(C.VAD_AGGRESSIVENESS)
            except ImportError:
                pass

        def cb(indata, frames, time_info, status):
            # TTS 중 받은 자기 소리를 애초에 큐에 넣지 않습니다. 재생이 끝난 뒤
            # 밀린 블록이 STT로 흘러가 에코 자막이 되는 경로도 차단합니다.
            if self.gate is not None and not self.gate.is_open():
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

        last_partial = ""
        silence_ms = 0
        block_ms = int(C.AUDIO_BLOCK / C.AUDIO_RATE * 1000)

        try:
            with sd.RawInputStream(samplerate=C.AUDIO_RATE, blocksize=C.AUDIO_BLOCK,
                                   dtype="int16", channels=1,
                                   device=self.device, callback=cb):
                while not self._quit.is_set():
                    try:
                        data = self._audio.get(timeout=0.3)
                    except queue.Empty:
                        continue

                    # TTS 가 스피커로 나가는 동안은 내 목소리를 받아쓰지 않습니다.
                    # 인식기도 리셋해서 반쯤 들어간 자기 음성을 버립니다.
                    if self.gate is not None and not self.gate.is_open():
                        rec.Reset()
                        last_partial, silence_ms = "", 0
                        continue

                    # ★ 무음도 반드시 vosk 에 넣어야 합니다.
                    #   VAD 로 무음을 걸러서 안 넣으면 인식기가 "발화가 끝났다"를
                    #   영영 알 수 없어 final 결과가 안 나옵니다.
                    #   VAD 는 '버리는 용도'가 아니라 '종료 판단 용도'로만 씁니다.
                    speech = self._has_speech(vad, data) if vad else True
                    silence_ms = 0 if speech else silence_ms + block_ms

                    if rec.AcceptWaveform(data):
                        text = json.loads(rec.Result()).get("text", "").strip()
                        if text:
                            self.out.put(("final", text))
                        last_partial, silence_ms = "", 0
                        continue

                    p = json.loads(rec.PartialResult()).get("partial", "").strip()
                    if p and p != last_partial:
                        last_partial = p
                        self.out.put(("partial", p))

                    # 무음이 충분히 이어지면 강제로 확정 (엔드포인팅)
                    if last_partial and silence_ms >= C.STT_ENDPOINT_MS:
                        text = json.loads(rec.FinalResult()).get("text", "").strip()
                        if text:
                            self.out.put(("final", text))
                        last_partial, silence_ms = "", 0
        except Exception as e:
            self.out.put(("error", str(e)))

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


if __name__ == "__main__":
    import time
    q = queue.Queue()
    w = SttWorker(q)
    w.start()
    print("말해보세요. Ctrl+C 로 종료.")
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
