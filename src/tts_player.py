# -*- coding: utf-8 -*-
"""
비동기 TTS 재생기

캐시된 wav 를 메모리에 미리 올려두고 워커 스레드에서 재생합니다.
메인 루프는 절대 블로킹되지 않습니다 (블로킹되면 영상이 끊깁니다).

    tts = TtsPlayer(); tts.start()
    tts.say("얼마예요?")
"""
import sys
import time
import queue
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.build_tts_cache import cache_path


class TtsPlayer(threading.Thread):
    daemon = True

    def __init__(self, preload=True, gate=None, stats=None):
        super().__init__(name="tts")
        from src.latency import NullTracker
        self.stats = stats or NullTracker()
        self.q = queue.Queue()
        # ★ 이름을 _stop 으로 두면 안 됩니다.
        # threading.Thread 에는 _stop() 이라는 내부 메서드가 있고, join() 이
        # 스레드 종료를 확인한 뒤 그걸 호출합니다. Event 로 덮어쓰면
        # join() 이 'Event object is not callable' 로 죽습니다.
        self._quit = threading.Event()
        self._cache = {}          # {stem: (samples, samplerate)}
        self._fallback = None
        if gate is None:
            from src.audio_gate import NullGate
            gate = NullGate()
        self.gate = gate
        if preload:
            self._preload()

    # ------------------------------------------------------------------
    def _preload(self):
        """시작 시 전부 메모리에 올려 디스크 I/O 지연도 제거합니다."""
        try:
            import soundfile as sf
        except ImportError:
            return
        if not C.TTS_CACHE.exists():
            return
        for f in C.TTS_CACHE.glob("*.wav"):
            try:
                data, sr = sf.read(f, dtype="float32")
                # 샘플레이트는 파일마다 따로 들고 있어야 합니다.
                # 하나로 뭉뚱그리면 마지막에 읽은 파일의 값이 전부에게 적용돼,
                # 다른 레이트로 만든 wav 하나만 섞여도 모든 발화가 느려지거나
                # 빨라집니다(24kHz 를 16kHz 로 재생 -> 1.5배 늘어진 목소리).
                self._cache[f.stem] = (data, int(sr))
            except Exception as e:
                print(f"[tts] {f.name} 읽기 실패: {e}")
        if self._cache:
            print(f"[tts] 캐시 {len(self._cache)}개 로드")
        else:
            print("[tts] 캐시 없음 -> `python -m src.build_tts_cache` 권장 "
                  "(없으면 OS 내장 음성으로 대체)")

    def say(self, text, origin_t=0.0):
        """
        origin_t : 이 발화를 촉발한 수어 단어가 확정된 시각(선택).
                   주면 '수어 확정 -> 실제 소리 출력' 전체 지연을 실측합니다.

        큐에 넣은 시각도 같이 싣습니다. 소리가 나기 시작할 때까지의 시간
        (큐 대기 + 오디오 장치 열기)이 Phase 1 지연의 마지막 조각입니다.
        """
        self.q.put((text, time.time(), origin_t))

    def stop(self):
        self._quit.set()
        self.q.put(None)

    @staticmethod
    def _unpack(item):
        """(text, t_say, origin_t) 또는 그냥 text 둘 다 받습니다."""
        if isinstance(item, tuple):
            return item
        return item, 0.0, 0.0

    # ------------------------------------------------------------------
    def run(self):
        while not self._quit.is_set():
            item = self.q.get()
            if item is None:
                break
            # 큐가 밀렸으면 오래된 발화는 버립니다 (대화가 어긋나는 것보다 낫습니다)
            while not self.q.empty():
                nxt = self.q.get_nowait()
                if nxt is None:
                    return
                item = nxt
            text, t_say, origin_t = self._unpack(item)
            self._play(text, t_say, origin_t)

    def _play(self, text, t_say=0.0, origin_t=0.0):
        # 재생하는 동안 마이크를 막습니다(반이중).
        # 스피커와 마이크가 안경테 위 몇 cm 거리라 이게 없으면
        # 자기가 한 말이 상대방 자막으로 다시 떠오릅니다.
        with self.gate.output():
            hit = self._cache.get(cache_path(text).stem)
            if hit is not None:
                data, sr = hit
                try:
                    import sounddevice as sd
                    self._mark(t_say, origin_t)
                    sd.play(data, sr, blocking=True)
                    return
                except Exception as e:
                    print(f"[tts] 재생 실패: {e}")
            self._mark(t_say, origin_t)
            self._speak_fallback(text)

    def _mark(self, t_say, origin_t):
        """소리가 나기 직전에 지연을 찍습니다."""
        if t_say:
            self.stats.since("tts_start", t_say)
        if origin_t:
            self.stats.since("end_to_end", origin_t)

    def _speak_fallback(self, text):
        """캐시에 없을 때 OS 내장 음성으로. macOS=Yuna, Windows=Heami."""
        try:
            import pyttsx3
            if self._fallback is None:
                eng = pyttsx3.init()
                for v in eng.getProperty("voices"):
                    s = f"{getattr(v, 'id', '')} {getattr(v, 'name', '')}".lower()
                    if "ko" in s or "korean" in s or "heami" in s or "yuna" in s:
                        eng.setProperty("voice", v.id)
                        break
                eng.setProperty("rate", 170)
                self._fallback = eng
            self._fallback.say(text)
            self._fallback.runAndWait()
        except Exception as e:
            print(f"[tts] '{text}' (음성 출력 불가: {e})")


if __name__ == "__main__":
    import time
    t = TtsPlayer()
    t.start()
    for s in ["안녕하세요", "얼마예요?", "감사합니다"]:
        t.say(s)
        time.sleep(1.5)
    t.stop()
