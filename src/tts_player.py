# -*- coding: utf-8 -*-
"""
비동기 TTS 재생기

캐시된 wav 를 메모리에 미리 올려두고 워커 스레드에서 재생합니다.
메인 루프는 절대 블로킹되지 않습니다 (블로킹되면 영상이 끊깁니다).

    tts = TtsPlayer(); tts.start()
    tts.say("얼마예요?")
"""
import sys
import queue
import re
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import clock
from src.build_tts_cache import (cache_key, legacy_aliases, legacy_cache_path,
                                  normalize_tts_text)


def is_korean_voice(voice):
    """언어 메타데이터도 확인하고 Kyoko 같은 이름의 'ko'는 제외합니다."""
    languages = getattr(voice, "languages", []) or []
    if isinstance(languages, (str, bytes)):
        languages = [languages]
    parts = [getattr(voice, "id", ""), getattr(voice, "name", "")] + list(languages)
    text = " ".join(p.decode("utf-8", errors="ignore") if isinstance(p, bytes)
                    else str(p) for p in parts).lower()
    return bool(re.search(r"(?:^|[^a-z])(?:ko(?:-kr|_kr)?|korean|heami|yuna|0412)(?:$|[^a-z])",
                          text))


class TtsPlayer(threading.Thread):

    def __init__(self, preload=True, gate=None, stats=None):
        super().__init__(name="tts", daemon=True)
        from src.latency import NullTracker
        self.stats = stats or NullTracker()
        self.q = queue.Queue()
        self.events = queue.Queue()
        self._com = None
        # ★ 이름을 _stop 으로 두면 안 됩니다.
        # threading.Thread 에는 _stop() 이라는 내부 메서드가 있고, join() 이
        # 스레드 종료를 확인한 뒤 그걸 호출합니다. Event 로 덮어쓰면
        # join() 이 'Event object is not callable' 로 죽습니다.
        self._quit = threading.Event()
        self._cache = {}          # {stem: (samples, samplerate)}
        self._aliases = {}        # {정규화한 문장: 기존 v1 파일 stem}
        self._fallback = None
        if gate is None:
            from src.audio_gate import NullGate
            gate = NullGate()
        self.gate = gate
        if preload:
            self._preload()

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
        self._aliases = legacy_aliases(set(self._cache))
        if self._cache:
            print(f"[tts] 캐시 {len(self._cache)}개 로드")
        else:
            print("[tts] 캐시 없음 -> `python -m src.build_tts_cache` 권장 "
                  "(없으면 OS 내장 음성으로 대체)")

    def say(self, text, origin_t=0.0):
        """
        origin_t : 이 발화를 촉발한 수어 단어가 확정된 시각(선택, clock.now() 기준).
                   주면 '수어 확정 -> 실제 소리 출력' 전체 지연을 실측합니다.

        큐에 넣은 시각도 같이 싣습니다. 소리가 나기 시작할 때까지의 시간
        (큐 대기 + 오디오 장치 열기)이 Phase 1 지연의 마지막 조각입니다.
        """
        if text and text.strip():
            self.q.put((text.strip(), clock.now(), origin_t))

    def stop(self):
        self._quit.set()
        self.q.put(None)

    @staticmethod
    def _unpack(item):
        """(text, t_say, origin_t) 또는 그냥 text 둘 다 받습니다."""
        if isinstance(item, tuple):
            return item
        return item, 0.0, 0.0

    def run(self):
        try:
            while not self._quit.is_set():
                item = self.q.get()
                if item is None:
                    break
                # 확정된 문장은 프레임이 아닙니다. FIFO로 한 번씩 모두 발화합니다.
                text, t_say, origin_t = self._unpack(item)
                try:
                    self._play(text, t_say, origin_t)
                except Exception as exc:
                    print(f"[tts] 음성 출력 실패: {exc}", flush=True)
                    self.events.put({"ok": False, "text": text, "error": str(exc)})
                else:
                    self.events.put({"ok": True, "text": text, "error": ""})
        finally:
            if self._fallback is not None:
                try:
                    self._fallback.stop()
                except Exception:
                    pass
                self._fallback = None
            if self._com is not None:
                self._com.CoUninitialize()

    def _play(self, text, t_say=0.0, origin_t=0.0):
        # 재생하는 동안 마이크를 막습니다(반이중).
        # 스피커와 마이크가 안경테 위 몇 cm 거리라 이게 없으면
        # 자기가 한 말이 상대방 자막으로 다시 떠오릅니다.
        with self.gate.output():
            hit = self._lookup(text)
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

    def _lookup(self, text):
        """v2 키 -> 정규화가 같은 기존 v1 파일 -> v1 원문 일치 순서로 찾습니다."""
        for stem in (cache_key(text), self._aliases.get(normalize_tts_text(text)),
                     legacy_cache_path(text).stem):
            if stem is not None and stem in self._cache:
                return self._cache[stem]
        return None

    def _mark(self, t_say, origin_t):
        """소리가 나기 직전에 지연을 찍습니다."""
        if t_say:
            self.stats.since("tts_start", t_say)
        if origin_t:
            self.stats.since("end_to_end", origin_t)

    def _speak_fallback(self, text):
        """캐시에 없을 때 OS 내장 음성으로. macOS=Yuna, Windows=Heami."""
        if sys.platform == "win32" and self._com is None:
            import pythoncom
            pythoncom.CoInitialize()
            self._com = pythoncom
        import pyttsx3
        if self._fallback is None:
            eng = pyttsx3.init()
            voice = next((v for v in eng.getProperty("voices") if is_korean_voice(v)), None)
            if voice is None:
                eng.stop()
                raise RuntimeError("OS 한국어 TTS 음성이 없습니다. 한국어 음성을 설치하거나 "
                                   "python -m src.build_tts_cache 로 WAV 캐시를 생성하세요.")
            eng.setProperty("voice", voice.id)
            eng.setProperty("rate", 170)
            self._fallback = eng
            print(f"[tts] OS 한국어 음성: {voice.name}", flush=True)
        eng = self._fallback
        errors, finished = [], []
        tokens = [eng.connect("error", lambda name, exception: errors.append(str(exception))),
                  eng.connect("finished-utterance", lambda name, completed: finished.append(completed))]
        try:
            eng.say(text)
            eng.runAndWait()
            if errors or not finished or not all(finished):
                raise RuntimeError("; ".join(errors) or "TTS 엔진이 발화를 완료하지 못했습니다")
        finally:
            for token in tokens:
                eng.disconnect(token)


def test_output(text="안녕하세요. 음성 출력 테스트입니다.", timeout=30):
    """앱과 같은 워커/캐시/OS 엔진으로 검사. 성공은 청취 확인을 대신하지 않습니다."""
    if not text or not text.strip():
        print("[tts] 테스트 문장을 입력하세요")
        return False
    player = TtsPlayer()
    player.start()
    player.say(text)
    try:
        result = player.events.get(timeout=timeout)
        if result["ok"]:
            print("[tts] 재생 처리가 완료됐습니다. 실제로 들리는지와 출력 장치를 확인하세요.")
        return result["ok"]
    except queue.Empty:
        print("[tts] 출력 시간 초과: OS 음성 엔진과 기본 스피커를 확인하세요")
        return False
    finally:
        player.stop()
        player.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(0 if test_output() else 1)
