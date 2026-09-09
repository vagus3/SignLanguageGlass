# -*- coding: utf-8 -*-
"""
반이중(half-duplex) 오디오 게이트

■ 이게 없으면 생기는 일
    착용자가 수어를 함
      -> TTS 가 스피커로 "얼마예요?" 를 송출
      -> 안경에 달린 마이크가 그 소리를 다시 주워담음
      -> STT 가 "얼마예요"를 인식
      -> HUD 에 자기가 방금 한 말이 상대방 말인 것처럼 뜸
      -> 최악의 경우 그게 또 트리거가 되어 무한 루프

    스피커와 마이크가 같은 안경테 위 몇 cm 거리에 붙어 있으므로
    이건 '혹시'가 아니라 '반드시' 일어납니다.

■ 해결
    제대로 하려면 음향 에코 제거(AEC)가 필요하지만 파이썬으로는 과합니다.
    말하는 동안 듣지 않는 반이중 방식이면 충분합니다.
    사람도 무전기도 그렇게 씁니다.

    tts 쪽:  with gate.output():  ...재생...
    stt 쪽:  if not gate.is_open(): 오디오 버림
"""
import time
import threading
from contextlib import contextmanager


class AudioGate:
    def __init__(self, tail_sec=0.4):
        self.tail_sec = tail_sec
        self._lock = threading.Lock()
        self._depth = 0          # 중첩 재생 대비 (카운터)
        self._until = 0.0        # 잔향/버퍼가 빠질 때까지의 여유 시간

    def begin_output(self):
        with self._lock:
            self._depth += 1

    def end_output(self):
        with self._lock:
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                self._until = time.time() + self.tail_sec

    def is_open(self):
        """True 면 마이크 입력을 받아도 되는 상태."""
        with self._lock:
            return self._depth == 0 and time.time() >= self._until

    @contextmanager
    def output(self):
        self.begin_output()
        try:
            yield
        finally:
            self.end_output()


class NullGate:
    """게이트를 쓰지 않을 때(유선 이어폰 사용 등)의 더미."""

    def is_open(self):
        return True

    @contextmanager
    def output(self):
        yield
