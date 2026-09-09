# -*- coding: utf-8 -*-
"""
지연 측정 하네스

이 프로젝트의 핵심 주장은 "체감 지연 1000ms 를 170ms 로 줄였다" 입니다.
그런데 --debug 오버레이는 순간값만 보여줘서, 보고서에 쓸 숫자가 안 나옵니다.
여기서 구간별로 모아 p50/p95 로 보고합니다.

    stats = LatencyTracker()
    with stats.span("holistic"):
        res = holistic.process(rgb)
    ...
    print(stats.report())
    stats.save_csv("latency.csv")

■ 왜 평균이 아니라 p95 인가
    지연은 정규분포가 아닙니다. 대부분 20ms 인데 가끔 300ms 가 튀는 식이라,
    평균을 쓰면 "괜찮아 보이는데 실제로는 뚝뚝 끊기는" 상태를 놓칩니다.
    사용자가 체감하는 건 평균이 아니라 꼬리 쪽입니다.

■ Phase 1 지연의 분해 (수어 -> 소리)
    recognition   확정을 만든 추론의 최신 프레임 캡처 -> 단어 확정
                  (특징 추출 + ONNX 추론 + 임계값/다수결 판정)
    phrase_latch  단어 확정 -> 발화 지시
                  (문장 규칙이 이어질지 기다리는 시간. 보통 여기가 제일 큽니다)
    tts_start     발화 지시 -> 스피커에서 실제로 소리가 나기 시작
    end_to_end    위 셋의 합

    이렇게 쪼개 두면 "느리다"가 아니라 "어디가 느리다"를 말할 수 있습니다.
"""
import time
import threading
import unicodedata
from contextlib import contextmanager

import numpy as np


def _w(s):
    """터미널 표시 폭. 한글·전각 문자는 두 칸을 차지합니다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s, width):
    return s + " " * max(0, width - _w(s))

# 보고서에 나올 순서. 여기 없는 이름은 뒤에 알파벳순으로 붙습니다.
ORDER = ["capture", "holistic", "feature", "infer", "sign_total",
         "render", "display",
         "recognition", "phrase_latch", "tts_start", "end_to_end"]

LABEL = {
    "capture":      "카메라 read()",
    "holistic":     "MediaPipe Holistic",
    "feature":      "특징 추출(246차원)",
    "infer":        "ONNX 추론",
    "sign_total":   "  = 수어 경로 합계",
    "render":       "HUD 렌더",
    "display":      "디스플레이 출력",
    "recognition":  "인식(프레임->단어확정)",
    "phrase_latch": "문장 조합 대기",
    "tts_start":    "TTS 재생 시작까지",
    "end_to_end":   "▶ 수어->소리 전체",
}


class LatencyTracker:
    """
    스레드 여러 개에서 동시에 기록해도 되도록 락을 겁니다.
    (비전/STT/TTS 스레드가 각자 자기 구간을 넣습니다)
    """

    def __init__(self, maxlen=20000, enabled=True):
        self.enabled = enabled
        self.maxlen = maxlen
        self._lock = threading.Lock()
        self._d = {}
        self.t_start = time.time()

    def add(self, name, ms):
        if not self.enabled:
            return
        with self._lock:
            a = self._d.setdefault(name, [])
            if len(a) < self.maxlen:
                a.append(float(ms))

    @contextmanager
    def span(self, name):
        """with stats.span("holistic"): ... 로 구간을 잽니다."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1000.0)

    def since(self, name, t0):
        """이미 갖고 있는 시작 시각(time.time() 기준)으로부터의 경과를 기록."""
        self.add(name, (time.time() - t0) * 1000.0)

    # ------------------------------------------------------------------
    def summary(self):
        """{이름: (n, p50, p95, max)}"""
        with self._lock:
            items = {k: np.asarray(v, np.float64) for k, v in self._d.items() if v}
        out = {}
        for k, a in items.items():
            out[k] = (len(a), float(np.percentile(a, 50)),
                      float(np.percentile(a, 95)), float(a.max()))
        return out

    def report(self, title="지연 측정 결과"):
        s = self.summary()
        if not s:
            return "지연 표본이 없습니다. (--stats 없이 돌렸거나 너무 짧게 실행)"

        keys = [k for k in ORDER if k in s] + sorted(k for k in s if k not in ORDER)
        w = max(_w(LABEL.get(k, k)) for k in keys) + 2
        total = w + 36
        el = time.time() - self.t_start

        lines = ["", "=" * total, f"  {title}   (실행 {el:.0f}초)", "=" * total,
                 "  " + _pad("구간", w)
                 + f"{'표본':>6}{'p50':>10}{'p95':>10}{'최대':>10}",
                 "  " + "-" * (total - 2)]
        for k in keys:
            n, p50, p95, mx = s[k]
            lines.append("  " + _pad(LABEL.get(k, k), w)
                         + f"{n:>6}{p50:>8.1f}ms{p95:>8.1f}ms{mx:>8.1f}ms")
        lines.append("=" * total)

        # 실시간 여부 판정: 수어 경로 한 바퀴가 프레임 간격 안에 들어오는가
        if "sign_total" in s:
            p95 = s["sign_total"][2]
            budget = 1000.0 / 30
            lines.append(
                f"  수어 경로 p95 {p95:.1f}ms / 30fps 예산 {budget:.1f}ms"
                + ("  -> 여유 있음" if p95 < budget else
                   "  -> 초과. 해상도나 model_complexity 를 낮추세요"))
        if "end_to_end" in s:
            n, p50, p95, _ = s["end_to_end"]
            lines.append(f"  수어->소리 전체: 중앙값 {p50:.0f}ms, p95 {p95:.0f}ms "
                         f"(표본 {n}회)")
        lines.append("")
        return "\n".join(lines)

    def save_csv(self, path):
        """구간별 원본 표본. 보고서 그래프용."""
        import csv
        from pathlib import Path
        with self._lock:
            d = {k: list(v) for k, v in self._d.items() if v}
        if not d:
            return None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = [k for k in ORDER if k in d] + sorted(k for k in d if k not in ORDER)
        with p.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["span", "index", "ms"])
            for k in keys:
                for i, v in enumerate(d[k]):
                    wr.writerow([k, i, f"{v:.3f}"])
        return p


class NullTracker:
    """--stats 없이 돌 때. 호출부에 if 문을 안 넣으려고 둡니다."""

    enabled = False

    def add(self, name, ms):
        pass

    @contextmanager
    def span(self, name):
        yield

    def since(self, name, t0):
        pass

    def summary(self):
        return {}

    def report(self, title=""):
        return ""

    def save_csv(self, path):
        return None
