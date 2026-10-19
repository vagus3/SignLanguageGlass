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

■ 최근 표본과 실행 전체 통계
    매 프레임 기록하는 구간은 30fps 에서 15분이면 2만 7천 개가 쌓입니다.
    구간마다 최근 maxlen 개만 링 버퍼로 남기고(최근 분위수는 정확),
    실행 전체의 표본 수·최댓값은 정확히 셉니다. 전체 분위수는 표본이
    maxlen 을 넘으면 로그 간격 히스토그램으로 근사하고 오차를 함께 표시합니다.
    정확한 전체 분위수가 필요하면 log_path(--stats 경로)의 전체 로그를 씁니다.

■ 시계
    since() 의 시작 시각은 clock.now() 로 잰 값이어야 합니다. time.time()
    (벽시계)과 섞으면 시계 보정 순간 음수나 수천 초짜리 지연이 찍힙니다.
"""
import csv
import math
import threading
import unicodedata
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src import clock


def _w(s):
    """터미널 표시 폭. 한글·전각 문자는 두 칸을 차지합니다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s, width):
    return s + " " * max(0, width - _w(s))


def _rpad(s, width):
    return " " * max(0, width - _w(s)) + s

# 보고서에 나올 순서. 여기 없는 이름은 뒤에 알파벳순으로 붙습니다.
ORDER = ["capture", "holistic", "feature", "infer", "sign_total",
         "render", "display",
         "recognition", "phrase_latch", "tts_start", "end_to_end"]

LABEL = {
    "cloud_roundtrip": "클라우드 요청 왕복",
    "cloud_response": "수어 종료->번역 수신",
    "pose_pack": "좌표 구간 패킹",
    "cloud_server_queue_wait": "서버 큐 대기",
    "cloud_server_recognize": "서버 CTC 인식",
    "cloud_server_language": "서버 언어 모델 생성",
    "cloud_server_server_total": "서버 전체",
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


# 전체 실행 분위수 근사: 칸 폭이 2%인 로그 간격 히스토그램. 칸의 기하 중앙값은
# 그 칸에 든 실제 값과 최대 약 1% 차이가 납니다.
_BIN_RATIO = 1.02
REL_ERROR = math.sqrt(_BIN_RATIO) - 1
_LOG_RATIO = math.log(_BIN_RATIO)
_MIN_MS = 1e-3          # 1µs 이하는 한 칸에 모읍니다(대표값 0)


def _rank(values, q):
    """하위 q% 순위의 실제 표본값(nearest-rank, numpy inverted_cdf)."""
    return float(np.percentile(np.asarray(values, np.float64), q, method="inverted_cdf"))


class _Span:
    """한 구간의 최근 표본(링 버퍼)과 실행 전체의 개수·최댓값·히스토그램."""

    __slots__ = ("recent", "count", "max", "bins")

    def __init__(self, maxlen):
        self.recent = deque(maxlen=maxlen)
        self.count = 0
        self.max = -math.inf
        self.bins = {}

    def add(self, ms):
        self.recent.append(ms)
        self.count += 1
        self.max = max(self.max, ms)
        k = -1 if ms <= _MIN_MS else int(math.log(ms / _MIN_MS) / _LOG_RATIO)
        self.bins[k] = self.bins.get(k, 0) + 1

    def whole_run(self, q):
        """(전체 실행 q 분위수, 정확 여부). 버려진 표본이 없으면 정확한 값입니다."""
        if self.count == len(self.recent):
            return _rank(self.recent, q), True
        target = max(1, math.ceil(q / 100 * self.count))
        seen = 0
        for k in sorted(self.bins):
            seen += self.bins[k]
            if seen >= target:
                return (0.0 if k < 0 else _MIN_MS * _BIN_RATIO ** (k + 0.5)), False
        raise AssertionError("histogram counts do not add up")


class LatencyTracker:
    """
    스레드 여러 개에서 동시에 기록해도 되도록 락을 겁니다.
    (비전/STT/TTS 스레드가 각자 자기 구간을 넣습니다)

    log_path 를 주면 모든 표본을 도착 순서대로 CSV 에 기록합니다(전체 로그).
    """

    def __init__(self, maxlen=20000, enabled=True, log_path=None):
        self.enabled = enabled
        self.maxlen = maxlen
        self._lock = threading.Lock()
        self._d = {}
        self.t_start = clock.now()
        self.log_path = Path(log_path) if log_path else None
        self._log = self._writer = None
        if self.log_path is not None and enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._log)
            self._writer.writerow(["span", "index", "ms"])

    def add(self, name, ms):
        if not self.enabled:
            return
        ms = float(ms)
        with self._lock:
            span = self._d.get(name)
            if span is None:
                span = self._d[name] = _Span(self.maxlen)
            span.add(ms)
            if self._writer is not None:
                self._writer.writerow([name, span.count - 1, f"{ms:.3f}"])

    @contextmanager
    def span(self, name):
        """with stats.span("holistic"): ... 로 구간을 잽니다."""
        if not self.enabled:
            yield
            return
        t0 = clock.now()
        try:
            yield
        finally:
            self.add(name, (clock.now() - t0) * 1000.0)

    def since(self, name, t0):
        """clock.now() 로 잰 시작 시각 t0 부터의 경과를 기록 (time.time() 과 섞지 않음)."""
        self.add(name, (clock.now() - t0) * 1000.0)

    def details(self):
        """{이름: {count, max, p50, p95, exact, recent_n, recent_p50, recent_p95}}

        count·max 는 실행 전체의 정확한 값입니다. p50·p95 는 exact 가 False 면
        상대 오차 ±REL_ERROR 이내의 근사입니다. recent_* 는 최근 recent_n 개의
        정확한 값입니다.
        """
        out = {}
        with self._lock:
            for k, v in self._d.items():
                if not v.count:
                    continue
                p50, exact = v.whole_run(50)
                p95, _ = v.whole_run(95)
                out[k] = {"count": v.count, "max": v.max, "p50": p50, "p95": p95,
                          "exact": exact, "recent_n": len(v.recent),
                          "recent_p50": _rank(v.recent, 50),
                          "recent_p95": _rank(v.recent, 95)}
        return out

    def summary(self):
        """{이름: (n, p50, p95, max)} — 실행 전체 기준 (근사 여부는 details())."""
        return {k: (v["count"], v["p50"], v["p95"], v["max"])
                for k, v in self.details().items()}

    def report(self, title="지연 측정 결과"):
        d = self.details()
        if not d:
            return "지연 표본이 없습니다. (--stats 없이 돌렸거나 너무 짧게 실행)"

        keys = [k for k in ORDER if k in d] + sorted(k for k in d if k not in ORDER)
        w = max(_w(LABEL.get(k, k)) for k in keys) + 2
        total = w + 53
        el = clock.now() - self.t_start

        def ms(v, approx=False):
            return f"{v:8.1f}ms" + ("≈" if approx else " ")

        lines = ["", "=" * total, f"  {title}   (실행 {el:.0f}초)", "=" * total,
                 "  " + _pad("구간", w) + _rpad("전체 표본", 9) + _rpad("p50", 12)
                 + _rpad("p95", 12) + _rpad("최대", 11) + _rpad("최근 p95", 11),
                 "  " + "-" * (total - 2)]
        for k in keys:
            v = d[k]
            approx = not v["exact"]
            lines.append("  " + _pad(LABEL.get(k, k), w) + f"{v['count']:>9}"
                         + ms(v["p50"], approx) + ms(v["p95"], approx)
                         + ms(v["max"]) + ms(v["recent_p95"]))
        lines.append("=" * total)
        lines.append(f"  p50·p95·최대: 실행 전체. 최근 p95: 구간별 마지막 {self.maxlen}개 표본의 정확한 값.")
        if any(not v["exact"] for v in d.values()):
            lines.append(f"  ≈: 표본이 {self.maxlen}개를 넘어 히스토그램으로 근사한 값 "
                         f"(nearest-rank 기준 상대 오차 ±{REL_ERROR * 100:.0f}% 이내).")
        if self.log_path is not None:
            lines.append(f"  전체 표본 원본: {self.log_path}")

        # 실시간 여부 판정: 수어 경로 한 바퀴가 프레임 간격 안에 들어오는가
        if "sign_total" in d:
            v = d["sign_total"]
            budget = 1000.0 / 30
            mark = "≈" if not v["exact"] else ""
            lines.append(
                f"  수어 경로 p95 {v['p95']:.1f}ms{mark} / 30fps 예산 {budget:.1f}ms"
                + ("  -> 여유 있음" if v["p95"] < budget else
                   "  -> 초과. 해상도나 model_complexity 를 낮추세요"))
        if "end_to_end" in d:
            v = d["end_to_end"]
            mark = "≈" if not v["exact"] else ""
            lines.append(f"  수어->소리 전체: 중앙값 {v['p50']:.0f}ms{mark}, "
                         f"p95 {v['p95']:.0f}ms{mark} (표본 {v['count']}회)")
        lines.append("")
        return "\n".join(lines)

    def save_csv(self, path):
        """표본 CSV 를 저장하고 경로를 돌려줍니다. 표본이 없으면 None.

        log_path 와 같은 경로면 전체 로그를 닫고 그 파일을 돌려줍니다. 그 밖의
        경로에는 구간별 최근 표본만 쓰며, index 는 실행 전체 기준 순번이라
        앞부분이 빠졌는지 알 수 있습니다.
        """
        p = Path(path)
        with self._lock:
            if self.log_path is not None and p.resolve() == self.log_path.resolve():
                self._close_log()
                return self.log_path if self._d else None
            d = {k: (v.count - len(v.recent), list(v.recent))
                 for k, v in self._d.items() if v.count}
        if not d:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = [k for k in ORDER if k in d] + sorted(k for k in d if k not in ORDER)
        with p.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["span", "index", "ms"])
            for k in keys:
                first, values = d[k]
                for i, v in enumerate(values, first):
                    wr.writerow([k, i, f"{v:.3f}"])
        return p

    def close(self):
        with self._lock:
            self._close_log()

    def _close_log(self):
        if self._log is not None:
            self._log.close()
            self._log = self._writer = None


class NullTracker:
    """--stats 없이 돌 때. 호출부에 if 문을 안 넣으려고 둡니다."""

    enabled = False
    log_path = None

    def add(self, name, ms):
        pass

    @contextmanager
    def span(self, name):
        yield

    def since(self, name, t0):
        pass

    def details(self):
        return {}

    def summary(self):
        return {}

    def report(self, title=""):
        return ""

    def save_csv(self, path):
        return None

    def close(self):
        pass
