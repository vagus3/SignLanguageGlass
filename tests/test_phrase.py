import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys, time
from src.predictor import PhraseBuilder

def drive(words_with_gaps, window=2.5, flush=1.0):
    """(단어, 앞 단어로부터의 간격초) 리스트를 가상 시계로 재생."""
    pb = PhraseBuilder(window_sec=window, flush_sec=flush)
    now = [1000.0]
    real = time.time
    time_mod = sys.modules["src.predictor"].time
    time_mod.time = lambda: now[0]
    spoken = []
    try:
        for w, gap in words_with_gaps:
            for _ in range(int(gap * 20)):       # 50ms 씩 진행하며 poll
                now[0] += 0.05
                s = pb.poll()
                if s: spoken.append(s)
            s = pb.feed(w)
            if s: spoken.append(s)
        for _ in range(int(4.0 * 20)):           # 마지막 4초 마무리
            now[0] += 0.05
            s = pb.poll()
            if s: spoken.append(s)
    finally:
        time_mod.time = real
    return spoken

cases = [
    ("이름 -> 1.5초 후 무엇  (사람이 보통 하는 속도)",
     [("이름", 0.0), ("무엇", 1.5)], ["이름이 뭐예요?"]),
    ("이름 -> 0.5초 후 무엇  (빠르게)",
     [("이름", 0.0), ("무엇", 0.5)], ["이름이 뭐예요?"]),
    ("이름 -> 3.0초 후 무엇  (너무 느림, 조합 포기)",
     [("이름", 0.0), ("무엇", 3.0)], ["이름", "무엇"]),
    ("어디 -> 1.8초 후 병원",
     [("어디", 0.0), ("병원", 1.8)], ["병원이 어디예요?"]),
    ("감사합니다 단독 (규칙 앞부분 아님 -> 빨리 발화)",
     [("감사합니다", 0.0)], ["감사합니다"]),
    ("얼마 단독 (1단어 규칙 -> 즉시)",
     [("얼마", 0.0)], ["얼마예요?"]),
]
bad = 0
for name, seq, want in cases:
    got = drive(seq)
    ok = got == want
    bad += not ok
    print(f"  {'OK ' if ok else 'FAIL'} {name}\n         -> {got}" + ("" if ok else f"   (기대 {want})"))
print("\n실패", bad)
sys.exit(1 if bad else 0)
