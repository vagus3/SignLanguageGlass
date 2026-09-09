# -*- coding: utf-8 -*-
"""
전체 테스트 실행

    python tests/run_all.py
    python tests/run_all.py -v          # 각 테스트의 출력까지 보기
    python tests/run_all.py adaptive    # 이름에 'adaptive' 가 들어간 것만

pytest 를 쓰지 않습니다. requirements.txt 만 깔린 환경(= 안경에 올라가는 그
환경)에서 그대로 돌아가야 하기 때문입니다. 라즈베리파이에서도 바로 됩니다.

각 테스트는 독립 프로세스로 돌립니다. 여러 테스트가 config 를 임시로
갈아끼우기 때문에(DATA_DIR, ADAPTIVE_WINDOW 등) 한 프로세스에서 몰아 돌리면
서로 오염됩니다.
"""
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

# 한글 폰트가 있는 환경에서만 의미 있는 테스트
OPTIONAL = {
    "test_hud_real": "실제 한글 TTF 필요 (macOS 시스템 폰트 사용)",
}


def main():
    verbose = "-v" in sys.argv
    pats = [a for a in sys.argv[1:] if not a.startswith("-")]

    files = sorted(p for p in HERE.glob("test_*.py"))
    if pats:
        files = [p for p in files if any(x in p.stem for x in pats)]
    if not files:
        print("실행할 테스트가 없습니다.")
        return 1

    print("=" * 60)
    print(f"  signglass 테스트  ({len(files)}개)")
    print("=" * 60)

    ok = skip = fail = 0
    failures = []
    for p in files:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(p)], cwd=str(ROOT),
                           capture_output=True, text=True)
        el = (time.perf_counter() - t0) * 1000
        if r.returncode == 0:
            ok += 1
            print(f"  PASS  {p.stem:<20} {el:>7.0f}ms")
        elif r.returncode == 77:
            skip += 1
            reason = (r.stdout or r.stderr).strip().splitlines()
            reason = reason[-1] if reason else "선택 의존성 없음"
            print(f"  SKIP  {p.stem:<20} {reason}")
        elif p.stem in OPTIONAL:
            skip += 1
            print(f"  SKIP  {p.stem:<20} {OPTIONAL[p.stem]}")
        else:
            fail += 1
            failures.append((p.stem, r.stdout, r.stderr))
            print(f"  FAIL  {p.stem:<20} {el:>7.0f}ms")
        if verbose and r.stdout and r.returncode != 77:
            print("\n".join("        " + l for l in r.stdout.splitlines()))

    for name, out, err in failures:
        print("\n" + "-" * 60)
        print(f"  {name} 실패 내용")
        print("-" * 60)
        print((out or "")[-2000:])
        print((err or "")[-2000:])

    print("=" * 60)
    print(f"  통과 {ok} / 건너뜀 {skip} / 실패 {fail}")
    print("=" * 60)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
