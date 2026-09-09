# -*- coding: utf-8 -*-
"""
AI Hub 분할압축 병합기 (Windows / macOS / Linux 공용, WSL 불필요)

    python tools/merge_parts.py <다운로드폴더>            # 미리보기
    python tools/merge_parts.py <다운로드폴더> --run       # 실제 병합
    python tools/merge_parts.py <폴더> --run --extract    # 병합 후 압축해제까지
    python tools/merge_parts.py <폴더> --run --delete-parts

■ AI Hub 가 안내하는 리눅스 명령
    find "폴더" -name "파일명.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "파일명.zip"

  핵심은 sort 의 -V (버전 정렬) 입니다.
  일반 정렬은 문자열 순이라  part1, part10, part11, part2 ...  로 섞이고,
  그 순서로 이어붙이면 파일이 조용히 깨집니다.
  AI Hub 안내문의 "병합된 파일 용량이 0" 이나 압축 해제 실패가 대개 이것입니다.

  이 스크립트는 part 뒤 숫자를 정수로 파싱해 정렬하므로
  자릿수가 섞여 있어도(part2 / part010) 항상 올바른 순서가 됩니다.
"""
import re
import sys
import zipfile
import argparse
from pathlib import Path
from collections import defaultdict

PART_RE = re.compile(r"^(?P<base>.+?)\.part(?P<num>\d+)$", re.IGNORECASE)
CHUNK = 8 * 1024 * 1024      # 8MB 씩. 통째로 읽으면 메모리가 터집니다


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def find_groups(root: Path):
    """
    .partNN 파일들을 원본 파일명 기준으로 묶습니다.
    반환: {합칠_경로: [(번호, 조각경로), ...]}
    """
    groups = defaultdict(list)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        m = PART_RE.match(p.name)
        if m:
            groups[p.parent / m.group("base")].append((int(m.group("num")), p))

    out = {}
    for target, parts in groups.items():
        parts.sort(key=lambda t: t[0])       # ★ 정수 정렬 (-V 와 동일)
        out[target] = parts
    return out


def check(parts):
    """조각 번호가 연속인지 확인. 빠진 게 있으면 병합해봐야 깨집니다."""
    nums = [n for n, _ in parts]
    lo, hi = min(nums), max(nums)
    expected = set(range(lo, hi + 1))
    missing = sorted(expected - set(nums))
    dup = len(nums) != len(set(nums))
    return missing, dup, lo, hi


def merge(target: Path, parts, verbose=True):
    total = sum(p.stat().st_size for _, p in parts)
    done = 0
    with target.open("wb") as out:
        for i, (num, p) in enumerate(parts, 1):
            with p.open("rb") as f:
                while True:
                    buf = f.read(CHUNK)
                    if not buf:
                        break
                    out.write(buf)
                    done += len(buf)
            if verbose:
                pct = done / total * 100 if total else 100
                print(f"\r  [{i}/{len(parts)}] {pct:5.1f}%  "
                      f"{human(done)} / {human(total)}", end="", flush=True)
    if verbose:
        print()
    return total


def verify_zip(path: Path):
    """zip 무결성 검사. 순서가 틀리면 여기서 잡힙니다."""
    if path.suffix.lower() != ".zip":
        return None
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                return f"손상된 항목: {bad}"
            return f"정상 ({len(z.namelist()):,}개 항목)"
    except zipfile.BadZipFile as e:
        return f"BadZipFile: {e}"
    except Exception as e:
        return f"검사 실패: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="AI Hub 다운로드 폴더")
    ap.add_argument("--run", action="store_true", help="실제로 병합 (없으면 미리보기)")
    ap.add_argument("--extract", action="store_true", help="병합 후 압축 해제")
    ap.add_argument("--delete-parts", action="store_true",
                    help="검증 통과한 조각 삭제 (용량 확보)")
    ap.add_argument("--force", action="store_true", help="기존 결과물 덮어쓰기")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.exists():
        raise SystemExit(f"경로가 없습니다: {root}")

    groups = find_groups(root)
    if not groups:
        raise SystemExit(
            f"'.partNN' 파일을 찾지 못했습니다: {root}\n"
            "AI Hub 다운로드가 끝난 폴더를 지정했는지 확인하세요.")

    print("=" * 62)
    print(f"  분할압축 그룹 {len(groups)}개 발견")
    print("=" * 62)

    ok_groups = []
    for target, parts in sorted(groups.items()):
        size = sum(p.stat().st_size for _, p in parts)
        missing, dup, lo, hi = check(parts)
        print(f"\n  {target.name}")
        print(f"    조각 {len(parts)}개 (part{lo}~part{hi}), 합계 {human(size)}")
        if missing:
            print(f"    [중단] 빠진 조각: {missing[:20]}")
            print(f"           다시 다운로드하세요. 이대로 합치면 깨집니다.")
            continue
        if dup:
            print(f"    [중단] 중복 번호가 있습니다.")
            continue
        if target.exists() and not a.force:
            print(f"    [건너뜀] 결과물이 이미 있습니다 (--force 로 덮어쓰기)")
            continue
        print(f"    [준비됨]")
        ok_groups.append((target, parts))

    if not a.run:
        print(f"\n{'=' * 62}")
        print(f"  미리보기입니다. 실제 병합하려면 --run 을 붙이세요.")
        print(f"  병합 대상 {len(ok_groups)}개")
        print("=" * 62)
        return

    print(f"\n{'=' * 62}\n  병합 시작\n{'=' * 62}")
    for target, parts in ok_groups:
        print(f"\n{target.name}")
        merge(target, parts)

        res = verify_zip(target)
        if res:
            print(f"  무결성: {res}")
            if not res.startswith("정상"):
                print("  ※ 조각 순서나 다운로드가 잘못됐을 가능성이 큽니다.")
                continue

        if a.extract and target.suffix.lower() == ".zip":
            dest = target.with_suffix("")
            dest.mkdir(exist_ok=True)
            print(f"  압축 해제 -> {dest}")
            with zipfile.ZipFile(target) as z:
                z.extractall(dest)

        if a.delete_parts and (res is None or res.startswith("정상")):
            for _, p in parts:
                p.unlink()
            print(f"  조각 {len(parts)}개 삭제 (용량 확보)")

    print(f"\n{'=' * 62}")
    print("  완료. 다음:")
    print("    python tools/inspect_aihub.py <이 폴더>")
    print("=" * 62)


if __name__ == "__main__":
    main()
