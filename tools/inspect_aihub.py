# -*- coding: utf-8 -*-
"""
AI Hub 수어 데이터셋 스캐너

    python tools/inspect_aihub.py <데이터_루트>
    python tools/inspect_aihub.py D:/aihub_sign --manifest

무엇을 알려주는가
    1. 각도(view)가 몇 개나 받아졌는가   <- 시점 일반화의 핵심 자산
    2. 영상 / 형태소 JSON / 키포인트 JSON 중 뭐가 있는가
    3. 키포인트가 OpenPose 형식이 맞는가, 좌표계는 무엇인가
    4. 어휘가 몇 종이고 클래스당 클립이 몇 개인가  <- 학습 가능 여부를 가름
    5. 촬영자(언어제공자)가 몇 명인가             <- 화자 단위 평가 가능 여부

AI Hub 전체는 53만 클립이라 보통 일부만 받게 됩니다.
"내가 받은 게 정확히 뭔지"를 먼저 아는 게 설계의 출발점입니다.
"""
import sys
import json
import argparse
import random
import csv
from pathlib import Path
from collections import Counter, defaultdict

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mts"}
KEYPOINT_HINT = ("hand_left_keypoints", "hand_right_keypoints",
                 "pose_keypoints", "face_keypoints", "people")
MORPH_HINT = ("metaData", "attributes", "sign_script", "start", "end")


# ------------------------------------------------------------------ 유틸
def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def schema(obj, depth=0, max_depth=4):
    """JSON 구조를 타입 요약으로 축약해서 보여줍니다."""
    pad = "  " * depth
    if depth >= max_depth:
        return f"{pad}..."
    if isinstance(obj, dict):
        out = []
        for k in list(obj.keys())[:12]:
            v = obj[k]
            if isinstance(v, (dict, list)):
                out.append(f"{pad}{k}:")
                out.append(schema(v, depth + 1, max_depth))
            else:
                s = repr(v)
                out.append(f"{pad}{k}: {type(v).__name__} = "
                           f"{s[:60]}{'...' if len(s) > 60 else ''}")
        if len(obj) > 12:
            out.append(f"{pad}... (+{len(obj) - 12} keys)")
        return "\n".join(out)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[] (빈 리스트)"
        if all(isinstance(x, (int, float)) for x in obj):
            return f"{pad}[숫자 {len(obj)}개]  예: {obj[:6]}"
        return f"{pad}[{len(obj)}개]\n" + schema(obj[0], depth + 1, max_depth)
    return f"{pad}{type(obj).__name__}"


def classify_json(path):
    """이 JSON 이 키포인트인지 형태소인지 판별."""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")[:4000]
    except Exception:
        return "unknown", None
    kind = "unknown"
    if any(h in txt for h in KEYPOINT_HINT):
        kind = "keypoint"
    elif any(h in txt for h in MORPH_HINT):
        kind = "morpheme"
    try:
        return kind, json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return kind, None


# ------------------------------------------------------------------ 스캔
def scan(root):
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"경로가 없습니다: {root}")

    videos, jsons, others = [], [], Counter()
    total_bytes = 0
    print(f"스캔 중: {root}  (파일이 많으면 몇 분 걸립니다)")
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass
        if ext in VIDEO_EXT:
            videos.append(p)
        elif ext == ".json":
            jsons.append(p)
        else:
            others[ext or "(확장자없음)"] += 1
    return root, videos, jsons, others, total_bytes


def guess_views(paths, root):
    """
    각도(view) 후보를 찾습니다.
    AI Hub 는 5각도 동시 촬영이라 보통 F/L/R/U/D 나 view1..5 같은 토큰이
    폴더명에 들어갑니다.

    파일명까지 넣으면 clip_001, clip_002 가 전부 다른 각도로 잡히므로
    (1) 디렉터리 조각만 보고 (2) 끝의 일련번호를 떼어내 묶습니다.
    """
    import re
    tokens = Counter()
    for p in paths[:20000]:
        for t in p.relative_to(root).parts[:-1]:      # 파일명 제외
            tl = t.lower()
            if any(k in tl for k in ("view", "cam", "angle", "front", "side",
                                     "top", "left", "right", "down", "up")) \
                    or re.fullmatch(r"[flrud]\d?", tl):
                base = re.sub(r"[_-]?\d+$", "", t)    # 끝 일련번호 제거
                tokens[base or t] += 1
    return tokens


# ------------------------------------------------------------------ 리포트
def report(root, videos, jsons, others, total_bytes, want_manifest):
    line = "=" * 62
    print(f"\n{line}\n  AI Hub 수어 데이터셋 스캔 결과\n{line}")
    print(f"  루트      : {root}")
    print(f"  총 용량   : {human(total_bytes)}")
    print(f"  영상      : {len(videos):,} 개")
    print(f"  JSON      : {len(jsons):,} 개")
    if others:
        print(f"  기타      : {dict(others.most_common(6))}")

    if not videos and not jsons:
        raise SystemExit("\n영상도 JSON 도 없습니다. 경로를 다시 확인하세요.")

    # ---------------- 각도 ----------------
    print(f"\n── 1. 각도(view) ──────────────────────────────────")
    tk = guess_views(videos or jsons, root)
    if tk:
        for t, c in tk.most_common(10):
            print(f"  {t:<24} {c:>7,}")
        print(f"\n  각도 후보 {len(tk)}종 발견.")
        if len(tk) < 3:
            print("  ※ AI Hub 는 5각도 동시 촬영입니다. 한두 각도만 받으셨다면\n"
                  "     나머지를 추가로 받으세요. 착용 시점 일반화에 가장 큰 도움이 됩니다.")
    else:
        print("  경로에서 각도 토큰을 못 찾았습니다.")
        print("  폴더 구조를 직접 확인해 각도가 나뉘어 있는지 보세요.")

    # ---------------- JSON 종류 ----------------
    print(f"\n── 2. JSON 종류 ───────────────────────────────────")
    kinds = Counter()
    samples = {}
    for p in random.sample(jsons, min(len(jsons), 120)) if jsons else []:
        k, obj = classify_json(p)
        kinds[k] += 1
        if k not in samples and obj is not None:
            samples[k] = (p, obj)
    for k, c in kinds.most_common():
        print(f"  {k:<12} {c:>4} / {min(len(jsons), 120)} 표본")

    for k, (p, obj) in samples.items():
        print(f"\n  [{k}] 예시 구조  ({p.relative_to(root)})")
        print(schema(obj, depth=2, max_depth=7))

    # ---------------- 키포인트 형식 검증 ----------------
    print(f"\n── 3. 키포인트 형식 ───────────────────────────────")
    if "keypoint" in samples:
        _, obj = samples["keypoint"]
        node = obj
        if isinstance(obj, dict) and "people" in obj:
            ppl = obj["people"]
            node = ppl[0] if isinstance(ppl, list) and ppl else ppl
        found = False
        if isinstance(node, dict):
            for key in ("hand_left_keypoints_2d", "hand_right_keypoints_2d",
                        "pose_keypoints_2d", "face_keypoints_2d"):
                if key in node:
                    v = node[key]
                    n = len(v) // 3 if isinstance(v, list) else "?"
                    print(f"  {key:<26} 값 {len(v):>5}개 -> 점 {n}개 (x,y,conf)")
                    found = True
        if found:
            print("\n  => OpenPose 형식입니다. MediaPipe 와 토폴로지가 다릅니다.")
            print("     손 21점은 순서까지 동일하지만, 포즈(25 vs 33)와")
            print("     얼굴(70 vs 468)은 호환되지 않습니다.")
            print("     src/features_v2.py 의 공통 부분집합을 쓰세요.")
        else:
            print("  예상한 OpenPose 필드를 못 찾았습니다. 위 구조를 보고 알려주세요.")
    else:
        print("  키포인트 JSON 이 표본에 없습니다. 영상만 받으셨다면")
        print("  MediaPipe 로 직접 추출해야 합니다 (재추출 경로).")

    # ---------------- 어휘 분포 ----------------
    print(f"\n── 4. 어휘 분포 ───────────────────────────────────")
    vocab = Counter()
    signers = set()
    rows = []
    scan_n = min(len(jsons), 3000)
    for p in (random.sample(jsons, scan_n) if jsons else []):
        k, obj = classify_json(p)
        if k != "morpheme" or obj is None:
            continue
        for w, sid in iter_morphemes(obj):
            vocab[w] += 1
            if sid:
                signers.add(sid)
            rows.append({"file": str(p.relative_to(root)), "word": w,
                         "signer": sid or ""})

    if vocab:
        print(f"  표본 {scan_n}개 JSON 에서 어휘 {len(vocab):,}종 발견")
        print(f"  상위 15개: ")
        for w, c in vocab.most_common(15):
            print(f"    {w:<16} {c:>5}")
        few = sum(1 for c in vocab.values() if c < 20)
        print(f"\n  클립 20개 미만인 어휘: {few:,} / {len(vocab):,}종")
        print("  ※ 클래스당 최소 20~30클립은 있어야 학습이 됩니다.")
        print("     3000종 전부를 쓰려 하지 말고, 시연 어휘 15개를 먼저 고른 뒤")
        print("     그 어휘의 클립만 추려서 시작하세요.")
    else:
        print("  형태소 JSON 에서 어휘를 파싱하지 못했습니다.")
        print("  위 [morpheme] 구조를 보고 iter_morphemes() 를 맞춰주세요.")

    if signers:
        print(f"\n  촬영자 후보 {len(signers)}명: {sorted(signers)[:10]}")
        print("  화자 단위 평가(GroupSplit)가 가능합니다. 반드시 이걸로 측정하세요.")

    # ---------------- 영상 속성 ----------------
    if videos:
        print(f"\n── 5. 영상 속성 (표본 5개) ────────────────────────")
        try:
            import cv2
            for p in random.sample(videos, min(5, len(videos))):
                c = cv2.VideoCapture(str(p))
                w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = c.get(cv2.CAP_PROP_FPS)
                n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
                c.release()
                dur = n / fps if fps else 0
                print(f"  {p.name:<38} {w}x{h} {fps:.0f}fps "
                      f"{n}프레임 {dur:.1f}초")
        except ImportError:
            print("  opencv 가 없어 건너뜁니다.")

    if want_manifest and rows:
        out = Path("data/manifest_aihub.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            wcsv = csv.DictWriter(f, fieldnames=["file", "word", "signer"])
            wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"\n매니페스트 저장: {out}  ({len(rows):,} 행)")

    print(f"\n{line}")
    print("  다음: 이 출력을 그대로 붙여넣어 주시면 학습 파이프라인을")
    print("        실제 구조에 맞춰 짜드립니다.")
    print(line)


def iter_morphemes(obj):
    """
    형태소 JSON 에서 (단어, 촬영자ID) 를 뽑습니다.
    AI Hub 공개 예시 기준이며, 실제 파일 구조가 다르면 여기만 고치면 됩니다.
      {"0": {"metaData": {...}, "data": [{"attributes":[{"name":"화장실"}]}, ...]}}
    """
    def walk(node, signer=None):
        if isinstance(node, dict):
            meta = node.get("metaData")
            if isinstance(meta, dict):
                signer = (meta.get("signer") or meta.get("performer")
                          or meta.get("name") or signer)
            attrs = node.get("attributes")
            if isinstance(attrs, list):
                for a in attrs:
                    if isinstance(a, dict) and a.get("name"):
                        yield str(a["name"]), signer
            for v in node.values():
                if isinstance(v, (dict, list)):
                    yield from walk(v, signer)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v, signer)

    yield from walk(obj)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="AI Hub 데이터를 풀어놓은 최상위 폴더")
    ap.add_argument("--manifest", action="store_true",
                    help="data/manifest_aihub.csv 로 저장")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    report(*scan(a.root), want_manifest=a.manifest)
