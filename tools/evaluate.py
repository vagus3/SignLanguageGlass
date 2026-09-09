# -*- coding: utf-8 -*-
"""
저장된 모델을 홀드아웃셋으로 평가합니다.

    python tools/evaluate.py                          # models/sign_lstm.onnx
    python tools/evaluate.py models/sign_lstm.onnx
    python tools/evaluate.py --split group            # 화자 단위 (진짜 성능)
    python tools/evaluate.py --holdout me             # 특정 촬영자만 테스트셋으로
    python tools/evaluate.py --csv report/eval.csv

■ 왜 train.py 로는 부족한가
    train.py 는 학습이 끝난 직후 그 자리에서 리포트를 찍습니다. 그래서
      1) 숫자를 다시 뽑으려면 재학습해야 하고
      2) 학습 중의 PyTorch 모델만 평가합니다.

    실제로 안경에서 도는 건 ONNX 입니다. 변환 과정에서 조용히 틀어질 수
    있는데(LSTM 초기 상태, 동적 축), 그러면 학습 정확도는 그대로인데 실행만
    이상해집니다. **배포하는 바로 그 파일을 재는 것**이 이 스크립트의 목적입니다.

■ sklearn 을 안 씁니다
    지표를 전부 numpy 로 계산합니다. requirements.txt 만 깔린 환경
    (= 안경에 올라가는 그 환경) 에서 그대로 돌아가야 하기 때문입니다.
    라즈베리파이에서도 바로 실행됩니다.
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.landmarks import FEATURE_DIM


# ------------------------------------------------------------------ 데이터
def load_dataset(seq_len, feat_dim):
    """data/raw/{라벨}/{촬영자}__{세션}_{번호}.npy 를 전부 읽습니다."""
    from src.dataset_contract import ensure_dataset_contract, signer_from_stem
    try:
        ensure_dataset_contract(create=False)
    except RuntimeError as e:
        raise SystemExit(str(e))
    X, y, groups, labels = [], [], [], []
    skipped = 0
    for label in C.VOCAB:
        d = C.DATA_DIR / label
        files = sorted(d.glob("*.npy")) if d.exists() else []
        if not files:
            continue
        valid = []
        for f in files:
            a = np.load(f)
            if a.shape != (seq_len, feat_dim):
                skipped += 1
                continue
            valid.append((f, a))
        if not valid:
            continue
        idx = len(labels)
        labels.append(label)
        for f, a in valid:
            X.append(a)
            y.append(idx)
            groups.append(signer_from_stem(f.stem))
    if not X:
        raise SystemExit(
            f"{C.DATA_DIR} 에 (seq={seq_len}, dim={feat_dim}) 형태의 .npy 가 "
            f"없습니다.\n  먼저 `python -m src.collect` 로 수집하세요.")
    if skipped:
        print(f"  [경고] shape 이 안 맞아 제외한 파일 {skipped}개")
    return (np.stack(X).astype(np.float32), np.array(y),
            np.array(groups), labels)


def split(X, y, groups, mode, holdout, seed, test_size=0.25):
    """
    반환: (테스트 인덱스, 설명 문자열, 경고 문자열 또는 None)

    화자 단위 분할이 왜 중요한지는 train.py 주석과 동일합니다.
    시퀀스 단위로 섞으면 같은 사람의 거의 같은 동작이 학습/테스트 양쪽에
    들어가 정확도가 부풀려집니다.
    """
    rng = np.random.default_rng(seed)
    signers = sorted(set(groups.tolist()))

    if holdout:
        want = set(holdout)
        unknown = want - set(signers)
        if unknown:
            raise SystemExit(f"그런 촬영자가 없습니다: {sorted(unknown)}\n"
                             f"  있는 촬영자: {signers}")
        te = np.where(np.isin(groups, list(want)))[0]
        return te, f"화자 단위 (테스트 촬영자: {sorted(want)})", None

    if mode == "group" or (mode == "auto" and len(signers) >= 2):
        n_te = max(1, int(round(len(signers) * test_size)))
        pick = set(rng.permutation(signers)[:n_te].tolist())
        te = np.where(np.isin(groups, list(pick)))[0]
        warn = ("촬영자가 2명뿐입니다. 테스트 화자가 1명이라 분산이 큽니다.\n"
                "     숫자와 함께 '촬영자 N명' 을 반드시 같이 적으세요."
                if len(signers) == 2 else None)
        return te, f"화자 단위 (테스트 촬영자: {sorted(pick)})", warn

    # 시퀀스 단위: 같은 사람의 거의 같은 동작이 학습/테스트 양쪽에 들어갑니다.
    te = np.sort(rng.permutation(len(y))[:max(1, int(len(y) * 0.2))])
    if len(signers) < 2:
        warn = ("촬영자가 1명뿐이라 화자 단위로 나눌 수가 없습니다.\n"
                "     이 숫자는 발표에 쓰지 마세요. 촬영자를 3명 이상 모으고\n"
                "     --split group 으로 다시 측정하세요.")
    else:
        warn = ("촬영자가 여럿인데 --split random 을 쓰셨습니다.\n"
                "     같은 사람의 거의 같은 동작이 학습/테스트 양쪽에 들어갑니다.\n"
                "     진짜 성능은 --split group 입니다.")
    return te, "시퀀스 단위 ★ 과대평가된 숫자입니다", warn


# ------------------------------------------------------------------ 지표
def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def prf(cm):
    """클래스별 precision / recall / f1 / support."""
    tp = np.diag(cm).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)     # 그 클래스로 예측한 횟수
    true = cm.sum(axis=1).astype(np.float64)     # 실제 그 클래스인 횟수
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
        r = np.where(true > 0, tp / np.maximum(true, 1), 0.0)
        f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return p, r, f, true.astype(np.int64)


def _w(s):
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s, n):
    return s + " " * max(0, n - _w(s))


def _trunc(s, n):
    """표시 폭 기준으로 자릅니다. 한글은 한 글자가 두 칸이라 [:6] 은 못 씁니다."""
    out, w = "", 0
    for c in s:
        cw = _w(c)
        if w + cw > n:
            break
        out += c
        w += cw
    return out


def report(y_true, y_pred, conf, labels):
    k = len(labels)
    cm = confusion(y_true, y_pred, k)
    p, r, f, sup = prf(cm)
    acc = float((y_true == y_pred).mean())

    w = max(max(_w(l) for l in labels), 8) + 2
    print("\n=== 클래스별 성능 ===")
    print("  " + _pad("어휘", w) + f"{'precision':>11}{'recall':>9}{'f1':>8}{'표본':>7}")
    print("  " + "-" * (w + 35))
    for i, l in enumerate(labels):
        mark = "  <- 주의" if sup[i] and r[i] < 0.6 else ""
        print("  " + _pad(l, w) + f"{p[i]:>11.3f}{r[i]:>9.3f}{f[i]:>8.3f}{sup[i]:>7}"
              + mark)
    print("  " + "-" * (w + 35))
    valid = sup > 0
    print("  " + _pad("macro 평균", w)
          + f"{p[valid].mean():>11.3f}{r[valid].mean():>9.3f}{f[valid].mean():>8.3f}"
          + f"{sup.sum():>7}")
    print(f"\n  정확도(accuracy) : {acc:.4f}   ({int((y_true==y_pred).sum())}/{len(y_true)})")

    # NEUTRAL 오출력: 가만히 있는데 단어를 뱉는 비율. 시연 체감에 제일 큽니다.
    if "NEUTRAL" in labels:
        ni = labels.index("NEUTRAL")
        m = y_true == ni
        if m.any():
            fa = float((y_pred[m] != ni).mean())
            print(f"  NEUTRAL 오출력   : {fa:.4f}  "
                  f"(무동작인데 단어를 뱉은 비율 — 낮을수록 좋습니다)")

    print("\n=== 혼동 행렬 (행=실제, 열=예측) ===")
    print(" " * 13 + " ".join(f"{i:>3}" for i in range(k)))
    for i in range(k):
        print(f"{i:>3} " + _pad(_trunc(labels[i], 8), 9)
              + " ".join(f"{v:>3}" for v in cm[i]))

    # 가장 헷갈리는 쌍
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.sum():
        pairs = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
        print("\n=== 가장 헷갈리는 쌍 ===")
        shown = 0
        for a, b in pairs:
            if off[a, b] == 0 or shown >= 5:
                break
            print(f"  {labels[a]} -> {labels[b]} : {off[a, b]}회")
            shown += 1
        print("  ※ 모델을 키우지 말고, 이 중 하나를 손 모양이 더 다른 단어로 "
              "교체하는 게 훨씬 효과적입니다.")

    # 신뢰도 임계값 스윕: CONF_THRESHOLD 를 정하는 근거
    print(f"\n=== 신뢰도 임계값별 (현재 설정 {C.CONF_THRESHOLD}) ===")
    print(f"  {'임계값':>8}{'통과율':>10}{'통과분 정확도':>16}")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = conf >= th
        cover = float(m.mean())
        a = float((y_true[m] == y_pred[m]).mean()) if m.any() else float("nan")
        star = "  <- 현재" if abs(th - C.CONF_THRESHOLD) < 1e-6 else ""
        print(f"  {th:>8.2f}{cover:>10.3f}{a:>16.3f}{star}")
    print("  ※ 임계값을 올리면 오출력은 줄지만 반응이 안 나오는 구간이 늘어납니다.")
    return acc, cm


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None,
                    help="기본값 config.MODEL_PATH (models/sign_lstm.onnx)")
    ap.add_argument("--split", default="auto",
                    choices=["auto", "random", "group"],
                    help="group=화자 단위(진짜 성능) / random=시퀀스 단위(과대평가)")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="테스트셋으로 뺄 촬영자 ID (예: --holdout kim lee)")
    ap.add_argument("--all", action="store_true",
                    help="분할 없이 전체 데이터로 평가 (학습셋 포함 — 참고용)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None, help="예측 결과를 CSV 로 저장")
    a = ap.parse_args()

    from src.runtime import SignRuntime

    model_path = Path(a.model or C.MODEL_PATH)
    if not model_path.exists():
        raise SystemExit(f"모델이 없습니다: {model_path}\n"
                         f"  학습/ONNX 생성:  python -m src.train")

    labels_json = None
    if C.LABELS_PATH.exists():
        labels_json = json.loads(C.LABELS_PATH.read_text(encoding="utf-8"))

    print("=" * 62)
    print(f"  모델 평가: {model_path.name}")
    print("=" * 62)
    rt = SignRuntime(model_path, labels=labels_json)

    seq_len = int(rt.spec.get("seq_len") or C.SEQ_LEN)
    feat_dim = int(rt.input_dim or rt.spec.get("input_dim") or FEATURE_DIM)
    print(f"  입력 사양: seq={seq_len}, dim={feat_dim}, "
          f"클래스={rt.n_classes or '?'}")

    X, y, groups, labels = load_dataset(seq_len, feat_dim)
    print(f"  데이터: {X.shape}, 클래스 {len(labels)}개, "
          f"촬영자 {sorted(set(groups.tolist()))}")

    if rt.n_classes and int(rt.n_classes) != len(labels):
        raise SystemExit(
            f"모델 출력 클래스 {rt.n_classes}개와 평가 데이터 클래스 {len(labels)}개가 "
            f"다릅니다.\n모델과 같은 labels.json/data 세트를 사용하세요.")
    if rt.labels and list(rt.labels) != labels:
        raise SystemExit(
            f"labels.json 순서와 평가 데이터 어휘 순서가 다릅니다.\n"
            f"  labels.json: {list(rt.labels)}\n"
            f"  데이터     : {labels}\n"
            f"이 상태의 정확도는 의미가 없으므로 평가를 중단합니다.")

    if a.all:
        te, how = np.arange(len(y)), "전체 (학습셋 포함 — 진짜 성능 아님)"
        warn = "학습에 쓴 데이터가 섞여 있습니다. 참고용 숫자입니다."
    else:
        te, how, warn = split(X, y, groups, a.split, a.holdout, a.seed)
    print(f"  분할: {how}")
    print(f"  테스트 표본 {len(te)}개")
    if warn:
        print(f"  ※ {warn}")

    # ---- 추론 ----
    y_true, y_pred, y_conf, ms = y[te], [], [], []
    for i in te:
        p = rt.predict(X[i][None, ...])
        y_pred.append(int(p.argmax()))
        y_conf.append(float(p.max()))
        ms.append(rt.last_ms)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)
    ms = np.array(ms)

    acc, cm = report(y_true, y_pred, y_conf, labels)

    print(f"\n=== 추론 속도 ({rt.backend}) ===")
    print(f"  중앙값 {np.median(ms):.2f}ms   p95 {np.percentile(ms, 95):.2f}ms")
    budget = C.STRIDE / float(getattr(C, "TRAIN_FPS", 30)) * 1000
    print(f"  추론 간격 예산 {budget:.0f}ms ({C.STRIDE}프레임마다)"
          + ("  -> 여유 있음" if np.median(ms) < budget * 0.5 else "  -> 빠듯합니다"))

    if a.csv:
        import csv as _csv
        out = Path(a.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            wr = _csv.writer(f)
            wr.writerow(["index", "signer", "true", "pred", "conf", "correct", "ms"])
            for j, i in enumerate(te):
                wr.writerow([int(i), groups[i], labels[y_true[j]],
                             labels[y_pred[j]], f"{y_conf[j]:.4f}",
                             int(y_true[j] == y_pred[j]), f"{ms[j]:.3f}"])
        print(f"\n  예측 결과 저장: {out}")

    print("\n" + "=" * 62)
    print(f"  결론: {how}  ->  정확도 {acc:.3f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
