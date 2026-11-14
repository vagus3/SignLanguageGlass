"""낱개 단어 클립으로 가변 길이 인식기(CTC)를 학습합니다.

python -m src.train_words data/processed/aihub_words/*/manifest.json --out models/words_v1

tools/prepare_aihub_words.py 가 만든 묶음(수어자 한 명씩)을 여러 개 받습니다.
src.train_continuous 는 낱개 단어를 거부합니다(문장이 아니므로). 여기서는 대신

- 단어 클립의 수어 구간(sign_intervals)만 잘라 1~3개를 이어 붙인 합성 문장으로 학습하고,
- 클립 앞뒤의 준비·복귀 동작을 '수어 아님'(빈 정답) 예시로 쓰고,
- 다른 사람·다른 카메라에서도 되도록 크기, 위치, 기울기, 속도, 프레임 수를 매번 바꿉니다.

이어 붙인 문장에는 실제 연속 수어의 단어 사이 변형이 없습니다. 결과 모델은
metadata.continuous=false, synthetic_sentences=true 로 표시됩니다.

기본은 모든 수어자를 학습에 씁니다. --val-signers 로 몇 명을 빼면 그 사람들로
'처음 보는 수어자' 정확도를 잽니다(최종 모델을 만들기 전 확인용).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.continuous_sign import CtcRecognizer, collate, save_checkpoint
from src.data_sources import attribution, manifest_sources
from src.pose_sequence import FEATURE_SETS, MAX_SECONDS, VERSION, PoseSequence
from src.train_continuous import evaluate

FRAME = 1 / 30
MIN_REST = 0.5          # '수어 아님' 예시로 쓸 최소 길이(초)
REST_GUARD = 0.15       # 수어 구간과 '수어 아님' 예시 사이에 두는 간격(초)


def load_words(manifests):
    """모든 묶음의 클립을 한 목록으로. 수어 구간이 없는 클립은 쓰지 않습니다."""
    rows, sources = [], []
    for manifest_path in manifests:
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("version") != VERSION:
            raise ValueError(f"{path}: {VERSION} 묶음만 지원합니다")
        for source in manifest_sources(manifest):
            if source not in sources:
                sources.append(source)
        for row in manifest["clips"]:
            if len(row.get("glosses") or []) != 1 or not row.get("sign_intervals"):
                raise ValueError(f"{row.get('id')}: 단어 하나와 수어 구간이 필요합니다")
            rows.append({**row, "label": row["glosses"][0],
                         "sequence": PoseSequence.load(path.parent / row["file"])})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("클립 ID가 겹칩니다 (같은 묶음을 두 번 넣었는지 확인하세요)")
    return rows, sources


def cut(sequence, start, end):
    """[start, end] 초 구간의 (시각, 좌표, 마스크). 시각은 0부터 다시 셉니다."""
    t = sequence.timestamps - sequence.timestamps[0]
    keep = (t >= start - 1e-9) & (t <= end + 1e-9)
    if keep.sum() < 2:
        return None
    return t[keep] - t[keep][0], sequence.coordinates[keep], sequence.mask[keep]


def sign_piece(row, rng, margin=0.3):
    """수어 구간과 앞뒤 여백(0~margin 초). rng 가 None 이면 여백은 고정 0.1초입니다."""
    start, end = row["sign_intervals"][0][0], row["sign_intervals"][-1][1]
    before, after = (0.1, 0.1) if rng is None else rng.uniform(0, margin, 2)
    return cut(row["sequence"], start - before, end + after)


def rest_pieces(row):
    """수어 구간 밖의 앞·뒤 조각 (준비, 복귀 동작). 짧으면 빠집니다."""
    start, end = row["sign_intervals"][0][0], row["sign_intervals"][-1][1]
    total = row["sequence"].timestamps[-1] - row["sequence"].timestamps[0]
    spans = [(0.0, start - REST_GUARD), (end + REST_GUARD, total)]
    pieces = [cut(row["sequence"], a, b) for a, b in spans if b - a >= MIN_REST]
    return [p for p in pieces if p is not None]


def join(pieces, image_size):
    """조각들을 한 시퀀스로 잇습니다. 조각 사이는 한 프레임 간격입니다."""
    times, offset = [], 0.0
    for t, _, _ in pieces:
        times.append(t + offset)
        offset += t[-1] + FRAME
    return PoseSequence(np.concatenate(times), np.concatenate([p[1] for p in pieces]),
                        np.concatenate([p[2] for p in pieces]), image_size)


def augment(sequence, rng):
    """같은 동작을 다른 사람·다른 카메라가 찍은 것처럼 바꿉니다.

    크기(카메라와의 거리), 위치, 기울기, 속도(빠른 수어), 프레임 수(느린 기기),
    손이 잠깐 안 잡히는 경우, 좌표 떨림. 화면 밖으로 나간 점은 '안 보임'이 됩니다.
    """
    t, p, m = sequence.timestamps, sequence.coordinates.astype(np.float64), sequence.mask.copy()
    w, h = sequence.image_size
    pixels = p * [w, h]
    center = pixels[m].mean(axis=0) if m.any() else np.array([w / 2, h / 2])
    angle = np.deg2rad(rng.uniform(-8, 8))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = rng.uniform(0.6, 1.25)
    target = np.array([rng.uniform(0.3, 0.7) * w, center[1] + rng.uniform(-0.15, 0.15) * h])
    pixels = (pixels - center) @ rotation.T * scale + target
    pixels += rng.normal(0, 0.002 * w, pixels.shape)
    p = pixels / [w, h]
    m &= ((p >= 0) & (p <= 1)).all(axis=-1)
    # 한쪽 손이 몇 프레임 동안 안 잡히는 경우 (MediaPipe 에서 흔합니다).
    if rng.random() < 0.3 and len(t) > 8:
        hand = 21 * int(rng.integers(2))
        first = int(rng.integers(len(t) - 6))
        m[first:first + int(rng.integers(2, 7)), hand:hand + 21] = False
    # 프레임 수: 30fps 원본을 12~30fps 로 솎아냅니다.
    keep = rng.random(len(t)) < rng.uniform(0.4, 1.0)
    keep[[0, -1]] = True
    t = (t - t[0]) * rng.uniform(0.75, 1.3)             # 속도
    if t[-1] > MAX_SECONDS:
        t *= MAX_SECONDS / t[-1]
    return PoseSequence(t[keep], np.where(m[..., None], p, 0)[keep], m[keep],
                        sequence.image_size).validate()


class Composer:
    """학습 예시를 매번 새로 만듭니다: 1~3단어 합성 문장, 가끔 '수어 아님'."""

    def __init__(self, rows, *, negative=0.12, lengths=(0.5, 0.3, 0.2), phrases=(),
                 phrase_rate=0.5):
        """phrases: 이 상황에서 나올 법한 문장들의 단어열. phrase_rate 만큼은 무작위 조합
        대신 이 문장들로 학습해, 모델이 자주 함께 나오는 단어를 익히게 합니다."""
        self.rows, self.negative, self.lengths = rows, negative, lengths
        self.phrases, self.phrase_rate = [list(p) for p in phrases], phrase_rate
        self.groups = defaultdict(list)      # 같은 수어자, 같은 카메라 방향끼리만 잇습니다
        for row in rows:
            self.groups[(row["signer_id"], row.get("view"))].append(row)
        self.keys = sorted(self.groups, key=str)

    def sample(self, rng, index=0):
        group = self.groups[self.keys[int(rng.integers(len(self.keys)))]]
        size = group[0]["sequence"].image_size
        if rng.random() < self.negative:
            rests = rest_pieces(group[int(rng.integers(len(group)))])
            if rests:
                piece = rests[int(rng.integers(len(rests)))]
                return {"id": f"syn{index}", "glosses": [],
                        "sequence": augment(join([piece], size), rng)}
        wanted = None
        if self.phrases and rng.random() < self.phrase_rate:
            by_label = defaultdict(list)
            for row in group:
                by_label[row["label"]].append(row)
            phrase = self.phrases[int(rng.integers(len(self.phrases)))]
            if all(label in by_label for label in phrase):
                wanted = [by_label[label][int(rng.integers(len(by_label[label])))]
                          for label in phrase]
        if wanted is None:
            count = 1 + int(rng.choice(len(self.lengths), p=self.lengths))
            wanted = [group[int(rng.integers(len(group)))] for _ in range(count)]
        chosen, pieces = [], []
        for row in wanted:
            if chosen and chosen[-1] == row["label"]:
                continue        # 같은 단어 반복은 실제 발화에 드뭅니다
            piece = sign_piece(row, rng)
            if piece is not None:
                chosen.append(row["label"])
                pieces.append(piece)
        if not pieces:
            return self.sample(rng, index)
        return {"id": f"syn{index}", "glosses": chosen,
                "sequence": augment(join(pieces, size), rng)}


def word_rows(rows):
    """평가용: 클립마다 수어 구간만 자른 예시 하나 (변형 없음)."""
    out = []
    for row in rows:
        piece = sign_piece(row, None)
        if piece is not None:
            out.append({"id": row["id"], "glosses": [row["label"]], "view": row.get("view"),
                        "sequence": join([piece], row["sequence"].image_size)})
    return out


def rest_rows(rows):
    """평가용: 준비·복귀 동작. 여기서 단어가 나오면 잘못된 인식입니다."""
    return [{"id": f"{row['id']}#rest{i}", "glosses": [],
             "sequence": join([piece], row["sequence"].image_size)}
            for row in rows for i, piece in enumerate(rest_pieces(row))]


def score(model, rows, labels, device="cpu"):
    """처음 보는 수어자에 대한 단어 정확도, 합성 문장 오류율, 잘못된 인식 비율."""
    words = evaluate(model, word_rows(rows), labels, device)["clips"]
    views = {r["id"]: r.get("view") for r in rows}
    by_view, wrong = defaultdict(list), defaultdict(int)
    for clip in words:
        ok = clip["prediction"] == clip["reference"]
        by_view[views[clip["id"]]].append(ok)
        if not ok:
            wrong[(clip["reference"][0], " ".join(clip["prediction"]) or "(없음)")] += 1
    rng = np.random.default_rng(1234)
    composer = Composer(rows, negative=0)
    sentences = evaluate(model, [composer.sample(rng, i) for i in range(300)], labels, device)
    rests = evaluate(model, rest_rows(rows), labels, device)["clips"]
    correct = [ok for flags in by_view.values() for ok in flags]
    return {"word_accuracy": float(np.mean(correct)) if correct else None,
            "word_accuracy_by_view": {str(v): float(np.mean(f)) for v, f in sorted(
                by_view.items(), key=lambda item: str(item[0]))},
            "sentence_wer": sentences["wer"],
            "false_output_rate": (float(np.mean([bool(c["prediction"]) for c in rests]))
                                  if rests else None),
            "clips": len(words), "rest_clips": len(rests),
            "most_confused": [{"reference": a, "prediction": b, "count": n} for (a, b), n in
                              sorted(wrong.items(), key=lambda item: -item[1])[:15]]}


def train(rows, labels, *, val_rows=(), epochs=40, batch=16, samples=None, seed=0,
          device="cpu", feature_set="2d", log=print, phrases=()):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CtcRecognizer(len(labels), feature_set=feature_set).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    composer = Composer(rows, phrases=phrases)
    samples = samples or len(rows)
    steps = epochs * -(-samples // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=steps)
    history, best, best_state = [], None, None
    for epoch in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for start in range(0, samples, batch):
            items = [composer.sample(rng, start + i) for i in range(min(batch, samples - start))]
            x, lengths, targets, target_lengths = collate(items, labels, feature_set)
            optimizer.zero_grad()
            loss = criterion(model(x.to(device), lengths), targets, lengths, target_lengths)
            if not torch.isfinite(loss):
                raise ValueError("손실이 유한하지 않습니다; 자료를 확인하세요")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            schedule.step()
            total, batches = total + float(loss.detach()), batches + 1
        entry = {"epoch": epoch + 1, "loss": total / batches}
        if val_rows:
            result = score(model, val_rows, labels, device)
            entry.update({k: result[k] for k in ("word_accuracy", "sentence_wer",
                                                 "false_output_rate")})
            if best is None or result["word_accuracy"] > best["word_accuracy"]:
                best = {**result, "epoch": epoch + 1}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(entry)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in entry.items()))
    if best_state is not None:      # 확인용 수어자가 있으면 가장 잘 맞힌 시점의 모델을 씁니다
        model.load_state_dict(best_state)
    return model.cpu(), history, best


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifests", nargs="+", help="prepare_aihub_words 결과의 manifest.json 들")
    p.add_argument("--out", required=True, help="새로 만들 결과 폴더")
    p.add_argument("--val-signers", default="",
                   help="학습에서 빼고 확인에 쓸 수어자 (예: REAL15,REAL16). 기본: 없음")
    p.add_argument("--phrases", help="상황에서 나올 법한 문장 목록 JSON "
                                     '([{"glosses": [...], "text": ...}]). 학습 예시의 절반을 '
                                     "이 문장들로 만듭니다")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--samples", type=int, help="한 에폭의 합성 예시 수 (기본: 학습 클립 수)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        p.error("결과 폴더가 이미 있습니다; 새 이름을 쓰세요")
    try:
        rows, sources = load_words(args.manifests)
    except ValueError as error:
        p.error(str(error))
    held = {s for s in args.val_signers.split(",") if s}
    signers = sorted({r["signer_id"] for r in rows})
    if held - set(signers):
        p.error(f"없는 수어자: {sorted(held - set(signers))} (있는 수어자: {signers})")
    train_rows = [r for r in rows if r["signer_id"] not in held]
    val_rows = [r for r in rows if r["signer_id"] in held]
    labels = ["<blank>"] + sorted({r["label"] for r in train_rows})
    val_rows = [r for r in val_rows if r["label"] in labels]
    print(f"학습 {len(train_rows)}클립 / 수어자 {len(signers) - len(held)}명, "
          f"확인 {len(val_rows)}클립 / {len(held)}명, 단어 {len(labels) - 1}개")
    phrases = []
    if args.phrases:
        phrases = [entry["glosses"] for entry in
                   json.loads(Path(args.phrases).read_text(encoding="utf-8"))
                   if all(g in labels for g in entry["glosses"])]
        print(f"상황 문장 {len(phrases)}개를 학습 예시의 절반에 사용")
    model, history, best = train(train_rows, labels, val_rows=val_rows, epochs=args.epochs,
                                 batch=args.batch, samples=args.samples, seed=args.seed,
                                 device=args.device, phrases=phrases)
    out.mkdir(parents=True)
    metadata = {"continuous": False, "synthetic_sentences": True, "data_sources": sources,
                "train_signers": [s for s in signers if s not in held],
                "val_signers": sorted(held), "seed": args.seed,
                "trained_with_phrases": len(phrases)}
    save_checkpoint(out / "ctc.pt", model, labels, metadata=metadata)
    report = {**metadata, "labels": labels[1:], "history": history, "validation": best,
              "auto_speech_enabled": False,
              "limitations": ["합성 문장: 실제 연속 수어의 단어 사이 변형 없음",
                              "OpenPose 좌표로 학습; MediaPipe 실제 카메라 평가 필요"]}
    (out / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    for line in attribution(sources):
        print("데이터 출처:", line)
    if best:
        print(f"처음 보는 수어자: 단어 정확도 {best['word_accuracy']:.1%}, 합성 문장 오류율 "
              f"{best['sentence_wer']:.1%}, 잘못된 인식 {best['false_output_rate']:.1%} "
              f"(에폭 {best['epoch']})")
    print(f"저장: {out}/ctc.pt")


if __name__ == "__main__":
    main()
