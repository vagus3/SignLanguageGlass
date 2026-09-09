# -*- coding: utf-8 -*-
"""
Step 1-2. LSTM 학습  [경로 A: 직접 수집한 데이터 전용]

    pip install -r requirements-train.txt     # <- 이게 먼저 필요합니다
    python -m src.train
    python -m src.train --split group         # 촬영자 3명 이상일 때

※ 두 가지 학습 경로가 있습니다. 헷갈리지 마세요.
    경로 A (이 파일)  : collect.py 로 직접 모은 데이터, landmarks.py 의 246차원
    경로 B (AI Hub)   : 제공 키포인트, features_v2.py 의 101차원

  경로 B 스크립트는 AI Hub 실제 폴더 구조를 확인한 뒤에 만듭니다.
  구조를 모르고 짜면 파서가 안 맞아서 어차피 다시 씁니다.

학습이 끝나면 TorchScript 백업과 배포용 ONNX를 함께 만들고, 두 출력의
최대 오차까지 검사합니다. main.py 는 ONNX만 읽습니다.

CPU로 충분합니다. 15개 클래스 x 40시퀀스 규모면 노트북에서 1~3분입니다.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.landmarks import FEATURE_DIM, FEATURE_VERSION, N_FLAGS


# ------------------------------------------------------------------ 데이터
def load_dataset():
    """파일명 {촬영자}__{세션}_{번호}.npy 에서 촬영자 ID도 읽습니다."""
    from src.dataset_contract import ensure_dataset_contract, signer_from_stem
    try:
        ensure_dataset_contract(create=False)
    except RuntimeError as e:
        raise SystemExit(str(e))
    X, y, groups, labels = [], [], [], []
    for label in C.VOCAB:
        d = C.DATA_DIR / label
        files = sorted(d.glob("*.npy")) if d.exists() else []
        if not files:
            print(f"  [건너뜀] {label}: 데이터 없음")
            continue
        valid = []
        for f in files:
            a = np.load(f)
            if a.shape != (C.SEQ_LEN, FEATURE_DIM):
                print(f"  [경고] {f.name} shape {a.shape} 불일치 - 제외")
                continue
            valid.append((f, a))
        if not valid:
            print(f"  [건너뜀] {label}: 올바른 shape 데이터 없음")
            continue
        idx = len(labels)
        labels.append(label)
        for f, a in valid:
            X.append(a)
            y.append(idx)
            groups.append(signer_from_stem(f.stem))
        print(f"  {label:<12} {len(valid):>3} 시퀀스")
    if not X:
        raise SystemExit("데이터가 없습니다. 먼저 `python -m src.collect` 를 돌리세요.")
    return (np.stack(X).astype(np.float32), np.array(y),
            np.array(groups), labels)


# ------------------------------------------------------------------ 증강
def augment(X, y, times, rng):
    """
    작은 데이터셋에서 일반화를 살리는 3가지.
      1) 가우시안 노이즈  - 랜드마크 지터 모사
      2) 시간 왜곡        - 수어 속도 개인차 모사
      3) 스케일/이동      - 카메라 거리·위치 변화 모사
    좌우 반전은 넣지 않습니다. 수어는 주손(dominant hand)이 의미를 가져서
    반전시키면 다른 동작이 되어버립니다.

    ★ 좌표 구간에만 적용합니다.
      벡터 끝 3칸은 검출 플래그(0/1)이고, 손이 안 잡힌 프레임은 해당 좌표
      63칸이 정확히 0 으로 채워져 있습니다. 여기까지 흔들면
        - 추론 때는 0.0/1.0 만 들어오는데 학습 때는 1.04 같은 값을 보게 되고
        - '손 없음'을 뜻하던 0 이 미세한 잡음으로 바뀌어 의미가 사라지며
        - 학습 때만 존재하는 가짜 미세 움직임을 모델이 외울 수 있습니다.
      둘 다 에러 없이 정확도만 갉아먹는 종류라 구간을 나눠서 처리합니다.
    """
    Xs, ys = [X], [y]
    n, T, D = X.shape
    coord = slice(0, D - N_FLAGS)          # 좌표 구간만 흔듭니다
    for _ in range(times):
        a = X.copy()
        c = a[:, :, coord]
        zero = c == 0.0                    # 미검출로 0 채워진 자리 (나중에 복원)

        c += rng.normal(0, 0.01, c.shape).astype(np.float32)              # 1
        c *= rng.uniform(0.93, 1.07, (n, 1, 1)).astype(np.float32)        # 3
        c += rng.uniform(-0.03, 0.03, (n, 1, c.shape[2])).astype(np.float32)
        c[zero] = 0.0

        for i in range(n):                                                # 2
            src = np.sort(rng.choice(T, T, replace=True))
            a[i] = a[i][src]               # 플래그도 프레임 단위로 같이 따라갑니다
        Xs.append(a)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


# ------------------------------------------------------------------ 모델
def build_model(n_classes):
    """TensorFlow/tf2onnx의 protobuf 충돌을 피하는 PyTorch LSTM."""
    import torch.nn as nn

    class SignLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm1 = nn.LSTM(FEATURE_DIM, 64, batch_first=True)
            self.lstm2 = nn.LSTM(64, 128, batch_first=True)
            self.lstm3 = nn.LSTM(128, 64, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.fc1 = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, 32)
            self.out = nn.Linear(32, n_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.drop(x)
            x, _ = self.lstm2(x)
            x = self.drop(x)
            x, _ = self.lstm3(x)
            x = self.drop(x[:, -1, :])
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.out(x)               # logits; runtime이 softmax 적용

    return SignLSTM()


def fit_model(model, Xtr, ytr, Xte, yte, class_weights, epochs, batch, seed):
    """작은 데이터셋용 CPU 학습 루프. 최고 검증 정확도 가중치를 복원합니다."""
    import copy
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다")
    if batch < 1:
        raise ValueError("batch는 1 이상이어야 합니다")

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long())
    loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=gen)

    weight = torch.ones(model.out.out_features, dtype=torch.float32)
    for i, w in class_weights.items():
        weight[i] = w
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-5)
    vx, vy = torch.from_numpy(Xte), torch.from_numpy(yte).long()

    best_acc, best_state, stale = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach()) * len(xb)
            seen += len(xb)

        model.eval()
        with torch.no_grad():
            vlogits = model(vx)
            val_loss = float(loss_fn(vlogits, vy))
            val_acc = float((vlogits.argmax(1) == vy).float().mean())
        scheduler.step(val_loss)

        if val_acc > best_acc + 1e-8:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(f"epoch {epoch:03d}  loss {train_loss/max(seen,1):.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}")
        if stale >= 40:
            print(f"조기 종료: 검증 정확도가 {stale} epoch 동안 개선되지 않았습니다.")
            break

    model.load_state_dict(best_state)
    return best_acc


def export_models(model, pt_path, onnx_path):
    """TorchScript와 ONNX를 저장하고 같은 입력에서 출력 일치를 검증합니다."""
    import torch

    model = model.cpu().eval()
    dummy = torch.zeros(1, C.SEQ_LEN, FEATURE_DIM, dtype=torch.float32)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(str(pt_path))
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # sidecar를 잃어도 로짓에 softmax를 정확히 적용할 수 있게 모델 안에도 기록.
    import onnx
    graph = onnx.load(str(onnx_path))
    prop = graph.metadata_props.add()
    prop.key, prop.value = "output_kind", "logits"
    onnx.save(graph, str(onnx_path))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    probes = [
        np.zeros((1, C.SEQ_LEN, FEATURE_DIM), np.float32),
        rng.normal(size=(1, C.SEQ_LEN, FEATURE_DIM)).astype(np.float32),
    ]
    max_err = 0.0
    for probe in probes:
        with torch.no_grad():
            ref = model(torch.from_numpy(probe)).numpy()
        got = sess.run(None, {sess.get_inputs()[0].name: probe})[0]
        if not np.isfinite(ref).all() or not np.isfinite(got).all():
            raise RuntimeError("ONNX 변환 검증 중 NaN/Inf 출력이 발생했습니다")
        err = float(np.max(np.abs(ref - got)))
        max_err = max(max_err, err)
        if err > 1e-4 or int(ref.argmax()) != int(got.argmax()):
            raise RuntimeError(f"ONNX 변환 출력 불일치: max error={err:.3e}")
    return max_err


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--augment", type=int, default=3, help="증강 배수 (0=끄기)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="auto", choices=["auto", "random", "group"],
                    help="group=화자 단위(진짜 성능) / random=시퀀스 단위(과대평가)")
    a = ap.parse_args()

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    import torch

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    print("데이터 로드")
    X, y, groups, labels = load_dataset()
    signers = sorted(set(groups))
    print(f"  -> {X.shape}, 클래스 {len(labels)}개, 촬영자 {signers}")

    # ★ 분할 방식이 정확도 숫자의 의미를 통째로 바꿉니다.
    #
    #   시퀀스 단위 무작위 분할(random)
    #     같은 사람의 거의 동일한 동작이 학습/테스트 양쪽에 들어갑니다.
    #     "val_accuracy 0.97" 이 나와도 다른 사람이 하면 0.3 일 수 있습니다.
    #
    #   화자 단위 분할(group)
    #     한 사람을 통째로 테스트셋으로 뺍니다. 이게 진짜 성능입니다.
    #     발표에 쓸 숫자는 반드시 이쪽이어야 합니다.
    use_group = (a.split == "group") or (a.split == "auto" and len(signers) >= 2)
    if use_group:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=a.seed)
        tr, te = next(gss.split(X, y, groups))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        print(f"  [화자 단위 분할] 테스트 촬영자: {sorted(set(groups[te]))}")
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=a.seed, stratify=y)
        if len(signers) < 2:
            print("  [경고] 촬영자가 1명뿐이라 시퀀스 단위로 분할했습니다.\n"
                  "         이 정확도는 과대평가입니다. 촬영자를 3명 이상 모으고\n"
                  "         --split group 으로 다시 측정하세요.")
    if a.augment:
        Xtr, ytr = augment(Xtr, ytr, a.augment, rng)
        print(f"  증강 후 학습셋 {Xtr.shape}")

    # ★ enumerate 로 키를 매기면 안 됩니다.
    #   화자 단위로 분할하면 특정 라벨이 학습셋에서 통째로 빠질 수 있습니다.
    #   그러면 np.unique(ytr) 가 [0,1,3,4] 처럼 되어 3번 클래스의 가중치가
    #   2번에게 붙습니다. 에러는 안 나고 학습만 조용히 틀어집니다.
    present = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=present, y=ytr)
    cw = {int(k): float(w) for k, w in zip(present, weights)}
    missing = [i for i in range(len(labels)) if i not in cw]
    if missing:
        raise SystemExit(
            f"학습셋에 없는 클래스: {[labels[i] for i in missing]}\n"
            f"이 상태로는 해당 클래스를 절대 학습할 수 없습니다. 촬영자별로 모든\n"
            f"클래스를 수집한 뒤 다시 실행하세요.")

    model = build_model(len(labels))
    print(model)
    best_acc = fit_model(model, Xtr, ytr, Xte, yte, cw,
                         a.epochs, a.batch, a.seed)

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = C.MODEL_DIR / "sign_lstm.pt"
    onnx_path = C.MODEL_PATH
    err = export_models(model, pt_path, onnx_path)
    C.LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # 전처리 사양을 모델 옆에 남깁니다.
    # runtime.SignRuntime 이 이 파일을 읽어 "학습 때 만든 차원"과 "지금 로드한
    # 모델이 기대하는 차원"을 대조합니다. 없으면 그 검증이 통째로 꺼지고,
    # 변환된 ONNX와 같은 stem으로 저장해 runtime이 자동으로 찾습니다.
    spec_path = C.MODEL_DIR / f"{onnx_path.stem}_preproc.json"
    spec_path.write_text(json.dumps({
        "note": "src/train.py 가 학습 시점에 기록. 손으로 고치지 마세요.",
        "feature": "landmarks.extract (pose15 + hand21x2 + face24 + flag3)",
        "feature_version": FEATURE_VERSION,
        "input_dim": FEATURE_DIM,
        "seq_len": C.SEQ_LEN,
        "output_kind": "logits",
        "labels": labels,
        "normalize": "body=어깨 기준, hands=손목/손바닥 기준+손목 위치, face=눈 간격 기준",
        "target_fps": C.TRAIN_FPS,
        "mirror": False,
        "signers": sorted(set(groups.tolist())),
        "split": "group" if use_group else "random",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(1).numpy()
    print("\n=== 분류 리포트 ===")
    class_ids = np.arange(len(labels))
    print(classification_report(yte, pred, labels=class_ids,
                                target_names=labels, zero_division=0))
    print("=== 혼동 행렬 ===")
    cm = confusion_matrix(yte, pred, labels=class_ids)
    print("      " + " ".join(f"{i:>3}" for i in range(len(labels))))
    for i, row in enumerate(cm):
        print(f"{i:>2} {labels[i][:4]:<4} " + " ".join(f"{v:>3}" for v in row))

    print(f"\n저장: {onnx_path} (배포) / {pt_path.name} (백업)")
    print(f"      {C.LABELS_PATH.name} / {spec_path.name}")
    print(f"ONNX 출력 일치 최대오차: {err:.3e}")
    print(f"최고 val_accuracy: {best_acc:.3f}")
    print("\n혼동 행렬에서 서로 헷갈리는 쌍이 보이면, 모델을 키우지 말고")
    print("그 두 어휘 중 하나를 손 모양이 더 다른 단어로 교체하세요.")


if __name__ == "__main__":
    main()
