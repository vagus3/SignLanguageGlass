"""Train CTC only on annotated continuous clips; keep held-out test untouched."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.continuous_sign import (CtcRecognizer, collate, load_dataset,
                                  split_dataset, save_checkpoint, greedy_decode, edit_distance)
from src.pose_sequence import FEATURE_SETS, VERSION_3D, needs_world


def fit(model, rows, labels, *, epochs=30, batch=8, seed=0, device="cpu"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    history = []
    for epoch in range(epochs):
        total = 0
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), batch):
            items = [rows[i] for i in order[start:start+batch]]
            x, lengths, targets, target_lengths = collate(items, labels, model.feature_set)
            optimizer.zero_grad()
            output = model(x.to(device), lengths)
            loss = criterion(output, targets.to(device), lengths, target_lengths)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite CTC loss; inspect target lengths/data")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach())
        history.append(total)
        print(f"epoch {epoch+1}/{epochs}: loss_sum={total:.4f}")
    return history


def evaluate(model, rows, labels, device="cpu"):
    model.eval()
    errors, words, results = 0, 0, []
    with torch.inference_mode():
        for row in rows:
            x, lengths, _, _ = collate([row], labels, model.feature_set)
            p = model(x.to(device), lengths)[:, 0].exp().cpu().numpy()
            prediction, score = greedy_decode(p, labels)
            errors += edit_distance(row["glosses"], prediction)
            words += len(row["glosses"])
            results.append({"id": row["id"], "reference": row["glosses"],
                            "prediction": prediction, "score": score})
    return {"wer": errors / words if words else None, "errors": errors,
            "reference_words": words, "clips": results}


def prepare_training(manifest, seed=0, feature_sets=("2d",)):
    """Rows, splits and vocabulary shared by every configuration trained on a manifest."""
    rows, skipped = load_dataset(manifest)
    for name in feature_sets:
        if needs_world(name) and rows[0]["sequence"].version != VERSION_3D:
            raise ValueError(f"--features {name} needs world coordinates; "
                             "prepare data with 3D keypoints")
    splits = split_dataset(rows, seed)
    if any(not r.get("continuous", False) for r in splits["train"]):
        raise ValueError("Continuous training requires continuous=true annotations; "
                         "isolated words are not sentences")
    labels = ["<blank>"] + sorted({g for r in splits["train"] for g in r["glosses"]})
    if len(labels) < 2:
        raise ValueError("Training vocabulary is empty")
    unknown = {g for s in ("val", "test") for r in splits[s] for g in r["glosses"]} - set(labels)
    if unknown:
        raise ValueError(f"Held-out labels absent from training vocabulary: {sorted(unknown)}")
    return skipped, splits, labels


def train_one(splits, labels, feature_set, out, *, epochs=30, batch=8, seed=0, device="cpu",
              skipped=()):
    torch.manual_seed(seed)
    model = CtcRecognizer(len(labels), feature_set=feature_set)
    history = fit(model, splits["train"], labels, epochs=epochs,
                  batch=batch, seed=seed, device=device)
    validation = evaluate(model, splits["val"], labels, device)
    report = {"feature_set": feature_set, "validation": validation, "skipped": list(skipped),
              "history": history,
              "splits": {s: [r["id"] for r in clips] for s, clips in splits.items()},
              "auto_speech_enabled": False, "test_evaluated": False}
    out = Path(out)
    out.mkdir(parents=True)
    save_checkpoint(out / "ctc.pt", model.cpu(), labels,
                    metadata={"splits": report["splits"], "split_seed": seed,
                              "continuous": True})
    (out / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--features", choices=sorted(FEATURE_SETS), default="2d",
                   help="3d* needs a pose3d-sequence-v2 manifest")
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1:
        p.error("epochs and batch must be positive")
    out = Path(args.out)
    if out.exists():
        p.error("Output exists; choose a new directory")
    try:
        skipped, splits, labels = prepare_training(args.manifest, args.seed, [args.features])
    except ValueError as error:
        p.error(str(error))
    train_one(splits, labels, args.features, out, epochs=args.epochs, batch=args.batch,
              seed=args.seed, device=args.device, skipped=skipped)
    print(f"Saved {out}/ctc.pt; automatic speech remains disabled until calibration")


if __name__ == "__main__":
    main()
