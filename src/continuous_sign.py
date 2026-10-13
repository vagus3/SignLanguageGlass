"""Variable-length pose encoder and CTC recognizer. No camera/vLLM dependency."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.pose_sequence import (PoseSequence, VERSION, VERSION_3D, VERSIONS, FEATURE_SETS,
                                needs_world)
from src.ctc_runtime import check_calibration, check_contract, greedy_decode  # noqa: F401


class CtcRecognizer(nn.Module):
    def __init__(self, classes, hidden=128, feature_set="2d"):
        super().__init__()
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        self.feature_set = feature_set
        self.encoder = nn.Sequential(nn.Linear(FEATURE_SETS[feature_set][1], hidden),
                                     nn.LayerNorm(hidden), nn.GELU())
        self.temporal = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True,
                                dropout=0.1, batch_first=True)
        self.output = nn.Linear(hidden * 2, classes)

    def forward(self, x, lengths):
        encoded = self.encoder(x)
        packed = pack_padded_sequence(encoded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed, _ = self.temporal(packed)
        decoded, _ = pad_packed_sequence(packed, batch_first=True,
                                         total_length=x.shape[1])
        return self.output(decoded).log_softmax(-1).transpose(0, 1)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j-1] + (x != y)))
        prev = row
    return prev[-1]


def load_dataset(manifest_path):
    """Only gloss-labelled clips train CTC; text-only rows are reported, not invented."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") not in VERSIONS:
        raise ValueError("Dataset version mismatch")
    rows, skipped = [], []
    for row in manifest["clips"]:
        if row.get("glosses") is None:
            skipped.append({"id": row["id"], "reason": "No ordered gloss annotation"})
            continue
        glosses = row["glosses"]
        if not isinstance(glosses, list) or any(not isinstance(g, str) or not g.strip()
                                               or g == "<blank>" for g in glosses):
            raise ValueError("glosses must be an ordered list of nonempty labels")
        for key in ("signer_id", "session_id", "group_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{row.get('id')}: missing {key}")
        sequence = PoseSequence.load(path.parent / row["file"])
        if sequence.version != manifest["version"]:
            raise ValueError(f"{row['id']}: clip version differs from manifest")
        minimum = len(glosses) + sum(a == b for a, b in zip(glosses, glosses[1:]))
        if len(sequence.timestamps) < minimum:
            raise ValueError(f"{row['id']}: too few frames for CTC target")
        rows.append({**row, "sequence": sequence})
    if not rows:
        raise ValueError("No gloss-labelled clips available")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate clip ID")
    return rows, skipped


def split_dataset(rows, seed=0):
    """Union signer and original motion groups before assigning any splits."""
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        for key in ("signer_id", "group_id"):
            token = (key, row[key])
            if token in owners:
                parent[root(i)] = root(owners[token])
            else:
                owners[token] = i
    components = {}
    for i in range(len(rows)):
        components.setdefault(root(i), []).append(i)
    assigned = [r.get("split") for r in rows]
    if any(assigned):
        if any(s not in ("train", "val", "test") for s in assigned):
            raise ValueError("Every row needs train/val/test if splits are explicit")
        for indices in components.values():
            if len({assigned[i] for i in indices}) != 1:
                raise ValueError("Signer or motion group crosses split boundary")
    else:
        if len(components) < 3:
            raise ValueError("Need at least 3 independent signer/motion components")
        groups = list(components.values())
        np.random.default_rng(seed).shuffle(groups)
        n_train = min(len(groups) - 2, max(1, int(len(groups) * 0.7)))
        n_val = max(1, int(len(groups) * 0.15))
        for j, indices in enumerate(groups):
            split = "train" if j < n_train else "val" if j < n_train+n_val else "test"
            for i in indices:
                assigned[i] = split
    result = {s: [r for r, chosen in zip(rows, assigned) if chosen == s]
              for s in ("train", "val", "test")}
    if not all(result.values()):
        raise ValueError("Train, validation and test must all be nonempty")
    return result


def collate(rows, vocabulary, feature_set="2d", cache=None):
    """cache: dict reused across epochs; features are deterministic per (clip, feature set)."""
    extract, dim = FEATURE_SETS[feature_set]
    if cache is None:
        arrays = [extract(r["sequence"]) for r in rows]
    else:
        arrays = []
        for r in rows:
            key = (r["id"], feature_set)
            if key not in cache:
                cache[key] = extract(r["sequence"])
            arrays.append(cache[key])
    lengths = torch.tensor([len(x) for x in arrays], dtype=torch.long)
    x = torch.zeros(len(rows), int(lengths.max()), dim)
    ids = {label: i for i, label in enumerate(vocabulary)}
    targets, target_lengths = [], []
    for i, (row, array) in enumerate(zip(rows, arrays)):
        x[i, :len(array)] = torch.from_numpy(array)
        targets.extend(ids[g] for g in row["glosses"])
        target_lengths.append(len(row["glosses"]))
    return x, lengths, torch.tensor(targets, dtype=torch.long), torch.tensor(target_lengths)


def save_checkpoint(path, model, labels, *, calibration=None, metadata=None):
    metadata = dict(metadata or {})
    if "model_id" not in metadata:
        digest = hashlib.sha256(json.dumps(labels).encode())
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        metadata["model_id"] = digest.hexdigest()[:16]
    feature_set = model.feature_set
    torch.save({"version": VERSION_3D if needs_world(feature_set) else VERSION,
                "feature_set": feature_set, "feature_dim": FEATURE_SETS[feature_set][1],
                "labels": labels,
                "hidden": model.temporal.hidden_size, "state_dict": model.state_dict(),
                "calibration": calibration, "metadata": metadata}, path)


class Recognizer:
    def __init__(self, path):
        # Only load trusted, locally trained checkpoints.
        data = torch.load(path, map_location="cpu", weights_only=True)
        # Checkpoints older than feature sets were always 2D.
        self.feature_set = data.get("feature_set", "2d")
        self.labels = data["labels"]
        check_contract(self.feature_set, data["version"], data["feature_dim"], self.labels)
        self.model = CtcRecognizer(len(self.labels), data["hidden"], self.feature_set)
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()
        self.calibration = data.get("calibration") or {}
        self.metadata = data.get("metadata", {})
        check_calibration(self.calibration)

    def predict(self, sequence):
        sequence.validate()
        if not sequence.mask[:, :42].any():
            return [], 0.0
        x = torch.from_numpy(FEATURE_SETS[self.feature_set][0](sequence))[None]
        with torch.inference_mode():
            probability = self.model(x, torch.tensor([x.shape[1]]))[:, 0].exp().numpy()
        return greedy_decode(probability, self.labels)
