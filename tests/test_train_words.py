"""Word-clip training: cropping, composed sentences, augmentation, held-out signers."""
import _path  # noqa
import sys
try:
    import torch
except ImportError:
    sys.exit(77)
import json
import tempfile
from pathlib import Path
import numpy as np
from src.continuous_sign import Recognizer
from src.pose_sequence import JOINT_NAMES, VERSION, PoseSequence
from src.continuous_sign import CtcRecognizer
from src.train_words import (Composer, augment, cut, load_words, rest_pieces, rest_rows,
                             score, sign_piece, train, word_rows)

torch.set_num_threads(1)
rng = np.random.default_rng(0)


def clip(label, signer, view="F", frames=120):
    """4 s clip; the sign occupies 1.5..2.5 s. Labels differ in where the hand moves."""
    t = np.arange(frames) / 30
    p = np.full((frames, 49, 2), 0.5, np.float32)
    p += rng.normal(0, 0.002, p.shape)
    active = (t >= 1.5) & (t <= 2.5)
    direction = {"가": (0.2, 0.0), "나": (0.0, 0.2)}[label]
    p[active, :42] += np.outer(np.sin((t[active] - 1.5) * np.pi), direction)[:, None, :]
    return {"id": f"{label}_{signer}_{view}", "signer_id": signer, "session_id": "fixture",
            "group_id": f"{label}_{signer}", "glosses": [label], "view": view,
            "continuous": False, "sign_intervals": [[1.5, 2.5]],
            "sequence": PoseSequence(t, p, np.ones((frames, 49), bool), (1920, 1080))}


with tempfile.TemporaryDirectory() as td:
    manifests = []
    for signer in ("s1", "s2", "s3"):
        folder = Path(td) / signer
        folder.mkdir()
        rows = []
        for i, label in enumerate(("가", "나")):
            row = clip(label, signer)
            row.pop("sequence").save(folder / f"clip_{i}.npz")
            rows.append({**row, "file": f"clip_{i}.npz"})
        (folder / "manifest.json").write_text(json.dumps(
            {"version": VERSION, "joint_names": JOINT_NAMES, "clips": rows,
             "sources": [{"id": "fixture", "name": "fixture"}]}, ensure_ascii=False),
            encoding="utf-8")
        manifests.append(folder / "manifest.json")
    rows, sources = load_words(manifests)
    assert len(rows) == 6 and len(sources) == 1, "sources are merged, not repeated"
    try:
        load_words(manifests + manifests[:1])
        raise AssertionError("The same bundle was accepted twice")
    except ValueError:
        pass

    # Cropping keeps the sign with a margin and restarts time at zero.
    t, p, m = sign_piece(rows[0], None)
    assert t[0] == 0 and abs(t[-1] - 1.2) < 0.05 and len(t) == len(p) == len(m)
    assert cut(rows[0]["sequence"], 3.99, 4.0) is None, "fewer than two frames"
    # Wide edges keep the raising and lowering of the hands, as a device segment does.
    lengths = [sign_piece(rows[0], rng, wide=(True, True))[0][-1] for _ in range(40)]
    assert max(lengths) > 2.5 and min(lengths) >= 1.0 - 0.04
    rests = rest_pieces(rows[0])
    assert len(rests) == 2 and all(piece[0][-1] < 1.4 for piece in rests)
    assert all(r["glosses"] == [] for r in rest_rows(rows)) and len(word_rows(rows)) == 6

    # Augmented sequences stay valid; points pushed out of the image become invisible.
    for _ in range(50):
        sequence = augment(rows[0]["sequence"], rng)
        assert 2 <= len(sequence.timestamps) <= 120
        assert (sequence.coordinates[~sequence.mask] == 0).all()

    # Composed examples: 0..3 words, never the same word twice in a row, one signer each.
    composer = Composer(rows)
    seen = set()
    for i in range(200):
        item = composer.sample(rng, i)
        glosses = item["glosses"]
        assert len(glosses) <= 3 and all(a != b for a, b in zip(glosses, glosses[1:]))
        seen.add(len(glosses))
    assert seen == {0, 1, 2, 3}, seen

    # A real sentence clip ("가 나", signs at 1.0..1.6 and 1.8..2.4 s) is used whole, and
    # its words are also cut out one by one without reaching into the neighbouring sign.
    sentence = {**clip("가", "s1"), "id": "sentence", "glosses": ["가", "나"],
                "sign_intervals": [[1.0, 1.6], [1.8, 2.4]]}
    sentence.update(labels=["가", "나"], label=None)
    mixed = Composer(rows + [sentence], sentence_rate=1.0, negative=0)
    cut_out = mixed.singles[("s1", "F")]["가"][-1]
    assert cut_out["sign_intervals"] == [[1.0, 1.6]] and abs(cut_out["room"][1] - 0.1) < 1e-9
    assert all(sign_piece(cut_out, rng)[0][-1] <= 0.6 + 0.3 + 0.1 + 0.04 for _ in range(30))
    whole = [mixed.sample(rng, i)["glosses"] for i in range(200)]
    assert ["가", "나"] in whole and any(len(g) == 3 for g in whole), "sentence, sentence + word"
    assert len(word_rows(rows + [sentence])) == 6, "sentences are not scored as single words"
    assert score(CtcRecognizer(2), rows + [sentence], ["<blank>", "가", "나"])[
        "real_sentence_wer"] is not None

    # Held-out signer: never used for training, and good enough on this separable fixture.
    train_rows = [r for r in rows if r["signer_id"] != "s3"]
    val_rows = [r for r in rows if r["signer_id"] == "s3"]
    labels = ["<blank>", "가", "나"]
    lines = []
    model, history, best = train(train_rows, labels, val_rows=val_rows, epochs=80, batch=8,
                                 samples=64, log=lines.append)
    assert len(history) == len(lines) == 80 and history[-1]["loss"] < history[0]["loss"]
    assert best["word_accuracy"] == 1.0, best
    assert best["clips"] == 2 and set(best["word_accuracy_by_view"]) == {"F"}
    result = score(model, val_rows, labels)
    assert result["word_accuracy"] == best["word_accuracy"], "the best epoch is what is kept"

    # Without held-out signers the last epoch is kept and nothing is reported.
    model, history, best = train(rows, labels, epochs=1, batch=8, samples=16,
                                 log=lambda line: None)
    assert best is None and "word_accuracy" not in history[0]

    from src.continuous_sign import save_checkpoint
    path = Path(td) / "ctc.pt"
    save_checkpoint(path, model, labels, metadata={"continuous": False,
                                                   "synthetic_sentences": True})
    loaded = Recognizer(path)
    assert loaded.metadata["continuous"] is False and loaded.labels == labels
    loaded.predict(word_rows(rows)[0]["sequence"])

print("[OK] Word clips: cropping, rest negatives, augmentation, composed sentences, "
      "held-out signer scoring, checkpoint")
