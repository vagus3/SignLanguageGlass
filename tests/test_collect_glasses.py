"""Glasses collector: hand labels from MediaPipe Hands, targets and resume counts."""
import _path  # noqa
from types import SimpleNamespace as NS
import numpy as np
from src.collect_glasses import done_counts, from_hands, targets


def hand(x, label, score=0.9):
    marks = NS(landmark=[NS(x=x, y=0.5 + i * 0.01) for i in range(21)])
    return marks, NS(classification=[NS(label=label, score=score)])


def results(*hands):
    return NS(multi_hand_landmarks=[h for h, _ in hands], multi_handedness=[c for _, c in hands])


# Unmirrored glasses view: Hands says "Left" for the wearer's right hand (it assumes a selfie).
points, mask = from_hands(results(hand(0.7, "Left"), hand(0.3, "Right")))
assert mask[:42].all() and not mask[42:].any(), "hands only; body joints never visible"
assert np.allclose(points[0, 0], 0.3) and np.allclose(points[21, 0], 0.7), "left=0..20, right=21..41"
# A mirrored camera keeps the label and flips x back to the unmirrored image.
points, mask = from_hands(results(hand(0.7, "Left")), image_is_mirrored=True)
assert mask[:21].all() and not mask[21:].any() and np.allclose(points[0, 0], 0.3)
# Two detections called the same hand: the more confident one wins; points outside are hidden.
points, mask = from_hands(results(hand(0.2, "Left", 0.6), hand(1.4, "Left", 0.95)))
assert not mask.any(), "the confident detection lies outside the image"
assert not from_hands(results())[1].any()

words, reps = targets("words")
assert reps == 10 and len(words) >= 30 and all(len(g) == 1 for g, _, _ in words)
phrases, reps = targets("phrases")
assert reps == 5 and phrases and all(len(g) > 1 for g, _, _ in phrases)
assert targets("negative")[0][0][0] == [] and targets("words", 3)[1] == 3
manifest = {"clips": [{"signer_id": "a", "glosses": ["119"]}, {"signer_id": "a", "glosses": ["119"]},
                      {"signer_id": "b", "glosses": ["119"]}]}
assert done_counts(manifest, "a") == {("119",): 2}, "resume counts are per signer"
print("[OK] glasses collector: hand labels, mirroring, targets, resume counts")
