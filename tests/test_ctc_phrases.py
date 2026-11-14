"""CTC sequence likelihood and correction by expected phrases (numpy only)."""
import _path  # noqa
import itertools
import numpy as np
from src.ctc_runtime import decode_with_phrases, greedy_decode, sequence_log_likelihood

labels = ["<blank>", "머리", "목", "아프다"]
rng = np.random.default_rng(0)


def collapse(path):
    return tuple(k for k, _ in itertools.groupby(path) if k != 0)


# The forward algorithm equals a brute-force sum over every alignment.
logits = rng.normal(0, 1, (5, 4))
log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
for tokens in ([], [1], [1, 3], [2, 2], [1, 2, 3]):
    total = sum(np.exp(sum(log_probs[t, k] for t, k in enumerate(path)))
                for path in itertools.product(range(4), repeat=5)
                if collapse(path) == tuple(tokens))
    assert np.isclose(sequence_log_likelihood(log_probs, tokens), np.log(total)), tokens
assert sequence_log_likelihood(log_probs[:2], [1, 1, 1]) == -np.inf, "too few frames"


def frames(*rows):
    p = np.array(rows, float)
    return np.log(p / p.sum(-1, keepdims=True))


# "목" and "머리" are nearly tied in the first word; the frame-by-frame best is "목".
unsure = frames([.05, .40, .50, .05], [.05, .42, .48, .05], [.9, .03, .03, .04],
                [.05, .02, .03, .90], [.05, .02, .03, .90])
assert greedy_decode(np.exp(unsure), labels)[0] == ["목", "아프다"]
phrases = [{"glosses": ["머리", "아프다"], "after": ["어디"]}]
# Without a bonus the more likely reading stays; with one the expected sentence wins.
assert decode_with_phrases(unsure, labels, phrases)[::2] == (["목", "아프다"], "greedy")
glosses, score, source = decode_with_phrases(unsure, labels, phrases, bonus=1.0)
assert (glosses, source) == (["머리", "아프다"], "phrase") and 0 < score < 1
# What the other person just said doubles the bonus of phrases that answer it.
assert decode_with_phrases(unsure, labels, phrases, bonus=0.2)[2] == "greedy"
assert decode_with_phrases(unsure, labels, phrases, bonus=0.2, heard="어디 아프세요")[2] == "phrase"

# A clear sign is not overwritten by a moderate bonus, and unknown labels are ignored.
clear = frames([.01, .01, .97, .01], [.9, .03, .03, .04], [.01, .01, .01, .97])
assert decode_with_phrases(clear, labels, phrases, bonus=4.0)[::2] == (["목", "아프다"], "greedy")
assert decode_with_phrases(clear, labels, [{"glosses": ["없는말"]}], bonus=99)[2] == "greedy"
# A phrase equal to the greedy result is reported as greedy, with the greedy score.
same = [{"glosses": ["목", "아프다"]}]
assert decode_with_phrases(clear, labels, same, bonus=9)[1:] == (0.97, "greedy")
# Nothing signed: an expected phrase needs evidence, not just a prior.
silent = frames(*[[.97, .01, .01, .01]] * 6)
assert decode_with_phrases(silent, labels, phrases, bonus=4.0) == ([], 0.0, "greedy")

print("[OK] CTC likelihood matches brute force; phrase correction, context bonus, "
      "clear signs and silence preserved")
