"""Latency stats keep recording past the sample window; whole-run and recent stats stay distinct."""
import _path  # noqa
import csv
import tempfile
from pathlib import Path
import numpy as np
from src.latency import LatencyTracker, NullTracker, REL_ERROR


def exact(values, q):
    return float(np.percentile(values, q, method="inverted_cdf"))


rng = np.random.default_rng(5)
# A slow first stretch (e.g. warm-up or a hot device) followed by a fast steady state.
values = np.concatenate([rng.lognormal(4.5, 0.3, 1000), rng.lognormal(3.0, 0.4, 3000)])
values[10] = 900.0          # The worst sample leaves the recent window early.

within = LatencyTracker(maxlen=5000)
for v in values:
    within.add("holistic", v)
d = within.details()["holistic"]
assert d["exact"] and d["count"] == 4000 and d["recent_n"] == 4000
assert d["p95"] == exact(values, 95) and d["max"] == 900.0

stats = LatencyTracker(maxlen=1000)
for v in values:
    stats.add("holistic", v)
d = stats.details()["holistic"]
assert d["count"] == 4000, "Samples past the window must still be counted"
assert d["max"] == 900.0, "Whole-run maximum must survive the ring buffer"
assert not d["exact"] and d["recent_n"] == 1000
for q, key in ((50, "p50"), (95, "p95")):
    truth = exact(values, q)
    assert abs(d[key] - truth) <= REL_ERROR * truth + 1e-9, (key, d[key], truth)
assert d["recent_p95"] == exact(values[-1000:], 95)
assert d["recent_p95"] < 0.5 * d["p95"], "This data must show recent and whole-run differ"
n, p50, p95, mx = stats.summary()["holistic"]
assert (n, p95, mx) == (4000, d["p95"], 900.0)
report = stats.report()
assert "≈" in report and "최근 p95" in report and "4000" in report and "±1%" in report

# The original failure: 15 minutes at 30 fps against a 20000-sample window.
soak = LatencyTracker()
for i in range(27000):
    soak.add("sign_total", 20.0 + (i % 7))
assert soak.summary()["sign_total"][0] == 27000

with tempfile.TemporaryDirectory() as td:
    log = Path(td) / "run" / "latency.csv"
    logged = LatencyTracker(maxlen=100, log_path=log)
    for i in range(1000):
        logged.add("render", float(i))
    assert "전체 표본 원본" in logged.report()
    assert logged.save_csv(log) == log, "The full log is the file --stats asked for"
    rows = list(csv.reader(log.open(encoding="utf-8")))
    assert rows[0] == ["span", "index", "ms"] and len(rows) == 1001
    assert rows[1] == ["render", "0", "0.000"] and rows[-1] == ["render", "999", "999.000"]
    logged.add("render", 1.0)          # After closing, recording continues without a log.
    logged.close()

    other = Path(td) / "recent.csv"
    assert stats.save_csv(other) == other
    rows = list(csv.reader(other.open(encoding="utf-8")))
    assert len(rows) == 1001 and rows[1][1] == "3000", "Recent rows keep whole-run indices"

null = NullTracker()
assert null.details() == {} and null.save_csv("unused.csv") is None
null.close()
print("[OK] whole-run count/max, ±1% whole-run percentiles, exact recent window, full log")
