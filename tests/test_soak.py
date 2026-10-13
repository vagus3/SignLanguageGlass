"""Short soak over real HTTP with every injected fault; the audit must catch unsafe speech."""
import _path  # noqa: F401
import sys
from unittest.mock import patch

try:
    import uvicorn  # noqa: F401
except ImportError:
    sys.exit(77)
from src.cloud_sign import CloudWorker
from tools.soak import run

FAST = dict(cycle=(1.2, 1.0), client_timeout=1.0, slow_delay=1.6, outage=(0.45, 0.6),
            slow_every=3, error_every=2, cancel_every=5)

report = run(24, **FAST)
assert report["passed"], report["problems"]
results = report["results"]
for status in ("ok", "expired", "fallback", "unavailable"):
    assert results.get(status), (status, results)       # every fault actually happened
assert report["spoken"] == results["ok"] and report["cancelled"] >= 1
assert sum(results.values()) == report["segments"], "every segment returned exactly once"
assert report["fps"] > 25 and report["threads"][-1] <= report["threads"][0] + 2

# The audit is not vacuous: a worker that forgets cancels and deadlines must fail it.
fresh = CloudWorker.poll


def careless(self):
    item = fresh(self)
    if item and item[0].get("status") == "expired":
        taken = self.results.taken[-1]
        return {"status": "ok", "text": "안녕하세요.", "auto_speech": True, "mock": False,
                "segment_id": taken["segment"]}, taken["origin_t"]
    return item


with patch.object(CloudWorker, "poll", careless):
    broken = run(14, **FAST)
assert not broken["passed"]
assert any("after cancel" in p or "after its deadline" in p for p in broken["problems"]), broken
print(f"[OK] soak: {report['segments']} segments over HTTP, results {results}, "
      f"spoken {report['spoken']}; audit catches speech after cancel/expiry")
