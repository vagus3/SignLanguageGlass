"""A local recognition that never returns must not leave the app stuck on '처리 중'."""
import _path  # noqa
import threading

from src import clock
from src.cloud_sign import CloudWorker


class Stuck:
    timeout = 0.05

    def __init__(self):
        self.release = threading.Event()

    def translate(self, payload):
        self.release.wait(5)
        return {"status": "ok", "text": "늦은 결과", "segment_id": payload["segment_id"],
                "auto_speech": False, "mock": False}


class Seq:
    def to_dict(self):
        return {}


client = Stuck()
worker = CloudWorker(client)
worker.GRACE = 0.05
worker.start()
assert worker.submit(Seq(), 0.0)
assert worker.busy, "while the request is in time, a new one is refused"
assert not worker.submit(Seq(), 0.0)
clock_deadline = worker.deadline
while clock.now() < clock_deadline + worker.GRACE + 0.05:
    pass
assert not worker.busy, "past the deadline the worker is released"
client.release.set()
worker.jobs.join() if False else None
# the late result is ignored, never shown as a fresh translation
import time
time.sleep(0.3)
item = worker.poll()
assert item is None or item[0]["status"] == "expired", item
worker.stop()
print("[OK] watchdog: a stuck local recognition is released after its deadline")
