"""Cloud controller: segment commands, recorder errors and what each server result shows or says."""
import _path  # noqa: F401
from src.cloud_control import READY, STATUS_TEXT, CloudController


class Recorder:
    def __init__(self):
        self.active, self.error, self.starts, self.cancels = False, None, [], 0
        self.finish_error = None

    def start(self, pre_roll=0.0):
        self.active = True
        self.starts.append(pre_roll)

    def cancel(self):
        self.active = False
        self.cancels += 1

    def finish(self):
        self.active = False
        if self.finish_error:
            raise ValueError(self.finish_error)
        return "sequence", 12.5


class Cloud:
    def __init__(self):
        self.busy, self.sent, self.cancels, self.results = False, [], 0, []

    def submit(self, sequence, origin_t):
        self.sent.append((sequence, origin_t))
        return True

    def cancel(self):
        self.cancels += 1

    def poll(self):
        return self.results.pop(0) if self.results else None


class Hud:
    def __init__(self):
        self.own_texts, self.statuses = [], []

    def own(self, text):
        self.own_texts.append(text)

    def status(self, text):
        self.statuses.append(text)


class Tts:
    def __init__(self):
        self.said = []

    def say(self, text, origin_t=0.0):
        self.said.append((text, origin_t))


lines = []
recorder, cloud, hud, tts = Recorder(), Cloud(), Hud(), Tts()
control = CloudController(cloud, recorder, hud, tts=tts, out=lambda *a: lines.append(" ".join(map(str, a))))
assert hud.statuses == [READY]

control.toggle()
assert recorder.active and recorder.starts == [0.0] and hud.statuses[-1] == "수어 녹화 중"
control.toggle()
assert cloud.sent == [("sequence", 12.5)] and hud.statuses[-1] == "수어 번역 중"
control.command("auto_start")
assert recorder.starts[-1] == 0.3, "Automatic starts keep the pre-roll"
control.command("auto_discard")
assert recorder.cancels == 1 and cloud.cancels == 0, "Discard keeps a pending translation"
control.command("start")
control.command("cancel")
assert recorder.cancels == 2 and cloud.cancels == 1 and hud.statuses[-1] == "수어 입력 취소"
cloud.busy = True
control.command("start")
assert not recorder.active and hud.statuses[-1] == "처리 중입니다"
cloud.busy = False
control.command("stop")              # nothing recording: ignored
assert len(cloud.sent) == 1
recorder.active, recorder.finish_error = True, "too_short"
control.command("stop")
assert hud.statuses[-1] == "수어를 다시 입력하세요" and lines[-1] == "[cloud] 구간 취소: too_short"

recorder.error = "too_long"
control.tick()
assert hud.statuses[-1] == "입력 취소: 다시 녹화하세요"
recorder.error = None

cloud.results = [({"status": "ok", "text": "병원이 어디예요?", "auto_speech": True, "mock": False}, 3.0)]
control.tick()
assert tts.said == [("병원이 어디예요?", 3.0)] and hud.own_texts[-1] == "병원이 어디예요?"
assert hud.statuses[-1] == READY and lines[-1] == "[cloud 번역] 병원이 어디예요?"
cloud.results = [({"status": "ok", "text": "안녕하세요.", "auto_speech": False, "mock": True}, 4.0)]
control.tick()
assert len(tts.said) == 1 and "텍스트만 표시" in lines[-1], "Unverified results are only shown"
cloud.results = [({"status": "fallback", "text": "병원 어디", "source": "words",
                   "auto_speech": False, "mock": False}, 5.0)]
control.tick()
assert hud.own_texts[-1] == "미리보기: 병원 어디 (다시 입력하세요)" and len(tts.said) == 1
cloud.results = [({"status": "fallback", "text": "이름이 뭐예요?", "source": "reviewed_rules",
                   "auto_speech": False, "mock": False}, 6.0)]
control.tick()
assert hud.own_texts[-1] == "미리보기: 이름이 뭐예요?" and len(tts.said) == 1
for status, message in STATUS_TEXT.items():
    cloud.results = [({"status": status, "text": "", "auto_speech": False, "mock": False}, 7.0)]
    control.tick()
    assert hud.statuses[-1] == message
control.tick()                          # no result: nothing changes
assert hud.statuses[-1] == STATUS_TEXT["expired"]
# A failed request names its reason in the operator's terminal, never on the HUD or aloud.
cloud.results = [({"status": "unavailable", "text": "", "auto_speech": False, "mock": False,
                   "error": "HTTP 409"}, 8.0)]
control.tick()
assert lines[-1] == "[cloud] 요청 실패: HTTP 409" and hud.statuses[-1] == STATUS_TEXT["unavailable"]
assert len(tts.said) == 1
print("[OK] cloud controller: commands, discard vs cancel, busy, errors, ok/fallback/status results")
