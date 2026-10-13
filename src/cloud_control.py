"""Segment commands and server results of the cloud backend, apart from main's I/O loop.

Commands come from SPACE/c, stdin or the auto-segmenter ("auto_start", ...);
results come from CloudWorker.poll(). The controller tells the HUD what to show
and the TTS what to say. It never touches the camera, windows or audio devices.
"""
from src.latency import NullTracker

STATUS_TEXT = {
    "no_sign": "수어가 감지되지 않았습니다",
    "uncertain": "수어를 다시 입력하세요",
    "unavailable": "번역 서버 연결을 확인하세요",
    "expired": "번역 요청이 취소 또는 만료됐습니다",
}
READY = "수어 입력 대기"


class CloudController:
    def __init__(self, cloud, recorder, hud, *, tts=None, stats=None, out=print):
        self.cloud, self.recorder, self.hud = cloud, recorder, hud
        self.tts, self.stats, self.out = tts, stats or NullTracker(), out
        self.hud.status(READY)

    def toggle(self):
        """SPACE: start a segment, or finish and send the current one."""
        self.command("stop" if self.recorder.active else "start")

    def command(self, command):
        automatic = command.startswith("auto_")
        command = command.removeprefix("auto_")
        if command == "discard":
            # Too short to be a sentence; unlike cancel, keep any pending translation.
            self.recorder.cancel()
            self.hud.status(READY)
        elif command == "cancel":
            self.recorder.cancel()
            self.cloud.cancel()
            self.hud.status("수어 입력 취소")
        elif command == "start":
            if self.cloud.busy:
                self.hud.status("처리 중입니다")
            elif not self.recorder.active:
                self.recorder.start(pre_roll=0.3 if automatic else 0.0)
                self.hud.status("수어 녹화 중")
        elif command == "stop":
            if not self.recorder.active:
                return
            try:
                sequence, origin_t = self.recorder.finish()
                if self.cloud.submit(sequence, origin_t):
                    self.hud.status("수어 번역 중")
            except ValueError as e:
                self.hud.status("수어를 다시 입력하세요")
                self.out("[cloud] 구간 취소:", e)

    def tick(self):
        """Once per main-loop pass: recorder errors and at most one finished result."""
        if self.recorder.error:
            self.hud.status("입력 취소: 다시 녹화하세요")
        item = self.cloud.poll()
        if not item:
            return
        result, origin_t = item
        status = result["status"]
        if status == "ok":
            text = result["text"]
            self.out("[cloud 번역]", text)
            self.hud.own(text)
            self.hud.status(READY)
            self.stats.since("cloud_response", origin_t)
            if result.get("auto_speech") and not result.get("mock"):
                if self.tts:
                    self.tts.say(text, origin_t=origin_t)
            else:
                self.out("[cloud] 미검증/모의 결과: 텍스트만 표시합니다.")
        elif status == "fallback":
            # Sentence generation failed: a preview for the signer, never spoken.
            text = result["text"]
            words_only = result.get("source") != "reviewed_rules"
            self.out("[cloud 미리보기]", text, f"({result.get('source')})")
            self.hud.own(f"미리보기: {text}" + (" (다시 입력하세요)" if words_only else ""))
            self.hud.status(READY)
        else:
            if result.get("error"):
                # 409: the server's --device-model is another model; 401: token; 429: busy.
                self.out("[cloud] 요청 실패:", result["error"])
            self.hud.status(STATUS_TEXT.get(status, ""))
