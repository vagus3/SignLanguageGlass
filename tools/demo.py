# -*- coding: utf-8 -*-
"""
데모 시나리오 러너 (카메라·모델 없이 대화 한 편을 재생)

    python tools/demo.py                        # 기본 시나리오, preview 창
    python tools/demo.py --record demo.mp4      # 발표용 영상으로 저장
    python tools/demo.py --scenario my.json
    python tools/demo.py --no-audio             # 소리 없이 화면만
    python tools/demo.py --display pi           # 실물 OLED 로

■ 왜 필요한가
    라이브 데모는 터집니다. 조명이 다르고, 카메라가 안 잡히고, 긴장해서
    수형이 흔들립니다. 발표 5분 전에 그걸 발견하면 손 쓸 방법이 없습니다.

    이 스크립트는 '출력 쪽'(HUD + 스피커)만 대본대로 재생합니다.
      - 리허설: 자막이 읽히는지, 발화 타이밍이 자연스러운지 미리 확인
      - 녹화:   보고서·발표에 넣을 영상을 안정적으로 확보
      - 폴백:   현장에서 인식이 안 될 때 대체 시연

    중요한 건 이게 가짜 화면이 아니라는 점입니다. 진짜 HudRenderer,
    진짜 PhraseBuilder, 진짜 TtsPlayer 를 그대로 씁니다. 그래서 여기서
    보이는 레이아웃과 들리는 소리가 실행 때와 동일합니다.

■ 시나리오 형식 (JSON 배열)
    [
      {"say":  "어디가 아프세요?", "face": [640, 300]},   # 상대가 말함 -> HUD
      {"sign": "병원"},                                   # 착용자가 수어 -> 스피커
      {"sign": "어디"},
      {"wait": 2.0},
      {"note": "화면에는 안 나오는 진행 메모"}
    ]
    say  : 상대방 발화 (자막으로 뜸).  face 를 같이 주면 말풍선 위치가 바뀜
    sign : 착용자의 수어 단어 (PhraseBuilder 를 거쳐 문장으로 발화)
    wait : 대기 초
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# 기본 시나리오: 병원에서의 짧은 대화 한 편.
# config.PHRASE_RULES 의 규칙이 실제로 걸리도록 짜여 있습니다.
DEFAULT = [
    {"note": "— 착용자가 병원 접수처에 도착 —"},
    {"sign": "안녕하세요"},
    {"wait": 1.6},
    {"say": "안녕하세요 무엇을 도와드릴까요", "face": [640, 300]},
    {"wait": 2.2},
    {"sign": "어디"},
    {"sign": "병원"},
    {"wait": 2.4},
    {"say": "내과는 삼층입니다", "face": [700, 310]},
    {"wait": 2.2},
    {"sign": "다시"},
    {"wait": 2.4},
    {"say": "삼 층 이라고 했습니다", "face": [560, 295]},
    {"wait": 2.2},
    {"sign": "감사합니다"},
    {"wait": 2.0},
    {"note": "— 끝 —"},
]


class Recorder:
    """HUD 프레임을 확대해 영상으로 저장합니다."""

    def __init__(self, path, fps, scale=6):
        import cv2
        self.cv2 = cv2
        self.scale = scale
        self.size = (C.OLED_W * scale, C.OLED_H * scale)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if p.suffix == ".mp4" else "MJPG"))
        self.w = cv2.VideoWriter(str(p), fourcc, fps, self.size)
        if not self.w.isOpened():
            raise RuntimeError(f"영상 파일을 열 수 없습니다: {p}")
        self.path = p
        self.n = 0

    def add(self, img):
        arr = (np.array(img, np.uint8) * 255)
        big = self.cv2.resize(arr, self.size,
                              interpolation=self.cv2.INTER_NEAREST)
        self.w.write(self.cv2.cvtColor(big, self.cv2.COLOR_GRAY2BGR))
        self.n += 1

    def close(self):
        self.w.release()
        return self.path, self.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="JSON 시나리오 파일")
    ap.add_argument("--display", default="preview",
                    choices=["preview", "serial", "pi", "null"])
    ap.add_argument("--record", default=None, help="영상으로 저장 (.mp4 / .avi)")
    ap.add_argument("--fps", type=int, default=20, help="렌더/녹화 프레임레이트")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="재생 속도 배율 (2.0 이면 두 배 빠르게 리허설)")
    a = ap.parse_args()

    from src.hud import HudRenderer, load_calib
    from src.oled_bridge import make_display
    from src.predictor import PhraseBuilder

    steps = DEFAULT
    if a.scenario:
        steps = json.loads(Path(a.scenario).read_text(encoding="utf-8"))

    load_calib()
    hud = HudRenderer()
    display = make_display(a.display)
    phrase = PhraseBuilder()
    rec = Recorder(a.record, a.fps) if a.record else None

    tts = None
    if not a.no_audio:
        from src.tts_player import TtsPlayer
        tts = TtsPlayer()
        tts.start()

    caption, cap_t = "", 0.0
    face = [640.0, 300.0]
    dt = 1.0 / a.fps

    def render_until(deadline):
        """마감 시각까지 매 프레임 렌더 + 출력 + 녹화."""
        nonlocal caption
        while time.time() < deadline:
            t0 = time.time()
            if caption and t0 - cap_t > C.HUD_HOLD_SEC:
                caption = ""
            img = hud.render(caption, face_xy=tuple(face),
                             cam_size=(C.CAM_W, C.CAM_H))
            display.show(img)
            if rec:
                rec.add(img)
            # 조합 대기가 끝난 단어를 흘려보냅니다 (main.py 와 동일한 규칙)
            pend = phrase.poll()
            if pend:
                print(f"        -> 발화: {pend}")
                if tts:
                    tts.say(pend)
            if a.display == "preview":
                import cv2
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    raise KeyboardInterrupt
            slack = dt - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)

    print(f"시나리오 {len(steps)}단계 재생 "
          f"({'녹화 ' + a.record if rec else 'preview'}, "
          f"{'무음' if a.no_audio else '소리 있음'})")
    if a.display == "preview":
        print("q 로 중단.")

    try:
        for st in steps:
            if "note" in st:
                print(f"\n{st['note']}")
                continue
            if "face" in st:
                face = [float(v) for v in st["face"]]
            if "say" in st:
                caption, cap_t = st["say"], time.time()
                print(f"[음성] {st['say']}")
                render_until(time.time() + 0.35 / a.speed)
            if "sign" in st:
                print(f"[수어] {st['sign']}")
                sentence = phrase.feed(st["sign"])
                if sentence:
                    print(f"        -> 발화: {sentence}")
                    if tts:
                        tts.say(sentence)
                render_until(time.time() + 0.35 / a.speed)
            if "wait" in st:
                render_until(time.time() + float(st["wait"]) / a.speed)
        # 마지막으로 남은 조합 대기를 흘려보낼 시간
        render_until(time.time() + 2.0 / a.speed)
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        if tts:
            tts.stop()
            tts.join(timeout=2.0)
        display.close()
        if rec:
            path, n = rec.close()
            print(f"\n녹화 저장: {path}  ({n}프레임, {n / a.fps:.1f}초)")
        if a.display == "preview":
            import cv2
            cv2.destroyAllWindows()
        print("완료")


if __name__ == "__main__":
    main()
