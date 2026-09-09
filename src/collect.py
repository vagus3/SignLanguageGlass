# -*- coding: utf-8 -*-
"""
Step 1-1. 수어 데이터 수집기

    python -m src.collect

키:
    SPACE  현재 라벨 1회 녹화 (3초 카운트다운 -> 1초를 30프레임으로 리샘플)
    n / p  다음 / 이전 라벨
    u      마지막 녹화 취소
    q      종료

저장 형식:  data/raw/{라벨}/{촬영자}__{세션}_{일련번호}.npy   shape (30, 246)
(프레임마다 파일을 쪼개지 않습니다. 파일 수가 수만 개가 되면 학습이 느려집니다)

촬영 팁
    - 배경/조명/옷/거리를 조금씩 바꿔가며 촬영해야 실전에서 버팁니다
    - NEUTRAL 은 "가만히 있기" 뿐 아니라 머리 긁기, 물 마시기, 대화 중 손짓 등
      수어가 아닌 동작을 다양하게 넣으세요. 오출력이 확 줍니다
    - 한 번에 몰아 찍지 말고 며칠에 나눠 찍으면 일반화가 좋아집니다
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src import landmarks as L


def label_dir(label):
    d = C.DATA_DIR / label
    d.mkdir(parents=True, exist_ok=True)
    return d


# 디렉터리 glob 을 매 프레임 15회씩 돌면 30fps 에서 초당 450회 파일시스템
# 접근입니다. 카운트는 캐시하고 저장/삭제할 때만 갱신합니다.
_COUNTS = {}


def refresh_counts():
    for v in C.VOCAB:
        _COUNTS[v] = len(list(label_dir(v).glob("*.npy")))


def count_of(label):
    return _COUNTS.get(label, 0)


def next_path(label):
    """
    파일명에 촬영자와 세션을 넣습니다: {촬영자}__{세션}_{번호}.npy

    ★ 이게 왜 중요한가
      학습/테스트를 시퀀스 단위로 무작위 분할하면, 같은 사람이 찍은 거의
      동일한 동작이 양쪽에 나뉘어 들어가 정확도가 거짓으로 부풀려집니다.
      "97% 정확도" 인데 다른 사람이 해보면 30% 인 상황이 여기서 나옵니다.
      촬영자를 기록해 두면 나중에 화자 단위로 나눠 진짜 성능을 볼 수 있습니다.
    """
    d = label_dir(label)
    from src.dataset_contract import capture_prefix
    sid = capture_prefix()
    n = 0
    while (d / f"{sid}_{n:03d}.npy").exists():
        n += 1
    return d / f"{sid}_{n:03d}.npy"


def put(img, text, xy, color=(255, 255, 255), scale=0.6, thick=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    from src.camera import open_camera
    from src.dataset_contract import ensure_dataset_contract

    try:
        ensure_dataset_contract(create=True)
    except RuntimeError as e:
        raise SystemExit(str(e))

    # main.py 와 같은 카메라 설정(수동 노출 포함)을 씁니다.
    # 수집할 때와 실행할 때 영상 특성이 다르면 정확도가 떨어집니다.
    cap = open_camera()
    if cap is None:
        print("카메라를 열 수 없습니다. 다른 앱이 점유 중인지 / OS 권한을 확인하세요.")
        return

    print(f"촬영자/세션: {C.SIGNER_ID} / {C.SESSION_ID}  "
          "(사람 또는 촬영일이 바뀌면 config.py를 수정하세요)")
    refresh_counts()
    holistic = L.make_holistic(C.MODEL_COMPLEXITY)
    li = 0
    last_saved = None
    state = "idle"          # idle | countdown | recording
    t_state = 0.0
    buf = []
    buf_t = []
    record_t0 = None
    record_sec = C.SEQ_LEN / float(C.TRAIN_FPS)

    print("SPACE=녹화  n/p=라벨이동  u=취소  q=종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)

        view = frame.copy()
        L.draw_debug(view, res)
        if C.MIRROR_PREVIEW:
            view = cv2.flip(view, 1)

        label = C.VOCAB[li]
        now = time.time()

        # ---------------- 상태 머신 ----------------
        if state == "countdown":
            left = 3.0 - (now - t_state)
            if left <= 0:
                state, buf, buf_t = "recording", [], []
                record_t0 = None
            else:
                put(view, f"{int(left) + 1}", (C.CAM_W // 2 - 30, C.CAM_H // 2),
                    (0, 255, 255), 3.0, 6)

        elif state == "recording":
            if record_t0 is None:
                record_t0 = now
            buf.append(L.extract(res))
            buf_t.append(now)
            elapsed = now - record_t0
            progress = min(C.SEQ_LEN, int(elapsed / record_sec * C.SEQ_LEN) + 1)
            put(view, f"REC {progress}/{C.SEQ_LEN}", (20, 110), (0, 0, 255))
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1),
                          (0, 0, 255), 6)
            if elapsed >= record_sec and len(buf) >= 2:
                arr = L.resample_timed(
                    buf, buf_t, C.SEQ_LEN,
                    start=record_t0, end=record_t0 + record_sec)
                p = next_path(label)
                np.save(p, arr)
                last_saved = p
                _COUNTS[label] = _COUNTS.get(label, 0) + 1
                actual_fps = (len(buf) - 1) / max(elapsed, 1e-6)
                print(f"저장 {p.relative_to(C.ROOT)}  {arr.shape}  "
                      f"원본 {len(buf)}프레임/{elapsed:.2f}초 ({actual_fps:.1f}fps)")
                state = "idle"

        # ---------------- HUD ----------------
        done = count_of(label)
        color = (0, 255, 0) if done >= C.SEQ_PER_LABEL else (255, 255, 255)
        put(view, f"[{li + 1}/{len(C.VOCAB)}]  {label}", (20, 40), color, 0.9, 2)
        put(view, f"{done} / {C.SEQ_PER_LABEL}", (20, 75), color)
        total = sum(count_of(v) for v in C.VOCAB)
        put(view, f"total {total}", (20, view.shape[0] - 20), (200, 200, 200), 0.5, 1)
        if not L.hands_visible(res) and state == "recording":
            put(view, "! 손이 안 잡힙니다", (20, 145), (0, 165, 255))

        cv2.imshow("signglass - collector", view)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break
        elif k == ord(" ") and state == "idle":
            state, t_state = "countdown", now
        elif k == ord("n"):
            li = (li + 1) % len(C.VOCAB)
        elif k == ord("p"):
            li = (li - 1) % len(C.VOCAB)
        elif k == ord("u") and last_saved and last_saved.exists():
            lbl = last_saved.parent.name
            last_saved.unlink()
            _COUNTS[lbl] = max(0, _COUNTS.get(lbl, 1) - 1)
            print(f"취소 {last_saved.name}")
            last_saved = None

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

    print("\n--- 수집 현황 ---")
    for v in C.VOCAB:
        c = count_of(v)
        mark = "OK " if c >= C.SEQ_PER_LABEL else "부족"
        print(f"  {mark} {v:<12} {c:>3} / {C.SEQ_PER_LABEL}")


if __name__ == "__main__":
    main()
