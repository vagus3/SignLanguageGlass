"""안경 카메라(착용자 시점)로 수어를 찍어 학습 자료를 모읍니다. 영상과 손 관절을 함께 저장합니다.

python -m src.collect_glasses --signer 홍길동 --session day1
python -m src.collect_glasses --signer 홍길동 --session day1 --mode phrases
python -m src.collect_glasses --signer 홍길동 --session day1 --mode negative

- words   : assets/glasses/vocabulary.json 의 단어를 차례로, 단어마다 --reps 번 (기본 10)
- phrases : assets/glasses/phrases.json 에서 두 단어 이상인 문장을, 문장마다 --reps 번 (기본 5)
- negative: 수어가 아닌 동작 (머리 긁기, 물 마시기, 가만히 있기). 정답 없음

SPACE 녹화 시작/끝, r 방금 것 지우고 다시, n 다음 단어로 건너뛰기, c 녹화 취소, q 끝내기.
손을 내린 채로 SPACE, 수어, 손을 내린 뒤 SPACE. 중간에 끊겨도 다시 실행하면 이어서 합니다.

얼굴과 어깨가 보이지 않는 시점이므로 Holistic 대신 손만 찾는 MediaPipe Hands 를 씁니다.
결과는 data/raw/glasses/ (깃허브에 올라가지 않음). 영상은 같은 자료로 다른 방식을
비교하거나 관절을 다시 뽑을 때 씁니다.
"""
import argparse
import json
from pathlib import Path
import uuid

import numpy as np

import config as C
from src import clock
from src.data_sources import source_entry
from src.pose_sequence import JOINTS, JOINT_NAMES, VERSION, PoseSequence

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = {"name": "mediapipe-hands", "viewpoint": "glasses", "max_num_hands": 2,
             "model_complexity": 1}


def from_hands(results, *, image_is_mirrored=False):
    """MediaPipe Hands 결과 -> 49개 관절 (손 42개만, 몸 7개는 안 보임).

    Hands 의 왼손/오른손 판정은 거울처럼 뒤집힌 셀카 화면을 가정합니다. 안경 카메라처럼
    뒤집지 않은 화면이면 판정을 바꿔야 착용자의 실제 왼손/오른손이 됩니다.
    같은 손으로 두 번 판정되면 더 확신 있는 쪽만 씁니다.
    """
    points, mask = np.zeros((JOINTS, 2), np.float32), np.zeros(JOINTS, bool)
    best = {}
    for hand, label in zip(results.multi_hand_landmarks or [],
                           results.multi_handedness or []):
        c = label.classification[0]
        name = c.label if image_is_mirrored else {"Left": "Right", "Right": "Left"}[c.label]
        if len(hand.landmark) == 21 and c.score > best.get(name, (-1, None))[0]:
            best[name] = (c.score, hand)
    for name, start in (("Left", 0), ("Right", 21)):
        if name in best:
            xy = np.array([[p.x, p.y] for p in best[name][1].landmark], np.float32)
            inside = np.isfinite(xy).all(1) & ((xy >= 0) & (xy <= 1)).all(1)
            points[start:start + 21] = np.where(inside[:, None], xy, 0)
            mask[start:start + 21] = inside
    if image_is_mirrored:
        points[mask, 0] = 1 - points[mask, 0]
    return points, mask


def targets(mode, reps=None):
    """[(정답 단어열, 보여 줄 글, 참고 클립)] 와 반복 횟수."""
    if mode == "negative":
        return [([], "수어 아님: 머리 긁기, 물 마시기, 가만히 있기 등", "")], reps or 30
    if mode == "phrases":
        entries = json.loads((ROOT / "assets/glasses/phrases.json").read_text(encoding="utf-8"))
        return [(e["glosses"], e["text"], "") for e in entries if len(e["glosses"]) > 1], reps or 5
    vocab = json.loads((ROOT / "assets/glasses/vocabulary.json").read_text(encoding="utf-8"))
    return [([w["gloss"]], w["gloss"], w.get("reference_clip", "")) for w in vocab["words"]], \
        reps or 10


def done_counts(manifest, signer):
    """이 사람이 이미 찍은 횟수 (단어열마다). 다시 실행하면 여기서 이어 갑니다."""
    counts = {}
    for clip in manifest["clips"]:
        if clip["signer_id"] == signer:
            key = tuple(clip["glosses"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def save_manifest(root, manifest):
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(root / "manifest.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signer", required=True, help="찍는 사람 (사람마다 다르게, 같은 사람은 같게)")
    p.add_argument("--session", required=True, help="촬영 날짜·장소 구분 (예: day1)")
    p.add_argument("--mode", choices=["words", "phrases", "negative"], default="words")
    p.add_argument("--reps", type=int, help="단어(문장)마다 횟수. 기본 단어 10, 문장 5, 수어 아님 30")
    p.add_argument("--out", default=str(C.DATA_DIR / "glasses"))
    p.add_argument("--camera", type=int, default=C.CAM_INDEX)
    p.add_argument("--mirrored", action="store_true",
                   help="카메라 화면이 거울처럼 뒤집혀 들어오면 (대부분의 안경 카메라는 아님)")
    args = p.parse_args()
    if not args.signer.strip() or not args.session.strip():
        p.error("--signer 와 --session 은 비울 수 없습니다")
    root = Path(args.out)
    (root / "video").mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    manifest = (json.loads(path.read_text(encoding="utf-8")) if path.exists() else
                {"version": VERSION, "joint_names": JOINT_NAMES, "extractor": EXTRACTOR,
                 "sources": [source_entry("team")], "clips": []})
    if manifest.get("extractor", {}).get("name") != EXTRACTOR["name"]:
        p.error(f"{path} 는 다른 방식으로 찍은 자료입니다; --out 을 새 폴더로 주세요")
    plan, reps = targets(args.mode, args.reps)
    counts = done_counts(manifest, args.signer)

    import cv2
    import mediapipe as mp
    from src.camera import CameraReader
    from src.landmarks import draw_hand_preview, make_preview_hands
    manifest["extractor"].setdefault("mediapipe", mp.__version__)
    cap = CameraReader(camera_index=args.camera)
    if not cap.ok():
        raise SystemExit("카메라를 열 수 없습니다")
    hands = make_preview_hands()
    index, announced = 0, None
    recording, writer, frames, last = False, None, [], None
    print("SPACE 시작/끝, r 다시, n 건너뛰기, c 취소, q 끝내기")
    try:
        while True:
            while index < len(plan) and counts.get(tuple(plan[index][0]), 0) >= reps:
                index += 1
            if index >= len(plan):
                print("모두 찍었습니다.")
                break
            glosses, prompt, reference = plan[index]
            ok, frame = cap.read()
            if not ok:
                continue
            t = clock.now()
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            points, mask = from_hands(res, image_is_mirrored=args.mirrored)
            if recording:
                writer.write(frame)
                frames.append((t, points, mask))
            view = draw_hand_preview(frame.copy(), res)
            done = counts.get(tuple(glosses), 0)
            cv2.putText(view, f"{'REC' if recording else 'READY'}  {done + 1}/{reps}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255) if recording else (0, 255, 255), 2)
            cv2.imshow("glasses collector", view)
            if announced != index:
                announced = index    # 단어가 바뀔 때 한 번만 안내
                print(f"\n[{index + 1}/{len(plan)}] {prompt}" +
                      (f"  (따라 할 AI Hub 동작: {reference})" if reference else ""))
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break
            if key == ord("c") and recording:
                recording = False
                writer.release()
                Path(writer_path).unlink(missing_ok=True)
                frames = []
                print("취소")
            elif key == ord("n") and not recording:
                index += 1
            elif key == ord("r") and not recording and last is not None:
                manifest["clips"] = [c for c in manifest["clips"] if c["id"] != last["id"]]
                for name in (last["file"], last["video"]):
                    (root / name).unlink(missing_ok=True)
                counts[tuple(last["glosses"])] -= 1
                index = next(i for i, item in enumerate(plan) if item[0] == last["glosses"])
                save_manifest(root, manifest)
                print("방금 것을 지웠습니다. 다시 찍으세요.")
                last = None
            elif key == ord(" "):
                if not recording:
                    identity = uuid.uuid4().hex
                    writer_path = str(root / "video" / f"{identity}.mp4")
                    writer = cv2.VideoWriter(writer_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             float(C.CAM_FPS), (w, h))
                    recording, frames, started = True, [], clock.unix_time()
                    continue
                recording = False
                writer.release()
                if len(frames) < 10:
                    Path(writer_path).unlink(missing_ok=True)
                    print("너무 짧아 저장하지 않았습니다")
                    frames = []
                    continue
                times = np.array([f[0] for f in frames])
                sequence = PoseSequence(times - times[0], np.array([f[1] for f in frames]),
                                        np.array([f[2] for f in frames]), (w, h)).validate()
                sequence.save(root / f"{identity}.npz")
                duration = float(sequence.timestamps[-1])
                last = {"id": identity, "file": f"{identity}.npz",
                        "video": f"video/{identity}.mp4", "signer_id": args.signer,
                        "session_id": args.session, "group_id": identity, "view": "glasses",
                        "glosses": list(glosses), "text": prompt if args.mode != "words" else "",
                        "continuous": len(glosses) > 1, "negative": not glosses,
                        # 손을 내린 채 시작하고 끝내므로 녹화 전체를 수어 구간으로 둡니다.
                        "sign_intervals": [[0.0, duration]] if glosses else [],
                        "capture_start": started, "duration_seconds": duration,
                        "hands_seen": float(sequence.mask[:, :42].any(1).mean())}
                manifest["clips"].append(last)
                save_manifest(root, manifest)
                counts[tuple(glosses)] = counts.get(tuple(glosses), 0) + 1
                print(f"  저장 {counts[tuple(glosses)]}/{reps}  ({duration:.1f}초, "
                      f"손 보임 {last['hands_seen']:.0%})")
                if last["hands_seen"] < 0.5:
                    print("  손이 절반 넘게 안 잡혔습니다. 팔을 앞으로 더 뻗거나 r 로 다시 찍으세요.")
                frames = []
    finally:
        if recording and writer is not None:
            writer.release()
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
