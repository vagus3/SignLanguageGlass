#!/usr/bin/env python
"""느린 노트북에서 어디가 느린지 숫자로 보기 (창을 띄우지 않습니다).

python tools/profile_pipeline.py --camera 0                  # 카메라 실제 초당 장수 + 단계별 시간
python tools/profile_pipeline.py --camera 0 --compare-formats # MJPG 요청 / 드라이버 기본 형식 비교
python tools/profile_pipeline.py --video some.mp4             # 카메라 없이 영상 파일로

카메라로 받은 프레임을 먼저 모은 뒤, main.py 와 같은 단계를 한 프레임씩 잽니다.
  capture        카메라가 실제로 준 초당 장수 (조명·USB·압축 형식에 좌우)
  holistic       손·몸 인식. 비전 스레드의 대부분
  face_detection 얼굴 검출
  video_overlay  영상 위 말풍선 그리기 (메인 스레드, 초당 VIDEO_OVERLAY_FPS 번)
  hud            안경 화면 그림
결과의 'vision max fps' 가 카메라보다 낮으면 CPU 가, 높으면 카메라가 병목입니다.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def grab_camera(index, seconds, fourcc):
    """(frames, delivered fps, negotiated format) with the given FOURCC request."""
    import cv2
    from src.camera import fourcc_name, open_camera
    saved = C.CAM_FOURCC
    C.CAM_FOURCC = fourcc
    try:
        cap = open_camera(verbose=False, camera_index=index)
    finally:
        C.CAM_FOURCC = saved
    if cap is None:
        raise RuntimeError(f"카메라 {index} 를 열 수 없습니다")
    try:
        for _ in range(5):                     # 노출·초점이 자리 잡는 첫 프레임은 버림
            cap.read()
        frames, start = [], time.perf_counter()
        while time.perf_counter() - start < seconds:
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        elapsed = time.perf_counter() - start
        name = fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("카메라에서 프레임을 받지 못했습니다")
    return frames, len(frames) / elapsed, name


def read_video(path, limit):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != C.CAM_W:          # 실행할 때 카메라 크기로 맞춥니다
            frame = cv2.resize(frame, (C.CAM_W, C.CAM_H), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"{path} 에서 프레임을 읽지 못했습니다")
    return frames


def measure(frames, warmup=5):
    """main.py 와 같은 순서로 단계별 시간(ms)을 잽니다."""
    import cv2
    import mediapipe as mp
    from src import landmarks as L
    from src.caption_state import CaptionEvent, CaptionState
    from src.face_tracker import FaceTracker
    from src.hud import HudContent, HudRenderer
    from src.video_overlay import VideoOverlay

    holistic = L.make_holistic(C.MODEL_COMPLEXITY)
    detector = mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                         min_detection_confidence=0.5)
    overlay, faces = VideoOverlay(), FaceTracker(C.FACE_LOST_SEC)
    captions = CaptionState(C.CAPTION_HOLD_SEC, C.CAPTION_PARTIAL_TIMEOUT, C.FACE_LOST_SEC)
    content = HudContent(HudRenderer())
    stages = {k: [] for k in ("convert", "holistic", "face_detection", "video_overlay", "hud")}
    try:
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            t1 = time.perf_counter()
            holistic.process(rgb)
            t2 = time.perf_counter()
            boxes = L.detection_face_boxes_px(detector.process(rgb), w, h)
            t3 = time.perf_counter()
            now = time.monotonic()
            faces.update(boxes, (w, h), now)
            if i == 0:                     # 말풍선이 보이는 상태로 잽니다
                captions.update(CaptionEvent("final", "어디가 아프세요? 병원에 같이 가 드릴까요?",
                                             1, now), faces.current(now))
            track = faces.current(now)
            overlay.render(frame, captions, track, now, mirror=C.MIRROR_PREVIEW,
                           tracks=faces.tracks(now), selected_id=faces.selected_id)
            t4 = time.perf_counter()
            content.partner("어디가 아프세요?", partial=False)
            content.frame(track.center if track else None, (w, h))
            t5 = time.perf_counter()
            if i >= warmup:
                for name, a, b in (("convert", t0, t1), ("holistic", t1, t2),
                                   ("face_detection", t2, t3), ("video_overlay", t3, t4),
                                   ("hud", t4, t5)):
                    stages[name].append((b - a) * 1000)
    finally:
        holistic.close()
        detector.close()
    return {name: {"p50": statistics.median(v), "p95": sorted(v)[max(0, int(len(v) * .95) - 1)]}
            for name, v in stages.items() if v}


def advice(report):
    lines = []
    stage = report["stages"]
    vision_ms = sum(stage[k]["p50"] for k in ("convert", "holistic", "face_detection"))
    vision_fps = 1000 / max(vision_ms, 1e-6)
    camera = report.get("camera_fps")
    if camera is not None and camera < 20:
        lines.append(f"카메라가 초당 {camera:.0f}장만 줍니다. 조명을 밝게 하고(어두우면 노출이 길어짐), "
                     "--compare-formats 로 MJPG 가 되는지 보세요. USB 허브 대신 본체 포트를 쓰세요.")
    if vision_fps < 15:
        lines.append(f"CPU 가 손·몸 인식을 초당 {vision_fps:.0f}번만 처리합니다. 전원을 연결하고 "
                     "윈도우 전원 모드를 '최고 성능'으로, 다른 프로그램(브라우저 등)을 닫으세요. "
                     "config.STT_THREADS = 1 도 시도해 보세요.")
    if stage["video_overlay"]["p50"] > 10:
        lines.append("말풍선 그리기가 느립니다. config.VIDEO_OVERLAY_FPS 를 15 로 낮춰 보세요.")
    if not lines:
        lines.append("병목이 보이지 않습니다. 실제 실행에서 끊기면 python main.py ... --stats 로 "
                     "STT·TTS 를 함께 켠 상태의 단계별 시간을 확인하세요.")
    return vision_fps, lines


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="카메라 번호 (기본: config.CAM_INDEX)")
    source.add_argument("--video", help="카메라 대신 쓸 영상 파일")
    p.add_argument("--seconds", type=float, default=4.0, help="카메라에서 받는 시간")
    p.add_argument("--frames", type=int, default=90, help="영상 파일에서 쓸 프레임 수")
    p.add_argument("--compare-formats", action="store_true",
                   help="MJPG 요청과 드라이버 기본 형식의 초당 장수를 비교")
    p.add_argument("--out", help="결과 JSON 저장 경로")
    args = p.parse_args()

    import cv2
    import mediapipe as mp
    report = {"system": {"os": f"{platform.system()} {platform.release()}",
                         "machine": platform.machine(), "cpus": os.cpu_count(),
                         "python": platform.python_version(), "opencv": cv2.__version__,
                         "mediapipe": mp.__version__}}
    try:
        if args.video:
            frames = read_video(args.video, args.frames)
            report["source"] = f"video {Path(args.video).name}"
        else:
            index = C.CAM_INDEX if args.camera is None else args.camera
            formats = [C.CAM_FOURCC] + ([None] if args.compare_formats and C.CAM_FOURCC else [])
            report["camera"] = {}
            for fourcc in formats:
                grabbed, fps, name = grab_camera(index, args.seconds, fourcc)
                label = f"요청 {fourcc}" if fourcc else "드라이버 기본"
                report["camera"][label] = {"fps": round(fps, 1), "format": name,
                                           "size": list(grabbed[0].shape[1::-1])}
                if fourcc == formats[0]:
                    frames, report["camera_fps"] = grabbed, fps
            report["source"] = f"camera {index}"
        report["frames"] = len(frames)
        report["stages"] = measure(frames)
    except RuntimeError as error:
        p.error(str(error))
    vision_fps, tips = advice(report)
    report["vision_max_fps"], report["advice"] = round(vision_fps, 1), tips

    s = report["system"]
    print(f"{s['os']} {s['machine']}, CPU {s['cpus']}개, OpenCV {s['opencv']}, "
          f"MediaPipe {s['mediapipe']} | {report['source']}, {report['frames']}프레임")
    for label, cam in report.get("camera", {}).items():
        print(f"  카메라 ({label}): 초당 {cam['fps']}장, 형식 {cam['format']}, "
              f"{cam['size'][0]}x{cam['size'][1]}")
    for name, v in report["stages"].items():
        print(f"  {name:15s} p50 {v['p50']:6.1f} ms   p95 {v['p95']:6.1f} ms")
    print(f"  vision max fps  {report['vision_max_fps']} (손·몸 인식 + 얼굴 검출 기준)")
    for tip in tips:
        print("  -", tip)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
