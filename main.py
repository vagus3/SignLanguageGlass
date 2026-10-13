# -*- coding: utf-8 -*-
"""
통합 실행

    python main.py
    python main.py --no-stt          # Phase 1(수어->음성)만
    python main.py --no-sign         # Phase 2(음성->자막)만
    python main.py --no-sign --fixed-caption  # 카메라 없는 안정형 Phase 2
    python main.py --display serial  # 실제 OLED 로 출력

스레드 구성
    [vision]  캡처 + Holistic + LSTM 추론      (C++ 구간에서 GIL 을 놓으므로 실제로 병렬)
    [stt]     sherpa-onnx(기본) / vosk 스트리밍
    [tts]     캐시 wav 재생
    [main]    HUD 렌더 + 화면 출력             (GUI 는 메인 스레드 전용)

지연을 만드는 최대 원인은 큐에 프레임이 쌓이는 것입니다.
그래서 비전 프레임 큐는 maxsize=1이고, 오디오 큐도 짧게 제한합니다.
"""
import sys
import time
import queue
import argparse
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from src import clock
from src import landmarks as L
from src.one_euro import OneEuroPoint


def resolve_camera_plan(no_sign, fixed_caption, sign_camera=None, face_camera=None):
    """CLI 조합을 실제 열 카메라 계획으로 바꾸고 장치 충돌을 차단합니다."""
    sign_index = C.CAM_INDEX if sign_camera is None else int(sign_camera)
    separate_face = face_camera is not None and not fixed_caption
    need_vision = (not no_sign) or (not fixed_caption and not separate_face)
    if not no_sign and separate_face and int(face_camera) == sign_index:
        raise ValueError(
            "수어/얼굴 카메라 번호가 같습니다. 두 카메라 구성은 서로 다른 번호를 쓰세요.")
    return need_vision, separate_face, sign_index


def draw_segment_overlay(cv2, frame, recorder, segmenter):
    """Recording state and, for webcam auto-segmentation, the rest line.

    Hands below the line (keyboard, lap) are ignored; drawn after any preview
    flip, which a horizontal line is unaffected by.
    """
    h, w = frame.shape[:2]
    line = getattr(segmenter, "rest_line", None)
    if line is not None and 0 < line < 1:
        y = int(line * h)
        cv2.line(frame, (0, y), (w, y), (255, 200, 0), 2)
        cv2.putText(frame, "hands down = end", (16, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    mode = "AUTO " if segmenter is not None else ""
    state, color = (("REC", (0, 0, 255)) if recorder.active else ("READY", (0, 255, 255)))
    cv2.putText(frame, mode + state, (w - 200, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ---------------------------------------------------------------- 비전 스레드
class VisionThread(threading.Thread):
    def __init__(self, out_q, enable_sign=True, stats=None, camera_index=None,
                 segment_recorder=None, pose_3d=False, segmenter=None, segment_events=None,
                 track_face=True):
        # daemon 을 클래스 속성으로 두면 Thread.daemon 프로퍼티를 가려서
        # 내부 _daemonic 플래그와 어긋납니다. 생성자로 넘겨야 안전합니다.
        super().__init__(name="vision", daemon=True)
        self.out = out_q
        self.enable_sign = enable_sign
        self.camera_index = camera_index
        self.segment_recorder = segment_recorder
        self.pose_3d = pose_3d
        self.segmenter, self.segment_events = segmenter, segment_events
        self.track_face = track_face    # 고정 자막·전방 얼굴 카메라면 계산할 필요가 없습니다
        from src.latency import NullTracker
        self.stats = stats or NullTracker()
        # ★ 이름을 _stop 으로 두면 안 됩니다.
        # threading.Thread 에는 _stop() 이라는 내부 메서드가 있고, join() 이
        # 스레드 종료를 확인한 뒤 그걸 호출합니다. Event 로 덮어쓰면
        # join() 이 'Event object is not callable' 로 죽습니다.
        self._quit = threading.Event()
        self.fps = 0.0
        self.ms_vision = 0.0     # 캡처 + Holistic 소요 시간

    def stop(self):
        self._quit.set()

    def run(self):
        import cv2
        from src.camera import CameraReader
        from src.pose_sequence import from_mediapipe, world_from_mediapipe

        cap = CameraReader(camera_index=self.camera_index)
        if not cap.ok():
            self.out.put(("error", "카메라를 열 수 없습니다"))
            return

        holistic = L.make_holistic(C.MODEL_COMPLEXITY)

        predictor = None
        if self.enable_sign:
            try:
                from src.predictor import SignPredictor
                predictor = SignPredictor()
                print(f"[vision] 모델 로드 완료 ({len(predictor.labels)} 클래스)")
            except Exception as e:
                print(f"[vision] 수어 모델 없음 -> Phase 2 만 동작합니다\n         {e}")

        smooth = OneEuroPoint(C.FILTER_MIN_CUTOFF, C.FILTER_BETA)
        t_prev, n = clock.now(), 0

        st = self.stats
        while not self._quit.is_set():
            t0 = clock.now()
            ok, frame = cap.read()
            if not ok:
                continue            # CameraReader 가 내부에서 sleep + 재연결 처리
            t_cap = clock.now()     # 이 프레임을 실제로 손에 넣은 시각 (경과 시간 시계)
            st.add("capture", (t_cap - t0) * 1000)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            t_hol = clock.now()
            st.add("holistic", (t_hol - t_cap) * 1000)
            self.ms_vision = (t_hol - t0) * 1000

            # Capture every processed frame before the lossy preview queue.
            if self.segment_recorder is not None:
                points, mask = from_mediapipe(res)
                world = world_from_mediapipe(res, (w, h)) if self.pose_3d else (None, None)
                self.segment_recorder.add(points, mask, t_cap, (w, h), *world)
                if self.segmenter is not None:
                    event = self.segmenter.update(points, mask, t_cap,
                                                  self.segment_recorder.active, (w, h))
                    if event:
                        self.segment_events.put("auto_" + event)

            word, word_t = None, 0.0
            if predictor is not None:
                with st.span("feature"):
                    feat = L.extract(res)
                # push 안에 ONNX 추론이 들어 있습니다. infer 는 런타임이 따로 잽니다.
                word = predictor.push(feat, t=t_cap)
                st.add("infer", predictor.last_ms)
                if word:
                    word_t = clock.now()
                    # 확정을 만든 추론에 쓰인 최신 프레임부터 확정까지
                    st.add("recognition",
                           (word_t - predictor.last_frame_t) * 1000)
            st.add("sign_total", (clock.now() - t0) * 1000)

            face = L.face_center_px(res, w, h) if self.track_face else None
            if face:
                face = smooth(face[0], face[1], clock.now())
            else:
                smooth.reset()

            n += 1
            if n >= 15:
                now = clock.now()
                self.fps = n / max(now - t_prev, 1e-6)
                t_prev, n = now, 0

            self._emit(("frame", {
                "frame": frame, "face": face, "word": word,
                "word_t": word_t,
                "size": (w, h), "fps": self.fps,
                "in_fps": getattr(predictor, "last_fps", 0.0),
                "conf": getattr(predictor, "last_conf", 0.0),
                "raw": getattr(predictor, "last_raw", None),
                "ms_vision": self.ms_vision,
                "ms_infer": getattr(predictor, "last_ms", 0.0),
            }))

        cap.release()
        holistic.close()

    def _emit(self, item):
        """큐를 쌓지 않고 항상 최신 프레임만 유지."""
        try:
            self.out.put_nowait(item)
        except queue.Full:
            try:
                self.out.get_nowait()
                self.out.put_nowait(item)
            except queue.Empty:
                pass


class FaceThread(threading.Thread):
    """선택형 전방 카메라 얼굴 추적. 수어 카메라와 시야를 분리합니다."""

    def __init__(self, out_q, camera_index):
        super().__init__(name="face", daemon=True)
        self.out = out_q
        self.camera_index = camera_index
        self._quit = threading.Event()

    def stop(self):
        self._quit.set()

    def run(self):
        import cv2
        import mediapipe as mp
        from src.camera import CameraReader

        cap = CameraReader(camera_index=self.camera_index)
        if not cap.ok():
            self._emit(("error", f"전방 카메라 {self.camera_index}를 열 수 없습니다"))
            return
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        smooth = OneEuroPoint(C.FILTER_MIN_CUTOFF, C.FILTER_BETA)
        try:
            while not self._quit.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = detector.process(rgb)
                face = L.detection_face_center_px(res, w, h)
                if face:
                    face = smooth(face[0], face[1], clock.now())
                else:
                    smooth.reset()
                self._emit(("face", {"face": face, "size": (w, h), "frame": frame}))
        except Exception as e:
            self._emit(("error", str(e)))
        finally:
            cap.release()
            detector.close()

    def _emit(self, item):
        try:
            self.out.put_nowait(item)
        except queue.Full:
            try:
                self.out.get_nowait()
                self.out.put_nowait(item)
            except queue.Empty:
                pass


def speak(sentence, tts, stats=None, word_t=0.0, show=None):
    """
    word_t : 그 발화를 촉발한 단어가 '확정된' 시각.
             확정 -> 발화 지시 사이의 시간이 곧 PhraseBuilder 의 조합 대기입니다.
             Phase 1 지연에서 보통 여기가 제일 큰 항목이라 따로 기록합니다.
    show   : 착용자 확인용으로 HUD 에 띄울 함수 (HudContent.own)
    """
    print(f"        -> 발화: {sentence}")
    if show is not None:
        show(sentence)
    if stats is not None and word_t:
        stats.since("phrase_latch", word_t)
    if tts:
        tts.say(sentence, origin_t=word_t)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--stt", choices=["sherpa", "vosk"], default=None,
                    help="음성 인식 엔진 (기본: config.STT_BACKEND)")
    ap.add_argument("--no-sign", action="store_true")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--sign-backend", choices=["local", "cloud"], default="local",
                    help="기존 ONNX 단어 인식 또는 서버의 연속 수어 번역")
    ap.add_argument("--cloud-controls", choices=["keyboard", "stdin"], default="keyboard",
                    help="cloud: SPACE/c 또는 표준입력 start/stop/cancel로 문장 구간 지정")
    ap.add_argument("--pose-3d", action="store_true",
                    help="cloud: 3D 좌표도 전송 (서버가 --features 3d 체크포인트일 때)")
    ap.add_argument("--recognize", choices=["server", "device"], default="server",
                    help="cloud: server=좌표를 보내 서버가 인식(기준선 A), "
                         "device=이 기기에서 ONNX로 인식해 글로스만 전송(비교 경로 B)")
    ap.add_argument("--onnx", metavar="PATH",
                    help="--recognize device 의 CTC ONNX (tools/export_ctc_onnx.py 결과)")
    ap.add_argument("--auto-segment", action="store_true",
                    help="cloud: 손 움직임으로 문장 구간 자동 시작/종료 (SPACE도 계속 사용 가능)")
    ap.add_argument("--viewpoint", choices=["webcam", "glasses"], default=C.SIGN_VIEWPOINT,
                    help="수어 카메라 시점: webcam=노트북 정면(손을 내리면 문장 끝), "
                         "glasses=안경 1인칭")
    ap.add_argument("--fixed-caption", action="store_true",
                    help="얼굴 추적 없이 HUD 하단 고정 자막 (카메라 불필요)")
    ap.add_argument("--sign-camera", type=int, default=None, metavar="INDEX",
                    help="수어/기본 카메라 번호 (기본: config.CAM_INDEX)")
    ap.add_argument("--face-camera", type=int, default=None, metavar="INDEX",
                    help="선택형 전방 얼굴 카메라 번호 (두 카메라 구성)")
    ap.add_argument("--display", default=None,
                    choices=["preview", "serial", "pi", "null"])
    ap.add_argument("--debug", action="store_true", help="카메라 뷰도 함께 표시")
    ap.add_argument("--no-gate", action="store_true",
                    help="반이중 게이트 해제 (유선 이어폰으로 들을 때만)")
    ap.add_argument("--stats", nargs="?", const="", default=None,
                    metavar="CSV",
                    help="지연을 구간별로 측정해 종료 시 p50/p95 를 출력합니다. "
                         "경로를 주면 실행 전체의 원본 표본을 CSV 로 계속 기록합니다.")
    a = ap.parse_args()
    if a.recognize == "device" and not a.onnx:
        ap.error("--recognize device 에는 --onnx 가 필요합니다")
    if a.sign_backend == "cloud" and not a.no_sign and a.cloud_controls == "keyboard":
        a.debug = True  # SPACE needs a visible OpenCV event window.

    import cv2
    from src.hud import HudContent, HudRenderer, load_calib
    from src.oled_bridge import make_display
    from src.predictor import PhraseBuilder
    from src.audio_gate import AudioGate, NullGate

    from src.latency import LatencyTracker, NullTracker
    stats = (LatencyTracker(log_path=a.stats or None) if a.stats is not None
             else NullTracker())

    load_calib()

    # 스피커 출력이 마이크로 되돌아오는 에코 루프를 막는 게이트.
    # 유선 이어폰으로 들을 때는 --no-gate 로 꺼도 됩니다.
    gate = NullGate() if a.no_gate else AudioGate(C.AUDIO_GATE_TAIL_SEC)

    cloud = recorder = segmenter = controller = None
    commands = queue.Queue()
    if a.sign_backend == "cloud" and not a.no_sign:
        from src.cloud_sign import CloudClient, CloudWorker, GlossClient, SegmentRecorder
        try:
            if a.recognize == "device":
                from src.ctc_runtime import OnnxRecognizer
                from src.pose_sequence import needs_world
                device = OnnxRecognizer(a.onnx)
                if needs_world(device.feature_set) != a.pose_3d:
                    ap.error(f"{a.onnx} 는 {device.feature_set} 특징입니다. "
                             f"--pose-3d 는 3D 모델과만 같이 씁니다")
                client = GlossClient(device)
                print(f"[cloud] 기기 인식(B): {device.feature_set}, "
                      f"{len(device.labels) - 1}개 글로스, 서버에는 글로스만 보냅니다")
            else:
                client = CloudClient()
            cloud = CloudWorker(client, stats=stats)
        except (OSError, ValueError) as e:
            ap.error(str(e))
        recorder = SegmentRecorder()
        if a.auto_segment:
            from src.segmenter import ActivitySegmenter
            segmenter = ActivitySegmenter(viewpoint=a.viewpoint,
                                          rest_ratio=C.SEGMENT_REST_RATIO)
        cloud.start()
        if a.cloud_controls == "stdin":
            def read_commands():
                for line in sys.stdin:
                    commands.put(line.strip().lower())
            threading.Thread(target=read_commands, daemon=True, name="cloud-controls").start()
            print("[cloud] start / stop / cancel / quit 입력")
        else:
            print("[cloud] SPACE=문장 시작/종료, c=취소. 자동 발화는 서버 검증 후 활성화됩니다.")

    # 수어도 얼굴 앵커도 사용하지 않는 안정형 Phase 2는 카메라가 필요 없습니다.
    try:
        need_vision, separate_face, sign_index = resolve_camera_plan(
            a.no_sign, a.fixed_caption, a.sign_camera, a.face_camera)
    except ValueError as e:
        ap.error(str(e))
    vq = queue.Queue(maxsize=1) if need_vision else None
    vision = (VisionThread(vq, enable_sign=not a.no_sign and cloud is None, stats=stats,
                           camera_index=sign_index, segment_recorder=recorder,
                           pose_3d=a.pose_3d, segmenter=segmenter, segment_events=commands,
                           track_face=not a.fixed_caption and not separate_face)
              if need_vision else None)
    if vision:
        vision.start()

    fq = queue.Queue(maxsize=1) if separate_face else None
    face_thread = FaceThread(fq, a.face_camera) if separate_face else None
    if face_thread:
        face_thread.start()

    sq = None
    stt = None
    if not a.no_stt:
        from src.stt_worker import SttWorker
        sq = queue.Queue()
        stt = SttWorker(sq, gate=gate, backend=a.stt)
        stt.start()

    tts = None
    if not a.no_tts:
        from src.tts_player import TtsPlayer
        tts = TtsPlayer(gate=gate, stats=stats)
        tts.start()

    hud = HudRenderer()
    content = HudContent(hud)
    display = make_display(a.display)
    phrase = PhraseBuilder()
    if cloud is not None:
        from src.cloud_control import CloudController
        controller = CloudController(cloud, recorder, content, tts=tts, stats=stats)

    pending_word_t = 0.0        # 아직 발화 안 된 단어가 확정된 시각
    last_face, last_size = None, (C.CAM_W, C.CAM_H)

    has_window = bool((a.debug and (vision or face_thread))
                      or getattr(display, "has_window", False))

    print("실행 중. " + ("q 로 종료." if has_window else "Ctrl+C 로 종료."))

    try:
        while True:
            # ---- 비전 ----
            try:
                if vq is None:
                    raise queue.Empty
                kind, payload = vq.get(timeout=0.05)
                if kind == "error":
                    print("ERROR:", payload)
                    break
                if face_thread is None:
                    last_face = payload["face"]
                    last_size = payload["size"]
                if payload["word"]:
                    print(f"[수어] {payload['word']}")
                    # 여기서 바로 발화하지 않습니다. 조합 규칙이 완성될 수도
                    # 있어서, 완성되면 문장으로 / 아니면 poll() 이 흘려보냅니다.
                    sentence = phrase.feed(payload["word"])
                    if sentence:
                        speak(sentence, tts, stats, payload["word_t"], show=content.own)
                    else:
                        pending_word_t = payload["word_t"]
                if a.debug:
                    v = payload["frame"]
                    if C.MIRROR_PREVIEW:
                        v = cv2.flip(v, 1)
                    txt = (f"{payload['fps']:.0f}fps  "
                           f"win {payload['in_fps']:.0f}fps  "
                           f"vis {payload['ms_vision']:.0f}ms  "
                           f"inf {payload['ms_infer']:.0f}ms  "
                           f"{payload['raw']} {payload['conf']:.2f}")
                    cv2.putText(v, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    if not gate.is_open():
                        cv2.putText(v, "MIC MUTED (TTS)", (16, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if recorder is not None:
                        draw_segment_overlay(cv2, v, recorder, segmenter)
                    cv2.imshow("camera (debug)", v)
            except queue.Empty:
                pass

            # 별도 전방 카메라가 있으면 얼굴 좌표는 이쪽이 최종 권한을 가집니다.
            if fq is not None:
                try:
                    kind, payload = fq.get_nowait()
                    if kind == "error":
                        print("[face]", payload, "-> 고정 자막으로 계속합니다")
                        fq = None
                        last_face = None
                    else:
                        last_face, last_size = payload["face"], payload["size"]
                        if a.debug:
                            fv = payload["frame"]
                            if last_face:
                                cv2.drawMarker(fv, (int(last_face[0]), int(last_face[1])),
                                               (0, 255, 255), cv2.MARKER_CROSS, 32, 2)
                            cv2.imshow("face camera (debug)", fv)
                except queue.Empty:
                    pass

            # 조합 대기가 끝난 단어를 흘려보냅니다 (이중 발화 방지의 나머지 절반)
            pend = phrase.poll()
            if pend:
                speak(pend, tts, stats, pending_word_t, show=content.own)
                pending_word_t = 0.0

            if controller is not None:
                while not commands.empty():
                    command = commands.get_nowait()
                    if command == "quit":
                        raise KeyboardInterrupt
                    controller.command(command)
                controller.tick()

            # ---- STT ----
            if sq is not None:
                while not sq.empty():
                    kind, text = sq.get_nowait()
                    if kind == "error":
                        print("[stt]", text)
                        sq = None
                        break
                    content.partner(text, partial=(kind == "partial"))
                    if kind == "final":
                        print(f"[음성] {text}")

            # ---- HUD ----
            # 실물 HUD 는 한 장에 더 최근 내용을, 노트북 미리보기는 두 영역을 함께.
            face_xy = None if a.fixed_caption else last_face
            with stats.span("render"):
                img = (content.dual(face_xy, last_size) if getattr(display, "dual", False)
                       else content.frame(face_xy, last_size))
            with stats.span("display"):
                display.show(img)

            # 창이 하나도 없으면 waitKey 가 키를 못 받고 루프가 CPU 를 100%
            # 태웁니다(--display null 이고 --debug 도 아닐 때). 그땐 sleep 으로
            # 양보하고 Ctrl+C 로 종료하게 둡니다.
            if has_window:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if controller is not None and a.cloud_controls == "keyboard":
                    if key == ord(" "):
                        controller.toggle()
                    elif key == ord("c"):
                        controller.command("cancel")
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.cancel()
        if cloud is not None:
            cloud.stop()
            cloud.join(timeout=2.0)
        # 비전 스레드는 카메라와 Holistic 을 들고 있습니다. 기다리지 않고
        # 프로세스를 내리면 카메라 장치가 열린 채로 남아, 다음 실행에서
        # "카메라를 열 수 없습니다" 가 뜹니다(macOS 에서 특히 잘 납니다).
        if vision:
            vision.stop()
            vision.join(timeout=2.0)
            if vision.is_alive():
                print("[main] 비전 스레드가 시간 안에 끝나지 않았습니다")
        if face_thread:
            face_thread.stop()
            face_thread.join(timeout=2.0)
            if face_thread.is_alive():
                print("[main] 얼굴 스레드가 시간 안에 끝나지 않았습니다")
        if stt:
            stt.stop()
            stt.join(timeout=2.0)
        if tts:
            tts.stop()
            tts.join(timeout=2.0)
        display.close()
        cv2.destroyAllWindows()
        print("종료")
        if a.stats is not None:
            print(stats.report())
            if a.stats:
                saved = stats.save_csv(a.stats)
                if saved:
                    print(f"  원본 표본 저장(실행 전체): {saved}")
            stats.close()


if __name__ == "__main__":
    main()
