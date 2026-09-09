#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
signglass 환경 점검 스크립트
---------------------------
설치가 끝난 뒤 가장 먼저 이걸 돌리세요.
전부 [OK] 가 나와야 Step 1 코딩을 시작할 수 있습니다.

    python check_env.py

옵션:
    --no-camera   카메라 테스트 건너뛰기
    --no-mic      마이크 테스트 건너뛰기
    --no-tts      TTS 테스트 건너뛰기 (소리가 납니다)
"""

import sys
import platform
import argparse

import config as C

PASS, WARN, FAIL = [], [], []


def ok(msg):
    print(f"  \033[92m[OK]\033[0m   {msg}")
    PASS.append(msg)


def warn(msg, fix=""):
    print(f"  \033[93m[WARN]\033[0m {msg}")
    if fix:
        print(f"         -> {fix}")
    WARN.append(msg)


def fail(msg, fix=""):
    print(f"  \033[91m[FAIL]\033[0m {msg}")
    if fix:
        print(f"         -> {fix}")
    FAIL.append(msg)


def section(title):
    print(f"\n── {title} " + "─" * max(0, 50 - len(title)))


# ---------------------------------------------------------------- 0. 시스템
def check_python():
    section("0. 파이썬")
    v = sys.version_info
    print(f"  {platform.python_version()} / {platform.system()} {platform.machine()}")
    if (v.major, v.minor) == (3, 11):
        ok("Python 3.11 (권장 버전)")
    elif (v.major, v.minor) in [(3, 9), (3, 10), (3, 12)]:
        warn(f"Python {v.major}.{v.minor} - 동작하지만 3.11 권장")
    else:
        fail(
            f"Python {v.major}.{v.minor} 는 지원 조합 밖입니다",
            "3.11 로 새 가상환경을 만드세요 (mediapipe 0.10.21 은 3.12 까지 지원)",
        )


# ------------------------------------------------------- 1. 핵심 라이브러리
def check_core():
    section("1. 핵심 라이브러리")

    # numpy: mediapipe 가 <2 를 요구
    numpy_ok = False
    try:
        import numpy as np

        if np.__version__.startswith("1."):
            ok(f"numpy {np.__version__}")
            numpy_ok = True
        else:
            fail(
                f"numpy {np.__version__} (2.x)",
                "pip install 'numpy==1.26.4' — mediapipe 가 numpy<2 를 요구합니다",
            )
    except ImportError:
        fail("numpy 없음", "pip install -r requirements.txt")
        # ★ 예전에는 여기서 return 해서 아래 mediapipe/opencv 검사가
        #   전부 생략됐습니다. numpy 가 없어도 나머지는 계속 검사합니다.
        #   (mediapipe 는 numpy 를 필요로 하므로 아래에서 다시 실패로 잡힙니다)

    # mediapipe: solutions.holistic 존재 여부가 이 프로젝트의 생명줄
    try:
        import mediapipe as mp

        ver = getattr(mp, "__version__", "unknown")
        if not hasattr(mp, "solutions"):
            fail(
                f"mediapipe {ver} 에 solutions 모듈이 없습니다",
                "pip install 'mediapipe==0.10.21' — 0.10.31+ 는 레거시 솔루션이 삭제됐습니다",
            )
        elif not hasattr(mp.solutions, "holistic"):
            fail(f"mediapipe {ver}: solutions.holistic 없음", "0.10.21 로 다운그레이드")
        else:
            ok(f"mediapipe {ver} — solutions.holistic 사용 가능")
            if not hasattr(mp.solutions, "face_detection"):
                warn("solutions.face_detection 없음 (Phase 2 말풍선 위치 검출에 필요)")
    except ImportError:
        fail("mediapipe 없음", "pip install -r requirements.txt")

    # opencv: mediapipe 가 contrib 판을 끌고 옴
    try:
        import cv2

        ok(f"opencv {cv2.__version__}")
    except ImportError:
        fail("opencv 없음", "mediapipe 설치 시 자동으로 깔립니다. 설치 로그를 확인하세요")

    # onnxruntime: 안경에서 실제로 모델을 돌리는 유일한 런타임
    try:
        import onnxruntime as ort

        provs = ort.get_available_providers()
        ok(f"onnxruntime {ort.__version__}  providers={provs}")
    except ImportError:
        fail("onnxruntime 없음", "pip install -r requirements.txt")

    # 학습 프레임워크는 '있으면 알려주고, 없어도 정상'입니다.
    # 안경 실행에는 필요 없고 requirements-train.txt 쪽에 있습니다.
    for name, label in (("torch", "PyTorch"), ("tensorflow", "TensorFlow")):
        try:
            m = __import__(name)
            print(f"  [정보] {label} {m.__version__} 설치됨 (학습·변환용)")
        except Exception:
            pass

    try:
        import sklearn

        ok(f"scikit-learn {sklearn.__version__}")
    except ImportError:
        warn("scikit-learn 없음 (학습 평가에 필요)")

    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401

        ok("pillow (HUD 비트맵 렌더링)")
    except ImportError:
        warn("pillow 없음")


# ------------------------------------------------------------- 2. 오디오 SW
def check_audio_libs():
    section("2. 오디오 / 음성 라이브러리")

    try:
        import sounddevice  # noqa: F401

        ok("sounddevice")
    except Exception as e:
        fail(f"sounddevice 사용 불가: {e}", "리눅스면 'sudo apt install libportaudio2' 필요")

    try:
        import vosk  # noqa: F401

        ok("vosk")
    except ImportError:
        warn("vosk 없음 (실시간 한국어 STT)")

    try:
        import webrtcvad  # noqa: F401

        ok("webrtcvad (발화 구간 검출)")
    except ImportError:
        warn("webrtcvad 없음", "pip install webrtcvad-wheels")

    try:
        import pyttsx3  # noqa: F401

        ok("pyttsx3")
    except ImportError:
        warn("pyttsx3 없음")

    try:
        import serial  # noqa: F401

        ok("pyserial (Step 2 OLED 브리지용)")
    except ImportError:
        warn("pyserial 없음 - Step 2 에서 필요")


# --------------------------------------------------------- 3. 카메라 실측
def check_camera():
    section("3. 카메라 실측")
    try:
        import cv2
        import time
    except ImportError:
        fail("opencv 없이는 카메라 테스트 불가")
        return

    from src.camera import open_camera
    cap = open_camera(verbose=False)
    if cap is None:
        fail(f"카메라(index {C.CAM_INDEX}) 열기 실패",
             "다른 앱이 점유 중인지, OS 카메라 권한이 켜져 있는지 확인")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n, t0 = 0, time.time()
    for _ in range(40):
        r, _f = cap.read()
        if r:
            n += 1
    fps = n / max(time.time() - t0, 1e-6)
    cap.release()

    if n == 0:
        fail("프레임을 한 장도 못 읽었습니다")
        return

    ok(f"해상도 {w}x{h}, 실측 {fps:.1f} fps")
    if fps < 15:
        warn(f"{fps:.1f} fps 는 수어 인식에 부족합니다", "해상도를 640x480 으로 낮춰보세요")
    if w < 1280:
        warn(f"{w}x{h} — 얼굴 위치 정밀도가 떨어질 수 있습니다 (Phase 2)")


# ------------------------------------------------- 4. Holistic 추론 속도
def check_holistic_speed():
    section("4. Holistic 추론 속도 (CPU)")
    try:
        import time
        import numpy as np
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            fail("solutions 없음 - 위 1번 항목을 먼저 해결하세요")
            return

        img = np.random.randint(0, 255, (C.CAM_H, C.CAM_W, 3), dtype=np.uint8)
        with mp.solutions.holistic.Holistic(model_complexity=C.MODEL_COMPLEXITY) as h:
            h.process(img)  # 워밍업
            t0 = time.time()
            for _ in range(10):
                h.process(img)
            ms = (time.time() - t0) / 10 * 1000

        if ms < 40:
            ok(f"프레임당 {ms:.0f}ms (약 {1000/ms:.0f} fps) — 실시간 가능")
        elif ms < 80:
            warn(f"프레임당 {ms:.0f}ms — 해상도를 640x480 으로 낮추길 권장")
        else:
            warn(f"프레임당 {ms:.0f}ms — 느립니다", "640x480 + model_complexity=0 조합 필수")
    except Exception as e:
        fail(f"Holistic 실행 실패: {e}")


# ---------------------------------------------------------- 5. 마이크 실측
def check_mic():
    section("5. 마이크 실측 (2초 녹음)")
    try:
        import sounddevice as sd
        import numpy as np

        ins = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        if not ins:
            fail("입력 장치 없음", "OS 마이크 권한 및 장치 연결 확인")
            return
        print(f"  기본 입력: {sd.query_devices(kind='input')['name']}")

        print("  말해보세요... (2초)")
        rec = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(rec**2)))
        peak = float(np.max(np.abs(rec)))

        if peak < 0.005:
            fail(f"거의 무음 (peak {peak:.4f})", "OS 입력 볼륨 / 음소거 / 권한 확인")
        elif rms < 0.01:
            warn(f"입력이 작습니다 (rms {rms:.4f})", "OS 마이크 볼륨을 올리세요")
        else:
            ok(f"정상 수음 (rms {rms:.4f}, peak {peak:.3f})")
        if peak > 0.99:
            warn("클리핑 발생 - 입력 볼륨을 낮추세요")
    except Exception as e:
        fail(f"마이크 테스트 실패: {e}")


# ------------------------------------------------------- 6. 한국어 TTS
def check_tts():
    section("6. 한국어 TTS")
    try:
        import pyttsx3

        eng = pyttsx3.init()
        voices = eng.getProperty("voices")
        ko = [v for v in voices if "ko" in str(getattr(v, "id", "")).lower()
              or "korean" in str(getattr(v, "name", "")).lower()
              or "heami" in str(getattr(v, "name", "")).lower()]
        if ko:
            ok(f"한국어 음성 발견: {ko[0].name}")
            eng.setProperty("voice", ko[0].id)
        else:
            warn(
                "OS에 한국어 TTS 음성이 없습니다",
                "윈도우: 설정 > 시간 및 언어 > 음성 > 음성 추가 에서 한국어 설치. "
                "또는 edge-tts 로 wav 를 미리 캐싱하세요 (권장)",
            )
        eng.say("환경 점검을 완료했습니다")
        eng.runAndWait()
        ok("TTS 재생 성공")
    except Exception as e:
        warn(f"pyttsx3 실패: {e}", "edge-tts 사전 캐싱 방식으로 대체 가능합니다")


# --------------------------------------------------------- 7. 실행 산출물
def check_artifacts():
    section("7. 전 구간 실행 산출물")

    samples = list(C.DATA_DIR.rglob("*.npy")) if C.DATA_DIR.exists() else []
    if samples:
        try:
            from src.dataset_contract import ensure_dataset_contract
            ensure_dataset_contract(create=False)
            ok(f"직접 수집 데이터 {len(samples)}개 + 전처리 계약 일치")
        except RuntimeError as e:
            fail(f"직접 수집 데이터 계약 오류: {e}")
    else:
        fail("직접 수집 데이터 없음", "python -m src.collect")

    if C.MODEL_PATH.exists():
        ok(f"수어 ONNX 모델: {C.MODEL_PATH.name}")
    else:
        fail(f"수어 ONNX 모델 없음: {C.MODEL_PATH}",
             "데이터 수집 후 python -m src.train으로 생성하세요")

    if C.LABELS_PATH.exists():
        ok(f"라벨 파일: {C.LABELS_PATH.name}")
    else:
        fail(f"라벨 파일 없음: {C.LABELS_PATH}", "src.train 실행 시 함께 생성됩니다")

    if C.VOSK_MODEL.exists() and any(C.VOSK_MODEL.iterdir()):
        ok(f"Vosk 한국어 모델: {C.VOSK_MODEL.name}")
    else:
        fail(f"Vosk 한국어 모델 없음: {C.VOSK_MODEL}",
             "vosk-model-small-ko-0.22를 내려받아 이 경로에 압축 해제하세요")

    wavs = list(C.TTS_CACHE.glob("*.wav")) if C.TTS_CACHE.exists() else []
    if wavs:
        ok(f"TTS 캐시 {len(wavs)}개")
    else:
        warn("TTS 캐시 없음", "python -m src.build_tts_cache")

    if C.HUD_FONT.exists():
        ok(f"HUD 한글 폰트: {C.HUD_FONT.name}")
    else:
        warn(f"HUD 픽셀 폰트 없음: {C.HUD_FONT}",
             "실물 OLED 전에 한글 저해상도 폰트를 넣고 가독성을 확인하세요")


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-mic", action="store_true")
    p.add_argument("--no-tts", action="store_true")
    a = p.parse_args()

    print("=" * 58)
    print("  signglass 환경 점검")
    print("=" * 58)

    check_python()
    check_core()
    check_audio_libs()
    if not a.no_camera:
        check_camera()
        check_holistic_speed()
    if not a.no_mic:
        check_mic()
    if not a.no_tts:
        check_tts()
    check_artifacts()

    print("\n" + "=" * 58)
    print(f"  통과 {len(PASS)} / 경고 {len(WARN)} / 실패 {len(FAIL)}")
    print("=" * 58)
    if FAIL:
        print("\n반드시 해결해야 할 항목:")
        for m in FAIL:
            print(f"  - {m}")
        sys.exit(1)
    print("\n환경 준비 완료. Step 1 (데이터 수집기) 로 넘어가세요.")


if __name__ == "__main__":
    main()
