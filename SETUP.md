# signglass 환경 설정 가이드

수어-음성 양방향 스마트글래스 프로젝트의 개발 환경을 구성합니다.
**하드웨어 없이** 여기까지 끝낼 수 있고, 비용은 0원입니다.

2026-09-08 현재 프로젝트 가상환경에는 PyTorch 2.4.1, ONNX 1.16.2,
scikit-learn 1.5.2가 설치됐고 학습→ONNX 테스트가 통과했습니다. Vosk 한국어 모델,
Galmuri11 v2.40.4 및 라이선스, Yuna TTS WAV 22개도 준비돼 있습니다.
실제 파일/디코딩 검사 결과는 `reports/assets.json`에 기록했습니다.

현재 설치된 MediaPipe는 바이너리에 arm64/x86_64가 모두 들어 있고 import는 성공하지만,
내부 WHEEL 메타데이터가 x86_64만 선언해 `pip check`에서 플랫폼 경고가 남습니다.
공식 배포 파일 선택은 universal2입니다. 이 경고를 숨기기 위해 메타데이터를 수정하지 않았습니다.
실기기 전체 경로를 검증할 때 함께 확인해야 합니다.

---

## 0. 왜 버전을 고정하는가

이 프로젝트가 첫날 막히는 이유는 거의 항상 아래 둘 중 하나입니다.

| 함정 | 증상 | 해결 |
|---|---|---|
| mediapipe 최신 버전 설치 | `module 'mediapipe' has no attribute 'solutions'` | `mediapipe==0.10.21` 고정 |
| numpy 2.x 설치 | mediapipe import 시 ABI 에러 | `numpy==1.26.4` 고정 |

구글은 2023년 3월에 MediaPipe 레거시 솔루션(Holistic 포함) 지원을 종료했고,
0.10.31 버전부터 파이썬 패키지에서 실제로 제거했습니다.
Tasks API에는 아직 Holistic 대체품이 없으므로 **구버전 고정이 유일한 길**입니다.

---

## 1. Python 3.11 준비

`python --version` 이 3.11.x 가 아니면 3.11을 따로 설치하세요.

- Windows: python.org 에서 3.11 설치 시 **"Add python.exe to PATH"** 체크
- macOS: `brew install python@3.11`
- Ubuntu: `sudo apt install python3.11 python3.11-venv`

> 프로젝트의 고정 MediaPipe/학습 조합은 Python 3.11을 기준으로 검증합니다.

---

## 2. 가상환경 + 설치

### Windows (PowerShell)
```powershell
mkdir signglass; cd signglass
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> 실행 정책 에러가 나면:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### macOS / Linux
```bash
mkdir signglass && cd signglass
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> Ubuntu 는 오디오용 시스템 패키지가 하나 더 필요합니다:
> `sudo apt install -y libportaudio2 libgl1`
>
> 학습은 PyTorch를 사용하며 macOS/Windows/Linux에서 같은 코드 경로를 씁니다.

**설치에 5~10분, 약 1.5GB 정도 걸립니다.**

---

## 3. 점검

```bash
python check_env.py
```

전부 `[OK]` 여야 다음 단계로 갑니다. 특히 이 두 줄을 확인하세요:

```
[OK]   mediapipe 0.10.21 — solutions.holistic 사용 가능
[OK]   프레임당 22ms (약 45 fps) — 실시간 가능
```

Holistic 이 40ms를 넘으면 카메라 해상도를 640x480 으로,
`model_complexity=0` 으로 낮추세요. 수어 인식엔 충분합니다.

---

## 4. 무료 리소스 내려받기

### 4-1. Vosk 한국어 STT 모델 (82MB, Apache 2.0)
```
https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip
```
압축을 풀어 `models/vosk-ko/` 에 넣으세요.

> Zeroth 테스트 기준 WER 약 28% 입니다. 완벽하진 않지만
> **오프라인 · 스트리밍 · 부분결과 지원**이라 실시간 자막에는 이쪽이 유리합니다.
> 정확도 비교가 필요하면 나중에 faster-whisper small 을 추가하세요.

### 4-2. 갈무리(Galmuri) 픽셀 한글 폰트 (무료)
128x64 OLED 에 한글을 그리려면 저해상도용 비트맵 폰트가 필요합니다.
일반 TTF 를 11px로 렌더하면 뭉개집니다. `assets/fonts/` 에 넣으세요.

### 4-3. 수어 어휘 레퍼런스
- 국립국어원 한국수어사전: `sldict.korean.go.kr`
  → 15개 어휘의 **정확한 수형**을 여기서 확인하고 촬영하세요.
- AI Hub 수어 데이터셋 (회원가입 후 무료)

---

## 5. 폴더 구조

```
signglass/
├── .venv/
├── config.py               ★ 모든 튜닝은 여기서만
├── main.py                 ★ 통합 실행
├── check_env.py
├── requirements.txt
├── models/
│   ├── vosk-ko/                # 4-1 에서 받은 STT 모델
│   ├── sign_lstm.pt            # TorchScript 백업
│   ├── sign_lstm.onnx          # 실제 실행 모델
│   └── labels.json
├── assets/
│   ├── fonts/Galmuri11.ttf     # 저해상도 한글 픽셀 폰트
│   └── tts_cache/*.wav         # 미리 뽑아둔 음성 (지연 0ms 핵심)
├── data/raw/dataset_meta.json             # 특징 버전/차원/길이/FPS 계약
├── data/raw/{어휘}/{촬영자}__{세션}_{번호}.npy  # shape (30, 246)
├── firmware/
│   └── oled_bridge.ino         # 아두이노 USB↔SPI 브리지
└── src/
    ├── landmarks.py            # 추출 + 정규화 (246차원)  ← 가장 중요
    ├── camera.py               # 공용 카메라 (수동 노출 + 재연결)
    ├── audio_gate.py           # 반이중 게이트 (에코 루프 차단)
    ├── one_euro.py             # 저지연 좌표 스무딩
    ├── calibrate.py            # 시차 보정 도구
    ├── collect.py              # Step 1-1 데이터 수집기
    ├── train.py                # Step 1-2 LSTM 학습
    ├── predictor.py            # 슬라이딩 추론 + 확정 + 문장화
    ├── stt_worker.py           # vosk 스트리밍 스레드
    ├── build_tts_cache.py      # TTS wav 사전 생성 (1회)
    ├── tts_player.py           # 비동기 재생
    ├── hud.py                  # 좌표변환 + 말풍선 + SSD1306 패킹
    └── oled_bridge.py          # preview / serial / null 백엔드
```

---

## 5-2. 실행 순서

```bash
python check_env.py                 # 0. 점검
python -m src.build_tts_cache       # 1. TTS 캐시 (1회, 인터넷 필요)
# macOS 내장 Yuna 음성으로 오프라인 캐시 생성:
# python -m src.build_tts_cache --engine macos
# 파일/모델 연결 검사 (실제 사람 STT 정확도와 별개):
# python tools/verify_assets.py --out reports/assets.json

# 2. 데이터 수집 — 사람/촬영일마다 SIGNER_ID / SESSION_ID 를 바꾸세요
python -m src.collect

pip install -r requirements-train.txt # 학습/ONNX 변환 의존성 (개발 PC에서만)
python -m src.train                 # 3. 학습 + ONNX 변환/검증 (CPU)
python -m src.train --split group   #    촬영자 3명 이상이면 이걸로 진짜 성능 측정

python -m src.calibrate --display serial  # 4. 실제 OLED 시차 보정 (Pi는 pi)
python main.py --debug              # 5. 통합 실행
```

부분 실행:

| 명령 | 용도 |
|---|---|
| `python main.py --no-stt` | Phase 1 (수어→음성)만 |
| `python main.py --no-sign` | Phase 2 (음성→자막)만 |
| `python main.py --no-sign --fixed-caption` | 카메라 없는 고정 자막 안정형 |
| `python main.py --sign-camera 0 --face-camera 1` | 하향 손/전방 얼굴 두 카메라 |
| `python main.py --no-gate` | 반이중 해제 (유선 이어폰으로 들을 때만) |
| `python main.py --display serial` | 실제 OLED 출력 (Step 3) |
| `python -m src.stt_worker` | STT 단독 테스트 |
| `python -m src.tts_player` | TTS 단독 테스트 |
| `python -m src.hud` | 말풍선 레이아웃 미리보기 |
| `python -m src.calibrate --display serial` | 실제 OLED 시차 보정값 조정·저장 |

`--debug` 화면에 `vis 21ms  inf 4ms` 처럼 구간별 지연이 표시됩니다.
`vis` 가 40ms 를 넘으면 해상도를 640×480으로 낮추세요.

`DISPLAY_BACKEND = "preview"` 이면 **하드웨어 없이** HUD 레이아웃·좌표 매핑·
가독성까지 전부 검증됩니다. Step 1~2 를 0원으로 끝낼 수 있는 지점입니다.

---

## 6. 자주 나는 에러

| 에러 | 원인 | 해결 |
|---|---|---|
| `no attribute 'solutions'` | mediapipe 버전 | `pip install mediapipe==0.10.21` |
| `numpy.dtype size changed` | numpy 2.x | `pip install numpy==1.26.4` |
| `PortAudio library not found` | 리눅스 시스템 패키지 | `sudo apt install libportaudio2` |
| 카메라 열기 실패 | 다른 앱이 점유 / OS 권한 | Zoom·Teams 종료, 카메라 권한 허용 |
| 카메라 첫 프레임 3~5초 지연 | Windows MSMF 백엔드 | `cv2.VideoCapture(0, cv2.CAP_DSHOW)` |
| 시간이 갈수록 영상이 밀림 | 큐에 프레임 누적 | `Queue(maxsize=1)` + 가득 차면 버리기 |
| 다른 스레드에서 `cv2.imshow` 시 크래시 | GUI 는 메인 스레드 전용 | 렌더링은 메인에서만 |
| pip 설치가 매우 느림 | 기본 인덱스 | `pip install -i https://pypi.org/simple ...` |

---

## 7. 데이터 수집 원칙

환경이 준비되면 **Step 1-1: 데이터 수집기**를 실행합니다.
그 전에 어휘 목록(10~15개)을 먼저 확정하세요. 어휘를 고를 때 기준:

1. **손 모양이 서로 확실히 다를 것** — 비슷한 수형은 15개 규모 모델이 절대 못 가릅니다
2. **정적 수형이 아니라 궤적이 있을 것** — LSTM의 장점을 살립니다
3. **시연 시나리오 하나로 엮일 것** — "안녕하세요 / 감사합니다 / 도와주세요 / 얼마예요" 같은
   실제 대화 한 편이 나오면 발표 임팩트가 완전히 달라집니다
4. **무동작(Neutral) 클래스를 반드시 포함** — 없으면 가만히 있어도 아무 단어나 뱉습니다
