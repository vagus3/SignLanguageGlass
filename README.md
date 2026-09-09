# signglass — 청각장애인 양방향 소통 스마트글래스

수어와 음성 사이를 **양방향으로** 잇는 안경형 장치입니다.

| 방향 | 이름 | 하는 일 |
|---|---|---|
| 착용자 → 상대방 | **Phase 1** | 수어를 인식해 안경 스피커로 **음성** 출력 |
| 상대방 → 착용자 | **Phase 2** | 상대의 말을 받아써서 상대 얼굴 옆에 **AR 말풍선** |

두 방향이 동시에 돌아가고, 서로의 소리가 되먹임되지 않도록 반이중 게이트가 사이에 있습니다.

---

## 지금 상태 (정직하게)

**프로토타입 소스는 대부분 구현됐지만, 제품의 핵심 가설과 전 구간 통합은 아직 검증되지 않았습니다.**

| | 상태 |
|---|---|
| 소스 모듈·회귀 테스트 | 🟡 20개 모두 통과 (PyTorch→ONNX 포함); 실기기 관통 테스트 미완료 |
| 학습 의존성 | ✅ PyTorch 2.4.1 / ONNX 1.16.2 / scikit-learn 1.5.2 설치 |
| 학습된 모델 `models/sign_lstm.onnx` | ❌ 없음 — 데이터 수집·학습 필요 |
| 직접 수집한 데이터 `data/raw/**/*.npy` | ❌ 0개 |
| AI Hub 샘플 | 🟡 3D 변환 완료: 100클립→21시퀀스/20동작 그룹; 한국어 뜻 매핑·추가 화자 필요 |
| vosk 한국어 STT 모델 `models/vosk-ko/` | ✅ 공식 모델 설치·디코딩 확인; 합성음성 3문장 오인식, 품질 평가 필요 |
| TTS 캐시 `assets/tts_cache/*.wav` | ✅ macOS Yuna 22개, 16kHz PCM 모노·무음 아님 확인 |
| 한글 픽셀 폰트 `assets/fonts/Galmuri11.ttf` | ✅ Galmuri v2.40.4 + LICENSE.txt 설치 |
| 하드웨어 (OLED / 광학계) | ❌ 미착수 |

가장 먼저 검증할 것은 모델 크기가 아니라 **실제 장착 위치에서 손 검출률과 얼굴 검출률이
각각 충분한가**입니다. 기본 한 카메라, `--face-camera`를 쓰는 하향 손/전방 얼굴 두 카메라,
카메라가 전혀 필요 없는 `--fixed-caption`까지 같은 코드에 유지합니다. 2주차 실측으로 최종
장착 구성을 고르되, 얼굴 AR 목표를 미리 버리지는 않습니다.

2명·12주 기준 권장 범위, 주차별 게이트, 평가 지표는
[`PROJECT_REVIEW.md`](PROJECT_REVIEW.md)에 정리했습니다.
경진대회 당일 전체/폴백 모드와 2인 역할은 [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md)를 따릅니다.
AI Hub 3D 활용 및 이전 검토 정정은 [`AIHUB_TRANSFER.md`](AIHUB_TRANSFER.md)에 정리했습니다.

---

## 아키텍처

### 스레드 구성

GUI는 메인 스레드 전용이고, 무거운 일은 전부 워커로 뺍니다.
MediaPipe와 ONNX는 C++ 구간에서 GIL을 놓으므로 실제로 병렬입니다.

```
┌─ [vision] ────────────────────────────────────────────┐
│  카메라 read → Holistic → 246차원 특징 → LSTM 추론      │
│  → 단어 확정                                            │
└──────────────────┬────────────────────────────────────┘
                   │ Queue(maxsize=1)  ← 가득 차면 오래된 것을 버림
  [face, 선택] 전방 카메라 → 경량 얼굴 검출 ────────────┤
                   ▼
┌─ [main] ──────────────────────────────────────────────┐
│  PhraseBuilder(단어열→문장) · HUD 렌더 · 디스플레이 출력 │
└───┬───────────────────────────────────┬───────────────┘
    │ say()                             ▲ Queue()
    ▼                                   │
┌─ [tts] ──────────────┐   ┌─ [stt] ────┴──────────────┐
│  캐시 wav 재생        │   │  vosk 스트리밍            │
└──────────┬───────────┘   └────────────▲──────────────┘
           │  게이트 닫음                │  게이트 열림일 때만 수용
           └────────►  AudioGate  ◄──────┘
                    (반이중: 말할 땐 안 듣는다)
```

비전·OLED 프레임 큐는 `maxsize=1`로 최신값만 유지하고, 오디오 큐는 약 1초로
제한합니다. 지연을 만드는 최대 원인은 처리 속도뿐 아니라 큐에 입력이 쌓이는 것입니다.

### Phase 1 — 수어 → 음성

```
MediaPipe Holistic
  │  pose 33 + 손 21×2 + 얼굴 468 랜드마크
  ▼
landmarks.extract()                         src/landmarks.py
  │  246차원 wearable-v3. 몸=어깨 기준, 손=손목/손바닥 기준
  │  어깨가 화면에서 잘려도 손 특징은 유지
  ▼
SignPredictor.push()                        src/predictor.py
  │  최근 1.0초를 30개로 리샘플 → ONNX LSTM → 5프레임마다 추론
  ▼
3단 확정
  │  ① 신뢰도 ≥ 0.80  ② 최근 5회 중 3회 다수결  ③ 확정 후 1.2초 쿨다운
  ▼
PhraseBuilder                               src/predictor.py
  │  "이름"+"무엇" → "이름이 뭐예요?"
  ▼
TtsPlayer → 안경 스피커                      src/tts_player.py
     미리 뽑아둔 wav 재생 (지연 0ms)
```

### Phase 2 — 음성 → AR 말풍선

```
안경 마이크
  ▼
SttWorker (vosk 스트리밍)                    src/stt_worker.py
  │  partial(말하는 중) / final(발화 확정)
  ▼                          landmarks.face_center_px()  ← 상대 얼굴 위치
HudRenderer.render()  ◄──────  OneEuroPoint 로 떨림 제거
  │                            hud.camera_to_hud() 로 시야·시차 보정
  │  말풍선을 얼굴 옆에. 자리 없으면 반대편. 시야 밖이면 화살표.
  ▼
oled_bridge → OLED                           src/oled_bridge.py
```

---

## 모듈 지도

```
config.py            ★ 모든 튜닝은 여기서만. 다른 모듈은 여기만 바라봄
main.py              ★ 통합 실행 (비전 스레드 + 메인 루프)

src/
  landmarks.py       ★ 랜드마크 추출·정규화 (246차원). 가장 중요한 모듈
  camera.py            공용 카메라 (수동 노출 + 끊김 재연결)
  one_euro.py          저지연 좌표 스무딩 (이동평균보다 지연이 적음)
  predictor.py       ★ 슬라이딩 추론 + 3단 확정 + 문장 조합
  runtime.py         ★ 통합 추론 런타임 (onnx / torch / keras)
  features_v2.py       AI Hub(OpenPose) ↔ MediaPipe 공통 부분집합, 101차원
  hand_shape3d.py      AI Hub 3D ↔ MediaPipe Hands world 공통 손 모양, 462차원
  hud.py             ★ 좌표변환 + 말풍선 렌더 + SSD1306 패킹
  oled_bridge.py       디스플레이 백엔드 (preview / serial / pi / null)
  stt_worker.py        vosk 스트리밍 STT 스레드
  tts_player.py        비동기 TTS 재생
  audio_gate.py        반이중 게이트 (에코 루프 차단)
  latency.py           지연 측정 하네스 (p50 / p95)
  collect.py           Step 1-1 데이터 수집기
  train.py             Step 1-2 LSTM 학습
  calibrate.py         시차(parallax) 보정 도구
  build_tts_cache.py   TTS wav 사전 생성 (1회)

tools/
  evaluate.py          ★ 저장된 ONNX 를 홀드아웃셋으로 평가
  demo.py              ★ 데모 시나리오 재생·녹화 (카메라·모델 불필요)
  to_onnx.py           외부/레거시 모델 → ONNX 변환 (기본 학습은 자동 변환)
  inspect_model.py     모델 형식 자동 판별·요약
  inspect_aihub.py     AI Hub 데이터셋 스캐너
  merge_parts.py       AI Hub 분할압축 병합
  check_mount.py       실제 장착 시점 손/얼굴 검출률과 FPS 측정
  prepare_aihub3d.py   3D 손 데이터 변환·중복 제거·원본 동작 그룹 감사
  verify_assets.py     모델/폰트/TTS 및 합성음성→STT 검사

tests/
  run_all.py           전체 테스트 실행 (pytest 불필요)
  test_*.py            회귀 테스트 20종 (학습 E2E는 선택 의존성 필요)

firmware/
  oled_bridge.ino      PC 구조에서만 필요한 USB↔SPI 브리지 펌웨어
```

```bash
python tests/run_all.py          # 전체
python tests/run_all.py -v       # 출력까지
python tests/run_all.py adaptive # 이름으로 골라서
```

테스트는 `pytest` 없이 표준 라이브러리만으로 돕니다. `requirements.txt`만 깔린
환경(= 안경에 올라가는 그 환경)에서 그대로 돌려야 하기 때문입니다.
각 테스트는 독립 프로세스로 실행됩니다 — 여러 테스트가 `config`를 임시로
갈아끼우기 때문에 한 프로세스에서 몰아 돌리면 서로 오염됩니다.

---

## 실행 구조 두 가지

### A. PC + 브리지 MCU

```
노트북 ──USB 시리얼──▶ 아두이노/Pico ──SPI──▶ OLED
(비전·STT·TTS 전부)      (그리기만)
```

PC에는 SPI 포트가 없어서 브리지가 필요합니다. 성능은 여유롭지만 안경이 노트북에 USB로 묶입니다.
`--display serial` / `requirements.txt` / `firmware/oled_bridge.ino`

### B. 라즈베리파이 온보드 (진짜 웨어러블)

```
Pi ──SPI(GPIO)──▶ OLED
(전부 Pi 에서)
```

**브리지 MCU도 펌웨어도 필요 없습니다.** `--display pi` / `requirements-pi.txt`

> **⚠ Pi에서는 mediapipe 버전이 다릅니다.**
> `mediapipe==0.10.21`은 Linux aarch64 휠이 **없어서** Pi에서 `pip install` 자체가 실패합니다.
> `0.10.15`에는 aarch64 휠이 cp39~cp312로 있고, `solutions.holistic`이 삭제된 건 0.10.31부터라
> 안전 구간입니다. `requirements-pi.txt`가 이걸 처리합니다.
>
> **⚠ 프레임레이트가 떨어지면 조용히 무너집니다.**
> 노트북 30fps로 학습한 모델을 Pi 10fps에서 돌리면 30프레임이 1초가 아니라 **3초**가 됩니다.
> 같은 수어인데 모델이 보는 시간 축이 3배가 되고, 에러는 하나도 안 납니다.
> `config.ADAPTIVE_WINDOW = True`(기본값)가 '최근 30프레임' 대신 **'최근 1.0초를 30개로
> 리샘플'**해서 이걸 막습니다.

---

## 설계 결정과 근거

이 프로젝트에서 "왜 이렇게 했는가"가 중요한 지점들입니다.

| 결정 | 근거 |
|---|---|
| **246차원** (얼굴 468점 전부가 아니라 24점) | 흔한 튜토리얼은 1662차원을 넣는데, 그러면 얼굴이 전체의 84%를 차지해 손 정보가 묻힙니다 |
| **착용형 손 정규화** | 몸은 어깨 기준, 손 모양은 손목/손바닥 기준으로 둡니다. 어깨가 잘려도 손이 0이 되지 않고, 손목 위치·크기도 보존합니다 |
| **얼굴은 눈 사이 거리로 따로 정규화** | 어깨 스케일로 나누면 표정 변화가 뭉개집니다. 한국수어의 의문·부정은 손이 아니라 표정입니다 |
| **STRIDE=5 슬라이딩 윈도우** | 30프레임을 다 채우고 추론하면 체감 지연 1초. 5프레임마다면 170ms |
| **3단 확정 (임계값+다수결+쿨다운)** | 한 번의 높은 신뢰도로 발화하면 손이 지나가기만 해도 말을 합니다 |
| **NEUTRAL 클래스 필수** | 없으면 가만히 있어도 아무 단어나 뱉습니다. 전체 데이터의 20~25%를 여기에 |
| **PhraseBuilder 지연 발화** | 즉시 발화하면 "이름" 뒤에 "이름이 뭐예요?"가 또 나와 두 번 들립니다 |
| **TTS 사전 캐싱** | 어휘가 고정이라 wav를 미리 뽑아둘 수 있습니다. 지연 0ms · 인터넷 불필요 · 시연 실패 0 |
| **반이중 게이트** | 스피커와 마이크가 안경테 위 몇 cm. 없으면 자기 말이 상대 자막으로 뜨고 무한 루프까지 갑니다 |
| **실행은 항상 ONNX** | torch 200MB / TF 600MB vs onnxruntime 50MB. 안경에 학습 프레임워크를 올리지 않습니다 |
| **One Euro Filter** | 이동평균은 부드럽게 할수록 지연이 그대로 늘어납니다. AR 오버레이 표준 |
| **화자 단위 평가** | 시퀀스 단위로 섞으면 "97%인데 남이 하면 30%"가 나옵니다 |
| **좌우 반전 증강 안 함** | 수어는 주손(dominant hand)이 의미를 가져서 뒤집으면 다른 동작이 됩니다 |

---

## 실행 및 팀 환경 준비

### 팀 협업 · GitHub 클론 후 준비

이 저장소는 **코드와 재현 절차만** GitHub으로 공유합니다. 각 컴퓨터에서 만들어지거나
용량·배포 조건을 따르는 파일은 `.gitignore`로 제외합니다. 따라서 `git clone`만으로는
STT와 수어 인식 전체를 실행할 수 없습니다.

| 항목 | GitHub 커밋 | 팀원이 준비할 방법 |
|---|---:|---|
| 소스코드·문서·`requirements*.txt`·테스트 | 포함 | `git clone` |
| Python 가상환경과 PyTorch / ONNX / Vosk **패키지** | 제외 | 아래 `pip install` 실행 |
| Vosk 한국어 **음성 모델** `models/vosk-ko/` | 제외 | [공식 다운로드](https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip)를 같은 경로에 압축 해제 |
| 학습 완료 ONNX 모델 `models/sign_lstm.onnx` | 제외 | [이 프로젝트 Releases](../../releases) / 팀 Drive에서 내려받거나 직접 학습 |
| AI Hub 원본·전처리 데이터 | 제외 | [AI Hub](https://aihub.or.kr/) 권한에 따라 별도 다운로드·전처리 |
| TTS wav 캐시·폰트 | 제외 | 아래 생성 명령 또는 팀 배포본 사용. 폰트는 [Galmuri Releases](https://github.com/quiple/galmuri/releases)에서 받음 |

여기서 **ONNX는 패키지와 모델 파일이 다릅니다.** `onnx`는 학습·변환용 Python
라이브러리이고, `sign_lstm.onnx`는 학습 후 만들어지는 실제 배포 모델입니다. 전자는
`requirements-train.txt`로 설치하고, 후자는 현재 GitHub에 올라가지 않습니다.

#### 새 팀원 초기 설정

```bash
git clone <저장소-주소>
cd signglass

# Python 3.11 권장
python3.11 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt    # 실행 환경
pip install -r requirements-train.txt  # 학습·ONNX 변환도 할 경우만

python check_env.py
```

그 다음 [Vosk 한국어 모델 ZIP](https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip)을
받아 압축을 풀고 폴더 이름을 `models/vosk-ko/`로 맞춥니다. [Galmuri v2.40.4 ZIP](https://github.com/quiple/galmuri/releases/download/v2.40.4/Galmuri-v2.40.4.zip)에서
`Galmuri11.ttf`와 `LICENSE.txt`를 `assets/fonts/`에 넣은 뒤 TTS 캐시를 생성합니다.

```bash
python -m src.build_tts_cache --engine macos  # macOS
# Windows에서는 --engine edge 사용 또는 팀에서 만든 wav 캐시를 전달
```

모델 학습이 끝나면 `sign_lstm.onnx`와 라벨/전처리 메타데이터를 **[이 프로젝트 Releases](../../releases)
또는 팀 공유 Drive**에 함께 배포하세요. `.venv`, AI Hub 원본, Vosk 모델을 Git에 강제로 넣으면
저장소가 무거워지고 배포·라이선스 관리가 어려워집니다. 커밋 전에는 아래 명령으로 제외 대상이
정상인지 확인합니다.

```bash
git status --short
git status --ignored  # .venv, models/vosk-ko, data/raw 등이 ignored로 보여야 함
```

### 준비 (한 번만)

```bash
python check_env.py                  # 환경 점검 — 전부 [OK] 여야 함
# models/vosk-ko/  ← vosk-model-small-ko-0.22 압축 해제
# assets/fonts/Galmuri11.ttf  ← 저해상도 한글 픽셀 폰트
python -m src.build_tts_cache        # TTS wav 사전 생성 (인터넷 필요, 1회)
```

### 데이터 → 모델

```bash
# 촬영자/촬영일마다 config.py 의 SIGNER_ID / SESSION_ID 를 바꾸세요
python -m src.collect                # SPACE 녹화 / n,p 라벨이동 / u 취소 / q 종료
pip install -r requirements-train.txt
python -m src.train --split group    # ONNX까지 생성. 촬영자 3명 이상 권장
python tools/evaluate.py --split group             # 배포하는 그 파일을 평가
```

### 실행

| 명령 | 용도 |
|---|---|
| `python main.py --debug` | 통합 실행 + 카메라 뷰 |
| `python main.py --no-stt` | Phase 1(수어→음성)만 |
| `python main.py --no-sign` | Phase 2(음성→자막)만 — **모델 없이도 동작** |
| `python main.py --no-sign --fixed-caption` | 카메라 없이 하단 고정 자막 — 현장 안정형 폴백 |
| `python main.py --sign-camera 0 --face-camera 1` | 하향 수어/전방 얼굴 두 카메라 통합 |
| `python main.py --stats` | 지연을 구간별로 측정해 종료 시 p50/p95 출력 |
| `python main.py --stats out.csv` | 위 + 원본 표본을 CSV로 (보고서 그래프용) |
| `python main.py --display pi` | 라즈베리파이 GPIO 직결 OLED |
| `python main.py --no-gate` | 반이중 해제 (유선 이어폰으로 들을 때만) |
| `python tools/demo.py --record demo.mp4` | 데모 시나리오 재생·녹화 |
| `python tools/check_mount.py --duration 30 --preview` | 장착 시점 손/얼굴 검출률 측정 |
| `python -m src.calibrate --display serial` | 실제 OLED 시차 보정 (Pi는 `--display pi`) |
| `python -m src.hud` | 말풍선 레이아웃 미리보기 |
| `python tests/run_all.py` | 회귀 테스트 |

`DISPLAY_BACKEND = "preview"`면 **하드웨어 없이** HUD 레이아웃·좌표 매핑·가독성이
전부 검증됩니다. Phase 1·2를 0원으로 끝낼 수 있는 지점입니다.

---

## 학습 데이터 두 경로

| | 경로 A — 직접 수집 | 경로 B — AI Hub |
|---|---|---|
| 특징 | `landmarks.py` **246차원 wearable-v3** | `hand_shape3d.py` **462차원 3D 손 모양** |
| 얼굴(비수지신호) | 검출 시 포함 | 원본에 존재하나 현재 손 모양 분기에는 미포함 |
| 수집 | `src/collect.py` | AI Hub 다운로드 |
| 학습 | `src/train.py` 있음, ONNX 변환 테스트 통과 | 학습 입력 변환 완료, 사전학습 모델은 미구현 |
| 상태 | 실제 장착 데이터 수집 필요 | 단어당 1동작 그룹: 추가 표본·뜻 매핑 필요 |

직접 수집 경로는 `data/raw/dataset_meta.json`에 특징 버전·차원·길이·FPS를 기록합니다.
파일명은 `{촬영자}__{세션}_{번호}.npy`이며, 메타데이터 없는 구형 배열이나 다른 특징 버전이
섞이면 학습 전에 중단합니다.

보유 AI Hub에는 2D뿐 아니라 손 3D와 카메라 보정값도 있습니다. 새 경로는 정규화한 3D
관절 간 거리로 손 모양을 표현해 강체 회전에 불변하게 만듭니다. 기존 `features_v2.py`의
2D 정렬만으로는 1인칭/3인칭 차이를 제거할 수 없습니다. 손 모양 사전학습에 AI Hub를 쓰고
착용 데이터로 방향·이동·검출 오차를 보완하는 경로를 검증할 수 있습니다.

```bash
python tools/merge_parts.py <다운로드폴더> --run --extract   # 분할압축 병합
python tools/inspect_aihub.py <데이터루트>                   # 구조·형식 스캔
python tools/prepare_aihub3d.py data/raw/New_sample --out data/processed/aihub3d-new
```

---

## 알려진 제약

- **MediaPipe 버전 고정 필수.** 0.10.31+에는 `solutions.holistic`이 삭제됐고
  Tasks API에 대체품이 없습니다. numpy도 `<2`여야 합니다.
- **macOS는 OpenCV 노출 제어가 안 먹습니다.** 수어 인식이 깨지는 최대 원인은
  해상도가 아니라 모션 블러라, 조명을 밝게 하거나 수집·시연을 Windows에서 통일하세요.
- **두 카메라는 선택 기능입니다.** 얼굴 AR 성공률은 높이지만 USB 대역폭·CPU·발열이 늘어납니다.
  실측이 기준을 못 넘으면 기능을 삭제하지 말고 `--fixed-caption`을 라이브 시연 기본값으로 둡니다.
- **wearable-v3 이전 데이터/모델과 섞지 마세요.** 특징 차원은 246으로 같아도 손 정규화 의미가
  다릅니다. 새 장착 시점으로 수집하고, 모델의 `feature_version` 검사를 통과시켜야 합니다.
- **시차 보정은 광학 결합기를 쓸 때만 의미가 있습니다.** 직시형이면 `calibrate.py`가
  사실상 놀게 됩니다.
- **`--display serial`을 아두이노 Uno/Nano로 쓸 때**, 펌웨어는 프레임버퍼에 직접
  수신하도록 되어 있어야 합니다(별도 1KB 버퍼를 두면 ATmega328P의 SRAM 2KB를 넘깁니다).
  현재 `firmware/oled_bridge.ino`는 그렇게 되어 있습니다.
- **광학용으로는 2.42인치 SSD1309를 권합니다.** 0.96인치와 같은 128×64·같은 드라이버라
  **코드 변경이 전혀 없고**, 픽셀이 커서 광학 정렬이 훨씬 쉽고 밝습니다.
