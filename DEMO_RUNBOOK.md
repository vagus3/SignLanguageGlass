# signglass 경진대회 시연 런북

목표: 2인 팀이 60~90초 양방향 대화를 안정적으로 보여주고, 문제가 생겨도 핵심 기능을
잃지 않은 채 즉시 폴백한다.

현재 시연 장치는 **노트북**이고 서버는 클라우드에 둔다. Pi·Arduino/Pico와 실제 OLED
배선을 시연의 필수 조건으로 두지 않는다. `preview`는 보존된 OLED 별도 미리보기 창이며,
영상 말풍선과 기록 창은 함께 사용할 수 있다. 아래 cloud 명령은 현재 CTC 기준선용으로,
도입 예정인 Qwen2.5-VL의 연결·품질 검증을 대신하지 않는다.

## 시연 모드 결정

새 기능을 삭제하지 않고, 실측 결과로 당일 기본 모드만 정한다.

| 우선순위 | 조건 | 실행 명령 |
|---|---|---|
| A. 노트북 기본 | 손·얼굴 검출률과 실제 번역 품질, 15분 소크 검증 | `python main.py --sign-backend cloud --auto-segment --viewpoint webcam --display preview --video-captions --transcript --stats reports/live.csv` |
| B. 두 카메라 비교 | 한 카메라로 손·얼굴을 함께 담기 어려울 때만 | `python main.py --sign-backend cloud --sign-camera 0 --face-camera 1 --display preview --video-captions --transcript --stats reports/live.csv` |
| C. 안정형 | 얼굴 추적 또는 말풍선 위치 불안정 | `python main.py --sign-backend cloud --fixed-caption --display preview --stats reports/live.csv` |
| D. 녹화 폴백 | 라이브 장치 복구가 30초를 넘음 | 같은 장치로 미리 찍은 무편집 60~90초 영상 재생 |

모드 A/B의 얼굴 앵커가 흔들려도 음성 자막 자체는 유지된다. 카메라가 전부 실패하면 모드 C는
`--no-sign --fixed-caption`으로 음성→자막만 계속 시연할 수 있다.

## 노트북 웹캠 시연 (연속 문장 번역)

노트북 내장 웹캠이 수어자를 정면으로 봅니다. 안경 1인칭과 달리 어깨가 보이므로
`--viewpoint webcam`(기본값)은 **손을 기준선 위로 올리면 문장 시작, 내리면 문장 끝**으로
구간을 나눕니다. 키보드 위에서 움직이는 손은 무시합니다.

~~~bash
# 터미널 1: 같은 노트북에서 서버 (모의 또는 학습한 체크포인트)
.venv/bin/python -m src.sign_server --mock
# .venv/bin/python -m src.sign_server --checkpoint models/continuous/ctc.pt --mapping mapping.json

# 터미널 2: 웹캠 + 자동 구간
.venv/bin/python main.py --sign-backend cloud --auto-segment --viewpoint webcam \
  --fixed-caption --display preview --stats reports/webcam.csv
~~~

- **얼굴 앵커 끄기**: 웹캠 한 대는 수어자 본인 얼굴을 보므로 `--fixed-caption`을 씁니다.
  상대 얼굴 말풍선은 안경/전방 카메라 구성에서만 의미가 있습니다.
- **화면 구도**: 머리부터 가슴 아래까지, 양 어깨가 모두 보이게 앉습니다. 수어하는 손이
  프레임 안에 있어야 합니다. 화면에 다른 사람이 들어오지 않게 합니다.
- **기준선 맞추기**: 프리뷰의 파란 선(`hands down = end`)이 가슴 아래와 책상 사이에
  오도록 앉는 위치를 조정합니다. 키보드·무릎 위 손이 선 아래에 있어야 합니다. 맞지 않으면
  `config.SEGMENT_REST_RATIO`(기본 1.0)를 0.8~1.3에서 바꿉니다. 당일 즉흥 변경은 금지이므로
  리허설 때 정합니다.
- **문장 사이**: 손을 선 아래로 1초 정도 내립니다. 우측 상단이 `AUTO REC` → `AUTO READY`로
  바뀌면 전송된 것입니다. 오검출되면 `c`로 취소하고, 자동이 불안하면 SPACE 수동으로 바꿉니다.
- **조명**: macOS는 OpenCV 노출 고정이 안 됩니다. 정면을 밝게 해 빠른 손의 모션 블러를 줄입니다.
- **데이터 시점 일치**: 시연 모델의 검증 데이터는 같은 노트북 웹캠으로
  `python -m src.collect_continuous --signer ... --session ...`(필요하면 `--with-3d`)로 모읍니다.
  정면 시점은 AI Hub 정면 촬영과 가깝지만, 같은 카메라 데이터로 확인하기 전에는
  정확도를 주장하지 않습니다.
- **정직한 표시**: 모의 서버와 미검증 모델은 텍스트만 표시하고 음성을 내지 않습니다.

리허설 기준: 시연 문장 각 10회에서 구간 누락·잘못 나뉨·키보드 오검출 횟수와 번역 결과를
기록합니다. 구간 오류가 2회 이상이면 당일 모드는 SPACE 수동으로 정합니다.

## 행사 48시간 전 동결 기준

다음 항목이 모두 충족되면 새 기능 추가를 멈춘다.

- `python tests/run_all.py`: 실패 0
- `python check_env.py`: 필수 산출물과 실제 장치 실패 0
- `python tools/check_mount.py --duration 30 --preview --json reports/mount.json`
- 미사용 화자 기준 macro F1, coverage, NEUTRAL 오출력 기록
- 수어→소리와 음성→자막 지연 p50/p95 기록
- 실제 배터리 전원으로 15분 연속 실행 3회, 크래시 0
- 전원 재부팅 후 3분 안에 팀원 둘 다 실행 가능

## 90초 데모 구성

| 시간 | 장면 | 보여줄 가치 |
|---|---|---|
| 0~10초 | “병원 접수에서 서로의 언어를 모르면?” 문제 제시 | 사용자 문제 |
| 10~35초 | 착용자가 `안녕하세요`, `도와주세요`, `병원` 수어 → 음성 | 수어→음성 |
| 35~60초 | 상대가 한국어로 안내 → HUD 자막 | 음성→자막 |
| 60~75초 | 얼굴 이동에 말풍선이 따라감 또는 고정 자막의 가독성 | 착용형 인터페이스 |
| 75~90초 | 미사용 화자 F1·오발화·지연 표 한 장 | 정직한 검증 |

자유 대화를 즉흥적으로 길게 하지 않는다. 고정 문장을 외워서 연기하는 것이 아니라, 사전에
정의한 사용 시나리오와 평가 범위를 재현하는 것이다.

## 2인 현장 역할

- 팀원 A: 안경 착용·수어 수행·모델 상태 확인. 예비 카메라와 ONNX/라벨 백업 담당.
- 팀원 B: 상대방 역할·실행 명령·오디오/OLED/전원 확인. 30초 복구 타이머와 영상 폴백 담당.
- 발표 전 역할을 한 번 바꿔 실행해 둘 다 전체 복구 절차를 익힌다.

## 가져갈 것

- 장치 본체, 예비 USB 카메라, OLED/브리지 MCU, 데이터 케이블 2개, 보조배터리/충전기
- 동일 환경이 설치된 예비 노트북 또는 복구 가능한 가상환경/의존성 파일
- `models/sign_lstm.onnx`, `labels.json`, `sign_lstm_preproc.json`, `sherpa-ko-174m-chunk32`·`vosk-ko` STT 모델, TTS 캐시, 폰트 백업
- 무편집 폴백 영상과 `reports/` 평가 결과를 로컬에 저장; 행사장 인터넷에 의존하지 않음

## 금지 사항

- 행사 당일 모델 재학습, 임계값 즉흥 변경, 운영체제/패키지 업데이트
- 학습에 포함된 사람의 정확도를 “사용자 독립 성능”으로 발표
- `tools/demo.py` 시뮬레이션 영상을 인식 성능 증거로 사용
- 실패를 숨기기 위해 범용 연속 수어 번역이라고 과장
