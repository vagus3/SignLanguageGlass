# MediaPipe 기준 인식 개선 및 재학습

기존 `models/words_v1`은 OpenPose 좌표로 학습한 임시 모델이다. 아래 변경은 실행 경로와 새 학습 입력을 개선한다.
기존 가중치의 실사용 정확도가 개선됐다고 검증한 결과는 아니다. 새 특징은 재학습한 모델에 적용한다.

## 적용한 변경

- 수어 인식이 켜져 있으면 디버그 관절은 **인식에 쓰는 Holistic 결과**를 그린다. 별도 Hands 그래프를 중복 실행하지 않는다.
- 자동 구간 분할은 검출 결측을 보존하는 적응형 필터를 거쳐 움직임을 측정한다. 빠른 왕복 동작의 짧은 방향 전환은 시작 후보를 즉시 취소하지 않는다.
- 새 특징 이름은 `mediapipe_body_v1`이다. 학습과 PyTorch/ONNX 추론 모두 같은 함수를 사용한다.
  30fps 리샘플링 → 흔들림 필터 → 손목 기준 손 모양 및 어깨 중심·너비 기준 위치/속도 순서다.
- 관절 결측과 0.2초를 넘는 촬영 공백을 보간해 채우지 않는다. 정지한 손에서 한 관절만 3 palm 이상 튀는 짧은 관측은 결측 처리한다.
- 새 학습 경로는 `extractor.name=mediapipe-holistic`인 데이터만 받는다. 양성 클립에 손 검출이 하나도 없으면 학습을 중단한다.
- 단어 학습은 `--task words`로 실제 단어 클립을 사용한다. 문장으로 이어 붙이지 않으며 결과 메타데이터에도 단어 모델로 기록한다.
- 기존 `2d`, `3d`, 246차원 로컬 모델의 특징 계약은 유지한다. 기존 가중치의 이름이나 JSON만 바꿔 새 특징을 넣으면 안 된다.

## 1. 학습 자료 준비

좌표 JSON만으로 MediaPipe 검출 결과를 복구할 수는 없다. **원본 영상에서 다시 추출하거나 실제 카메라로 수집**한다.
재추출과 실행은 같은 MediaPipe 버전 및 `config.MODEL_COMPLEXITY`로 맞춘다.

영상과 정답이 있다면 `annotations.json`을 작성한다. 영상 경로는 이 파일의 위치를 기준으로 해석한다.

```json
{
  "clips": [
    {
      "id": "hospital-person1-take1",
      "video": "videos/hospital-person1-take1.mp4",
      "signer_id": "person1",
      "session_id": "day1",
      "group_id": "hospital-person1-take1",
      "continuous": false,
      "glosses": ["병원"]
    }
  ]
}
```

위 한 행은 형식 예시다. 학습에는 모든 목표 단어와 수어가 없는 무동작/일상동작 자료(negative)를 포함한다.
수어가 없는 자료의 `glosses`는 `[]`이고 `negative: true`를 지정한다. 무동작이라고 단어 정답을 임의 부여하지 않는다.
화자와 동일 촬영 동작 그룹은 train/val/test 사이에 중복될 수 없다. 자동 분할에는 독립 화자/동작 그룹 묶음이 최소 3개 필요하다.
한 사람의 여러 take에 다른 화자 ID를 붙여 조건을 통과시키면 안 된다.

```powershell
python tools/extract_mediapipe.py annotations.json --out data/processed/mediapipe-words --workers 2
```

직접 카메라로 수집하는 경우:

```powershell
python -m src.collect_continuous --out data/raw/mediapipe-words --signer person1 --session day1 --glosses 병원
python -m src.collect_continuous --out data/raw/mediapipe-words --signer person1 --session day1 --glosses
```

SPACE로 시작/종료, c로 취소한다. 첫 명령은 단어 클립, 둘째는 무동작·일상동작 자료다.
수집기는 MediaPipe 추출기 정보를 저장하고, 다른 추출기·버전·모델 복잡도의 자료와 섞는 것을 막는다.

## 2. 단어 모델 재학습

학습 환경에는 `requirements-train.txt`의 PyTorch 등이 필요하다. 현재 실행용 `.venv`에는 PyTorch가 없다.
처음에는 수형이 분명한 5~10개 단어로 파이프라인을 검증하고, 성공한 뒤 어휘를 늘린다.

```powershell
python -m src.train_continuous data/processed/mediapipe-words/manifest.json --out models/mediapipe-words-v1 --task words --features mediapipe_body_v1 --epochs 40
python tools/evaluate_continuous.py data/processed/mediapipe-words/manifest.json --checkpoint models/mediapipe-words-v1/ctc.pt --split test --out reports/mediapipe-words-test.json
python tools/export_ctc_onnx.py models/mediapipe-words-v1/ctc.pt --out models/mediapipe-words-v1/ctc.onnx
```

직접 수집한 자료를 쓰면 manifest 경로를 `data/raw/mediapipe-words/manifest.json`으로 바꾼다.
실제 연속 문장 자료는 `--task continuous`를 사용한다. 학습에는 정답 글로스의 시각순 배열이 필요하다.
테스트셋을 반복해서 보고 모델을 고르지 말고, 조정은 validation에서 수행한다.

## 3. 실시간 확인

기존 cloud/device 경로에서 `--onnx models/mediapipe-words-v1/ctc.onnx`를 지정하면 새 특징이 자동 선택된다.
서버의 `--device-model`도 같은 새 모델의 JSON으로 맞춰야 한다.
첫 확인은 `--auto-segment`를 빼고 SPACE로 구간을 지정한다. 단어 인식이 안정된 뒤 자동 구간을 켠다.
`--debug --stats reports/mediapipe-live.csv --no-stt --no-tts --fixed-caption`으로 영상 처리와 수어에 집중해 측정할 수 있다.
현재 서버/용어집/번역기 설정은 계속 필요하다. 인식과 문장 생성·음성 출력은 각각 결과를 확인한다.

## 검증 범위

- 합성 재현에서 ±3.2px 반복 관절 흔들림이 자동 녹화를 시작시키지 않는 것을 확인했다.
- 이동·일정한 크기 변화 후 특징 일치, 같은 직선 궤적의 10/30fps 입력 일치, 결측·긴 공백·이상치 처리를 검사했다.
- 새 특징의 Torch 없는 ONNX 실행 경로 선택, 기존 특징 계약, 인식용 관절 표시, 자동/수동 구간 통합을 검사했다.
- 실제 카메라 정확도와 전체 FPS 개선은 원본 영상·실제 촬영 자료로 별도 검증해야 한다.
