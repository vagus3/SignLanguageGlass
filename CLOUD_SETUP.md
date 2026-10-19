# 현재 구현: 좌표 기반 연속 수어 번역

이 문서는 **현재 CTC + 텍스트 언어 모델 기준선**의 실행 절차다. 현재 클라이언트는
노트북이며 Pi·Arduino/Pico를 필수로 사용하지 않는다. 개발 목표인 Qwen2.5-VL-7B/72B는
아직 연결되지 않았으며, VLM 입력·서빙 경로를 별도로 구현해야 한다.
전체 방향과 OLED 별도 미리보기의 역할은 [README](README.md#현재-개발-방향)를 따른다.

장치는 MediaPipe 좌표를 추출하고, 서버는 CTC로 단어열을 인식한 뒤 vLLM으로
한국어 문장을 생성합니다. vLLM은 언어 모델 서빙 엔진이며 좌표 인식 모델은
별도의 PyTorch 모델입니다. 첫 버전은 버튼으로 구분하는 최대 15초 문장을 처리합니다.

실제 문장 데이터·학습된 체크포인트·GPU vLLM 서버는 별도로 준비해야 합니다.
보유 WORD 샘플과 모의 서버는 실제 연속 번역 성능의 증거가 아닙니다.

## 1. 로컬 환경과 모의 API

~~~bash
source .venv/bin/activate
python -m pip install -r requirements-server.txt
python -m src.sign_server --mock
~~~

다른 터미널에서 장치 경로를 실행합니다.

~~~bash
source .venv/bin/activate
python main.py --sign-backend cloud --no-stt --no-tts --fixed-caption
~~~

SPACE=녹화 시작/종료, c=취소, q=종료입니다. 모의 결과는 HUD와 콘솔에만
표시합니다. 헤드리스 장치는 아래 명령으로 start/stop/cancel/quit를 표준입력에
보낼 수 있습니다. 물리 버튼은 같은 동작을 호출하도록 연결하면 됩니다.

~~~bash
python main.py --sign-backend cloud --cloud-controls stdin --display null --fixed-caption
python tools/send_pose.py data/processed/continuous/clip_000000.npz
~~~

두 번째 명령은 카메라 없이 저장된 구간을 전송합니다.

## 2. 데이터 준비와 학습

공통 규격은 pose2d-sequence-v1입니다. 왼손 21점, 오른손 21점,
목·오른/왼 어깨·팔꿈치·손목 7점의 순서로 총 49점을 저장합니다.
좌표는 원본 비반전 영상의 x/width, y/height이며 이미지 크기도 저장합니다.
목은 양 어깨 중점입니다. 검출되지 않거나 영상 밖인 관절은 mask=false로 둡니다.
상대방 얼굴은 포함하지 않습니다.

서버 특징은 회전을 제거하지 않은 손 로컬 좌표, 이미지 공간 좌표, 유효 마스크,
인접 프레임의 속도, 시간 간격·경과 시간입니다. 이미지 비율을 복원해 거리를 계산하고,
0.25초 이상 공백에서는 속도를 만들지 않습니다. 원본 3D 데이터는 별도로 보존합니다.

AI Hub/OpenPose JSON에 다음 주석 목록을 준비합니다. 이미지 크기와 FPS는
실제 메타데이터에서 확인하세요. 아래 숫자는 형식 예시입니다.

~~~json
{
  "image_size": [1920, 1080],
  "fps": 30,
  "clips": [{
    "id": "sentence001-front",
    "keypoints_dir": "sentence001/keypoints",
    "signer_id": "person01",
    "session_id": "day01",
    "group_id": "sentence001-original-motion",
    "continuous": true,
    "glosses": ["NAME", "WHAT"],
    "text": "이름이 뭐예요?",
    "split": "train"
  }]
}
~~~

- glosses는 실제 수어 순서의 단어열입니다. 한국어 문장을 단어로 나누어 대신
  넣지 않습니다. glosses가 없으면 CTC 학습에서 제외됩니다.
- 같은 동작의 다른 시점은 같은 group_id로 묶습니다. 촬영자와 원본 동작 그룹은
  train/val/test 경계를 넘을 수 없습니다.
- split은 모든 행에 명시하거나 모두 생략합니다. 생략하면 촬영자·동작 그룹의
  연결 성분을 기준으로 나눕니다. 최소 3개 독립 성분이 필요합니다.
- 평가 어휘는 학습에도 존재해야 합니다. 일상 동작은 glosses=[], negative=true입니다.
  NEUTRAL을 출력 단어로 넣지 않습니다.
- 연속 문장은 continuous=true로 표시합니다. 단어 샘플을 이어 붙인 결과를
  실제 연속 수어 성능으로 보고하지 않습니다.
- 15초를 넘는 문장은 정답에 맞는 경계를 주석으로 나누어 준비합니다.
  이미지 크기가 시점마다 다르면 행별 image_size를 지정합니다.

~~~bash
python tools/prepare_continuous.py annotations.json --out data/processed/continuous
python -m src.train_continuous data/processed/continuous/manifest.json \
  --out models/continuous --epochs 30 --batch 8
python tools/evaluate_continuous.py data/processed/continuous/manifest.json \
  --checkpoint models/continuous/ctc.pt --out reports/continuous-test.json
~~~

출력 경로가 이미 있으면 덮어쓰지 않습니다. 학습은 기본 CPU이고 CUDA를 사용하려면
해당 GPU에 맞는 PyTorch 환경과 --device cuda를 준비합니다.

착용 좌표를 직접 수집할 수도 있습니다.

~~~bash
python -m src.collect_continuous --signer person01 --session day01 \
  --glosses NAME WHAT --text "이름이 뭐예요?"
python -m src.collect_continuous --signer person01 --session day01 --glosses
~~~

두 번째 명령은 일상 동작 표본입니다. 기본 저장 위치는 data/raw/continuous/입니다.
SPACE로 녹화하고, 미사용 촬영자·날짜·배경·조명을 나누어 데이터를 준비합니다.
안경 시점의 표정 미관측·가림·AI Hub와 실시간 좌표 추정 차이는 별도 평가해야 합니다.

### 3D 방향·깊이 분기 (pose3d-sequence-v2)

2D 규격에 3D world 좌표(49점, x 오른쪽·y 아래·z 카메라에서 멀어지는 방향)를
더한 선택형 규격입니다. 2D 특징 331차원 뒤에 손별 손바닥 법선, 손가락 방향,
어깨 중심 기준 손목 위치·속도(어깨폭 정규화), 유효 플래그 28차원을 붙입니다.
방향은 손 관절끼리만, 위치·깊이는 몸 관절끼리만 계산하므로 손과 몸의 3D 원점이
달라도 됩니다.

~~~bash
python tools/prepare_continuous.py annotations.json --out data/processed/continuous3d --with-3d
python -m src.train_continuous data/processed/continuous3d/manifest.json \
  --out models/continuous3d --features 3d
python -m src.collect_continuous --with-3d --signer person01 --session day01 --glosses NAME WHAT
python main.py --sign-backend cloud --pose-3d   # 서버가 3D 체크포인트일 때
~~~

- AI Hub의 `*_keypoints_3d`를 사용하되 카메라 좌표라고 가정하지 않습니다.
  기존 샘플 감사에서는 서로 다른 시점이 공통 3D 좌표를 공유했습니다.
  `--with-3d` 변환 시 annotations.json의 최상위 또는 각 클립에
  `coordinate_frame_3d`를 반드시 지정합니다.
  - `"camera"`: 이미 x 오른쪽·y 아래·z 카메라에서 멀어지는 좌표임을 확인한 입력.
  - `"world"`: `world_to_camera`도 지정해야 합니다. 유한한 3×4 `[R|t]`이며
    `p_camera = R @ p_world + t`입니다. R은 직교행렬, 행렬식 +1이어야 하고
    t는 원본 좌표와 같은 단위입니다. 클립별 값이 최상위 값보다 우선합니다.
  - 변환은 미러 보정 전에 적용하고, 변환 후에도 결측 관절은 0으로 유지합니다.
    원본 좌표계와 적용한 행렬은 출력 manifest의 `source_coordinate_frame_3d`,
    `source_world_to_camera`에 기록됩니다.
  `camparam.CameraMatrix`의 이름만 보고 외부 파라미터로 사용하지 마세요.
  실제 보정 형식·방향을 확인해서 행렬을 지정해야 합니다. 좌표계가 불명확하거나
  3D 값이 없는 입력은 거부합니다. 이전에 좌표계를 확인하지 않고 만든 3D 데이터셋은
  다시 변환·학습·검증해야 합니다.
- 실시간 MediaPipe Holistic은 손 3D가 영상 기준 근사값(z는 대략 x 축척)이고
  몸은 `pose_world_landmarks`(엉덩이 원점)입니다. 손바닥 방향은 근사값이므로
  AI Hub 학습 모델을 착용 데이터로 검증하기 전에는 성능 향상을 가정하지 않습니다.
- 3D 데이터셋으로 `--features 2d` 모델도 학습할 수 있어 같은 데이터로 비교할 수 있습니다.
  3D 체크포인트는 2D 요청을 거부합니다.

### 특징 구성 비교 실험

특징을 블록(hand_shape, position, mask, velocity, timing, palm_orientation, wrist_3d)으로
나누고, 블록 조합에 이름을 붙인 구성을 `--features`로 고릅니다:
`2d`, `3d`, `2d_no_hand_shape`, `2d_no_position`, `2d_no_velocity`, `3d_orientation`, `3d_wrist`.

~~~bash
python tools/compare_features.py data/processed/continuous3d/manifest.json \
  --out reports/features-seed0 --features 2d 2d_no_velocity 3d_orientation 3d --seed 0
~~~

모든 구성은 같은 클립·분할·어휘·seed·학습 설정을 사용합니다. 결과는
`comparison.json`/`comparison.md`(validation WER, 출력률, 일상 동작 오출력)와 구성별
`<구성>/ctc.pt`입니다. test는 평가하지 않습니다. validation으로 구성을 고른 뒤
`evaluate_continuous.py`로 test를 한 번만 평가합니다. 데이터가 적으면 seed를 바꿔
반복하고, 차이가 일관된 경우에만 결론을 냅니다.

### 장치 내 인식 비교용 ONNX (REVIEW_AND_PLAN P1-1)

~~~bash
python tools/export_ctc_onnx.py models/continuous/ctc.pt --out models/continuous/ctc.onnx
~~~

패딩 없는 한 구간(1, 프레임, 특징)을 길이 제한 없이 받는 ONNX와 사이드카 JSON(라벨·특징
구성·좌표 규격·보정·메타데이터)을 만듭니다. 2·37·150·900프레임에서 PyTorch와
log-확률(허용 1e-4)과 탐욕 디코딩이 같을 때만 저장합니다. 장치에서는 torch 없이
`src/ctc_runtime.OnnxRecognizer`로 같은 판단을 합니다. 배치 결정(서버 인식 A / 장치 인식 B)은
목표 장치에서 카메라·STT·TTS를 함께 돌린 부하·발열·전체 지연을 재고 정합니다.

### 장치 인식 경로 B와 배치 비교

~~~bash
# 서버: 같은 내보내기의 체크포인트(A)와 사이드카(B)를 함께 지정
python -m src.sign_server --host 0.0.0.0 --checkpoint models/continuous/ctc.pt \
  --device-model models/continuous/ctc.json --mapping glossary.json
# 장치(B): 이 기기에서 ONNX로 인식하고 글로스만 보냄 (기본은 A: --recognize server)
python main.py --sign-backend cloud --auto-segment --recognize device \
  --onnx models/continuous/ctc.onnx
# 배치 비교: 시연 노트북에서 실제 서버를 대상으로
python tools/benchmark_paths.py data/processed/aihub_sample_extractors/mediapipe/manifest.json \
  --onnx models/continuous/ctc.onnx --url "$SIGN_SERVER_URL" --out reports/ab.json
~~~

`POST /v1/sign/compose` 입력은 version(`gloss-sequence-v1`), session_id, segment_id,
glosses(최대 64개), score(0~1), recognizer(사이드카의 model_id)입니다. 서버는 model_id가
`--device-model`과 다르면 409, 그 모델 라벨에 없는 글로스면 422로 거절합니다. 그 뒤 임계값·
용어집·문장 생성·발화 게이트를 A와 같은 코드로 적용하고, 캐시·중복 판정·동시 요청 제한도
같습니다. 응답 형식은 `/v1/sign/translate`와 같습니다.

`--mock` 서버는 B 요청 중 model_id가 `mock`인 것만 받습니다. 실제 ONNX로 B를 시험하려면
같은 내보내기로 `--device-model ... --mock-language` 서버를 띄웁니다. 실패하면 장치 터미널에
`[cloud] 요청 실패: HTTP 409`처럼 이유가 나옵니다. 401은 토큰, 409는 다른 모델, 422는 형식,
429는 처리 중입니다.

`--mock-language`는 실제 인식기에 모의 문장 생성기를 붙입니다. 결과는 mock으로 표시되어
발화되지 않으며, 학습 전에도 무작위 초기화 모델로 배치 지연을 잴 수 있습니다(가중치는
지연과 무관). 측정 도구는 같은 클립·서버·모델로 두 경로의 준비 시간, 전송 바이트, 왕복,
서버 시간, CPU 시간, 최고 메모리와 두 경로의 글로스 일치 수를 기록합니다.

## 3. 실제 vLLM과 서버

GPU Linux 환경에 별도 가상환경을 만들고 설치합니다. vLLM을 장치 또는 CTC의
torch==2.4.1 환경과 합치지 않습니다. 아래 명령은 모델 다운로드와 실제 실행을
수행하며 GPU 메모리·처리량은 해당 장치에서 확인해야 합니다.

~~~bash
python -m pip install -r requirements-vllm.txt
vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --api-key YOUR_LOCAL_VLLM_KEY
~~~

Apple Silicon 로컬 검사는 API·좌표 처리 검증이며 GPU 모델 실행 완료로 간주하지 않습니다.
각 수어 ID에 한국어 뜻을 연결한 mapping.json을 준비합니다.

~~~json
{"NAME": "이름", "WHAT": "무엇"}
~~~

검수 문장과 의미 검사를 쓰려면 같은 `--mapping` 자리에 glossary.json을 줍니다.

~~~json
{
  "glosses": {
    "NAME": {"ko": "이름"},
    "WHAT": {"ko": "무엇"},
    "HOSPITAL": {"ko": "병원", "kind": "place", "aihub": ["WORD0000"]},
    "NOT": {"ko": "아니다", "kind": "negation"}
  },
  "sentences": [
    {"glosses": ["NAME", "WHAT"], "text": "이름이 뭐예요?",
     "reviewed_by": "검수자", "reviewed_on": "2026-10-12"}
  ]
}
~~~

문장은 `src/translation.py`의 순서로 만듭니다.

1. 뜻이 없는 글로스가 있거나 인식 점수가 임계값보다 낮으면 `uncertain`(재입력 안내).
2. 글로스열이 검수 문장과 정확히 같으면 그 문장을 쓰고 언어 모델을 부르지 않습니다.
3. 그 밖에는 언어 모델 문장에 보조 검사를 적용합니다. 입력 숫자가 그대로인지,
   `negation` 글로스가 있으면 부정 표현이 남았는지, `person`·`place`·`number` 글로스의
   뜻이 문장에 들어 있는지 봅니다. 활용·동의어·부정 범위는 판단하지 못하므로
   사람 의미 검수를 대신하지 않습니다.
4. 생성이 실패하거나 검사에 걸리면 `fallback`입니다. 검수 문장들로 글로스 전체가
   차례대로 덮이면 그 문장들(`source: reviewed_rules`)을, 아니면 단어 나열
   (`source: words`)을 미리보기로만 표시하며 자동 발화하지 않습니다.

검수 문장도 인식 보정 게이트를 면제받지 않습니다. 보정의 `mapping_hash`는 glossary의
뜻·종류·검수 문장에서 계산하므로(`Glossary.policy_hash`), 문장을 고치면 보정을 다시 합니다.
검수 문장과 종류가 없는 mapping.json은 예전과 같은 해시라 기존 보정이 그대로 유효합니다.

~~~bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=YOUR_LOCAL_VLLM_KEY
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
python -m src.sign_server --checkpoint models/continuous/ctc.pt --mapping mapping.json
~~~

언어 모델에는 인식된 단어열만 보냅니다. 구조화 JSON·사용한 입력 인덱스·숫자 변경
검사를 적용하지만 의미 보존은 보장되지 않으므로 실제 출력 검수가 필요합니다.

클라우드에서는 TLS 프록시 뒤에 API를 두고 토큰을 설정합니다. vLLM 키는 서버에만 둡니다.

~~~bash
export SIGN_SERVER_TOKEN=YOUR_SIGN_API_TOKEN
python -m src.sign_server --host 0.0.0.0 --checkpoint models/continuous/ctc.pt \
  --mapping mapping.json
# 장치 환경:
export SIGN_SERVER_URL=https://YOUR_SIGN_API_HOST
export SIGN_SERVER_TOKEN=YOUR_SIGN_API_TOKEN
~~~

장치 클라이언트는 원격 HTTP를 거부하며 localhost HTTP만 개발용으로 허용합니다.
토큰과 .env 파일을 Git에 추가하지 않습니다.

요청 본문은 간결 JSON입니다. 좌표는 서버가 읽는 float32로 정확히 되돌아가는 가장 짧은
소수로 씁니다(`0.41240000724792480` 대신 `0.4124`, 손실 없음). 그 위에 gzip(레벨 1,
`Content-Encoding: gzip`)을 적용합니다. AI Hub 샘플 100개(평균 139프레임, 2D)에서 본문은
평균 262KB → 163KB, gzip 후 106KB → 56KB입니다. 1KB 미만 본문(글로스 요청)은 압축하지
않습니다. 서버는 압축 전과 해제 후 각각 8MB로 제한하고, 손상된 gzip은 400, 지원하지 않는
인코딩은 415로 거절하며, 압축하지 않은 JSON도 계속 받습니다. gzip을 지원하지 않는 이전
서버에는 `SIGN_SERVER_GZIP=0`으로 일반 JSON을 보냅니다.

좌표 반올림(`SIGN_SERVER_ROUND="4"` 또는 2D·3D를 따로 `"4,3"`)은 손실이 있어 기본으로
꺼져 있습니다. 같은 샘플에서 소수 4자리는 gzip 30KB, 3자리는 18KB까지 줄지만 손이 작게
잡힌 프레임에서는 손바닥 길이로 정규화한 손 모양 특징이 최대 1.0(3자리는 10)까지
바뀝니다. 학습한 모델로 아래 도구를 돌려 글로스·임계값 판정이 바뀌지 않는 설정만 켭니다.

~~~bash
python tools/evaluate_rounding.py data/processed/continuous/manifest.json \
  --checkpoint models/continuous/ctc.pt --out reports/rounding.json
~~~

### 자동 문장 구간 (--auto-segment)

~~~bash
python main.py --sign-backend cloud --auto-segment
~~~

SPACE 없이 손 움직임으로 구간을 나눕니다(`src/segmenter.py`). 규칙은 카메라 시점
(`--viewpoint`, 기본 `config.SIGN_VIEWPOINT = "webcam"`)에 따라 다릅니다.

| | webcam (노트북 정면, 시연) | glasses (안경 1인칭) |
|---|---|---|
| 수어 손 | 어깨 높이 + 어깨폭 × `SEGMENT_REST_RATIO`(1.0) 선보다 위의 손 | 화면에 보이는 손 |
| 무시 | 선 아래 손 (키보드 타이핑, 무릎 위) | - |
| 문장 끝 | 손을 선 아래로 내림 | 손이 화면 밖으로 나감 |
| 속도 기준 | 어깨폭 × 0.25 (손바닥 길이 환산) | 손바닥 길이 |

- 움직임은 손목 이동과 손목 기준 관절의 상대 이동 중 큰 값입니다.
  상대 이동은 양 프레임에서 유효한 관절 중 가장 많이 움직인 4개의 평균으로,
  손가락 동작·손 회전을 반영하고 단일 관절의 추정 잡음을 줄입니다.
- 시작: 수어 손이 손바닥 길이 기준 1.5배/초 이상으로 0.3초 이상 움직임.
  시작 전 0.3초를 함께 녹화합니다.
- 종료: 수어 손이 0.8초 이상 없거나, 있어도 1.5초 이상 거의 정지.
  0.5초 미만 구간은 번역하지 않고 버립니다. 14초에 도달하면 나누어 보냅니다.
- webcam에서 어깨가 검출되지 않은 프레임은 glasses 규칙으로 판단합니다.
- 디버그 프리뷰에 기준선(`hands down = end`)과 `AUTO READY/REC`가 표시됩니다.
- SPACE로 시작한 구간은 자동으로 끝내지 않습니다. 자동 구간을 SPACE로 끝낼 수도 있습니다.

규칙 기반이며 임계값은 실제 녹화로 확인하지 않았습니다. 시연 자리에서 기준선 위치를
확인하고, 자동 발화 검증과 별개로 구간 검출 오류를 측정해야 합니다.

## 4. 자동 발화 검증

새 모델은 텍스트 미리보기만 허용합니다. 발화 활성화에는 검증 분할의 최소 600초
실제 일상 동작과 실제 LLM 출력의 의미 검수가 필요합니다. 일상 동작에는
capture_start(Unix 초), signer_id, session_id를 기록합니다. 겹치는 녹화 구간으로
검증 시간을 부풀리면 검사를 중단합니다. CTC 점수를 정답 확률로 해석하지 않습니다.

~~~bash
python tools/evaluate_continuous.py data/processed/continuous/manifest.json \
  --checkpoint models/continuous/ctc.pt --split val --mapping mapping.json \
  --compare-llm --out reports/continuous-val.json
~~~

실제 리포트의 clip ID·prediction·llm_text를 확인해 review.json을 작성합니다.
아래는 형식 예시이며 모의 결과로 발화를 활성화하면 안 됩니다.

~~~json
{
  "reviewer": "수어와 한국어 의미를 검수한 사람",
  "clips": {
    "실제-검증-clip-id": {
      "prediction": ["NAME", "WHAT"],
      "text": "이름이 뭐예요?",
      "meaning_preserved": true
    }
  }
}
~~~

~~~bash
python tools/evaluate_continuous.py data/processed/continuous/manifest.json \
  --checkpoint models/continuous/ctc.pt --split val --mapping mapping.json \
  --compare-llm --semantic-review review.json \
  --calibrate-out models/continuous/ctc-calibrated.pt \
  --out reports/continuous-calibration.json
~~~

10분당 오출력 2회 이하와 의미 검수를 통과하는 임계값 중 출력 비율이 가장 높은 값을
선택합니다. 통과하지 못하면 자동 발화는 비활성입니다. 통과한 calibrated 체크포인트를
서버에 지정하세요. 뜻 매핑이나 언어 모델을 바꾸면 검증 설정과 불일치해 발화가 꺼집니다.

test 분할에서 WER·오발화·포기율·의미 오류를 독립적으로 보고하세요.
실제 장치의 지연 p50/p95, FPS, 발열·전력과 15분 연속 동작도 확인해야 합니다.
TTS 지연의 끝점은 재생 호출 직전이므로 실제 음향 출력 시각은 별도 녹음으로 확인합니다.

## API와 검사

GET /health는 규격, 모의 여부, 인식 모델·장치 모델(B) 준비 여부, 언어 모델 구성 여부,
경로별 발화 활성 여부와 `data_sources`(학습 데이터 출처 표기 문구)를 반환합니다. 실제 vLLM
가용성은 번역 요청으로 확인합니다. `POST /v1/sign/compose`는 위 "장치 인식 경로 B" 절을 봅니다.
검증 오류(422)는 위치·종류·메시지만 최대 10개 돌려주며 입력값을 되돌려 보내지 않습니다.

POST /v1/sign/translate 입력은 version, session_id, segment_id,
timestamps(T), coordinates(T,49,2), mask(T,49 boolean), image_size([width,height])입니다.
응답은 같은 ID, status(ok/fallback/no_sign/uncertain/unavailable), glosses, text, source,
issues, auto_speech, mock, model_version, timings_ms를 포함합니다. no_sign·uncertain·unavailable의
text는 비어 있고, fallback의 text는 미리보기 전용입니다.
발화는 ok + auto_speech=true + mock=false에만 허용합니다. 본문은 받은 바이트와 gzip 해제 후
각각 최대 8MiB, 프레임은 900개입니다. 형식이 틀린 JSON·NaN·정수 마스크는 422입니다.

장치는 미완료 요청 한 개를 유지하고, 총 30초 내에서 연결 오류를 같은 ID로 한 번 재시도합니다.
서버는 성공 결과를 5분간 캐시하며 같은 ID의 다른 입력은 409로 거부합니다. 같은 입력인지는
검증한 배열 내용으로 판단합니다. 공백·키 순서·gzip 여부·`0`과 `0.0` 표기, 마스크가 꺼진
점의 값은 판정에 영향을 주지 않습니다.
취소한 요청은 완료까지 자리를 유지해 다음 요청이 쌓이지 않게 합니다.
캐시는 단일 프로세스 메모리이므로 API는 uvicorn worker 한 개로 운영합니다.
다중 프로세스 확장에는 공유 중복 처리 저장소가 필요합니다.

~~~bash
python tests/run_all.py continuous cloud_pipeline pose_sequence -v
python tests/run_all.py
~~~

검사는 ASGI API·HTTP 직렬화·vLLM 요청 형식, 학습·저장·재로딩, 데이터 누수와
취소·만료·재시도·중복 처리를 확인합니다. 실제 GPU·카메라·마이크·OLED 검증과 구분합니다.
