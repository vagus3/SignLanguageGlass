# 좌표 기반 연속 수어 번역

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
python main.py --sign-backend cloud --cloud-controls stdin --display pi --fixed-caption
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

GET /health는 규격, 모의 여부, 인식 모델 준비 여부, 언어 모델 구성 여부,
발화 활성 여부를 반환합니다. 실제 vLLM 가용성은 번역 요청으로 확인합니다.

POST /v1/sign/translate 입력은 version, session_id, segment_id,
timestamps(T), coordinates(T,49,2), mask(T,49 boolean), image_size([width,height])입니다.
응답은 같은 ID, status(ok/no_sign/uncertain/unavailable), glosses, text,
auto_speech, mock, model_version, timings_ms를 포함합니다. 오류 상태의 text는 비어 있습니다.
발화는 ok + auto_speech=true + mock=false에만 허용합니다. 본문은 최대 4MiB, 프레임은 900개입니다.

장치는 미완료 요청 한 개를 유지하고, 총 30초 내에서 연결 오류를 같은 ID로 한 번 재시도합니다.
서버는 성공 결과를 5분간 캐시하며 같은 ID의 다른 입력은 거부합니다.
취소한 요청은 완료까지 자리를 유지해 다음 요청이 쌓이지 않게 합니다.
캐시는 단일 프로세스 메모리이므로 API는 uvicorn worker 한 개로 운영합니다.
다중 프로세스 확장에는 공유 중복 처리 저장소가 필요합니다.

~~~bash
python tests/run_all.py continuous cloud_pipeline pose_sequence -v
python tests/run_all.py
~~~

검사는 ASGI API·HTTP 직렬화·vLLM 요청 형식, 학습·저장·재로딩, 데이터 누수와
취소·만료·재시도·중복 처리를 확인합니다. 실제 GPU·카메라·마이크·OLED 검증과 구분합니다.
