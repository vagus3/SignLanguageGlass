# AI Hub 수어영상 팀 실행 가이드

대상은 AI Hub **수어 영상** 데이터셋(`datasetkey=103`)이다. 전체 2~3TB를 받지 않고,
REAL/WORD의 형태소와 keypoint만 `filekey` 단위로 내려받아 462차원 3D 특징으로 변환한다.

## 원칙

- 원천 영상(`*_video.zip`)은 받지 않는다.
- 첫 실험은 형태소 110MB와 keypoint 11GB 한 묶음만 사용한다.
- API Key는 Git, 채팅, Google Drive에 올리지 않는다.
- AI Hub 원본 ZIP은 공용 Drive로 재배포하지 않는다. 각 팀원이 다운로드 승인을 받는다.
- Drive에는 변환 결과를 올리되, 가공 데이터 공유 범위도 AI Hub 이용정책을 확인한다.

## 파일 목록

AI Hub 다운로드 창에서 아래 경로를 연다.

```text
004.수어영상
└── 1.Training
    └── 라벨링데이터
        └── REAL
            └── WORD
```

`원천데이터/REAL/WORD`에 보이는 같은 번호의 `*_video.zip`은 선택하지 않는다.

### 첫 시험에 필요한 파일

| 용도 | 파일 | 용량 | filekey |
|---|---|---:|---:|
| 훈련 어휘 매핑 | `01_real_word_morpheme.zip` | 110MB | `39601` |
| 훈련 3D keypoint 첫 묶음 | `01_real_word_keypoint.zip` | 11GB | `39600` |
| 검증 어휘 매핑 | `01_real_word_morpheme.zip` | 14MB | `39478` |
| 검증 3D keypoint | `09_real_word_keypoint.zip` | 21GB | `39479` |

검증 데이터는 첫 파이프라인 확인 뒤 받는다. 학습과 검증을 합치지 않는다.

### 훈련 전체 목록: 사이트에서 선택할 정확한 파일명

| 순서 | 다운로드할 파일명 | 표시 용량 | filekey | 현재 상태 |
|---:|---|---:|---:|---|
| 어휘 | `01_real_word_morpheme.zip` | 110MB | `39601` | 완료 |
| 01 | `01_real_word_keypoint.zip` | 11GB | `39600` | 변환 완료 |
| 02 | `02_real_word_keypoint.zip` | 9GB | `39602` | 진행 중 |
| 03 | `03_real_word_keypoint.zip` | 11GB | `39603` | 대기 |
| 04 | `04_real_word_keypoint.zip` | 14GB | `39604` | 대기 |
| 05 | `05_real_word_keypoint.zip` | 10GB | `39605` | 대기 |
| 06 | `06_real_word_keypoint.zip` | 11GB | `39606` | 대기 |
| 07 | `07_real_word_keypoint.zip` | 10GB | `39607` | 대기 |
| 08 | `08_real_word_keypoint.zip` | 11GB | `39608` | 대기 |
| 09 | `09_real_word_keypoint.zip` | 11GB | `39609` | 대기 |
| 10 | `10_real_word_keypoint.zip` | 11GB | `39610` | 대기 |
| 11 | `11_real_word_keypoint.zip` | 11GB | `39611` | 대기 |
| 12 | `12_real_word_keypoint.zip` | 10GB | `39612` | 대기 |
| 13 | `13_real_word_keypoint.zip` | 11GB | `39613` | 대기 |
| 14 | `14_real_word_keypoint.zip` | 12GB | `39614` | 대기 |
| 15 | `15_real_word_keypoint.zip` | 11GB | `39615` | 대기 |
| 16 | `16_real_word_keypoint.zip` | 10GB | `39616` | 대기 |

표시 용량 합계는 약 173GB다. 한 번에 하나씩 처리하고 검증 후 원본을 지운다.
처음부터 전체 filekey를 한 명령에 넣지 않는다.

### 검증 데이터: 훈련과 경로를 혼동하지 않기

검증 파일은 아래 별도 경로에 있다.

```text
004.수어영상/2.Validation/라벨링데이터/REAL/WORD/
```

| 용도 | 다운로드할 파일명 | 표시 용량 | filekey |
|---|---|---:|---:|
| 검증 어휘 | `01_real_word_morpheme.zip` | 14MB | `39478` |
| 검증 keypoint | `09_real_word_keypoint.zip` | 21GB | `39479` |

훈련의 `09_real_word_keypoint.zip`(`39609`)과 검증의 동명 파일(`39479`)은
경로와 filekey가 다르다. 검증 데이터는 학습 데이터에 합치지 않는다.

### 초기 실험에서 제외

- `1.Training/원천데이터/REAL/WORD/*_real_word_video.zip`: `39546`~`39577`
- `2.Validation/원천데이터/REAL/01_real_word_video.zip`: `39483` (118GB)
- 문장(SEN) 영상과 keypoint
- CROWD 영상과 keypoint
- SYN 영상과 keypoint

손 모양 사전학습이 유효한지 확인한 뒤 필요할 때만 범위를 넓힌다.

## 팀원 PC 준비

Python 3.11을 사용한다.

```bash
git clone <프로젝트 저장소 URL>
cd wobbegong
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

MediaPipe 연결 확인:

```bash
python - <<'PY'
import mediapipe as mp
from src.landmarks import make_holistic
print(mp.__version__)
model = make_holistic()
model.close()
print('MediaPipe OK')
PY
```

## AI Hub 승인과 API Key

팀원마다 데이터셋 페이지에서 다운로드 승인을 받고 API Key를 발급한다.

```bash
mkdir -p ~/.config/signglass
printf '%s' '여기에_실제_AI_Hub_API_Key' > ~/.config/signglass/aihub-api-key
chmod 600 ~/.config/signglass/aihub-api-key
```

파일에는 실제 Key만 넣는다. `AIHUB_API_KEY=`나 따옴표를 넣지 않는다.

공식 다운로더 설치:

```bash
mkdir -p ~/.local/bin
curl -fsSL -o ~/.local/bin/aihubshell https://api.aihub.or.kr/api/aihubshell.do
chmod 700 ~/.local/bin/aihubshell
```

## 어휘 사전 생성

형태소 파일 `39601`을 받은 뒤 ZIP 전체를 풀지 않고 3,000개 어휘를 추출한다.

```bash
mkdir -p data/raw/aihub103/metadata
cd data/raw/aihub103/metadata
~/.local/bin/aihubshell -mode d -datasetkey 103 -filekey 39601 \
  -aihubapikey "$(<~/.config/signglass/aihub-api-key)"
cd ../../../..

python tools/extract_aihub_vocab.py \
  data/raw/aihub103/metadata/004.수어영상/1.Training/라벨링데이터/REAL/WORD/01_real_word_morpheme.zip \
  --out data/processed/aihub103/word_vocab.json
```

## keypoint 분할 다운로드와 변환

### macOS 분할 ZIP 병합 확인

공식 `aihubshell` 일부 버전은 `part0, part1, part10, part11, part2...` 순서로
병합할 수 있다. 현재 이 컴퓨터의 `~/.local/bin/aihubshell`에는 자연수 정렬과 ZIP CRC
검사를 적용했다. 다른 팀원은 다운로드 전에 아래 문자열이 표시되는지 확인한다.

```bash
rg 'sort -zV|unzip -tq' ~/.local/bin/aihubshell
```

두 줄이 없다면 큰 keypoint ZIP 다운로드를 시작하지 말고 스크립트를 먼저 보완해야 한다.
CRC 검증 전 part 파일을 삭제하는 버전은 병합 실패 시 전체 파일을 다시 받아야 할 수 있다.

첫 묶음만 실행한다.

```bash
AIHUB_API_KEY="$(<~/.config/signglass/aihub-api-key)" \
python tools/aihub_shards.py \
  --dataset-key 103 \
  --file-keys 39600 \
  --work-dir data/raw/aihub103/shards \
  --out-dir data/processed/aihub103/shards \
  --aihubshell ~/.local/bin/aihubshell
```

도구는 다음 순서로 동작한다.

1. filekey 전용 폴더에 다운로드
2. 분할 ZIP 병합 결과를 안전하게 압축 해제
3. AI Hub 3D 관절을 `(30, 462)` 특징으로 변환
4. `dataset.npz`와 `manifest.json` 계약 검증
5. `.complete` 체크포인트 기록

재실행하면 `.complete`가 있는 shard는 건너뛴다.

## 원본 ZIP 삭제

처음에는 원본을 유지한다. 다음 파일이 모두 있고 검증을 통과한 뒤에만 해당
`filekey-...` 작업 폴더를 삭제한다.

```text
data/processed/aihub103/shards/filekey-39600/dataset.npz
data/processed/aihub103/shards/filekey-39600/manifest.json
data/processed/aihub103/shards/filekey-39600/.complete
```

다음 shard부터 자동 정리하려면 `--delete-source`를 추가한다. 이 옵션은 변환 검증 후
해당 `data/raw/.../filekey-NNNNN` 폴더만 삭제한다.

어휘 사전을 확인한 뒤 형태소 ZIP도 삭제할 수 있다. `word_vocab.json`은 유지한다.

## Google Drive에 저장

권장 공유 대상:

```text
aihub103/
├── word_vocab.json
└── shards/
    └── filekey-39600/
        ├── dataset.npz
        ├── manifest.json
        └── .complete
```

원본 ZIP과 압축 해제 JSON은 올리지 않는다.

`rclone`에 Google Drive remote를 `gdrive`라는 이름으로 설정한 예:

```bash
rclone copy data/processed/aihub103 \
  gdrive:signglass/aihub103 \
  --progress --checksum
```

다른 팀원이 받을 때:

```bash
rclone copy gdrive:signglass/aihub103 \
  data/processed/aihub103 \
  --progress --checksum
```

Drive 동기화 뒤에는 로컬 `manifest.json`의 시퀀스 수와 `dataset.npz`의 첫 차원이
일치하는지 다시 확인한다.

## MediaPipe 실제 카메라 데이터

AI Hub와 같은 462차원 특징으로 수집한다.

```bash
source .venv/bin/activate
python tools/capture_mediapipe3d.py WORD0001 \
  --signer person01 --mirrored
```

- 손이 검출된 최근 30프레임을 유지한다.
- `Space`로 저장하고 `q`로 종료한다.
- 서로 다른 사람 ID로 수집해야 화자 분리 평가가 가능하다.

## 현재 학습 범위

이 파이프라인의 462차원은 손 관절 쌍 거리 기반의 **손 모양 사전학습 입력**이다.
손바닥 방향, 손목 궤적, 몸 기준 위치가 제거되므로 이것만으로 3,000개 수어를 완전히
구분한다고 가정하면 안 된다. 첫 shard로 학습 가능성을 측정한 다음 실제 착용 카메라
데이터를 더해 미세조정한다.
