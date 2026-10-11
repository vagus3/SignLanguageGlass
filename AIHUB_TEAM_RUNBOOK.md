# AI Hub 수어영상 팀 실행 가이드

대상은 AI Hub **수어 영상** 데이터셋(`datasetkey=103`)이다. 전체 2~3TB를 받지 않고,
REAL/WORD의 형태소와 keypoint만 `filekey` 단위로 내려받아 462차원 3D 특징으로 변환한다.

## 가이드를 읽기 전에

이 문서의 터미널 명령은 **Mac 기준**이다. Windows PowerShell에 그대로 붙여 넣지 않는다.

아래 순서대로 프로젝트를 열고, 필요한 프로그램을 설치하고, 카메라 연결을 확인한 뒤
AI Hub 데이터를 준비한다. MediaPipe 카메라 연결은 AI Hub 데이터를 받기 전에도 확인할 수 있다.
친

지금 이 컴퓨터는 이미 설치와 API Key 설정을 끝냈고 남은 다운로드 큐를 실행 중이다.
**이 컴퓨터에서는 설치나 전체 다운로드 명령을 또 실행하지 않는다.**

용어는 이것만 알아두면 된다.

- **MediaPipe:** 카메라 화면에서 손 관절 좌표를 찾아주는 프로그램.
- **keypoint ZIP:** AI Hub가 미리 추출해 둔 관절 좌표. 이번 학습에 사용할 원본.
- **morpheme ZIP:** `WORD0001` 같은 번호가 어떤 한국어 단어인지 알려주는 자료.
- **dataset.npz:** 원본 좌표를 우리 프로그램이 쓰기 쉽게 바꾼 작은 학습 파일.
- **API Key:** AI Hub 다운로드 인증용 비밀번호. 다른 사람에게 주지 않는다.

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

## 1. 프로젝트 열기 — 처음 설치할 때

먼저 프로젝트 코드가 들어 있는 폴더를 받는다. 폴더 안에 `README.md`,
`requirements.txt`, `src`, `tools`가 보여야 한다. 폴더 이름은 `signglass`일 수도,
`wobbegong`일 수도 있다. 이름보다 **안에 들어 있는 파일**이 중요하다.

Mac에서 「터미널」 앱을 연다. `cd `를 입력한 다음 Finder에서 프로젝트 폴더를
터미널로 끌어다 놓고 Enter를 누른다. 폴더 경로를 직접 외워서 입력할 필요 없다.

다음 명령으로 올바른 폴더인지 확인한다.

```bash
ls requirements.txt
```

`requirements.txt`라고 나오면 성공이다. `No such file`이면 폴더를 잘못 연 것이다.
이후 명령은 모두 **이 프로젝트 폴더에서** 실행한다.

## 2. 필요한 프로그램 설치 — 처음 한 번만

Python **3.11**이 설치되어 있어야 한다. 먼저 확인한다.

```bash
python3.11 --version
```

`Python 3.11.x`가 나오면 다음으로 넘어간다. `command not found`이면 Python 3.11을
먼저 설치해야 한다. 설치하지 않은 상태에서 아래 명령을 계속 실행하지 않는다.

아래 블록 전체를 터미널에 붙여 넣는다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`.venv`는 **이 프로젝트 전용 프로그램 설치 공간**이다. 다른 프로젝트 설정에 영향을
덜 주기 위해 사용한다. 설치 중 글자가 많이 나오는 것은 정상이며 오류 없이 끝나면 된다.

터미널을 새로 열 때는 설치를 반복하지 않고, 프로젝트 폴더에 들어가 아래 한 줄만 실행한다.

```bash
source .venv/bin/activate
```

MediaPipe가 설치됐는지 확인하려면 아래 블록을 그대로 실행한다.

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

마지막에 `MediaPipe OK`가 나오면 성공이다.

## 3. 카메라 테스트 — AI Hub 데이터 없이 가능

프로젝트 폴더에서 아래 명령을 실행한다.

```bash
python tools/capture_mediapipe3d.py WORD0001 --signer person01 --mirrored
```

카메라 접근 허용 창이 뜨면 허용한다. 화면에 손이 보이고 `frames 30/30`이 표시되면
Space를 눌러 한 샘플을 저장한다. `q`를 누르면 종료된다.

- `WORD0001`: 저장할 단어 번호. 현재 사전에서 「고민」에 해당한다. 실제 그 수어를 촬영할 때만 해당 학습 라벨로 사용한다.
- `person01`: 촬영하는 사람의 구분 이름. 다른 는 `person02`처럼 바꾼다.
- 저장 위치: 프로젝트의 `data/mediapipe3d/WORD0001/` 폴더.

단순 연결 테스트로 아무 손동작을 저장했다면 그 파일을 정답 학습 자료로 사용하지 않는다.
**여기까지 되면 MediaPipe 연결 테스트는 끝이다.** 데이터 다운로드나 AI 학습까지 완료됐다는 뜻은 아니다.

## 4. AI Hub API Key 저장

AI Hub 「수어 영상」 데이터셋의 다운로드 승인을 받고 API Key를 발급받는다.
팀원마다 자기 계정과 자기 Key를 사용한다.

아래 명령은 **Mac 기본 터미널(zsh)** 기준이다. 전체 블록을 붙여 넣는다.

```zsh
mkdir -p ~/.config/signglass
read -s "SIGNGLASS_KEY?AI Hub API Key를 붙여 넣고 Enter: "
printf '%s' "$SIGNGLASS_KEY" > ~/.config/signglass/aihub-api-key
chmod 600 ~/.config/signglass/aihub-api-key
unset SIGNGLASS_KEY
echo
```

`AI Hub API Key를 붙여 넣고 Enter:`라는 질문이 나오면 **발급받은 실제 키만**
붙여 넣고 Enter를 누른다. 보안을 위해 입력한 글자가 화면에 안 보이는 것이 정상이다.

예를 들어 키가 `EXAMPLE-1234-ABCD`라면 질문에 `EXAMPLE-1234-ABCD`만 입력한다.
이 문자열은 설명용 가짜 키이므로 실제로 사용하지 않는다.
`AIHUB_API_KEY=`나 따옴표, 파일 이름을 함께 입력하지 않는다.

파일을 직접 만들 필요 없다. 위 명령이 `~/.config/signglass/aihub-api-key`를 만들고 저장한다.
여기서 `~`는 내 사용자 폴더이며, `aihub-api-key`는 **키 값이 아니라 파일 이름**이다.

키를 화면에 출력하지 않고 저장 여부만 확인한다.

```bash
test -s ~/.config/signglass/aihub-api-key && echo '키 파일 저장됨'
```

`키 파일 저장됨`은 파일이 비어 있지 않다는 뜻이다. 실제 인증 성공 여부는 다운로드 때 확인된다.

## 5. AI Hub 다운로드 프로그램 설치

아래 블록을 그대로 실행한다.

```bash
mkdir -p ~/.local/bin
curl -fsSL -o ~/.local/bin/aihubshell https://api.aihub.or.kr/api/aihubshell.do
chmod 700 ~/.local/bin/aihubshell
```

이것은 **데이터를 받는 프로그램만 설치**하는 명령이다. 아직 데이터는 받지 않는다.
다음 「macOS 분할 ZIP 병합 확인」도 반드시 읽는다. 공식 프로그램을 새로 설치하면
이 컴퓨터에 적용한 병합 오류 수정까지 자동 설치되는 것은 아니다.

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

## Google Drive로 팀원에게 전달하기 — 쉬운 방법

이 작업은 자동 업로드 설정이 아니라 **파일을 직접 올리고 받는 방법**이다.
먼저 AI Hub 이용 조건에 따라 가공 데이터의 팀 내 공유가 허용되는지 확인한다.
이 가이드가 공유 허가를 대신하지는 않는다.

Finder에서 프로젝트의 `data/processed/aihub103/` 폴더를 연다.
Google Drive에서 팀 공유 폴더를 만들고 다음 파일만 같은 폴더 구조로 올린다.

- `word_vocab.json`: 단어 번호와 한국어 이름 연결표.
- 각 `shards/filekey-숫자/`의 `dataset.npz`, `manifest.json`, `.complete`.
- 검증이 완료되면 `validation/` 안의 어휘 사전과 같은 세 가지 변환 파일.

`.complete`는 이름이 점으로 시작하는 숨김 파일이다. Mac Finder에서
`Command + Shift + .`을 누르면 볼 수 있다. 업로드 중인 폴더를 완료됐다고 알리지 않는다.
원본 ZIP, 압축 해제 JSON, API Key, 다운로드 로그는 올리지 않는다.

업로드가 끝나면다음 순서로 파일을 받아 프로젝트에 넣는다.

1. 팀 공유 Drive에서 위 변환 파일들을 받는다.
2. 프로젝트 안에 `data/processed/aihub103/` 폴더를 만들고 넣는다.
3. Drive에서 폴더째 받아 ZIP이 생겼다면 압축을 푼다. `aihub103/aihub103/`처럼 폴더가 두 겹이 되지 않도록 한다.
4. 최종 위치가 아래 예시와 같은지 확인한다.

```text
프로젝트폴더/data/processed/aihub103/word_vocab.json
프로젝트폴더/data/processed/aihub103/shards/filekey-39600/dataset.npz
```

이렇게 받는 **AI Hub API Key 설정과 원본 다운로드 단계를 건너뛴다.**
프로그램 설치는 자기 PC에서 해야 한다. `.venv` 폴더 PC에 복사해 쓰지 않는다.

파일을 넣은 뒤 프로젝트 폴더의 터미널에서 아래 명령으로 정상적으로 읽히는지 확인한다.

```bash
python - <<'PY'
from pathlib import Path
from tools.aihub_shards import validate_output
folders = sorted(Path('data/processed/aihub103').rglob('filekey-*'))
if not folders:
    raise SystemExit('받은 파일의 폴더 위치를 다시 확인하세요.')
for folder in folders:
    print(folder.name, validate_output(folder), '개 시퀀스 확인 완료')
PY
```

`확인 완료`가 나오면 학습 입력 파일의 구조와 샘플 수가 맞는 것이다.
이 검사는 다운로드 전후 파일이 바이트 단위로 완전히 동일한지 검사하는 체크섬 검사는 아니다.

같은 파일을 여러 번 올리고 받을 때는 `rclone`이라는 파일 전송 프로그램으로
터미널에서 복사할 수도 있다. 위에서 설명한 직접 업로드/다운로드 방법을 사용했다면
아래 명령은 실행하지 않아도 된다. `gdrive`는 예시 연결 이름이며,
rclone 설정에서 Google Drive에 로그인하고 이 이름으로 연결을 만들어 둔 경우에만 실행된다.

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

## MediaPipe 실제 카메라 데이터 — 추가 수집

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
