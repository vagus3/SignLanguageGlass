# 직접 해야 하는 테스트 목록

자동 테스트(`python tests/run_all.py`)는 코드 계약을 확인합니다. 아래 항목은 사람, 실제 장비,
실제 데이터나 네트워크가 있어야 결과가 의미 있습니다. 결과 파일은 `reports/`에 모읍니다.
Git에는 올리지 않습니다.

## 0. 시작 전 공통 준비

- `python check_env.py`가 전부 `[OK]`인지 확인합니다. 기본 STT 모델은 README "STT 모델" 명령으로 받습니다.
- **서버를 먼저 새 코드로 배포**합니다.
  - 새 클라이언트는 gzip 본문을 보내고, 기기 인식 경로(B)는 `/v1/sign/compose`를 씁니다.
  - 예전 서버를 잠시 써야 하면 장치에서 `SIGN_SERVER_GZIP=0`을 설정합니다.
- 서버가 뜨면 `curl -H "Authorization: Bearer $SIGN_SERVER_TOKEN" $SIGN_SERVER_URL/health`로
  확인합니다. 볼 값은 `recognizer_ready`, `device_model_ready`, `data_sources`입니다.

## 1. 의미 검수

검수할 문장은 지금 시연 어휘(`config.VOCAB`, 한국어 단어 글로스)로 만듭니다. 검수 파일은 모두
`reports/` 안에 둡니다. 이 폴더는 깃에서 제외되므로, 검수자 이름이 공개 저장소에 올라가지 않습니다.
용어집을 저장소에 올릴지는 팀이 정한 뒤 옮깁니다.

1. **언어 모델 준비:** 시연에 쓸 vLLM 서버를 켜고 `VLLM_BASE_URL`, `VLLM_MODEL`을 설정합니다.
   - 보정은 모델 이름에 묶이므로, 시연과 같은 모델로 검수해야 합니다.
   - 서버가 꺼져 있거나 모델 이름이 다르면, 4단계 내보내기가 검수표를 만들지 않고 이유를 알려 줍니다.
2. **용어집 초안 만들기:**
   ~~~bash
   python tools/review_sentences.py draft --out reports/glossary.json
   ~~~
   - 단어마다 `ko`(뜻)를 확인하고 고칩니다.
   - 숫자·부정·인물·장소 단어에는 `kind`를 넣습니다(`number`/`negation`/`person`/`place`).
3. **검수할 글로스열 정하기:** 시연에서 나올 조합을 `reports/review_sequences.json`에 적습니다.
   - 자유 조합이므로 대본의 문장과 자주 섞일 조합을 넣습니다.
   - 형식은 `[["이름","무엇"], ["병원","어디"], ...]`입니다.
   - `config.PHRASE_RULES` 6개 패턴은 자동으로 포함됩니다.
4. **검수표 내보내기:**
   ~~~bash
   python tools/review_sentences.py export --glossary reports/glossary.json \
     --sequences reports/review_sequences.json --out reports/review_sheet.csv
   ~~~
5. **검수:** 엑셀에서 `reports/review_sheet.csv`를 엽니다.
   - 시스템이 실제로 말할 문장은 `text` 칸이고, 예전 규칙 문장은 `rule_text` 칸입니다.
   - 행마다 `meaning_preserved`(Y/N)를 적고, 필요하면 `corrected_text`(고친 문장), `reviewer`(검수자), `note`(메모)를 채웁니다.
   - `source`가 `words`/`reviewed_rules`/`uncertain`인 행은 말할 문장이 없습니다. 승인하지 말고 `corrected_text`에 문장을 씁니다.
   - 첫 줄(열 이름)은 고치지 않습니다. 저장은 엑셀의 "CSV UTF-8"과 그냥 "CSV" 둘 다 됩니다.
6. **결과 가져오기:**
   ~~~bash
   python tools/review_sentences.py import reports/review_sheet.csv --glossary reports/glossary.json \
     --out-glossary reports/glossary.reviewed.json --out-review reports/review.json
   python -m src.build_tts_cache --engine macos --glossary reports/glossary.reviewed.json
   ~~~
   - 승인·수정한 문장은 검수 문장(번역 메모리)이 됩니다. 같은 글로스열이 들어오면 언어 모델 없이 그 문장을 말합니다.
   - 두 번째 명령은 그 문장들의 음성을 미리 합성해 둡니다.
   - 서버는 `--mapping reports/glossary.reviewed.json`으로 띄웁니다.
- **통과 기준:** 시연 대본의 글로스열이 모두 Y이거나 고친 문장을 가지고, 해결 안 된 N이 없습니다.
- **주의:**
  - 검수 문장이 바뀌면 `policy_hash`가 바뀌어 기존 자동 발화 보정이 무효가 됩니다(5번에서 다시 합니다).
  - AI 허브 데이터로 학습하면 글로스가 AI 허브 단어 번호(WORD ID)가 됩니다. 그때 용어집의 단어를 번호로 옮깁니다. 뜻 판단은 그대로 유효합니다.

## 2. 실제 목소리 STT 비교

지금 기본값은 sherpa-onnx `models/sherpa-ko-174m-chunk32`입니다. 근거는 아래와 같고, 실제 사람 목소리 비교로 확정합니다.

| 데이터 | vosk | sherpa 공식 | sherpa 174M |
|---|---|---|---|
| 합성음성 22문장 CER | 58.0% | 49.0% | 13.0% |
| 실제 음성 4문장 CER | 47.4% | 11.8% | 0.0% |

1. 시연 마이크로 시연 장소에서 녹음합니다. 2명 이상이 각자 20~30문장을 말하고, 대본의 상대방 대사를 포함합니다.
   - 변환: `ffmpeg -i in.m4a -ar 16000 -ac 1 data/raw/stt_samples/team/001.wav`
2. 정답 파일 `refs.txt`에 한 줄씩 `001.wav 어디가 아프세요?` 형식으로 적습니다.
3. 비교:
   ~~~bash
   python tools/compare_stt.py --wavs data/raw/stt_samples/team --refs refs.txt -v \
     --engines vosk sherpa:models/sherpa-ko-174m-chunk32 sherpa:models/sherpa-ko-2024-06-16 \
     --out reports/stt_team.json
   ~~~
   - 비교용 공식 모델(`sherpa-ko-2024-06-16`)은 받는 파일이 418MB이고, int8 부품만 꺼냅니다.
     ~~~bash
     curl -L -o /tmp/ko.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2
     n=sherpa-onnx-streaming-zipformer-korean-2024-06-16; mkdir -p models/sherpa-ko-2024-06-16
     tar -xjf /tmp/ko.tar.bz2 --strip-components=1 -C models/sherpa-ko-2024-06-16 \
       $n/encoder-epoch-99-avg-1.int8.onnx $n/decoder-epoch-99-avg-1.int8.onnx \
       $n/joiner-epoch-99-avg-1.int8.onnx $n/tokens.txt
     ~~~
- **통과 기준:** CER이 가장 낮고 RTF가 0.5 미만이며, 발화 끝 → 확정 자막(final p50)이 대화에 무리 없는 엔진을 고릅니다.
  - 174M chunk-32는 구조상 약 0.64초 지연이 있습니다. 확정이 늦게 느껴지면 같은 저장소의 chunk-16(약 0.32초)을 비교합니다.
  - 고른 엔진은 `config.STT_BACKEND`, `SHERPA_MODEL`에 반영합니다.
- 실행 중 에코 확인: TTS가 말하는 동안 자막이 생기지 않아야 합니다. 스피커 시연일 때 확인합니다.

## 3. 실제 웹캠 자동 구간

~~~bash
python -m src.sign_server --mock          # 터미널 1
python main.py --sign-backend cloud --auto-segment --viewpoint webcam --debug   # 터미널 2
~~~

화면의 기준선을 보면서 문장 20번을 시도하고, 아래를 셉니다.
- 손을 올려 시작하고 내려 끝나는가
- 첫 동작이 잘리지 않는가
- 문장 중간의 짧은 멈춤에서 끊기지 않는가
- 키보드·턱 괴기·머리 만지기에서 시작되지 않는가

- **통과 기준:** 놓침 0, 잘림·분할 1회 이하, 일상 동작 10분당 오검출 2회 이하.
- 실패하면 정량 평가로 임계값을 조정합니다. AI Hub 문장 데이터의 형태소 시간이나 손으로 표시한 녹화의
  `sign_intervals`를 씁니다.
  ~~~bash
  python tools/evaluate_segments.py data/processed/continuous/manifest.json --split val --grid
  python tools/evaluate_segments.py data/processed/continuous/manifest.json --split test
  ~~~
  - 고른 값을 test 분할에서 한 번 더 확인하고 `src/segmenter.py` 기본값에 반영합니다.

## 4. 시연 노트북에서 배치 비교(A/B)

학습 전에도 지연은 잴 수 있습니다(무작위 가중치, 실제 크기 모델).

~~~bash
python tools/benchmark_paths.py data/processed/aihub_sample_extractors/mediapipe/manifest.json \
  --make-dummy models/bench
# 서버(클라우드):
python -m src.sign_server --host 0.0.0.0 --checkpoint models/bench/ctc.pt \
  --device-model models/bench/ctc.json --mapping models/bench/mapping.json --mock-language
# 시연 노트북:
python tools/benchmark_paths.py data/processed/aihub_sample_extractors/mediapipe/manifest.json \
  --onnx models/bench/ctc.onnx --url "$SIGN_SERVER_URL" --out reports/ab_wifi.json
~~~

- 행사장과 비슷한 Wi-Fi, 휴대폰 핫스팟에서 각각 한 번씩 측정합니다.
- 같은 시간에 `main.py`를 켜 둔 상태로도 한 번 잽니다(카메라·STT와 CPU 경쟁).
- 이 Mac에서 localhost로 잰 참고값(네트워크 없음)은 맨 아래에 있습니다. 네트워크가 끼면 A의 업로드가 차이를 키웁니다.
- **기록할 것:** 경로별 total p50/p95, 전송 KB, 클라이언트 CPU, 실패 수, 노트북 발열·팬 소음(체감).
  B가 더 유리하면 `--recognize device --onnx ...`로 시연합니다.

## 5. 일상 동작 600초와 자동 발화 보정

자동 발화는 보정을 통과한 모델에서만 켜집니다(CLOUD_SETUP 4절).

~~~bash
python -m src.collect_continuous --glosses --signer S1 --session day2   # 빈 glosses = 일상 동작
~~~

- 키보드·마우스·물 마시기·머리 만지기를 15초 이하 구간으로 나눠 600초 이상 모읍니다(서로 겹치지 않게).
- 학습 → 검증 분할 평가 → 1번의 `reports/review.json`을 `--semantic-review`로 넣어 보정합니다.
- 글로스열 단위 승인은 **예측이 정답과 같은 클립에만** 인정됩니다. 틀린 인식은 클립 단위 검수가 필요합니다.

## 6. 실제 장치 15분 연속 동작

~~~bash
python main.py --sign-backend cloud --auto-segment --stats reports/live.csv   # 15분 × 3회
~~~

- **통과 기준:**
  - 크래시 0
  - FPS 유지
  - 활동 모니터에서 메모리가 계속 늘지 않음
  - 에코 자막 없음
  - 지연 p95를 종료 출력과 `reports/live.csv`로 기록
- 참고: 장비 없이 HTTP·스레드·장애 처리만 확인하는 무인 소크는 `python tools/soak.py --minutes 15`입니다(결과는 맨 아래).

## 7. 네트워크 단절·취소 리허설

- 번역 요청 중 Wi-Fi를 끕니다. HUD에 "번역 서버 연결을 확인하세요"가 뜨고, 아무 소리도 나지 않아야 합니다.
- 다시 켠 뒤 다음 문장이 정상인지 봅니다.
- 처리 중 `c`(취소)를 누릅니다. 늦게 온 결과도 말하지 않아야 합니다.
- 서버를 끈 상태로 시작합니다. 크래시 없이 상태만 표시돼야 합니다.

## 8. 실물 OLED (`--display serial`)

- 내 번역이 길면 잘리지 않고 페이지가 넘어가는지 봅니다(오른쪽 아래 점).
- 상대 자막과 내 결과 중 최신 것이 보이는지 봅니다.
- 2.42인치·광학계에서 글자 크기와 유지 시간(`HUD_OWN_HOLD_SEC`, `HUD_PAGE_SEC`)이 충분한지 봅니다.

## 9. TTS 실제 소리 시점

- 지연 측정의 끝점은 재생 호출 직전입니다. 휴대폰으로 화면과 스피커를 함께 녹화해, 수어 끝 → 소리 시작을 프레임 단위로 잽니다.

## 10. AI Hub 데이터 링크를 받은 뒤 (Phase 1)

1. `tools/inspect_aihub.py`로 형식을 확인하고, 형태소 JSON → annotations 변환기를 만듭니다.
   - 글로스 순서, `sign_intervals`, WORD ID → 한국어 표제어를 담습니다.
2. 화자 단위로 나눈 CTC 기준 모델을 학습하고, 학습에 안 쓴 화자로 WER·coverage·10분당 오출력을 봅니다.
3. 일부 클립을 `tools/extract_mediapipe.py`로 다시 추출합니다. `tools/compare_extractors.py`로 비교한 뒤 기준 모델과 성능을 비교합니다.
4. 학습한 모델로 `tools/evaluate_rounding.py`를 돌려 좌표 반올림 허용 여부를 정합니다.
5. 형태소 구간으로 3번의 자동 구간 평가를 합니다.
6. 1번 검수 결과를 WORD ID 용어집으로 옮기고, `tools/export_ctc_onnx.py`로 B 경로 모델을 만듭니다.

## 11. 발표 자료 데이터 출처

발표 슬라이드·포스터·보고서 마지막에 다음 문구를 넣습니다(README "데이터 출처·라이선스").

> 이 프로젝트는 한국지능정보사회진흥원의 사업결과물인 AI 허브 '수어 영상' 데이터를
> 활용했습니다(https://aihub.or.kr/aidata/7965).

STT 모델 등 외부 자원 표도 README에서 옮깁니다.

---

## 참고: 이 Mac에서 자동으로 측정한 값

| 항목 | 결과 |
|---|---|
| 서버 파싱+검증 (2D 150 / 3D 150 / 3D 450프레임) | 3.9 / 9.5 / 42.2ms → 2.4 / 6.0 / 16.9ms |
| 중복 판정 지문 | 4.8 / 14.0 / 35.3ms → 0.02 / 0.06 / 0.18ms |
| HUD 렌더 (말풍선 / 두 영역) | 1.4 / 2.9ms → 0.002 / 0.006ms (같은 화면 재사용) |
| 좌표 본문 (샘플 100개, 2D, gzip) | 106KB → 56KB (무손실 float32 표기) |
| A/B localhost (샘플 100개, 실제 크기 무작위 모델, 네트워크 없음) | A: total p50/p95 12.9/30.1ms, 전송 57KB · B: 3.7/7.0ms, 0.2KB · 글로스 일치 100/100 |
| 무인 소크 15분 (HTTP, 장애 주입) | 통과. 27,000프레임 30fps, 구간 180개: ok 116(전부 발화) · 만료 38(취소 16 포함) · 미리보기 10 · 서버 단절 16. 잘못된 발화 0, 결과 유실 0, 메모리 증가 없음(49→41MB), 스레드 3~4 |
