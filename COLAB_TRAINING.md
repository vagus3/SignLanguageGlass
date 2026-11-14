# AI Hub SEN JSON → Colab 학습

현재 보유한 `C:\AIHub\01_syn_sen_keypoint\SEN`에는 `keypoint`와 **`morpheme` 원본 라벨이 모두 있습니다.**
REAL 요약 DOCX를 SYN 정답으로 대입할 필요가 없습니다. `tools/prepare_sen.py`가 클립 전체 ID와
라벨의 `metaData.name`을 확인하고, `data[].start/end` 순으로 `attributes[0].name`을 연결합니다.
비수지 속성과 형태소 구간도 manifest에 보존합니다. 원본 영상 다운로드는 변환의 필수 조건이 아닙니다.

현재 SYN 좌표의 해상도는 사용자가 확인한 **2048×1152**, FPS는
[AI Hub의 키포인트 설명](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)에 따른 30입니다.
다른 데이터에는 해당 데이터의 실제 해상도와 FPS를 사용하세요.

## PC에서 변환

프로젝트 루트의 PowerShell에서 실행합니다. NumPy만으로 변환할 수 있으며 PyTorch는 학습할 때 필요합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/prepare_sen.py `
  C:\AIHub\01_syn_sen_keypoint `
  --out data/processed/aihub-sen-new `
  --image-size 2048 1152 --fps 30 --view F --skip-invalid `
  --colab-out reports/colab-sen-new
```

출력 경로가 이미 있으면 중단합니다. 재실행에는 새 경로를 지정하세요.
현재 변환 결과는 `data/processed/aihub-sen-training/`, 업로드용 번들은
`reports/colab-sen-training/`에 준비되어 있습니다.

| 현재 정면 데이터 검사 | 결과 |
|---|---:|
| 정확히 짝이 맞는 원본 라벨 | 500클립 |
| 학습 입력 변환 완료 | 498클립 / 61,344프레임 |
| 시간 범위 불일치로 제외 | 2클립 |
| 특징의 NaN/Inf·좌표 범위 초과 | 0 |

제외된 클립은 `NIA_SL_SEN0175_SYN05_F`(라벨 영상 3.6초, 키포인트 0.9초)와
`NIA_SL_SEN0545_SYN05_F`(라벨 영상 3.433초, 키포인트 2.7초)입니다.
순차적인 파일명만 검사하면 이런 뒷부분 누락을 놓치므로 라벨의 영상 길이도 비교합니다.
상세 사유는 `manifest.json`의 `excluded_from_conversion`에 있습니다.

## 출력과 오류 처리

- 각 `clip_*.npz`: 원래 프레임 간격의 timestamps, 정규화한 `(T,49,2)` 좌표, `(T,49)` 마스크,
  이미지 크기, 입력 버전. CTC 특징 추출 시 `(T,331)`이 됩니다.
- `manifest.json`: 정답 단어열, 형태소 시각·속성, 원본 클립·동작 그룹, 출처, 시퀀스 길이,
  좌표 유효 비율, 누락 프레임 범위, 변환/제외 건수.
- 기본은 오류 시 중단하며 완성되지 않은 출력은 게시하지 않습니다. `--skip-invalid`를 명시하면
  잘못된 클립의 사유를 남기고 정상 클립만 변환합니다. 라벨 누락/오류는 `index_audit.excluded`에 남습니다.
- 프레임 번호는 숫자로 정렬하고 중복을 거부합니다. 한 손이나 몸 좌표가 비어 있으면 마스크로 보존하며,
  잘못된 배열 길이는 오류로 처리합니다. 양손이 모두 무효인 클립은 제외합니다.
- 2D 세 번째 값은 무시합니다. 유한하며 `(0,0)`이 아닌 x/y와 영상 범위로 마스크를 만듭니다.
  마스크는 검출 신뢰도나 가림 검증 결과가 아닙니다. 빈 3D 필드를 z=0으로 채우지 않습니다.
- `--view all`은 다섯 시점을 처리합니다. 같은 원본 동작은 항상 같은 `group_id`를 유지합니다.
  `--limit 20`으로 작은 변환을 먼저 확인할 수 있으며 기본 `--limit 0`은 전체입니다.

해상도를 아직 모르는 새 데이터는 `--image-size`를 생략해 원시 x/y 캐시를 만듭니다.
캐시 버전 `sen-raw-xy-v1`은 바로 CTC에 넣지 않습니다. 해상도 확인 후 JSON 재처리 없이 정규화합니다.

```powershell
.\.venv\Scripts\python.exe tools/prepare_sen.py C:\AIHub\01_syn_sen_keypoint `
  --out data/processed/sen-xy-cache --skip-invalid
.\.venv\Scripts\python.exe tools/prepare_sen.py data/processed/sen-xy-cache `
  --from-cache --image-size 2048 1152 --out data/processed/sen-normalized
```

기존 `prepare_aihub3d.py`는 WORD의 실제 3D 손 모양 데이터용입니다. 현재 SYN 2D 데이터에는
`prepare_sen.py`를 사용합니다. DOCX용 `prepare_sen_colab.export_index`는 이전 실험의 재현용으로 유지합니다.

## Colab에서 학습

1. `reports/colab-sen-training/`의 `colab_code.zip`, `sen_training_data.zip`을 Drive의
   `내 드라이브/SignLanguageGlass`에 올립니다. 예전 `reports/colab-input/colab_code.zip` 대신 새 번들을 사용하세요.
2. [notebooks/aihub_sen_colab.ipynb](notebooks/aihub_sen_colab.ipynb)를 Colab에서 열고 GPU 런타임을 선택합니다.
3. 기본 `USE_PREPARED_DATA=True`, `IMAGE_SIZE=[2048,1152]`, `FPS=30`을 확인합니다.
   NPZ와 코드 ZIP만 압축 해제하므로 원본 JSON 수만 개를 다시 읽지 않습니다.
4. 20클립·3 epoch로 실행을 확인한 뒤 `MAX_CLIPS=0`으로 바꾸어 변환된 498클립 전체를 학습합니다.
5. 결과 `ctc.pt`, `training.json`, `manifest.json`, `data_audit.json`은
   Drive의 `SignLanguageGlass/runs/<실행시각>/`에 저장됩니다.

원본 JSON을 Colab에서 처리하려면 `USE_PREPARED_DATA=False`로 바꾸고 `keypoints.zip`에
`SEN/keypoint`와 `SEN/morpheme`을 함께 넣으세요. 이 경로도 같은 변환·검증 함수를 사용합니다.
Drive 연결과 GPU 자원은 [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html)를 참고하세요.

현재는 SYN05 한 원본 토큰입니다. 정확한 클립 라벨 연결은 확인했지만 실제 화자/캐릭터 정체와
미사용 화자 성능은 검증하지 않았습니다. 노트북은 전체 선택 데이터의 학습 손실을 확인하며,
검증·테스트 정확도를 계산하지 않습니다. 여러 시점을 무작위로 나눠 독립 성능으로 보고하지 마세요.
독립 train/val/test 데이터가 확보되면 [CLOUD_SETUP.md](CLOUD_SETUP.md)의 정식 평가 경로를 사용합니다.

CTC 체크포인트는 로컬 단어용 `models/sign_lstm.onnx`와 입력 규격이 다릅니다.
학습과 실시간 인식 성능 측정, 앱 연결은 변환 이후의 별도 단계입니다.
