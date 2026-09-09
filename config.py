# -*- coding: utf-8 -*-
"""
signglass 중앙 설정
모든 모듈이 여기만 바라봅니다. 튜닝은 이 파일에서만 하세요.
"""
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
ASSET_DIR = ROOT / "assets"
TTS_CACHE = ASSET_DIR / "tts_cache"
FONT_DIR = ASSET_DIR / "fonts"

MODEL_PATH = MODEL_DIR / "sign_lstm.onnx"   # 실행은 항상 ONNX. 변환: tools/to_onnx.py
LABELS_PATH = MODEL_DIR / "labels.json"
VOSK_MODEL = MODEL_DIR / "vosk-ko"
CALIB_PATH = MODEL_DIR / "hud_calib.json"

# 촬영자 ID. 데이터 파일명에 기록되어 '화자 단위 평가'에 쓰입니다.
# 촬영자가 바뀔 때마다 반드시 바꿔주세요. 이걸 안 하면 정확도가 거짓으로 나옵니다.
SIGNER_ID = "me"
# 같은 사람도 날짜·장소가 바뀌면 새 세션 ID를 쓰세요. 파일명에 함께 기록되어
# 같은 날 연속 촬영한 거의 동일한 샘플을 구분할 수 있습니다.
SESSION_ID = "session1"

IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"

# ============================================================
# 1. 어휘
# ============================================================
# 규칙 4가지:
#   1) 손 모양이 서로 확실히 다를 것
#   2) 정적 수형이 아니라 궤적이 있을 것 (LSTM 의 장점)
#   3) 시연 대화 한 편으로 엮일 것
#   4) NEUTRAL(무동작)을 반드시 포함 - 없으면 가만히 있어도 단어를 뱉습니다
#
# 국립국어원 한국수어사전(sldict.korean.go.kr)에서 정확한 수형을 확인하고 촬영하세요.
VOCAB = [
    "NEUTRAL",      # 무동작 - 필수. 전체 데이터의 20~25% 를 여기에 쓰세요
    "안녕하세요",
    "감사합니다",
    "죄송합니다",
    "도와주세요",
    "이름",
    "무엇",
    "어디",
    "얼마",
    "네",
    "아니요",
    "모르겠어요",
    "천천히",
    "다시",
    "병원",
]

# 수어 단어열 -> 자연스러운 한국어 문장 (Phase 1 후처리)
# 한국수어는 한국어의 단어 치환이 아니라 별개 문법 체계라서 이 레이어가 필요합니다.
PHRASE_RULES = [
    (("이름", "무엇"), "이름이 뭐예요?"),
    (("어디", "병원"), "병원이 어디예요?"),
    (("얼마",), "얼마예요?"),
    (("천천히",), "조금만 천천히 말해 주세요."),
    (("다시",), "다시 한 번 말씀해 주시겠어요?"),
    (("모르겠어요",), "잘 모르겠어요."),
]

# ============================================================
# 2. 시퀀스 / 추론
# ============================================================
SEQ_LEN = 30            # 모델이 받는 시퀀스 길이(프레임)
STRIDE = 5              # 5프레임마다 추론 -> 체감 지연 1000ms 를 170ms 로

# ★ 학습 데이터를 찍은 프레임레이트. SEQ_LEN 을 '시간'으로 환산하는 기준입니다.
#   SEQ_LEN 30 / TRAIN_FPS 30 = 1.0초 창
TRAIN_FPS = 30

# ★ 시간 기반 창 (프레임레이트가 달라져도 견디게 하는 스위치)
#
#   False 면 '최근 30프레임'을 그대로 씁니다. 노트북 30fps 로 학습하고
#   노트북 30fps 로 실행할 때는 이게 정확합니다.
#
#   True 면 '최근 1.0초'를 모아 30개로 리샘플해서 넣습니다.
#   라즈베리파이처럼 10fps 밖에 안 나오는 기기에서 반드시 필요합니다.
#   안 켜면 30프레임이 1초가 아니라 3초가 되어, 같은 수어인데 모델이 보는
#   시간 축이 3배로 늘어납니다. 에러는 안 나고 정확도만 무너집니다.
ADAPTIVE_WINDOW = True
CONF_THRESHOLD = 0.80   # 확정 임계값
VOTE_WINDOW = 5         # 최근 5회 예측 중
VOTE_MIN = 3            # 3회 이상 같아야 확정
COOLDOWN_SEC = 1.2      # 같은 단어 연속 발화 방지(debounce)

SEQ_PER_LABEL = 40      # 라벨당 수집할 시퀀스 개수 (최소 30, 많을수록 좋음)

# ============================================================
# 3. 카메라
# ============================================================
CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720
CAM_FPS = 30
MODEL_COMPLEXITY = 0    # 0 = 가장 빠름. 수어 인식엔 충분합니다
MIRROR_PREVIEW = True   # 프리뷰만 좌우 반전(셀피 뷰). 좌표 계산에는 영향 없음

# 수동 노출: 수어 인식이 깨지는 최대 원인은 해상도가 아니라 '모션 블러'입니다.
# 자동 노출이면 조금만 어두워도 셔터가 길어져 빠른 손이 뭉개지고,
# MediaPipe 가 손가락 관절을 놓칩니다. 셔터를 고정하고 조명으로 밝기를 메우세요.
# ※ macOS AVFoundation 은 OpenCV 노출 제어를 대부분 무시합니다(정상). 아래 참고.
CAM_MANUAL_EXPOSURE = True
CAM_EXPOSURE = -6       # 값이 작을수록 빠른 셔터. -5 ~ -8 사이에서 찾으세요
CAM_GAIN = None         # None = 건드리지 않음. 어두우면 100~200 시도

def camera_backend():
    """OS별 최적 백엔드. Windows MSMF 는 첫 프레임이 3~5초 늦습니다."""
    import cv2
    if IS_WIN:
        return cv2.CAP_DSHOW
    if IS_MAC:
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY

# ============================================================
# 4. 오디오 / STT
# ============================================================
AUDIO_RATE = 16000
AUDIO_BLOCK = 4000      # vosk 권장 블록
VAD_AGGRESSIVENESS = 2  # 0(관대) ~ 3(엄격)
STT_ENDPOINT_MS = 600   # 이만큼 무음이 이어지면 발화 종료로 보고 확정
TTS_VOICE = "ko-KR-SunHiNeural"   # edge-tts 캐시 생성용

# 반이중(half-duplex) 게이트
# 스피커로 나간 TTS 음성을 마이크가 다시 주워서 자막으로 띄우는
# 에코 루프를 막습니다. 이거 없으면 시연에서 100% 터집니다.
AUDIO_GATE_TAIL_SEC = 0.4   # TTS 끝난 뒤 이 시간만큼 더 마이크를 막음

# ============================================================
# 5. HUD / 디스플레이
# ============================================================
OLED_W, OLED_H = 128, 64
HUD_FONT = FONT_DIR / "Galmuri11.ttf"   # 저해상도 한글 픽셀 폰트
HUD_FONT_SIZE = 11
# 줄바꿈은 폰트 메트릭(실제 픽셀 폭)으로 합니다. 이 값은 폰트를 못 읽었을 때를
# 대비한 안전 상한일 뿐이라, 평소에는 걸리지 않을 만큼 넉넉해야 합니다.
# 여기를 9 같은 작은 값으로 두면 숫자·영문이 섞인 자막이 절반만 차고 넘어갑니다.
HUD_MAX_CHARS_PER_LINE = 24
HUD_MAX_LINES = 3
HUD_HOLD_SEC = 4.0           # 자막 유지 시간

# 카메라 시야(110도) 중 디스플레이가 실제로 덮는 영역의 비율.
# 얼굴이 이 영역 밖이면 말풍선 대신 방향 화살표를 띄웁니다.
HUD_FOV_X = (0.28, 0.72)
HUD_FOV_Y = (0.20, 0.80)

# 카메라와 눈의 위치 차이(시차) 1회성 보정값. calibrate 로 구한 뒤 여기에 적으세요.
HUD_CALIB_OFFSET = (0, 0)     # 픽셀 단위 (dx, dy)
HUD_CALIB_SCALE = (1.0, 1.0)

# One Euro Filter (이동평균보다 같은 부드러움에서 지연이 적습니다)
FILTER_MIN_CUTOFF = 1.0
FILTER_BETA = 0.02

# ============================================================
# 6. 디스플레이 백엔드
# ============================================================
#   "preview" : PC 창에 6배 확대 (하드웨어 0원, Step 1~2)
#   "serial"  : PC + 아두이노/Pico 브리지 경유 실제 OLED (Step 3)
#   "pi"      : 라즈베리파이 GPIO 직결 (브리지 MCU 불필요)
#   "null"    : 출력 없음
DISPLAY_BACKEND = "preview"
SERIAL_PORT = None      # None 이면 자동 탐색 (COM3 / /dev/cu.usbmodem...)
SERIAL_BAUD = 921600
