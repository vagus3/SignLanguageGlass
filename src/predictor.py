# -*- coding: utf-8 -*-
"""
슬라이딩 윈도우 추론 + 확정 로직

지연을 줄이는 핵심:
    30프레임을 다 채우고 추론하면 체감 지연이 1초입니다.
    링 버퍼를 유지하면서 STRIDE(5프레임)마다 추론하면 170ms 로 줄어듭니다.

오출력을 줄이는 3단 방어:
    1) confidence >= 0.80
    2) 최근 5회 예측 중 3회 이상 같은 단어일 때만 확정 (다수결)
    3) 확정 후 1.2초 쿨다운 (같은 단어 연타 방지)

프레임레이트가 달라져도 견디는 법 (config.ADAPTIVE_WINDOW):
    '최근 30프레임'이 아니라 '최근 1.0초'를 모아 30개로 리샘플해서 넣습니다.
    노트북(30fps)에서 학습한 모델을 라즈베리파이(10fps)에 올리면, 30프레임이
    1초가 아니라 3초가 됩니다. 같은 수어인데 모델이 보는 시간 축이 3배로
    늘어나 정확도가 무너지는데 에러는 하나도 안 납니다. 찾기 제일 어려운
    종류라 아예 시간 기준으로 창을 잡습니다.
"""
import sys
import json
import time
from pathlib import Path
from collections import deque

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.landmarks import FEATURE_DIM, FEATURE_VERSION, N_FLAGS


class SignPredictor:
    """
    모델 로딩은 SignRuntime 에 위임합니다.
    여기서 tensorflow/torch 를 직접 import 하지 않는 게 핵심입니다.
    안경 실행 경로에는 onnxruntime 만 올라갑니다.
    """

    def __init__(self, model_path=None, labels_path=None):
        from src.runtime import SignRuntime

        model_path = Path(model_path or C.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} 가 없습니다.\n"
                f"  학습/ONNX 생성:  python -m src.train")

        labels = None
        labels_path = Path(labels_path or C.LABELS_PATH)
        if labels_path.exists():
            labels = json.loads(labels_path.read_text(encoding="utf-8"))

        self.rt = SignRuntime(model_path, labels=labels)
        # labels.json 이 없어도 *_preproc.json 안에 라벨이 있을 수 있습니다.
        # 실제 런타임이 사용할 라벨을 기준으로 계약을 검사해야 합니다.
        self._check_contract(self.rt.labels)
        n = self.rt.n_classes or 1000
        self.labels = self.rt.labels or [str(i) for i in range(n)]

        # 시간 기반 창: 창 길이와 추론 간격을 '초'로 환산해 둡니다.
        self.adaptive = getattr(C, "ADAPTIVE_WINDOW", False)
        fps = float(getattr(C, "TRAIN_FPS", 30) or 30)
        self.win_sec = C.SEQ_LEN / fps          # 예: 30/30 = 1.0초
        self.stride_sec = C.STRIDE / fps        # 예:  5/30 = 0.167초

        # (t, feature) 쌍을 담습니다. 고정 프레임 모드면 30개로 충분하고,
        # 시간 기반 모드면 고프레임 카메라도 담기게 넉넉히 잡습니다.
        self.buf = deque(maxlen=max(8 * C.SEQ_LEN, 256) if self.adaptive
                         else C.SEQ_LEN)
        self.votes = deque(maxlen=C.VOTE_WINDOW)
        self._since = 0
        self._t_infer = 0.0
        self._last_word = None
        self._last_t = 0.0
        self.last_conf = 0.0
        self.last_raw = None
        self.last_ms = 0.0
        self.last_frame_t = 0.0    # 이번 추론에 쓰인 가장 최신 프레임의 캡처 시각
        self.last_fps = 0.0        # 창 안에서 실제로 들어온 프레임레이트

    @staticmethod
    def _fail(msg):
        raise RuntimeError(msg)

    def _check_contract(self, labels):
        """
        모델과 이 실행 경로가 같은 약속을 지키고 있는지 시작할 때 대조합니다.

        여기서 안 잡으면 어떻게 되는가:
          차원이 다르면   -> onnxruntime 이 shape 에러를 내거나(운이 좋은 경우),
                            동적 축이면 조용히 통과하고 결과만 쓰레기가 됩니다.
          라벨 수가 다르면 -> argmax 인덱스가 엉뚱한 단어를 가리킵니다.
                            "감사합니다" 를 하면 "병원" 이 나오는데 에러는 없습니다.
        둘 다 원인 찾기 제일 어려운 부류라, 시끄럽게 죽는 쪽이 낫습니다.
        """
        want = self.rt.input_dim
        if want and int(want) != FEATURE_DIM:
            self._fail(
                f"특징 차원 불일치: 모델 {want} vs landmarks.py {FEATURE_DIM}\n"
                f"  이 실행 경로는 landmarks.extract() 의 {FEATURE_DIM}차원을 넣습니다.\n"
                f"  다른 전처리(features_v2 등)로 학습한 모델이면 추론 쪽도 같이 바꾸세요.")

        model_feature = getattr(self.rt, "spec", {}).get("feature_version")
        if model_feature and model_feature != FEATURE_VERSION:
            self._fail(
                f"특징 버전 불일치: 모델 {model_feature} vs 실행 {FEATURE_VERSION}\n"
                f"  차원은 같아도 손 정규화 의미가 다릅니다. 현재 수집 데이터로 재학습하세요.")

        n = self.rt.n_classes
        if n and labels and len(labels) != int(n):
            self._fail(
                f"라벨 개수 불일치: {C.LABELS_PATH.name} {len(labels)}개 vs 모델 {n}개\n"
                f"  config.VOCAB 을 고친 뒤 재학습을 안 했거나, labels.json 이\n"
                f"  다른 모델의 것입니다. 이대로 돌리면 엉뚱한 단어를 발화합니다.")
        if n and not labels:
            print(f"[predictor] {C.LABELS_PATH} 가 없습니다. 숫자 인덱스로 표시합니다.")

    def push(self, feature, t=None):
        """
        프레임 특징 1개 투입.
        확정된 단어가 나오면 문자열, 아니면 None 반환.

        t : 그 프레임을 캡처한 시각(초). 생략하면 현재 시각.
            시간 기반 창과 지연 측정이 이 값을 씁니다. 추론이 끝난 뒤가 아니라
            '캡처한 순간'을 넘겨야 지연 숫자가 정직해집니다.
        """
        now = time.time() if t is None else float(t)
        self.buf.append((now, feature))

        seq = self._window(now)
        if seq is None:
            return None
        self.last_frame_t = now

        x = seq[None, ...].astype(np.float32)
        prob = self.rt.predict(x)
        self.last_ms = self.rt.last_ms
        i = int(prob.argmax())
        word, conf = self.labels[i], float(prob[i])
        self.last_conf, self.last_raw = conf, word

        self.votes.append(word if conf >= C.CONF_THRESHOLD else None)
        return self._confirm(word, conf)

    # ------------------------------------------------------------------
    def _window(self, now):
        """
        추론에 넣을 (SEQ_LEN, D) 시퀀스. 아직 때가 아니면 None.
        """
        if not self.adaptive:
            self._since += 1
            if len(self.buf) < C.SEQ_LEN or self._since < C.STRIDE:
                return None
            self._since = 0
            self.last_fps = C.TRAIN_FPS
            return np.stack([f for _, f in self.buf])

        # 창 밖으로 나간 프레임을 버립니다. 다만 왼쪽 끝을 보간하려면
        # 경계 바로 바깥의 프레임 하나는 남겨 둬야 합니다.
        cutoff = now - self.win_sec
        while len(self.buf) > 2 and self.buf[1][0] <= cutoff:
            self.buf.popleft()

        if now - self._t_infer < self.stride_sec:
            return None
        if len(self.buf) < 2 or now - self.buf[0][0] < self.win_sec:
            return None                      # 아직 한 창을 못 채웠습니다
        self._t_infer = now
        self.last_fps = (len(self.buf) - 1) / max(now - self.buf[0][0], 1e-6)
        return self._resample(cutoff, now)

    def _resample(self, t0, t1):
        """
        실제 캡처 시각 기준으로 [t0, t1] 을 SEQ_LEN 개로 균등 리샘플합니다.

        프레임 인덱스가 아니라 '시각'으로 보간하는 이유: 카메라는 프레임을
        고르게 주지 않습니다. 한 프레임이 밀리면 인덱스 기준 보간은 그 구간을
        실제보다 짧게 봅니다.
        """
        n = len(self.buf)
        ts = np.fromiter((t for t, _ in self.buf), np.float64, n)
        X = np.stack([f for _, f in self.buf]).astype(np.float32)

        grid = np.linspace(t0, t1, C.SEQ_LEN)
        hi = np.searchsorted(ts, grid, side="left").clip(1, n - 1)
        lo = hi - 1
        w = ((grid - ts[lo]) / np.maximum(ts[hi] - ts[lo], 1e-9)).clip(0.0, 1.0)
        out = (1.0 - w[:, None]).astype(np.float32) * X[lo] \
            + w[:, None].astype(np.float32) * X[hi]

        # 검출 플래그는 0/1 이라 보간하면 0.5 가 생깁니다. 학습 때 본 적 없는
        # 값이므로 가까운 쪽 프레임의 값을 그대로 가져옵니다.
        near = np.where(w < 0.5, lo, hi)
        out[:, -N_FLAGS:] = X[near][:, -N_FLAGS:]
        return out

    def _confirm(self, word, conf):
        if conf < C.CONF_THRESHOLD:
            return None
        if self.votes.count(word) < C.VOTE_MIN:
            return None
        if word == "NEUTRAL":
            self._last_word = None       # 무동작이 지나가면 다음 같은 단어를 허용
            return None

        now = time.time()
        if word == self._last_word and (now - self._last_t) < C.COOLDOWN_SEC:
            return None
        self._last_word, self._last_t = word, now
        self.votes.clear()
        return word

    def reset(self):
        self.buf.clear()
        self.votes.clear()
        self._since = 0
        self._t_infer = 0.0


class PhraseBuilder:
    """
    수어 단어열 -> 자연스러운 한국어 문장.

    한국수어는 한국어 어순과 다른 별개 문법 체계라서, 인식된 단어를
    그대로 이어붙이면 어색한 나열이 됩니다.

    ■ 왜 즉시 발화하면 안 되는가 (중요)
        단어가 들어오자마자 발화하면 "이름" -> "무엇" 순서로 수어했을 때
            1) "이름"        을 발화하고
            2) 곧이어 "이름이 뭐예요?" 를 또 발화합니다.
        상대방은 같은 말을 두 번 듣게 됩니다.

        그래서 단어를 바로 뱉지 않고 잠깐 물고 있다가(latch),
          - 규칙이 완성되면        -> 문장으로 발화
          - 시간 안에 안 오면      -> 그때서야 단어 그대로 발화
        합니다. 이 지연은 사람이 수어를 이어서 하는 시간과 겹치므로
        체감 지연이 늘지 않습니다.

    ■ 얼마나 기다려야 하는가 (대기 시간이 두 종류인 이유)
        물고 있는 단어가 어떤 규칙의 '앞부분'이면    -> window_sec 까지 기다림
        어떤 규칙의 시작도 아니면                    -> flush_sec 만 기다림

        예: "이름" 은 ("이름","무엇") 의 앞부분이므로 다음 수어를 기다립니다.
            "감사합니다" 는 어떤 규칙도 시작하지 않으므로 곧바로 흘려보냅니다.

        두 값을 구분하지 않고 짧은 쪽(flush_sec=1.0초)만 쓰면, 두 단어를
        1초 안에 이어서 수어해야만 문장 규칙이 걸립니다. 실제로는 단어 사이가
        1.5~2초라 규칙이 거의 안 걸리고, window_sec 은 아무 일도 하지 않습니다.

    사용법
        s = pb.feed(word)   # 단어가 확정될 때
        s = pb.poll()       # 매 프레임 (시간 초과분을 흘려보냄)
    """

    def __init__(self, window_sec=2.5, flush_sec=1.0):
        self.window_sec = window_sec    # 규칙이 이어질 수 있을 때의 대기 시간
        self.flush_sec = flush_sec      # 이어질 규칙이 없을 때의 대기 시간
        self.pending = []               # [(word, t), ...]
        self._rules = sorted(C.PHRASE_RULES, key=lambda r: -len(r[0]))
        # 어떤 규칙의 '진짜 앞부분'(자기 자신은 제외)만 모아 둡니다.
        # 규칙이 늘어도 poll() 이 매 프레임 전체를 훑지 않게 하기 위한 캐시입니다.
        self._prefixes = {pat[:i]
                          for pat, _ in C.PHRASE_RULES
                          for i in range(1, len(pat))}

    def feed(self, word):
        """규칙이 완성되면 문장을, 아니면 None(대기)을 반환."""
        now = time.time()
        self.pending = [(w, t) for w, t in self.pending
                        if now - t < self.window_sec]
        self.pending.append((word, now))

        seq = tuple(w for w, _ in self.pending)
        for pat, sentence in self._rules:   # 긴 규칙 우선
            if len(seq) >= len(pat) and seq[-len(pat):] == pat:
                self.pending.clear()
                return sentence
        return None

    def _may_extend(self, seq):
        """물고 있는 단어열의 꼬리가 어떤 규칙의 앞부분이면 True."""
        return any(seq[-i:] in self._prefixes
                   for i in range(1, len(seq) + 1))

    def poll(self):
        """메인 루프에서 매 프레임 호출. 조합 대기가 끝난 단어를 흘려보냅니다."""
        if not self.pending:
            return None
        seq = tuple(w for w, _ in self.pending)
        wait = self.window_sec if self._may_extend(seq) else self.flush_sec
        if time.time() - self.pending[-1][1] < wait:
            return None
        text = " ".join(seq)
        self.pending.clear()
        return text

    def reset(self):
        self.pending.clear()
