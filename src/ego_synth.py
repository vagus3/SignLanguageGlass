"""AI Hub 3D 관절을 안경 카메라(착용자 시점)에서 본 2D 손 관절로 바꿔 학습 예시를 만듭니다.

AI Hub '수어 영상'은 앞에서 찍은 영상이지만 관절 JSON 에 손·얼굴·몸의 3D 좌표(카메라 기준
mm, 일부 클립은 m)가 들어 있습니다. 눈 사이에 가상 카메라를 두고 아래로 기울여 내려다보면
안경에서 본 손 모양과 위치가 됩니다. 얼굴·어깨는 안경에 보이지 않으므로 몸 관절 7개는
'안 보임'으로 두고 손 42개만 채웁니다.

카메라 기울기, 시야각, 팔을 앞으로 뻗은 정도는 예시마다 무작위로 바꿉니다. 실제 안경의
장착 각도와 사람마다 다른 자세를 미리 알 수 없기 때문입니다.

입력 자료 (data/raw/aihub103_3d, tools 의 pick_3d 결과):
  word/REALxx/NIA_SL_WORDxxxx_REALxx_F.npz  단어 클립, 구간은 word3d_intervals.json
  sen/REALxx/NIA_SL_SENxxxx_REALxx_F.npz    긴급 문장 클립, 단어별 구간은 aihub_sentences
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.pose_sequence import JOINTS, MAX_SECONDS, PoseSequence

W, H = 1280, 720
FPS = 30.0
# 착용자는 팔을 앞으로 뻗고 수어합니다(촬영 안내와 같음). 0cm 까지 넣으면 손이 화면 밖인
# 예시가 많아져, 조금만 가려져도 '수어 아님'이라고 답하게 됩니다.
VIEW = {"pitch": (25.0, 45.0), "fov": (70.0, 90.0), "reach": (100.0, 250.0)}


def to_mm(pose, *hands):
    """클립에 따라 3D 단위가 m 입니다 (어깨 너비가 5 보다 작으면 m 로 보고 1000 배)."""
    width = np.linalg.norm(pose[..., 2, :] - pose[..., 5, :], axis=-1)
    width = np.nanmedian(width) if np.ndim(width) else width
    k = 1000.0 if np.isfinite(width) and width < 5 else 1.0
    return [np.asarray(a, np.float64) * k for a in (pose, *hands)]


def head_frame(pose):
    """한 클립의 평균 머리 방향: 눈 가운데, 오른쪽, 위, 앞 (카메라 좌표 mm).

    BODY_25: 1 목, 2 오른어깨, 5 왼어깨, 8 골반 가운데, 15/16 눈. 위쪽은 몸통(골반->목)
    방향입니다. 프레임마다 다시 재면 3D 잡음으로 화면 전체가 흔들리므로 클립 평균을 씁니다.
    """
    ok = np.isfinite(pose[:, [2, 5, 15, 16]]).all(axis=(1, 2))
    if ok.sum() < 3:
        return None
    p = pose[ok]
    eye = np.nanmean(p[:, [15, 16]], axis=(0, 1))
    right = np.nanmean(p[:, 15] - p[:, 16], 0)
    torso = p[:, 1] - p[:, 8]
    up = np.nanmean(torso, 0) if np.isfinite(torso).all(1).sum() >= 3 else \
        np.nanmean(p[:, [15, 16]].mean(1) - (p[:, 2] + p[:, 5]) / 2, 0)
    right /= np.linalg.norm(right)
    up -= right * (up @ right)
    up /= np.linalg.norm(up)
    forward = np.cross(up, right)
    if forward[2] > 0:           # 사람은 원래 카메라 쪽(-z)을 봅니다
        forward = -forward
    return eye, right, up, forward


def project(points, frame, *, pitch, fov, reach=0.0, forward_mm=20.0):
    """(N, 3) mm -> (N, 2) 화면 비율 좌표와 보임 여부. reach 만큼 손을 앞으로 옮깁니다."""
    eye, right, up, forward = frame
    a = np.deg2rad(pitch)
    z = np.cos(a) * forward - np.sin(a) * up
    x = right
    y = np.cross(z, x)
    rel = points + forward * reach - (eye + forward * forward_mm)
    cam = np.stack([rel @ x, rel @ y, rel @ z], -1)
    f = (W / 2) / np.tan(np.deg2rad(fov) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = f * cam[..., 0] / cam[..., 2] / W + 0.5
        v = f * cam[..., 1] / cam[..., 2] / H + 0.5
    ok = (np.isfinite(u) & np.isfinite(v) & (cam[..., 2] > 50)
          & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))
    return np.stack([u, v], -1), ok


PALM = [0, 1, 5, 9, 13, 17]       # 손목과 손바닥 쪽 관절: 손등에 가려지지 않는 것으로 봅니다


def hidden_fingers(hand, frame, *, reach=0.0, forward_mm=20.0, margin=8.0):
    """(T, 21) 카메라에서 볼 때 손바닥 면 뒤쪽에 있어 손등에 가려지는 손가락 관절.

    손등 쪽에서 내려다보면 접은 손가락이 가려지고, MediaPipe 는 그 위치를 짐작합니다.
    손바닥 면(손목, 검지·새끼 뿌리)을 기준으로 카메라와 반대편에 margin mm 넘게 있으면 가려짐.
    """
    eye, _, _, forward = frame
    camera = eye + forward * forward_mm
    p = hand + forward * reach
    normal = np.cross(p[:, 5] - p[:, 0], p[:, 17] - p[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
    side_cam = np.einsum("tk,tk->t", camera - p[:, 0], normal)
    side = np.einsum("tjk,tk->tj", p - p[:, :1], normal)
    out = (side * side_cam[:, None] < 0) & (np.abs(side) > margin)
    out[:, PALM] = False
    return out & np.isfinite(p).all(-1)


def load_clips(root, word_ids, word_intervals, sentence_manifests, vocabulary):
    """학습에 쓸 3D 클립 목록. vocabulary 의 단어만 남깁니다.

    word_ids: {"라벨": ["WORDxxxx"]} (word3d_ids.json), vocabulary: assets/glasses/vocabulary.json
    반환 행: {"id", "signer", "glosses", "intervals", "pose", "left", "right", "frame"}
    """
    root = Path(root)
    # 같은 단어가 AI Hub 에 번호 둘로 들어 있으면(예: 없다 WORD1637·WORD1384) aihub_word_extra 로 함께 씁니다.
    gloss_of = {}
    for w in vocabulary["words"]:
        for number in [w.get("aihub_word"), *w.get("aihub_word_extra", [])]:
            if number:
                gloss_of[number] = w["gloss"]
    wanted = {w["gloss"] for w in vocabulary["words"]}
    rows = []
    for path in sorted((root / "word").glob("*/*.npz")):
        number = path.stem.split("_")[2]
        if number in gloss_of and path.stem in word_intervals:
            spans = word_intervals[path.stem]
            rows.append({"id": path.stem, "glosses": [gloss_of[number]],
                         "intervals": [[spans[0][0], spans[-1][1]]], "path": path})
    spans_of = {}
    for manifest in sentence_manifests:
        for clip in json.loads(Path(manifest).read_text(encoding="utf-8"))["clips"]:
            if clip.get("view") == "F":
                spans_of[clip["id"]] = (clip["glosses"], clip["sign_intervals"])
    for path in sorted((root / "sen").glob("*/*.npz")):
        if path.stem in spans_of:
            glosses, spans = spans_of[path.stem]
            rows.append({"id": path.stem, "glosses": list(glosses), "intervals": spans,
                         "path": path})
    out = []
    for row in rows:
        data = np.load(row.pop("path"))
        pose, left, right = to_mm(data["pose"], data["left"], data["right"])
        frame = head_frame(pose)
        if frame is None:
            continue
        keep = [i for i, g in enumerate(row["glosses"]) if g in wanted]
        if not keep and len(row["glosses"]) == 1:
            continue
        out.append({**row, "signer": row["id"].split("_")[3], "frame": frame,
                    "times": data["frame"] / FPS, "left": left, "right": right,
                    "usable": keep})
    return out


def piece(row, start, end):
    """[start, end] 초 구간의 (시각, 왼손, 오른손) 3D. 시각은 0 부터."""
    t = row["times"] - row["times"][0]
    k = (t >= start - 1e-9) & (t <= end + 1e-9)
    if k.sum() < 2:
        return None
    return t[k] - t[k][0], row["left"][k], row["right"][k], row["frame"]


def render(pieces, view, rng=None):
    """이어 붙인 3D 조각들을 한 번의 카메라 설정으로 2D 시퀀스(49개 관절)로 만듭니다."""
    times, coords, masks, offset = [], [], [], 0.0
    for t, left, right, frame in pieces:
        n = len(t)
        c, m = np.zeros((n, JOINTS, 2), np.float32), np.zeros((n, JOINTS), bool)
        for start, hand in ((0, left), (21, right)):
            uv, ok = project(hand, frame, **view)
            ok &= np.isfinite(hand).all(-1)
            if rng is not None:             # 가려진 손가락은 MediaPipe 처럼 짐작한 값으로
                guess = hidden_fingers(hand, frame, reach=view.get("reach", 0.0))
                palm = np.linalg.norm(uv[:, 9] - uv[:, 0], axis=-1)[:, None, None]
                palm = np.where(np.isfinite(palm), palm, 0.05)
                noise = rng.normal(0, 1, uv.shape) * palm * 0.25
                wild = rng.random(guess.shape) < 0.15
                noise[wild] *= 3
                uv = np.where(guess[..., None], uv + noise, uv)
            seen = ok.sum(1) >= 15          # 손 대부분이 화면 안에 있을 때만 검출된 것으로
            c[:, start:start + 21] = np.where((ok & seen[:, None])[..., None], uv, 0)
            m[:, start:start + 21] = ok & seen[:, None]
        times.append(t + offset)
        offset += t[-1] + 1 / FPS
        coords.append(c)
        masks.append(m)
    t, c, m = np.concatenate(times), np.concatenate(coords), np.concatenate(masks)
    if rng is not None:
        t, c, m = jitter(t, c, m, rng)
    if t[-1] - t[0] > MAX_SECONDS:
        t = (t - t[0]) * MAX_SECONDS / (t[-1] - t[0])
    return PoseSequence(t - t[0], c, m, (W, H)).validate()


def jitter(t, c, m, rng):
    """좌표 떨림, 손이 잠깐 안 잡힘, 프레임 수(느린 기기), 속도 차이."""
    c = c + rng.normal(0, 0.003, c.shape).astype(np.float32)
    m = m & ((c >= 0) & (c <= 1)).all(-1)
    if rng.random() < 0.3 and len(t) > 8:
        hand = 21 * int(rng.integers(2))
        first = int(rng.integers(len(t) - 6))
        m[first:first + int(rng.integers(2, 7)), hand:hand + 21] = False
    keep = rng.random(len(t)) < rng.uniform(0.5, 1.0)
    keep[[0, -1]] = True
    t = (t - t[0]) * rng.uniform(0.8, 1.25)
    c = np.where(m[..., None], c, 0).astype(np.float32)
    return t[keep], c[keep], m[keep]


def random_view(rng):
    return {k: float(rng.uniform(*v)) for k, v in VIEW.items()}


class EgoComposer:
    """학습 예시: 상황 문장, 단어 1~3개 조합, 실제 긴급 문장, 수어 아님(준비·복귀 동작).

    phrases: [{"glosses", "text"}] (assets/glasses/phrases.json). 정답 글은
    목록에 있는 단어열이면 그 문장, 없으면 단어를 띄어 쓴 것, 수어 아님은 NO_SIGN 입니다.
    """
    NO_SIGN = "(수어 아님)"

    def __init__(self, rows, phrases, *, negative=0.12, phrase_rate=0.5, margin=0.25):
        self.text = {tuple(p["glosses"]): p["text"] for p in phrases}
        self.phrases = [list(p["glosses"]) for p in phrases]
        self.negative, self.phrase_rate, self.margin = negative, phrase_rate, margin
        self.singles = defaultdict(lambda: defaultdict(list))   # 사람 -> 단어 -> (행, 구간)
        self.rests = defaultdict(list)
        for row in rows:
            spans = row["intervals"]
            for i in row["usable"]:
                self.singles[row["signer"]][row["glosses"][i]].append((row, i))
            total = row["times"][-1] - row["times"][0]
            for a, b in ((0.0, spans[0][0] - 0.15), (spans[-1][1] + 0.15, total)):
                if b - a >= 0.5:
                    self.rests[row["signer"]].append((row, a, b))
        self.signers = sorted(s for s in self.singles if self.singles[s])

    def target(self, glosses):
        if not glosses:
            return self.NO_SIGN
        return self.text.get(tuple(glosses)) or " ".join(glosses) + "."

    def cut(self, row, i, rng):
        spans = row["intervals"]
        start, end = spans[i]
        before = start - spans[i - 1][1] if i else start
        after = spans[i + 1][0] - end if i + 1 < len(spans) else row["times"][-1] - end
        a = rng.uniform(0, min(self.margin, max(before, 0) / 2)) if rng is not None else 0.1
        b = rng.uniform(0, min(self.margin, max(after, 0) / 2)) if rng is not None else 0.1
        return piece(row, start - a, end + b)

    def sample(self, rng, signer=None):
        signer = signer or self.signers[int(rng.integers(len(self.signers)))]
        words = self.singles[signer]
        view = random_view(rng)
        if rng.random() < self.negative and self.rests[signer]:
            row, a, b = self.rests[signer][int(rng.integers(len(self.rests[signer])))]
            part = piece(row, a, b)
            if part is not None:
                return {"glosses": [], "text": self.NO_SIGN, "sequence": render([part], view, rng)}
        glosses = None
        if rng.random() < self.phrase_rate:
            phrase = self.phrases[int(rng.integers(len(self.phrases)))]
            if all(g in words for g in phrase):
                glosses = phrase
        if glosses is None:
            # 클립이 적은 단어(단어 자료에만 있는 단어, 사람당 1개)를 더 자주 뽑습니다.
            names = sorted(words)
            weight = np.array([1 / np.sqrt(len(words[n])) for n in names])
            count = 1 + int(rng.choice(3, p=[0.55, 0.3, 0.15]))
            glosses = [names[int(rng.choice(len(names), p=weight / weight.sum()))]
                       for _ in range(count)]
            glosses = [g for k, g in enumerate(glosses) if k == 0 or g != glosses[k - 1]]
        parts = []
        for g in glosses:
            row, i = words[g][int(rng.integers(len(words[g])))]
            part = self.cut(row, i, rng)
            if part is None:
                return self.sample(rng, signer)
            parts.append(part)
        return {"glosses": glosses, "text": self.target(glosses),
                "sequence": render(parts, view, rng)}
