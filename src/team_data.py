"""팀이 안경 카메라로 찍은 자료(src.collect_glasses 결과)로 학습 예시를 만듭니다.

AI Hub 를 안경 시점으로 바꾼 가짜 자료(src.ego_synth)와 같은 모양의 예시
{"glosses", "text", "sequence"} 를 돌려주므로, 둘을 섞어 미세조정할 수 있습니다.

- 찍은 클립은 실제 MediaPipe Hands 결과라 손 떨림, 놓침, 가려진 손가락이 이미 들어 있습니다.
  여기서는 작은 회전·크기·위치·속도 변화와 프레임 솎기만 더합니다.
- 정답 글은 ego_synth 와 같은 규칙입니다: 상황 문장 목록에 있으면 그 문장, 아니면 단어를
  띄어 쓴 것, 수어 아님(negative)은 "(수어 아님)".
- 단어 하나짜리 클립을 같은 사람끼리 이어 붙여 상황 문장 예시도 만듭니다.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.ego_synth import EgoComposer, FPS
from src.pose_sequence import MAX_SECONDS, PoseSequence


def load_team(manifests, vocabulary=None):
    """collect_glasses manifest 들 -> 행 목록. vocabulary 에 없는 단어가 든 클립은 뺍니다."""
    wanted = {w["gloss"] for w in vocabulary["words"]} if vocabulary else None
    rows = []
    for manifest in manifests:
        path = Path(manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("extractor", {}).get("name") != "mediapipe-hands":
            raise ValueError(f"{path}: 안경 촬영 도구(collect_glasses)로 찍은 자료가 아닙니다")
        for clip in data["clips"]:
            glosses = list(clip.get("glosses") or [])
            if wanted is not None and any(g not in wanted for g in glosses):
                continue
            rows.append({"id": clip["id"], "signer": clip["signer_id"], "glosses": glosses,
                         "sequence": PoseSequence.load(path.parent / clip["file"])})
    return rows


def _jitter(sequence, rng):
    """실제 카메라 자료에 더하는 가벼운 변형: 회전 ±5도, 크기 ±10%, 위치 ±4%, 속도, 프레임 수."""
    t, p, m = sequence.timestamps, sequence.coordinates.astype(np.float64), sequence.mask.copy()
    angle = np.deg2rad(rng.uniform(-5, 5))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([0.5, 0.5])
    p = (p - center) @ rot.T * rng.uniform(0.9, 1.1) + center + rng.uniform(-0.04, 0.04, 2)
    m &= ((p >= 0) & (p <= 1)).all(-1)
    keep = rng.random(len(t)) < rng.uniform(0.6, 1.0)
    keep[[0, -1]] = True
    t = (t - t[0]) * rng.uniform(0.85, 1.2)
    if t[-1] > MAX_SECONDS:
        t *= MAX_SECONDS / t[-1]
    p = np.where(m[..., None], p, 0).astype(np.float32)
    return PoseSequence(t[keep], p[keep], m[keep], sequence.image_size).validate()


def _join(sequences):
    times, coords, masks, offset = [], [], [], 0.0
    for s in sequences:
        t = s.timestamps - s.timestamps[0]
        times.append(t + offset)
        offset += t[-1] + 1 / FPS
        coords.append(s.coordinates)
        masks.append(s.mask)
    t = np.concatenate(times)
    if t[-1] > MAX_SECONDS:
        t *= MAX_SECONDS / t[-1]
    return PoseSequence(t, np.concatenate(coords), np.concatenate(masks),
                        sequences[0].image_size).validate()


class TeamComposer:
    """팀 촬영 클립으로 예시를 만듭니다. 단어마다 고르게 뽑고, 가끔 상황 문장으로 잇습니다."""

    def __init__(self, rows, phrases, *, phrase_rate=0.3):
        self.text = {tuple(p["glosses"]): p["text"] for p in phrases}
        self.phrases = [list(p["glosses"]) for p in phrases if len(p["glosses"]) > 1]
        self.phrase_rate = phrase_rate
        self.by_key = defaultdict(list)                   # 단어열 -> 클립들
        self.words = defaultdict(lambda: defaultdict(list))   # 사람 -> 단어 -> 클립들
        for row in rows:
            self.by_key[tuple(row["glosses"])].append(row)
            if len(row["glosses"]) == 1:
                self.words[row["signer"]][row["glosses"][0]].append(row)
        self.keys = sorted(self.by_key)
        self.signers = sorted({r["signer"] for r in rows})
        if not self.keys:
            raise ValueError("팀 촬영 자료가 비어 있습니다")

    def target(self, glosses):
        """ego_synth.EgoComposer 와 같은 정답 규칙."""
        if not glosses:
            return EgoComposer.NO_SIGN
        return self.text.get(tuple(glosses)) or " ".join(glosses) + "."

    def sample(self, rng, signer=None):
        if self.phrases and rng.random() < self.phrase_rate:
            person = signer or self.signers[int(rng.integers(len(self.signers)))]
            phrase = self.phrases[int(rng.integers(len(self.phrases)))]
            have = self.words[person]
            if all(g in have for g in phrase):
                parts = [_jitter(have[g][int(rng.integers(len(have[g])))]["sequence"], rng)
                         for g in phrase]
                return {"glosses": phrase, "text": self.target(phrase), "sequence": _join(parts)}
        candidates = self.keys
        if signer is not None:
            candidates = [k for k in self.keys if any(r["signer"] == signer
                                                      for r in self.by_key[k])] or self.keys
        key = candidates[int(rng.integers(len(candidates)))]
        pool = [r for r in self.by_key[key] if signer is None or r["signer"] == signer] \
            or self.by_key[key]
        row = pool[int(rng.integers(len(pool)))]
        return {"glosses": list(key), "text": self.target(list(key)),
                "sequence": _jitter(row["sequence"], rng)}


class MixedComposer:
    """팀 자료와 AI Hub 가짜 안경 자료를 섞습니다. 팀 자료만 쓰면 팀원 손에만 맞춰집니다."""

    def __init__(self, team, aihub, team_rate=0.5):
        self.team, self.aihub, self.team_rate = team, aihub, team_rate
        self.signers = list(getattr(aihub, "signers", [])) + list(team.signers)

    def sample(self, rng, signer=None):
        if signer is not None:
            source = self.team if signer in self.team.signers else self.aihub
            return source.sample(rng, signer)
        return (self.team if rng.random() < self.team_rate else self.aihub).sample(rng)
