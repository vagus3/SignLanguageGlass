"""Qwen 을 수어 관절로 미세조정해, 관절 시퀀스를 보고 한국어 문장을 바로 쓰게 합니다.

python -m src.train_qwen --out models/glasses_qwen_v1 --val-signers REAL07,REAL08

구조:  손 관절(장면마다 331개) -> [관절 앞단: 압축 + 양방향 LSTM] -> 4장면씩 평균
       -> [연결층] -> Qwen 이 읽는 형식 -> Qwen2.5 (+LoRA) -> "119를 불러 주세요."

- 관절 앞단은 처음부터 학습합니다. Qwen 은 원래 가중치를 그대로 두고 LoRA(작은 추가
  가중치)만 학습합니다.
- 관절 앞단 위에 단어 판정층(CTC)을 하나 더 두고 같이 학습합니다. 앞단이 빨리 손동작을
  배우게 돕고, 단어 후보와 점수를 따로 꺼낼 수 있게 합니다.
- 학습 예시는 src.ego_synth 가 AI Hub 3D 관절을 안경 시점으로 바꿔 매번 새로 만듭니다.
  팀이 안경으로 찍은 자료(--team, src.collect_glasses 결과)가 있으면 섞습니다 (src.team_data).

미세조정 (이미 학습한 모델에 팀 촬영 자료를 더해 이어 학습):
python -m src.train_qwen --init models/glasses_qwen_v1 --team data/raw/glasses/manifest.json \
    --out models/glasses_qwen_v2 --steps 800 --lr-lora 1e-4 --lr-pose 1e-4 --val-team-signers 홍길동

결과 폴더: adapter/ (LoRA), pose.pt (앞단·연결층·판정층·설정), training.json
"""
import argparse
import glob
import json
import math
from src import clock
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.continuous_sign import CtcRecognizer
from src.data_sources import attribution, source_entry
from src.pose_sequence import FEATURE_SETS, PoseSequence

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "수어를 한국어 문장으로 옮기세요.\n수어:"
ANSWER = "\n문장:"
HEARD = "\n상대가 방금 한 말: {}"


class PoseQwen(nn.Module):
    def __init__(self, llm, glosses, *, feature_set="2d", hidden=128, stride=4):
        super().__init__()
        self.glosses, self.feature_set, self.stride = list(glosses), feature_set, stride
        self.pose = CtcRecognizer(len(self.glosses) + 1, hidden, feature_set)
        width = llm.config.hidden_size
        self.project = nn.Sequential(nn.Linear(2 * hidden, width), nn.GELU(),
                                     nn.Linear(width, width))
        self.llm = llm

    def encode(self, x, lengths):
        """(관절 토큰, 토큰 수, CTC 로그확률 [T, B, C])."""
        h = self.pose.encoder(x)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.pose.temporal(packed)[0], batch_first=True,
                                   total_length=x.shape[1])
        ctc = self.pose.output(h).log_softmax(-1).transpose(0, 1)
        pooled = nn.functional.avg_pool1d(h.transpose(1, 2), self.stride, self.stride,
                                          ceil_mode=True).transpose(1, 2)
        counts = torch.div(lengths + self.stride - 1, self.stride, rounding_mode="floor")
        return self.project(pooled), counts, ctc

    def embed(self, ids):
        return self.llm.get_input_embeddings()(ids)


def features(sequence, feature_set="2d"):
    return torch.from_numpy(FEATURE_SETS[feature_set][0](sequence))


def batch_inputs(model, tokenizer, items, device, *, with_target=True):
    """관절 토큰을 프롬프트 사이에 끼워 넣은 입력 (오른쪽 채움). 정답 글자만 손실에 씁니다."""
    xs = [features(i["sequence"], model.feature_set) for i in items]
    lengths = torch.tensor([len(x) for x in xs])
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True).to(device)
    tokens, counts, ctc = model.encode(x, lengths.to(device))
    rows, labels = [], []
    for k, item in enumerate(items):
        prompt = PROMPT if not item.get("heard") else HEARD.format(item["heard"]) + "\n" + PROMPT
        before = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
        after = tokenizer(ANSWER, return_tensors="pt").input_ids[0].to(device)
        parts = [model.embed(before), tokens[k, :counts[k]], model.embed(after)]
        label = [torch.full((len(before) + int(counts[k]) + len(after),), -100, device=device)]
        if with_target:
            target = tokenizer(" " + item["text"] + tokenizer.eos_token,
                               return_tensors="pt").input_ids[0].to(device)
            parts.append(model.embed(target))
            label.append(target)
        rows.append(torch.cat(parts))
        labels.append(torch.cat(label))
    width = max(len(r) for r in rows)
    pad = model.embed(torch.tensor([tokenizer.pad_token_id], device=device))[0]
    if with_target:      # 학습: 오른쪽 채움
        embeds = torch.stack([torch.cat([r, pad.expand(width - len(r), -1)]) for r in rows])
        mask = torch.stack([torch.arange(width, device=device) < len(r) for r in rows]).long()
        label = torch.stack([torch.cat([l, torch.full((width - len(l),), -100, device=device)])
                             for l in labels])
    else:                # 생성: 왼쪽 채움 (마지막 위치에서 이어 쓰므로)
        embeds = torch.stack([torch.cat([pad.expand(width - len(r), -1), r]) for r in rows])
        mask = torch.stack([torch.arange(width, device=device) >= width - len(r)
                            for r in rows]).long()
        label = None
    return embeds, mask, label, ctc, lengths


def ctc_targets(model, items):
    index = {g: i + 1 for i, g in enumerate(model.glosses)}
    ids = [[index[g] for g in i["glosses"] if g in index] for i in items]
    flat = torch.tensor([t for row in ids for t in row], dtype=torch.long)
    return flat, torch.tensor([len(row) for row in ids], dtype=torch.long)


@torch.inference_mode()
def generate(model, tokenizer, items, device, max_new_tokens=24):
    """[(문장, 평균 글자 확률, CTC 단어열)]."""
    model.eval()
    embeds, mask, _, ctc, lengths = batch_inputs(model, tokenizer, items, device,
                                                 with_target=False)
    # transformers 의 generate 는 맥 GPU(MPS)에서 inputs_embeds 와 함께 쓰면 위치 계산이
    # 틀려 엉뚱한 글을 냅니다 (같은 입력의 forward 는 CPU 와 일치). 짧은 문장만 쓰므로
    # 위치를 직접 주며 한 글자씩 다시 계산합니다.
    eos = tokenizer.eos_token_id
    finished = torch.zeros(len(items), dtype=torch.bool, device=device)
    picked, logps = [], []
    for _ in range(max_new_tokens):
        position = (mask.cpu().cumsum(1) - 1).clamp(min=0).to(device)
        # 마지막 위치의 확률만 필요하므로 logits_to_keep=1 (전체 위치 x 어휘 15만 개를 만들면
        # 배치 16개에서 메모리 수 GB 를 씁니다)
        logits = model.llm(inputs_embeds=embeds, attention_mask=mask, position_ids=position,
                           logits_to_keep=1).logits[:, -1].float()
        logp = logits.log_softmax(-1)
        token = logp.argmax(-1)
        token = torch.where(finished, torch.full_like(token, eos), token)
        picked.append(token)
        logps.append(torch.where(finished, torch.zeros_like(token, dtype=logp.dtype),
                                 logp.gather(-1, token[:, None])[:, 0]))
        counted = ~finished
        finished = finished | (token == eos)
        logps[-1] = logps[-1] * counted
        if bool(finished.all()):
            break
        embeds = torch.cat([embeds, model.embed(token[:, None])], 1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
    sequences = torch.stack(picked, 1).cpu()
    steps = (sequences != eos).cumsum(1).max(1).values + 1      # eos 까지의 글자 수
    total = torch.stack(logps, 1).sum(1).cpu()
    scores = (total / steps.clamp(min=1)).exp().tolist()
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    results = []
    for k, text in enumerate(texts):
        path = ctc[:lengths[k], k].argmax(-1).tolist()
        words = [model.glosses[t - 1] for i, t in enumerate(path)
                 if t and (i == 0 or t != path[i - 1])]
        results.append((text.strip(), float(scores[k]), words))
    return results


def evaluate(model, tokenizer, items, device, batch=16):
    rows = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        for item, (text, score, words) in zip(chunk, generate(model, tokenizer, chunk, device)):
            rows.append({"reference": item["text"], "glosses": item["glosses"],
                         "output": text, "score": score, "ctc": words})

    def rate(subset, key):
        subset = [r for r in rows if subset(r)]
        return (float(np.mean([key(r) for r in subset])), len(subset)) if subset else (None, 0)
    phrase = lambda r: r["reference"] in model.phrase_texts       # noqa: E731
    report = {"exact": rate(lambda r: True, lambda r: r["output"] == r["reference"]),
              "phrase_exact": rate(phrase, lambda r: r["output"] == r["reference"]),
              "other_exact": rate(lambda r: r["glosses"] and not phrase(r),
                                  lambda r: r["output"] == r["reference"]),
              "no_sign_correct": rate(lambda r: not r["glosses"],
                                      lambda r: r["output"] == r["reference"]),
              "ctc_exact": rate(lambda r: True, lambda r: r["ctc"] == r["glosses"])}
    return report, rows


def sentence_manifests():
    """문장 구간 파일: 학습용 묶음들 + 검증 수어자(REAL17, REAL18) 문장."""
    return sorted(glob.glob(str(ROOT / "data/processed/aihub_sentences/*/manifest.json"))
                  + glob.glob(str(ROOT / "data/processed/aihub_sentences_val/manifest.json")))


def fixed_items(composer, signers, count, seed=1234):
    rng = np.random.default_rng(seed)
    return [composer.sample(rng, signers[i % len(signers)]) for i in range(count)]


def pretrain_pose(model, composer, *, steps, batch, device, val_items=(), log=print, seed=0,
                  lr=2e-3):
    """1단계: 관절 앞단만 단어 판정(CTC)으로 학습합니다. Qwen 없이 돌아 빠릅니다.

    빈 상태의 앞단을 처음부터 Qwen 과 함께 학습하면 손동작을 배우기 전에 Qwen 이 정답 문장의
    빈도만 외웁니다. 앞단이 먼저 손동작을 구분하게 만든 뒤 2단계에서 Qwen 에 연결합니다.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    pose = model.pose
    optimizer = torch.optim.AdamW(pose.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    started = clock.now()
    for step in range(1, steps + 1):
        pose.train()
        items = [composer.sample(rng) for _ in range(batch)]
        xs = [features(i["sequence"], model.feature_set) for i in items]
        lengths = torch.tensor([len(x) for x in xs])
        x = nn.utils.rnn.pad_sequence(xs, batch_first=True).to(device)
        flat, target_lengths = ctc_targets(model, items)
        loss = loss_fn(pose(x, lengths.to(device)).float().cpu(), flat, lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pose.parameters(), 5)
        optimizer.step()
        schedule.step()
        if step % 100 == 0 or step == steps:
            line = f"pose step={step} ctc={float(loss):.3f} {(clock.now() - started) / step:.2f}s/step"
            if val_items and (step % 500 == 0 or step == steps):
                line += f" 확인 단어 정확도={pose_accuracy(model, val_items, device):.3f}"
            log(line)


@torch.inference_mode()
def pose_accuracy(model, items, device, batch=32):
    model.pose.eval()
    right = 0
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        xs = [features(i["sequence"], model.feature_set) for i in chunk]
        lengths = torch.tensor([len(x) for x in xs])
        out = model.pose(nn.utils.rnn.pad_sequence(xs, batch_first=True).to(device),
                         lengths.to(device)).cpu()
        for k, item in enumerate(chunk):
            path = out[:lengths[k], k].argmax(-1).tolist()
            words = [model.glosses[t - 1] for i, t in enumerate(path)
                     if t and (i == 0 or t != path[i - 1])]
            right += words == item["glosses"]
    return right / max(len(items), 1)


def train(model, tokenizer, composer, *, steps, batch, device, val_items=(), log=print,
          eval_every=200, seed=0, lr_lora=2e-4, lr_pose=3e-4, ctc_weight=0.3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lora = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    pose = list(model.pose.parameters()) + list(model.project.parameters())
    optimizer = torch.optim.AdamW([{"params": lora, "lr": lr_lora},
                                   {"params": pose, "lr": lr_pose}], weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[lr_lora, lr_pose],
                                                   total_steps=steps, pct_start=0.1)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    history, best, best_state, started = [], None, None, clock.now()
    for step in range(1, steps + 1):
        model.train()
        items = [composer.sample(rng) for _ in range(batch)]
        embeds, mask, labels, ctc, lengths = batch_inputs(model, tokenizer, items, device)
        # 정답 글자 위치의 어휘 확률만 계산합니다. llm(labels=...) 는 모든 위치 x 어휘 15만 개의
        # 확률을 만들어 메모리를 수 GB 쓰고 단계마다 모양이 달라 MPS 메모리가 계속 불어납니다
        # (같은 손실: 정답 글자 평균 교차 엔트로피).
        base = model.llm.get_base_model() if hasattr(model.llm, "get_base_model") else model.llm
        hidden = base.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        keep = labels[:, 1:] != -100
        logits = base.lm_head(hidden[:, :-1][keep]).float()
        lm = nn.functional.cross_entropy(logits, labels[:, 1:][keep])
        flat, target_lengths = ctc_targets(model, items)
        aux = ctc_loss(ctc.float().cpu(), flat, lengths, target_lengths)
        loss = lm + ctc_weight * aux.to(lm.device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora + pose, 1.0)
        optimizer.step()
        schedule.step()
        if step % 20 == 0 and str(device).startswith("mps"):
            torch.mps.empty_cache()      # 단계마다 입력 길이가 달라 맥 GPU 가 메모리를 쌓아 둡니다
        if step % 25 == 0 or step == 1:
            log(f"step={step} loss={float(lm):.3f} ctc={float(aux):.3f} "
                f"{(clock.now() - started) / step:.2f}s/step")
        if val_items and (step % eval_every == 0 or step == steps):
            report, _ = evaluate(model, tokenizer, val_items, device)
            entry = {"step": step, **{k: v[0] for k, v in report.items()}}
            history.append(entry)
            log("  확인: " + " ".join(f"{k}={v:.3f}" for k, v in entry.items()
                                     if isinstance(v, float)))
            key = entry["exact"] or 0
            if best is None or key > best["exact"]:
                best = entry
                best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                              if p.requires_grad}
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return history, best


def build(glosses, base=BASE, device="cpu", lora_rank=16, local_only=False):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=local_only)
    llm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, attn_implementation="eager",
                                               local_files_only=local_only)
    llm = get_peft_model(llm, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                                         lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj",
                                                         "o_proj"]))
    return PoseQwen(llm, glosses).to(device), tokenizer


def build_from(path, device="cpu", local_only=True, glosses=None):
    """이미 학습한 모델을 이어 학습할 수 있게 불러옵니다 (LoRA 학습 가능 상태).

    glosses 가 저장된 단어 목록과 다르면(단어 추가) 단어 판정층만 새로 만들고, 관절 앞단·
    연결층·LoRA 는 그대로 이어 씁니다. 이때는 1단계로 새 판정층을 먼저 맞춰야 합니다."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(path)
    spec = torch.load(path / "pose.pt", map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["base"], local_files_only=local_only)
    llm = AutoModelForCausalLM.from_pretrained(spec["base"], torch_dtype=torch.float32,
                                               attn_implementation="eager",
                                               local_files_only=local_only)
    llm = PeftModel.from_pretrained(llm, path / "adapter", is_trainable=True)
    model = PoseQwen(llm, glosses or spec["glosses"], feature_set=spec["feature_set"],
                     hidden=spec["hidden"], stride=spec["stride"])
    own = model.pose.state_dict()
    model.pose.load_state_dict({k: v for k, v in spec["pose"].items()
                                if k in own and own[k].shape == v.shape}, strict=False)
    model.project.load_state_dict(spec["project"])
    return model.to(device), tokenizer


def save(model, tokenizer, out, base, metadata):
    out = Path(out)
    out.mkdir(parents=True)
    model.llm.save_pretrained(out / "adapter")
    torch.save({"base": base, "glosses": model.glosses, "feature_set": model.feature_set,
                "stride": model.stride, "hidden": model.pose.temporal.hidden_size,
                "pose": model.pose.state_dict(), "project": model.project.state_dict(),
                "phrase_texts": sorted(model.phrase_texts), "metadata": metadata},
               out / "pose.pt")


def load(path, device="cpu", local_only=True, attention="eager", merge=False):
    """학습한 모델을 불러옵니다. merge=True 면 LoRA 를 Qwen 가중치에 합쳐 실행을 빠르게 합니다
    (더 학습하지 않을 때만)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(path)
    spec = torch.load(path / "pose.pt", map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["base"], local_files_only=local_only)
    llm = AutoModelForCausalLM.from_pretrained(spec["base"], torch_dtype=torch.float32, attn_implementation=attention,
                                               local_files_only=local_only)
    llm = PeftModel.from_pretrained(llm, path / "adapter")
    if merge:
        llm = llm.merge_and_unload()
    model = PoseQwen(llm, spec["glosses"], feature_set=spec["feature_set"],
                     hidden=spec["hidden"], stride=spec["stride"])
    model.pose.load_state_dict(spec["pose"])
    model.project.load_state_dict(spec["project"])
    model.phrase_texts = set(spec["phrase_texts"])
    return model.to(device).eval(), tokenizer, spec


def main():
    from src.ego_synth import EgoComposer, load_clips
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default=BASE)
    p.add_argument("--val-signers", default="", help="학습에서 뺄 수어자 (예: REAL07,REAL08)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--val-count", type=int, default=160)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pretrain-steps", type=int, default=8000,
                   help="1단계: 관절 앞단만 단어 판정으로 학습하는 횟수 (0 이면 건너뜀)")
    p.add_argument("--pretrain-batch", type=int, default=32)
    p.add_argument("--team", nargs="*", default=[], help="팀 안경 촬영 manifest.json (collect_glasses)")
    p.add_argument("--team-rate", type=float, default=0.5, help="학습 예시 중 팀 자료 비율")
    p.add_argument("--val-team-signers", default="",
                   help="팀 촬영 중 학습에서 뺄 사람 (처음 보는 사람 성능 확인용)")
    p.add_argument("--init", help="이어 학습할 모델 폴더 (앞단·연결층·LoRA 를 불러오고 1단계는 건너뜀)")
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-pose", type=float, default=3e-4)
    args = p.parse_args()
    if Path(args.out).exists():
        p.error("결과 폴더가 이미 있습니다")
    vocabulary = json.loads((ROOT / "assets/glasses/vocabulary.json").read_text(encoding="utf-8"))
    phrases = json.loads((ROOT / "assets/glasses/phrases.json").read_text(encoding="utf-8"))
    rows = load_clips(ROOT / "data/raw/aihub103_3d",
                      json.loads((ROOT / "data/processed/word3d_ids.json").read_text()),
                      json.loads((ROOT / "data/processed/word3d_intervals.json").read_text()),
                      sentence_manifests(),
                      vocabulary)
    held = {s for s in args.val_signers.split(",") if s}
    train_rows = [r for r in rows if r["signer"] not in held]
    val_rows = [r for r in rows if r["signer"] in held]
    composer = EgoComposer(train_rows, phrases)
    glosses = [w["gloss"] for w in vocabulary["words"]]
    team_val = None
    if args.team:
        from src.team_data import MixedComposer, TeamComposer, load_team
        team_rows = load_team(args.team, vocabulary)
        held_team = {s for s in args.val_team_signers.split(",") if s}
        team = TeamComposer([r for r in team_rows if r["signer"] not in held_team], phrases)
        if held_team:
            team_val = TeamComposer([r for r in team_rows if r["signer"] in held_team], phrases)
        composer = MixedComposer(team, composer, args.team_rate)
        print(f"팀 촬영 {len(team_rows)}클립 / {len(team.signers)}명 섞음 (비율 {args.team_rate}), "
              f"확인용 팀원 {sorted(held_team)}")
    if args.init:
        old = torch.load(Path(args.init) / "pose.pt", map_location="cpu",
                         weights_only=True)["glosses"]
        model, tokenizer = build_from(args.init, args.device, glosses=glosses)
        if old == glosses:
            args.pretrain_steps = 0
        else:       # 단어가 늘었으면 새 판정층을 1단계로 먼저 맞춥니다 (앞단은 이어 씀)
            print(f"단어 목록 변경 {len(old)} -> {len(glosses)}개: 1단계 {args.pretrain_steps}회 후 이어 학습")
    else:
        model, tokenizer = build(glosses, args.base, args.device)
    model.phrase_texts = {ph["text"] for ph in phrases}
    print(f"학습 {len(train_rows)}클립 / 수어자 {len(composer.signers)}명, "
          f"확인 수어자 {sorted(held)}, 단어 {len(glosses)}개, 장치 {args.device}")
    val_items = []
    if team_val is not None:      # 실제 안경 카메라 자료로 확인하는 것이 우선입니다
        val_items = fixed_items(team_val, team_val.signers, args.val_count)
    elif val_rows:
        val = EgoComposer(val_rows, phrases)
        val_items = fixed_items(val, val.signers, args.val_count)
    if args.pretrain_steps:      # 작은 LSTM 은 맥 GPU 보다 CPU 가 빠릅니다 (0.09 대 0.5초/단계)
        model.pose.to("cpu")
        pretrain_pose(model, composer, steps=args.pretrain_steps, batch=args.pretrain_batch,
                      device="cpu", val_items=val_items, seed=args.seed)
        model.pose.to(args.device)
    history, best = train(model, tokenizer, composer, steps=args.steps, batch=args.batch,
                          device=args.device, val_items=val_items, eval_every=args.eval_every,
                          seed=args.seed, lr_lora=args.lr_lora, lr_pose=args.lr_pose)
    sources = [source_entry("aihub-sign-video")] + ([source_entry("team")] if args.team else [])
    metadata = {"train_signers": composer.signers, "val_signers": sorted(held),
                "steps": args.steps, "batch": args.batch, "seed": args.seed,
                "pretrain_steps": args.pretrain_steps, "init": args.init,
                "team_manifests": [str(t) for t in args.team], "team_rate": args.team_rate,
                "viewpoint": "glasses-synthetic-from-aihub-3d", "data_sources": sources}
    save(model, tokenizer, args.out, args.base, metadata)
    (Path(args.out) / "training.json").write_text(json.dumps(
        {**metadata, "history": history, "validation": best}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    for line in attribution(sources):
        print("데이터 출처:", line)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()


def candidate_texts(model, phrases, vocabulary_words):
    """고를 수 있는 답: 상황 문장 + 문장이 없는 단어 하나("지갑.") + 수어 아님. {글: 단어열}."""
    from src.ego_synth import EgoComposer
    options = {p["text"]: list(p["glosses"]) for p in phrases}
    single = {tuple(p["glosses"]) for p in phrases if len(p["glosses"]) == 1}
    for word in vocabulary_words:
        if (word,) not in single:
            options[word + "."] = [word]
    options[EgoComposer.NO_SIGN] = []
    return options


@torch.inference_mode()
def rank(model, tokenizer, sequence, options, device, heard=None, shortlist=8, with_score=False):
    """각 답이 이 관절 시퀀스에서 나올 확률 (서로 비교한 비율)과 단어 판정층의 단어열.

    문장을 자유롭게 쓰게 하지 않고 목록 안에서만 고르므로, 하지 않은 말을 지어내지 않습니다.
    질문+관절 부분은 모든 답에 공통이므로 한 번만 계산해 두고(KV 캐시) 답 글자만 이어 계산합니다.
    shortlist: 단어 판정층이 보기에 그럴듯한 답 몇 개만 Qwen 이 비교합니다 (빠르게).
    """
    from src.ctc_runtime import sequence_log_likelihood
    model.eval()
    prefix, mask, _, ctc, lengths = batch_inputs(
        model, tokenizer, [{"sequence": sequence, "text": "", "glosses": [], "heard": heard}],
        device, with_target=False)
    log_probs = ctc[:lengths[0], 0].float().cpu().numpy()
    index = {g: i + 1 for i, g in enumerate(model.glosses)}
    texts = list(options)
    if shortlist and len(texts) > shortlist:
        prior = {t: sequence_log_likelihood(log_probs, [index[g] for g in options[t] if g in index])
                 for t in texts}
        texts = sorted(texts, key=lambda t: -prior[t])[:shortlist]
    out = model.llm(inputs_embeds=prefix, attention_mask=mask, use_cache=True)
    first = out.logits[0, -1].float().log_softmax(-1)
    del out.logits
    ids = [tokenizer(" " + t + tokenizer.eos_token).input_ids for t in texts]
    width, n = max(len(i) for i in ids), len(ids)
    tokens = torch.full((n, width), tokenizer.pad_token_id, dtype=torch.long)
    real = torch.zeros((n, width), dtype=torch.bool)
    for k, row in enumerate(ids):
        tokens[k, :len(row)], real[k, :len(row)] = torch.tensor(row), True
    tokens, real = tokens.to(device), real.to(device)
    cache = out.past_key_values
    cache.batch_repeat_interleave(n)
    start = prefix.shape[1]
    position = torch.arange(start, start + width, device=device)[None].expand(n, -1)
    attention = torch.cat([torch.ones((n, start), dtype=torch.long, device=device),
                           real.long()], 1)
    logits = model.llm(inputs_embeds=model.embed(tokens), attention_mask=attention,
                       position_ids=position, past_key_values=cache).logits[:, :-1]
    total = first[tokens[:, 0]]
    if width > 1:       # log_softmax 전체를 만들지 않고 필요한 글자만: 고른 값 - logsumexp
        chosen = logits.gather(-1, tokens[:, 1:, None])[..., 0].float()
        later = chosen - torch.logsumexp(logits.float(), -1)
        total = total + (later * real[:, 1:]).sum(1)
    posterior = total.softmax(0).cpu().tolist()
    path = log_probs.argmax(-1).tolist()
    words = [model.glosses[t - 1] for i, t in enumerate(path)
             if t and (i == 0 or t != path[i - 1])]
    if with_score:      # 단어 판정층 점수: 본 단어 중 가장 자신 없는 단어의 확률
        best = [float(np.exp(log_probs[i, t])) for i, t in enumerate(path)
                if t and (i == 0 or t != path[i - 1])]
        return dict(zip(texts, posterior)), words, (min(best) if best else 0.0)
    return dict(zip(texts, posterior)), words
