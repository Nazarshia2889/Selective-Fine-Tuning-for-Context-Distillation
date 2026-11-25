"""Task accuracy (with/without context), recovery fraction, and MMLU forgetting."""
from __future__ import annotations

import random

import torch

from sfcd.config import EvalCfg, ModelCfg
from sfcd.model import autocast, build_prompt, generate
from sfcd.tasks.base import Example, Task


def eval_task(model, tok, task: Task, examples: list[Example], context_text: str | None,
              eval_cfg: EvalCfg, model_cfg: ModelCfg) -> tuple[float, list[str]]:
    prompts = [build_prompt(tok, task, context_text, e.query) for e in examples]
    preds = generate(model, tok, prompts, task.max_new_tokens, eval_cfg.gen_batch_size, model_cfg)
    return task.accuracy(preds, examples), preds


def recovery(student: float, zero_shot: float, icl: float, trivial: float = 0.0) -> float | None:
    """Fraction of the ICL improvement recovered without the context, measured from the better of the
    zero-shot score and the trivial baseline (e.g. always answering the majority label)."""
    floor = max(zero_shot, trivial)
    gap = icl - floor
    return None if abs(gap) < 1e-9 else (student - floor) / gap


class _MMLUTask(Task):
    name = "mmlu"
    system_prompt = "You are a helpful assistant. Answer with the letter of the correct option only."

    def build(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def is_correct(self, prediction, example):  # pragma: no cover
        raise NotImplementedError


def load_mmlu_subset(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [ds[i] for i in idx]


MMLU_VERSION = 2  # v2: option-text scoring (v1: first-token A/B/C/D logits)
LETTERS = ["A", "B", "C", "D"]


class _ClozeTask(_MMLUTask):
    system_prompt = "You are a helpful assistant. Answer the question."


def _letter_prompts(tok, items: list[dict]) -> list[str]:
    task = _MMLUTask()
    out = []
    for it in items:
        opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, it["choices"]))
        out.append(build_prompt(tok, task, None, f"{it['question']}\n\n{opts}\n\nAnswer:"))
    return out


@torch.no_grad()
def option_text_scores(model, tok, items: list[dict], model_cfg: ModelCfg, batch_seqs: int = 8) -> list[list[float]]:
    """Mean log-prob per token of each option's text as the answer (question shown without options).
    Sequences are right-padded: in a causal model, padding after a sequence can't change the scores of
    the tokens before it, so batching doesn't change the result."""
    device = next(model.parameters()).device
    task = _ClozeTask()
    was_training = model.training
    model.eval()
    pad = getattr(tok, "pad_token_id", None) or 0
    seqs = []  # (item, choice, prompt len, option len, ids)
    for i, it in enumerate(items):
        p = tok(build_prompt(tok, task, None, f"{it['question']}\n\nAnswer:"), add_special_tokens=False)["input_ids"]
        for ci, c in enumerate(it["choices"]):
            o = tok(" " + str(c).strip(), add_special_tokens=False)["input_ids"]
            seqs.append((i, ci, len(p), len(o), p + o))
    scores = [[0.0] * len(it["choices"]) for it in items]
    bs = batch_seqs  # full-vocab logits are large (248k vocab), so keep batches small
    for s in range(0, len(seqs), bs):
        chunk = seqs[s:s + bs]
        L = max(len(x[4]) for x in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x[4])] = torch.tensor(x[4])
            att[j, :len(x[4])] = 1
        ids, att = ids.to(device), att.to(device)
        with autocast(model_cfg, device):
            logits = model(input_ids=ids, attention_mask=att).logits
        for j, (i, ci, lp_, lo, _) in enumerate(chunk):
            lg = logits[j, lp_ - 1:lp_ - 1 + lo].float()  # positions that predict the option tokens
            tgt = ids[j, lp_:lp_ + lo]
            scores[i][ci] = float(lg.log_softmax(-1).gather(-1, tgt[:, None]).mean())
        del logits
    model.train(was_training)
    return scores


@torch.no_grad()
def letter_preds(model, tok, items: list[dict], eval_cfg: EvalCfg, model_cfg: ModelCfg) -> list[int]:
    """v1 scoring: argmax over A/B/C/D logits at the first answer token (sensitive to answer format)."""
    device = next(model.parameters()).device
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    prompts = _letter_prompts(tok, items)
    was_training = model.training
    model.eval()
    tok.padding_side = "left"
    preds = []
    bs = eval_cfg.mmlu_batch_size
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with autocast(model_cfg, device):
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        preds += logits[:, letter_ids].argmax(-1).tolist()
    model.train(was_training)
    return preds


def mmlu_eval(model, tok, items: list[dict], eval_cfg: EvalCfg, model_cfg: ModelCfg) -> dict:
    """acc = option-text accuracy (knowledge, robust to answer-format changes); acc_letter = v1 score."""
    if not items:
        return {"acc": None, "acc_letter": None, "preds": [], "version": MMLU_VERSION}
    gold = [it["answer"] for it in items]
    ot = option_text_scores(model, tok, items, model_cfg)
    preds = [max(range(len(r)), key=r.__getitem__) for r in ot]
    lp = letter_preds(model, tok, items, eval_cfg, model_cfg)
    n = len(items)
    return {"acc": sum(a == g for a, g in zip(preds, gold)) / n,
            "acc_letter": sum(a == g for a, g in zip(lp, gold)) / n,
            "preds": preds, "version": MMLU_VERSION}


def eval_mmlu(model, tok, items: list[dict], eval_cfg: EvalCfg, model_cfg: ModelCfg,
              return_preds: bool = False):
    """MMLU accuracy (optionally with per-question predicted letter indices)."""
    r = mmlu_eval(model, tok, items, eval_cfg, model_cfg)
    return (r["acc"], r["preds"]) if return_preds else r["acc"]
