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


def recovery(student: float, zero_shot: float, icl: float) -> float | None:
    """Fraction of the ICL improvement recovered without the context."""
    gap = icl - zero_shot
    return None if abs(gap) < 1e-9 else (student - zero_shot) / gap


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


@torch.no_grad()
def mmlu_eval(model, tok, items: list[dict], eval_cfg: EvalCfg, model_cfg: ModelCfg) -> dict:
    """Next-token letter scoring after the (non-thinking) assistant prefix: argmax over A/B/C/D."""
    if not items:
        return {"acc": None, "preds": []}
    task = _MMLUTask()
    device = next(model.parameters()).device
    letters = ["A", "B", "C", "D"]
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
    prompts = []
    for it in items:
        opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, it["choices"]))
        prompts.append(build_prompt(tok, task, None, f"{it['question']}\n\n{opts}\n\nAnswer:"))
    was_training = model.training
    model.eval()
    tok.padding_side = "left"
    logps = []
    bs = eval_cfg.mmlu_batch_size
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with autocast(model_cfg, device):
            logits = model(**enc).logits[:, -1, :]
        logps.append(torch.log_softmax(logits[:, letter_ids].float(), dim=-1).cpu())
    model.train(was_training)
    lp = torch.cat(logps)
    gold = torch.tensor([it["answer"] for it in items])
    preds = lp.argmax(-1)
    return {"acc": (preds == gold).float().mean().item(), "preds": preds.tolist()}


def eval_mmlu(model, tok, items: list[dict], eval_cfg: EvalCfg, model_cfg: ModelCfg,
              return_preds: bool = False):
    """MMLU accuracy (optionally with per-question predicted letter indices)."""
    r = mmlu_eval(model, tok, items, eval_cfg, model_cfg)
    return (r["acc"], r["preds"]) if return_preds else r["acc"]
