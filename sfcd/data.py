"""SFT data: student prompt (no context) + teacher response, loss only on response tokens."""
from __future__ import annotations

import random

import torch

from sfcd.model import build_prompt
from sfcd.tasks.base import Task


def make_sft_examples(tok, task: Task, records: list[dict], max_len: int) -> list[dict]:
    """records: [{"query":..., "teacher_response":...}] -> tokenized examples."""
    out = []
    eos = tok.eos_token or ""
    for r in records:
        p = tok(build_prompt(tok, task, None, r["query"]), add_special_tokens=False)["input_ids"]
        a = tok(r["teacher_response"] + eos, add_special_tokens=False)["input_ids"]
        ids = (p + a)[:max_len]
        labels = ([-100] * len(p) + a)[:max_len]
        if any(l != -100 for l in labels):
            out.append({"input_ids": ids, "labels": labels})
    return out


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def batches(examples: list[dict], batch_size: int, pad_id: int, seed: int, shuffle: bool = True):
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for s in range(0, len(idx), batch_size):
        yield collate([examples[i] for i in idx[s:s + batch_size]], pad_id)


def micro_batches(batch: dict[str, torch.Tensor], micro_size: int):
    """Split a batch into micro-batches, each with weight = its share of the batch's target tokens,
    so that sum(weight * micro_loss) equals the full-batch mean token loss (HF loss is a per-token mean)."""
    n_tok = lambda labels: int((labels[:, 1:] != -100).sum())
    total = max(1, n_tok(batch["labels"]))
    B = batch["input_ids"].shape[0]
    for s in range(0, B, micro_size):
        mb = {k: v[s:s + micro_size] for k, v in batch.items()}
        # trim padding columns this micro-batch doesn't need
        L = int(mb["attention_mask"].sum(1).max())
        mb = {k: v[:, :L] for k, v in mb.items()}
        w = n_tok(mb["labels"]) / total
        if w > 0:
            yield mb, w
