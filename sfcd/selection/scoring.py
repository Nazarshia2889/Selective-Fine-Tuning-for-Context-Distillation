"""Per-weight saliency: mean |dL/dW| of the distillation (SFT) loss over N batches, no context."""
from __future__ import annotations

import time

import torch

from sfcd.config import ModelCfg
from sfcd.data import micro_batches
from sfcd.model import LinearInfo, autocast


def compute_grad_scores(model, linears: list[LinearInfo], batch_iter, n_batches: int,
                        model_cfg: ModelCfg, micro_batch_size: int = 4) -> tuple[dict[str, torch.Tensor], dict]:
    """Returns ({param_name: fp32 cpu tensor of mean |grad|}, meta)."""
    device = next(model.parameters()).device
    params = dict(model.named_parameters())
    targets = {li.param_name: params[li.param_name] for li in linears}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in targets.values():
        p.requires_grad_(True)

    acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in targets.items()}
    model.train()
    t0, seen, losses = time.time(), 0, []
    for batch in batch_iter:
        if seen >= n_batches:
            break
        loss = 0.0
        for mb, w in micro_batches(batch, micro_batch_size):  # accumulate the full-batch gradient
            mb = {k: v.to(device) for k, v in mb.items()}
            with autocast(model_cfg, device):
                part = model(**mb).loss * w
            part.backward()
            loss += part.detach()
        for n, p in targets.items():
            acc[n].add_(p.grad.detach().float().abs())
            p.grad = None
        losses.append(float(loss))
        seen += 1
    if seen == 0:
        raise ValueError("no batches available for gradient scoring")
    scores = {n: (a / seen).cpu() for n, a in acc.items()}
    for p in targets.values():
        p.requires_grad_(False)
    meta = {"n_batches": seen, "score_time_s": time.time() - t0, "mean_loss": sum(losses) / seen}
    return scores, meta
