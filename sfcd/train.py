"""SFT loop over the trainable units only. Reports peak memory and wall time."""
from __future__ import annotations

import math
import time
from typing import Callable

import torch

from sfcd.config import ModelCfg, TrainCfg
from sfcd.data import batches, micro_batches
from sfcd.model import autocast


def train_sft(model, trainable: list[torch.nn.Parameter], examples: list[dict], pad_id: int,
              cfg: TrainCfg, lr: float, model_cfg: ModelCfg,
              log: Callable[[dict], None] | None = None) -> dict:
    from transformers import get_cosine_schedule_with_warmup

    device = next(model.parameters()).device
    torch.manual_seed(cfg.seed)
    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg.weight_decay,
                            fused=device.type == "cuda")
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0, step, losses = time.time(), 0, []
    for epoch in range(cfg.epochs):
        for batch in batches(examples, cfg.batch_size, pad_id, seed=cfg.seed + epoch):
            loss = 0.0
            for mb, w in micro_batches(batch, cfg.micro_batch_size):
                mb = {k: v.to(device) for k, v in mb.items()}
                with autocast(model_cfg, device):
                    part = model(**mb).loss * w
                part.backward()
                loss += part.detach()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss))
            if log:
                log({"train/loss": losses[-1], "train/grad_norm": float(gnorm),
                     "train/lr": sched.get_last_lr()[0], "train/epoch": epoch, "train/step": step})
    if device.type == "cuda":
        torch.cuda.synchronize()
    stats = {
        "train_time_s": time.time() - t0,
        "train_steps": step,
        "final_loss": sum(losses[-steps_per_epoch:]) / max(1, len(losses[-steps_per_epoch:])),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
    }
    del opt
    return stats
