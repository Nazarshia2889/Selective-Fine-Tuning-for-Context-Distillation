"""Controlled training-cost benchmark: one GPU container per task, every setting trained back to back.

For one context of a task, each share of weights is trained at its picked learning rate, plus full
fine-tuning of all parameters and of the eligible linears only. Settings run round-robin, `repeats`
times, after one discarded warm-up run (kernel compilation / autotuning). Only training is timed;
nothing is evaluated or saved. Peak memory is torch.cuda.max_memory_allocated during training.
"""
from __future__ import annotations

import logging
import platform
import sys

import torch

from sfcd.config import BaseCfg, RunCfg
from sfcd.model import eligible_linears, get_device, load_model
from sfcd.paths import Paths

log = logging.getLogger("sfcd")


def environment(model) -> dict:
    """Versions, GPU, and which linear-attention / conv kernels the model's code actually calls."""
    import transformers

    env = {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    try:
        import fla  # noqa: F401
        env["fla_import"] = "ok"
    except Exception as e:  # report why the fused path is unavailable
        env["fla_import"] = repr(e)[:300]
    kernels = {}
    for mod in model.modules():
        if "GatedDeltaNet" in type(mod).__name__:
            src = sys.modules[type(mod).__module__]
            for holder in (mod, src):
                for name in dir(holder):
                    if "gated_delta_rule" in name or "causal_conv1d" in name:
                        f = getattr(holder, name)
                        kernels[name] = None if f is None else f"{getattr(f, '__module__', '?')}.{getattr(f, '__name__', '?')}"
            break
    env["kernels"] = kernels
    env["fast_path"] = bool(kernels) and all(v and "torch_" not in v.split(".")[-1] for k, v in kernels.items()
                                             if "gated_delta_rule" in k)
    return env


def _one(paths: Paths, run: RunCfg) -> dict:
    from sfcd.pipeline import _free, _sft_examples
    from sfcd.selection.selectors import select
    from sfcd.sparse.params import apply_selection, count_params
    from sfcd.train import train_sft

    base, device = run.base, get_device()
    model, tok = load_model(base.model, device)
    linears = eligible_linears(model)
    mask = None
    if run.granularity != "full":
        blob = torch.load(paths.scores(run.task, run.context_id), map_location="cpu")
        mask = select(blob["scores"], linears, run.granularity, run.budget_blocks, device=device,
                      budget_frac=run.budget_frac)
        del blob
        _free()
    trainable = apply_selection(model, linears, run.granularity, mask, base.model.full_scope)
    n = count_params(trainable)
    examples = _sft_examples(paths, base, run.task, run.context_id, tok)
    out = {"n_trained": n, "n_eligible": sum(li.numel for li in linears)}
    try:
        stats = train_sft(model, trainable, examples, tok.pad_token_id, base.train, run.lr, base.model)
        out |= {k: stats[k] for k in ("train_time_s", "train_steps", "peak_mem_gb", "final_loss")} | {"oom": False}
    except torch.OutOfMemoryError:
        out |= {"oom": True, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    del model, trainable, mask
    _free()
    return out


def run_bench(paths: Paths, specs: list[dict], repeats: int = 3) -> dict:
    """specs: [{"label", "run": RunCfg dict}]. Returns env + one row per (spec, repeat)."""
    runs = [(s["label"], RunCfg.model_validate(s["run"])) for s in specs]
    m, _ = load_model(runs[0][1].base.model, get_device())
    env = environment(m)
    del m
    log.info("environment: %s", env)
    _one(paths, runs[0][1])  # warm-up, discarded
    rows = []
    for rep in range(repeats):
        for label, run in runs:
            r = _one(paths, run)
            rows.append({"label": label, "repeat": rep, "granularity": run.granularity,
                         "full_scope": run.base.model.full_scope if run.granularity == "full" else "",
                         "lr": run.lr, **r})
            log.info("bench %s rep %d: %s", label, rep, r)
    return {"env": env, "rows": rows}
