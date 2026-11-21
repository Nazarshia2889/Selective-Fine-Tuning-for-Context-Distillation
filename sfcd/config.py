"""Typed configuration: base settings shared by every run, a single RunCfg, and sweep expansion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

Granularity = Literal["weight", "row", "block", "full"]
Selector = Literal["topk", "random", "none"]


class ModelCfg(BaseModel):
    name: str = "Qwen/Qwen3.5-0.8B"
    # Master weights are kept in fp32 for every method; compute runs under autocast(compute_dtype).
    # This keeps peak-memory comparisons between methods apples-to-apples.
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    max_seq_len: int = 1024
    # Scope of full fine-tuning: "all" parameters (incl. embeddings/norms) or only the "eligible" linears.
    full_scope: Literal["all", "eligible"] = "all"


class DataCfg(BaseModel):
    n_candidates: int = 5  # candidate contexts generated per task (filtered down to n_keep)
    n_keep: int = 3
    n_train: int = 500
    n_val: int = 100
    n_test: int = 200
    min_gap: float = 0.20  # teacher (ICL) must beat zero-shot by this much on val
    seed: int = 0


class TrainCfg(BaseModel):
    epochs: int = 3
    batch_size: int = 16
    micro_batch_size: int = 4  # gradient accumulation; same math as batch_size, less memory
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 0


class SelectionCfg(BaseModel):
    score_batches: int = 8
    score_batch_size: int = 16
    micro_batch_size: int = 4
    random_seed: int = 0


class EvalCfg(BaseModel):
    gen_batch_size: int = 32
    mmlu_n: int = 500  # 0 disables MMLU
    mmlu_seed: int = 0
    mmlu_batch_size: int = 16


class WandbCfg(BaseModel):
    project: str = "sfcd"
    entity: Optional[str] = None
    mode: Literal["online", "offline", "disabled"] = "online"


class BaseCfg(BaseModel):
    model: ModelCfg = Field(default_factory=ModelCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    selection: SelectionCfg = Field(default_factory=SelectionCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    wandb: WandbCfg = Field(default_factory=WandbCfg)
    save_delta: bool = True


class RunCfg(BaseModel):
    base: BaseCfg
    stage: Literal["main", "lr_tune", "smoke"] = "main"
    task: str
    context_id: int
    granularity: Granularity
    budget_blocks: Optional[int] = None  # None for full fine-tuning
    selector: Selector = "topk"
    lr: float

    def run_id(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def name(self) -> str:
        if self.granularity == "full":
            core = "full"
        else:
            core = f"{self.granularity}-b{self.budget_blocks}-{self.selector}"
        return f"{self.task}-c{self.context_id}-{core}-lr{self.lr:g}"


class LrTuneCfg(BaseModel):
    pilot_task: str = "knowledge"
    pilot_context: int = 0
    selector: Selector = "topk"
    budget_blocks: int = 2
    max_mmlu_drop: float = 0.02  # lr must keep the MMLU drop within this
    grid: dict[str, list[float]]


class SweepCfg(BaseModel):
    base_config: str = "configs/base.yaml"
    tasks: list[str]
    granularities: list[Literal["weight", "row", "block"]]
    budgets_blocks: list[int]
    selectors: list[Literal["topk", "random"]]
    include_full: bool = True
    default_lr: dict[str, float]
    lr_tune: LrTuneCfg


# ----------------------------------------------------------------------------- loading


def _set_dotted(d: dict, key: str, value) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def parse_overrides(overrides: list[str] | None) -> dict:
    """Turn ["train.epochs=1", "wandb.mode=disabled"] into a nested dict (values parsed as YAML)."""
    out: dict = {}
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        _set_dotted(out, key.strip(), yaml.safe_load(raw))
    return out


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_base(path: str | Path | None = None, overrides: list[str] | None = None) -> BaseCfg:
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    return BaseCfg.model_validate(_deep_merge(raw or {}, parse_overrides(overrides)))


def load_sweep(path: str | Path) -> SweepCfg:
    return SweepCfg.model_validate(yaml.safe_load(Path(path).read_text()))


# ----------------------------------------------------------------------------- expansion


def expand_lr_tune(sweep: SweepCfg, base: BaseCfg) -> list[RunCfg]:
    lt = sweep.lr_tune
    runs = []
    for gran, lrs in lt.grid.items():
        for lr in lrs:
            full = gran == "full"
            runs.append(RunCfg(
                base=base, stage="lr_tune", task=lt.pilot_task, context_id=lt.pilot_context,
                granularity=gran, budget_blocks=None if full else lt.budget_blocks,
                selector="none" if full else lt.selector, lr=lr,
            ))
    return runs


def expand_main(
    sweep: SweepCfg,
    base: BaseCfg,
    kept: dict[str, list[int]] | None = None,
    best_lr: dict[str, float] | None = None,
) -> list[RunCfg]:
    """kept: task -> retained context ids (defaults to range(n_keep)). best_lr: granularity -> lr."""
    lrs = {**sweep.default_lr, **(best_lr or {})}
    runs = []
    for task in sweep.tasks:
        ctx_ids = (kept or {}).get(task, list(range(base.data.n_keep)))
        for cid in ctx_ids:
            for gran in sweep.granularities:
                for b in sweep.budgets_blocks:
                    for sel in sweep.selectors:
                        runs.append(RunCfg(base=base, stage="main", task=task, context_id=cid,
                                           granularity=gran, budget_blocks=b, selector=sel, lr=lrs[gran]))
            if sweep.include_full:
                runs.append(RunCfg(base=base, stage="main", task=task, context_id=cid,
                                   granularity="full", budget_blocks=None, selector="none", lr=lrs["full"]))
    return runs


def expand_stage(stage: str, sweep: SweepCfg, base: BaseCfg, state: dict) -> list[RunCfg]:
    """state: {"kept": task -> context ids, "best_lr": granularity -> lr} (either may be None)."""
    if stage == "lr_tune":
        return expand_lr_tune(sweep, base)
    if stage == "main":
        return expand_main(sweep, base, state.get("kept"), state.get("best_lr"))
    raise ValueError(f"unknown run stage {stage!r}")
