"""Selection overlap (Jaccard) across contexts/tasks, recomputed from cached gradient scores."""
from __future__ import annotations

import itertools

import pandas as pd
import torch

from sfcd.model import LinearInfo
from sfcd.paths import Paths
from sfcd.selection.selectors import select


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else float("nan")


def overlap_table(paths: Paths, linears: list[LinearInfo], contexts: list[tuple[str, int]],
                  granularities=("weight", "row", "block"), budgets=(1, 2, 4), device="cpu") -> pd.DataFrame:
    """linears must describe the model the scores came from (see sfcd.model.eligible_linears).
    Computing it needs only the model config, e.g. via a meta-device model."""
    rows = []
    for gran, b in itertools.product(granularities, budgets):
        masks = {}
        for task, cid in contexts:
            scores = torch.load(paths.scores(task, cid), map_location="cpu")["scores"]
            masks[(task, cid)] = select(scores, linears, gran, b, device=device).to_flat_bool(linears)
            del scores
        for (ka, ma), (kb, mb) in itertools.combinations(masks.items(), 2):
            rows.append({"granularity": gran, "budget_blocks": b, "a": f"{ka[0]}/c{ka[1]}",
                         "b": f"{kb[0]}/c{kb[1]}", "same_task": ka[0] == kb[0], "jaccard": jaccard(ma, mb)})
    return pd.DataFrame(rows)


def linears_from_config(model_name: str) -> list[LinearInfo]:
    """Eligible linears without loading weights (meta device)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from sfcd.model import eligible_linears

    cfg = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return eligible_linears(model)
