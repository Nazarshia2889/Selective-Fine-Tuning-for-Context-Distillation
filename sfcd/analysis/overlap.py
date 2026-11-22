"""Selection overlap (Jaccard) across contexts/tasks, recomputed from cached gradient scores."""
from __future__ import annotations

import itertools
import re

import pandas as pd
import torch

from sfcd.model import LinearInfo
from sfcd.paths import Paths
from sfcd.selection.selectors import select


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else float("nan")


def overlap_table(paths: Paths, contexts: list[tuple[str, int]],
                  granularities=("weight", "row", "block"), budgets=(1, 2, 4), device="cpu") -> pd.DataFrame:
    """Jaccard overlap of top-gradient selections between every pair of contexts."""
    first = torch.load(paths.scores(*contexts[0]), map_location="cpu")["scores"]
    linears = linears_from_scores(first)
    del first
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


def linears_from_scores(scores: dict[str, torch.Tensor]) -> list[LinearInfo]:
    """Eligible linears recovered from a saved score file (keys are '<module>.weight', in model order)."""
    out = []
    for name, t in scores.items():
        m = re.search(r"\.(\d+)\.", name)
        out.append(LinearInfo(name[: -len(".weight")], int(m.group(1)), t.shape[0], t.shape[1]))
    return out
