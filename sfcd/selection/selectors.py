"""Turn per-weight scores into a selection mask at a given granularity and budget.

Budget is expressed in blocks: budget_params = budget_blocks * (eligible params / n_layers), so
weight- and row-level selections are parameter-matched to the block-level ones.
Random selection reuses the exact same code path with i.i.d. uniform scores.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from sfcd.model import LinearInfo


@dataclass
class Mask:
    granularity: str  # weight | row | block
    weight_idx: dict[str, torch.Tensor] = field(default_factory=dict)  # param -> sorted flat indices
    rows: dict[str, torch.Tensor] = field(default_factory=dict)  # param -> sorted row indices
    blocks: list[int] = field(default_factory=list)

    def n_selected(self, linears: list[LinearInfo]) -> int:
        if self.granularity == "weight":
            return sum(int(v.numel()) for v in self.weight_idx.values())
        if self.granularity == "row":
            by = {li.param_name: li for li in linears}
            return sum(int(v.numel()) * by[n].in_features for n, v in self.rows.items())
        return sum(li.numel for li in linears if li.layer in self.blocks)

    def checksum(self) -> str:
        h = hashlib.sha1(self.granularity.encode())
        for d in (self.weight_idx, self.rows):
            for k in sorted(d):
                h.update(k.encode())
                h.update(d[k].cpu().numpy().tobytes())
        h.update(str(sorted(self.blocks)).encode())
        return h.hexdigest()[:12]

    def to_flat_bool(self, linears: list[LinearInfo]) -> torch.Tensor:
        """Concatenated boolean mask over all eligible weights (for overlap analysis)."""
        parts = []
        for li in linears:
            m = torch.zeros(li.numel, dtype=torch.bool)
            if self.granularity == "weight" and li.param_name in self.weight_idx:
                m[self.weight_idx[li.param_name]] = True
            elif self.granularity == "row" and li.param_name in self.rows:
                m.view(li.out_features, li.in_features)[self.rows[li.param_name]] = True
            elif self.granularity == "block" and li.layer in self.blocks:
                m[:] = True
            parts.append(m)
        return torch.cat(parts)


def budget_params(linears: list[LinearInfo], budget_blocks: int | None = None,
                  budget_frac: float | None = None) -> int:
    """Number of weights to train: budget_blocks blocks' worth, or a fraction of all eligible weights."""
    total = sum(li.numel for li in linears)
    if budget_frac is not None:
        return max(1, round(budget_frac * total))
    n_layers = len({li.layer for li in linears})
    return round(budget_blocks * total / n_layers)


def random_scores(linears: list[LinearInfo], seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {li.param_name: torch.rand(li.out_features, li.in_features, generator=g) for li in linears}


def _split_sorted(idx: torch.Tensor, linears: list[LinearInfo], sizes: list[int]) -> dict[str, torch.Tensor]:
    """Split sorted global indices into per-parameter local indices given per-param unit counts."""
    offsets = torch.tensor([0] + sizes, device=idx.device).cumsum(0)
    bounds = torch.searchsorted(idx, offsets)
    out = {}
    for i, li in enumerate(linears):
        local = idx[bounds[i]:bounds[i + 1]] - offsets[i]
        if local.numel():
            out[li.param_name] = local.cpu()
    return out


def select(scores: dict[str, torch.Tensor], linears: list[LinearInfo], granularity: str,
           budget_blocks: int | None, device: str | torch.device = "cpu", budget_frac: float | None = None) -> Mask:
    if granularity == "block" and budget_frac is not None:
        raise ValueError("block selection takes whole blocks; use budget_blocks")
    budget = budget_params(linears, budget_blocks, budget_frac)

    if granularity == "weight":
        flat = torch.cat([scores[li.param_name].to(device).flatten() for li in linears])
        k = min(budget, flat.numel())
        idx = torch.topk(flat, k, sorted=False).indices
        del flat
        idx = idx.sort().values
        return Mask("weight", weight_idx=_split_sorted(idx, linears, [li.numel for li in linears]))

    if granularity == "row":
        # mean |grad| per row, so rows of different widths are comparable
        row_scores = torch.cat([scores[li.param_name].to(device).mean(dim=1) for li in linears])
        row_sizes = torch.cat([torch.full((li.out_features,), li.in_features, device=device) for li in linears])
        order = torch.argsort(row_scores, descending=True)
        cum = row_sizes[order].cumsum(0)
        first_reach = int(torch.searchsorted(cum, torch.tensor([budget], device=device))[0])
        n = min(first_reach + 1, order.numel())
        idx = order[:n].sort().values
        return Mask("row", rows=_split_sorted(idx, linears, [li.out_features for li in linears]))

    if granularity == "block":
        layers = sorted({li.layer for li in linears})
        tot = {l: 0.0 for l in layers}
        cnt = {l: 0 for l in layers}
        for li in linears:
            tot[li.layer] += float(scores[li.param_name].double().sum())
            cnt[li.layer] += li.numel
        mean = {l: tot[l] / cnt[l] for l in layers}
        chosen = sorted(sorted(layers, key=lambda l: -mean[l])[:budget_blocks])
        return Mask("block", blocks=chosen)

    raise ValueError(f"unknown granularity {granularity!r}")
