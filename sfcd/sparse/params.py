"""Memory-real selective fine-tuning.

The base weights stay frozen; only the selected units get trainable fp32 deltas, so gradients and
AdamW state scale with the budget rather than with the model size.

- block: requires_grad on eligible linear weights of the selected blocks (no wrapper needed)
- row:   RowDeltaLinear adds x @ delta^T into the selected output columns (no dense gradient)
- weight: WeightDeltaLinear builds W + scatter(delta) each forward. This materialises a transient
          dense copy of W (and its gradient) per wrapped layer: the honest overhead of
          unstructured sparsity, which is visible in the peak-memory metric.
- full:  ordinary fine-tuning of all parameters (or only eligible ones, per ModelCfg.full_scope)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sfcd.model import LinearInfo
from sfcd.selection.selectors import Mask


class RowDeltaLinear(nn.Module):
    def __init__(self, base: nn.Linear, rows: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("rows", rows.to(base.weight.device, torch.long), persistent=False)
        self.delta = nn.Parameter(torch.zeros(len(rows), base.in_features, device=base.weight.device,
                                              dtype=torch.float32))

    def forward(self, x):
        y = self.base(x)
        d = F.linear(x, self.delta)
        return y.index_add(y.dim() - 1, self.rows, d.to(y.dtype))

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        self.base.weight.index_add_(0, self.rows, self.delta.to(self.base.weight.dtype))
        return self.base

    def export(self) -> dict[str, torch.Tensor]:
        return {"rows": self.rows.to(torch.int32).cpu(), "delta": self.delta.detach().cpu()}


class WeightDeltaLinear(nn.Module):
    def __init__(self, base: nn.Linear, flat_idx: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("idx", flat_idx.to(base.weight.device, torch.long), persistent=False)
        self.delta = nn.Parameter(torch.zeros(len(flat_idx), device=base.weight.device, dtype=torch.float32))

    def effective_weight(self):
        w = self.base.weight
        return w.flatten().index_add(0, self.idx, self.delta.to(w.dtype)).view_as(w)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.base.bias)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        self.base.weight.view(-1).index_add_(0, self.idx, self.delta.to(self.base.weight.dtype))
        return self.base

    def export(self) -> dict[str, torch.Tensor]:
        return {"idx": self.idx.to(torch.int32).cpu(), "delta": self.delta.detach().cpu()}


_WRAPPERS = (RowDeltaLinear, WeightDeltaLinear)


def _set_module(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def apply_selection(model: nn.Module, linears: list[LinearInfo], granularity: str,
                    mask: Mask | None, full_scope: str = "all") -> list[nn.Parameter]:
    """Freeze the model, attach trainable units, and return the trainable parameter list."""
    for p in model.parameters():
        p.requires_grad_(False)

    if granularity == "full":
        if full_scope == "all":
            params = list(model.parameters())
        else:
            named = dict(model.named_parameters())
            params = [named[li.param_name] for li in linears]
        for p in params:
            p.requires_grad_(True)
        return params

    assert mask is not None and mask.granularity == granularity
    if granularity == "block":
        named = dict(model.named_parameters())
        params = [named[li.param_name] for li in linears if li.layer in mask.blocks]
        for p in params:
            p.requires_grad_(True)
        return params

    params = []
    for li in linears:
        sel = (mask.rows if granularity == "row" else mask.weight_idx).get(li.param_name)
        if sel is None or sel.numel() == 0:
            continue
        base = model.get_submodule(li.name)
        wrapper = RowDeltaLinear(base, sel) if granularity == "row" else WeightDeltaLinear(base, sel)
        _set_module(model, li.name, wrapper)
        params.append(wrapper.delta)
    return params


def export_delta(model: nn.Module, granularity: str, trainable: list[nn.Parameter]) -> dict[str, torch.Tensor]:
    """The per-context artifact you would actually store: only what changed."""
    if granularity in ("row", "weight"):
        out = {}
        for name, mod in model.named_modules():
            if isinstance(mod, _WRAPPERS):
                for k, v in mod.export().items():
                    out[f"{name}.{k}"] = v
        return out
    ids = {id(p) for p in trainable}
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if id(p) in ids}


def delta_nbytes(tensors: dict[str, torch.Tensor]) -> int:
    """Raw tensor bytes (indices as int32, values as fp32); the saved file adds a small header."""
    return sum(t.numel() * t.element_size() for t in tensors.values())


def merge_into_base(model: nn.Module) -> nn.Module:
    """Fold row/weight deltas back into plain nn.Linear modules (for fast eval)."""
    wrapped = [(n, m) for n, m in model.named_modules() if isinstance(m, _WRAPPERS)]
    for name, mod in wrapped:
        _set_module(model, name, mod.merged())
    return model


def count_params(params: list[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)
