"""Learning-rate selection used in the paper (one-standard-error rule with a teacher cap).

For one group of runs (a task and a share of weights, or a task and a granularity), each learning rate
has a mean validation accuracy over the task's contexts. Let p be the best of these and n the number of
validation questions behind it. We pick the smallest learning rate whose mean validation accuracy is at
least

    min(p - sqrt(p (1 - p) / n),  teacher validation accuracy)

i.e. within one binomial standard error of the best (Hastie, Tibshirani & Friedman, ESL §7.10), or at
least as good as the teacher, whichever is easier to reach. Only validation accuracy is used.
"""
from __future__ import annotations

import math
from collections import defaultdict


def summarize(runs: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    """Average runs (one per context) into settings keyed by group_keys + lr.
    Each run needs: lr, student_val, icl_val, n_val, plus the group keys."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[tuple(r[k] for k in group_keys) + (float(r["lr"]),)].append(r)
    out = []
    for key, rs in groups.items():
        out.append({**{k: v for k, v in zip(group_keys, key)}, "lr": key[-1], "contexts": len(rs),
                    "val": sum(float(r["student_val"]) for r in rs) / len(rs),
                    "teacher_val": sum(float(r["icl_val"]) for r in rs) / len(rs),
                    "n_val": sum(int(r["n_val"]) for r in rs), "runs": rs})
    return out


def pick_lr(settings: list[dict]) -> dict:
    """settings: one group's learning rates (from summarize). Returns the picked setting."""
    best = sorted(settings, key=lambda s: (-s["val"], s["lr"]))[0]
    p = best["val"]
    threshold = min(p - math.sqrt(p * (1 - p) / best["n_val"]), best["teacher_val"])
    ok = [s for s in settings if s["val"] >= threshold - 1e-9]
    return min(ok, key=lambda s: s["lr"])


def picks(runs: list[dict], group_keys: tuple[str, ...]) -> dict[tuple, dict]:
    """group (without lr) -> picked setting."""
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for s in summarize(runs, group_keys):
        by_group[tuple(s[k] for k in group_keys)].append(s)
    return {g: pick_lr(ss) for g, ss in by_group.items()}
