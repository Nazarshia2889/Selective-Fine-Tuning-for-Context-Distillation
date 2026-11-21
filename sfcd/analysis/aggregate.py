"""Collect results/<stage>/*.json into a flat table and a summary grouped by method."""
from __future__ import annotations

import pandas as pd

from sfcd.paths import Paths, read_json

COLS = ["task", "context_id", "granularity", "budget_blocks", "selector", "lr",
        "student_test", "zero_shot_test", "icl_test", "recovery_test", "mmlu", "mmlu_forgetting",
        "n_trainable_params", "frac_eligible_trained", "peak_mem_gb", "train_time_s", "delta_bytes",
        "sel_score_time_s", "run_id"]


def load_results(paths: Paths, stage: str = "main") -> pd.DataFrame:
    d = paths.results_dir(stage)
    rows = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        r = read_json(f)
        c = r["config"]
        rows.append({**{k: c[k] for k in ("task", "context_id", "granularity", "budget_blocks", "selector", "lr")},
                     **{k: r.get(k) for k in COLS if k not in c}})
    return pd.DataFrame(rows, columns=COLS)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["task", "granularity", "budget_blocks", "selector"]
    agg = {"recovery_test": ["mean", "std"], "student_test": "mean", "mmlu_forgetting": "mean",
           "frac_eligible_trained": "mean", "peak_mem_gb": "mean", "train_time_s": "mean",
           "delta_bytes": "mean", "run_id": "count"}
    out = df.fillna({"budget_blocks": 0}).groupby(keys, dropna=False).agg(agg)
    out.columns = ["_".join(c).rstrip("_") for c in out.columns]
    return out.reset_index()


def write_tables(paths: Paths, stage: str = "main") -> dict[str, str]:
    df = load_results(paths, stage)
    summ = summarize(df)
    out_dir = paths.root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stage}_runs.csv", index=False)
    summ.to_csv(out_dir / f"{stage}_summary.csv", index=False)
    return {"runs.csv": df.to_csv(index=False), "summary.csv": summ.to_csv(index=False)}
