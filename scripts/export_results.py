"""Download the saved results from the Modal volume and write the CSV tables behind every paper figure.

    python scripts/export_results.py            # download + write results/paper/*.csv
    python scripts/export_results.py --no-fetch # rebuild the CSVs from an existing download

Tables (one row per run unless noted):
  phase1_runs.csv   phase 1: individual weights at each share of weights and learning rate
  gran_runs.csv     phase 2: weights, rows and blocks at 4.2%, plus full fine-tuning
  math_general.csv  MATH accuracy of every math student (and the untrained model)
  cost_runs.csv     training time, peak memory and storage of every run
  loss_curves.csv   training loss per step (run_id, step, loss)
  cost_bench.csv    controlled timing/memory benchmark (one container per task, 3 repeats per setting)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "outputs" / "remote"
OUT = ROOT / "results" / "paper"
VOLUME = "sfcd-vol"

TASKS = {"knowledge": "Knowledge", "arbitrary_labels": "Arbitrary labels", "math_methods": "Math (digit sum mod 9)"}
GRAN = {"weight": "Weights", "row": "Rows", "block": "Blocks", "full": "Full fine-tuning"}
KEPT = {"knowledge": [0, 1, 2], "arbitrary_labels": [0, 2, 3], "math_methods": [2]}
MATH_N_VAL = 300  # sfcd/tasks/math_methods.py: n_val


def fetch() -> None:
    """modal volume get for everything the tables need (results JSONs, baselines, diagnostics)."""
    items = ["results/sparsity", "results/granularity", "diagnostics/math_bench",
             "diagnostics/math_bench_students.json", "diagnostics/delta_norms_sparsity.json",
             "diagnostics/delta_norms_granularity.json"]
    items += [f"diagnostics/cost_bench_{t}.json" for t in KEPT]
    items += [f"data/{t}/c{c}/{f}" for t, cs in KEPT.items() for c in cs for f in ("baselines.json", "context.json", "test.jsonl")]
    for it in items:
        dest = REMOTE / it
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("fetching", it)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", VOLUME, it, str(dest.parent)], check=True,
                       stdout=subprocess.DEVNULL)


def _load(pattern: str) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(str(REMOTE / pattern), recursive=True))]


def _write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / name} ({len(rows)} rows)")


def build() -> None:
    base = {(t, c): json.load(open(REMOTE / f"data/{t}/c{c}/baselines.json")) for t, cs in KEPT.items() for c in cs}
    change = {}
    for f in ("delta_norms_sparsity.json", "delta_norms_granularity.json"):
        p = REMOTE / "diagnostics" / f
        if p.exists():
            change.update({r["run_id"]: r["l2"] for r in json.load(open(p))})
    students = json.load(open(REMOTE / "diagnostics/math_bench_students.json"))
    math = {k: v["overall"] for k, v in students.items()}
    for f in glob.glob(str(REMOTE / "diagnostics/math_bench/**/*.json"), recursive=True):
        math[os.path.basename(f)[:-5]] = json.load(open(f))["overall"]

    runs = []
    for stage in ("sparsity", "granularity"):
        for r in _load(f"results/{stage}/**/*.json"):
            c = r["config"]
            if c["task"] not in TASKS or c["context_id"] not in KEPT[c["task"]]:
                continue  # earlier task versions
            runs.append((stage, r))

    p1, gran, cost, curves = [], [], [], []
    for stage, r in runs:
        c = r["config"]
        b = base[(c["task"], c["context_id"])]
        n_val = MATH_N_VAL if c["task"] == "math_methods" else c["base"]["data"]["n_val"]
        row = {
            "run_id": r["run_id"], "stage": stage, "task": TASKS[c["task"]], "context": c["context_id"],
            "granularity": GRAN[c["granularity"]], "fkey": round(r["frac_eligible_trained"] * 100, 4), "lr": c["lr"],
            "n_trained": r["n_trainable_params"],
            "student_val": r["student_val"], "student_test": r["student_test"],
            "icl_val": b["icl_val"], "icl_test": b["icl_test"], "zero_shot_test": b["zero_shot_test"],
            "trivial_test": b.get("trivial_test", 0.0), "n_val": n_val, "recovery": r["recovery_test"],
            "mmlu": r["mmlu"], "mmlu_base": r["mmlu_base"], "mmlu_drop": r["mmlu_forgetting"],
            "change_l2": change.get(r["run_id"], ""),
            "math_acc": math[r["run_id"]]["acc"] if r["run_id"] in math else "",
            "math_base": math["base"]["acc"] if c["task"] == "math_methods" else "",
        }
        if stage == "sparsity" and c["granularity"] == "weight":
            p1.append(row)
        if stage == "granularity" or abs(row["fkey"] - 4.1667) < 1e-3:
            gran.append(row)
        cost.append({k: row[k] for k in ("run_id", "task", "context", "granularity", "fkey", "lr", "n_trained")}
                    | {"train_time_s": r["train_time_s"], "train_steps": r["train_steps"], "peak_mem_gb": r["peak_mem_gb"],
                       "storage_mb": r["delta_bytes"] / 1e6, "answer_tokens": r.get("n_train_answer_tokens")})
        curves += [{"run_id": r["run_id"], "step": i, "loss": l} for i, l in enumerate(r.get("loss_curve") or [])]

    m = [{"run_id": "", "model": "Untrained model", "granularity": "", "fkey": 0, "lr": "",
          "math_acc": math["base"]["acc"], "hit_limit": math["base"]["hit_limit"]}]
    for row in gran + p1:
        if row["math_acc"] != "" and not any(x["run_id"] == row["run_id"] for x in m):
            m.append({"run_id": row["run_id"], "model": f"{row['granularity']} {row['fkey']:g}%", "granularity": row["granularity"],
                      "fkey": row["fkey"], "lr": row["lr"], "math_acc": row["math_acc"], "hit_limit": math[row["run_id"]]["hit_limit"]})
    _write("phase1_runs.csv", p1)
    _write("gran_runs.csv", gran)
    _write("math_general.csv", m)
    _write("cost_runs.csv", cost)
    _write("loss_curves.csv", curves)
    _write("cost_bench.csv", cost_bench())
    _write("inference_tokens.csv", inference_tokens())


def cost_bench() -> list[dict]:
    """Rows of diagnostics/cost_bench_<task>.json (sfcd/analysis/cost_bench.py); fkey = % of eligible weights."""
    rows = []
    for t in KEPT:
        d = json.load(open(REMOTE / f"diagnostics/cost_bench_{t}.json"))
        for r in d["rows"]:
            method = f"Full fine-tuning ({r['full_scope']})" if r["granularity"] == "full" else "Weights"
            rows.append({"task": TASKS[t], "method": method, "fkey": round(r["n_trained"] / r["n_eligible"] * 100, 4),
                         "lr": r["lr"], "repeat": r["repeat"], "n_trained": r["n_trained"], "oom": r["oom"],
                         "train_time_s": r.get("train_time_s", ""), "peak_mem_gb": r["peak_mem_gb"],
                         "gpu": d["env"]["gpu"], "torch": d["env"]["torch"], "transformers": d["env"]["transformers"]})
    return rows


def inference_tokens() -> list[dict]:
    """Prompt length (tokens) per test query, with and without the context, using the real chat prompt."""
    from sfcd.config import load_base
    from sfcd.model import build_prompt, load_tokenizer
    from sfcd.tasks import get_task

    base = load_base(ROOT / "configs" / "base.yaml")
    tok = load_tokenizer(base.model.name)
    rows = []
    for t, cs in KEPT.items():
        task = get_task(t)
        for c in cs:
            ctx = json.load(open(REMOTE / f"data/{t}/c{c}/context.json"))["text"]
            for line in open(REMOTE / f"data/{t}/c{c}/test.jsonl"):
                q = json.loads(line)["query"]
                n = lambda text: len(tok(build_prompt(tok, task, text, q), add_special_tokens=False)["input_ids"])
                rows.append({"task": TASKS[t], "context": c, "with_context": n(ctx), "without_context": n(None)})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch:
        fetch()
    build()
