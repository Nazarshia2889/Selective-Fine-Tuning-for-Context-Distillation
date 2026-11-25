"""Modal launcher. Every stage is a thin wrapper around sfcd.pipeline; artifacts live on a Volume.

    modal run modal_app.py --stage prep          # data -> teacher -> filter -> mmlu -> score
    modal run modal_app.py --stage lr_tune       # 12 runs on the pilot context
    modal run modal_app.py --stage select_lr
    modal run modal_app.py --stage main          # 171 runs (162 selective + 9 full), fanned out
    modal run modal_app.py --stage aggregate     # writes ./results/*.csv locally
    modal run modal_app.py --stage main --dry-run --limit 3

Requires a Modal secret named "wandb" containing WANDB_API_KEY (or pass --overrides wandb.mode=disabled).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = os.environ.get("SFCD_APP", "sfcd")
VOLUME_NAME = os.environ.get("SFCD_VOLUME", "sfcd-vol")
GPU = os.environ.get("SFCD_GPU", "A100-40GB")
MAX_CONTAINERS = int(os.environ.get("SFCD_MAX_CONTAINERS", "20"))
ROOT = "/vol"

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        # torch 2.9 + flash-linear-attention: fused kernels for Qwen3.5's linear-attention layers
        # (~2.3x faster training than the PyTorch fallback; same math up to float rounding)
        "torch==2.9.1", "transformers>=4.57", "accelerate", "datasets", "safetensors",
        "pydantic>=2", "pyyaml", "pandas", "wandb", "flash-linear-attention", "einops",
    )
    .env({"HF_HOME": f"{ROOT}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("sfcd")
)
app = modal.App(APP_NAME, image=image)
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])

gpu_kwargs = dict(gpu=GPU, volumes={ROOT: vol}, timeout=4 * 3600, retries=2, max_containers=MAX_CONTAINERS)


def _setup():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    vol.reload()
    from sfcd.paths import Paths

    return Paths(ROOT)


def _base(d: dict):
    from sfcd.config import BaseCfg

    return BaseCfg.model_validate(d)


@app.function(volumes={ROOT: vol}, timeout=3600, memory=8192)
def build_data(base: dict, tasks: list[str]):
    from sfcd import pipeline

    pipeline.stage_build_data(_setup(), _base(base), tasks)
    vol.commit()


@app.function(**gpu_kwargs)
def teacher(base: dict, task: str, cid: int) -> dict:
    from sfcd import pipeline

    out = pipeline.stage_teacher(_setup(), _base(base), task, cid)
    vol.commit()
    return out


@app.function(volumes={ROOT: vol}, timeout=600)
def filter_contexts(base: dict, tasks: list[str]) -> dict:
    from sfcd import pipeline

    out = pipeline.stage_filter(_setup(), _base(base), tasks)
    vol.commit()
    return out


@app.function(**gpu_kwargs)
def mmlu_base(base: dict) -> dict | None:
    from sfcd import pipeline

    out = pipeline.stage_mmlu_base(_setup(), _base(base))
    vol.commit()
    return out


@app.function(**{**gpu_kwargs, "timeout": 6 * 3600, "retries": 0})
def prep_remote(base: dict, tasks: list[str]) -> dict:
    """Whole prep for the given tasks in one container (data -> teacher -> filter -> score), so it
    doesn't depend on the local client staying connected. Kept contexts of other tasks are untouched."""
    import json

    from sfcd import pipeline
    from sfcd.model import load_model

    paths = _setup()
    b = _base(base)
    pipeline.stage_build_data(paths, b, tasks)
    vol.commit()
    mt = load_model(b.model)
    for t in tasks:
        for c in pipeline.candidate_ids(t, b):
            pipeline.stage_teacher(paths, b, t, c, mt)
            vol.commit()
    kept = pipeline.stage_filter(paths, b, tasks)
    for t in tasks:
        for c in kept[t]:
            pipeline.stage_score(paths, b, t, c, mt)
    vol.commit()
    details = pipeline.read_json(paths.kept())["details"]
    out = {"kept": {t: kept[t] for t in tasks},
           "details": {k: v for k, v in details.items() if k.split("/")[0] in tasks}}
    pipeline.write_json(paths.root / "diagnostics" / f"prep_{'_'.join(tasks)}.json", out)
    vol.commit()
    print(json.dumps(out, indent=1))
    return out


@app.function(**gpu_kwargs)
def score(base: dict, task: str, cid: int):
    from sfcd import pipeline

    pipeline.stage_score(_setup(), _base(base), task, cid)
    vol.commit()


@app.function(**gpu_kwargs, secrets=[wandb_secret])
def train_eval(run: dict, git_sha: str | None = None, force: bool = False) -> dict:
    from sfcd import pipeline
    from sfcd.config import RunCfg

    if git_sha:
        os.environ["SFCD_GIT_SHA"] = git_sha
    res = pipeline.stage_train_eval(_setup(), RunCfg.model_validate(run), force=force)
    vol.commit()
    return {k: res.get(k) for k in ("name", "recovery_test", "student_test", "peak_mem_gb", "train_time_s")}


@app.function(volumes={ROOT: vol}, timeout=600)
def select_lr(max_mmlu_drop: float) -> dict:
    from sfcd import pipeline

    out = pipeline.stage_select_lr(_setup(), max_mmlu_drop)
    vol.commit()
    return out


@app.function(**{**gpu_kwargs, "timeout": 6 * 3600})
def reeval(task: str, cid: int, stage: str = "main") -> dict:
    from sfcd import pipeline

    out = pipeline.stage_reeval_context(_setup(), task, cid, stage)
    vol.commit()
    return out


@app.function(volumes={ROOT: vol}, timeout=600)
def progress() -> str:
    paths = _setup()
    lines = []
    for stage in ("lr_tune", "main", "sparsity", "granularity"):
        d = paths.results_dir(stage)
        lines.append(f"{stage}: {len(list(d.glob('*.json'))) if d.exists() else 0} finished")
    return "\n".join(lines)


@app.function(volumes={ROOT: vol}, timeout=600)
def read_state() -> dict:
    from sfcd import pipeline

    return pipeline.read_state(_setup())


@app.function(volumes={ROOT: vol}, timeout=1800, memory=4096)
def aggregate(stage: str = "main") -> dict:
    from sfcd.analysis.aggregate import write_tables

    out = write_tables(_setup(), stage)
    vol.commit()
    return out


@app.function(volumes={ROOT: vol}, timeout=3 * 3600, memory=65536)
def overlap(contexts: list[tuple[str, int]]) -> str:
    from sfcd.analysis.overlap import overlap_table

    paths = _setup()
    df = overlap_table(paths, [tuple(c) for c in contexts])
    (paths.root / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.root / "tables" / "overlap.csv", index=False)
    vol.commit()
    return df.to_csv(index=False)


@app.function(**gpu_kwargs)
def diagnose_forgetting(base: dict, task: str, cid: int) -> dict:
    from sfcd.analysis.forgetting import diagnose
    from sfcd.paths import write_json

    paths = _setup()
    out = diagnose(paths, _base(base), task, cid)
    write_json(paths.root / "diagnostics" / f"forgetting_{task}_c{cid}.json", out)
    vol.commit()
    return out


@app.function(**gpu_kwargs)
def env_check(base: dict, task: str, cid: int, write: bool = False) -> dict:
    from sfcd import pipeline
    from sfcd.paths import write_json

    paths = _setup()
    out = pipeline.stage_env_check(paths, _base(base), task, cid, write)
    write_json(paths.root / "diagnostics" / f"env_check_{task}_c{cid}.json", out)
    vol.commit()
    return out


@app.function(**gpu_kwargs)
def mmlu_batch_check(base: dict) -> dict:
    from sfcd import pipeline
    from sfcd.paths import write_json

    paths = _setup()
    out = pipeline.mmlu_batching_check(paths, _base(base))
    write_json(paths.root / "diagnostics" / "mmlu_batch_check.json", out)
    vol.commit()
    return out


@app.function(volumes={ROOT: vol}, timeout=1800, memory=4096)
def select_by_val() -> dict:
    from sfcd import pipeline

    out = pipeline.stage_select_by_val(_setup())
    vol.commit()
    return out


@app.function(**{**gpu_kwargs, "timeout": 6 * 3600})
def rescore_mmlu(task: str, cid: int) -> dict:
    from sfcd import pipeline

    out = pipeline.stage_rescore_mmlu(_setup(), task, cid)
    vol.commit()
    return out


@app.function(**gpu_kwargs)
def math_check(base: dict, cids: list[int]) -> dict:
    from sfcd.analysis.math_check import check
    from sfcd.paths import write_json

    paths = _setup()
    out = check(paths, _base(base), cids)
    write_json(paths.root / "diagnostics" / "math_check.json", out)
    vol.commit()
    return out


@app.function(**gpu_kwargs)
def math_bench(base: dict, decodings: list[str] | None = None, per_level: int = 10) -> dict:
    from sfcd.analysis.math_bench import run
    from sfcd.paths import write_json

    paths = _setup()
    out = run(_base(base), per_level=per_level, decodings=decodings)
    name = "math_bench_decoding.json" if decodings else "math_bench.json"
    write_json(paths.root / "diagnostics" / name, out)
    vol.commit()
    return out


@app.function(**{**gpu_kwargs, "timeout": 6 * 3600})
def probe_math_methods(base: dict) -> list:
    from sfcd.analysis.probe import probe
    from sfcd.paths import write_json
    from sfcd.tasks.math_methods import FAMILIES, MathMethodsTask

    paths = _setup()
    rows = probe(_base(base), {"math_methods": MathMethodsTask()}, list(range(len(FAMILIES))), 50)
    write_json(paths.root / "diagnostics" / "probe_math_methods.json", rows)
    vol.commit()
    return rows


@app.function(**{**gpu_kwargs, "timeout": 6 * 3600})
def math_bench_students(run_ids: list[str]) -> dict:
    from sfcd.analysis.math_bench import run_students
    from sfcd.paths import write_json

    paths = _setup()
    out = run_students(paths, run_ids)
    write_json(paths.root / "diagnostics" / "math_bench_students.json", out)
    vol.commit()
    return out


@app.function(**gpu_kwargs)
def math_bench_run(run_id: str, stage: str = "sparsity") -> dict:
    """General-math check for one student (no base pass); result in diagnostics/math_bench/<run_id>.json."""
    from sfcd.analysis.math_bench import run_students
    from sfcd.paths import write_json

    paths = _setup()
    out = run_students(paths, [run_id], stage=stage, include_base=False)[run_id]
    write_json(paths.root / "diagnostics" / "math_bench" / f"{run_id}.json", out)
    vol.commit()
    return {"run_id": run_id, "acc": out["overall"]["acc"]}


@app.function(volumes={ROOT: vol}, timeout=3600, memory=16384)
def delta_norms(stage: str = "sparsity", tasks: list[str] | None = None) -> list[dict]:
    """Size of each saved weight change (no GPU): L2 norm, mean |change| per trained weight, max |change|."""
    import torch
    from safetensors.torch import load_file

    from sfcd.paths import read_json, write_json

    paths = _setup()
    rows = []
    base_params = None
    for f in sorted(paths.results_dir(stage).glob("*.json")):
        r = read_json(f)
        c = r["config"]
        if tasks and c["task"] not in tasks:
            continue
        dp = paths.delta(stage, r["run_id"])
        if not dp.exists():
            continue
        d = load_file(str(dp))
        if c["granularity"] in ("block", "full"):  # saved as the new weights, not the change: subtract the base
            if base_params is None:
                from sfcd.config import BaseCfg
                from sfcd.model import load_model

                m, _ = load_model(BaseCfg.model_validate(c["base"]).model, torch.device("cpu"))
                base_params = {n: p.detach() for n, p in m.named_parameters()}
            vals = [v.float() - base_params[k].float() for k, v in d.items()]
        else:
            vals = [v.float() for k, v in d.items() if k.endswith(".delta")] or [v.float() for v in d.values()]
        sq = sum(float((v ** 2).sum()) for v in vals)
        n = sum(v.numel() for v in vals)
        rows.append({"run_id": r["run_id"], "task": c["task"], "context_id": c["context_id"], "lr": c["lr"],
                     "frac": r.get("frac_eligible_trained"), "n_trained": n, "l2": sq ** 0.5,
                     "mean_abs": sum(float(v.abs().sum()) for v in vals) / max(1, n),
                     "max_abs": max(float(v.abs().max()) for v in vals),
                     "student_test": r.get("student_test"), "recovery_test": r.get("recovery_test"),
                     "mmlu_forgetting": r.get("mmlu_forgetting")})
    write_json(paths.root / "diagnostics" / f"delta_norms_{stage}.json", rows)
    vol.commit()
    return rows


MMLU_CHECK_RUNS = "716e98e20a41,eea5f8070579,91586ed005d6,a5aa948c7ab7,7cb41a89f935,8d7dd188f60f"


@app.function(**gpu_kwargs)
def mmlu_check(run_ids: list[str], stage: str = "main") -> dict:
    from sfcd.analysis.mmlu_check import check
    from sfcd.paths import write_json

    paths = _setup()
    out = check(paths, run_ids, stage)
    write_json(paths.root / "diagnostics" / "mmlu_check.json", out)
    vol.commit()
    return out


PROBE_SPECS = [  # label, task, constructor kwargs
    ("base7_small_8shots", "base_arith", {"variant": "small", "n_shots": 8}),
    ("base7_small_16shots", "base_arith", {"variant": "small", "n_shots": 16}),
    ("base7_small_32shots", "base_arith", {"variant": "small", "n_shots": 32}),
    ("operator", "rule_arith", {"rule": "operator"}),
    ("procedure", "rule_arith", {"rule": "procedure"}),
]


@app.function(**gpu_kwargs)
def probe_tasks(base: dict, specs: list, context_ids: list[int], n: int) -> list[dict]:
    from sfcd.analysis.probe import probe
    from sfcd.tasks import _REGISTRY

    _setup()  # HF cache on the volume
    return probe(_base(base), {label: _REGISTRY[t](**kw) for label, t, kw in specs}, context_ids, n)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _report(results):
    fails = [r for r in results if isinstance(r, Exception)]
    for r in results:
        print("FAILED:" if isinstance(r, Exception) else "ok:", r)
    print(f"{len(results) - len(fails)}/{len(results)} succeeded")


@app.function(**{**gpu_kwargs, "timeout": 3 * 3600, "retries": 0})
def cost_bench(task: str, specs: list[dict], repeats: int = 3) -> dict:
    """Training time and peak memory, back to back in one container; diagnostics/cost_bench_<task>.json."""
    from sfcd.analysis.cost_bench import run_bench
    from sfcd.paths import write_json

    paths = _setup()
    out = run_bench(paths, specs, repeats)
    write_json(paths.root / "diagnostics" / f"cost_bench_{task}.json", out)
    vol.commit()
    return {"task": task, "fast_path": out["env"]["fast_path"]}


def _cost_bench_specs(base) -> dict[str, list[dict]]:
    """Per task: each phase-1 share at its picked lr (one-SE rule on results/paper/phase1_runs.csv),
    plus full fine-tuning of all parameters and of the eligible linears only."""
    import csv

    from sfcd.analysis.select_rule import picks
    from sfcd.config import RunCfg

    names = {"Knowledge": "knowledge", "Arbitrary labels": "arbitrary_labels", "Math (digit sum mod 9)": "math_methods"}
    ctx = {"knowledge": 0, "arbitrary_labels": 0, "math_methods": 2}
    fracs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.02]
    rows = list(csv.DictReader(open("results/paper/phase1_runs.csv")))
    specs: dict[str, list[dict]] = {t: [] for t in ctx}
    for (task, fkey), s in sorted(picks(rows, ("task", "fkey")).items(), key=lambda kv: float(kv[0][1])):
        t, f = names[task], float(fkey) / 100
        if abs(f - 1 / 24) < 1e-4:
            budget = {"budget_blocks": 1}
        else:
            budget = {"budget_frac": min(fracs, key=lambda x: abs(x - f))}
        run = RunCfg(base=base, stage="sparsity", task=t, context_id=ctx[t], granularity="weight", lr=s["lr"], **budget)
        specs[t].append({"label": f"weights {float(fkey):g}%", "run": run.model_dump()})
    for scope in ("all", "eligible"):
        b = base.model_copy(deep=True)
        b.model.full_scope = scope
        for t in ctx:
            run = RunCfg(base=b, stage="granularity", task=t, context_id=ctx[t], granularity="full", lr=3e-6)
            specs[t].append({"label": f"full ({scope})", "run": run.model_dump()})
    return specs


@app.local_entrypoint()
def main(stage: str = "prep", sweep: str = "configs/sweep.yaml", overrides: str = "",
         dry_run: bool = False, limit: int = 0, force: bool = False, tasks: str = "", spawn: bool = False,
         run_ids: str = MMLU_CHECK_RUNS):
    # Only sfcd.config is imported locally (needs pydantic + pyyaml, not torch).
    from sfcd.config import expand_stage, load_base, load_sweep

    sw = load_sweep(sweep)
    if tasks:  # e.g. --tasks knowledge,base_arith restricts every stage to these tasks
        sw.tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    base = load_base(sw.base_config, [o for o in overrides.split(",") if o])
    bd = base.model_dump()
    cands = [(t, c) for t in sw.tasks for c in range(base.data.n_candidates)]

    if stage in ("lr_tune", "main", "pilot", "sparsity", "range_check", "granularity"):
        state = {} if dry_run else read_state.remote()
        runs = expand_stage(stage, sw, base, state)
        if limit:
            runs = runs[:limit]
        print(f"{stage}: {len(runs)} runs")
        if dry_run:
            for r in runs:
                print("  ", r.name(), r.run_id())
            return
        sha = _git_sha()
        payload = [r.model_dump() for r in runs]
        if spawn:  # submit everything and return; pair with `modal run --detach` so a sleeping laptop doesn't matter
            train_eval.spawn_map(payload, kwargs={"git_sha": sha, "force": force})
            print(f"submitted {len(runs)} runs; results land on the volume (results/{stage}/) and in W&B.")
            print("check progress with: modal run modal_app.py --stage progress")
            return
        _report(list(train_eval.map(payload, kwargs={"git_sha": sha, "force": force}, return_exceptions=True)))
        return

    if stage == "cost_bench":  # spawned, one container per task; results in diagnostics/cost_bench_<task>.json
        for t, sp in _cost_bench_specs(base).items():
            print(t, [x["label"] + " lr=" + str(x["run"]["lr"]) for x in sp])
            if not dry_run:
                print("  spawned:", cost_bench.spawn(t, sp, 3).object_id)
        return
    if dry_run:
        print(f"stage {stage}; candidates: {cands}")
        return

    if stage == "prep_spawn":  # server-side prep; result in diagnostics/prep_<tasks>.json
        print("spawned prep:", prep_remote.spawn(bd, sw.tasks).object_id)
        return
    if stage in ("data", "prep"):
        build_data.remote(bd, sw.tasks)
    if stage in ("teacher", "prep"):
        _report(list(teacher.starmap([(bd, t, c) for t, c in cands], return_exceptions=True)))
    if stage in ("filter", "prep"):
        print("kept:", filter_contexts.remote(bd, sw.tasks))
    if stage in ("mmlu", "prep"):
        print("mmlu base:", mmlu_base.remote(bd))
    if stage in ("score", "prep"):
        kept = read_state.remote()["kept"] or {}
        jobs = [(bd, t, c) for t, cs in kept.items() for c in cs]
        _report(list(score.starmap(jobs, return_exceptions=True)))
    if stage == "reeval":
        kept = read_state.remote()["kept"] or {}
        jobs = [(t, c) for t, cs in kept.items() for c in cs]
        if spawn:
            reeval.spawn_map([j[0] for j in jobs], [j[1] for j in jobs])
            print(f"submitted {len(jobs)} re-eval jobs")
            return
        _report(list(reeval.starmap(jobs, return_exceptions=True)))
    if stage == "rescore_mmlu":
        kept = read_state.remote()["kept"] or {}
        jobs = [(t, c) for t, cs in kept.items() for c in cs]
        if spawn:
            rescore_mmlu.spawn_map([j[0] for j in jobs], [j[1] for j in jobs])
            print(f"submitted {len(jobs)} MMLU rescoring jobs")
            return
        _report(list(rescore_mmlu.starmap(jobs, return_exceptions=True)))
    if stage == "progress":
        print(progress.remote())
    if stage == "env_check":  # spawned; results in diagnostics/ on the volume
        kept = read_state.remote()["kept"] or {}
        jobs = [(t, c) for t, cs in kept.items() for c in cs]
        env_check.spawn_map([bd] * len(jobs), [j[0] for j in jobs], [j[1] for j in jobs])
        print("spawned env checks:", jobs, "and mmlu batching check", mmlu_batch_check.spawn(bd).object_id)
        return
    if stage == "rebaseline":  # recompute baselines in the current environment for the given tasks
        kept = read_state.remote()["kept"] or {}
        jobs = [(t, c) for t in sw.tasks for c in kept.get(t, [])]
        env_check.spawn_map([bd] * len(jobs), [j[0] for j in jobs], [j[1] for j in jobs], [True] * len(jobs))
        print("spawned rebaseline:", jobs)
        return
    if stage == "math_bench":  # spawned; result in diagnostics/math_bench.json
        print("spawned MATH benchmark:", math_bench.spawn(bd).object_id)
        return
    if stage == "decoding_check":  # spawned; result in diagnostics/math_bench_decoding.json
        names = ["greedy", "qwen_recommended", "qwen_benchmark", "greedy_presence"]
        print("spawned decoding check:", math_bench.spawn(bd, names, 5).object_id)
        return
    if stage == "delta_norms":  # spawned; result in diagnostics/delta_norms_sparsity.json
        print("spawned:", delta_norms.spawn("sparsity").object_id)
        return
    if stage == "delta_norms_granularity":  # spawned; result in diagnostics/delta_norms_granularity.json
        print("spawned:", delta_norms.spawn("granularity").object_id)
        return
    if stage in ("math_bench_each", "math_bench_each_granularity"):  # spawned, one container per student
        ids = [r.strip() for r in run_ids.split(",") if r.strip()]
        run_stage = "granularity" if stage.endswith("granularity") else "sparsity"
        math_bench_run.spawn_map(ids, kwargs={"stage": run_stage})
        print(f"spawned {len(ids)} MATH checks")
        return
    if stage == "math_bench_students":  # spawned; --run-ids = students; result in diagnostics/
        ids = [r.strip() for r in run_ids.split(",") if r.strip()]
        print("spawned:", math_bench_students.spawn(ids).object_id)
        return
    if stage == "math_check":  # spawned; result in diagnostics/math_check.json
        print("spawned math check:", math_check.spawn(bd, [0, 1, 2]).object_id)
        return
    if stage == "select_by_val":
        print("chosen lrs:", select_by_val.remote())
    if stage == "select_lr":
        print("best lr:", select_lr.remote(sw.lr_tune.max_mmlu_drop))
    if stage == "aggregate":
        out = Path("results")
        out.mkdir(exist_ok=True)
        for name, text in aggregate.remote().items():
            (out / name).write_text(text)
        print("wrote", [str(out / n) for n in ("runs.csv", "summary.csv")])
    if stage == "diagnose":
        import json

        out = diagnose_forgetting.remote(bd, sw.tasks[0], 0)
        Path("results").mkdir(exist_ok=True)
        Path(f"results/forgetting_{sw.tasks[0]}.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    if stage == "mmlu_check":
        import json

        ids = [r.strip() for r in run_ids.split(",") if r.strip()]
        if spawn:  # result lands on the volume at diagnostics/mmlu_check.json
            print("spawned", mmlu_check.spawn(ids).object_id)
            return
        out = mmlu_check.remote(ids)
        Path("results").mkdir(exist_ok=True)
        Path("results/mmlu_check.json").write_text(json.dumps(out, indent=2))
        print("wrote results/mmlu_check.json")
    if stage == "probe":
        import json

        rows = probe_tasks.remote(bd, PROBE_SPECS, [0, 1], 50)
        Path("results").mkdir(exist_ok=True)
        Path("results/probe.json").write_text(json.dumps(rows, indent=2))
        for r in rows:
            print(f"{r['variant']:20s} c{r['context_id']}  with context={r['icl']:.2f}  without={r['zero_shot']:.2f}  gap={r['gap']:+.2f}")
    if stage == "probe_math_methods":  # spawned; result in diagnostics/probe_math_methods.json
        print("spawned:", probe_math_methods.spawn(bd).object_id)
        return
    if stage == "probe_method":
        import json

        rows = probe_tasks.remote(bd, [("method_math", "method_math", {})], [0, 1, 2, 3, 4], 50)
        Path("results").mkdir(exist_ok=True)
        Path("results/probe_method.json").write_text(json.dumps(rows, indent=2))
        from sfcd.tasks.method_math import FAMILIES as fams
        for r in rows:
            print(f"{fams[r['context_id']]:15s} with context={r['icl']:.2f}  without={r['zero_shot']:.2f}  gap={r['gap']:+.2f}")
    if stage == "overlap":
        kept = read_state.remote()["kept"] or {}
        ctxs = [(t, c) for t, cs in kept.items() for c in cs]
        Path("results").mkdir(exist_ok=True)
        Path("results/overlap.csv").write_text(overlap.remote(ctxs))
        print("wrote results/overlap.csv")
