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
        "torch==2.8.0", "transformers>=4.57", "accelerate", "datasets", "safetensors",
        "pydantic>=2", "pyyaml", "pandas", "wandb",
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


@app.function(volumes={ROOT: vol}, timeout=600)
def progress() -> str:
    paths = _setup()
    lines = []
    for stage in ("lr_tune", "main"):
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
def overlap(model_name: str, contexts: list[tuple[str, int]]) -> str:
    from sfcd.analysis.overlap import linears_from_config, overlap_table

    paths = _setup()
    df = overlap_table(paths, linears_from_config(model_name), [tuple(c) for c in contexts])
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


@app.local_entrypoint()
def main(stage: str = "prep", sweep: str = "configs/sweep.yaml", overrides: str = "",
         dry_run: bool = False, limit: int = 0, force: bool = False, tasks: str = "", spawn: bool = False):
    # Only sfcd.config is imported locally (needs pydantic + pyyaml, not torch).
    from sfcd.config import expand_stage, load_base, load_sweep

    sw = load_sweep(sweep)
    if tasks:  # e.g. --tasks knowledge,base_arith restricts every stage to these tasks
        sw.tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    base = load_base(sw.base_config, [o for o in overrides.split(",") if o])
    bd = base.model_dump()
    cands = [(t, c) for t in sw.tasks for c in range(base.data.n_candidates)]

    if stage in ("lr_tune", "main"):
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

    if dry_run:
        print(f"stage {stage}; candidates: {cands}")
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
    if stage == "progress":
        print(progress.remote())
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
    if stage == "probe":
        import json

        rows = probe_tasks.remote(bd, PROBE_SPECS, [0, 1], 50)
        Path("results").mkdir(exist_ok=True)
        Path("results/probe.json").write_text(json.dumps(rows, indent=2))
        for r in rows:
            print(f"{r['variant']:20s} c{r['context_id']}  with context={r['icl']:.2f}  without={r['zero_shot']:.2f}  gap={r['gap']:+.2f}")
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
        Path("results/overlap.csv").write_text(overlap.remote(base.model.name, ctxs))
        print("wrote results/overlap.csv")
