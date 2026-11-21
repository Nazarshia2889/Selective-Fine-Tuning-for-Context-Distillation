"""Idempotent pipeline stages. Each reads/writes artifacts under a Paths root, so the same code runs
locally (./outputs/...) and on Modal (a Volume mount). Every stage skips work already on disk.

    build_data -> teacher -> filter -> mmlu_base -> score -> train_eval (lr_tune, then main) -> aggregate
"""
from __future__ import annotations

import gc
import hashlib
import logging
import os
import time

import torch

from sfcd.config import BaseCfg, RunCfg
from sfcd.data import batches, make_sft_examples
from sfcd.evaluate import eval_task, load_mmlu_subset, mmlu_eval, recovery
from sfcd.model import eligible_linears, get_device, load_model
from sfcd.paths import Paths, read_json, read_jsonl, write_json, write_jsonl
from sfcd.tasks import Context, Example, get_task

log = logging.getLogger("sfcd")


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_context(paths: Paths, task: str, cid: int) -> tuple[Context, dict[str, list[Example]]]:
    d = paths.ctx_dir(task, cid)
    ctx = Context.from_dict(read_json(d / "context.json"))
    splits = {s: [Example.from_dict(r) for r in read_jsonl(d / f"{s}.jsonl")] for s in ("train", "val", "test")}
    return ctx, splits


# ----------------------------------------------------------------------------- data + teacher


def stage_build_data(paths: Paths, base: BaseCfg, tasks: list[str], task_factory=get_task) -> None:
    for name in tasks:
        task = task_factory(name)
        for cid in range(base.data.n_candidates):
            d = paths.ctx_dir(name, cid)
            if (d / "test.jsonl").exists():
                continue
            ctx, splits = task.build(cid, base.data)
            write_json(d / "context.json", ctx.to_dict())
            for s, exs in splits.items():
                write_jsonl(d / f"{s}.jsonl", [e.to_dict() for e in exs])
            log.info("built %s c%d (%s)", name, cid, {s: len(v) for s, v in splits.items()})


def stage_teacher(paths: Paths, base: BaseCfg, task_name: str, cid: int, model_tok=None) -> dict:
    """Teacher responses on train queries (with context) + zero-shot / ICL baselines on val & test."""
    d = paths.ctx_dir(task_name, cid)
    if (d / "baselines.json").exists() and (d / "teacher.jsonl").exists():
        return read_json(d / "baselines.json")
    task = get_task(task_name)
    ctx, splits = load_context(paths, task_name, cid)
    model, tok = model_tok or load_model(base.model)

    t0 = time.time()
    _, responses = eval_task(model, tok, task, splits["train"], ctx.text, base.eval, base.model)
    gen_time = time.time() - t0
    write_jsonl(d / "teacher.jsonl", [
        {"query": e.query, "answer": e.answer, "teacher_response": r, "teacher_correct": task.is_correct(r, e)}
        for e, r in zip(splits["train"], responses)
    ])
    out = {"teacher_train_acc": task.accuracy(responses, splits["train"]), "teacher_gen_time_s": gen_time}
    for split in ("val", "test"):
        out[f"zero_shot_{split}"], _ = eval_task(model, tok, task, splits[split], None, base.eval, base.model)
        out[f"icl_{split}"], _ = eval_task(model, tok, task, splits[split], ctx.text, base.eval, base.model)
    write_json(d / "baselines.json", out)
    log.info("teacher %s c%d: %s", task_name, cid, out)
    return out


def stage_filter(paths: Paths, base: BaseCfg, tasks: list[str]) -> dict:
    kept, details = {}, {}
    for t in tasks:
        kept[t] = []
        for cid in range(base.data.n_candidates):
            b = read_json(paths.ctx_dir(t, cid) / "baselines.json")
            gap = b["icl_val"] - b["zero_shot_val"]
            details[f"{t}/c{cid}"] = {"gap_val": gap, **b}
            if gap >= base.data.min_gap and len(kept[t]) < base.data.n_keep:
                kept[t].append(cid)
        if len(kept[t]) < base.data.n_keep:
            log.warning("task %s: only %d/%d contexts pass the %.0f%% gap filter",
                        t, len(kept[t]), base.data.n_keep, 100 * base.data.min_gap)
    write_json(paths.kept(), {"kept": kept, "details": details})
    return kept


def stage_mmlu_base(paths: Paths, base: BaseCfg, model_tok=None) -> dict | None:
    if paths.mmlu_base().exists():
        return read_json(paths.mmlu_base())
    if base.eval.mmlu_n <= 0:
        return None
    model, tok = model_tok or load_model(base.model)
    r = mmlu_eval(model, tok, load_mmlu_subset(base.eval.mmlu_n, base.eval.mmlu_seed), base.eval, base.model)
    out = {"mmlu": r["acc"], "n": base.eval.mmlu_n, "seed": base.eval.mmlu_seed}
    write_json(paths.mmlu_base(), out)
    return out


# ----------------------------------------------------------------------------- selection scores


def _sft_examples(paths: Paths, base: BaseCfg, task_name: str, cid: int, tok) -> list[dict]:
    records = read_jsonl(paths.ctx_dir(task_name, cid) / "teacher.jsonl")
    return make_sft_examples(tok, get_task(task_name), records, base.model.max_seq_len)


def stage_score(paths: Paths, base: BaseCfg, task_name: str, cid: int, model_tok=None) -> None:
    from sfcd.selection.scoring import compute_grad_scores

    out = paths.scores(task_name, cid)
    if out.exists():
        return
    model, tok = model_tok or load_model(base.model)
    examples = _sft_examples(paths, base, task_name, cid, tok)
    sc = base.selection
    it = batches(examples, sc.score_batch_size, tok.pad_token_id, seed=sc.random_seed)
    scores, meta = compute_grad_scores(model, eligible_linears(model), it, sc.score_batches, base.model,
                                       sc.micro_batch_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({"scores": scores, "meta": meta}, tmp)
    tmp.replace(out)
    log.info("scores %s c%d: %s", task_name, cid, meta)


# ----------------------------------------------------------------------------- train + eval


def _random_seed(run: RunCfg) -> int:
    key = f"{run.task}|{run.context_id}|{run.granularity}|{run.budget_blocks}|{run.base.selection.random_seed}"
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def _git_sha() -> str | None:
    if os.environ.get("SFCD_GIT_SHA"):
        return os.environ["SFCD_GIT_SHA"]
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def stage_train_eval(paths: Paths, run: RunCfg, force: bool = False) -> dict:
    from sfcd.selection.selectors import random_scores, select
    from sfcd.sparse.params import apply_selection, count_params, delta_nbytes, export_delta, merge_into_base
    from sfcd.train import train_sft

    rid = run.run_id()
    out_path = paths.result(run.stage, rid)
    if out_path.exists() and not force:
        return read_json(out_path)
    base = run.base
    task = get_task(run.task)
    device = get_device()
    _, splits = load_context(paths, run.task, run.context_id)
    baselines = read_json(paths.ctx_dir(run.task, run.context_id) / "baselines.json")

    model, tok = load_model(base.model, device)
    linears = eligible_linears(model)
    n_eligible = sum(li.numel for li in linears)
    n_total = sum(p.numel() for p in model.parameters())

    # --- selection
    mask, sel_meta = None, {}
    if run.granularity != "full":
        t0 = time.time()
        if run.selector == "random":
            scores = random_scores(linears, _random_seed(run))
        else:
            blob = torch.load(paths.scores(run.task, run.context_id), map_location="cpu")
            scores, sel_meta = blob["scores"], blob["meta"]
        mask = select(scores, linears, run.granularity, run.budget_blocks, device=device)
        del scores
        sel_meta["select_time_s"] = time.time() - t0
        _free()

    trainable = apply_selection(model, linears, run.granularity, mask, base.model.full_scope)
    n_trainable = count_params(trainable)

    # --- logging
    wb = None
    if base.wandb.mode != "disabled":
        import wandb

        wb = wandb.init(project=base.wandb.project, entity=base.wandb.entity, mode=base.wandb.mode,
                        name=run.name(), group=f"{run.stage}/{run.task}", id=f"{run.stage}-{rid}",
                        resume="allow", config={**run.model_dump(), "run_id": rid}, reinit=True)

    examples = _sft_examples(paths, base, run.task, run.context_id, tok)
    stats = train_sft(model, trainable, examples, tok.pad_token_id, base.train, run.lr, base.model,
                      log=(lambda d: wb.log(d)) if wb else None)

    # --- storage
    delta = export_delta(model, run.granularity, trainable)
    nbytes = delta_nbytes(delta)
    if base.save_delta:
        from safetensors.torch import save_file

        p = paths.delta(run.stage, rid)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file(delta, str(p))
    del delta, trainable
    _free()

    # --- eval
    merge_into_base(model)
    test_acc, test_preds = eval_task(model, tok, task, splits["test"], None, base.eval, base.model)
    val_acc, _ = eval_task(model, tok, task, splits["val"], None, base.eval, base.model)
    mm = {"acc": None}
    if base.eval.mmlu_n > 0:
        mm = mmlu_eval(model, tok, load_mmlu_subset(base.eval.mmlu_n, base.eval.mmlu_seed), base.eval, base.model)
    mb = read_json(paths.mmlu_base()) if paths.mmlu_base().exists() else {}
    drop = lambda b, a: (b - a) if a is not None and b is not None else None

    result = {
        "run_id": rid, "name": run.name(), "config": run.model_dump(), "git_sha": _git_sha(),
        "mask_checksum": mask.checksum() if mask else None,
        "mask_blocks": mask.blocks if mask and mask.granularity == "block" else None,
        "n_total_params": n_total, "n_eligible_params": n_eligible, "n_trainable_params": n_trainable,
        "frac_eligible_trained": n_trainable / n_eligible,
        **{f"sel_{k}": v for k, v in sel_meta.items()},
        **stats,
        "delta_bytes": nbytes,
        "student_test": test_acc, "student_val": val_acc,
        "zero_shot_test": baselines["zero_shot_test"], "icl_test": baselines["icl_test"],
        "recovery_test": recovery(test_acc, baselines["zero_shot_test"], baselines["icl_test"]),
        "mmlu": mm["acc"], "mmlu_base": mb.get("mmlu"),
        "mmlu_forgetting": drop(mb.get("mmlu"), mm["acc"]),
        "sample_preds": [{"query": e.query, "gold": e.answer, "pred": p}
                         for e, p in list(zip(splits["test"], test_preds))[:10]],
    }
    write_json(out_path, result)
    if wb:
        wb.summary.update({k: v for k, v in result.items() if isinstance(v, (int, float)) and v is not None})
        wb.finish()
    del model
    _free()
    return result


def stage_select_lr(paths: Paths, max_mmlu_drop: float = 0.02) -> dict:
    """Per granularity: the lr with the best no-context val accuracy among lrs whose MMLU drop is at
    most max_mmlu_drop. If none qualifies, fall back to the lr that forgets least."""
    by_gran: dict[str, list[dict]] = {}
    d = paths.results_dir("lr_tune")
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        r = read_json(f)
        by_gran.setdefault(r["config"]["granularity"], []).append(r)
    best, report = {}, {}
    for g, rs in by_gran.items():
        forget = lambda r: r.get("mmlu_forgetting") or 0.0
        ok = [r for r in rs if forget(r) <= max_mmlu_drop]
        if ok:
            pick = max(ok, key=lambda r: (r["student_val"], -forget(r)))
        else:
            pick = min(rs, key=forget)
            log.warning("granularity %s: no lr keeps the MMLU drop <= %.3f; using least-forgetting lr", g, max_mmlu_drop)
        best[g] = pick["config"]["lr"]
        report[g] = [{"lr": r["config"]["lr"], "val": r["student_val"], "mmlu_drop": forget(r),
                      "eligible": r in ok} for r in sorted(rs, key=lambda r: r["config"]["lr"])]
    write_json(paths.best_lr(), best)
    write_json(paths.results_dir() / "lr_tune_report.json", {"max_mmlu_drop": max_mmlu_drop, "per_granularity": report})
    return best


def read_state(paths: Paths) -> dict:
    """Small summary the launcher needs to expand the main sweep."""
    kept = read_json(paths.kept())["kept"] if paths.kept().exists() else None
    best_lr = read_json(paths.best_lr()) if paths.best_lr().exists() else None
    return {"kept": kept, "best_lr": best_lr}
