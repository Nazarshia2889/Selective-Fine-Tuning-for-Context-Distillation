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
from sfcd.evaluate import MMLU_VERSION, eval_task, load_mmlu_subset, mmlu_eval, recovery
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


def candidate_ids(task_name: str, base: BaseCfg) -> list[int]:
    ids = get_task(task_name).candidate_ids
    return list(ids) if ids is not None else list(range(base.data.n_candidates))


def n_keep(task_name: str, base: BaseCfg) -> int:
    k = get_task(task_name).n_keep
    return k if k is not None else base.data.n_keep


def stage_build_data(paths: Paths, base: BaseCfg, tasks: list[str], task_factory=get_task) -> None:
    for name in tasks:
        task = task_factory(name)
        for cid in candidate_ids(name, base):
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
        out[f"trivial_{split}"] = task.trivial_baseline(splits[split])
    write_json(d / "baselines.json", out)
    log.info("teacher %s c%d: %s", task_name, cid, out)
    return out


def stage_filter(paths: Paths, base: BaseCfg, tasks: list[str]) -> dict:
    """Keep the n_keep candidate contexts with the largest val margin of the teacher over
    max(zero-shot, trivial baseline), among those with margin >= min_gap. Only the given tasks are
    refiltered; other tasks' entries in kept.json are kept."""
    prev = read_json(paths.kept()) if paths.kept().exists() else {"kept": {}, "details": {}}
    kept, details = dict(prev["kept"]), dict(prev["details"])
    for t in tasks:
        gaps = {}
        for cid in candidate_ids(t, base):
            b = read_json(paths.ctx_dir(t, cid) / "baselines.json")
            gap = b["icl_val"] - max(b["zero_shot_val"], b.get("trivial_val", 0.0))
            details[f"{t}/c{cid}"] = {"gap_val": gap, **b}
            if gap >= base.data.min_gap:
                gaps[cid] = gap
        kept[t] = sorted(sorted(gaps, key=lambda c: (-gaps[c], c))[:n_keep(t, base)])
        if len(kept[t]) < n_keep(t, base):
            log.warning("task %s: only %d/%d contexts pass the %.0f%% gap filter",
                        t, len(kept[t]), n_keep(t, base), 100 * base.data.min_gap)
    write_json(paths.kept(), {"kept": kept, "details": details})
    return kept


def stage_mmlu_base(paths: Paths, base: BaseCfg, model_tok=None) -> dict | None:
    if paths.mmlu_base().exists():
        old = read_json(paths.mmlu_base())
        if old.get("mmlu_version") == MMLU_VERSION:
            return old
    if base.eval.mmlu_n <= 0:
        return None
    model, tok = model_tok or load_model(base.model)
    r = mmlu_eval(model, tok, load_mmlu_subset(base.eval.mmlu_n, base.eval.mmlu_seed), base.eval, base.model)
    out = {"mmlu": r["acc"], "mmlu_letter": r["acc_letter"], "mmlu_version": MMLU_VERSION,
           "n": base.eval.mmlu_n, "seed": base.eval.mmlu_seed}
    write_json(paths.mmlu_base(), out)
    return out


# ----------------------------------------------------------------------------- selection scores


def _sft_examples(paths: Paths, base: BaseCfg, task_name: str, cid: int, tok) -> list[dict]:
    records = read_jsonl(paths.ctx_dir(task_name, cid) / "teacher.jsonl")
    return make_sft_examples(tok, get_task(task_name), records, base.model.max_seq_len)


def answer_tokens(ex: dict) -> int:
    return sum(l != -100 for l in ex["labels"])


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
    budget = run.budget_blocks if run.budget_frac is None else f"f{run.budget_frac:g}"
    key = f"{run.task}|{run.context_id}|{run.granularity}|{budget}|{run.base.selection.random_seed}"
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
        mask = select(scores, linears, run.granularity, run.budget_blocks, device=device, budget_frac=run.budget_frac)
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
    mm = {"acc": None, "acc_letter": None}
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
        "n_train_examples": len(examples), "n_train_answer_tokens": sum(answer_tokens(e) for e in examples),
        "student_test": test_acc, "student_val": val_acc,
        "zero_shot_test": baselines["zero_shot_test"], "icl_test": baselines["icl_test"],
        "recovery_test": recovery(test_acc, baselines["zero_shot_test"], baselines["icl_test"],
                                  baselines.get("trivial_test", 0.0)),
        "trivial_test": baselines.get("trivial_test", 0.0),
        "mmlu": mm["acc"], "mmlu_base": mb.get("mmlu"),
        "mmlu_forgetting": drop(mb.get("mmlu"), mm["acc"]),
        "mmlu_letter": mm["acc_letter"], "mmlu_base_letter": mb.get("mmlu_letter"), "mmlu_version": MMLU_VERSION,
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


def _forget(r: dict) -> float:
    return r.get("mmlu_forgetting") or 0.0


def pick_lr(rs: list[dict], max_mmlu_drop: float) -> tuple[dict, bool]:
    """Best no-context val accuracy among runs whose MMLU drop is at most max_mmlu_drop; if none
    qualifies, the run that forgets least. Returns (run, whether any run qualified)."""
    ok = [r for r in rs if _forget(r) <= max_mmlu_drop]
    if ok:
        return max(ok, key=lambda r: (r["student_val"], -_forget(r))), True
    return min(rs, key=_forget), False


def _lr_report(rs: list[dict], max_mmlu_drop: float) -> list[dict]:
    return [{"lr": r["config"]["lr"], "val": r["student_val"], "mmlu_drop": _forget(r),
             "eligible": _forget(r) <= max_mmlu_drop} for r in sorted(rs, key=lambda r: r["config"]["lr"])]


def stage_select_lr(paths: Paths, max_mmlu_drop: float = 0.02) -> dict:
    """Per granularity, on the lr-tune pilot: see pick_lr."""
    by_gran: dict[str, list[dict]] = {}
    d = paths.results_dir("lr_tune")
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        r = read_json(f)
        by_gran.setdefault(r["config"]["granularity"], []).append(r)
    best, report = {}, {}
    for g, rs in by_gran.items():
        pick, ok = pick_lr(rs, max_mmlu_drop)
        if not ok:
            log.warning("granularity %s: no lr keeps the MMLU drop <= %.3f; using least-forgetting lr", g, max_mmlu_drop)
        best[g] = pick["config"]["lr"]
        report[g] = _lr_report(rs, max_mmlu_drop)
    write_json(paths.best_lr(), best)
    write_json(paths.results_dir() / "lr_tune_report.json", {"max_mmlu_drop": max_mmlu_drop, "per_granularity": report})
    return best


def budget_key(cfg: dict) -> str:
    return f"f{cfg['budget_frac']:g}" if cfg.get("budget_frac") is not None else f"b{cfg['budget_blocks']}"


def method_key(cfg: dict) -> str:
    """Everything that identifies a method except the lr and the context."""
    return f"{cfg['task']}|{cfg['granularity']}|{cfg['selector']}|{budget_key(cfg) if cfg['granularity'] != 'full' else 'full'}"


def stage_select_by_val(paths: Paths, stages=("sparsity", "granularity")) -> dict:
    """Per method (task, granularity, selector, budget): the lr with the best mean no-context val accuracy
    over that task's contexts. Only lrs run on every context of the method compete. MMLU is not used."""
    groups: dict[str, dict[float, dict[int, dict]]] = {}
    for stage in stages:
        d = paths.results_dir(stage)
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            r = read_json(f)
            c = r["config"]
            groups.setdefault(method_key(c), {}).setdefault(c["lr"], {})[c["context_id"]] = {**r, "_stage": stage}
    best, report = {}, {}
    for k, by_lr in sorted(groups.items()):
        n_ctx = max(len(v) for v in by_lr.values())
        full = {lr: v for lr, v in by_lr.items() if len(v) == n_ctx}
        mean_val = {lr: sum(r["student_val"] for r in v.values()) / n_ctx for lr, v in full.items()}
        lr_best = max(mean_val, key=lambda lr: (mean_val[lr], -lr))  # ties -> smaller lr
        grid = sorted(by_lr)
        best[k] = lr_best
        report[k] = {"lr": lr_best, "n_contexts": n_ctx, "at_grid_edge": lr_best in (grid[0], grid[-1]),
                     "mean_val": mean_val,
                     "runs": {cid: {"stage": r["_stage"], "run_id": r["run_id"]} for cid, r in full[lr_best].items()}}
    write_json(paths.results_dir() / "selected_lr.json", {"lr": best, "report": report})
    return best


def read_state(paths: Paths) -> dict:
    """Small summary the launcher needs to expand the main sweep."""
    kept = read_json(paths.kept())["kept"] if paths.kept().exists() else None
    best_lr = read_json(paths.best_lr()) if paths.best_lr().exists() else None
    return {"kept": kept, "best_lr": best_lr}


# ----------------------------------------------------------------------------- re-evaluation

EVAL_VERSION = 2  # v2: generation stops at <|im_end|> (v1 let students run on past it)


def apply_saved_delta(model, granularity: str, delta: dict[str, torch.Tensor]) -> None:
    """Inverse of sparse.params.export_delta, applied to a plain (unwrapped) model in place."""
    with torch.no_grad():
        if granularity in ("weight", "row"):
            suffix = ".idx" if granularity == "weight" else ".rows"
            for key, idx in delta.items():
                if not key.endswith(suffix):
                    continue
                name = key[: -len(suffix)]
                w = model.get_submodule(name).weight
                d = delta[f"{name}.delta"].to(w.device, w.dtype)
                idx = idx.to(w.device, torch.long)
                (w.view(-1) if granularity == "weight" else w).index_add_(0, idx, d)
        else:
            params = dict(model.named_parameters())
            for name, v in delta.items():
                params[name].copy_(v.to(params[name].device, params[name].dtype))


def stage_reeval_context(paths: Paths, task_name: str, cid: int, stage: str = "main") -> dict:
    """Recompute baselines and every saved run of one context with the current eval code (no training)."""
    from safetensors.torch import load_file

    files = []
    for f in sorted(paths.results_dir(stage).glob("*.json")):
        r = read_json(f)
        if r["config"]["task"] == task_name and r["config"]["context_id"] == cid:
            files.append((f, r))
    if not files:
        return {"task": task_name, "context_id": cid, "n": 0}
    base = BaseCfg.model_validate(files[0][1]["config"]["base"])
    task = get_task(task_name)
    ctx, splits = load_context(paths, task_name, cid)
    model, tok = load_model(base.model)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}

    # baselines (teacher training answers are left untouched)
    bpath = paths.ctx_dir(task_name, cid) / "baselines.json"
    b = read_json(bpath)
    if b.get("eval_version") != EVAL_VERSION:
        b_old = {k: b[k] for k in ("zero_shot_val", "icl_val", "zero_shot_test", "icl_test")}
        for split in ("val", "test"):
            b[f"zero_shot_{split}"], _ = eval_task(model, tok, task, splits[split], None, base.eval, base.model)
            b[f"icl_{split}"], _ = eval_task(model, tok, task, splits[split], ctx.text, base.eval, base.model)
        b["v1"], b["eval_version"] = b_old, EVAL_VERSION
        write_json(bpath, b)

    done = 0
    for f, r in files:
        if r.get("eval_version") == EVAL_VERSION:
            continue
        g = r["config"]["granularity"]
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(orig[n])
        apply_saved_delta(model, g, load_file(str(paths.delta(stage, r["run_id"]))))
        test_acc, test_preds = eval_task(model, tok, task, splits["test"], None, base.eval, base.model)
        val_acc, _ = eval_task(model, tok, task, splits["val"], None, base.eval, base.model)
        r["v1"] = {k: r.get(k) for k in ("student_test", "student_val", "zero_shot_test", "icl_test", "recovery_test")}
        r.update({
            "student_test": test_acc, "student_val": val_acc,
            "zero_shot_test": b["zero_shot_test"], "icl_test": b["icl_test"],
            "recovery_test": recovery(test_acc, b["zero_shot_test"], b["icl_test"]),
            "sample_preds": [{"query": e.query, "gold": e.answer, "pred": p}
                             for e, p in list(zip(splits["test"], test_preds))[:10]],
            "eval_version": EVAL_VERSION,
        })
        write_json(f, r)
        done += 1
        log.info("re-eval %s: test %.3f (was %.3f)", r["name"], test_acc, r["v1"]["student_test"])
    return {"task": task_name, "context_id": cid, "n": len(files), "reevaluated": done,
            "zero_shot_test": b["zero_shot_test"], "icl_test": b["icl_test"], "baseline_v1": b.get("v1")}


def stage_rescore_mmlu(paths: Paths, task_name: str, cid: int, stages=("lr_tune", "main")) -> dict:
    """Rescore MMLU (option-text) for every saved run of one context, from its saved delta. Old score kept
    as mmlu_letter. The base-model score is recomputed too (deterministic, so concurrent writers agree)."""
    from safetensors.torch import load_file

    files = []
    for stage in stages:
        d = paths.results_dir(stage)
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            r = read_json(f)
            if r["config"]["task"] == task_name and r["config"]["context_id"] == cid:
                files.append((stage, f, r))
    if not files:
        return {"task": task_name, "context_id": cid, "n": 0}
    base = BaseCfg.model_validate(files[0][2]["config"]["base"])
    model, tok = load_model(base.model)
    mb = stage_mmlu_base(paths, base, (model, tok))
    items = load_mmlu_subset(base.eval.mmlu_n, base.eval.mmlu_seed)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}
    done = 0
    for stage, f, r in files:
        if r.get("mmlu_version") == MMLU_VERSION:
            continue
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(orig[n])
        apply_saved_delta(model, r["config"]["granularity"], load_file(str(paths.delta(stage, r["run_id"]))))
        mm = mmlu_eval(model, tok, items, base.eval, base.model)
        r.update({"mmlu": mm["acc"], "mmlu_base": mb["mmlu"], "mmlu_forgetting": mb["mmlu"] - mm["acc"],
                  "mmlu_letter": r.get("mmlu_letter", r.get("mmlu")),
                  "mmlu_base_letter": mb.get("mmlu_letter"), "mmlu_version": MMLU_VERSION})
        write_json(f, r)
        done += 1
        log.info("mmlu %s: %.3f (letter %.3f)", r["name"], mm["acc"], r["mmlu_letter"])
    return {"task": task_name, "context_id": cid, "n": len(files), "rescored": done, "mmlu_base": mb}


def stage_env_check(paths: Paths, base: BaseCfg, task_name: str, cid: int, write: bool = False) -> dict:
    """Recompute the base model's no-context / with-context accuracy in the current environment and
    compare to the stored baselines. write=True replaces them (old values kept under "env1"), so new
    runs measure recovery against baselines from the same software environment."""
    b = read_json(paths.ctx_dir(task_name, cid) / "baselines.json")
    task = get_task(task_name)
    ctx, splits = load_context(paths, task_name, cid)
    model, tok = load_model(base.model)
    out = {"task": task_name, "context_id": cid}
    for split in ("val", "test"):
        out[f"zero_shot_{split}"], _ = eval_task(model, tok, task, splits[split], None, base.eval, base.model)
        out[f"icl_{split}"], _ = eval_task(model, tok, task, splits[split], ctx.text, base.eval, base.model)
        for k in (f"zero_shot_{split}", f"icl_{split}"):
            out[f"{k}_stored"] = b[k]
    if write and "env1" not in b:
        keys = ("zero_shot_val", "icl_val", "zero_shot_test", "icl_test")
        b["env1"] = {k: b[k] for k in keys}
        b.update({k: out[k] for k in keys})
        b.setdefault("trivial_val", task.trivial_baseline(splits["val"]))
        b.setdefault("trivial_test", task.trivial_baseline(splits["test"]))
        write_json(paths.ctx_dir(task_name, cid) / "baselines.json", b)
    return out


def mmlu_batching_check(paths: Paths, base: BaseCfg, n_items: int = 40) -> dict:
    """Batched vs one-at-a-time option-text scores on the real model, plus time for the full subset."""
    from sfcd.evaluate import option_text_scores

    model, tok = load_model(base.model)
    items = load_mmlu_subset(base.eval.mmlu_n, base.eval.mmlu_seed)
    a = option_text_scores(model, tok, items[:n_items], base.model)
    b = option_text_scores(model, tok, items[:n_items], base.model, batch_seqs=1)
    diff = max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    same_pick = sum(max(range(4), key=ra.__getitem__) == max(range(4), key=rb.__getitem__) for ra, rb in zip(a, b))
    t0 = time.time()
    r = mmlu_eval(model, tok, items, base.eval, base.model)
    return {"max_abs_diff": diff, "same_pick": f"{same_pick}/{n_items}", "full_mmlu_eval_s": time.time() - t0,
            "mmlu": r["acc"], "mmlu_letter": r["acc_letter"]}
