"""Forgetting diagnostic: train one context under several settings in a single container and measure
task accuracy, MMLU, MMLU letter distribution, and how far the weights moved."""
from __future__ import annotations

import time
from collections import Counter

import torch

from sfcd.config import BaseCfg
from sfcd.evaluate import eval_mmlu, eval_task, load_mmlu_subset
from sfcd.model import eligible_linears, load_model
from sfcd.paths import Paths
from sfcd.pipeline import _sft_examples, load_context
from sfcd.selection.selectors import random_scores, select
from sfcd.sparse.params import apply_selection, count_params, merge_into_base
from sfcd.tasks import get_task
from sfcd.train import train_sft

DEFAULT_CONFIGS = [
    {"label": "sanity: weight 4% zero update", "granularity": "weight", "selector": "topk", "lr": 0.0, "epochs": 0},
    {"label": "weight topk lr1e-5", "granularity": "weight", "selector": "topk", "lr": 1e-5, "epochs": 3},
    {"label": "weight topk lr3e-5", "granularity": "weight", "selector": "topk", "lr": 3e-5, "epochs": 3},
    {"label": "weight topk lr1e-4", "granularity": "weight", "selector": "topk", "lr": 1e-4, "epochs": 3},
    {"label": "weight topk lr3e-4 (test run)", "granularity": "weight", "selector": "topk", "lr": 3e-4, "epochs": 3},
    {"label": "weight topk lr3e-4 1 epoch", "granularity": "weight", "selector": "topk", "lr": 3e-4, "epochs": 1},
    {"label": "weight random lr3e-4", "granularity": "weight", "selector": "random", "lr": 3e-4, "epochs": 3},
    {"label": "row topk lr3e-4", "granularity": "row", "selector": "topk", "lr": 3e-4, "epochs": 3},
    {"label": "block topk lr1e-4", "granularity": "block", "selector": "topk", "lr": 1e-4, "epochs": 3},
    {"label": "full lr3e-5", "granularity": "full", "selector": "none", "lr": 3e-5, "epochs": 3},
]


def diagnose(paths: Paths, base: BaseCfg, task_name: str, cid: int, configs=None, budget_blocks: int = 1) -> dict:
    configs = configs or DEFAULT_CONFIGS
    task = get_task(task_name)
    _, splits = load_context(paths, task_name, cid)
    items = load_mmlu_subset(base.eval.mmlu_n, base.eval.mmlu_seed)
    model, tok = load_model(base.model)
    linears = eligible_linears(model)
    eligible = {li.param_name for li in linears}
    examples = _sft_examples(paths, base, task_name, cid, tok)
    grad_scores = torch.load(paths.scores(task_name, cid), map_location="cpu")["scores"]
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}

    # padding sanity: batched (left-padded) vs one-at-a-time MMLU on the base model
    sub = items[:100]
    acc_b, preds_b = eval_mmlu(model, tok, sub, base.eval, base.model, return_preds=True)
    acc_1, preds_1 = eval_mmlu(model, tok, sub, base.eval.model_copy(update={"mmlu_batch_size": 1}),
                               base.model, return_preds=True)
    padding = {"batched_acc": acc_b, "unbatched_acc": acc_1,
               "agreement": sum(a == b for a, b in zip(preds_b, preds_1)) / len(sub)}
    base_mmlu, base_preds = eval_mmlu(model, tok, items, base.eval, base.model, return_preds=True)
    gold_dist = Counter("ABCD"[it["answer"]] for it in items)

    rows = []
    for c in configs:
        t0 = time.time()
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(orig[n])
        mask = None
        if c["granularity"] != "full":
            sc = grad_scores if c["selector"] == "topk" else random_scores(linears, 0)
            mask = select(sc, linears, c["granularity"], budget_blocks, device=orig[linears[0].param_name].device)
        trainable = apply_selection(model, linears, c["granularity"], mask, base.model.full_scope)
        stats = {}
        if c["lr"] > 0 and c["epochs"] > 0:
            tcfg = base.train.model_copy(update={"epochs": c["epochs"]})
            stats = train_sft(model, trainable, examples, tok.pad_token_id, tcfg, c["lr"], base.model)
        n_trainable = count_params(trainable)
        del trainable
        merge_into_base(model)

        # how far did the weights move?
        num = den = changed_abs = changed_ref = 0.0
        n_changed = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in eligible:
                    continue
                d = (p - orig[n]).float()
                num += float(d.pow(2).sum())
                den += float(orig[n].float().pow(2).sum())
                nz = d != 0
                n_changed += int(nz.sum())
                changed_abs += float(d[nz].abs().sum())
                changed_ref += float(orig[n][nz].float().abs().sum())
        test_acc, _ = eval_task(model, tok, task, splits["test"], None, base.eval, base.model)
        mmlu, preds = eval_mmlu(model, tok, items, base.eval, base.model, return_preds=True)
        dist = Counter("ABCD"[p] for p in preds)
        rows.append({
            **c, "n_trainable": n_trainable, "final_loss": stats.get("final_loss"),
            "test_acc": test_acc, "mmlu": mmlu, "mmlu_drop": base_mmlu - mmlu,
            "agree_with_base": sum(a == b for a, b in zip(preds, base_preds)) / len(items),
            "top_letter_frac": max(dist.values()) / len(items),
            "letter_dist": dict(sorted(dist.items())),
            "rel_frob_change": (num / den) ** 0.5 if den else 0.0,
            "mean_rel_change_of_changed": changed_abs / changed_ref if changed_ref else 0.0,
            "n_weights_changed": n_changed,
            "seconds": time.time() - t0,
        })
        print(rows[-1], flush=True)
    return {"padding_check": padding, "base_mmlu": base_mmlu, "gold_letter_dist": dict(sorted(gold_dist.items())),
            "base_letter_dist": dict(sorted(Counter("ABCD"[p] for p in base_preds).items())), "runs": rows}
