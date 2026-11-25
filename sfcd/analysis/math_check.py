"""Why does the no-context model fail method_math? Wrong math, or length/format?

For each kept math context, on the val split, with and without the context: answer length, whether
generation hit max_new_tokens, whether a final answer could be parsed, strict correctness (the task's
grader), and lenient correctness (the gold number is the last number in the text).
"""
from __future__ import annotations

import re

from sfcd.config import BaseCfg
from sfcd.model import build_prompt, generate, load_model
from sfcd.paths import Paths
from sfcd.tasks import get_task

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_PARSE = re.compile(r"Answer:\s*\**\s*\$?\s*-?\d|\\boxed\{")


def _stats(task, tok, preds, examples) -> dict:
    rows = []
    for p, e in zip(preds, examples):
        n_tok = len(tok(p, add_special_tokens=False)["input_ids"])
        nums = _NUM.findall(p.replace(",", ""))
        rows.append({"n_tokens": n_tok, "hit_limit": n_tok >= task.max_new_tokens - 2,
                     "parsed": bool(_PARSE.search(p)), "correct": task.is_correct(p, e),
                     "lenient": bool(nums) and nums[-1] == e.answer})
    n = len(rows)
    agg = {k: sum(r[k] for r in rows) / n for k in ("hit_limit", "parsed", "correct", "lenient")}
    agg["mean_tokens"] = sum(r["n_tokens"] for r in rows) / n
    wrong = [i for i, r in enumerate(rows) if not r["correct"]]
    agg["wrong_because"] = {
        "hit_limit": sum(rows[i]["hit_limit"] for i in wrong) / max(1, len(wrong)),
        "no_parsable_answer": sum(not rows[i]["parsed"] for i in wrong) / max(1, len(wrong)),
        "right_number_wrong_format": sum(rows[i]["lenient"] for i in wrong) / max(1, len(wrong)),
    }
    return agg, rows


def check(paths: Paths, base: BaseCfg, cids: list[int], n_samples: int = 4) -> dict:
    from sfcd.pipeline import load_context

    task = get_task("method_math")
    model, tok = load_model(base.model)
    out = {}
    for cid in cids:
        ctx, splits = load_context(paths, "method_math", cid)
        ex = splits["val"]
        res = {"family": ctx.meta.get("family")}
        for cond, text in (("no_context", None), ("with_context", ctx.text)):
            prompts = [build_prompt(tok, task, text, e.query) for e in ex]
            preds = generate(model, tok, prompts, task.max_new_tokens, base.eval.gen_batch_size, base.model)
            agg, rows = _stats(task, tok, preds, ex)
            wrong = [i for i, r in enumerate(rows) if not r["correct"]][:n_samples]
            res[cond] = {**agg, "samples_wrong": [{"query": ex[i].query, "gold": ex[i].answer,
                                                   "pred_tail": preds[i][-400:], **rows[i]} for i in wrong]}
        out[f"c{cid}"] = res
    return out
