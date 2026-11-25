"""Teacher-only probe: can the model do a task *with* the context (and not without)? Nothing is written
to the data folder; used to pick task variants before running the real pipeline."""
from __future__ import annotations

from sfcd.config import BaseCfg
from sfcd.evaluate import eval_task
from sfcd.model import load_model


def probe(base: BaseCfg, tasks: dict, context_ids: list[int], n: int = 50) -> list[dict]:
    """tasks: label -> Task instance. Evaluates on the first n val queries of each context."""
    model, tok = load_model(base.model)
    rows = []
    for label, task in tasks.items():
        for cid in context_ids:
            ctx, splits = task.build(cid, base.data)
            exs = splits["val"][:n]
            icl, icl_preds = eval_task(model, tok, task, exs, ctx.text, base.eval, base.model)
            zero, zero_preds = eval_task(model, tok, task, exs, None, base.eval, base.model)
            unfinished = lambda p: len(tok(p, add_special_tokens=False)["input_ids"]) >= task.max_new_tokens - 2
            zw = [p for p, e in zip(zero_preds, exs) if not task.is_correct(p, e)]
            iw = [p for p, e in zip(icl_preds, exs) if not task.is_correct(p, e)]
            rows.append({"variant": label, "context_id": cid, "family": ctx.meta.get("family"),
                         "icl": icl, "zero_shot": zero, "gap": icl - zero,
                         "zero_shot_unfinished_share_of_errors": sum(map(unfinished, zw)) / max(1, len(zw)),
                         "icl_unfinished_share_of_errors": sum(map(unfinished, iw)) / max(1, len(iw)),
                         "example_query": exs[0].query, "example_gold": exs[0].answer,
                         "example_pred": icl_preds[0][-500:],
                         "zero_shot_wrong_examples": [{"query": e.query, "gold": e.answer, "pred_tail": p[-400:]}
                                                      for p, e in zip(zero_preds, exs)
                                                      if not task.is_correct(p, e)][:3]})
            print(rows[-1], flush=True)
    return rows
