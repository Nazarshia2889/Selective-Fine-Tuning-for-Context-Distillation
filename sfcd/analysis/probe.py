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
            zero, _ = eval_task(model, tok, task, exs, None, base.eval, base.model)
            rows.append({"variant": label, "context_id": cid, "icl": icl, "zero_shot": zero, "gap": icl - zero,
                         "example_query": exs[0].query, "example_gold": exs[0].answer,
                         "example_pred": icl_preds[0]})
            print(rows[-1], flush=True)
    return rows
