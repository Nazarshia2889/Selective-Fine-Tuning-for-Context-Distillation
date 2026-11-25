"""Where does the model fail at math on its own? MATH (Hendrycks et al. 2021) test problems, no context,
non-thinking mode, by subject and difficulty level.

Only problems whose reference answer is a single number (integer, fraction or decimal) are used, so
grading is exact. Reports accuracy, how often generation hit the token limit, and per-subject/level
breakdowns, plus a few wrong answers per subject.
"""
from __future__ import annotations

import random
import re
from fractions import Fraction

from sfcd.config import BaseCfg
from sfcd.model import build_prompt, generate, load_model
from sfcd.tasks.base import Task

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
            "prealgebra", "precalculus"]


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def to_number(s: str | None) -> Fraction | None:
    if s is None:
        return None
    s = s.strip().strip("$").replace(",", "").replace("\\!", "").replace(" ", "")
    s = re.sub(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", s)
    s = re.sub(r"\^\{?\\circ\}?|\\%|%|\\text\{[^}]*\}", "", s)
    if s.startswith("-\\frac"):
        s = re.sub(r"-\\frac\{(\d+)\}\{(\d+)\}", r"-\1/\2", s)
    if not re.fullmatch(r"-?\d+(\.\d+)?(/\d+)?", s):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


class _MathTask(Task):
    name = "math_bench"
    system_prompt = ("You are a helpful assistant. Solve the problem step by step, then give the final "
                     "answer as \\boxed{<answer>}.")
    max_new_tokens = 1536

    def build(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def is_correct(self, prediction, example):  # pragma: no cover
        raise NotImplementedError


DECODING = {  # name -> sampling settings for model.generate (None = greedy)
    "greedy": None,
    "qwen_recommended": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 2.0},
    "qwen_benchmark": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    "greedy_presence": {"presence_penalty": 1.5},
}


def run(base: BaseCfg, per_level: int = 10, seed: int = 0, max_new_tokens: int = 1536,
        decodings: list[str] | None = None) -> dict:
    """With decodings=None: one greedy pass (as before). Otherwise one result per decoding name."""
    if decodings:
        model_tok = load_model(base.model)
        return {name: _run(base, per_level, seed, max_new_tokens, DECODING[name], model_tok)
                for name in decodings}
    return _run(base, per_level, seed, max_new_tokens)


def run_students(paths, run_ids: list[str], stage: str = "sparsity", per_level: int = 5, seed: int = 0,
                 max_new_tokens: int = 1536, include_base: bool = True) -> dict:
    """The same MATH subset, greedy, for the base model and for each saved student (base + its delta).
    per_level=5, seed=0 is the 175-problem subset of the decoding check (greedy base: 40.0%)."""
    import torch
    from safetensors.torch import load_file

    from sfcd.paths import read_json
    from sfcd.pipeline import apply_saved_delta

    results = {rid: read_json(paths.result(stage, rid)) for rid in run_ids}
    base = BaseCfg.model_validate(next(iter(results.values()))["config"]["base"])
    model, tok = load_model(base.model)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}
    out = {"base": _run(base, per_level, seed, max_new_tokens, None, (model, tok))} if include_base else {}
    for rid, r in results.items():
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(orig[n])
        apply_saved_delta(model, r["config"]["granularity"], load_file(str(paths.delta(stage, rid))))
        out[rid] = {"name": r["name"], **_run(base, per_level, seed, max_new_tokens, None, (model, tok))}
    return out


def _run(base: BaseCfg, per_level: int, seed: int, max_new_tokens: int, sampling=None, model_tok=None) -> dict:
    from datasets import load_dataset

    rng = random.Random(seed)
    items = []
    for subj in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
        by_level: dict[str, list[dict]] = {}
        for r in ds:
            gold = to_number(last_boxed(r["solution"]))
            if gold is not None:
                by_level.setdefault(r["level"], []).append({**r, "subject": subj, "gold": gold})
        for lvl in sorted(by_level):
            pool = by_level[lvl]
            items += rng.sample(pool, min(per_level, len(pool)))
    task = _MathTask()
    task.max_new_tokens = max_new_tokens
    model, tok = model_tok or load_model(base.model)
    prompts = [build_prompt(tok, task, None, it["problem"]) for it in items]
    preds = generate(model, tok, prompts, max_new_tokens, base.eval.gen_batch_size, base.model,
                     sampling=sampling, seed=seed)
    rows = []
    for it, p in zip(items, preds):
        n_tok = len(tok(p, add_special_tokens=False)["input_ids"])
        got = to_number(last_boxed(p))
        rows.append({"subject": it["subject"], "level": it["level"], "correct": got == it["gold"],
                     "answered": got is not None, "hit_limit": n_tok >= max_new_tokens - 2, "n_tokens": n_tok,
                     "problem": it["problem"][:300], "gold": str(it["gold"]), "pred_tail": p[-300:]})

    def agg(rs):
        n = len(rs)
        return {"n": n, "acc": sum(r["correct"] for r in rs) / n, "answered": sum(r["answered"] for r in rs) / n,
                "hit_limit": sum(r["hit_limit"] for r in rs) / n,
                "mean_tokens": sum(r["n_tokens"] for r in rs) / n} if n else {"n": 0}

    out = {"overall": agg(rows), "by_subject": {}, "by_subject_level": {}, "wrong_samples": {}}
    for subj in SUBJECTS:
        rs = [r for r in rows if r["subject"] == subj]
        out["by_subject"][subj] = agg(rs)
        out["by_subject_level"][subj] = {lvl: agg([r for r in rs if r["level"] == lvl])
                                         for lvl in sorted({r["level"] for r in rs})}
        out["wrong_samples"][subj] = [r for r in rs if not r["correct"]][:4]
    return out
