"""Multi-step reasoning with an unfamiliar rule taught only through worked examples.

rule="operator":  a ★ b = p*a + b - q            (p, q differ per context)
rule="procedure": reverse the digits, add k, multiply by m   (k, m differ per context)

Without the context the model has never seen the rule; with it, each step is ordinary arithmetic.
"""
from __future__ import annotations

import re

from sfcd.config import DataCfg
from sfcd.tasks.base import Context, Example, Splits, Task, split_unique

OPERATOR_PARAMS = [(2, 3), (3, 1), (2, 7), (3, 5), (4, 2)]  # context i -> (p, q)
PROCEDURE_PARAMS = [(5, 2), (7, 3), (3, 2), (9, 2), (4, 3)]  # context i -> (k, m)


class RuleArithTask(Task):
    name = "rule_arith"
    system_prompt = ("You are a helpful assistant. Solve the problem step by step, "
                     "then give the final line as 'Answer: <number>'.")
    max_new_tokens = 120

    def __init__(self, rule: str = "operator", n_shots: int = 8):
        if rule not in ("operator", "procedure"):
            raise ValueError(rule)
        self.rule, self.n_shots = rule, n_shots

    # --- operator
    def _op(self, cid: int, a: int, b: int) -> tuple[str, str, int]:
        p, q = OPERATOR_PARAMS[cid % len(OPERATOR_PARAMS)]
        s1, s2 = p * a, p * a + b
        ans = s2 - q
        steps = [f"Step 1: {p} × {a} = {s1}", f"Step 2: {s1} + {b} = {s2}", f"Step 3: {s2} − {q} = {ans}"]
        return f"Problem: {a} ★ {b}", "\n".join(steps + [f"Answer: {ans}"]), ans

    # --- procedure
    def _proc(self, cid: int, n: int) -> tuple[str, str, int]:
        k, m = PROCEDURE_PARAMS[cid % len(PROCEDURE_PARAMS)]
        r = int(str(n)[::-1])
        s = r + k
        ans = s * m
        steps = [f"Step 1: reverse the digits of {n}: {r}", f"Step 2: {r} + {k} = {s}", f"Step 3: {s} × {m} = {ans}"]
        return f"Problem: transform {n}", "\n".join(steps + [f"Answer: {ans}"]), ans

    def build(self, context_id: int, cfg: DataCfg) -> tuple[Context, Splits]:
        rng = self.rng(context_id, cfg)
        need = self.n_shots + cfg.n_train + cfg.n_val + cfg.n_test
        if self.rule == "operator":
            items: set = set()
            while len(items) < need:
                items.add((rng.randint(10, 99), rng.randint(10, 99)))
            make = lambda it: self._op(context_id, *it)
            header = "Here are worked examples of the ★ operation."
        else:
            pool = [n for n in range(100, 10000) if n % 10]  # no trailing zero, so reversal is unambiguous
            items = set(rng.sample(pool, need))
            make = lambda n: self._proc(context_id, n)
            header = "Here are worked examples of how to transform a number."
        shots, train, val, test = split_unique(sorted(items), [self.n_shots, cfg.n_train, cfg.n_val, cfg.n_test], rng)
        lines = [header] + [f"{q}\n{sol}" for q, sol, _ in map(make, shots)]
        context = Context(self.name, context_id, "\n\n".join(lines), meta={"rule": self.rule})
        ex = lambda its: [Example(q, str(ans)) for q, _, ans in map(make, its)]
        return context, {"train": ex(train), "val": ex(val), "test": ex(test)}

    def is_correct(self, prediction: str, example: Example) -> bool:
        m = re.findall(r"Answer:\s*\$?(-?[0-9]+)", prediction.replace(",", ""))
        return bool(m) and m[-1] == example.answer
