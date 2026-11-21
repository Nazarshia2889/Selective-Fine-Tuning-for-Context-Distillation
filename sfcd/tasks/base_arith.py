"""Multi-step reasoning: addition in an unfamiliar base, with 8 worked step-by-step solutions.

The base is stated only in the context, so without context the model defaults to base-10.

Variants (chosen by probing what Qwen3.5-0.8B can actually follow with the context):
  original:     3-4 digit operands, plain column-by-column steps
  small:        2-3 digit operands, plain steps
  padded:       3-4 digit operands, both padded to the same width, every step names its digit
                position, and the final digit reversal is written out
  padded_small: padded format with 2-3 digit operands
"""
from __future__ import annotations

import re

from sfcd.config import DataCfg
from sfcd.tasks.base import Context, Example, Splits, Task, split_unique

BASES = [7, 5, 9, 6, 8, 4, 3]  # candidate context i uses BASES[i % len(BASES)]
VARIANTS = {  # name -> (min digits, max digits, padded format)
    "original": (3, 4, False),
    "small": (2, 3, False),
    "padded": (3, 4, True),
    "padded_small": (2, 3, True),
}
DEFAULT_VARIANT = "original"


def to_base(n: int, b: int) -> str:
    if n == 0:
        return "0"
    digits = []
    while n:
        n, r = divmod(n, b)
        digits.append(str(r))
    return "".join(reversed(digits))


def _columns(x: int, y: int, b: int):
    xs, ys = to_base(x, b), to_base(y, b)
    width = max(len(xs), len(ys))
    return xs.zfill(width), ys.zfill(width), width


def worked_solution(x: int, y: int, b: int, padded: bool = False) -> str:
    xp, yp, width = _columns(x, y, b)
    xd, yd = xp[::-1], yp[::-1]
    carry, out, steps = 0, [], []
    for i in range(width):
        a, c = int(xd[i]), int(yd[i])
        s = a + c + carry
        terms = f"{a} + {c}" + (f" + {carry}" if carry else "")
        new_carry, digit = divmod(s, b)
        where = f"Step {i + 1} (digit {i + 1} from the right)" if padded else f"Step {i + 1}"
        if new_carry:
            steps.append(f"{where}: {terms} = {s} = {new_carry}*{b} + {digit}, write {digit}, carry {new_carry}")
        else:
            steps.append(f"{where}: {terms} = {s}, write {digit}, carry 0")
        out.append(str(digit))
        carry = new_carry
    if carry:
        steps.append(f"Step {width + 1}: write the final carry {carry}")
        out.append(str(carry))
    answer = "".join(reversed(out)).lstrip("0") or "0"
    assert answer == to_base(x + y, b)
    if padded:
        steps.append(f"Digits written, right to left: {', '.join(out)}")
        steps.append(f"Read left to right: {''.join(reversed(out))}")
    return "\n".join(steps + [f"Answer: {answer}"])


class BaseArithTask(Task):
    name = "base_arith"
    system_prompt = ("You are a helpful assistant. Solve the addition problem step by step, "
                     "then give the final line as 'Answer: <number>'.")
    max_new_tokens = 200
    n_shots = 8

    def __init__(self, variant: str = DEFAULT_VARIANT, n_shots: int = 8):
        self.n_shots = n_shots
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        self.variant = variant
        self.min_digits, self.max_digits, self.padded = VARIANTS[variant]

    def base_for(self, context_id: int) -> int:
        return BASES[context_id % len(BASES)]

    def _problem(self, x: int, y: int, b: int) -> str:
        if self.padded:
            xp, yp, _ = _columns(x, y, b)
            return f"Problem: {xp} + {yp}"
        return f"Problem: {to_base(x, b)} + {to_base(y, b)}"

    def build(self, context_id: int, cfg: DataCfg) -> tuple[Context, Splits]:
        rng = self.rng(context_id, cfg)
        b = self.base_for(context_id)
        lo, hi = b ** (self.min_digits - 1), b ** self.max_digits - 1
        need = self.n_shots + cfg.n_train + cfg.n_val + cfg.n_test
        if (hi - lo + 1) ** 2 < need:
            raise ValueError(f"base {b} with {self.variant!r} has too few distinct problems")
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < need:
            pairs.add((rng.randint(lo, hi), rng.randint(lo, hi)))
        shots, train, val, test = split_unique(sorted(pairs), [self.n_shots, cfg.n_train, cfg.n_val, cfg.n_test], rng)

        lines = [f"All numbers below are written in base {b}. Here are worked examples of addition in base {b}."]
        for x, y in shots:
            lines.append(f"{self._problem(x, y, b)}\n{worked_solution(x, y, b, self.padded)}")
        context = Context(self.name, context_id, "\n\n".join(lines), meta={"base": b, "variant": self.variant})
        ex = lambda ps: [Example(self._problem(x, y, b), to_base(x + y, b), meta={"x": x, "y": y}) for x, y in ps]
        return context, {"train": ex(train), "val": ex(val), "test": ex(test)}

    def is_correct(self, prediction: str, example: Example) -> bool:
        m = re.findall(r"Answer:\s*\$?([0-9]+)", prediction)
        return bool(m) and m[-1].lstrip("0") == example.answer.lstrip("0")
