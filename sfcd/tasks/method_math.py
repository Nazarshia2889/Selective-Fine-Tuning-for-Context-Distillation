"""Multi-step reasoning: real math problem types that need a specific method.

Each context is one problem family, shown through 8 worked solutions; queries are new instances of
the same family with fresh numbers. The model often fails these alone, and the examples teach the
method (not an invented rule).

  0 trailing_zeros: trailing zeros of n!       (count factors of 5: n/5 + n/25 + ...)
  1 work_rate:      two pipes filling a tank   (add rates, invert)
  2 digit_mod9:     remainder of N mod 9       (digit sum)
  3 mod_power:      remainder of a^n mod m     (remainders repeat in a cycle)   [backup]
  4 last_digit:     last digit of a^n          (last digits repeat in a cycle)  [backup]

Order matters: the filter keeps the first 3 candidates that pass. Probe (50 val problems, gain with
context): trailing_zeros +74, work_rate +44, digit_mod9 +26, mod_power +20, last_digit +16.
"""
from __future__ import annotations

import re
from fractions import Fraction

from sfcd.config import DataCfg
from sfcd.tasks.base import Context, Example, Splits, Task, split_unique

FAMILIES = ["trailing_zeros", "work_rate", "digit_mod9", "mod_power", "last_digit"]
ORD = {1: "1st", 2: "2nd", 3: "3rd"}


def _ordinal(i: int) -> str:
    return ORD.get(i, f"{i}th")


def _cycle(a: int, m: int) -> list[int]:
    """Repeating remainders of a^1, a^2, ... mod m (assumes the sequence is purely periodic)."""
    seen, out, x = {}, [], a % m
    while x not in seen:
        seen[x] = len(out)
        out.append(x)
        x = (x * a) % m
    return out[seen[x]:] if seen[x] == 0 else out  # purely periodic for the bases we use


def _cycle_solution(a: int, n: int, m: int, what: str) -> tuple[str, str]:
    cyc = _cycle(a, m)
    L = len(cyc)
    r = n % L
    pos = L if r == 0 else r
    ans = cyc[pos - 1]
    note = f" (remainder 0 means the last one in the cycle)" if r == 0 else ""
    steps = [
        f"The {what} of powers of {a} repeat in a cycle: {', '.join(map(str, cyc))} (length {L}).",
        f"{n} divided by {L} leaves remainder {r}{note}.",
        f"So we take the {_ordinal(pos)} number in the cycle: {ans}.",
    ]
    return "\n".join(steps), str(ans)


class MethodMathTask(Task):
    name = "method_math"
    system_prompt = ("You are a helpful assistant. Solve the problem step by step, "
                     "then give the final line as 'Answer: <answer>'.")
    max_new_tokens = 512  # the model writes long explanations; 160 truncated before 'Answer:'

    def __init__(self, n_shots: int = 8):
        self.n_shots = n_shots

    def family(self, context_id: int) -> str:
        return FAMILIES[context_id % len(FAMILIES)]

    # each generator: rng -> (question key)   and   key -> (question, solution, answer)
    def _sample(self, fam: str, rng):
        if fam == "last_digit":
            return (rng.choice([2, 3, 4, 7, 8, 9]), rng.randint(20, 999))
        if fam == "trailing_zeros":
            return (rng.randint(25, 3000),)
        if fam == "mod_power":
            m = rng.choice([7, 11, 13])
            a = rng.choice([x for x in range(2, 10) if x % m])
            return (a, rng.randint(20, 999), m)
        if fam == "work_rate":
            a, b = rng.sample(range(2, 41), 2)
            return (a, b)
        if fam == "digit_mod9":
            return (rng.randint(10**6, 10**9 - 1),)
        raise ValueError(fam)

    def _render(self, fam: str, key) -> tuple[str, str, str]:
        if fam == "last_digit":
            a, n = key
            sol, ans = _cycle_solution(a, n, 10, "last digits")
            return f"What is the last digit of {a}^{n}?", sol, ans
        if fam == "trailing_zeros":
            (n,) = key
            terms, p = [], 5
            while p <= n:
                terms.append((p, n // p))
                p *= 5
            total = sum(t for _, t in terms)
            lines = ["Each trailing zero needs a factor 10 = 2 × 5, and factors of 5 are the rarer ones, so count them."]
            lines += [f"Multiples of {p} up to {n}: {n} ÷ {p} = {t} (rounded down)" for p, t in terms]
            lines.append(f"Total: {' + '.join(str(t) for _, t in terms)} = {total}")
            return f"How many zeros does {n}! end with?", "\n".join(lines), str(total)
        if fam == "mod_power":
            a, n, m = key
            sol, ans = _cycle_solution(a, n, m, f"remainders mod {m}")
            return f"What is the remainder when {a}^{n} is divided by {m}?", sol, ans
        if fam == "work_rate":
            a, b = key
            t = Fraction(a * b, a + b)
            ans = str(t.numerator) if t.denominator == 1 else f"{t.numerator}/{t.denominator}"
            lines = [f"Pipe A fills 1/{a} of the tank per hour and pipe B fills 1/{b} per hour.",
                     f"Together: 1/{a} + 1/{b} = {b}/{a * b} + {a}/{a * b} = {a + b}/{a * b} of the tank per hour.",
                     f"Time = {a * b}/{a + b} hours = {ans} hours."]
            q = (f"Pipe A fills a tank in {a} hours and pipe B fills it in {b} hours. Working together, "
                 f"how many hours do they take? Give the answer as a whole number or a fraction in lowest terms.")
            return q, "\n".join(lines), ans
        if fam == "digit_mod9":
            (N,) = key
            digits = [int(d) for d in str(N)]
            s = sum(digits)
            lines = ["A number has the same remainder mod 9 as the sum of its digits.",
                     f"Digit sum: {' + '.join(map(str, digits))} = {s}",
                     f"{s} = 9 × {s // 9} + {s % 9}, so the remainder is {s % 9}."]
            return f"What is the remainder when {N} is divided by 9?", "\n".join(lines), str(s % 9)
        raise ValueError(fam)

    def build(self, context_id: int, cfg: DataCfg) -> tuple[Context, Splits]:
        rng = self.rng(context_id, cfg)
        fam = self.family(context_id)
        need = self.n_shots + cfg.n_train + cfg.n_val + cfg.n_test
        keys: set = set()
        tries = 0
        while len(keys) < need:
            keys.add(self._sample(fam, rng))
            tries += 1
            if tries > 100 * need:
                raise ValueError(f"family {fam} has too few distinct problems")
        shots, train, val, test = split_unique(sorted(keys), [self.n_shots, cfg.n_train, cfg.n_val, cfg.n_test], rng)
        lines = ["Here are worked examples of how to solve this kind of problem."]
        for k in shots:
            q, sol, ans = self._render(fam, k)
            lines.append(f"Problem: {q}\n{sol}\nAnswer: {ans}")
        context = Context(self.name, context_id, "\n\n".join(lines), meta={"family": fam})
        ex = lambda ks: [Example(f"Problem: {self._render(fam, k)[0]}", self._render(fam, k)[2]) for k in ks]
        return context, {"train": ex(train), "val": ex(val), "test": ex(test)}

    def is_correct(self, prediction: str, example: Example) -> bool:
        text = prediction.replace(",", "")
        text = re.sub(r"\\[dt]?frac\{(\d+)\}\{(\d+)\}", r"\1/\2", text)  # \frac{a}{b} -> a/b
        m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+)?)", text)
        m = m or re.findall(r"\\boxed\{\s*(-?[0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+)?)\s*\}", text)
        if not m:
            return False
        got = m[-1].replace(" ", "")
        try:
            return Fraction(got) == Fraction(example.answer)
        except (ValueError, ZeroDivisionError):
            return False
