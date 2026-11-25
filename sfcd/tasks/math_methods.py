"""Math problem types where the method must come from the context (v2 of method_math).

Same format as method_math (8 worked solutions of one problem type in the context; queries are new
instances), with two changes after the first version turned out to mostly teach brevity:
  - answers may run to 1536 tokens (512 cut off the no-context model before it finished), and
  - a context is only kept if most no-context errors are wrong answers, not unfinished ones
    (max_zero_shot_truncation, used by the filter).

Candidates 0-4 are the original types; 5-9 are new, from number theory and counting, the weakest
MATH subjects for this model that can be generated in bulk with exact answers:
  5 josephus:        survivor of the Josephus problem, every 2nd person removed (n = 2^m + L -> 2L + 1)
  6 crt:             smallest x with given remainders mod two coprime numbers (step through one congruence)
  7 mod_inverse:     inverse of a mod a prime (extended Euclid)
  8 incl_excl:       count of 1..N divisible by at least one of three numbers (inclusion-exclusion)
  9 last_two_digits: last two digits of a^n (powers repeat mod 100)
"""
from __future__ import annotations

from math import gcd

from sfcd.tasks.method_math import FAMILIES as V1_FAMILIES
from sfcd.tasks.method_math import MethodMathTask, _cycle_solution

FAMILIES = V1_FAMILIES + ["josephus", "crt", "mod_inverse", "incl_excl", "last_two_digits"]
PRIMES = [p for p in range(11, 100) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
CRT_MODULI = [3, 4, 5, 7, 8, 9, 11, 13]
IE_BASES = [2, 3, 5, 7, 11, 13]


def josephus_solution(n: int) -> tuple[str, str]:
    m = n.bit_length() - 1
    L = n - 2 ** m
    ans = 2 * L + 1
    lines = ["With every 2nd person removed, write n as the largest power of 2 not above it, plus a remainder L:",
             f"{n} = 2^{m} + {L} = {2 ** m} + {L}.",
             "The survivor is person 2L + 1.",
             f"2 × {L} + 1 = {ans}."]
    return "\n".join(lines), str(ans)


def crt_solution(a: int, p: int, b: int, q: int) -> tuple[str, str]:
    """x ≡ a (mod p), x ≡ b (mod q); step through numbers ≡ b mod q until one is ≡ a mod p."""
    lines = [f"Numbers that leave remainder {b} when divided by {q} are {b}, {b + q}, {b + 2 * q}, ...",
             f"Check each one's remainder when divided by {p} until it is {a}:"]
    x = b if b > 0 else q
    while True:
        r = x % p
        lines.append(f"{x} ÷ {p} leaves remainder {r}" + (" ✓" if r == a else ""))
        if r == a:
            break
        x += q
    lines.append(f"The smallest such positive number is {x}.")
    return "\n".join(lines), str(x)


def mod_inverse_solution(a: int, p: int) -> tuple[str, str]:
    """Extended Euclid, written as a table of (remainder, coefficient of a) pairs."""
    lines = [f"Run the Euclidean algorithm on {p} and {a}, tracking each remainder as a multiple of {a} mod {p}.",
             f"Start: {p} ≡ 0 × {a}, and {a} = 1 × {a}."]
    r0, t0, r1, t1 = p, 0, a, 1
    while r1 != 1:
        qq = r0 // r1
        r2, t2 = r0 - qq * r1, t0 - qq * t1
        lines.append(f"{r0} = {qq} × {r1} + {r2}, so {r2} ≡ {t0} − {qq} × {t1} = {t2} times {a}.")
        r0, t0, r1, t1 = r1, t1, r2, t2
    ans = t1 % p
    lines.append(f"So 1 ≡ {t1} × {a} (mod {p}), and {t1} mod {p} = {ans}.")
    lines.append(f"Check: {a} × {ans} = {a * ans} = {p} × {a * ans // p} + 1.")
    return "\n".join(lines), str(ans)


def incl_excl_solution(N: int, a: int, b: int, c: int) -> tuple[str, str]:
    s1 = [N // a, N // b, N // c]
    s2 = [N // (a * b), N // (a * c), N // (b * c)]
    s3 = N // (a * b * c)
    ans = sum(s1) - sum(s2) + s3
    lines = ["Use inclusion-exclusion: add the counts for each number, subtract the counts for each pair "
             "(multiples of the product), and add back the count for all three.",
             f"Singles: {N}÷{a} = {s1[0]}, {N}÷{b} = {s1[1]}, {N}÷{c} = {s1[2]} (rounded down); sum {sum(s1)}.",
             f"Pairs: {N}÷{a * b} = {s2[0]}, {N}÷{a * c} = {s2[1]}, {N}÷{b * c} = {s2[2]}; sum {sum(s2)}.",
             f"All three: {N}÷{a * b * c} = {s3}.",
             f"Total: {sum(s1)} − {sum(s2)} + {s3} = {ans}."]
    return "\n".join(lines), str(ans)


class MathMethodsTask(MethodMathTask):
    """Used in the study with a single context, digit_mod9 (candidate 2): in the probe (50 val problems,
    1536 tokens) it was one of only two of the ten types where the context taught a method the model
    lacked (+32 points, 12% of no-context errors unfinished). Being one context, it gets larger val/test
    sets so its single number is less noisy."""
    name = "math_methods"
    max_new_tokens = 1536
    candidate_ids = [2]  # digit_mod9
    n_keep = 1
    n_val, n_test = 300, 500

    def family(self, context_id: int) -> str:
        return FAMILIES[context_id % len(FAMILIES)]

    def build(self, context_id: int, cfg):
        return super().build(context_id, cfg.model_copy(update={"n_val": self.n_val, "n_test": self.n_test}))

    def _sample(self, fam: str, rng):
        if fam == "josephus":
            return (rng.randint(20, 5000),)
        if fam == "crt":
            p, q = rng.sample(CRT_MODULI, 2)
            while gcd(p, q) != 1:
                p, q = rng.sample(CRT_MODULI, 2)
            return (rng.randrange(p), p, rng.randrange(1, q), q)
        if fam == "mod_inverse":
            p = rng.choice(PRIMES)
            return (rng.randint(2, p - 1), p)
        if fam == "incl_excl":
            a, b, c = sorted(rng.sample(IE_BASES, 3))
            return (rng.randint(100, 10000), a, b, c)
        if fam == "last_two_digits":
            return (rng.choice([3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37]), rng.randint(20, 999))
        return super()._sample(fam, rng)

    def _render(self, fam: str, key) -> tuple[str, str, str]:
        if fam == "josephus":
            (n,) = key
            sol, ans = josephus_solution(n)
            q = (f"{n} people stand in a circle, numbered 1 to {n}. Starting with person 1, every second "
                 f"person is removed (person 2 first, then 4, and so on around the circle) until one remains. "
                 f"Which number survives?")
            return q, sol, ans
        if fam == "crt":
            a, p, b, q = key
            sol, ans = crt_solution(a, p, b, q)
            return (f"What is the smallest positive integer that leaves remainder {a} when divided by {p} "
                    f"and remainder {b} when divided by {q}?"), sol, ans
        if fam == "mod_inverse":
            a, p = key
            sol, ans = mod_inverse_solution(a, p)
            return f"Find the integer x between 1 and {p - 1} such that {a}x leaves remainder 1 when divided by {p}.", sol, ans
        if fam == "incl_excl":
            N, a, b, c = key
            sol, ans = incl_excl_solution(N, a, b, c)
            return f"How many integers from 1 to {N} are divisible by at least one of {a}, {b}, or {c}?", sol, ans
        if fam == "last_two_digits":
            a, n = key
            sol, ans = _cycle_solution(a, n, 100, "last two digits")
            return f"What are the last two digits of {a}^{n}? Give them as a number from 0 to 99.", sol, ans
        return super()._render(fam, key)
