"""Is a student's MMLU drop real forgetting, or just the first-token readout breaking?

For the base model and each saved student, on the same MMLU subset:
  1. letter spread under the standard scoring (argmax over A/B/C/D logits at the first answer token)
  2. what the first token wants over the full vocabulary (top tokens, probability mass on A-D)
  3. generation-based accuracy: greedy-generate a few tokens and parse the first standalone letter
"""
from __future__ import annotations

import re
from collections import Counter

import torch

from sfcd.config import BaseCfg
from sfcd.evaluate import _MMLUTask, load_mmlu_subset, option_text_scores
from sfcd.model import autocast, build_prompt, generate, load_model
from sfcd.paths import Paths, read_json

LETTERS = ["A", "B", "C", "D"]
_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_letter(text: str) -> int | None:
    m = _LETTER_RE.search(text)
    return LETTERS.index(m.group(1)) if m else None


@torch.no_grad()
def _first_token_stats(model, tok, prompts, base: BaseCfg) -> dict:
    device = next(model.parameters()).device
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    tok.padding_side = "left"
    model.eval()
    argmax_letter, top1, mass = [], [], []
    bs = base.eval.mmlu_batch_size
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with autocast(base.model, device):
            logits = model(**enc).logits[:, -1, :].float()
        p = logits.softmax(-1)
        argmax_letter += logits[:, letter_ids].argmax(-1).tolist()
        top1 += p.argmax(-1).tolist()
        mass += p[:, letter_ids].sum(-1).tolist()
    return {"argmax_letter": argmax_letter, "top1": top1, "letter_mass": mass}


def _summarize(model, tok, items, prompts, base: BaseCfg, n_gen_tokens: int) -> dict:
    gold = [it["answer"] for it in items]
    ot = option_text_scores(model, tok, items, base.model)
    ot_pred = [max(range(len(r)), key=r.__getitem__) for r in ot]
    ft = _first_token_stats(model, tok, prompts, base)
    gens = generate(model, tok, prompts, n_gen_tokens, base.eval.gen_batch_size, base.model)
    parsed = [parse_letter(g) for g in gens]
    n = len(items)
    top = Counter(tok.decode([t]) for t in ft["top1"]).most_common(8)
    return {
        "mmlu_option_text": sum(p == g for p, g in zip(ot_pred, gold)) / n,
        "option_text_spread": {LETTERS[k]: v / n for k, v in sorted(Counter(ot_pred).items())},
        "mmlu_first_token": sum(a == g for a, g in zip(ft["argmax_letter"], gold)) / n,
        "letter_spread": {LETTERS[k]: v / n for k, v in sorted(Counter(ft["argmax_letter"]).items())},
        "gold_spread": {LETTERS[k]: v / n for k, v in sorted(Counter(gold).items())},
        "top1_tokens": [[t, c / n] for t, c in top],
        "mean_letter_mass": sum(ft["letter_mass"]) / n,
        "mmlu_generate": sum(p == g for p, g in zip(parsed, gold)) / n,
        "mmlu_generate_parsed_only": (sum(p == g for p, g in zip(parsed, gold) if p is not None)
                                      / max(1, sum(p is not None for p in parsed))),
        "parse_rate": sum(p is not None for p in parsed) / n,
        "gen_spread": {(LETTERS[k] if k is not None else "none"): v / n
                       for k, v in Counter(parsed).most_common()},
        "sample_generations": gens[:12],
    }


def check(paths: Paths, run_ids: list[str], stage: str = "main", n_gen_tokens: int = 16) -> dict:
    from safetensors.torch import load_file

    from sfcd.pipeline import apply_saved_delta

    results = {rid: read_json(paths.result(stage, rid)) for rid in run_ids}
    base = BaseCfg.model_validate(next(iter(results.values()))["config"]["base"])
    items = load_mmlu_subset(base.eval.mmlu_n, base.eval.mmlu_seed)
    task = _MMLUTask()
    model, tok = load_model(base.model)
    prompts = []
    for it in items:
        opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, it["choices"]))
        prompts.append(build_prompt(tok, task, None, f"{it['question']}\n\n{opts}\n\nAnswer:"))
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}

    out = {"base": {"label": "base model", **_summarize(model, tok, items, prompts, base, n_gen_tokens)}}
    for rid, r in results.items():
        c = r["config"]
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(orig[n])
        apply_saved_delta(model, c["granularity"], load_file(str(paths.delta(stage, rid))))
        label = f"{c['task']} c{c['context_id']} {c['granularity']} b{c['budget_blocks']} {c['selector']} lr{c['lr']:g}"
        out[rid] = {"label": label, "mmlu_reported": r.get("mmlu"),
                    **_summarize(model, tok, items, prompts, base, n_gen_tokens)}
    return out
