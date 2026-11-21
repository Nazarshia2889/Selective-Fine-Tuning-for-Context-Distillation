"""End-to-end smoke run of the whole pipeline in-process (no Modal), on a tiny model.

    python scripts/run_local.py                         # tiny model, CPU-friendly, synthetic tasks
    python scripts/run_local.py --model Qwen/Qwen3.5-0.8B --root outputs/real --set data.n_train=64
"""
from __future__ import annotations

import argparse
import logging

from sfcd import pipeline
from sfcd.config import RunCfg, load_base
from sfcd.paths import Paths

SMOKE_OVERRIDES = [
    "data.n_candidates=1", "data.n_keep=1", "data.n_train=32", "data.n_val=8", "data.n_test=16",
    "data.min_gap=-1.0",  # tiny random models never pass the real 20pp filter
    "train.epochs=1", "train.batch_size=8",
    "selection.score_batches=2", "selection.score_batch_size=8",
    "eval.gen_batch_size=16", "eval.mmlu_n=0", "wandb.mode=disabled", "model.max_seq_len=512",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    ap.add_argument("--root", default="outputs/smoke")
    ap.add_argument("--tasks", nargs="+", default=["knowledge", "base_arith"])
    ap.add_argument("--set", nargs="*", default=[], help="extra dotted overrides, e.g. train.epochs=2")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    base = load_base("configs/base.yaml", SMOKE_OVERRIDES + [f"model.name={args.model}"] + args.set)
    paths = Paths(args.root)
    pipeline.stage_build_data(paths, base, args.tasks)
    for t in args.tasks:
        for c in range(base.data.n_candidates):
            pipeline.stage_teacher(paths, base, t, c)
    kept = pipeline.stage_filter(paths, base, args.tasks)
    pipeline.stage_mmlu_base(paths, base)
    for t, cs in kept.items():
        for c in cs:
            pipeline.stage_score(paths, base, t, c)

    t, c = args.tasks[0], kept[args.tasks[0]][0]
    runs = [RunCfg(base=base, stage="smoke", task=t, context_id=c, granularity=g, budget_blocks=1,
                   selector=s, lr=1e-3) for g in ("weight", "row", "block") for s in ("topk", "random")]
    runs.append(RunCfg(base=base, stage="smoke", task=t, context_id=c, granularity="full", selector="none", lr=1e-4))
    for r in runs:
        res = pipeline.stage_train_eval(paths, r)
        print(f"{res['name']:45s} trainable={res['frac_eligible_trained']:.3f} "
              f"test={res['student_test']:.3f} delta={res['delta_bytes'] / 1e6:.2f}MB "
              f"time={res['train_time_s']:.1f}s")


if __name__ == "__main__":
    main()
