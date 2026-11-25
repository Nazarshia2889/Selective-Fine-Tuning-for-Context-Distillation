"""Benchmark Qwen3.5 with and without the fused linear-attention kernels (flash-linear-attention,
causal-conv1d) on real method_math training data. Checks speed and that losses/outputs agree.

    modal run scripts/modal_bench_kernels.py
"""
from __future__ import annotations

import modal

ROOT = "/vol"
vol = modal.Volume.from_name("sfcd-vol")
PKGS = ["torch==2.8.0", "transformers>=4.57", "accelerate", "datasets", "safetensors",
        "pydantic>=2", "pyyaml", "pandas", "wandb"]

slow = (modal.Image.debian_slim(python_version="3.11").uv_pip_install(*PKGS)
        .env({"HF_HOME": f"{ROOT}/hf", "TOKENIZERS_PARALLELISM": "false"}).add_local_python_source("sfcd"))
FAST_PKGS = ["torch==2.9.1", *PKGS[1:], "flash-linear-attention", "einops"]
fast = (modal.Image.debian_slim(python_version="3.11").uv_pip_install(*FAST_PKGS)
        .env({"HF_HOME": f"{ROOT}/hf", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_python_source("sfcd"))
app = modal.App("sfcd-bench")


def _bench() -> dict:
    import importlib.util
    import time

    import torch

    from sfcd.config import BaseCfg
    from sfcd.data import collate, make_sft_examples, micro_batches
    from sfcd.model import autocast, build_prompt, generate, load_model
    from sfcd.paths import read_jsonl
    from sfcd.tasks import get_task

    base = BaseCfg()
    model, tok = load_model(base.model)
    task = get_task("method_math")
    recs = read_jsonl(f"{ROOT}/data/method_math/c0/teacher.jsonl")[:64]
    exs = make_sft_examples(tok, task, recs, base.model.max_seq_len)
    model.train()
    losses, times = [], []
    for step in range(4):  # 4 batches of 16, micro-batched by 4 (as in training)
        batch = collate(exs[step * 16:(step + 1) * 16], tok.pad_token_id)
        torch.cuda.synchronize(); t0 = time.time()
        tot = 0.0
        for mb, w in micro_batches(batch, 4):
            mb = {k: v.cuda() for k, v in mb.items()}
            with autocast(base.model, torch.device("cuda")):
                loss = model(**mb).loss * w
            loss.backward()
            tot += float(loss)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t0); losses.append(tot)
    prompts = [build_prompt(tok, task, None, r["query"]) for r in recs[:32]]
    generate(model, tok, prompts, 128, 32, base.model)  # warm-up (kernel compilation / autotuning)
    torch.cuda.synchronize(); t0 = time.time()
    outs = generate(model, tok, prompts, 128, 32, base.model)
    torch.cuda.synchronize(); gen_t = time.time() - t0
    try:
        import fla  # noqa: F401
        fla_ok = "ok"
    except Exception as e:  # report why the fused path is unavailable
        fla_ok = repr(e)[:300]
    return {"fla_import": fla_ok, "torch": torch.__version__,
            "train_step_s": times, "losses": losses, "gen_32x128_s": gen_t, "gen_head": outs[:3]}


def _save(name: str, out: dict) -> dict:
    import json
    from pathlib import Path

    p = Path(ROOT) / "diagnostics" / f"bench_{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    vol.commit()
    return out


@app.function(image=slow, gpu="A100-40GB", volumes={ROOT: vol}, timeout=1800)
def bench_slow() -> dict:
    return _save("slow", _bench())


@app.function(image=fast, gpu="A100-40GB", volumes={ROOT: vol}, timeout=1800)
def bench_fast() -> dict:
    return _save("fast", _bench())


@app.local_entrypoint()
def main():
    import json

    a, b = bench_slow.spawn(), bench_fast.spawn()
    ra, rb = a.get(), b.get()
    print(json.dumps({"slow": ra, "fast": rb}, indent=1))
    print("train step speedup (excl. first):", sum(ra["train_step_s"][1:]) / sum(rb["train_step_s"][1:]))
    print("generation speedup:", ra["gen_32x128_s"] / rb["gen_32x128_s"])
    print("max loss diff:", max(abs(x - y) for x, y in zip(ra["losses"], rb["losses"])))
