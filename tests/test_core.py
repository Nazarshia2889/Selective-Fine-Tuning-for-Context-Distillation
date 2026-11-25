"""Offline tests on a random-init tiny Qwen2 (no downloads)."""
from __future__ import annotations

import copy

import pytest
import torch

from sfcd.config import DataCfg, ModelCfg, TrainCfg, expand_lr_tune, expand_main, load_base, load_sweep
from sfcd.model import eligible_linears
from sfcd.selection.selectors import budget_params, random_scores, select
from sfcd.sparse.params import apply_selection, export_delta, merge_into_base
from sfcd.tasks.base_arith import BaseArithTask, to_base, worked_solution
from sfcd.tasks.knowledge import KnowledgeTask
from sfcd.tasks.label_mapping import LabelMappingTask, synthetic_pool


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen2ForCausalLM(cfg).eval()


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 128, (2, 12), generator=g)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}


# ----------------------------------------------------------------------------- config / sweep


def test_sweep_has_183_runs():
    sw = load_sweep("configs/sweep.yaml")
    base = load_base(sw.base_config)
    assert len(expand_lr_tune(sw, base)) == 12
    main = expand_main(sw, base)
    assert len(main) == 171
    assert len({r.run_id() for r in main}) == 171


def test_overrides():
    base = load_base("configs/base.yaml", ["train.epochs=1", "wandb.mode=disabled"])
    assert base.train.epochs == 1 and base.wandb.mode == "disabled"


# ----------------------------------------------------------------------------- tasks


def test_base_arith():
    assert to_base(10, 7) == "13"
    sol = worked_solution(int("66", 7), int("1", 7), 7)
    assert sol.endswith("Answer: 100")
    task = BaseArithTask()
    ctx, sp = task.build(0, DataCfg())
    assert [len(sp[s]) for s in ("train", "val", "test")] == [500, 100, 200]
    assert not ({e.query for e in sp["train"]} & {e.query for e in sp["test"]})
    e = sp["test"][0]
    assert task.is_correct(f"Step 1: ...\nAnswer: {e.answer}", e)
    assert "base 7" in ctx.text and "base" not in e.query


def test_knowledge_heldout_wordings():
    task = KnowledgeTask()
    ctx, sp = task.build(0, DataCfg())
    assert [len(sp[s]) for s in ("train", "val", "test")] == [500, 100, 200]
    assert not ({e.query for e in sp["train"]} & {e.query for e in sp["test"] + sp["val"]})
    e = sp["test"][0]
    assert e.answer in ctx.text and task.is_correct(f"It is {e.answer}.", e)
    assert task.build(0, DataCfg())[0].text == ctx.text  # deterministic


def test_label_mapping_offline():
    task = LabelMappingTask(pool_loader=synthetic_pool)
    ctx, sp = task.build(0, DataCfg())
    assert ctx.text.count("Label:") == 16
    e = sp["test"][0]
    assert task.is_correct(f"{e.answer.capitalize()}.", e) and not task.is_correct("banana", e)


# ----------------------------------------------------------------------------- selection


@pytest.mark.parametrize("gran", ["weight", "row", "block"])
@pytest.mark.parametrize("b", [1, 2])
def test_budget_matched(tiny, gran, b):
    lin = eligible_linears(tiny)
    assert len({li.layer for li in lin}) == 4
    mask = select(random_scores(lin, 0), lin, gran, b)
    target = budget_params(lin, b)
    n = mask.n_selected(lin)
    if gran == "block":
        assert len(mask.blocks) == b
    tol = 0.01 if gran == "weight" else max(li.in_features for li in lin) / target
    assert abs(n - target) / target <= tol


def test_topk_picks_highest(tiny):
    lin = eligible_linears(tiny)
    scores = {li.param_name: torch.zeros(li.out_features, li.in_features) for li in lin}
    hot = lin[5]
    scores[hot.param_name][3] = 1.0  # one hot row
    m = select(scores, lin, "row", 1)
    assert 3 in m.rows[hot.param_name].tolist()
    mb = select(scores, lin, "block", 1)
    assert mb.blocks == [hot.layer]


def test_random_deterministic(tiny):
    lin = eligible_linears(tiny)
    a = select(random_scores(lin, 7), lin, "weight", 1).checksum()
    b = select(random_scores(lin, 7), lin, "weight", 1).checksum()
    c = select(random_scores(lin, 8), lin, "weight", 1).checksum()
    assert a == b != c


# ----------------------------------------------------------------------------- sparse params


@pytest.mark.parametrize("gran", ["weight", "row", "block", "full"])
def test_only_selected_weights_change(tiny, gran):
    model = copy.deepcopy(tiny)
    lin = eligible_linears(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mask = None if gran == "full" else select(random_scores(lin, 1), lin, gran, 1)
    params = apply_selection(model, lin, gran, mask, full_scope="eligible")
    with torch.no_grad():
        out0 = model(**_batch()).logits
    torch.testing.assert_close(out0, tiny(**_batch()).logits)  # zero deltas == identity

    opt = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(2):
        model(**_batch()).loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        out_wrapped = model(**_batch(1)).logits
    exported = export_delta(model, gran, params)
    assert exported
    merge_into_base(model)
    with torch.no_grad():
        torch.testing.assert_close(model(**_batch(1)).logits, out_wrapped, atol=1e-5, rtol=1e-4)

    after = model.state_dict()
    for li in lin:
        diff = (after[li.param_name] != before[li.param_name])
        if gran == "weight":
            allowed = torch.zeros(li.numel, dtype=torch.bool)
            if li.param_name in mask.weight_idx:
                allowed[mask.weight_idx[li.param_name]] = True
            allowed = allowed.view_as(diff)
        elif gran == "row":
            allowed = torch.zeros_like(diff)
            if li.param_name in mask.rows:
                allowed[mask.rows[li.param_name]] = True
        elif gran == "block":
            allowed = torch.full_like(diff, li.layer in mask.blocks)
        else:
            allowed = torch.ones_like(diff)
        assert not (diff & ~allowed).any(), li.param_name
    assert any((after[li.param_name] != before[li.param_name]).any() for li in lin)
    # frozen non-eligible params untouched
    for k in before:
        if k not in {li.param_name for li in lin}:
            assert torch.equal(before[k], after[k]), k


def test_trainable_count_matches_budget(tiny):
    model = copy.deepcopy(tiny)
    lin = eligible_linears(model)
    mask = select(random_scores(lin, 2), lin, "weight", 1)
    params = apply_selection(model, lin, "weight", mask)
    assert sum(p.numel() for p in params) == mask.n_selected(lin)
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.endswith("delta"))


def test_train_loop_and_scoring(tiny):
    from sfcd.selection.scoring import compute_grad_scores
    from sfcd.train import train_sft

    model = copy.deepcopy(tiny)
    lin = eligible_linears(model)
    b = _batch()
    scores, meta = compute_grad_scores(model, lin, iter([b, b, b]), 2, ModelCfg())
    assert meta["n_batches"] == 2 and all((s >= 0).all() for s in scores.values())
    mask = select(scores, lin, "row", 1)
    params = apply_selection(model, lin, "row", mask)
    ex = [{"input_ids": b["input_ids"][0].tolist(), "labels": b["labels"][0].tolist()}] * 4
    stats = train_sft(model, params, ex, 0, TrainCfg(epochs=2, batch_size=2), 1e-2, ModelCfg())
    assert stats["train_steps"] == 4


# ----------------------------------------------------------------------------- lr selection


def test_select_lr_respects_forgetting(tmp_path):
    from sfcd.paths import Paths, write_json
    from sfcd.pipeline import stage_select_lr

    paths = Paths(tmp_path)
    fake = [("weight", 1e-5, 0.86, 0.00), ("weight", 3e-5, 0.99, 0.01), ("weight", 1e-4, 1.00, 0.17),
            ("full", 1e-6, 0.50, 0.05), ("full", 3e-6, 0.70, 0.10)]
    for i, (g, lr, val, drop) in enumerate(fake):
        write_json(paths.result("lr_tune", f"r{i}"), {"config": {"granularity": g, "lr": lr},
                                                      "student_val": val, "mmlu_forgetting": drop})
    best = stage_select_lr(paths, max_mmlu_drop=0.02)
    assert best["weight"] == 3e-5  # 1e-4 has higher val acc but forgets too much
    assert best["full"] == 1e-6  # nothing qualifies -> least forgetting


def test_micro_batches_match_full_batch_gradient(tiny):
    from sfcd.data import collate, micro_batches

    model = copy.deepcopy(tiny)
    g = torch.Generator().manual_seed(3)
    exs = []
    for L, p in [(10, 3), (14, 6), (7, 2), (12, 4), (9, 5)]:  # varied lengths and prompt sizes
        ids = torch.randint(0, 128, (L,), generator=g).tolist()
        exs.append({"input_ids": ids, "labels": [-100] * p + ids[p:]})
    batch = collate(exs, pad_id=0)

    model.zero_grad()
    model(**batch).loss.backward()
    full = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()
    for mb, w in micro_batches(batch, 2):
        (model(**mb).loss * w).backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad, full[n], atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("gran", ["weight", "row", "block", "full"])
def test_saved_delta_roundtrip(tiny, gran):
    from sfcd.pipeline import apply_saved_delta

    model = copy.deepcopy(tiny)
    lin = eligible_linears(model)
    mask = None if gran == "full" else select(random_scores(lin, 4), lin, gran, 1)
    params = apply_selection(model, lin, gran, mask)
    opt = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(2):
        model(**_batch()).loss.backward()
        opt.step()
        opt.zero_grad()
    delta = export_delta(model, gran, params)
    merge_into_base(model)

    fresh = copy.deepcopy(tiny)
    apply_saved_delta(fresh, gran, delta)
    for (n, a), (_, b) in zip(model.state_dict().items(), fresh.state_dict().items()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5, msg=n)


class _CharTok:
    """Minimal offline tokenizer: one token per character (ids mod 128)."""
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) % 128 for ch in text]}


def test_option_text_scores_match_manual(tiny):
    from sfcd.evaluate import option_text_scores, _ClozeTask
    from sfcd.model import build_prompt

    tok = _CharTok()
    items = [{"question": "2+2?", "choices": ["four", "five", "a long answer", "six"], "answer": 0}]
    cfg = ModelCfg(compute_dtype="float32")
    got = option_text_scores(tiny, tok, items, cfg, batch_seqs=3)[0]
    p = tok(build_prompt(tok, _ClozeTask(), None, "2+2?\n\nAnswer:"))["input_ids"]
    for c, g in zip(items[0]["choices"], got):
        o = tok(" " + c)["input_ids"]
        x = torch.tensor([p + o])
        with torch.no_grad():
            lp = tiny(x).logits[0, :-1].log_softmax(-1).gather(-1, x[0, 1:, None])[:, 0]
        assert abs(float(lp[len(p) - 1:].mean()) - g) < 1e-5


def test_old_run_id_unchanged():
    import json
    from sfcd.config import RunCfg

    r = json.load(open("tests/fixtures/old_run.json"))
    assert RunCfg.model_validate(r["config"]).run_id() == r["run_id"]


def test_phase_stage_counts():
    from sfcd.config import expand_pilot, expand_sparsity

    sw = load_sweep("configs/sweep.yaml")
    base = load_base(sw.base_config)
    kept = {"arbitrary_labels": [0, 2, 3], "knowledge": [0, 1, 2], "math_methods": [2]}
    sp = expand_sparsity(sw, base, kept)
    assert len(sp) == 7 * 6 * 4  # 7 contexts x 6 budgets x 4 lrs
    assert len({r.run_id() for r in sp}) == len(sp)
    assert sw.sparsity.lrs_for_fraction(0.000001)[0] == 1e-3
    pilot = expand_pilot(sw, base)
    assert len(pilot) == 3 * 2 * 6
    # pilot-grid runs inside the lean bands: 0.01% with 3e-4 and 1e-3, for knowledge and arbitrary_labels c0
    # (math_methods c0 isn't a kept context)
    assert len({r.run_id() for r in pilot} & {r.run_id() for r in sp}) == 4


def test_select_by_val_uses_mean_val_only(tmp_path):
    from sfcd.paths import Paths, write_json
    from sfcd.pipeline import stage_select_by_val

    paths = Paths(tmp_path)
    vals = {1e-5: [0.5, 0.6, 0.7], 1e-4: [0.9, 0.9, 0.8]}  # 1e-4 better on average
    for lr, vs in vals.items():
        for cid, v in enumerate(vs):
            cfg = {"task": "knowledge", "context_id": cid, "granularity": "weight", "selector": "topk",
                   "budget_blocks": None, "budget_frac": 0.001, "lr": lr}
            write_json(paths.result("sparsity", f"{lr}-{cid}"),
                       {"run_id": f"{lr}-{cid}", "config": cfg, "student_val": v, "mmlu_forgetting": 0.5})
    best = stage_select_by_val(paths)
    assert best == {"knowledge|weight|topk|f0.001": 1e-4}


def test_budget_frac(tiny):
    lin = eligible_linears(tiny)
    total = sum(li.numel for li in lin)
    assert budget_params(lin, budget_frac=0.01) == round(0.01 * total)
    g = torch.Generator().manual_seed(0)
    scores = {li.param_name: torch.rand(li.out_features, li.in_features, generator=g) for li in lin}
    m = select(scores, lin, "weight", None, budget_frac=0.01)
    assert m.n_selected(lin) == round(0.01 * total)
    with pytest.raises(ValueError):
        select(scores, lin, "block", None, budget_frac=0.01)


def test_arbitrary_labels_offline():
    from sfcd.tasks.arbitrary_labels import ArbitraryLabelTask, synthetic_pool

    t = ArbitraryLabelTask(pool_loader=synthetic_pool)
    cfg = DataCfg()
    for cid, n_cls in [(0, 4), (2, 6), (4, 14)]:
        ctx, sp = t.build(cid, cfg)
        assert ctx.meta["n_shots"] >= 3 * n_cls
        assert len({e.answer for e in sp["test"]}) == n_cls
        assert abs(t.trivial_baseline(sp["test"]) - 1 / n_cls) < 0.02
        assert t.is_correct(sp["test"][0].answer + ".", sp["test"][0])
    assert t.build(0, cfg)[0].text == t.build(0, cfg)[0].text  # deterministic


def test_recovery_uses_trivial_floor():
    from sfcd.evaluate import recovery

    assert recovery(0.5, 0.0, 0.6) == pytest.approx(0.5 / 0.6)
    assert recovery(0.5, 0.0, 0.6, trivial=0.5) == pytest.approx(0.0)  # constant-label student: no credit
    assert recovery(0.55, 0.0, 0.6, trivial=0.5) == pytest.approx(0.5)


def test_filter_keeps_other_tasks(tmp_path):
    from sfcd.paths import Paths, read_json, write_json
    from sfcd.pipeline import stage_filter

    paths = Paths(tmp_path)
    write_json(paths.kept(), {"kept": {"knowledge": [0, 1, 2]}, "details": {}})
    base = load_base("configs/base.yaml")
    # margins over max(zero-shot, trivial): 0.05 (fails), 0.35, 0.33, 0.63, 0.28 -> best three are 1, 2, 3
    for cid, (icl, triv) in enumerate([(0.30, 0.25), (0.60, 0.25), (0.50, 0.17), (0.70, 0.07), (0.35, 0.07)]):
        write_json(paths.ctx_dir("arbitrary_labels", cid) / "baselines.json",
                   {"zero_shot_val": 0.0, "icl_val": icl, "trivial_val": triv})
    kept = stage_filter(paths, base, ["arbitrary_labels"])
    assert kept["arbitrary_labels"] == [1, 2, 3] and kept["knowledge"] == [0, 1, 2]
    assert read_json(paths.kept())["kept"]["knowledge"] == [0, 1, 2]


def test_range_check_runs_are_phase1_runs():
    from sfcd.config import expand_range_check, expand_sparsity

    sw = load_sweep("configs/sweep.yaml")
    base = load_base(sw.base_config)
    rc = expand_range_check(sw, base)
    assert len(rc) == 20
    current = [r for r in rc if r.task in sw.tasks]  # method_math (v1) has since been replaced
    assert current and {r.run_id() for r in current} <= {r.run_id() for r in expand_sparsity(sw, base)}


def test_presence_penalty_only_hits_generated_tokens():
    from sfcd.model import PresencePenalty

    ids = torch.tensor([[1, 2, 3, 5, 5]])  # prompt = first 3 tokens, generated = [5, 5]
    scores = torch.zeros(1, 8)
    out = PresencePenalty(1.5, prompt_len=3)(ids, scores)
    assert out[0, 5] == -1.5 and out[0, 1] == 0 and out[0, 2] == 0 and (out[0, [0, 4, 6, 7]] == 0).all()


def test_pick_lr_one_se_rule_with_teacher_cap():
    from sfcd.analysis.select_rule import picks

    def run(lr, val, teacher=0.99, n=100):
        return {"task": "t", "fkey": 1.0, "lr": lr, "student_val": val, "icl_val": teacher, "n_val": n}

    # best 0.80 (lr 1e-3), SE = sqrt(.8*.2/100) = 0.04 -> threshold 0.76 -> smallest lr with val >= 0.76 is 3e-4
    rs = [run(1e-4, 0.70), run(3e-4, 0.77), run(1e-3, 0.80), run(3e-3, 0.60)]
    assert picks(rs, ("task", "fkey"))[("t", 1.0)]["lr"] == 3e-4
    # teacher cap: teacher val 0.65 is below best - SE, so the smallest lr reaching 0.65 wins
    rs = [run(1e-4, 0.70, teacher=0.65), run(3e-4, 0.77, teacher=0.65), run(1e-3, 0.80, teacher=0.65)]
    assert picks(rs, ("task", "fkey"))[("t", 1.0)]["lr"] == 1e-4
