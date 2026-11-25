"""Make every paper figure (PDF) and generated table (LaTeX) from results/paper/*.csv.

    python scripts/make_figures.py

Writes paper/figures/*.pdf and paper/tables/*.tex. Picked runs use sfcd.analysis.select_rule.
Error bars are the lowest and highest of a task's contexts (3 contexts; math has 1, so no bar).
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sfcd.analysis.select_rule import picks, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "paper"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"

TASKS = ["Knowledge", "Arbitrary labels", "Math (digit sum mod 9)"]
SHORT = {"Knowledge": "Knowledge Retrieval", "Arbitrary labels": "Label Mapping", "Math (digit sum mod 9)": "Mathematical Reasoning"}
COLOR = {"Knowledge": "#1f6fb2", "Arbitrary labels": "#d9822b", "Math (digit sum mod 9)": "#3a8f5c"}
GRANS = ["Weights", "Rows", "Blocks"]
GCOLOR = {"Weights": "#4c4c4c", "Rows": "#8c8c8c", "Blocks": "#c8c8c8"}
FULL_W, COL_W = 6.75, 3.25  # NeurIPS text width / column-ish width, inches
SHARES = [0.0001, 0.001, 0.01, 0.1, 1.0, 4.1667]
GPU_GB = 40  # A100-40GB used for every run

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "lines.markersize": 3.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def load(name: str) -> list[dict]:
    return list(csv.DictReader(open(DATA / name)))


def f(x):
    return None if x in ("", None) else float(x)


def share_label(s: float) -> str:
    return "0%" if s == 0 else ("4.2%" if s > 4 else f"{s:g}%")


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def ref_lines(ax, teacher=None, guess=None, untrained=None):
    if teacher is not None:
        ax.axhline(teacher, color="black", ls="--", lw=0.8, label="With context (teacher)")
    if untrained is not None:
        ax.axhline(untrained, color="0.45", ls=":", lw=0.9, label="Untrained")
    if guess is not None:
        ax.axhline(guess, color="0.7", ls="-", lw=0.7, label="Guessing")


def fig_legend(fig, ax, ncol=4, y=-0.06):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


def per_task_base(rows: list[dict]) -> dict:
    out = {}
    for t in TASKS:
        ctx = {}
        for r in rows:
            if r["task"] == t:
                ctx.setdefault(r["context"], r)
        cs = list(ctx.values())
        out[t] = {k: st.mean(float(c[col]) for c in cs) for k, col in
                  [("teacher", "icl_test"), ("untrained", "zero_shot_test"), ("guess", "trivial_test"), ("mmlu_base", "mmlu_base")]}
    return out


def picked_stats(p1: list[dict]) -> dict:
    """(task, share) -> dict of mean/min/max for the picked lr's runs."""
    out = {}
    for (t, fk), s in picks(p1, ("task", "fkey")).items():
        rs = s["runs"]
        d = {"lr": s["lr"], "n_trained": int(rs[0]["n_trained"])}
        for col in ("student_test", "mmlu", "recovery", "change_l2", "math_acc"):
            vals = [f(r[col]) for r in rs if f(r[col]) is not None]
            if vals:
                d[col] = (st.mean(vals), min(vals), max(vals))
        out[(t, float(fk))] = d
    return out


# ----------------------------------------------------------------------------- F2: lr -> change -> forgetting
def fig_lr_change_forgetting(p1):
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.3))
    grays = plt.cm.Greys([0.35 + 0.6 * i / (len(SHARES) - 1) for i in range(len(SHARES))])
    for sh, c in zip(SHARES, grays):
        g = defaultdict(list)
        for r in p1:
            if abs(float(r["fkey"]) - sh) < 1e-6 and r["change_l2"]:
                g[float(r["lr"])].append(float(r["change_l2"]))
        lrs = sorted(g)
        a.plot(lrs, [st.mean(g[l]) for l in lrs], "o-", color=c, label=share_label(sh))
    a.set(xscale="log", yscale="log", xlabel="Learning rate", ylabel=r"Change size $\|\Delta\theta\|_2$")
    a.legend(title="Share of weights", ncol=2, frameon=False, title_fontsize=7)
    a.set_title("(a) Change size grows with the learning rate", loc="left")
    for t in TASKS:
        rs = [r for r in p1 if r["task"] == t and r["change_l2"]]
        b.scatter([float(r["change_l2"]) for r in rs], [float(r["mmlu"]) * 100 for r in rs], s=7, alpha=0.7,
                  color=COLOR[t], label=SHORT[t], linewidths=0)
    base = st.mean(float(r["mmlu_base"]) for r in p1) * 100
    b.axhline(base, color="0.45", ls=":", lw=0.9, label="Untrained")
    b.set(xscale="log", xlabel=r"Change size $\|\Delta\theta\|_2$ (log)", ylabel="MMLU accuracy (%)", ylim=(20, 40))
    b.legend(frameon=False, ncol=2)
    b.set_title("(b) Larger changes forget more", loc="left")
    save(fig, "f2_lr_change_forgetting")


# ----------------------------------------------------------------------------- F3/F4: picked runs by share
def _share_axis(ax, shares):
    xs = list(range(len(shares) + 1))
    ax.set_xticks(xs, [share_label(0)] + [share_label(s) for s in shares], rotation=45, ha="right")
    ax.set_xlabel("Share of weights trained")
    return xs


def _picked_panel(ax, t, ps, col, base0, scale=100):
    shares = sorted(s for (tt, s) in ps if tt == t and col in ps[(tt, s)])
    xs = _share_axis(ax, shares)
    m = [base0 * scale] + [ps[(t, s)][col][0] * scale for s in shares]
    lo = [0] + [(ps[(t, s)][col][0] - ps[(t, s)][col][1]) * scale for s in shares]
    hi = [0] + [(ps[(t, s)][col][2] - ps[(t, s)][col][0]) * scale for s in shares]
    ax.errorbar(xs[1:], m[1:], yerr=[lo[1:], hi[1:]], fmt="o-", color=COLOR[t], capsize=2, elinewidth=0.8,
                label="Student (picked LR)")
    ax.plot([xs[0]], [m[0]], "s", color="0.45", label="Untrained (0%)")
    ax.plot(xs[:2], m[:2], "-", color=COLOR[t], lw=0.8, alpha=0.6)


def fig_recovery(p1):
    ps, base = picked_stats(p1), per_task_base(p1)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2), sharey=True)
    for ax, t in zip(axes, TASKS):
        _picked_panel(ax, t, ps, "student_test", base[t]["untrained"])
        ref_lines(ax, teacher=base[t]["teacher"] * 100)
        ax.set_title(SHORT[t])
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("Test accuracy without context (%)")
    fig.tight_layout(w_pad=0.6)
    fig_legend(fig, axes[1], y=-0.08)
    save(fig, "f3_accuracy_by_share")


def fig_forgetting(p1, mathrows):
    ps, base = picked_stats(p1), per_task_base(p1)
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.2))
    for ax, t in zip(axes[:3], TASKS):
        _picked_panel(ax, t, ps, "mmlu", base[t]["mmlu_base"])
        ref_lines(ax, untrained=base[t]["mmlu_base"] * 100)
        ax.set(title=f"MMLU: {SHORT[t]}", ylim=(20, 40))
    axes[0].set_ylabel("Accuracy (%)")
    mb = next(float(r["math_acc"]) for r in mathrows if not r["run_id"])
    _picked_panel(axes[3], TASKS[2], ps, "math_acc", mb)
    ref_lines(axes[3], untrained=mb * 100)
    axes[3].set(title=f"MATH: {SHORT[TASKS[2]]}", ylim=(0, 50))
    fig.tight_layout(w_pad=0.6)
    fig_legend(fig, axes[0], y=-0.1)
    save(fig, "f4_forgetting_by_share")


# ----------------------------------------------------------------------------- F5: efficiency
def fig_efficiency(cost, tokens):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    pos = {s: i for i, s in enumerate(SHARES)}
    FULLX = len(SHARES)
    for k, (ax, col, lab) in enumerate(zip(axes[:2], ["peak_mem_gb", "train_time_s"],
                                           ["Peak GPU memory during training (GB)", "Training time (s), median"])):
        tasks = [TASKS[0]] if col == "storage_mb" else TASKS
        for j, t in enumerate(tasks):
            off = 0 if col == "storage_mb" else (j - 1) * 0.12
            c = "0.3" if col == "storage_mb" else COLOR[t]  # storage is identical for every task (8 bytes per trained weight)
            xs, ys, lo, hi = [], [], [], []
            for s in SHARES:
                v = sorted(float(r[col]) for r in cost if r["task"] == t and r["granularity"] == "Weights" and abs(float(r["fkey"]) - s) < 1e-3)
                if not v:
                    continue
                q = st.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3
                mid = st.median(v) if col == "train_time_s" else st.mean(v)
                xs.append(pos[s] + off), ys.append(mid)
                lo.append(mid - q[0] if col == "train_time_s" else 0), hi.append(q[2] - mid if col == "train_time_s" else 0)
            ax.errorbar(xs, ys, yerr=[lo, hi] if col == "train_time_s" else None, fmt="o-", color=c, capsize=2, elinewidth=0.8,
                        label=SHORT[t] if col != "storage_mb" else None)
            fv = [float(r[col]) for r in cost if r["task"] == t and r["granularity"] == "Full fine-tuning"]
            if fv:
                ax.plot([FULLX + off], [st.median(fv)], "s", color=c, ms=4)
            elif col == "peak_mem_gb":  # full fine-tuning did not fit on the 40 GB GPU
                ax.plot([FULLX + off], [GPU_GB], "x", color=c, ms=6, mew=1.5)
                ax.annotate("out of\nmemory", (FULLX + off, GPU_GB), xytext=(-4, -16), textcoords="offset points",
                            ha="right", fontsize=6, color=c)
        ax.set_xticks(list(range(len(SHARES))) + [FULLX], [share_label(s) for s in SHARES] + ["Full"], rotation=45, ha="right")
        ax.set_xlabel("Share of weights trained")
        ax.set_title(lab)
        if col == "storage_mb":
            ax.set_yscale("log")
        elif col == "peak_mem_gb":
            ax.axhline(GPU_GB, color="0.6", ls="--", lw=0.7)
            ax.text(-0.3, GPU_GB + 0.8, "GPU limit (40 GB)", fontsize=6, color="0.4")
            ax.set_ylim(0, 45)
        else:
            ax.set_ylim(0, None)
    # inference: prompt tokens per query with and without the context
    ax = axes[2]
    for i, t in enumerate(TASKS):
        w = st.mean(int(r["with_context"]) for r in tokens if r["task"] == t)
        o = st.mean(int(r["without_context"]) for r in tokens if r["task"] == t)
        ax.bar(i - 0.2, w, 0.38, color=COLOR[t], edgecolor="black", lw=0.4)
        ax.bar(i + 0.2, o, 0.38, color="white", edgecolor=COLOR[t], lw=1.0, hatch="////")
        ax.text(i, w + 60, f"{w / o:.0f}x", ha="center", fontsize=7)
    ax.set_xticks(range(len(TASKS)), [SHORT[t].replace(" ", "\n", 1) for t in TASKS], fontsize=6.5)
    ax.set_title("Input tokens per query")
    from matplotlib.patches import Patch
    ax.legend_handles = [Patch(facecolor="0.55", edgecolor="black", lw=0.4, label="With context (teacher)"),
                         Patch(facecolor="white", edgecolor="0.3", hatch="////", label="Without context (student)")]
    axes[0].plot([], [], "s", color="0.4", label="Full fine-tuning")
    fig.tight_layout(w_pad=0.8)
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = ax.legend_handles, [h.get_label() for h in ax.legend_handles]
    fig.legend(h0 + h2, l0 + l2, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.14))
    save(fig, "f5_efficiency")


# ----------------------------------------------------------------------------- F6: granularity
def fig_granularity(gran):
    rows = [r for r in gran if r["granularity"] in GRANS]
    pk = picks(rows, ("task", "granularity"))
    base = per_task_base(rows)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2), gridspec_kw={"width_ratios": [3, 3, 1.2]})
    width = 0.26
    for ax, col, title, tasks in [(axes[0], "student_test", "Task accuracy without context (%)", TASKS),
                                  (axes[1], "mmlu", "MMLU (%)", TASKS), (axes[2], "math_acc", "MATH (%)", TASKS[2:])]:
        for i, t in enumerate(tasks):
            for j, g in enumerate(GRANS):
                rs = pk[(t, g)]["runs"]
                v = [float(r[col]) * 100 for r in rs if r[col] != ""]
                m = st.mean(v)
                ax.bar(i + (j - 1) * width, m, width * 0.92, color=GCOLOR[g], edgecolor="black", lw=0.4, label=g if i == 0 else None)
                if len(v) > 1:
                    ax.errorbar(i + (j - 1) * width, m, yerr=[[m - min(v)], [max(v) - m]], color="black", capsize=1.5, elinewidth=0.6)
            ref = {"student_test": base[t]["teacher"] * 100, "mmlu": base[t]["mmlu_base"] * 100,
                   "math_acc": next((float(r["math_base"]) * 100 for r in rows if r["task"] == t and r["math_base"]), None)}[col]
            if ref is not None:
                ax.hlines(ref, i - 1.6 * width, i + 1.6 * width, color="black", ls="--" if col == "student_test" else ":", lw=0.8)
        ax.set_xticks(range(len(tasks)), [SHORT[t] for t in tasks], rotation=0 if len(tasks) == 1 else 15)
        ax.set_title(title)
        if col == "mmlu":
            ax.set_ylim(20, 40)
    axes[0].plot([], [], "k--", lw=0.8, label="With context (teacher)")
    axes[0].plot([], [], "k:", lw=0.8, label="Untrained")
    fig.tight_layout(w_pad=0.8)
    fig_legend(fig, axes[0], ncol=5, y=-0.08)
    save(fig, "f6_granularity")


# ----------------------------------------------------------------------------- F7: every lr per context (appendix)
def fig_boxplots(p1, mathrows):
    ps = picked_stats(p1)
    for t in TASKS:
        ctxs = sorted({r["context"] for r in p1 if r["task"] == t}, key=int)
        metrics = [("student_test", "Task accuracy (%)", (0, 102)), ("mmlu", "MMLU (%)", (20, 40))]
        if t == TASKS[2]:
            metrics.append(("math_acc", "MATH (%)", (0, 50)))
        fig, axes = plt.subplots(len(metrics), len(ctxs), figsize=(FULL_W if len(ctxs) > 1 else COL_W, 1.9 * len(metrics)),
                                 squeeze=False)
        for j, cx in enumerate(ctxs):
            rs = [r for r in p1 if r["task"] == t and r["context"] == cx]
            shares = sorted({float(r["fkey"]) for r in rs})
            for i, (col, lab, yl) in enumerate(metrics):
                ax = axes[i][j]
                data, picked = [], []
                for s in shares:
                    v = [(float(r["lr"]), float(r[col]) * 100) for r in rs if abs(float(r["fkey"]) - s) < 1e-6 and r[col] != ""]
                    data.append([y for _, y in v])
                    plr = ps[(t, s)]["lr"]
                    picked.append(next((y for l, y in v if abs(l - plr) / plr < 1e-6), None))
                ax.boxplot(data, positions=range(len(shares)), widths=0.5, showfliers=False,
                           medianprops={"color": "black", "lw": 0.9}, boxprops={"lw": 0.6}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
                for k, v in enumerate(data):
                    ax.plot([k] * len(v), v, ".", color="0.5", ms=3)
                    if picked[k] is not None:
                        ax.plot([k], [picked[k]], "o", color=COLOR[t], ms=4)
                ax.set_xticks(range(len(shares)), [share_label(s) for s in shares], rotation=45, ha="right")
                ax.set_ylim(*yl)
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 0:
                    ax.set_title(f"Context {j + 1}" if len(ctxs) > 1 else SHORT[t])
        fig.supxlabel("Share of weights trained", fontsize=8)
        fig.tight_layout()
        save(fig, f"f7_every_lr_{t.split()[0].lower()}")


# ----------------------------------------------------------------------------- F8: training loss (appendix)
def fig_loss(p1, curves):
    by_run = defaultdict(list)
    for c in curves:
        by_run[c["run_id"]].append((int(c["step"]), float(c["loss"])))
    sel = {(t, float(fk)): v for (t, fk), v in picks(p1, ("task", "fkey")).items()}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.1), sharey=False)
    grays = plt.cm.viridis([i / (len(SHARES) - 1) for i in range(len(SHARES))])
    for ax, t in zip(axes, TASKS):
        for s, c in zip(SHARES, grays):
            if (t, s) not in sel:
                continue
            ids = [r["run_id"] for r in sel[(t, s)]["runs"]]
            cs = [sorted(by_run[i]) for i in ids if by_run.get(i)]
            if not cs:
                continue
            n = min(len(x) for x in cs)
            ax.plot(range(n), [st.mean(x[k][1] for x in cs) for k in range(n)], color=c, lw=0.9, label=share_label(s))
        ax.set(yscale="log", xlabel="Training step", title=SHORT[t])
    axes[0].set_ylabel("Training loss (log)")
    axes[0].legend(frameon=False, title="Share", title_fontsize=7)
    fig.tight_layout()
    save(fig, "f8_training_loss")


# ----------------------------------------------------------------------------- tables
def table_tasks(p1):
    base = per_task_base(p1)
    n_ctx = {t: len({r["context"] for r in p1 if r["task"] == t}) for t in TASKS}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Task & Contexts & Val / test per context & Without context & With context (teacher) \\", r"\midrule"]
    split = {"Knowledge": "100 / 200", "Arbitrary labels": "100 / 200", "Math (digit sum mod 9)": "300 / 500"}
    for t in TASKS:
        b = base[t]
        lines.append(f"{SHORT[t]} & {n_ctx[t]} & {split[t]} & {b['untrained']*100:.1f}\\% & {b['teacher']*100:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _wt("t1_tasks", lines)


def table_picked(p1):
    ps = picked_stats(p1)
    lines = [r"\begin{tabular}{llrlrrr}", r"\toprule",
             r"Task & Share & Weights & LR & Accuracy & Recovery & MMLU \\", r"\midrule"]
    for t in TASKS:
        for s in sorted(s for (tt, s) in ps if tt == t):
            d = ps[(t, s)]
            lines.append(f"{SHORT[t]} & {share_label(s).replace('%', chr(92) + '%')} & {d['n_trained']:,} & {d['lr']:.0e} & {d['student_test'][0]*100:.1f}\\% & "
                         f"{d['recovery'][0]*100:.0f}\\% & {d['mmlu'][0]*100:.1f}\\% \\\\".replace("e-0", "e-"))
        lines.append(r"\midrule" if t != TASKS[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    _wt("t_picked_runs", lines)


def table_efficiency(cost):
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Task & Method (4.2\% budget) & Peak memory (GB) & Storage (MB) & Train time (s, median) \\", r"\midrule"]
    for t in TASKS:
        for g in GRANS + ["Full fine-tuning"]:
            rs = [r for r in cost if r["task"] == t and r["granularity"] == g and (g != "Weights" or abs(float(r["fkey"]) - 4.1667) < 1e-3)]
            if not rs:
                lines.append(f"{SHORT[t]} & {g} & \\multicolumn{{3}}{{c}}{{out of memory (40 GB)}} \\\\")
                continue
            lines.append(f"{SHORT[t]} & {g} & {st.mean(float(r['peak_mem_gb']) for r in rs):.1f} & "
                         f"{st.mean(float(r['storage_mb']) for r in rs):,.0f} & {st.median(float(r['train_time_s']) for r in rs):.0f} \\\\")
        lines.append(r"\midrule" if t != TASKS[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    _wt("t3_efficiency", lines)


def _wt(name, lines):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / f"{name}.tex")


if __name__ == "__main__":
    p1, gran, mathrows, cost = load("phase1_runs.csv"), load("gran_runs.csv"), load("math_general.csv"), load("cost_runs.csv")
    curves = load("loss_curves.csv")
    fig_lr_change_forgetting(p1)
    fig_recovery(p1)
    fig_forgetting(p1, mathrows)
    fig_efficiency(cost, load("inference_tokens.csv"))
    fig_granularity(gran)
    fig_boxplots(p1, mathrows)
    fig_loss(p1, curves)
    table_tasks(p1)
    table_picked(p1)
    table_efficiency(cost)
