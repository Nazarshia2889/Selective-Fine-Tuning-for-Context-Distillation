"""Arbitrary-label classification (semantically unrelated labels, as in Wei et al. 2023, "Larger language
models do in-context learning differently"): topic classification where every class is renamed to an
unrelated word, so the mapping can only be learned from the in-context examples.

Compared with the first label_mapping task: only 4+ classes (a student that always gives one label gets
<= 25%), more examples in the context (>= 3 per class), and recovery is measured from the constant-label
baseline (trivial_baseline), not from 0%.
"""
from __future__ import annotations

import random

from sfcd.config import DataCfg
from sfcd.tasks.base import Context, Example, Splits, Task

DATASETS = {
    # name: (hf path, text columns joined with ". ", label column)
    "ag_news": ("fancyzhx/ag_news", ["text"], "label"),
    "dbpedia": ("fancyzhx/dbpedia_14", ["title", "content"], "label"),
}
# candidate context i -> (dataset, original class ids used, arbitrary label words, shots per class)
CANDIDATES = [
    ("ag_news", [0, 1, 2, 3], ["river", "stone", "cloud", "lamp"], 8),
    ("ag_news", [0, 1, 2, 3], ["tiger", "violin", "marble", "pepper"], 8),
    ("dbpedia", [0, 2, 3, 6, 9, 12], ["kettle", "zebra", "anchor", "tulip", "comet", "velvet"], 6),
    ("dbpedia", [1, 4, 5, 7, 10, 13], ["cactus", "harbor", "pencil", "falcon", "lemon", "jacket"], 6),
    ("dbpedia", list(range(14)), ["apple", "bridge", "candle", "dragon", "eagle", "forest", "guitar",
                                  "hammer", "island", "jungle", "kite", "ladder", "mirror", "needle"], 3),
]
MAX_WORDS = 60


def _load_pool(source: str) -> list[tuple[str, int]]:
    from datasets import load_dataset

    path, cols, lcol = DATASETS[source]
    ds = load_dataset(path, split="train")
    out = []
    for r in ds:
        text = ". ".join(str(r[c]).strip() for c in cols if str(r[c]).strip())
        if 0 < len(text.split()) <= MAX_WORDS:
            out.append((text, int(r[lcol])))
    return out


class ArbitraryLabelTask(Task):
    name = "arbitrary_labels"
    system_prompt = "You are a helpful assistant. Reply with only the label."
    max_new_tokens = 8

    def __init__(self, pool_loader=_load_pool):
        self._pool_loader = pool_loader

    def build(self, context_id: int, cfg: DataCfg) -> tuple[Context, Splits]:
        rng = self.rng(context_id, cfg)
        source, classes, words, per_class = CANDIDATES[context_id % len(CANDIDATES)]
        by_label: dict[int, list[str]] = {i: [] for i in range(len(classes))}
        cls_index = {c: i for i, c in enumerate(classes)}
        for text, lab in self._pool_loader(source):
            if lab in cls_index:
                by_label[cls_index[lab]].append(text)
        for v in by_label.values():
            v.sort()
            rng.shuffle(v)
        n_labels = len(classes)

        def take_balanced(n: int) -> list[tuple[str, int]]:
            out = [(by_label[i % n_labels].pop(), i % n_labels) for i in range(n)]
            rng.shuffle(out)
            return out

        shots = take_balanced(per_class * n_labels)
        train, val, test = (take_balanced(n) for n in (cfg.n_train, cfg.n_val, cfg.n_test))
        lines = ["Classify each input using the labels shown in these examples."]
        lines += [f"Input: {t}\nLabel: {words[l]}" for t, l in shots]
        context = Context(self.name, context_id, "\n\n".join(lines),
                          meta={"source": source, "classes": classes, "labels": words, "n_shots": len(shots)})
        ex = lambda items: [Example(f"Input: {t}\nLabel:", words[l], meta={"orig_label": l}) for t, l in items]
        return context, {"train": ex(train), "val": ex(val), "test": ex(test)}

    def is_correct(self, prediction: str, example: Example) -> bool:
        toks = prediction.strip().lower().replace(".", " ").replace(",", " ").split()
        return bool(toks) and toks[0].strip("\"'*:") == example.answer.lower()

    def trivial_baseline(self, examples: list[Example]) -> float:
        """Accuracy of always answering the most common label (what a student gets for free once it
        knows the label words, without learning the mapping)."""
        counts: dict[str, int] = {}
        for e in examples:
            counts[e.answer] = counts.get(e.answer, 0) + 1
        return max(counts.values()) / len(examples) if examples else 0.0


def synthetic_pool(source: str, n: int = 6000, seed: int = 0) -> list[tuple[str, int]]:
    """Offline stand-in pool for tests (no dataset download)."""
    rng = random.Random(seed)
    n_cls = 4 if source == "ag_news" else 14
    return [(f"sentence {i} about topic {i % n_cls} {rng.random():.3f}", i % n_cls) for i in range(n)]
