"""Label mapping: 16 sentiment/topic examples relabeled with arbitrary words (e.g. positive -> "apple")."""
from __future__ import annotations

import random

from sfcd.config import DataCfg
from sfcd.tasks.base import Context, Example, Splits, Task

DATASETS = {
    # name: (hf path, config, text column, label column)
    "sst2": ("stanfordnlp/sst2", None, "sentence", "label"),
    "ag_news": ("fancyzhx/ag_news", None, "text", "label"),
}
# candidate context i -> (source dataset, arbitrary label words indexed by original label id)
CANDIDATES = [
    ("sst2", ["orange", "apple"]),             # 0=negative, 1=positive
    ("ag_news", ["river", "stone", "cloud", "lamp"]),  # world, sports, business, sci/tech
    ("sst2", ["seven", "blue"]),
    ("ag_news", ["tiger", "violin", "marble", "pepper"]),
    ("sst2", ["north", "candle"]),
]


def _load_pool(source: str) -> list[tuple[str, int]]:
    from datasets import load_dataset

    path, conf, tcol, lcol = DATASETS[source]
    ds = load_dataset(path, conf, split="train")
    return [(str(r[tcol]).strip(), int(r[lcol])) for r in ds if 0 < len(str(r[tcol]).split()) <= 60]


class LabelMappingTask(Task):
    name = "label_mapping"
    system_prompt = "You are a helpful assistant. Reply with only the label."
    max_new_tokens = 8
    n_shots = 16

    def __init__(self, pool_loader=_load_pool):
        self._pool_loader = pool_loader

    def build(self, context_id: int, cfg: DataCfg) -> tuple[Context, Splits]:
        rng = self.rng(context_id, cfg)
        source, words = CANDIDATES[context_id % len(CANDIDATES)]
        pool = self._pool_loader(source)
        n_labels = len(words)
        by_label: dict[int, list[str]] = {i: [] for i in range(n_labels)}
        for text, lab in pool:
            by_label[lab].append(text)
        for v in by_label.values():
            rng.shuffle(v)

        def take_balanced(n: int) -> list[tuple[str, int]]:
            out = []
            for i in range(n):
                lab = i % n_labels
                out.append((by_label[lab].pop(), lab))
            rng.shuffle(out)
            return out

        shots = take_balanced(self.n_shots)
        train, val, test = (take_balanced(n) for n in (cfg.n_train, cfg.n_val, cfg.n_test))
        lines = ["Classify each input using the labels shown in these examples."]
        lines += [f"Input: {t}\nLabel: {words[l]}" for t, l in shots]
        context = Context(self.name, context_id, "\n\n".join(lines), meta={"source": source, "labels": words})
        ex = lambda items: [Example(f"Input: {t}\nLabel:", words[l], meta={"orig_label": l}) for t, l in items]
        return context, {"train": ex(train), "val": ex(val), "test": ex(test)}

    def is_correct(self, prediction: str, example: Example) -> bool:
        toks = prediction.strip().lower().replace(".", " ").replace(",", " ").split()
        return bool(toks) and toks[0].strip("\"'*:") == example.answer.lower()


def synthetic_pool(source: str, n: int = 4000, seed: int = 0) -> list[tuple[str, int]]:
    """Offline stand-in pool for tests/smoke runs (no dataset download)."""
    rng = random.Random(seed)
    n_labels = 2 if source == "sst2" else 4
    vocab = [["awful", "boring", "bad"], ["great", "lovely", "fun"], ["market", "stocks"], ["science", "chips"]]
    return [(f"sentence {i} is {rng.choice(vocab[i % n_labels])}", i % n_labels) for i in range(n)]
