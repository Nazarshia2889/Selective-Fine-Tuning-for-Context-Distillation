"""Task interface. A task deterministically builds (context, train/val/test queries) for a context id."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from sfcd.config import DataCfg


@dataclass
class Example:
    query: str
    answer: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Example":
        return Example(query=d["query"], answer=d["answer"], meta=d.get("meta", {}))


@dataclass
class Context:
    task: str
    context_id: int
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Context":
        return Context(**d)


Splits = dict[str, list[Example]]  # keys: train, val, test


class Task(ABC):
    name: str
    system_prompt: str = "You are a helpful assistant. Answer concisely."
    max_new_tokens: int = 32
    # Optional per-task overrides of the data settings (None = use DataCfg):
    candidate_ids: list[int] | None = None  # which candidate contexts to build/run (default range(n_candidates))
    n_keep: int | None = None  # how many contexts the filter keeps

    def rng(self, context_id: int, cfg: DataCfg) -> random.Random:
        return random.Random(f"{self.name}-{cfg.seed}-{context_id}")

    @abstractmethod
    def build(self, context_id: int, cfg: DataCfg) -> tuple[Context, Splits]:
        ...

    @abstractmethod
    def is_correct(self, prediction: str, example: Example) -> bool:
        ...

    def user_message(self, context_text: str | None, query: str) -> str:
        return f"{context_text}\n\n{query}" if context_text else query

    def trivial_baseline(self, examples: list[Example]) -> float:
        """Accuracy available without learning the task (e.g. always giving the majority label).
        Recovery is measured from max(zero-shot, this)."""
        return 0.0

    def accuracy(self, predictions: list[str], examples: list[Example]) -> float:
        if not examples:
            return 0.0
        return sum(self.is_correct(p, e) for p, e in zip(predictions, examples)) / len(examples)


def split_unique(items: list, sizes: list[int], rng: random.Random) -> list[list]:
    """Shuffle and cut into disjoint chunks of the requested sizes (raises if too few)."""
    items = list(items)
    rng.shuffle(items)
    if sum(sizes) > len(items):
        raise ValueError(f"need {sum(sizes)} unique items, only have {len(items)}")
    out, i = [], 0
    for s in sizes:
        out.append(items[i:i + s])
        i += s
    return out
