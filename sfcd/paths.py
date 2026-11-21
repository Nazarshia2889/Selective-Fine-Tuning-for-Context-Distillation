"""Artifact layout under a single root (a Modal Volume mount on Modal, ./outputs/<name> locally)."""
from __future__ import annotations

import json
from pathlib import Path


class Paths:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def ctx_dir(self, task: str, cid: int) -> Path:
        return self.root / "data" / task / f"c{cid}"

    def kept(self) -> Path:
        return self.root / "data" / "kept.json"

    def mmlu_base(self) -> Path:
        return self.root / "baselines" / "mmlu_base.json"

    def scores(self, task: str, cid: int) -> Path:
        return self.root / "scores" / f"{task}_c{cid}.pt"

    def result(self, stage: str, run_id: str) -> Path:
        return self.root / "results" / stage / f"{run_id}.json"

    def results_dir(self, stage: str | None = None) -> Path:
        return self.root / "results" / stage if stage else self.root / "results"

    def delta(self, stage: str, run_id: str) -> Path:
        return self.root / "deltas" / stage / f"{run_id}.safetensors"

    def best_lr(self) -> Path:
        return self.root / "results" / "best_lr.json"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
