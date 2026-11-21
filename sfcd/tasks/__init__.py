from sfcd.tasks.base import Context, Example, Task
from sfcd.tasks.base_arith import BaseArithTask
from sfcd.tasks.knowledge import KnowledgeTask
from sfcd.tasks.label_mapping import LabelMappingTask
from sfcd.tasks.method_math import MethodMathTask
from sfcd.tasks.rule_arith import RuleArithTask

_REGISTRY = {
    "label_mapping": LabelMappingTask,
    "knowledge": KnowledgeTask,
    "base_arith": BaseArithTask,
    "rule_arith": RuleArithTask,
    "method_math": MethodMathTask,
}


def get_task(name: str) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def register_task(name: str, cls) -> None:
    _REGISTRY[name] = cls


__all__ = ["Context", "Example", "Task", "get_task", "register_task"]
