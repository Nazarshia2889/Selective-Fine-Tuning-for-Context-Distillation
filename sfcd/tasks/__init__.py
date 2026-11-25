from sfcd.tasks.arbitrary_labels import ArbitraryLabelTask
from sfcd.tasks.base import Context, Example, Task
from sfcd.tasks.base_arith import BaseArithTask
from sfcd.tasks.knowledge import KnowledgeTask
from sfcd.tasks.label_mapping import LabelMappingTask
from sfcd.tasks.math_methods import MathMethodsTask
from sfcd.tasks.method_math import MethodMathTask
from sfcd.tasks.rule_arith import RuleArithTask

_REGISTRY = {
    "label_mapping": LabelMappingTask,
    "arbitrary_labels": ArbitraryLabelTask,
    "knowledge": KnowledgeTask,
    "base_arith": BaseArithTask,
    "rule_arith": RuleArithTask,
    "method_math": MethodMathTask,
    "math_methods": MathMethodsTask,
}


def get_task(name: str) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def register_task(name: str, cls) -> None:
    _REGISTRY[name] = cls


__all__ = ["Context", "Example", "Task", "get_task", "register_task"]
