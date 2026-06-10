"""JAX 训练与评估对手适配器。"""

from .opponents import (
    EvalContext,
    ExternalScriptOpponent,
    ExternalScriptJaxEvalOpponent,
    JaxEvalOpponent,
    NearestJaxEvalOpponent,
    default_eval_opponents,
    eval_metric_fields,
)

__all__ = [
    "EvalContext",
    "ExternalScriptOpponent",
    "ExternalScriptJaxEvalOpponent",
    "JaxEvalOpponent",
    "NearestJaxEvalOpponent",
    "default_eval_opponents",
    "eval_metric_fields",
]
