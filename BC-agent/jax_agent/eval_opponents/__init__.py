"""JAX 训练与评估对手适配器。"""

from .opponents import (
    CheckpointJaxEvalOpponent,
    EvalContext,
    ExternalScriptJaxEvalOpponent,
    JaxEvalOpponent,
    NearestJaxEvalOpponent,
    default_eval_opponents,
    eval_metric_fields,
)

__all__ = [
    "CheckpointJaxEvalOpponent",
    "EvalContext",
    "ExternalScriptJaxEvalOpponent",
    "JaxEvalOpponent",
    "NearestJaxEvalOpponent",
    "default_eval_opponents",
    "eval_metric_fields",
]
