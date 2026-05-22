"""JAX training/eval opponent adapters."""

from .opponents import (
    EvalContext,
    ExternalScriptJaxEvalOpponent,
    JaxEvalOpponent,
    NearestJaxEvalOpponent,
    default_eval_opponents,
    eval_metric_fields,
)

__all__ = [
    "EvalContext",
    "ExternalScriptJaxEvalOpponent",
    "JaxEvalOpponent",
    "NearestJaxEvalOpponent",
    "default_eval_opponents",
    "eval_metric_fields",
]
