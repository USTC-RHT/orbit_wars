"""rollout JIT 内使用的 JAX 原生训练对手。"""

from .opponents import (
    nearest_opponent_actions,
    selfplay_policy_actions,
)
from .producer_jax import producer_opponent_actions

__all__ = [
    "nearest_opponent_actions",
    "producer_opponent_actions",
    "selfplay_policy_actions",
]
