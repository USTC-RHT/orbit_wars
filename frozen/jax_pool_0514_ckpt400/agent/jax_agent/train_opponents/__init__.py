"""JAX-native training opponents used inside rollout JIT."""

from .opponents import (
    nearest_opponent_actions,
    selfplay_policy_actions,
)

__all__ = [
    "nearest_opponent_actions",
    "selfplay_policy_actions",
]
