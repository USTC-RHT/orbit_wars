"""Load fixed JAX checkpoint policies as training/eval opponents."""

from .opponents import CheckpointOpponent, load_checkpoint_opponent

__all__ = ["CheckpointOpponent", "load_checkpoint_opponent"]
