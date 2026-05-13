"""Experimental full-JAX Orbit Wars training stack.

This package is intentionally parallel to the existing PyTorch/Kaggle path.
The first milestone is a fast, fixed-array simulator plus a JAX PPO loop that
can be parity-tested and benchmarked without disturbing the current trainer.
"""

