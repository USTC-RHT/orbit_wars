"""Benchmark the experimental JAX environment step."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

from .build_initial_state_bank import build_bank  # noqa: E402
from .fast_env import EnvParams, load_state_bank, select_from_bank, step_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_0509_cnn_no_combat.yaml")
    parser.add_argument("--bank", default="outputs/jax_initial_state_bank.npz")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    params = EnvParams(episode_steps=cfg.env.episode_steps, max_planets=cfg.env.max_planets, max_actions=cfg.env.max_planets)
    bank_path = Path(args.bank)
    if not bank_path.exists():
        build_bank(bank_path, 256, cfg.seed, params)
    bank = load_state_bank(bank_path, params)
    state = select_from_bank(bank, jnp.arange(args.num_envs) % bank.planets.shape[0])
    actions = jnp.zeros((args.num_envs, 2, params.max_actions, 3), dtype=jnp.float32)

    @jax.jit
    def loop(st):
        def body(_, carry):
            return step_batch(carry, actions, params)

        return jax.lax.fori_loop(0, args.steps, body, st)

    state = loop(state)
    jax.block_until_ready(state.planets)
    start = time.perf_counter()
    state = loop(state)
    jax.block_until_ready(state.planets)
    elapsed = time.perf_counter() - start
    env_steps = args.num_envs * args.steps
    print(f"env_steps={env_steps} elapsed={elapsed:.3f}s sps={env_steps / elapsed:.1f} devices={jax.devices()}")


if __name__ == "__main__":
    main()

