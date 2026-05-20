"""Small parity smoke test against the Kaggle interpreter.

This is intentionally a smoke test, not the final exhaustive parity suite.
It compares no-op trajectories, including comet spawn/movement when a state
bank with precomputed comet paths is supplied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

from .fast_env import EnvParams, load_state_bank, select_from_bank, state_from_observation, step_batch  # noqa: E402


def extract_observation(player_state):
    if isinstance(player_state, dict):
        return player_state.get("observation", player_state)
    return getattr(player_state, "observation", player_state)


def read(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_0509_cnn_no_combat.yaml")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--bank", default=None)
    parser.add_argument("--bank-index", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)

    from kaggle_environments import make

    params = EnvParams(episode_steps=cfg.env.episode_steps, max_planets=cfg.env.max_planets, max_actions=cfg.env.max_planets)
    env = make("orbit_wars", configuration={"seed": args.seed, "randomSeed": args.seed}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])
    obs = extract_observation(states[0])
    if args.bank is None:
        if args.steps >= 49:
            raise SystemExit("steps >= 49 requires --bank with precomputed comet paths")
        jax_state = state_from_observation(obs, params)
        jax_state = jax.tree.map(lambda x: x[None, ...] if getattr(x, "ndim", 0) >= 0 else x, jax_state)
    else:
        bank = load_state_bank(args.bank, params)
        jax_state = select_from_bank(bank, jnp.asarray([args.bank_index], dtype=jnp.int32))
    actions = jnp.zeros((1, 2, params.max_actions, 3), dtype=jnp.float32)

    for _ in range(args.steps):
        states = env.step([[], []])
        jax_state = step_batch(jax_state, actions, params)

    obs = extract_observation(states[0])
    kg_planets = np.asarray(read(obs, "planets", []), dtype=np.float32)
    jx_planets = np.asarray(jax_state.planets[0])
    alive = np.asarray(jax_state.planet_alive[0])
    jx_planets = jx_planets[alive][: len(kg_planets)]
    common = min(len(kg_planets), len(jx_planets))
    pos_diff = np.abs(kg_planets[:common, 2:4] - jx_planets[:common, 2:4]).max(initial=0.0)
    ships_diff = np.abs(kg_planets[:common, 5] - jx_planets[:common, 5]).max(initial=0.0)
    owner_match = bool(np.array_equal(kg_planets[:common, 1].astype(np.int32), jx_planets[:common, 1].astype(np.int32)))
    comet_count_kg = int((kg_planets[:, 4] == 1.0).sum())
    comet_count_jx = int((jx_planets[:, 4] == 1.0).sum())
    print(
        f"seed={args.seed} steps={args.steps} planets={common} "
        f"pos_max_abs_diff={pos_diff:.6f} ships_max_abs_diff={ships_diff:.6f} "
        f"owner_match={owner_match} comets={comet_count_kg}/{comet_count_jx}"
    )


if __name__ == "__main__":
    main()
