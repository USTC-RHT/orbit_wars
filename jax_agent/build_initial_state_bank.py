"""为 JAX 模拟器生成可复用的 Orbit Wars 初始状态池。

训练时 reset 直接从 state bank 采样，避免每次都调用 Kaggle Python 环境。
彗星未来路径也在这里预生成，保证 JAX fast_env 可以在纯数组逻辑里复现比赛环境。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from .fast_env import COMETS_PER_SPAWN, COMET_SPAWN_STEPS, MAX_COMET_PATH, MAX_COMET_SPAWNS, EnvParams, state_from_observation


def extract_observation(player_state):
    if isinstance(player_state, dict):
        return player_state.get("observation", player_state)
    return getattr(player_state, "observation", player_state)


def build_bank(output: Path, seeds: int, start_seed: int, params: EnvParams) -> None:
    from kaggle_environments import make

    states = []
    comet_payloads = []
    orbit_wars = load_orbit_wars_module()
    for seed in range(start_seed, start_seed + seeds):
        env = make("orbit_wars", configuration={"seed": int(seed), "randomSeed": int(seed)}, debug=False)
        env.reset(num_agents=2)
        player_states = env.step([[], []])
        obs = extract_observation(player_states[0])
        state = state_from_observation(obs, params)
        states.append(state)
        comet_payloads.append(precompute_comets_for_seed(orbit_wars, seed, np.asarray(state.initial_planets), float(state.angular_velocity)))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        planets=np.stack([np.asarray(s.planets) for s in states]),
        initial_planets=np.stack([np.asarray(s.initial_planets) for s in states]),
        planet_alive=np.stack([np.asarray(s.planet_alive) for s in states]),
        fleets=np.stack([np.asarray(s.fleets) for s in states]),
        fleet_alive=np.stack([np.asarray(s.fleet_alive) for s in states]),
        step=np.stack([np.asarray(s.step) for s in states]),
        angular_velocity=np.stack([np.asarray(s.angular_velocity) for s in states]),
        next_fleet_id=np.stack([np.asarray(s.next_fleet_id) for s in states]),
        comet_paths=np.stack([payload["paths"] for payload in comet_payloads]),
        comet_lengths=np.stack([payload["lengths"] for payload in comet_payloads]),
        comet_ships=np.stack([payload["ships"] for payload in comet_payloads]),
        comet_valid=np.stack([payload["valid"] for payload in comet_payloads]),
    )
    print(f"wrote {output} seeds={seeds} max_planets={params.max_planets} max_fleets={params.max_fleets}")


def load_orbit_wars_module():
    try:
        from kaggle_environments.envs.orbit_wars import orbit_wars
        return orbit_wars
    except ImportError:
        pass

    import types

    path = Path(__file__).resolve().parents[2] / "orbit_wars.py"
    source = path.read_text(encoding="utf-8")
    source = source.split("dir_path = path.dirname(__file__)", 1)[0]
    module = types.ModuleType("orbit_wars_local")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def precompute_comets_for_seed(orbit_wars, seed: int, initial_planets: np.ndarray, angular_velocity: float) -> dict[str, np.ndarray]:
    paths_out = np.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=np.float32)
    lengths_out = np.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=np.int32)
    ships_out = np.zeros((MAX_COMET_SPAWNS,), dtype=np.float32)
    valid_out = np.zeros((MAX_COMET_SPAWNS,), dtype=bool)
    planets_list = initial_planets.tolist()
    planets_list = [row for row in planets_list if row[4] > 0]
    for spawn_i, spawn_step in enumerate(COMET_SPAWN_STEPS.tolist()):
        comet_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        comet_paths = orbit_wars.generate_comet_paths(
            planets_list,
            angular_velocity,
            spawn_step,
            set(),
            4.0,
            rng=comet_rng,
        )
        if not comet_paths:
            continue
        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        valid_out[spawn_i] = True
        ships_out[spawn_i] = float(comet_ships)
        for comet_i, path in enumerate(comet_paths[:COMETS_PER_SPAWN]):
            arr = np.asarray(path[:MAX_COMET_PATH], dtype=np.float32)
            paths_out[spawn_i, comet_i, : len(arr)] = arr
            lengths_out[spawn_i, comet_i] = len(arr)
    return {"paths": paths_out, "lengths": lengths_out, "ships": ships_out, "valid": valid_out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/jax_initial_state_bank.npz")
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=512)
    args = parser.parse_args()
    build_bank(
        Path(args.output),
        args.seeds,
        args.start_seed,
        EnvParams(max_planets=args.max_planets, max_fleets=args.max_fleets, max_actions=args.max_planets),
    )


if __name__ == "__main__":
    main()
