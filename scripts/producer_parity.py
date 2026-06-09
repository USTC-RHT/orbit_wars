#!/usr/bin/env python
"""Benchmark and parity harness for the external producer and its JAX port.

This script is deliberately separate from training.  It gathers real Kaggle
observations, records the original producer actions, and optionally compares a
candidate JAX implementation once one is available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_DIR = REPO_ROOT / "agents" / "external" / "slawekbiel-producer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jax_agent.fast_env import EnvParams, state_from_observation  # noqa: E402
from jax_agent.train_opponents.producer_jax import (  # noqa: E402
    empty_producer_memory_for_state,
    producer_opponent_actions,
    producer_opponent_plan_with_memory,
    update_batched_producer_memory_after_step_jax,
)


def load_external_producer():
    module_name = "_producer_parity_external"
    main_py = PRODUCER_DIR / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load producer from {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(PRODUCER_DIR))
    spec.loader.exec_module(module)
    return module


def canonical_moves(moves: list[list[Any]], *, angle_ndigits: int = 6) -> list[tuple[int, float, int]]:
    out: list[tuple[int, float, int]] = []
    for row in moves or []:
        if len(row) < 3:
            continue
        ships = int(row[2])
        if ships <= 0:
            continue
        out.append((int(row[0]), round(float(row[1]), angle_ndigits), ships))
    return out


def moves_match(a: list[list[Any]], b: list[list[Any]], *, angle_tol: float = 1.0e-5) -> bool:
    if len(a or []) != len(b or []):
        return False
    for lhs, rhs in zip(a or [], b or [], strict=True):
        if int(lhs[0]) != int(rhs[0]):
            return False
        if abs(float(lhs[1]) - float(rhs[1])) > angle_tol:
            return False
        if int(lhs[2]) != int(rhs[2]):
            return False
    return True


def jax_moves_from_obs(obs: dict[str, Any], *, max_planets: int, max_fleets: int) -> list[list[float]]:
    state = state_from_observation(obs, EnvParams(max_planets=max_planets, max_fleets=max_fleets))
    actions = producer_opponent_actions(state, max_actions=max_planets)
    rows = jax.device_get(actions)
    moves: list[list[float]] = []
    for row in rows:
        ships = int(row[2])
        if ships <= 0:
            continue
        moves.append([int(row[0]), float(row[1]), ships])
    return moves


def collect_observations(*, seeds: list[int], max_turns: int) -> list[dict[str, Any]]:
    from kaggle_environments import make as kag_make

    observations: list[dict[str, Any]] = []
    for seed in seeds:
        env = kag_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        turns = 0
        while states[0]["status"] == "ACTIVE" and turns < max_turns:
            observations.append(states[1]["observation"])
            # Use empty learner/opponent actions while collecting diverse early
            # frames.  Full self-play collection can be added once the JAX port
            # emits actions.
            states = env.step([[], []])
            turns += 1
    return observations


def benchmark_and_compare_live(
    producer,
    *,
    seeds: list[int],
    max_turns: int,
    max_planets: int,
    max_fleets: int,
    angle_tol: float,
    max_mismatches: int,
    stateful_jax: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from kaggle_environments import make as kag_make

    if hasattr(producer, "_RUNTIME"):
        producer._RUNTIME.reset()

    # Compile once on a real observation shape. Each later call reuses the same
    # fixed fast-env shapes.
    jax_fn = jax.jit(lambda st: producer_opponent_actions(st, max_actions=max_planets))
    jax_stateful_fn = jax.jit(lambda st, mem: producer_opponent_plan_with_memory(st, mem, max_actions=max_planets))
    jax_memory_update_fn = jax.jit(lambda mem, entries, prev, nxt: update_batched_producer_memory_after_step_jax(mem, entries, prev, nxt))
    producer_timings: list[float] = []
    jax_timings: list[float] = []
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for seed in seeds:
        env = kag_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        if hasattr(producer, "_RUNTIME"):
            producer._RUNTIME.reset()
        jax_memory = None
        turns = 0
        while states[0]["status"] == "ACTIVE" and turns < max_turns:
            obs = states[1]["observation"]
            step_value = obs.get("step", None) if isinstance(obs, dict) else getattr(obs, "step", None)
            explicit_step0 = step_value is not None and int(step_value) == 0
            if explicit_step0 and hasattr(producer, "_RUNTIME"):
                producer._RUNTIME.reset()

            state = state_from_observation(obs, EnvParams(max_planets=max_planets, max_fleets=max_fleets))
            start = time.perf_counter()
            jax_entries = None
            if stateful_jax:
                if jax_memory is None or explicit_step0:
                    jax_memory = empty_producer_memory_for_state(state)
                jax_rows, jax_entries = jax_stateful_fn(state, jax_memory)
                jax_rows.block_until_ready()
            else:
                jax_rows = jax_fn(state).block_until_ready()
            jax_elapsed = time.perf_counter() - start
            jax_timings.append(jax_elapsed)
            jax_moves: list[list[float]] = []
            for row in jax.device_get(jax_rows):
                ships = int(row[2])
                if ships > 0:
                    jax_moves.append([int(row[0]), float(row[1]), ships])

            start = time.perf_counter()
            producer_moves = producer.agent(obs)
            producer_elapsed = time.perf_counter() - start
            producer_timings.append(producer_elapsed)

            matched = moves_match(producer_moves, jax_moves, angle_tol=angle_tol)
            row = {
                "seed": seed,
                "turn": turns,
                "step": int(step_value) if step_value is not None else turns,
                "player": int(obs.get("player", 1)),
                "producer_actions": canonical_moves(producer_moves),
                "jax_actions": canonical_moves(jax_moves),
                "matched": matched,
                "producer_elapsed_ms": producer_elapsed * 1000.0,
                "jax_elapsed_ms": jax_elapsed * 1000.0,
                "fleet_count": len(obs.get("fleets", []) or []),
            }
            rows.append(row)
            if not matched:
                mismatches.append(row)
                if len(mismatches) >= int(max_mismatches):
                    break
            next_states = env.step([[], producer_moves])
            if stateful_jax and jax_memory is not None and jax_entries is not None:
                next_obs = next_states[1]["observation"]
                next_state = state_from_observation(next_obs, EnvParams(max_planets=max_planets, max_fleets=max_fleets))
                jax_memory = jax_memory_update_fn(jax_memory, jax_entries, state, next_state)
            states = next_states
            turns += 1

    prod_ms = [x * 1000.0 for x in producer_timings]
    jax_ms = [x * 1000.0 for x in jax_timings]
    summary = {
        "observations": len(rows),
        "matched": len(mismatches) == 0,
        "mismatches": len(mismatches),
        "max_mismatches": int(max_mismatches),
        "producer_mean_ms": statistics.mean(prod_ms) if prod_ms else 0.0,
        "producer_p95_ms": sorted(prod_ms)[max(int(len(prod_ms) * 0.95) - 1, 0)] if prod_ms else 0.0,
        "jax_mean_ms": statistics.mean(jax_ms) if jax_ms else 0.0,
        "jax_p95_ms": sorted(jax_ms)[max(int(len(jax_ms) * 0.95) - 1, 0)] if jax_ms else 0.0,
        "jax_calls_per_second": (len(jax_ms) / sum(jax_timings)) if jax_timings else 0.0,
        "angle_tol": angle_tol,
        "stateful_jax": bool(stateful_jax),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="12345,12346,12347")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--dump", type=Path, default=None)
    parser.add_argument("--compare-jax", action="store_true")
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=256)
    parser.add_argument("--angle-tol", type=float, default=1.0e-5)
    parser.add_argument("--max-mismatches", type=int, default=1)
    parser.add_argument("--stateful-jax", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    producer = load_external_producer()
    if args.compare_jax:
        summary, rows = benchmark_and_compare_live(
            producer,
            seeds=seeds,
            max_turns=args.max_turns,
            max_planets=args.max_planets,
            max_fleets=args.max_fleets,
            angle_tol=args.angle_tol,
            max_mismatches=args.max_mismatches,
            stateful_jax=args.stateful_jax,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.dump is not None:
            args.dump.parent.mkdir(parents=True, exist_ok=True)
            args.dump.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
            print(f"wrote {args.dump}")
        if not summary["matched"]:
            raise SystemExit(1)
        return

    observations = collect_observations(seeds=seeds, max_turns=args.max_turns)
    if hasattr(producer, "_RUNTIME"):
        producer._RUNTIME.reset()

    timings: list[float] = []
    rows: list[dict[str, Any]] = []
    for idx, obs in enumerate(observations):
        step_value = obs.get("step", None) if isinstance(obs, dict) else getattr(obs, "step", None)
        if idx == 0 or (step_value is not None and int(step_value) == 0):
            if hasattr(producer, "_RUNTIME"):
                producer._RUNTIME.reset()
        start = time.perf_counter()
        moves = producer.agent(obs)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        rows.append(
            {
                "index": idx,
                "step": int(step_value) if step_value is not None else idx,
                "player": int(obs.get("player", 1)),
                "actions": canonical_moves(moves),
                "elapsed_ms": elapsed * 1000.0,
            }
        )

    ms = [x * 1000.0 for x in timings]
    summary = {
        "observations": len(observations),
        "mean_ms": statistics.mean(ms) if ms else 0.0,
        "median_ms": statistics.median(ms) if ms else 0.0,
        "p95_ms": sorted(ms)[max(int(len(ms) * 0.95) - 1, 0)] if ms else 0.0,
        "max_ms": max(ms) if ms else 0.0,
        "calls_per_second": (len(ms) / sum(timings)) if timings else 0.0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
