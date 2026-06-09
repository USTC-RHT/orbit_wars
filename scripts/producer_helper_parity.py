#!/usr/bin/env python
"""Helper-level parity checks for the JAX producer port."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_DIR = REPO_ROOT / "agents" / "external" / "slawekbiel-producer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PRODUCER_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCER_DIR))

from orbit_lite.geometry import fleet_speed as torch_fleet_speed  # noqa: E402
from orbit_lite.adapter import single_obs_to_tensor  # noqa: E402
from orbit_lite.distance_cache import build_distance_cache as torch_build_distance_cache  # noqa: E402
from orbit_lite.intercept_aim import intercept_angle as torch_intercept_angle  # noqa: E402
from orbit_lite.movement import MovementConfig, PlanetMovement, _build_future_from_obs as torch_build_future_from_obs  # noqa: E402
from orbit_lite.movement_step import disambiguate_duplicate_launches as torch_disambiguate_duplicate_launches  # noqa: E402
from orbit_lite.planner_core import _candidate_indices as torch_candidate_indices  # noqa: E402
from orbit_lite.planner_core import _greedy_select as torch_greedy_select  # noqa: E402
from orbit_lite.planner_core import build_target_shortlist as torch_build_target_shortlist  # noqa: E402
from orbit_lite.planner_core import capture_floor as torch_capture_floor  # noqa: E402
from orbit_lite.planner_core import entries_to_sparse_payload as torch_entries_to_sparse_payload  # noqa: E402
from orbit_lite.planner_core import make_launch_set as torch_make_launch_set  # noqa: E402
from orbit_lite.planner_core import reachable_mask as torch_reachable_mask  # noqa: E402
from orbit_lite.planner_core import safe_drain as torch_safe_drain  # noqa: E402
from orbit_lite.planner_core import score_candidates as torch_score_candidates  # noqa: E402
from orbit_lite.obs import parse_obs as torch_parse_obs  # noqa: E402
from main import ProducerLiteConfig, cheap_enemy_pressure as torch_cheap_enemy_pressure  # noqa: E402
from main import plan_lite_waves as torch_plan_lite_waves  # noqa: E402

from jax_agent.fast_env import EnvParams, state_from_observation  # noqa: E402
from jax_agent.train_opponents.producer_jax import (  # noqa: E402
    GarrisonStatusJax,
    ProducerJaxConfig,
    build_distance_cache_jax,
    build_target_shortlist_jax,
    candidate_indices_jax,
    capture_floor_jax,
    cheap_enemy_pressure_jax,
    fleet_speed_jax,
    future_positions_jax,
    garrison_status_from_obs_jax,
    greedy_select_jax,
    intercept_angle_jax,
    make_launch_set_jax,
    disambiguate_duplicate_launches_jax,
    entries_to_action_array_jax,
    entries_to_fixed_action_array_jax,
    plan_lite_waves_jax,
    plan_attack_waves_jax,
    producer_obs_from_state,
    reachable_mask_jax,
    safe_drain_jax,
    score_candidates_jax,
    stable_argmax_jax,
)


class DummyObs:
    def __init__(self, planets: torch.Tensor, alive: torch.Tensor, player_id: int) -> None:
        self.P = int(planets.shape[0])
        self.device = planets.device
        self.ships = planets[:, 5]
        self.owner_abs = planets[:, 1].to(torch.long)
        self.alive = alive
        self.player_id = player_id


class DummyCache:
    def __init__(self, cross_dist0: torch.Tensor) -> None:
        self.cross_dist = cross_dist0.unsqueeze(0)


def status_to_jax(status) -> GarrisonStatusJax:
    return GarrisonStatusJax(
        owner=jnp.asarray(status.owner.cpu().numpy()),
        ships=jnp.asarray(status.ships.cpu().numpy()),
        pre_combat_owner=(
            None if status.pre_combat_owner is None else jnp.asarray(status.pre_combat_owner.cpu().numpy())
        ),
        pre_combat_ships=(
            None if status.pre_combat_ships is None else jnp.asarray(status.pre_combat_ships.cpu().numpy())
        ),
        arrivals_by_owner=(
            None if status.arrivals_by_owner is None else jnp.asarray(status.arrivals_by_owner.cpu().numpy())
        ),
    )


def assert_close(name: str, got, expected, atol: float = 1.0e-5, rtol: float = 0.0) -> None:
    got_np = np.asarray(got)
    exp_np = np.asarray(expected)
    if got_np.shape != exp_np.shape or not np.allclose(got_np, exp_np, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(f"{name} mismatch\nJAX={got_np}\nTorch={exp_np}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--movement-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    ships = np.concatenate([
        np.array([1.0, 2.0, 10.0, 100.0, 1000.0, 5000.0], dtype=np.float32),
        rng.uniform(1.0, 900.0, size=32).astype(np.float32),
    ])
    assert_close("fleet_speed", fleet_speed_jax(jnp.asarray(ships)), torch_fleet_speed(torch.tensor(ships)))

    for _ in range(args.cases):
        p = int(rng.integers(2, 48))
        cap = int(rng.integers(1, p + 1))
        values = rng.normal(size=p).astype(np.float32)
        # Force ties often; stable tie-break is part of producer behavior.
        values[rng.choice(p, size=max(1, p // 5), replace=False)] = 1.0
        mask = rng.random(size=p) > 0.25
        j_idx, j_exists = candidate_indices_jax(jnp.asarray(values), jnp.asarray(mask), cap)
        t_idx, t_exists = torch_candidate_indices(torch.tensor(values), torch.tensor(mask), cap)
        if list(map(int, np.asarray(j_idx))) != list(map(int, t_idx.cpu().numpy())):
            raise AssertionError(f"candidate_indices idx mismatch values={values} mask={mask} cap={cap}")
        if list(map(bool, np.asarray(j_exists))) != list(map(bool, t_exists.cpu().numpy())):
            raise AssertionError(f"candidate_indices exists mismatch values={values} mask={mask} cap={cap}")

        j_arg = int(stable_argmax_jax(jnp.asarray(values)))
        t_arg = int(torch.where(torch.tensor(values) == torch.tensor(values).max(), torch.arange(p), torch.full((p,), p)).argmin())
        if j_arg != t_arg:
            raise AssertionError(f"stable_argmax mismatch values={values}: jax={j_arg} torch={t_arg}")

    for _ in range(args.cases):
        p = int(rng.integers(4, 32))
        planets = np.zeros((p, 7), dtype=np.float32)
        planets[:, 0] = np.arange(p, dtype=np.float32)
        planets[:, 1] = rng.choice([-1, 0, 1], size=p).astype(np.float32)
        planets[:, 2:4] = rng.uniform(0, 100, size=(p, 2)).astype(np.float32)
        planets[:, 5] = rng.uniform(0, 200, size=p).astype(np.float32)
        alive = rng.random(size=p) > 0.1
        dx = planets[:, None, 2] - planets[None, :, 2]
        dy = planets[:, None, 3] - planets[None, :, 3]
        cross_dist0 = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        j_pressure = cheap_enemy_pressure_jax(
            jnp.asarray(planets),
            jnp.asarray(alive),
            jnp.asarray(cross_dist0),
            horizon=18.0,
            player_id=1,
        )
        t_pressure = torch_cheap_enemy_pressure(
            DummyObs(torch.tensor(planets), torch.tensor(alive), 1),
            DummyCache(torch.tensor(cross_dist0)),
            horizon=18.0,
            player_id=1,
        )
        assert_close("cheap_enemy_pressure", j_pressure, t_pressure, atol=1.0e-2, rtol=5.0e-5)

    for _ in range(args.cases):
        p = int(rng.integers(8, 32))
        c = int(rng.integers(8, 40))
        l = int(rng.integers(1, 4))
        t = int(rng.integers(3, 10))
        w = int(rng.integers(2, 7))
        cand_src = rng.integers(0, p, size=(c, l), dtype=np.int64)
        cand_send = rng.integers(1, 40, size=(c, l)).astype(np.float32)
        cand_angle = rng.uniform(-3.14, 3.14, size=(c, l)).astype(np.float32)
        cand_eta = rng.uniform(1.0, 18.0, size=(c, l)).astype(np.float32)
        cand_active = rng.random(size=(c, l)) > 0.25
        cand_tgt_slot = rng.integers(0, p, size=(c,), dtype=np.int64)
        cand_tgt_short = rng.integers(0, t, size=(c,), dtype=np.int64)
        cand_is_def = rng.random(size=(c,)) > 0.7
        target_exists = rng.random(size=(t,)) > 0.15
        source_budget = rng.integers(10, 120, size=(p,)).astype(np.float32)
        score = rng.normal(loc=2.0, scale=3.0, size=(c,)).astype(np.float32)
        score[rng.choice(c, size=max(1, c // 6), replace=False)] = 3.0
        score[~target_exists[cand_tgt_short]] = -np.inf

        j_entries, j_leftover = greedy_select_jax(
            p=p,
            w=w,
            score=jnp.asarray(score),
            cand_src=jnp.asarray(cand_src),
            cand_send=jnp.asarray(cand_send),
            cand_angle=jnp.asarray(cand_angle),
            cand_eta=jnp.asarray(cand_eta),
            cand_active=jnp.asarray(cand_active),
            cand_tgt_slot=jnp.asarray(cand_tgt_slot),
            cand_tgt_short=jnp.asarray(cand_tgt_short),
            cand_is_def=jnp.asarray(cand_is_def),
            source_budget=jnp.asarray(source_budget),
            target_exists=jnp.asarray(target_exists),
            roi_threshold=1.5,
        )
        t_entries, t_leftover = torch_greedy_select(
            P=p,
            W=w,
            device=torch.device("cpu"),
            dtype=torch.float32,
            score=torch.tensor(score),
            cand_src=torch.tensor(cand_src, dtype=torch.long),
            cand_send=torch.tensor(cand_send),
            cand_angle=torch.tensor(cand_angle),
            cand_eta=torch.tensor(cand_eta),
            cand_active=torch.tensor(cand_active),
            cand_tgt_slot=torch.tensor(cand_tgt_slot, dtype=torch.long),
            cand_tgt_short=torch.tensor(cand_tgt_short, dtype=torch.long),
            cand_is_def=torch.tensor(cand_is_def),
            source_budget=torch.tensor(source_budget),
            target_exists=torch.tensor(target_exists),
            roi_threshold=1.5,
        )
        if not np.array_equal(np.asarray(j_entries.valid), t_entries.valid.cpu().numpy()):
            raise AssertionError("greedy valid mismatch")
        if not np.array_equal(np.asarray(j_entries.source_slots), t_entries.source_slots.cpu().numpy()):
            raise AssertionError("greedy source mismatch")
        if not np.array_equal(np.asarray(j_entries.target_slots), t_entries.target_slots.cpu().numpy()):
            raise AssertionError("greedy target mismatch")
        assert_close("greedy ships", j_entries.ships, t_entries.ships, atol=1.0e-5)
        assert_close("greedy angle", j_entries.angle, t_entries.angle, atol=1.0e-5)
        assert_close("greedy eta", j_entries.eta, t_entries.eta, atol=1.0e-5)
        assert_close("greedy leftover", j_leftover, t_leftover, atol=1.0e-5)

    if args.movement_cases > 0:
        from kaggle_environments import make as kag_make

        env = kag_make("orbit_wars", configuration={"seed": args.seed, "randomSeed": args.seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        checked = 0
        env_params = EnvParams(max_planets=64, max_fleets=256)
        while states[0]["status"] == "ACTIVE" and checked < args.movement_cases:
            obs = states[1]["observation"]
            obs_tensors = single_obs_to_tensor(obs, player_id=1)
            torch_future = torch_build_future_from_obs(obs_tensors, 18)
            state = state_from_observation(obs, env_params)
            producer_obs = producer_obs_from_state(state, player_id=1, player_count=2)
            x, y, alive = future_positions_jax(producer_obs, 18)
            assert_close("future_x", x, torch_future["x"], atol=1.0e-4, rtol=1.0e-5)
            assert_close("future_y", y, torch_future["y"], atol=1.0e-4, rtol=1.0e-5)
            if not np.array_equal(np.asarray(alive), torch_future["alive_by_step"].cpu().numpy()):
                raise AssertionError("future alive_by_step mismatch")

            movement = PlanetMovement.from_obs_tensors(
                obs_tensors,
                config=MovementConfig(movement_horizon=18, track_fleets=True, player_count=2),
            )
            status = movement.garrison_status(max_horizon=18)
            expected_status_jax = status_to_jax(status)
            status_jax = garrison_status_from_obs_jax(producer_obs, x, y, alive, horizon=18, player_count=2)
            if not np.array_equal(np.asarray(status_jax.owner), status.owner.cpu().numpy()):
                raise AssertionError("garrison owner mismatch")
            assert_close("garrison ships", status_jax.ships, status.ships, atol=1.0e-5)
            if not np.array_equal(
                np.asarray(status_jax.pre_combat_owner),
                status.pre_combat_owner.cpu().numpy(),
            ):
                raise AssertionError("garrison pre_combat_owner mismatch")
            assert_close("garrison pre_combat_ships", status_jax.pre_combat_ships, status.pre_combat_ships, atol=1.0e-5)
            assert_close("garrison arrivals", status_jax.arrivals_by_owner, expected_status_jax.arrivals_by_owner, atol=1.0e-5)
            torch_cache = torch_build_distance_cache(movement, max_k=18)
            j_cross, j_cache_alive = build_distance_cache_jax(x, y, alive, max_k=18)
            planets = np.asarray(producer_obs.planets)
            alive_np = np.asarray(producer_obs.planet_alive)
            valid_slots = np.flatnonzero(alive_np & (planets[:, 0] >= 0))
            if valid_slots.size >= 2:
                src_np = rng.choice(valid_slots, size=min(6, valid_slots.size), replace=False).astype(np.int64)
                tgt_np = rng.choice(valid_slots, size=min(7, valid_slots.size), replace=False).astype(np.int64)
                sizes_np = rng.uniform(1.0, 200.0, size=(src_np.shape[0], tgt_np.shape[0], 3)).astype(np.float32)
                eta_np = rng.uniform(2.0, 18.0, size=(tgt_np.shape[0],)).astype(np.float32)
                j_reach = reachable_mask_jax(
                    x,
                    y,
                    producer_obs.planets[:, 4],
                    source_idx=jnp.asarray(src_np),
                    target_idx=jnp.asarray(tgt_np),
                    fleet_sizes=jnp.asarray(sizes_np),
                    eta_cap=jnp.asarray(eta_np),
                )
                t_reach = torch_reachable_mask(
                    movement,
                    source_idx=torch.tensor(src_np, dtype=torch.long),
                    target_idx=torch.tensor(tgt_np, dtype=torch.long),
                    fleet_sizes=torch.tensor(sizes_np),
                    eta_cap=torch.tensor(eta_np),
                )
                if not np.array_equal(np.asarray(j_reach), t_reach.cpu().numpy()):
                    raise AssertionError("reachable_mask mismatch")

                sizes2_np = sizes_np[..., 0]
                j_aim = intercept_angle_jax(
                    x,
                    y,
                    producer_obs.planets[:, 4],
                    alive[0],
                    jnp.asarray(src_np)[:, None],
                    jnp.asarray(tgt_np)[None, :],
                    jnp.asarray(sizes2_np),
                )
                t_aim = torch_intercept_angle(
                    movement,
                    torch.tensor(src_np, dtype=torch.long).unsqueeze(1),
                    torch.tensor(tgt_np, dtype=torch.long).unsqueeze(0),
                    torch.tensor(sizes2_np),
                )
                if not np.array_equal(np.asarray(j_aim["viable"]), t_aim["viable"].cpu().numpy()):
                    raise AssertionError(
                        f"intercept viable mismatch\nJAX={np.asarray(j_aim['viable'])}\nTorch={t_aim['viable']}"
                    )
                viable_np = np.asarray(j_aim["viable"])
                if viable_np.any():
                    assert_close(
                        "intercept_angle",
                        np.asarray(j_aim["angle"])[viable_np],
                        t_aim["angle"].cpu().numpy()[viable_np],
                        atol=2.0e-5,
                        rtol=1.0e-5,
                    )
                assert_close("intercept_eta", j_aim["eta"], t_aim["eta"], atol=1.0e-5, rtol=0.0)

                active2_np = np.asarray(j_reach)[..., 0]
                j_aim_active = intercept_angle_jax(
                    x,
                    y,
                    producer_obs.planets[:, 4],
                    alive[0],
                    jnp.asarray(src_np)[:, None],
                    jnp.asarray(tgt_np)[None, :],
                    jnp.asarray(sizes2_np),
                    active=jnp.asarray(active2_np),
                )
                t_aim_active = torch_intercept_angle(
                    movement,
                    torch.tensor(src_np, dtype=torch.long).unsqueeze(1),
                    torch.tensor(tgt_np, dtype=torch.long).unsqueeze(0),
                    torch.tensor(sizes2_np),
                    active=torch.tensor(active2_np),
                )
                if not np.array_equal(np.asarray(j_aim_active["viable"]), t_aim_active["viable"].cpu().numpy()):
                    raise AssertionError("intercept active viable mismatch")
                active_viable_np = np.asarray(j_aim_active["viable"])
                if active_viable_np.any():
                    assert_close(
                        "intercept_active_angle",
                        np.asarray(j_aim_active["angle"])[active_viable_np],
                        t_aim_active["angle"].cpu().numpy()[active_viable_np],
                        atol=2.0e-5,
                        rtol=1.0e-5,
                    )
                assert_close("intercept_active_eta", j_aim_active["eta"], t_aim_active["eta"], atol=1.0e-5, rtol=0.0)

                source_ships_np = planets[src_np, 5].astype(np.float32)
                j_drain = safe_drain_jax(
                    status_jax,
                    source_idx=jnp.asarray(src_np),
                    source_ships=jnp.asarray(source_ships_np),
                    h_eff=jnp.asarray(18.0, dtype=jnp.float32),
                    player_id=1,
                )
                t_drain = torch_safe_drain(
                    status,
                    source_idx=torch.tensor(src_np, dtype=torch.long),
                    source_ships=torch.tensor(source_ships_np),
                    H_eff=torch.tensor(18.0),
                    player_id=1,
                )
                assert_close("safe_drain", j_drain, t_drain, atol=1.0e-5)

                target_np = tgt_np.astype(np.int64)
                j_floor = capture_floor_jax(
                    status_jax,
                    target_idx=jnp.asarray(target_np),
                    k_max=18,
                    capture_overhead=1.0,
                    player_id=1,
                )
                t_floor = torch_capture_floor(
                    status,
                    target_idx=torch.tensor(target_np, dtype=torch.long),
                    k_max=18,
                    capture_overhead=1.0,
                    player_id=1,
                )
                assert_close("capture_floor", j_floor, t_floor, atol=1.0e-5)

                score_c = min(24, src_np.shape[0] * tgt_np.shape[0])
                sc_src = rng.choice(src_np, size=score_c, replace=True).astype(np.int64).reshape(score_c, 1)
                sc_tgt = rng.choice(tgt_np, size=score_c, replace=True).astype(np.int64).reshape(score_c, 1)
                sc_ships = rng.uniform(1.0, 80.0, size=(score_c, 1)).astype(np.float32)
                sc_eta = rng.uniform(1.0, 18.0, size=(score_c, 1)).astype(np.float32)
                sc_valid = rng.random(size=(score_c, 1)) > 0.2
                j_launches = make_launch_set_jax(
                    source_slots=jnp.asarray(sc_src),
                    target_slots=jnp.asarray(sc_tgt),
                    ships=jnp.asarray(sc_ships),
                    eta=jnp.asarray(sc_eta),
                    valid=jnp.asarray(sc_valid),
                    player_id=1,
                )
                t_launches = torch_make_launch_set(
                    source_slots=torch.tensor(sc_src, dtype=torch.long),
                    target_slots=torch.tensor(sc_tgt, dtype=torch.long),
                    ships=torch.tensor(sc_ships),
                    eta=torch.tensor(sc_eta),
                    valid=torch.tensor(sc_valid),
                    player_id=1,
                )
                j_score = score_candidates_jax(
                    status_jax,
                    prod=jnp.asarray(movement.planet_prod.cpu().numpy()),
                    alive_by_step=jnp.asarray(movement.alive_by_step[:19].cpu().numpy()),
                    player_count=2,
                    launches=j_launches,
                    player_id=1,
                )
                t_score = torch_score_candidates(
                    status,
                    prod=movement.planet_prod,
                    alive_by_step=movement.alive_by_step[:19],
                    player_count=2,
                    launches=t_launches,
                    player_id=1,
                )
                assert_close("score_candidates", j_score, t_score, atol=2.0e-4, rtol=2.0e-5)

                torch_obs = torch_parse_obs(obs_tensors)
                torch_source_mask = torch_obs.owned & torch_obs.alive & (torch_obs.ships >= 4.0)
                j_owner = producer_obs.planets[:, 1].astype(jnp.int32)
                j_source_mask = (
                    producer_obs.planet_alive
                    & (producer_obs.planets[:, 0] >= 0.0)
                    & (j_owner == 1)
                    & (producer_obs.planets[:, 5] >= 4.0)
                )
                t_tgt, t_exists = torch_build_target_shortlist(
                    torch_obs,
                    obs_tensors,
                    status,
                    torch_cache,
                    config=ProducerLiteConfig(),
                    K_eta=18,
                    H=18,
                    prod=movement.planet_prod,
                    source_mask=torch_source_mask,
                )
                j_tgt, j_exists = build_target_shortlist_jax(
                    producer_obs,
                    status_jax,
                    j_cross,
                    j_cache_alive,
                    config=ProducerJaxConfig(),
                    k_eta=18,
                    h=18,
                    prod=jnp.asarray(movement.planet_prod.cpu().numpy()),
                    source_mask=j_source_mask,
                )
                if list(map(int, np.asarray(j_tgt))) != list(map(int, t_tgt.cpu().numpy())):
                    raise AssertionError(f"target shortlist idx mismatch\nJAX={np.asarray(j_tgt)}\nTorch={t_tgt}")
                if list(map(bool, np.asarray(j_exists))) != list(map(bool, t_exists.cpu().numpy())):
                    raise AssertionError(f"target shortlist exists mismatch\nJAX={np.asarray(j_exists)}\nTorch={t_exists}")

                j_wave_entries, _j_leftover_wave = plan_attack_waves_jax(
                    producer_obs,
                    status_jax,
                    x,
                    y,
                    alive,
                    j_cross,
                    j_cache_alive,
                    prod=jnp.asarray(movement.planet_prod.cpu().numpy()),
                    config=ProducerJaxConfig(enable_regroup=False),
                    player_count=2,
                )
                t_wave_entries = torch_plan_lite_waves(
                    movement=movement,
                    obs=torch_obs,
                    obs_tensors=obs_tensors,
                    cache=torch_cache,
                    garrison_status=status,
                    prod=movement.planet_prod,
                    alive_by_step=movement.alive_by_step[:19],
                    config=dataclasses.replace(ProducerLiteConfig(), enable_regroup=False),
                    player_count=2,
                )
                j_valid = np.asarray(j_wave_entries.valid)
                t_valid = t_wave_entries.valid.cpu().numpy()
                if int(j_valid.sum()) != int(t_valid.sum()):
                    raise AssertionError(
                        f"attack wave valid count mismatch JAX={j_valid.sum()} Torch={t_valid.sum()}"
                    )
                assert_close(
                    "attack_wave_source",
                    np.asarray(j_wave_entries.source_slots)[j_valid],
                    t_wave_entries.source_slots.cpu().numpy()[t_valid],
                    atol=0,
                )
                assert_close(
                    "attack_wave_target",
                    np.asarray(j_wave_entries.target_slots)[j_valid],
                    t_wave_entries.target_slots.cpu().numpy()[t_valid],
                    atol=0,
                )
                assert_close(
                    "attack_wave_ships",
                    np.asarray(j_wave_entries.ships)[j_valid],
                    t_wave_entries.ships.cpu().numpy()[t_valid],
                    atol=1.0e-5,
                )
                assert_close(
                    "attack_wave_angle",
                    np.asarray(j_wave_entries.angle)[j_valid],
                    t_wave_entries.angle.cpu().numpy()[t_valid],
                    atol=2.0e-5,
                    rtol=1.0e-5,
                )
                assert_close(
                    "attack_wave_eta",
                    np.asarray(j_wave_entries.eta)[j_valid],
                    t_wave_entries.eta.cpu().numpy()[t_valid],
                    atol=1.0e-5,
                )

                j_full_entries = disambiguate_duplicate_launches_jax(
                    plan_lite_waves_jax(
                        producer_obs,
                        status_jax,
                        x,
                        y,
                        alive,
                        j_cross,
                        j_cache_alive,
                        prod=jnp.asarray(movement.planet_prod.cpu().numpy()),
                        config=ProducerJaxConfig(enable_regroup=True),
                        player_count=2,
                    )
                )
                t_full_entries = torch_disambiguate_duplicate_launches(
                    torch_plan_lite_waves(
                        movement=movement,
                        obs=torch_obs,
                        obs_tensors=obs_tensors,
                        cache=torch_cache,
                        garrison_status=status,
                        prod=movement.planet_prod,
                        alive_by_step=movement.alive_by_step[:19],
                        config=ProducerLiteConfig(),
                        player_count=2,
                    )
                )
                j_full_valid = np.asarray(j_full_entries.valid)
                t_full_valid = t_full_entries.valid.cpu().numpy()
                if int(j_full_valid.sum()) != int(t_full_valid.sum()):
                    raise AssertionError(
                        f"full entries valid count mismatch JAX={j_full_valid.sum()} Torch={t_full_valid.sum()}"
                    )
                assert_close(
                    "full_source",
                    np.asarray(j_full_entries.source_slots)[j_full_valid],
                    t_full_entries.source_slots.cpu().numpy()[t_full_valid],
                    atol=0,
                )
                assert_close(
                    "full_target",
                    np.asarray(j_full_entries.target_slots)[j_full_valid],
                    t_full_entries.target_slots.cpu().numpy()[t_full_valid],
                    atol=0,
                )
                assert_close(
                    "full_ships",
                    np.asarray(j_full_entries.ships)[j_full_valid],
                    t_full_entries.ships.cpu().numpy()[t_full_valid],
                    atol=1.0e-5,
                )
                assert_close(
                    "full_angle",
                    np.asarray(j_full_entries.angle)[j_full_valid],
                    t_full_entries.angle.cpu().numpy()[t_full_valid],
                    atol=2.0e-5,
                    rtol=1.0e-5,
                )
                assert_close(
                    "full_eta",
                    np.asarray(j_full_entries.eta)[j_full_valid],
                    t_full_entries.eta.cpu().numpy()[t_full_valid],
                    atol=1.0e-5,
                )

                j_action_rows = entries_to_action_array_jax(
                    j_full_entries,
                    producer_obs.planets[:, 0].astype(jnp.int32),
                )
                j_fixed_rows = entries_to_fixed_action_array_jax(
                    j_full_entries,
                    producer_obs.planets[:, 0].astype(jnp.int32),
                    max_actions=64,
                )
                t_payload = torch_entries_to_sparse_payload(
                    t_full_entries,
                    planet_ids=obs_tensors["planets"][..., 0].long(),
                )
                j_rows_valid = np.asarray(j_action_rows)[j_full_valid]
                t_count = int(t_payload["counts"].item())
                if int(j_rows_valid.shape[0]) != t_count:
                    raise AssertionError(f"action row count mismatch JAX={j_rows_valid.shape[0]} Torch={t_count}")
                if t_count > 0:
                    assert_close(
                        "action_from_planet_id",
                        j_rows_valid[:, 0].astype(np.int32),
                        t_payload["from_planet_id"].cpu().numpy(),
                        atol=0,
                    )
                    assert_close(
                        "action_angle",
                        j_rows_valid[:, 1],
                        t_payload["angle"].cpu().numpy(),
                        atol=2.0e-5,
                        rtol=1.0e-5,
                    )
                    assert_close(
                        "action_num_ships",
                        j_rows_valid[:, 2],
                        t_payload["num_ships"].cpu().numpy(),
                        atol=1.0e-5,
                    )
                    assert_close(
                        "fixed_action_rows",
                        np.asarray(j_fixed_rows)[:t_count],
                        j_rows_valid,
                        atol=2.0e-5,
                        rtol=1.0e-5,
                    )
                fixed_padding = np.asarray(j_fixed_rows)[t_count:]
                assert_close("fixed_action_padding", fixed_padding, np.zeros_like(fixed_padding), atol=0.0)
            states = env.step([[], []])
            checked += 1

    print(f"producer helper parity passed: cases={args.cases} movement_cases={args.movement_cases}")


if __name__ == "__main__":
    main()
