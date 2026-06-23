"""Pure NumPy + ONNX Runtime bridge for feature-row BC policies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .numpy_inference import (
    BOARD_SIZE,
    CENTER,
    COMETS_PER_SPAWN,
    INBOUND_HORIZON,
    LOCAL_PLANET_RADIUS,
    MAP_CHANNELS,
    MAP_GRID_SIZE,
    MAX_COMET_PATH,
    MAX_COMET_SPAWNS,
    MIN_LAUNCH,
    ROTATION_RADIUS_LIMIT,
    SHIP_RATIOS,
    SUN_RADIUS,
    _build_global_features_np,
    _build_map_features_np,
    _comet_planet_mask_np,
    _fleet_speed_np,
    _fleet_speed_scalar,
    _point_segment_dist_sq_np,
    _proto_score_np,
    _StateNP,
    _to_np,
    aim_with_prediction_np,
    obs_to_np_state,
)

FIGHT_PLANET_DIM = 28
FIGHT_RELATIVE_DIM = 22
DEFAULT_GARRISON_HORIZON = 18
LAUNCH_SURFACE_OFFSET = 0.1
TARGET_HIT_SURFACE_OFFSET = 0.0
REACHABILITY_EPS = 1e-4
FRIENDLY_REGROUP_MAX_TIME = 7.0
FRIENDLY_REGROUP_PRESSURE_DELTA_MIN = 0.25
CAPTURE_OVERHEAD = 1.0
NEG_INF = -1.0e9


class _BCFeatureParams(NamedTuple):
    max_planets: int
    ship_bucket_count: int
    max_ships: float
    max_production: float
    episode_steps: int


class _FuturePositions(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    alive: np.ndarray
    comet: np.ndarray


class _GarrisonStatus(NamedTuple):
    owner: np.ndarray
    ships: np.ndarray
    pre_combat_owner: np.ndarray
    pre_combat_ships: np.ndarray
    arrivals_by_owner: np.ndarray
    alive: np.ndarray


class _FlipStatus(NamedTuple):
    will_flip: np.ndarray
    flip_turn: np.ndarray
    urgency: np.ndarray


def _clip01(x: np.ndarray | float) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _circle_collision_hit_np(d2: np.ndarray, radius: np.ndarray | float) -> np.ndarray:
    return d2 < np.square(radius)


def _mirror_planet_indices_np(count: int) -> np.ndarray:
    idx = np.arange(int(count), dtype=np.int32)
    mirror = (idx // 4) * 4 + (3 - (idx % 4))
    return np.where(mirror < int(count), mirror, idx).astype(np.int32)


def _safe_logits_np(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, logits, NEG_INF).astype(np.float32, copy=False)
    empty = ~mask.any(axis=-1, keepdims=True)
    return np.where(empty, 0.0, masked).astype(np.float32, copy=False)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _ort_providers(ort: Any) -> list[str]:
    requested = os.environ.get("ORBIT_WARS_ONNX_PROVIDERS", "").strip()
    if requested:
        return [p.strip() for p in requested.split(",") if p.strip()]
    return ["CPUExecutionProvider"]


def _preload_ort_cuda_dependencies(ort: Any) -> None:
    if os.environ.get("ORBIT_WARS_ONNX_PRELOAD_CUDA", "1").lower() in {"0", "false", "no"}:
        return
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    preload_dirs: list[str | None] = []
    try:
        site_root = Path(getattr(ort, "__file__", "")).resolve().parents[1]
        preload_dirs.extend(str(path) for path in sorted((site_root / "nvidia").glob("*/lib")) if path.is_dir())
    except Exception:
        pass
    preload_dirs.extend(["", None])
    for directory in preload_dirs:
        try:
            preload(cuda=True, cudnn=True, msvc=False, directory=directory)
        except TypeError:
            try:
                preload(cuda=True, cudnn=True, directory=directory)
            except Exception:
                continue
        except Exception:
            continue


def _env_int(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 1)
    except (TypeError, ValueError):
        return int(default)


def _has_owned_ships(observation: Any, player: int) -> bool:
    rows = observation.get("planets", []) if isinstance(observation, dict) else getattr(observation, "planets", [])
    for row in rows or []:
        try:
            owner = int(row[1])
            ships = float(row[5])
        except Exception:
            owner = int(getattr(row, "owner", -1))
            ships = float(getattr(row, "ship_count", 0.0) or 0.0)
        if owner == int(player) and ships > 0.0:
            return True
    return False


def _canonicalize_owner_for_player_np(state: _StateNP, player: int) -> _StateNP:
    if int(player) == 0:
        return state

    def flip_owner(owner: np.ndarray) -> np.ndarray:
        return np.where(owner == 0.0, 1.0, np.where(owner == 1.0, 0.0, owner))

    planets = state.planets.copy()
    initial_planets = state.initial_planets.copy()
    fleets = state.fleets.copy()
    planets[:, 1] = flip_owner(planets[:, 1])
    initial_planets[:, 1] = flip_owner(initial_planets[:, 1])
    fleets[:, 1] = flip_owner(fleets[:, 1])
    return state._replace(planets=planets, initial_planets=initial_planets, fleets=fleets)


def _planet_positions_alive_over_horizon_np(state: _StateNP, horizon: int) -> _FuturePositions:
    planets = state.planets
    init = state.initial_planets
    turns = np.arange(int(horizon) + 1, dtype=np.float32)
    turns_i = np.arange(int(horizon) + 1, dtype=np.int32)
    comet_mask = _comet_planet_mask_np(state)

    dx = init[:, 2] - CENTER
    dy = init[:, 3] - CENTER
    orbit_r = np.sqrt(dx * dx + dy * dy)
    rotating = (orbit_r + init[:, 4] < ROTATION_RADIUS_LIMIT) & state.planet_alive & (~comet_mask)
    init_angle = np.arctan2(dy, dx)
    future_step = float(state.step) + turns - 1.0
    phi = init_angle[:, None] + float(state.angular_velocity) * future_step[None, :]
    px = CENTER + orbit_r[:, None] * np.cos(phi)
    py = CENTER + orbit_r[:, None] * np.sin(phi)
    px[:, 0] = planets[:, 2]
    py[:, 0] = planets[:, 3]
    px = np.where(rotating[:, None], px, planets[:, 2:3])
    py = np.where(rotating[:, None], py, planets[:, 3:4])
    alive = np.broadcast_to(state.planet_alive[:, None], (planets.shape[0], int(horizon) + 1)).copy()

    comet_slots = state.comet_slots.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    safe_slots = np.where(comet_slots >= 0, comet_slots, 0).astype(np.int32)
    group_active = state.comet_path_index[:, None] >= -1
    active = (state.comet_slots >= 0) & group_active
    active_flat = active.reshape(-1)

    path_index = state.comet_path_index[:, None, None] + turns_i[None, None, :]
    path_index = np.broadcast_to(path_index, (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, int(horizon) + 1))
    valid_t = active[:, :, None] & (path_index >= 0) & (path_index < state.comet_lengths[:, :, None])
    gather = np.clip(path_index, 0, MAX_COMET_PATH - 1)
    path_pos = np.take_along_axis(
        state.comet_paths,
        np.broadcast_to(gather[..., None], (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, int(horizon) + 1, 2)),
        axis=2,
    )
    current_xy = state.planets[safe_slots, 2:4].reshape(MAX_COMET_SPAWNS, COMETS_PER_SPAWN, 1, 2)
    path_pos = np.where(valid_t[..., None], path_pos, current_xy)
    flat_pos = path_pos.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN, int(horizon) + 1, 2)
    flat_valid = valid_t.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN, int(horizon) + 1)

    slot_onehot = np.zeros((MAX_COMET_SPAWNS * COMETS_PER_SPAWN, planets.shape[0]), dtype=np.float32)
    for i, slot in enumerate(safe_slots):
        if active_flat[i] and 0 <= slot < planets.shape[0]:
            slot_onehot[i, slot] = 1.0
    comet_has = slot_onehot.sum(axis=0) > 0.0
    flat_pos_for_einsum = flat_pos.astype(np.float16).astype(np.float32)
    comet_px = slot_onehot.T @ flat_pos_for_einsum[:, :, 0]
    comet_py = slot_onehot.T @ flat_pos_for_einsum[:, :, 1]
    comet_alive = (slot_onehot.T @ flat_valid.astype(np.float32)) > 0.0
    comet_alive[:, 0] = np.where(comet_has, state.planet_alive, comet_alive[:, 0])

    px = np.where(comet_has[:, None], comet_px, px)
    py = np.where(comet_has[:, None], comet_py, py)
    alive = np.where(comet_has[:, None], comet_alive, alive)
    return _FuturePositions(px.astype(np.float32), py.astype(np.float32), alive.astype(bool), comet_has.astype(bool))


def _feature_planet_positions_alive_over_horizon_np(state: _StateNP, horizon: int) -> _FuturePositions:
    planets = state.planets
    init = state.initial_planets
    turns = np.arange(int(horizon) + 1, dtype=np.float32)
    turns_i = np.arange(int(horizon) + 1, dtype=np.int32)
    comet_mask = _comet_planet_mask_np(state)

    dx = init[:, 2] - CENTER
    dy = init[:, 3] - CENTER
    orbit_r = np.sqrt(dx * dx + dy * dy)
    rotating = (orbit_r + init[:, 4] < ROTATION_RADIUS_LIMIT) & state.planet_alive & (~comet_mask)
    phi_cur = np.arctan2(planets[:, 3] - CENTER, planets[:, 2] - CENTER)
    phi = phi_cur[:, None] + float(state.angular_velocity) * turns[None, :]
    px = CENTER + orbit_r[:, None] * np.cos(phi)
    py = CENTER + orbit_r[:, None] * np.sin(phi)
    px = np.where(rotating[:, None], px, planets[:, 2:3])
    py = np.where(rotating[:, None], py, planets[:, 3:4])
    alive = np.broadcast_to(state.planet_alive[:, None], (planets.shape[0], int(horizon) + 1)).copy()

    comet_slots = state.comet_slots.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    safe_slots = np.where(comet_slots >= 0, comet_slots, 0).astype(np.int32)
    group_active = state.comet_path_index[:, None] >= -1
    active = (state.comet_slots >= 0) & group_active
    active_flat = active.reshape(-1)
    path_index = state.comet_path_index[:, None, None] + turns_i[None, None, :]
    path_index = np.broadcast_to(path_index, (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, int(horizon) + 1))
    valid_t = active[:, :, None] & (path_index >= 0) & (path_index < state.comet_lengths[:, :, None])
    gather = np.clip(path_index, 0, MAX_COMET_PATH - 1)
    path_pos = np.take_along_axis(
        state.comet_paths,
        np.broadcast_to(gather[..., None], (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, int(horizon) + 1, 2)),
        axis=2,
    )
    current_xy = state.planets[safe_slots, 2:4].reshape(MAX_COMET_SPAWNS, COMETS_PER_SPAWN, 1, 2)
    path_pos = np.where(valid_t[..., None], path_pos, current_xy)
    flat_pos = path_pos.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN, int(horizon) + 1, 2)
    flat_valid = valid_t.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN, int(horizon) + 1)
    slot_onehot = np.zeros((MAX_COMET_SPAWNS * COMETS_PER_SPAWN, planets.shape[0]), dtype=np.float32)
    for i, slot in enumerate(safe_slots):
        if active_flat[i] and 0 <= slot < planets.shape[0]:
            slot_onehot[i, slot] = 1.0
    comet_has = slot_onehot.sum(axis=0) > 0.0
    flat_pos_for_einsum = flat_pos.astype(np.float16).astype(np.float32)
    comet_px = slot_onehot.T @ flat_pos_for_einsum[:, :, 0]
    comet_py = slot_onehot.T @ flat_pos_for_einsum[:, :, 1]
    comet_alive = (slot_onehot.T @ flat_valid.astype(np.float32)) > 0.0
    comet_alive[:, 0] = np.where(comet_has, state.planet_alive, comet_alive[:, 0])
    px = np.where(comet_has[:, None], comet_px, px)
    py = np.where(comet_has[:, None], comet_py, py)
    alive = np.where(comet_has[:, None], comet_alive, alive)
    return _FuturePositions(px.astype(np.float32), py.astype(np.float32), alive.astype(bool), comet_has.astype(bool))


def _predict_fleet_first_contacts_np(
    state: _StateNP,
    horizon: int,
    positions: _FuturePositions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fleets = state.fleets
    planets = state.planets
    F = fleets.shape[0]
    P = planets.shape[0]
    steps = np.arange(1, int(horizon) + 1, dtype=np.int32)
    ts = steps.astype(np.float32)
    ships = np.maximum(fleets[:, 6], 1.0)
    speed = _fleet_speed_np(ships)
    vx = np.cos(fleets[:, 4]) * speed
    vy = np.sin(fleets[:, 4]) * speed
    fx_prev = fleets[:, 2:3] + vx[:, None] * (ts[None, :] - 1.0)
    fy_prev = fleets[:, 3:4] + vy[:, None] * (ts[None, :] - 1.0)
    fx_cur = fleets[:, 2:3] + vx[:, None] * ts[None, :]
    fy_cur = fleets[:, 3:4] + vy[:, None] * ts[None, :]
    in_bounds = (fx_cur >= 0.0) & (fx_cur <= BOARD_SIZE) & (fy_cur >= 0.0) & (fy_cur <= BOARD_SIZE)
    sun_hit = _circle_collision_hit_np(
        _point_segment_dist_sq_np(CENTER, CENTER, fx_prev, fy_prev, fx_cur, fy_cur),
        SUN_RADIUS,
    )

    px_old = positions.x[:, :-1]
    py_old = positions.y[:, :-1]
    px_new = positions.x[:, 1:]
    py_new = positions.y[:, 1:]
    alive_start = positions.alive[:, :-1]
    alive_now = positions.alive[:, 1:]

    move_d2 = _point_segment_dist_sq_np(
        px_old[None, :, :],
        py_old[None, :, :],
        fx_prev[:, None, :],
        fy_prev[:, None, :],
        fx_cur[:, None, :],
        fy_cur[:, None, :],
    )
    move_hit = _circle_collision_hit_np(move_d2, planets[None, :, None, 4])
    move_hit = move_hit & alive_start[None, :, :] & state.fleet_alive[:, None, None]
    move_hit = move_hit & in_bounds[:, None, :] & (~sun_hit[:, None, :])
    any_move_hit = move_hit.any(axis=1)
    survived_move = state.fleet_alive[:, None] & in_bounds & (~sun_hit) & (~any_move_hit)

    sweep_x = px_new - px_old
    sweep_y = py_new - py_old
    sweep_active = (sweep_x * sweep_x + sweep_y * sweep_y) > 1e-9
    sweep_d2 = _point_segment_dist_sq_np(
        fx_cur[:, None, :],
        fy_cur[:, None, :],
        px_old[None, :, :],
        py_old[None, :, :],
        px_new[None, :, :],
        py_new[None, :, :],
    )
    rotating_sweep_active = sweep_active & alive_now & (~positions.comet[:, None])
    rotating_hit = _circle_collision_hit_np(sweep_d2, planets[None, :, None, 4])
    rotating_hit = rotating_hit & rotating_sweep_active[None, :, :] & survived_move[:, None, :]
    any_rotating_hit = rotating_hit.any(axis=1)

    survived_rotating = survived_move & (~any_rotating_hit)
    comet_sweep_active = sweep_active & alive_now & positions.comet[:, None]
    comet_hit = _circle_collision_hit_np(sweep_d2, planets[None, :, None, 4])
    comet_hit = comet_hit & comet_sweep_active[None, :, :] & survived_rotating[:, None, :]

    large = np.int32(4 * (int(horizon) + 1) + 1)
    lethal_keys = np.where(state.fleet_alive[:, None] & ((~in_bounds) | sun_hit), 4 * steps[None, :], large)
    first_lethal_key = lethal_keys.min(axis=1)
    move_keys = (4 * steps).reshape(1, 1, int(horizon))
    rotating_keys = (4 * steps + 1).reshape(1, 1, int(horizon))
    comet_keys = (4 * steps + 2).reshape(1, 1, int(horizon))
    target_keys = np.minimum(
        np.where(move_hit, move_keys, large),
        np.minimum(np.where(rotating_hit, rotating_keys, large), np.where(comet_hit, comet_keys, large)),
    )
    first_per_planet = target_keys.min(axis=2)
    first_target_key = first_per_planet.min(axis=1)
    target_idx = first_per_planet.argmin(axis=1).astype(np.int32)
    valid = (first_target_key < large) & (first_target_key < first_lethal_key)
    first_t = (first_target_key // 4).astype(np.int32)
    return np.clip(target_idx, 0, P - 1), valid, np.clip(first_t, 0, int(horizon))


def _fleet_arrivals_by_owner_np(state: _StateNP, horizon: int, positions: _FuturePositions) -> np.ndarray:
    target, valid, first_t = _predict_fleet_first_contacts_np(state, horizon, positions)
    P = state.planets.shape[0]
    arrivals = np.zeros((P, int(horizon) + 1, 2), dtype=np.float32)
    owner = state.fleets[:, 1].astype(np.int32)
    owner_valid = (owner >= 0) & (owner < 2)
    valid = valid & owner_valid
    for i in range(state.fleets.shape[0]):
        if valid[i]:
            arrivals[int(np.clip(target[i], 0, P - 1)), int(np.clip(first_t[i], 0, horizon)), int(owner[i])] += float(state.fleets[i, 6])
    arrivals[:, 0, :] = 0.0
    return arrivals


def _garrison_status_from_arrivals_np(state: _StateNP, arrivals_by_owner: np.ndarray, alive_by_step: np.ndarray) -> _GarrisonStatus:
    horizon = arrivals_by_owner.shape[1] - 1
    P = state.planets.shape[0]
    owner = np.where(alive_by_step[:, 0], state.planets[:, 1].astype(np.int32), -1)
    ships = np.where(alive_by_step[:, 0], state.planets[:, 5].astype(np.float32), 0.0)
    prod = state.planets[:, 6].astype(np.float32)
    owner_hist = np.zeros((P, horizon + 1), dtype=np.int32)
    ships_hist = np.zeros((P, horizon + 1), dtype=np.float32)
    pre_owner_hist = np.zeros((P, horizon + 1), dtype=np.int32)
    pre_ships_hist = np.zeros((P, horizon + 1), dtype=np.float32)
    owner_hist[:, 0] = owner
    ships_hist[:, 0] = ships
    pre_owner_hist[:, 0] = owner
    pre_ships_hist[:, 0] = ships
    for k in range(1, horizon + 1):
        alive_before = alive_by_step[:, k - 1]
        alive_now = alive_by_step[:, k]
        produces = alive_before & (owner >= 0)
        ships = ships + np.where(produces, prod, 0.0)
        pre_owner = np.where(alive_now, owner, -1).astype(np.int32)
        pre_ships = np.where(alive_now, ships, 0.0).astype(np.float32)

        combat0 = arrivals_by_owner[:, k, 0]
        combat1 = arrivals_by_owner[:, k, 1]
        has0 = combat0 > 0.0
        has1 = combat1 > 0.0
        top_owner = np.where(combat0 >= combat1, 0, 1).astype(np.int32)
        top_ships = np.maximum(combat0, combat1)
        second_ships = np.minimum(combat0, combat1)
        survivor = np.where(has0 & has1, top_ships - second_ships, top_ships)
        survivor_positive = (has0 | has1) & (survivor > 0.0) & alive_now

        same_owner = pre_owner == top_owner
        new_ships_same = pre_ships + survivor
        new_ships_attack = pre_ships - survivor
        captured = new_ships_attack < 0.0
        next_owner = np.where(survivor_positive & (~same_owner) & captured, top_owner, pre_owner)
        next_ships = np.where(
            survivor_positive,
            np.where(same_owner, new_ships_same, np.where(captured, -new_ships_attack, new_ships_attack)),
            pre_ships,
        )
        owner = np.where(alive_now, next_owner, -1).astype(np.int32)
        ships = np.where(alive_now, next_ships, 0.0).astype(np.float32)
        owner_hist[:, k] = owner
        ships_hist[:, k] = ships
        pre_owner_hist[:, k] = pre_owner
        pre_ships_hist[:, k] = pre_ships
    return _GarrisonStatus(owner_hist, ships_hist, pre_owner_hist, pre_ships_hist, arrivals_by_owner, alive_by_step)


def _garrison_status_np(state: _StateNP, horizon: int = DEFAULT_GARRISON_HORIZON) -> _GarrisonStatus:
    positions = _planet_positions_alive_over_horizon_np(state, horizon)
    arrivals = _fleet_arrivals_by_owner_np(state, horizon, positions)
    return _garrison_status_from_arrivals_np(state, arrivals, positions.alive)


def _safe_drain_all_sources_np(status: _GarrisonStatus, source_ships: np.ndarray, h_eff: float, player_id: int = 0) -> np.ndarray:
    horizon = status.ships.shape[1] - 1
    if horizon <= 0:
        return np.zeros_like(source_ships, dtype=np.float32)
    traj_ships = status.ships[:, 1:]
    traj_owner = status.owner[:, 1:]
    turns = np.arange(1, horizon + 1, dtype=np.float32)
    held = (traj_owner == int(player_id)) & (traj_ships > 0.0) & (turns[None, :] <= float(h_eff))
    min_slack = np.where(held, traj_ships, np.inf).min(axis=1)
    cap = np.where(np.isfinite(min_slack), min_slack, source_ships.astype(np.float32))
    return np.clip(np.minimum(cap, source_ships), 0.0, None).astype(np.float32)


def _target_flip_status_from_garrison_np(state: _StateNP, status: _GarrisonStatus, player_id: int = 0) -> _FlipStatus:
    horizon = status.owner.shape[1] - 1
    current_owner = state.planets[:, 1].astype(np.int32)
    owned_now = state.planet_alive & (current_owner == int(player_id))
    if horizon <= 0:
        return _FlipStatus(np.zeros_like(owned_now), np.ones_like(current_owner), np.zeros_like(state.planets[:, 5]))
    future_owner = status.owner[:, 1:]
    flips = owned_now[:, None] & (future_owner != int(player_id))
    will_flip = flips.any(axis=1)
    first_flip = flips.astype(np.int32).argmax(axis=1).astype(np.int32) + 1
    flip_turn = np.where(will_flip, first_flip, horizon + 1).astype(np.int32)
    remaining = np.maximum(float(horizon) - flip_turn.astype(np.float32), 0.0)
    urgency_raw = state.planets[:, 5].astype(np.float32) + state.planets[:, 6].astype(np.float32) * remaining
    urgency = np.where(will_flip, urgency_raw, 0.0).astype(np.float32)
    return _FlipStatus(will_flip, flip_turn, urgency)


def _precompute_inbound_from_garrison_np(state: _StateNP, status: _GarrisonStatus) -> np.ndarray:
    arrivals = status.arrivals_by_owner.astype(np.float32)
    horizon = arrivals.shape[1] - 1
    default = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    if horizon <= 0:
        return np.broadcast_to(default[None, :], (state.planets.shape[0], 6)).copy()
    future_arrivals = arrivals[:, 1:, :]
    turns = np.arange(1, horizon + 1, dtype=np.float32)

    def owner_features(by_step: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ships = by_step.sum(axis=1)
        has_step = by_step > 0.0
        has_any = has_step.any(axis=1)
        eta_min = np.where(has_any, np.where(has_step, turns[None, :], 1.0e9).min(axis=1) / float(horizon), 1.0)
        eta_mean = np.where(
            ships > 0.0,
            (by_step * turns[None, :]).sum(axis=1) / np.maximum(ships, 1.0) / float(horizon),
            1.0,
        )
        return ships, eta_min, eta_mean

    my_ships, my_min, my_mean = owner_features(future_arrivals[:, :, 0])
    enemy_ships, enemy_min, enemy_mean = owner_features(future_arrivals[:, :, 1])
    inbound = np.stack([my_ships, enemy_ships, my_min, my_mean, enemy_min, enemy_mean], axis=-1).astype(np.float32)
    return np.where(state.planet_alive[:, None], inbound, default[None, :]).astype(np.float32)


def _enemy_pressure_np(state: _StateNP, horizon: float = float(DEFAULT_GARRISON_HORIZON), player_id: int = 0) -> np.ndarray:
    planets = state.planets
    alive = state.planet_alive
    owner = planets[:, 1].astype(np.int32)
    ships = planets[:, 5].astype(np.float32)
    xy = planets[:, 2:4]
    dist = np.sqrt(np.maximum(np.sum(np.square(xy[:, None, :] - xy[None, :, :]), axis=-1), 0.0))
    speeds = _fleet_speed_np(np.maximum(ships, 1.0))
    reach_dist = np.maximum(speeds[:, None] * float(horizon), 1e-6)
    decay = np.clip(1.0 - dist / reach_dist, 0.0, 1.0)
    enemy_source = alive & (owner >= 0) & (owner != int(player_id))
    not_self = ~np.eye(planets.shape[0], dtype=bool)
    valid = enemy_source[:, None] & alive[None, :] & not_self
    return np.where(valid, ships[:, None] * decay, 0.0).sum(axis=0).astype(np.float32)


def _reachable_full_send_np(
    src: np.ndarray,
    planets: np.ndarray,
    future_px: np.ndarray,
    future_py: np.ndarray,
    future_alive: np.ndarray,
    ships: float,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = future_px.shape[1] - 1
    ax = future_px[:, :-1]
    ay = future_py[:, :-1]
    bx = future_px[:, 1:]
    by = future_py[:, 1:]
    segment_alive = future_alive[:, :-1] & future_alive[:, 1:]
    d2 = _point_segment_dist_sq_np(src[2], src[3], ax, ay, bx, by)
    seg_dist = np.sqrt(np.maximum(d2, 0.0))
    gap = src[4] + planets[:, 4] + LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET
    surface_dist = np.maximum(seg_dist - gap[:, None], 0.0)
    turns = np.arange(1, horizon + 1, dtype=np.float32)
    speed = _fleet_speed_scalar(max(float(ships), 1.0))
    reachable_by_turn = (surface_dist <= (speed * turns[None, :] * (1.0 + REACHABILITY_EPS))) & segment_alive
    can_reach = (float(ships) >= MIN_LAUNCH) & reachable_by_turn.any(axis=1)
    eta = np.where(reachable_by_turn, turns[None, :], INBOUND_HORIZON + 1.0).min(axis=1)
    return can_reach, np.where(can_reach, eta, INBOUND_HORIZON + 1.0).astype(np.float32)


def _relative_eta_garrison_features_np(status: _GarrisonStatus, eta_turns: np.ndarray, params: _BCFeatureParams) -> np.ndarray:
    owner = status.pre_combat_owner
    ships = status.pre_combat_ships
    arrivals = status.arrivals_by_owner.astype(np.float32)
    horizon = owner.shape[1] - 1
    eta_ceil = np.ceil(eta_turns)
    eta_out = eta_ceil > float(horizon)
    k = np.clip(eta_ceil.astype(np.int32), 0, horizon)
    idx = np.arange(owner.shape[0], dtype=np.int32)
    owner_at_eta = owner[idx, k]
    ships_at_eta = ships[idx, k]
    step_ids = np.arange(horizon + 1, dtype=np.int32)
    step_mask = step_ids[None, :] <= k[:, None]
    my_arrivals = np.where(step_mask, arrivals[:, :, 0], 0.0).sum(axis=1)
    enemy_arrivals = np.where(step_mask, arrivals[:, :, 1], 0.0).sum(axis=1)
    return np.stack(
        [
            (owner_at_eta == -1).astype(np.float32),
            (owner_at_eta == 0).astype(np.float32),
            (owner_at_eta == 1).astype(np.float32),
            _clip01(ships_at_eta / params.max_ships),
            _clip01(my_arrivals / params.max_ships),
            _clip01(enemy_arrivals / params.max_ships),
            eta_out.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def _relative_capture_floor_features_np(
    status: _GarrisonStatus,
    eta_turns: np.ndarray,
    send_ships: float,
    params: _BCFeatureParams,
    player_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    owner = status.owner
    ships = status.ships
    horizon = owner.shape[1] - 1
    k = np.clip(np.ceil(eta_turns).astype(np.int32), 0, horizon)
    idx = np.arange(owner.shape[0], dtype=np.int32)
    owner_at_eta = owner[idx, k]
    defenders_at_eta = ships[idx, k]
    capture_floor = np.ceil(np.maximum(defenders_at_eta + CAPTURE_OVERHEAD, 1.0))
    floor = np.where(owner_at_eta == int(player_id), 1.0, capture_floor)
    return _clip01(floor / params.max_ships), np.clip((float(send_ships) - floor) / params.max_ships, -1.0, 1.0).astype(np.float32)


def _build_planet_fight_features_np(
    state: _StateNP,
    src_idx: int,
    inbound: np.ndarray,
    safe_drain: np.ndarray,
    params: _BCFeatureParams,
) -> np.ndarray:
    src = state.planets[src_idx]
    owner = int(src[1])
    alive = bool(state.planet_alive[src_idx])
    is_comet = bool(_comet_planet_mask_np(state)[src_idx])
    init = state.initial_planets[src_idx]
    orbit_r = np.sqrt((init[2] - CENTER) ** 2 + (init[3] - CENTER) ** 2)
    rotating = bool((orbit_r + init[4] < ROTATION_RADIUS_LIMIT) and alive and (not is_comet))
    is_static = bool(alive and (not rotating) and (not is_comet))
    topology = _local_topology_features_np(state, src_idx, params)
    inb = inbound[src_idx]
    safe_drain_ratio = np.clip(safe_drain[src_idx] / max(float(src[5]), 1.0), 0.0, 1.0)
    full_send_risk_ratio = np.clip((float(src[5]) - safe_drain[src_idx]) / max(float(src[5]), 1.0), 0.0, 1.0)
    return np.array(
        [
            1.0,
            float(alive),
            float(owner == -1),
            float(owner == 0),
            float(owner == 1),
            src[2] / BOARD_SIZE,
            src[3] / BOARD_SIZE,
            src[4] / 5.0,
            np.clip(src[5] / params.max_ships, 0.0, 1.0),
            src[6] / params.max_production,
            float(is_static),
            float(rotating),
            float(is_comet),
            np.sqrt((src[2] - CENTER) ** 2 + (src[3] - CENTER) ** 2) / BOARD_SIZE,
            *topology.tolist(),
            np.clip(inb[0] / params.max_ships, 0.0, 1.0),
            np.clip(inb[1] / params.max_ships, 0.0, 1.0),
            inb[2],
            inb[3],
            inb[4],
            inb[5],
            safe_drain_ratio,
            full_send_risk_ratio,
        ],
        dtype=np.float32,
    )


def _local_topology_features_np(state: _StateNP, src_idx: int, params: _BCFeatureParams) -> np.ndarray:
    planets = state.planets
    src = planets[src_idx]
    owner = planets[:, 1].astype(np.int32)
    valid = state.planet_alive & (np.arange(params.max_planets) != int(src_idx))
    dx = planets[:, 2] - src[2]
    dy = planets[:, 3] - src[3]
    dist = np.sqrt(dx * dx + dy * dy)
    masks = [valid & (owner == 0), valid & (owner == 1), valid & (owner == -1)]

    def nearest(mask: np.ndarray) -> float:
        return float(np.where(mask, dist, 1.0e9).min() / BOARD_SIZE) if mask.any() else 1.0

    def count_in_radius(mask: np.ndarray) -> float:
        return float((mask & (dist <= LOCAL_PLANET_RADIUS)).sum() / params.max_planets)

    return np.array([nearest(masks[0]), nearest(masks[1]), nearest(masks[2]), count_in_radius(masks[0]), count_in_radius(masks[1]), count_in_radius(masks[2])], dtype=np.float32)


def _bc_ship_buckets_np(source_ships: np.ndarray, bucket_count: int) -> np.ndarray:
    ships = np.floor(source_ships).astype(np.float32)
    ratios = SHIP_RATIOS[: int(bucket_count)].astype(np.float32)
    raw = np.round(ships[:, None] * ratios[None, :])
    raw = np.minimum(np.maximum(raw, 1.0), ships[:, None])
    return np.where(ships[:, None] > 0.0, raw, 0.0).astype(np.float32)


def _feature_ship_buckets_scalar_np(source_ships: float, bucket_count: int) -> np.ndarray:
    ships = np.floor(float(source_ships))
    ratios = SHIP_RATIOS[: int(bucket_count)].astype(np.float32)
    raw = np.minimum(np.maximum(np.round(ships * ratios), MIN_LAUNCH), ships)
    return np.where(ships >= MIN_LAUNCH, raw, np.zeros_like(raw)).astype(np.float32)


def _build_relative_fight_features_np(
    state: _StateNP,
    src_idx: int,
    status: _GarrisonStatus,
    enemy_pressure: np.ndarray,
    flip_status: _FlipStatus,
    future: _FuturePositions,
    params: _BCFeatureParams,
) -> np.ndarray:
    src = state.planets[src_idx]
    planets = state.planets
    dx = planets[:, 2] - src[2]
    dy = planets[:, 3] - src[3]
    dist = np.sqrt(dx * dx + dy * dy)
    owner = planets[:, 1].astype(np.int32)
    is_source = np.arange(params.max_planets) == int(src_idx)
    is_neutral = owner == -1
    is_friend = (owner == 0) & (~is_source)
    is_enemy = owner == 1
    target_valid = state.planet_alive & (~is_source) & (is_friend | is_enemy | is_neutral)
    ship_buckets = _feature_ship_buckets_scalar_np(float(src[5]), params.ship_bucket_count)
    max_bucket = max(float(ship_buckets.max()), 1.0)
    _reachable, eta_turns_raw = _reachable_full_send_np(src, planets, future.x, future.y, future.alive, max_bucket)
    eta_turns = np.where(target_valid, eta_turns_raw, INBOUND_HORIZON + 1.0).astype(np.float32)
    eta_norm = eta_turns / float(INBOUND_HORIZON)
    eta_garrison = _relative_eta_garrison_features_np(status, eta_turns, params)
    target_ships_at_eta = eta_garrison[:, 3] * params.max_ships
    my_arrivals = eta_garrison[:, 4] * params.max_ships
    enemy_arrivals = eta_garrison[:, 5] * params.max_ships
    capture_margin_at_eta = np.clip((max_bucket - target_ships_at_eta) / params.max_ships, -1.0, 1.0)
    pressure_delta = enemy_pressure - enemy_pressure[int(np.clip(src_idx, 0, enemy_pressure.shape[0] - 1))]
    pressure_delta_norm = np.clip(pressure_delta / params.max_ships, -1.0, 1.0)
    target_enemy_pressure_norm = _clip01(enemy_pressure / params.max_ships)
    arrival_balance_at_eta = np.clip((my_arrivals - enemy_arrivals) / params.max_ships, -1.0, 1.0)
    contestedness_at_eta = _clip01((my_arrivals + enemy_arrivals) / params.max_ships)
    capture_floor_at_eta, capture_surplus_at_eta = _relative_capture_floor_features_np(status, eta_turns, max_bucket, params, player_id=0)
    rel = np.stack(
        [
            is_source.astype(np.float32),
            dx / BOARD_SIZE,
            dy / BOARD_SIZE,
            dist / BOARD_SIZE,
            _clip01((planets[:, 5] + 1.0) / max(float(src[5]), 1.0)),
            np.full(params.max_planets, np.clip(src[5] / params.max_ships, 0.0, 1.0), dtype=np.float32),
            _proto_score_np(src, planets) / 100.0,
            eta_norm,
            eta_garrison[:, 0],
            eta_garrison[:, 1],
            eta_garrison[:, 2],
            eta_garrison[:, 3],
            eta_garrison[:, 4],
            eta_garrison[:, 5],
            eta_garrison[:, 6],
            pressure_delta_norm,
            target_enemy_pressure_norm,
            capture_margin_at_eta,
            arrival_balance_at_eta,
            contestedness_at_eta,
            capture_floor_at_eta,
            capture_surplus_at_eta,
        ],
        axis=-1,
    ).astype(np.float32)
    rel = np.where(target_valid[:, None], rel, 0.0)
    return np.concatenate([np.zeros((1, FIGHT_RELATIVE_DIM), dtype=np.float32), rel], axis=0)


def _build_bc_policy_rows_np(state: _StateNP, params: _BCFeatureParams) -> dict[str, np.ndarray]:
    P = params.max_planets
    status = _garrison_status_np(state, DEFAULT_GARRISON_HORIZON)
    safe_drain = _safe_drain_all_sources_np(status, state.planets[:, 5], h_eff=float(DEFAULT_GARRISON_HORIZON), player_id=0)
    enemy_pressure = _enemy_pressure_np(state, horizon=float(DEFAULT_GARRISON_HORIZON), player_id=0)
    flip_status = _target_flip_status_from_garrison_np(state, status, player_id=0)
    inbound = _precompute_inbound_from_garrison_np(state, status)
    future = _feature_planet_positions_alive_over_horizon_np(state, int(INBOUND_HORIZON))

    owner = state.planets[:, 1].astype(np.int32)
    source_mask = state.planet_alive & (owner == 0)
    source_ids = state.planets[:, 0].astype(np.int32)
    source_ships = state.planets[:, 5].astype(np.float32)
    candidate_mask = np.concatenate(
        [np.zeros((P, 1), dtype=bool), state.planet_alive[None, :] & (~np.eye(P, dtype=bool))],
        axis=1,
    )
    ship_buckets = _bc_ship_buckets_np(source_ships, params.ship_bucket_count)
    amount_mask = candidate_mask[:, :, None] & (ship_buckets > 0.0)[:, None, :]
    global_row = _build_global_features_np(state, P, params.max_ships, params.max_production, params.episode_steps)
    planet = np.zeros((P, FIGHT_PLANET_DIM), dtype=np.float32)
    relative = np.zeros((P, P + 1, FIGHT_RELATIVE_DIM), dtype=np.float32)
    map_features = np.zeros((P, MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE), dtype=np.float32)
    for i in np.flatnonzero(source_mask):
        idx = int(i)
        planet[idx] = _build_planet_fight_features_np(state, idx, inbound, safe_drain, params)
        relative[idx] = _build_relative_fight_features_np(state, idx, status, enemy_pressure, flip_status, future, params)
        map_features[idx] = _build_map_features_np(state, idx, params.max_ships)
    return {
        "planet": planet,
        "relative": relative,
        "global_features": np.broadcast_to(global_row[None, :], (P, global_row.shape[0])).astype(np.float32).copy(),
        "map_features": map_features,
        "candidate_mask": candidate_mask,
        "amount_mask": amount_mask,
        "source_mask": source_mask.astype(bool),
        "source_ids": source_ids,
        "ship_buckets": ship_buckets,
    }


def _bc_rows_to_player0_view_np(rows: dict[str, np.ndarray], player: int, max_planets: int) -> dict[str, np.ndarray]:
    if int(player) == 0:
        return rows
    planet_perm = _mirror_planet_indices_np(max_planets)
    candidate_perm = np.concatenate([np.array([0], dtype=np.int32), planet_perm + 1])
    mirrored = {k: np.array(v, copy=True) for k, v in rows.items()}
    mirrored["planet"][:, 5] = 1.0 - mirrored["planet"][:, 5]
    mirrored["planet"][:, 6] = 1.0 - mirrored["planet"][:, 6]
    mirrored["relative"] = mirrored["relative"][:, candidate_perm, :]
    mirrored["relative"][:, :, 1] *= -1.0
    mirrored["relative"][:, :, 2] *= -1.0
    mirrored["candidate_mask"] = mirrored["candidate_mask"][:, candidate_perm]
    mirrored["amount_mask"] = mirrored["amount_mask"][:, candidate_perm, :]
    mirrored["map_features"] = mirrored["map_features"][:, :, ::-1, ::-1]
    return mirrored


def _target_from_player0_view_np(target: np.ndarray, player: int, max_planets: int) -> np.ndarray:
    if int(player) == 0:
        return target.astype(np.int32, copy=False)
    planet_perm = _mirror_planet_indices_np(max_planets)
    target_slot = np.clip(target - 1, 0, max_planets - 1).astype(np.int32)
    mapped = np.where(target > 0, planet_perm[target_slot] + 1, 0).astype(np.int32)
    return mapped


def _read_meta(model_dir: Path) -> dict[str, Any]:
    meta_path = model_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing BC ONNX meta: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _first_model_dir(root: Path) -> Path:
    direct_meta = root / "meta.json"
    if direct_meta.is_file():
        return root
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing BC ONNX manifest/meta under: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models") or manifest.get("checkpoints") or []
    if not models:
        raise ValueError(f"empty BC ONNX manifest: {manifest_path}")
    first = models[0]
    if "name" in first:
        return root / str(first["name"])
    if "path" in first:
        return root / Path(str(first["path"])).parent
    if "checkpoint" in first:
        return root / Path(str(first["checkpoint"])).stem
    raise ValueError(f"cannot resolve BC ONNX model dir from manifest row: {first!r}")


@dataclass
class BCNumpyOnnxPolicyAgent:
    session: Any
    params: _BCFeatureParams
    max_fleets: int
    dropped_relative_feature_dims: tuple[int, ...] = ()
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()

    @classmethod
    def from_dir(cls, root: str | Path, *, max_fleets: int = 512) -> "BCNumpyOnnxPolicyAgent":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for BC ONNX policy eval") from exc
        providers = _ort_providers(ort)
        if any(str(provider).startswith("CUDAExecutionProvider") for provider in providers):
            _preload_ort_cuda_dependencies(ort)
        model_dir = _first_model_dir(Path(root))
        meta = _read_meta(model_dir)
        raw = meta.get("raw_config") or {}
        env = raw.get("env", {}) if isinstance(raw, dict) else {}
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = _env_int("ORBIT_WARS_ONNX_INTRA_THREADS", 1)
        sess_options.inter_op_num_threads = _env_int("ORBIT_WARS_ONNX_INTER_THREADS", 1)
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(
            str(model_dir / str(meta.get("onnx_file", "bc_policy.onnx"))),
            sess_options,
            providers=providers,
        )
        input_names = tuple(inp.name for inp in session.get_inputs())
        output_names = tuple(out.name for out in session.get_outputs())
        return cls(
            session=session,
            params=_BCFeatureParams(
                max_planets=int(meta.get("max_planets", env.get("max_planets", 45))),
                ship_bucket_count=int(meta.get("ship_bucket_count", env.get("ship_bucket_count", 8))),
                max_ships=float(env.get("max_ships", 500.0)),
                max_production=float(env.get("max_production", 5.0)),
                episode_steps=int(env.get("episode_steps", 500)),
            ),
            max_fleets=int(max_fleets),
            dropped_relative_feature_dims=tuple(int(x) for x in (meta.get("dropped_relative_feature_dims") or ())),
            input_names=input_names,
            output_names=output_names,
        )

    def build_rows(self, observation: Any, player: int) -> tuple[_StateNP, dict[str, np.ndarray]]:
        state = _to_np(obs_to_np_state(observation, self.params.max_planets, self.max_fleets))
        state = _canonicalize_owner_for_player_np(state, int(player))
        rows = _build_bc_policy_rows_np(state, self.params)
        return state, _bc_rows_to_player0_view_np(rows, int(player), self.params.max_planets)

    def _policy_inputs(self, rows_np: dict[str, np.ndarray], selected_targets: np.ndarray) -> dict[str, np.ndarray]:
        relative = rows_np["relative"].astype(np.float32, copy=bool(self.dropped_relative_feature_dims))
        if self.dropped_relative_feature_dims:
            relative[..., list(self.dropped_relative_feature_dims)] = 0.0
        feed = {
            "planet": rows_np["planet"].astype(np.float32, copy=False),
            "relative": relative,
            "global_features": rows_np["global_features"].astype(np.float32, copy=False),
            "candidate_mask": rows_np["candidate_mask"].astype(np.bool_, copy=False),
            "map_features": rows_np["map_features"].astype(np.float32, copy=False),
            "selected_targets": selected_targets.astype(np.int32, copy=False),
        }
        if self.input_names:
            feed = {name: value for name, value in feed.items() if name in self.input_names}
        return feed

    def _run_fire_target(self, rows_np: dict[str, np.ndarray], selected_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fire_logits, target_logits = self.session.run(
            ["fire_logits", "target_logits"],
            self._policy_inputs(rows_np, selected_targets),
        )
        return np.asarray(fire_logits, dtype=np.float32), np.asarray(target_logits, dtype=np.float32)

    def _run_amount(self, rows_np: dict[str, np.ndarray], selected_targets: np.ndarray) -> np.ndarray:
        (amount_logits,) = self.session.run(
            ["amount_logits"],
            self._policy_inputs(rows_np, selected_targets),
        )
        return np.asarray(amount_logits, dtype=np.float32)

    def _run_greedy(self, rows_np: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outputs = self.session.run(
            ["fire_logits", "target_logits", "amount_logits", "greedy_target"],
            self._policy_inputs(rows_np, np.zeros((self.params.max_planets,), dtype=np.int32)),
        )
        fire_logits, target_logits, amount_logits, greedy_target = outputs
        return (
            np.asarray(fire_logits, dtype=np.float32),
            np.asarray(target_logits, dtype=np.float32),
            np.asarray(amount_logits, dtype=np.float32),
            np.asarray(greedy_target, dtype=np.int32),
        )

    def act(self, observation: Any, *, player: int | None = None) -> list[list[float | int]]:
        if player is None:
            if isinstance(observation, dict):
                player = int(observation.get("player", 0) or 0)
            else:
                player = int(getattr(observation, "player", 0) or 0)
        if not _has_owned_ships(observation, int(player)):
            return []
        state, rows_np = self.build_rows(observation, int(player))
        use_greedy = "greedy_target" in self.output_names and "selected_targets" not in self.input_names
        zero_target = np.zeros((self.params.max_planets,), dtype=np.int32)
        if use_greedy:
            fire_logits, target_logits, amount_logits, greedy_target = self._run_greedy(rows_np)
        else:
            fire_logits, target_logits = self._run_fire_target(rows_np, zero_target)
        can_fire = rows_np["source_mask"] & rows_np["candidate_mask"][:, 1:].any(axis=-1)
        fire = (_sigmoid_np(fire_logits) >= 0.5) & can_fire
        if not bool(np.any(fire)):
            return []
        if use_greedy:
            target = greedy_target.astype(np.int32, copy=False)
        else:
            target_mask = rows_np["candidate_mask"].copy()
            target_mask[:, 0] = False
            target = np.argmax(_safe_logits_np(target_logits, target_mask), axis=-1).astype(np.int32)
        target = np.where(fire, target, 0).astype(np.int32)

        if not use_greedy:
            amount_logits = self._run_amount(rows_np, target)
        selected_amount_mask = np.take_along_axis(rows_np["amount_mask"], target[:, None, None], axis=1).squeeze(1)
        amount = np.argmax(_safe_logits_np(amount_logits, selected_amount_mask), axis=-1).astype(np.int32)
        amount = np.where(fire, amount, 0).astype(np.int32)

        target_env = _target_from_player0_view_np(target, int(player), self.params.max_planets)
        target_slot = np.clip(target_env[None] - 1, 0, self.params.max_planets - 1).astype(np.int32)
        ships = np.take_along_axis(rows_np["ship_buckets"][None], amount[None, :, None], axis=2).squeeze(-1)
        target_valid = np.take_along_axis(rows_np["candidate_mask"][None], target_env[None, :, None], axis=2).squeeze(-1)
        valid = fire[None] & rows_np["source_mask"][None] & target_valid & (target_env[None] > 0) & (ships > 0.0)
        valid_rows = np.flatnonzero(valid[0])
        if valid_rows.size == 0:
            return []
        source_planets = state.planets[valid_rows][None]
        target_planets = state.planets[target_slot[0, valid_rows]][None]
        selected_ships = ships[:, valid_rows]
        angle, _hit = aim_with_prediction_np(
            source_planets,
            target_planets,
            target_slot[:, valid_rows],
            state,
            selected_ships,
        )
        return [
            [float(rows_np["source_ids"][idx]), float(angle[0, j]), int(selected_ships[0, j])]
            for j, idx in enumerate(valid_rows)
        ]
