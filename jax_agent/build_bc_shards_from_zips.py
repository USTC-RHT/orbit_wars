"""Build per-planet BC shards directly from official daily replay zip files.

The official replay action is ``[from_planet_id, angle, ships]`` and does not
store an explicit target. This script infers the first planet hit by that launch
using the same fixed-shape state representation as training, then writes shards
compatible with ``train_bc_jax.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .fast_env import (
    BOARD_SIZE,
    CENTER,
    COMETS_PER_SPAWN,
    MAX_COMET_PATH,
    MAX_COMET_SPAWNS,
    SUN_RADIUS,
    EnvParams,
    comet_arrays_from_observation,
)
from .features_jax import SHIP_RATIOS


STATE_FIELDS = (
    "planets",
    "initial_planets",
    "planet_alive",
    "fleets",
    "fleet_alive",
    "step",
    "angular_velocity",
    "next_fleet_id",
    "done",
    "winner",
    "overflow_fleets",
    "comet_paths",
    "comet_lengths",
    "comet_ships",
    "comet_valid",
    "comet_path_index",
    "comet_slots",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/manifests/orbit_2p_manifest.csv"))
    parser.add_argument("--zip-dir", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/daily_zips_cli"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/bc_shards/official_2p_winners_2026-06-01_2026-06-16"))
    parser.add_argument("--max-rows", type=int, default=0, help="0 = no row limit")
    parser.add_argument("--max-replays", type=int, default=0, help="0 = no replay limit")
    parser.add_argument("--max-replays-per-zip", type=int, default=0, help="0 = no per-day replay limit")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--noop-mode", choices=["all", "none", "stride"], default="all")
    parser.add_argument("--noop-stride", type=int, default=0, help="Only used when --noop-mode=stride.")
    parser.add_argument("--max-action-tokens", type=int, default=16)
    parser.add_argument("--learn-players", choices=["winners", "all"], default="winners")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--trajectory-cache", action="store_true")
    parser.add_argument("--max-planets", type=int, default=45)
    parser.add_argument("--max-fleets", type=int, default=512)
    parser.add_argument("--report-interval", type=float, default=10.0)
    return parser.parse_args()


def fleet_speed_np(ships: float, max_speed: float = 6.0) -> float:
    ships = max(float(ships), 1.0)
    ratio = min(max(math.log(ships) / math.log(1000.0), 0.0), 1.0)
    return min(1.0 + (max_speed - 1.0) * (ratio**1.5), max_speed)


def point_segment_distance_sq_np(px, py, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    denom = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / denom, 0.0, 1.0)
    cx = x0 + t * dx
    cy = y0 + t * dy
    ddx = px - cx
    ddy = py - cy
    return ddx * ddx + ddy * ddy


def comet_planet_mask_np(state: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros((state["planets"].shape[0],), dtype=bool)
    slots = state["comet_slots"].reshape(-1)
    active = ((state["comet_path_index"][:, None] >= -1) & (state["comet_slots"] >= 0)).reshape(-1)
    valid = (slots >= 0) & active
    mask[slots[valid].astype(np.int32)] = True
    return mask


def planet_positions_at_turn(state: dict[str, np.ndarray], turn: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planets = state["planets"]
    initial = state["initial_planets"]
    alive = state["planet_alive"].copy()
    dx = initial[:, 2] - CENTER
    dy = initial[:, 3] - CENTER
    orbit_r = np.sqrt(dx * dx + dy * dy)
    comet_mask = comet_planet_mask_np(state)
    rotating = (orbit_r + initial[:, 4] < 50.0) & alive & (~comet_mask)
    phi_cur = np.arctan2(planets[:, 3] - CENTER, planets[:, 2] - CENTER)
    phi = phi_cur + float(state["angular_velocity"]) * turn
    px = np.where(rotating, CENTER + orbit_r * np.cos(phi), planets[:, 2])
    py = np.where(rotating, CENTER + orbit_r * np.sin(phi), planets[:, 3])

    for group in range(MAX_COMET_SPAWNS):
        path_index = int(state["comet_path_index"][group])
        if path_index < -1:
            continue
        for comet in range(COMETS_PER_SPAWN):
            slot = int(state["comet_slots"][group, comet])
            length = int(state["comet_lengths"][group, comet])
            if slot < 0 or length <= 0:
                continue
            fi = path_index + float(turn)
            i0 = int(math.floor(fi))
            if i0 < 0 or i0 >= length:
                alive[slot] = False
                continue
            i1 = min(i0 + 1, length - 1)
            frac = fi - i0
            pos0 = state["comet_paths"][group, comet, i0]
            pos1 = state["comet_paths"][group, comet, i1]
            pos = pos0 + frac * (pos1 - pos0) if i1 > i0 else pos0
            px[slot] = pos[0]
            py[slot] = pos[1]
            alive[slot] = True
    return px, py, alive


def infer_target_slot(state: dict[str, np.ndarray], source_slot: int, angle: float, ships: float, horizon: int) -> int:
    planets = state["planets"]
    if source_slot < 0 or source_slot >= planets.shape[0] or not state["planet_alive"][source_slot]:
        return -1
    src = planets[source_slot]
    speed = fleet_speed_np(ships)
    cos_a = math.cos(float(angle))
    sin_a = math.sin(float(angle))
    fx0 = float(src[2]) + cos_a * (float(src[4]) + 0.1)
    fy0 = float(src[3]) + sin_a * (float(src[4]) + 0.1)

    for turn in range(1, int(horizon) + 1):
        fx_prev = fx0 + (turn - 1) * speed * cos_a
        fy_prev = fy0 + (turn - 1) * speed * sin_a
        fx_cur = fx0 + turn * speed * cos_a
        fy_cur = fy0 + turn * speed * sin_a
        if fx_cur < 0.0 or fx_cur > BOARD_SIZE or fy_cur < 0.0 or fy_cur > BOARD_SIZE:
            return -1
        if point_segment_distance_sq_np(CENTER, CENTER, fx_prev, fy_prev, fx_cur, fy_cur) < SUN_RADIUS * SUN_RADIUS:
            return -1

        px_old, py_old, alive_old = planet_positions_at_turn(state, float(turn - 1))
        px_new, py_new, alive_new = planet_positions_at_turn(state, float(turn))
        radii = planets[:, 4]
        move_d2 = point_segment_distance_sq_np(px_old, py_old, fx_prev, fy_prev, fx_cur, fy_cur)
        move_hit = (move_d2 < radii * radii) & alive_old
        sweep_active = ((px_new - px_old) ** 2 + (py_new - py_old) ** 2) > 1e-9
        sweep_d2 = point_segment_distance_sq_np(fx_cur, fy_cur, px_old, py_old, px_new, py_new)
        sweep_hit = (sweep_d2 < radii * radii) & alive_new & sweep_active
        move_hit[source_slot] = False
        sweep_hit[source_slot] = False
        if move_hit.any():
            return int(np.argmax(move_hit))
        if sweep_hit.any():
            return int(np.argmax(sweep_hit))
    return -1


def planet_trajectory(state: dict[str, np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = [planet_positions_at_turn(state, float(turn)) for turn in range(int(horizon) + 1)]
    px = np.stack([item[0] for item in positions], axis=0)
    py = np.stack([item[1] for item in positions], axis=0)
    alive = np.stack([item[2] for item in positions], axis=0)
    return px, py, alive


def infer_target_slot_with_trajectory(
    state: dict[str, np.ndarray],
    source_slot: int,
    angle: float,
    ships: float,
    horizon: int,
    future_px: np.ndarray,
    future_py: np.ndarray,
    future_alive: np.ndarray,
) -> int:
    planets = state["planets"]
    if source_slot < 0 or source_slot >= planets.shape[0] or not state["planet_alive"][source_slot]:
        return -1
    src = planets[source_slot]
    speed = fleet_speed_np(ships)
    cos_a = math.cos(float(angle))
    sin_a = math.sin(float(angle))
    fx0 = float(src[2]) + cos_a * (float(src[4]) + 0.1)
    fy0 = float(src[3]) + sin_a * (float(src[4]) + 0.1)
    radii = planets[:, 4]

    for turn in range(1, int(horizon) + 1):
        fx_prev = fx0 + (turn - 1) * speed * cos_a
        fy_prev = fy0 + (turn - 1) * speed * sin_a
        fx_cur = fx0 + turn * speed * cos_a
        fy_cur = fy0 + turn * speed * sin_a
        if fx_cur < 0.0 or fx_cur > BOARD_SIZE or fy_cur < 0.0 or fy_cur > BOARD_SIZE:
            return -1
        if point_segment_distance_sq_np(CENTER, CENTER, fx_prev, fy_prev, fx_cur, fy_cur) < SUN_RADIUS * SUN_RADIUS:
            return -1

        px_old = future_px[turn - 1]
        py_old = future_py[turn - 1]
        alive_old = future_alive[turn - 1]
        px_new = future_px[turn]
        py_new = future_py[turn]
        alive_new = future_alive[turn]
        move_d2 = point_segment_distance_sq_np(px_old, py_old, fx_prev, fy_prev, fx_cur, fy_cur)
        move_hit = (move_d2 < radii * radii) & alive_old
        sweep_active = ((px_new - px_old) ** 2 + (py_new - py_old) ** 2) > 1e-9
        sweep_d2 = point_segment_distance_sq_np(fx_cur, fy_cur, px_old, py_old, px_new, py_new)
        sweep_hit = (sweep_d2 < radii * radii) & alive_new & sweep_active
        move_hit[source_slot] = False
        sweep_hit[source_slot] = False
        if move_hit.any():
            return int(np.argmax(move_hit))
        if sweep_hit.any():
            return int(np.argmax(sweep_hit))
    return -1


def action_amount_bucket(source_ships: float, action_ships: float, bucket_count: int = 8) -> int:
    ships = math.floor(float(source_ships))
    if ships <= 0 or action_ships <= 0:
        return -1
    ratios = np.asarray(SHIP_RATIOS[:bucket_count], dtype=np.float32)
    buckets = np.clip(np.round(ships * ratios), 1.0, ships)
    if not (buckets > 0).any():
        return -1
    return int(np.argmin(np.abs(buckets - float(action_ships))))


def canonicalize_state_for_player(state: dict[str, np.ndarray], player: int) -> dict[str, np.ndarray]:
    if int(player) == 0:
        return state
    out = {k: v.copy() for k, v in state.items()}

    def flip_owner(values):
        return np.where(values == 0.0, 1.0, np.where(values == 1.0, 0.0, values))

    out["planets"][:, 1] = flip_owner(out["planets"][:, 1])
    out["initial_planets"][:, 1] = flip_owner(out["initial_planets"][:, 1])
    out["fleets"][:, 1] = flip_owner(out["fleets"][:, 1])
    return out


def state_dict_from_observation(obs: dict[str, Any], step: int, params: EnvParams, player: int) -> dict[str, np.ndarray]:
    obs = dict(obs)
    obs["step"] = int(step)
    planets_np = np.zeros((params.max_planets, 7), dtype=np.float32)
    initial_np = np.zeros((params.max_planets, 7), dtype=np.float32)
    fleets_np = np.zeros((params.max_fleets, 7), dtype=np.float32)
    planet_alive = np.zeros((params.max_planets,), dtype=bool)
    fleet_alive = np.zeros((params.max_fleets,), dtype=bool)

    planets = list(obs.get("planets", []) or [])[: params.max_planets]
    initial = list(obs.get("initial_planets", planets) or planets)[: params.max_planets]
    fleets = list(obs.get("fleets", []) or [])[: params.max_fleets]
    planet_id_to_slot: dict[int, int] = {}
    if planets:
        planets_np[: len(planets)] = np.asarray(planets, dtype=np.float32)
        planet_alive[: len(planets)] = True
        planet_id_to_slot = {int(row[0]): i for i, row in enumerate(planets)}
    if initial:
        initial_np[: len(initial)] = np.asarray(initial, dtype=np.float32)
    else:
        initial_np[:] = planets_np
    if fleets:
        fleets_np[: len(fleets)] = np.asarray(fleets, dtype=np.float32)
        fleet_alive[: len(fleets)] = True

    comet_paths, comet_lengths, comet_ships, comet_valid, comet_path_index, comet_slots = comet_arrays_from_observation(
        obs,
        planet_id_to_slot,
        planets_np,
    )
    state = {
        "planets": planets_np,
        "initial_planets": initial_np,
        "planet_alive": planet_alive,
        "fleets": fleets_np,
        "fleet_alive": fleet_alive,
        "step": np.asarray(int(obs.get("step", 0) or 0), dtype=np.int32),
        "angular_velocity": np.asarray(float(obs.get("angular_velocity", 0.0) or 0.0), dtype=np.float32),
        "next_fleet_id": np.asarray(int(obs.get("next_fleet_id", 0) or 0), dtype=np.int32),
        "done": np.asarray(False),
        "winner": np.asarray(-1, dtype=np.int32),
        "overflow_fleets": np.asarray(0, dtype=np.int32),
        "comet_paths": comet_paths,
        "comet_lengths": comet_lengths,
        "comet_ships": comet_ships,
        "comet_valid": comet_valid,
        "comet_path_index": comet_path_index,
        "comet_slots": comet_slots,
    }
    return canonicalize_state_for_player(state, player)


def empty_labels(max_action_tokens: int, max_planets: int) -> dict[str, np.ndarray]:
    return {
        "end_action": np.zeros((max_action_tokens,), dtype=bool),
        "src_action": np.zeros((max_action_tokens,), dtype=np.int32),
        "tgt_action": np.zeros((max_action_tokens,), dtype=np.int32),
        "amt_action": np.zeros((max_action_tokens,), dtype=np.int32),
        "executed": np.zeros((max_action_tokens,), dtype=bool),
        "token_valid": np.zeros((max_action_tokens,), dtype=bool),
        "source_valid": np.ones((max_planets,), dtype=bool),
    }


def label_actions(
    state: dict[str, np.ndarray],
    actions: Any,
    max_action_tokens: int,
    horizon: int,
    use_trajectory_cache: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    labels = empty_labels(max_action_tokens, int(state["planets"].shape[0]))
    stats = {
        "raw_actions": 0,
        "kept_actions": 0,
        "skipped_actions": 0,
        "invalid_action_actions": 0,
        "missing_source_actions": 0,
        "wrong_owner_actions": 0,
        "no_target_actions": 0,
        "comet_hit_actions": 0,
        "bad_amount_actions": 0,
    }
    rows = []
    alive = np.asarray(state["planet_alive"], dtype=bool)
    planet_ids = state["planets"][:, 0].astype(np.int32)
    id_to_slot = {int(planet_ids[i]): int(i) for i in np.flatnonzero(alive)}
    trajectory: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    comet_mask: np.ndarray | None = None
    for action in actions or []:
        stats["raw_actions"] += 1
        if not isinstance(action, (list, tuple)) or len(action) < 3:
            stats["skipped_actions"] += 1
            stats["invalid_action_actions"] += 1
            continue
        src_id = int(action[0])
        angle = float(action[1])
        ships = float(action[2])
        src_slot = id_to_slot.get(src_id, -1)
        if src_slot < 0:
            stats["skipped_actions"] += 1
            stats["missing_source_actions"] += 1
            continue
        if int(state["planets"][src_slot, 1]) != 0:
            stats["skipped_actions"] += 1
            stats["wrong_owner_actions"] += 1
            continue
        if use_trajectory_cache and trajectory is None:
            trajectory = planet_trajectory(state, horizon)
        if trajectory is None:
            target_slot = infer_target_slot(state, src_slot, angle, ships, horizon)
        else:
            target_slot = infer_target_slot_with_trajectory(state, src_slot, angle, ships, horizon, *trajectory)
        amount = action_amount_bucket(float(state["planets"][src_slot, 5]), ships)
        if target_slot < 0:
            labels["source_valid"][src_slot] = False
            stats["skipped_actions"] += 1
            stats["no_target_actions"] += 1
            continue
        if comet_mask is None:
            comet_mask = comet_planet_mask_np(state)
        if comet_mask[target_slot]:
            labels["source_valid"][src_slot] = False
            stats["skipped_actions"] += 1
            stats["comet_hit_actions"] += 1
            continue
        if amount < 0:
            stats["skipped_actions"] += 1
            stats["bad_amount_actions"] += 1
            continue
        rows.append((src_slot, target_slot, amount, ships))
        stats["kept_actions"] += 1

    rows = sorted(rows, key=lambda item: (item[0], -item[3], item[1]))[: max_action_tokens - 1]
    for i, (src_slot, target_slot, amount, _ships) in enumerate(rows):
        labels["src_action"][i] = src_slot
        labels["tgt_action"][i] = target_slot
        labels["amt_action"][i] = amount
        labels["executed"][i] = True
        labels["token_valid"][i] = True
    end_idx = min(len(rows), max_action_tokens - 1)
    labels["end_action"][end_idx] = True
    labels["token_valid"][end_idx] = True
    return labels, stats


def append_row(buffer: dict[str, list[np.ndarray]], state: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> None:
    for name in STATE_FIELDS:
        buffer[f"state__{name}"].append(state[name])
    for name, value in labels.items():
        buffer[f"label__{name}"].append(value)


def new_buffer() -> dict[str, list[np.ndarray]]:
    return {f"state__{name}": [] for name in STATE_FIELDS} | {
        f"label__{name}": []
        for name in ("end_action", "src_action", "tgt_action", "amt_action", "executed", "token_valid", "source_valid")
    }


def write_shard(buffer: dict[str, list[np.ndarray]], output_dir: Path, shard_index: int, meta: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.stack(values, axis=0) for name, values in buffer.items()}
    arrays["meta__json"] = np.asarray(json.dumps(meta, default=str, sort_keys=True))
    path = output_dir / f"bc_shard_{shard_index:05d}.npz"
    np.savez_compressed(path, **arrays)
    return path


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def write_progress(
    output_dir: Path,
    args: argparse.Namespace,
    total: dict[str, int],
    shards: list[dict[str, Any]],
    started: float,
    zip_i: int,
    zip_count: int,
    zip_path: str,
    done: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    elapsed = time.monotonic() - started
    progress = {
        "args": jsonable_args(args),
        "stats": dict(total),
        "shards_written": len(shards),
        "last_shard": shards[-1] if shards else None,
        "elapsed_sec": elapsed,
        "replay_per_sec": total["replays"] / max(elapsed, 1e-9),
        "row_per_sec": total["rows"] / max(elapsed, 1e-9),
        "keep_rate": total["kept_actions"] / max(total["raw_actions"], 1),
        "no_target_rate": total.get("no_target_actions", 0) / max(total["raw_actions"], 1),
        "zip_index": zip_i,
        "zip_count": zip_count,
        "zip_path": str(zip_path),
        "done": done,
    }
    (output_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, sort_keys=True) + "\n")


def file_snapshot(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    stat = p.stat()
    return {
        "path": str(p),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def write_input_snapshot(output_dir: Path, args: argparse.Namespace, rows: list[dict[str, str]], by_zip: dict[str, list[dict[str, str]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    zips = []
    for zip_path, zip_rows in by_zip.items():
        snap = file_snapshot(zip_path)
        snap["replays"] = len(zip_rows)
        zips.append(snap)
    snapshot = {
        "args": jsonable_args(args),
        "manifest": file_snapshot(args.manifest),
        "manifest_rows_loaded": len(rows),
        "zip_count": len(by_zip),
        "zips": zips,
        "note": "Inputs are read-only. Raw replay zips and source manifest are not modified by this builder.",
    }
    (output_dir / "input_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    shutil.copy2(args.manifest, output_dir / "source_manifest_copy.csv")


def manifest_rows(path: Path, max_replays: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if max_replays > 0:
        rows = rows[:max_replays]
    return rows


def selected_players(rewards: Any, mode: str) -> list[int]:
    if mode == "all" or not isinstance(rewards, list) or not rewards:
        return [0, 1]
    best = max(rewards)
    return [idx for idx, reward in enumerate(rewards[:2]) if reward == best and reward > 0]


def process_replay(replay: dict[str, Any], params: EnvParams, args: argparse.Namespace) -> tuple[list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]], dict[str, int]]:
    steps = replay.get("steps") or []
    if not steps or len(steps[0]) != 2:
        return [], {
            "replays": 0,
            "rows": 0,
            "raw_actions": 0,
            "kept_actions": 0,
            "skipped_actions": 0,
            "invalid_action_actions": 0,
            "missing_source_actions": 0,
            "wrong_owner_actions": 0,
            "no_target_actions": 0,
            "comet_hit_actions": 0,
            "bad_amount_actions": 0,
        }
    players = selected_players(replay.get("rewards"), args.learn_players)
    rows = []
    stats = {
        "replays": 1,
        "rows": 0,
        "raw_actions": 0,
        "kept_actions": 0,
        "skipped_actions": 0,
        "invalid_action_actions": 0,
        "missing_source_actions": 0,
        "wrong_owner_actions": 0,
        "no_target_actions": 0,
        "comet_hit_actions": 0,
        "bad_amount_actions": 0,
    }
    for t in range(1, len(steps)):
        prev_step = steps[t - 1]
        cur_step = steps[t]
        step_num = int((prev_step[0].get("observation") or {}).get("step") or (t - 1))
        for player in players:
            if player >= len(prev_step) or player >= len(cur_step):
                continue
            obs = prev_step[player].get("observation") or {}
            actions = cur_step[player].get("action") or []
            labels: dict[str, np.ndarray]
            state = state_dict_from_observation(obs, step_num, params, player)
            labels, label_stats = label_actions(
                state,
                actions,
                int(args.max_action_tokens),
                int(args.horizon),
                use_trajectory_cache=bool(getattr(args, "trajectory_cache", False)),
            )
            for key in (
                "raw_actions",
                "kept_actions",
                "skipped_actions",
                "invalid_action_actions",
                "missing_source_actions",
                "wrong_owner_actions",
                "no_target_actions",
                "comet_hit_actions",
                "bad_amount_actions",
            ):
                stats[key] += label_stats[key]
            has_action = bool(labels["executed"].any())
            raw_action_count = len(actions) if isinstance(actions, list) else 0
            keep_noop = (
                raw_action_count == 0
                and (
                    args.noop_mode == "all"
                    or (args.noop_mode == "stride" and int(args.noop_stride) > 0 and step_num % int(args.noop_stride) == 0)
                )
            )
            if has_action or keep_noop:
                rows.append((state, labels))
                stats["rows"] += 1
    return rows, stats


def main() -> None:
    args = parse_args()
    params = EnvParams(max_planets=int(args.max_planets), max_fleets=int(args.max_fleets), max_actions=int(args.max_planets))
    rows = manifest_rows(args.manifest, int(args.max_replays))
    buffer = new_buffer()
    shard_index = 1
    total = {
        "replays": 0,
        "rows": 0,
        "raw_actions": 0,
        "kept_actions": 0,
        "skipped_actions": 0,
        "invalid_action_actions": 0,
        "missing_source_actions": 0,
        "wrong_owner_actions": 0,
        "no_target_actions": 0,
        "comet_hit_actions": 0,
        "bad_amount_actions": 0,
    }
    shards: list[dict[str, Any]] = []
    started = time.monotonic()
    last_report = started

    by_zip: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_zip.setdefault(row["zip_path"], []).append(row)
    if int(args.max_replays_per_zip) > 0:
        by_zip = {path: zip_rows[: int(args.max_replays_per_zip)] for path, zip_rows in by_zip.items()}
    write_input_snapshot(args.output_dir, args, rows, by_zip)

    audit_log = args.output_dir / "audit_log.jsonl"
    replay_audit = args.output_dir / "replay_audit.jsonl"
    append_jsonl(
        audit_log,
        {
            "event": "run_start",
            "time": time.time(),
            "args": jsonable_args(args),
            "manifest_rows_loaded": len(rows),
            "zip_count": len(by_zip),
        },
    )

    stop = False
    for zip_i, (zip_path, zip_rows) in enumerate(by_zip.items(), start=1):
        if stop:
            break
        append_jsonl(
            audit_log,
            {
                "event": "zip_start",
                "time": time.time(),
                "zip_index": zip_i,
                "zip_count": len(by_zip),
                "zip_path": zip_path,
                "zip_replays": len(zip_rows),
                "input": file_snapshot(zip_path),
            },
        )
        with zipfile.ZipFile(zip_path) as zf:
            for row in zip_rows:
                if int(args.max_rows) > 0 and total["rows"] >= int(args.max_rows):
                    stop = True
                    break
                try:
                    replay = json.loads(zf.read(row["json_name"]))
                except Exception:
                    total["skipped_actions"] += 1
                    append_jsonl(
                        replay_audit,
                        {
                            "event": "replay_error",
                            "time": time.time(),
                            "zip_path": zip_path,
                            "json_name": row.get("json_name"),
                            "episode_id": row.get("episode_id"),
                        },
                    )
                    continue
                replay_rows, stats = process_replay(replay, params, args)
                for key, value in stats.items():
                    if key == "rows":
                        continue
                    total[key] += int(value)
                appended_rows = 0
                for state, labels in replay_rows:
                    if int(args.max_rows) > 0 and total["rows"] >= int(args.max_rows):
                        stop = True
                        break
                    append_row(buffer, state, labels)
                    total["rows"] += 1
                    appended_rows += 1
                    if len(buffer["state__step"]) >= int(args.shard_size):
                        meta = {"args": vars(args), "stats": dict(total), "shard_index": shard_index}
                        path = write_shard(buffer, args.output_dir, shard_index, meta)
                        shards.append({"path": str(path), "rows": int(len(buffer["state__step"]))})
                        print(f"[shard] wrote {path} rows={shards[-1]['rows']} total_rows={total['rows']}", flush=True)
                        append_jsonl(
                            audit_log,
                            {
                                "event": "shard_write",
                                "time": time.time(),
                                "path": str(path),
                                "shard_index": shard_index,
                                "rows": int(len(buffer["state__step"])),
                                "total": dict(total),
                            },
                        )
                        write_progress(args.output_dir, args, total, shards, started, zip_i, len(by_zip), zip_path)
                        shard_index += 1
                        buffer = new_buffer()
                append_jsonl(
                    replay_audit,
                    {
                        "event": "replay_done",
                        "time": time.time(),
                        "zip_index": zip_i,
                        "zip_path": zip_path,
                        "date": row.get("date"),
                        "json_name": row.get("json_name"),
                        "episode_id": row.get("episode_id"),
                        "player_count": row.get("player_count"),
                        "selected_mode": args.learn_players,
                        "candidate_rows": int(stats.get("rows", 0)),
                        "appended_rows": int(appended_rows),
                        "raw_actions": int(stats.get("raw_actions", 0)),
                        "kept_actions": int(stats.get("kept_actions", 0)),
                        "skipped_actions": int(stats.get("skipped_actions", 0)),
                        "invalid_action_actions": int(stats.get("invalid_action_actions", 0)),
                        "missing_source_actions": int(stats.get("missing_source_actions", 0)),
                        "wrong_owner_actions": int(stats.get("wrong_owner_actions", 0)),
                        "no_target_actions": int(stats.get("no_target_actions", 0)),
                        "comet_hit_actions": int(stats.get("comet_hit_actions", 0)),
                        "bad_amount_actions": int(stats.get("bad_amount_actions", 0)),
                        "total_rows_after": int(total["rows"]),
                    },
                )
                now = time.monotonic()
                if now - last_report >= float(args.report_interval):
                    elapsed = now - started
                    rate = total["replays"] / max(elapsed, 1e-9)
                    keep_rate = total["kept_actions"] / max(total["raw_actions"], 1)
                    no_target_rate = total["no_target_actions"] / max(total["raw_actions"], 1)
                    print(
                        f"[progress] zips={zip_i}/{len(by_zip)} replays={total['replays']} rows={total['rows']} "
                        f"raw_actions={total['raw_actions']} kept_actions={total['kept_actions']} "
                        f"keep_rate={keep_rate:.3f} no_target={no_target_rate:.3f} rate={rate:.1f} replay/s",
                        flush=True,
                    )
                    append_jsonl(
                        audit_log,
                        {
                            "event": "progress",
                            "time": time.time(),
                            "zip_index": zip_i,
                            "zip_count": len(by_zip),
                            "zip_path": zip_path,
                            "elapsed_sec": elapsed,
                            "total": dict(total),
                            "replay_per_sec": rate,
                            "keep_rate": keep_rate,
                            "no_target_rate": no_target_rate,
                        },
                    )
                    write_progress(args.output_dir, args, total, shards, started, zip_i, len(by_zip), zip_path)
                    last_report = now
        print(f"[zip] {Path(zip_path).name} cumulative_replays={total['replays']} rows={total['rows']}", flush=True)
        append_jsonl(
            audit_log,
            {
                "event": "zip_done",
                "time": time.time(),
                "zip_index": zip_i,
                "zip_count": len(by_zip),
                "zip_path": zip_path,
                "total": dict(total),
            },
        )
        write_progress(args.output_dir, args, total, shards, started, zip_i, len(by_zip), zip_path)

    if buffer["state__step"]:
        meta = {"args": vars(args), "stats": dict(total), "shard_index": shard_index}
        path = write_shard(buffer, args.output_dir, shard_index, meta)
        shards.append({"path": str(path), "rows": int(len(buffer["state__step"]))})
        print(f"[shard] wrote {path} rows={shards[-1]['rows']} total_rows={total['rows']}", flush=True)
        append_jsonl(
            audit_log,
            {
                "event": "shard_write",
                "time": time.time(),
                "path": str(path),
                "shard_index": shard_index,
                "rows": int(len(buffer["state__step"])),
                "total": dict(total),
            },
        )
        write_progress(args.output_dir, args, total, shards, started, len(by_zip), len(by_zip), "final")

    manifest = {
        "args": jsonable_args(args),
        "stats": total,
        "shards": shards,
        "elapsed_sec": time.monotonic() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_progress(args.output_dir, args, total, shards, started, len(by_zip), len(by_zip), "done", done=True)
    append_jsonl(
        audit_log,
        {
            "event": "run_done",
            "time": time.time(),
            "manifest": str(args.output_dir / "manifest.json"),
            "elapsed_sec": manifest["elapsed_sec"],
            "total": dict(total),
            "shards": len(shards),
        },
    )
    print(f"[done] output={args.output_dir} stats={total}", flush=True)


if __name__ == "__main__":
    main()
