"""Build per-planet BC feature-row shards from official daily replay zips.

This is the closer-to-discussion preprocessing path:

    replay board observation -> normalized feature rows -> per-planet labels

The raw replay zips and source manifest are opened read-only. All derived
features, labels, manifests, and audit logs are written under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any, NamedTuple
from zlib import crc32

import jax
import jax.numpy as jnp
import numpy as np

from .build_bc_shards_from_zips import (
    append_jsonl,
    file_snapshot,
    jsonable_args,
    manifest_rows,
    process_replay,
    write_input_snapshot,
)
from .bc_fight_feature_extras import FIGHT_PLANET_DIM, FIGHT_RELATIVE_DIM, build_fight_feature_extras, build_fight_feature_extras_for_sources
from .fast_env import BOARD_SIZE, EnvParams, EnvState
from .features_jax import GLOBAL_DIM, FeatureParams, SHIP_RATIOS, build_global_features, build_map_features

BC_MAP_CHANNELS = 6
SOURCE_SLOT_BUCKETS = (1, 2, 4, 8, 16, 32, 45)


class LightEncoded(NamedTuple):
    source_mask: jnp.ndarray
    source_slots: jnp.ndarray
    source_ids: jnp.ndarray
    source_ships: jnp.ndarray
    candidate_ids: jnp.ndarray
    ship_buckets: jnp.ndarray
    global_features: jnp.ndarray
    map_features: jnp.ndarray


def encode_light_sources(state: EnvState, source_slots: jnp.ndarray, params: FeatureParams) -> LightEncoded:
    planets = state.planets
    source_slots = source_slots.astype(jnp.int32)
    alive = state.planet_alive
    owner = planets[:, 1].astype(jnp.int32)
    source_mask = alive[source_slots] & (owner[source_slots] == 0)
    all_source_ids = planets[:, 0].astype(jnp.int32)
    source_ids = all_source_ids[source_slots]
    source_ships = planets[source_slots, 5]
    candidate_row = jnp.concatenate([jnp.asarray([-1], dtype=jnp.int32), all_source_ids])
    candidate_ids = jnp.broadcast_to(candidate_row[None, :], (source_slots.shape[0], params.max_planets + 1))
    ship_buckets = jnp.zeros((source_slots.shape[0], params.ship_bucket_count), dtype=jnp.float32)
    global_row = build_global_features(state, params)
    global_features = jnp.broadcast_to(global_row[None, :], (source_slots.shape[0], GLOBAL_DIM))
    map_features = jax.vmap(lambda src_idx: build_map_features(state, src_idx, params))(source_slots)
    return LightEncoded(
        source_mask=source_mask,
        source_slots=source_slots,
        source_ids=source_ids,
        source_ships=source_ships,
        candidate_ids=candidate_ids,
        ship_buckets=ship_buckets,
        global_features=global_features,
        map_features=map_features,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/manifests/orbit_2p_manifest.csv"))
    parser.add_argument("--zip-dir", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/daily_zips_cli"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/bc_feature_rows/official_2p_winners_full"))
    parser.add_argument("--max-rows", type=int, default=0, help="0 = no feature-row limit")
    parser.add_argument("--max-replays", type=int, default=0, help="0 = no replay limit")
    parser.add_argument("--max-replays-per-zip", type=int, default=0, help="0 = no per-day replay limit")
    parser.add_argument("--zip-shard-count", type=int, default=1, help="Split daily zips across N independent workers.")
    parser.add_argument("--zip-shard-index", type=int, default=0, help="This worker's 0-based zip shard index.")
    parser.add_argument("--shard-size", type=int, default=32768)
    parser.add_argument("--compression", choices=["compressed", "store"], default="compressed")
    parser.add_argument("--feature-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--board-batch-size", type=int, default=1)
    parser.add_argument("--noop-mode", choices=["all", "none", "stride"], default="all")
    parser.add_argument("--noop-stride", type=int, default=0, help="Only used when --noop-mode=stride.")
    parser.add_argument("--noop-row-keep-ratio", type=float, default=1.0, help="Deterministically keep this fraction of non-fire feature rows.")
    parser.add_argument("--source-filter-mode", choices=["late", "early"], default="late")
    parser.add_argument("--max-action-tokens", type=int, default=16)
    parser.add_argument("--learn-players", choices=["winners", "all"], default="winners")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--trajectory-cache", action="store_true")
    parser.add_argument("--max-planets", type=int, default=45)
    parser.add_argument("--max-fleets", type=int, default=512)
    parser.add_argument("--ship-bucket-count", type=int, default=8)
    parser.add_argument("--map-grid-size", type=int, default=8)
    parser.add_argument("--max-ships", type=float, default=500.0)
    parser.add_argument("--max-production", type=float, default=5.0)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--target-filter-enabled", action="store_true")
    parser.add_argument("--attack-top-k", type=int, default=10)
    parser.add_argument("--defense-top-k", type=int, default=5)
    parser.add_argument("--report-interval", type=float, default=30.0)
    return parser.parse_args()


def state_dict_to_env_state(state: dict[str, np.ndarray]) -> EnvState:
    return EnvState(**{name: jnp.asarray(value) for name, value in state.items()})


def stack_state_dicts(states: list[dict[str, np.ndarray]]) -> EnvState:
    names = states[0].keys()
    return EnvState(**{name: jnp.asarray(np.stack([state[name] for state in states], axis=0)) for name in names})


def pad_chunk(chunk: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]], size: int) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    if len(chunk) >= size:
        return chunk
    if not chunk:
        return chunk
    return chunk + [chunk[-1]] * (size - len(chunk))


def new_buffer() -> dict[str, list[np.ndarray]]:
    return {
        "feature__source_id": [],
        "feature__source_ships": [],
        "feature__candidate_ids": [],
        "feature__ship_buckets": [],
        "feature__candidate_mask": [],
        "feature__amount_mask": [],
        "feature__global": [],
        "feature__map": [],
        "feature__fight_planet": [],
        "feature__fight_relative": [],
        "label__fire": [],
        "label__target": [],
        "label__amount": [],
        "meta__episode_id": [],
        "meta__step": [],
        "meta__source_slot": [],
    }


def per_planet_labels_np(labels: dict[str, np.ndarray], max_planets: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fire = np.zeros((max_planets,), dtype=bool)
    target = np.zeros((max_planets,), dtype=np.int32)
    amount = np.zeros((max_planets,), dtype=np.int32)
    valid = labels["token_valid"] & labels["executed"] & (~labels["end_action"])
    for i in np.flatnonzero(valid):
        src = int(labels["src_action"][i])
        if src < 0 or src >= max_planets or fire[src]:
            continue
        fire[src] = True
        target[src] = int(labels["tgt_action"][i]) + 1
        amount[src] = int(labels["amt_action"][i])
    return fire, target, amount


def bc_ship_buckets_np(source_ships: float, bucket_count: int) -> np.ndarray:
    ships = np.floor(float(source_ships))
    ratios = np.asarray(SHIP_RATIOS[:bucket_count], dtype=np.float32)
    if ships <= 0:
        return np.zeros((bucket_count,), dtype=np.float32)
    return np.clip(np.round(ships * ratios), 1.0, ships).astype(np.float32)


def bc_masks_np(state: EnvState, src: int, ship_buckets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alive = np.asarray(state.planet_alive, dtype=bool)
    max_planets = int(alive.shape[0])
    target_slots = np.arange(max_planets)
    per_planet_mask = alive & (target_slots != int(src))
    candidate_mask = np.concatenate([np.asarray([False]), per_planet_mask]).astype(bool)

    positive_amount = np.asarray(ship_buckets > 0, dtype=bool)
    amount_mask = np.broadcast_to(positive_amount[None, :], (max_planets + 1, positive_amount.shape[0])).copy()
    amount_mask &= candidate_mask[:, None]
    return candidate_mask, amount_mask


def accum_many_np(grid: np.ndarray, channels: np.ndarray, x: np.ndarray, y: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid_size = int(grid.shape[-1])
    in_bounds = (x >= 0.0) & (x <= BOARD_SIZE) & (y >= 0.0) & (y <= BOARD_SIZE) & (values > 0.0)
    ix = np.clip((x / BOARD_SIZE * grid_size).astype(np.int32), 0, grid_size - 1)
    iy = np.clip((y / BOARD_SIZE * grid_size).astype(np.int32), 0, grid_size - 1)
    valid = np.asarray(in_bounds, dtype=bool)
    if valid.any():
        np.add.at(grid, (channels[valid], iy[valid], ix[valid]), values[valid])
    return np.clip(grid, 0.0, 1.0)


def bc_map_np(state: EnvState, src: int, max_ships: float, grid_size: int) -> np.ndarray:
    grid = np.zeros((BC_MAP_CHANNELS, int(grid_size), int(grid_size)), dtype=np.float32)
    planets = np.asarray(state.planets)
    planet_alive = np.asarray(state.planet_alive, dtype=bool)
    owner = planets[:, 1].astype(np.int32)
    idx = np.arange(planets.shape[0], dtype=np.int32)
    channels = np.where(idx == int(src), 0, np.where(owner == 0, 1, np.where(owner == -1, 3, 2))).astype(np.int32)
    grid = accum_many_np(grid, channels, planets[:, 2], planets[:, 3], planet_alive.astype(np.float32))

    fleets = np.asarray(state.fleets)
    fleet_alive = np.asarray(state.fleet_alive, dtype=bool)
    fleet_channels = np.where(fleets[:, 1].astype(np.int32) == 0, 4, 5).astype(np.int32)
    fleet_values = np.clip(fleets[:, 6] / max(float(max_ships), 1.0), 0.0, 1.0) * fleet_alive.astype(np.float32)
    return accum_many_np(grid, fleet_channels, fleets[:, 2], fleets[:, 3], fleet_values)


def keep_noop_row(episode_id: str, step: int, source_slot: int, ratio: float) -> bool:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio >= 1.0:
        return True
    if ratio <= 0.0:
        return False
    key = f"{episode_id}:{int(step)}:{int(source_slot)}".encode("utf-8")
    return (crc32(key) & 0xFFFFFFFF) / 2**32 < ratio


def selected_source_slots_np(
    state: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    episode_id: str,
    step: int,
    max_planets: int,
    noop_row_keep_ratio: float,
    limit: int | None = None,
) -> np.ndarray:
    fire, _target, _amount = per_planet_labels_np(labels, max_planets)
    alive = np.asarray(state["planet_alive"], dtype=bool)
    owner = np.asarray(state["planets"])[:, 1].astype(np.int32)
    source_valid = np.asarray(labels.get("source_valid", np.ones((max_planets,), dtype=bool)), dtype=bool)
    source_mask = alive & (owner == 0) & source_valid
    noop_keep = np.asarray(
        [keep_noop_row(episode_id, step, src, noop_row_keep_ratio) for src in range(max_planets)],
        dtype=bool,
    )
    keep = source_mask & (fire | noop_keep)
    slots = np.flatnonzero(keep).astype(np.int32)
    if limit is not None and int(limit) >= 0:
        slots = slots[: int(limit)]
    return slots


def pad_source_slots_np(source_slots: np.ndarray, max_planets: int) -> tuple[np.ndarray, int]:
    real_count = int(source_slots.shape[0])
    if real_count <= 0:
        return source_slots.astype(np.int32), 0
    bucket = next((size for size in SOURCE_SLOT_BUCKETS if real_count <= size), int(max_planets))
    bucket = min(bucket, int(max_planets))
    padded = np.empty((bucket,), dtype=np.int32)
    padded[:real_count] = source_slots[:real_count]
    padded[real_count:] = int(source_slots[0])
    return padded, real_count


def append_feature_rows(
    buffer: dict[str, list[np.ndarray]],
    state: EnvState,
    encoded: Any,
    fight_extras: Any,
    labels: dict[str, np.ndarray],
    episode_id: str,
    step: int,
    max_planets: int,
    map_grid_size: int,
    max_ships: float,
    row_count: int,
    noop_row_keep_ratio: float,
    limit: int | None = None,
) -> int:
    fire, target, amount = per_planet_labels_np(labels, max_planets)
    source_mask = np.asarray(encoded.source_mask, dtype=bool)
    source_valid = np.asarray(labels.get("source_valid", np.ones((max_planets,), dtype=bool)), dtype=bool)
    source_slots = np.asarray(encoded.source_slots, dtype=np.int32)
    source_ids = np.asarray(encoded.source_ids)
    source_ships = np.asarray(encoded.source_ships)
    candidate_ids = np.asarray(encoded.candidate_ids)
    global_features = np.asarray(encoded.global_features)
    encoded_map = np.asarray(encoded.map_features)
    fight_planet = np.asarray(fight_extras.planet)
    fight_relative = np.asarray(fight_extras.relative)
    bucket_count = int(np.asarray(encoded.ship_buckets).shape[-1])

    rows = 0
    for row_i, src in enumerate(source_slots[: int(row_count)]):
        if limit is not None and rows >= limit:
            break
        src = int(src)
        if not bool(source_mask[row_i]) or not bool(source_valid[src]):
            continue
        if not bool(fire[src]) and not keep_noop_row(episode_id, step, src, noop_row_keep_ratio):
            continue
        ship_buckets = bc_ship_buckets_np(float(source_ships[row_i]), bucket_count)
        candidate_mask, amount_mask = bc_masks_np(state, int(src), ship_buckets)
        buffer["feature__source_id"].append(np.asarray(source_ids[row_i], dtype=np.int32))
        buffer["feature__source_ships"].append(np.asarray(source_ships[row_i], dtype=np.float32))
        buffer["feature__candidate_ids"].append(candidate_ids[row_i])
        buffer["feature__ship_buckets"].append(ship_buckets)
        buffer["feature__candidate_mask"].append(candidate_mask)
        buffer["feature__amount_mask"].append(amount_mask)
        buffer["feature__global"].append(global_features[row_i])
        buffer["feature__map"].append(encoded_map[row_i])
        buffer["feature__fight_planet"].append(fight_planet[row_i])
        buffer["feature__fight_relative"].append(fight_relative[row_i])
        buffer["label__fire"].append(np.asarray(fire[src], dtype=bool))
        buffer["label__target"].append(np.asarray(target[src], dtype=np.int32))
        buffer["label__amount"].append(np.asarray(amount[src], dtype=np.int32))
        buffer["meta__episode_id"].append(np.asarray(int(episode_id), dtype=np.int64))
        buffer["meta__step"].append(np.asarray(step, dtype=np.int32))
        buffer["meta__source_slot"].append(np.asarray(src, dtype=np.int32))
        rows += 1
    return rows


def slice_tree(tree: Any, index: int) -> Any:
    return jax.tree_util.tree_map(lambda x: x[index], tree)


def write_shard(
    buffer: dict[str, list[np.ndarray]],
    output_dir: Path,
    shard_index: int,
    meta: dict[str, Any],
    compression: str,
    feature_dtype: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.stack(values, axis=0) for name, values in buffer.items()}
    if feature_dtype == "float16":
        arrays = {
            name: value.astype(np.float16) if name.startswith("feature__") and np.issubdtype(value.dtype, np.floating) else value
            for name, value in arrays.items()
        }
    arrays["meta__json"] = np.asarray(json.dumps(meta, default=str, sort_keys=True))
    path = output_dir / f"bc_feature_rows_{shard_index:05d}.npz"
    if compression == "compressed":
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return path


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
    elapsed = time.monotonic() - started
    progress = {
        "args": jsonable_args(args),
        "stats": dict(total),
        "shards_written": len(shards),
        "last_shard": shards[-1] if shards else None,
        "elapsed_sec": elapsed,
        "replay_per_sec": total["replays"] / max(elapsed, 1e-9),
        "feature_row_per_sec": total["feature_rows"] / max(elapsed, 1e-9),
        "keep_rate": total["kept_actions"] / max(total["raw_actions"], 1),
        "no_target_rate": total["no_target_actions"] / max(total["raw_actions"], 1),
        "comet_hit_rate": total["comet_hit_actions"] / max(total["raw_actions"], 1),
        "zip_index": zip_i,
        "zip_count": zip_count,
        "zip_path": str(zip_path),
        "done": done,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    env_params = EnvParams(max_planets=int(args.max_planets), max_fleets=int(args.max_fleets), max_actions=int(args.max_planets))
    feature_params = FeatureParams(
        max_planets=int(args.max_planets),
        ship_bucket_count=int(args.ship_bucket_count),
        max_ships=float(args.max_ships),
        max_production=float(args.max_production),
        episode_steps=int(args.episode_steps),
        target_filter_enabled=bool(args.target_filter_enabled),
        attack_top_k=int(args.attack_top_k),
        defense_top_k=int(args.defense_top_k),
    )
    encode_all = jax.jit(
        lambda state: encode_light_sources(
            state,
            jnp.arange(feature_params.max_planets, dtype=jnp.int32),
            feature_params,
        )
    )
    fight_all = jax.jit(lambda state: build_fight_feature_extras(state, feature_params))
    encode_sources = jax.jit(lambda state, source_slots: encode_light_sources(state, source_slots, feature_params))
    fight_sources = jax.jit(lambda state, source_slots: build_fight_feature_extras_for_sources(state, source_slots, feature_params))

    rows = manifest_rows(args.manifest, int(args.max_replays))
    by_zip: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_zip.setdefault(row["zip_path"], []).append(row)
    if int(args.max_replays_per_zip) > 0:
        by_zip = {path: zip_rows[: int(args.max_replays_per_zip)] for path, zip_rows in by_zip.items()}
    zip_shard_count = max(int(args.zip_shard_count), 1)
    zip_shard_index = int(args.zip_shard_index)
    if zip_shard_index < 0 or zip_shard_index >= zip_shard_count:
        raise ValueError(f"zip_shard_index must be in [0, {zip_shard_count}), got {zip_shard_index}")
    if zip_shard_count > 1:
        by_zip = {
            path: zip_rows
            for i, (path, zip_rows) in enumerate(by_zip.items())
            if i % zip_shard_count == zip_shard_index
        }

    write_input_snapshot(args.output_dir, args, rows, by_zip)
    audit_log = args.output_dir / "audit_log.jsonl"
    replay_audit = args.output_dir / "replay_audit.jsonl"
    append_jsonl(audit_log, {"event": "run_start", "time": time.time(), "args": jsonable_args(args), "zip_count": len(by_zip)})

    total = {
        "replays": 0,
        "board_rows": 0,
        "feature_rows": 0,
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
    buffer = new_buffer()
    shards: list[dict[str, Any]] = []
    shard_index = 1
    started = time.monotonic()
    last_report = started
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
                if int(args.max_rows) > 0 and total["feature_rows"] >= int(args.max_rows):
                    stop = True
                    break
                try:
                    replay = json.loads(zf.read(row["json_name"]))
                except Exception as exc:
                    append_jsonl(replay_audit, {"event": "replay_error", "time": time.time(), "row": row, "error": str(exc)})
                    continue

                board_rows, stats = process_replay(replay, env_params, args)
                for key, value in stats.items():
                    if key == "rows":
                        total["board_rows"] += int(value)
                    else:
                        total[key] += int(value)

                appended_feature_rows = 0
                board_batch_size = max(int(args.board_batch_size), 1)
                for batch_start in range(0, len(board_rows), board_batch_size):
                    if int(args.max_rows) > 0 and total["feature_rows"] >= int(args.max_rows):
                        stop = True
                        break
                    chunk = board_rows[batch_start : batch_start + board_batch_size]
                    for state_dict, labels in chunk:
                        if int(args.max_rows) > 0 and total["feature_rows"] >= int(args.max_rows):
                            stop = True
                            break
                        step = int(np.asarray(state_dict["step"]))
                        limit = None
                        if int(args.max_rows) > 0:
                            limit = max(int(args.max_rows) - int(total["feature_rows"]), 0)
                        if limit == 0:
                            stop = True
                            break
                        env_state = state_dict_to_env_state(state_dict)
                        if str(args.source_filter_mode) == "early":
                            source_slots = selected_source_slots_np(
                                state_dict,
                                labels,
                                str(row["episode_id"]),
                                step,
                                int(args.max_planets),
                                float(args.noop_row_keep_ratio),
                                limit=limit,
                            )
                            if source_slots.size == 0:
                                continue
                            padded_source_slots, source_row_count = pad_source_slots_np(source_slots, int(args.max_planets))
                            source_slots_jax = jnp.asarray(padded_source_slots, dtype=jnp.int32)
                            encoded = encode_sources(env_state, source_slots_jax)
                            fight_extras = fight_sources(env_state, source_slots_jax)
                        else:
                            source_row_count = int(args.max_planets)
                            encoded = encode_all(env_state)
                            fight_extras = fight_all(env_state)
                        added = append_feature_rows(
                            buffer,
                            env_state,
                            encoded,
                            fight_extras,
                            labels,
                            str(row["episode_id"]),
                            step,
                            int(args.max_planets),
                            int(args.map_grid_size),
                            float(args.max_ships),
                            int(source_row_count),
                            float(args.noop_row_keep_ratio),
                            limit=limit,
                        )
                        total["feature_rows"] += added
                        appended_feature_rows += added

                        while len(buffer["label__fire"]) >= int(args.shard_size):
                            shard_buffer = {name: values[: int(args.shard_size)] for name, values in buffer.items()}
                            buffer = {name: values[int(args.shard_size) :] for name, values in buffer.items()}
                            meta = {"args": jsonable_args(args), "stats": dict(total), "shard_index": shard_index}
                            path = write_shard(
                                shard_buffer,
                                args.output_dir,
                                shard_index,
                                meta,
                                str(args.compression),
                                str(args.feature_dtype),
                            )
                            shards.append({"path": str(path), "rows": int(args.shard_size)})
                            append_jsonl(audit_log, {"event": "shard_write", "time": time.time(), "path": str(path), "rows": int(args.shard_size), "total": dict(total)})
                            print(f"[shard] wrote {path} rows={int(args.shard_size)} total_feature_rows={total['feature_rows']}", flush=True)
                            write_progress(args.output_dir, args, total, shards, started, zip_i, len(by_zip), zip_path)
                            shard_index += 1
                    if stop:
                        break

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
                        "board_rows": int(stats.get("rows", 0)),
                        "feature_rows": int(appended_feature_rows),
                        "raw_actions": int(stats.get("raw_actions", 0)),
                        "kept_actions": int(stats.get("kept_actions", 0)),
                        "skipped_actions": int(stats.get("skipped_actions", 0)),
                        "no_target_actions": int(stats.get("no_target_actions", 0)),
                        "comet_hit_actions": int(stats.get("comet_hit_actions", 0)),
                        "bad_amount_actions": int(stats.get("bad_amount_actions", 0)),
                        "total_feature_rows_after": int(total["feature_rows"]),
                    },
                )

                now = time.monotonic()
                if now - last_report >= float(args.report_interval):
                    keep_rate = total["kept_actions"] / max(total["raw_actions"], 1)
                    no_target_rate = total["no_target_actions"] / max(total["raw_actions"], 1)
                    comet_hit_rate = total["comet_hit_actions"] / max(total["raw_actions"], 1)
                    elapsed = now - started
                    print(
                        f"[progress] zips={zip_i}/{len(by_zip)} replays={total['replays']} "
                        f"feature_rows={total['feature_rows']} raw_actions={total['raw_actions']} "
                        f"keep_rate={keep_rate:.3f} no_target={no_target_rate:.3f} "
                        f"comet_hit={comet_hit_rate:.3f} "
                        f"rate={total['feature_rows'] / max(elapsed, 1e-9):.1f} rows/s",
                        flush=True,
                    )
                    append_jsonl(audit_log, {"event": "progress", "time": time.time(), "zip_index": zip_i, "zip_path": zip_path, "total": dict(total)})
                    write_progress(args.output_dir, args, total, shards, started, zip_i, len(by_zip), zip_path)
                    last_report = now

        append_jsonl(audit_log, {"event": "zip_done", "time": time.time(), "zip_index": zip_i, "zip_path": zip_path, "total": dict(total)})
        write_progress(args.output_dir, args, total, shards, started, zip_i, len(by_zip), zip_path)
        print(f"[zip] {Path(zip_path).name} cumulative_replays={total['replays']} feature_rows={total['feature_rows']}", flush=True)

    if buffer["label__fire"]:
        meta = {"args": jsonable_args(args), "stats": dict(total), "shard_index": shard_index}
        path = write_shard(buffer, args.output_dir, shard_index, meta, str(args.compression), str(args.feature_dtype))
        shards.append({"path": str(path), "rows": len(buffer["label__fire"])})
        append_jsonl(audit_log, {"event": "shard_write", "time": time.time(), "path": str(path), "rows": len(buffer["label__fire"]), "total": dict(total)})
        print(f"[shard] wrote {path} rows={len(buffer['label__fire'])} total_feature_rows={total['feature_rows']}", flush=True)

    manifest = {
        "args": jsonable_args(args),
        "stats": total,
        "feature_schema": {
            "feature__fight_planet": [FIGHT_PLANET_DIM],
            "feature__fight_relative": [int(args.max_planets) + 1, FIGHT_RELATIVE_DIM],
            "feature__candidate_mask": [int(args.max_planets) + 1],
            "feature__amount_mask": [int(args.max_planets) + 1, int(args.ship_bucket_count)],
            "feature__global": [GLOBAL_DIM],
            "feature__map": [BC_MAP_CHANNELS, int(args.map_grid_size), int(args.map_grid_size)],
            "labels": {"fire": "bool", "target": "0=no-fire placeholder else target_slot+1", "amount": "ship bucket index"},
        },
        "shards": shards,
        "elapsed_sec": time.monotonic() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_progress(args.output_dir, args, total, shards, started, len(by_zip), len(by_zip), "done", done=True)
    append_jsonl(audit_log, {"event": "run_done", "time": time.time(), "manifest": str(args.output_dir / "manifest.json"), "total": dict(total), "shards": len(shards)})
    print(f"[done] output={args.output_dir} stats={total}", flush=True)


if __name__ == "__main__":
    main()
