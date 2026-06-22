"""BC feature-row flip to a player-0 geometric view.

The source shards were already owner-canonicalized during replay preprocessing:
for a learned player-1 row, owner 1 was relabeled to owner 0, but coordinates
were left in the real map frame. This converter applies the Fight-branch 2p
view transform to those player-1-view rows:

    x/y rotate 180 degrees around board center
    planet candidate slots mirror within each 4-slot symmetry group
    target labels and source-slot metadata follow the same slot permutation

It does not mutate input shards.

The stored active-comet ETA feature is known to be wrong in old shards and is
not recoverable from feature rows alone. The converter leaves that source data
unchanged; BC training drops that feature before normalization/model input.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


FIGHT_PLANET_X = 5
FIGHT_PLANET_Y = 6
FIGHT_REL_DX = 1
FIGHT_REL_DY = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        type=str,
        required=True,
        help="Input bc_feature_rows_*.npz glob.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for converted shards.",
    )
    parser.add_argument("--compression", choices=["compressed", "store"], default="compressed")
    parser.add_argument("--limit-shards", type=int, default=0, help="0 = convert all shards.")
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-approximate-feature-flip",
        action="store_true",
        help="Run while retaining the known-buggy ETA feature. Downstream BC training must drop it.",
    )
    return parser.parse_args()


def mirror_planet_indices(count: int) -> np.ndarray:
    idx = np.arange(int(count), dtype=np.int32)
    mirror = (idx // 4) * 4 + (3 - (idx % 4))
    return np.where(mirror < count, mirror, idx).astype(np.int32, copy=False)


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.shape[0], dtype=perm.dtype)
    return inv


def player1_view_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "meta__learn_player" in arrays:
        return arrays["meta__learn_player"].astype(np.int32) == 1
    required = ("meta__winner_name", "meta__player1_name")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"missing metadata required to identify player-1-view rows: {missing}")
    return arrays["meta__winner_name"].astype(str) == arrays["meta__player1_name"].astype(str)


def transform_player1_rows(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    rel = arrays["feature__fight_relative"]
    max_planets = int(rel.shape[1] - 1)
    planet_perm = mirror_planet_indices(max_planets)
    inv_planet_perm = inverse_permutation(planet_perm)
    candidate_perm = np.concatenate([np.asarray([0], dtype=np.int32), planet_perm + 1])

    p1_rows = player1_view_mask(arrays)
    p1_count = int(p1_rows.sum())
    total_rows = int(p1_rows.shape[0])
    if p1_count <= 0:
        return {"rows": total_rows, "player1_rows": 0, "target_remapped": 0}

    planet = arrays["feature__fight_planet"].copy()
    planet[p1_rows, FIGHT_PLANET_X] = 1.0 - planet[p1_rows, FIGHT_PLANET_X]
    planet[p1_rows, FIGHT_PLANET_Y] = 1.0 - planet[p1_rows, FIGHT_PLANET_Y]
    arrays["feature__fight_planet"] = planet

    rel = rel.copy()
    rel_p1 = rel[p1_rows][:, candidate_perm, :]
    rel_p1[:, :, FIGHT_REL_DX] = -rel_p1[:, :, FIGHT_REL_DX]
    rel_p1[:, :, FIGHT_REL_DY] = -rel_p1[:, :, FIGHT_REL_DY]
    rel[p1_rows] = rel_p1
    arrays["feature__fight_relative"] = rel

    for name in ("feature__candidate_mask", "feature__amount_mask", "feature__candidate_ids"):
        if name in arrays:
            value = arrays[name].copy()
            value[p1_rows] = value[p1_rows][:, candidate_perm, ...]
            arrays[name] = value

    if "feature__map" in arrays:
        fmap = arrays["feature__map"].copy()
        fmap[p1_rows] = fmap[p1_rows, :, ::-1, ::-1]
        arrays["feature__map"] = np.ascontiguousarray(fmap)

    target = arrays["label__target"].copy()
    target_p1 = target[p1_rows]
    fire_target = target_p1 > 0
    old_slot = np.clip(target_p1[fire_target] - 1, 0, max_planets - 1)
    target_p1[fire_target] = inv_planet_perm[old_slot] + 1
    target[p1_rows] = target_p1
    arrays["label__target"] = target

    if "meta__source_slot" in arrays:
        source_slot = arrays["meta__source_slot"].copy()
        source_p1 = source_slot[p1_rows]
        valid_source = (source_p1 >= 0) & (source_p1 < max_planets)
        source_p1[valid_source] = inv_planet_perm[source_p1[valid_source]]
        source_slot[p1_rows] = source_p1
        arrays["meta__source_slot"] = source_slot

    return {
        "rows": total_rows,
        "player1_rows": p1_count,
        "target_remapped": int(fire_target.sum()),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def write_npz(path: Path, arrays: dict[str, np.ndarray], compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer = np.savez_compressed if compression == "compressed" else np.savez
    with tmp.open("wb") as f:
        writer(f, **arrays)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.allow_approximate_feature_flip:
        raise SystemExit(
            "Refusing to run with the known-buggy active-comet ETA feature. "
            "Pass --allow-approximate-feature-flip only when downstream training drops relative dim 7."
        )
    input_paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if int(args.limit_shards) > 0:
        input_paths = input_paths[: int(args.limit_shards)]
    if not input_paths:
        raise FileNotFoundError(f"no shards matched input glob: {args.input_glob}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_glob": args.input_glob,
        "output_dir": str(args.output_dir),
        "compression": args.compression,
        "limit_shards": int(args.limit_shards),
        "shard_count": len(input_paths),
        "transform": {
            "player1_row_mask": "meta__learn_player == 1 when present; otherwise meta__winner_name == meta__player1_name",
            "geometry": "180-degree rotation to player-0 view",
            "planet_slot_permutation": "Fight branch mirror: (idx // 4) * 4 + (3 - idx % 4)",
            "owner_swap": "not applied here; input rows were already owner-canonicalized",
            "buggy_eta_norm": "retained in output; train_bc_jax drops relative dim 7",
        },
        "started_at": time.time(),
    }
    write_json(args.output_dir / "conversion_manifest.json", manifest)

    totals = {
        "shards_done": 0,
        "shards_total": len(input_paths),
        "rows": 0,
        "player1_rows": 0,
        "target_remapped": 0,
        "skipped_existing": 0,
    }
    started = time.monotonic()

    for shard_i, in_path in enumerate(input_paths, start=1):
        out_path = args.output_dir / in_path.name
        if out_path.exists() and args.resume and not args.overwrite:
            totals["skipped_existing"] += 1
            totals["shards_done"] += 1
            continue
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"output shard already exists; pass --resume or --overwrite: {out_path}")

        arrays = load_npz(in_path)
        stats = transform_player1_rows(arrays)
        arrays["meta__player0_view_json"] = np.asarray(
            json.dumps(
                {
                    "source_path": str(in_path),
                    "player1_rows": stats["player1_rows"],
                    "target_remapped": stats["target_remapped"],
                    "buggy_eta_norm": "retained in output; train_bc_jax drops relative dim 7",
                    "transform": "owner-canonicalized player-1-view rows rotated/permuted to player-0 geometry",
                },
                sort_keys=True,
            )
        )
        write_npz(out_path, arrays, str(args.compression))

        totals["shards_done"] += 1
        totals["rows"] += int(stats["rows"])
        totals["player1_rows"] += int(stats["player1_rows"])
        totals["target_remapped"] += int(stats["target_remapped"])

        if shard_i == 1 or shard_i == len(input_paths) or shard_i % max(int(args.report_every), 1) == 0:
            elapsed = time.monotonic() - started
            rate = totals["shards_done"] / max(elapsed, 1e-9)
            eta = (len(input_paths) - totals["shards_done"]) / max(rate, 1e-9)
            progress = {
                **totals,
                "elapsed_sec": elapsed,
                "shards_per_sec": rate,
                "eta_sec": eta,
                "done": totals["shards_done"] >= len(input_paths),
                "last_input": str(in_path),
                "last_output": str(out_path),
            }
            write_json(args.output_dir / "conversion_progress.json", progress)
            print(
                f"[convert] shard={totals['shards_done']}/{len(input_paths)} "
                f"rows={totals['rows']} p1={totals['player1_rows']} "
                f"target_remap={totals['target_remapped']} "
                f"rate={rate:.3f} shard/s eta={eta / 60.0:.1f}m",
                flush=True,
            )

    elapsed = time.monotonic() - started
    done = {**totals, "elapsed_sec": elapsed, "done": True, "finished_at": time.time()}
    write_json(args.output_dir / "conversion_progress.json", done)
    print(
        f"[convert] done shards={totals['shards_done']} rows={totals['rows']} "
        f"player1_rows={totals['player1_rows']} elapsed={elapsed / 60.0:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
