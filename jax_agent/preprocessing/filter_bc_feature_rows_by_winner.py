"""按 winner/fire-rate 过滤并落盘 BC feature-row shards。

用于把训练配置中的筛选逻辑提前物化成一个新数据目录，便于复现实验和减少训练时
反复扫描原始 shards 的成本。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..bc.train_bc_jax import (
    build_allowed_winners,
    compute_winner_fire_rate_stats,
    filter_load_keys,
    load_shard_arrays,
    read_raw_config,
    row_filter_mask,
    shard_paths,
    split_paths,
    winner_fire_rate_filter_config,
)


class _Args:
    winner_fire_rate_min = None
    winner_fire_rate_max = None
    winner_fire_rate_min_rows = None
    winner_fire_rate_min_episodes = None
    exclude_unqualified_winners = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-glob", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--winner-stats-json", type=Path, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--compression", choices=("stored", "compressed"), default="stored")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-every", type=int, default=100)
    return parser.parse_args()


def save_npz(path: Path, arrays: dict[str, np.ndarray], compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compression == "compressed" else np.savez
    writer(path, **arrays)


def load_winner_stats_json(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("stats", payload) if isinstance(payload, dict) else payload
    if isinstance(stats, dict):
        return {
            str(name): {
                "rows": int(item["rows"]),
                "fire": int(item.get("fire", item.get("fires", 0))),
                "episodes": int(item["episodes"]),
                "fire_rate": float(item["fire_rate"]),
            }
            for name, item in stats.items()
        }
    if isinstance(stats, list):
        out: dict[str, dict[str, object]] = {}
        for item in stats:
            name = str(item.get("winner", item.get("name", "")))
            if not name:
                raise ValueError(f"winner stats row is missing winner/name: {item!r}")
            out[name] = {
                "rows": int(item["rows"]),
                "fire": int(item.get("fire", item.get("fires", 0))),
                "episodes": int(item["episodes"]),
                "fire_rate": float(item["fire_rate"]),
            }
        return out
    raise ValueError(f"unsupported winner stats JSON shape: {path}")


def main() -> None:
    args = parse_args()
    raw_cfg = read_raw_config(args.config)
    dataset_pattern = args.dataset_glob or str(raw_cfg.get("bc", {}).get("dataset_glob", ""))
    paths = shard_paths(dataset_pattern, args.max_shards)
    val_fraction = float(args.val_fraction if args.val_fraction is not None else raw_cfg.get("bc", {}).get("val_fraction", 0.0))
    train_paths, _val_paths = split_paths(paths, val_fraction)

    filter_cfg = winner_fire_rate_filter_config(raw_cfg, _Args())
    if filter_cfg is None:
        raise ValueError("bc.winner_fire_rate_filter.enabled must be true in config, or pass a config with filter settings")

    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.glob("bc_feature_rows_*.npz")) and not args.overwrite:
        raise FileExistsError(f"output shards already exist; pass --overwrite to replace: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old in out_dir.glob("bc_feature_rows_*.npz"):
            old.unlink()

    start = time.time()
    if args.winner_stats_json is not None:
        stats = load_winner_stats_json(args.winner_stats_json)
        print(f"filter-materialize: loaded winner stats {args.winner_stats_json} winners={len(stats)}", flush=True)
    else:
        stats = compute_winner_fire_rate_stats(train_paths)
    allowed_winners, summary = build_allowed_winners(stats, filter_cfg)
    (out_dir / "winner_fire_rate_filter_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "allowed_winners.json").write_text(
        json.dumps(sorted(allowed_winners), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    stats_elapsed = time.time() - start
    print(
        "filter-materialize: allowed "
        f"winners={summary['winners_kept']}/{summary['winners_total']} "
        f"rows={summary['rows_kept']}/{summary['rows_total']} "
        f"kept_fraction={summary['kept_fraction']:.4f} "
        f"stats_elapsed={stats_elapsed:.1f}s",
        flush=True,
    )

    # Copy every key in the source shards so the materialized dataset preserves
    # metadata for later analysis/conversion, not only the train-time arrays.
    total_rows = kept_rows = written_shards = empty_shards = 0
    write_start = time.time()
    for i, path in enumerate(paths, start=1):
        with np.load(path) as source:
            keys = tuple(source.files)
        data = load_shard_arrays(path, filter_load_keys(keys, allowed_winners))
        keep = row_filter_mask(data, allowed_winners)
        rows = int(next(iter(data.values())).shape[0])
        kept = int(keep.sum())
        total_rows += rows
        kept_rows += kept
        if kept > 0:
            out = {
                name: (value[keep] if value.ndim > 0 and int(value.shape[0]) == rows else value)
                for name, value in data.items()
            }
            save_npz(out_dir / path.name, out, args.compression)
            written_shards += 1
        else:
            empty_shards += 1
        if i == 1 or i == len(paths) or (args.report_every > 0 and i % args.report_every == 0):
            elapsed = time.time() - write_start
            rate = i / max(elapsed, 1e-9)
            eta = (len(paths) - i) / max(rate, 1e-9)
            print(
                f"filter-materialize: shard={i}/{len(paths)} file={path.name} "
                f"kept={kept}/{rows} kept_total={kept_rows}/{total_rows} "
                f"written={written_shards} empty={empty_shards} "
                f"rate={rate:.2f} shard/s eta={eta / 60.0:.1f}m",
                flush=True,
            )

    total_elapsed = time.time() - start
    result = {
        "source_glob": dataset_pattern,
        "output_dir": str(out_dir),
        "source_shards": len(paths),
        "written_shards": written_shards,
        "empty_shards": empty_shards,
        "rows_total": total_rows,
        "rows_kept": kept_rows,
        "kept_fraction": kept_rows / max(total_rows, 1),
        "stats_elapsed_sec": stats_elapsed,
        "write_elapsed_sec": time.time() - write_start,
        "total_elapsed_sec": total_elapsed,
        "compression": args.compression,
    }
    (out_dir / "materialize_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"filter-materialize: done {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
