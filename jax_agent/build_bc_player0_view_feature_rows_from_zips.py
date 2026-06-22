"""Build BC feature-row shards from raw replay zips in player-0 view.

This is an orchestration wrapper around the two existing preprocessing stages:

1. ``build_bc_feature_rows_from_zips`` builds owner-canonical selected-player
   rows from raw Kaggle replay zips.
2. ``convert_bc_feature_rows_to_player0_view`` mirrors player-1 winner rows into
   the same geometric view as player 0.

The final output directory is directly usable by ``train_bc_jax.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--zip-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Owner-canonical feature-row dir. Default: sibling '<output-dir>__owner_canonical'.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Only run the player0-view conversion from intermediate-dir.")
    parser.add_argument("--delete-intermediate", action="store_true", help="Delete intermediate-dir after successful conversion.")

    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--max-replays-per-zip", type=int, default=0)
    parser.add_argument("--zip-shard-count", type=int, default=1)
    parser.add_argument("--zip-shard-index", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=32768)
    parser.add_argument("--compression", choices=["compressed", "store"], default="compressed")
    parser.add_argument("--feature-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--board-batch-size", type=int, default=64)
    parser.add_argument("--noop-mode", choices=["all", "none", "stride"], default="all")
    parser.add_argument("--noop-stride", type=int, default=0)
    parser.add_argument("--noop-row-keep-ratio", type=float, default=1.0)
    parser.add_argument("--source-filter-mode", choices=["late", "early"], default="late")
    parser.add_argument("--max-action-tokens", type=int, default=16)
    parser.add_argument("--learn-players", choices=["winners", "all"], default="winners")
    parser.add_argument("--learn-player-names", nargs="*", default=None)
    parser.add_argument("--learn-player-names-file", type=Path, default=None)
    parser.add_argument(
        "--leaderboard-stats-json",
        type=Path,
        default=None,
        help="Optional leaderboard_top100_bc_data_stats.json used with --leaderboard-top-n.",
    )
    parser.add_argument("--leaderboard-top-n", type=int, default=0)
    parser.add_argument(
        "--winner-fire-rate-stats-json",
        type=Path,
        default=None,
        help="Optional winner_fire_rate_stats_player0_view.json used to filter selected player names.",
    )
    parser.add_argument("--winner-fire-rate-min", type=float, default=None)
    parser.add_argument("--winner-fire-rate-max", type=float, default=None)
    parser.add_argument("--winner-min-rows", type=int, default=0)
    parser.add_argument("--winner-min-episodes", type=int, default=0)
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

    parser.add_argument("--limit-shards", type=int, default=0, help="0 = convert all built shards.")
    parser.add_argument("--convert-report-every", type=int, default=10)
    parser.add_argument("--resume-convert", action="store_true")
    parser.add_argument("--overwrite-convert", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def add_bool_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def leaderboard_top_names(path: Path, top_n: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("top100") or payload.get("top10") or []
    names = [str(row["team_name"]) for row in rows[: int(top_n)] if row.get("team_name")]
    if len(names) < int(top_n):
        raise ValueError(f"leaderboard file only yielded {len(names)} names for top_n={top_n}: {path}")
    return names


def winner_fire_rate_stats(path: Path) -> dict[str, dict[str, float | int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("stats", payload) if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        return {
            str(name): {
                "rows": int(item["rows"]),
                "episodes": int(item["episodes"]),
                "fire_rate": float(item["fire_rate"]),
            }
            for name, item in rows.items()
        }
    if isinstance(rows, list):
        out: dict[str, dict[str, float | int]] = {}
        for item in rows:
            name = str(item.get("winner", item.get("name", "")))
            if name:
                out[name] = {
                    "rows": int(item["rows"]),
                    "episodes": int(item["episodes"]),
                    "fire_rate": float(item["fire_rate"]),
                }
        return out
    raise ValueError(f"unsupported winner fire-rate stats JSON: {path}")


def filter_names_by_winner_fire_rate(
    names: list[str],
    stats_path: Path,
    *,
    min_rate: float | None,
    max_rate: float | None,
    min_rows: int,
    min_episodes: int,
) -> list[str]:
    stats = winner_fire_rate_stats(stats_path)
    kept: list[str] = []
    dropped: list[dict[str, object]] = []
    for name in names:
        item = stats.get(name)
        keep = item is not None
        if keep and int(item["rows"]) < int(min_rows):
            keep = False
        if keep and int(item["episodes"]) < int(min_episodes):
            keep = False
        if keep and min_rate is not None and float(item["fire_rate"]) < float(min_rate):
            keep = False
        if keep and max_rate is not None and float(item["fire_rate"]) >= float(max_rate):
            keep = False
        if keep:
            kept.append(name)
        else:
            dropped.append({"name": name, "stats": item})
    print(
        "[player0-view-pipeline] fire-rate name filter "
        f"kept={len(kept)}/{len(names)} dropped={len(dropped)} "
        f"min_rate={min_rate} max_rate={max_rate} min_rows={min_rows} min_episodes={min_episodes}",
        flush=True,
    )
    if dropped:
        print(
            "[player0-view-pipeline] dropped names: "
            + ", ".join(str(item["name"]) for item in dropped),
            flush=True,
        )
    if not kept:
        raise ValueError("fire-rate name filter dropped every selected player")
    return kept


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    intermediate_dir = args.intermediate_dir
    if intermediate_dir is None:
        intermediate_dir = output_dir.with_name(output_dir.name + "__owner_canonical")

    learn_player_names = list(args.learn_player_names or [])
    if args.learn_player_names_file is not None:
        with args.learn_player_names_file.open(encoding="utf-8") as f:
            learn_player_names.extend(line.strip() for line in f if line.strip() and not line.lstrip().startswith("#"))
    if int(args.leaderboard_top_n) > 0:
        if args.leaderboard_stats_json is None:
            raise ValueError("--leaderboard-top-n requires --leaderboard-stats-json")
        learn_player_names.extend(leaderboard_top_names(args.leaderboard_stats_json, int(args.leaderboard_top_n)))
    learn_player_names = list(dict.fromkeys(learn_player_names))
    if args.winner_fire_rate_stats_json is not None:
        if not learn_player_names:
            learn_player_names = sorted(winner_fire_rate_stats(args.winner_fire_rate_stats_json))
        learn_player_names = filter_names_by_winner_fire_rate(
            learn_player_names,
            args.winner_fire_rate_stats_json,
            min_rate=args.winner_fire_rate_min,
            max_rate=args.winner_fire_rate_max,
            min_rows=int(args.winner_min_rows),
            min_episodes=int(args.winner_min_episodes),
        )

    started = time.time()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest": str(args.manifest),
        "zip_dir": str(args.zip_dir),
        "intermediate_dir": str(intermediate_dir),
        "output_dir": str(output_dir),
        "learn_players": str(args.learn_players),
        "learn_player_names": learn_player_names,
        "started_at": started,
        "stages": [
            "build_bc_feature_rows_from_zips",
            "convert_bc_feature_rows_to_player0_view",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "player0_view_pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.skip_build:
        build_cmd = [
            sys.executable,
            "-m",
            "jax_agent.build_bc_feature_rows_from_zips",
            "--manifest",
            str(args.manifest),
            "--zip-dir",
            str(args.zip_dir),
            "--output-dir",
            str(intermediate_dir),
            "--max-rows",
            str(args.max_rows),
            "--max-replays",
            str(args.max_replays),
            "--max-replays-per-zip",
            str(args.max_replays_per_zip),
            "--zip-shard-count",
            str(args.zip_shard_count),
            "--zip-shard-index",
            str(args.zip_shard_index),
            "--shard-size",
            str(args.shard_size),
            "--compression",
            str(args.compression),
            "--feature-dtype",
            str(args.feature_dtype),
            "--board-batch-size",
            str(args.board_batch_size),
            "--noop-mode",
            str(args.noop_mode),
            "--noop-stride",
            str(args.noop_stride),
            "--noop-row-keep-ratio",
            str(args.noop_row_keep_ratio),
            "--source-filter-mode",
            str(args.source_filter_mode),
            "--max-action-tokens",
            str(args.max_action_tokens),
            "--learn-players",
            str(args.learn_players),
            "--horizon",
            str(args.horizon),
            "--max-planets",
            str(args.max_planets),
            "--max-fleets",
            str(args.max_fleets),
            "--ship-bucket-count",
            str(args.ship_bucket_count),
            "--map-grid-size",
            str(args.map_grid_size),
            "--max-ships",
            str(args.max_ships),
            "--max-production",
            str(args.max_production),
            "--episode-steps",
            str(args.episode_steps),
            "--attack-top-k",
            str(args.attack_top_k),
            "--defense-top-k",
            str(args.defense_top_k),
            "--report-interval",
            str(args.report_interval),
        ]
        add_bool_flag(build_cmd, "--trajectory-cache", bool(args.trajectory_cache))
        add_bool_flag(build_cmd, "--target-filter-enabled", bool(args.target_filter_enabled))
        if learn_player_names:
            build_cmd.append("--learn-player-names")
            build_cmd.extend(learn_player_names)
        run(build_cmd)

    convert_cmd = [
        sys.executable,
        "-m",
        "jax_agent.convert_bc_feature_rows_to_player0_view",
        "--input-glob",
        str(intermediate_dir / "bc_feature_rows_*.npz"),
        "--output-dir",
        str(output_dir),
        "--compression",
        str(args.compression),
        "--limit-shards",
        str(args.limit_shards),
        "--report-every",
        str(args.convert_report_every),
        "--allow-approximate-feature-flip",
    ]
    add_bool_flag(convert_cmd, "--resume", bool(args.resume_convert))
    add_bool_flag(convert_cmd, "--overwrite", bool(args.overwrite_convert))
    run(convert_cmd)

    if args.delete_intermediate:
        shutil.rmtree(intermediate_dir)

    done = {
        **manifest,
        "finished_at": time.time(),
        "elapsed_sec": time.time() - started,
        "delete_intermediate": bool(args.delete_intermediate),
    }
    (output_dir / "player0_view_pipeline_manifest.json").write_text(
        json.dumps(done, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[player0-view-pipeline] done output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
