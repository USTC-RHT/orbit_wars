"""Feature-row behavioral cloning trainer.

This trains directly on shards produced by ``build_bc_feature_rows_from_zips``:

    feature rows -> normalizer stats from train split -> fire/target/amount heads

The online RL encoder is intentionally not used here. The feature-row schema is
the BC contract.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
    _WANDB_IMPORT_ERROR = None
except ImportError as exc:
    _WANDB_AVAILABLE = False
    _WANDB_IMPORT_ERROR = exc


NEG_INF = -1.0e9
TRAIN_ARRAY_KEYS = (
    "feature__fight_planet",
    "feature__fight_relative",
    "feature__global",
    "feature__map",
    "feature__candidate_mask",
    "feature__amount_mask",
    "label__fire",
    "label__target",
    "label__amount",
)
FILTER_META_KEYS = (
    "meta__winner_name",
    "meta__learn_player_name",
    "meta__episode_id",
)
NORMALIZER_ARRAY_KEYS = (
    "feature__fight_planet",
    "feature__fight_relative",
    "feature__global",
    "feature__map",
    "feature__candidate_mask",
)
DEFAULT_DROPPED_RELATIVE_FEATURE_DIMS = (7,)  # Old active-comet ETA feature; drop from BC inputs.
DROPPED_RELATIVE_FEATURE_DIMS = DEFAULT_DROPPED_RELATIVE_FEATURE_DIMS


class FeatureRowBatch(NamedTuple):
    planet: jnp.ndarray
    relative: jnp.ndarray
    global_features: jnp.ndarray
    map_features: jnp.ndarray
    candidate_mask: jnp.ndarray
    amount_mask: jnp.ndarray
    fire: jnp.ndarray
    target: jnp.ndarray
    amount: jnp.ndarray


class BCOutput(NamedTuple):
    fire_logits: jnp.ndarray
    target_logits: jnp.ndarray
    amount_logits: jnp.ndarray


def read_raw_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def raw_get(raw: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = raw.get(section, {}).get(key, default)
    return default if value is None else value


def parse_int_list(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return (int(value),)


def parse_learn_result_filter(value: Any) -> str:
    mode = str(value or "all").strip().lower()
    aliases = {
        "": "all",
        "both": "all",
        "win": "winner",
        "wins": "winner",
        "won": "winner",
        "loss": "loser",
        "lose": "loser",
        "lost": "loser",
        "losses": "loser",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"all", "winner", "loser"}:
        raise ValueError(f"bc.learn_result_filter must be one of all/winner/loser, got {value!r}")
    return mode


def setup_wandb(raw_cfg: dict[str, Any], *, run_name: str, save_dir: Path) -> bool:
    wandb_cfg = raw_cfg.get("wandb", {}) if isinstance(raw_cfg.get("wandb", {}), dict) else {}
    enabled = bool(wandb_cfg.get("enabled", False))
    mode = str(wandb_cfg.get("mode", "online"))
    if not enabled or mode == "disabled":
        return False
    if not _WANDB_AVAILABLE:
        print(f"wandb: unavailable; continuing without W&B: {_WANDB_IMPORT_ERROR!r}", flush=True)
        return False

    tags = wandb_cfg.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        run = _wandb.init(
            project=str(wandb_cfg.get("project", "orbit-wars-bc")),
            entity=wandb_cfg.get("entity", None) or None,
            name=str(wandb_cfg.get("name", run_name) or run_name),
            config=raw_cfg,
            mode=mode,
            group=wandb_cfg.get("group", None) or None,
            tags=list(tags),
            dir=str(save_dir),
        )
        print(f"wandb: enabled project={wandb_cfg.get('project', 'orbit-wars-bc')} mode={mode}", flush=True)
        if getattr(run, "url", None):
            print(f"wandb: run_url={run.url}", flush=True)
        return True
    except Exception as exc:
        print(f"wandb: init failed; continuing without W&B: {exc!r}", flush=True)
        return False


def log_wandb(use_wandb: bool, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    if not use_wandb:
        return
    try:
        _wandb.log(dict(metrics), step=step)
    except Exception as exc:
        print(f"wandb: log failed; continuing without interrupting training: {exc!r}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_jax_BC_1.yaml")
    parser.add_argument("--dataset-glob", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--shuffle-buffer-rows", type=int, default=None)
    parser.add_argument("--prefetch-batches", type=int, default=None)
    parser.add_argument("--no-shuffle-shards", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-rows-per-shard", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--normalizer-path", type=Path, default=None)
    parser.add_argument("--normalizer-sample-shards", type=int, default=None)
    parser.add_argument("--normalizer-sample-seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a BC checkpoint path.")
    parser.add_argument("--winner-fire-rate-min", type=float, default=None)
    parser.add_argument("--winner-fire-rate-max", type=float, default=None)
    parser.add_argument("--winner-fire-rate-min-rows", type=int, default=None)
    parser.add_argument("--winner-fire-rate-min-episodes", type=int, default=None)
    parser.add_argument(
        "--exclude-unqualified-winners",
        action="store_true",
        help="Drop winners below the fire-rate stats sample floor instead of keeping them.",
    )
    return parser.parse_args()


def shard_paths(pattern: str, max_shards: int | None = None) -> list[Path]:
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if max_shards is not None and max_shards > 0:
        paths = paths[: int(max_shards)]
    if not paths:
        raise FileNotFoundError(f"no BC feature-row shards matched: {pattern}")
    return paths


def split_paths(paths: list[Path], val_fraction: float) -> tuple[list[Path], list[Path]]:
    if len(paths) > 1 and val_fraction > 0.0:
        val_count = max(1, int(round(len(paths) * val_fraction)))
        val_count = min(val_count, len(paths) - 1)
    else:
        val_count = 0
    train_paths = paths[:-val_count] if val_count else paths
    val_paths = paths[-val_count:] if val_count else paths[-1:]
    return train_paths, val_paths


def sample_normalizer_paths(paths: list[Path], sample_shards: int, seed: int) -> list[Path]:
    if int(sample_shards) <= 0 or int(sample_shards) >= len(paths):
        return paths
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(paths), size=int(sample_shards), replace=False))
    return [paths[int(i)] for i in indices]


def load_shard_arrays(path: Path, keys: tuple[str, ...] = TRAIN_ARRAY_KEYS) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in keys}


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def unique_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(keys))


def filter_load_keys(
    base_keys: tuple[str, ...],
    allowed_winners: frozenset[str] | None,
    learn_result_filter: str = "all",
) -> tuple[str, ...]:
    if allowed_winners is None and learn_result_filter == "all":
        return base_keys
    return unique_keys(base_keys + FILTER_META_KEYS)


def winner_fire_rate_filter_config(raw_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    cfg = raw_get(raw_cfg, "bc", "winner_fire_rate_filter", {})
    cfg = dict(cfg) if isinstance(cfg, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    if args.winner_fire_rate_min is not None or args.winner_fire_rate_max is not None:
        enabled = True
    if not enabled:
        return None
    min_rate = args.winner_fire_rate_min if args.winner_fire_rate_min is not None else parse_optional_float(cfg.get("min_rate"))
    max_rate = args.winner_fire_rate_max if args.winner_fire_rate_max is not None else parse_optional_float(cfg.get("max_rate"))
    if min_rate is None and max_rate is None:
        raise ValueError("winner fire-rate filter is enabled, but neither min_rate nor max_rate is set")
    min_rows = args.winner_fire_rate_min_rows
    if min_rows is None:
        min_rows = parse_optional_int(cfg.get("min_rows", 0))
    min_episodes = args.winner_fire_rate_min_episodes
    if min_episodes is None:
        min_episodes = parse_optional_int(cfg.get("min_episodes", 0))
    include_unqualified = bool(cfg.get("include_unqualified", True)) and not bool(args.exclude_unqualified_winners)
    allow_winners_raw = cfg.get("allow_winners", cfg.get("allowed_winners", None))
    allow_winners = None
    if allow_winners_raw not in (None, ""):
        if isinstance(allow_winners_raw, str):
            allow_winners = [allow_winners_raw]
        else:
            allow_winners = [str(name) for name in allow_winners_raw]
    return {
        "min_rate": min_rate,
        "max_rate": max_rate,
        "min_rows": int(min_rows or 0),
        "min_episodes": int(min_episodes or 0),
        "include_unqualified": include_unqualified,
        "allow_winners": allow_winners,
    }


def compute_winner_fire_rate_stats(
    paths: list[Path],
    *,
    report_every: int = 500,
    use_wandb: bool = False,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, int] = {}
    fires: dict[str, int] = {}
    episodes: dict[str, set[int]] = {}
    start = time.time()
    total = len(paths)
    print(f"winner-fire-rate: start stats shards={total}", flush=True)
    for i, path in enumerate(paths, start=1):
        data = load_shard_arrays(path, ("label__fire", "meta__winner_name", "meta__episode_id"))
        winners = data["meta__winner_name"].astype(str)
        fire = data["label__fire"].astype(np.int64)
        episode_id = data["meta__episode_id"]
        values, inverse = np.unique(winners, return_inverse=True)
        row_counts = np.bincount(inverse, minlength=len(values))
        fire_counts = np.bincount(inverse, weights=fire, minlength=len(values)).astype(np.int64)
        for j, value in enumerate(values):
            name = str(value)
            rows[name] = rows.get(name, 0) + int(row_counts[j])
            fires[name] = fires.get(name, 0) + int(fire_counts[j])
            eps = episodes.setdefault(name, set())
            eps.update(map(int, np.unique(episode_id[winners == value])))
        if i == 1 or i == total or (int(report_every) > 0 and i % int(report_every) == 0):
            elapsed = time.time() - start
            rate = i / max(elapsed, 1e-9)
            eta = (total - i) / max(rate, 1e-9)
            print(
                f"winner-fire-rate: shard={i}/{total} file={path.name} "
                f"rate={rate:.2f} shard/s eta={eta / 60.0:.1f}m",
                flush=True,
            )
            log_wandb(
                use_wandb,
                {
                    "winner_filter/stats_shards_done": i,
                    "winner_filter/stats_progress": i / max(total, 1),
                    "winner_filter/stats_eta_minutes": eta / 60.0,
                },
                step=0,
            )
    return {
        name: {
            "rows": int(rows[name]),
            "fire": int(fires.get(name, 0)),
            "episodes": len(episodes.get(name, set())),
            "fire_rate": float(fires.get(name, 0)) / max(int(rows[name]), 1),
        }
        for name in rows
    }


def build_allowed_winners(
    stats: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[frozenset[str], dict[str, Any]]:
    min_rate = cfg["min_rate"]
    max_rate = cfg["max_rate"]
    min_rows = int(cfg["min_rows"])
    min_episodes = int(cfg["min_episodes"])
    include_unqualified = bool(cfg["include_unqualified"])
    allow_winners = cfg.get("allow_winners")
    allow_winners_set = set(map(str, allow_winners)) if allow_winners else None
    allowed: set[str] = set()
    kept_rows = kept_fire = kept_episodes = 0
    dropped_rows = dropped_fire = dropped_episodes = 0
    qualified_count = 0
    unqualified_count = 0
    for name, item in stats.items():
        rows = int(item["rows"])
        fire = int(item["fire"])
        episodes = int(item["episodes"])
        rate = float(item["fire_rate"])
        qualified = rows >= min_rows and episodes >= min_episodes
        if qualified:
            qualified_count += 1
            keep = True
            if min_rate is not None and rate < float(min_rate):
                keep = False
            if max_rate is not None and rate >= float(max_rate):
                keep = False
        else:
            unqualified_count += 1
            keep = include_unqualified
        if keep and allow_winners_set is not None and name not in allow_winners_set:
            keep = False
        if keep:
            allowed.add(name)
            kept_rows += rows
            kept_fire += fire
            kept_episodes += episodes
        else:
            dropped_rows += rows
            dropped_fire += fire
            dropped_episodes += episodes
    total_rows = kept_rows + dropped_rows
    total_fire = kept_fire + dropped_fire
    summary = {
        "enabled": True,
        "min_rate": min_rate,
        "max_rate": max_rate,
        "min_rows": min_rows,
        "min_episodes": min_episodes,
        "include_unqualified": include_unqualified,
        "allow_winners": sorted(allow_winners_set) if allow_winners_set is not None else None,
        "allow_winners_missing": sorted(allow_winners_set - set(stats)) if allow_winners_set is not None else [],
        "winners_total": len(stats),
        "winners_qualified": qualified_count,
        "winners_unqualified": unqualified_count,
        "winners_kept": len(allowed),
        "winners_dropped": len(stats) - len(allowed),
        "rows_total": total_rows,
        "rows_kept": kept_rows,
        "rows_dropped": dropped_rows,
        "fire_total": total_fire,
        "fire_kept": kept_fire,
        "fire_dropped": dropped_fire,
        "episodes_kept_summed": kept_episodes,
        "episodes_dropped_summed": dropped_episodes,
        "kept_fraction": kept_rows / max(total_rows, 1),
        "kept_fire_rate": kept_fire / max(kept_rows, 1),
        "dropped_fire_rate": dropped_fire / max(dropped_rows, 1),
    }
    if not allowed:
        raise ValueError(f"winner fire-rate filter dropped every winner: {summary}")
    return frozenset(allowed), summary


def row_filter_mask(
    data: Mapping[str, np.ndarray],
    allowed_winners: frozenset[str] | None,
    learn_result_filter: str = "all",
) -> np.ndarray:
    first = next(iter(data.values()))
    rows = int(first.shape[0])
    keep = np.ones((rows,), dtype=bool)
    if allowed_winners is not None:
        if "meta__winner_name" not in data:
            raise KeyError("winner fire-rate filtering requires meta__winner_name in BC shards")
        keep &= np.isin(data["meta__winner_name"].astype(str), list(allowed_winners))
    if learn_result_filter != "all":
        missing = [name for name in ("meta__learn_player_name", "meta__winner_name") if name not in data]
        if missing:
            raise KeyError(f"bc.learn_result_filter={learn_result_filter!r} requires metadata keys: {missing}")
        is_learn_winner = data["meta__learn_player_name"].astype(str) == data["meta__winner_name"].astype(str)
        if learn_result_filter == "winner":
            keep &= is_learn_winner
        elif learn_result_filter == "loser":
            keep &= ~is_learn_winner
        else:
            raise ValueError(f"unsupported learn_result_filter: {learn_result_filter!r}")
    return keep


def zero_disabled_relative_np(relative: np.ndarray, *, copy: bool = True) -> np.ndarray:
    if not DROPPED_RELATIVE_FEATURE_DIMS:
        return relative
    value = np.array(relative, copy=copy)
    value[..., DROPPED_RELATIVE_FEATURE_DIMS] = 0.0
    return value


def zero_disabled_relative_jax(relative: jnp.ndarray) -> jnp.ndarray:
    relative = jnp.asarray(relative)
    if not DROPPED_RELATIVE_FEATURE_DIMS:
        return relative
    return relative.at[..., DROPPED_RELATIVE_FEATURE_DIMS].set(0.0)


class MLP(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        return nn.relu(x)


class FeatureRowBCPolicy(nn.Module):
    ship_bucket_count: int
    hidden_size: int = 128
    use_cnn: bool = True
    target_head: str = "dot"
    target_num_heads: int = 8

    @nn.compact
    def __call__(
        self,
        planet: jnp.ndarray,
        relative: jnp.ndarray,
        global_features: jnp.ndarray,
        candidate_mask: jnp.ndarray,
        map_features: jnp.ndarray,
        selected_targets: jnp.ndarray | None = None,
    ) -> BCOutput:
        planet_h = MLP(self.hidden_size)(planet)
        relative_h = MLP(self.hidden_size)(relative)
        global_h = MLP(self.hidden_size)(global_features)

        valid = candidate_mask.astype(jnp.float32)
        has_valid = candidate_mask.any(axis=1, keepdims=True)
        masked = jnp.where(candidate_mask[..., None], relative_h, jnp.finfo(jnp.float32).min)
        max_pool = jnp.where(has_valid, masked.max(axis=1), 0.0)
        mean_pool = (relative_h * valid[..., None]).sum(axis=1) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)

        context_parts = [planet_h, global_h, 0.5 * (max_pool + mean_pool)]
        if self.use_cnn:
            x = jnp.transpose(map_features, (0, 2, 3, 1))
            x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
            x = x.mean(axis=(1, 2))
            context_parts.append(nn.relu(nn.Dense(16)(x)))

        context_h = jnp.concatenate(context_parts, axis=-1)

        target_head = str(self.target_head or "dot").lower().replace("-", "_")
        if target_head in {"cross_attention", "attention", "mh_attention", "multihead_attention"}:
            heads = max(int(self.target_num_heads), 1)
            head_dim = max((int(self.hidden_size) + heads - 1) // heads, 1)
            attn_size = heads * head_dim
            q = nn.relu(nn.Dense(self.hidden_size, name="target_query_mlp_0")(context_h))
            q = nn.Dense(attn_size, name="target_query")(q).reshape((context_h.shape[0], heads, head_dim))
            k = nn.Dense(attn_size, name="target_key")(relative_h).reshape((relative_h.shape[0], relative_h.shape[1], heads, head_dim))
            v = nn.Dense(attn_size, name="target_value")(relative_h).reshape((relative_h.shape[0], relative_h.shape[1], heads, head_dim))
            attn_logits = jnp.einsum("bhd,bchd->bhc", q, k) / jnp.sqrt(float(head_dim))
            masked_attn_logits = jnp.where(candidate_mask[:, None, :], attn_logits, NEG_INF)
            attn_weights = jax.nn.softmax(masked_attn_logits, axis=-1)
            attended = jnp.einsum("bhc,bchd->bhd", attn_weights, v).reshape((context_h.shape[0], attn_size))
            attended = nn.relu(nn.Dense(self.hidden_size, name="target_attended_proj")(attended))
            head_logits = jnp.swapaxes(masked_attn_logits, 1, 2)
            head_mix = nn.Dense(1, use_bias=False, name="target_head_mix")(head_logits).squeeze(-1)
            pair_context = nn.relu(nn.Dense(self.hidden_size, name="target_pair_context")(jnp.concatenate([context_h, attended], axis=-1)))
            pair_context = jnp.broadcast_to(pair_context[:, None, :], relative_h.shape)
            pair = jnp.concatenate([relative_h, pair_context, relative_h * pair_context], axis=-1)
            pair_logits = nn.Dense(1, name="target_pair_score")(
                nn.relu(nn.Dense(self.hidden_size, name="target_pair_mlp")(pair))
            ).squeeze(-1)
            target_logits = head_mix + pair_logits
        else:
            q = nn.Dense(self.hidden_size)(context_h)
            q = nn.relu(q)
            q = nn.Dense(self.hidden_size)(q)
            q = q / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q), axis=-1, keepdims=True), 1e-12))
            keys = nn.Dense(self.hidden_size)(relative_h)
            keys = keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(keys), axis=-1, keepdims=True), 1e-12))
            target_logits = jnp.einsum("bh,bch->bc", q, keys) / jnp.sqrt(float(self.hidden_size))
        target_logits = jnp.where(candidate_mask, target_logits, NEG_INF)

        fire_h = nn.relu(nn.Dense(self.hidden_size)(context_h))
        fire_logits = nn.Dense(1)(fire_h).squeeze(-1)

        if selected_targets is None:
            selected_targets = jnp.zeros((planet.shape[0],), dtype=jnp.int32)
        selected_targets = jnp.clip(selected_targets.astype(jnp.int32), 0, relative.shape[1] - 1)
        row = jnp.arange(selected_targets.shape[0])
        selected_h = relative_h[row, selected_targets]
        amount_h = jnp.concatenate([context_h, selected_h], axis=-1)
        amount_h = nn.relu(nn.Dense(self.hidden_size)(amount_h))
        amount_logits = nn.Dense(self.ship_bucket_count)(amount_h)
        return BCOutput(fire_logits, target_logits, amount_logits)


def safe_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    masked = jnp.where(mask, logits, NEG_INF)
    empty = ~mask.any(axis=-1, keepdims=True)
    return jnp.where(empty, 0.0, masked)


def masked_mean(value: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    weight = mask.astype(jnp.float32)
    return (value * weight).sum() / jnp.maximum(weight.sum(), 1.0)


def categorical_nll(logits: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), label[..., None], axis=-1).squeeze(-1)


def topk_contains(logits: jnp.ndarray, label: jnp.ndarray, k: int) -> jnp.ndarray:
    order = jnp.argsort(logits, axis=-1)[:, -int(k) :]
    return (order == label[:, None]).any(axis=-1)


def zeros_stats() -> dict[str, np.ndarray]:
    return {
        "planet_sum": np.zeros((28,), dtype=np.float64),
        "planet_sumsq": np.zeros((28,), dtype=np.float64),
        "planet_count": np.asarray(0.0, dtype=np.float64),
        "relative_sum": np.zeros((22,), dtype=np.float64),
        "relative_sumsq": np.zeros((22,), dtype=np.float64),
        "relative_count": np.asarray(0.0, dtype=np.float64),
        "global_sum": np.zeros((11,), dtype=np.float64),
        "global_sumsq": np.zeros((11,), dtype=np.float64),
        "global_count": np.asarray(0.0, dtype=np.float64),
        "map_sum": np.zeros((6,), dtype=np.float64),
        "map_sumsq": np.zeros((6,), dtype=np.float64),
        "map_count": np.asarray(0.0, dtype=np.float64),
    }


def accumulate_stats(stats: dict[str, np.ndarray], data: Mapping[str, np.ndarray]) -> None:
    planet = data["feature__fight_planet"].astype(np.float64)
    stats["planet_sum"] += planet.sum(axis=0)
    stats["planet_sumsq"] += np.square(planet).sum(axis=0)
    stats["planet_count"] += planet.shape[0]

    relative = zero_disabled_relative_np(data["feature__fight_relative"].astype(np.float64), copy=False)
    rel_mask = data["feature__candidate_mask"].astype(bool)
    valid_rel = relative[rel_mask]
    if valid_rel.size:
        stats["relative_sum"] += valid_rel.sum(axis=0)
        stats["relative_sumsq"] += np.square(valid_rel).sum(axis=0)
        stats["relative_count"] += valid_rel.shape[0]

    global_f = data["feature__global"].astype(np.float64)
    stats["global_sum"] += global_f.sum(axis=0)
    stats["global_sumsq"] += np.square(global_f).sum(axis=0)
    stats["global_count"] += global_f.shape[0]

    maps = data["feature__map"].astype(np.float64)
    stats["map_sum"] += maps.sum(axis=(0, 2, 3))
    stats["map_sumsq"] += np.square(maps).sum(axis=(0, 2, 3))
    stats["map_count"] += maps.shape[0] * maps.shape[2] * maps.shape[3]


def finalize_normalizer(stats: dict[str, np.ndarray], eps: float = 1e-6) -> dict[str, np.ndarray]:
    def finish(prefix: str):
        count = max(float(stats[f"{prefix}_count"]), 1.0)
        mean = stats[f"{prefix}_sum"] / count
        var = np.maximum(stats[f"{prefix}_sumsq"] / count - np.square(mean), eps * eps)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

    planet_mean, planet_std = finish("planet")
    relative_mean, relative_std = finish("relative")
    global_mean, global_std = finish("global")
    map_mean, map_std = finish("map")
    return {
        "planet_mean": planet_mean,
        "planet_std": planet_std,
        "relative_mean": relative_mean,
        "relative_std": relative_std,
        "global_mean": global_mean,
        "global_std": global_std,
        "map_mean": map_mean.reshape((6, 1, 1)),
        "map_std": map_std.reshape((6, 1, 1)),
        "counts": np.asarray(
            [stats["planet_count"], stats["relative_count"], stats["global_count"], stats["map_count"]],
            dtype=np.float64,
        ),
    }


def compute_normalizer(
    paths: list[Path],
    *,
    report_every: int = 100,
    use_wandb: bool = False,
    allowed_winners: frozenset[str] | None = None,
    learn_result_filter: str = "all",
) -> dict[str, np.ndarray]:
    stats = zeros_stats()
    start = time.time()
    total = len(paths)
    rows_kept = 0
    print(f"normalizer: start shards={total}", flush=True)
    log_wandb(use_wandb, {"normalizer/total_shards": total, "normalizer/shards_done": 0}, step=0)
    for i, path in enumerate(paths, start=1):
        keys = filter_load_keys(NORMALIZER_ARRAY_KEYS, allowed_winners, learn_result_filter)
        data = load_shard_arrays(path, keys)
        keep = row_filter_mask(data, allowed_winners, learn_result_filter)
        if keep.any():
            rows_kept += int(keep.sum())
            accumulate_stats(stats, {name: data[name][keep] for name in NORMALIZER_ARRAY_KEYS})
        if i == 1 or i == total or (int(report_every) > 0 and i % int(report_every) == 0):
            elapsed = time.time() - start
            rate = i / max(elapsed, 1e-9)
            eta = (total - i) / max(rate, 1e-9)
            progress = i / max(total, 1)
            print(
                f"normalizer: shard={i}/{total} file={path.name} "
                f"rate={rate:.2f} shard/s eta={eta / 60.0:.1f}m",
                flush=True,
            )
            log_wandb(
                use_wandb,
                {
                    "normalizer/shards_done": i,
                    "normalizer/progress": progress,
                    "normalizer/rate_shards_per_sec": rate,
                    "normalizer/eta_minutes": eta / 60.0,
                    "normalizer/elapsed_minutes": elapsed / 60.0,
                    "normalizer/rows_kept": rows_kept,
                },
                step=0,
            )
    if rows_kept <= 0:
        raise ValueError("normalizer saw zero rows after winner fire-rate filtering")
    return finalize_normalizer(stats)


def identity_normalizer() -> dict[str, np.ndarray]:
    return {
        "planet_mean": np.zeros((28,), dtype=np.float32),
        "planet_std": np.ones((28,), dtype=np.float32),
        "relative_mean": np.zeros((22,), dtype=np.float32),
        "relative_std": np.ones((22,), dtype=np.float32),
        "global_mean": np.zeros((11,), dtype=np.float32),
        "global_std": np.ones((11,), dtype=np.float32),
        "map_mean": np.zeros((6, 1, 1), dtype=np.float32),
        "map_std": np.ones((6, 1, 1), dtype=np.float32),
        "counts": np.zeros((4,), dtype=np.float64),
    }


def load_normalizer_npz(path: Path) -> dict[str, np.ndarray]:
    required = tuple(identity_normalizer().keys())
    with np.load(path) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise KeyError(f"normalizer npz missing keys {missing}: {path}")
        return {name: np.asarray(data[name]) for name in required}


def jax_normalizer(normalizer: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {name: jnp.asarray(value) for name, value in normalizer.items()}


def normalize_batch(batch: FeatureRowBatch, norm: dict[str, jnp.ndarray]) -> FeatureRowBatch:
    planet = (batch.planet - norm["planet_mean"]) / jnp.maximum(norm["planet_std"], 1e-6)
    relative_raw = zero_disabled_relative_jax(batch.relative)
    relative = (relative_raw - norm["relative_mean"]) / jnp.maximum(norm["relative_std"], 1e-6)
    relative = jnp.where(batch.candidate_mask[..., None], relative, 0.0)
    global_features = (batch.global_features - norm["global_mean"]) / jnp.maximum(norm["global_std"], 1e-6)
    map_features = (batch.map_features - norm["map_mean"]) / jnp.maximum(norm["map_std"], 1e-6)
    return batch._replace(planet=planet, relative=relative, global_features=global_features, map_features=map_features)


def batch_from_arrays(data: Mapping[str, np.ndarray], idx: np.ndarray) -> FeatureRowBatch:
    return FeatureRowBatch(
        planet=np.asarray(data["feature__fight_planet"][idx], dtype=np.float32),
        relative=zero_disabled_relative_np(np.asarray(data["feature__fight_relative"][idx], dtype=np.float32), copy=False),
        global_features=np.asarray(data["feature__global"][idx], dtype=np.float32),
        map_features=np.asarray(data["feature__map"][idx], dtype=np.float32),
        candidate_mask=np.asarray(data["feature__candidate_mask"][idx], dtype=bool),
        amount_mask=np.asarray(data["feature__amount_mask"][idx], dtype=bool),
        fire=np.asarray(data["label__fire"][idx], dtype=bool),
        target=np.asarray(data["label__target"][idx], dtype=np.int32),
        amount=np.asarray(data["label__amount"][idx], dtype=np.int32),
    )


def allocate_shuffle_buffer(sample: Mapping[str, np.ndarray], capacity: int) -> dict[str, np.ndarray]:
    return {
        name: np.empty((int(capacity),) + sample[name].shape[1:], dtype=sample[name].dtype)
        for name in TRAIN_ARRAY_KEYS
    }


def append_to_shuffle_buffer(
    buffer: dict[str, np.ndarray],
    count: int,
    data: Mapping[str, np.ndarray],
    rows: np.ndarray,
) -> int:
    if len(rows) == 0:
        return int(count)
    dst = slice(int(count), int(count) + len(rows))
    for name in TRAIN_ARRAY_KEYS:
        buffer[name][dst] = data[name][rows]
    return int(count) + len(rows)


def batch_from_shuffle_buffer(buffer: Mapping[str, np.ndarray], idx: np.ndarray) -> FeatureRowBatch:
    return batch_from_arrays(buffer, idx)


def remove_from_shuffle_buffer(buffer: dict[str, np.ndarray], count: int, idx: np.ndarray) -> int:
    new_count = int(count)
    for pos in sorted((int(i) for i in idx), reverse=True):
        last = new_count - 1
        if pos != last:
            for name in TRAIN_ARRAY_KEYS:
                buffer[name][pos] = buffer[name][last]
        new_count -= 1
    return new_count


def bc_loss_and_metrics(model, params, batch: FeatureRowBatch, weights: dict[str, float]):
    zero_target = jnp.zeros_like(batch.target)
    base_out = model.apply(
        {"params": params},
        batch.planet,
        batch.relative,
        batch.global_features,
        batch.candidate_mask,
        batch.map_features,
        zero_target,
    )
    fire_f = batch.fire.astype(jnp.float32)
    fire_nll = optax.sigmoid_binary_cross_entropy(base_out.fire_logits, fire_f)
    fire_weight = jnp.where(batch.fire, 1.0, float(weights["noop_weight"]))
    fire_loss = (fire_nll * fire_weight).sum() / jnp.maximum(fire_weight.sum(), 1.0)

    target_valid = jnp.take_along_axis(batch.candidate_mask, batch.target[:, None], axis=1).squeeze(1)
    target_mask = batch.fire & target_valid
    target_logits = safe_logits(base_out.target_logits, batch.candidate_mask)
    target_loss = masked_mean(categorical_nll(target_logits, batch.target), target_mask)

    action_out = model.apply(
        {"params": params},
        batch.planet,
        batch.relative,
        batch.global_features,
        batch.candidate_mask,
        batch.map_features,
        batch.target,
    )
    amount_valid = jnp.take_along_axis(batch.amount_mask, batch.target[:, None, None], axis=1).squeeze(1)
    chosen_amount_valid = jnp.take_along_axis(amount_valid, batch.amount[:, None], axis=1).squeeze(1)
    amount_mask = target_mask & chosen_amount_valid
    amount_logits = safe_logits(action_out.amount_logits, amount_valid)
    amount_loss = masked_mean(categorical_nll(amount_logits, batch.amount), amount_mask)

    loss = (
        float(weights["fire_weight"]) * fire_loss
        + float(weights["target_weight"]) * target_loss
        + float(weights["amount_weight"]) * amount_loss
    )

    fire_pred = base_out.fire_logits > 0.0
    target_pred = jnp.argmax(target_logits, axis=-1)
    amount_pred = jnp.argmax(amount_logits, axis=-1)
    fire_tp = (fire_pred & batch.fire).astype(jnp.float32).sum()
    fire_fp = (fire_pred & (~batch.fire)).astype(jnp.float32).sum()
    fire_fn = ((~fire_pred) & batch.fire).astype(jnp.float32).sum()
    amount_err = jnp.abs(amount_pred.astype(jnp.float32) - batch.amount.astype(jnp.float32))
    metrics = {
        "loss": loss,
        "fire_loss": fire_loss,
        "target_loss": target_loss,
        "amount_loss": amount_loss,
        "fire_acc": (fire_pred == batch.fire).astype(jnp.float32).mean(),
        "fire_precision": fire_tp / jnp.maximum(fire_tp + fire_fp, 1.0),
        "fire_recall": fire_tp / jnp.maximum(fire_tp + fire_fn, 1.0),
        "target_acc": masked_mean((target_pred == batch.target).astype(jnp.float32), target_mask),
        "target_top3": masked_mean(topk_contains(target_logits, batch.target, 3).astype(jnp.float32), target_mask),
        "amount_acc": masked_mean((amount_pred == batch.amount).astype(jnp.float32), amount_mask),
        "amount_mae": masked_mean(amount_err, amount_mask),
        "launch_rate": batch.fire.astype(jnp.float32).mean(),
        "pred_launch_rate": fire_pred.astype(jnp.float32).mean(),
        "target_kept": masked_mean(target_valid.astype(jnp.float32), batch.fire),
        "amount_kept": masked_mean(chosen_amount_valid.astype(jnp.float32), target_mask),
        "rows": jnp.asarray(batch.fire.shape[0], dtype=jnp.float32),
    }
    return loss, metrics


def make_train_state(rng, model, sample_batch: FeatureRowBatch, lr: float, weight_decay: float) -> TrainState:
    variables = model.init(
        rng,
        sample_batch.planet,
        sample_batch.relative,
        sample_batch.global_features,
        sample_batch.candidate_mask,
        sample_batch.map_features,
        jnp.zeros((sample_batch.fire.shape[0],), dtype=jnp.int32),
    )
    tx = optax.adamw(lr, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def make_train_step(model, weights, normalizer: dict[str, jnp.ndarray]):
    @jax.jit(donate_argnums=(0,))
    def train_step(train_state, batch: FeatureRowBatch):
        batch = normalize_batch(batch, normalizer)

        def loss_fn(params):
            return bc_loss_and_metrics(model, params, batch, weights)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        del loss
        return train_state.apply_gradients(grads=grads), metrics

    return train_step


def make_eval_step(model, weights, normalizer: dict[str, jnp.ndarray]):
    @jax.jit
    def eval_step(train_state, batch: FeatureRowBatch):
        batch = normalize_batch(batch, normalizer)
        _, metrics = bc_loss_and_metrics(model, train_state.params, batch, weights)
        return metrics

    return eval_step


def epoch_indices(
    data: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    train: bool,
    noop_keep_ratio: float,
    max_rows_per_shard: int | None,
    allowed_winners: frozenset[str] | None = None,
    learn_result_filter: str = "all",
    shuffle: bool | None = None,
) -> np.ndarray:
    if shuffle is None:
        shuffle = train
    rows = int(data["label__fire"].shape[0])
    keep = row_filter_mask(data, allowed_winners, learn_result_filter)
    if train and noop_keep_ratio < 1.0:
        fire = data["label__fire"].astype(bool)
        keep_noop = rng.random(rows) < float(noop_keep_ratio)
        keep = keep & (fire | keep_noop)
        if not keep.any():
            keep = row_filter_mask(data, allowed_winners, learn_result_filter)
    idx = np.flatnonzero(keep)
    if shuffle:
        rng.shuffle(idx)
    if max_rows_per_shard is not None and int(max_rows_per_shard) > 0:
        idx = idx[: int(max_rows_per_shard)]
    return idx.astype(np.int32, copy=False)


def add_metric_sums(sums: dict[str, float], metrics: Mapping[str, Any], weight: int) -> None:
    for key, value in metrics.items():
        sums[key] = sums.get(key, 0.0) + float(value) * float(weight)


def average_metric_sums(sums: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(int(count), 1) for key, value in sums.items()}


def add_metric_sums_device(sums: dict[str, jnp.ndarray], metrics: Mapping[str, Any], weight: int) -> None:
    weight_arr = jnp.asarray(float(weight), dtype=jnp.float32)
    for key, value in metrics.items():
        weighted = jnp.asarray(value) * weight_arr
        sums[key] = sums.get(key, jnp.asarray(0.0, dtype=weighted.dtype)) + weighted


def average_metric_sums_device(sums: dict[str, jnp.ndarray], count: int) -> dict[str, jnp.ndarray]:
    denom = jnp.maximum(jnp.asarray(float(count), dtype=jnp.float32), 1.0)
    return {key: value / denom for key, value in sums.items()}


def train_batch_stream(
    paths: list[Path],
    batch_size: int,
    rng: np.random.Generator,
    noop_keep_ratio: float,
    max_rows_per_shard: int | None,
    allowed_winners: frozenset[str] | None = None,
    learn_result_filter: str = "all",
    initial_completed_passes: int = 0,
    shuffle_shards: bool = True,
    shuffle_buffer_rows: int = 0,
):
    completed_passes = int(initial_completed_passes)
    buffer_capacity = int(shuffle_buffer_rows)
    if buffer_capacity < int(batch_size):
        raise ValueError("shuffle_buffer_rows must be >= batch_size")
    buffer: dict[str, np.ndarray] | None = None
    buffer_count = 0

    pass_paths: list[Path] = []
    path_cursor = 0
    current_data: dict[str, np.ndarray] | None = None
    current_order = np.zeros((0,), dtype=np.int32)
    current_cursor = 0
    current_path = paths[0]

    def reset_pass():
        nonlocal pass_paths, path_cursor
        pass_paths = list(paths)
        if shuffle_shards:
            shard_order = rng.permutation(len(paths))
            pass_paths = [paths[int(i)] for i in shard_order]
        path_cursor = 0

    def load_next_kept_shard():
        nonlocal completed_passes, path_cursor
        attempts = 0
        while attempts < max(len(paths) * 2, 1):
            if path_cursor >= len(pass_paths):
                completed_passes += 1
                reset_pass()
            path = pass_paths[path_cursor]
            path_cursor += 1
            attempts += 1
            data = load_shard_arrays(path, filter_load_keys(TRAIN_ARRAY_KEYS, allowed_winners, learn_result_filter))
            order = epoch_indices(
                data,
                rng,
                train=True,
                noop_keep_ratio=noop_keep_ratio,
                max_rows_per_shard=max_rows_per_shard,
                allowed_winners=allowed_winners,
                learn_result_filter=learn_result_filter,
                shuffle=True,
            )
            if len(order) > 0:
                return path, data, order
        raise ValueError("no train rows were produced; check BC shard contents and noop_keep_ratio")

    def refill(target_count: int):
        nonlocal buffer, buffer_count, current_data, current_order, current_cursor, current_path
        while buffer_count < int(target_count):
            if current_data is None or current_cursor >= len(current_order):
                current_path, current_data, current_order = load_next_kept_shard()
                current_cursor = 0
                if buffer is None:
                    buffer = allocate_shuffle_buffer(current_data, buffer_capacity)
            free = buffer_capacity - buffer_count
            if free <= 0:
                break
            available = len(current_order) - current_cursor
            take = min(free, available)
            rows = current_order[current_cursor : current_cursor + take]
            buffer_count = append_to_shuffle_buffer(buffer, buffer_count, current_data, rows)
            current_cursor += take

    reset_pass()
    refill(buffer_capacity)
    if buffer is None or buffer_count < batch_size:
        raise ValueError("shuffle buffer could not be initialized with enough train rows")

    while True:
        batch_idx = rng.choice(buffer_count, size=int(batch_size), replace=False).astype(np.int32)
        batch = batch_from_shuffle_buffer(buffer, batch_idx)
        buffer_count = remove_from_shuffle_buffer(buffer, buffer_count, batch_idx)
        refill(buffer_capacity)
        yield batch, completed_passes, current_path


def prefetch_batch_stream(stream, prefetch_batches: int):
    depth = int(prefetch_batches)
    if depth <= 0:
        yield from stream
        return

    sentinel = object()
    q: queue.Queue[Any] = queue.Queue(maxsize=depth)

    def worker() -> None:
        try:
            for batch, completed_passes, path in stream:
                q.put((jax.device_put(batch), completed_passes, path))
        except BaseException as exc:
            q.put(exc)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=worker, name="bc-prefetch", daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def run_eval_batches(
    paths: list[Path],
    train_state,
    eval_step,
    batch_size: int,
    rng,
    eval_batches: int,
    max_rows_per_shard: int | None,
    allowed_winners: frozenset[str] | None = None,
    learn_result_filter: str = "all",
):
    sums: dict[str, float] = {}
    count = 0
    batches = 0
    path_index = 0
    empty_cycles = 0
    while batches < int(eval_batches):
        path = paths[path_index % len(paths)]
        path_index += 1
        data = load_shard_arrays(path, filter_load_keys(TRAIN_ARRAY_KEYS, allowed_winners, learn_result_filter))
        order = epoch_indices(
            data,
            rng,
            train=False,
            noop_keep_ratio=1.0,
            max_rows_per_shard=max_rows_per_shard,
            allowed_winners=allowed_winners,
            learn_result_filter=learn_result_filter,
            shuffle=True,
        )
        if len(order) < batch_size:
            empty_cycles += 1
            if empty_cycles > len(paths):
                raise ValueError("no eval batches were produced; check batch_size and BC validation shards")
            continue
        empty_cycles = 0
        for start in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            metrics = jax.device_get(eval_step(train_state, batch_from_arrays(data, idx)))
            batch_rows = int(len(idx))
            count += batch_rows
            batches += 1
            add_metric_sums(sums, metrics, batch_rows)
            if batches >= int(eval_batches):
                break
    return average_metric_sums(sums, count)


def sample_batch_for_init(
    paths: list[Path],
    normalizer: dict[str, jnp.ndarray],
    batch_size: int,
    allowed_winners: frozenset[str] | None,
    learn_result_filter: str = "all",
) -> FeatureRowBatch:
    for path in paths:
        data = load_shard_arrays(path, filter_load_keys(TRAIN_ARRAY_KEYS, allowed_winners, learn_result_filter))
        idx = epoch_indices(
            data,
            np.random.default_rng(0),
            train=False,
            noop_keep_ratio=1.0,
            max_rows_per_shard=None,
            allowed_winners=allowed_winners,
            learn_result_filter=learn_result_filter,
            shuffle=False,
        )
        if len(idx) > 0:
            sample_idx = idx[: min(int(batch_size), len(idx))]
            return normalize_batch(batch_from_arrays(data, sample_idx), normalizer)
    raise ValueError("no sample rows available after BC row filtering")


def load_checkpoint_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"BC checkpoint payload must be a dict: {path}")
    return payload


def restore_train_state(train_state, payload: dict[str, Any]):
    params = payload.get("params")
    if params is None:
        raise KeyError("resume checkpoint is missing 'params'")
    train_state = train_state.replace(params=jax.tree_util.tree_map(jnp.asarray, params))
    if "opt_state" in payload:
        train_state = train_state.replace(opt_state=jax.tree_util.tree_map(jnp.asarray, payload["opt_state"]))
    if "step" in payload:
        train_state = train_state.replace(step=jnp.asarray(payload["step"]))
    return train_state


def metrics_best_val(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return float("inf")
    best = float("inf")
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_loss" in row:
                best = min(best, float(row["val_loss"]))
    return best


def save_checkpoint(
    path: Path,
    train_state,
    normalizer: dict[str, np.ndarray],
    raw_cfg,
    metrics: dict[str, float],
    *,
    completed_passes: int,
    rng,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": jax.device_get(train_state.params),
        "opt_state": jax.device_get(train_state.opt_state),
        "step": int(jax.device_get(train_state.step)),
        "normalizer": normalizer,
        "raw_config": raw_cfg,
        "metrics": metrics,
        "completed_passes": int(completed_passes),
        "rng": jax.device_get(rng),
        "best_val": float(best_val),
        "checkpoint_type": "feature_row_bc",
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    np.savez(path.parent / "normalizer.npz", **normalizer)


def write_metrics(path: Path, step: int, metrics: dict[str, float], *, completed_passes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"step": int(step), "completed_passes": int(completed_passes), "time": time.time(), **metrics}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def resolve_online_opponent_dir(root: Path, name: str) -> Path:
    direct = Path(name)
    if (direct / "main.py").exists():
        return direct
    aliases = {
        "nearest": "nearest-sniper",
        "nearest-sniper": "nearest-sniper",
        "producer": "slawekbiel-producer",
        "slawekbiel": "slawekbiel-producer",
    }
    candidates = [name]
    alias = aliases.get(str(name))
    if alias is not None:
        candidates.append(alias)
    for candidate_name in candidates:
        for subdir in ("external", "mine", "baselines"):
            candidate = root / subdir / candidate_name
            if (candidate / "main.py").exists():
                return candidate
    raise FileNotFoundError(f"could not resolve online eval opponent: {name}")


def make_online_eval_agent(model, train_state, normalizer: dict[str, jnp.ndarray], raw_cfg: dict[str, Any], *, max_fleets: int):
    # Imported lazily to avoid paying Kaggle/runtime setup cost when online eval is disabled.
    from .bc_policy_runtime import BCPolicyAgent, build_env_params_from_raw, env_config_from_raw
    from .features_jax import FeatureParams

    env_params = build_env_params_from_raw(raw_cfg, max_fleets=int(max_fleets))
    env_cfg = env_config_from_raw(raw_cfg)
    feature_params = FeatureParams.from_train_config(type("Cfg", (), {"env": env_cfg})())
    ship_bucket_count = int(raw_get(raw_cfg, "env", "ship_bucket_count", 8))
    return BCPolicyAgent(
        model=model,
        params=train_state.params,
        normalizer=normalizer,
        env_params=env_params,
        feature_params=feature_params,
        ship_bucket_count=ship_bucket_count,
    )


def run_online_eval(
    *,
    model,
    train_state,
    normalizer: dict[str, jnp.ndarray],
    raw_cfg: dict[str, Any],
    opponent_name: str,
    agents_root: Path,
    games: int,
    seed: int,
    bc_player: int,
    max_fleets: int,
) -> dict[str, float]:
    from .eval_opponents.opponents import ExternalScriptOpponent

    try:
        from kaggle_environments import make as kaggle_make
    except ImportError as exc:
        raise RuntimeError("kaggle_environments is required for online BC eval") from exc

    agent = make_online_eval_agent(model, train_state, normalizer, raw_cfg, max_fleets=int(max_fleets))
    opponent_dir = resolve_online_opponent_dir(Path(agents_root), str(opponent_name))
    opponent = ExternalScriptOpponent(opponent_dir)
    bc_idx = int(bc_player)
    opp_idx = 1 - bc_idx
    wins = losses = draws = 0
    reward_sum = 0.0
    turns_sum = 0.0
    start = time.time()
    print(
        f"[online-eval] start opponent={opponent_dir} games={int(games)} bc_player={bc_idx} seed={int(seed)}",
        flush=True,
    )

    for game_idx in range(int(games)):
        game_seed = int(seed) + game_idx
        opponent.reset_env(game_idx, game_seed)
        env = kaggle_make("orbit_wars", configuration={"seed": game_seed, "randomSeed": game_seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        while states[0]["status"] == "ACTIVE":
            bc_action = agent.act(states[bc_idx]["observation"], player=bc_idx)
            opp_action = opponent.act_for_env(game_idx, states[opp_idx]["observation"])
            joint = [None, None]
            joint[bc_idx] = bc_action
            joint[opp_idx] = opp_action
            states = env.step(joint)

        reward = float(states[bc_idx].get("reward", 0.0) or 0.0)
        reward_sum += reward
        turns = float(len(env.steps) if hasattr(env, "steps") else 0)
        turns_sum += turns
        if reward > 0:
            wins += 1
            outcome = "win"
        elif reward < 0:
            losses += 1
            outcome = "loss"
        else:
            draws += 1
            outcome = "draw"
        print(
            f"[online-eval] game={game_idx + 1}/{int(games)} seed={game_seed} "
            f"outcome={outcome} reward={reward:.1f} turns={turns:.0f}",
            flush=True,
        )

    total = max(int(games), 1)
    elapsed = time.time() - start
    metrics = {"online_win_rate": wins / total}
    print(
        f"[online-eval] done wins={wins} losses={losses} draws={draws} "
        f"win_rate={metrics['online_win_rate']:.3f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return metrics


def online_eval_opponent_names(raw_cfg: dict[str, Any]) -> list[str]:
    value = raw_get(raw_cfg, "eval", "opponents", None)
    if value is None:
        return [str(raw_get(raw_cfg, "eval", "opponent", "slawekbiel-producer"))]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def online_eval_metric_prefix(name: str) -> str:
    aliases = {
        "producer": "producer",
        "slawekbiel": "producer",
        "slawekbiel-producer": "producer",
        "fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid": "ckpt48900",
        "ckpt_48900": "ckpt48900",
        "ckpt48900": "ckpt48900",
    }
    label = aliases.get(str(name), str(name))
    out = []
    for ch in label.lower():
        out.append(ch if ch.isalnum() else "_")
    return "_".join("".join(out).strip("_").split("_"))


def prefix_online_eval_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    win_rate = metrics.get("online_win_rate")
    return {} if win_rate is None else {f"online_{prefix}_win_rate": float(win_rate)}


def main() -> None:
    global DROPPED_RELATIVE_FEATURE_DIMS
    args = parse_args()
    config_path = Path(args.config).expanduser()
    raw_cfg = read_raw_config(args.config)
    seed = int(raw_cfg.get("seed", 42))
    bc_cfg = raw_cfg.setdefault("bc", {})
    if isinstance(bc_cfg, dict):
        DROPPED_RELATIVE_FEATURE_DIMS = parse_int_list(
            bc_cfg.get("dropped_relative_feature_dims", DEFAULT_DROPPED_RELATIVE_FEATURE_DIMS)
        )
        bc_cfg["dropped_relative_feature_dims"] = list(DROPPED_RELATIVE_FEATURE_DIMS)
        if args.dataset_glob is not None:
            bc_cfg["dataset_glob"] = str(args.dataset_glob)
        if (
            args.winner_fire_rate_min is not None
            or args.winner_fire_rate_max is not None
            or args.winner_fire_rate_min_rows is not None
            or args.winner_fire_rate_min_episodes is not None
            or args.exclude_unqualified_winners
        ):
            filter_cfg = dict(bc_cfg.get("winner_fire_rate_filter", {}) or {})
            filter_cfg["enabled"] = True
            if args.winner_fire_rate_min is not None:
                filter_cfg["min_rate"] = float(args.winner_fire_rate_min)
            if args.winner_fire_rate_max is not None:
                filter_cfg["max_rate"] = float(args.winner_fire_rate_max)
            if args.winner_fire_rate_min_rows is not None:
                filter_cfg["min_rows"] = int(args.winner_fire_rate_min_rows)
            if args.winner_fire_rate_min_episodes is not None:
                filter_cfg["min_episodes"] = int(args.winner_fire_rate_min_episodes)
            if args.exclude_unqualified_winners:
                filter_cfg["include_unqualified"] = False
            bc_cfg["winner_fire_rate_filter"] = filter_cfg
    resume_payload = load_checkpoint_payload(args.resume)
    dataset_pattern = args.dataset_glob or str(raw_get(raw_cfg, "bc", "dataset_glob", ""))
    paths = shard_paths(dataset_pattern, args.max_shards)
    train_paths, val_paths = split_paths(paths, float(raw_get(raw_cfg, "bc", "val_fraction", 0.1)))
    batch_size = int(args.batch_size or raw_get(raw_cfg, "bc", "batch_size", 512))
    lr = float(args.lr or raw_get(raw_cfg, "bc", "lr", 3e-4))
    max_steps = int(args.max_steps or raw_get(raw_cfg, "bc", "max_steps", 0))
    if max_steps <= 0:
        raise ValueError("step-based BC training requires bc.max_steps or --max-steps > 0")
    log_every_steps = int(args.log_every_steps or raw_get(raw_cfg, "bc", "log_every_steps", 100))
    eval_every_steps = int(args.eval_every_steps or raw_get(raw_cfg, "bc", "eval_every_steps", 1000))
    eval_batches = int(args.eval_batches or raw_get(raw_cfg, "bc", "eval_batches", 64))
    checkpoint_every_steps = int(args.checkpoint_every_steps or raw_get(raw_cfg, "bc", "checkpoint_every_steps", 5000))
    shuffle_shards = bool(raw_get(raw_cfg, "bc", "shuffle_shards", True)) and not bool(args.no_shuffle_shards)
    prefetch_batches = int(args.prefetch_batches if args.prefetch_batches is not None else raw_get(raw_cfg, "bc", "prefetch_batches", 0))
    shuffle_buffer_rows_raw = args.shuffle_buffer_rows
    if shuffle_buffer_rows_raw is None:
        shuffle_buffer_rows_raw = raw_get(raw_cfg, "bc", "shuffle_buffer_rows", None)
    if shuffle_buffer_rows_raw is None:
        raise ValueError("step-based BC training requires bc.shuffle_buffer_rows or --shuffle-buffer-rows")
    shuffle_buffer_rows = int(shuffle_buffer_rows_raw)
    if eval_batches <= 0:
        raise ValueError("step-based BC training requires bc.eval_batches or --eval-batches > 0")
    if shuffle_buffer_rows < batch_size:
        raise ValueError("bc.shuffle_buffer_rows or --shuffle-buffer-rows must be >= batch_size")
    if prefetch_batches < 0:
        raise ValueError("bc.prefetch_batches or --prefetch-batches must be >= 0")
    hidden_size = int(raw_get(raw_cfg, "model", "hidden_size", 128))
    use_cnn = bool(raw_get(raw_cfg, "model", "use_cnn", True))
    target_head = str(raw_get(raw_cfg, "model", "target_head", "dot"))
    target_num_heads = int(raw_get(raw_cfg, "model", "target_num_heads", 8))
    weight_decay = float(raw_get(raw_cfg, "bc", "weight_decay", 0.0))
    noop_keep_ratio = float(raw_get(raw_cfg, "bc", "noop_keep_ratio", 0.25))
    learn_result_filter = parse_learn_result_filter(raw_get(raw_cfg, "bc", "learn_result_filter", "all"))
    bc_cfg["learn_result_filter"] = learn_result_filter
    weights = {
        "noop_weight": float(raw_get(raw_cfg, "bc", "noop_weight", 1.0)),
        "fire_weight": float(raw_get(raw_cfg, "bc", "fire_weight", 1.0)),
        "target_weight": float(raw_get(raw_cfg, "bc", "target_weight", 1.0)),
        "amount_weight": float(raw_get(raw_cfg, "bc", "amount_weight", 0.5)),
    }
    online_eval_enabled = bool(raw_get(raw_cfg, "eval", "enabled", False))
    online_eval_every_steps = int(raw_get(raw_cfg, "eval", "every_steps", 0))
    online_eval_games = int(raw_get(raw_cfg, "eval", "games", 2))
    online_eval_seed = int(raw_get(raw_cfg, "eval", "seed", 300000))
    online_eval_opponents = online_eval_opponent_names(raw_cfg)
    online_eval_agents_root = Path(str(raw_get(raw_cfg, "eval", "agents_root", "agents")))
    online_eval_bc_player = int(raw_get(raw_cfg, "eval", "bc_player", 0))
    online_eval_max_fleets = int(raw_get(raw_cfg, "eval", "max_fleets", 512))
    online_eval_fail_on_error = bool(raw_get(raw_cfg, "eval", "fail_on_error", False))
    if online_eval_enabled and online_eval_every_steps <= 0:
        raise ValueError("online eval is enabled, so eval.every_steps must be > 0")
    if online_eval_bc_player not in (0, 1):
        raise ValueError(f"eval.bc_player must be 0 or 1, got {online_eval_bc_player}")

    save_root = Path(str(args.save_dir or raw_cfg.get("save_dir", "/mnt/kaggle/bc_runs")))
    run_name = str(args.run_name or raw_cfg.get("run_name", "bc_feature_rows"))
    save_dir = save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    use_wandb = setup_wandb(raw_cfg, run_name=run_name, save_dir=save_dir)

    tic = time.time()
    filter_cfg = winner_fire_rate_filter_config(raw_cfg, args)
    allowed_winners: frozenset[str] | None = None
    if filter_cfg is not None:
        winner_stats = compute_winner_fire_rate_stats(train_paths, use_wandb=use_wandb)
        allowed_winners, filter_summary = build_allowed_winners(winner_stats, filter_cfg)
        bc_cfg["winner_fire_rate_filter_summary"] = filter_summary
        (save_dir / "winner_fire_rate_filter_summary.json").write_text(
            json.dumps(filter_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "winner-fire-rate: "
            f"min={filter_summary['min_rate']} max={filter_summary['max_rate']} "
            f"min_rows={filter_summary['min_rows']} min_episodes={filter_summary['min_episodes']} "
            f"include_unqualified={filter_summary['include_unqualified']} "
            f"winners={filter_summary['winners_kept']}/{filter_summary['winners_total']} "
            f"rows={filter_summary['rows_kept']}/{filter_summary['rows_total']} "
            f"fire={filter_summary['fire_kept']}/{filter_summary['fire_total']} "
            f"kept_fire_rate={filter_summary['kept_fire_rate']:.4f}",
            flush=True,
        )
        log_wandb(use_wandb, {f"winner_filter/{k}": v for k, v in filter_summary.items()}, step=0)
    normalize_enabled = bool(raw_get(raw_cfg, "state_norm", "enabled", True)) and not args.no_normalize
    normalizer_sample_shards = args.normalizer_sample_shards
    if normalizer_sample_shards is None:
        normalizer_sample_shards = parse_optional_int(raw_get(raw_cfg, "state_norm", "sample_shards", 0))
    normalizer_sample_seed = args.normalizer_sample_seed
    if normalizer_sample_seed is None:
        normalizer_sample_seed = parse_optional_int(raw_get(raw_cfg, "state_norm", "sample_seed", seed))
    normalizer_path = args.normalizer_path
    if normalizer_path is None:
        normalizer_path_raw = raw_get(raw_cfg, "state_norm", "normalizer_path", None)
        if normalizer_path_raw in (None, ""):
            normalizer_path_raw = raw_get(raw_cfg, "state_norm", "path", None)
        if normalizer_path_raw not in (None, ""):
            normalizer_path = Path(str(normalizer_path_raw)).expanduser()
    if normalizer_path is not None and not normalizer_path.is_absolute():
        normalizer_path = (config_path.parent / normalizer_path).resolve()
    if resume_payload is not None and "normalizer" in resume_payload and normalize_enabled:
        normalizer_np = resume_payload["normalizer"]
        print(f"resume: loaded normalizer from {args.resume}", flush=True)
    elif normalize_enabled and normalizer_path is not None:
        if not normalizer_path.is_file():
            raise FileNotFoundError(f"configured normalizer_path does not exist: {normalizer_path}")
        normalizer_np = load_normalizer_npz(normalizer_path)
        print(f"normalizer: loaded from {normalizer_path}", flush=True)
        log_wandb(use_wandb, {"normalizer/loaded_from_path": 1}, step=0)
    else:
        normalizer_paths = sample_normalizer_paths(train_paths, int(normalizer_sample_shards or 0), int(normalizer_sample_seed or seed))
        if normalize_enabled and len(normalizer_paths) != len(train_paths):
            print(
                f"normalizer: sampled shards={len(normalizer_paths)}/{len(train_paths)} "
                f"seed={int(normalizer_sample_seed or seed)}",
                flush=True,
            )
        normalizer_np = (
            identity_normalizer()
            if not normalize_enabled
            else compute_normalizer(
                normalizer_paths,
                use_wandb=use_wandb,
                allowed_winners=allowed_winners,
                learn_result_filter=learn_result_filter,
            )
        )
        if not normalize_enabled:
            print("normalizer: disabled; using identity normalizer", flush=True)
            log_wandb(use_wandb, {"normalizer/disabled": 1}, step=0)
    normalizer = jax_normalizer(normalizer_np)
    print(
        f"BC feature-row start shards={len(paths)} train={len(train_paths)} val={len(val_paths)} "
        f"max_steps={max_steps} batch={batch_size} eval_batches={eval_batches} "
        f"shuffle_shards={shuffle_shards} shuffle_buffer_rows={shuffle_buffer_rows} "
        f"prefetch_batches={prefetch_batches} "
        f"learn_result_filter={learn_result_filter} "
        f"dropped_relative_dims={list(DROPPED_RELATIVE_FEATURE_DIMS)} "
        f"norm_counts={normalizer_np['counts'].tolist()}",
        flush=True,
    )
    if online_eval_enabled:
        print(
            f"online eval enabled every_steps={online_eval_every_steps} games={online_eval_games} "
            f"opponents={online_eval_opponents} bc_player={online_eval_bc_player}",
            flush=True,
        )

    sample_batch = sample_batch_for_init(
        train_paths,
        normalizer,
        batch_size,
        allowed_winners,
        learn_result_filter=learn_result_filter,
    )

    ship_bucket_count = int(sample_batch.amount_mask.shape[-1])
    model = FeatureRowBCPolicy(
        ship_bucket_count=ship_bucket_count,
        hidden_size=hidden_size,
        use_cnn=use_cnn,
        target_head=target_head,
        target_num_heads=target_num_heads,
    )
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    train_state = make_train_state(init_rng, model, sample_batch, lr=lr, weight_decay=weight_decay)
    completed_passes = 0
    if resume_payload is not None:
        train_state = restore_train_state(train_state, resume_payload)
        completed_passes = int(resume_payload.get("completed_passes", 0) or 0)
        if "rng" in resume_payload:
            rng = jnp.asarray(resume_payload["rng"])
        print(
            f"resume: restored {args.resume} step={int(jax.device_get(train_state.step))} "
            f"completed_passes={completed_passes}",
            flush=True,
        )
    train_step = make_train_step(model, weights, normalizer)
    eval_step = make_eval_step(model, weights, normalizer)

    np.savez(save_dir / "normalizer.npz", **normalizer_np)
    (save_dir / "train_config_snapshot.yaml").write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")

    metrics_path = save_dir / "metrics.jsonl"
    best_val = metrics_best_val(metrics_path)
    if resume_payload is not None:
        best_val = min(
            best_val,
            float(resume_payload.get("best_val", float("inf"))),
            float(resume_payload.get("metrics", {}).get("val_loss", float("inf"))),
        )
    start_step = int(jax.device_get(train_state.step))
    if start_step >= max_steps:
        print(f"BC resume checkpoint already at step {start_step}; target max_steps={max_steps}, nothing to do.", flush=True)
        if use_wandb:
            _wandb.finish()
        return

    data_seed = seed + 1_000_003 * (start_step + 1)
    data_rng = np.random.default_rng(data_seed)
    stream = train_batch_stream(
        train_paths,
        batch_size,
        data_rng,
        noop_keep_ratio,
        args.max_rows_per_shard,
        allowed_winners=allowed_winners,
        learn_result_filter=learn_result_filter,
        initial_completed_passes=completed_passes,
        shuffle_shards=shuffle_shards,
        shuffle_buffer_rows=shuffle_buffer_rows,
    )
    stream = prefetch_batch_stream(stream, prefetch_batches)
    log_sums: dict[str, jnp.ndarray] = {}
    log_count = 0
    log_start_step = start_step
    log_start_time = time.time()
    last_metrics: dict[str, float] = {}
    step = start_step

    while step < max_steps:
        batch, completed_passes, _path = next(stream)
        train_state, train_metrics_raw = train_step(train_state, batch)
        step += 1
        batch_rows = int(batch.fire.shape[0])
        add_metric_sums_device(log_sums, train_metrics_raw, batch_rows)
        log_count += batch_rows

        final_step = step >= max_steps
        log_due = log_every_steps > 0 and (step % log_every_steps == 0 or final_step)
        eval_due = eval_every_steps > 0 and (step % eval_every_steps == 0 or final_step)
        online_due = online_eval_enabled and (step % online_eval_every_steps == 0 or final_step)
        checkpoint_due = checkpoint_every_steps > 0 and (step % checkpoint_every_steps == 0 or final_step)
        if not (log_due or eval_due or online_due or checkpoint_due):
            continue

        train_metrics = {
            key: float(value)
            for key, value in jax.device_get(average_metric_sums_device(log_sums, log_count)).items()
        }
        metrics = {f"train_{key}": value for key, value in train_metrics.items()}
        metrics["total_elapsed_sec"] = time.time() - tic
        metrics["steps_per_sec"] = (step - log_start_step) / max(time.time() - log_start_time, 1e-9)
        metrics["rows_per_sec"] = metrics["steps_per_sec"] * float(batch_size)

        improved = False
        if eval_due:
            val_metrics = run_eval_batches(
                val_paths,
                train_state,
                eval_step,
                batch_size,
                np.random.default_rng(seed + 2_000_003 + step),
                eval_batches,
                args.max_rows_per_shard,
                allowed_winners=allowed_winners,
                learn_result_filter=learn_result_filter,
            )
            metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
            improved = metrics["val_loss"] < best_val
            if improved:
                best_val = metrics["val_loss"]

        if online_due:
            for opponent_name in online_eval_opponents:
                prefix = online_eval_metric_prefix(opponent_name)
                try:
                    online_metrics = run_online_eval(
                        model=model,
                        train_state=train_state,
                        normalizer=normalizer,
                        raw_cfg=raw_cfg,
                        opponent_name=opponent_name,
                        agents_root=online_eval_agents_root,
                        games=online_eval_games,
                        seed=online_eval_seed + step,
                        bc_player=online_eval_bc_player,
                        max_fleets=online_eval_max_fleets,
                    )
                    metrics.update(prefix_online_eval_metrics(prefix, online_metrics))
                except Exception as exc:
                    if online_eval_fail_on_error:
                        raise
                    metrics[f"online_{prefix}_eval_error"] = 1.0
                    print(f"[online-eval/{prefix}] failed at step={step}: {exc!r}", flush=True)

        write_metrics(metrics_path, step, metrics, completed_passes=completed_passes)
        log_wandb(use_wandb, {"completed_passes": completed_passes, **metrics}, step=step)
        last_metrics = metrics
        online_text = ""
        online_parts = []
        for opponent_name in online_eval_opponents:
            prefix = online_eval_metric_prefix(opponent_name)
            wr_key = f"online_{prefix}_win_rate"
            if wr_key not in metrics:
                continue
            online_parts.append(f"{prefix}_wr={metrics[wr_key]:.3f}")
        if online_parts:
            online_text = " online " + " ".join(online_parts)
        if eval_due:
            print(
                f"step={step:08d}/{max_steps} loss={metrics['train_loss']:.4f} val={metrics['val_loss']:.4f} "
                f"fire_acc={metrics['val_fire_acc']:.3f} fire_rec={metrics['val_fire_recall']:.3f} "
                f"target={metrics['val_target_acc']:.3f}/{metrics['val_target_top3']:.3f} "
                f"amount={metrics['val_amount_acc']:.3f} mae={metrics['val_amount_mae']:.3f} "
                f"launch={metrics['val_launch_rate']:.3f}->{metrics['val_pred_launch_rate']:.3f} "
                f"steps/s={metrics['steps_per_sec']:.2f} rows/s={metrics['rows_per_sec']:.0f}{online_text}",
                flush=True,
            )
        elif log_due:
            print(
                f"step={step:08d}/{max_steps} loss={metrics['train_loss']:.4f} "
                f"fire_acc={metrics['train_fire_acc']:.3f} fire_rec={metrics['train_fire_recall']:.3f} "
                f"target={metrics['train_target_acc']:.3f}/{metrics['train_target_top3']:.3f} "
                f"amount={metrics['train_amount_acc']:.3f} mae={metrics['train_amount_mae']:.3f} "
                f"launch={metrics['train_launch_rate']:.3f}->{metrics['train_pred_launch_rate']:.3f} "
                f"steps/s={metrics['steps_per_sec']:.2f} rows/s={metrics['rows_per_sec']:.0f}{online_text}",
                flush=True,
            )
        elif online_due:
            print(
                f"step={step:08d}/{max_steps} loss={metrics['train_loss']:.4f} "
                f"steps/s={metrics['steps_per_sec']:.2f} rows/s={metrics['rows_per_sec']:.0f}{online_text}",
                flush=True,
            )

        if log_due or eval_due or online_due:
            log_sums = {}
            log_count = 0
            log_start_step = step
            log_start_time = time.time()

        if checkpoint_due or improved:
            save_checkpoint(
                save_dir / "ckpt_last.pkl",
                train_state,
                normalizer_np,
                raw_cfg,
                metrics,
                completed_passes=completed_passes,
                rng=rng,
                best_val=best_val,
            )
        if improved:
            save_checkpoint(
                save_dir / "ckpt_best.pkl",
                train_state,
                normalizer_np,
                raw_cfg,
                metrics,
                completed_passes=completed_passes,
                rng=rng,
                best_val=best_val,
            )
        if checkpoint_due:
            save_checkpoint(
                save_dir / f"ckpt_step_{step:08d}.pkl",
                train_state,
                normalizer_np,
                raw_cfg,
                metrics,
                completed_passes=completed_passes,
                rng=rng,
                best_val=best_val,
            )

    if last_metrics:
        print(f"BC done at step={step}; wrote {save_dir / 'ckpt_last.pkl'}", flush=True)
    if use_wandb:
        _wandb.finish()


if __name__ == "__main__":
    main()
