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
import sys
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
NORMALIZER_ARRAY_KEYS = (
    "feature__fight_planet",
    "feature__fight_relative",
    "feature__global",
    "feature__map",
    "feature__candidate_mask",
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_jax_BC_1.yaml")
    parser.add_argument("--dataset-glob", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-rows-per-shard", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a BC checkpoint path.")
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


def load_shard_arrays(path: Path, keys: tuple[str, ...] = TRAIN_ARRAY_KEYS) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in keys}


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

    relative = data["feature__fight_relative"].astype(np.float64)
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


def compute_normalizer(paths: list[Path]) -> dict[str, np.ndarray]:
    stats = zeros_stats()
    for path in paths:
        accumulate_stats(stats, load_shard_arrays(path, NORMALIZER_ARRAY_KEYS))
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


def jax_normalizer(normalizer: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {name: jnp.asarray(value) for name, value in normalizer.items()}


def normalize_batch(batch: FeatureRowBatch, norm: dict[str, jnp.ndarray]) -> FeatureRowBatch:
    planet = (batch.planet - norm["planet_mean"]) / jnp.maximum(norm["planet_std"], 1e-6)
    relative = (batch.relative - norm["relative_mean"]) / jnp.maximum(norm["relative_std"], 1e-6)
    relative = jnp.where(batch.candidate_mask[..., None], relative, 0.0)
    global_features = (batch.global_features - norm["global_mean"]) / jnp.maximum(norm["global_std"], 1e-6)
    map_features = (batch.map_features - norm["map_mean"]) / jnp.maximum(norm["map_std"], 1e-6)
    return batch._replace(planet=planet, relative=relative, global_features=global_features, map_features=map_features)


def batch_from_arrays(data: Mapping[str, np.ndarray], idx: np.ndarray) -> FeatureRowBatch:
    return FeatureRowBatch(
        planet=jnp.asarray(data["feature__fight_planet"][idx], dtype=jnp.float32),
        relative=jnp.asarray(data["feature__fight_relative"][idx], dtype=jnp.float32),
        global_features=jnp.asarray(data["feature__global"][idx], dtype=jnp.float32),
        map_features=jnp.asarray(data["feature__map"][idx], dtype=jnp.float32),
        candidate_mask=jnp.asarray(data["feature__candidate_mask"][idx], dtype=bool),
        amount_mask=jnp.asarray(data["feature__amount_mask"][idx], dtype=bool),
        fire=jnp.asarray(data["label__fire"][idx], dtype=bool),
        target=jnp.asarray(data["label__target"][idx], dtype=jnp.int32),
        amount=jnp.asarray(data["label__amount"][idx], dtype=jnp.int32),
    )


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


def make_train_step(model, weights):
    @jax.jit
    def train_step(train_state, batch: FeatureRowBatch):
        def loss_fn(params):
            return bc_loss_and_metrics(model, params, batch, weights)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        del loss
        return train_state.apply_gradients(grads=grads), metrics

    return train_step


def make_eval_step(model, weights):
    @jax.jit
    def eval_step(train_state, batch: FeatureRowBatch):
        _, metrics = bc_loss_and_metrics(model, train_state.params, batch, weights)
        return metrics

    return eval_step


def epoch_indices(
    data: Mapping[str, np.ndarray],
    rng,
    train: bool,
    noop_keep_ratio: float,
    max_rows_per_shard: int | None,
) -> tuple[np.ndarray, Any]:
    rows = int(data["label__fire"].shape[0])
    keep = np.ones((rows,), dtype=bool)
    if train and noop_keep_ratio < 1.0:
        fire = data["label__fire"].astype(bool)
        rng, keep_key = jax.random.split(rng)
        keep_noop = np.asarray(jax.random.uniform(keep_key, (rows,)) < float(noop_keep_ratio))
        keep = fire | keep_noop
        if not keep.any():
            keep[:] = True
    idx = np.flatnonzero(keep)
    if max_rows_per_shard is not None and int(max_rows_per_shard) > 0:
        idx = idx[: int(max_rows_per_shard)]
    if train:
        rng, order_key = jax.random.split(rng)
        idx = np.asarray(jax.random.permutation(order_key, idx))
    return idx, rng


def run_epoch(
    paths: list[Path],
    train_state,
    step_fn,
    normalizer: dict[str, jnp.ndarray],
    batch_size: int,
    rng,
    train: bool,
    noop_keep_ratio: float,
    max_rows_per_shard: int | None = None,
):
    sums: dict[str, float] = {}
    count = 0
    for path in paths:
        data = load_shard_arrays(path)
        order, rng = epoch_indices(
            data,
            rng,
            train=train,
            noop_keep_ratio=noop_keep_ratio,
            max_rows_per_shard=max_rows_per_shard,
        )
        if len(order) == 0:
            continue
        starts = range(0, len(order) - batch_size + 1, batch_size)
        if len(order) < batch_size:
            starts = range(0, 1)
        for start in starts:
            idx = order[start : start + batch_size]
            if len(idx) < batch_size and count > 0:
                continue
            batch = normalize_batch(batch_from_arrays(data, idx), normalizer)
            if train:
                train_state, metrics = step_fn(train_state, batch)
            else:
                metrics = step_fn(train_state, batch)
            metrics = jax.device_get(metrics)
            batch_rows = int(len(idx))
            count += batch_rows
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value) * batch_rows
    return train_state, rng, {k: v / max(count, 1) for k, v in sums.items()}


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
    epoch: int,
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
        "epoch": int(epoch),
        "rng": jax.device_get(rng),
        "best_val": float(best_val),
        "checkpoint_type": "feature_row_bc",
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    np.savez(path.parent / "normalizer.npz", **normalizer)


def write_metrics(path: Path, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"epoch": epoch, "time": time.time(), **metrics}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    raw_cfg = read_raw_config(args.config)
    resume_payload = load_checkpoint_payload(args.resume)
    dataset_pattern = args.dataset_glob or str(raw_get(raw_cfg, "bc", "dataset_glob", ""))
    paths = shard_paths(dataset_pattern, args.max_shards)
    train_paths, val_paths = split_paths(paths, float(raw_get(raw_cfg, "bc", "val_fraction", 0.1)))
    epochs = int(args.epochs or raw_get(raw_cfg, "bc", "epochs", 1))
    batch_size = int(args.batch_size or raw_get(raw_cfg, "bc", "batch_size", 512))
    lr = float(args.lr or raw_get(raw_cfg, "bc", "lr", 3e-4))
    hidden_size = int(raw_get(raw_cfg, "model", "hidden_size", 128))
    use_cnn = bool(raw_get(raw_cfg, "model", "use_cnn", True))
    weight_decay = float(raw_get(raw_cfg, "bc", "weight_decay", 0.0))
    noop_keep_ratio = float(raw_get(raw_cfg, "bc", "noop_keep_ratio", 0.25))
    weights = {
        "noop_weight": float(raw_get(raw_cfg, "bc", "noop_weight", 1.0)),
        "fire_weight": float(raw_get(raw_cfg, "bc", "fire_weight", 1.0)),
        "target_weight": float(raw_get(raw_cfg, "bc", "target_weight", 1.0)),
        "amount_weight": float(raw_get(raw_cfg, "bc", "amount_weight", 0.5)),
    }

    tic = time.time()
    if resume_payload is not None and "normalizer" in resume_payload and not args.no_normalize:
        normalizer_np = resume_payload["normalizer"]
        print(f"resume: loaded normalizer from {args.resume}", flush=True)
    else:
        normalizer_np = identity_normalizer() if args.no_normalize else compute_normalizer(train_paths)
    normalizer = jax_normalizer(normalizer_np)
    print(
        f"BC feature-row start shards={len(paths)} train={len(train_paths)} val={len(val_paths)} "
        f"epochs={epochs} batch={batch_size} norm_counts={normalizer_np['counts'].tolist()}",
        flush=True,
    )

    sample_data = load_shard_arrays(train_paths[0])
    sample_idx = np.arange(min(batch_size, int(sample_data["label__fire"].shape[0])))
    sample_batch = normalize_batch(batch_from_arrays(sample_data, sample_idx), normalizer)

    ship_bucket_count = int(sample_batch.amount_mask.shape[-1])
    model = FeatureRowBCPolicy(ship_bucket_count=ship_bucket_count, hidden_size=hidden_size, use_cnn=use_cnn)
    rng = jax.random.PRNGKey(int(raw_cfg.get("seed", 42)))
    rng, init_rng = jax.random.split(rng)
    train_state = make_train_state(init_rng, model, sample_batch, lr=lr, weight_decay=weight_decay)
    start_epoch = 1
    if resume_payload is not None:
        train_state = restore_train_state(train_state, resume_payload)
        resumed_epoch = int(resume_payload.get("epoch", 0) or 0)
        start_epoch = resumed_epoch + 1
        if "rng" in resume_payload:
            rng = jnp.asarray(resume_payload["rng"])
        print(
            f"resume: restored {args.resume} epoch={resumed_epoch} "
            f"step={int(jax.device_get(train_state.step))} next_epoch={start_epoch}",
            flush=True,
        )
    train_step = make_train_step(model, weights)
    eval_step = make_eval_step(model, weights)

    save_root = Path(str(args.save_dir or raw_cfg.get("save_dir", "/mnt/f/kaggle-orbit-wars-data/bc_runs")))
    run_name = str(args.run_name or raw_cfg.get("run_name", "bc_feature_rows"))
    save_dir = save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_dir / "normalizer.npz", **normalizer_np)
    (save_dir / "train_config_snapshot.yaml").write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")

    checkpoint_every = int(raw_cfg.get("checkpoint_every", 1))
    metrics_path = save_dir / "metrics.jsonl"
    best_val = metrics_best_val(metrics_path)
    if resume_payload is not None:
        best_val = min(
            best_val,
            float(resume_payload.get("best_val", float("inf"))),
            float(resume_payload.get("metrics", {}).get("val_loss", float("inf"))),
        )
    if start_epoch > epochs:
        print(f"BC resume checkpoint already at epoch {start_epoch - 1}; target epochs={epochs}, nothing to do.", flush=True)
        return

    for epoch in range(start_epoch, epochs + 1):
        epoch_tic = time.time()
        train_state, rng, train_metrics = run_epoch(
            train_paths,
            train_state,
            train_step,
            normalizer,
            batch_size,
            rng,
            train=True,
            noop_keep_ratio=noop_keep_ratio,
            max_rows_per_shard=args.max_rows_per_shard,
        )
        train_state, rng, val_metrics = run_epoch(
            val_paths,
            train_state,
            eval_step,
            normalizer,
            batch_size,
            rng,
            train=False,
            noop_keep_ratio=1.0,
            max_rows_per_shard=args.max_rows_per_shard,
        )
        metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        elapsed = time.time() - epoch_tic
        metrics["epoch_elapsed_sec"] = elapsed
        metrics["total_elapsed_sec"] = time.time() - tic
        write_metrics(metrics_path, epoch, metrics)
        print(
            f"epoch={epoch:04d} loss={metrics['train_loss']:.4f} val={metrics['val_loss']:.4f} "
            f"fire_acc={metrics['val_fire_acc']:.3f} fire_rec={metrics['val_fire_recall']:.3f} "
            f"target={metrics['val_target_acc']:.3f}/{metrics['val_target_top3']:.3f} "
            f"amount={metrics['val_amount_acc']:.3f} mae={metrics['val_amount_mae']:.3f} "
            f"launch={metrics['val_launch_rate']:.3f}->{metrics['val_pred_launch_rate']:.3f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        improved = metrics["val_loss"] < best_val
        if improved:
            best_val = metrics["val_loss"]
        save_checkpoint(
            save_dir / "ckpt_last.pkl",
            train_state,
            normalizer_np,
            raw_cfg,
            metrics,
            epoch=epoch,
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
                epoch=epoch,
                rng=rng,
                best_val=best_val,
            )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(
                save_dir / f"ckpt_epoch_{epoch:04d}.pkl",
                train_state,
                normalizer_np,
                raw_cfg,
                metrics,
                epoch=epoch,
                rng=rng,
                best_val=best_val,
            )

    print(f"BC done, wrote {save_dir / 'ckpt_best.pkl'}", flush=True)


if __name__ == "__main__":
    main()
