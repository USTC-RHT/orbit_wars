from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from jax_agent.config import load_config
from jax_agent.fast_env import EnvParams, EnvState
from jax_agent.features_jax import FLEET_DIM, FeatureParams, encode_batch
from jax_agent.model_jax import AutoregressiveActorCritic
from jax_agent.normalization_jax import NormalizerParams, init_normalizer, normalize_encoded
from jax_agent.ppo_jax import make_auto_train_state, safe_logits
from jax_agent.train_jax import (
    dynamic_ship_buckets,
    flip_state,
    kaggle_obs_to_jax_state,
)
from true_physics import Planet as PhysicsPlanet
from true_physics import aim_with_prediction


@dataclass(frozen=True)
class PhysicsContext:
    planets: list[PhysicsPlanet]
    planets_by_id: dict[int, PhysicsPlanet]
    initial_by_id: dict[int, PhysicsPlanet]
    comets: list[dict]
    comet_ids: set[int]
    angular_velocity: float


def _read(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _iter_replay_objects(paths: list[str]) -> Iterable[tuple[str, dict]]:
    for pattern in paths:
        for raw in sorted(glob.glob(pattern, recursive=True)):
            path = Path(raw)
            if path.name.endswith("-logs.json") or path.name.endswith(".meta.json") or path.name == "_metadata.json":
                continue
            if path.suffix == ".json":
                try:
                    yield str(path), json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
            elif path.suffixes[-2:] == [".tar", ".gz"] or path.name.endswith(".tar.gz"):
                with tarfile.open(path, "r:gz") as tar:
                    for member in tar:
                        if not member.isfile() or not member.name.endswith(".json"):
                            continue
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        try:
                            yield f"{path}:{member.name}", json.load(f)
                        except Exception:
                            continue


def _canonical_state_and_actions(obs: dict, actions: list, env_params: EnvParams) -> tuple[EnvState, list[list[float]]]:
    state = kaggle_obs_to_jax_state(obs, env_params)
    player = int(_read(obs, "player", 0) or 0)
    out: list[list[float]] = []
    for row in actions or []:
        if row is None or len(row) < 3:
            continue
        src = float(row[0])
        angle = float(row[1])
        ships = float(row[2])
        if ships <= 0:
            continue
        if player == 1:
            angle = math.atan2(-math.sin(angle), -math.cos(angle))
        out.append([src, angle, ships])
    if player == 1:
        state = flip_state(state)
    return state, out


def _angle_error(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _planet_from_row(row: np.ndarray) -> PhysicsPlanet:
    return PhysicsPlanet(
        id=int(round(float(row[0]))),
        owner=int(round(float(row[1]))),
        x=float(row[2]),
        y=float(row[3]),
        radius=float(row[4]),
        ships=float(row[5]),
        production=float(row[6]),
    )


def _physics_context_from_state(state: EnvState) -> PhysicsContext:
    planets_arr = np.asarray(state.planets[0], dtype=np.float64)
    initial_arr = np.asarray(state.initial_planets[0], dtype=np.float64)
    alive = np.asarray(state.planet_alive[0], dtype=bool)

    planets: list[PhysicsPlanet] = []
    planets_by_id: dict[int, PhysicsPlanet] = {}
    initial_by_id: dict[int, PhysicsPlanet] = {}
    for slot, is_alive in enumerate(alive):
        if not is_alive:
            continue
        planet = _planet_from_row(planets_arr[slot])
        initial = _planet_from_row(initial_arr[slot])
        planets.append(planet)
        planets_by_id[planet.id] = planet
        initial_by_id[initial.id] = initial

    comet_paths = np.asarray(state.comet_paths[0], dtype=np.float64)
    comet_lengths = np.asarray(state.comet_lengths[0], dtype=np.int32)
    comet_path_index = np.asarray(state.comet_path_index[0], dtype=np.int32)
    comet_slots = np.asarray(state.comet_slots[0], dtype=np.int32)
    comets: list[dict] = []
    comet_ids: set[int] = set()
    for group_idx in range(comet_slots.shape[0]):
        path_index = int(comet_path_index[group_idx])
        if path_index < -1:
            continue
        group_ids: list[int] = []
        group_paths: list[list[list[float]]] = []
        for comet_idx in range(comet_slots.shape[1]):
            slot = int(comet_slots[group_idx, comet_idx])
            if slot < 0 or slot >= planets_arr.shape[0] or not alive[slot]:
                continue
            length = int(comet_lengths[group_idx, comet_idx])
            if length <= 0:
                continue
            pid = int(round(float(planets_arr[slot, 0])))
            path = comet_paths[group_idx, comet_idx, :length].tolist()
            group_ids.append(pid)
            group_paths.append(path)
            comet_ids.add(pid)
        if group_ids:
            comets.append({"planet_ids": group_ids, "paths": group_paths, "path_index": path_index})

    return PhysicsContext(
        planets=planets,
        planets_by_id=planets_by_id,
        initial_by_id=initial_by_id,
        comets=comets,
        comet_ids=comet_ids,
        angular_velocity=float(np.asarray(state.angular_velocity[0])),
    )


def _find_source_slot(state: EnvState, from_id: float) -> int | None:
    planets = np.asarray(state.planets[0])
    alive = np.asarray(state.planet_alive[0])
    matches = np.where((alive) & (planets[:, 0].astype(np.int32) == int(round(from_id))) & (planets[:, 1] == 0))[0]
    return int(matches[0]) if len(matches) else None


def _infer_target_slot(
    state: EnvState,
    physics: PhysicsContext,
    source_slot: int,
    angle: float,
    ships: float,
    candidate_mask_row: np.ndarray,
) -> int | None:
    planets = np.asarray(state.planets[0], dtype=np.float64)
    alive = np.asarray(state.planet_alive[0], dtype=bool)
    p = planets.shape[0]
    mask = np.asarray(candidate_mask_row[1 : p + 1], dtype=bool) & alive
    mask[source_slot] = False
    if not mask.any():
        return None

    src_id = int(round(float(planets[source_slot, 0])))
    src = physics.planets_by_id.get(src_id)
    if src is None:
        return None

    best_slot: int | None = None
    best_score = math.inf
    for target_slot in np.where(mask)[0]:
        target_id = int(round(float(planets[target_slot, 0])))
        target = physics.planets_by_id.get(target_id)
        if target is None:
            continue
        aim = aim_with_prediction(
            src=src,
            target=target,
            ships=max(1, int(round(float(ships)))),
            initial_by_id=physics.initial_by_id,
            ang_vel=physics.angular_velocity,
            comets=physics.comets,
            comet_ids=physics.comet_ids,
            planets=physics.planets,
        )
        if aim is None:
            continue
        theta, eta, _hit_x, _hit_y = aim
        score = float(_angle_error(np.asarray([theta], dtype=np.float64), float(angle))[0])
        # Deterministic tie-breaker; ETA is not a label, but nearer valid hits
        # make angle ties less brittle on symmetric maps.
        score += 1e-5 * float(eta)
        if score < best_score:
            best_score = score
            best_slot = int(target_slot)

    return best_slot


def _closest_amount_bucket(remaining: float, ships: float, bucket_count: int) -> int | None:
    buckets = np.asarray(dynamic_ship_buckets(jnp.asarray([remaining], dtype=jnp.float32), bucket_count))[0]
    valid = buckets > 0
    if not valid.any():
        return None
    err = np.where(valid, np.abs(buckets - float(ships)), np.inf)
    return int(np.argmin(err))


def _empty_label(max_tokens: int) -> dict[str, np.ndarray]:
    return {
        "end_action": np.zeros((max_tokens,), dtype=np.bool_),
        "src_action": np.zeros((max_tokens,), dtype=np.int32),
        "tgt_action": np.zeros((max_tokens,), dtype=np.int32),
        "amt_action": np.zeros((max_tokens,), dtype=np.int32),
        "executed": np.zeros((max_tokens,), dtype=np.bool_),
        "token_valid": np.zeros((max_tokens,), dtype=np.bool_),
    }


def _label_one(
    state: EnvState,
    actions: list[list[float]],
    feat_params: FeatureParams,
    max_tokens: int,
) -> tuple[dict[str, np.ndarray], int, int]:
    encoded = encode_batch(state, feat_params)
    candidate_mask = np.asarray(encoded.candidate_mask[0])
    remaining = np.concatenate([[0.0], np.asarray(encoded.source_ships[0], dtype=np.float32)])
    physics = _physics_context_from_state(state)
    label = _empty_label(max_tokens)
    used = 0
    skipped = 0
    for row in actions:
        if used >= max_tokens:
            break
        source_slot = _find_source_slot(state, row[0])
        if source_slot is None:
            skipped += 1
            continue
        src_token = source_slot + 1
        if src_token >= remaining.shape[0] or remaining[src_token] < 4.0:
            skipped += 1
            continue
        target_slot = _infer_target_slot(state, physics, source_slot, row[1], row[2], candidate_mask[source_slot])
        if target_slot is None:
            skipped += 1
            continue
        tgt_token = target_slot + 1
        if tgt_token <= 0 or tgt_token >= candidate_mask.shape[1] or not candidate_mask[source_slot, tgt_token]:
            skipped += 1
            continue
        bucket = _closest_amount_bucket(float(remaining[src_token]), row[2], feat_params.ship_bucket_count)
        if bucket is None:
            skipped += 1
            continue
        buckets = np.asarray(dynamic_ship_buckets(jnp.asarray([remaining[src_token]], dtype=jnp.float32), feat_params.ship_bucket_count))[0]
        ships_sent = float(buckets[bucket])
        if ships_sent <= 0:
            skipped += 1
            continue
        label["token_valid"][used] = True
        label["src_action"][used] = src_token
        label["tgt_action"][used] = tgt_token
        label["amt_action"][used] = bucket
        label["executed"][used] = True
        remaining[src_token] = max(0.0, remaining[src_token] - ships_sent)
        used += 1
    if used < max_tokens:
        label["token_valid"][used] = True
        label["end_action"][used] = True
    return label, used, skipped


def _concat_states(states: list[EnvState]) -> EnvState:
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *states)


def build_dataset(args, cfg, env_params: EnvParams, feat_params: FeatureParams):
    random.seed(int(args.seed))
    states: list[EnvState] = []
    labels: list[dict[str, np.ndarray]] = []
    stats = {"replays": 0, "rows": 0, "nonempty_rows": 0, "executed_tokens": 0, "skipped_actions": 0}
    for _name, replay in _iter_replay_objects(args.replay_glob):
        steps = replay.get("steps") or []
        if not steps or len(steps[0]) != 2:
            continue
        stats["replays"] += 1
        stride = max(int(args.stride), 1)
        for step_idx, step in enumerate(steps):
            if step_idx % stride != 0:
                continue
            for entry in step:
                if len(states) >= int(args.max_samples):
                    break
                obs = entry.get("observation") if isinstance(entry, dict) else None
                if not isinstance(obs, dict):
                    continue
                action = entry.get("action") or []
                if not action and not args.include_empty:
                    continue
                state, actions = _canonical_state_and_actions(obs, action, env_params)
                label, used, skipped = _label_one(state, actions, feat_params, int(cfg.model.max_action_tokens))
                if used == 0 and not args.include_empty:
                    continue
                states.append(state)
                labels.append(label)
                stats["rows"] += 1
                stats["nonempty_rows"] += int(used > 0)
                stats["executed_tokens"] += used
                stats["skipped_actions"] += skipped
            if len(states) >= int(args.max_samples):
                break
        if len(states) >= int(args.max_samples):
            break
    if not states:
        raise RuntimeError("No BC samples found. Pass --replay-glob pointing at replay JSON files or episodes.tar.gz.")
    batch_state = _concat_states(states)
    stacked = {k: jnp.asarray(np.stack([lab[k] for lab in labels], axis=0)) for k in labels[0]}
    return batch_state, stacked, stats


def make_model(cfg, sample):
    layers = int(getattr(cfg.model, "num_transformer_layers", 2)) if bool(getattr(cfg.model, "use_transformer", False)) else 0
    return AutoregressiveActorCritic(
        planet_dim=sample.planet_features.shape[-1],
        relative_dim=sample.relative_features.shape[-1],
        global_dim=sample.global_features.shape[-1],
        relative_count=cfg.env.max_planets + 1,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        use_cnn=cfg.model.use_cnn,
        fleet_dim=FLEET_DIM,
        max_fleet_tokens=feat_params_from_cfg(cfg).max_fleet_tokens,
        num_transformer_layers=layers,
        num_heads=int(getattr(cfg.model, "num_heads", 4)),
        max_action_tokens=int(getattr(cfg.model, "max_action_tokens", 16)),
        end_bias_init=float(getattr(cfg.model, "end_bias_init", -2.0)),
    )


def feat_params_from_cfg(cfg) -> FeatureParams:
    return FeatureParams.from_train_config(cfg)


def make_loss_fn(model, encoded, labels):
    initial_ships = jnp.concatenate([jnp.zeros((encoded.source_ships.shape[0], 1), dtype=encoded.source_ships.dtype), encoded.source_ships], axis=1)

    def loss_fn(params):
        out = model.apply(
            {"params": params},
            encoded.planet_features,
            encoded.relative_features,
            encoded.global_features,
            encoded.candidate_mask,
            encoded.amount_mask,
            encoded.source_mask,
            labels["end_action"],
            labels["src_action"],
            labels["tgt_action"],
            labels["amt_action"],
            labels["executed"],
            initial_ships,
            map_features=encoded.map_features,
            fleet_features=encoded.fleet_features,
            fleet_mask=encoded.fleet_mask,
        )
        token_valid = labels["token_valid"].astype(jnp.float32)
        executed = labels["executed"].astype(jnp.float32)
        end_loss = optax.sigmoid_binary_cross_entropy(out.end_logits, labels["end_action"].astype(jnp.float32))
        end_loss = (end_loss * token_valid).sum() / jnp.maximum(token_valid.sum(), 1.0)
        src_logp = jax.nn.log_softmax(safe_logits(out.src_logits), axis=-1)
        tgt_logp = jax.nn.log_softmax(safe_logits(out.tgt_logits), axis=-1)
        amt_logp = jax.nn.log_softmax(safe_logits(out.amount_logits), axis=-1)
        src_loss = -jnp.take_along_axis(src_logp, labels["src_action"][..., None], axis=-1).squeeze(-1)
        tgt_loss = -jnp.take_along_axis(tgt_logp, labels["tgt_action"][..., None], axis=-1).squeeze(-1)
        amt_loss = -jnp.take_along_axis(amt_logp, labels["amt_action"][..., None], axis=-1).squeeze(-1)
        denom = jnp.maximum(executed.sum(), 1.0)
        action_loss = ((src_loss + tgt_loss + amt_loss) * executed).sum() / denom
        loss = end_loss + action_loss
        metrics = {
            "loss": loss,
            "end_loss": end_loss,
            "action_loss": action_loss,
            "end_acc": (((out.end_logits > 0.0) == labels["end_action"]) * labels["token_valid"]).sum() / jnp.maximum(labels["token_valid"].sum(), 1),
            "src_acc": ((jnp.argmax(out.src_logits, axis=-1) == labels["src_action"]) * labels["executed"]).sum() / jnp.maximum(labels["executed"].sum(), 1),
            "tgt_acc": ((jnp.argmax(out.tgt_logits, axis=-1) == labels["tgt_action"]) * labels["executed"]).sum() / jnp.maximum(labels["executed"].sum(), 1),
            "amt_acc": ((jnp.argmax(out.amount_logits, axis=-1) == labels["amt_action"]) * labels["executed"]).sum() / jnp.maximum(labels["executed"].sum(), 1),
        }
        return loss, metrics

    return loss_fn


def _default_replay_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "replays" / "kaggle").is_dir():
            return parent
        lab = parent / "orbit-wars-lab"
        if (lab / "replays" / "kaggle").is_dir():
            return lab
    return here.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior clone the local JAX autoregressive Transformer from replay actions.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("bc_config.yaml"))
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("checkpoints") / "ckpt_bc_latest.pkl")
    args = parser.parse_args()

    if not args.replay_glob:
        root = _default_replay_root()
        args.replay_glob = [
            str(root / "replays" / "kaggle" / "**" / "episode_*.json"),
            str(root / "runs" / "2026-06-08-003" / "replays" / "*.json"),
        ]

    cfg = load_config(args.config)
    cfg.seed = int(args.seed)
    cfg.ppo.epochs = int(args.epochs)
    env_params = EnvParams(
        episode_steps=cfg.env.episode_steps,
        max_planets=cfg.env.max_planets,
        max_fleets=512,
        max_actions=cfg.env.max_planets,
    )
    feat_params = feat_params_from_cfg(cfg)
    norm_params = NormalizerParams.from_train_config(cfg)
    states, labels, data_stats = build_dataset(args, cfg, env_params, feat_params)
    encoded_raw = encode_batch(states, feat_params)
    normalizer_state = init_normalizer()
    encoded = normalize_encoded(encoded_raw, normalizer_state, norm_params)
    model = make_model(cfg, encoded)
    rng = jax.random.PRNGKey(int(args.seed))
    train_state = make_auto_train_state(
        rng,
        model,
        encoded,
        cfg.ppo.lr,
        cfg.ppo.max_grad_norm,
        max_action_tokens=int(cfg.model.max_action_tokens),
        lr_schedule="constant",
    )
    loss_fn = make_loss_fn(model, encoded, labels)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    for epoch in range(1, int(args.epochs) + 1):
        (loss, metrics), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        host = {k: float(v) for k, v in jax.device_get(metrics).items()}
        print(
            f"epoch={epoch} loss={host['loss']:.4f} end={host['end_loss']:.4f} action={host['action_loss']:.4f} "
            f"acc(end/src/tgt/amt)={host['end_acc']:.3f}/{host['src_acc']:.3f}/{host['tgt_acc']:.3f}/{host['amt_acc']:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": "producer_bc_transformer_v1",
        "params": jax.device_get(train_state.params),
        "normalizer_state": jax.device_get(normalizer_state),
        "cfg": cfg,
        "data_stats": data_stats,
        "label_stats": {k: int(np.asarray(v).sum()) if v.dtype == jnp.bool_ else list(np.asarray(v).shape) for k, v in labels.items()},
    }
    with args.out.open("wb") as f:
        pickle.dump(payload, f)
    print(f"saved {args.out}")
    print(json.dumps(data_stats, indent=2))


if __name__ == "__main__":
    main()
