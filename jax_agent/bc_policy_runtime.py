"""Runtime bridge from feature-row BC checkpoints to Kaggle actions."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .bc_fight_feature_extras import build_fight_feature_extras
from .build_bc_feature_rows_from_zips import encode_light_sources
from .config import EnvConfig
from .fast_env import EnvParams, EnvState, state_from_observation
from .features_jax import FeatureParams, SHIP_RATIOS
from .train_bc_jax import FeatureRowBCPolicy, FeatureRowBatch, identity_normalizer, jax_normalizer, normalize_batch
from .train_jax import aim_with_prediction_jax, fit_action_slots

NEG_INF = -1.0e9


class BCPolicyRows(NamedTuple):
    planet: jnp.ndarray
    relative: jnp.ndarray
    global_features: jnp.ndarray
    map_features: jnp.ndarray
    candidate_mask: jnp.ndarray
    amount_mask: jnp.ndarray
    source_mask: jnp.ndarray
    source_ids: jnp.ndarray
    source_slots: jnp.ndarray
    source_ships: jnp.ndarray
    ship_buckets: jnp.ndarray


def _raw_get(raw: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = raw.get(section, {}).get(key, default)
    return default if value is None else value


def env_config_from_raw(raw: dict[str, Any]) -> EnvConfig:
    cfg = EnvConfig()
    env = raw.get("env", {}) if isinstance(raw, dict) else {}
    if not isinstance(env, dict):
        return cfg
    for key, value in env.items():
        if hasattr(cfg, key):
            current = getattr(cfg, key)
            if isinstance(current, bool):
                setattr(cfg, key, bool(value))
            elif isinstance(current, int):
                setattr(cfg, key, int(value))
            elif isinstance(current, float):
                setattr(cfg, key, float(value))
            else:
                setattr(cfg, key, value)
    return cfg


def build_env_params_from_raw(raw: dict[str, Any], *, max_fleets: int = 512) -> EnvParams:
    return EnvParams(
        episode_steps=int(_raw_get(raw, "env", "episode_steps", 500)),
        board_size=float(_raw_get(raw, "env", "board_size", 100.0)),
        max_planets=int(_raw_get(raw, "env", "max_planets", 45)),
        max_fleets=int(max_fleets),
        max_actions=int(_raw_get(raw, "env", "max_planets", 45)),
    )


def canonicalize_state_owner_for_player(state: EnvState, player: int) -> EnvState:
    """Put the requested player in owner slot 0 without rotating coordinates.

    BC feature rows were built by owner-canonicalizing the learner side; keeping
    real coordinates means predicted angles can be emitted directly to Kaggle.
    """

    if int(player) == 0:
        return state

    def flip_owner(owner):
        return jnp.where(owner == 0.0, 1.0, jnp.where(owner == 1.0, 0.0, owner))

    planets = state.planets.at[:, 1].set(flip_owner(state.planets[:, 1]))
    initial_planets = state.initial_planets.at[:, 1].set(flip_owner(state.initial_planets[:, 1]))
    fleets = state.fleets.at[:, 1].set(flip_owner(state.fleets[:, 1]))
    return state._replace(planets=planets, initial_planets=initial_planets, fleets=fleets)


def obs_to_bc_state(obs: Any, env_params: EnvParams, player: int | None = None) -> EnvState:
    state = state_from_observation(obs, env_params)
    if player is None:
        if isinstance(obs, dict):
            player = int(obs.get("player", 0) or 0)
        else:
            player = int(getattr(obs, "player", 0) or 0)
    return canonicalize_state_owner_for_player(state, int(player))


def batch_state(state: EnvState) -> EnvState:
    return EnvState(**{name: value[None] for name, value in state._asdict().items()})


def bc_ship_buckets(source_ships: jnp.ndarray, bucket_count: int) -> jnp.ndarray:
    ships = jnp.floor(source_ships).astype(jnp.float32)
    ratios = jnp.asarray(SHIP_RATIOS[: int(bucket_count)], dtype=jnp.float32)
    raw = jnp.round(ships[:, None] * ratios[None, :])
    return jnp.where(ships[:, None] > 0.0, jnp.clip(raw, 1.0, ships[:, None]), 0.0)


def build_bc_policy_rows(state: EnvState, params: FeatureParams, bucket_count: int) -> BCPolicyRows:
    source_slots = jnp.arange(int(params.max_planets), dtype=jnp.int32)
    encoded = encode_light_sources(state, source_slots, params)
    fight = build_fight_feature_extras(state, params)

    source_ships = encoded.source_ships.astype(jnp.float32)
    ship_buckets = bc_ship_buckets(source_ships, int(bucket_count))
    alive = state.planet_alive
    per_planet_mask = alive[None, :] & (source_slots[None, :] != source_slots[:, None])
    candidate_mask = jnp.concatenate(
        [jnp.zeros((int(params.max_planets), 1), dtype=bool), per_planet_mask],
        axis=1,
    )
    amount_mask = candidate_mask[:, :, None] & (ship_buckets > 0.0)[:, None, :]
    return BCPolicyRows(
        planet=fight.planet,
        relative=fight.relative,
        global_features=encoded.global_features,
        map_features=encoded.map_features,
        candidate_mask=candidate_mask,
        amount_mask=amount_mask,
        source_mask=encoded.source_mask,
        source_ids=encoded.source_ids,
        source_slots=encoded.source_slots,
        source_ships=source_ships,
        ship_buckets=ship_buckets,
    )


def normalize_rows(rows: BCPolicyRows, normalizer: dict[str, jnp.ndarray]) -> FeatureRowBatch:
    dummy_bool = jnp.zeros((rows.planet.shape[0],), dtype=bool)
    dummy_i32 = jnp.zeros((rows.planet.shape[0],), dtype=jnp.int32)
    batch = FeatureRowBatch(
        planet=rows.planet,
        relative=rows.relative,
        global_features=rows.global_features,
        map_features=rows.map_features,
        candidate_mask=rows.candidate_mask,
        amount_mask=rows.amount_mask,
        fire=dummy_bool,
        target=dummy_i32,
        amount=dummy_i32,
    )
    return normalize_batch(batch, normalizer)


def safe_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    masked = jnp.where(mask, logits, NEG_INF)
    empty = ~mask.any(axis=-1, keepdims=True)
    return jnp.where(empty, 0.0, masked)


def bc_decisions(model: FeatureRowBCPolicy, params: Any, rows: BCPolicyRows, normalizer: dict[str, jnp.ndarray]):
    batch = normalize_rows(rows, normalizer)
    zero_target = jnp.zeros((batch.planet.shape[0],), dtype=jnp.int32)
    base_out = model.apply(
        {"params": params},
        batch.planet,
        batch.relative,
        batch.global_features,
        batch.candidate_mask,
        batch.map_features,
        zero_target,
    )
    can_fire = rows.source_mask & batch.candidate_mask[:, 1:].any(axis=-1)
    fire = (jax.nn.sigmoid(base_out.fire_logits) >= 0.5) & can_fire

    target_mask = batch.candidate_mask.at[:, 0].set(False)
    target_logits = safe_logits(base_out.target_logits, target_mask)
    target = jnp.where(fire, jnp.argmax(target_logits, axis=-1), 0).astype(jnp.int32)

    action_out = model.apply(
        {"params": params},
        batch.planet,
        batch.relative,
        batch.global_features,
        batch.candidate_mask,
        batch.map_features,
        target,
    )
    selected_amount_mask = jnp.take_along_axis(batch.amount_mask, target[:, None, None], axis=1).squeeze(1)
    amount_logits = safe_logits(action_out.amount_logits, selected_amount_mask)
    amount = jnp.where(fire, jnp.argmax(amount_logits, axis=-1), 0).astype(jnp.int32)
    return fire, target, amount


def bc_actions_from_decisions(
    state: EnvState,
    rows: BCPolicyRows,
    fire: jnp.ndarray,
    target: jnp.ndarray,
    amount: jnp.ndarray,
    env_params: EnvParams,
) -> jnp.ndarray:
    state_b = batch_state(state)
    fire_b = fire[None]
    target_b = target[None]
    amount_b = amount[None]
    target_slot = jnp.clip(target_b - 1, 0, int(env_params.max_planets) - 1).astype(jnp.int32)
    target_planets = jnp.take_along_axis(state_b.planets, target_slot[..., None], axis=1)
    source_planets = state_b.planets
    ships = jnp.take_along_axis(rows.ship_buckets[None], amount_b[..., None], axis=2).squeeze(-1)
    target_valid = jnp.take_along_axis(rows.candidate_mask[None], target_b[..., None], axis=2).squeeze(-1)
    valid = fire_b & rows.source_mask[None] & target_valid & (target_b > 0) & (ships > 0.0)
    angle, _hit = aim_with_prediction_jax(source_planets, target_planets, target_slot, state_b, ships)
    actions = jnp.stack(
        [
            jnp.where(valid, rows.source_ids[None].astype(jnp.float32), 0.0),
            jnp.where(valid, angle, 0.0),
            jnp.where(valid, ships, 0.0),
        ],
        axis=-1,
    )
    return fit_action_slots(actions, int(env_params.max_actions))


def jax_actions_to_kaggle(actions_arr) -> list[list[float | int]]:
    arr = np.asarray(actions_arr[0])
    return [[float(row[0]), float(row[1]), int(row[2])] for row in arr if float(row[2]) > 0.0]


@dataclass
class BCPolicyAgent:
    model: FeatureRowBCPolicy
    params: Any
    normalizer: dict[str, jnp.ndarray]
    env_params: EnvParams
    feature_params: FeatureParams
    ship_bucket_count: int

    def __post_init__(self) -> None:
        self._act_jit = jax.jit(self._act_arrays)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, max_fleets: int = 512) -> "BCPolicyAgent":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        raw = payload.get("raw_config") or payload.get("config") or {}
        env_params = build_env_params_from_raw(raw, max_fleets=max_fleets)
        env_cfg = env_config_from_raw(raw)
        feature_params = FeatureParams.from_train_config(type("Cfg", (), {"env": env_cfg})())
        ship_bucket_count = int(_raw_get(raw, "env", "ship_bucket_count", 8))
        hidden_size = int(_raw_get(raw, "model", "hidden_size", 128))
        use_cnn = bool(_raw_get(raw, "model", "use_cnn", True))
        model = FeatureRowBCPolicy(ship_bucket_count=ship_bucket_count, hidden_size=hidden_size, use_cnn=use_cnn)
        normalizer = payload.get("normalizer") or identity_normalizer()
        return cls(
            model=model,
            params=jax.tree_util.tree_map(jnp.asarray, payload["params"]),
            normalizer=jax_normalizer(normalizer),
            env_params=env_params,
            feature_params=feature_params,
            ship_bucket_count=ship_bucket_count,
        )

    @classmethod
    def random_from_config(cls, raw: dict[str, Any] | None = None, *, seed: int = 0, max_fleets: int = 512) -> "BCPolicyAgent":
        raw = raw or {}
        env_params = build_env_params_from_raw(raw, max_fleets=max_fleets)
        env_cfg = env_config_from_raw(raw)
        feature_params = FeatureParams.from_train_config(type("Cfg", (), {"env": env_cfg})())
        ship_bucket_count = int(_raw_get(raw, "env", "ship_bucket_count", 8))
        hidden_size = int(_raw_get(raw, "model", "hidden_size", 128))
        use_cnn = bool(_raw_get(raw, "model", "use_cnn", True))
        model = FeatureRowBCPolicy(ship_bucket_count=ship_bucket_count, hidden_size=hidden_size, use_cnn=use_cnn)
        state0 = state_from_observation({"planets": [], "fleets": [], "step": 0}, env_params)
        rows = build_bc_policy_rows(state0, feature_params, ship_bucket_count)
        batch = normalize_rows(rows, jax_normalizer(identity_normalizer()))
        variables = model.init(
            jax.random.PRNGKey(int(seed)),
            batch.planet,
            batch.relative,
            batch.global_features,
            batch.candidate_mask,
            batch.map_features,
            jnp.zeros((batch.planet.shape[0],), dtype=jnp.int32),
        )
        return cls(
            model=model,
            params=variables["params"],
            normalizer=jax_normalizer(identity_normalizer()),
            env_params=env_params,
            feature_params=feature_params,
            ship_bucket_count=ship_bucket_count,
        )

    def _act_arrays(self, state: EnvState) -> jnp.ndarray:
        rows = build_bc_policy_rows(state, self.feature_params, self.ship_bucket_count)
        fire, target, amount = bc_decisions(self.model, self.params, rows, self.normalizer)
        return bc_actions_from_decisions(state, rows, fire, target, amount, self.env_params)

    def act(self, observation: Any, *, player: int | None = None) -> list[list[float | int]]:
        state = obs_to_bc_state(observation, self.env_params, player=player)
        actions = self._act_jit(state)
        return jax_actions_to_kaggle(jax.device_get(actions))
