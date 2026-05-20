"""Running state normalization for JAX feature batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from .features_jax import EncodedBatch, GLOBAL_DIM


SELF_NORMALIZE_INDICES = jnp.asarray([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=jnp.int32)
CANDIDATE_NORMALIZE_INDICES = jnp.asarray([6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=jnp.int32)
GLOBAL_NORMALIZE_INDICES = jnp.arange(GLOBAL_DIM, dtype=jnp.int32)


@dataclass(frozen=True)
class NormalizerParams:
    enabled: bool = True
    clip: float = 5.0
    eps: float = 1e-8

    @classmethod
    def from_train_config(cls, cfg) -> "NormalizerParams":
        return cls(
            enabled=bool(cfg.state_norm.enabled),
            clip=float(cfg.state_norm.clip),
            eps=float(cfg.state_norm.eps),
        )


class NormalizerState(NamedTuple):
    self_mean: jnp.ndarray
    self_var: jnp.ndarray
    self_count: jnp.ndarray
    candidate_mean: jnp.ndarray
    candidate_var: jnp.ndarray
    candidate_count: jnp.ndarray
    global_mean: jnp.ndarray
    global_var: jnp.ndarray
    global_count: jnp.ndarray


def init_normalizer() -> NormalizerState:
    return NormalizerState(
        self_mean=jnp.zeros((len(SELF_NORMALIZE_INDICES),), dtype=jnp.float32),
        self_var=jnp.ones((len(SELF_NORMALIZE_INDICES),), dtype=jnp.float32),
        self_count=jnp.asarray(1e-4, dtype=jnp.float32),
        candidate_mean=jnp.zeros((len(CANDIDATE_NORMALIZE_INDICES),), dtype=jnp.float32),
        candidate_var=jnp.ones((len(CANDIDATE_NORMALIZE_INDICES),), dtype=jnp.float32),
        candidate_count=jnp.asarray(1e-4, dtype=jnp.float32),
        global_mean=jnp.zeros((len(GLOBAL_NORMALIZE_INDICES),), dtype=jnp.float32),
        global_var=jnp.ones((len(GLOBAL_NORMALIZE_INDICES),), dtype=jnp.float32),
        global_count=jnp.asarray(1e-4, dtype=jnp.float32),
    )


def update_and_normalize(encoded: EncodedBatch, state: NormalizerState, params: NormalizerParams) -> tuple[EncodedBatch, NormalizerState]:
    if not params.enabled:
        return encoded, state

    source_mask = encoded.source_mask.astype(jnp.float32)
    self_values = encoded.self_features[..., SELF_NORMALIZE_INDICES]
    self_mean, self_var, self_count = masked_moments(self_values, source_mask)
    new_self_mean, new_self_var, new_self_count = update_moments(
        state.self_mean,
        state.self_var,
        state.self_count,
        self_mean,
        self_var,
        self_count,
    )

    real_planets = (encoded.candidate_features[..., 1] > 0.5).astype(jnp.float32) * source_mask[..., None]
    candidate_values = encoded.candidate_features[..., CANDIDATE_NORMALIZE_INDICES]
    cand_mean, cand_var, cand_count = masked_moments(candidate_values, real_planets)
    new_cand_mean, new_cand_var, new_cand_count = update_moments(
        state.candidate_mean,
        state.candidate_var,
        state.candidate_count,
        cand_mean,
        cand_var,
        cand_count,
    )

    global_values = encoded.global_features[..., GLOBAL_NORMALIZE_INDICES]
    global_mean, global_var, global_count = masked_moments(global_values, source_mask)
    new_global_mean, new_global_var, new_global_count = update_moments(
        state.global_mean,
        state.global_var,
        state.global_count,
        global_mean,
        global_var,
        global_count,
    )

    new_state = NormalizerState(
        new_self_mean,
        new_self_var,
        new_self_count,
        new_cand_mean,
        new_cand_var,
        new_cand_count,
        new_global_mean,
        new_global_var,
        new_global_count,
    )
    return normalize_encoded(encoded, new_state, params), new_state


def masked_moments(values: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mask = mask.astype(jnp.float32)
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    count = mask.sum()
    safe_count = jnp.maximum(count, 1.0)
    mean = (values * mask).sum(axis=tuple(range(values.ndim - 1))) / safe_count
    var = (jnp.square(values - mean) * mask).sum(axis=tuple(range(values.ndim - 1))) / safe_count
    mean = jnp.where(count > 0.0, mean, jnp.zeros_like(mean))
    var = jnp.where(count > 0.0, var, jnp.ones_like(var))
    return mean, var, count


def update_moments(mean, var, count, batch_mean, batch_var, batch_count):
    total = count + batch_count
    delta = batch_mean - mean
    safe_total = jnp.maximum(total, 1e-8)
    new_mean = mean + delta * batch_count / safe_total
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * count * batch_count / safe_total
    new_var = m2 / safe_total
    return jnp.where(batch_count > 0.0, new_mean, mean), jnp.where(batch_count > 0.0, new_var, var), total


def normalize_encoded(encoded: EncodedBatch, state: NormalizerState, params: NormalizerParams) -> EncodedBatch:
    self_features = encoded.self_features
    candidate_features = encoded.candidate_features
    global_features = encoded.global_features

    self_norm = normalize_values(self_features[..., SELF_NORMALIZE_INDICES], state.self_mean, state.self_var, params)
    cand_norm = normalize_values(candidate_features[..., CANDIDATE_NORMALIZE_INDICES], state.candidate_mean, state.candidate_var, params)
    global_norm = normalize_values(global_features[..., GLOBAL_NORMALIZE_INDICES], state.global_mean, state.global_var, params)

    self_features = self_features.at[..., SELF_NORMALIZE_INDICES].set(self_norm)
    candidate_features = candidate_features.at[..., CANDIDATE_NORMALIZE_INDICES].set(
        jnp.where((candidate_features[..., 1] > 0.5)[..., None], cand_norm, candidate_features[..., CANDIDATE_NORMALIZE_INDICES])
    )
    global_features = global_features.at[..., GLOBAL_NORMALIZE_INDICES].set(global_norm)
    return EncodedBatch(
        self_features=self_features,
        candidate_features=candidate_features,
        global_features=global_features,
        map_features=encoded.map_features,
        candidate_mask=encoded.candidate_mask,
        amount_mask=encoded.amount_mask,
        source_ids=encoded.source_ids,
        source_ships=encoded.source_ships,
        candidate_ids=encoded.candidate_ids,
        ship_buckets=encoded.ship_buckets,
        source_mask=encoded.source_mask,
        fleet_features=encoded.fleet_features,
        fleet_mask=encoded.fleet_mask,
    )


def normalize_values(values: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray, params: NormalizerParams) -> jnp.ndarray:
    return jnp.clip((values - mean) / jnp.sqrt(var + params.eps), -params.clip, params.clip)
