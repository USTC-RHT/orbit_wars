"""JAX 特征批次的运行时归一化。

训练时只对连续数值特征维护 running mean/var，mask、owner、是否存活等离散语义特征保持原值。
归一化状态会随 checkpoint 一起保存，评估和网页推理必须使用同一份统计量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from .features_jax import EncodedBatch, GLOBAL_DIM


# 只归一化连续特征列；布尔/类别列不参与 running statistics。
PLANET_NORMALIZE_INDICES = jnp.asarray([5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=jnp.int32)
RELATIVE_NORMALIZE_INDICES = jnp.asarray([1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)
GLOBAL_NORMALIZE_INDICES = jnp.arange(GLOBAL_DIM, dtype=jnp.int32)


@dataclass(frozen=True)
class NormalizerParams:
    """归一化开关和数值稳定参数。"""
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
    """三类特征各自维护 mean/var/count。"""
    planet_mean: jnp.ndarray
    planet_var: jnp.ndarray
    planet_count: jnp.ndarray
    relative_mean: jnp.ndarray
    relative_var: jnp.ndarray
    relative_count: jnp.ndarray
    global_mean: jnp.ndarray
    global_var: jnp.ndarray
    global_count: jnp.ndarray


def init_normalizer() -> NormalizerState:
    """创建空的归一化统计量，count 用很小正数避免除零。"""
    return NormalizerState(
        planet_mean=jnp.zeros((len(PLANET_NORMALIZE_INDICES),), dtype=jnp.float32),
        planet_var=jnp.ones((len(PLANET_NORMALIZE_INDICES),), dtype=jnp.float32),
        planet_count=jnp.asarray(1e-4, dtype=jnp.float32),
        relative_mean=jnp.zeros((len(RELATIVE_NORMALIZE_INDICES),), dtype=jnp.float32),
        relative_var=jnp.ones((len(RELATIVE_NORMALIZE_INDICES),), dtype=jnp.float32),
        relative_count=jnp.asarray(1e-4, dtype=jnp.float32),
        global_mean=jnp.zeros((len(GLOBAL_NORMALIZE_INDICES),), dtype=jnp.float32),
        global_var=jnp.ones((len(GLOBAL_NORMALIZE_INDICES),), dtype=jnp.float32),
        global_count=jnp.asarray(1e-4, dtype=jnp.float32),
    )


def update_and_normalize(encoded: EncodedBatch, state: NormalizerState, params: NormalizerParams) -> tuple[EncodedBatch, NormalizerState]:
    """用当前 batch 更新 running statistics，并返回归一化后的特征。"""
    if not params.enabled:
        return encoded, state

    source_mask = encoded.source_mask.astype(jnp.float32)
    planet_values = encoded.planet_features[..., PLANET_NORMALIZE_INDICES]
    planet_mean, planet_var, planet_count = masked_moments(planet_values, source_mask)
    new_planet_mean, new_planet_var, new_planet_count = update_moments(
        state.planet_mean,
        state.planet_var,
        state.planet_count,
        planet_mean,
        planet_var,
        planet_count,
    )

    real_planets = (encoded.candidate_ids >= 0).astype(jnp.float32) * source_mask[..., None]
    relative_values = encoded.relative_features[..., RELATIVE_NORMALIZE_INDICES]
    cand_mean, cand_var, cand_count = masked_moments(relative_values, real_planets)
    new_cand_mean, new_cand_var, new_cand_count = update_moments(
        state.relative_mean,
        state.relative_var,
        state.relative_count,
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
        new_planet_mean,
        new_planet_var,
        new_planet_count,
        new_cand_mean,
        new_cand_var,
        new_cand_count,
        new_global_mean,
        new_global_var,
        new_global_count,
    )
    return normalize_encoded(encoded, new_state, params), new_state


def masked_moments(values: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """在有效 mask 上计算 batch 均值、方差和样本数。"""
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
    """合并历史统计量和当前 batch 统计量，等价于并行版 Welford 更新。"""
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
    """按保存的 running statistics 归一化 EncodedBatch。"""
    planet_features = encoded.planet_features
    relative_features = encoded.relative_features
    global_features = encoded.global_features

    planet_norm = normalize_values(planet_features[..., PLANET_NORMALIZE_INDICES], state.planet_mean, state.planet_var, params)
    cand_norm = normalize_values(relative_features[..., RELATIVE_NORMALIZE_INDICES], state.relative_mean, state.relative_var, params)
    global_norm = normalize_values(global_features[..., GLOBAL_NORMALIZE_INDICES], state.global_mean, state.global_var, params)

    planet_features = planet_features.at[..., PLANET_NORMALIZE_INDICES].set(planet_norm)
    relative_features = relative_features.at[..., RELATIVE_NORMALIZE_INDICES].set(
        jnp.where((encoded.candidate_ids >= 0)[..., None], cand_norm, relative_features[..., RELATIVE_NORMALIZE_INDICES])
    )
    global_features = global_features.at[..., GLOBAL_NORMALIZE_INDICES].set(global_norm)
    return EncodedBatch(
        planet_features=planet_features,
        relative_features=relative_features,
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
    """标准化并裁剪到固定范围，防止极端输入破坏网络激活。"""
    return jnp.clip((values - mean) / jnp.sqrt(var + params.eps), -params.clip, params.clip)
