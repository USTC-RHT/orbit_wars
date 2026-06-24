"""JAX 在线特征归一化。

这里用于 RL 编码路径的 running normalizer：按有效来源星球/候选星球统计均值方差，
再把可连续化的维度标准化。布尔 one-hot、mask、no-op 标记等离散维度不参与归一化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from .features_jax import EncodedBatch, GLOBAL_DIM


# 只归一化连续数值维度；类别/布尔标记保持原值。
SELF_NORMALIZE_INDICES = jnp.asarray([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=jnp.int32)
CANDIDATE_NORMALIZE_INDICES = jnp.asarray([6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=jnp.int32)
GLOBAL_NORMALIZE_INDICES = jnp.arange(GLOBAL_DIM, dtype=jnp.int32)


@dataclass(frozen=True)
class NormalizerParams:
    """归一化运行参数。"""

    enabled: bool = True
    clip: float = 5.0  # z-score 截断范围，防止极端局势冲爆网络输入
    eps: float = 1e-8  # 方差保护项

    @classmethod
    def from_train_config(cls, cfg) -> "NormalizerParams":
        return cls(
            enabled=bool(cfg.state_norm.enabled),
            clip=float(cfg.state_norm.clip),
            eps=float(cfg.state_norm.eps),
        )


class NormalizerState(NamedTuple):
    """三类特征的 running mean/var/count。"""

    self_mean: jnp.ndarray       # 来源星球连续维度均值
    self_var: jnp.ndarray        # 来源星球连续维度方差
    self_count: jnp.ndarray      # 来源星球有效样本数
    candidate_mean: jnp.ndarray  # 真实候选星球连续维度均值（不含 no-op）
    candidate_var: jnp.ndarray   # 真实候选星球连续维度方差
    candidate_count: jnp.ndarray # 真实候选星球有效样本数
    global_mean: jnp.ndarray     # 全局特征均值
    global_var: jnp.ndarray      # 全局特征方差
    global_count: jnp.ndarray    # 全局特征有效样本数


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
    """用当前 batch 更新 running stats，并返回归一化后的 batch。"""

    if not params.enabled:
        return encoded, state

    # self 只统计有效我方来源星球，避免 padding 星球污染均值方差。
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

    # candidate 只统计真实星球候选；slot 0 no-op 和无效来源都跳过。
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
    """按 mask 计算最后一维特征的均值和方差。"""

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
    """并联合并历史 running moments 与当前 batch moments。"""

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
    """把 EncodedBatch 中选定连续维度替换成 clipped z-score。"""

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
        self_features,
        candidate_features,
        global_features,
        encoded.map_features,
        encoded.candidate_mask,
        encoded.amount_mask,
        encoded.source_ids,
        encoded.source_ships,
        encoded.candidate_ids,
        encoded.ship_buckets,
        encoded.source_mask,
    )


def normalize_values(values: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray, params: NormalizerParams) -> jnp.ndarray:
    return jnp.clip((values - mean) / jnp.sqrt(var + params.eps), -params.clip, params.clip)
