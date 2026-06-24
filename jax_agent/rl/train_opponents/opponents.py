"""JAX 原生训练对手。

这些函数会在 JIT 编译的 rollout 路径里调用，因此不能依赖 Python/Kaggle 外部 agent。
训练对手必须能表示成 JAX 数组程序，rollout 才能保持高吞吐。
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ...common.features_jax import FeatureParams, encode_batch
from ...common.normalization_jax import normalize_encoded

MIN_LAUNCH_LOCAL = 10.0


def nearest_opponent_actions(state, player: int, params: FeatureParams):
    """最近目标规则基线：每个来源星球向最近非己方目标发射。"""

    return jax.vmap(lambda planets, alive: _nearest_opponent_single(planets, alive, player, params))(
        state.planets,
        state.planet_alive,
    )


def _nearest_opponent_single(planets, alive, player: int, params: FeatureParams):
    owner = planets[:, 1].astype(jnp.int32)
    sources = alive & (owner == player)
    targets = alive & (owner != player)
    dx = planets[None, :, 2] - planets[:, None, 2]
    dy = planets[None, :, 3] - planets[:, None, 3]
    dist = jnp.sqrt(dx * dx + dy * dy)
    masked_dist = jnp.where(targets[None, :] & (~jnp.eye(params.max_planets, dtype=bool)), dist, 1e9)
    tgt_idx = jnp.argmin(masked_dist, axis=1)
    tgt = planets[tgt_idx]
    src = planets
    ships = jnp.minimum(src[:, 5], jnp.maximum(tgt[:, 5] + 1.0, jnp.floor(src[:, 5] * 0.5)))
    valid = sources & (src[:, 5] >= MIN_LAUNCH_LOCAL) & (masked_dist.min(axis=1) < 1e8)
    angle = jnp.arctan2(tgt[:, 3] - src[:, 3], tgt[:, 2] - src[:, 2])
    return jnp.stack(
        [
            jnp.where(valid, src[:, 0], 0.0),
            jnp.where(valid, angle, 0.0),
            jnp.where(valid, ships, 0.0),
        ],
        axis=-1,
    )


def selfplay_policy_actions(
    state,
    opponent_params,
    model,
    normalizer_state,
    feat_params: FeatureParams,
    norm_params,
    rng,
    *,
    flip_state_fn: Callable,
    policy_sample_params_fn: Callable,
    actions_from_decisions_fn: Callable,
):
    """冻结策略自博弈对手；当前自回归主线已使用 train_jax.py 内的新版实现。"""

    flipped = flip_state_fn(state)
    raw_encoded = encode_batch(flipped, feat_params)
    encoded = normalize_encoded(raw_encoded, normalizer_state, norm_params)
    fire, target, amount, _, _ = policy_sample_params_fn(opponent_params, model, encoded, rng)
    return actions_from_decisions_fn(flipped, encoded, fire, target, amount, feat_params)
