"""Orbit Wars 策略网络（Flax 实现）。

架构说明：
- 三头输出：fire（是否发射）、target（指针注意力选星球）、amount（派兵档位）。
- legacy 路径保持原 per-source MLP + 单头 cross-attn，用于旧 checkpoint fallback。
- Transformer 路径使用 [CLS | planet tokens | fleet tokens] 编码全局关系，
  再用 source contextual embedding 对 planet tokens 做 pointer 选目标。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import flax.linen as nn
import jax
import jax.numpy as jnp


HIDDEN_INIT = nn.initializers.orthogonal(math.sqrt(2.0))
ACTOR_LOGIT_INIT = nn.initializers.orthogonal(0.01)
VALUE_OUT_INIT = nn.initializers.orthogonal(1.0)
BIAS_ZERO = nn.initializers.zeros


def hidden_dense(features: int) -> nn.Dense:
    return nn.Dense(features, kernel_init=HIDDEN_INIT, bias_init=BIAS_ZERO)


def actor_logit_dense(features: int) -> nn.Dense:
    return nn.Dense(features, kernel_init=ACTOR_LOGIT_INIT, bias_init=BIAS_ZERO)


def value_out_dense(features: int) -> nn.Dense:
    return nn.Dense(features, kernel_init=VALUE_OUT_INIT, bias_init=BIAS_ZERO)


@dataclass(frozen=True)
class PolicyOutput:
    """网络前向输出。"""
    fire_logits: jnp.ndarray    # [批量]：是否发射的 Bernoulli logit
    target_logits: jnp.ndarray  # [批量, P+1]：候选星球 attention 分数
    amount_logits: jnp.ndarray  # [批量, B]：派兵档位 logits
    value: jnp.ndarray          # [批量]：状态价值估计


class MLP(nn.Module):
    """两层 MLP + LayerNorm + ReLU，用于平行编码每个元素。"""
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = hidden_dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = hidden_dense(self.hidden_size)(x)
        return nn.relu(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder block for planet/fleet tokens."""
    hidden_size: int
    num_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, mask=None):
        h = nn.LayerNorm()(x)
        attn_mask = mask[:, None, None, :] if mask is not None else None
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout_rate=0.0,
            deterministic=True,
        )(h, h, mask=attn_mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = hidden_dense(self.hidden_size * self.mlp_ratio)(h)
        h = nn.gelu(h)
        h = hidden_dense(self.hidden_size)(h)
        return x + h


class ActorCritic(nn.Module):
    """主策略网络，封装 fire/target/amount/value 四个头。"""
    self_dim: int          # 来源星球特征维度 (SELF_DIM=18)
    candidate_dim: int     # 候选星球特征维度 (CANDIDATE_DIM=26)
    global_dim: int        # 全局局势特征维度 (GLOBAL_DIM=11)
    candidate_count: int   # 候选数量 = max_planets + 1 (slot 0 为 no-op)
    ship_bucket_count: int # 派兵档位数
    hidden_size: int = 128 # 隐层宽度
    use_cnn: bool = False  # 是否启用 CNN 小地图分支
    map_channels: int = 6  # CNN 输入通道数
    map_grid_size: int = 8 # CNN 输入分辨率
    use_transformer: bool = False
    fleet_dim: int = 16
    max_fleet_tokens: int = 32
    num_transformer_layers: int = 2
    num_heads: int = 4
    value_use_context: bool = True  # True=用 context_h（per-source，推荐）；False=用 cls_per_source（旧行为，兼容旧 ckpt）

    @nn.compact
    def __call__(
        self,
        self_features,
        candidate_features,
        global_features,
        candidate_mask,
        map_features=None,
        selected_targets=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        if self.use_transformer:
            if self_features.ndim == 3:
                return self._forward_transformer_grouped(
                    self_features,
                    candidate_features,
                    global_features,
                    candidate_mask,
                    map_features,
                    selected_targets,
                    fleet_features,
                    fleet_mask,
                )
            return self._forward_transformer(
                self_features,
                candidate_features,
                global_features,
                candidate_mask,
                map_features,
                selected_targets,
                fleet_features,
                fleet_mask,
            )
        return self._forward_legacy(
            self_features,
            candidate_features,
            global_features,
            candidate_mask,
            map_features,
            selected_targets,
        )

    def _forward_legacy(self, self_features, candidate_features, global_features, candidate_mask, map_features=None, selected_targets=None):
        # 将三类输入分别编码为等长 hidden 向量
        self_h = MLP(self.hidden_size)(self_features)
        cand_h = MLP(self.hidden_size)(candidate_features)
        global_h = MLP(self.hidden_size)(global_features)
        # 全图池化：对存活星球做 max/mean 混合池，得到全局感知
        # safe_mask 保证当没有星球时至少保留 slot 0 参与池化
        planet_mask = candidate_features[..., 1] > 0.5
        safe_mask = jnp.where(
            planet_mask.any(axis=-1, keepdims=True),
            planet_mask,
            jnp.arange(self.candidate_count) == 0,
        )
        masked = jnp.where(safe_mask[..., None], cand_h, jnp.finfo(jnp.float32).min)
        max_pool = masked.max(axis=-2)
        mean_pool = (cand_h * safe_mask[..., None]).sum(axis=-2) / jnp.maximum(
            safe_mask.sum(axis=-1, keepdims=True),
            1,
        )
        map_h = 0.5 * (max_pool + mean_pool)  # 全图池化结果
        context_parts = [self_h, global_h, map_h]
        if self.use_cnn:
            if map_features is None:
                map_features = jnp.zeros(
                    self_h.shape[:-1] + (self.map_channels, self.map_grid_size, self.map_grid_size),
                    dtype=self_h.dtype,
                )
            # Flax Conv expects NHWC.
            leading_shape = map_features.shape[:-3]
            x = map_features.reshape((-1,) + map_features.shape[-3:])
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
            x = x.mean(axis=(1, 2))
            x = nn.relu(hidden_dense(16)(x))
            x = x.reshape(leading_shape + (x.shape[-1],))
            context_parts.append(x)
        context_h = jnp.concatenate(context_parts, axis=-1)  # [B, hidden*3 (+ cnn_out)]

        # 单头 scaled dot-product cross-attention 实现目标选择
        # query = 来源星球 context 经非线性变换 + L2 归一化
        # 注意：用 sqrt(max(sum(x²), eps²)) 而非 max(norm(x), eps)。
        # 后者在 x=0 时反向传播会产生 0/0=NaN（norm 梯度 = x/norm(x)，
        # 即使 max 截断使前向为 eps，反向仍通过 d(norm)/dx=x/0 产生 NaN 梯度）。
        q = hidden_dense(self.hidden_size)(context_h)
        q = nn.relu(q)
        q = actor_logit_dense(self.hidden_size)(q)
        q = q / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q), axis=-1, keepdims=True), 1e-12))
        # key = 候选星球编码 + L2 归一化；slot 0 用可学习 no-op key 覆盖
        keys = actor_logit_dense(self.hidden_size)(cand_h)
        noop_key = self.param("noop_key", nn.initializers.zeros, (self.hidden_size,))
        keys = keys.at[..., 0, :].set(noop_key)
        keys = keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(keys), axis=-1, keepdims=True), 1e-12))
        # logit = q · key / sqrt(d)，非法目标置 -inf
        target_logits = jnp.einsum("...h,...ch->...c", q, keys) / jnp.sqrt(float(self.hidden_size))
        target_logits = jnp.where(candidate_mask, target_logits, -1.0e9)

        # fire 头：二分类逻辑回归
        fire_logits = hidden_dense(self.hidden_size)(context_h)
        fire_logits = nn.relu(fire_logits)
        fire_logits = actor_logit_dense(1)(fire_logits).squeeze(-1)

        # amount 头：拼接已选目标和 context，输出派兵档位 logits
        if selected_targets is None:
            selected_targets = jnp.zeros(self_h.shape[:-1], dtype=jnp.int32)
        selected_h = jnp.take_along_axis(cand_h, selected_targets[..., None, None], axis=-2).squeeze(-2)
        amount_h = jnp.concatenate([context_h, selected_h], axis=-1)
        amount_h = nn.relu(hidden_dense(self.hidden_size)(amount_h))
        amount_logits = actor_logit_dense(self.ship_bucket_count)(amount_h)

        # critic 头：当前状态价值
        value_h = nn.relu(hidden_dense(self.hidden_size)(context_h))
        value = value_out_dense(1)(value_h).squeeze(-1)
        return PolicyOutput(fire_logits, target_logits, amount_logits, value)

    def _forward_transformer(
        self,
        self_features,
        candidate_features,
        global_features,
        candidate_mask,
        map_features=None,
        selected_targets=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        batch_size = self_features.shape[0]
        hidden = self.hidden_size
        planet_h = MLP(hidden)(candidate_features)
        if fleet_features is None:
            fleet_features = jnp.zeros((batch_size, self.max_fleet_tokens, self.fleet_dim), dtype=planet_h.dtype)
        if fleet_mask is None:
            fleet_mask = jnp.zeros((batch_size, self.max_fleet_tokens), dtype=bool)
        fleet_h = MLP(hidden)(fleet_features)
        cls = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, hidden))
        cls = jnp.broadcast_to(cls, (batch_size, 1, hidden))
        type_embed = self.param("type_embed", nn.initializers.normal(0.02), (3, hidden))
        cls = cls + type_embed[0]
        planet_h = planet_h + type_embed[1]
        fleet_h = fleet_h + type_embed[2]
        tokens = jnp.concatenate([cls, planet_h, fleet_h], axis=1)
        cls_mask = jnp.ones((batch_size, 1), dtype=bool)
        planet_alive = candidate_features[..., 1] > 0.5
        planet_alive = planet_alive.at[:, 0].set(True)
        token_mask = jnp.concatenate([cls_mask, planet_alive, fleet_mask], axis=1)
        x = tokens
        for _ in range(self.num_transformer_layers):
            x = TransformerEncoderLayer(hidden, self.num_heads)(x, mask=token_mask)
        x = nn.LayerNorm()(x)
        cls_out = x[:, 0, :]
        planet_out = x[:, 1:1 + self.candidate_count, :]

        self_h = MLP(hidden)(self_features)
        global_h = MLP(hidden)(global_features)
        source_token_mask = candidate_features[..., 2:3] * (candidate_features[..., 1:2] > 0.5)
        source_ctx = (planet_out * source_token_mask).sum(axis=1) / jnp.maximum(source_token_mask.sum(axis=1), 1.0)
        context_parts = [self_h, global_h, source_ctx, cls_out]
        if self.use_cnn:
            if map_features is None:
                map_features = jnp.zeros((batch_size, self.map_channels, self.map_grid_size, self.map_grid_size), dtype=self_h.dtype)
            m = jnp.transpose(map_features, (0, 2, 3, 1))
            m = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(m))
            m = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(m))
            m = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(m))
            m = m.mean(axis=(1, 2))
            m = nn.relu(hidden_dense(16)(m))
            context_parts.append(m)
        context_h = jnp.concatenate(context_parts, axis=-1)

        q = hidden_dense(hidden)(context_h)
        q = nn.relu(q)
        q = actor_logit_dense(hidden)(q)
        q = q / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q), axis=-1, keepdims=True), 1e-12))
        keys = actor_logit_dense(hidden)(planet_out)
        noop_key = self.param("noop_key", nn.initializers.zeros, (hidden,))
        keys = keys.at[:, 0, :].set(noop_key)
        keys = keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(keys), axis=-1, keepdims=True), 1e-12))
        target_logits = jnp.einsum("bh,bch->bc", q, keys) / jnp.sqrt(float(hidden))
        target_logits = jnp.where(candidate_mask, target_logits, -1.0e9)

        fire_logits = hidden_dense(hidden)(context_h)
        fire_logits = nn.relu(fire_logits)
        fire_logits = actor_logit_dense(1)(fire_logits).squeeze(-1)

        if selected_targets is None:
            selected_targets = jnp.zeros((batch_size,), dtype=jnp.int32)
        row = jnp.arange(selected_targets.shape[0])
        selected_h = planet_out[row, selected_targets]
        amount_h = jnp.concatenate([context_h, selected_h], axis=-1)
        amount_h = nn.relu(hidden_dense(hidden)(amount_h))
        amount_logits = actor_logit_dense(self.ship_bucket_count)(amount_h)

        value_h = nn.relu(hidden_dense(hidden)(cls_out))
        value = value_out_dense(1)(value_h).squeeze(-1)
        return PolicyOutput(fire_logits, target_logits, amount_logits, value)

    def _forward_transformer_grouped(
        self,
        self_features,
        candidate_features,
        global_features,
        candidate_mask,
        map_features=None,
        selected_targets=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        batch_size, source_count = self_features.shape[:2]
        hidden = self.hidden_size
        planet_features = self._shared_planet_features(candidate_features[:, 0])
        planet_h = MLP(hidden)(planet_features)
        if fleet_features is None:
            fleet_features = jnp.zeros((batch_size, source_count, self.max_fleet_tokens, self.fleet_dim), dtype=planet_h.dtype)
        if fleet_mask is None:
            fleet_mask = jnp.zeros((batch_size, source_count, self.max_fleet_tokens), dtype=bool)
        shared_fleet_features = fleet_features[:, 0]
        shared_fleet_mask = fleet_mask[:, 0]
        fleet_h = MLP(hidden)(shared_fleet_features)
        cls = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, hidden))
        cls = jnp.broadcast_to(cls, (batch_size, 1, hidden))
        type_embed = self.param("type_embed", nn.initializers.normal(0.02), (3, hidden))
        cls = cls + type_embed[0]
        planet_h = planet_h + type_embed[1]
        fleet_h = fleet_h + type_embed[2]
        tokens = jnp.concatenate([cls, planet_h, fleet_h], axis=1)
        cls_mask = jnp.ones((batch_size, 1), dtype=bool)
        planet_alive = planet_features[..., 1] > 0.5
        planet_alive = planet_alive.at[:, 0].set(True)
        token_mask = jnp.concatenate([cls_mask, planet_alive, shared_fleet_mask], axis=1)
        x = tokens
        for _ in range(self.num_transformer_layers):
            x = TransformerEncoderLayer(hidden, self.num_heads)(x, mask=token_mask)
        x = nn.LayerNorm()(x)
        cls_out = x[:, 0, :]
        planet_out = x[:, 1:1 + self.candidate_count, :]

        self_h = MLP(hidden)(self_features)
        global_h = MLP(hidden)(global_features)
        source_token_mask = candidate_features[..., 2:3] * (candidate_features[..., 1:2] > 0.5)
        source_ctx = (planet_out[:, None, :, :] * source_token_mask).sum(axis=2) / jnp.maximum(source_token_mask.sum(axis=2), 1.0)
        cls_per_source = jnp.broadcast_to(cls_out[:, None, :], (batch_size, source_count, hidden))
        context_parts = [self_h, global_h, source_ctx, cls_per_source]
        if self.use_cnn:
            if map_features is None:
                map_features = jnp.zeros((batch_size, source_count, self.map_channels, self.map_grid_size, self.map_grid_size), dtype=self_h.dtype)
            m = jnp.reshape(map_features, (batch_size * source_count,) + map_features.shape[2:])
            m = jnp.transpose(m, (0, 2, 3, 1))
            m = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(m))
            m = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(m))
            m = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(m))
            m = m.mean(axis=(1, 2))
            m = nn.relu(hidden_dense(16)(m))
            m = jnp.reshape(m, (batch_size, source_count, 16))
            context_parts.append(m)
        context_h = jnp.concatenate(context_parts, axis=-1)

        relative_h = MLP(hidden)(self._relative_candidate_features(candidate_features))
        q = hidden_dense(hidden)(context_h)
        q = nn.relu(q)
        q = actor_logit_dense(hidden)(q)
        q = q / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q), axis=-1, keepdims=True), 1e-12))
        base_keys = actor_logit_dense(hidden)(planet_out)
        keys = base_keys[:, None, :, :] + relative_h
        noop_key = self.param("noop_key", nn.initializers.zeros, (hidden,))
        keys = keys.at[:, :, 0, :].set(noop_key)
        keys = keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(keys), axis=-1, keepdims=True), 1e-12))
        target_logits = jnp.einsum("bph,bpch->bpc", q, keys) / jnp.sqrt(float(hidden))
        target_logits = jnp.where(candidate_mask, target_logits, -1.0e9)

        fire_logits = hidden_dense(hidden)(context_h)
        fire_logits = nn.relu(fire_logits)
        fire_logits = actor_logit_dense(1)(fire_logits).squeeze(-1)

        if selected_targets is None:
            selected_targets = jnp.zeros((batch_size, source_count), dtype=jnp.int32)
        amount_keys = base_keys[:, None, :, :] + relative_h
        selected_h = jnp.take_along_axis(amount_keys, selected_targets[..., None, None], axis=2).squeeze(2)
        amount_h = jnp.concatenate([context_h, selected_h], axis=-1)
        amount_h = nn.relu(hidden_dense(hidden)(amount_h))
        amount_logits = actor_logit_dense(self.ship_bucket_count)(amount_h)

        # value_use_context=True（推荐）：用 context_h，per-source 独立，与 legacy 路径一致。
        # value_use_context=False（旧行为）：用 cls_per_source，同 batch 内所有 source 共享相同 value，
        #   param shape 更小，可兼容旧 checkpoint。
        value_input = context_h if self.value_use_context else cls_per_source
        value_h = nn.relu(hidden_dense(hidden)(value_input))
        value = value_out_dense(1)(value_h).squeeze(-1)
        return PolicyOutput(fire_logits, target_logits, amount_logits, value)

    def _shared_planet_features(self, candidate_features):
        """Strip source-relative fields before the state-level Transformer."""
        features = candidate_features
        relative_dims = jnp.asarray([2, 8, 9, 10, 16, 17, 24, 25], dtype=jnp.int32)
        features = features.at[..., relative_dims].set(0.0)
        alive = features[..., 1] > 0.5
        is_neutral = features[..., 3] > 0.5
        is_enemy = features[..., 5] > 0.5
        is_mine = alive & (~is_neutral) & (~is_enemy)
        features = features.at[..., 4].set(is_mine.astype(features.dtype))
        features = features.at[..., 0, 4].set(0.0)
        return features

    def _relative_candidate_features(self, candidate_features):
        """Keep only source-relative target features for per-source pointer keys."""
        keep_dims = jnp.asarray([2, 8, 9, 10, 16, 17, 24, 25], dtype=jnp.int32)
        out = jnp.zeros_like(candidate_features)
        return out.at[..., keep_dims].set(candidate_features[..., keep_dims])
