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


def hidden_dense(features: int, name: str | None = None) -> nn.Dense:
    return nn.Dense(features, kernel_init=HIDDEN_INIT, bias_init=BIAS_ZERO, name=name)


def actor_logit_dense(features: int, name: str | None = None) -> nn.Dense:
    return nn.Dense(features, kernel_init=ACTOR_LOGIT_INIT, bias_init=BIAS_ZERO, name=name)


def value_out_dense(features: int, name: str | None = None) -> nn.Dense:
    return nn.Dense(features, kernel_init=VALUE_OUT_INIT, bias_init=BIAS_ZERO, name=name)


@dataclass(frozen=True)
class PolicyOutput:
    """网络前向输出。"""
    fire_logits: jnp.ndarray    # [批量]：是否发射的 Bernoulli logit
    target_logits: jnp.ndarray  # [批量, P+1]：候选星球 attention 分数
    amount_logits: jnp.ndarray  # [批量, B]：派兵档位 logits
    value: jnp.ndarray          # [批量]：状态价值估计


@dataclass(frozen=True)
class AutoPolicyOutput:
    """Autoregressive actor forward outputs for a fixed action prefix."""
    end_logits: jnp.ndarray     # [B, K]
    src_logits: jnp.ndarray     # [B, K, P+1]
    tgt_logits: jnp.ndarray     # [B, K, P+1]
    amount_logits: jnp.ndarray  # [B, K, A]
    value: jnp.ndarray          # [B]


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
    planet_dim: int          # 星球局部特征维度
    relative_dim: int     # source-target 相对特征维度 (RELATIVE_DIM=8)
    global_dim: int        # 全局局势特征维度 (GLOBAL_DIM=11)
    relative_count: int   # 候选数量 = max_planets + 1 (slot 0 为 no-op)
    ship_bucket_count: int # 派兵档位数
    hidden_size: int = 128 # 隐层宽度
    use_cnn: bool = False  # 是否启用 CNN 小地图分支
    map_channels: int = 6  # CNN 输入通道数
    map_grid_size: int = 8 # CNN 输入分辨率
    use_transformer: bool = False
    fleet_dim: int = 28
    max_fleet_tokens: int = 32
    num_transformer_layers: int = 2
    num_heads: int = 4
    value_use_context: bool = True  # True=用 context_h（per-source，推荐）；False=用 cls_per_source（旧行为，兼容旧 ckpt）

    @nn.compact
    def __call__(
        self,
        planet_features,
        relative_features,
        global_features,
        candidate_mask,
        map_features=None,
        selected_targets=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        if self.use_transformer:
            if planet_features.ndim == 3:
                return self._forward_transformer_grouped(
                    planet_features,
                    relative_features,
                    global_features,
                    candidate_mask,
                    map_features,
                    selected_targets,
                    fleet_features,
                    fleet_mask,
                )
            return self._forward_transformer(
                planet_features,
                relative_features,
                global_features,
                candidate_mask,
                map_features,
                selected_targets,
                fleet_features,
                fleet_mask,
            )
        return self._forward_legacy(
            planet_features,
            relative_features,
            global_features,
            candidate_mask,
            map_features,
            selected_targets,
        )

    def _forward_legacy(self, planet_features, relative_features, global_features, candidate_mask, map_features=None, selected_targets=None):
        # 将三类输入分别编码为等长 hidden 向量
        self_h = MLP(self.hidden_size)(planet_features)
        cand_h = MLP(self.hidden_size)(relative_features)
        global_h = MLP(self.hidden_size)(global_features)
        # 全图池化：对存活星球做 max/mean 混合池，得到全局感知
        # safe_mask 保证当没有星球时至少保留 slot 0 参与池化
        safe_mask = candidate_mask
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
        planet_features,
        relative_features,
        global_features,
        candidate_mask,
        map_features=None,
        selected_targets=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        batch_size = planet_features.shape[0]
        hidden = self.hidden_size
        planet_h = MLP(hidden)(relative_features)
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
        planet_alive = candidate_mask
        planet_alive = planet_alive.at[:, 0].set(True)
        token_mask = jnp.concatenate([cls_mask, planet_alive, fleet_mask], axis=1)
        x = tokens
        for _ in range(self.num_transformer_layers):
            x = TransformerEncoderLayer(hidden, self.num_heads)(x, mask=token_mask)
        x = nn.LayerNorm()(x)
        cls_out = x[:, 0, :]
        planet_out = x[:, 1:1 + self.relative_count, :]

        self_h = MLP(hidden)(planet_features)
        global_h = MLP(hidden)(global_features)
        source_token_mask = relative_features[..., 0:1]
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
        planet_features,
        relative_features,
        global_features,
        candidate_mask,
        map_features=None,
        selected_targets=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        batch_size, source_count = planet_features.shape[:2]
        hidden = self.hidden_size
        transformer_planet_features = self._planet_features_with_noop(planet_features)
        planet_h = MLP(hidden)(transformer_planet_features)
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
        planet_alive = transformer_planet_features[..., 1] > 0.5
        planet_alive = planet_alive.at[:, 0].set(True)
        token_mask = jnp.concatenate([cls_mask, planet_alive, shared_fleet_mask], axis=1)
        x = tokens
        for _ in range(self.num_transformer_layers):
            x = TransformerEncoderLayer(hidden, self.num_heads)(x, mask=token_mask)
        x = nn.LayerNorm()(x)
        cls_out = x[:, 0, :]
        planet_out = x[:, 1:1 + self.relative_count, :]

        planet_h_per_source = MLP(hidden)(planet_features)
        global_h = MLP(hidden)(global_features)
        source_ctx = planet_out[:, 1:1 + source_count, :]
        cls_per_source = jnp.broadcast_to(cls_out[:, None, :], (batch_size, source_count, hidden))
        context_parts = [planet_h_per_source, global_h, source_ctx, cls_per_source]
        if self.use_cnn:
            if map_features is None:
                map_features = jnp.zeros((batch_size, source_count, self.map_channels, self.map_grid_size, self.map_grid_size), dtype=planet_h_per_source.dtype)
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

        relative_h = MLP(hidden)(relative_features)
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

    def _planet_features_with_noop(self, planet_features):
        noop = jnp.zeros(planet_features.shape[:1] + (1, planet_features.shape[-1]), dtype=planet_features.dtype)
        return jnp.concatenate([noop, planet_features], axis=1)


class AutoregressiveActorCritic(nn.Module):
    """State-level autoregressive actor.

    The module returns logits for all K positions conditioned on the provided
    rollout prefix. During rollout the caller repeatedly feeds the prefix,
    samples step t, validates physics, then marks `executed[t]` before decoding
    the next token. PPO recomputation uses the stored sequence in one call.
    """

    planet_dim: int
    relative_dim: int
    global_dim: int
    relative_count: int
    ship_bucket_count: int
    hidden_size: int = 128
    use_cnn: bool = False
    map_channels: int = 6
    map_grid_size: int = 8
    fleet_dim: int = 28
    max_fleet_tokens: int = 32
    num_transformer_layers: int = 2
    num_heads: int = 4
    max_action_tokens: int = 16
    end_bias_init: float = -2.0

    @nn.compact
    def __call__(
        self,
        planet_features,
        relative_features,
        global_features,
        candidate_mask,
        amount_mask,
        source_mask,
        end_actions,
        src_actions,
        tgt_actions,
        amt_actions,
        executed,
        initial_ships,
        map_features=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        batch_size, source_count = planet_features.shape[:2]
        hidden = self.hidden_size
        cls_out, planet_out, relative_h = self._encode(
            planet_features,
            relative_features,
            global_features,
            candidate_mask,
            map_features,
            fleet_features,
            fleet_mask,
        )
        global_h = MLP(hidden, name="auto_global_mlp")(global_features[:, 0])
        state_emb = nn.relu(hidden_dense(hidden, name="auto_state_emb")(jnp.concatenate([cls_out, global_h], axis=-1)))
        h0 = nn.tanh(hidden_dense(hidden, name="auto_init_h")(state_emb))
        bos = self.param("bos_token", nn.initializers.normal(0.02), (hidden,))
        end_tok = self.param("end_token", nn.initializers.normal(0.02), (hidden,))
        invalid_tok = self.param("invalid_token", nn.initializers.normal(0.02), (hidden,))
        amount_emb = self.param("amount_emb", nn.initializers.normal(0.02), (self.ship_bucket_count, hidden))
        step_emb = self.param("step_emb", nn.initializers.normal(0.02), (self.max_action_tokens, hidden))

        key_proj = actor_logit_dense(hidden, name="auto_key_proj")(planet_out)
        key_proj = key_proj / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(key_proj), axis=-1, keepdims=True), 1e-12))
        src_key_proj = self._source_key_proj(planet_out, map_features)
        decoder_cell = nn.GRUCell(features=hidden, name="decoder_gru")
        end_out = actor_logit_dense(1, name="auto_end_out")
        src_pre = hidden_dense(hidden, name="auto_src_pre")
        src_out = actor_logit_dense(hidden, name="auto_src_out")
        tgt_pre = hidden_dense(hidden, name="auto_tgt_pre")
        tgt_out = actor_logit_dense(hidden, name="auto_tgt_out")
        amt_pre = hidden_dense(hidden, name="auto_amt_pre")
        amt_out = actor_logit_dense(self.ship_bucket_count, name="auto_amt_out")
        end_bias = self.param("end_bias", nn.initializers.constant(self.end_bias_init), (1,))
        pointer_logit_scale = self.param(
            "pointer_logit_scale",
            nn.initializers.constant(math.log(10.0)),
            (1,),
        )
        pointer_logit_scale = jnp.clip(jnp.exp(pointer_logit_scale[0]), 1.0, 20.0)

        batch_idx = jnp.arange(batch_size)

        def token_embed(src_t, tgt_t, amt_t, end_t, executed_t, step_i):
            src_h = planet_out[batch_idx, jnp.clip(src_t, 0, self.relative_count - 1)]
            tgt_h = planet_out[batch_idx, jnp.clip(tgt_t, 0, self.relative_count - 1)]
            raw = src_h + tgt_h + amount_emb[jnp.clip(amt_t, 0, self.ship_bucket_count - 1)] + step_emb[step_i]
            raw = raw / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(raw), axis=-1, keepdims=True), 1e-12))
            return jnp.where(
                end_t[:, None],
                end_tok[None, :],
                jnp.where(executed_t[:, None], raw, invalid_tok[None, :]),
            )

        def decode_step(carry, item):
            h, prev_tok, remaining = carry
            step_i, end_t, src_t, tgt_t, amt_t, executed_t = item
            decoder_input = jnp.concatenate([prev_tok, state_emb], axis=-1)
            h_new, _ = decoder_cell(h, decoder_input)

            end_logits = end_out(h_new).squeeze(-1)
            # Set the initial bias through a named parameter so it can be
            # monitored and adjusted independently.
            end_logits = end_logits + end_bias[0]

            q_src = src_out(nn.relu(src_pre(h_new)))
            q_src = q_src / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q_src), axis=-1, keepdims=True), 1e-12))
            src_logits = jnp.einsum("bh,bch->bc", q_src, src_key_proj) * pointer_logit_scale
            source_mine = jnp.concatenate(
                [jnp.zeros((batch_size, 1), dtype=bool), source_mask.astype(bool)],
                axis=1,
            )
            src_budget = remaining >= 10.0
            src_mask = source_mine & src_budget
            src_logits = jnp.where(src_mask, src_logits, -1.0e9)
            src_logits = src_logits.at[:, 0].set(-1.0e9)

            src_safe = jnp.clip(src_t, 0, self.relative_count - 1)
            src_h = planet_out[batch_idx, src_safe]
            rel_for_src = relative_h[batch_idx, jnp.clip(src_safe - 1, 0, source_count - 1)]
            q_tgt = tgt_out(nn.relu(tgt_pre(h_new + src_h)))
            q_tgt = q_tgt / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q_tgt), axis=-1, keepdims=True), 1e-12))
            tgt_keys = key_proj + rel_for_src
            tgt_keys = tgt_keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(tgt_keys), axis=-1, keepdims=True), 1e-12))
            tgt_logits = jnp.einsum("bh,bch->bc", q_tgt, tgt_keys) * pointer_logit_scale
            src_candidate_mask = candidate_mask[batch_idx, jnp.clip(src_safe - 1, 0, source_count - 1)]
            tgt_mask = src_candidate_mask.at[:, 0].set(False)
            tgt_logits = jnp.where(tgt_mask, tgt_logits, -1.0e9)

            tgt_safe = jnp.clip(tgt_t, 0, self.relative_count - 1)
            tgt_h = planet_out[batch_idx, tgt_safe]
            selected_rel_h = jnp.take_along_axis(rel_for_src, tgt_safe[:, None, None], axis=1).squeeze(1)
            amt_h = nn.relu(amt_pre(jnp.concatenate([h_new, src_h, tgt_h, selected_rel_h], axis=-1)))
            amount_logits = amt_out(amt_h)
            source_amount_mask = amount_mask[batch_idx, jnp.clip(src_safe - 1, 0, source_count - 1)]
            selected_amount_mask = jnp.take_along_axis(source_amount_mask, tgt_safe[:, None, None], axis=1).squeeze(1)
            dynamic_amount_mask = self._dynamic_amount_mask(remaining[batch_idx, src_safe])
            selected_amount_mask = selected_amount_mask & dynamic_amount_mask
            amount_logits = jnp.where(selected_amount_mask, amount_logits, -1.0e9)

            ships_sent = self._ships_for_bucket(remaining[batch_idx, src_safe], amt_t)
            remaining = remaining.at[batch_idx, src_safe].add(jnp.where(executed_t, -ships_sent, 0.0))
            next_tok = token_embed(src_safe, tgt_safe, amt_t, end_t, executed_t, step_i)
            return (h_new, next_tok, remaining), (end_logits, src_logits, tgt_logits, amount_logits)

        init = (h0, jnp.broadcast_to(bos[None, :], (batch_size, hidden)), initial_ships)
        carry = init
        end_logs = []
        src_logs = []
        tgt_logs = []
        amt_logs = []
        for step_i in range(self.max_action_tokens):
            carry, step_out = decode_step(
                carry,
                (
                    jnp.asarray(step_i, dtype=jnp.int32),
                    end_actions[:, step_i],
                    src_actions[:, step_i],
                    tgt_actions[:, step_i],
                    amt_actions[:, step_i],
                    executed[:, step_i],
                ),
            )
            e, s, t, a = step_out
            end_logs.append(e)
            src_logs.append(s)
            tgt_logs.append(t)
            amt_logs.append(a)
        end_logits = jnp.stack(end_logs, axis=0)
        src_logits = jnp.stack(src_logs, axis=0)
        tgt_logits = jnp.stack(tgt_logs, axis=0)
        amount_logits = jnp.stack(amt_logs, axis=0)
        value_h = nn.relu(hidden_dense(hidden, name="auto_value_h")(jnp.concatenate([cls_out, global_h], axis=-1)))
        value = value_out_dense(1, name="auto_value_out")(value_h).squeeze(-1)
        return AutoPolicyOutput(
            jnp.swapaxes(end_logits, 0, 1),
            jnp.swapaxes(src_logits, 0, 1),
            jnp.swapaxes(tgt_logits, 0, 1),
            jnp.swapaxes(amount_logits, 0, 1),
            value,
        )

    @nn.compact
    def auto_start(
        self,
        planet_features,
        relative_features,
        global_features,
        candidate_mask,
        initial_ships,
        map_features=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        """Encode state once and initialize the incremental GRU decoder."""
        hidden = self.hidden_size
        cls_out, planet_out, relative_h = self._encode(
            planet_features,
            relative_features,
            global_features,
            candidate_mask,
            map_features,
            fleet_features,
            fleet_mask,
        )
        global_h = MLP(hidden, name="auto_global_mlp")(global_features[:, 0])
        state_emb = nn.relu(hidden_dense(hidden, name="auto_state_emb")(jnp.concatenate([cls_out, global_h], axis=-1)))
        h = nn.tanh(hidden_dense(hidden, name="auto_init_h")(state_emb))
        bos = self.param("bos_token", nn.initializers.normal(0.02), (hidden,))
        prev_tok = jnp.broadcast_to(bos[None, :], (planet_features.shape[0], hidden))
        key_proj = actor_logit_dense(hidden, name="auto_key_proj")(planet_out)
        key_proj = key_proj / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(key_proj), axis=-1, keepdims=True), 1e-12))
        src_key_proj = self._source_key_proj(planet_out, map_features)
        value_h = nn.relu(hidden_dense(hidden, name="auto_value_h")(jnp.concatenate([cls_out, global_h], axis=-1)))
        value = value_out_dense(1, name="auto_value_out")(value_h).squeeze(-1)
        return state_emb, h, prev_tok, initial_ships, planet_out, relative_h, key_proj, src_key_proj, value

    @nn.compact
    def auto_step_base(self, state_emb, h, prev_tok, remaining, relative_features, source_mask, src_key_proj):
        """Run one GRU step and produce END/SRC logits."""
        hidden = self.hidden_size
        decoder_cell = nn.GRUCell(features=hidden, name="decoder_gru")
        decoder_input = jnp.concatenate([prev_tok, state_emb], axis=-1)
        h_new, _ = decoder_cell(h, decoder_input)
        end_logits = actor_logit_dense(1, name="auto_end_out")(h_new).squeeze(-1)
        end_bias = self.param("end_bias", nn.initializers.constant(self.end_bias_init), (1,))
        end_logits = end_logits + end_bias[0]
        q_src = actor_logit_dense(hidden, name="auto_src_out")(nn.relu(hidden_dense(hidden, name="auto_src_pre")(h_new)))
        q_src = q_src / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q_src), axis=-1, keepdims=True), 1e-12))
        pointer_logit_scale = self._pointer_logit_scale()
        src_logits = jnp.einsum("bh,bch->bc", q_src, src_key_proj) * pointer_logit_scale
        source_mine = jnp.concatenate(
            [jnp.zeros((h_new.shape[0], 1), dtype=bool), source_mask.astype(bool)],
            axis=1,
        )
        src_budget = remaining >= 10.0
        src_mask = source_mine & src_budget
        src_logits = jnp.where(src_mask, src_logits, -1.0e9)
        src_logits = src_logits.at[:, 0].set(-1.0e9)
        return h_new, end_logits, src_logits

    @nn.compact
    def auto_tgt_logits(self, h_new, src_t, planet_out, relative_h, key_proj, candidate_mask):
        """Produce target logits for the current sampled source."""
        hidden = self.hidden_size
        batch_idx = jnp.arange(h_new.shape[0])
        source_count = relative_h.shape[1]
        src_safe = jnp.clip(src_t, 0, self.relative_count - 1)
        src_h = planet_out[batch_idx, src_safe]
        rel_for_src = relative_h[batch_idx, jnp.clip(src_safe - 1, 0, source_count - 1)]
        q_tgt = actor_logit_dense(hidden, name="auto_tgt_out")(nn.relu(hidden_dense(hidden, name="auto_tgt_pre")(h_new + src_h)))
        q_tgt = q_tgt / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q_tgt), axis=-1, keepdims=True), 1e-12))
        tgt_keys = key_proj + rel_for_src
        tgt_keys = tgt_keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(tgt_keys), axis=-1, keepdims=True), 1e-12))
        tgt_logits = jnp.einsum("bh,bch->bc", q_tgt, tgt_keys) * self._pointer_logit_scale()
        src_candidate_mask = candidate_mask[batch_idx, jnp.clip(src_safe - 1, 0, source_count - 1)]
        tgt_mask = src_candidate_mask.at[:, 0].set(False)
        return jnp.where(tgt_mask, tgt_logits, -1.0e9)

    @nn.compact
    def auto_amt_logits(self, h_new, src_t, tgt_t, planet_out, relative_h, remaining, amount_mask):
        """Produce amount logits for the current sampled source/target pair."""
        hidden = self.hidden_size
        batch_idx = jnp.arange(h_new.shape[0])
        source_count = relative_h.shape[1]
        src_safe = jnp.clip(src_t, 0, self.relative_count - 1)
        tgt_safe = jnp.clip(tgt_t, 0, self.relative_count - 1)
        src_h = planet_out[batch_idx, src_safe]
        tgt_h = planet_out[batch_idx, tgt_safe]
        rel_for_src = relative_h[batch_idx, jnp.clip(src_safe - 1, 0, source_count - 1)]
        selected_rel_h = jnp.take_along_axis(rel_for_src, tgt_safe[:, None, None], axis=1).squeeze(1)
        amt_h = nn.relu(hidden_dense(hidden, name="auto_amt_pre")(jnp.concatenate([h_new, src_h, tgt_h, selected_rel_h], axis=-1)))
        amount_logits = actor_logit_dense(self.ship_bucket_count, name="auto_amt_out")(amt_h)
        source_amount_mask = amount_mask[batch_idx, jnp.clip(src_safe - 1, 0, amount_mask.shape[1] - 1)]
        selected_amount_mask = jnp.take_along_axis(source_amount_mask, tgt_safe[:, None, None], axis=1).squeeze(1)
        dynamic_amount_mask = self._dynamic_amount_mask(remaining[batch_idx, src_safe])
        selected_amount_mask = selected_amount_mask & dynamic_amount_mask
        return jnp.where(selected_amount_mask, amount_logits, -1.0e9)

    @nn.compact
    def auto_token_embed(self, src_t, tgt_t, amt_t, end_t, executed_t, step_i, planet_out):
        """Build the token fed into the next GRU step."""
        hidden = self.hidden_size
        batch_idx = jnp.arange(src_t.shape[0])
        end_tok = self.param("end_token", nn.initializers.normal(0.02), (hidden,))
        invalid_tok = self.param("invalid_token", nn.initializers.normal(0.02), (hidden,))
        amount_emb = self.param("amount_emb", nn.initializers.normal(0.02), (self.ship_bucket_count, hidden))
        step_emb = self.param("step_emb", nn.initializers.normal(0.02), (self.max_action_tokens, hidden))
        src_safe = jnp.clip(src_t, 0, self.relative_count - 1)
        tgt_safe = jnp.clip(tgt_t, 0, self.relative_count - 1)
        src_h = planet_out[batch_idx, src_safe]
        tgt_h = planet_out[batch_idx, tgt_safe]
        raw = src_h + tgt_h + amount_emb[jnp.clip(amt_t, 0, self.ship_bucket_count - 1)] + step_emb[step_i]
        raw = raw / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(raw), axis=-1, keepdims=True), 1e-12))
        return jnp.where(
            end_t[:, None],
            end_tok[None, :],
            jnp.where(executed_t[:, None], raw, invalid_tok[None, :]),
        )

    def _pointer_logit_scale(self):
        pointer_logit_scale = self.param(
            "pointer_logit_scale",
            nn.initializers.constant(math.log(10.0)),
            (1,),
        )
        return jnp.clip(jnp.exp(pointer_logit_scale[0]), 1.0, 20.0)

    def _source_key_proj(self, planet_out, map_features):
        hidden = self.hidden_size
        batch_size = planet_out.shape[0]
        planet_keys = actor_logit_dense(hidden, name="auto_src_key_proj")(planet_out)
        if self.use_cnn:
            if map_features is None:
                source_count = self.relative_count - 1
                map_features = jnp.zeros(
                    (batch_size, source_count, self.map_channels, self.map_grid_size, self.map_grid_size),
                    dtype=planet_out.dtype,
                )
            source_count = map_features.shape[1]
            m = jnp.reshape(map_features, (batch_size * source_count,) + map_features.shape[2:])
            m = jnp.transpose(m, (0, 2, 3, 1))
            m = nn.relu(nn.Conv(16, (3, 3), padding="SAME", name="auto_src_map_conv1")(m))
            m = nn.relu(nn.Conv(32, (3, 3), padding="SAME", name="auto_src_map_conv2")(m))
            m = nn.relu(nn.Conv(16, (3, 3), padding="SAME", name="auto_src_map_conv3")(m))
            m = m.mean(axis=(1, 2))
            map_h = nn.relu(hidden_dense(hidden, name="auto_src_map_h")(m))
            map_h = jnp.reshape(map_h, (batch_size, source_count, hidden))
            src_planet_h = planet_out[:, 1:1 + source_count]
            src_keys = actor_logit_dense(hidden, name="auto_src_map_key")(
                nn.relu(hidden_dense(hidden, name="auto_src_map_fuse")(jnp.concatenate([src_planet_h, map_h], axis=-1)))
            )
            planet_keys = planet_keys.at[:, 1:1 + source_count, :].set(src_keys)
        return planet_keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(planet_keys), axis=-1, keepdims=True), 1e-12))

    def _encode(self, planet_features, relative_features, global_features, candidate_mask, map_features, fleet_features, fleet_mask):
        batch_size, source_count = planet_features.shape[:2]
        hidden = self.hidden_size
        transformer_planet_features = self._planet_features_with_noop(planet_features)
        planet_h = MLP(hidden)(transformer_planet_features)
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
        tokens = jnp.concatenate([cls + type_embed[0], planet_h + type_embed[1], fleet_h + type_embed[2]], axis=1)
        cls_mask = jnp.ones((batch_size, 1), dtype=bool)
        planet_alive = transformer_planet_features[..., 1] > 0.5
        planet_alive = planet_alive.at[:, 0].set(True)
        token_mask = jnp.concatenate([cls_mask, planet_alive, shared_fleet_mask], axis=1)
        x = tokens
        for _ in range(self.num_transformer_layers):
            x = TransformerEncoderLayer(hidden, self.num_heads)(x, mask=token_mask)
        x = nn.LayerNorm()(x)
        cls_out = x[:, 0, :]
        planet_out = x[:, 1:1 + self.relative_count, :]
        relative_h = MLP(hidden)(relative_features)
        return cls_out, planet_out, relative_h

    def _ships_for_bucket(self, ships, bucket):
        buckets = self._bucket_values(ships)
        return jnp.take_along_axis(buckets, jnp.clip(bucket, 0, self.ship_bucket_count - 1)[:, None], axis=-1).squeeze(-1)

    def _bucket_values(self, ships):
        ratios = jnp.asarray([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=jnp.float32)
        ratios = ratios[: self.ship_bucket_count]
        s = jnp.floor(ships)
        raw = jnp.clip(jnp.round(s[:, None] * ratios[None, :]), 10.0, s[:, None])
        return jnp.where(s[:, None] >= 10.0, raw, jnp.zeros_like(raw))

    def _dynamic_amount_mask(self, ships):
        buckets = self._bucket_values(ships)
        positive = buckets >= 10.0
        return positive

    def _planet_features_with_noop(self, planet_features):
        noop = jnp.zeros(planet_features.shape[:1] + (1, planet_features.shape[-1]), dtype=planet_features.dtype)
        return jnp.concatenate([noop, planet_features], axis=1)
