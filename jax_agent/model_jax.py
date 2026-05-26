"""Orbit Wars 自回归策略网络（Flax 实现）。

整体结构：
1. Encoder：把星球、舰队和全局信息编码成 Transformer token。
2. Decoder：GRU 按 token 逐步生成 end/src/tgt/amount。
3. Critic：从全局状态表示输出状态价值，用于 PPO 的 value loss。
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
    """隐藏层 Dense：使用 orthogonal 初始化，适合 PPO 稳定训练。"""
    return nn.Dense(features, kernel_init=HIDDEN_INIT, bias_init=BIAS_ZERO, name=name)


def actor_logit_dense(features: int, name: str | None = None) -> nn.Dense:
    """策略 logits 输出层：较小初始化，让初始策略更接近均匀探索。"""
    return nn.Dense(features, kernel_init=ACTOR_LOGIT_INIT, bias_init=BIAS_ZERO, name=name)


def value_out_dense(features: int, name: str | None = None) -> nn.Dense:
    """价值函数输出层初始化。"""
    return nn.Dense(features, kernel_init=VALUE_OUT_INIT, bias_init=BIAS_ZERO, name=name)


@dataclass(frozen=True)
class AutoPolicyOutput:
    """固定动作前缀下，所有 decoder 位置的策略 logits 和 value。"""
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
    """前置 LayerNorm 的 Transformer 编码块，用于星球/舰队 token 交互。"""
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


class AutoregressiveActorCritic(nn.Module):
    """状态级自回归 actor-critic。

    前向时给定已经采样出的动作前缀，模型一次性返回 K 个位置的 logits。
    rollout 时外层代码逐 token 采样、做物理层校验，并把通过校验的动作标记为 executed。
    PPO 更新时使用存下来的整段序列一次性重算 log-prob。
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
        target_action_mask=None,
        amount_action_mask=None,
        map_features=None,
        fleet_features=None,
        fleet_mask=None,
    ):
        """重算一整段动作序列的 logits，供 PPO update 使用。"""
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
            # end head 的初始偏置单独作为参数，便于监控和调整结束倾向。
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
            if target_action_mask is not None:
                tgt_mask = tgt_mask & target_action_mask[:, step_i]
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
            if amount_action_mask is not None:
                selected_amount_mask = selected_amount_mask & amount_action_mask[:, step_i]
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
        """只编码一次棋盘状态，并初始化增量式 GRU decoder。"""
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
        """运行一步 GRU，并输出 end/src logits。"""
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
        """在给定 src 后输出目标星球 logits。"""
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
        """在给定 src+tgt 后输出派兵档位 logits，并应用动态剩余兵力约束。"""
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
        """把当前 token 的执行结果编码成下一步 GRU 输入。"""
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
        """可学习的 pointer logit 温度，限制范围避免注意力过尖或过平。"""
        pointer_logit_scale = self.param(
            "pointer_logit_scale",
            nn.initializers.constant(math.log(10.0)),
            (1,),
        )
        return jnp.clip(jnp.exp(pointer_logit_scale[0]), 1.0, 20.0)

    def _source_key_proj(self, planet_out, map_features):
        """构造 source 选择 head 的 key；启用 CNN 时融合局部地图特征。"""
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
        """编码棋盘状态，返回全局 CLS、星球 token 和 source-target 相对表示。"""
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
        """根据当前剩余兵力和档位索引得到实际派兵数。"""
        buckets = self._bucket_values(ships)
        return jnp.take_along_axis(buckets, jnp.clip(bucket, 0, self.ship_bucket_count - 1)[:, None], axis=-1).squeeze(-1)

    def _bucket_values(self, ships):
        """按比例生成动态派兵档位，最小合法发射数为 10。"""
        ratios = jnp.asarray([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=jnp.float32)
        ratios = ratios[: self.ship_bucket_count]
        s = jnp.floor(ships)
        raw = jnp.clip(jnp.round(s[:, None] * ratios[None, :]), 10.0, s[:, None])
        return jnp.where(s[:, None] >= 10.0, raw, jnp.zeros_like(raw))

    def _dynamic_amount_mask(self, ships):
        """动态 amount mask：只允许剩余兵力能覆盖的正数档位。"""
        buckets = self._bucket_values(ships)
        positive = buckets >= 10.0
        return positive

    def _planet_features_with_noop(self, planet_features):
        """在 planet token 前补 no-op slot，使动作索引 0 始终可用。"""
        noop = jnp.zeros(planet_features.shape[:1] + (1, planet_features.shape[-1]), dtype=planet_features.dtype)
        return jnp.concatenate([noop, planet_features], axis=1)
