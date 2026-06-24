"""Orbit Wars 共享策略网络（Flax 实现）。

架构说明：
- ActorCritic 用于 RL/PPO：fire + target pointer + amount + value。
- FeatureRowBCPolicy 用于 BC 训练、BC runtime 和 ONNX 导出。
- 三头输出：fire（是否发射）、target（指针注意力选星球）、amount（派兵档位）。
- context 由三部分拼接：self_h + global_h + map_h（+ CNN）。
- map_h = max/mean 混合池化候选层表示，得到全局感知。
- BC 目标选择支持 dot head 或 multi-head cross-attention + pairwise MLP。
- 候选星球之间没有 self-attention，与标准 Transformer 的区别在此。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

NEG_INF = -1.0e9


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
        x = nn.Dense(self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        return nn.relu(x)


class BCOutput(NamedTuple):
    """BC 三个监督头的未归一化输出。"""

    fire_logits: jnp.ndarray    # [批量]：是否发射
    target_logits: jnp.ndarray  # [批量, P+1]：no-op + 星球候选分数
    amount_logits: jnp.ndarray  # [批量, B]：派兵档位分数


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

    @nn.compact
    def __call__(self, self_features, candidate_features, global_features, candidate_mask, map_features=None, selected_targets=None):
        # 将三类输入分别编码为等长 hidden 向量
        self_h = MLP(self.hidden_size)(self_features)
        cand_h = MLP(self.hidden_size)(candidate_features)
        global_h = MLP(self.hidden_size)(global_features)
        # 全图池化：对存活星球做 max/mean 混合池，得到全局感知
        # safe_mask 保证当没有星球时至少保留 slot 0 参与池化
        planet_mask = candidate_features[..., 1] > 0.5
        safe_mask = jnp.where(planet_mask.any(axis=-1, keepdims=True), planet_mask, jnp.arange(self.candidate_count)[None, :] == 0)
        masked = jnp.where(safe_mask[..., None], cand_h, jnp.finfo(jnp.float32).min)
        max_pool = masked.max(axis=1)
        mean_pool = (cand_h * safe_mask[..., None]).sum(axis=1) / jnp.maximum(safe_mask.sum(axis=1, keepdims=True), 1)
        map_h = 0.5 * (max_pool + mean_pool)  # 全图池化结果
        context_parts = [self_h, global_h, map_h]
        if self.use_cnn:
            if map_features is None:
                map_features = jnp.zeros((self_h.shape[0], self.map_channels, self.map_grid_size, self.map_grid_size), dtype=self_h.dtype)
            # Flax Conv 使用 NHWC；外部 feature-row contract 存的是 NCHW。
            x = jnp.transpose(map_features, (0, 2, 3, 1))
            x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
            x = x.mean(axis=(1, 2))
            x = nn.relu(nn.Dense(16)(x))
            context_parts.append(x)
        context_h = jnp.concatenate(context_parts, axis=-1)  # [B, hidden*3 (+ cnn_out)]

        # 单头 scaled dot-product cross-attention 实现目标选择
        # query = 来源星球 context 经非线性变换 + L2 归一化
        # 注意：用 sqrt(max(sum(x²), eps²)) 而非 max(norm(x), eps)。
        # 后者在 x=0 时反向传播会产生 0/0=NaN（norm 梯度 = x/norm(x)，
        # 即使 max 截断使前向为 eps，反向仍通过 d(norm)/dx=x/0 产生 NaN 梯度）。
        q = nn.Dense(self.hidden_size)(context_h)
        q = nn.relu(q)
        q = nn.Dense(self.hidden_size)(q)
        q = q / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q), axis=-1, keepdims=True), 1e-12))
        # key = 候选星球编码 + L2 归一化；slot 0 用可学习 no-op key 覆盖
        keys = nn.Dense(self.hidden_size)(cand_h)
        noop_key = self.param("noop_key", nn.initializers.zeros, (self.hidden_size,))
        keys = keys.at[:, 0, :].set(noop_key)
        keys = keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(keys), axis=-1, keepdims=True), 1e-12))
        # logit = q · key / sqrt(d)，非法目标置 -inf
        target_logits = jnp.einsum("bh,bch->bc", q, keys) / jnp.sqrt(float(self.hidden_size))
        target_logits = jnp.where(candidate_mask, target_logits, -1.0e9)

        # fire 头：二分类逻辑回归
        fire_logits = nn.Dense(self.hidden_size)(context_h)
        fire_logits = nn.relu(fire_logits)
        fire_logits = nn.Dense(1)(fire_logits).squeeze(-1)

        # amount 头：拼接已选目标和 context，输出派兵档位 logits
        if selected_targets is None:
            selected_targets = jnp.zeros((self_h.shape[0],), dtype=jnp.int32)
        row = jnp.arange(selected_targets.shape[0])
        selected_h = cand_h[row, selected_targets]
        amount_h = jnp.concatenate([context_h, selected_h], axis=-1)
        amount_h = nn.relu(nn.Dense(self.hidden_size)(amount_h))
        amount_logits = nn.Dense(self.ship_bucket_count)(amount_h)

        # critic 头：当前状态价值
        value_h = nn.relu(nn.Dense(self.hidden_size)(context_h))
        value = nn.Dense(1)(value_h).squeeze(-1)
        return PolicyOutput(fire_logits, target_logits, amount_logits, value)


class FeatureRowBCPolicy(nn.Module):
    """特征行（feature-row）行为克隆策略网络。

    这是 BC 训练、checkpoint runtime、ONNX 导出的唯一共享模型定义。
    参数名和 layer name 必须保持稳定：已保存 checkpoint 的参数树依赖这里。
    """

    ship_bucket_count: int       # 派兵档位数量
    hidden_size: int = 128       # 所有 MLP/attention 分支的主宽度
    use_cnn: bool = True         # 是否使用 map_features 的 CNN 分支
    target_head: str = "dot"     # "dot" 或 "cross_attention"
    target_num_heads: int = 8    # cross_attention 的头数

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
        # 三类 feature-row 输入分别编码，然后拼成来源星球上下文。
        planet_h = MLP(self.hidden_size)(planet)
        relative_h = MLP(self.hidden_size)(relative)
        global_h = MLP(self.hidden_size)(global_features)

        # 候选池化只看有效目标，得到当前来源星球看到的全图摘要。
        valid = candidate_mask.astype(jnp.float32)
        has_valid = candidate_mask.any(axis=1, keepdims=True)
        masked = jnp.where(candidate_mask[..., None], relative_h, jnp.finfo(jnp.float32).min)
        max_pool = jnp.where(has_valid, masked.max(axis=1), 0.0)
        mean_pool = (relative_h * valid[..., None]).sum(axis=1) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)

        context_parts = [planet_h, global_h, 0.5 * (max_pool + mean_pool)]
        if self.use_cnn:
            # 输入 contract 是 [B, C, H, W]，Flax Conv 需要 [B, H, W, C]。
            x = jnp.transpose(map_features, (0, 2, 3, 1))
            x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
            x = x.mean(axis=(1, 2))
            context_parts.append(nn.relu(nn.Dense(16)(x)))

        context_h = jnp.concatenate(context_parts, axis=-1)

        target_head = str(self.target_head or "dot").lower().replace("-", "_")
        if target_head in {"cross_attention", "attention", "mh_attention", "multihead_attention"}:
            # 多头 cross-attention：来源 context 做 query，候选 relative 做 key/value。
            # 后面再用 pairwise MLP 给每个候选补一个非线性打分。
            heads = max(int(self.target_num_heads), 1)
            head_dim = max((int(self.hidden_size) + heads - 1) // heads, 1)
            attn_size = heads * head_dim
            q = nn.relu(nn.Dense(self.hidden_size, name="target_query_mlp_0")(context_h))
            q = nn.Dense(attn_size, name="target_query")(q).reshape((context_h.shape[0], heads, head_dim))
            k = nn.Dense(attn_size, name="target_key")(relative_h).reshape((relative_h.shape[0], relative_h.shape[1], heads, head_dim))
            v = nn.Dense(attn_size, name="target_value")(relative_h).reshape((relative_h.shape[0], relative_h.shape[1], heads, head_dim))
            attn_logits = jnp.einsum("bhd,bchd->bhc", q, k) / jnp.sqrt(float(head_dim))
            masked_attn_logits = jnp.where(candidate_mask[:, None, :], attn_logits, NEG_INF)
            attn_weights = jax.nn.softmax(masked_attn_logits, axis=-1)
            attended = jnp.einsum("bhc,bchd->bhd", attn_weights, v).reshape((context_h.shape[0], attn_size))
            attended = nn.relu(nn.Dense(self.hidden_size, name="target_attended_proj")(attended))
            head_logits = jnp.swapaxes(masked_attn_logits, 1, 2)
            head_mix = nn.Dense(1, use_bias=False, name="target_head_mix")(head_logits).squeeze(-1)
            pair_context = nn.relu(nn.Dense(self.hidden_size, name="target_pair_context")(jnp.concatenate([context_h, attended], axis=-1)))
            pair_context = jnp.broadcast_to(pair_context[:, None, :], relative_h.shape)
            pair = jnp.concatenate([relative_h, pair_context, relative_h * pair_context], axis=-1)
            pair_logits = nn.Dense(1, name="target_pair_score")(
                nn.relu(nn.Dense(self.hidden_size, name="target_pair_mlp")(pair))
            ).squeeze(-1)
            target_logits = head_mix + pair_logits
        else:
            # 轻量 dot 头：context 和候选 key 做归一化点积，速度更快。
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
        # amount 依赖目标：训练时用 label target，greedy 导出时用模型 argmax target。
        amount_h = jnp.concatenate([context_h, selected_h], axis=-1)
        amount_h = nn.relu(nn.Dense(self.hidden_size)(amount_h))
        amount_logits = nn.Dense(self.ship_bucket_count)(amount_h)
        return BCOutput(fire_logits, target_logits, amount_logits)

    @nn.compact
    def greedy_single_pass(
        self,
        planet: jnp.ndarray,
        relative: jnp.ndarray,
        global_features: jnp.ndarray,
        candidate_mask: jnp.ndarray,
        map_features: jnp.ndarray,
    ):
        """ONNX 单图推理入口：先 argmax target，再用该目标预测 amount。"""

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

        target_head = str(self.target_head or "dot").lower().replace("-", "_")
        if target_head in {"cross_attention", "attention", "mh_attention", "multihead_attention"}:
            # 与 __call__ 保持同一套参数名，保证 greedy ONNX 和训练 checkpoint 对齐。
            heads = max(int(self.target_num_heads), 1)
            head_dim = max((int(self.hidden_size) + heads - 1) // heads, 1)
            attn_size = heads * head_dim
            q = nn.relu(nn.Dense(self.hidden_size, name="target_query_mlp_0")(context_h))
            q = nn.Dense(attn_size, name="target_query")(q).reshape((context_h.shape[0], heads, head_dim))
            k = nn.Dense(attn_size, name="target_key")(relative_h).reshape((relative_h.shape[0], relative_h.shape[1], heads, head_dim))
            v = nn.Dense(attn_size, name="target_value")(relative_h).reshape((relative_h.shape[0], relative_h.shape[1], heads, head_dim))
            attn_logits = jnp.einsum("bhd,bchd->bhc", q, k) / jnp.sqrt(float(head_dim))
            masked_attn_logits = jnp.where(candidate_mask[:, None, :], attn_logits, NEG_INF)
            attn_weights = jax.nn.softmax(masked_attn_logits, axis=-1)
            attended = jnp.einsum("bhc,bchd->bhd", attn_weights, v).reshape((context_h.shape[0], attn_size))
            attended = nn.relu(nn.Dense(self.hidden_size, name="target_attended_proj")(attended))
            head_logits = jnp.swapaxes(masked_attn_logits, 1, 2)
            head_mix = nn.Dense(1, use_bias=False, name="target_head_mix")(head_logits).squeeze(-1)
            pair_context = nn.relu(nn.Dense(self.hidden_size, name="target_pair_context")(jnp.concatenate([context_h, attended], axis=-1)))
            pair_context = jnp.broadcast_to(pair_context[:, None, :], relative_h.shape)
            pair = jnp.concatenate([relative_h, pair_context, relative_h * pair_context], axis=-1)
            pair_logits = nn.Dense(1, name="target_pair_score")(
                nn.relu(nn.Dense(self.hidden_size, name="target_pair_mlp")(pair))
            ).squeeze(-1)
            target_logits = head_mix + pair_logits
        else:
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

        safe_target_logits = jnp.where(candidate_mask, target_logits, NEG_INF)
        empty = ~candidate_mask.any(axis=-1, keepdims=True)
        safe_target_logits = jnp.where(empty, 0.0, safe_target_logits)
        target_for_amount = jnp.argmax(safe_target_logits, axis=-1).astype(jnp.int32)
        row = jnp.arange(target_for_amount.shape[0])
        selected_h = relative_h[row, target_for_amount]
        amount_h = jnp.concatenate([context_h, selected_h], axis=-1)
        amount_h = nn.relu(nn.Dense(self.hidden_size)(amount_h))
        amount_logits = nn.Dense(self.ship_bucket_count)(amount_h)
        return fire_logits, target_logits, amount_logits, target_for_amount
