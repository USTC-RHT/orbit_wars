# Transformer 架构迁移指南

> 目标：把 `jax_agent` 的策略网络从「per-source MLP + 单头 cross-attn 选目标」改造为
> 「全局 Transformer Encoder + Pointer Network」架构，让模型显式建模星球与舰队之间的关系。
>
> 阅读对象：负责实施改造的 AI Coding Agent。
> 前置阅读：`jax_agent/GUIDE.md`（项目总览）、`jax_agent/model_jax.py`（旧架构）、
> `jax_agent/features_jax.py`（特征编码）、`jax_agent/ppo_jax.py`（PPO 更新）、
> `jax_agent/train_jax.py`（rollout & 训练循环）。

---

## 1. 设计目标与不动点

### 1.1 想达成什么

- **关系建模**：让每颗星球在编码阶段就能看到所有其他星球与所有相关舰队（self-attention）。
- **舰队作为 first-class entity**：舰队作为独立 token 进入网络，而非被聚合成 `planet_features` 中的若干标量。
- **全局 critic**：用一个 CLS token 通过 attention 自适应聚合全局状态价值，替换当前的 `max+mean pool → value_head`。
- **Pointer 选目标**：保留 cross-attention pointer，但 q/K 输入升级为 Transformer 输出后的 contextual embedding。

### 1.2 必须保持不变的接口（critical）

以下接口**不要改动**，否则会破坏 rollout / PPO update / kaggle 推理三处调用：

1. **`PolicyOutput` 字段**：`fire_logits / target_logits / amount_logits / value` 名称、形状、语义不变。
2. **`ActorCritic.__call__` 签名**：参数列表与现在完全一致：
   ```python
   def __call__(self, planet_features, candidate_features, global_features,
                candidate_mask, map_features=None, selected_targets=None)
   ```
   这样 `train_jax.policy_sample`、`policy_sample_params`、`ppo_jax.ppo_update_step` 都不需要改调用方。
3. **`fast_env.EnvState`**：环境状态结构不动。
4. **`EncodedBatch` 的现有字段名不动**，只**新增** `fleet_features` 与 `fleet_mask` 两个字段。
5. **PPO loss 公式不动**（`log_prob_entropy`、value clipping、log_ratio clip 全部保留）。

> **设计原则**：把舰队 token 与 Transformer 包裹在 `ActorCritic.__call__` 内部，调用方完全无感。
> 唯一需要新增的是 `fleet_features / fleet_mask` 两个 kwarg，且必须有默认值 `None` 以兼容旧 checkpoint 推理路径（kaggle 提交时也可能用旧编码）。

### 1.3 复杂度预算

- 新增 token 数 K（fleet token 数）建议 **K = 32**。
- Transformer 层数 N 建议 **N = 2**，hidden = 128，heads = 4。
- 训练 sps 退化预期：当前 ≈40k → 目标 ≥15k（≤2.5× slowdown）。若退化超过 4×，先降到 N=1, K=16。

---

## 2. 整体架构图

```
                ┌────────────────────────┐
                │   原始 EnvState (JAX)  │
                └───────────┬────────────┘
                            │
                ┌───────────▼────────────┐
                │  features_jax.encode_  │
                │  single (per-state,    │
                │   vmap over source)    │
                └───────────┬────────────┘
                            │ 新增 fleet_features / fleet_mask
                            │
        ┌───────────────────▼────────────────────┐
        │           ActorCritic.__call__          │
        │  ┌──────────────────────────────────┐   │
        │  │  Token Embedding                  │   │
        │  │  ─ planet_h = MLP(cand_features)  │   │
        │  │  ─ fleet_h  = MLP(fleet_features) │   │
        │  │  ─ cls_h    = learned [1,H]       │   │
        │  │  ─ + type_embed (planet/fleet/cls)│   │
        │  └────────────────┬──────────────────┘   │
        │                   │                       │
        │  tokens = [cls | planet × P | fleet × K] │
        │  mask   = [True | planet_alive | fleet_alive] │
        │                   │                       │
        │  ┌────────────────▼──────────────────┐   │
        │  │  Transformer Encoder × N          │   │
        │  │  (MHA + FFN + Pre-LayerNorm)      │   │
        │  └────────────────┬──────────────────┘   │
        │                   │                       │
        │  cls_out, planet_out, fleet_out  (split) │
        │                   │                       │
        │  ┌────────────────┼─────────────────┐    │
        │  │                │                 │    │
        │  ▼                ▼                 ▼    │
        │ Value Head    Fire Head          Pointer  Amount │
        │ (cls_out)    (context_h)       (q·K planet_out) │
        │                                                  │
        │  context_h = concat(MLP(self_f),                 │
        │                     MLP(global_f),               │
        │                     planet_out[src_idx])         │
        └──────────────────────────────────────────────────┘
```

**关键说明**：

- **Self-Attention**：所有 token（cls + planet + fleet）互相 attend，fleet 的输出**只用于影响 planet 与 cls 的表示**，本身不参与决策头。
- **Cross-Attention 的 q/K**：q 来自 `context_h`（自身 + 全局 + Transformer 后的本星球 embedding），K 仅来自 `planet_out`（fleet 不在 pointer 候选里——游戏不允许把船派到舰队上）。
- **CLS token**：替换现在的 `max+mean pool`。CLS 的输出经 self-attention 聚合所有 token，是 critic 的输入。

---

## 3. 文件级改动清单

### 3.1 必须修改的文件

| 文件 | 性质 | 改动范围 |
|------|------|----------|
| `jax_agent/model_jax.py` | 重写 | 增加 Transformer encoder，重构 `ActorCritic.__call__` |
| `jax_agent/features_jax.py` | 扩展 | 新增 `build_fleet_features` 与 `FLEET_DIM`，扩展 `EncodedBatch` |
| `jax_agent/train_jax.py` | 小改 | `policy_sample` / `policy_sample_params` 透传新 kwarg；`ActorCritic(...)` 构造新增超参 |
| `jax_agent/ppo_jax.py` | 小改 | `ppo_update_step` 的 `model.apply(...)` 透传新 kwarg；`flatten_rollout` 加新字段 |
| `jax_agent/cfg_jax_selfplay.yaml` | 小改 | 新增 `model.use_transformer / num_layers / num_heads / max_fleet_tokens` |

### 3.2 不需要改的文件（确认）

- `jax_agent/fast_env.py` — 状态结构已包含 `fleets / fleet_alive`，无需改动
- `jax_agent/reward_jax.py` — 奖励函数与网络无关
- `jax_agent/eval_*.py` — 通过 `model.apply` 调用，自动受益

---

## 4. 特征工程详解（`features_jax.py`）

### 4.1 新增常量

在文件头部添加：

```python
FLEET_DIM = 14            # 单支舰队 token 特征维度
MAX_FLEET_TOKENS = 32     # 进入网络的舰队 token 数（top-K by relevance）
```

> **为何不是 `params.max_fleets=512`？**
> Self-attention 复杂度 O(T²)。512 + 45 + 1 = 558 个 token 太大。
> 取与当前局面最相关的 K 支舰队即可（按距离/威胁排序）。

### 4.2 新增 EncodedBatch 字段

```python
class EncodedBatch(NamedTuple):
    planet_features: jnp.ndarray
    candidate_features: jnp.ndarray
    global_features: jnp.ndarray
    map_features: jnp.ndarray
    candidate_mask: jnp.ndarray
    amount_mask: jnp.ndarray
    source_ids: jnp.ndarray
    source_ships: jnp.ndarray
    candidate_ids: jnp.ndarray
    ship_buckets: jnp.ndarray
    source_mask: jnp.ndarray
    # 新增：
    fleet_features: jnp.ndarray   # [max_planets, MAX_FLEET_TOKENS, FLEET_DIM]  per-source 行
    fleet_mask: jnp.ndarray       # [max_planets, MAX_FLEET_TOKENS] bool
```

### 4.3 `build_fleet_features` 实现要点

签名：
```python
def build_fleet_features(state: EnvState, src_idx: jnp.ndarray,
                          params: FeatureParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    """返回 (fleet_features[K, FLEET_DIM], fleet_mask[K])"""
```

**Step 1 — 计算每支舰队与源星球的"相关度"**：
```python
src = state.planets[src_idx]
src_x, src_y = src[2], src[3]
fleets = state.fleets                        # [F, 7]
alive = state.fleet_alive                    # [F]
dist = jnp.hypot(fleets[:, 2] - src_x, fleets[:, 3] - src_y)
# 死槽位推到无穷远
score = jnp.where(alive, dist, jnp.inf)
top_k_idx = jnp.argsort(score)[:MAX_FLEET_TOKENS]   # [K]
sel = fleets[top_k_idx]                              # [K, 7]
sel_alive = alive[top_k_idx]                         # [K]
```

**Step 2 — 构造 14 维特征向量**：

| 索引 | 含义 | 计算 |
|------|------|------|
| 0 | `is_mine` | `1.0 if owner==0 else 0.0` |
| 1 | `x_norm` | `sel[:, 2] / BOARD_SIZE` |
| 2 | `y_norm` | `sel[:, 3] / BOARD_SIZE` |
| 3 | `cos_angle` | `jnp.cos(sel[:, 4])` |
| 4 | `sin_angle` | `jnp.sin(sel[:, 4])` |
| 5 | `ships_norm` | `clip01(sel[:, 6] / params.max_ships)` |
| 6 | `speed_norm` | `fleet_speed(sel[:, 6]) / 6.0` |
| 7 | `dist_to_sun_norm` | `hypot(x-CENTER, y-CENTER) / BOARD_SIZE` |
| 8 | `from_x_norm` | 用 `sel[:, 5]` 索引 `state.planets[:, 2] / BOARD_SIZE` |
| 9 | `from_y_norm` | 同上，y |
| 10 | `from_is_mine` | `state.planets[from_id, 1] == 0` |
| 11 | `from_ships_norm` | `clip01(state.planets[from_id, 5] / params.max_ships)` |
| 12 | `dx_to_src_norm` | `(sel[:, 2] - src_x) / BOARD_SIZE`（**带方向**信息，让 src 知道这支舰队相对自己在哪） |
| 13 | `dy_to_src_norm` | `(sel[:, 3] - src_y) / BOARD_SIZE` |

> **注意**：用 `cos/sin(angle)` 不要直接用 `angle`。神经网络处理周期变量必须解开包裹。

**Step 3 — 安全索引 from_id**：
```python
from_id = sel[:, 5].astype(jnp.int32)
# 防止 from_id < 0 或越界（fast_env 应保证有效，但加保险）
from_id = jnp.clip(from_id, 0, params.max_planets - 1)
from_planets = state.planets[from_id]   # [K, 7]
```

**Step 4 — Mask 处理**：
```python
features = jnp.where(sel_alive[:, None], features, 0.0)
return features, sel_alive
```

### 4.4 集成到 `build_source_decision`

```python
def build_source_decision(state, src_idx, global_f, inbound, params):
    src = state.planets[src_idx]
    self_f = build_planet_features(state, src_idx, inbound, params)
    cand_f, cand_mask, amount_mask, candidate_ids, ship_buckets = \
        build_candidate_features(state, src_idx, inbound, params)
    map_f = build_map_features(state, src_idx, params)
    fleet_f, fleet_m = build_fleet_features(state, src_idx, params)   # 新增
    return EncodedBatch(
        ...,                             # 原有字段
        fleet_features=fleet_f,
        fleet_mask=fleet_m,
    )
```

> **重要**：`fleet_features` 是 per-source 计算的（因为 top-K 排序依赖 src 距离）。
> 这意味着 `[B, max_planets, K, FLEET_DIM]` 的存储成本。
> 如果显存吃紧，**替代方案**：改为 per-state 全局 top-K（按舰队 ships 大小或威胁排序），所有 source 共享。
> 起步建议先用 per-source 版本（信息更丰富），观测到 OOM 再切换。

### 4.5 `target_filter_enabled` 路径

如果当前已启用 target_filter（`cfg.env.target_filter_enabled=true`），fleet token 不应受其影响——它们和 planet candidate 是不同维度。直接构造即可。

---

## 5. 模型实现（`model_jax.py`）

### 5.1 新增模块

```python
class TransformerEncoderLayer(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None):
        # Pre-LayerNorm 风格（更稳定）
        h = nn.LayerNorm()(x)
        # mask: [B, T] bool. 转 attention mask: [B, 1, T, T]
        attn_mask = None
        if mask is not None:
            # padding mask: 不允许 attend 到死 token
            attn_mask = mask[:, None, None, :]   # [B, 1, 1, T]
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout_rate=self.dropout,
            deterministic=True,
        )(h, h, mask=attn_mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.hidden_size * self.mlp_ratio)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_size)(h)
        return x + h
```

### 5.2 重构 `ActorCritic`

新增超参（带默认值，保持向后兼容）：

```python
class ActorCritic(nn.Module):
    planet_dim: int
    candidate_dim: int
    global_dim: int
    candidate_count: int
    ship_bucket_count: int
    hidden_size: int = 128
    use_cnn: bool = False
    map_channels: int = 6
    map_grid_size: int = 8
    # 新增：
    use_transformer: bool = False
    fleet_dim: int = 14
    max_fleet_tokens: int = 32
    num_transformer_layers: int = 2
    num_heads: int = 4
```

> 通过 `use_transformer=False` 默认，旧 checkpoint 推理路径仍能跑（fallback 到老架构）。

### 5.3 `__call__` 主体

```python
@nn.compact
def __call__(self, planet_features, candidate_features, global_features,
             candidate_mask, map_features=None, selected_targets=None,
             fleet_features=None, fleet_mask=None):
    if self.use_transformer:
        return self._forward_transformer(
            planet_features, candidate_features, global_features,
            candidate_mask, fleet_features, fleet_mask,
            map_features, selected_targets,
        )
    # 旧路径：保留现有实现，便于 A/B 对照
    return self._forward_legacy(...)
```

`_forward_transformer` 核心逻辑：

```python
def _forward_transformer(self, planet_features, cand_features, global_features,
                          cand_mask, fleet_features, fleet_mask,
                          map_features, selected_targets):
    B = planet_features.shape[0]
    H = self.hidden_size
    P = self.candidate_count                  # max_planets + 1（含 noop slot 0）
    K = self.max_fleet_tokens

    # 1) 三类 token embedding
    planet_h = MLP(H)(cand_features)          # [B, P, H]
    fleet_h  = MLP(H)(fleet_features) if fleet_features is not None \
               else jnp.zeros((B, K, H), dtype=planet_h.dtype)
    cls_h    = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, H))
    cls_h    = jnp.broadcast_to(cls_h, (B, 1, H))

    # 2) Type embedding（让 attention 区分实体类型）
    type_embed = self.param("type_embed", nn.initializers.normal(0.02), (3, H))
    cls_h    = cls_h    + type_embed[0]
    planet_h = planet_h + type_embed[1]
    fleet_h  = fleet_h  + type_embed[2]

    # 3) 拼接 + mask
    tokens = jnp.concatenate([cls_h, planet_h, fleet_h], axis=1)  # [B, 1+P+K, H]
    cls_mask = jnp.ones((B, 1), dtype=bool)
    if fleet_mask is None:
        fleet_mask = jnp.zeros((B, K), dtype=bool)
    # candidate_mask 是 [B, P]（与 cand_features 一致），slot 0 是 noop 应为 True 才能 attend
    # 注意：原 candidate_mask 表示"是否合法目标"；这里我们要"是否存活 token"，slot 0 永远 True
    planet_attn_mask = candidate_mask.at[:, 0].set(True) if False else candidate_mask
    # 更准确：用 cand_features[..., 1] > 0.5 提取 alive bit（dim 1 是 alive flag）
    planet_alive = cand_features[..., 1] > 0.5
    planet_alive = planet_alive.at[:, 0].set(True)   # slot 0 (no-op) 始终活
    attn_mask = jnp.concatenate([cls_mask, planet_alive, fleet_mask], axis=1)  # [B, 1+P+K]

    # 4) Transformer encoder
    x = tokens
    for _ in range(self.num_transformer_layers):
        x = TransformerEncoderLayer(self.hidden_size, self.num_heads)(x, mask=attn_mask)
    x = nn.LayerNorm()(x)

    # 5) 拆分输出
    cls_out    = x[:, 0, :]                   # [B, H]
    planet_out = x[:, 1:1+P, :]               # [B, P, H]
    # fleet_out = x[:, 1+P:, :]               # 不使用

    # 6) Context for per-source decision heads
    self_h   = MLP(H)(planet_features)          # [B, H]
    global_h = MLP(H)(global_features)        # [B, H]
    # 当前 source planet 的 contextual embedding（src 在 batch 维已 vmap，slot 0 之后是 planet_idx+1）
    # 注意：现在的 vmap 把 source planet 摊平到 batch 维，每行对应一颗 source。
    # 我们没法直接知道 src_idx → 用 planet_features 的"is source"信号已编码在 self_h 里。
    # 取 cls_out 作为全局 context 即可（足够）。
    context_h = jnp.concatenate([self_h, global_h, cls_out], axis=-1)  # [B, 3H]

    # 7) Pointer Network 选目标（cross-attention，仅 planet token 参与）
    q = nn.Dense(H)(context_h)
    q = nn.relu(q)
    q = nn.Dense(H)(q)
    q = q / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(q), -1, keepdims=True), 1e-12))
    keys = nn.Dense(H)(planet_out)
    noop_key = self.param("noop_key", nn.initializers.zeros, (H,))
    keys = keys.at[:, 0, :].set(noop_key)
    keys = keys / jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(keys), -1, keepdims=True), 1e-12))
    target_logits = jnp.einsum("bh,bch->bc", q, keys) / jnp.sqrt(float(H))
    target_logits = jnp.where(candidate_mask, target_logits, -1.0e9)

    # 8) Fire / Amount / Value 头
    fire_logits = nn.Dense(H)(context_h)
    fire_logits = nn.relu(fire_logits)
    fire_logits = nn.Dense(1)(fire_logits).squeeze(-1)

    if selected_targets is None:
        selected_targets = jnp.zeros((B,), dtype=jnp.int32)
    row = jnp.arange(B)
    selected_h = planet_out[row, selected_targets]
    amount_h = jnp.concatenate([context_h, selected_h], axis=-1)
    amount_h = nn.relu(nn.Dense(H)(amount_h))
    amount_logits = nn.Dense(self.ship_bucket_count)(amount_h)

    value_h = nn.relu(nn.Dense(H)(cls_out))
    value = nn.Dense(1)(value_h).squeeze(-1)

    return PolicyOutput(fire_logits, target_logits, amount_logits, value)
```

### 5.4 关键实现细节（务必遵守）

1. **数值安全**：保留 L2 norm 的 `jnp.maximum(jnp.sum(jnp.square(x), -1, True), 1e-12)` 写法。**不要**用 `jnp.linalg.norm(x) + eps`，会在 x=0 处反向 NaN。
2. **`candidate_mask` 的双重含义**：
   - 在**legal target**意义上：slot 0 (noop) 是否可选取决于 fire；不能简单设 True。
   - 在**alive token**意义上（attention mask）：slot 0 必须为 True，否则 attention 报 NaN。
   - 上面代码用 `cand_features[..., 1] > 0.5` 提取 alive bit 单独构造 attention mask，**不要复用 `candidate_mask`**。
3. **Pre-LayerNorm**：与 GPT/ViT 一致，比 Post-LN 更稳定，PPO 训练不易梯度爆炸。
4. **Dropout=0**：RL 中通常关闭 dropout，否则同一 obs 两次前向不一致，破坏 PPO importance sampling。
5. **Fleet token 全死**：当 episode 早期没有舰队时，`fleet_mask` 全 False，attention 对 fleet 全 mask。
   - **务必保证 cls + planet alive 至少有 True**，否则 softmax 全 -inf → NaN。
   - 已通过 `cls_mask = jnp.ones(...)` 和 `slot 0 强制 True` 保证。

---

## 6. 训练代码改动（`train_jax.py`）

### 6.1 `policy_sample` / `policy_sample_params`

两处 `model.apply(...)` 调用都需要透传 fleet kwargs：

```python
out1 = model.apply(
    {"params": params},
    flat.planet_features, flat.candidate_features, flat.global_features,
    flat.candidate_mask, flat.map_features, zeros,
    fleet_features=flat.fleet_features,   # 新增
    fleet_mask=flat.fleet_mask,           # 新增
)
```

第二次 forward（`out2`）同样补上。

### 6.2 `ActorCritic(...)` 构造（约 line 350）

```python
model = ActorCritic(
    planet_dim=PLANET_DIM,
    candidate_dim=CANDIDATE_DIM,
    global_dim=GLOBAL_DIM,
    candidate_count=feat_params.max_planets + 1,
    ship_bucket_count=feat_params.ship_bucket_count,
    hidden_size=cfg.model.hidden_size,
    use_cnn=cfg.model.use_cnn,
    # 新增：
    use_transformer=cfg.model.use_transformer,
    fleet_dim=FLEET_DIM,
    max_fleet_tokens=MAX_FLEET_TOKENS,
    num_transformer_layers=cfg.model.num_transformer_layers,
    num_heads=cfg.model.num_heads,
)
```

### 6.3 `make_train_state` 的 `sample` 形状

`make_train_state` 用 sample dict 初始化网络。`sample` 必须包含 `fleet_features / fleet_mask` 字段，否则 `model.init(...)` 拿不到形状。检查 `ppo_jax.make_train_state` 调用方，给 sample 添加：

```python
sample["fleet_features"] = jnp.zeros((1, MAX_FLEET_TOKENS, FLEET_DIM), dtype=jnp.float32)
sample["fleet_mask"]     = jnp.zeros((1, MAX_FLEET_TOKENS), dtype=bool)
```

并在 `model.init(rng, **sample)` 处确保 kwargs 传递。

### 6.4 Rollout 存储

`Rollout` NamedTuple 需要新增字段：
```python
fleet_features: jnp.ndarray   # [T, num_envs, max_planets, K, FLEET_DIM]
fleet_mask: jnp.ndarray       # [T, num_envs, max_planets, K]
```
`flatten_rollout` 同步加入键。

### 6.5 `kaggle_obs_to_jax_state` 路径

kaggle 推理路径会调用 `encode_single`，`fleet_features` 自动产生，无需特殊处理。

---

## 7. PPO 更新改动（`ppo_jax.py`）

### 7.1 `flatten_rollout` 新增字段

```python
data = {
    ...,
    "fleet_features": flat(rollout.fleet_features),
    "fleet_mask": flat(rollout.fleet_mask),
}
```

### 7.2 `ppo_update_step` 内 `model.apply` 调用

```python
action_outputs = train_state.apply_fn(
    {"params": params},
    batch["planet_features"],
    batch["candidate_features"],
    batch["global_features"],
    batch["candidate_mask"],
    batch["map_features"],
    batch["target"],
    fleet_features=batch["fleet_features"],   # 新增
    fleet_mask=batch["fleet_mask"],           # 新增
)
```

`value_outputs` 调用（如果有独立 forward）同步加。

### 7.3 显存检查

新增 batch 项形状 `[mb_size, K, FLEET_DIM]` = `[4096, 32, 14]` ≈ 7MB float32。可以接受。

---

## 8. 配置文件改动（`cfg_jax_selfplay.yaml`）

在 `model:` section 增加：

```yaml
model:
  hidden_size: 128
  use_cnn: false
  # 新增
  use_transformer: true
  num_transformer_layers: 2
  num_heads: 4
```

`max_fleet_tokens` 与 `fleet_dim` 在 `features_jax.py` 顶层 const 定义即可，不放 yaml（避免散落配置）。

---

## 9. 实施步骤（按顺序）

> **每一步完成后必须跑一次 smoke test**：`timeout 120 python -m jax_agent.train_jax --config cfg_jax_selfplay.yaml`，
> 看是否能完成 1 个 update。

1. **Step 1 — 特征层**：
   - 在 `features_jax.py` 添加 `FLEET_DIM / MAX_FLEET_TOKENS / build_fleet_features`
   - 扩展 `EncodedBatch` 与 `build_source_decision`
   - 写一个 `pytest` 用例验证 shape 正确：`test_fleet_features_shape`

2. **Step 2 — 模型层**：
   - 在 `model_jax.py` 添加 `TransformerEncoderLayer` 与 `_forward_transformer`
   - 保留 `_forward_legacy` 作为 `use_transformer=False` 的 fallback
   - 写单测：`test_transformer_forward_shape`、`test_transformer_handles_no_fleets`

3. **Step 3 — 训练层串通**：
   - 修改 `make_train_state` 的 sample 构造
   - 修改 `policy_sample / policy_sample_params / ppo_update_step / Rollout / flatten_rollout`
   - 修改 cfg yaml 加 `use_transformer: false`（先不开）
   - 跑 smoke test：确认旧路径仍能跑（说明未破坏）

4. **Step 4 — 开启 Transformer**：
   - 改 cfg `use_transformer: true`，跑 smoke test 1 update
   - 检查 sps 退化倍数（应 ≤2.5×）
   - 检查 KL / loss / value 是否在合理范围（KL < 0.1, 无 NaN）

5. **Step 5 — 全量训练**：
   - 跑 200 updates，对比 baseline（旧架构 200 updates 的 nearest win_rate）
   - 若 win_rate 不退步且 entropy 收敛健康，切到 5000 updates 完整训练

---

## 10. 调试与回滚

### 10.1 常见 NaN 来源

| 症状 | 可能原因 | 对策 |
|------|---------|------|
| `ppo/loss` NaN | attention mask 全 False（fleet 全死） | 确保 cls_mask 永远 True |
| `ppo/grads` NaN | L2 norm 在 x=0 处反向 | 用 `sqrt(max(sum_sq, 1e-12))` 写法 |
| `target_logits` NaN | candidate_mask 全 False | 确保 noop slot 0 在 candidate_mask 中至少有路径 |
| KL 突然爆炸 | Transformer 训练初期参数尺度不对 | LayerNorm 默认 init 即可；如不行试 `nn.initializers.normal(0.02)` for cls/type embed |

### 10.2 回滚机制

- 旧架构以 `_forward_legacy` 形式保留在 `model_jax.py` 中
- cfg `use_transformer: false` 即可一键回退
- 训练 checkpoint 不兼容（参数 shape 不同），保留 `weights/` 目录下旧 ckpt 单独管理

### 10.3 性能对比基准

迁移前请记录当前 baseline 数字（建议 200 updates 处的）：
- `sps`（steps per second）
- `eval/nearest_win_rate`
- `eval/sigma_win_rate`
- `eval/tamrazov_win_rate`（若有）
- `train/approx_kl` 平均值
- `train/entropy` 平均值

迁移后用同一 seed 训练同样步数，对比每一项。

---

## 11. 后续可选优化（不在本次范围）

- **Per-source planet token 替换**：当前 src 信息走 `planet_features → MLP`。可改为直接取 `planet_out[src_idx]` 作为 src embedding（需要 features 层把 src_idx 传到模型，或在 vmap 外做）。
- **Fleet-as-target**：游戏不允许直接打舰队，但可以加一个辅助 head 预测"哪支敌方舰队最威胁"，作为 auxiliary loss 帮助 representation 学习。
- **Centralized critic**：critic 改为看双方完整状态（去对称、看 enemy planet 详情），actor 仍只看己方。
- **RoPE / 相对位置**：把星球间相对距离作为 attention bias，进一步增强空间感知。

---

## 12. Checklist（提交前必查）

- [ ] `features_jax.py` 中 `FLEET_DIM` 与 `model.fleet_dim` 一致
- [ ] `make_train_state` 的 `sample` 包含 `fleet_features / fleet_mask`，且 `model.init` 接受
- [ ] `policy_sample` 与 `policy_sample_params` 两处都补了 kwargs
- [ ] `ppo_update_step` 的 `model.apply` 补了 kwargs
- [ ] `Rollout` NamedTuple 与 `flatten_rollout` 同步增加字段
- [ ] cfg yaml 新增 `use_transformer / num_transformer_layers / num_heads`
- [ ] 旧 ckpt 路径用 `use_transformer: false` 验证可加载
- [ ] smoke test 跑 1 update 不 NaN
- [ ] 200 updates 训练 KL / entropy 健康
- [ ] sps 退化在预期范围内（≤2.5×）
