# Orbit Wars JAX Transformer PPO · 模型架构详解

本文档完整描述当前 Transformer self-play 配置下 `ActorCritic` 网络（`jax_agent/model_jax.py`）的端到端结构，重点是把每一层的张量形状与作用都标清楚，便于后续画图与调试。

---

## 0. 关键超参与符号

| 符号 | 含义 | 当前值 |
|------|------|--------|
| `B` | batch size（=`num_envs`） | 64 |
| `S` | per-state source 数（每方星球槽位） | 与 `max_planets` 同形状 |
| `P` | 候选目标数（含 slot 0 = no-op） | `max_planets + 1` = 46 |
| `F` | 舰队 token 数 | `max_fleet_tokens` = 16 |
| `H` | 隐层宽度 `hidden_size` | 128 |
| `L` | Transformer 层数 | 2 |
| `A` | 注意力头数 `num_heads` | 4 |
| `Bk` | 派兵档位数 `ship_bucket_count` | 8 |
| `Cm` | map 通道数 | 6 |
| `G` | map 网格分辨率 | 8 |
| `D_self / D_cand / D_global / D_fleet` | 各特征维度 | 18 / 26 / 11 / 16 |

> **核心观察**：网络是 *per-source* 的——同一时刻每个我方源星球都做一次独立决策，因此 self / candidate / map 这些和 source 绑定的特征会比 global / planet token 多一个 `S` 维度。
> 代码里两条对应路径：
> - `_forward_transformer`：source 维度展平后的 batched 调用
> - `_forward_transformer_grouped`：保留 source 维度，**当前训练实际走的就是这条**

---

## 1. 输入张量一览（grouped 模式）

| 名称 | shape | 来源 |
|------|-------|------|
| `self_features` | `[B, S, 18]` | source 自身 + 局势统计 |
| `candidate_features` | `[B, S, P, 26]` | source × 候选目标 pairwise |
| `global_features` | `[B, S, 11]` | 全局局势（每个 source 一份，但内容相同） |
| `fleet_features` | `[B, S, F, 16]` | 最多 16 个舰队 token |
| `fleet_mask` | `[B, S, F]` (bool) | 舰队槽位是否有效 |
| `map_features` | `[B, S, 6, 8, 8]` | CNN 小地图 |
| `candidate_mask` | `[B, S, P]` (bool) | 合法目标 mask（含 no-op） |
| `selected_targets` | `[B, S]` (int32) | amount 头条件输入；训练时 = 当前采样的 target |

> grouped 模式下，`candidate_features[:, 0]`（任一 source 视角）会被去掉相对字段后作为**共享 planet token 输入**进入 Transformer。也就是说 **state-level 的 token 编码在所有 source 之间共享**，只在最后的 pointer 步骤里再加回 source-relative 增量。

---

## 2. 三段式数据流总览

```
┌──────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│ A. Token Enc │ → │ B. Transformer   │ → │ C. Per-source Heads │
│  + CNN map   │   │  (state encoder) │   │  fire/tgt/amt/value │
└──────────────┘   └──────────────────┘   └─────────────────────┘
```

- **A** 把 planet/fleet/CNN 编成等宽向量（`H = 128`）。
- **B** 用 2 层 Transformer 让 token 互相看见全局信息（共享于所有 source）。
- **C** 取每个 source 的 context（self + global + source token + CLS + CNN）做四个头。

---

## 3. 详细分层结构

### 3.1 A 段 · Token & CNN 编码

#### (a) Planet token 编码（共享）

```
candidate_features[:, 0]                  # [B, P, 26]
   └─ _shared_planet_features            # 把相对字段(2,8,9,10,16,17,24,25)置零, 重置 is_mine
   └─ MLP(H)                              # 两层 Dense(H)+LN+ReLU+Dense(H)+ReLU
   └─ + type_embed[1]                    # planet 类型嵌入
   = planet_h                            # [B, P, H]
```

#### (b) Fleet token 编码

```
fleet_features[:, 0]                      # [B, F, 16]
   └─ MLP(H)
   └─ + type_embed[2]
   = fleet_h                             # [B, F, H]
```

#### (c) CLS token

```
cls_token (learned param, [1,1,H])       # broadcast 到 [B, 1, H]
   └─ + type_embed[0]
```

#### (d) Token 拼接 & mask

```
tokens     = concat([cls, planet_h, fleet_h], dim=1)        # [B, 1+P+F, H]
token_mask = concat([1, planet_alive, fleet_mask], dim=1)   # [B, 1+P+F]
            # planet_alive[:, 0] 强制 True (no-op slot)
```

#### (e) CNN 小地图分支（逐 source 独立）

```
map_features                              # [B, S, 6, 8, 8]
   └─ flatten 到 [B*S, 6, 8, 8] → NHWC
   └─ Conv(16, 3x3)+ReLU
   └─ Conv(32, 3x3)+ReLU
   └─ Conv(16, 3x3)+ReLU
   └─ GlobalAvgPool (mean over H,W)
   └─ Dense(16)+ReLU
   └─ reshape 回 [B, S, 16]
   = map_h
```

> 注意：CNN 是 **per-source** 计算的，因为不同 source 的 source channel 不同。

---

### 3.2 B 段 · Transformer State Encoder

```
x = tokens                                # [B, 1+P+F, H]
for l in range(L=2):
    x = TransformerEncoderLayer(H=128, heads=A=4)(x, mask=token_mask)
x = LayerNorm(x)

cls_out    = x[:, 0, :]                   # [B, H]      → 全局 summary
planet_out = x[:, 1:1+P, :]               # [B, P, H]   → 每个候选星球的 contextual embedding
                                          # (fleet_out 不会再被显式用到)
```

`TransformerEncoderLayer` 是 **Pre-LN** 结构：

```
h = LN(x)
h = MultiHeadDotProductAttention(heads=4, qkv=H, out=H)(h, h, mask)
x = x + h
h = LN(x)
h = Dense(4H) → GELU → Dense(H)
x = x + h
```

mask 通过 `mask[:, None, None, :]` 广播到 `[B, 1, 1, T]`，效果是让 padding token 不参与 attention。

---

### 3.3 C 段 · Per-source Decision Context

> 这一段开始**恢复 source 维度**。所有特征拼成一个长 context 向量 `context_h`，供四个头共享。

```
self_h     = MLP(H)(self_features)                      # [B, S, H]
global_h   = MLP(H)(global_features)                    # [B, S, H]

# source_ctx：把候选 P 维里"是否是 source 自身"那一位作为权重，
# 在 transformer planet_out 上做加权平均，得到"该 source 对应的 token 表示"。
source_token_mask = candidate_features[..., 2:3] *
                    (candidate_features[..., 1:2] > 0.5)  # [B, S, P, 1]
source_ctx = (planet_out[:, None, :, :] * source_token_mask).sum(axis=2)
             / max(source_token_mask.sum(axis=2), 1)      # [B, S, H]

cls_per_source = broadcast(cls_out, [B, S, H])           # [B, S, H]

context_parts = [self_h, global_h, source_ctx, cls_per_source]
if use_cnn:
    context_parts.append(map_h)                          # [B, S, 16]

context_h = concat(context_parts, dim=-1)
            # shape = [B, S, 4H + 16] = [B, S, 528]
```

`context_h` 同时喂给 fire / target / amount / value 四个头。

---

### 3.4 C 段 · 四个输出头

#### Head 1 · Fire（是否发射）

```
h = Dense(H, init=ortho√2)(context_h) → ReLU
h = Dense(1, init=ortho 0.01)(h)
fire_logits = squeeze(h)                  # [B, S]   Bernoulli logit
```

#### Head 2 · Target（指针注意力选目标）

target 头是 **source-relative 增强的 pointer attention**：

```
# query：source context 投影 + L2 归一化
q = Dense(H)(context_h) → ReLU
q = Dense(H, ortho 0.01)(q)
q = q / sqrt(max(||q||², 1e-12))          # [B, S, H]

# state-level base key（所有 source 共享）
base_keys = Dense(H, ortho 0.01)(planet_out)            # [B, P, H]

# source-relative 增量：从 candidate_features 中只保留相对字段
relative_feat = _relative_candidate_features(candidate_features)  # 只保 [2,8,9,10,16,17,24,25]
relative_h    = MLP(H)(relative_feat)                   # [B, S, P, H]

# 合并并 L2 归一化
keys = base_keys[:, None, :, :] + relative_h            # [B, S, P, H]
keys[:, :, 0, :] = noop_key                             # slot 0 = 可学习 no-op key
keys = keys / sqrt(max(||keys||², 1e-12))

# scaled dot-product → 用 candidate_mask 屏蔽非法目标
target_logits = einsum("bph,bpch->bpc", q, keys) / sqrt(H)
target_logits = where(candidate_mask, target_logits, -1e9)   # [B, S, P]
```

> **关键点**：state-level 的 planet 表示（`planet_out`/`base_keys`）在所有 source 共享，节省算力；
> 而每个 source 看同一个目标星球时的"距离/相对方位"等差异，通过 `relative_h` 加在 key 上来恢复。

#### Head 3 · Amount（派兵档位）

amount 头的 key 与 target 头共享（`amount_keys = base_keys + relative_h`），
然后 gather 出当前已采样目标的表示与 context 拼接：

```
amount_keys = base_keys[:, None, :, :] + relative_h     # [B, S, P, H]
selected_h  = take_along_axis(amount_keys, selected_targets[..., None, None], axis=2).squeeze(2)
              # [B, S, H]：当前 target 对应的 key

amount_h     = concat([context_h, selected_h], dim=-1)  # [B, S, 4H+16+H]
amount_h     = Dense(H) → ReLU
amount_logits = Dense(Bk=8, ortho 0.01)(amount_h)       # [B, S, 8]
```

> 因为依赖 `selected_targets`，amount 头在采样阶段是**串行**的：先采 target，再用真实 target 算 amount logits。

#### Head 4 · Value（状态价值）

```
# value_use_context = True（默认/推荐）：每 source 独立 value，与 legacy 一致
value_input = context_h                                  # [B, S, 4H+16]
value_h     = Dense(H)(value_input) → ReLU
value       = Dense(1, ortho 1.0)(value_h).squeeze(-1)   # [B, S]

# value_use_context = False：兼容老 checkpoint，所有 source 共享 cls_per_source
```

> 多 source 的 value 在 PPO 中会按 `source_mask` 求平均/求和，得到该 state 的标量 value（具体处理见 `ppo_jax.py`）。

---

## 4. 输出形状汇总

| 输出 | shape | 含义 |
|------|-------|------|
| `fire_logits` | `[B, S]` | 每个 source 是否发射的 Bernoulli logit |
| `target_logits` | `[B, S, P]` | 每个 source 选择各候选目标的分数 |
| `amount_logits` | `[B, S, Bk=8]` | 每个 source 派兵档位 logits |
| `value` | `[B, S]` | 每个 source 的状态价值估计 |

采样后转换成游戏动作：

```
[from_planet_id, angle_rad, ships]
```

其中 `angle_rad` 由物理层基于 target 星球与派兵档位求出。

---

## 5. 参数初始化与正则

| 模块 | 初始化 |
|------|--------|
| 隐层 Dense kernel | `orthogonal(√2)` |
| Actor head 输出 kernel | `orthogonal(0.01)` （小初值，防止初期策略尖锐） |
| Value 输出 kernel | `orthogonal(1.0)` |
| 所有 bias | `zeros` |
| `cls_token`, `type_embed` | `normal(0.02)` |
| `noop_key` | `zeros` |

LayerNorm：标准 Flax 实现；MLP 内部使用 `LN → ReLU → Dense → ReLU`。

---

## 6. 一句话记忆

> **planet/fleet 全局共享一份 Transformer 表示；每个 source 用自己的 self+global+source-token+CLS+CNN 拼成 context，对共享 planet keys 做 source-relative pointer attention，得到 fire/target/amount 三头 + value。**
