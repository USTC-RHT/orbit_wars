# Orbit Wars Actor-Critic 设计指南

> 针对当前项目 `jax_agent/model_jax.py::ActorCritic` 的设计建议。
> 既给出**当前实现为什么这么设计**的解释，也指出**接下来可以怎么改进**。

---

## 0. 设计目标（先想清楚要什么）

Orbit Wars 不是一个"单步单动作"的环境，每个 state 下：


1. **同时存在多个 source（我方星球槽位）**，每个 source 都要独立决定是否发射、打哪、派多少。
2. **目标是一个"集合"而不是固定枚举**（星球数量可变 → pointer 机制）。
3. **动作空间结构化、有依赖**：是否发射 → 选目标 → 派兵档位。
4. **物理可达性是约束**（aim 失败会被改成 no-op）。

所以你的 Actor-Critic **不能照搬 Atari/MuJoCo 那种"flatten + MLP"**；
应该围绕三点：**(a) Per-source 决策、(b) Pointer 选目标、(c) 结构化动作分层采样**。

---

## 1. 顶层结构：Shared Trunk + Per-Source Heads

推荐保持 **共享 Encoder + 多头输出** 的双塔风格：

```
              ┌────────────┐
              │  Encoder   │  ← 状态级共享 (Transformer + CNN)
              └──────┬─────┘
        ┌────────────┴────────────┐
        ▼                         ▼
   per-source Context        State Summary (CLS)
        │                         │
   ┌────┼─────┬────────┐          │
   ▼    ▼    ▼        ▼          ▼
  Fire Tgt  Amt      Value (Critic)
   ↑   ↑    ↑
   Actor heads
```

**为什么 actor 和 critic 共享 trunk？**

- 状态特征（局势、星球、舰队、地图）对 actor / critic 都有用，共享能大幅省算力。
- 在 RL 里这通常**不会让 critic 拖累 actor**——因为 PPO 用 advantage 而不是 raw value 做 policy gradient，价值估计的偏置不会直接污染策略方向。

**为什么 critic 不另起一塔？**

- 你已经有 grouped Transformer（state-level token 表示共享）。critic 单独一塔意味着你要再跑一次完整的 Transformer，显存和时间都翻倍。
- 如果以后训练发现 *critic underfit / actor 抢容量*，再考虑把 value head 加深，或在最后两层 detach gradient（`stop_gradient`）以隔离梯度。

---

## 2. Actor 的设计要点

Actor 的关键问题是：**怎么把"从哪个 source 发射、打哪、派多少"这三个有依赖的离散动作建好模？**

### 2.1 三头分解（推荐保留）

```
p(a | s) = p(fire | s) · p(target | s, fire=1) · p(amount | s, fire=1, target)
```

对应你现在的：

| Head | 分布 | 输入 | 备注 |
|------|------|------|------|
| `fire_logits` | Bernoulli | `context_h` | source 维独立 |
| `target_logits` | Categorical (P+1) | `context_h` 做 query, 候选 token 做 key (pointer) | slot 0 = no-op |
| `amount_logits` | Categorical (Bk=8) | `context_h ⊕ selected_target_key` | 条件依赖采样到的 target |

**优点**：
- 每个头都有清晰的"含义"，便于看 `entropy_fire / entropy_target / entropy_amount` 单独诊断。
- amount 显式条件在 target 上 → 模型可以学到"打远星球需要多派兵"这类先验。

**注意点**：
- 联合 logp **要按依赖关系写**（不要简单相加）：

```python
logp = log_p_fire + fire_bool * (log_p_target + log_p_amount)
entropy = entropy_fire + p_fire * (entropy_target + entropy_amount)
```

- 不发射的样本，target/amount 不应进入 loss（你现在用 `policy_mask` 屏蔽 + 物理不可达回退到 no-op，是对的）。

### 2.2 Pointer 选目标（推荐保留并增强）

你现在的 target head 是 **scaled dot-product pointer + L2 归一化**：

```python
q     = L2norm( Dense(Dense(context_h)) )            # source query
keys  = L2norm( Dense(planet_out) + MLP(rel_feat) )  # planet keys (含 source-relative 增量)
logits = (q · k) / sqrt(H)
logits = where(candidate_mask, logits, -1e9)
```

**为什么 L2 归一化？** 防止 `||q||` 漂移让 logits 数量级失控（PPO 里这点很重要：logit 太大会让 ratio 一边倒，clip 触发频繁）。

**可以再增强的几点**：

1. **多头 pointer**：现在是单头 dot-product；可以做 *multi-head* 然后取均值或拼接，对"同时打多个目标"这种需求更有表达力（你现在是单源单目标，用单头够了）。
2. **加 bias logit**：候选星球可以加一个**普通 Dense(planet_out, 1)** 的标量 bias 进 target_logits，作为"无视 query 的固有吸引力"。这一项类似 Transformer 里的 *content vs position bias*，能让模型更稳。
3. **no-op 槽更聪明**：现在 slot 0 是一个可学习 `noop_key`。可以给 no-op 一个**条件偏置**——比如当 source 上 inbound 我方援军很多时，no-op logit 自动+；这本质上是把 fire head 的功能复制了一份，但能让 target 分布在边缘场景下更稳。

### 2.3 Amount 头（建议小调整）

amount 现在的输入是 `concat(context_h, selected_h)`：
- `context_h` 是 per-source 的（包含自身舰船数等）。
- `selected_h` 是当前选到的 target 的 key（含距离信息）。

**够用了**，但有两个潜在问题：

1. amount 的 mask（`amount_mask`）是 `[T, B, P, P+1, B]`——它依赖于 target，因为可达性会过滤档位。**实际训练时只对采样到的 target 应用这一行 mask 即可**（你已经这么做了）。
2. amount 离散化为 8 档可能不够细（30%~100%）。可以试试 **bucket=12 + 等比划分**，或者改成 **Beta 分布**（连续动作）：

```python
α, β = softplus(Dense2(amount_h)) + 1.0
fraction = sample(Beta(α, β))   # 0~1 的派兵比例
```

Beta 比 Categorical 更平滑，但 PPO 的 entropy 解析式麻烦些。**建议先把 bucket 加到 12 看看**。

### 2.4 Per-source 共享参数（重要）

**不要为每个 source 学一套独立参数**——星球槽位是排列对称的。
你现在的做法是对的：用同一份 `context_h → heads` 的网络，输入 `[B, S, …]` 一起算，参数对所有 source 共享。

---

## 3. Critic 的设计要点

### 3.1 标量 value vs Per-source value

你当前 `value_use_context=True`，**per-source 输出 `[B, S]`**。这有两条路：

**方案 A**（当前）：每个 source 一个 value，PPO 时按 `policy_mask` 平均/求和。
- 优点：actor / critic 维度对齐，advantage 可按 source 计算。
- 缺点：实际 reward 是 state-level 的，把它"摊到"每个 source 上，会稍微噪声化。

**方案 B**（建议尝试）：critic 输出**标量 state value**，advantage 在 source 维度上广播。
- 用 `cls_out`（已经是全局 summary）经一层 MLP 输出 `[B]`。
- 优点：value 学习 signal 更清晰；和真正的 episode reward 直接对应。
- 缺点：actor 三头 loss 是 per-source 的，需要把 `advantage[B] → broadcast 到 [B, S]` 再用 `policy_mask` 过滤；这其实就是你当前 code 反过来做。

**推荐：保持 A 不变，但实验时把 B 当成 ablation 试一下**。Orbit Wars 因为多 source 同步执行，per-source advantage 能给"打这个 vs 打那个"提供更细的反馈，但在 reward 稀疏（terminal_weight=1.0，其他全 0）时可能噪声更大。

### 3.2 Value Head 容量

现在 value 头是：

```python
value_h = Dense(H=128)(context_h) → ReLU
value   = Dense(1, ortho 1.0)(value_h)
```

**够浅**。如果你看到 `value_loss` 一直降不下来 / `explained_var` 偏低，可以：

- 加深：`Dense(H) → ReLU → Dense(H) → ReLU → Dense(1)`
- 或者给 critic 单独一个 small Transformer head（`cls_out → 2-layer MLP → value`），完全不和 actor 头共享 ReLU 之后的特征。

### 3.3 Value Clipping（已开启，保留）

你 PPO 里已经做了 value clipping：

```
v_clipped = old_v + clip(v - old_v, -clip, clip)
loss = max((v - ret)^2, (v_clipped - ret)^2)
```

这是 stable-baselines 的标准做法，**保留**。在 self-play 这种非平稳环境下，value 跳变剧烈，clip 能避免一两个异常 batch 把 value head 打飞。

---

## 4. Actor / Critic 的耦合方式

### 4.1 共享 Encoder，独立 Head

```
encoder_out  →  actor_branch (three heads)
              ↘
                critic_branch (one head)
```

**这是你现在的结构**，最稳。

### 4.2 是否给 critic 加 stop_gradient？

```python
value_h = Dense(H)(jax.lax.stop_gradient(context_h))
```

当出现 `value_loss * vf_coef` 比 `policy_loss` 大很多、actor 学得慢时可以试。
但通常 **不需要**——`vf_coef = 0.5` + value clipping 已经足够。

### 4.3 学习率与梯度裁剪

- Actor 头初始化用 `orthogonal(0.01)`（你已经做了），让初期策略接近均匀，避免 logit 数量级爆炸。
- Critic 输出用 `orthogonal(1.0)`（你已经做了），允许 value 一开始就有较大动态范围。
- 全网梯度做 `max_grad_norm = 0.5` 裁剪（你已经做了）。

---

## 5. 当前实现 vs 设计建议对照表

| 维度 | 当前实现 | 是否合理 | 备注 / 改进方向 |
|------|----------|----------|------------------|
| Trunk 共享 | Transformer state encoder + CNN，actor/critic 共享 | ✅ | 必要时给 critic 加深一层 |
| Source 处理 | per-source `[B, S, …]`，参数共享 | ✅ | — |
| Fire head | Bernoulli + 浅 MLP | ✅ | — |
| Target head | Pointer attn + L2 + source-relative key | ✅ | 可加 content bias / 多头 |
| Amount head | Cat(8) + 条件 selected target key | ✅ | 试 bucket=12 或 Beta 分布 |
| Value head | per-source Dense(H)→Dense(1) | ✅ | underfit 时加深 |
| Value clipping | 开启 | ✅ | — |
| GAE | λ=0.95, γ=0.99 | ✅ | — |
| Entropy | 三头独立系数 | ✅ | 用 entropy schedule 防早熟 |
| 初始化 | orthogonal √2 / 0.01 / 1.0 | ✅ | — |
| KL 早停 | 1.5×target_kl | ✅ | self-play 非平稳，必要 |
| Per-source value | True（推荐） | ✅ | 也可作为 ablation 跑一次标量 value |

---

## 6. 推荐的 4 步迭代路径

如果你想"把当前 actor-critic 再推一步"，按风险从低到高：

1. **Critic 加深 1 层**：`Dense(H) → ReLU → Dense(H) → ReLU → Dense(1)`。
   - 风险：极低；好处：value underfit 时立竿见影。
2. **Target head 加 content bias**：`Dense(planet_out, 1)` 加到 target_logits。
   - 风险：低；好处：target 分布在弱 query 时更稳。
3. **Amount → bucket=12 等比划分**（如 0.2, 0.3, …, 1.0 + 全部）。
   - 风险：中；改动 amount_mask 与 ship_buckets 同步。
4. **试一次标量 value（cls_out → MLP → scalar）**。
   - 风险：中；改动 PPO advantage 广播逻辑。

每步先跑 200~400 update，对比 `win_rate / value_loss / entropy_*` 三项决定是否保留。

---

## 7. 一个干净的 Actor-Critic 接口模板

如果未来要重写或抽成可复用模块，建议接口长这样：

```python
class ActorCritic(nn.Module):
    encoder: Encoder            # Transformer + CNN，输出 cls_out, planet_out, source_ctx
    fire_head: FireHead         # context_h -> [B, S]
    target_head: PointerHead    # context_h, planet_out, rel_feat, mask -> [B, S, P]
    amount_head: AmountHead     # context_h, selected_key -> [B, S, Bk]
    value_head: ValueHead       # context_h (or cls_out) -> [B, S] or [B]

    def __call__(self, obs, selected_target=None):
        cls_out, planet_out, source_ctx = self.encoder(obs)
        ctx = build_context(self_h, global_h, source_ctx, cls_out, map_h)

        fire   = self.fire_head(ctx)
        target = self.target_head(ctx, planet_out, rel_feat, candidate_mask)
        amount = self.amount_head(ctx, gather_target(planet_out, selected_target))
        value  = self.value_head(ctx)

        return PolicyOutput(fire, target, amount, value)
```

把每个头单独成 `nn.Module`，方便：

- 改一个头不影响其他头的 ckpt 兼容性
- 单独做单元测试（喂随机 ctx，看输出 shape / logits 范围）
- 未来加 distributional value、Beta amount、multi-head pointer 都只改对应模块

---

## 8. 总结

> **共享 Trunk（Transformer + CNN） + Per-source Context + 三头分解 Actor + per-source Value 头**
> 是当前问题（结构化动作 + 排列对称多 source + 物理约束）下的合理选择。
>
> 短期改进重点放在 **critic 容量** 与 **target 头 content bias**；
> 中期可以试 **Beta 分布的 amount 头** 与 **标量 value ablation**。
