# Autoregressive Actor —— AlphaStar 风格设计文档

> 目标：把当前 "per-source 三头独立采样" 的 actor，替换成一条 **autoregressive action sequence**：
> `(src_1, tgt_1, amt_1) -> (src_2, tgt_2, amt_2) -> ... -> END`，最多 `K` 个 token，遇 END 提前停止。
> 灵感来源：AlphaStar 的 *function head + arguments* 自回归结构（Vinyals et al. 2019, "Grandmaster level in StarCraft II using multi-agent RL"）。

---

## 0. TL;DR

| 维度 | 现状（per-source） | 新设计（autoregressive） |
|---|---|---|
| Action shape | `[B, 2, 45, 3]`，每个 source 独立采 (fire, target, amount) | `[B, 2, K, 3]`，sequence-of-actions，END 早停 |
| Decision dependencies | source 之间完全独立 | 第 t 个 action 看到前 t-1 个的 src/tgt/amt |
| 表达力 | 难以表达"集中调度"、"协同 multi-source attack" | 可学组合策略；动作数自适应 |
| Encoder cost | 1× transformer / state | 不变（encoder 仍 1×/state） |
| Decoder cost | 单次 FFN per source | K 步 decoder（K 通常 ≤ 16） |
| PPO action | per-source token | per-state action sequence；token-level ratio |
| Critic | per-source value（context-based） | 单一 `V(s)` from CLS（更干净） |

**核心论点**：Orbit Wars 的最优策略经常是"用 3 颗星集中打掉一颗敌方"，per-source 独立 Bernoulli 无法表达"我开火当且仅当队友也开火"。Autoregressive head + END token 直接对齐 AlphaStar 已经验证过的范式，并且在工程上完全可在 JAX/Flax + PPO 内闭合（固定 K 即可 jit）。

---

## 1. 现状回顾

### 1.1 数据流（grouped transformer 路径）

```
EnvState
  encode_batch (features_jax)
    planet_features    [B, P, SELF=18]
    candidate_features [B, P, P+1, CAND=26]
    global_features  [B, P, GLOBAL=11]
    fleet_features   [B, P, K_fleet=32, FLEET=16]
    map_features     [B, P, 6, 8, 8]
    candidate_mask   [B, P, P+1]
    amount_mask      [B, P, P+1, BUCKETS=8]
    source_mask      [B, P]
  model_jax.ActorCritic._forward_transformer_grouped
    encoder (1×): planet_tokens + fleet_tokens + CLS  -> cls_out, planet_out
    per source: context = [self_h, global_h, source_ctx, cls_per_source, map_h]
    heads (per source, parallel):
      fire_logits     [B, P]
      target_logits   [B, P, P+1]
      amount_logits   [B, P, BUCKETS]
      value           [B, P]
```

### 1.2 痛点

1. **No cross-source coordination**：每个 source 的 fire 是独立 Bernoulli，模型无法表达 "如果 P3 打 P7，那 P5 也应该打 P7"。这是 PPO 在 self-play pool 训练到 ckpt400+ 后 plateau 的核心限制。
2. **Per-source value**：`V(s, source)` 的"语义"不干净，advantage 信号在不发射的 source 上是噪声（虽然 `policy_mask` 已经过滤了一部分）。
3. **熵分布不可控**：fire entropy 必须 per-source 独立调，但 source 数随 episode 阶段变化，导致熵尺度漂移（见 `TRANSFORMER_RL_STABILITY_PLAN.md` P0 中提到的 entropy 监控难点）。
4. **`max_actions` 浪费**：env 已经接受 `[B, 2, max_actions=64, 3]`，目前 actor 一次最多产 `max_planets=45` 个动作，但其中绝大多数是 fire=0。Decoder 范式天然只输出"想发射的那几条"。

---

## 2. 新架构总览（AlphaStar-style）

```
encode_batch (unchanged)
        │
        ▼
┌────────────────────────────────────┐
│ Shared Encoder (1× per state)      │   ← 不变
│  - planet tokens + fleet tokens    │
│  - Transformer encoder × N         │
│  - cls_out, planet_out             │
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ Autoregressive Action Decoder      │   ← NEW
│  K 步 unroll, mask 早停, 共享权重  │
└────────────────────────────────────┘
        │
   ┌────┴────┬────────┬─────────┬────────┐
   ▼         ▼        ▼         ▼        ▼
 SrcHead  EndHead  TgtHead   AmtHead  ValueHead
(pointer) (binary) (pointer) (cat-8)  (scalar, from cls_out)
```

### 2.1 解码循环（per env state, K 步）

```python
# 初始：仅 encoder 输出
state_emb = MLP(cls_out)                      # [B, H]    全局摘要
h_dec     = init_decoder_state(state_emb)     # [B, H]    decoder 隐状态
prev_tok  = LEARNED_BOS                       # [B, H]    起始 token

remaining_ships = planet_ships.copy()         # [B, P+1]  动态预算
end_emitted     = zeros(B, dtype=bool)        # 早停 mask

actions = []     # 收集 (src, tgt, amt) tokens
end_log = []
src_log, tgt_log, amt_log = [], [], []

for t in range(K):
    # 1. 把 prev token + state_emb 喂给 decoder cell
    h_dec = DecoderCell(prev_tok, state_emb, h_dec)

    # 2. END head (Bernoulli)：是否提前停止
    end_logit = EndHead(h_dec)                # [B]
    end_t     = sample(end_logit)
    end_emitted = end_emitted | end_t

    # 3. Src head: pointer over my-planet tokens (+ no-op slot)
    src_logits = pointer(h_dec, planet_out, key_kind="src")
    src_logits = mask_invalid_sources(src_logits, remaining_ships, source_mask)
    src_logits = mask_if_end(src_logits, end_emitted)
    src_t      = sample(src_logits)

    # 4. Tgt head: pointer over all alive planets, 输入加入 src embedding
    tgt_query  = h_dec + Wsrc(planet_out[src_t]) + Erel(src_t, *)
    tgt_logits = pointer(tgt_query, planet_out, key_kind="tgt")
    tgt_logits = mask_invalid_targets(tgt_logits, src_t)
    tgt_t      = sample(tgt_logits)

    # 5. Amount head: 分类 over BUCKETS, 输入加入 (src, tgt)
    amt_query  = h_dec + Wsrc(planet_out[src_t]) + Wtgt(planet_out[tgt_t])
    amt_logits = AmountHead(amt_query)        # [B, 8]
    amt_logits = mask_unaffordable(amt_logits, remaining_ships[src_t])
    amt_t      = sample(amt_logits)

    # 6. 更新预算 & 收集
    ships_sent = SHIP_RATIOS[amt_t] * remaining_ships[src_t]
    remaining_ships = remaining_ships.at[src_t].add(-ships_sent)
    actions.append((src_t, tgt_t, amt_t))
    src_log.append(...); tgt_log.append(...); amt_log.append(...); end_log.append(...)

    # 7. 下一个 prev_tok = token embedding
    prev_tok = TokEmb(src_t, tgt_t, amt_t)

# value 来自 encoder 端 cls_out（不在 decoder 里求）
value = ValueHead(cls_out)
```

**关键点**：
- 整个循环 K 是常数 → `jax.lax.scan` jit 友好，shape 完全静态。
- `end_emitted` 一旦置 1，后续 step 的 (src, tgt, amt) 都被 mask 为 "no-op token"，log-prob 不计入。
- Decoder cell 可以选 **GRU**（轻）或 **1-layer Transformer decoder block with causal self-attention**（更强，但 K 小所以代价低）。第一版用 GRU。

### 2.2 与 env 的接口

- env `step_batch` 接受 `[B, 2, max_actions, 3]`，**不需要任何改动**。
- Actor 输出 `(src_id, angle_rad, ships)` 三元组列表，长度 K，未发射的 slot 填 `(0, 0, 0)`，env 已经会跳过 `from_planet_id <= 0` 的 slot（见 `_launch_all`）。
- 配置 `max_actions = K`（建议 K=16，覆盖 99%+ 的 episode 步内动作密度）。

---

## 3. 概率模型与 PPO Loss

### 3.1 联合 log-prob

定义第 t 步的 token log-prob：

$$
\log \pi_t = \log p(\text{end}_t) + (1-\text{end}_t)\big[\log p(\text{src}_t) + \log p(\text{tgt}_t) + \log p(\text{amt}_t)\big]
$$

整条 sequence：

$$
\log \pi(a_{1:K} \mid s) = \sum_{t=1}^{K} \log \pi_t \cdot m_t
$$

其中 `m_t = 1` 当 token 在 END 发出**之前**（含发出 END 的那一步），END 之后的 m_t = 0。

### 3.2 PPO loss（推荐：token-level ratio + state-level advantage）

```
A_t shared across all tokens of step t          # state-level GAE
ratio_{t,i} = exp(logp_new_{t,i} - logp_old_{t,i})   # token i in step t
L_policy = - mean_{t,i in valid_tokens} min(
    A_t * ratio_{t,i},
    A_t * clip(ratio_{t,i}, 1-ε, 1+ε)
)
```

**为什么 token-level 而不是 sequence-level**：
- 序列长度可变 → sequence-level ratio = `exp(sum log_ratios)` 方差极大，PPO clip 几乎总是 saturate。
- AlphaStar 也是 per-argument PPO clip。
- 风险：相邻 token 高度相关（同一 step 内 src/tgt/amt），ratio 不独立 → 比 IID 假设方差大一点，但 PPO clip 本身能容忍。

**Value loss**：和现在一样 clipped MSE，但只 per-state（不再 per-source）：

```
V_pred = old_value + clip(V_new - old_value, -ε, ε)
L_value = (returns - V_pred)^2
```

### 3.3 Entropy bonus

分四路：

```
H_end    = mean over all decoded steps        # 包括 END 之后的 step？只算 pre-END
H_src    = mean over pre-END steps with end=0
H_tgt    = mean over pre-END steps with end=0
H_amt    = mean over pre-END steps with end=0

ent_bonus = c_end * H_end + c_src * H_src + c_tgt * H_tgt + c_amt * H_amt
```

每路独立 schedule（参考 `TRANSFORMER_RL_STABILITY_PLAN.md` 的 per-head entropy 监控）。

### 3.4 关键工程细节：mask 同步

**最容易出 bug 的地方**：rollout 时用了 `remaining_ships` 动态 mask，PPO 重算 logp 时必须用**完全相同**的 mask，否则 ratio 偏差，KL/clipfrac 爆炸。

解决方案：
1. Rollout 时记录每个 step 的 **实际 mask**：`src_mask_t, tgt_mask_t, amt_mask_t, end_mask_t`（bool 张量）。
2. 这些 mask 完全可以从 `remaining_ships` 推导，但**预算的演化依赖采样到的 src/amt**，不可避免地需要在 PPO 里"重放"。
3. 由于 src/amt **已经在 rollout 里采样过并存了下来**，重放是**确定性**的：用存好的 `(src_t, amt_t)_{1..K}` 重算 `remaining_ships_{1..K}`，再生成 mask。这一步用 `jax.lax.scan` 走 K 步即可。

> ⚠️ 这是和现状最大的实现差异：**logp 重算不能 vmap 掉时间维**，必须 scan。但 K ≤ 16，单次 minibatch 内 scan 的开销可控。

---

## 4. 网络模块详细规格

> H = hidden_size = 128（沿用），P = max_planets = 45，B_envs = num_envs

### 4.1 Encoder（不变）

复用 `_forward_transformer_grouped` 的前半段，输出：

- `cls_out`: `[B, H]`
- `planet_out`: `[B, P+1, H]`
- 也保留 `relative_h_table`（per source-target 的 source-relative MLP）以备 tgt head 使用

### 4.2 Decoder Cell

```python
class DecoderCell(nn.Module):
    hidden_size: int
    def setup(self):
        self.gru = nn.GRUCell(features=self.hidden_size)
        self.in_proj = nn.Dense(self.hidden_size)
    def __call__(self, prev_tok, state_emb, h):
        # prev_tok: [B, H], state_emb: [B, H], h: [B, H]
        x = self.in_proj(jnp.concatenate([prev_tok, state_emb], axis=-1))
        h_new, _ = self.gru(h, x)
        return h_new
```

**升级路径**：如果 GRU 表达力不足，换成 1-layer Transformer decoder block（causal self-attention over 已生成 tokens，cross-attention 到 `[cls_out + planet_out + fleet_out]`）。

### 4.3 Heads

| Head | 输入 | 输出 | 备注 |
|---|---|---|---|
| `EndHead` | `h_dec` | `[B]` (logit) | 初始化时 bias = -2.0，避免立刻 END |
| `SrcHead` | `h_dec`, `planet_out` | `[B, P+1]` | pointer attention; mask 非我方/船数不足/已 END |
| `TgtHead` | `h_dec + Wsrc(planet_out[src])`, `planet_out`, `relative_h_table[src]` | `[B, P+1]` | pointer with source-relative key mixing（沿用现有 grouped transformer 的设计） |
| `AmountHead` | `h_dec + Wsrc(p_src) + Wtgt(p_tgt)` | `[B, BUCKETS=8]` | MLP; mask 不够船的档位（沿用 `amount_mask`） |
| `ValueHead` | `cls_out` | `[B]` | 与 decoder 解耦 |

### 4.4 Token embedding（feedback）

```
TokEmb(src, tgt, amt) = LayerNorm(
    planet_out[src] +
    planet_out[tgt] +
    AmountEmb[amt] +
    StepPosEmb[t]
)
```

`StepPosEmb` 是 K 个可学位置嵌入；让 decoder 知道当前是第几步。

END token 的 feedback 用一个 learned `END_TOK_EMB`。

### 4.5 Mask 规则

| 步骤 | Mask 内容 | 数据来源 |
|---|---|---|
| `end_mask_t` | always True（END 永远可选） | — |
| `src_mask_t` | source 必须我方存活 ∧ `remaining_ships[src] >= MIN_LAUNCH` | `source_mask`, `remaining_ships_t` |
| `tgt_mask_t` | target 是任意存活星球（含敌/中立/友），但 ≠ src；no-op slot 0 不允许（END 替代了 no-op） | `candidate_mask`, `src_t` |
| `amt_mask_t` | 至少要送出 `MIN_LAUNCH` 船，且 ≤ 当前 `remaining_ships[src]` | `amount_mask`（基于动态预算重算） |
| post-END | 全部 token 强制取 0；logp 设为 0；不计入 loss/entropy | `end_emitted` |

**关键简化**：no-op slot 在新架构里被 END 替代。`fire=False` 不再是 per-source 决定，而是"我已经发了 t 个动作，不想再发了"由 END 表达。

### 4.6 物理层处理策略（V1：采样后验证，不做 hard mask）

第一版**不把完整物理可达性前置成 action mask**。原因：

- 当前 `aim_with_prediction_jax` 的主要成本来自 60 点粗搜索 + 12 次二分 + 旋转星球/彗星位置预测，而不是最后的碰撞判断。
- 实测在 `B=64` 下，`T=44 × A=8` 的 exact sun-only 仍约 50ms/decoder step，完整物理约 53ms/decoder step，省不了多少。
- 全量 `src × target × amount` hard mask 会把 rollout 和 PPO update 的计算量放大到不可承受。

因此 V1 采用和当前 per-source 路径一致的思路：

```python
# 采样得到 (src_t, tgt_t, amt_t) 后，才做物理验证
ships_t = dynamic_ship_bucket(remaining_ships[src_t], amt_t)
angle_t, aim_hit_t = aim_with_prediction_jax(src_t, tgt_t, ships_t)

executed_t = (
    token_valid_t
    & (~end_action_t)
    & src_valid_t
    & tgt_valid_t
    & amt_valid_t
    & aim_hit_t
)

action_slot_t = where(executed_t, [src_id, angle_t, ships_t], [0, 0, 0])
remaining_ships = where(executed_t, remaining_ships.at[src_t].add(-ships_t), remaining_ships)
invalid_t = token_valid_t & (~end_action_t) & (~executed_t)
prev_tok = where(
    end_action_t, END_TOK_EMB,
    where(executed_t, TokEmb(src_t, tgt_t, amt_t), INVALID_TOK_EMB),
)
```

关键约束：

1. **invalid token 不进入 `src/tgt/amt` policy loss**。否则策略 logp 对应的是 `(src,tgt,amt)`，环境执行却是 no-op，会污染 PPO ratio。
2. **invalid token 不扣预算**。预算必须反映真实进入环境的动作。
3. **invalid token 的 decoder feedback 用 `INVALID_TOK_EMB`**，不要继续喂原 `(src,tgt,amt)`，否则后续 token 会以为刚才已经发射成功。END token 仍使用 `END_TOK_EMB`。
4. **END head 仍正常学习**。END 决定的是是否停止发动作；物理 invalid 只是参数 token 的执行失败。

V2 可选优化：如果 `invalid_action_rate` 长期很高，再做局部 hard mask：

```text
选定 src 后，只对 top-k targets × amount buckets 做完整物理验证
```

但不建议第一版直接做所有敌/中立目标的全量物理 mask。

---

## 5. Rollout 数据结构

```python
class AutoRollout(NamedTuple):
    # encoder 输入（per state）
    planet_features:      [T, B, P, SELF]
    candidate_features: [T, B, P, P+1, CAND]
    global_features:    [T, B, P, GLOBAL]
    map_features:       [T, B, P, C, H, W]
    fleet_features:     [T, B, K_fleet, FLEET]
    fleet_mask:         [T, B, K_fleet]
    candidate_mask:     [T, B, P+1]
    source_mask:        [T, B, P]
    initial_ships:      [T, B, P+1]    # 第 t 个 env step 开始时各星球可用船数（仅我方）

    # decoder 序列（per token，K 步）
    end_action:   [T, B, K]            # bool
    src_action:   [T, B, K]            # int
    tgt_action:   [T, B, K]            # int
    amt_action:   [T, B, K]            # int

    # 物理执行结果（V1：采样后验证）
    executed:     [T, B, K]            # bool，真实进入 env 的动作 token
    invalid:      [T, B, K]            # bool，非 END 但物理层丢弃/变 no-op

    # token-level log-prob（4 路独立，便于 token-level ratio）
    old_logp_end: [T, B, K]
    old_logp_src: [T, B, K]
    old_logp_tgt: [T, B, K]
    old_logp_amt: [T, B, K]

    # token-level valid mask
    token_valid:  [T, B, K]            # END 之前的 step（含发 END 那步）

    # state-level
    old_value: [T, B]
    reward:    [T, B]
    done:      [T, B]
```

**注意 reward 形状**：从 `[T, B, P]` 退化为 `[T, B]`。因为现在每个 env state 只产一个 trajectory transition，per-source 维度消失。这同时简化了 GAE。

---

## 6. PPO Update 细节

### 6.1 重新计算 logp

```python
def recompute_logp(params, batch):
    # 1. encoder forward (一次)
    cls_out, planet_out = encode(params, batch)
    state_emb = MLP(cls_out)

    # 2. K 步 scan，重放 decoder
    def scan_step(carry, t):
        h_dec, remaining_ships, prev_tok = carry
        h_dec = DecoderCell(prev_tok, state_emb, h_dec)

        # 用 batch 里存好的 actions 算 logp
        end_logit = EndHead(h_dec)
        logp_end  = bernoulli_logp(end_logit, batch.end_action[:, t])

        src_logits = src_pointer(h_dec, planet_out)
        src_logits = mask_src(src_logits, remaining_ships, batch.source_mask)
        logp_src   = categorical_logp(src_logits, batch.src_action[:, t])

        tgt_logits = tgt_pointer(h_dec, planet_out, batch.src_action[:, t])
        tgt_logits = mask_tgt(tgt_logits, batch.src_action[:, t])
        logp_tgt   = categorical_logp(tgt_logits, batch.tgt_action[:, t])

        amt_logits = amount_head(h_dec, batch.src_action[:, t], batch.tgt_action[:, t])
        amt_logits = mask_amt(amt_logits, remaining_ships, batch.src_action[:, t])
        logp_amt   = categorical_logp(amt_logits, batch.amt_action[:, t])

        # 更新预算（用存好的 executed/src/amt）
        ships_sent = SHIP_RATIOS[batch.amt_action[:, t]] * remaining_ships[arange(B), batch.src_action[:, t]]
        ships_sent = where(batch.executed[:, t], ships_sent, 0.0)
        remaining_ships = remaining_ships.at[arange(B), batch.src_action[:, t]].add(-ships_sent)

        prev_tok_raw = token_embed(batch.src_action[:, t], batch.tgt_action[:, t], batch.amt_action[:, t], t)
        prev_tok = where(
            batch.end_action[:, t, None],
            END_TOK_EMB,
            where(batch.executed[:, t, None], prev_tok_raw, INVALID_TOK_EMB),
        )
        return (h_dec, remaining_ships, prev_tok), (logp_end, logp_src, logp_tgt, logp_amt,
                                                    entropy_end, entropy_src, entropy_tgt, entropy_amt)

    init = (zeros(B, H), batch.initial_ships, BOS_EMB)
    _, (logp_end, logp_src, logp_tgt, logp_amt, H_end, H_src, H_tgt, H_amt) = \
        jax.lax.scan(scan_step, init, jnp.arange(K))
    # logp_* shape: [K, B]
    return logp_end, logp_src, logp_tgt, logp_amt, H_*
```

### 6.2 Loss

```python
# token-level ratios
log_ratio_end = (logp_end_new - logp_end_old) * token_valid * 1                # END token 始终算
log_ratio_src = (logp_src_new - logp_src_old) * executed                       # 只有真实执行的参数 token 才算
log_ratio_tgt = ... ; log_ratio_amt = ...

# 数值安全（沿用现有 clip(-5, log(10))）
def safe_ratio(lr): return jnp.exp(jnp.clip(lr, -5.0, jnp.log(10.0)))

# 每个 token-stream 独立 PPO clip，然后求和
def ppo_token_loss(adv, log_ratio, mask):
    ratio = safe_ratio(log_ratio)
    pg1 = adv * ratio
    pg2 = adv * jnp.clip(ratio, 1 - ε, 1 + ε)
    return -(jnp.minimum(pg1, pg2) * mask).sum() / jnp.maximum(mask.sum(), 1.0)

# 把 state-level advantage broadcast 到 K
adv_K = adv[:, None].repeat(K, axis=1)   # [B, K]

policy_loss = (ppo_token_loss(adv_K, log_ratio_end, end_valid_mask)
             + ppo_token_loss(adv_K, log_ratio_src, executed_mask)
             + ppo_token_loss(adv_K, log_ratio_tgt, executed_mask)
             + ppo_token_loss(adv_K, log_ratio_amt, executed_mask)) / 4.0
```

### 6.3 KL 早停

四路 KL 加权平均，沿用现有 `target_kl = 0.05` 阈值。**额外**：加 `approx_kl_end` 单独监控——如果 END head 崩了，整条 sequence 全废。

---

## 7. 监控指标（接住 `TRANSFORMER_RL_STABILITY_PLAN.md`）

新增：

| 指标 | 含义 |
|---|---|
| `seq_len_mean` | 每个 state 实际发出动作数（END 出现位置） |
| `seq_len_p50` / `p95` | 分位数，监控是否塌缩到 K=1（一发 END）或撑到 K |
| `entropy_end` / `_src` / `_tgt` / `_amt` | per-head entropy |
| `approx_kl_end` / `_src` / `_tgt` / `_amt` | per-head KL |
| `clipfrac_end` / `_src` / `_tgt` / `_amt` | per-head clipfrac |
| `budget_exhaust_rate` | 预算耗尽（无可用 src）触发被动 END 的比例 |
| `end_bias` | EndHead 当前 bias（用于诊断"过早 END"问题） |
| `repeated_src_rate` | 同一 state 内同 src 出现多次的比例 |
| `invalid_action_rate` | 非 END token 被物理层丢弃的比例 |
| `executed_action_rate` | 非 END token 中真实进入 env 的比例 |

---

## 8. 推荐超参 & 训练流程

### 8.1 第一版 (PoC, 1-2 天)

| 参数 | 值 |
|---|---|
| `K` (max decoder steps) | 16 |
| Decoder type | GRU |
| `H` | 128（同 encoder） |
| `END_bias_init` | -2.0 |
| `c_end` (END entropy coef) | 0.02 |
| `c_src` | 0.01 |
| `c_tgt` | 0.01 |
| `c_amt` | 0.005 |
| LR | 与现状相同，3e-4 warmup_cosine |
| PPO epochs | 4 |
| clip ε | 0.2 |
| target_kl | 0.05 |
| minibatch_size | `4096 / 4`（per-state batch 缩水到 1/45） |

### 8.2 训练阶段

1. **Phase A — Imitation warmup（强烈推荐）**：用现有强基线 policy（例如 ckpt400 legacy per-source，或后续更强 transformer checkpoint）跑 ~1k 局，把每 env step 的 `[fire, target, amount]_{1..45}` 序列化（按 source_id 排序，fire=True 的留下）+ 末尾追加 END。BC loss 在 decoder 上预训练 ~5 epochs。
   - 这一步**直接解决 END 冷启动**，并继承现有 policy 的胜率。
2. **Phase B — PPO self-play**：用 BC 初始化，沿用现有 `cfg_jax_pool_0514_resume.yaml` 的对手池设置。前 100 updates 固定 `c_end=0.05`，之后线性衰减到 0.01。
3. **Phase C — Curriculum on K**：初期可以 K=8（强制简短序列，更容易学），稳定后扩到 K=16。

### 8.3 Ablation 清单

- [ ] GRU vs 1-layer Transformer decoder（K=16 时 transformer 应该更强）
- [ ] State-level value vs multi-query value（attend `[cls + planet_out]`）
- [ ] Token feedback：是否需要 `tgt_embed`（也许只 src+amt 就够）
- [ ] Sequence-level ratio vs token-level ratio
- [ ] BC warmup 5 epoch vs 20 epoch
- [ ] `c_end` schedule 形式

---

## 9. 风险 & 缓解

| 风险 | 描述 | 缓解 |
|---|---|---|
| END 塌缩 | 训练初期 END head 立刻输出 1，每步只发 0 个动作 | END bias init -2；BC warmup；前期高 `c_end` |
| END 永不触发 | END logit 漂到极负，K 步全用完，浪费且训练慢 | 监控 `seq_len_p50`；如果一直 > K-2，加 `c_end` |
| Mask 不一致 | rollout / recompute 的 remaining_ships 路径分叉，ratio 偏 | 单元测试：随机 seed 下 `logp_new == logp_old` 在 same params 时 1e-6 一致 |
| 物理 invalid 过多 | 大量 token 被 `aim_with_prediction_jax` 丢弃，导致有效梯度少、探索浪费 | V1 先监控 `invalid_action_rate`；若长期 >20%-30%，再加 top-k target 物理 hard mask 或 sun-risk logit bias |
| Decoder 状态污染 | invalid token 仍用原 `(src,tgt,amt)` feedback，后续 token 以为发射成功 | invalid 后使用 `INVALID_TOK_EMB`，预算不扣，PPO 重放同样按 `executed` 更新 |
| Scan 开销 | K=16 的 lax.scan 在每个 PPO minibatch 都跑一次 | 实测：H=128, K=16 单次 ≈ 0.3ms/state on A100，可接受 |
| 优势 broadcast 偏置 | state-level adv 给所有 token 用，长序列方差大 | 用 PPO clip 兜底；监控 `clipfrac_*` per head |
| Repeated src | 同一 src 被选 K 次，预算耗尽前 mask 没生效 | mask 由 remaining_ships 推导，理论上 `MIN_LAUNCH` 用尽后自动屏蔽；加 `repeated_src_rate` 监控 |

---

## 10. 文件改动清单

| 文件 | 改动范围 | 说明 |
|---|---|---|
| `model_jax.py` | **大改** | 新增 `AutoregressiveActorCritic` Module；保留 `ActorCritic` 作为 fallback |
| `features_jax.py` | 小改 | `EncodedBatch` 不再需要 per-source `planet_features`（因为不再 per-source 决策）；可以删 P 维度但建议先保留并加 `initial_ships: [B, P+1]` 字段 |
| `ppo_jax.py` | **大改** | `Rollout` -> `AutoRollout`；`ppo_update` 重写 logp 重算逻辑（scan）；`log_prob_entropy` 替换为 `recompute_logp` |
| `train_jax.py` | 中改 | `policy_sample` 替换为 `policy_sample_autoregressive`（K 步 scan）；`actions_from_decisions` 改成处理 K-step sequence；config 加 `model.actor_type: autoregressive` |
| `eval_opponents/opponents.py` | 小改 | 透传新的 sample 函数 |
| `train_opponents/opponents.py` | 小改 | 同上 |
| `bench_jax.py` | 小改 | sample 函数切换 |
| `agent.yaml` / 配置 | 加字段 | `actor_type`, `max_action_tokens=16`, `decoder_type`, `end_bias_init`, `c_end`, etc. |
| `frozen/` 旧 ckpt | 兼容 | 旧 ckpt 不兼容新模型；保留 `ActorCritic` legacy 路径用于对战池 |

**向后兼容**：
- `ActorCritic` 不删，作为 opponent pool 中旧 checkpoint 的推理网络。
- 训练入口 `train_jax.py` 根据 `cfg.model.actor_type` 二选一构造 actor。
- 对战环节（self-play / opponent eval）必须能让"新 actor"和"旧 actor"互打——这是 baseline 验证的关键。

---

## 11. 实现里程碑

1. **M1 — 静态前向（0.5 天）**
   - 实现 `AutoregressiveActorCritic.__call__`（不带采样，只返回 logits）
   - `parity_smoke` 风格的 shape/finite 测试
2. **M2 — 采样 + env 闭环（1 天）**
   - `policy_sample_autoregressive`，跑通 `bench_jax.py`（随机 init 不报错，FPS 不比旧版差 2x）
3. **M3 — PPO loss + scan logp（1 天）**
   - `recompute_logp` + token-level PPO loss
   - 单元测试：`logp_old == logp_new`（same params）
4. **M4 — BC warmup（0.5 天）**
   - 用 ckpt400 生成 BC 数据集，跑 5 epoch
   - 验证 BC 后 zero-shot 胜率 ≥ ckpt400 - 5%
5. **M5 — 短程 PPO smoke（1 天）**
   - 1000 updates，监控 `seq_len_mean`、4 路 entropy/KL/clipfrac
   - 期待：胜率不崩；END head 不塌缩
6. **M6 — 长程训练（持续）**
   - 加入 self-play pool；和 ckpt400 / 现有 transformer ckpt 互打

每个里程碑都先看监控指标，再决定下一步。

---

## 12. 参考

- Vinyals et al. 2019, "Grandmaster level in StarCraft II using multi-agent reinforcement learning" — function head + arguments + autoregressive sampling
- Vinyals et al. 2015, "Pointer Networks" — pointer attention for variable-size targets
- AlphaStar supplementary materials —— PPO + V-trace 混合，KL 对 BC 的正则（适合我们的 BC warmup 阶段）
- 项目内：`TRANSFORMER_ARCHITECTURE.md`、`TRANSFORMER_RL_STABILITY_PLAN.md`
