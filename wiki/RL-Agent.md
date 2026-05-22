# RL Agent 设计

> **所属页面**：[Home](Home.md) > RL Agent 设计  
> **位置**：`rl-agent/`

---

## 1. 设计哲学

本 RL Agent 采用 PPO 算法训练，流程为：  
**收集 rollout → 计算 GAE → PPO clipped loss 多轮 minibatch 更新**。

核心设计原则：

- **自回归多命令决策**：每回合由 decoder 连续生成 `END` 或 `(source, target, amount)` 命令
- **条件式三级选择**：先选 source，再基于 source 选 target，最后基于 source-target pair 选派兵量
- **物理计算后置**：网络只选"打谁/派多少"，真实的发射角度和碰撞检测在动作确定后再计算
- **Reward 精细塑形**：终局胜负（主信号）+ 多维局势变化 shaping + 事件驱动奖惩

---

## 2. 目录结构

```
rl-agent/
├── README.md             项目说明
├── agent.yaml            orbit-wars-lab Agent 元信息
├── cfg_jax_*.yaml        JAX 训练配置
├── main.py               Kaggle/Web 推理入口
├── FEATURES.md           特征工程详细文档
└── jax_agent/
    ├── train_jax.py      rollout、评估、日志、checkpoint 主循环
    ├── model_jax.py      AutoregressiveActorCritic 网络
    ├── ppo_jax.py        PPO loss、GAE、参数更新
    ├── features_jax.py   JAX 特征编码与 mask
    ├── fast_env.py       JAX 并行环境
    └── eval_opponents/   评估对手与外部 agent 包装
```

---

## 3. 动作空间

网络不直接输出角度，而是在每个 decoder step 输出一条离散命令。命令可以是结束符，也可以是一组 `(source, target, amount)`：

| 动作维度 | 类型 | 含义 |
|----------|------|------|
| `end` | Bernoulli/Categorical | 当前 step 是否结束本回合命令生成 |
| `source` | Categorical | 从哪个己方星球发兵 |
| `target` | Categorical | 打哪个目标槽位 |
| `amount` | Categorical | 派兵比例档位 |

### 3.1 目标槽位定义

```
target slot 0       = no-op / padding 类目标位
target slot 1..45   = 当前全图所有星球/彗星（按 id 排序，不足则 padding）
```

### 3.2 合法性 Mask

`features_jax.py` 会为 source、target、amount 分别生成 mask：

| Mask | 形状 | 规则 |
|------|------|------|
| `source_mask` | `[B, P]` | 只有己方且有可用兵力的星球可作为 source |
| `candidate_mask` | `[B, P, P + 1]` | 每个 source 只允许合法 target |
| `amount_mask` | `[B, P, P + 1, 8]` | 每个 source-target pair 只允许可执行的派兵比例 |

Decoder 每执行一条命令后会更新该 source 的剩余兵力，后续命令的 source/amount mask 会基于新的剩余兵力重新计算。

---

## 4. 特征工程

当前 JAX 特征由 `jax_agent/features_jax.py` 编码，直接服务于 autoregressive actor。核心张量如下：

| 特征组 | 形状 | 说明 |
|--------|------|------|
| `planet_features` | `[B, P, 26]` | 每个星球自己的状态特征 |
| `relative_features` | `[B, P, P + 1, 8]` | 每个 source 到每个 target 的相对特征 |
| `global_features` | `[B, P, 11]` | 全局游戏状态，模型取 `global_features[:, 0]` |
| `fleet_features` | `[B, P, F, 28]` | 舰队 token 特征 |
| `fleet_mask` | `[B, P, F]` | 舰队 token 是否有效 |
| `map_features` | `[B, P, 6, 8, 8]` | 每个 source 视角的局部地图 |
| `source_mask` | `[B, P]` | 哪些星球可作为 source |
| `candidate_mask` | `[B, P, P + 1]` | 每个 source 的合法 target |
| `amount_mask` | `[B, P, P + 1, 8]` | 每个 source-target 的合法派兵档 |

### 4.1 planet_features

`planet_features` 描述星球自身，不绑定某个 source。它包含坐标、半径、船数、生产力、阵营、旋转/彗星状态、inbound 压力等字段，是 Transformer 中 planet token 的输入。

### 4.2 relative_features

`relative_features[src, tgt]` 描述 source 到 target 的关系，例如相对位移、距离、相对兵力、最快到达时间、目标先验价值等。它不进入 Transformer token，而是在 target head 和 amount head 中加回去，让同一个目标在不同 source 视角下有不同 key。

### 4.3 global / fleet / map

`global_features[:, 0]` 提供全局局势摘要，例如回合进度、双方星球数、总船数、总生产力、舰队压力等。

`fleet_features` 作为舰队 token 输入 Transformer，让模型看到正在飞行的舰队。`map_features` 由 source head 的 CNN 使用，帮助 source 选择感知局部空间结构。

---

## 5. 网络结构

当前 JAX 路线使用 `jax_agent/model_jax.py` 中的 `AutoregressiveActorCritic`。模型先用 Transformer 编码整局状态，再用 GRU decoder 自回归地产生多条命令，每条命令为：

```
END 或 (source, target, amount)
```

默认每回合最多生成 `max_action_tokens = 12` 条命令。

### 5.1 Encoder

星球 token、舰队 token 和 `CLS` token 一起进入 Transformer；全局特征单独走 MLP，再和 `CLS` 输出合成 decoder 的状态 context。

```python
# =====================
# Encoder
# =====================
planet_h = MLP(planet_own_features)      # 每个星球自己的特征
fleet_h  = MLP(fleet_features_global)    # 每支舰队自己的特征

tokens = [CLS, planet_h..., fleet_h...]
encoded = Transformer(tokens)

cls_out = encoded[CLS]
planet_out = encoded[planet_tokens]

# 独立 context
global_ctx = MLP(global_features[:, 0])
state_ctx = MLP(concat(cls_out, global_ctx))
```

`relative_features` 不进入 Transformer，而是单独经过 MLP 得到 `relative_h`，在 target head 和 amount head 中补回 source-target 的相对距离、方向、攻防关系等信息。

### 5.2 Decoder

Decoder 每一步先用上一步 token 和 `state_ctx` 更新 GRU hidden state，然后依次输出 source、target、amount。

```python
# =====================
# Decoder
# =====================
# h ：GRU decoder 当前步的 hidden state。
h = GRU(prev_tok, state_emb)

# src head
src_cnn = cnn(src_map_feature)
src_key_proj = MLP(concat(src_cnn, planet_out))
src_logits = pointer(
  query = MLP(h),
  keys = src_key_proj
)

# tgt head
src_h = planet_out[src]
relative_h = MLP(relative_features[src, :])
tgt_logits = pointer(
  query = MLP(h + src_h), 
  keys = MLP(planet_out) + relative_h
)

# amount head
tgt_h = planet_out[tgt]
pair_h = relative_h[tgt]
amount_logits = MLP(concat(h, src_h, tgt_h, pair_h))
```

**设计细节**：
- source head 使用 `source_mask` 和当前剩余兵力 mask，不能发兵的星球不会被选中
- target head 使用 `candidate_mask[src]`，只允许合法目标
- amount head 使用 `amount_mask[src, tgt]` 和剩余兵力 mask，避免选择不可执行的派兵比例
- decoder 会在每条已执行命令后更新 source 剩余兵力，因此同一回合后续命令能感知前面已经派出的兵

### 训练 vs 推理

| 阶段 | 动作序列 | 选择方式 |
|------|----------|----------|
| 训练 rollout | `END / source / target / amount` 序列 | 按策略分布采样 |
| PPO 更新 | replay rollout 中保存的 token | 重算 log-prob、entropy、value |
| Kaggle/eval/网页推理 | 同一套 decoder | 通常使用 deterministic/argmax |

---

## 6. Reward 设计

每步总奖励由以下分量线性求和：

$$
r_t = r_{\text{terminal}} + r_{\text{shaped}} + r_{\text{neutral}} + r_{\text{enemy}} + r_{\text{lose}} + r_{\text{wasted}} + r_{\text{time}} + r_{\text{launch}}
$$

### 6.1 终局 Reward（`terminal_weight`）

游戏结束时触发，由环境返回的 `reward` 字段乘以权重：

| 结果 | 原始值 | 实际权重 |
|------|--------|----------|
| 获胜 | `+1` | `× terminal_weight` |
| 失败 | `-1` | `× terminal_weight` |
| 平局/异常 | `0` | — |

### 6.2 局势变化 Shaping（`shaped_reward`）

每步根据己方局势快照的差值给出连续奖励（裁剪至 `±step_clip`）：

| 局势变量 | 权重参数 | 说明 |
|----------|----------|------|
| 星球上船数变化 | `ships_weight` | 鼓励积累驻守兵力 |
| 总产能变化 | `production_weight` | 鼓励占领高产能星球 |
| 己方星球数变化 | `planet_weight` | 鼓励扩张版图 |

`step_clip`（默认 0.20）将单步 shaping 幅度压住，防止密集奖励淹没终局胜负。

### 6.3 事件驱动奖惩

| 分量 | 权重参数 | 触发条件 | 说明 |
|------|----------|----------|------|
| **中立星球占领奖励** | `neutral_capture_weight` | 中立星球变为己方 | 早期（前 `neutral_capture_early_steps` 步）按 ETA 指数衰减加分，鼓励快速扩张 |
| **敌方星球占领奖励** | `enemy_capture_weight` | 敌方星球变为己方 | 加分 = `weight × (1 + production)`，高产能星球价值更高 |
| **丢失星球惩罚** | `lose_planet_penalty` | 己方星球变为非己方 | 惩罚 = `−penalty × (1 + production)`，保护高产能星球 |
| **无效攻击惩罚** | `wasted_attack_penalty` | 己方舰队消失但目标未被占领 | 惩罚 = `−penalty × ships`，减少无意义消耗 |
| **时间惩罚** | `time_penalty` | 每步固定扣分 | 鼓励尽快决策，减少拖延 |
| **发射惩罚** | `launch_penalty` | 每发射一支舰队扣分 | 惩罚 = `−penalty × 发射数`，抑制盲目乱发 |

> 默认配置中 `neutral_capture_weight`、`enemy_capture_weight`、`lose_planet_penalty`、`wasted_attack_penalty`、`time_penalty` 均为 `0.0`；`launch_penalty` 默认为 `0.002`（小惩罚以抑制盲目乱发），各权重可通过 YAML 配置文件调整。

---

## 7. PPO 训练细节

### 7.1 Loss 函数

$$
\mathcal{L} = \mathcal{L}_{policy} + c_v \cdot \mathcal{L}_{value} - c_e \cdot \mathcal{H}
$$

| 分量 | 说明 |
|------|------|
| $\mathcal{L}_{policy}$ | PPO clipped objective |
| $\mathcal{L}_{value}$ | return 与 critic value 的 MSE（支持 value clipping） |
| $\mathcal{H}$ | 策略熵（含 end / source / target / amount），鼓励探索 |
| `approx_kl` | 监控新旧策略距离，超过阈值提前停止本轮更新 |

### 7.2 GAE 计算

在 `jax_agent/ppo_jax.py` 中：

$$
A_t = \delta_t + \gamma \lambda \cdot A_{t+1}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

> 每个环境步对应一段自回归动作序列，reward 仍按环境步计算，PPO replay 时用 rollout 中保存的 token 重新计算整段序列的 log-prob。

### 7.3 PPO 中的 log-prob

当前实现不会先把整段动作序列合成一个总 `logp_seq` 再做 PPO ratio，而是保存并重算四类 token log-prob：

$$
\log \pi_{end}(e_t | h_t)
$$

$$
\log \pi_{src}(src_t | h_t),\quad
\log \pi_{tgt}(tgt_t | h_t, src_t),\quad
\log \pi_{amt}(amt_t | h_t, src_t, tgt_t)
$$

其中 `end` 对所有 `token_valid` step 参与 loss，`src/tgt/amt` 只在 `executed=True` 的 step 参与 loss。`ppo_jax.py` 分别计算 `end/src/tgt/amt` 的 PPO clipped loss，然后取平均：

```python
policy_loss = (l_end + l_src + l_tgt + l_amt) / 4.0
```

因此从概率建模角度看，一条已执行命令的 log-prob 可以理解为这些条件概率的和；但当前 PPO 优化实现是“按 head/token 分开做 ratio”，不是“整段序列求和后做一个总 ratio”。

---

## 8. 训练命令

### 8.1 快速启动

```bash
cd /home/dministrator/kaggle/rl-agent
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python -m jax_agent.train_jax \
  --config cfg_jax_fixed_pool_0521.yaml
```

### 8.2 完整参数示例

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python -m jax_agent.train_jax \
  --config cfg_jax_fixed_pool_0521.yaml
```

| 参数 | 说明 |
|------|------|
| `--config` | 指定 YAML 配置 |
| `ppo.total_updates` | 总更新轮次 |
| `env.num_envs` | 并行环境数 |
| `ppo.rollout_steps` | 每次 rollout 收集步数 |
| `opponents.enabled` | 训练/评估使用的对手池 |
| `checkpoint.resume_from` | 从指定 `.pkl` checkpoint 继续训练 |
| `wandb.enabled` | 启用 Weights & Biases 记录训练曲线 |

### 8.3 评估 checkpoint

当前评估通常由训练循环按配置自动触发，固定对手在 `jax_agent/eval_opponents/opponents.py` 中注册。训练中的 checkpoint 会按配置周期性保存，评估指标同步写入 W&B。

### 8.4 查看训练曲线

当前主要通过 W&B 查看训练曲线；本地日志保存在 `outputs/{run_name}_jax/wandb/`。

---

## 9. 训练输出目录

训练输出默认写入 `outputs/{run_name}_jax/`：

```
outputs/{run_name}_jax/
├── ckpt_000100.pkl       按配置间隔保存的 checkpoint
├── ckpt_000200.pkl
├── config.yaml           当前运行配置快照
└── wandb/                W&B 本地日志
```

核心指标会同步写入 W&B，包括 PPO loss、entropy、approx KL、episode reward、win rate，以及固定对手评估指标。

---

## 相关页面

- [Agent 全览](Agent-Zoo.md)
- [开发指南](Development-Guide.md)
- [评估与排行榜](Evaluation.md)
