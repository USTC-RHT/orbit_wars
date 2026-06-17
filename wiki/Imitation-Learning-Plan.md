# 模仿学习落地计划

> **所属页面**：[Home](Home.md) > 模仿学习落地计划  
> **目标**：在现有 JAX PPO agent 之前加入 BC / imitation learning 阶段，把随机探索起点换成可打的监督学习策略。  
> **推荐路线**：先复用当前 autoregressive actor 做 BC MVP，再视效果决定是否新增 per-planet imitation head。

---

## 1. 总目标

做一条可反复迭代的训练流水线：

```text
高质量 replay
  -> 状态/动作解析
  -> BC 监督训练
  -> online eval
  -> PPO / lagged self-play 精修
  -> Kaggle submission
```

第一阶段的成功标准不是“BC 直接夺冠”，而是：

- BC agent 能稳定击败弱 baseline。
- 比随机初始化 PPO 更快达到可用水平。
- 特征、mask、网络结构能被快速验证。
- BC checkpoint 能平滑接上当前 PPO 训练循环。

---

## 2. 当前代码可以复用的部分

| 现有模块 | 复用方式 |
|----------|----------|
| `jax_agent/features_jax.py` | 继续生成 normalized feature rows、relative features、fleet/map/global 特征和 masks |
| `jax_agent/model_jax.py` | 复用 `AutoregressiveActorCritic` 的 end/source/target/amount heads |
| `jax_agent/train_jax.py` | 复用 observation -> `EnvState`、player flip、动作转 Kaggle、eval 对手接口 |
| `jax_agent/ppo_jax.py` | 参考 PPO 里重算 logits / masked log-prob 的写法 |
| `cfg_jax_*.yaml` | 新增 BC config，保持 env/model/state_norm 参数一致 |
| `eval_opponents/` | BC checkpoint 训练后直接纳入在线评估 |

第一版尽量不要改推理动作格式。BC 训练出来的 checkpoint 应该能直接被 PPO resume 或 eval 读取。

---

## 3. 阶段 0：文档与决策

**目标**：确定 BC 的第一版训练定义。

决策建议：

- 第一版使用 autoregressive teacher forcing。
- 训练标签为 `END/source/target/amount`。
- 多动作排序规则固定为 `source_id -> ships desc -> target_id`。
- target 学 planet slot，不学角度。
- amount 学现有 8 个 `SHIP_RATIOS` 桶。
- no-op 通过 end/token_valid 和 launch 样本加权处理。

产物：

- `wiki/BC-Training-Notes.md`
- `wiki/Imitation-Learning-Plan.md`
- 后续新增 `cfg_jax_BC_mvp.yaml`

---

## 4. 阶段 1：Replay 数据集

**目标**：把 Kaggle replay 变成可训练的离线数据集。

### 4.1 数据收集

优先级：

1. daily episodes datasets。
2. top-30 / top-50 玩家对局。
3. 强规则 agent / planner agent 的本地自生成对局。
4. 当前自己 checkpoint 的高胜率对局。

建议目录：

```text
data/bc/
  raw/
    episodes/
  manifests/
    bc_manifest.csv
  processed/
    shards/
```

manifest 至少包含：

```text
episode_id, player, opponent, winner, teacher_rank, opponent_rank,
num_steps, source_path, split, quality_weight
```

### 4.2 训练/验证切分

- 按 episode 切分，不按 step 切分。
- `train/val/test = 90/5/5` 起步。
- 验证集保留强强对局，不要只放普通局。
- 固定 split seed，确保每次实验可比。

### 4.3 质量统计

每次生成数据后输出：

- episode 数、step 数、action token 数。
- 胜者样本比例。
- no-op / launch 比例。
- 平均每回合 launch 数。
- invalid action 比例。
- 超过 `max_action_tokens` 的截断率。
- target 类型分布：neutral / enemy / self。

---

## 5. 阶段 2：状态与标签抽取

**目标**：对每个 replay step 生成 `EncodedBatch` 输入和 BC 标签。

### 5.1 状态重建

优先复用现有解析逻辑：

- replay observation -> `parse_observation`
- observation -> `kaggle_obs_to_jax_state`
- player 1 视角 -> `flip_state`
- `encode_batch(state, feat_params)` 生成训练输入

注意 BC 训练和 PPO 后续必须保持同一套：

- `EnvParams`
- `FeatureParams`
- target mask 开关
- `max_planets`
- `ship_bucket_count`
- state normalizer

### 5.2 动作反解

每条 Kaggle action 通常是：

```text
[source_id, angle, ships]
```

需要反解成：

```text
source_slot
target_slot
amount_bucket
```

建议做法：

1. 用 `source_id` 找 source slot。
2. 用当前 step 的 planet/fleet/physics，推断该 angle 最可能命中的 target。
3. 如果无法稳定反解 target，先丢弃该动作并计数。
4. `ships / source_ships` 映射到最近的 `SHIP_RATIOS` 桶。
5. 根据现有 masks 检查 label 是否合法。

如果某些强 agent 使用复杂瞄准导致 target 反解不准，可以先训练 source/amount/launch，再单独处理 target 标签质量。

### 5.3 序列标签

生成固定长度 `K=max_action_tokens`：

```text
end_actions: [K] bool
src_actions: [K] int
tgt_actions: [K] int
amt_actions: [K] int
executed:    [K] bool
token_valid: [K] bool
```

规则：

- 第一个 END 之前的 token 是真实 launch。
- 第一个 END token 参与 end loss。
- END 后全部 padding，不参与 source/target/amount loss。
- 空动作回合：第 0 个 token 是 END。
- 超长回合：截断到 K，并记录 `truncated=true`。

---

## 6. 阶段 3：BC 训练 MVP

**目标**：新增一个独立 BC 训练入口，训练现有 `AutoregressiveActorCritic`。

建议新增：

```text
jax_agent/
  bc_dataset.py
  train_bc_jax.py
cfg_jax_BC_mvp.yaml
```

### 6.1 训练循环

训练循环最小需求：

- shard dataloader，按 episode/step 采样。
- fit/update state normalizer。
- forward 当前 model，teacher forcing 输入 replay 动作前缀。
- masked cross entropy / BCE loss。
- metrics 写 CSV 或 wandb。
- 定期保存 checkpoint。
- 定期跑 offline validation。

### 6.2 Loss 权重初值

可以从保守值起步：

```yaml
bc:
  lr: 2.0e-4
  batch_size: 2048
  epochs: 5
  weight_decay: 1.0e-4
  label_smoothing: 0.02
  end_weight: 0.5
  source_weight: 1.0
  target_weight: 1.0
  amount_weight: 0.5
  launch_token_weight: 2.0
  idle_token_weight: 0.25
```

这些不是最终答案，只是为了第一版避免 no-op 淹没真实 launch。后续根据 `launch_precision/recall` 调。

### 6.3 Offline 验证门槛

第一版可以设这些最低门槛：

- validation loss 不再明显过拟合。
- active token 的 target top-3 明显高于随机。
- launch recall 不接近 0。
- invalid label 比例可解释，最好小于 1%。
- 截断率足够低，最好小于 0.5%。

---

## 7. 阶段 4：Online 评估

**目标**：确认 BC 学到的是可执行策略，而不是只会拟合日志。

评估列表：

1. `nearest` baseline。
2. starter / random。
3. Producer / planner 类外部 agent。
4. 当前 PPO checkpoint，例如 `ckpt_032700`、`ckpt_048900`。
5. 如果数据来自单个 teacher，再测 vs teacher。

每次 BC checkpoint 记录：

```text
checkpoint
dataset_manifest_hash
bc_val_loss
launch_recall
target_top3
eval_nearest_win_rate
eval_producer_win_rate
eval_ckpt_win_rate
avg_launches_per_turn
invalid_action_rate
```

BC 如果 offline accuracy 高但 online 很差，优先检查：

- target 反解是否错。
- action 顺序是否噪声大。
- no-op 是否坍缩。
- 数据是否只覆盖 teacher 自己会到达的局面。
- deterministic 推理是否太死，可尝试温度采样。

---

## 8. 阶段 5：BC 接 PPO

**目标**：让 PPO 从 BC policy 开始，而不是从随机策略开始。

建议新增 PPO 配置：

```text
cfg_jax_RL_from_BC_mixed_self.yaml
```

关键设置：

- `init_from_checkpoint: checkpoints/.../bc_ckpt.pkl`
- actor / encoder 从 BC 加载。
- critic 允许重新初始化，或先做 critic warm-up。
- PPO 初期 actor lr 降低，例如 `2e-5 ~ 5e-5`。
- 保留 entropy bonus，防止 BC 后策略过早确定化。
- 前几千 update 可以加入小 KL-to-BC，但必须逐渐衰减。
- 对手池使用 lagged-self + 历史 checkpoint，不要只打 frozen teacher。

### 8.1 Critic warm-up

如果 actor 已经会玩但 critic 为空，PPO 早期可能误更新 actor。可选方案：

- 先 rollout BC policy，固定 actor，只训练 value head 若干轮。
- 或 PPO 前期提高 `vf_coef`，降低 actor lr。
- 或用 Monte Carlo return / GAE target 对 critic 做短预训练。

### 8.2 何时停止依赖 BC

BC 只是脚手架，不是天花板。可以按阶段衰减：

| 阶段 | 约束 |
|------|------|
| PPO 早期 | 小 KL-to-BC，保护不立刻忘掉 |
| PPO 中期 | KL 权重线性降到 0，主要靠 self-play |
| PPO 后期 | 完全按 win-rate / exploitability 选择 checkpoint |

---

## 9. 阶段 6：进阶 imitation learning

BC MVP 跑通后，可以继续做：

- **Outcome-weighted BC**：赢局、优势扩大后的动作权重更高。
- **Q-filtered BC**：只学习估计价值更高的 demonstrator 动作。
- **DAgger**：当前模型 self-play 产生 off-distribution 状态，再让 teacher / 搜索器标注。
- **Search distillation**：本地 planner 对同一状态搜索多个候选，把搜索结果蒸馏给网络。
- **Per-planet head**：专门解决 launch/no-launch 和动作集合建模。
- **Hybrid BC + RL loss**：同一 batch 混合 replay CE 和 PPO loss，逐步降低 CE 权重。

---

## 10. 风险清单

| 风险 | 表现 | 应对 |
|------|------|------|
| no-op 坍缩 | agent 几乎不发兵 | launch 加权、idle 下采样、按 active token 评估 |
| 标签反解错误 | offline target acc 上不去 | 改进 angle->target 反解，先人工抽样核对 |
| 动作顺序噪声 | source acc 低、序列 loss 高 | 固定排序，或改 per-planet head |
| 数据质量差 | clone 像平均玩家 | 只学强者/胜者，加质量权重 |
| 分布偏移 | offline 好、online 崩 | DAgger、self-play RL、online eval 提前介入 |
| teacher ceiling | 打不过 teacher | PPO / outcome signal / lagged self-play |
| critic mismatch | PPO 接上后退化 | critic warm-up，降低 actor lr，保留 entropy |
| 参数过大 | train loss 降、val/online 不升 | 先扩大数据，再扩大模型 |

---

## 11. 推荐执行顺序

1. 写 replay manifest 和数据质量统计脚本。
2. 抽 100 局 replay，完成 state/action 反解并人工 spot check。
3. 生成小 shard，训练 `train_bc_jax.py` smoke test。
4. 跑 1k-5k 局数据，看 offline metrics 是否合理。
5. 接 online eval，确认不是 no-op agent。
6. 扩到 top daily episodes 数据。
7. 调 no-op 权重、action 排序、target 反解。
8. 保存最佳 BC checkpoint。
9. 新建 PPO from BC config，做 critic warm-up。
10. 用 lagged self-play / checkpoint pool 继续训练。

第一版要快：宁可先用 5k 干净 replay 跑通闭环，也不要一上来处理几十万局但不知道标签是否可信。
