# Behavioral Cloning 训练要点

> **所属页面**：[Home](Home.md) > Behavioral Cloning 训练要点  
> **参考来源**：Kaggle 讨论帖 "Struggling with RL? Something that helped me." 及其评论区经验  
> **适用范围**：Orbit Wars / `rl-agent` 的 JAX PPO 训练路线

---

## 1. 核心结论

BC（Behavioral Cloning，本质是监督学习）适合用来给 RL 打底：先让模型学会“像强玩家一样做出合理动作”，再把这个策略交给 PPO / self-play 去继续优化。它解决的是随机策略一开始乱探索、policy collapse、训练信号不稳定的问题。

这位大佬强调的关键不是“复制 replay”，而是“学习好棋长什么样”。如果把所有动作都无脑当标签，模型会同时学到强动作、坏习惯、噪声和平均玩家的保守策略。BC 的上限主要由数据质量、标签设计和分布偏移决定。

对当前仓库来说，最值得借用的点有四个：

- **BC 是 RL warm start**：先训练 actor 到一个能打的水平，再接 PPO 精修。
- **动作要拆小**：不要把一整回合的联合动作当成一个巨大类别；拆成 source / target / amount / end 或按每个 planet 独立决策。
- **不要让网络学确定性物理**：网络只决定从哪里打、打哪里、派多少；角度、可达性、mask、瞄准由代码精确计算。
- **重点处理 no-op/hold**：不发兵是最多的标签，直接训练会让 launch head 坍缩成永远不动。

---

## 2. 数据选择：学强者，不学平均

### 2.1 数据来源

优先使用 daily episodes / 高分玩家 replay，而不是随便抓全量对局。讨论里提到有人直接从 daily episode datasets 下载了大量 replay；这比单独爬某一个 bot 更适合作为第一版数据源。

推荐给每条样本保留这些元数据：

| 字段 | 用途 |
|------|------|
| `episode_id` | 按整局切分 train/val/test，避免同局泄漏 |
| `step` | 分析早中晚期表现 |
| `player` | 训练前统一翻到 player 0 视角 |
| `winner` | 只学胜者或给胜者更高权重 |
| `teacher_name/rank/score` | 数据质量过滤和加权 |
| `opponent_name/rank/score` | 避免只学虐菜局 |
| `raw_action_count` | 统计动作密度、异常 replay |

### 2.2 过滤原则

第一版不要贪全量，先做干净数据：

- 只取 top-N 玩家或高 LB / 高 TrueSkill 玩家对局。
- 优先学习胜者视角；败者动作可以低权重保留，或先完全丢弃。
- 丢弃异常局：非法动作过多、过早结束、缺 step、动作格式不完整。
- 按 `episode_id` 切分验证集，不能按单步随机切分。
- 保留对手强度信息，避免模型只学到“强者虐弱者”的展开。

### 2.3 不要盲目克隆

move accuracy 上升不等于 win-rate 上升。评论区有人克隆 #1 agent，动作准确率持续变好，但对 greedy Producer 胜率卡在约 17%。原因是 teacher 可能用了 1-ply search，BC 只能蒸馏它最后的动作，学不到搜索过程；模型一旦走到 teacher replay 里没出现过的局面，错误会滚雪球。

所以 BC 数据要尽量表达“好局面下的好决策”：

- 胜者动作权重大于败者动作。
- 强强对局权重大于强弱对局。
- 关键决策步权重大于无意义 no-op。
- 可考虑 outcome-weighted BC：赢局、终局优势扩大后的动作权重更高。
- 后续可做 DAgger：让当前 clone 自己产生偏离分布的局面，再由强 teacher / 搜索器补标签。

---

## 3. 动作标签设计

### 3.1 大佬建议的 factorization

讨论里最明确的设计是“每颗 planet 独立做小决策”：

1. `launch_gate`：这个 source 是否发兵。
2. `target_head`：如果发兵，target 是哪个 planet，使用 pointer / categorical。
3. `amount_head`：如果发兵，派多少，使用离散比例档或连续比例。

`hold/do nothing` 不建议塞进 target 类别里，而是单独做 launch/no-launch gate。这样 target head 只在真实 launch 样本上学习，不会被海量 idle 标签污染。

### 3.2 当前仓库的可复用路线

当前 `rl-agent` 已经有 autoregressive actor：

```text
每回合最多 K 个 token：
END 或 (source, target, amount)
```

因此第一版 BC 有两条路线：

| 路线 | 做法 | 优点 | 风险 |
|------|------|------|------|
| 复用 autoregressive actor | 把 replay 动作转成 `end/source/target/amount` 序列，用 teacher forcing 训练当前模型 | 最容易接 PPO，checkpoint 可以直接 warm start | 同一回合多个动作的顺序可能有噪声 |
| 新增 per-planet BC head | 每个己方 planet 独立输出 launch/target/amount | 更贴合大佬建议，no-op 更好处理 | 需要再把 per-planet policy 接回现有推理/PPO |

建议先走 **复用 autoregressive actor**，因为它最贴合现有 JAX PPO 代码；如果 no-op 或动作顺序问题很严重，再补 per-planet BC head。

### 3.3 动作顺序要规范化

Orbit Wars 同一回合的多条 launch 在语义上更像一个集合，但 autoregressive 模型需要序列。不要直接相信 replay 里的动作顺序稳定，建议训练前固定排序：

1. 先按 `source_id` 升序。
2. 同 source 多条动作时，按 `ships` 降序或 `target_id` 升序。
3. 超过 `max_action_tokens` 的动作截断，并统计截断率。
4. 序列末尾补 `END`，后面 token 标成无效，不参与 source/target/amount loss。

这样模型学的是稳定规范，而不是 replay 输出顺序的偶然性。

### 3.4 target 不学角度

不要让网络回归发射角。当前代码已经有正确方向：网络选择 target slot，物理层再根据旋转星球、彗星、轨迹预测计算角度。确定性可解的问题应该放在环境/动作后处理里，让模型专注于战略决策。

标签转换时应保存：

- `source_slot`
- `target_slot`
- `amount_bucket`
- `end_action`
- `executed/token_valid`

角度只用于反推 action 是否命中某个 target；训练标签最终仍应是 target id / slot。

---

## 4. 特征设计

大佬提到 preprocessing 是把 board 变成 normalized feature rows，并且需要一些 relational info。纯 per-planet 特征很难判断 reachability、谁在争夺、从哪个 source 打哪个 target 更合理。

当前 `features_jax.py` 已经非常适合作为 BC 输入：

| 特征组 | 对 BC 的价值 |
|--------|--------------|
| `planet_features` | owner、ships、production、位置、inbound 压力、局部拓扑 |
| `relative_features` | source-target 距离、方向、相对兵力、ETA、目标先验 |
| `global_features` | 回合进度、双方总船数/星球数/产能 |
| `fleet_features` | 正在飞行的舰队，避免模型只看静态星球 |
| `map_features` | source 视角局部空间结构 |
| `source_mask/candidate_mask/amount_mask` | 把非法动作从训练和推理里都排除 |

BC 里尤其要坚持这条线：**能精确算的就精确算**。比如可发兵 source、合法 target、最小发兵量、派兵比例上限、ETA、轨迹命中，都不要交给网络硬学。

---

## 5. Loss 与 no-op 处理

### 5.1 Autoregressive BC loss

如果复用现有 actor，推荐 loss 结构：

```text
L_bc =
  w_end * BCE(end_logits, end_label)
+ w_src * CE(src_logits, src_label)       only when token_valid and not end
+ w_tgt * CE(tgt_logits, tgt_label)       only when token_valid and not end
+ w_amt * CE(amount_logits, amt_label)    only when token_valid and not end
+ optional entropy / label smoothing
```

注意：

- `source/target/amount` 只在非 END token 上训练。
- invalid / padding token 不参与 loss。
- logits 必须套现有 mask，避免模型为了拟合脏标签学非法动作。
- `amount` 第一版用当前 `SHIP_RATIOS` 离散桶，不急着回归连续 ships。

### 5.2 Per-planet BC loss

如果后续新增 per-planet head，可以用：

```text
L_planet =
  w_launch * BCE(launch_logits, launch_label)
+ I_launch * w_tgt * CE(target_logits, target_label)
+ I_launch * w_amt * CE(amount_logits, amount_label)
```

这里 `I_launch` 很重要：不 launch 的样本不训练 target/amount。

### 5.3 no-op 坍缩的处理

不发兵通常远多于发兵，直接训练会得到“永远 no-op”的模型。可组合使用：

- **正样本加权**：launch 样本给更高权重。
- **idle 下采样**：随机丢掉大量普通 idle，但保留“难 idle”。
- **balanced batch**：每个 batch 保证一定比例 launch token / launch planet。
- **focal loss**：让模型少被简单 idle 样本支配。
- **按阶段统计**：早期扩张、中期攻防、后期收官的 launch rate 分开看。

不要把所有 idle 都删光。好的 agent 也需要学会等待，特别是低兵力 source、危险 target、敌方 incoming 压力很大时的 no-op。

---

## 6. 模型容量与训练策略

讨论里提到 25M 参数是因为有大量数据支撑；数据少时盲目放大只会过拟合。当前模型 `hidden_size=128`、Transformer + CNN + fleet tokens 已经可以先跑 MVP。

建议顺序：

1. 小模型 / 当前模型结构先验证数据管线。
2. 看 offline validation loss 和 online win-rate 是否同步改善。
3. 数据规模上来后再扩 hidden size、层数、fleet tokens。
4. 每次只改一个变量，避免不知道提升来自数据、模型还是过滤规则。

训练技巧：

- 用 AdamW，先从 `1e-4 ~ 3e-4` 搜索；大 batch 可更稳。
- 使用 label smoothing 或小 entropy bonus，避免 BC 把策略熵压到太低。
- early stopping 看 validation NLL 和 online eval，不只看 train loss。
- 保存 normalizer 参数；BC 和后续 PPO 必须使用一致的归一化统计。

---

## 7. 评估指标

### 7.1 Offline 指标

只看 overall action accuracy 不够，要拆开：

| 指标 | 说明 |
|------|------|
| `end_bce / end_acc` | 序列长度和停止时机是否学对 |
| `src_top1/top3` | 是否知道该从哪里发兵 |
| `target_top1/top3` | 是否知道该打哪里 |
| `amount_acc / bucket_mae` | 派兵比例是否接近 |
| `launch_precision/recall` | 是否解决 no-op 坍缩 |
| `active_token_nll` | 只看真实 launch token 的 NLL |
| `invalid_rate` | mask 后仍产生非法动作的比例 |

还要按 `step` 分桶：早期扩张、中期攻防、后期收官。BC 很可能早期开局学得好，但中后期一旦偏离 teacher 分布就崩。

### 7.2 Online 指标

最终以实战为准：

- vs `nearest` / starter baseline。
- vs Producer / planner 类强规则 agent。
- vs 被模仿的 teacher 或同级 checkpoint。
- vs 当前 PPO checkpoint。
- deterministic 和 stochastic 两种推理都测。

同时记录行为统计：

- 每回合平均 launch 数。
- no-op 回合比例。
- 平均派兵比例。
- 目标类型分布：neutral / enemy / self reinforcement。
- early expansion 速度。
- invalid / truncated action rate。

---

## 8. BC 接 PPO 的注意点

BC clone 通常比 demonstrator 弱，因为它只见过 teacher 轨迹里的状态。想超过 teacher，需要 outcome signal，也就是 RL / self-play。

BC 接 PPO 时最容易出两个问题：

1. **策略熵被 BC 压太低**：PPO 没探索空间，后续学不动。
2. **actor 已训练、critic 还是空的**：actor-critic 方法可能在 critic 追上之前把好策略“洗掉”。

推荐接法：

- 从 BC checkpoint 初始化 actor 和 encoder。
- normalizer 沿用 BC 统计，或在 PPO 开始时小心 warm-up。
- critic 可以先用 rollout 的 Monte Carlo / GAE return 单独 warm-up 几轮。
- PPO 初期 actor lr 小一点，critic lr 可相对正常。
- 保留 entropy bonus，不要一开始就 deterministic。
- 可短期加 KL-to-BC 约束，但不要长期把 teacher 固定成锚点。
- self-play 对手应逐步更新，使用 lagged-self / checkpoint pool，而不是永远打一个 frozen teacher。

讨论里的经验也指向这一点：固定 teacher、固定 KL anchor 容易卡在 teacher 水平附近；真正突破需要训练对象跟着自己一起变强。

---

## 9. 第一版落地 Checklist

- [ ] 收集 top replay，生成 episode manifest。
- [ ] 统一 player 0 视角，复用 `flip_state` 逻辑。
- [ ] replay step 转 `EnvState`，复用 `encode_batch`。
- [ ] action 转 `end/source/target/amount` teacher-forcing 标签。
- [ ] 固定多动作排序规则。
- [ ] 统计 no-op / launch 比例、非法标签、截断率。
- [ ] 实现 masked BC loss。
- [ ] 训练当前 `AutoregressiveActorCritic`。
- [ ] offline 分 head 验证。
- [ ] online 打 baseline / Producer / 当前 checkpoint。
- [ ] 保存 BC actor、normalizer、配置、数据 manifest hash。
- [ ] 用 BC checkpoint 初始化 PPO，并做 critic warm-up / entropy 保护。
