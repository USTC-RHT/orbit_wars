# Transformer + PPO 稳定训练计划

> 目标：不要再等到 `entropy` 崩溃、`KL` 爆炸或 NaN 出现才发现问题。
> 后续训练要把 `clipfrac` 当作提前预警信号，用 schedule、分头 entropy、自动降速和容量门控把问题挡在崩溃前。

## 结论

当前项目最大的问题不是某一个单点 bug，而是训练系统还像普通 MLP PPO 一样运行 Transformer：

- PPO 只有整体 `entropy` / `KL` / `clipfrac`，看不见 fire、target、amount 哪个 head 先坏。
- 学习率只有线性退火，没有 warmup，也没有按风险指标自动降速。
- `ent_coef` 是单一系数，不能分别保护 fire、target、amount 三个不同分布。
- 训练是否继续主要看 NaN、KL、eval 胜率，太滞后。
- Transformer 容量、对手难度、优化强度一起变化时，很难定位是谁导致发散。

后续不要先加更复杂模型或对手池。第一阶段应该先补“稳定性控制面板”和“训练刹车”。

---

## 架构项取舍

这些改动不是同一个优先级。当前首要问题是 PPO + Transformer 的优化控制不足，不是单纯表达力不够。

| 项目 | 当前状态 | 是否需要 | 优先级 | 结论 |
|---|---|---|---|---|
| `TypedInputProjection` | 现在已有 planet / fleet / self / global 的独立 MLP，并额外加了 token type embedding | 可做 | P2 | 不是当前发散主因。后续可以整理成更明确的 typed projection，但先别和 schedule / entropy / 对手改造混在一起。 |
| `sun mask` | 当前太阳/轨道/安全信息主要在特征和动作执行物理过滤里 | 谨慎做 | P2 | 不建议把完整物理引擎放到推理前 hard mask。可以先做 sun risk feature 或轻量 logit bias；hard mask 必须保证和 rollout 记录的 logp 完全同步。 |
| `MLP FireHead` | 当前 fire head 已经是 `Dense -> ReLU -> small-init Dense` | 已有基础版 | P1.5 | 保留即可。先观察 `entropy_fire` / `fire_rate`，如果 fire head 仍塌缩，再加更深 MLP 或 head-specific entropy。 |
| `per-source TargetMix` | grouped Transformer 已经用 `global planet key + source-relative MLP` 混合 | 已需要且已有 | P1.5 | 这是解决“Transformer 只跑一次但目标选择仍知道 source-relative 信息”的关键结构，应该保留。后续可升级成 gated mix。 |
| `multi-query ValueHead` | 当前 grouped Transformer 默认使用 global CLS value broadcast；legacy 是 per-source value | 可做 | P2 | 更贴合 actor-critic 的是全局 `V(s)`，但可以用多个 learned query attend 全局 tokens 来增强 critic。不要回到纯 per-source value 当主路线。 |
| `update_c=True` | 代码里没有这个概念 | 暂不做 | P3 | 如果指 PopArt / critic target normalization，当前回报尺度还不大，先用 value 健康指标判断是否需要。不要在没定位前加。 |
| `MLP 4x expansion` | Transformer block 里 `mlp_ratio=4` 已经存在 | 保留 | 已完成 | 这是标准 Transformer FFN 配置。慢或不稳时优先降 layer/head/hidden，不优先改掉 4x。 |

推荐顺序：

```text
1. 先补 per-head entropy、clipfrac rolling、value 健康指标。
2. 再做 warmup + cosine、entropy schedule、clipfrac governor。
3. 之后才动 TypedInputProjection / multi-query ValueHead 这类结构项。
4. sun 相关先做 feature/logit bias，不急着做 hard mask。
```

---

## P0：必须先做的稳定性监控

### 1. 把 `clipfrac` 作为一等指标

现在代码已经记录整体 `clipfrac`：

```text
losses/clipfrac
metrics.csv: clipfrac
```

但还不够。需要新增 rolling 统计：

| 指标 | 用途 |
|---|---|
| `clipfrac_ma_20` | 最近 20 update 的平均剪切比例 |
| `clipfrac_ma_100` | 最近 100 update 的慢速趋势 |
| `clipfrac_slope_100` | 判断是否单调爬升 |
| `clipfrac_alert` | 是否进入预警状态 |

建议阈值：

```text
clipfrac_ma_20 < 0.15      健康
0.15 <= clipfrac < 0.25    注意，观察 slope
0.25 <= clipfrac < 0.35    预警，降低 lr 或减少 epoch
clipfrac >= 0.35           高危，保存 checkpoint 并自动降速/暂停
```

核心判断不是单点高，而是：

```text
clipfrac_ma_100 持续上升 + entropy 下降 + KL 变大
```

这比等 NaN 或等 eval 掉下去更早。

### 2. 拆分三类 entropy

当前 `entropy` 是：

```python
fire_entropy + fire * (target_entropy + amount_entropy)
```

这会把三个不同问题混在一起。需要在 `log_prob_entropy` 或旁路函数里输出：

| 指标 | 说明 |
|---|---|
| `entropy_fire` | 是否发射的 Bernoulli 熵 |
| `entropy_target` | 目标选择熵，只统计 fire=True 且合法目标 |
| `entropy_amount` | 派兵数量熵，只统计 fire=True |
| `fire_rate` | 当前策略发射比例 |
| `valid_launch_rate` | 真实执行发射比例 |
| `invalid_fire_rate` | fire=True 但物理过滤成 no-op 的比例 |

预警规则：

```text
entropy_fire 快速下降 + fire_rate 变极端       fire head 塌缩
entropy_target 快速下降 + clipfrac 上升        pointer head 过早锐化
entropy_amount 快速下降 + amount 分布单一      amount head 过早固定
invalid_fire_rate 上升                         策略和物理可执行动作错配
```

### 3. 增加 value / advantage 健康指标

`clipfrac` 上升通常不是 actor 单独的问题，value head 锐化也会把 advantage 分布推坏。需要记录：

| 指标 | 说明 |
|---|---|
| `value_mean_abs` | value 输出绝对值均值 |
| `value_std` | value 输出标准差 |
| `return_std` | GAE return 标准差 |
| `adv_std_before_norm` | 归一化前 advantage 标准差 |
| `explained_variance` | value 对 return 的解释度 |
| `value_clipfrac` | value clipping 命中比例 |

如果出现：

```text
value_std 上升
adv_std 上升
clipfrac 上升
policy_loss 变大
```

优先怀疑 critic 更新过强，而不是先调 actor。

### 4. 记录优化器强度

需要记录：

| 指标 | 说明 |
|---|---|
| `lr/current` | 当前学习率 |
| `grad/global_norm` | clip 前梯度范数 |
| `grad/clipped_global_norm` | clip 后梯度范数 |
| `params/global_norm` | 参数范数 |
| `update/global_norm` | 参数更新量 |
| `update_to_param_ratio` | 更新量 / 参数量 |

如果 `clipfrac` 上升但 `grad_norm` 已经长期打满 `max_grad_norm`，说明不是再加梯度裁剪能解决，而是 lr/epoch/model capacity 过强。

---

## P1：学习率 schedule 改造

### 1. 默认使用 warmup + cosine decay

Transformer 策略不应该从第一个 update 就用满 lr。建议把 `PPOConfig` 扩展为：

```yaml
ppo:
  lr: 1.0e-5
  lr_schedule: warmup_cosine   # constant | linear | warmup_cosine
  warmup_updates: 100
  min_lr_ratio: 0.1
```

行为：

```text
0 ~ warmup_updates:
  lr 从 0 线性升到 base_lr

warmup_updates ~ total_updates:
  lr 按 cosine decay 降到 base_lr * min_lr_ratio
```

优先级：

1. Transformer 从零训练：必须 warmup。
2. 从 legacy checkpoint fine-tune：可以短 warmup。
3. 纯 legacy MLP：可以继续 linear decay。

### 2. 加自适应 LR governor

固定 schedule 不够，后续需要一个训练期“刹车”：

```text
如果 clipfrac_ma_20 > 0.25 且 clipfrac_slope_100 > 0:
    lr *= 0.5

如果 clipfrac_ma_20 > 0.35 或 max_abs_log_ratio 持续 > 5:
    保存 checkpoint
    lr *= 0.25
    暂时 epochs = 1 或跳过本轮更新
```

这个可以先在 Python 训练循环外层实现，不必一开始塞进 JIT。

### 3. KL 早停保留，但不能作为唯一保护

当前 `target_kl=0.03` + KL early stop 是必要的，但它偏晚。建议：

```text
clipfrac 是早期预警
KL 是中期保护
NaN assert 是最后防线
```

也就是说，看到 `clipfrac` 爬升就处理，不等 KL 爆。

---

## P1：entropy schedule 改造

### 1. 单一 `ent_coef` 改成 per-head entropy

现在的：

```yaml
ent_coef: 0.05
```

建议扩展为：

```yaml
ppo:
  ent_coef_fire: 0.02
  ent_coef_target: 0.04
  ent_coef_amount: 0.01
  ent_schedule: linear_decay
  ent_min_ratio: 0.25
```

原因：

- `fire` 是 Bernoulli，塌缩后最容易导致“永远不发射/永远发射”。
- `target` 是 pointer 分布，是 Transformer 最容易锐化的地方。
- `amount` 通常不需要和 target 一样高的探索。

### 2. 根据 head 状态动态保护

简单规则：

```text
如果 entropy_fire 过低且 fire_rate 极端：
    临时提高 ent_coef_fire

如果 entropy_target 过低且 clipfrac 上升：
    临时提高 ent_coef_target 或降低 lr

如果 amount 分布单一但胜率没有提升：
    提高 ent_coef_amount
```

第一版可以只做日志，不自动调；第二版再做自动调。

---

## P1：训练停止和回滚策略

### 1. 每次训练必须有健康门控

训练循环里加 health gate：

```text
healthy:
  clipfrac_ma_20 < 0.30
  approx_kl < 1.5 * target_kl
  entropy_fire > fire_entropy_floor
  no non-finite values
```

触发 unhealthy 时：

1. 保存 `bad_ckpt`。
2. 保存最近 `good_ckpt` 的路径。
3. 自动写一行告警到 log。
4. 可选：停止训练或降低 lr 继续。

### 2. 最近好 checkpoint 的定义

不要只保存固定间隔 checkpoint。还要保存：

```text
best_eval_checkpoint
last_healthy_checkpoint
pre_alert_checkpoint
```

这样一旦 `clipfrac` 开始爬升，可以回到刚爬升前，而不是回到几百 update 以前。

---

## P2：PPO 更新强度改造

### 1. epochs 不要固定 4

当前 `epochs=4`，虽然有 KL early stop，但 clipfrac 高时第一轮就可能已经太重。建议：

```text
clipfrac_ma_20 < 0.15      epochs=4
0.15~0.25                 epochs=2
>0.25                     epochs=1
>0.35                     skip update 或 lr 降到 1/4
```

### 2. minibatch 按 state 而不是 source 更严格

当前代码已经按 state 做 minibatch：

```python
states_per_minibatch = minibatch_size // source_count
```

这点是对的。后续需要确认：

- 同一个 state 的所有 source 同进一个 minibatch。
- global value 的样本不要被每个 source 重复放大太多。

如果 Transformer 使用 global value，value loss 最好按 state 聚合，而不是按 source 重复计权。否则 critic 可能被放大 P 倍，只是现在 mask/denom 把它平均了，但语义上仍需要再审一次。

---

## P2：模型容量推进顺序

不要同时改架构、对手池、lr、entropy。建议容量推进顺序：

### Stage A：legacy 稳定基线

目标：确保训练管线、eval、Kaggle submission 全通。

```yaml
use_transformer: false
use_cnn: false
```

保留作为所有实验的 sanity baseline。

### Stage B：小 Transformer

```yaml
use_transformer: true
use_cnn: false
hidden_size: 96 或 128
num_transformer_layers: 1
num_heads: 2
max_fleet_tokens: 8 或 16
```

目标不是最高胜率，而是：

```text
clipfrac 不持续爬升
entropy_fire 不崩
KL 不长期贴近早停线
eval 至少能稳定打赢 nearest
```

### Stage C：标准 Transformer

```yaml
hidden_size: 128
num_transformer_layers: 2
num_heads: 4
max_fleet_tokens: 16
```

只有 Stage B 健康后再上。

### Stage D：更大模型或 CNN

CNN 和更多 fleet tokens 都是后续增强，不应在稳定性没解决前混进来。

---

## P2：对手策略改造

### 1. 先固定对手，再加 self-play

Transformer 从零训练时，对手变化会让分布漂移更严重。建议顺序：

1. `nearest`
2. `ckpt400 fixed`
3. `ckpt400 + nearest eval`
4. 单一 lag self-play
5. 小池子 self-play

不要一开始就 PFSP + Transformer + 多对手。

### 2. 对手切换也要看稳定指标

升阶条件不只看 win_rate：

```text
nearest_win_rate >= threshold
clipfrac_ma_100 < 0.20
entropy_fire 不低于 floor
KL 稳定
```

否则可能是策略短期压过 nearest，但优化已经在坏掉。

---

## 推荐实施顺序

### 第 1 步：补日志，不改训练行为

改动：

- 拆 `entropy_fire / entropy_target / entropy_amount`
- 记录 `fire_rate / valid_launch_rate / invalid_fire_rate`
- 记录 `clipfrac_ma_20 / clipfrac_ma_100 / clipfrac_slope_100`
- 记录 `lr/current`
- 记录 value 健康指标

验收：

```text
metrics.csv 和 wandb 能画出每个 head 的曲线。
至少跑 300 update，能看清谁先变化。
```

### 第 2 步：warmup + cosine

改动：

- `PPOConfig` 增加 `lr_schedule / warmup_updates / min_lr_ratio`
- `make_train_state` 支持 warmup_cosine
- wandb 记录当前 lr

默认 Transformer 配置：

```yaml
ppo:
  lr: 1.0e-5
  lr_schedule: warmup_cosine
  warmup_updates: 100
  min_lr_ratio: 0.1
```

### 第 3 步：per-head entropy

改动：

- PPO loss 拆分 entropy
- 配置增加 `ent_coef_fire / ent_coef_target / ent_coef_amount`
- 默认先保守：

```yaml
ent_coef_fire: 0.02
ent_coef_target: 0.04
ent_coef_amount: 0.01
```

### 第 4 步：clipfrac governor

改动：

- Python 训练循环维护 rolling stats。
- 高危时自动降 lr 或停止。
- 保存 `last_healthy_checkpoint`。

第一版建议只自动停止，不自动调参，避免隐藏实验结果。

### 第 5 步：重跑小 Transformer ablation

矩阵：

| 实验 | 模型 | 对手 | 目标 |
|---|---|---|---|
| A | legacy | ckpt400 | sanity |
| B | 1-layer Transformer | nearest | 稳定性 |
| C | 1-layer Transformer | ckpt400 | 稳定性 |
| D | 2-layer Transformer | ckpt400 | 上限 |
| E | 2-layer Transformer | single lag self-play | 自博弈稳定性 |

所有实验先看 `clipfrac`，再看 win-rate。

---

## 推荐默认配置草案

用于下一轮 Transformer 稳定性实验：

```yaml
model:
  hidden_size: 128
  use_cnn: false
  use_transformer: true
  num_transformer_layers: 1
  num_heads: 2
  max_fleet_tokens: 16

ppo:
  num_envs: 64
  rollout_steps: 128
  epochs: 4
  minibatch_size: 4096
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.15
  vf_coef: 0.5
  lr: 1.0e-5
  lr_schedule: warmup_cosine
  warmup_updates: 100
  min_lr_ratio: 0.1
  max_grad_norm: 0.5
  target_kl: 0.03
  norm_adv: true
  ent_coef_fire: 0.02
  ent_coef_target: 0.04
  ent_coef_amount: 0.01
```

如果 `clipfrac_ma_20 > 0.25`：

```text
先减 lr，不先改 reward。
先降 epochs，不先加模型容量。
先看 per-head entropy，不先换对手池。
```

---

## 成功标准

一轮训练不再只看最终 win_rate。健康标准应该是：

```text
1. clipfrac 没有持续单调爬升到 0.30+
2. entropy_fire 不提前塌到接近 0
3. KL 不长期依赖早停压住
4. value_loss 不持续膨胀
5. eval 胜率提高时，训练指标也健康
6. checkpoint 能本地完整 eval 和 Kaggle validation
```

如果胜率涨了但 `clipfrac` 正在爬升，那不是“训练很好”，而是“快要出事”。
