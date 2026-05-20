# JAX 训练代码修改指南

## 1. 状态 / 特征

### 特征维度常量
**文件：`jax_agent/features_jax.py`，第 38-44 行**

```
SELF_DIM = 18         # 来源星球自身特征
CANDIDATE_DIM = 26    # 候选星球特征
GLOBAL_DIM = 11       # 全局局势特征
MAP_CHANNELS = 6      # CNN 小地图通道数
MAP_GRID_SIZE = 8     # CNN 小地图分辨率
```

修改维度后必须同步修改下面的特征构建函数，否则维度不匹配会在 JIT 编译时报错。

### 来源星球特征（self_features）
**文件：`jax_agent/features_jax.py`，`build_self_features()`，第 146 行**

18 维特征：驻军数、生产力、owner 独热、是否连通 sigma 等。

### 候选星球特征（candidate_features）
**文件：`jax_agent/features_jax.py`，`build_candidate_features()`，第 198 行**

26 维特征：距离、角度、敌我兵力差、inbound 威胁等。包含 slot 0（no-op）。

### 全局特征（global_features）
**文件：`jax_agent/features_jax.py`，`build_global_features()`，第 310 行**

11 维：全局船数比、星球数比、步数进度等。

### 特征编码参数
**文件：`jax_agent/features_jax.py`，`FeatureParams`，第 49 行**
**配置项：`cfg.env.*`（yaml 中 `env:` 块）**

| 参数 | 作用 |
|---|---|
| `max_planets` | 最大星球数，决定特征矩阵 shape |
| `ship_bucket_count` | 派兵档位数（默认 8）|
| `target_filter_enabled` | 是否过滤不可达目标 |
| `attack_top_k` / `defense_top_k` | 攻击/防御候选保留数 |

### 状态归一化
**文件：`jax_agent/normalization_jax.py`，`update_and_normalize()` / `normalize_encoded()`**
**配置项：`cfg.state_norm.clip`、`cfg.state_norm.eps`**

在线 running mean/var 归一化，每步在 rollout 中更新。修改 clip 值影响特征截断力度。

---

## 2. 动作

### 派兵档位
**文件：`jax_agent/features_jax.py`，第 44 行**

```python
SHIP_RATIOS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
```

修改派兵比例档位（增删档位时需同步修改 `ship_bucket_count`）。

### 最小发射阈值
**文件：`jax_agent/features_jax.py`，第 43 行**

```python
MIN_LAUNCH = 10.0  # 驻军不足此数时禁止所有派兵档位
```

### 动作掩码（amount_mask）
**文件：`jax_agent/features_jax.py`，`build_candidate_features()`，第 198 行**

决定哪些（目标, 档位）组合合法：驻军充足 + 目标合法才放开。

### 动作→游戏格式转换
**文件：`jax_agent/train_jax.py`，`actions_from_decisions()`，第 825 行**

将网络输出的 (fire, target_slot, amount_bucket) 转为游戏接口格式 `[from_planet, angle, ships]`。修改发射逻辑（如瞄准算法、最小派兵量）在此处。

### 瞄准预测（aim_with_prediction_jax）
**文件：`jax_agent/train_jax.py`，`aim_with_prediction_jax()`**

60 步轨迹预测锁定旋转星球/彗星目标，输出发射角度。

---

## 3. 奖励

### 终局奖励
**文件：`jax_agent/reward_jax.py`，`terminal_reward_player0()`，第 83 行**

```python
# 当前逻辑：
return jnp.where(s0 > s1, 1.0, -1.0)  # 赢 +1，其余（平局/负）-1
```

修改此处改变胜负平局的奖励值。

### 奖励塑形
**文件：`jax_agent/reward_jax.py`，`RewardParams`，第 17 行**
**配置项：`cfg.reward.*`（yaml 中 `reward:` 块）**

| 参数 | 作用 | 当前值 |
|---|---|---|
| `terminal_weight` | 终局奖励系数 | 1.0 |
| `ships_weight` | 船只增量奖励 | 0.0 |
| `production_weight` | 生产力增量奖励 | 0.0 |
| `planet_weight` | 星球数增量奖励 | 0.0 |
| `neutral_capture_weight` | 夺取中立星球奖励 | 0.0 |
| `enemy_capture_weight` | 占领敌方星球奖励 | 0.0 |
| `lose_planet_penalty` | 失去星球惩罚 | 0.0 |
| `wasted_attack_penalty` | 攻击己方星球惩罚 | 0.0 |
| `time_penalty` | 每步时间惩罚（鼓励速胜）| 0.0 |

各奖励分项的计算逻辑在 `reward_from_transition()` 函数中，第 63 行起。

---

## 4. 对手

### 对手模式选择
**配置项：`cfg.opponent`（yaml 顶层）**

| 值 | 含义 |
|---|---|
| `"nearest"` | 纯 nearest 基线对手 |
| `"self"` | 纯自博弈（实时镜像）|
| `"pool"` | 对手池模式（推荐）|

### Pool 模式：Phase 1（nearest / selfplay / selfplay-lag 各 1/3）
**文件：`jax_agent/train_jax.py`，第 503-527 行（主循环内）**

```python
if u < 1/3:   → train_step_nearest
elif u < 2/3: → train_step_selfplay (opponent_params)
else:         → train_step_selfplay (lagged_params)
```

修改概率分配改这三个边界值。

### Pool 模式：Phase 2（selfplay / selfplay-lag 各 1/2）
自动升阶触发后生效，训练仍用 self-play，sigma 通过 eval 跟踪。

### 升阶阈值
**配置项（yaml）：**

```yaml
promote_nearest_win_rate: 1.0    # nearest win_rate 达到此值（全胜）才触发升阶
promote_consecutive_evals: 3     # 连续几次评估都满足才升阶
```

### 对手权重同步频率
**配置项（yaml）：**

```yaml
self_play_update_interval: 50  # selfplay 对手权重每 N 步同步
pool_lag_interval: 50          # lagged 对手权重每 N 步同步
```

### 对手动作生成
**文件：`jax_agent/train_jax.py`，`collect_and_update()` 内 body 函数，第 600-610 行**

self-play 时翻转局面（`flip_state`）后用冻结权重推理；nearest 时直接调用 `nearest_opponent_actions()`。

---

## 5. 评估

### vs nearest（JAX 快速评估，不影响训练速度）
**文件：`jax_agent/train_jax.py`，`eval_vs_nearest_jax()`，第 160 行**

在 JAX fast_env 中跑 `num_games` 局，统计 win/draw/loss（按 winner 字段）。

### vs 外部 agent（通过 kaggle_environments 运行，较慢）
**文件：`jax_agent/train_jax.py`，`eval_vs_external()`，第 208 行**

通用接口，传入 `opponent_dir` 路径即可。已有的外部 agent：
- `orbit-wars-lab/agents/external/sigmaborov-reinforce`
- `orbit-wars-lab/agents/external/tamrazov-starwars`

### 评估触发逻辑
**文件：`jax_agent/train_jax.py`，主循环，第 554 行起**

顺序：nearest → sigma → tamrazov。结果写入 wandb `eval/` 命名空间 + metrics.csv。

### 评估频率 / 局数
**配置项（yaml）：**

```yaml
eval:
  enabled: true
  every: 200    # 每 N 次 update 评估一次
  games: 10     # 每次评估局数
  seed: 200000
```

---

## 6. 网络结构

**文件：`jax_agent/model_jax.py`，`ActorCritic`，第 43 行**

三头输出：
- `fire_logits`：Bernoulli（是否发射）
- `target_logits`：cross-attention 对所有候选星球打分
- `amount_logits`：categorical 选派兵档位

修改网络宽度：`cfg.model.hidden_size`（默认 128）。
启用 CNN 分支：`cfg.model.use_cnn: true`（增加 8x8 小地图感知）。

---

## 7. PPO 超参

**文件：`jax_agent/ppo_jax.py`，`PPOParams`，第 38 行**
**配置项：`cfg.ppo.*`**

| 参数 | 当前值 | 说明 |
|---|---|---|
| `ent_coef` | 0.05 | 熵系数，过低会导致熵崩（log_prob→-inf→NaN）|
| `clip_coef` | 0.25 | PPO clip ε |
| `norm_adv` | true | 是否对 advantage 做 batch 内归一化 |
| `max_grad_norm` | 0.5 | 梯度裁剪（optax.clip_by_global_norm）|

NaN 处理：梯度层用 `nan_to_num` 静默清零；loss/logp/entropy 层保留 `_assert_all_finite` 报错。
