# jax_pool_0514 ckpt400 配置说明

这个文档记录 `outputs/jax_pool_0514_jax/ckpt_000400.pkl` 里保存的真实原始配置。
后面的“当前续训配置”是后来为了接着训练单独改的，不属于原始训练配置。

## Checkpoint

- 路径：`outputs/jax_pool_0514_jax/ckpt_000400.pkl`
- 保存 update：`400`
- 原始 run name：`jax_pool_0514`
- 原始 opponent：`pool`
- 参数量：约 `305,674`

## Checkpoint 内保存的原始配置

下面这份 YAML 是直接从 `ckpt_000400.pkl["cfg"]` 导出的原始配置，不是手写推测。

```yaml
seed: 42
run_name: jax_pool_0514
device: auto
save_dir: outputs
checkpoint_every: 100
log_every: 1
opponent: pool
opponent_pool:
- nearest
opponent_pool_weights:
- 1.0
self_play_update_interval: 50
pool_lag_interval: 50
promote_nearest_win_rate: 1.0
promote_consecutive_evals: 2
self_play_deterministic: true
alternate_player_sides: true

env:
  board_size: 100.0
  episode_steps: 500
  ship_bucket_count: 8
  max_planets: 45
  max_ships: 500.0
  max_production: 5.0
  target_filter_enabled: true
  attack_top_k: 6
  defense_top_k: 3

reward:
  terminal_weight: 1.0
  ships_weight: 0.0
  production_weight: 0.0
  planet_weight: 0.0
  time_penalty: 0.0
  launch_penalty: 0.0
  neutral_capture_weight: 0.0
  neutral_capture_eta_scale: 12.0
  neutral_capture_early_steps: 150
  enemy_capture_weight: 0.0
  lose_planet_penalty: 0.0
  wasted_attack_penalty: 0.0
  step_clip: 0.0

model:
  hidden_size: 128
  use_cnn: false

state_norm:
  enabled: true
  clip: 5.0
  eps: 1.0e-08

eval:
  enabled: true
  every: 100
  games: 20
  seed: 200000

ppo:
  rollout_steps: 128
  num_envs: 64
  total_updates: 5000
  epochs: 4
  minibatch_size: 4096
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.25
  ent_coef: 0.05
  vf_coef: 0.5
  lr: 0.0003
  max_grad_norm: 0.5
  anneal_lr: true
  norm_adv: true
  clip_vloss: true
  target_kl: 0.03

wandb:
  enabled: true
  project: orbit-wars-rl
  entity: null
  mode: online
  group: null
  tags: []
```

## 模型结构

这个版本不是 CNN，也不是 Transformer，而是 legacy per-source MLP + pointer attention。

严格按 checkpoint 原始配置说，`model` 里只有：

```yaml
hidden_size: 128
use_cnn: false
```

当时的配置类还没有 `use_transformer` 字段；在当前代码里加载旧 ckpt 时，缺省等价于 `use_transformer: false`。

每个我方星球作为一个 source，独立输出一行动作决策：

```text
planet_features      -> MLP -> self_h
candidate_features -> MLP -> cand_h
global_features    -> MLP -> global_h

cand_h over planets -> masked max/mean pool -> map_h

[self_h, global_h, map_h] -> context_h

context_h -> fire head              -> 是否发射
context_h x cand_h -> pointer score -> 目标星球
[context_h, selected_target_h]      -> 派兵档位
context_h -> value head             -> source 行 value
```

关键点：

- `use_cnn: False`
- `use_transformer: False`
- `hidden_size: 128`
- 目标选择不是普通分类头，而是单头 scaled dot-product pointer：

```python
q = f(context_h)
keys = g(cand_h)
target_logits = q · keys / sqrt(hidden_size)
```

## 原始环境配置摘录

```yaml
env:
  board_size: 100.0
  episode_steps: 500
  ship_bucket_count: 8
  max_planets: 45
  max_ships: 500.0
  max_production: 5.0
  target_filter_enabled: true
  attack_top_k: 6
  defense_top_k: 3
```

`attack_top_k=6` 和 `defense_top_k=3` 是当时目标过滤的配置。

## 原始奖励配置摘录

这个版本只用了终局奖励，没有 dense shaping：

```yaml
reward:
  terminal_weight: 1.0
  ships_weight: 0.0
  production_weight: 0.0
  planet_weight: 0.0
  time_penalty: 0.0
  launch_penalty: 0.0
  neutral_capture_weight: 0.0
  enemy_capture_weight: 0.0
  lose_planet_penalty: 0.0
  wasted_attack_penalty: 0.0
  step_clip: 0.0
```

## 原始 PPO 配置摘录

```yaml
ppo:
  num_envs: 64
  rollout_steps: 128
  total_updates: 5000
  epochs: 4
  minibatch_size: 4096
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.25
  ent_coef: 0.05
  vf_coef: 0.5
  lr: 3.0e-4
  max_grad_norm: 0.5
  anneal_lr: true
  norm_adv: true
  clip_vloss: true
  target_kl: 0.03
```

## 原始 pool 逻辑说明

原始配置里没有 `pool_mode`、`pool_start_phase`、`pool_max_size` 这些后来新增字段。
因此原始 ckpt 只能确认这些字段：

- `self_play_update_interval: 50`
- `pool_lag_interval: 50`
- `promote_nearest_win_rate: 1.0`
- `promote_consecutive_evals: 2`

结合当时训练代码/日志语义，它对应的是旧版二阶段 pool：

旧版 fixed pool 的阶段含义：

```text
Phase 1:
  nearest / selfplay-current / selfplay-lag

Phase 2:
  selfplay-current / selfplay-lag
```

## 当前续训配置

当前续训配置文件：

`cfg_jax_pool_0514_resume.yaml`

这部分不是原始配置，是为了从 ckpt400 接着训新增/修改的配置。

它从 ckpt400 初始化：

```yaml
init_from_checkpoint: "outputs/jax_pool_0514_jax/ckpt_000400.pkl"
start_update_from_checkpoint: true
```

并且直接从 Phase 2 开始：

```yaml
opponent: "pool"
pool_mode: fixed
pool_start_phase: 2
```

也就是说当前训练只采样：

```text
selfplay-current
selfplay-lag
```

nearest 不参与训练，只保留做 eval benchmark。

## 当前续训相对原始配置的变化

为了从 ckpt400 更稳地 fine-tune，当前续训把 PPO 调保守了一些：

| 项 | 原始 ckpt400 | 当前续训 |
|---|---:|---:|
| `lr` | `3.0e-4` | `1.0e-4` |
| `clip_coef` | `0.25` | `0.15` |
| `ent_coef` | `0.05` | `0.02` |
| `target_kl` | `0.03` | `0.03` |
| `epochs` | `4` | `4` |

这些改动的目的不是改变模型能力，而是避免从旧 checkpoint 接着训时策略突然漂移太大。

## 续训限制

当前 checkpoint 只保存了：

```text
params
normalizer_state
update
cfg
```

没有保存 optimizer state。因此当前续训是 fine-tune：

- 模型参数从 ckpt400 继续
- normalizer 从 ckpt400 继续
- update 编号从 401 继续
- Adam optimizer 动量重新初始化
