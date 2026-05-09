# RL Agent

这是一个面向 Orbit Wars 的 PPO 强化学习 agent。它没有直接套 SB3，
但训练结构刻意整理成类似 SB3 的风格：先收集 rollout，再计算 GAE，
最后用 PPO clipped loss 做多轮 minibatch 更新。

核心设计：

- 每颗我方星球独立产生一行决策。
- 网络从全图目标槽位里选择“打谁/不打”，再为选中目标选择“派多少船”。
- 动作由两个离散分布组成：`target_index` 和 `amount_bucket`。
- 物理层不参与全图枚举；只在最终选中一条动作后计算发射角、ETA 和碰撞。
- reward 使用终局胜负 + 少量局势变化 shaping。
- PPO 更新集中在 `src/ppo.py`，方便后续改 loss、entropy、KL 早停等。

## 目录

```text
RL/
  README.md           # 项目说明
  agent.yaml          # orbit-wars-lab agent 元信息
  default_cfg.yaml    # 默认训练配置
  main.py             # Kaggle/web 推理入口
  train.py            # 训练命令入口
  eval.py             # checkpoint 评估入口
  plot_metrics.py     # 训练曲线绘图
  src/
    config.py         # 配置 dataclass 和 YAML 加载
    state.py          # Kaggle observation 解析
    features.py       # 特征工程、全图目标槽位、派兵档位
    model.py          # Actor-Critic 网络
    ppo.py            # 动作分布、PPO loss、参数更新
    opponents.py      # random / nearest / self-play 对手
    env_wrapper.py    # 环境封装和 reward 计算
    train_loop.py     # rollout、GAE、日志、checkpoint
  artifacts/          # 训练输出目录，checkpoint 和 metrics 会写到这里
```

## 快速训练

```bash
cd /home/dministrator/kaggle
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python RL/train.py \
  --config RL/default_cfg.yaml
```

跑一个 1000 update 实验：

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python RL/train.py \
  --config RL/default_cfg.yaml \
  --run-name sb3_style_nearest_1000 \
  --updates 1000 \
  --num-envs 8 \
  --rollout-steps 32 \
  --device auto \
  --opponent nearest
```

如果机器有 CUDA，日志开头会显示：

```text
device=cuda
```

## 动作空间

网络不是直接输出角度，而是先输出两个离散选择：

```text
target_index   # 0 表示 no-op，其余表示候选目标
amount_bucket  # 派兵比例档位
```

现在不再手写候选目标。`features.py` 会把全图当前星球 padding 到固定槽位：

```text
slot 0       = no-op
slot 1..45   = 当前全图星球/彗星
```

`target_index` 直接指向这些槽位。`features.py` 只做轻量合法性 mask：

- padding 不可选。
- 当前来源星球自己不可选。
- 来源星球没有船时不能发兵。
- 派兵档位必须大于 0。

真正的物理可达性不在全图上预先枚举。`opponents.py::action_from_indices`
只对网络最终选中的 `src + target + ships` 调 physics，成功后还原成环境动作：

```text
[source_planet_id, angle, ships]
```

当前 `default_cfg.yaml` 里 `ship_bucket_count: 8`，也就是 8 个派兵档位。
档位覆盖 `1` 到当前来源星球全部船数。

## Reward

终局奖励来自环境：

- 赢：`+1`
- 输：`-1`
- 平/异常：`0`

每一步还会给一点 shaped reward：

- 我方星球数增加，加分。
- 我方星球上舰船数、飞行中舰队数、生产力上升，加分。
- 每步 shaping 会被 `step_clip` 裁剪，避免盖过终局胜负。

这让 PPO 不必等到几百回合后才知道前面的扩张动作是否有价值。

## 网络结构

`src/model.py` 里的 `ActorCritic` 是三路编码 + dot-product target selector：

```text
self_features      -> self_encoder   -> self_h
candidate_features -> target_encoder -> target_h
global_features    -> global_encoder -> global_h

target_h -- masked max/mean pool --> map_h

query = MLP(self_h + global_h + map_h)
keys  = Linear(target_h)

target_logits = dot(query, keys)
sample target_i

amount_logits = MLP(self_h + global_h + map_h + target_h_i)
critic        = MLP(self_h + global_h + map_h)
```

训练时先从 target categorical 采样目标，再把采样出来的目标 embedding
传给 amount head，得到该目标对应的派兵分布。两个 `log_prob` 相加作为
PPO 的动作概率。

训练和推理的动作选择不一样：

- 训练 rollout：`target` 和 `amount` 都从 categorical 中采样，用来保持探索。
- PPO 更新：使用 rollout 时已经采样出的 `target/amount` 重新计算 log_prob，不重新采样。
- Kaggle/web 推理和 `eval.py`：`target` 和 `amount` 都用 argmax，行为更稳定。

## PPO Loss

`src/ppo.py` 里实现了接近 SB3 风格的 PPO 更新：

```text
loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
```

- `policy_loss`: PPO clipped objective
- `value_loss`: return 和 critic value 的均方误差，可做 value clipping
- `entropy`: 鼓励探索
- `approx_kl`: 监控新旧策略距离，超过阈值会提前停止本轮更新

GAE 在 `src/train_loop.py::compute_gae` 里计算：

```text
advantage_t = delta_t + gamma * lambda * advantage_{t+1}
```

## 评估

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python RL/eval.py \
  --config RL/default_cfg.yaml \
  --checkpoint RL/artifacts/sb3_style_nearest_1000/ckpt_last.pt \
  --games 20 \
  --seed 2000 \
  --opponent nearest \
  --device cpu
```

## 画图

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python RL/plot_metrics.py \
  --csv RL/artifacts/sb3_style_nearest_1000/metrics.csv \
  --window 20
```
