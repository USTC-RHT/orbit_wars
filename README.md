# Orbit Wars RL Agent

这是 Orbit Wars Kaggle 比赛用的强化学习 / 行为克隆 agent 仓库。

项目目标是从 replay 数据中提取固定形状特征，训练 BC 策略作为强先验，再继续做
RL/PPO、自博弈评估，并最终导出 Kaggle 可提交的 ONNX agent 包。

## 当前主线

当前推荐工作分支是 `BC`。

`BC` 分支已经把源码按工作流重新整理：

```text
jax_agent/
  common/         共享环境、特征、模型、normalizer、物理辅助
  preprocessing/  raw replay zip -> feature-row 数据构建
  bc/             行为克隆训练、checkpoint runtime、BC 评估
  rl/             PPO / self-play 训练
  onnx/           ONNX 导出与 NumPy 推理辅助
  evaluation/     外部 / 打包对手适配
data_stats/       leaderboard、winner fire-rate、selected5 对比等统计 JSON
scripts/          常用启动脚本
wiki/             训练、打包、架构和问题记录
```

老分支可能仍是 `jax_agent/*.py` 平铺结构；README 的分支说明覆盖整个仓库历史。

## 常用入口

预处理 raw replay zips：

```bash
python -m jax_agent.preprocessing.build_bc_feature_rows_from_zips ...
```

构建 selected5 player0-view 数据：

```bash
bash scripts/start_selected5_player0_view_preprocess_fast.sh
```

训练 BC：

```bash
python -m jax_agent.bc.train_bc_jax --config cfg_jax_BC_4090_winner.yaml
```

训练 RL / PPO：

```bash
python -m jax_agent.rl.train_jax --config cfg_jax_RL_1.yaml
```

导出 BC ONNX：

```bash
python -m jax_agent.onnx.export_bc_onnx --checkpoint <ckpt.pkl> --out-dir <models_dir>
```

## 配置文件

- `cfg_jax_BC_4090_winner.yaml`：Top 玩家 winner-only BC 训练配置。
- `cfg_jax_BC_4090_all.yaml`：winner + loser / all rows BC 训练配置。
- `cfg_jax_BC_4090.yaml`：BC 实验配置模板。
- `cfg_jax_RL_1.yaml`：PPO / self-play 训练配置。

## 分支地图

| 分支 | 用途 |
|---|---|
| `BC` | 当前行为克隆主线；feature-row 数据、cross-attention BC、ONNX 导出、提交包调试。 |
| `main` | 早期 RL 主线；fire actor 二分类头、reward shaping overhaul。 |
| `v1-base` | 最早的基础 RL agent。 |
| `v2-inbound` | 在基础 RL 上加入 inbound features 和 noop bias。 |
| `fast-env` | 固定形状 JAX 环境和高速 rollout 方向。 |
| `Go_for_4P` | 面向 4P 的 reward / agent 实验。 |
| `2P-Improve` | 2P 改进分支；包含早期 torch-agent / BC agent 尝试。 |
| `autoregressive-actor-v1` | 自回归 actor 的第一版实验。 |
| `transformer-pointer-v2` | Transformer pointer 目标选择实验。 |
| `Transformer-V1` | Transformer 模型方向的早期整理和中文注释。 |
| `reachability-amount-mask` | 可达性 mask、amount mask、自博弈采样实验。 |
| `target-mask-attack-ships` | 攻击船数相关 target mask 实验。 |
| `target-mask-post-discard-attack-ships` | post-discard target mask 和 lag normalizer 修复。 |
| `proto-target-filter-topk6-defense3` | proto target filter：进攻 top6、防守 top3。 |
| `planned-launch-features` | planned launch coordination features 实验。 |
| `Fight` | Fight/producer 风格物理特征、对手与评估实验。 |
| `Fight-no-amount-head` | Fight 分支去掉 amount head 的对照实验。 |

## 维护约定

- `common/` 是共享 contract：特征、模型、normalizer、物理辅助都从这里复用。
- BC 训练、BC runtime、ONNX 导出不要各自复制模型定义。
- 统计 JSON 放在 `data_stats/`；训练/预处理输出目录里的 manifest/progress/meta 保持在各自输出目录。
- Kaggle 提交包和冻结 agent 不随源码重构随意改动，避免破坏已验证包。
- 如果 README 更新需要同步到其它分支，优先在一个分支改好后用 `git cherry-pick` 或 merge 传播。
