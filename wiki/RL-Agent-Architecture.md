# RL Agent 源码分层

> **所属页面**：[Home](Home.md) > RL Agent 源码分层
> **适用范围**：`rl-agent/jax_agent` 的物理目录结构、入口命令和模块职责。

## 主流程

```text
raw replay zips
  -> jax_agent.preprocessing
  -> BC feature-row shards
  -> jax_agent.bc
  -> BC checkpoint
  -> jax_agent.onnx
  -> Kaggle ONNX runtime package

BC checkpoint / random policy
  -> jax_agent.rl
  -> PPO / self-play checkpoint
```

## 目录职责

| 目录 | 职责 | 典型文件 |
|------|------|----------|
| `jax_agent/common/` | 通用架构层：环境、特征、BC/RL 模型、归一化、garrison/pressure 物理辅助 | `fast_env.py`, `features_jax.py`, `bc_fight_feature_extras.py`, `model_jax.py`, `normalization_jax.py` |
| `jax_agent/preprocessing/` | 数据预处理：raw zip、manifest、player0 view、winner/fire-rate 过滤、state bank | `build_bc_feature_rows_from_zips.py`, `build_bc_player0_view_feature_rows_from_zips.py`, `convert_bc_feature_rows_to_player0_view.py` |
| `jax_agent/bc/` | BC 训练与 checkpoint runtime：feature-row 训练、JAX checkpoint 推理、ONNX runtime bridge、BC smoke eval | `train_bc_jax.py`, `bc_policy_runtime.py`, `bc_numpy_runtime.py`, `bc_onnx_runtime.py` |
| `jax_agent/rl/` | RL/PPO 训练：rollout、PPO update、reward、训练对手 | `train_jax.py`, `ppo_jax.py`, `reward_jax.py`, `train_opponents/` |
| `jax_agent/onnx/` | 导出与纯 NumPy 推理辅助：ONNX export、NumPy feature/aim helpers | `export_bc_onnx.py`, `numpy_inference.py` |
| `jax_agent/evaluation/` | 评估对手适配层 | `eval_opponents/` |

## 常用入口

### 预处理 raw replay zips

```bash
python -m jax_agent.preprocessing.build_bc_feature_rows_from_zips ...
```

selected5 player0-view 快速重建脚本：

```bash
bash scripts/start_selected5_player0_view_preprocess_fast.sh
```

### 训练 BC

```bash
python -m jax_agent.bc.train_bc_jax --config cfg_jax_BC_4090_winner.yaml
```

### 训练 RL

```bash
python -m jax_agent.rl.train_jax --config cfg_jax_RL_1.yaml
```

### 导出 ONNX

```bash
python -m jax_agent.onnx.export_bc_onnx ...
```

## 依赖方向

```text
common
  <- preprocessing
  <- bc
  <- rl
  <- onnx
  <- evaluation

preprocessing -> bc 只允许少量过滤/统计复用
bc -> rl 目前仅复用 aiming/action slot helper
onnx -> bc 读取 BC 模型定义并导出
```

原则：

- `common/` 不依赖 `bc/`、`rl/`、`onnx/`。
- 特征、归一化和模型定义统一放在 `common/`；`bc/` 只负责训练循环、loss、checkpoint 和 runtime glue。
- 预处理生成的是 feature-row contract；BC 训练不重新定义特征 schema。
- Kaggle 提交包里的冻结 agent 不参与源码重构，避免破坏已验证提交包。
- 不保留旧顶层模块入口，例如不再使用 `jax_agent.train_bc_jax` 或 `jax_agent.features_jax`。

## 注释与维护约定

- 关键共享模块需要保留中文模块注释和字段注释，尤其是 feature-row schema、normalizer 统计对象、模型 head、视角镜像逻辑。
- `jax_agent/common/model_jax.py` 是 BC/RL 模型定义的唯一来源；BC 训练、checkpoint runtime、ONNX 导出都直接 import 这里的模型。
- `jax_agent/common/features_jax.py` 和 `jax_agent/common/bc_fight_feature_extras.py` 是特征定义的唯一来源；预处理和 runtime 只做调用/组装，不再复制特征逻辑。
- 修改 `FeatureRowBatch`、`FightFeatureExtras`、`BCPolicyRows` 字段时，需要同步检查预处理、训练、导出和提交 runtime。
