# Orbit Wars BC 分支 Wiki

> **范围**：JAX Orbit Wars agent，以及行为克隆 / imitation learning 相关工作。
> **数据根目录**：`/mnt/f/kaggle-orbit-wars-data`

## 页面

| 页面 | 说明 |
|------|-------|
| [RL Agent Architecture](RL-Agent-Architecture.md) | 源码物理分层、入口命令和模块职责 |
| [Behavioral Cloning Training Notes](BC-Training-Notes.md) | 数据质量、label、loss 和 BC 常见坑 |
| [Kaggle BC Discussion Translation](BC-Discussion-Translation.md) | BC / imitation-learning 讨论的中文翻译 |
| [Kaggle Submission Packaging Notes](Kaggle-Submission-Packaging-Notes.md) | Kaggle tarball 中 vendored ONNX Runtime 的注意事项 |

## 仓库结构

```text
rl-agent/
  jax_agent/
    common/         共享环境、特征、模型、normalizer、物理辅助
    preprocessing/  raw replay zip -> feature-row 数据构建
    bc/             行为克隆训练与 runtime
    rl/             PPO/self-play 训练
    onnx/           ONNX 导出与 NumPy 推理辅助
    evaluation/     对手适配器
  wiki/             BC 与 imitation-learning 笔记
  cfg_jax_*.yaml    当前 JAX 训练配置
```
