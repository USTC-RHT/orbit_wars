# jax_pool_0514_ckpt400 Frozen Project

这个文件夹是 `outputs/jax_pool_0514_jax/ckpt_000400.pkl` 对应 agent 的冻结版。

目的：

- 保留一个不受当前 Transformer / PFSP / schedule 实验影响的干净版本。
- 方便重新打包 Kaggle submission 或放到 Orbit Wars Lab 里可视化。
- 记录 ckpt400 当时的真实配置，避免后续把新配置误认为原始配置。

## 目录

```text
agent/
  agent.yaml              # Orbit Wars Lab 元数据
  main.py                 # Kaggle / Orbit Wars Lab agent 入口
  weights/ckpt_000400.pkl # jax_pool_0514 checkpoint 400
  jax_agent/              # 运行该 agent 所需的 JAX 推理代码
  src/                    # 最小兼容层：checkpoint 反序列化和 observation 解析

JAX_POOL_0514_CKPT400.md  # ckpt400 原始配置说明
cfg_jax_fixed_pool_legacy.yaml
cfg_jax_pool_0514_resume.yaml
```

## 重要性质

- 这个 checkpoint 是 legacy JAX MLP agent。
- 不是 CNN。
- 不是 Transformer。
- `agent/src/` 已裁剪为最小集合，只保留 `config.py` / `state.py`。
- 原始配置里 `model.use_cnn: false`，旧配置类没有 `use_transformer` 字段；在当前语义里等价于 `use_transformer: false`。
- 模型选择目标的方式是 per-source MLP + pointer attention。

## 打包 submission

从这个目录执行：

```bash
cd rl-agent/frozen/jax_pool_0514_ckpt400/agent
zip -r ../jax-pool-0514-ckpt400-frozen.zip . -x '**/__pycache__/*' '*.pyc'
```

生成的 zip 在：

```text
rl-agent/frozen/jax_pool_0514_ckpt400/jax-pool-0514-ckpt400-frozen.zip
```

## 放到 Orbit Wars Lab

可以直接复制 `agent/` 到 lab 的 agent 目录，例如：

```bash
rsync -a --delete rl-agent/frozen/jax_pool_0514_ckpt400/agent/ \
  orbit-wars-lab/agents/mine/jax-pool-0514-ckpt400-frozen/
```

`agent/` 里已经带了 `agent.yaml`。
