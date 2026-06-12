# 提交包打包注意事项

## 固定基准包

`ckpt48900 producer hybrid` 后续打包统一以这份 tarball 作为干净基准：

```bash
/home/dministrator/kaggle/mine__fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid.tar.gz
```

不要再用 `submissions/` 或 `orbit-wars-lab/submissions/` 里的旧包当基准。这两个目录是输出目录，后续打包会覆盖里面的同名提交包。

对应的工作 agent 目录是：

```bash
/home/dministrator/kaggle/rl-agent/agents/mine/fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid
```

## 允许同步的范围

从基准包刷新提交包时，包结构、模型文件、`main.py`、`producer_main.py`、`orbit_lite/`、`flatbuffers/`、`onnxruntime/` 都应保留基准包里的版本。

物理层和 JAX 训练逻辑的源头统一是顶层目录：

```bash
/home/dministrator/kaggle/rl-agent/jax_agent
```

提交包里的同名物理/训练文件要同步成这里的版本。除非明确决定扩大改动范围，否则只同步下面几个文件：

```text
jax_agent/fast_env.py
jax_agent/features_jax.py
jax_agent/train_jax.py
jax_agent/numpy_inference.py
```

其中前三个文件应直接从顶层 `/home/dministrator/kaggle/rl-agent/jax_agent/` 复制到提交包。`numpy_inference.py` 是 ONNX 提交运行时里对应 JAX 瞄准/物理逻辑的 NumPy 镜像，顶层 `jax_agent` 里通常没有这个文件；只要 `fast_env.py`、`features_jax.py` 或 `train_jax.py` 里的物理/特征/瞄准逻辑变了，就要同步检查并更新 `numpy_inference.py`。

## ONNX Runtime ABI 警告

当前提交包内置的 ONNX Runtime 二进制是给 Python 3.11 用的：

```text
onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so
```

本地常用环境可能是 Python 3.12 或 3.13。`cpython-311` 的 `.so` 不能被 Python 3.12/3.13 直接 import。

需要特别注意：

- 如果本地用 Python 3.12 做冒烟测试，直接导入包内置的 ONNX Runtime 可能会失败。
- `main.py` 里有 ONNX Runtime 导入保护逻辑：非 Python 3.11 时，会把包内 `cpython-311` 的 vendored 路径从 `sys.path` 中移除，然后尝试导入当前环境里的 ONNX Runtime。
- 如果比赛运行环境是 Python 3.12，需要运行环境本身安装兼容 Python 3.12 的 `onnxruntime`，或者重新 vendor 一个 Python 3.12 wheel。
- 如果 Kaggle 运行这个 agent 时用的是 Python 3.11，当前内置 ONNX Runtime 是匹配的。
- 不要随便删除或改写 ONNX Runtime 导入保护逻辑；如果要改，需要同时测试 Python 3.11 和 Python 3.12 路径。

## 重建流程

先从固定基准包重建工作 agent 目录，再把提交包中的物理层/JAX 训练文件同步为顶层 `jax_agent` 的版本：

```bash
BASE=/home/dministrator/kaggle/mine__fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid.tar.gz
TARGET=/home/dministrator/kaggle/rl-agent/agents/mine/fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid

rm -rf "$TARGET"
mkdir -p "$TARGET"
tar -xzf "$BASE" -C "$TARGET"

cp /home/dministrator/kaggle/rl-agent/jax_agent/fast_env.py "$TARGET/jax_agent/fast_env.py"
cp /home/dministrator/kaggle/rl-agent/jax_agent/features_jax.py "$TARGET/jax_agent/features_jax.py"
cp /home/dministrator/kaggle/rl-agent/jax_agent/train_jax.py "$TARGET/jax_agent/train_jax.py"
# numpy_inference.py 需要从维护好的提交运行时版本复制；复制前先确认物理/瞄准逻辑已同步。
```

打包时排除本地元数据和缓存：

```bash
NAME=mine__fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid.tar.gz
cd "$TARGET"
tar --exclude='./agent.yaml' --exclude='./__pycache__' --exclude='*/__pycache__' --exclude='*.pyc' -czf "/tmp/$NAME" .
cp -f "/tmp/$NAME" /home/dministrator/kaggle/submissions/"$NAME"
cp -f "/tmp/$NAME" /home/dministrator/kaggle/orbit-wars-lab/submissions/"$NAME"
```

## 验证清单

提交前先跑物理碰撞测试：

```bash
python -m pytest tests/test_fast_env_collision.py -q
```

如果默认 Python 没装 `pytest`，用已知可用的环境：

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python -m pytest tests/test_fast_env_collision.py -q
```

检查新包相对基准包只差允许同步的文件：

```bash
rm -rf /tmp/ow_base_48900 /tmp/ow_new_48900
mkdir -p /tmp/ow_base_48900 /tmp/ow_new_48900
tar -xzf /home/dministrator/kaggle/mine__fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid.tar.gz -C /tmp/ow_base_48900
tar -xzf /home/dministrator/kaggle/submissions/mine__fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid.tar.gz -C /tmp/ow_new_48900
diff -qr /tmp/ow_base_48900 /tmp/ow_new_48900
```

预期差异应只包含：

```text
jax_agent/fast_env.py
jax_agent/features_jax.py
jax_agent/train_jax.py
jax_agent/numpy_inference.py
```
