# Kaggle 提交包打包注意事项

> **所属页面**：[Home](Home.md) > Kaggle 提交包打包注意事项
> **适用范围**：Orbit Wars ONNX / NumPy runtime agent 的 Kaggle submission tarball

## 1. ONNX Runtime 必须随提交包一起带上

Kaggle 提交运行时不会使用本机 conda 环境，也不会自动 `pip install onnxruntime`。如果 agent 的 `main.py` 或 runtime 里有：

```python
import onnxruntime as ort
```

提交包里必须 vendored 一个可直接 import 的 `onnxruntime/` 目录。不要只打包 `.onnx` 模型文件。

已检查的历史提交包：

```text
/home/dministrator/kaggle/mine__fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid.tar.gz
```

它确实包含：

```text
onnxruntime/
onnxruntime/__init__.py
onnxruntime/capi/onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so
onnxruntime/capi/libonnxruntime.so.1.26.0
onnxruntime/capi/libonnxruntime_providers_shared.so
flatbuffers/
models/ckpt_048900/*.onnx
```

这个包大小约 `21M`，总文件数 `403`。其中 `onnxruntime/capi/*.so` 是关键；只有 Python 源码目录但没有这些 native shared libraries，导入 ORT 会失败。

## 2. Python ABI 要匹配 Kaggle 运行时

历史包里带的是：

```text
onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so
```

这表示它面向 Linux x86_64 + CPython 3.11。重新 vendoring ORT 时，要确认 Kaggle competition runtime 的 Python 版本仍然匹配。若 Kaggle 运行时换成 Python 3.12/3.13，这个 `.so` 可能无法 import，需要重新复制对应版本的 onnxruntime wheel 内容。

## 3. 打包前检查清单

在 agent 目录打 tarball 前，至少确认：

```bash
find AGENT_DIR -maxdepth 2 -type d -name onnxruntime -o -name flatbuffers
find AGENT_DIR -path '*/onnxruntime/capi/*.so*' -type f
find AGENT_DIR/models -name '*.onnx' -o -name '*.onnx.data' -o -name 'manifest.json' -o -name 'meta.json'
```

提交包生成后，再检查 tarball：

```bash
tar -tzf submission.tar.gz | rg '^(\\./)?onnxruntime/|^(\\./)?flatbuffers/|\\.onnx(\\.data)?$'
tar -tzvf submission.tar.gz | rg 'onnxruntime/capi/.*\\.so'
```

如果这些命令看不到 `onnxruntime/capi/libonnxruntime.so.*` 和 `onnxruntime_pybind11_state...so`，这个包大概率不能在 Kaggle 上跑 ONNX。

## 4. 本地验证不要被 conda 环境误导

本机 `orbit-wars-lab` conda 环境里装了 `onnxruntime`，所以本地评估即使 agent 目录没有 vendored ORT 也可能正常运行。但 Kaggle 提交包是隔离运行的，应该用临时目录模拟：

```bash
tmp=$(mktemp -d)
tar -xzf submission.tar.gz -C "$tmp"
PYTHONPATH="$tmp" python - <<'PY'
import onnxruntime as ort
print(ort.__version__, ort.get_available_providers())
PY
```

更严格时，用一个没有安装 `onnxruntime` 的 Python 环境做这个测试。

## 5. 当前 BC ONNX agent 的特别提醒

当前这些本地打包目录可以在 conda 环境中跑，但目录本身尚未 vendored `onnxruntime/`：

```text
orbit-wars-lab/agents/mine/bc-feature-row-step70000-onnx
orbit-wars-lab/agents/mine/bc-rl2-n1akl7wj-step10000-onnx
orbit-wars-lab/agents/mine/bc-rl2-latest-step10000-onnx
```

如果要把它们提交到 Kaggle，需要在生成 `.tar.gz` 前复制 `onnxruntime/` 和必要依赖目录，例如 `flatbuffers/`，并重新做 tarball 级别检查。

