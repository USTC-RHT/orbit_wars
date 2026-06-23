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

## 6. Kaggle file-agent 下不要依赖 `__file__`

2026-06-24 发现一次真实提交问题：本地直接 import agent 正常，Kaggle 比赛中却不发射动作。根因不是模型输出为空，而是 Kaggle 的 Python file-agent runner 执行 `main.py` 时使用类似：

```python
exec(code_object, env)
```

这个 `env` 不包含 `__file__`。如果 `main.py` 只靠：

```python
Path(globals().get("__file__", ".")).resolve().parent
```

来定位提交包目录，线上会退到错误 cwd，导致找不到 `jax_agent/` 或 `models/manifest.json`。如果入口再捕获异常并返回 `[]`，比赛里就表现为“agent 不发射动作”。

旧的可用提交包：

```text
/home/dministrator/kaggle/mine__fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid.tar.gz
```

能发射动作的关键是它在每次 `agent()` 调用开始时读取 Kaggle 注入的：

```python
configuration["__raw_path__"]
```

并用这个路径修正 agent 目录。

推荐入口逻辑：

```python
def _has_manifest(path: Path) -> bool:
    return (path / "models" / "manifest.json").is_file()


def _resolve_agent_dir() -> Path:
    path = globals().get("__file__")
    if path:
        candidate = Path(path).resolve().parent
        if _has_manifest(candidate):
            return candidate

    # Kaggle exec(main.py, env={}) 没有 __file__，但执行期间会把 agent
    # 目录临时 append 到 sys.path，所以可以从 sys.path 恢复。
    for entry in reversed(sys.path):
        if not entry:
            continue
        candidate = Path(entry).resolve()
        if _has_manifest(candidate):
            return candidate

    return Path.cwd().resolve()


def _set_agent_dir_from_config(config: Any = None) -> None:
    global _HERE
    raw_path = config.get("__raw_path__") if isinstance(config, dict) else getattr(config, "__raw_path__", None)
    if raw_path:
        candidate = Path(raw_path).resolve().parent
        if _has_manifest(candidate):
            _HERE = candidate
            sys.path.insert(0, str(_HERE))
```

`agent()` 入口第一步必须调用：

```python
_set_agent_dir_from_config(configuration)
```

## 7. 用 Kaggle 官方 file-agent 路径复现

不要只用 `importlib.util.spec_from_file_location()` 直接 import `main.py`，那会设置 `__file__`，容易掩盖线上问题。应该至少用 Kaggle 自己的 `Agent` 包装层验证：

```python
from kaggle_environments import make
from kaggle_environments.agent import Agent

env = make("orbit_wars", debug=False)
env.reset(2)
agents = [Agent("/path/to/agent/main.py", env), Agent("/path/to/agent/main.py", env)]

for _ in range(80):
    actions, logs = [], []
    for i, a in enumerate(agents):
        action, log = a.act(env.state[i].observation)
        actions.append(action)
        logs.append(log)
    env.step(actions, logs)
```

检查项：

```text
status 不应出现 ERROR/TIMEOUT/INVALID
logs 里不应有 ModuleNotFoundError
前几十步应该能看到 owned fleet 出现
单步 duration 应低于 Orbit Wars actTimeout=1s，或至少不要持续耗尽 remainingOverageTime
```

本次修复后的 pathfix 包：

```text
/home/dministrator/kaggle/submissions/mine__bc-feature-row-step70000-onnx-selective-const55-const51-fp16storage-producer4p-pathfix.tar.gz
sha256: 87dc0ec8f61cbe0faace8a9e6a9a3824fdfa31c500ae5b27618c0fc7706045d0
Kaggle submission id: 53989142
```
