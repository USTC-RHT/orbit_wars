#!/usr/bin/env bash
set -euo pipefail

# Install CUDA 12.4 PyTorch for torch-backed external eval opponents without
# downgrading the CuDNN runtime required by JAX 0.10.0.
#
# Usage:
#   bash scripts/install_eval_torch_cu124.sh
#   PYTHON=/opt/conda/envs/orbit-wars-lab/bin/python bash scripts/install_eval_torch_cu124.sh

PYTHON_BIN="${PYTHON:-python}"
TORCH_VERSION="2.6.0+cu124"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

echo "[install_eval_torch_cu124] python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

"$PYTHON_BIN" -m pip install \
  filelock \
  fsspec \
  jinja2 \
  networkx \
  "sympy==1.13.1" \
  "triton==3.2.0" \
  typing-extensions

"$PYTHON_BIN" -m pip install \
  --force-reinstall \
  --no-deps \
  "torch==${TORCH_VERSION}" \
  --extra-index-url "${TORCH_INDEX}"

# JAX 0.10.0's CUDA plugin is compiled against CuDNN 9.8+. The normal torch
# dependency set pins CuDNN 9.1, so restore a compatible runtime explicitly.
"$PYTHON_BIN" -m pip install --upgrade "nvidia-cudnn-cu12>=9.8,<10"

"$PYTHON_BIN" - <<'PY'
import importlib

import torch

print(f"[install_eval_torch_cu124] torch={torch.__version__} compiled_cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    x = torch.tensor([1.0, 2.0], device="cuda")
    print(f"[install_eval_torch_cu124] torch_cuda_sum={float((x * x).sum().cpu().item())}")

try:
    jax = importlib.import_module("jax")
except Exception as exc:
    print(f"[install_eval_torch_cu124] jax_import=FAILED {exc!r}")
    raise
else:
    print(f"[install_eval_torch_cu124] jax={jax.__version__} devices={jax.devices()}")
PY
