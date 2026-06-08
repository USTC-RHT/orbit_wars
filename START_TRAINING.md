# Cloud Training Setup

The local training environment currently used for this project is:

- Conda env: `orbit-wars-lab`
- Python: `3.12.13`
- JAX: `0.10.0`
- JAX backend on this machine: `gpu`
- Local GPU: `NVIDIA GeForce RTX 3070 Ti`, 8 GB

## Create Environment

From the `rl-agent` directory:

```bash
conda env create -f environment.yml
conda activate orbit-wars-lab
```

If the cloud image already has a suitable conda env, install only the pip deps:

```bash
python -m pip install -r requirements.txt
```

## Verify JAX GPU

```bash
python - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
print(jax.default_backend())
PY
```

Expected output should include a CUDA/GPU device and `gpu` backend.

## Start Current Training

```bash
python -m jax_agent.train_jax --config cfg_jax_fixed_pool_4090.yaml
```

For wandb online logging:

```bash
wandb login
```

If the cloud platform cannot access wandb, change `wandb.mode` in the config to
`offline`, or set `wandb.enabled: false`.

## Notes

- `cfg_jax_fixed_pool_4090.yaml` is currently set to `total_updates: 50000`
  and targets multi-GPU training. For a smaller local run, use
  `cfg_jax_fixed_pool_0525.yaml`.
- The config evaluates against `nearest`, `sigma`, `tamrazov`, and
  `pool_ckpt900`; bundled external opponents live under `rl-agent/agents/...`.
- The requirements pin the Kaggle environment by Git commit to reduce mismatch
  between local physics tests and cloud training.
