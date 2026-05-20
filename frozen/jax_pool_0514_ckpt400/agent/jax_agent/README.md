# JAX Agent Fast Path

Experimental full-JAX Orbit Wars training stack. It is intentionally parallel
to the existing PyTorch/Kaggle trainer and does not replace `train.py`.

## What Works In V1

- Fixed-array environment state:
  - `planets: [B, P, 7]`
  - `fleets: [B, F, 7]`
  - alive masks plus scalar step/winner fields
- JIT/vmap environment step:
  - launch
  - production
  - fleet movement
  - sun/planet collision
  - rotating planet sweep
  - combat resolution
  - comet spawn/path movement/expiration from precomputed paths
  - terminal scoring
- JAX feature encoder, Flax actor-critic, Optax PPO.
- Running state normalization mirrors the existing Python normalizer.
- Reward shaping mirrored from `src.env_wrapper`:
  - shaped ships/production/planet reward
  - neutral capture reward
  - enemy capture reward
  - lost planet penalty
  - wasted attack penalty
  - time/launch/terminal reward
- Proto target filter is wired to `env.target_filter_enabled`.
- Inbound features are computed by a JAX port of `features.precompute_inbound`
  for regular/rotating planets.
- Cached initial-state bank generated from the Kaggle environment.
- Smoke parity before first comet spawn.

## Known Gaps

- Fast rollout supports JIT-friendly opponents only. Python agents such as
  `sigmaborov-reinforce` should stay in eval until ported.
- Current local JAX install is CPU-only in `orbit-wars-lab`; GPU SPS requires
  CUDA-enabled `jaxlib`.

## Commands

Generate/reset a cached initial state bank:

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python -m jax_agent.build_initial_state_bank \
  --output outputs/jax_initial_state_bank.npz \
  --seeds 256 \
  --start-seed 100000
```

Run no-op parity smoke:

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python -m jax_agent.parity_smoke \
  --config cfg_0509_cnn_no_combat.yaml \
  --seed 12345 \
  --steps 10
```

Benchmark env step:

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python -m jax_agent.bench_jax \
  --config cfg_0509_cnn_no_combat.yaml \
  --num-envs 256 \
  --steps 1000
```

Run a short JAX PPO smoke:

```bash
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python -m jax_agent.train_jax \
  --config cfg_0509_cnn_no_combat.yaml \
  --updates 5 \
  --num-envs 128 \
  --rollout-steps 32
```
