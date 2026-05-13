# JAX Parity Status

Goal: make `jax_agent` match the existing RL training logic before using it as
the main training path.

## Matched Now

- Fixed-array state mirrors current observation rows for planets/fleets.
- Core physics:
  - launch
  - production
  - fleet speed
  - sun collision with `SUN_RADIUS=10`
  - planet collision
  - rotating planet sweep
  - combat resolution
  - terminal scoring
- Policy structure:
  - fire Bernoulli
  - target categorical
  - autoregressive amount categorical
- CNN map encoder structure.
- Proto target filter config is applied.
- `features.precompute_inbound` fills self/candidate inbound feature slots
  instead of using hard-coded defaults, including comet path positions.
- Defense target filtering uses the same defense-pressure formula as Python.
- Reward shaping mirrors the current Python reward components.
- Online state normalization mirrors the Python running mean/std indices and
  rollout timing.
- Comet spawn/path/movement/expiration is backed by precomputed state-bank
  paths and participates in planet sweep collision.
- `aim_with_prediction_jax` handles rotating/comet targets, sun blocking, and
  checks that the selected target is the first collision object among planets.
- `parity_smoke` no-op test matches Kaggle through the first comet spawn.

## Not Yet Fully Matched

- Python-only opponents, especially `sigmaborov-reinforce`, are not JIT-compatible
  and need a JAX port or a separate Python eval path.

## Next Parity Tasks

1. Add frame-by-frame parity tests for launch, collision, combat, and comet cases.
