# Orbit Wars BC Branch Wiki

> **Scope**: JAX Orbit Wars agent plus Behavioral Cloning / imitation-learning work.
> **Data root**: `/mnt/f/kaggle-orbit-wars-data`

## Pages

| Page | Notes |
|------|-------|
| [Behavioral Cloning Training Notes](BC-Training-Notes.md) | Data quality, labels, losses, and BC pitfalls |
| [Kaggle BC Discussion Translation](BC-Discussion-Translation.md) | Chinese translation of the BC / imitation-learning discussion |
| [Imitation Learning Plan](Imitation-Learning-Plan.md) | Implementation path from replay zips to BC checkpoint and PPO warm start |

## Repository Layout

```text
rl-agent/
  jax_agent/        JAX model, features, environment, PPO training code
  tests/            Focused regression tests
  wiki/             BC and imitation-learning notes
  cfg_jax_*.yaml    Current JAX training configs
```
