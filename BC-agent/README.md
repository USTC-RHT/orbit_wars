# BC Agent

Standalone behavior-cloning agent for Orbit Wars.

It reuses the local `jax_agent` autoregressive Transformer and trains it from
replay actions. The first smoke path reads Kaggle replay JSON files or
`episodes.tar.gz` archives, canonicalizes player 1 into player 0 perspective,
infers target tokens from action angles with the true Orbit Wars aiming and
first-collision physics, and trains the
`end/src/tgt/amount` decoder heads.

Smoke train:

```bash
cd /home/dministrator/kaggle/rl-agent/BC-agent
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python train_bc.py --replay-glob '/home/dministrator/kaggle/orbit-wars-lab/replays/kaggle/**/episode_*.json' --max-samples 64 --stride 2 --epochs 3
```

The default checkpoint path is `checkpoints/ckpt_bc_latest.pkl`.
