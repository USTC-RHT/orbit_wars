"""Checkpoint-backed JAX policy opponents.

These helpers let a learner use a checkpoint from a different architecture as a
fixed opponent.  For example, a Transformer learner can train against the old
legacy MLP `jax_pool_0514/ckpt_000400.pkl` because the opponent owns its own
ActorCritic module and feature parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pickle

from ..features_jax import FLEET_DIM, FeatureParams
from ..model_jax import ActorCritic


@dataclass(frozen=True)
class CheckpointOpponent:
    """A fixed JAX policy loaded from a training checkpoint."""

    path: Path
    update: int | None
    params: Any
    normalizer_state: Any
    cfg: Any
    feat_params: FeatureParams
    model: ActorCritic


def load_checkpoint_opponent(path: str | Path) -> CheckpointOpponent:
    """Load params, normalizer, feature config, and model for a checkpoint opponent."""

    ckpt_path = Path(path).resolve()
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)

    cfg = ckpt["cfg"]
    normalizer_state = ckpt["normalizer_state"]
    feat_params = FeatureParams.from_train_config(cfg)
    model_cfg = cfg.model
    # Infer feature widths from the checkpoint normalizer so checkpoints from
    # older and current feature layouts instantiate matching model shapes.
    norm_width = int(normalizer_state.planet_mean.shape[0])
    planet_dim = {16: 18, 12: 14, 18: 26}.get(norm_width, norm_width)
    relative_width = int(normalizer_state.relative_mean.shape[0])
    relative_dim = {18: 26, 7: 8}.get(relative_width, relative_width)
    model = ActorCritic(
        planet_dim=planet_dim,
        relative_dim=relative_dim,
        global_dim=11,
        relative_count=cfg.env.max_planets + 1,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=model_cfg.hidden_size,
        use_cnn=getattr(model_cfg, "use_cnn", False),
        use_transformer=getattr(model_cfg, "use_transformer", False),
        fleet_dim=FLEET_DIM,
        max_fleet_tokens=feat_params.max_fleet_tokens,
        num_transformer_layers=getattr(model_cfg, "num_transformer_layers", 2),
        num_heads=getattr(model_cfg, "num_heads", 4),
    )
    return CheckpointOpponent(
        path=ckpt_path,
        update=ckpt.get("update"),
        params=ckpt["params"],
        normalizer_state=normalizer_state,
        cfg=cfg,
        feat_params=feat_params,
        model=model,
    )
