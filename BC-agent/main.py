from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

def _guess_here() -> Path:
    raw_file = globals().get("__file__")
    if raw_file:
        return Path(raw_file).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd / "BC-agent",
        cwd / "rl-agent" / "BC-agent",
        cwd / "agents" / "mine" / "BC-agent",
        cwd / "orbit-wars-lab" / "agents" / "mine" / "BC-agent",
        cwd / "agents" / "mine" / "producer-bc-transformer",
        cwd / "orbit-wars-lab" / "agents" / "mine" / "producer-bc-transformer",
    ]
    for candidate in candidates:
        if (candidate / "checkpoints" / "ckpt_bc_latest.pkl").is_file():
            return candidate
    return cwd


_HERE = _guess_here()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp

from jax_agent.fast_env import EnvParams, empty_state
from jax_agent.features_jax import FLEET_DIM, GLOBAL_DIM, PLANET_DIM, RELATIVE_DIM, FeatureParams
from jax_agent.model_jax import AutoregressiveActorCritic
from jax_agent.normalization_jax import NormalizerParams
from jax_agent.train_jax import (
    auto_deterministic_policy_actions,
    flip_actions,
    flip_state,
    jax_actions_to_kaggle,
    kaggle_obs_to_jax_state,
)


_RUNTIME = None


def _read(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _is_four_player(observation: Any) -> bool:
    owners: set[int] = set()
    for key in ("planets", "fleets"):
        for row in _read(observation, key, []) or []:
            if len(row) >= 2 and int(row[1]) >= 0:
                owners.add(int(row[1]))
    return len(owners) >= 3 or any(owner >= 2 for owner in owners)


class _BcRuntime:
    def __init__(self) -> None:
        ckpt_path = _HERE / "checkpoints" / "ckpt_bc_latest.pkl"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"missing BC checkpoint: {ckpt_path}")
        with ckpt_path.open("rb") as f:
            ckpt = pickle.load(f)
        self.cfg = ckpt["cfg"]
        self.params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
        self.normalizer_state = jax.tree_util.tree_map(jnp.asarray, ckpt["normalizer_state"])
        self.feat_params = FeatureParams.from_train_config(self.cfg)
        self.norm_params = NormalizerParams.from_train_config(self.cfg)
        self.env_params = EnvParams(
            episode_steps=int(self.cfg.env.episode_steps),
            max_planets=int(self.cfg.env.max_planets),
            max_fleets=512,
            max_actions=int(self.cfg.env.max_planets),
        )
        layers = (
            int(getattr(self.cfg.model, "num_transformer_layers", 2))
            if bool(getattr(self.cfg.model, "use_transformer", False))
            else 0
        )
        self.model = AutoregressiveActorCritic(
            planet_dim=PLANET_DIM,
            relative_dim=RELATIVE_DIM,
            global_dim=GLOBAL_DIM,
            relative_count=int(self.cfg.env.max_planets) + 1,
            ship_bucket_count=int(self.cfg.env.ship_bucket_count),
            hidden_size=int(self.cfg.model.hidden_size),
            use_cnn=bool(self.cfg.model.use_cnn),
            fleet_dim=FLEET_DIM,
            max_fleet_tokens=int(getattr(self.cfg.model, "max_fleet_tokens", 32)),
            num_transformer_layers=layers,
            num_heads=int(getattr(self.cfg.model, "num_heads", 4)),
            max_action_tokens=int(getattr(self.cfg.model, "max_action_tokens", 16)),
            end_bias_init=float(getattr(self.cfg.model, "end_bias_init", -2.0)),
        )
        self.train_state = SimpleNamespace(params=self.params, apply_fn=self.model.apply)
        self._act_jit = jax.jit(self._act_state)
        dummy = jax.tree_util.tree_map(lambda x: x[None], empty_state(self.env_params))
        self._act_jit(dummy).block_until_ready()

    def _act_state(self, state):
        return auto_deterministic_policy_actions(
            self.train_state,
            self.model,
            state,
            self.feat_params,
            self.normalizer_state,
            self.norm_params,
        )

    def act(self, observation: Any) -> list:
        if _is_four_player(observation):
            return []
        player = int(_read(observation, "player", 0) or 0)
        state = kaggle_obs_to_jax_state(observation, self.env_params)
        if player == 1:
            state = flip_state(state)
        actions = self._act_jit(state)
        if player == 1:
            actions = flip_actions(actions)
        return jax_actions_to_kaggle(actions)


def agent(obs, config=None):
    del config
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _BcRuntime()
    try:
        return _RUNTIME.act(obs)
    except Exception:
        return []


try:
    _RUNTIME = _BcRuntime()
except Exception:
    _RUNTIME = None
