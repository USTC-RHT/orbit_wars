"""JAX pool PPO agent - jax_pool_0514 ckpt_000400 for Orbit Wars Lab."""

from __future__ import annotations

import pickle
import sys
import threading
from pathlib import Path
from typing import Any

_HERE = Path(globals().get("__file__", ".")).resolve().parent
# 计算 rl-agent 根路径：orbit-wars-lab/agents/mine/jax-pool-0514-ckpt400/ -> up 4 levels -> kaggle/
# 如果 __file__ 不在 exec 环境中（kaggle_environments exec 模式），_HERE.parents 可能不够深
_RL_ROOT = _HERE

_INFER_FN = None
_ENV_PARAMS = None
_LOAD_ERROR: BaseException | None = None
_LOAD_THREAD: threading.Thread | None = None
_LOAD_LOCK = threading.Lock()


def _set_agent_dir(config: Any = None) -> None:
    global _HERE, _RL_ROOT
    raw_path = config.get("__raw_path__") if isinstance(config, dict) else getattr(config, "__raw_path__", None)
    if raw_path:
        _HERE = Path(raw_path).resolve().parent
        _RL_ROOT = _HERE


def _load_model() -> None:
    global _INFER_FN, _ENV_PARAMS, _LOAD_ERROR
    try:
        if str(_RL_ROOT) not in sys.path:
            sys.path.insert(0, str(_RL_ROOT))

        import jax
        import jax.numpy as jnp

        from jax_agent.fast_env import (
            EnvParams,
            EnvState,
            MAX_COMET_SPAWNS,
            COMETS_PER_SPAWN,
            MAX_COMET_PATH,
        )
        from jax_agent.features_jax import FeatureParams, encode_batch, flatten_sources
        from jax_agent.model_jax import ActorCritic
        from jax_agent.normalization_jax import NormalizerParams, normalize_encoded
        from jax_agent.ppo_jax import safe_logits
        from jax_agent.train_jax import (
            actions_from_decisions,
            jax_actions_to_kaggle,
            kaggle_obs_to_jax_state,
        )

        ckpt_path = _HERE / "weights" / "ckpt_000400.pkl"
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        cfg = ckpt["cfg"]
        params = ckpt["params"]
        norm_state = ckpt["normalizer_state"]

        env_params = EnvParams(
            episode_steps=cfg.env.episode_steps,
            max_planets=cfg.env.max_planets,
        )
        feat_params = FeatureParams(
            max_planets=cfg.env.max_planets,
            ship_bucket_count=cfg.env.ship_bucket_count,
            max_ships=cfg.env.max_ships,
            max_production=cfg.env.max_production,
            episode_steps=cfg.env.episode_steps,
            target_filter_enabled=cfg.env.target_filter_enabled,
            attack_top_k=cfg.env.attack_top_k,
            defense_top_k=cfg.env.defense_top_k,
        )
        norm_params = NormalizerParams.from_train_config(cfg)

        model = ActorCritic(
            self_dim=18,
            candidate_dim=26,
            global_dim=11,
            candidate_count=cfg.env.max_planets + 1,
            ship_bucket_count=cfg.env.ship_bucket_count,
            hidden_size=cfg.model.hidden_size,
            use_cnn=cfg.model.use_cnn,
        )

        @jax.jit
        def _infer(obs_state):
            raw_enc = encode_batch(obs_state, feat_params)
            enc = normalize_encoded(raw_enc, norm_state, norm_params)
            flat = flatten_sources(enc)
            zeros = jnp.zeros((flat.self_features.shape[0],), dtype=jnp.int32)
            out1 = model.apply(
                {"params": params},
                flat.self_features,
                flat.candidate_features,
                flat.global_features,
                flat.candidate_mask,
                flat.map_features,
                zeros,
            )
            fire = jax.nn.sigmoid(out1.fire_logits) > 0.5
            has_target = flat.candidate_mask.at[:, 0].set(False).any(axis=-1) & flat.source_mask
            fire = fire & has_target
            target_logits = jnp.where(flat.candidate_mask.at[:, 0].set(False), out1.target_logits, -1.0e9)
            target_logits = jnp.where(fire[:, None], target_logits, -1.0e9).at[:, 0].set(
                jnp.where(fire, target_logits[:, 0], 0.0)
            )
            target = jnp.argmax(safe_logits(target_logits), axis=-1)
            out2 = model.apply(
                {"params": params},
                flat.self_features,
                flat.candidate_features,
                flat.global_features,
                flat.candidate_mask,
                flat.map_features,
                target,
            )
            selected_amount_mask = jnp.take_along_axis(flat.amount_mask, target[:, None, None], axis=1).squeeze(1)
            amount_logits = jnp.where(selected_amount_mask, out2.amount_logits, -1.0e9)
            amount_logits = jnp.where((~selected_amount_mask.any(axis=-1))[:, None], 0.0, amount_logits)
            amount = jnp.argmax(amount_logits, axis=-1)
            acts = actions_from_decisions(
                obs_state, enc,
                fire.reshape(obs_state.planets.shape[:2]),
                target.reshape(obs_state.planets.shape[:2]),
                amount.reshape(obs_state.planets.shape[:2]),
                feat_params,
            )
            return acts

        _INFER_FN = _infer
        _ENV_PARAMS = env_params
        # 预热 JIT：用 dummy EnvState 触发编译
        import numpy as np
        P = env_params.max_planets
        F = env_params.max_fleets
        dummy_state = EnvState(
            planets=jnp.zeros((1, P, 7), dtype=jnp.float32),
            initial_planets=jnp.zeros((1, P, 7), dtype=jnp.float32),
            planet_alive=jnp.zeros((1, P), dtype=bool),
            fleets=jnp.zeros((1, F, 7), dtype=jnp.float32),
            fleet_alive=jnp.zeros((1, F), dtype=bool),
            step=jnp.zeros(1, dtype=jnp.int32),
            angular_velocity=jnp.zeros(1, dtype=jnp.float32),
            next_fleet_id=jnp.zeros(1, dtype=jnp.int32),
            done=jnp.zeros(1, dtype=bool),
            winner=jnp.full(1, -1, dtype=jnp.int32),
            overflow_fleets=jnp.zeros(1, dtype=jnp.int32),
            comet_paths=jnp.zeros((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=jnp.float32),
            comet_lengths=jnp.zeros((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=jnp.int32),
            comet_ships=jnp.zeros((1, MAX_COMET_SPAWNS), dtype=jnp.float32),
            comet_valid=jnp.zeros((1, MAX_COMET_SPAWNS), dtype=bool),
            comet_path_index=jnp.full((1, MAX_COMET_SPAWNS), -2, dtype=jnp.int32),
            comet_slots=jnp.full((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN), -1, dtype=jnp.int32),
        )
        _infer(dummy_state)  # 触发 JIT 编译
    except Exception as e:  # noqa: BLE001
        _LOAD_ERROR = e


def _ensure_loaded(config: Any = None) -> None:
    global _LOAD_THREAD
    _set_agent_dir(config)
    with _LOAD_LOCK:
        if _INFER_FN is None and _LOAD_ERROR is None and _LOAD_THREAD is None:
            _LOAD_THREAD = threading.Thread(target=_load_model, daemon=True)
            _LOAD_THREAD.start()
    if _LOAD_THREAD is not None:
        _LOAD_THREAD.join()


def agent(observation, configuration):
    config = configuration
    _ensure_loaded(config)

    if _LOAD_ERROR is not None:
        return []
    if _INFER_FN is None:
        return []

    try:
        import sys
        if str(_RL_ROOT) not in sys.path:
            sys.path.insert(0, str(_RL_ROOT))

        import jax
        import jax.numpy as jnp
        from jax_agent.train_jax import (
            flip_state,
            kaggle_obs_to_jax_state,
            jax_actions_to_kaggle,
        )
        from jax_agent.fast_env import EnvParams

        player = observation.player
        jax_state = kaggle_obs_to_jax_state(observation, _ENV_PARAMS)
        if player == 1:
            jax_state = flip_state(jax_state)
        acts_jax = _INFER_FN(jax_state)
        return jax_actions_to_kaggle(acts_jax)
    except Exception:  # noqa: BLE001
        return []
