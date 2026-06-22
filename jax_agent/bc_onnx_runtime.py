"""ONNX Runtime bridge for feature-row BC policies.

This uses the current BC feature builder for `fight` rows, then runs the policy
network through ONNX Runtime. The reusable NumPy inference helpers copied from
older ONNX agents are used for aiming/action formatting, while the BC-specific
fight features remain in JAX for now.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from .bc_policy_runtime import (
    bc_rows_to_player0_view,
    build_bc_policy_rows,
    build_env_params_from_raw,
    canonicalize_state_owner_for_player,
    env_config_from_raw,
    target_from_player0_view,
)
from .fast_env import state_from_observation
from .features_jax import FeatureParams
from .numpy_inference import actions_to_kaggle_np, aim_with_prediction_np


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _safe_logits_np(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, logits, -1.0e9).astype(np.float32, copy=False)
    empty = ~mask.any(axis=-1, keepdims=True)
    return np.where(empty, 0.0, masked).astype(np.float32, copy=False)


def _read_meta(model_dir: Path) -> dict[str, Any]:
    meta_path = model_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing BC ONNX meta: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _first_model_dir(root: Path) -> Path:
    direct_meta = root / "meta.json"
    if direct_meta.is_file():
        return root
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing BC ONNX manifest/meta under: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models") or manifest.get("checkpoints") or []
    if not models:
        raise ValueError(f"empty BC ONNX manifest: {manifest_path}")
    first = models[0]
    if "name" in first:
        return root / str(first["name"])
    if "path" in first:
        return root / Path(str(first["path"])).parent
    if "checkpoint" in first:
        return root / Path(str(first["checkpoint"])).stem
    raise ValueError(f"cannot resolve BC ONNX model dir from manifest row: {first!r}")


@dataclass
class BCOnnxPolicyAgent:
    session: Any
    raw_config: dict[str, Any]
    env_params: Any
    feature_params: FeatureParams
    ship_bucket_count: int
    max_planets: int

    def __post_init__(self) -> None:
        self._build_rows_jit = jax.jit(
            lambda state: build_bc_policy_rows(state, self.feature_params, self.ship_bucket_count)
        )

    @classmethod
    def from_dir(cls, root: str | Path, *, max_fleets: int = 512) -> "BCOnnxPolicyAgent":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for BC ONNX policy eval") from exc

        model_dir = _first_model_dir(Path(root))
        meta = _read_meta(model_dir)
        raw = meta.get("raw_config") or {}
        env_params = build_env_params_from_raw(raw, max_fleets=max_fleets)
        env_cfg = env_config_from_raw(raw)
        feature_params = FeatureParams.from_train_config(type("Cfg", (), {"env": env_cfg})())
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(
            str(model_dir / str(meta.get("onnx_file", "bc_policy.onnx"))),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        return cls(
            session=session,
            raw_config=raw,
            env_params=env_params,
            feature_params=feature_params,
            ship_bucket_count=int(meta.get("ship_bucket_count", 8)),
            max_planets=int(meta.get("max_planets", env_params.max_planets)),
        )

    def _run_policy(
        self,
        rows_np: dict[str, np.ndarray],
        selected_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fire_logits, target_logits, amount_logits = self.session.run(
            None,
            {
                "planet": rows_np["planet"].astype(np.float32, copy=False),
                "relative": rows_np["relative"].astype(np.float32, copy=False),
                "global_features": rows_np["global_features"].astype(np.float32, copy=False),
                "candidate_mask": rows_np["candidate_mask"].astype(np.bool_, copy=False),
                "map_features": rows_np["map_features"].astype(np.float32, copy=False),
                "selected_targets": selected_targets.astype(np.int32, copy=False),
            },
        )
        return (
            np.asarray(fire_logits, dtype=np.float32),
            np.asarray(target_logits, dtype=np.float32),
            np.asarray(amount_logits, dtype=np.float32),
        )

    def act(self, observation: Any, *, player: int | None = None) -> list[list[float | int]]:
        if player is None:
            if isinstance(observation, dict):
                player = int(observation.get("player", 0) or 0)
            else:
                player = int(getattr(observation, "player", 0) or 0)
        state = state_from_observation(observation, self.env_params)
        state = canonicalize_state_owner_for_player(state, int(player))
        rows = self._build_rows_jit(state)
        model_rows = bc_rows_to_player0_view(rows, int(player))
        rows_np = {
            "planet": np.asarray(jax.device_get(model_rows.planet)),
            "relative": np.asarray(jax.device_get(model_rows.relative)),
            "global_features": np.asarray(jax.device_get(model_rows.global_features)),
            "map_features": np.asarray(jax.device_get(model_rows.map_features)),
            "candidate_mask": np.asarray(jax.device_get(model_rows.candidate_mask)),
            "amount_mask": np.asarray(jax.device_get(model_rows.amount_mask)),
            "source_mask": np.asarray(jax.device_get(model_rows.source_mask)),
            "source_ids": np.asarray(jax.device_get(model_rows.source_ids)),
            "ship_buckets": np.asarray(jax.device_get(model_rows.ship_buckets)),
        }
        env_rows_np = {
            "candidate_mask": np.asarray(jax.device_get(rows.candidate_mask)),
            "source_mask": np.asarray(jax.device_get(rows.source_mask)),
            "source_ids": np.asarray(jax.device_get(rows.source_ids)),
            "ship_buckets": np.asarray(jax.device_get(rows.ship_buckets)),
        }

        zero_target = np.zeros((self.max_planets,), dtype=np.int32)
        fire_logits, target_logits, _ = self._run_policy(rows_np, zero_target)
        can_fire = rows_np["source_mask"] & rows_np["candidate_mask"][:, 1:].any(axis=-1)
        fire = (_sigmoid_np(fire_logits) >= 0.5) & can_fire

        target_mask = rows_np["candidate_mask"].copy()
        target_mask[:, 0] = False
        target = np.argmax(_safe_logits_np(target_logits, target_mask), axis=-1).astype(np.int32)
        target = np.where(fire, target, 0).astype(np.int32)

        _, _, amount_logits = self._run_policy(rows_np, target)
        selected_amount_mask = np.take_along_axis(
            rows_np["amount_mask"],
            target[:, None, None],
            axis=1,
        ).squeeze(1)
        amount = np.argmax(_safe_logits_np(amount_logits, selected_amount_mask), axis=-1).astype(np.int32)
        amount = np.where(fire, amount, 0).astype(np.int32)

        target_env = np.asarray(
            jax.device_get(target_from_player0_view(jnp.asarray(target), int(player), self.max_planets)),
            dtype=np.int32,
        )
        target_slot = np.clip(target_env[None] - 1, 0, self.max_planets - 1).astype(np.int32)
        source_planets = np.asarray(jax.device_get(state.planets))[None]
        target_planets = np.take_along_axis(source_planets, target_slot[..., None], axis=1)
        ships = np.take_along_axis(env_rows_np["ship_buckets"][None], amount[None, :, None], axis=2).squeeze(-1)
        target_valid = np.take_along_axis(env_rows_np["candidate_mask"][None], target_env[None, :, None], axis=2).squeeze(-1)
        valid = fire[None] & env_rows_np["source_mask"][None] & target_valid & (target_env[None] > 0) & (ships > 0.0)
        angle, _hit = aim_with_prediction_np(source_planets, target_planets, target_slot, state, ships)
        actions = np.stack(
            [
                np.where(valid, env_rows_np["source_ids"][None].astype(np.float32), 0.0),
                np.where(valid, angle, 0.0),
                np.where(valid, ships, 0.0),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        return actions_to_kaggle_np(actions)
