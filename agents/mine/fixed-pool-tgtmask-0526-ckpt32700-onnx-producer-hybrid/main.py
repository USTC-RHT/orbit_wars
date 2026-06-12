"""Orbit Wars hybrid agent.

2-player games use checkpoint 32700 from the JAX fixed-pool target-mask run.
4-player games delegate to the public producer agent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

_HERE = Path(globals().get("__file__", ".")).resolve().parent
if not sys.path or sys.path[0] != str(_HERE):
    sys.path.insert(0, str(_HERE))


def _drop_foreign_jax_agent_modules() -> None:
    local_pkg = (_HERE / "jax_agent").resolve()
    for name, module in list(sys.modules.items()):
        if name != "jax_agent" and not name.startswith("jax_agent."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            sys.modules.pop(name, None)
            continue
        try:
            Path(module_file).resolve().relative_to(local_pkg)
        except ValueError:
            sys.modules.pop(name, None)


_ERROR_LOGGED = False
_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
_PRELOAD_THREAD = None
_PRODUCER = None
_ACTIVE_2P_CHECKPOINTS = ("ckpt_032700",)


@dataclass(frozen=True)
class _EnvParams:
    episode_steps: int
    max_planets: int

    @property
    def max_fleets(self) -> int:
        return 512


@dataclass(frozen=True)
class _FeatureParams:
    max_planets: int
    ship_bucket_count: int
    max_ships: float
    max_production: float
    episode_steps: int
    target_filter_enabled: bool
    attack_top_k: int
    defense_top_k: int
    max_fleet_tokens: int


@dataclass(frozen=True)
class _NormalizerParams:
    enabled: bool
    clip: float
    eps: float


def _set_agent_dir(config: Any = None) -> None:
    global _HERE
    raw_path = config.get("__raw_path__") if isinstance(config, dict) else getattr(config, "__raw_path__", None)
    if raw_path:
        _HERE = Path(raw_path).resolve().parent
    if not sys.path or sys.path[0] != str(_HERE):
        sys.path.insert(0, str(_HERE))


def _read_field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _rows(obj: Any, key: str) -> list:
    return list(_read_field(obj, key, []) or [])


def _row_value(row: Any, idx: int, attr: str, default=None):
    try:
        return row[idx]
    except Exception:
        return getattr(row, attr, default)


def _is_four_player(observation: Any, configuration: Any = None) -> bool:
    for key in ("agentCount", "num_agents", "numAgents", "players"):
        value = _read_field(configuration, key, None)
        if value is not None:
            try:
                return int(value) >= 4
            except Exception:
                pass

    owners: set[int] = set()
    for row in _rows(observation, "planets"):
        owner = int(_row_value(row, 1, "owner", -1))
        if owner >= 0:
            owners.add(owner)
    for row in _rows(observation, "fleets"):
        owner = int(_row_value(row, 1, "owner", -1))
        if owner >= 0:
            owners.add(owner)
    return len(owners) >= 3 or any(owner >= 2 for owner in owners)


def _has_sendable_source(observation: Any) -> bool:
    player = int(_read_field(observation, "player", 0) or 0)
    for row in _rows(observation, "planets"):
        owner = int(_row_value(row, 1, "owner", -1))
        ships = float(_row_value(row, 5, "ship_count", 0.0) or 0.0)
        if owner == player and ships >= 10.0:
            return True
    return False


def _load_producer():
    global _PRODUCER
    if _PRODUCER is not None:
        return _PRODUCER
    path = _HERE / "producer_main.py"
    spec = importlib.util.spec_from_file_location("_hybrid_producer_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load producer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _PRODUCER = module
    return module


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _safe_logits(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, -1.0e9).astype(np.float32, copy=False)


def _import_onnxruntime():
    existing = sys.modules.get("onnxruntime")
    if existing is not None and hasattr(existing, "SessionOptions"):
        return existing
    for name in list(sys.modules):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            del sys.modules[name]

    abi_tag = f"cpython-{sys.version_info[0]}{sys.version_info[1]}"
    removed: list[tuple[int, str]] = []
    for entry in list(sys.path):
        if not entry:
            continue
        try:
            capi = Path(entry) / "onnxruntime" / "capi"
        except TypeError:
            continue
        has_ort_binary = any(capi.glob("onnxruntime_pybind11_state.cpython-*.so"))
        has_matching_binary = any(capi.glob(f"onnxruntime_pybind11_state.{abi_tag}-*.so"))
        if has_ort_binary and not has_matching_binary:
            while entry in sys.path:
                idx = sys.path.index(entry)
                removed.append((idx, sys.path.pop(idx)))
    try:
        import onnxruntime as ort
        if not hasattr(ort, "SessionOptions"):
            raise ImportError("imported onnxruntime without SessionOptions")
        return ort
    finally:
        for idx, entry in sorted(removed):
            sys.path.insert(min(idx, len(sys.path)), entry)


@dataclass
class _CheckpointRuntime:
    name: str
    meta: dict
    norm_state: Any
    norm_params: Any
    start: Any
    step: Any
    tgt: Any
    amt: Any
    tok: Any


class _OnnxEnsembleRuntime:
    def __init__(self, root: Path):
        ort = _import_onnxruntime()
        _drop_foreign_jax_agent_modules()

        from jax_agent.numpy_inference import (
            encode_batch_np,
            dynamic_ship_buckets_np,
            aim_with_prediction_np,
            actions_to_kaggle_np,
            flip_actions_np,
            flip_state_np,
            load_normalizer_np,
            normalize_encoded_np,
            obs_to_np_state,
        )

        manifest = json.loads((root / "models" / "manifest.json").read_text(encoding="utf-8"))
        self.checkpoints: list[_CheckpointRuntime] = []
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3

        for item in manifest["checkpoints"]:
            ckpt_dir = root / "models" / Path(item["checkpoint"]).stem
            if _ACTIVE_2P_CHECKPOINTS and ckpt_dir.name not in _ACTIVE_2P_CHECKPOINTS:
                continue
            meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
            norm_npz = np.load(ckpt_dir / "normalizer.npz")
            norm_state = load_normalizer_np(norm_npz)
            norm_params = _NormalizerParams(
                enabled=bool(meta["state_norm_enabled"]),
                clip=float(meta["state_norm_clip"]),
                eps=float(meta["state_norm_eps"]),
            )
            providers = ["CPUExecutionProvider"]
            self.checkpoints.append(
                _CheckpointRuntime(
                    name=Path(item["checkpoint"]).stem,
                    meta=meta,
                    norm_state=norm_state,
                    norm_params=norm_params,
                    start=ort.InferenceSession(str(ckpt_dir / "auto_start.onnx"), sess_options, providers=providers),
                    step=ort.InferenceSession(str(ckpt_dir / "auto_step_base.onnx"), sess_options, providers=providers),
                    tgt=ort.InferenceSession(str(ckpt_dir / "auto_tgt_logits.onnx"), sess_options, providers=providers),
                    amt=ort.InferenceSession(str(ckpt_dir / "auto_amt_logits.onnx"), sess_options, providers=providers),
                    tok=ort.InferenceSession(str(ckpt_dir / "auto_token_embed.onnx"), sess_options, providers=providers),
                )
            )

        meta0 = self.checkpoints[0].meta
        self.max_planets = int(meta0["max_planets"])
        self.max_action_tokens = int(meta0["max_action_tokens"])
        self.ship_bucket_count = int(meta0["ship_bucket_count"])
        self.env_params = _EnvParams(
            episode_steps=int(meta0["episode_steps"]),
            max_planets=self.max_planets,
        )
        self.feat_params = _FeatureParams(
            max_planets=self.max_planets,
            ship_bucket_count=self.ship_bucket_count,
            max_ships=float(meta0["max_ships"]),
            max_production=float(meta0["max_production"]),
            episode_steps=int(meta0["episode_steps"]),
            target_filter_enabled=bool(meta0["target_filter_enabled"]),
            attack_top_k=int(meta0["attack_top_k"]),
            defense_top_k=int(meta0["defense_top_k"]),
            max_fleet_tokens=int(meta0["max_fleet_tokens"]),
        )
        # numpy + ONNX Runtime: no JAX dependency on the match path.
        self._encode_batch = encode_batch_np
        self._normalize_encoded = normalize_encoded_np
        self._obs_to_np_state = obs_to_np_state
        self._flip_state = flip_state_np
        self._flip_actions = flip_actions_np
        self._actions_to_kaggle = actions_to_kaggle_np
        self._dynamic_ship_buckets_np = dynamic_ship_buckets_np
        self._aim_with_prediction_np = aim_with_prediction_np

    def _encoded_inputs(self, encoded, initial_ships: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "planet_features": np.asarray(encoded.planet_features, dtype=np.float32),
            "relative_features": np.asarray(encoded.relative_features, dtype=np.float32),
            "global_features": np.asarray(encoded.global_features, dtype=np.float32),
            "candidate_mask": np.asarray(encoded.candidate_mask, dtype=np.bool_),
            "initial_ships": initial_ships.astype(np.float32, copy=False),
            "map_features": np.asarray(encoded.map_features, dtype=np.float32),
            "fleet_features": np.asarray(encoded.fleet_features, dtype=np.float32),
            "fleet_mask": np.asarray(encoded.fleet_mask, dtype=np.bool_),
        }

    def act(self, observation: Any) -> list[list[float | int]]:
        player = int(_read_field(observation, "player", 0) or 0)
        state = self._obs_to_np_state(observation, self.max_planets, self.env_params.max_fleets)
        if player == 1:
            state = self._flip_state(state)

        raw_encoded = self._encode_batch(state, self.feat_params)

        states = []
        encoded_for_actions = None
        for ckpt in self.checkpoints:
            encoded = self._normalize_encoded(raw_encoded, ckpt.norm_state, ckpt.norm_params)
            if encoded_for_actions is None:
                encoded_for_actions = encoded
            initial_ships = np.concatenate(
                [
                    np.zeros((1, 1), dtype=np.float32),
                    np.asarray(encoded.source_ships, dtype=np.float32),
                ],
                axis=1,
            )
            out = ckpt.start.run(
                None,
                self._encoded_inputs(encoded, initial_ships),
            )
            states.append(
                {
                    "ckpt": ckpt,
                    "encoded": encoded,
                    "state_emb": out[0],
                    "h": out[1],
                    "prev_tok": out[2],
                    "remaining": out[3],
                    "planet_out": out[4],
                    "relative_h": out[5],
                    "key_proj": out[6],
                    "src_key_proj": out[7],
                }
            )

        assert encoded_for_actions is not None
        actions = np.zeros((1, self.max_planets, 3), dtype=np.float32)
        ended = np.zeros((1,), dtype=np.bool_)
        batch_idx = np.arange(1)
        source_ids_np = np.asarray(encoded_for_actions.source_ids)
        source_mask_np = np.asarray(encoded_for_actions.source_mask, dtype=np.bool_)

        for step_i in range(self.max_action_tokens):
            step_outputs = []
            for st in states:
                out = st["ckpt"].step.run(
                    None,
                    {
                        "state_emb": st["state_emb"],
                        "h": st["h"],
                        "prev_tok": st["prev_tok"],
                        "remaining": st["remaining"],
                        "relative_features": np.asarray(st["encoded"].relative_features, dtype=np.float32),
                        "source_mask": np.asarray(st["encoded"].source_mask, dtype=np.bool_),
                        "src_key_proj": st["src_key_proj"],
                    },
                )
                step_outputs.append(out)
                st["h_new"] = out[0]

            tv = ~ended
            end_logits = np.mean([out[1] for out in step_outputs], axis=0)
            src_logits = np.mean([out[2] for out in step_outputs], axis=0)
            end_t = tv & (_sigmoid(end_logits) > 0.5)
            src_t = np.argmax(_safe_logits(src_logits), axis=-1).astype(np.int32)
            src_for_heads = np.where(tv & (~end_t), src_t, 0).astype(np.int32)

            tgt_logits_list = []
            for st in states:
                tgt_logits_list.append(
                    st["ckpt"].tgt.run(
                        None,
                        {
                            "h_new": st["h_new"],
                            "src_t": src_for_heads,
                            "planet_out": st["planet_out"],
                            "relative_h": st["relative_h"],
                            "key_proj": st["key_proj"],
                            "candidate_mask": np.asarray(st["encoded"].candidate_mask, dtype=np.bool_),
                        },
                    )[0]
                )
            tgt_logits = np.mean(tgt_logits_list, axis=0)
            tgt_t = np.argmax(_safe_logits(tgt_logits), axis=-1).astype(np.int32)
            tgt_for_heads = np.where(tv & (~end_t), tgt_t, 0).astype(np.int32)

            amt_logits_list = []
            for st in states:
                amt_logits_list.append(
                    st["ckpt"].amt.run(
                        None,
                        {
                            "h_new": st["h_new"],
                            "src_t": src_for_heads,
                            "tgt_t": tgt_for_heads,
                            "planet_out": st["planet_out"],
                            "relative_h": st["relative_h"],
                            "remaining": st["remaining"],
                            "amount_mask": np.asarray(st["encoded"].amount_mask, dtype=np.float32),
                        },
                    )[0]
                )
            amt_logits = np.mean(amt_logits_list, axis=0)
            amt_t = np.argmax(_safe_logits(amt_logits), axis=-1).astype(np.int32)

            src_slot = np.clip(src_t - 1, 0, self.max_planets - 1).astype(np.int32)
            tgt_slot = np.clip(tgt_t - 1, 0, self.max_planets - 1).astype(np.int32)
            remaining_ref = states[0]["remaining"]
            source_buckets = self._dynamic_ship_buckets_np(
                remaining_ref[batch_idx, np.clip(src_t, 0, self.max_planets)],
                self.ship_bucket_count,
            )
            ships = np.take_along_axis(source_buckets, amt_t[:, None], axis=-1).squeeze(-1)

            src_planets = np.asarray(state.planets)[batch_idx, src_slot]
            tgt_planets = np.asarray(state.planets)[batch_idx, tgt_slot]
            angle_np, aim_hit_np = self._aim_with_prediction_np(
                src_planets[:, None, :],
                tgt_planets[:, None, :],
                tgt_slot[:, None],
                state,
                ships[:, None].astype(np.float32),
            )
            src_valid = (src_t > 0) & source_mask_np[batch_idx, src_slot]
            tgt_valid = tgt_t > 0
            exec_t = tv & (~end_t) & src_valid & tgt_valid & aim_hit_np[:, 0]

            if bool(exec_t[0]):
                actions[0, step_i, 0] = float(source_ids_np[0, src_slot[0]])
                actions[0, step_i, 1] = float(angle_np[0, 0])
                actions[0, step_i, 2] = float(ships[0])

            for st in states:
                if bool(exec_t[0]):
                    st["remaining"][0, int(np.clip(src_t[0], 0, self.max_planets))] -= float(ships[0])
                st["prev_tok"] = st["ckpt"].tok.run(
                    None,
                    {
                        "src_t": src_for_heads,
                        "tgt_t": tgt_for_heads,
                        "amt_t": np.where(tv & (~end_t), amt_t, 0).astype(np.int32),
                        "end_t": end_t.astype(np.bool_),
                        "executed_t": exec_t.astype(np.bool_),
                        "step_i": np.asarray(step_i, dtype=np.int32),
                        "planet_out": st["planet_out"],
                    },
                )[0]
                st["h"] = st["h_new"]
            ended = ended | end_t
            if bool(np.all(ended)):
                break

        actions_out = actions
        if player == 1:
            actions_out = self._flip_actions(actions)
        return self._actions_to_kaggle(actions_out)


def _load_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = _OnnxEnsembleRuntime(_HERE)
    return _RUNTIME


def _start_preload() -> None:
    global _PRELOAD_THREAD
    if _RUNTIME is not None:
        return
    if _PRELOAD_THREAD is not None and _PRELOAD_THREAD.is_alive():
        return
    _PRELOAD_THREAD = threading.Thread(target=_load_runtime, daemon=True)
    _PRELOAD_THREAD.start()


def agent(observation: Any, configuration: Any = None) -> list[list[float | int]]:
    global _ERROR_LOGGED
    try:
        _set_agent_dir(configuration)
        if _is_four_player(observation, configuration):
            return _load_producer().agent(observation)
        _start_preload()
        if not _has_sendable_source(observation):
            return []
        return _load_runtime().act(observation)
    except Exception:
        if not _ERROR_LOGGED:
            traceback.print_exc(file=sys.stderr)
            _ERROR_LOGGED = True
        raise


__all__ = ["agent"]
