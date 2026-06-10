"""Kaggle-style runtime for converted Torch autoregressive checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .conversion import load_converted_checkpoint
from .numpy_inference import (
    NormalizerStateNP,
    actions_to_kaggle_np,
    aim_with_prediction_np,
    dynamic_ship_buckets_np,
    encode_batch_np,
    flip_actions_np,
    flip_state_np,
    normalize_encoded_np,
    obs_to_np_state,
)


def _read_field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _safe_logits_np(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, -1.0e9).astype(np.float32, copy=False)


def _auto_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, device=device)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


@dataclass(frozen=True)
class EnvParamsTorchRuntime:
    episode_steps: int
    max_planets: int

    @property
    def max_fleets(self) -> int:
        return 512


@dataclass(frozen=True)
class FeatureParamsTorchRuntime:
    max_planets: int
    ship_bucket_count: int
    max_ships: float
    max_production: float
    episode_steps: int
    target_filter_enabled: bool
    attack_top_k: int
    defense_top_k: int
    max_fleet_tokens: int
    fleet_selection: str


@dataclass(frozen=True)
class NormalizerParamsTorchRuntime:
    enabled: bool
    clip: float
    eps: float


@dataclass
class _CheckpointRuntime:
    name: str
    model: torch.nn.Module
    metadata: dict[str, Any]
    norm_state: NormalizerStateNP
    norm_params: NormalizerParamsTorchRuntime


class TorchAutoregRuntime:
    """Deterministic action runtime for one or more converted checkpoints."""

    def __init__(self, checkpoint_paths: Iterable[str | Path], *, device: str | torch.device | None = "auto") -> None:
        self.device = _auto_device(device)
        self.checkpoints: list[_CheckpointRuntime] = []
        for path_value in checkpoint_paths:
            path = Path(path_value)
            model, payload = load_converted_checkpoint(path, map_location=self.device)
            model.to(self.device)
            model.eval()
            metadata = payload["metadata"]
            self.checkpoints.append(
                _CheckpointRuntime(
                    name=path.stem,
                    model=model,
                    metadata=metadata,
                    norm_state=_normalizer_from_payload(payload["normalizer"]),
                    norm_params=NormalizerParamsTorchRuntime(**metadata["state_norm"]),
                )
            )
        if not self.checkpoints:
            raise ValueError("TorchAutoregRuntime requires at least one checkpoint")

        meta0 = self.checkpoints[0].metadata
        env_meta = meta0["env"]
        model_meta = meta0["model"]
        feature_meta = meta0.get("feature", {})
        self.max_planets = int(env_meta["max_planets"])
        self.max_action_tokens = int(model_meta["max_action_tokens"])
        self.ship_bucket_count = int(env_meta["ship_bucket_count"])
        self.env_params = EnvParamsTorchRuntime(
            episode_steps=int(env_meta["episode_steps"]),
            max_planets=self.max_planets,
        )
        self.feat_params = FeatureParamsTorchRuntime(
            max_planets=self.max_planets,
            ship_bucket_count=self.ship_bucket_count,
            max_ships=float(env_meta["max_ships"]),
            max_production=float(env_meta["max_production"]),
            episode_steps=int(env_meta["episode_steps"]),
            target_filter_enabled=bool(env_meta["target_filter_enabled"]),
            attack_top_k=int(env_meta["attack_top_k"]),
            defense_top_k=int(env_meta["defense_top_k"]),
            max_fleet_tokens=int(model_meta["max_fleet_tokens"]),
            fleet_selection=str(feature_meta.get("fleet_selection", model_meta.get("fleet_selection", "size"))),
        )

    def act(self, observation: Any) -> list[list[float | int]]:
        player = int(_read_field(observation, "player", 0) or 0)
        state = obs_to_np_state(observation, self.max_planets, self.env_params.max_fleets)
        if player == 1:
            state = flip_state_np(state)

        raw_encoded = encode_batch_np(state, self.feat_params)
        states = []
        encoded_for_actions = None
        for ckpt in self.checkpoints:
            encoded = normalize_encoded_np(raw_encoded, ckpt.norm_state, ckpt.norm_params)
            if encoded_for_actions is None:
                encoded_for_actions = encoded
            initial_ships_np = np.concatenate(
                [
                    np.zeros((1, 1), dtype=np.float32),
                    np.asarray(encoded.source_ships, dtype=np.float32),
                ],
                axis=1,
            )
            with torch.inference_mode():
                start = ckpt.model.auto_start(
                    _to_tensor(np.asarray(encoded.planet_features, dtype=np.float32), self.device),
                    _to_tensor(np.asarray(encoded.relative_features, dtype=np.float32), self.device),
                    _to_tensor(np.asarray(encoded.global_features, dtype=np.float32), self.device),
                    _to_tensor(np.asarray(encoded.candidate_mask, dtype=np.bool_), self.device),
                    _to_tensor(initial_ships_np, self.device),
                    map_features=_to_tensor(np.asarray(encoded.map_features, dtype=np.float32), self.device),
                    fleet_features=_to_tensor(np.asarray(encoded.fleet_features, dtype=np.float32), self.device),
                    fleet_mask=_to_tensor(np.asarray(encoded.fleet_mask, dtype=np.bool_), self.device),
                )
            states.append(
                {
                    "ckpt": ckpt,
                    "encoded": encoded,
                    "state_emb": start[0],
                    "h": start[1],
                    "prev_tok": start[2],
                    "remaining": start[3].clone(),
                    "planet_out": start[4],
                    "relative_h": start[5],
                    "key_proj": start[6],
                    "src_key_proj": start[7],
                }
            )

        assert encoded_for_actions is not None
        actions = np.zeros((1, self.max_planets, 3), dtype=np.float32)
        ended = np.zeros((1,), dtype=np.bool_)
        batch_idx = np.arange(1)
        source_ids_np = np.asarray(encoded_for_actions.source_ids)
        source_mask_np = np.asarray(encoded_for_actions.source_mask, dtype=np.bool_)

        for step_i in range(self.max_action_tokens):
            h_news = []
            end_logits_list = []
            src_logits_list = []
            with torch.inference_mode():
                for st in states:
                    encoded = st["encoded"]
                    h_new, end_logits, src_logits = st["ckpt"].model.auto_step_base(
                        st["state_emb"],
                        st["h"],
                        st["prev_tok"],
                        st["remaining"],
                        _to_tensor(np.asarray(encoded.relative_features, dtype=np.float32), self.device),
                        _to_tensor(np.asarray(encoded.source_mask, dtype=np.bool_), self.device),
                        st["src_key_proj"],
                    )
                    st["h_new"] = h_new
                    h_news.append(h_new)
                    end_logits_list.append(end_logits)
                    src_logits_list.append(src_logits)

            del h_news
            tv = ~ended
            end_logits_np = np.mean([_to_numpy(x) for x in end_logits_list], axis=0)
            src_logits_np = np.mean([_to_numpy(x) for x in src_logits_list], axis=0)
            end_t = tv & (_sigmoid_np(end_logits_np) > 0.5)
            src_t = np.argmax(_safe_logits_np(src_logits_np), axis=-1).astype(np.int32)
            src_for_heads = np.where(tv & (~end_t), src_t, 0).astype(np.int32)

            tgt_logits_list = []
            with torch.inference_mode():
                for st in states:
                    encoded = st["encoded"]
                    tgt_logits = st["ckpt"].model.auto_tgt_logits(
                        st["h_new"],
                        _to_tensor(src_for_heads, self.device),
                        st["planet_out"],
                        st["relative_h"],
                        st["key_proj"],
                        _to_tensor(np.asarray(encoded.candidate_mask, dtype=np.bool_), self.device),
                    )
                    tgt_logits_list.append(tgt_logits)
            tgt_logits_np = np.mean([_to_numpy(x) for x in tgt_logits_list], axis=0)
            tgt_t = np.argmax(_safe_logits_np(tgt_logits_np), axis=-1).astype(np.int32)
            tgt_for_heads = np.where(tv & (~end_t), tgt_t, 0).astype(np.int32)

            amt_logits_list = []
            with torch.inference_mode():
                for st in states:
                    encoded = st["encoded"]
                    amount_logits = st["ckpt"].model.auto_amt_logits(
                        st["h_new"],
                        _to_tensor(src_for_heads, self.device),
                        _to_tensor(tgt_for_heads, self.device),
                        st["planet_out"],
                        st["relative_h"],
                        st["remaining"],
                        _to_tensor(np.asarray(encoded.amount_mask, dtype=np.bool_), self.device),
                    )
                    amt_logits_list.append(amount_logits)
            amt_logits_np = np.mean([_to_numpy(x) for x in amt_logits_list], axis=0)
            amt_t = np.argmax(_safe_logits_np(amt_logits_np), axis=-1).astype(np.int32)

            src_slot = np.clip(src_t - 1, 0, self.max_planets - 1).astype(np.int32)
            tgt_slot = np.clip(tgt_t - 1, 0, self.max_planets - 1).astype(np.int32)
            remaining_ref = _to_numpy(states[0]["remaining"])
            source_buckets = dynamic_ship_buckets_np(
                remaining_ref[batch_idx, np.clip(src_t, 0, self.max_planets)],
                self.ship_bucket_count,
            )
            ships = np.take_along_axis(source_buckets, amt_t[:, None], axis=-1).squeeze(-1)

            src_planets = np.asarray(state.planets)[batch_idx, src_slot]
            tgt_planets = np.asarray(state.planets)[batch_idx, tgt_slot]
            angle_np, aim_hit_np = aim_with_prediction_np(
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

            with torch.inference_mode():
                src_for_heads_t = _to_tensor(src_for_heads, self.device)
                tgt_for_heads_t = _to_tensor(tgt_for_heads, self.device)
                amt_for_tok_t = _to_tensor(np.where(tv & (~end_t), amt_t, 0).astype(np.int32), self.device)
                end_t_t = _to_tensor(end_t.astype(np.bool_), self.device)
                exec_t_t = _to_tensor(exec_t.astype(np.bool_), self.device)
                for st in states:
                    if bool(exec_t[0]):
                        st["remaining"][0, int(np.clip(src_t[0], 0, self.max_planets))] -= float(ships[0])
                    st["prev_tok"] = st["ckpt"].model.auto_token_embed(
                        src_for_heads_t,
                        tgt_for_heads_t,
                        amt_for_tok_t,
                        end_t_t,
                        exec_t_t,
                        step_i,
                        st["planet_out"],
                    )
                    st["h"] = st["h_new"]
            ended = ended | end_t
            if bool(np.all(ended)):
                break

        actions_out = actions
        if player == 1:
            actions_out = flip_actions_np(actions)
        return actions_to_kaggle_np(actions_out)


def _normalizer_from_payload(normalizer: dict[str, torch.Tensor]) -> NormalizerStateNP:
    values = {name: _to_numpy(tensor).astype(np.float32, copy=False) for name, tensor in normalizer.items()}
    return NormalizerStateNP(
        planet_mean=values["planet_mean"],
        planet_var=values["planet_var"],
        planet_count=values["planet_count"],
        relative_mean=values["relative_mean"],
        relative_var=values["relative_var"],
        relative_count=values["relative_count"],
        global_mean=values["global_mean"],
        global_var=values["global_var"],
        global_count=values["global_count"],
    )
