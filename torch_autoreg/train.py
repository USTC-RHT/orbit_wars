"""Fine-tune a converted Torch autoregressive checkpoint."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.distributions import Bernoulli, Categorical
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch_autoreg.conversion import load_converted_checkpoint, normalizer_to_torch_dict  # noqa: E402
from torch_autoreg.numpy_inference import (  # noqa: E402
    actions_to_kaggle_np,
    aim_with_prediction_np,
    dynamic_ship_buckets_np,
    encode_batch_np,
    flip_actions_np,
    flip_state_np,
    normalize_encoded_np,
    obs_to_np_state,
    update_and_normalize_encoded_np,
)
from torch_autoreg.ppo import ppo_update, safe_logits, stack_records  # noqa: E402
from torch_autoreg.runtime import (  # noqa: E402
    EnvParamsTorchRuntime,
    FeatureParamsTorchRuntime,
    NormalizerParamsTorchRuntime,
    TorchAutoregRuntime,
    _normalizer_from_payload,
)


def _read_field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_lr_schedule(args: argparse.Namespace) -> str:
    schedule = str(getattr(args, "lr_schedule", "auto") or "auto").lower()
    if schedule == "auto":
        return "linear" if bool(getattr(args, "anneal_lr", False)) else "constant"
    return schedule


def _schedule_value(update: int, total_updates: int, schedule: str, min_ratio: float) -> float:
    name = (schedule or "constant").lower()
    if name == "constant" or total_updates <= 1:
        return 1.0
    progress = min(max((update - 1) / max(total_updates - 1, 1), 0.0), 1.0)
    floor = max(float(min_ratio), 0.0)
    if name in {"linear", "linear_decay"}:
        return floor + (1.0 - floor) * (1.0 - progress)
    if name in {"cosine", "cosine_decay"}:
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))
    return 1.0


def _entropy_scale(args: argparse.Namespace, update: int) -> float:
    return _schedule_value(
        update,
        int(getattr(args, "updates", 1)),
        str(getattr(args, "ent_schedule", "constant") or "constant"),
        float(getattr(args, "ent_min_ratio", 1.0)),
    )


def _current_lr(args: argparse.Namespace, optimizer_step: int, total_steps: int, warmup_steps: int) -> float:
    schedule = _resolve_lr_schedule(args)
    base_lr = float(args.lr)
    min_ratio = float(getattr(args, "min_lr_ratio", 0.0))
    if schedule == "constant" or total_steps <= 0:
        return base_lr
    if schedule == "linear":
        progress = min(max(optimizer_step / max(total_steps, 1), 0.0), 1.0)
        return base_lr * (min_ratio + (1.0 - min_ratio) * (1.0 - progress))
    if schedule in {"warmup_cosine", "cosine_warmup"}:
        if warmup_steps > 0 and optimizer_step < warmup_steps:
            return base_lr * optimizer_step / max(warmup_steps, 1)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((optimizer_step - warmup_steps) / decay_steps, 0.0), 1.0)
        return base_lr * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)))
    return base_lr


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    defaults = _flatten_yaml_config(data)
    base_dir = path.resolve().parent
    for key in ("checkpoint", "out_dir", "producer_dir"):
        if key in defaults:
            value = Path(defaults[key])
            if not value.is_absolute():
                defaults[key] = base_dir / value
    return defaults


def _normalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def _flatten_yaml_config(data: dict[str, Any]) -> dict[str, Any]:
    top = _normalize_keys(data)
    out: dict[str, Any] = {}
    direct_keys = {
        "checkpoint",
        "out_dir",
        "producer_dir",
        "updates",
        "num_envs",
        "rollout_steps",
        "seed",
        "log_every",
        "learner_player",
        "device",
        "lr",
        "lr_schedule",
        "warmup_updates",
        "min_lr_ratio",
        "anneal_lr",
        "gamma",
        "gae_lambda",
        "clip_coef",
        "ent_coef",
        "ent_coef_fire",
        "ent_coef_target",
        "ent_coef_amount",
        "ent_schedule",
        "ent_min_ratio",
        "vf_coef",
        "epochs",
        "minibatch_size",
        "max_grad_norm",
        "norm_adv",
        "target_kl",
        "checkpoint_every",
        "eval_every",
        "eval_games",
        "eval_player",
        "eval_seed",
        "freeze_normalizer",
        "torch_threads",
    }
    for key in direct_keys:
        if key in top:
            out[key] = top[key]
    if "save_dir" in top and "out_dir" not in out:
        out["out_dir"] = top["save_dir"]

    train = top.get("train")
    if isinstance(train, dict):
        train = _normalize_keys(train)
        for src, dst in {
            "checkpoint": "checkpoint",
            "out_dir": "out_dir",
            "save_dir": "out_dir",
            "total_updates": "updates",
            "updates": "updates",
            "checkpoint_every": "checkpoint_every",
            "log_every": "log_every",
            "freeze_normalizer": "freeze_normalizer",
        }.items():
            if src in train:
                out[dst] = train[src]

    rollout = top.get("rollout")
    if isinstance(rollout, dict):
        rollout = _normalize_keys(rollout)
        for key in ["num_envs", "rollout_steps", "learner_player"]:
            if key in rollout:
                out[key] = rollout[key]

    opponent = top.get("opponent")
    if isinstance(opponent, dict):
        opponent = _normalize_keys(opponent)
        if "agent_dir" in opponent:
            out["producer_dir"] = opponent["agent_dir"]
        if "producer_dir" in opponent:
            out["producer_dir"] = opponent["producer_dir"]
        if "learner_player" in opponent:
            out["learner_player"] = opponent["learner_player"]

    ppo = top.get("ppo")
    if isinstance(ppo, dict):
        ppo = _normalize_keys(ppo)
        if "total_updates" in ppo:
            out["updates"] = ppo["total_updates"]
        elif "updates" in ppo:
            out["updates"] = ppo["updates"]
        for key in [
            "lr",
            "lr_schedule",
            "warmup_updates",
            "min_lr_ratio",
            "anneal_lr",
            "gamma",
            "gae_lambda",
            "clip_coef",
            "ent_coef",
            "ent_coef_fire",
            "ent_coef_target",
            "ent_coef_amount",
            "ent_schedule",
            "ent_min_ratio",
            "vf_coef",
            "epochs",
            "minibatch_size",
            "max_grad_norm",
            "norm_adv",
            "target_kl",
        ]:
            if key in ppo:
                out[key] = ppo[key]

    eval_cfg = top.get("eval")
    if isinstance(eval_cfg, dict):
        eval_cfg = _normalize_keys(eval_cfg)
        enabled = bool(eval_cfg.get("enabled", True))
        if "every" in eval_cfg:
            out["eval_every"] = int(eval_cfg["every"]) if enabled else 0
        elif not enabled:
            out["eval_every"] = 0
        if "games" in eval_cfg:
            out["eval_games"] = eval_cfg["games"]
        if "player" in eval_cfg:
            out["eval_player"] = eval_cfg["player"]
        if "seed" in eval_cfg:
            out["eval_seed"] = eval_cfg["seed"]
    return out


def _path_arg(value: Any | None) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


class ExternalScriptOpponent:
    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir.resolve()
        self.main_py = self.agent_dir / "main.py"
        if not self.main_py.exists():
            raise FileNotFoundError(f"external opponent entry not found: {self.main_py}")
        self.modules: dict[int, Any] = {}
        self.call_with_config_by_env: dict[int, bool] = {}
        self.load_counter = 0

    def reset(self, seed: int) -> None:
        self.reset_env(-1, seed)

    def reset_env(self, env_index: int, seed: int) -> None:
        self.load_counter += 1
        module_name = f"_torch_train_producer_{id(self)}_{env_index}_{self.load_counter}"
        spec = importlib.util.spec_from_file_location(module_name, self.main_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load external opponent: {self.main_py}")
        module = importlib.util.module_from_spec(spec)
        old_path = list(sys.path)
        try:
            sys.modules[module_name] = module
            sys.path.insert(0, str(self.agent_dir))
            spec.loader.exec_module(module)
        finally:
            sys.path = old_path
        agent = getattr(module, "agent", None)
        if agent is None or not callable(agent):
            raise AttributeError(f"external opponent has no callable agent(obs): {self.main_py}")
        random.seed(int(seed))
        try:
            call_with_config = len(inspect.signature(agent).parameters) >= 2
        except (TypeError, ValueError):
            call_with_config = False
        self.modules[int(env_index)] = module
        self.call_with_config_by_env[int(env_index)] = call_with_config

    def act(self, observation: Any) -> list[list[float | int]]:
        return self.act_for_env(-1, observation)

    def act_for_env(self, env_index: int, observation: Any) -> list[list[float | int]]:
        env_index = int(env_index)
        if env_index not in self.modules:
            self.reset_env(env_index, 0)
        module = self.modules[env_index]
        if self.call_with_config_by_env.get(env_index, False):
            moves = module.agent(observation, {"__raw_path__": str(self.main_py)})
        else:
            moves = module.agent(observation)
        return list(moves or [])


class TrainRuntime:
    def __init__(self, model: torch.nn.Module, payload: dict[str, Any], *, device: torch.device, update_normalizer: bool = True) -> None:
        self.model = model
        self.payload = payload
        self.device = device
        self.update_normalizer = bool(update_normalizer)
        meta = payload["metadata"]
        env_meta = meta["env"]
        model_meta = meta["model"]
        feature_meta = meta.get("feature", {})
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
        self.norm_state = _normalizer_from_payload(payload["normalizer"])
        self.norm_params = NormalizerParamsTorchRuntime(**meta["state_norm"])

    def _encode_observation(self, observation: Any, *, update_normalizer: bool) -> tuple[Any, Any, int]:
        player = int(_read_field(observation, "player", 0) or 0)
        state = obs_to_np_state(observation, self.max_planets, self.env_params.max_fleets)
        if player == 1:
            state = flip_state_np(state)
        raw_encoded = encode_batch_np(state, self.feat_params)
        if update_normalizer:
            encoded, self.norm_state = update_and_normalize_encoded_np(raw_encoded, self.norm_state, self.norm_params)
        else:
            encoded = normalize_encoded_np(raw_encoded, self.norm_state, self.norm_params)
        return encoded, state, player

    def estimate_value(self, observation: Any) -> torch.Tensor:
        encoded, _state, _player = self._encode_observation(observation, update_normalizer=False)
        initial_ships_np = np.concatenate(
            [
                np.zeros((1, 1), dtype=np.float32),
                np.asarray(encoded.source_ships, dtype=np.float32),
            ],
            axis=1,
        )
        with torch.inference_mode():
            value = self.model.auto_start(
                torch.as_tensor(np.asarray(encoded.planet_features, dtype=np.float32), device=self.device),
                torch.as_tensor(np.asarray(encoded.relative_features, dtype=np.float32), device=self.device),
                torch.as_tensor(np.asarray(encoded.global_features, dtype=np.float32), device=self.device),
                torch.as_tensor(np.asarray(encoded.candidate_mask, dtype=np.bool_), device=self.device),
                torch.as_tensor(initial_ships_np, device=self.device),
                map_features=torch.as_tensor(np.asarray(encoded.map_features, dtype=np.float32), device=self.device),
                fleet_features=torch.as_tensor(np.asarray(encoded.fleet_features, dtype=np.float32), device=self.device),
                fleet_mask=torch.as_tensor(np.asarray(encoded.fleet_mask, dtype=np.bool_), device=self.device),
            )[-1]
        return value.squeeze(0).detach().cpu()

    def sample_action(self, observation: Any) -> tuple[list[list[float | int]], dict[str, Any]]:
        encoded, state, player = self._encode_observation(observation, update_normalizer=self.update_normalizer)
        initial_ships_np = np.concatenate(
            [
                np.zeros((1, 1), dtype=np.float32),
                np.asarray(encoded.source_ships, dtype=np.float32),
            ],
            axis=1,
        )
        planet_features = torch.as_tensor(np.asarray(encoded.planet_features, dtype=np.float32), device=self.device)
        relative_features = torch.as_tensor(np.asarray(encoded.relative_features, dtype=np.float32), device=self.device)
        global_features = torch.as_tensor(np.asarray(encoded.global_features, dtype=np.float32), device=self.device)
        candidate_mask = torch.as_tensor(np.asarray(encoded.candidate_mask, dtype=np.bool_), device=self.device)
        amount_mask = torch.as_tensor(np.asarray(encoded.amount_mask, dtype=np.bool_), device=self.device)
        source_mask = torch.as_tensor(np.asarray(encoded.source_mask, dtype=np.bool_), device=self.device)
        map_features = torch.as_tensor(np.asarray(encoded.map_features, dtype=np.float32), device=self.device)
        fleet_features = torch.as_tensor(np.asarray(encoded.fleet_features, dtype=np.float32), device=self.device)
        fleet_mask = torch.as_tensor(np.asarray(encoded.fleet_mask, dtype=np.bool_), device=self.device)
        initial_ships = torch.as_tensor(initial_ships_np, device=self.device)

        with torch.inference_mode():
            state_emb, h, prev_tok, remaining, planet_out, relative_h, key_proj, src_key_proj, value = self.model.auto_start(
                planet_features,
                relative_features,
                global_features,
                candidate_mask,
                initial_ships,
                map_features=map_features,
                fleet_features=fleet_features,
                fleet_mask=fleet_mask,
            )

            k = self.max_action_tokens
            end_action = torch.zeros((1, k), dtype=torch.bool, device=self.device)
            src_action = torch.zeros((1, k), dtype=torch.long, device=self.device)
            tgt_action = torch.zeros((1, k), dtype=torch.long, device=self.device)
            amt_action = torch.zeros((1, k), dtype=torch.long, device=self.device)
            executed = torch.zeros((1, k), dtype=torch.bool, device=self.device)
            invalid = torch.zeros((1, k), dtype=torch.bool, device=self.device)
            token_valid = torch.zeros((1, k), dtype=torch.bool, device=self.device)
            old_logp_end = torch.zeros((1, k), dtype=torch.float32, device=self.device)
            old_logp_src = torch.zeros((1, k), dtype=torch.float32, device=self.device)
            old_logp_tgt = torch.zeros((1, k), dtype=torch.float32, device=self.device)
            old_logp_amt = torch.zeros((1, k), dtype=torch.float32, device=self.device)

            actions = np.zeros((1, self.max_planets, 3), dtype=np.float32)
            ended = torch.zeros((1,), dtype=torch.bool, device=self.device)
            batch_idx_np = np.arange(1)
            source_ids_np = np.asarray(encoded.source_ids)
            source_mask_np = np.asarray(encoded.source_mask, dtype=np.bool_)
            state_planets_np = np.asarray(state.planets)

            for step_i in range(k):
                h_new, end_logits, src_logits = self.model.auto_step_base(
                    state_emb, h, prev_tok, remaining, relative_features, source_mask, src_key_proj
                )
                tv = ~ended
                token_valid[:, step_i] = tv

                end_dist = Bernoulli(logits=end_logits)
                sampled_end = end_dist.sample().bool()
                end_t = tv & sampled_end
                end_action[:, step_i] = end_t
                old_logp_end[:, step_i] = end_dist.log_prob(end_t.float())

                src_dist = Categorical(logits=safe_logits(src_logits))
                src_t = src_dist.sample()
                logp_src_t = src_dist.log_prob(src_t)
                src_for_heads = torch.where(tv & (~end_t), src_t, torch.zeros_like(src_t))
                src_action[:, step_i] = src_for_heads

                tgt_logits = self.model.auto_tgt_logits(h_new, src_for_heads, planet_out, relative_h, key_proj, candidate_mask)
                tgt_dist = Categorical(logits=safe_logits(tgt_logits))
                tgt_t = tgt_dist.sample()
                logp_tgt_t = tgt_dist.log_prob(tgt_t)
                tgt_for_heads = torch.where(tv & (~end_t), tgt_t, torch.zeros_like(tgt_t))
                tgt_action[:, step_i] = tgt_for_heads

                amt_logits = self.model.auto_amt_logits(h_new, src_for_heads, tgt_for_heads, planet_out, relative_h, remaining, amount_mask)
                amt_dist = Categorical(logits=safe_logits(amt_logits))
                amt_t = amt_dist.sample()
                logp_amt_t = amt_dist.log_prob(amt_t)
                amt_for_tok = torch.where(tv & (~end_t), amt_t, torch.zeros_like(amt_t))
                amt_action[:, step_i] = amt_for_tok

                src_t_np = _to_numpy(src_t).astype(np.int32)
                tgt_t_np = _to_numpy(tgt_t).astype(np.int32)
                amt_t_np = _to_numpy(amt_t).astype(np.int32)
                end_t_np = _to_numpy(end_t).astype(np.bool_)
                src_slot = np.clip(src_t_np - 1, 0, self.max_planets - 1).astype(np.int32)
                tgt_slot = np.clip(tgt_t_np - 1, 0, self.max_planets - 1).astype(np.int32)
                remaining_np = _to_numpy(remaining)
                source_buckets = dynamic_ship_buckets_np(
                    remaining_np[batch_idx_np, np.clip(src_t_np, 0, self.max_planets)],
                    self.ship_bucket_count,
                )
                ships = np.take_along_axis(source_buckets, amt_t_np[:, None], axis=-1).squeeze(-1)
                src_planets = state_planets_np[batch_idx_np, src_slot]
                tgt_planets = state_planets_np[batch_idx_np, tgt_slot]
                angle_np, aim_hit_np = aim_with_prediction_np(
                    src_planets[:, None, :],
                    tgt_planets[:, None, :],
                    tgt_slot[:, None],
                    state,
                    ships[:, None].astype(np.float32),
                )
                src_valid = (src_t_np > 0) & source_mask_np[batch_idx_np, src_slot]
                tgt_valid = tgt_t_np > 0
                exec_np = (~_to_numpy(ended)) & (~end_t_np) & src_valid & tgt_valid & aim_hit_np[:, 0]
                invalid_np = (~_to_numpy(ended)) & (~end_t_np) & (~exec_np)
                exec_t = torch.as_tensor(exec_np, dtype=torch.bool, device=self.device)
                executed[:, step_i] = exec_t
                invalid[:, step_i] = torch.as_tensor(invalid_np, dtype=torch.bool, device=self.device)
                old_logp_src[:, step_i] = torch.where(exec_t, logp_src_t, torch.zeros_like(logp_src_t))
                old_logp_tgt[:, step_i] = torch.where(exec_t, logp_tgt_t, torch.zeros_like(logp_tgt_t))
                old_logp_amt[:, step_i] = torch.where(exec_t, logp_amt_t, torch.zeros_like(logp_amt_t))

                if bool(exec_np[0]):
                    actions[0, step_i, 0] = float(source_ids_np[0, src_slot[0]])
                    actions[0, step_i, 1] = float(angle_np[0, 0])
                    actions[0, step_i, 2] = float(ships[0])
                    remaining[0, int(np.clip(src_t_np[0], 0, self.max_planets))] -= float(ships[0])

                prev_tok = self.model.auto_token_embed(src_for_heads, tgt_for_heads, amt_for_tok, end_t, exec_t, step_i, planet_out)
                h = h_new
                ended = ended | end_t
                if bool(torch.all(ended).item()):
                    break

        actions_out = flip_actions_np(actions) if player == 1 else actions
        record = {
            "planet_features": planet_features.squeeze(0).detach().cpu(),
            "relative_features": relative_features.squeeze(0).detach().cpu(),
            "global_features": global_features.squeeze(0).detach().cpu(),
            "map_features": map_features.squeeze(0).detach().cpu(),
            "fleet_features": fleet_features.squeeze(0).detach().cpu(),
            "fleet_mask": fleet_mask.squeeze(0).detach().cpu(),
            "candidate_mask": candidate_mask.squeeze(0).detach().cpu(),
            "amount_mask": amount_mask.squeeze(0).detach().cpu(),
            "source_mask": source_mask.squeeze(0).detach().cpu(),
            "initial_ships": initial_ships.squeeze(0).detach().cpu(),
            "end_action": end_action.squeeze(0).detach().cpu(),
            "src_action": src_action.squeeze(0).detach().cpu(),
            "tgt_action": tgt_action.squeeze(0).detach().cpu(),
            "amt_action": amt_action.squeeze(0).detach().cpu(),
            "executed": executed.squeeze(0).detach().cpu(),
            "invalid": invalid.squeeze(0).detach().cpu(),
            "old_logp_end": old_logp_end.squeeze(0).detach().cpu(),
            "old_logp_src": old_logp_src.squeeze(0).detach().cpu(),
            "old_logp_tgt": old_logp_tgt.squeeze(0).detach().cpu(),
            "old_logp_amt": old_logp_amt.squeeze(0).detach().cpu(),
            "token_valid": token_valid.squeeze(0).detach().cpu(),
            "old_value": value.squeeze(0).detach().cpu(),
        }
        return actions_to_kaggle_np(actions_out), record


class RolloutEnv:
    def __init__(
        self,
        *,
        env_index: int,
        learner_player: int,
        opponent: ExternalScriptOpponent,
    ) -> None:
        self.env_index = int(env_index)
        self.learner_player_cfg = int(learner_player)
        self.opponent = opponent
        self.env: Any | None = None
        self.states: Any | None = None
        self.learner_idx = 0
        self.opponent_idx = 1
        self.episode_index = 0
        self.episode_steps = 0

    def reset(self, seed: int) -> None:
        from kaggle_environments import make as kag_make

        if self.learner_player_cfg < 0:
            self.learner_idx = (self.env_index + self.episode_index) % 2
        else:
            self.learner_idx = 1 if self.learner_player_cfg == 1 else 0
        self.opponent_idx = 1 - self.learner_idx
        self.opponent.reset_env(self.env_index, int(seed))
        self.env = kag_make("orbit_wars", configuration={"seed": int(seed), "randomSeed": int(seed)}, debug=False)
        self.env.reset(num_agents=2)
        self.states = self.env.step([[], []])
        self.episode_steps = 0
        self.episode_index += 1

    def step(self, runtime: TrainRuntime) -> tuple[dict[str, Any], bool, float]:
        if self.env is None or self.states is None:
            raise RuntimeError("Call reset before step.")
        learner_action, record = runtime.sample_action(self.states[self.learner_idx]["observation"])
        opponent_action = self.opponent.act_for_env(self.env_index, self.states[self.opponent_idx]["observation"])
        joint_actions: list[Any] = [None, None]
        joint_actions[self.learner_idx] = learner_action
        joint_actions[self.opponent_idx] = opponent_action
        self.states = self.env.step(joint_actions)
        self.episode_steps += 1
        done = self.states[0]["status"] != "ACTIVE"
        reward = float(self.states[self.learner_idx].get("reward", 0.0) or 0.0) if done else 0.0
        record["reward"] = torch.tensor(reward, dtype=torch.float32)
        record["done"] = torch.tensor(done, dtype=torch.bool)
        record["bootstrap_value"] = torch.tensor(0.0, dtype=torch.float32)
        record["rollout_boundary"] = torch.tensor(done, dtype=torch.bool)
        return record, done, reward

    def bootstrap_value(self, runtime: TrainRuntime) -> torch.Tensor:
        if self.states is None:
            return torch.tensor(0.0, dtype=torch.float32)
        return runtime.estimate_value(self.states[self.learner_idx]["observation"]).to(dtype=torch.float32)


def collect_rollout(
    runtime: TrainRuntime,
    envs: list[RolloutEnv],
    *,
    rollout_steps: int,
    next_seed: int,
) -> tuple[list[dict[str, Any]], int, dict[str, float]]:
    records_by_env: list[list[dict[str, Any]]] = [[] for _ in envs]
    wins = losses = draws = 0
    finished_steps: list[int] = []
    finished_rewards: list[float] = []

    for _step in range(int(rollout_steps)):
        for env_idx, env in enumerate(envs):
            record, done, reward = env.step(runtime)
            records_by_env[env_idx].append(record)
            if done:
                finished_steps.append(env.episode_steps)
                finished_rewards.append(reward)
                if reward > 0.0:
                    wins += 1
                elif reward < 0.0:
                    losses += 1
                else:
                    draws += 1
                env.reset(next_seed)
                next_seed += 1

    for env_idx, env in enumerate(envs):
        env_records = records_by_env[env_idx]
        if env_records and not bool(env_records[-1]["rollout_boundary"]):
            env_records[-1]["bootstrap_value"] = env.bootstrap_value(runtime)
            env_records[-1]["rollout_boundary"] = torch.tensor(True, dtype=torch.bool)

    records = [record for env_records in records_by_env for record in env_records]
    episodes = wins + losses + draws
    stats = {
        "envs": float(len(envs)),
        "rollout_steps": float(rollout_steps),
        "records": float(len(records)),
        "episodes": float(episodes),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": float(wins / max(episodes, 1)),
        "mean_steps": float(np.mean(finished_steps) if finished_steps else 0.0),
        "mean_reward": float(np.mean(finished_rewards) if finished_rewards else 0.0),
    }
    return records, next_seed, stats


def evaluate_deterministic(
    checkpoint: Path,
    producer_dir: Path,
    *,
    games: int,
    seed: int,
    learner_player: int,
    device: str,
) -> dict[str, float]:
    from kaggle_environments import make as kag_make

    runtime = TorchAutoregRuntime([checkpoint], device=device)
    opponent = ExternalScriptOpponent(producer_dir)
    wins = losses = draws = 0
    steps_list: list[int] = []
    for game_idx in range(int(games)):
        learner_idx = game_idx % 2 if learner_player < 0 else (1 if learner_player == 1 else 0)
        opponent_idx = 1 - learner_idx
        game_seed = int(seed) + game_idx
        opponent.reset(game_seed)
        env = kag_make("orbit_wars", configuration={"seed": game_seed, "randomSeed": game_seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        done = states[0]["status"] != "ACTIVE"
        steps = 0
        while not done:
            learner_action = runtime.act(states[learner_idx]["observation"])
            opponent_action = opponent.act(states[opponent_idx]["observation"])
            joint_actions: list[Any] = [None, None]
            joint_actions[learner_idx] = learner_action
            joint_actions[opponent_idx] = opponent_action
            states = env.step(joint_actions)
            done = states[0]["status"] != "ACTIVE"
            steps += 1
        reward = float(states[learner_idx].get("reward", 0.0) or 0.0)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
        steps_list.append(steps)
    return {
        "eval_games": float(games),
        "eval_wins": float(wins),
        "eval_losses": float(losses),
        "eval_draws": float(draws),
        "eval_win_rate": float(wins / max(games, 1)),
        "eval_mean_steps": float(np.mean(steps_list) if steps_list else 0.0),
    }


def save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any],
    update: int,
    normalizer_state: Any,
    optimizer_step: int,
) -> None:
    metadata = dict(payload["metadata"])
    metadata["update"] = int(update)
    metadata["fine_tune"] = {
        "source_checkpoint": metadata.get("source_checkpoint", ""),
        "latest_update": int(update),
        "opponent": "slawekbiel-producer",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "normalizer": normalizer_to_torch_dict(normalizer_state),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": int(update),
            "optimizer_step": int(optimizer_step),
        },
        path,
    )


def write_metrics_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "update",
                "sps",
                "episode_reward",
                "win_rate",
                "mean_return",
                "episodes_per_rollout",
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "entropy_end",
                "entropy_src",
                "entropy_tgt",
                "entropy_amt",
                "approx_kl",
                "clipfrac",
                "max_abs_log_ratio",
                "n_epochs_run",
                "lr",
                "ent_scale",
                "eval_win_rate",
                "eval_wins",
                "eval_losses",
                "eval_draws",
                "eval_mean_steps",
            ]
        )


def append_metrics(
    path: Path,
    update: int,
    sps: float,
    stats: dict[str, float],
    metrics: dict[str, float],
    eval_stats: dict[str, float] | None,
) -> None:
    eval_stats = eval_stats or {}
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                update,
                f"{sps:.6f}",
                f"{stats.get('mean_reward', 0.0):.8f}",
                f"{stats.get('win_rate', 0.0):.8f}",
                f"{stats.get('mean_reward', 0.0):.8f}",
                f"{stats.get('episodes', 0.0):.1f}",
                f"{metrics.get('loss', 0.0):.8f}",
                f"{metrics.get('policy_loss', 0.0):.8f}",
                f"{metrics.get('value_loss', 0.0):.8f}",
                f"{metrics.get('entropy', 0.0):.8f}",
                f"{metrics.get('entropy_end', 0.0):.8f}",
                f"{metrics.get('entropy_src', 0.0):.8f}",
                f"{metrics.get('entropy_tgt', 0.0):.8f}",
                f"{metrics.get('entropy_amt', 0.0):.8f}",
                f"{metrics.get('approx_kl', 0.0):.8f}",
                f"{metrics.get('clipfrac', 0.0):.8f}",
                f"{metrics.get('max_abs_log_ratio', 0.0):.8f}",
                f"{metrics.get('n_epochs_run', 0.0):.1f}",
                f"{metrics.get('lr', 0.0):.12f}",
                f"{metrics.get('ent_scale', 1.0):.8f}",
                "" if "eval_win_rate" not in eval_stats else f"{eval_stats['eval_win_rate']:.8f}",
                "" if "eval_wins" not in eval_stats else int(eval_stats["eval_wins"]),
                "" if "eval_losses" not in eval_stats else int(eval_stats["eval_losses"]),
                "" if "eval_draws" not in eval_stats else int(eval_stats["eval_draws"]),
                "" if "eval_mean_steps" not in eval_stats else f"{eval_stats['eval_mean_steps']:.8f}",
            ]
        )


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, remaining_argv = pre_parser.parse_known_args()
    config_defaults = _load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=pre_args.config, help="YAML config; CLI flags override it")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "torch_autoreg_producer_ft")
    parser.add_argument("--producer-dir", type=Path, default=REPO_ROOT / "agents" / "external" / "slawekbiel-producer")
    parser.add_argument("--updates", type=int, default=10, help="final update index, matching jax_agent total_updates")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--learner-player", type=int, choices=[-1, 0, 1], default=1, help="-1 alternates by game")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--lr-schedule", default="auto", choices=["auto", "constant", "linear", "warmup_cosine", "cosine_warmup"])
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ent-coef-fire", type=float, default=0.005)
    parser.add_argument("--ent-coef-target", type=float, default=0.01)
    parser.add_argument("--ent-coef-amount", type=float, default=0.005)
    parser.add_argument("--ent-schedule", default="constant", choices=["constant", "linear", "linear_decay", "cosine", "cosine_decay"])
    parser.add_argument("--ent-min-ratio", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--eval-player", type=int, choices=[-1, 0, 1], default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument(
        "--freeze-normalizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep checkpoint normalizer fixed during rollout collection",
    )
    parser.add_argument("--torch-threads", type=int, default=4, help="CPU torch intra-op threads; 0 keeps PyTorch default")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_argv)
    args.checkpoint = _path_arg(args.checkpoint)
    args.out_dir = _path_arg(args.out_dir)
    args.producer_dir = _path_arg(args.producer_dir)
    if args.checkpoint is None:
        parser.error("--checkpoint is required, either in YAML config or on the command line")

    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, payload = load_converted_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt_update = payload.get("update")
    if ckpt_update is None:
        fine_tune_meta = payload.get("metadata", {}).get("fine_tune", {})
        ckpt_update = fine_tune_meta.get("latest_update", payload.get("metadata", {}).get("update"))
    start_update = int(ckpt_update) + 1 if ckpt_update is not None else 1
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    ent_coef_fire = args.ent_coef if float(args.ent_coef_fire) < 0.0 else args.ent_coef_fire
    ent_coef_target = args.ent_coef if float(args.ent_coef_target) < 0.0 else args.ent_coef_target
    ent_coef_amount = args.ent_coef if float(args.ent_coef_amount) < 0.0 else args.ent_coef_amount
    rollout_state_count = int(args.num_envs) * int(args.rollout_steps)
    ppo_states_per_minibatch = min(max(int(args.minibatch_size), 1), max(rollout_state_count, 1))
    ppo_num_minibatches = max(rollout_state_count // ppo_states_per_minibatch, 1)
    optimizer_steps_per_update = int(args.epochs) * ppo_num_minibatches
    total_optimizer_steps = int(args.updates) * optimizer_steps_per_update
    warmup_steps = int(getattr(args, "warmup_updates", 0)) * optimizer_steps_per_update
    optimizer_step = int(payload.get("optimizer_step", max(start_update - 1, 0) * optimizer_steps_per_update))
    runtime = TrainRuntime(model, payload, device=device, update_normalizer=not args.freeze_normalizer)
    opponent = ExternalScriptOpponent(args.producer_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.csv"
    write_metrics_header(metrics_path)
    print(
        f"[train] checkpoint={args.checkpoint} device={device} total_updates={args.updates} "
        f"envs={args.num_envs} rollout_steps={args.rollout_steps} "
        f"lr_schedule={_resolve_lr_schedule(args)} ent_schedule={args.ent_schedule} "
        f"learner={'alternate' if args.learner_player < 0 else f'p{args.learner_player}'} "
        f"start_update={start_update}",
        flush=True,
    )

    last_update = int(args.updates)
    next_seed = int(args.seed)
    rollout_envs: list[RolloutEnv] = []
    if last_update >= start_update:
        for env_index in range(int(args.num_envs)):
            env = RolloutEnv(env_index=env_index, learner_player=args.learner_player, opponent=opponent)
            env.reset(next_seed)
            next_seed += 1
            rollout_envs.append(env)
    for update in range(start_update, last_update + 1):
        start_time = time.perf_counter()
        lr_now = _current_lr(args, optimizer_step, total_optimizer_steps, warmup_steps)
        _set_optimizer_lr(optimizer, lr_now)
        ent_scale = _entropy_scale(args, update)
        model.eval()
        records, next_seed, stats = collect_rollout(
            runtime,
            rollout_envs,
            rollout_steps=args.rollout_steps,
            next_seed=next_seed,
        )
        rollout = stack_records(records, gamma=args.gamma, gae_lambda=args.gae_lambda)
        model.train()
        metrics = ppo_update(
            model,
            optimizer,
            rollout,
            clip_coef=args.clip_coef,
            ent_coef=args.ent_coef,
            ent_coef_fire=ent_coef_fire,
            ent_coef_target=ent_coef_target,
            ent_coef_amount=ent_coef_amount,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            norm_adv=args.norm_adv,
            target_kl=args.target_kl,
            ent_scale=ent_scale,
        )
        optimizer_step += int(metrics.get("n_epochs_run", float(args.epochs))) * ppo_num_minibatches
        metrics["lr"] = lr_now
        metrics["ent_scale"] = ent_scale
        elapsed = time.perf_counter() - start_time
        samples = int(args.num_envs) * int(args.rollout_steps)
        sps = samples / max(elapsed, 1.0e-9)
        eval_stats = None
        needs_save = update % max(int(args.checkpoint_every), 1) == 0
        needs_eval = args.eval_every > 0 and update % args.eval_every == 0
        if needs_save or needs_eval:
            save_path = args.out_dir / f"ckpt_{update:06d}.pt"
            if needs_save:
                save_training_checkpoint(save_path, model, optimizer, payload, update, runtime.norm_state, optimizer_step)
            save_training_checkpoint(args.out_dir / "ckpt_last.pt", model, optimizer, payload, update, runtime.norm_state, optimizer_step)
            if needs_save:
                print(f"[ckpt] saved {save_path}", flush=True)
        if needs_eval:
            eval_seed = args.seed if args.eval_seed is None else args.eval_seed
            eval_player = args.learner_player if args.eval_player is None else args.eval_player
            eval_stats = evaluate_deterministic(
                args.out_dir / "ckpt_last.pt",
                args.producer_dir,
                games=args.eval_games,
                seed=eval_seed + update,
                learner_player=eval_player,
                device=str(device),
            )
        line = (
            f"update={update} sps={sps:.1f} loss={metrics.get('loss', 0.0):.4f} "
            f"ep_reward={stats.get('mean_reward', 0.0):.3f} winrate={stats.get('win_rate', 0.0):.3f} "
            f"return={stats.get('mean_reward', 0.0):.3f} eps={stats.get('episodes', 0.0):.0f} "
            f"ent={metrics.get('entropy', 0.0):.3f} kl={metrics.get('approx_kl', 0.0):.5f} "
            f"logr={metrics.get('max_abs_log_ratio', 0.0):.2f} "
            f"epochs={metrics.get('n_epochs_run', args.epochs):.0f} "
            f"lr={metrics.get('lr', args.lr):.2e} ent_scale={metrics.get('ent_scale', 1.0):.2f} "
            f"invalid={metrics.get('invalid_action_rate', 0.0):.3f} "
            f"exec={metrics.get('executed_action_rate', 0.0):.3f} "
            f"seq={metrics.get('seq_len_mean', 0.0):.2f} "
            f"logr_heads="
            f"{metrics.get('max_abs_log_ratio_end', 0.0):.1f}/"
            f"{metrics.get('max_abs_log_ratio_src', 0.0):.1f}/"
            f"{metrics.get('max_abs_log_ratio_tgt', 0.0):.1f}/"
            f"{metrics.get('max_abs_log_ratio_amt', 0.0):.1f}"
            f"{'' if eval_stats is None else f' eval_winrate={eval_stats['eval_win_rate']:.3f}'}"
        )
        if update % max(int(args.log_every), 1) == 0:
            print(line, flush=True)
        append_metrics(metrics_path, update, sps, stats, metrics, eval_stats)


if __name__ == "__main__":
    main()
