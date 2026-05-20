"""Orbit Wars JAX PPO 训练入口。

核心设计：
- collect_and_update: jax.lax.scan 单一 jit 内完成整段 rollout + GAE + PPO 更新。
- policy_sample: 两次前向传播（第一次采 fire/target，第二次带 target 采 amount）。
- actions_from_decisions: 将网络决策转换为游戏动作格式 [from, angle, ships]。
- aim_with_prediction_jax: 60 步轨迹预测锁定目标（旋转星球/彗星均支持）。
- target_is_first_collision_jax: 验证目标星球是否为舰队路径上第一个碰撜体。
- 环境并行度 B = num_envs，星球并行度 P = max_planets。
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.state import parse_observation  # noqa: E402

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from .build_initial_state_bank import build_bank  # noqa: E402
from .checkpoint_opponents import load_checkpoint_opponent  # noqa: E402
from .eval_opponents import EvalContext, default_eval_opponents, eval_metric_fields as build_eval_metric_fields  # noqa: E402
from .fast_env import (  # noqa: E402
    COMETS_PER_SPAWN,
    MAX_COMET_PATH,
    MAX_COMET_SPAWNS,
    EnvParams,
    EnvState,
    comet_arrays_from_observation,
    comet_planet_mask,
    load_state_bank,
    point_segment_distance_sq,
    scores,
    select_from_bank,
    step_batch,
)
from .features_jax import FLEET_DIM, FeatureParams, encode_batch, flatten_sources  # noqa: E402
from .model_jax import ActorCritic  # noqa: E402
from .normalization_jax import NormalizerParams, init_normalizer, normalize_encoded, update_and_normalize  # noqa: E402
from .ppo_jax import PPOParams, Rollout, compute_gae, log_prob_entropy, make_train_state, ppo_update, safe_logits  # noqa: E402
from .reward_jax import RewardParams, reward_from_transition  # noqa: E402
from .train_opponents import nearest_opponent_actions, selfplay_policy_actions  # noqa: E402

NEG_INF = -1.0e9


def _resolve_lr_schedule(cfg) -> str:
    schedule = str(getattr(cfg.ppo, "lr_schedule", "auto") or "auto").lower()
    if schedule == "auto":
        return "linear" if getattr(cfg.ppo, "anneal_lr", False) else "constant"
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


def _entropy_scale(cfg, update: int) -> float:
    return _schedule_value(
        update,
        int(getattr(cfg.ppo, "total_updates", 1)),
        str(getattr(cfg.ppo, "ent_schedule", "constant") or "constant"),
        float(getattr(cfg.ppo, "ent_min_ratio", 1.0)),
    )


def _current_lr(cfg, optimizer_step: int, total_steps: int, warmup_steps: int) -> float:
    schedule = _resolve_lr_schedule(cfg)
    base_lr = float(cfg.ppo.lr)
    min_ratio = float(getattr(cfg.ppo, "min_lr_ratio", 0.0))
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


def flip_state(state: EnvState) -> EnvState:
    """将棋盘视角从 player 0 翻转到 player 1（owner 0↔1 互换）。

    用于自博弈：把当前局面翻转后，网络以为自己是 player 0，实际输出的是 player 1 的动作。
    """
    # 星球 owner：0→1，1→0，中立 -1 不变
    po = state.planets[..., 1]
    new_po = jnp.where(po == 0.0, 1.0, jnp.where(po == 1.0, 0.0, po))
    new_planets = state.planets.at[..., 1].set(new_po)
    # 舰队 owner：只有 0/1，直接取反
    new_fo = 1.0 - state.fleets[..., 1]
    new_fleets = state.fleets.at[..., 1].set(new_fo)
    return state._replace(planets=new_planets, fleets=new_fleets)


def policy_sample_params(params, model, encoded, rng):
    """与 policy_sample 相同，但直接接受 params pytree（用于冻结对手）。"""
    zeros = jnp.zeros(encoded.self_features.shape[:-1], dtype=jnp.int32)
    out1 = model.apply(
        {"params": params},
        encoded.self_features, encoded.candidate_features, encoded.global_features,
        encoded.candidate_mask, encoded.map_features, zeros,
        fleet_features=encoded.fleet_features,
        fleet_mask=encoded.fleet_mask,
    )
    rng_fire, rng_target, rng_amount = jax.random.split(rng, 3)
    has_target = encoded.candidate_mask.at[..., 0].set(False).any(axis=-1) & encoded.source_mask
    fire_logits = jnp.where(has_target, out1.fire_logits, -1.0e8)
    fire = jax.random.uniform(rng_fire, fire_logits.shape) < jax.nn.sigmoid(fire_logits)
    target_logits = jnp.where(encoded.candidate_mask.at[..., 0].set(False), out1.target_logits, NEG_INF)
    target_logits = jnp.where(fire[..., None], target_logits, NEG_INF).at[..., 0].set(jnp.where(fire, target_logits[..., 0], 0.0))
    target_logits = safe_logits(target_logits)
    target = jax.random.categorical(rng_target, target_logits, axis=-1)
    out2 = model.apply(
        {"params": params},
        encoded.self_features, encoded.candidate_features, encoded.global_features,
        encoded.candidate_mask, encoded.map_features, target,
        fleet_features=encoded.fleet_features,
        fleet_mask=encoded.fleet_mask,
    )
    selected_amount_mask = jnp.take_along_axis(encoded.amount_mask, target[..., None, None], axis=-2).squeeze(-2)
    amount_logits = jnp.where(selected_amount_mask, out2.amount_logits, NEG_INF)
    amount_logits = jnp.where((~selected_amount_mask.any(axis=-1))[..., None], 0.0, amount_logits)
    amount = jax.random.categorical(rng_amount, amount_logits, axis=-1)
    return fire, target, amount, None, out1.value


def kaggle_obs_to_jax_state(obs, env_params: EnvParams) -> EnvState:
    """将 kaggle_environments 单步 observation 转换为 batch=1 的 EnvState。

    用于在 Python 游戏循环中把 kaggle obs 喂给 JAX 模型。
    同步解析 observation.comets，保证网页推理阶段的彗星特征和瞄准可用。
    """
    P = env_params.max_planets
    F = env_params.max_fleets
    state = parse_observation(obs)

    planets = np.zeros((P, 7), dtype=np.float32)
    planet_alive = np.zeros(P, dtype=bool)
    planet_id_to_slot: dict[int, int] = {}
    for p in state.planets:
        if p.id < P:
            planets[p.id] = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            planet_alive[p.id] = True
            planet_id_to_slot[p.id] = p.id

    initial_planets = planets.copy()
    for p in state.initial_planets:
        if p.id < P:
            initial_planets[p.id] = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]

    fleets = np.zeros((F, 7), dtype=np.float32)
    fleet_alive = np.zeros(F, dtype=bool)
    for i, fl in enumerate(state.fleets[:F]):
        fleets[i] = [fl.id, fl.owner, fl.x, fl.y, fl.angle, fl.from_planet_id, fl.ships]
        fleet_alive[i] = True
    comet_paths, comet_lengths, comet_ships, comet_valid, comet_path_index, comet_slots = comet_arrays_from_observation(
        obs,
        planet_id_to_slot,
        planets,
    )

    return EnvState(
        planets=jnp.array(planets[None]),          # [1, P, 7]
        initial_planets=jnp.array(initial_planets[None]),
        planet_alive=jnp.array(planet_alive[None]),
        fleets=jnp.array(fleets[None]),            # [1, F, 7]
        fleet_alive=jnp.array(fleet_alive[None]),
        step=jnp.array([state.step], dtype=jnp.int32),
        angular_velocity=jnp.array([state.angular_velocity], dtype=jnp.float32),
        next_fleet_id=jnp.array([int(read_obs_field(obs, "next_fleet_id", len(state.fleets)) or 0)], dtype=jnp.int32),
        done=jnp.array([False]),
        winner=jnp.array([-1], dtype=jnp.int32),
        overflow_fleets=jnp.zeros(1, dtype=jnp.int32),
        comet_paths=jnp.array(comet_paths[None]),
        comet_lengths=jnp.array(comet_lengths[None]),
        comet_ships=jnp.array(comet_ships[None]),
        comet_valid=jnp.array(comet_valid[None]),
        comet_path_index=jnp.array(comet_path_index[None]),
        comet_slots=jnp.array(comet_slots[None]),
    )


def read_obs_field(obs, key: str, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def jax_actions_to_kaggle(actions_arr) -> list:
    """将 JAX actions_from_decisions 输出 [1, P, 3] 转为 kaggle 格式 [[from, angle, ships], ...]。"""
    arr = np.array(actions_arr[0])  # [P, 3]
    return [[float(r[0]), float(r[1]), int(r[2])] for r in arr if float(r[2]) > 0]


def deterministic_policy_actions(train_state, state, feat_params, norm_state, norm_params):
    """argmax 推理路径，用于网页/外部评估/JAX 内部 eval 保持一致。"""
    return deterministic_policy_actions_params(train_state.params, train_state.apply_fn, state, feat_params, norm_state, norm_params)


def deterministic_policy_actions_params(params, apply_fn, state, feat_params, norm_state, norm_params):
    """argmax 推理路径，直接接受 params/apply_fn，用于固定 checkpoint 对手。"""
    raw_enc = encode_batch(state, feat_params)
    enc = normalize_encoded(raw_enc, norm_state, norm_params)
    zeros = jnp.zeros(enc.self_features.shape[:-1], dtype=jnp.int32)
    out1 = apply_fn(
        {"params": params},
        enc.self_features,
        enc.candidate_features,
        enc.global_features,
        enc.candidate_mask,
        enc.map_features,
        zeros,
        fleet_features=enc.fleet_features,
        fleet_mask=enc.fleet_mask,
    )
    has_target = enc.candidate_mask.at[..., 0].set(False).any(axis=-1) & enc.source_mask
    fire = (jax.nn.sigmoid(out1.fire_logits) > 0.5) & has_target
    target_logits = jnp.where(enc.candidate_mask.at[..., 0].set(False), out1.target_logits, NEG_INF)
    target_logits = jnp.where(fire[..., None], target_logits, NEG_INF).at[..., 0].set(
        jnp.where(fire, target_logits[..., 0], 0.0)
    )
    target = jnp.argmax(safe_logits(target_logits), axis=-1)
    out2 = apply_fn(
        {"params": params},
        enc.self_features,
        enc.candidate_features,
        enc.global_features,
        enc.candidate_mask,
        enc.map_features,
        target,
        fleet_features=enc.fleet_features,
        fleet_mask=enc.fleet_mask,
    )
    selected_amount_mask = jnp.take_along_axis(enc.amount_mask, target[..., None, None], axis=-2).squeeze(-2)
    amount_logits = jnp.where(selected_amount_mask, out2.amount_logits, NEG_INF)
    amount_logits = jnp.where((~selected_amount_mask.any(axis=-1))[..., None], 0.0, amount_logits)
    amount = jnp.argmax(amount_logits, axis=-1)
    return actions_from_decisions(
        state,
        enc,
        fire,
        target,
        amount,
        feat_params,
    )


def main() -> None:
    """训练入口：解析参数、加载配置、初始化环境/模型/状态，循环执行 collect_and_update。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_jax_default.yaml")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--bank", default="outputs/jax_initial_state_bank.npz")
    parser.add_argument("--bank-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.updates is not None:
        cfg.ppo.total_updates = args.updates
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.rollout_steps is not None:
        cfg.ppo.rollout_steps = args.rollout_steps

    env_params = EnvParams(
        episode_steps=cfg.env.episode_steps,
        max_planets=cfg.env.max_planets,
        max_fleets=512,
        max_actions=cfg.env.max_planets,
    )
    feat_params = FeatureParams.from_train_config(cfg)
    norm_params = NormalizerParams.from_train_config(cfg)
    reward_params = RewardParams.from_train_config(cfg)
    ent_coef_fire = cfg.ppo.ent_coef if getattr(cfg.ppo, "ent_coef_fire", -1.0) < 0.0 else cfg.ppo.ent_coef_fire
    ent_coef_target = cfg.ppo.ent_coef if getattr(cfg.ppo, "ent_coef_target", -1.0) < 0.0 else cfg.ppo.ent_coef_target
    ent_coef_amount = cfg.ppo.ent_coef if getattr(cfg.ppo, "ent_coef_amount", -1.0) < 0.0 else cfg.ppo.ent_coef_amount
    ppo_params = PPOParams(
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_coef=cfg.ppo.clip_coef,
        ent_coef=cfg.ppo.ent_coef,
        ent_coef_fire=ent_coef_fire,
        ent_coef_target=ent_coef_target,
        ent_coef_amount=ent_coef_amount,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        epochs=cfg.ppo.epochs,
        minibatch_size=min(cfg.ppo.minibatch_size, cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.env.max_planets),
        norm_adv=cfg.ppo.norm_adv,
        target_kl=getattr(cfg.ppo, "target_kl", 0.05),
    )
    bank_path = Path(args.bank)
    if not bank_path.exists():
        build_bank(bank_path, args.bank_size, cfg.seed, env_params)

    bank = load_state_bank(bank_path, env_params)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, rng_reset, rng_model = jax.random.split(rng, 3)
    state = reset_from_bank(bank, rng_reset, cfg.ppo.num_envs)
    raw_sample = encode_batch(state, feat_params)
    sample = raw_sample if getattr(cfg.model, "use_transformer", False) else flatten_sources(raw_sample)
    rollout_state_count = cfg.ppo.num_envs * cfg.ppo.rollout_steps
    ppo_source_count = cfg.env.max_planets
    ppo_states_per_minibatch = min(max(cfg.ppo.minibatch_size // ppo_source_count, 1), rollout_state_count)
    ppo_num_minibatches = max(rollout_state_count // ppo_states_per_minibatch, 1)
    optimizer_steps_per_update = cfg.ppo.epochs * ppo_num_minibatches
    total_optimizer_steps = cfg.ppo.total_updates * optimizer_steps_per_update
    warmup_steps = int(getattr(cfg.ppo, "warmup_updates", 0)) * optimizer_steps_per_update
    lr_schedule_name = _resolve_lr_schedule(cfg)
    model = ActorCritic(
        self_dim=sample.self_features.shape[-1],
        candidate_dim=sample.candidate_features.shape[-1],
        global_dim=sample.global_features.shape[-1],
        candidate_count=cfg.env.max_planets + 1,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        use_cnn=cfg.model.use_cnn,
        use_transformer=getattr(cfg.model, "use_transformer", False),
        fleet_dim=FLEET_DIM,
        max_fleet_tokens=feat_params.max_fleet_tokens,
        num_transformer_layers=getattr(cfg.model, "num_transformer_layers", 2),
        num_heads=getattr(cfg.model, "num_heads", 4),
        value_use_context=getattr(cfg.model, "value_use_context", False),
    )
    train_state = make_train_state(
        rng_model, model, sample, cfg.ppo.lr, cfg.ppo.max_grad_norm,
        total_steps=total_optimizer_steps if lr_schedule_name != "constant" else 0,
        lr_schedule=lr_schedule_name,
        warmup_steps=warmup_steps,
        min_lr_ratio=getattr(cfg.ppo, "min_lr_ratio", 0.0),
    )
    normalizer_state = init_normalizer()
    start_update = 1
    init_checkpoint_params = None
    init_checkpoint_normalizer_state = None

    def _resolve_config_path(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else Path(args.config).resolve().parent / path

    def _load_checkpoint(path_value: str) -> tuple[dict, Path]:
        ckpt_path = _resolve_config_path(path_value)
        with ckpt_path.open("rb") as f:
            return pickle.load(f), ckpt_path

    def _assert_param_compatible(ckpt_params, ckpt_path: Path) -> None:
        expected_leaves, expected_tree = jax.tree_util.tree_flatten(train_state.params)
        loaded_leaves, loaded_tree = jax.tree_util.tree_flatten(ckpt_params)
        if loaded_tree != expected_tree:
            raise ValueError(f"Checkpoint param tree does not match current model: {ckpt_path}")
        mismatches = [
            (i, getattr(exp, "shape", None), getattr(got, "shape", None))
            for i, (exp, got) in enumerate(zip(expected_leaves, loaded_leaves, strict=True))
            if getattr(exp, "shape", None) != getattr(got, "shape", None)
        ]
        if mismatches:
            raise ValueError(
                f"Checkpoint param shapes do not match current model: {ckpt_path}; "
                f"first mismatches={mismatches[:5]}"
            )

    init_ckpt = getattr(cfg, "init_from_checkpoint", None)
    if init_ckpt:
        ckpt, ckpt_path = _load_checkpoint(init_ckpt)
        ckpt_params = ckpt["params"]
        _assert_param_compatible(ckpt_params, ckpt_path)
        train_state = train_state.replace(params=ckpt_params)
        if "normalizer_state" in ckpt:
            normalizer_state = ckpt["normalizer_state"]
        init_checkpoint_params = ckpt_params
        init_checkpoint_normalizer_state = ckpt.get("normalizer_state", normalizer_state)
        if getattr(cfg, "start_update_from_checkpoint", True):
            start_update = int(ckpt.get("update", 0)) + 1
        print(
            f"[init] loaded checkpoint {ckpt_path} "
            f"(ckpt_update={ckpt.get('update', 'unknown')}, start_update={start_update}); "
            "optimizer state is freshly initialized"
        )
    opponent_checkpoint = None
    opponent_ckpt = getattr(cfg, "opponent_checkpoint", None)
    if opponent_ckpt:
        opponent_checkpoint = load_checkpoint_opponent(_resolve_config_path(opponent_ckpt))
        print(
            f"[opponent] loaded checkpoint opponent {opponent_checkpoint.path} "
            f"(ckpt_update={opponent_checkpoint.update if opponent_checkpoint.update is not None else 'unknown'})"
        )
    elif init_ckpt and init_checkpoint_params is not None:
        opponent_checkpoint = load_checkpoint_opponent(_resolve_config_path(init_ckpt))

    eval_checkpoint = opponent_checkpoint
    eval_ckpt = getattr(cfg, "eval_checkpoint", None)
    if eval_ckpt:
        eval_checkpoint = load_checkpoint_opponent(_resolve_config_path(eval_ckpt))
        print(
            f"[eval] loaded checkpoint eval opponent {eval_checkpoint.path} "
            f"(ckpt_update={eval_checkpoint.update if eval_checkpoint.update is not None else 'unknown'})"
        )

    run_dir = Path(cfg.save_dir) / f"{cfg.run_name}_jax"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    eval_opponents = default_eval_opponents(
        ROOT,
        checkpoint_opponent=eval_checkpoint,
        enabled_names=getattr(cfg.eval, "opponents", None),
    )
    eval_metric_fields = build_eval_metric_fields(eval_opponents)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
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
            "entropy_fire",
            "entropy_target",
            "entropy_amount",
            "approx_kl",
            "clipfrac",
            "max_abs_log_ratio",
            "n_epochs_run",
            "lr",
            "ent_scale",
            *eval_metric_fields,
        ])
    print(f"JAX trainer run_dir={run_dir} envs={cfg.ppo.num_envs} rollout_steps={cfg.ppo.rollout_steps} device={jax.devices()}")

    # 初始化 wandb（仅当配置开启且库可用时）
    use_wandb = _WANDB_AVAILABLE and getattr(cfg.wandb, "enabled", False)
    if use_wandb:
        _wandb.init(
            project=getattr(cfg.wandb, "project", "orbit-wars-rl"),
            entity=getattr(cfg.wandb, "entity", None) or None,
            name=cfg.run_name,
            config={
                "num_envs": cfg.ppo.num_envs,
                "rollout_steps": cfg.ppo.rollout_steps,
                "total_updates": cfg.ppo.total_updates,
                "gamma": cfg.ppo.gamma,
                "gae_lambda": cfg.ppo.gae_lambda,
                "clip_coef": cfg.ppo.clip_coef,
                "ent_coef": cfg.ppo.ent_coef,
                "ent_coef_fire": ent_coef_fire,
                "ent_coef_target": ent_coef_target,
                "ent_coef_amount": ent_coef_amount,
                "ent_schedule": getattr(cfg.ppo, "ent_schedule", "constant"),
                "ent_min_ratio": getattr(cfg.ppo, "ent_min_ratio", 1.0),
                "vf_coef": cfg.ppo.vf_coef,
                "lr": cfg.ppo.lr,
                "lr_schedule": lr_schedule_name,
                "warmup_updates": getattr(cfg.ppo, "warmup_updates", 0),
                "min_lr_ratio": getattr(cfg.ppo, "min_lr_ratio", 0.0),
                "max_grad_norm": cfg.ppo.max_grad_norm,
                "epochs": cfg.ppo.epochs,
                "hidden_size": cfg.model.hidden_size,
                "use_cnn": cfg.model.use_cnn,
                "use_transformer": getattr(cfg.model, "use_transformer", False),
                "num_transformer_layers": getattr(cfg.model, "num_transformer_layers", 2),
                "num_heads": getattr(cfg.model, "num_heads", 4),
                "max_fleet_tokens": feat_params.max_fleet_tokens,
                "max_planets": cfg.env.max_planets,
                "opponent": getattr(cfg, "opponent", "nearest"),
            },
            mode=getattr(cfg.wandb, "mode", "online"),
            group=getattr(cfg.wandb, "group", None) or None,
            tags=getattr(cfg.wandb, "tags", []) or [],
            dir=str(run_dir),
        )
        print(f"wandb run: {_wandb.run.url}")
    else:
        print("wandb disabled (set wandb.enabled: true in config to enable)")

    opponent_name = getattr(cfg, "opponent", "nearest")
    self_play = opponent_name == "self"
    checkpoint_play = opponent_name == "checkpoint"
    use_pool = opponent_name == "pool"
    # 对手池模式：
    # - current: 全程对当前策略自博弈，每个 update 前同步 opponent_params。
    # - lag_checkpoint: 全程/分阶段对 pool_lag_interval 步前 checkpoint。
    # - fixed_pool: 旧版固定池，Phase 1 nearest/current/lag，Phase 2 current/lag。
    # - pfsp: 胜率加权采样，优先训练当前打不过的对手。
    raw_pool_mode = str(getattr(cfg, "pool_mode", "pfsp") or "pfsp").lower()
    pool_mode_aliases = {
        "current": "current",
        "lag_checkpoint": "lag_checkpoint",
        "fixed_lag_checkpoint": "lag_checkpoint",
        "fixed": "fixed_pool",
        "fixed_pool": "fixed_pool",
        "pfsp": "pfsp",
        "snapshot": "snapshot",
    }
    if raw_pool_mode not in pool_mode_aliases:
        raise ValueError(
            f"Unknown pool_mode={raw_pool_mode!r}; expected one of "
            "current, lag_checkpoint, fixed_pool, pfsp, snapshot"
        )
    pool_mode = pool_mode_aliases[raw_pool_mode]
    current_pool = pool_mode == "current"
    lag_checkpoint_pool = pool_mode == "lag_checkpoint"
    fixed_pool = pool_mode == "fixed_pool"
    pfsp_pool = pool_mode == "pfsp"
    snapshot_pool = pool_mode == "snapshot"
    phased_pool = lag_checkpoint_pool or fixed_pool
    lagged_pool = lag_checkpoint_pool or fixed_pool or pfsp_pool
    pool_phase = int(getattr(cfg, "pool_start_phase", 1)) if phased_pool else 1
    pool_include_nearest = bool(getattr(cfg, "pool_include_nearest", True)) if pfsp_pool else False
    pool_lag_interval = int(getattr(cfg, "pool_lag_interval", 50)) if lagged_pool else 0
    snapshot_interval = int(getattr(cfg, "snapshot_interval", 200)) if snapshot_pool else 0
    snapshot_current_ratio = float(getattr(cfg, "snapshot_current_ratio", 0.2)) if snapshot_pool else 0.0
    pool_max_size = int(getattr(cfg, "pool_max_size", 4)) if pfsp_pool else 0
    promote_nearest_win_rate = float(getattr(cfg, "promote_nearest_win_rate", 0.8)) if phased_pool else 0.0
    promote_consecutive_evals = int(getattr(cfg, "promote_consecutive_evals", 3)) if phased_pool else 0
    recent_nearest_win_rates: list[float] = []
    pool_rng = random.Random(cfg.seed + 12345)
    if use_pool:
        if current_pool:
            print("opponent=pool current (sync_every_update=true)")
        elif lag_checkpoint_pool:
            print(f"opponent=pool lag_checkpoint (start_phase={pool_phase}, lag_interval={pool_lag_interval}, promote_at={promote_nearest_win_rate:.2f}x{promote_consecutive_evals})")
        elif fixed_pool:
            print(f"opponent=pool fixed_pool (start_phase={pool_phase}, lag_interval={pool_lag_interval}, promote_at={promote_nearest_win_rate:.2f}x{promote_consecutive_evals})")
        elif snapshot_pool:
            print(f"opponent=pool snapshot (interval={snapshot_interval}, current_ratio={snapshot_current_ratio:.0%})")
        else:
            print(f"opponent=pool PFSP (lag_interval={pool_lag_interval}, max_size={pool_max_size}, "
                  f"include_nearest={pool_include_nearest})")
    else:
        print(f"opponent={'checkpoint' if checkpoint_play else ('self_play' if self_play else 'nearest')}")
    opponent_update_freq = cfg.self_play_update_interval  # 默认 25
    eval_enabled = getattr(cfg.eval, "enabled", True)
    eval_interval = cfg.eval.every            # 每多少 update 评估一次
    eval_games = cfg.eval.games               # 每次评估局数
    eval_seed = cfg.eval.seed
    checkpoint_every = getattr(cfg, "checkpoint_every", 100)
    checkpoint_model = opponent_checkpoint.model if opponent_checkpoint is not None else model
    checkpoint_feat_params = opponent_checkpoint.feat_params if opponent_checkpoint is not None else feat_params
    checkpoint_normalizer_state = opponent_checkpoint.normalizer_state if opponent_checkpoint is not None else normalizer_state

    # 冻结对手权重（初始与学习器相同，每 opponent_update_freq 步同步一次）
    opponent_params = (
        jax.tree_util.tree_map(lambda x: x, opponent_checkpoint.params)
        if checkpoint_play and opponent_checkpoint is not None
        else jax.tree_util.tree_map(lambda x: x, train_state.params)
    )
    if checkpoint_play and opponent_checkpoint is None:
        raise ValueError("opponent='checkpoint' requires opponent_checkpoint or init_from_checkpoint")
    # lag_snapshot 保存“上一个 interval 起点的策略”；新 lag_checkpoint/PFSP 接口到下一个
    # interval 时才发布，因此 lag 对手是真正间隔 pool_lag_interval 的 checkpoint。
    lag_snapshot_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
    lagged_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
    snapshot_pool_entries: list[dict] = []  # snapshot pool 快照列表
    pool_entries: list[dict] = []
    if use_pool and pfsp_pool:
        pool_entries = [{"id": "selfplay-current", "type": "current", "params": None, "winrate": 0.5}]
        if pool_include_nearest:
            pool_entries.insert(0, {"id": "nearest", "type": "nearest", "params": None, "winrate": 0.5})

    def _copy_params(params):
        return jax.tree_util.tree_map(lambda x: x, params)

    def _pfsp_sample(entries: list[dict]) -> dict:
        weights = [(1.0 - float(entry["winrate"]) + 0.1) ** 2 for entry in entries]
        return pool_rng.choices(entries, weights=weights, k=1)[0]

    def _pool_summary(entries: list[dict]) -> str:
        return ", ".join(f"{entry['id']}:{float(entry['winrate']):.2f}" for entry in entries)

    def _add_lag_to_pool(update_idx: int, params) -> None:
        pool_entries.append({
            "id": f"selfplay-lag-{max(update_idx - pool_lag_interval, 0):06d}",
            "type": "lag",
            "params": _copy_params(params),
            "winrate": 0.5,
        })
        print(f"[pool] update={update_idx} 加入 50 步前 checkpoint，对手池: {_pool_summary(pool_entries)}")
        _prune_pool(update_idx)

    def _prune_pool(update_idx: int) -> None:
        if len(pool_entries) <= pool_max_size:
            return
        removable = [entry for entry in pool_entries if entry["id"] != "selfplay-current"]
        if not removable:
            return
        strong = [entry for entry in removable if float(entry["winrate"]) > 0.60]
        if strong:
            victim = max(strong, key=lambda entry: float(entry["winrate"]))
            reason = f"胜率超过 60% 的对手 {victim['id']} (winrate={float(victim['winrate']):.3f})"
        else:
            lag_entries = [entry for entry in removable if entry["type"] == "lag"]
            if lag_entries:
                victim = min(lag_entries, key=lambda entry: str(entry["id"]))
                reason = f"最旧 lag checkpoint {victim['id']} (winrate={float(victim['winrate']):.3f})"
            else:
                victim = min(removable, key=lambda entry: float(entry["winrate"]))
                reason = f"最低胜率对手 {victim['id']} (winrate={float(victim['winrate']):.3f})"
        pool_entries.remove(victim)
        print(f"[pool] update={update_idx} 对手池超出 {pool_max_size}，移除{reason}，"
              f"对手池: {_pool_summary(pool_entries)}")

    fixed_phase1_opponents = ["nearest", "selfplay-current", "selfplay-lag"]
    fixed_phase2_opponents = ["selfplay-current", "selfplay-lag"]
    current_opponents = ["selfplay-current"]
    lag_checkpoint_phase1_opponents = list(getattr(cfg, "pool_lag_checkpoint_phase1_opponents", ["selfplay-lag"]))
    lag_checkpoint_phase2_opponents = list(getattr(cfg, "pool_lag_checkpoint_phase2_opponents", ["selfplay-current", "selfplay-lag"]))
    valid_fixed_opponents = {"nearest", "selfplay-current", "selfplay-lag"}
    if lag_checkpoint_pool:
        for phase_name, phase_opponents in (
            ("pool_lag_checkpoint_phase1_opponents", lag_checkpoint_phase1_opponents),
            ("pool_lag_checkpoint_phase2_opponents", lag_checkpoint_phase2_opponents),
        ):
            if not phase_opponents:
                raise ValueError(f"{phase_name} must contain at least one opponent")
            invalid = sorted(set(phase_opponents) - valid_fixed_opponents)
            if invalid:
                raise ValueError(f"{phase_name} contains unknown opponents: {invalid}")

    # JIT 编译的训练步（JAX-native 对手），供 pool 模式按需调用。
    # rollout 收集 + GAE 和 PPO 更新分别 JIT；PPO 的 KL 早停在 JIT 内完成。
    _jit_nearest = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, model, feat_params, cfg.ppo.rollout_steps, "nearest",
        )
    )
    _jit_selfplay = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, model, feat_params, cfg.ppo.rollout_steps, "selfplay",
        )
    )
    _jit_checkpoint = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, checkpoint_model, checkpoint_feat_params, cfg.ppo.rollout_steps, "checkpoint",
        )
    )
    # 兼容旧配置（opponent: nearest / self）
    _jit_step = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout(
            ts,
            opp_p,
            opp_ns,
            st,
            ns,
            key,
            bank,
            env_params,
            feat_params,
            norm_params,
            reward_params,
            ppo_params,
            model,
            checkpoint_model if checkpoint_play else model,
            checkpoint_feat_params if checkpoint_play else feat_params,
            cfg.ppo.rollout_steps,
            "checkpoint" if checkpoint_play else ("selfplay" if self_play else "nearest"),
        )
    )

    def _run_step(jit_fn, ts, opp_p, opp_ns, st, ns, key, ent_scale):
        """JIT 收集 rollout，然后在 Python 层执行 PPO 更新（含 KL 早停）。"""
        ts, st, ns, rollout, returns, advantages, ppo_rng, ep_metrics = jit_fn(ts, opp_p, opp_ns, st, ns, key)
        ts, ppo_metrics = ppo_update(ts, rollout, returns, advantages, ppo_params, ppo_rng, ent_scale)
        return ts, st, ns, {**ppo_metrics, **ep_metrics}

    def _run_named_opponent(opponent_id: str, ts, st, ns, key, ent_scale):
        if opponent_id == "nearest":
            return _run_step(_jit_nearest, ts, opponent_params, ns, st, ns, key, ent_scale)
        if opponent_id == "selfplay-current":
            return _run_step(_jit_selfplay, ts, opponent_params, ns, st, ns, key, ent_scale)
        if opponent_id == "selfplay-lag":
            return _run_step(_jit_selfplay, ts, lagged_params, ns, st, ns, key, ent_scale)
        raise ValueError(f"Unknown training opponent: {opponent_id}")

    for update in range(start_update, cfg.ppo.total_updates + 1):
        selected_pool_entry = None
        selected_pool_id = ""

        # 每隔 opponent_update_freq 步把当前权重同步给 self-play/current 对手
        if (use_pool and current_pool) or ((not checkpoint_play) and (update - 1) % max(opponent_update_freq, 1) == 0):
            opponent_params = _copy_params(train_state.params)
            if self_play or use_pool:
                print(f"[pool] update={update} 同步 selfplay-current 对手权重")

        start = time.perf_counter()
        rng, step_key = jax.random.split(rng)
        ent_scale = _entropy_scale(cfg, update)

        if use_pool:
            if current_pool or lag_checkpoint_pool or fixed_pool:
                if current_pool:
                    candidates = current_opponents
                elif lag_checkpoint_pool:
                    candidates = lag_checkpoint_phase1_opponents if pool_phase == 1 else lag_checkpoint_phase2_opponents
                else:
                    candidates = fixed_phase1_opponents if pool_phase == 1 else fixed_phase2_opponents
                selected_pool_id = pool_rng.choice(candidates)
                train_state, state, normalizer_state, metrics = _run_named_opponent(
                    selected_pool_id,
                    train_state,
                    state,
                    normalizer_state,
                    step_key,
                    ent_scale,
                )
            elif snapshot_pool:
                # 20% 打当前策略，80% 打历史快照；若快照池为空则全打当前
                if snapshot_pool_entries and pool_rng.random() >= snapshot_current_ratio:
                    snap = pool_rng.choice(snapshot_pool_entries)
                    selected_pool_id = snap["id"]
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, snap["params"], normalizer_state, state, normalizer_state, step_key, ent_scale)
                else:
                    selected_pool_id = "selfplay-current"
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, opponent_params, normalizer_state, state, normalizer_state, step_key, ent_scale)
            else:
                selected_pool_entry = _pfsp_sample(pool_entries)
                selected_pool_id = str(selected_pool_entry["id"])
                if selected_pool_entry["type"] == "nearest":
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_nearest, train_state, opponent_params, normalizer_state, state, normalizer_state, step_key, ent_scale)
                elif selected_pool_entry["type"] == "current":
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, opponent_params, normalizer_state, state, normalizer_state, step_key, ent_scale)
                else:
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, selected_pool_entry["params"], normalizer_state, state, normalizer_state, step_key, ent_scale)
        else:
            # 兼容旧配置（opponent: nearest / self）
            if self_play and update % opponent_update_freq == 1:
                pass  # already synced above
            step_jit = _jit_checkpoint if checkpoint_play else _jit_step
            step_opp_norm = checkpoint_normalizer_state if checkpoint_play else normalizer_state
            train_state, state, normalizer_state, metrics = _run_step(
                step_jit, train_state, opponent_params, step_opp_norm, state, normalizer_state, step_key, ent_scale)
        metrics = jax.device_get(metrics)
        current_lr = _current_lr(cfg, int(jax.device_get(train_state.step)), total_optimizer_steps, warmup_steps)
        metrics["lr"] = current_lr
        metrics["ent_scale"] = ent_scale
        if use_pool and pfsp_pool and selected_pool_entry is not None and float(metrics["episodes_per_rollout"]) > 0.0:
            old_wr = float(selected_pool_entry["winrate"])
            selected_pool_entry["winrate"] = 0.95 * old_wr + 0.05 * float(metrics["win_rate"])
            metrics["pool_selected_winrate"] = selected_pool_entry["winrate"]
        elapsed = time.perf_counter() - start
        samples = cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.env.max_planets
        sps = samples / max(elapsed, 1e-9)
        line = (
            f"update={update} sps={sps:.1f} loss={metrics['loss']:.4f} "
            f"ep_reward={metrics['episode_reward']:.3f} winrate={metrics['win_rate']:.3f} "
            f"return={metrics['mean_return']:.3f} eps={metrics['episodes_per_rollout']:.0f} "
            f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.5f} "
            f"logr={metrics.get('max_abs_log_ratio', 0.0):.2f} "
            f"epochs={metrics.get('n_epochs_run', cfg.ppo.epochs):.0f} "
            f"lr={metrics['lr']:.2e} ent_scale={metrics['ent_scale']:.2f}"
        )
        if use_pool:
            if pfsp_pool:
                selected_wr = float(selected_pool_entry["winrate"]) if selected_pool_entry is not None else 0.0
                line += f" opp={selected_pool_id} opp_wr={selected_wr:.3f}"
            else:
                line += f" opp={selected_pool_id}"
        print(line, flush=True)

        # snapshot pool：每 snapshot_interval 步存一个快照
        if use_pool and snapshot_pool and snapshot_interval > 0 and update % snapshot_interval == 0:
            snap_id = f"snap-{update:06d}"
            snapshot_pool_entries.append({"id": snap_id, "params": _copy_params(train_state.params)})
            print(f"[pool] update={update} 存入快照 {snap_id}，快照池大小={len(snapshot_pool_entries)}")

        if use_pool and lagged_pool and pool_lag_interval > 0 and update % pool_lag_interval == 0:
            if fixed_pool:
                lagged_params = _copy_params(train_state.params)
                print(f"[pool] update={update} 同步 fixed_pool lagged 对手权重 (lag={pool_lag_interval})")
            elif lag_checkpoint_pool:
                lagged_params = _copy_params(lag_snapshot_params)
                lag_snapshot_params = _copy_params(train_state.params)
                print(f"[pool] update={update} 发布 lag_checkpoint 对手权重 (lag={pool_lag_interval})")
            elif pfsp_pool:
                _add_lag_to_pool(update, lag_snapshot_params)
                lag_snapshot_params = _copy_params(train_state.params)

        eval_metrics = {field: "" for field in eval_metric_fields}
        if eval_enabled and eval_interval > 0 and update % eval_interval == 0:
            print(f"[eval] update={update} 开始评估 ({eval_games} 局)...")
            rng, eval_rng = jax.random.split(rng)

            eval_ctx = EvalContext(
                train_state=train_state,
                model=model,
                bank=bank,
                env_params=env_params,
                feat_params=feat_params,
                normalizer_state=normalizer_state,
                norm_params=norm_params,
                num_games=eval_games,
                base_seed=eval_seed + update,
                rng=eval_rng,
                deterministic_policy_actions=deterministic_policy_actions,
                deterministic_policy_actions_params=deterministic_policy_actions_params,
                nearest_opponent_actions=nearest_opponent_actions,
                reset_from_bank=reset_from_bank,
                flip_state=flip_state,
                kaggle_obs_to_jax_state=kaggle_obs_to_jax_state,
                jax_actions_to_kaggle=jax_actions_to_kaggle,
            )
            eval_results: dict[str, dict] = {}
            for eval_opponent in eval_opponents:
                if not eval_opponent.exists():
                    continue
                result = eval_opponent.evaluate(eval_ctx)
                if not result:
                    continue
                eval_results[eval_opponent.metric_prefix] = result
                eval_metrics.update(eval_opponent.metric_values(result))
            near_res = eval_results.get("nearest", {})

            # 升阶检测（Pool Phase 1 → Phase 2）：连续 N 次 nearest win_rate >= threshold
            if use_pool and phased_pool and pool_phase == 1 and near_res:
                recent_nearest_win_rates.append(float(near_res["win_rate"]))
                if len(recent_nearest_win_rates) > promote_consecutive_evals:
                    recent_nearest_win_rates.pop(0)
                if (len(recent_nearest_win_rates) >= promote_consecutive_evals
                        and all(r >= promote_nearest_win_rate for r in recent_nearest_win_rates)):
                    pool_phase = 2
                    print(f"[pool] update={update} ★ 升阶 Phase 2：连续 {promote_consecutive_evals} 次 "
                          f"nearest win_rate >= {promote_nearest_win_rate:.2f}，切换为 selfplay-current/selfplay-lag 训练 + tamrazov-starwars eval")

            if use_wandb:
                log_dict: dict = {}
                for prefix, result in eval_results.items():
                    log_dict[f"eval/{prefix}_win_rate"] = float(result["win_rate"])
                if use_pool:
                    if current_pool:
                        fixed_size = len(current_opponents)
                    elif lag_checkpoint_pool:
                        fixed_size = len(lag_checkpoint_phase1_opponents if pool_phase == 1 else lag_checkpoint_phase2_opponents)
                    elif fixed_pool:
                        fixed_size = len(fixed_phase1_opponents if pool_phase == 1 else fixed_phase2_opponents)
                    else:
                        fixed_size = len(pool_entries)
                    if phased_pool:
                        log_dict["pool/phase"] = float(pool_phase)
                    log_dict["pool/size"] = float(len(pool_entries) if pfsp_pool else fixed_size)
                    if phased_pool and recent_nearest_win_rates:
                        log_dict["pool/recent_nearest_win_rate"] = float(np.mean(recent_nearest_win_rates))
                _wandb.log(log_dict, step=update)

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                update,
                f"{sps:.6f}",
                f"{metrics['episode_reward']:.8f}",
                f"{metrics['win_rate']:.8f}",
                f"{metrics['mean_return']:.8f}",
                f"{metrics['episodes_per_rollout']:.1f}",
                f"{metrics['loss']:.8f}",
                f"{metrics['policy_loss']:.8f}",
                f"{metrics['value_loss']:.8f}",
                f"{metrics['entropy']:.8f}",
                f"{metrics.get('entropy_fire', 0.0):.8f}",
                f"{metrics.get('entropy_target', 0.0):.8f}",
                f"{metrics.get('entropy_amount', 0.0):.8f}",
                f"{metrics['approx_kl']:.8f}",
                f"{metrics['clipfrac']:.8f}",
                f"{metrics.get('max_abs_log_ratio', 0.0):.8f}",
                f"{metrics.get('n_epochs_run', cfg.ppo.epochs):.1f}",
                f"{metrics.get('lr', cfg.ppo.lr):.12f}",
                f"{metrics.get('ent_scale', 1.0):.8f}",
                *[eval_metrics[field] for field in eval_metric_fields],
            ])
        if use_wandb:
            _wandb.log({
                "charts/win_rate": float(metrics["win_rate"]),
                "charts/episode_reward": float(metrics["episode_reward"]),
                "charts/mean_return": float(metrics["mean_return"]),
                "charts/episodes_per_rollout": float(metrics["episodes_per_rollout"]),
                "losses/total_loss": float(metrics["loss"]),
                "losses/policy_loss": float(metrics["policy_loss"]),
                "losses/value_loss": float(metrics["value_loss"]),
                "losses/entropy": float(metrics["entropy"]),
                "entropy/fire": float(metrics.get("entropy_fire", 0.0)),
                "entropy/target": float(metrics.get("entropy_target", 0.0)),
                "entropy/amount": float(metrics.get("entropy_amount", 0.0)),
                "entropy/scale": float(metrics.get("ent_scale", 1.0)),
                "losses/approx_kl": float(metrics["approx_kl"]),
                "losses/clipfrac": float(metrics["clipfrac"]),
                "losses/max_abs_log_ratio": float(metrics.get("max_abs_log_ratio", 0.0)),
                "losses/n_epochs_run": float(metrics.get("n_epochs_run", cfg.ppo.epochs)),
                "lr/current": float(metrics.get("lr", cfg.ppo.lr)),
            }, step=update)
            if use_pool:
                if current_pool:
                    fixed_size = len(current_opponents)
                elif lag_checkpoint_pool:
                    fixed_size = len(lag_checkpoint_phase1_opponents if pool_phase == 1 else lag_checkpoint_phase2_opponents)
                elif fixed_pool:
                    fixed_size = len(fixed_phase1_opponents if pool_phase == 1 else fixed_phase2_opponents)
                else:
                    fixed_size = len(pool_entries)
                pool_log = {"pool/size": float(len(pool_entries) if pfsp_pool else fixed_size)}
                if pfsp_pool:
                    selected_wr = float(selected_pool_entry["winrate"]) if selected_pool_entry is not None else 0.0
                    pool_log["pool/selected_win_rate"] = selected_wr
                _wandb.log(pool_log, step=update)

        # ── 定期保存 checkpoint ───────────────────────────────────
        if update % checkpoint_every == 0:
            ckpt_path = run_dir / f"ckpt_{update:06d}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "params": jax.device_get(train_state.params),
                    "normalizer_state": jax.device_get(normalizer_state),
                    "update": update,
                    "cfg": cfg,
                }, f)
            print(f"[ckpt] saved {ckpt_path}")


def reset_from_bank(bank: EnvState, rng, num_envs: int) -> EnvState:
    """从预建 state bank 中随机采样 num_envs 个初始状态。"""
    size = bank.planets.shape[0]
    idx = jax.random.randint(rng, (num_envs,), 0, size)
    return select_from_bank(bank, idx)


def collect_rollout(
    train_state,
    opponent_params,
    opponent_normalizer_state,
    state,
    normalizer_state,
    rng,
    bank,
    env_params,
    feat_params,
    norm_params,
    reward_params,
    ppo_params,
    model,
    opponent_model,
    opponent_feat_params,
    rollout_steps: int,
    opponent_kind: str = "nearest",
):
    """收集 rollout，计算 GAE 和 episode 统计（JIT 安全，不含 PPO 更新）。

    1. lax.scan 循环 rollout_steps 步：特征展开 + 采样 + step 环境 + 奖励 + done 处理。
    2. 计算最后状态价值 + GAE 估计。
    3. 计算 episode 统计（win_rate, episode_reward 等）。
    注：PPO 更新由调用方单独执行；ppo_update 自身是 JIT 编译的，
    KL 早停在 JIT 内用 while_loop 完成。

    opponent_params: 冻结的对手权重 pytree（opponent_kind 为 selfplay/checkpoint 时使用）。
    """
    rng, rollout_key = jax.random.split(rng)

    num_envs = state.planets.shape[0]
    max_planets = feat_params.max_planets
    window = reward_params.failed_capture_window  # 通常 50
    use_failed_capture = reward_params.failed_capture_penalty != 0.0

    # 环形历史缓冲区：记录过去 window 步内每个 (env, slot) 的发射动作和目标
    fire_hist   = jnp.zeros((window, num_envs, max_planets), dtype=bool)
    target_hist = jnp.zeros((window, num_envs, max_planets), dtype=jnp.int32)
    step_cnt    = jnp.array(0, dtype=jnp.int32)

    def body(carry, _):
        """lax.scan 循环体：处理单步。"""
        ts, st, ns, key, fire_hist, target_hist, step_cnt = carry
        key, action_key, opp_key, reset_key = jax.random.split(key, 4)
        raw_encoded = encode_batch(st, feat_params)
        encoded, ns = update_and_normalize(raw_encoded, ns, norm_params)
        fire_2d, target_2d, amount_2d, logp, value = policy_sample(ts, model, encoded, action_key)
        learner_actions, learner_policy_mask = actions_from_decisions_with_policy_mask(
            st,
            encoded,
            fire_2d,
            target_2d,
            amount_2d,
            feat_params,
        )
        # 对手动作：训练对手必须是 JAX-native，保证 rollout 仍可 JIT。
        if opponent_kind == "selfplay" or opponent_kind == "checkpoint":
            opp_norm_state = opponent_normalizer_state if opponent_kind == "checkpoint" else ns
            opp_model = opponent_model if opponent_kind == "checkpoint" else model
            opp_feat_params = opponent_feat_params if opponent_kind == "checkpoint" else feat_params
            opponent_actions = selfplay_policy_actions(
                st,
                opponent_params,
                opp_model,
                opp_norm_state,
                opp_feat_params,
                norm_params,
                opp_key,
                flip_state_fn=flip_state,
                policy_sample_params_fn=policy_sample_params,
                actions_from_decisions_fn=actions_from_decisions,
            )
        else:
            opponent_actions = nearest_opponent_actions(st, player=1, params=feat_params)
        joint = jnp.stack([learner_actions, opponent_actions], axis=1)  # 双方动作合并
        next_state = step_batch(st, joint, env_params)
        reward_env, reward_parts = reward_from_transition(st, next_state, learner_actions, reward_params)

        # ── failed-capture 惩罚 ─────────────────────────────────────────────────
        # 读出 window 步前在同一环形 slot 的发射记录
        hist_slot = step_cnt % window
        old_fire   = jax.lax.dynamic_index_in_dim(fire_hist,   hist_slot, axis=0, keepdims=False)  # [B, P]
        old_target = jax.lax.dynamic_index_in_dim(target_hist, hist_slot, axis=0, keepdims=False)  # [B, P]
        # 查当前目标星球 owner：planets [B, P_all, 7]，第 1 列 owner（0=player0, 1=player1）
        # learner 是 player 0
        cur_owner = st.planets[:, :, 1].astype(jnp.int32)   # [B, P_all]
        # 把 old_target（候选 index，范围 0..max_planets）映射到 owner，slot 0（noop）对应 owner=-1
        safe_target = jnp.clip(old_target, 0, max_planets - 1)
        target_owner = jnp.take_along_axis(cur_owner, safe_target, axis=-1)  # [B, P]
        is_noop_target = (old_target == 0)
        # 判断失败：window 步前有发射 & 目标不是 noop & 目标现在不属于 player0
        failed = old_fire & (~is_noop_target) & (target_owner != 0)   # [B, P]
        source_mask_1d = encoded.source_mask   # [B, P] bool
        failed = failed & source_mask_1d
        failed_penalty = jnp.where(
            use_failed_capture,
            failed.astype(jnp.float32).sum(axis=-1) * reward_params.failed_capture_penalty,
            0.0,
        )  # [B]
        reward_env = reward_env - failed_penalty

        # 更新环形缓冲区（done 时清掉该 env 行，防止跨局污染）
        done_env = next_state.done  # [B]
        fire_hist_new   = fire_hist.at[hist_slot].set(jnp.where(done_env[:, None], False,  fire_2d))
        target_hist_new = target_hist.at[hist_slot].set(jnp.where(done_env[:, None], 0,    target_2d))
        step_cnt_new = step_cnt + 1
        # ───────────────────────────────────────────────────────────────────────

        reward = jnp.repeat(reward_env[:, None], feat_params.max_planets, axis=1)  # 扩展到 P 维
        done = jnp.repeat(next_state.done[:, None], feat_params.max_planets, axis=1)
        reset_state = reset_from_bank(bank, reset_key, st.planets.shape[0])
        next_state = replace_done(next_state, reset_state)  # 局结环境自动重置
        row = Rollout(
            encoded.self_features,
            encoded.candidate_features,
            encoded.global_features,
            encoded.map_features,
            encoded.fleet_features,
            encoded.fleet_mask,
            encoded.candidate_mask,
            encoded.amount_mask,
            encoded.source_mask,
            learner_policy_mask,
            fire_2d,
            target_2d,
            amount_2d,
            logp,
            value,
            reward,
            done,
        )
        return (ts, next_state, ns, key, fire_hist_new, target_hist_new, step_cnt_new), row

    (train_state, state, normalizer_state, rng, _fh, _th, _sc), rollout = jax.lax.scan(
        body,
        (train_state, state, normalizer_state, rollout_key, fire_hist, target_hist, step_cnt),
        None,
        length=rollout_steps,
    )
    last_raw_encoded = encode_batch(state, feat_params)
    last_encoded = normalize_encoded(last_raw_encoded, normalizer_state, norm_params)
    last_out = train_state.apply_fn(
        {"params": train_state.params},
        last_encoded.self_features,
        last_encoded.candidate_features,
        last_encoded.global_features,
        last_encoded.candidate_mask,
        last_encoded.map_features,
        jnp.zeros(last_encoded.self_features.shape[:-1], dtype=jnp.int32),
        fleet_features=last_encoded.fleet_features,
        fleet_mask=last_encoded.fleet_mask,
    )
    last_values = last_out.value
    returns, advantages = compute_gae(rollout.reward, rollout.done, rollout.old_value, last_values, ppo_params)
    # rollout.done: [T, B, P]，每步每 env 的结束标志（P 维相同）
    # rollout.reward: [T, B, P]，只有 done=True 那步有 ±1，其余为 0
    finished = rollout.done.any(axis=-1)          # [T, B] 该步是否有局结束
    ep_reward = rollout.reward.mean(axis=-1)       # [T, B] 对 P 维平均（P 维全同）
    # episode_reward：只统计已结束局的终局奖励均值（非零步），反映真实胜负
    ep_reward_sum = jnp.where(finished, ep_reward, 0.0).sum()
    episodes = jnp.maximum(finished.sum(), 1)
    episode_reward = ep_reward_sum / episodes
    # win_rate：结束局中学习方获胜（reward > 0）的比率
    wins = ((ep_reward > 0.0) & finished).sum()
    win_rate = wins / episodes
    # mean_return：rollout 内所有样本 return（GAE 回报）的均值，反映 value 估计水平
    mean_return = returns.mean()
    ep_metrics = {
        "episode_reward": episode_reward,   # 已结束局的终局奖励均值（接近 ±1）
        "win_rate": win_rate,               # 已结束局中的胜率
        "mean_return": mean_return,         # GAE 回报均值
        "episodes_per_rollout": finished.sum().astype(jnp.float32),  # 每次 rollout 完成的局数
    }
    # 返回 rollout 数据供 Python 层调用 ppo_update（rng 供 ppo_update 中 minibatch shuffle）
    return train_state, state, normalizer_state, rollout, returns, advantages, rng, ep_metrics


def policy_sample(train_state, model, encoded, rng):
    """两阶段采样策略。

    第一步：传入 zeros target 前向 -> 采 fire/target。
    第二步：传入真实 target 前向 -> 采 amount，并计算联合 logp。
    与 log_prob_entropy 训练路径完全一致。
    """
    zeros = jnp.zeros(encoded.self_features.shape[:-1], dtype=jnp.int32)
    out1 = train_state.apply_fn(
        {"params": train_state.params},
        encoded.self_features,
        encoded.candidate_features,
        encoded.global_features,
        encoded.candidate_mask,
        encoded.map_features,
        zeros,
        fleet_features=encoded.fleet_features,
        fleet_mask=encoded.fleet_mask,
    )
    rng_fire, rng_target, rng_amount = jax.random.split(rng, 3)
    has_target = encoded.candidate_mask.at[..., 0].set(False).any(axis=-1) & encoded.source_mask
    fire_logits = jnp.where(has_target, out1.fire_logits, -1.0e8)  # 没有可用目标时强制 fire=0
    fire = jax.random.uniform(rng_fire, fire_logits.shape) < jax.nn.sigmoid(fire_logits)
    # 屏蔽 no-op slot，使 target 必须是阵中星球
    target_logits = jnp.where(encoded.candidate_mask.at[..., 0].set(False), out1.target_logits, NEG_INF)
    target_logits = jnp.where(fire[..., None], target_logits, NEG_INF).at[..., 0].set(jnp.where(fire, target_logits[..., 0], 0.0))
    target_logits = safe_logits(target_logits)
    target = jax.random.categorical(rng_target, target_logits, axis=-1)
    out2 = train_state.apply_fn(
        {"params": train_state.params},
        encoded.self_features,
        encoded.candidate_features,
        encoded.global_features,
        encoded.candidate_mask,
        encoded.map_features,
        target,
        fleet_features=encoded.fleet_features,
        fleet_mask=encoded.fleet_mask,
    )
    selected_amount_mask = jnp.take_along_axis(encoded.amount_mask, target[..., None, None], axis=-2).squeeze(-2)
    amount_logits = jnp.where(selected_amount_mask, out2.amount_logits, NEG_INF)
    amount_logits = jnp.where((~selected_amount_mask.any(axis=-1))[..., None], 0.0, amount_logits)
    amount = jax.random.categorical(rng_amount, amount_logits, axis=-1)
    logp, _entropy, _entropy_fire, _entropy_target, _entropy_amount = log_prob_entropy(
        out2,
        fire,
        target,
        amount,
        encoded.candidate_mask,
        encoded.amount_mask,
    )
    return fire, target, amount, logp, out1.value


def actions_from_decisions(state, encoded, fire, target, amount, params: FeatureParams):
    """将网络决策转换为游戏动作 [from_id, angle_rad, ships]。

    fire=False 或 target=0 (属于 no-op) 时输出全 0。
    """
    actions, _policy_mask = actions_from_decisions_with_policy_mask(state, encoded, fire, target, amount, params)
    return actions


def actions_from_decisions_with_policy_mask(state, encoded, fire, target, amount, params: FeatureParams):
    """转换动作，并返回 PPO policy mask。

    fire=True 但物理瞄准不可达的动作会被安全层变成 no-op；这类样本不应
    用 raw fire/target/amount 的 logp 更新 policy。真实 no-op(fire=False)
    和真实执行动作继续参与 policy loss。
    """
    b, p = fire.shape
    source_ids = encoded.source_ids.astype(jnp.float32)
    ships = jnp.take_along_axis(encoded.ship_buckets, amount[..., None], axis=-1).squeeze(-1)
    target_idx = jnp.clip(target - 1, 0, params.max_planets - 1)
    batch_idx = jnp.arange(b)[:, None]
    target_planets = state.planets[batch_idx, target_idx]
    src_planets = state.planets
    angle, aim_hit = aim_with_prediction_jax(
        src_planets,
        target_planets,
        target_idx,
        state,
        ships,
    )
    valid = fire & (target > 0) & encoded.source_mask & aim_hit
    attempted_invalid = fire & (target > 0) & encoded.source_mask & (~aim_hit)
    policy_mask = encoded.source_mask & (~attempted_invalid)
    actions = jnp.stack(
        [
            jnp.where(valid, source_ids, 0.0),
            jnp.where(valid, angle, 0.0),
            jnp.where(valid, ships, 0.0),
        ],
        axis=-1,
    )
    return actions, policy_mask


def aim_with_prediction_jax(src, target, target_idx, state: EnvState, ships):
    """与 src/physics.py 的 aim_with_prediction 同步的 JAX 批量瞄准。

    流程保持一致：
    1. 在 60 回合视野内粗扫描 f(T) 的正到负符号变化。
    2. 二分 12 次得到连续命中时间 T*。
    3. 只验证 floor(T*)-1/floor(T*)/floor(T*)+1 三个整数回合。
    4. 每个候选回合都按引擎顺序检查太阳、出界、第一碰撞和目标碰撞。
    """

    horizon = 60
    n_coarse = 60
    n_bisect = 12
    clearance = 0.1
    speed = fleet_speed_local(jnp.maximum(ships, 1.0))
    max_turns = target_max_turns_jax(state, target_idx, horizon).astype(jnp.float32)
    can_search = max_turns >= 1.0

    samples = jnp.arange(1, n_coarse + 1, dtype=jnp.float32)
    dt = max_turns[..., None] / float(n_coarse)
    ts = samples * dt
    f_cur = aim_f_jax(src, target, target_idx, state, speed, ts)
    f_prev0 = aim_f_jax(src, target, target_idx, state, speed, jnp.full_like(max_turns, 1e-6))
    f_prev = jnp.concatenate([f_prev0[..., None], f_cur[..., :-1]], axis=-1)
    crossing = (f_prev > 0.0) & (f_cur <= 0.0) & can_search[..., None]
    has_interval = crossing.any(axis=-1)
    first_i = jnp.argmax(crossing, axis=-1).astype(jnp.float32)
    t_lo = first_i * (max_turns / float(n_coarse))
    t_hi = (first_i + 1.0) * (max_turns / float(n_coarse))

    def bisect_body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        f_lo = aim_f_jax(src, target, target_idx, state, speed, lo)
        f_mid = aim_f_jax(src, target, target_idx, state, speed, mid)
        use_left = (f_lo * f_mid) <= 0.0
        return jnp.where(use_left, lo, mid), jnp.where(use_left, mid, hi)

    t_lo, t_hi = jax.lax.fori_loop(0, n_bisect, bisect_body, (t_lo, t_hi))
    t_star = 0.5 * (t_lo + t_hi)
    t_floor = jnp.floor(t_star).astype(jnp.int32)
    candidates = jnp.stack(
        [
            jnp.maximum(t_floor - 1, 1),
            jnp.maximum(t_floor, 1),
            t_floor + 1,
        ],
        axis=-1,
    )
    candidate_turns = candidates.astype(jnp.float32)
    candidate_in_range = (candidate_turns >= 1.0) & (candidate_turns <= max_turns[..., None])

    aim_t = candidate_turns - 1.0
    px, py, target_pos_valid = target_pos_at_time_jax(state, target_idx, aim_t)
    angle = jnp.arctan2(py - src[..., None, 3], px - src[..., None, 2])
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    fx0 = src[..., None, 2] + cos_a * (src[..., None, 4] + clearance)
    fy0 = src[..., None, 3] + sin_a * (src[..., None, 4] + clearance)
    fx_t = fx0 + candidate_turns * speed[..., None] * cos_a
    fy_t = fy0 + candidate_turns * speed[..., None] * sin_a

    sun_hit = point_segment_distance_sq(50.0, 50.0, fx0, fy0, fx_t, fy_t) < 100.0
    first_collision_ok = target_is_first_collision_jax(
        src,
        target_idx,
        state,
        fx0,
        fy0,
        speed[..., None],
        cos_a,
        sin_a,
        candidates,
    )

    fx_prev = fx0 + (candidate_turns - 1.0) * speed[..., None] * cos_a
    fy_prev = fy0 + (candidate_turns - 1.0) * speed[..., None] * sin_a
    move_d2 = point_segment_distance_sq(px, py, fx_prev, fy_prev, fx_t, fy_t)
    move_hits_target = move_d2 < target[..., None, 4] ** 2

    px_next, py_next, target_next_valid = target_pos_at_time_jax(state, target_idx, candidate_turns)
    target_moved = ((px_next - px) ** 2 + (py_next - py) ** 2) > 1e-9
    sweep_d2 = point_segment_distance_sq(fx_t, fy_t, px, py, px_next, py_next)
    sweep_hits_target = target_moved & target_next_valid & (sweep_d2 < target[..., None, 4] ** 2)

    candidate_valid = (
        has_interval[..., None]
        & candidate_in_range
        & target_pos_valid
        & (~sun_hit)
        & first_collision_ok
        & (move_hits_target | sweep_hits_target)
        & (ships[..., None] > 0.0)
    )
    has_valid = candidate_valid.any(axis=-1)
    chosen = jnp.argmax(candidate_valid, axis=-1)
    chosen_angle = jnp.take_along_axis(angle, chosen[..., None], axis=-1).squeeze(-1)
    direct_angle = jnp.arctan2(target[..., 3] - src[..., 3], target[..., 2] - src[..., 2])
    return jnp.where(has_valid, chosen_angle, direct_angle), has_valid


def aim_f_jax(src, target, target_idx, state: EnvState, speed, turns):
    """f(T)=dist(src,target(T))-(src.radius+clearance+T*speed)-target.radius。"""
    px, py, valid = target_pos_at_time_jax(state, target_idx, turns)
    src_x = _expand_like_time(src[..., 2], turns)
    src_y = _expand_like_time(src[..., 3], turns)
    src_radius = _expand_like_time(src[..., 4], turns)
    speed_t = _expand_like_time(speed, turns)
    target_radius = _expand_like_time(target[..., 4], turns)
    dist = jnp.sqrt((px - src_x) ** 2 + (py - src_y) ** 2)
    value = dist - (src_radius + 0.1 + turns * speed_t) - target_radius
    return jnp.where(valid, value, jnp.inf)


def target_pos_at_time_jax(state: EnvState, target_idx, turns):
    """目标星球在浮点 turns 后的位置，支持普通星球和彗星线性插值。"""
    b, p = target_idx.shape
    batch_idx = jnp.arange(b)[:, None]
    planets = state.planets[batch_idx, target_idx]
    initial = state.initial_planets[batch_idx, target_idx]
    alive = state.planet_alive[batch_idx, target_idx]

    base_x = _expand_like_time(planets[..., 2], turns)
    base_y = _expand_like_time(planets[..., 3], turns)
    dx = initial[..., 2] - 50.0
    dy = initial[..., 3] - 50.0
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    phi_cur = jnp.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    phi = _expand_like_time(phi_cur, turns) + _expand_like_time(state.angular_velocity[:, None], turns) * turns
    rot_x = 50.0 + _expand_like_time(orbit_r, turns) * jnp.cos(phi)
    rot_y = 50.0 + _expand_like_time(orbit_r, turns) * jnp.sin(phi)
    comet_mask = jax.vmap(comet_planet_mask)(state)
    is_comet = comet_mask[batch_idx, target_idx]
    rotating = (orbit_r + initial[..., 4] < 50.0) & alive & (~is_comet)
    px = jnp.where(_expand_like_time(rotating, turns), rot_x, base_x)
    py = jnp.where(_expand_like_time(rotating, turns), rot_y, base_y)
    valid = _expand_like_time(alive, turns)

    comet_x, comet_y, comet_valid = comet_target_pos_at_time_jax(state, target_idx, turns)
    is_comet_t = _expand_like_time(is_comet, turns)
    px = jnp.where(is_comet_t, comet_x, px)
    py = jnp.where(is_comet_t, comet_y, py)
    valid = jnp.where(is_comet_t, comet_valid, valid)
    return px, py, valid


def comet_target_pos_at_time_jax(state: EnvState, target_idx, turns):
    """彗星目标在浮点 turns 后的位置。非彗星返回无效占位。"""
    extra = turns.ndim - target_idx.ndim
    one_shape = (1,) * extra
    slots = state.comet_slots.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    target_e = target_idx.reshape(target_idx.shape + one_shape + (1, 1))
    active = (
        (state.comet_slots >= 0)
        & (state.comet_path_index[:, :, None] >= -1)
        & (state.comet_lengths > 0)
    ).reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    match = (slots == target_e) & active

    path_index = state.comet_path_index.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, 1))
    lengths = state.comet_lengths.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    fi = path_index + turns[..., None, None]
    i0 = jnp.floor(fi).astype(jnp.int32)
    frac = fi - i0.astype(jnp.float32)
    valid_i0 = (i0 >= 0) & (i0 < lengths)
    i1 = jnp.minimum(i0 + 1, lengths - 1)
    has_i1 = i0 + 1 < lengths
    i0_safe = jnp.clip(i0, 0, MAX_COMET_PATH - 1)
    i1_safe = jnp.clip(i1, 0, MAX_COMET_PATH - 1)
    path_shape = (state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2)
    paths = state.comet_paths.reshape(path_shape)
    one0 = jax.nn.one_hot(i0_safe, MAX_COMET_PATH, dtype=jnp.float32)
    one1 = jax.nn.one_hot(i1_safe, MAX_COMET_PATH, dtype=jnp.float32)
    pos0 = (one0[..., None] * paths).sum(axis=-2)
    pos1 = (one1[..., None] * paths).sum(axis=-2)
    pos = jnp.where(has_i1[..., None], pos0 + frac[..., None] * (pos1 - pos0), pos0)
    weights = (match & valid_i0).astype(jnp.float32)
    has = weights.sum(axis=(-2, -1)) > 0.0
    px = (pos[..., 0] * weights).sum(axis=(-2, -1))
    py = (pos[..., 1] * weights).sum(axis=(-2, -1))
    return px, py, has


def target_max_turns_jax(state: EnvState, target_idx, horizon: int):
    """普通目标搜索 horizon；彗星目标按 src/physics.py 限制为 remaining_life-1。"""
    b, p = target_idx.shape
    slots = state.comet_slots[:, None, :, :]
    target_e = target_idx[:, :, None, None]
    active = (state.comet_slots[:, None, :, :] >= 0) & (state.comet_path_index[:, None, :, None] >= -1)
    match = (slots == target_e) & active
    remaining = state.comet_lengths[:, None, :, :] - state.comet_path_index[:, None, :, None]
    comet_remaining = jnp.where(match, remaining, 0).max(axis=(-2, -1))
    comet_has = match.any(axis=(-2, -1))
    comet_max = jnp.minimum(jnp.asarray(horizon, dtype=jnp.int32), jnp.maximum(0, comet_remaining - 1))
    return jnp.where(comet_has, comet_max, horizon)


def target_is_first_collision_jax(src, target_idx, state: EnvState, fx0, fy0, speed, cos_a, sin_a, hit_turn):
    """按引擎阶段验证候选回合的第一碰撞对象就是目标。"""
    horizon = 60
    turns_i = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    turns = turns_i.astype(jnp.float32)
    planets = state.planets
    planet_alive = state.planet_alive
    px_t, py_t = all_planet_pos_at_turns(state, horizon)
    fx_prev = fx0[..., None] + (turns - 1.0) * speed[..., None] * cos_a[..., None]
    fy_prev = fy0[..., None] + (turns - 1.0) * speed[..., None] * sin_a[..., None]
    fx_cur = fx0[..., None] + turns * speed[..., None] * cos_a[..., None]
    fy_cur = fy0[..., None] + turns * speed[..., None] * sin_a[..., None]
    active_step = turns_i.reshape((1,) * hit_turn.ndim + (horizon,)) <= hit_turn[..., None]
    in_bounds = (fx_cur >= 0.0) & (fx_cur <= 100.0) & (fy_cur >= 0.0) & (fy_cur <= 100.0)
    in_bounds_until_hit = (in_bounds | (~active_step)).all(axis=-1)

    src_axes = (None,) * (hit_turn.ndim - 1)
    px_old = px_t[(slice(None),) + src_axes + (slice(None), slice(None, -1))]
    py_old = py_t[(slice(None),) + src_axes + (slice(None), slice(None, -1))]
    px_new = px_t[(slice(None),) + src_axes + (slice(None), slice(1, None))]
    py_new = py_t[(slice(None),) + src_axes + (slice(None), slice(1, None))]
    radii = planets[(slice(None),) + src_axes + (slice(None), 4)]
    alive = planet_alive[(slice(None),) + src_axes + (slice(None), None)]

    move_d2 = point_segment_distance_sq(
        px_old,
        py_old,
        fx_prev[..., None, :],
        fy_prev[..., None, :],
        fx_cur[..., None, :],
        fy_cur[..., None, :],
    )
    move_hit = (move_d2 < radii[..., None] ** 2) & alive & active_step[..., None, :]

    sweep_active = ((px_new - px_old) ** 2 + (py_new - py_old) ** 2) > 1e-9
    sweep_d2 = point_segment_distance_sq(
        fx_cur[..., None, :],
        fy_cur[..., None, :],
        px_old,
        py_old,
        px_new,
        py_new,
    )
    sweep_hit = (sweep_d2 < radii[..., None] ** 2) & alive & sweep_active & active_step[..., None, :]

    large = jnp.int32(2 * (horizon + 1) + 1)
    move_keys = (2 * turns_i).reshape((1,) * (move_hit.ndim - 1) + (horizon,))
    sweep_keys = (2 * turns_i + 1).reshape((1,) * (sweep_hit.ndim - 1) + (horizon,))
    first_move = jnp.where(move_hit, move_keys, large).min(axis=-1)
    first_sweep = jnp.where(sweep_hit, sweep_keys, large).min(axis=-1)
    first_by_planet = jnp.minimum(first_move, first_sweep)
    first_planet = first_by_planet.argmin(axis=-1)
    first_key = first_by_planet.min(axis=-1)
    target_key_lo = 2 * hit_turn
    target_key_hi = 2 * hit_turn + 1
    return (
        in_bounds_until_hit
        & (first_key < large)
        & (first_planet == target_idx[..., None])
        & (first_key >= target_key_lo)
        & (first_key <= target_key_hi)
    )


def _expand_like_time(x, turns):
    """把 [B,P] 标量特征扩展到 turns 的 [B,P,...] 形状。"""
    return x.reshape(x.shape + (1,) * (turns.ndim - x.ndim))


def all_planet_pos_at_turns(state: EnvState, horizon: int):
    """计算未来 horizon 步内全部星球位置 [B, P, horizon+1]。

    step 0 为当前位置；静止星球固定，旋转星球按角速度追踪，彗星使用预存路径。
    """
    planets = state.planets
    initial_planets = state.initial_planets
    planet_alive = state.planet_alive
    turns = jnp.arange(horizon + 1, dtype=jnp.float32)
    dx = initial_planets[..., 2] - 50.0
    dy = initial_planets[..., 3] - 50.0
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    comet_mask = jax.vmap(comet_planet_mask)(state)
    rotating = (orbit_r + initial_planets[..., 4] < 50.0) & planet_alive & (~comet_mask)
    phi_cur = jnp.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    phi = phi_cur[..., None] + state.angular_velocity[:, None, None] * turns
    px = 50.0 + orbit_r[..., None] * jnp.cos(phi)
    py = 50.0 + orbit_r[..., None] * jnp.sin(phi)
    px = jnp.where(rotating[..., None], px, planets[..., None, 2])
    py = jnp.where(rotating[..., None], py, planets[..., None, 3])

    comet_slots = state.comet_slots.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    safe_slots = jnp.where(comet_slots >= 0, comet_slots, 0)
    group_active = state.comet_path_index[:, :, None] >= -1
    active = (state.comet_slots >= 0) & group_active
    active_flat = active.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    path_index = state.comet_path_index[:, :, None, None] + jnp.arange(horizon + 1, dtype=jnp.int32)[None, None, None, :]
    path_index = jnp.broadcast_to(path_index, (state.planets.shape[0], MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1))
    valid_t = active[..., None] & (path_index >= 0) & (path_index < state.comet_lengths[..., None])
    path_index = jnp.clip(path_index, 0, MAX_COMET_PATH - 1)
    gather_index = jnp.broadcast_to(
        path_index[..., None],
        (state.planets.shape[0], MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1, 2),
    )
    path_pos = jnp.take_along_axis(state.comet_paths, gather_index, axis=3)
    path_pos = jnp.where(valid_t[..., None], path_pos, -1000.0)
    flat_pos = path_pos.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN, horizon + 1, 2)
    slot_onehot = jax.nn.one_hot(safe_slots, state.planets.shape[1], dtype=jnp.float32) * active_flat[..., None].astype(jnp.float32)
    comet_has = slot_onehot.sum(axis=1) > 0.0
    comet_px = jnp.einsum("bkp,bkt->bpt", slot_onehot, flat_pos[..., 0])
    comet_py = jnp.einsum("bkp,bkt->bpt", slot_onehot, flat_pos[..., 1])
    px = jnp.where(comet_has[..., None], comet_px, px)
    py = jnp.where(comet_has[..., None], comet_py, py)
    return px, py


def fleet_speed_local(ships):
    """根据舰数计算舰队速度：与 orbit_wars.py fleet_speed 完全一致。"""
    ratio = jnp.clip(jnp.log(jnp.maximum(ships, 1.0)) / jnp.log(1000.0), 0.0, 1.0)
    return jnp.minimum(1.0 + 5.0 * (ratio**1.5), 6.0)


def replace_done(state: EnvState, reset: EnvState) -> EnvState:
    """局结环境用 reset 收筞替换。"""
    mask = state.done

    def choose(a, b):
        """按 done mask 选择环境 a（未结局）或 b（重置）。"""
        return jnp.where(mask.reshape((-1,) + (1,) * (a.ndim - 1)), b, a) if a.ndim > 0 else jnp.where(mask, b, a)

    return EnvState(
        planets=choose(state.planets, reset.planets),
        initial_planets=choose(state.initial_planets, reset.initial_planets),
        planet_alive=choose(state.planet_alive, reset.planet_alive),
        fleets=choose(state.fleets, reset.fleets),
        fleet_alive=choose(state.fleet_alive, reset.fleet_alive),
        step=choose(state.step, reset.step),
        angular_velocity=choose(state.angular_velocity, reset.angular_velocity),
        next_fleet_id=choose(state.next_fleet_id, reset.next_fleet_id),
        done=choose(state.done, reset.done),
        winner=choose(state.winner, reset.winner),
        overflow_fleets=choose(state.overflow_fleets, reset.overflow_fleets),
        comet_paths=choose(state.comet_paths, reset.comet_paths),
        comet_lengths=choose(state.comet_lengths, reset.comet_lengths),
        comet_ships=choose(state.comet_ships, reset.comet_ships),
        comet_valid=choose(state.comet_valid, reset.comet_valid),
        comet_path_index=choose(state.comet_path_index, reset.comet_path_index),
        comet_slots=choose(state.comet_slots, reset.comet_slots),
    )


if __name__ == "__main__":
    main()
