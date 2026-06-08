"""Orbit Wars JAX PPO 训练入口。

核心设计：
- autoregressive actor 逐 token 生成多条 [source, angle, ships] 动作。
- rollout 收集与 PPO 更新走 state-level AutoRollout。
- aim_with_prediction_jax: 60 步轨迹预测锁定目标（旋转星球/彗星均支持）。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .config import load_config  # noqa: E402
from .state import parse_observation  # noqa: E402

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from .build_initial_state_bank import build_bank  # noqa: E402
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
from .features_jax import FLEET_DIM, FeatureParams, encode_batch  # noqa: E402
from .model_jax import AutoregressiveActorCritic  # noqa: E402
from .normalization_jax import NormalizerParams, init_normalizer, normalize_encoded, update_and_normalize  # noqa: E402
from .ppo_jax import (
    AutoRollout,
    PPOParams,
    compute_gae,
    compute_gae_state,
    make_auto_train_state,
    make_sharded_ppo_update_auto,
    safe_logits,
)  # noqa: E402
from .reward_jax import RewardParams, reward_from_transition  # noqa: E402
from .train_opponents import nearest_opponent_actions, selfplay_policy_actions  # noqa: E402

NEG_INF = -1.0e9


def _is_array_leaf(value) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _put_array_tree(tree, sharding):
    if sharding is None:
        return tree
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding) if _is_array_leaf(x) else x,
        tree,
    )


def _put_leading_axis_tree(tree, sharding, leading_size: int):
    if sharding is None:
        return tree

    def place(x):
        if _is_array_leaf(x) and getattr(x, "ndim", 0) > 0 and int(x.shape[0]) == int(leading_size):
            return jax.device_put(x, sharding)
        return x

    return jax.tree_util.tree_map(place, tree)


def _tree_to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) if _is_array_leaf(x) else x, tree)


def _assert_tree_compatible(name: str, expected, loaded, ckpt_path: Path) -> None:
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    loaded_leaves, loaded_tree = jax.tree_util.tree_flatten(loaded)
    if loaded_tree != expected_tree:
        raise ValueError(f"{name} tree does not match current training state: {ckpt_path}")
    mismatches = [
        (i, getattr(exp, "shape", None), getattr(got, "shape", None))
        for i, (exp, got) in enumerate(zip(expected_leaves, loaded_leaves, strict=True))
        if getattr(exp, "shape", None) != getattr(got, "shape", None)
    ]
    if mismatches:
        raise ValueError(
            f"{name} shapes do not match current training state: {ckpt_path}; "
            f"first mismatches={mismatches[:5]}"
        )


def _checkpoint_train_state_payload(train_state):
    return {
        "step": jax.device_get(train_state.step),
        "params": jax.device_get(train_state.params),
        "opt_state": jax.device_get(train_state.opt_state),
    }


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def _check_nccl_shared_memory(device_count: int) -> None:
    """Fail early for the common Docker /dev/shm limit that breaks NCCL."""
    if device_count < 2:
        return
    if os.environ.get("NCCL_SHM_DISABLE", "").strip() == "1":
        print("[multi-device] NCCL_SHM_DISABLE=1 is set; NCCL shared memory is disabled and multi-GPU may be slower")
        return

    shm_path = Path("/dev/shm")
    if not shm_path.exists():
        return

    stats = os.statvfs(shm_path)
    total = stats.f_frsize * stats.f_blocks
    free = stats.f_frsize * stats.f_bavail
    min_total = max(512 * 1024 * 1024, device_count * 256 * 1024 * 1024)
    min_free = max(256 * 1024 * 1024, device_count * 128 * 1024 * 1024)
    if total >= min_total and free >= min_free:
        return

    raise RuntimeError(
        "Multi-GPU NCCL needs more /dev/shm space. "
        f"Detected /dev/shm total={_format_bytes(total)}, free={_format_bytes(free)}, "
        f"devices={device_count}. Start the container with a larger shared-memory mount, for example "
        "`docker run --gpus all --ipc=host ...` or `docker run --gpus all --shm-size=16g ...`. "
        "If you cannot change the container, set `NCCL_SHM_DISABLE=1` as a slower workaround."
    )


def _make_train_shardings(cfg):
    requested = bool(getattr(cfg, "multi_device", False)) or str(getattr(cfg, "device", "auto")).lower() in {
        "all",
        "data_parallel",
        "multi",
        "multi_device",
        "multi_gpu",
    }
    devices = jax.local_devices()
    if not requested:
        return None, None, None, devices
    if len(devices) < 2:
        print("[multi-device] requested but only one local JAX device is visible; using single-device jit")
        return None, None, None, devices
    if cfg.ppo.num_envs % len(devices) != 0:
        raise ValueError(
            f"multi_device requires ppo.num_envs ({cfg.ppo.num_envs}) to be divisible by "
            f"local device count ({len(devices)})."
        )
    _check_nccl_shared_memory(len(devices))
    mesh = Mesh(np.asarray(devices), ("data",))
    data_sharding = NamedSharding(mesh, P("data"))
    rollout_sharding = NamedSharding(mesh, P(None, "data"))
    replicated_sharding = NamedSharding(mesh, P())
    print(
        f"[multi-device] enabled devices={len(devices)} "
        f"envs_per_device={cfg.ppo.num_envs // len(devices)} devices={devices}"
    )
    return data_sharding, rollout_sharding, replicated_sharding, devices


def _collect_rollout_out_shardings(data_sharding, rollout_sharding, replicated_sharding):
    if data_sharding is None:
        return None
    return (
        replicated_sharding,  # train_state
        data_sharding,        # EnvState [B, ...]
        replicated_sharding,  # normalizer_state
        rollout_sharding,     # AutoRollout [T, B, ...]
        rollout_sharding,     # returns [T, B]
        rollout_sharding,     # advantages [T, B]
        replicated_sharding,  # rng
        replicated_sharding,  # metrics dict
    )


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


def _flip_owner(owner):
    return jnp.where(owner == 0.0, 1.0, jnp.where(owner == 1.0, 0.0, owner))


def _rotate_angle_pi(angle):
    return jnp.arctan2(-jnp.sin(angle), -jnp.cos(angle))


def _mirror_planet_indices(count: int) -> jnp.ndarray:
    idx = jnp.arange(count, dtype=jnp.int32)
    mirror = (idx // 4) * 4 + (3 - (idx % 4))
    return jnp.where(mirror < count, mirror, idx)


def flip_actions(actions: jnp.ndarray) -> jnp.ndarray:
    """把规范视角动作旋转 180 度，转回真实地图坐标。"""
    ships = actions[..., 2]
    angle = jnp.where(ships > 0.0, _rotate_angle_pi(actions[..., 1]), 0.0)
    return actions.at[..., 1].set(angle)


def flip_state(state: EnvState) -> EnvState:
    """将 player 1 局面转成 player 0 规范视角。

    Orbit Wars 的 2p 初始点关于地图中心 180 度对称。player 1 推理时需要
    owner 互换、坐标旋转、舰队角度旋转、对称星球槽位重排，才能看到与
    player 0 一致的视角。星球行内的真实 id 不改，输出动作仍可直接映射
    回真实环境。
    """
    planet_perm = _mirror_planet_indices(state.planets.shape[-2])
    new_planet_alive = state.planet_alive[..., planet_perm]

    planets = state.planets[..., planet_perm, :]
    po = planets[..., 1]
    new_po = _flip_owner(po)
    new_planets = planets.at[..., 1].set(new_po)
    new_planets = new_planets.at[..., 2].set(100.0 - planets[..., 2])
    new_planets = new_planets.at[..., 3].set(100.0 - planets[..., 3])

    initial_planets = state.initial_planets[..., planet_perm, :]
    ipo = initial_planets[..., 1]
    new_initial_planets = initial_planets.at[..., 1].set(_flip_owner(ipo))
    new_initial_planets = new_initial_planets.at[..., 2].set(100.0 - initial_planets[..., 2])
    new_initial_planets = new_initial_planets.at[..., 3].set(100.0 - initial_planets[..., 3])

    new_fo = _flip_owner(state.fleets[..., 1])
    new_fleets = state.fleets.at[..., 1].set(new_fo)
    new_fleets = new_fleets.at[..., 2].set(100.0 - state.fleets[..., 2])
    new_fleets = new_fleets.at[..., 3].set(100.0 - state.fleets[..., 3])
    new_fleets = new_fleets.at[..., 4].set(_rotate_angle_pi(state.fleets[..., 4]))

    new_comet_paths = state.comet_paths.at[..., 0].set(100.0 - state.comet_paths[..., 0])
    new_comet_paths = new_comet_paths.at[..., 1].set(100.0 - state.comet_paths[..., 1])
    safe_comet_slots = jnp.clip(state.comet_slots, 0, state.planets.shape[-2] - 1)
    new_comet_slots = jnp.where(state.comet_slots >= 0, planet_perm[safe_comet_slots], -1)

    return state._replace(
        planets=new_planets,
        initial_planets=new_initial_planets,
        planet_alive=new_planet_alive,
        fleets=new_fleets,
        comet_paths=new_comet_paths,
        comet_slots=new_comet_slots,
    )


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


def auto_deterministic_policy_actions(train_state, model, state, feat_params, norm_state, norm_params):
    """评估用贪心自回归策略。

    逻辑与 auto_policy_sample 对齐，但所有 head 都使用 argmax，不采样。
    返回动作形状为 [B, max_planets, 3]。
    """
    raw_enc = encode_batch(state, feat_params)
    encoded = normalize_encoded(raw_enc, norm_state, norm_params)
    b = encoded.planet_features.shape[0]
    k = model.max_action_tokens
    initial_ships = jnp.concatenate(
        [jnp.zeros((b, 1), dtype=encoded.source_ships.dtype), encoded.source_ships], axis=1
    )
    actions = jnp.zeros((b, feat_params.max_planets, 3), dtype=jnp.float32)
    ended = jnp.zeros((b,), dtype=bool)

    state_emb, h, prev_tok, remaining, planet_out, relative_h, key_proj, src_key_proj, _value = train_state.apply_fn(
        {"params": train_state.params},
        encoded.planet_features,
        encoded.relative_features,
        encoded.global_features,
        encoded.candidate_mask,
        initial_ships,
        map_features=encoded.map_features,
        fleet_features=encoded.fleet_features,
        fleet_mask=encoded.fleet_mask,
        method=model.auto_start,
    )
    batch_idx = jnp.arange(b)

    for t in range(k):
        h_new, end_logits_t, src_logits_t = train_state.apply_fn(
            {"params": train_state.params},
            state_emb, h, prev_tok, remaining,
            encoded.relative_features, encoded.source_mask, src_key_proj,
            method=model.auto_step_base,
        )
        tv = ~ended
        end_t = tv & (jax.nn.sigmoid(end_logits_t) > 0.5)
        src_t = jnp.argmax(safe_logits(src_logits_t), axis=-1)

        src_for_heads = jnp.where(tv & (~end_t), src_t, 0)
        tgt_logits_t = train_state.apply_fn(
            {"params": train_state.params},
            h_new, src_for_heads, planet_out, relative_h, key_proj,
            encoded.candidate_mask,
            method=model.auto_tgt_logits,
        )
        tgt_t = jnp.argmax(safe_logits(tgt_logits_t), axis=-1)

        tgt_for_heads = jnp.where(tv & (~end_t), tgt_t, 0)
        amt_logits_t = train_state.apply_fn(
            {"params": train_state.params},
            h_new, src_for_heads, tgt_for_heads, planet_out,
            relative_h, remaining, encoded.amount_mask,
            method=model.auto_amt_logits,
        )
        amt_t = jnp.argmax(safe_logits(amt_logits_t), axis=-1)

        src_slot = jnp.clip(src_t - 1, 0, feat_params.max_planets - 1)
        tgt_slot = jnp.clip(tgt_t - 1, 0, feat_params.max_planets - 1)
        source_buckets = dynamic_ship_buckets(
            remaining[batch_idx, jnp.clip(src_t, 0, feat_params.max_planets)],
            feat_params.ship_bucket_count,
        )
        ships = jnp.take_along_axis(source_buckets, amt_t[:, None], axis=-1).squeeze(-1)
        src_planets = state.planets[batch_idx, src_slot]
        tgt_planets = state.planets[batch_idx, tgt_slot]
        angle, aim_hit = aim_with_prediction_jax(
            src_planets[:, None, :], tgt_planets[:, None, :], tgt_slot[:, None], state, ships[:, None]
        )
        angle = angle[:, 0]
        aim_hit = aim_hit[:, 0]
        src_valid = (src_t > 0) & encoded.source_mask[batch_idx, src_slot]
        tgt_valid = tgt_t > 0
        exec_t = tv & (~end_t) & src_valid & tgt_valid & aim_hit

        source_ids = encoded.source_ids[batch_idx, src_slot].astype(jnp.float32)
        action_t = jnp.stack(
            [
                jnp.where(exec_t, source_ids, 0.0),
                jnp.where(exec_t, angle, 0.0),
                jnp.where(exec_t, ships, 0.0),
            ],
            axis=-1,
        )
        actions = actions.at[:, t, :].set(action_t)
        remaining = remaining.at[batch_idx, jnp.clip(src_t, 0, feat_params.max_planets)].add(
            jnp.where(exec_t, -ships, 0.0)
        )

        prev_tok = train_state.apply_fn(
            {"params": train_state.params},
            src_for_heads, tgt_for_heads,
            jnp.where(tv & (~end_t), amt_t, 0),
            end_t, exec_t, t, planet_out,
            method=model.auto_token_embed,
        )
        ended = ended | end_t
        h = h_new

    return actions


def auto_opponent_policy_actions(
    state,
    opponent_params,
    model,
    opp_fp,
    opp_ns,
    norm_params,
    max_action_tokens: int,
    *,
    flip_state_fn,
    rng,
):
    """自博弈训练用的冻结参数对手。

    它先把棋盘翻转到 player 1 视角，再用冻结的 opponent_params 生成动作。
    与 learner 的 ``auto_policy_sample`` 保持一致：end / src / tgt / amount
    四个 head 都按当前 logits 采样（categorical / bernoulli），不再 argmax，
    用于训练时给学习者提供有探索的对手。评估路径仍使用
    ``auto_deterministic_policy_actions`` 走 argmax。
    """
    flipped = flip_state_fn(state)
    raw_enc = encode_batch(flipped, opp_fp)
    encoded = normalize_encoded(raw_enc, opp_ns, norm_params)
    b = encoded.planet_features.shape[0]
    k = max_action_tokens
    initial_ships = jnp.concatenate(
        [jnp.zeros((b, 1), dtype=encoded.source_ships.dtype), encoded.source_ships], axis=1
    )
    actions = jnp.zeros((b, opp_fp.max_planets, 3), dtype=jnp.float32)
    ended = jnp.zeros((b,), dtype=bool)

    state_emb, h, prev_tok, remaining, planet_out, relative_h, key_proj, src_key_proj, _value = model.apply(
        {"params": opponent_params},
        encoded.planet_features, encoded.relative_features, encoded.global_features,
        encoded.candidate_mask, initial_ships,
        map_features=encoded.map_features, fleet_features=encoded.fleet_features, fleet_mask=encoded.fleet_mask,
        method=model.auto_start,
    )
    batch_idx = jnp.arange(b)

    for t in range(k):
        h_new, end_logits_t, src_logits_t = model.apply(
            {"params": opponent_params},
            state_emb, h, prev_tok, remaining, encoded.relative_features, encoded.source_mask, src_key_proj,
            method=model.auto_step_base,
        )
        rng, key_end, key_src, key_tgt, key_amt = jax.random.split(rng, 5)
        tv = ~ended
        sampled_end = jax.random.uniform(key_end, (b,)) < jax.nn.sigmoid(end_logits_t)
        end_t = tv & sampled_end
        src_t = jax.random.categorical(key_src, safe_logits(src_logits_t), axis=-1)
        src_for_heads = jnp.where(tv & (~end_t), src_t, 0)

        tgt_logits_t = model.apply(
            {"params": opponent_params},
            h_new, src_for_heads, planet_out, relative_h, key_proj, encoded.candidate_mask,
            method=model.auto_tgt_logits,
        )
        tgt_t = jax.random.categorical(key_tgt, safe_logits(tgt_logits_t), axis=-1)
        tgt_for_heads = jnp.where(tv & (~end_t), tgt_t, 0)

        amt_logits_t = model.apply(
            {"params": opponent_params},
            h_new, src_for_heads, tgt_for_heads, planet_out,
            relative_h, remaining, encoded.amount_mask,
            method=model.auto_amt_logits,
        )
        amt_t = jax.random.categorical(key_amt, safe_logits(amt_logits_t), axis=-1)

        src_slot = jnp.clip(src_t - 1, 0, opp_fp.max_planets - 1)
        tgt_slot = jnp.clip(tgt_t - 1, 0, opp_fp.max_planets - 1)
        source_buckets = dynamic_ship_buckets(
            remaining[batch_idx, jnp.clip(src_t, 0, opp_fp.max_planets)], opp_fp.ship_bucket_count
        )
        ships = jnp.take_along_axis(source_buckets, amt_t[:, None], axis=-1).squeeze(-1)
        src_planets = flipped.planets[batch_idx, src_slot]
        tgt_planets = flipped.planets[batch_idx, tgt_slot]
        angle, aim_hit = aim_with_prediction_jax(
            src_planets[:, None, :], tgt_planets[:, None, :], tgt_slot[:, None], flipped, ships[:, None]
        )
        angle = angle[:, 0]
        aim_hit = aim_hit[:, 0]
        src_valid = (src_t > 0) & encoded.source_mask[batch_idx, src_slot]
        tgt_valid = tgt_t > 0
        exec_t = tv & (~end_t) & src_valid & tgt_valid & aim_hit
        source_ids = encoded.source_ids[batch_idx, src_slot].astype(jnp.float32)
        action_t = jnp.stack(
            [jnp.where(exec_t, source_ids, 0.0), jnp.where(exec_t, angle, 0.0), jnp.where(exec_t, ships, 0.0)],
            axis=-1,
        )
        actions = actions.at[:, t, :].set(action_t)
        remaining = remaining.at[batch_idx, jnp.clip(src_t, 0, opp_fp.max_planets)].add(
            jnp.where(exec_t, -ships, 0.0)
        )
        prev_tok = model.apply(
            {"params": opponent_params},
            src_for_heads, tgt_for_heads,
            jnp.where(tv & (~end_t), amt_t, 0),
            end_t, exec_t, t, planet_out,
            method=model.auto_token_embed,
        )
        ended = ended | end_t
        h = h_new

    return flip_actions(actions)


def main() -> None:
    """训练入口：解析参数、加载配置、初始化环境/模型/状态，循环执行 collect_and_update。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_jax_fixed_pool_4090.yaml")
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
        clip_vloss=cfg.ppo.clip_vloss,
        target_kl=getattr(cfg.ppo, "target_kl", 0.05),
    )
    bank_path = Path(args.bank)
    if not bank_path.exists():
        build_bank(bank_path, args.bank_size, cfg.seed, env_params)

    bank = load_state_bank(bank_path, env_params)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, rng_reset, rng_model = jax.random.split(rng, 3)
    state = reset_from_bank(bank, rng_reset, cfg.ppo.num_envs)
    sample = encode_batch(state, feat_params)
    rollout_state_count = cfg.ppo.num_envs * cfg.ppo.rollout_steps
    ppo_source_count = 1
    ppo_states_per_minibatch = min(max(cfg.ppo.minibatch_size // ppo_source_count, 1), rollout_state_count)
    ppo_num_minibatches = max(rollout_state_count // ppo_states_per_minibatch, 1)
    optimizer_steps_per_update = cfg.ppo.epochs * ppo_num_minibatches
    total_optimizer_steps = cfg.ppo.total_updates * optimizer_steps_per_update
    warmup_steps = int(getattr(cfg.ppo, "warmup_updates", 0)) * optimizer_steps_per_update
    lr_schedule_name = _resolve_lr_schedule(cfg)
    model = AutoregressiveActorCritic(
        planet_dim=sample.planet_features.shape[-1],
        relative_dim=sample.relative_features.shape[-1],
        global_dim=sample.global_features.shape[-1],
        relative_count=cfg.env.max_planets + 1,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        use_cnn=cfg.model.use_cnn,
        fleet_dim=FLEET_DIM,
        max_fleet_tokens=feat_params.max_fleet_tokens,
        num_transformer_layers=(
            int(getattr(cfg.model, "num_transformer_layers", 2))
            if bool(getattr(cfg.model, "use_transformer", False))
            else 0
        ),
        num_heads=getattr(cfg.model, "num_heads", 4),
        max_action_tokens=int(getattr(cfg.model, "max_action_tokens", 16)),
        end_bias_init=float(getattr(cfg.model, "end_bias_init", -2.0)),
    )
    train_state = make_auto_train_state(
        rng_model, model, sample, cfg.ppo.lr, cfg.ppo.max_grad_norm,
        max_action_tokens=int(getattr(cfg.model, "max_action_tokens", 16)),
        total_steps=total_optimizer_steps if lr_schedule_name != "constant" else 0,
        lr_schedule=lr_schedule_name,
        warmup_steps=warmup_steps,
        min_lr_ratio=getattr(cfg.ppo, "min_lr_ratio", 0.0),
        weight_decay=getattr(cfg.ppo, "weight_decay", 0.0),
    )
    normalizer_state = init_normalizer()

    def deterministic_policy_actions(ts, st, fp, ns, np):  # noqa: E501
        return auto_deterministic_policy_actions(ts, model, st, fp, ns, np)

    start_update = 1
    init_ckpt_path: Path | None = None
    resume_runtime_state: dict | None = None
    resumed_full_state = False

    def _resolve_config_path(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else Path(args.config).resolve().parent / path

    def _load_checkpoint(path_value: str) -> tuple[dict, Path]:
        ckpt_path = _resolve_config_path(path_value)
        with ckpt_path.open("rb") as f:
            return pickle.load(f), ckpt_path

    def _restore_train_state_from_checkpoint(ckpt: dict, ckpt_path: Path):
        nonlocal train_state
        saved_train_state = ckpt.get("train_state")
        if saved_train_state is None:
            ckpt_params = ckpt["params"]
            _assert_tree_compatible("Checkpoint params", train_state.params, ckpt_params, ckpt_path)
            train_state = train_state.replace(params=_tree_to_jax(ckpt_params))
            return False

        if isinstance(saved_train_state, dict):
            ckpt_params = saved_train_state["params"]
            ckpt_opt_state = saved_train_state.get("opt_state")
            ckpt_step = saved_train_state.get("step", train_state.step)
        else:
            ckpt_params = saved_train_state.params
            ckpt_opt_state = saved_train_state.opt_state
            ckpt_step = saved_train_state.step

        _assert_tree_compatible("Checkpoint params", train_state.params, ckpt_params, ckpt_path)
        if ckpt_opt_state is None:
            train_state = train_state.replace(params=_tree_to_jax(ckpt_params), step=jnp.asarray(ckpt_step))
            return False
        _assert_tree_compatible("Checkpoint optimizer state", train_state.opt_state, ckpt_opt_state, ckpt_path)
        train_state = train_state.replace(
            step=jnp.asarray(ckpt_step),
            params=_tree_to_jax(ckpt_params),
            opt_state=_tree_to_jax(ckpt_opt_state),
        )
        return True

    init_ckpt = getattr(cfg, "init_from_checkpoint", None)
    if init_ckpt:
        ckpt, ckpt_path = _load_checkpoint(init_ckpt)
        init_ckpt_path = ckpt_path
        resumed_full_state = _restore_train_state_from_checkpoint(ckpt, ckpt_path)
        if "normalizer_state" in ckpt:
            _assert_tree_compatible("Checkpoint normalizer state", normalizer_state, ckpt["normalizer_state"], ckpt_path)
            normalizer_state = _tree_to_jax(ckpt["normalizer_state"])
        if resumed_full_state and "training_state" in ckpt:
            runtime_state = ckpt["training_state"]
            if "env_state" in runtime_state:
                _assert_tree_compatible("Checkpoint env state", state, runtime_state["env_state"], ckpt_path)
                state = _tree_to_jax(runtime_state["env_state"])
            if "rng" in runtime_state:
                rng = _tree_to_jax(runtime_state["rng"])
            resume_runtime_state = runtime_state
        if getattr(cfg, "start_update_from_checkpoint", True):
            start_update = int(ckpt.get("update", 0)) + 1
        resume_mode = "full training state" if resumed_full_state and resume_runtime_state is not None else "params/init state"
        print(
            f"[init] loaded checkpoint {ckpt_path} "
            f"(ckpt_update={ckpt.get('update', 'unknown')}, start_update={start_update}); "
            f"resume_mode={resume_mode}"
        )

    data_sharding, rollout_sharding, replicated_sharding, _train_devices = _make_train_shardings(cfg)

    def _place_train_runtime(ts, st, ns):
        if data_sharding is None:
            return ts, st, ns
        ts = _put_array_tree(ts, replicated_sharding)
        st = _put_leading_axis_tree(st, data_sharding, cfg.ppo.num_envs)
        ns = _put_array_tree(ns, replicated_sharding)
        return ts, st, ns

    train_state, state, normalizer_state = _place_train_runtime(train_state, state, normalizer_state)
    if replicated_sharding is not None:
        bank = _put_array_tree(bank, replicated_sharding)

    run_dir = Path(cfg.save_dir) / f"{cfg.run_name}_jax"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    eval_opponents = default_eval_opponents(
        REPO_ROOT,
        enabled_names=getattr(cfg.eval, "opponents", None),
        init_checkpoint_path=init_ckpt_path,
    )
    eval_metric_fields = build_eval_metric_fields(eval_opponents)
    append_metrics = resumed_full_state and metrics_path.exists() and start_update > 1
    with metrics_path.open("a" if append_metrics else "w", newline="", encoding="utf-8") as f:
        if not append_metrics:
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
                "weight_decay": getattr(cfg.ppo, "weight_decay", 0.0),
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
                "pool_mode": getattr(cfg, "pool_mode", "pfsp"),
                "openai_five_current_ratio": getattr(cfg, "openai_five_current_ratio", 0.8),
                "openai_five_snapshot_interval": getattr(cfg, "openai_five_snapshot_interval", 100),
                "openai_five_max_past_opponents": getattr(cfg, "openai_five_max_past_opponents", 24),
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
    if opponent_name == "checkpoint":
        raise ValueError("opponent='checkpoint' was removed; use self-play/pool or external eval opponents")
    use_pool = opponent_name == "pool"
    # 对手池模式：
    # - current: 全程对当前策略自博弈，每个 update 前同步 opponent_params。
    # - lag_checkpoint: 全程/分阶段对 pool_lag_interval 步前 checkpoint。
    # - fixed_pool: 旧版固定池，Phase 1 nearest/current/lag，Phase 2 current/lag。
    # - pfsp: 胜率加权采样，优先训练当前打不过的对手。
    # - openai_five: 80% 对最新参数，20% 对历史快照；历史对手按 softmax quality 采样。
    raw_pool_mode = str(getattr(cfg, "pool_mode", "pfsp") or "pfsp").lower()
    pool_mode_aliases = {
        "current": "current",
        "lag_checkpoint": "lag_checkpoint",
        "fixed_lag_checkpoint": "lag_checkpoint",
        "fixed": "fixed_pool",
        "fixed_pool": "fixed_pool",
        "pfsp": "pfsp",
        "snapshot": "snapshot",
        "openai_five": "openai_five",
        "openai-five": "openai_five",
        "oai_five": "openai_five",
    }
    if raw_pool_mode not in pool_mode_aliases:
        raise ValueError(
            f"Unknown pool_mode={raw_pool_mode!r}; expected one of "
            "current, lag_checkpoint, fixed_pool, pfsp, snapshot, openai_five"
        )
    pool_mode = pool_mode_aliases[raw_pool_mode]
    current_pool = pool_mode == "current"
    lag_checkpoint_pool = pool_mode == "lag_checkpoint"
    fixed_pool = pool_mode == "fixed_pool"
    pfsp_pool = pool_mode == "pfsp"
    snapshot_pool = pool_mode == "snapshot"
    openai_five_pool = pool_mode == "openai_five"
    phased_pool = lag_checkpoint_pool or fixed_pool
    lagged_pool = lag_checkpoint_pool or fixed_pool or pfsp_pool
    pool_phase = int(getattr(cfg, "pool_start_phase", 1)) if phased_pool else 1
    pool_include_nearest = bool(getattr(cfg, "pool_include_nearest", True)) if pfsp_pool else False
    pool_lag_interval = int(getattr(cfg, "pool_lag_interval", 50)) if lagged_pool else 0
    fixed_pool_current_ratio = float(getattr(cfg, "fixed_pool_current_ratio", 0.5)) if fixed_pool else 0.0
    snapshot_interval = int(getattr(cfg, "snapshot_interval", 200)) if snapshot_pool else 0
    snapshot_current_ratio = float(getattr(cfg, "snapshot_current_ratio", 0.2)) if snapshot_pool else 0.0
    pool_max_size = int(getattr(cfg, "pool_max_size", 4)) if pfsp_pool else 0
    openai_five_current_ratio = (
        min(max(float(getattr(cfg, "openai_five_current_ratio", 0.8)), 0.0), 1.0)
        if openai_five_pool else 0.0
    )
    openai_five_snapshot_interval = (
        max(int(getattr(cfg, "openai_five_snapshot_interval", 100)), 0)
        if openai_five_pool else 0
    )
    openai_five_quality_lr = (
        max(float(getattr(cfg, "openai_five_quality_lr", 0.01)), 0.0)
        if openai_five_pool else 0.0
    )
    openai_five_temperature = (
        max(float(getattr(cfg, "openai_five_temperature", 1.0)), 1.0e-6)
        if openai_five_pool else 1.0
    )
    openai_five_max_past_opponents = (
        max(int(getattr(cfg, "openai_five_max_past_opponents", 24)), 0)
        if openai_five_pool else 0
    )
    openai_five_include_nearest = (
        bool(getattr(cfg, "openai_five_include_nearest", True))
        if openai_five_pool else False
    )
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
            print(
                f"opponent=pool fixed_pool (start_phase={pool_phase}, lag_interval={pool_lag_interval}, "
                f"current_ratio={fixed_pool_current_ratio:.0%}, promote_at={promote_nearest_win_rate:.2f}x{promote_consecutive_evals})"
            )
        elif snapshot_pool:
            print(f"opponent=pool snapshot (interval={snapshot_interval}, current_ratio={snapshot_current_ratio:.0%})")
        elif openai_five_pool:
            print(
                "opponent=pool openai_five "
                f"(current_ratio={openai_five_current_ratio:.0%}, "
                f"past_ratio={1.0 - openai_five_current_ratio:.0%}, "
                f"snapshot_interval={openai_five_snapshot_interval}, "
                f"max_past={openai_five_max_past_opponents}, "
                f"include_nearest={openai_five_include_nearest}, "
                f"quality_lr={openai_five_quality_lr:.4f}, "
                f"temperature={openai_five_temperature:.3f})"
            )
        else:
            print(f"opponent=pool PFSP (lag_interval={pool_lag_interval}, max_size={pool_max_size}, "
                  f"include_nearest={pool_include_nearest})")
    else:
        print(f"opponent={'self_play' if self_play else 'nearest'}")
    opponent_update_freq = cfg.self_play_update_interval  # 默认 25
    eval_enabled = getattr(cfg.eval, "enabled", True)
    eval_interval = cfg.eval.every            # 每多少 update 评估一次
    _nearest_every = int(getattr(cfg.eval, "nearest_every", 0))
    _slow_every    = int(getattr(cfg.eval, "slow_every",    0))
    eval_nearest_interval = _nearest_every if _nearest_every > 0 else eval_interval   # nearest 专属间隔
    eval_slow_interval    = _slow_every    if _slow_every    > 0 else eval_interval    # 外部脚本对手间隔
    eval_games = cfg.eval.games               # 每次评估局数
    eval_seed = cfg.eval.seed
    log_every = int(getattr(cfg, "log_every", 1))
    checkpoint_every = getattr(cfg, "checkpoint_every", 100)

    # 冻结对手策略（权重 + 输入归一化统计量，每 opponent_update_freq 步同步一次）。
    opponent_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
    opponent_normalizer_state = jax.tree_util.tree_map(lambda x: x, normalizer_state)
    # lag_snapshot 保存“上一个 interval 起点的策略”；新 lag_checkpoint/PFSP 接口到下一个
    # interval 时才发布，因此 lag 对手是真正间隔 pool_lag_interval 的 checkpoint。
    lag_snapshot_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
    lag_snapshot_normalizer_state = jax.tree_util.tree_map(lambda x: x, normalizer_state)
    lagged_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
    lagged_normalizer_state = jax.tree_util.tree_map(lambda x: x, normalizer_state)
    snapshot_pool_entries: list[dict] = []  # snapshot pool 快照列表
    pool_entries: list[dict] = []
    if use_pool and pfsp_pool:
        pool_entries = [{"id": "selfplay-current", "type": "current", "params": None, "winrate": 0.5}]
        if pool_include_nearest:
            pool_entries.insert(0, {"id": "nearest", "type": "nearest", "params": None, "winrate": 0.5})

    def _copy_params(params):
        return jax.tree_util.tree_map(lambda x: x, params)

    def _place_replicated_runtime_tree(value):
        return _put_array_tree(_tree_to_jax(value), replicated_sharding)

    if resume_runtime_state is not None:
        if "opponent_params" in resume_runtime_state:
            opponent_params = _place_replicated_runtime_tree(resume_runtime_state["opponent_params"])
        if "opponent_normalizer_state" in resume_runtime_state:
            opponent_normalizer_state = _place_replicated_runtime_tree(resume_runtime_state["opponent_normalizer_state"])
        if "lag_snapshot_params" in resume_runtime_state:
            lag_snapshot_params = _place_replicated_runtime_tree(resume_runtime_state["lag_snapshot_params"])
        if "lag_snapshot_normalizer_state" in resume_runtime_state:
            lag_snapshot_normalizer_state = _place_replicated_runtime_tree(resume_runtime_state["lag_snapshot_normalizer_state"])
        if "lagged_params" in resume_runtime_state:
            lagged_params = _place_replicated_runtime_tree(resume_runtime_state["lagged_params"])
        if "lagged_normalizer_state" in resume_runtime_state:
            lagged_normalizer_state = _place_replicated_runtime_tree(resume_runtime_state["lagged_normalizer_state"])
        if "snapshot_pool_entries" in resume_runtime_state:
            snapshot_pool_entries = _place_replicated_runtime_tree(resume_runtime_state["snapshot_pool_entries"])
        if "pool_entries" in resume_runtime_state:
            pool_entries = _place_replicated_runtime_tree(resume_runtime_state["pool_entries"])
        if "pool_phase" in resume_runtime_state:
            pool_phase = int(resume_runtime_state["pool_phase"])
        if "recent_nearest_win_rates" in resume_runtime_state:
            recent_nearest_win_rates = list(resume_runtime_state["recent_nearest_win_rates"])
        if "pool_rng_state" in resume_runtime_state:
            pool_rng.setstate(resume_runtime_state["pool_rng_state"])
        print(
            f"[resume] restored runtime state: pool_phase={pool_phase} "
            f"pool_entries={len(pool_entries)} snapshots={len(snapshot_pool_entries)}"
        )
        missing_runtime_normalizer = any(
            params_key in resume_runtime_state and normalizer_key not in resume_runtime_state
            for params_key, normalizer_key in (
                ("opponent_params", "opponent_normalizer_state"),
                ("lag_snapshot_params", "lag_snapshot_normalizer_state"),
                ("lagged_params", "lagged_normalizer_state"),
            )
        )
        if missing_runtime_normalizer:
            print("[resume] warning: old checkpoint lacks frozen opponent normalizer state; using current normalizer fallback")
        restored_entries = list(snapshot_pool_entries) + list(pool_entries)
        if any(entry.get("type") == "lag" and "normalizer_state" not in entry for entry in restored_entries):
            print("[resume] warning: old lag pool entries do not contain frozen normalizer_state; using current normalizer fallback")
        if any("params" in entry and entry.get("params") is not None and "normalizer_state" not in entry for entry in snapshot_pool_entries):
            print("[resume] warning: old snapshot entries do not contain frozen normalizer_state; using current normalizer fallback")

    def _pfsp_sample(entries: list[dict]) -> dict:
        weights = [(1.0 - float(entry["winrate"]) + 0.1) ** 2 for entry in entries]
        return pool_rng.choices(entries, weights=weights, k=1)[0]

    def _pool_summary(entries: list[dict]) -> str:
        return ", ".join(f"{entry['id']}:{float(entry['winrate']):.2f}" for entry in entries)

    def _entry_normalizer_state(entry: dict, fallback):
        return entry.get("normalizer_state", fallback)

    def _ensure_openai_five_nearest_entry() -> None:
        if not (use_pool and openai_five_pool and openai_five_include_nearest):
            return
        if any(entry.get("type") == "nearest" or entry.get("id") == "nearest" for entry in snapshot_pool_entries):
            return
        snapshot_pool_entries.insert(0, {
            "id": "nearest",
            "type": "nearest",
            "quality": 0.0,
            "games": 0.0,
            "current_win_rate": 0.5,
        })

    def _openai_five_past_probs(entries: list[dict]) -> list[float]:
        if not entries:
            return []
        qualities = [float(entry.get("quality", 0.0)) for entry in entries]
        max_quality = max(qualities)
        weights = [
            math.exp((quality - max_quality) / openai_five_temperature)
            for quality in qualities
        ]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return [1.0 / len(entries)] * len(entries)
        return [weight / total for weight in weights]

    def _openai_five_sample_past(entries: list[dict]) -> tuple[dict, float]:
        probs = _openai_five_past_probs(entries)
        if not probs:
            raise ValueError("OpenAI Five opponent pool has no past opponents to sample")
        idx = pool_rng.choices(range(len(entries)), weights=probs, k=1)[0]
        return entries[idx], float(probs[idx])

    def _openai_five_pool_summary(entries: list[dict]) -> str:
        if not entries:
            return "empty"
        probs = _openai_five_past_probs(entries)
        items = zip(entries, probs, strict=True)
        ranked = sorted(items, key=lambda item: float(item[0].get("quality", 0.0)), reverse=True)
        return ", ".join(
            f"{entry['id']}:q={float(entry.get('quality', 0.0)):.3f}/p={prob:.3f}"
            for entry, prob in ranked[:6]
        )

    def _wandb_metric_name_part(value) -> str:
        raw = str(value)
        cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
        return cleaned or "unknown"

    def _openai_five_quality_wandb_metrics() -> dict[str, float]:
        if not (use_pool and openai_five_pool and snapshot_pool_entries):
            return {}
        log_dict = {}
        for entry in snapshot_pool_entries:
            opponent_id = _wandb_metric_name_part(entry.get("id", "unknown"))
            log_dict[f"pool_quality/{opponent_id}"] = float(entry.get("quality", 0.0))
        return log_dict

    def _add_openai_five_snapshot(update_idx: int) -> None:
        max_quality = max(
            (float(entry.get("quality", 0.0)) for entry in snapshot_pool_entries),
            default=0.0,
        )
        snapshot_pool_entries.append({
            "id": f"oai5-past-{update_idx:06d}",
            "params": _copy_params(train_state.params),
            "normalizer_state": _copy_params(normalizer_state),
            "quality": max_quality,
            "games": 0.0,
            "current_win_rate": 0.5,
        })
        print(
            f"[pool] update={update_idx} OpenAI Five 历史池加入当前快照，"
            f"past_size={len(snapshot_pool_entries)}，分布: "
            f"{_openai_five_pool_summary(snapshot_pool_entries)}"
        )
        _prune_openai_five_pool(update_idx)

    def _prune_openai_five_pool(update_idx: int) -> None:
        if openai_five_max_past_opponents <= 0:
            return
        while len(snapshot_pool_entries) > openai_five_max_past_opponents:
            victim = min(
                snapshot_pool_entries,
                key=lambda entry: (float(entry.get("quality", 0.0)), str(entry.get("id", ""))),
            )
            snapshot_pool_entries.remove(victim)
            print(
                f"[pool] update={update_idx} OpenAI Five 历史池超出 "
                f"{openai_five_max_past_opponents}，移除 {victim['id']} "
                f"(quality={float(victim.get('quality', 0.0)):.3f})"
            )

    def _update_openai_five_quality(entry: dict, selected_prob: float, metrics: dict) -> None:
        episodes = float(metrics.get("episodes_per_rollout", 0.0))
        if episodes <= 0.0 or selected_prob <= 0.0 or not snapshot_pool_entries:
            return
        current_win_rate = float(metrics["win_rate"])
        n = max(len(snapshot_pool_entries), 1)
        old_quality = float(entry.get("quality", 0.0))
        delta = openai_five_quality_lr * current_win_rate / max(n * selected_prob, 1.0e-8)
        entry["quality"] = old_quality - delta
        entry["games"] = float(entry.get("games", 0.0)) + episodes
        old_wr = float(entry.get("current_win_rate", 0.5))
        entry["current_win_rate"] = 0.95 * old_wr + 0.05 * current_win_rate
        metrics["pool_selected_quality"] = float(entry["quality"])
        metrics["pool_selected_probability"] = float(selected_prob)

    _ensure_openai_five_nearest_entry()
    _prune_openai_five_pool(start_update - 1)

    def _add_lag_to_pool(update_idx: int, params, normalizer) -> None:
        pool_entries.append({
            "id": f"selfplay-lag-{max(update_idx - pool_lag_interval, 0):06d}",
            "type": "lag",
            "params": _copy_params(params),
            "normalizer_state": _copy_params(normalizer),
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

    # JIT 编译的 autoregressive 训练步（JAX-native 对手），供 pool 模式按需调用。
    max_action_tokens = int(getattr(cfg.model, "max_action_tokens", 16))
    collect_in_shardings = None
    collect_out_shardings = _collect_rollout_out_shardings(data_sharding, rollout_sharding, replicated_sharding)
    if data_sharding is not None:
        collect_in_shardings = (
            replicated_sharding,  # train_state
            replicated_sharding,  # opponent params
            replicated_sharding,  # opponent normalizer
            data_sharding,        # EnvState [B, ...]
            replicated_sharding,  # normalizer_state
            replicated_sharding,  # rng
        )
    ppo_update_fn = make_sharded_ppo_update_auto(ppo_params, max_action_tokens, replicated_sharding, rollout_sharding)

    _jit_nearest = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout_auto(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, cfg.ppo.rollout_steps, "nearest",
            max_action_tokens=max_action_tokens,
        ),
        in_shardings=collect_in_shardings,
        out_shardings=collect_out_shardings,
    )
    _jit_selfplay = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout_auto(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, cfg.ppo.rollout_steps, "selfplay",
            max_action_tokens=max_action_tokens,
        ),
        in_shardings=collect_in_shardings,
        out_shardings=collect_out_shardings,
    )
    _jit_step = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout_auto(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, cfg.ppo.rollout_steps,
            "selfplay" if self_play else "nearest",
            max_action_tokens=max_action_tokens,
        ),
        in_shardings=collect_in_shardings,
        out_shardings=collect_out_shardings,
    )

    def _run_step(jit_fn, ts, opp_p, opp_ns, st, ns, key, ent_scale):
        """JIT 收集 rollout，然后在 Python 层执行 PPO 更新（含 KL 早停）。"""
        ts, st, ns, rollout, returns, advantages, ppo_rng, ep_metrics = jit_fn(ts, opp_p, opp_ns, st, ns, key)
        ts, ppo_metrics = ppo_update_fn(
            ts,
            rollout,
            returns,
            advantages,
            ppo_rng,
            ent_scale,
        )
        ts, st, ns = _place_train_runtime(ts, st, ns)
        return ts, st, ns, {**ppo_metrics, **ep_metrics}

    def _run_named_opponent(opponent_id: str, ts, st, ns, key, ent_scale):
        if opponent_id == "nearest":
            return _run_step(_jit_nearest, ts, opponent_params, opponent_normalizer_state, st, ns, key, ent_scale)
        if opponent_id == "selfplay-current":
            return _run_step(_jit_selfplay, ts, opponent_params, opponent_normalizer_state, st, ns, key, ent_scale)
        if opponent_id == "selfplay-lag":
            return _run_step(_jit_selfplay, ts, lagged_params, lagged_normalizer_state, st, ns, key, ent_scale)
        raise ValueError(f"Unknown training opponent: {opponent_id}")

    for update in range(start_update, cfg.ppo.total_updates + 1):
        selected_pool_entry = None
        selected_pool_id = ""
        selected_pool_probability = 0.0

        # 每隔 opponent_update_freq 步把当前权重同步给 self-play/current 对手
        if (use_pool and (current_pool or openai_five_pool)) or ((update - 1) % max(opponent_update_freq, 1) == 0):
            opponent_params = _copy_params(train_state.params)
            opponent_normalizer_state = _copy_params(normalizer_state)
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
                elif fixed_pool and pool_phase == 2:
                    candidates = fixed_phase2_opponents
                    selected_pool_id = pool_rng.choices(
                        candidates,
                        weights=[fixed_pool_current_ratio, 1.0 - fixed_pool_current_ratio],
                        k=1,
                    )[0]
                else:
                    candidates = fixed_phase1_opponents if pool_phase == 1 else fixed_phase2_opponents
                if not selected_pool_id:
                    selected_pool_id = pool_rng.choice(candidates)
                train_state, state, normalizer_state, metrics = _run_named_opponent(
                    selected_pool_id,
                    train_state,
                    state,
                    normalizer_state,
                    step_key,
                    ent_scale,
                )
            elif openai_five_pool:
                use_past = snapshot_pool_entries and pool_rng.random() >= openai_five_current_ratio
                if use_past:
                    selected_pool_entry, selected_pool_probability = _openai_five_sample_past(snapshot_pool_entries)
                    selected_pool_id = str(selected_pool_entry["id"])
                    if selected_pool_entry.get("type") == "nearest":
                        train_state, state, normalizer_state, metrics = _run_step(
                            _jit_nearest, train_state, opponent_params,
                            opponent_normalizer_state, state, normalizer_state, step_key, ent_scale)
                    else:
                        train_state, state, normalizer_state, metrics = _run_step(
                            _jit_selfplay, train_state, selected_pool_entry["params"],
                            _entry_normalizer_state(selected_pool_entry, normalizer_state),
                            state, normalizer_state, step_key, ent_scale)
                else:
                    selected_pool_id = "selfplay-current"
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, opponent_params,
                        opponent_normalizer_state, state, normalizer_state, step_key, ent_scale)
            elif snapshot_pool:
                # 20% 打当前策略，80% 打历史快照；若快照池为空则全打当前
                if snapshot_pool_entries and pool_rng.random() >= snapshot_current_ratio:
                    snap = pool_rng.choice(snapshot_pool_entries)
                    selected_pool_id = snap["id"]
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, snap["params"],
                        _entry_normalizer_state(snap, normalizer_state),
                        state, normalizer_state, step_key, ent_scale)
                else:
                    selected_pool_id = "selfplay-current"
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, opponent_params,
                        opponent_normalizer_state, state, normalizer_state, step_key, ent_scale)
            else:
                selected_pool_entry = _pfsp_sample(pool_entries)
                selected_pool_id = str(selected_pool_entry["id"])
                if selected_pool_entry["type"] == "nearest":
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_nearest, train_state, opponent_params,
                        opponent_normalizer_state, state, normalizer_state, step_key, ent_scale)
                elif selected_pool_entry["type"] == "current":
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, opponent_params,
                        opponent_normalizer_state, state, normalizer_state, step_key, ent_scale)
                else:
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, selected_pool_entry["params"],
                        _entry_normalizer_state(selected_pool_entry, normalizer_state),
                        state, normalizer_state, step_key, ent_scale)
        else:
            if self_play and update % opponent_update_freq == 1:
                pass  # 上面已经同步过 current 对手。
            train_state, state, normalizer_state, metrics = _run_step(
                _jit_step, train_state, opponent_params, opponent_normalizer_state, state, normalizer_state, step_key, ent_scale)
        metrics = jax.device_get(metrics)
        current_lr = _current_lr(cfg, int(jax.device_get(train_state.step)), total_optimizer_steps, warmup_steps)
        metrics["lr"] = current_lr
        metrics["ent_scale"] = ent_scale
        if use_pool and pfsp_pool and selected_pool_entry is not None and float(metrics["episodes_per_rollout"]) > 0.0:
            old_wr = float(selected_pool_entry["winrate"])
            selected_pool_entry["winrate"] = 0.95 * old_wr + 0.05 * float(metrics["win_rate"])
            metrics["pool_selected_winrate"] = selected_pool_entry["winrate"]
        if use_pool and openai_five_pool and selected_pool_entry is not None:
            _update_openai_five_quality(selected_pool_entry, selected_pool_probability, metrics)
        elapsed = time.perf_counter() - start
        samples = cfg.ppo.num_envs * cfg.ppo.rollout_steps
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
            elif openai_five_pool:
                line += f" opp={selected_pool_id}"
                if selected_pool_entry is not None:
                    line += (
                        f" opp_q={float(selected_pool_entry.get('quality', 0.0)):.3f}"
                        f" opp_p={selected_pool_probability:.3f}"
                    )
            else:
                line += f" opp={selected_pool_id}"
        line += (
            f" invalid={metrics.get('invalid_action_rate', 0.0):.3f}"
            f" exec={metrics.get('executed_action_rate', 0.0):.3f}"
            f" seq={metrics.get('seq_len_mean', 0.0):.2f}"
            f" logr_heads="
            f"{metrics.get('max_abs_log_ratio_end', 0.0):.1f}/"
            f"{metrics.get('max_abs_log_ratio_src', 0.0):.1f}/"
            f"{metrics.get('max_abs_log_ratio_tgt', 0.0):.1f}/"
            f"{metrics.get('max_abs_log_ratio_amt', 0.0):.1f}"
        )
        if update % log_every == 0:
            print(line, flush=True)

        # snapshot pool：每 snapshot_interval 步存一个快照
        if use_pool and snapshot_pool and snapshot_interval > 0 and update % snapshot_interval == 0:
            snap_id = f"snap-{update:06d}"
            snapshot_pool_entries.append({
                "id": snap_id,
                "params": _copy_params(train_state.params),
                "normalizer_state": _copy_params(normalizer_state),
            })
            print(f"[pool] update={update} 存入快照 {snap_id}，快照池大小={len(snapshot_pool_entries)}")

        if (
            use_pool
            and openai_five_pool
            and openai_five_snapshot_interval > 0
            and update % openai_five_snapshot_interval == 0
        ):
            _add_openai_five_snapshot(update)

        if use_pool and lagged_pool and pool_lag_interval > 0 and update % pool_lag_interval == 0:
            if fixed_pool:
                lagged_params = _copy_params(lag_snapshot_params)
                lagged_normalizer_state = _copy_params(lag_snapshot_normalizer_state)
                lag_snapshot_params = _copy_params(train_state.params)
                lag_snapshot_normalizer_state = _copy_params(normalizer_state)
                print(f"[pool] update={update} 发布 fixed_pool lagged 对手权重 (lag={pool_lag_interval})")
            elif lag_checkpoint_pool:
                lagged_params = _copy_params(lag_snapshot_params)
                lagged_normalizer_state = _copy_params(lag_snapshot_normalizer_state)
                lag_snapshot_params = _copy_params(train_state.params)
                lag_snapshot_normalizer_state = _copy_params(normalizer_state)
                print(f"[pool] update={update} 发布 lag_checkpoint 对手权重 (lag={pool_lag_interval})")
            elif pfsp_pool:
                _add_lag_to_pool(update, lag_snapshot_params, lag_snapshot_normalizer_state)
                lag_snapshot_params = _copy_params(train_state.params)
                lag_snapshot_normalizer_state = _copy_params(normalizer_state)

        eval_metrics = {field: "" for field in eval_metric_fields}
        _do_nearest = eval_enabled and eval_nearest_interval > 0 and update % eval_nearest_interval == 0
        _do_slow   = eval_enabled and eval_slow_interval   > 0 and update % eval_slow_interval   == 0
        if _do_nearest or _do_slow:
            eval_player = 1 if int(getattr(cfg.eval, "player", 0)) == 1 else 0
            print(f"[eval] update={update} 开始评估 ({eval_games} 局, learner=player{eval_player})...")
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
                nearest_opponent_actions=nearest_opponent_actions,
                reset_from_bank=reset_from_bank,
                flip_state=flip_state,
                flip_actions=flip_actions,
                kaggle_obs_to_jax_state=kaggle_obs_to_jax_state,
                jax_actions_to_kaggle=jax_actions_to_kaggle,
                eval_player=eval_player,
            )
            eval_results: dict[str, dict] = {}
            for eval_opponent in eval_opponents:
                if not eval_opponent.exists():
                    continue
                _is_nearest = eval_opponent.metric_prefix == "nearest"
                if _is_nearest and not _do_nearest:
                    continue
                if not _is_nearest and not _do_slow:
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
                    elif openai_five_pool:
                        fixed_size = len(snapshot_pool_entries) + 1
                    else:
                        fixed_size = len(pool_entries)
                    if phased_pool:
                        log_dict["pool/phase"] = float(pool_phase)
                    log_dict["pool/size"] = float(len(pool_entries) if pfsp_pool else fixed_size)
                    if openai_five_pool:
                        log_dict["pool/past_size"] = float(len(snapshot_pool_entries))
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
                f"{metrics.get('entropy_end', 0.0):.8f}",
                f"{metrics.get('entropy_src', 0.0):.8f}",
                f"{metrics.get('entropy_tgt', 0.0):.8f}",
                f"{metrics.get('entropy_amt', 0.0):.8f}",
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
                "entropy/scale": float(metrics.get("ent_scale", 1.0)),
                "losses/approx_kl": float(metrics["approx_kl"]),
                "losses/clipfrac": float(metrics["clipfrac"]),
                "losses/max_abs_log_ratio": float(metrics.get("max_abs_log_ratio", 0.0)),
                "losses/n_epochs_run": float(metrics.get("n_epochs_run", cfg.ppo.epochs)),
                "lr/current": float(metrics.get("lr", cfg.ppo.lr)),
                "charts/sps": float(sps),
                "charts/seq_len_mean": float(metrics.get("seq_len_mean", 0.0)),
                # 自回归 actor 专属指标
                **({
                    "entropy/end": float(metrics.get("entropy_end", 0.0)),
                    "entropy/src": float(metrics.get("entropy_src", 0.0)),
                    "entropy/tgt": float(metrics.get("entropy_tgt", 0.0)),
                    "entropy/amt": float(metrics.get("entropy_amt", 0.0)),
                    "losses/max_abs_log_ratio_end": float(metrics.get("max_abs_log_ratio_end", 0.0)),
                    "losses/max_abs_log_ratio_src": float(metrics.get("max_abs_log_ratio_src", 0.0)),
                    "losses/max_abs_log_ratio_tgt": float(metrics.get("max_abs_log_ratio_tgt", 0.0)),
                    "losses/max_abs_log_ratio_amt": float(metrics.get("max_abs_log_ratio_amt", 0.0)),
                    "autoregressive/invalid_action_rate": float(metrics.get("invalid_action_rate", 0.0)),
                    "autoregressive/executed_action_rate": float(metrics.get("executed_action_rate", 0.0)),
                    "autoregressive/seq_len_mean": float(metrics.get("seq_len_mean", 0.0)),
                }),
            }, step=update)
            if use_pool:
                if current_pool:
                    fixed_size = len(current_opponents)
                elif lag_checkpoint_pool:
                    fixed_size = len(lag_checkpoint_phase1_opponents if pool_phase == 1 else lag_checkpoint_phase2_opponents)
                elif fixed_pool:
                    fixed_size = len(fixed_phase1_opponents if pool_phase == 1 else fixed_phase2_opponents)
                elif openai_five_pool:
                    fixed_size = len(snapshot_pool_entries) + 1
                else:
                    fixed_size = len(pool_entries)
                pool_log = {"pool/size": float(len(pool_entries) if pfsp_pool else fixed_size)}
                if pfsp_pool:
                    selected_wr = float(selected_pool_entry["winrate"]) if selected_pool_entry is not None else 0.0
                    pool_log["pool/selected_win_rate"] = selected_wr
                if openai_five_pool:
                    pool_log["pool/past_size"] = float(len(snapshot_pool_entries))
                    pool_log["pool/current_ratio"] = float(openai_five_current_ratio)
                    pool_log.update(_openai_five_quality_wandb_metrics())
                _wandb.log(pool_log, step=update)

        # ── 定期保存 checkpoint ───────────────────────────────────
        if update % checkpoint_every == 0:
            ckpt_path = run_dir / f"ckpt_{update:06d}.pkl"
            params_host = jax.device_get(train_state.params)
            normalizer_host = jax.device_get(normalizer_state)
            training_state = {
                "rng": jax.device_get(rng),
                "env_state": jax.device_get(state),
                "opponent_params": jax.device_get(opponent_params),
                "opponent_normalizer_state": jax.device_get(opponent_normalizer_state),
                "lag_snapshot_params": jax.device_get(lag_snapshot_params),
                "lag_snapshot_normalizer_state": jax.device_get(lag_snapshot_normalizer_state),
                "lagged_params": jax.device_get(lagged_params),
                "lagged_normalizer_state": jax.device_get(lagged_normalizer_state),
                "snapshot_pool_entries": jax.device_get(snapshot_pool_entries),
                "pool_entries": jax.device_get(pool_entries),
                "pool_phase": int(pool_phase),
                "recent_nearest_win_rates": list(recent_nearest_win_rates),
                "pool_rng_state": pool_rng.getstate(),
            }
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "checkpoint_version": 3,
                    # Legacy fields are kept for eval/export scripts and old init workflows.
                    "params": params_host,
                    "normalizer_state": normalizer_host,
                    "train_state": _checkpoint_train_state_payload(train_state),
                    "training_state": training_state,
                    "update": update,
                    "cfg": cfg,
                }, f)
            print(f"[ckpt] saved {ckpt_path}")


def reset_from_bank(bank: EnvState, rng, num_envs: int) -> EnvState:
    """从预建 state bank 中随机采样 num_envs 个初始状态。"""
    size = bank.planets.shape[0]
    idx = jax.random.randint(rng, (num_envs,), 0, size)
    return select_from_bank(bank, idx)


def collect_rollout_auto(
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
    rollout_steps: int,
    opponent_kind: str = "nearest",
    max_action_tokens: int = 16,
):
    """收集状态级自回归 rollout。

    支持 nearest 规则对手，以及自回归 self-play / pool 对手。
    """
    rng, rollout_key = jax.random.split(rng)

    def body(carry, _):
        ts, st, ns, key = carry
        key, action_key, opp_key, reset_key = jax.random.split(key, 4)
        raw_encoded = encode_batch(st, feat_params)
        encoded, ns = update_and_normalize(raw_encoded, ns, norm_params)
        sample = auto_policy_sample(ts, model, st, encoded, action_key, feat_params, max_action_tokens)
        learner_actions = sample["actions"]
        if opponent_kind == "selfplay":
            opponent_actions = auto_opponent_policy_actions(
                st,
                opponent_params,
                model,
                feat_params,
                opponent_normalizer_state,
                norm_params,
                max_action_tokens,
                flip_state_fn=flip_state,
                rng=opp_key,
            )
        else:
            opponent_actions = nearest_opponent_actions(st, player=1, params=feat_params)
        joint = jnp.stack([learner_actions, opponent_actions], axis=1)
        next_state = step_batch(st, joint, env_params)
        reward_env, _reward_parts = reward_from_transition(st, next_state, learner_actions, reward_params)
        done = next_state.done
        reset_state = reset_from_bank(bank, reset_key, st.planets.shape[0])
        next_state = replace_done(next_state, reset_state)
        row = AutoRollout(
            encoded.planet_features,
            encoded.relative_features,
            encoded.global_features,
            encoded.map_features,
            encoded.fleet_features[:, 0],
            encoded.fleet_mask[:, 0],
            encoded.candidate_mask,
            encoded.amount_mask,
            encoded.source_mask,
            sample["initial_ships"],
            sample["end_action"],
            sample["src_action"],
            sample["tgt_action"],
            sample["amt_action"],
            sample["executed"],
            sample["invalid"],
            sample["old_logp_end"],
            sample["old_logp_src"],
            sample["old_logp_tgt"],
            sample["old_logp_amt"],
            sample["token_valid"],
            sample["value"],
            reward_env,
            done,
        )
        return (ts, next_state, ns, key), row

    (train_state, state, normalizer_state, rng), rollout = jax.lax.scan(
        body,
        (train_state, state, normalizer_state, rollout_key),
        None,
        length=rollout_steps,
    )
    last_raw_encoded = encode_batch(state, feat_params)
    last_encoded = normalize_encoded(last_raw_encoded, normalizer_state, norm_params)
    b = last_encoded.planet_features.shape[0]
    k = max_action_tokens
    last_initial_ships = jnp.concatenate([jnp.zeros((b, 1), dtype=last_encoded.source_ships.dtype), last_encoded.source_ships], axis=1)
    last_out = train_state.apply_fn(
        {"params": train_state.params},
        last_encoded.planet_features,
        last_encoded.relative_features,
        last_encoded.global_features,
        last_encoded.candidate_mask,
        last_encoded.amount_mask,
        last_encoded.source_mask,
        jnp.zeros((b, k), dtype=bool),
        jnp.zeros((b, k), dtype=jnp.int32),
        jnp.zeros((b, k), dtype=jnp.int32),
        jnp.zeros((b, k), dtype=jnp.int32),
        jnp.zeros((b, k), dtype=bool),
        last_initial_ships,
        map_features=last_encoded.map_features,
        fleet_features=last_encoded.fleet_features,
        fleet_mask=last_encoded.fleet_mask,
    )
    returns, advantages = compute_gae_state(rollout.reward, rollout.done, rollout.old_value, last_out.value, ppo_params)
    finished = rollout.done
    ep_reward = rollout.reward
    ep_reward_sum = jnp.where(finished, ep_reward, 0.0).sum()
    episodes = jnp.maximum(finished.sum(), 1)
    episode_reward = ep_reward_sum / episodes
    wins = ((ep_reward > 0.0) & finished).sum()
    win_rate = wins / episodes
    ep_metrics = {
        "episode_reward": episode_reward,
        "win_rate": win_rate,
        "mean_return": returns.mean(),
        "episodes_per_rollout": finished.sum().astype(jnp.float32),
    }
    return train_state, state, normalizer_state, rollout, returns, advantages, rng, ep_metrics


def _bernoulli_log_prob_local(logits, action_bool):
    """rollout 采样阶段使用的 Bernoulli log-prob。"""
    action = action_bool.astype(jnp.float32)
    return -jax.nn.softplus(jnp.where(action_bool, -logits, logits))


def _categorical_log_prob_local(logits, action):
    """rollout 采样阶段使用的离散动作 log-prob。"""
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1).squeeze(-1)


def dynamic_ship_buckets(ships, bucket_count: int):
    """根据当前剩余兵力动态生成派兵档位。"""
    ratios = jnp.asarray([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=jnp.float32)[:bucket_count]
    s = jnp.floor(ships)
    raw = jnp.clip(jnp.round(s[:, None] * ratios[None, :]), 10.0, s[:, None])
    return jnp.where(s[:, None] >= 10.0, raw, jnp.zeros_like(raw))


def auto_policy_sample(train_state, model, state, encoded, rng, feat_params: FeatureParams, max_action_tokens: int):
    """自回归 rollout 采样，并在采样后做物理层校验。"""
    b = encoded.planet_features.shape[0]
    k = max_action_tokens
    initial_ships = jnp.concatenate([jnp.zeros((b, 1), dtype=encoded.source_ships.dtype), encoded.source_ships], axis=1)
    end_action = jnp.zeros((b, k), dtype=bool)
    src_action = jnp.zeros((b, k), dtype=jnp.int32)
    tgt_action = jnp.zeros((b, k), dtype=jnp.int32)
    amt_action = jnp.zeros((b, k), dtype=jnp.int32)
    executed = jnp.zeros((b, k), dtype=bool)
    invalid = jnp.zeros((b, k), dtype=bool)
    token_valid = jnp.zeros((b, k), dtype=bool)
    old_logp_end = jnp.zeros((b, k), dtype=jnp.float32)
    old_logp_src = jnp.zeros((b, k), dtype=jnp.float32)
    old_logp_tgt = jnp.zeros((b, k), dtype=jnp.float32)
    old_logp_amt = jnp.zeros((b, k), dtype=jnp.float32)
    actions = jnp.zeros((b, feat_params.max_planets, 3), dtype=jnp.float32)
    ended = jnp.zeros((b,), dtype=bool)
    state_emb, h, prev_tok, remaining, planet_out, relative_h, key_proj, src_key_proj, value = train_state.apply_fn(
        {"params": train_state.params},
        encoded.planet_features,
        encoded.relative_features,
        encoded.global_features,
        encoded.candidate_mask,
        initial_ships,
        map_features=encoded.map_features,
        fleet_features=encoded.fleet_features,
        fleet_mask=encoded.fleet_mask,
        method=model.auto_start,
    )
    batch_idx = jnp.arange(b)

    def body(t, carry):
        (key, ended, h, prev_tok, remaining, end_action, src_action, tgt_action, amt_action, executed, invalid,
         token_valid, old_logp_end, old_logp_src, old_logp_tgt, old_logp_amt, actions) = carry
        key, key_end, key_src, key_tgt, key_amt = jax.random.split(key, 5)
        h_new, end_logits_t, src_logits_t = train_state.apply_fn(
            {"params": train_state.params},
            state_emb,
            h,
            prev_tok,
            remaining,
            encoded.relative_features,
            encoded.source_mask,
            src_key_proj,
            method=model.auto_step_base,
        )
        tv = ~ended
        sampled_end = jax.random.uniform(key_end, (b,)) < jax.nn.sigmoid(end_logits_t)
        end_t = tv & sampled_end
        src_logits_t = safe_logits(src_logits_t)
        src_t = jax.random.categorical(key_src, src_logits_t, axis=-1)

        src_for_heads = jnp.where(tv & (~end_t), src_t, 0)
        tgt_logits_t = train_state.apply_fn(
            {"params": train_state.params},
            h_new,
            src_for_heads,
            planet_out,
            relative_h,
            key_proj,
            encoded.candidate_mask,
            method=model.auto_tgt_logits,
        )
        tgt_logits_t = safe_logits(tgt_logits_t)
        tgt_t = jax.random.categorical(key_tgt, tgt_logits_t, axis=-1)

        tgt_for_heads = jnp.where(tv & (~end_t), tgt_t, 0)
        amt_logits_t = train_state.apply_fn(
            {"params": train_state.params},
            h_new,
            src_for_heads,
            tgt_for_heads,
            planet_out,
            relative_h,
            remaining,
            encoded.amount_mask,
            method=model.auto_amt_logits,
        )
        amt_logits_t = safe_logits(amt_logits_t)
        amt_t = jax.random.categorical(key_amt, amt_logits_t, axis=-1)

        src_slot = jnp.clip(src_t - 1, 0, feat_params.max_planets - 1)
        tgt_slot = jnp.clip(tgt_t - 1, 0, feat_params.max_planets - 1)
        source_buckets = dynamic_ship_buckets(remaining[batch_idx, jnp.clip(src_t, 0, feat_params.max_planets)], feat_params.ship_bucket_count)
        ships = jnp.take_along_axis(source_buckets, amt_t[:, None], axis=-1).squeeze(-1)
        src_planets = state.planets[batch_idx, src_slot]
        tgt_planets = state.planets[batch_idx, tgt_slot]
        angle, aim_hit = aim_with_prediction_jax(src_planets[:, None, :], tgt_planets[:, None, :], tgt_slot[:, None], state, ships[:, None])
        angle = angle[:, 0]
        aim_hit = aim_hit[:, 0]
        src_valid = (src_t > 0) & encoded.source_mask[batch_idx, src_slot]
        tgt_valid = tgt_t > 0
        exec_t = tv & (~end_t) & src_valid & tgt_valid & aim_hit
        invalid_t = tv & (~end_t) & (~exec_t)
        source_ids = encoded.source_ids[batch_idx, src_slot].astype(jnp.float32)
        action_t = jnp.stack(
            [
                jnp.where(exec_t, source_ids, 0.0),
                jnp.where(exec_t, angle, 0.0),
                jnp.where(exec_t, ships, 0.0),
            ],
            axis=-1,
        )
        actions = actions.at[:, t, :].set(action_t)
        remaining = remaining.at[batch_idx, jnp.clip(src_t, 0, feat_params.max_planets)].add(jnp.where(exec_t, -ships, 0.0))
        prev_tok = train_state.apply_fn(
            {"params": train_state.params},
            src_for_heads,
            tgt_for_heads,
            jnp.where(tv & (~end_t), amt_t, 0),
            end_t,
            exec_t,
            t,
            planet_out,
            method=model.auto_token_embed,
        )
        end_action = end_action.at[:, t].set(end_t)
        src_action = src_action.at[:, t].set(src_for_heads)
        tgt_action = tgt_action.at[:, t].set(tgt_for_heads)
        amt_action = amt_action.at[:, t].set(jnp.where(tv & (~end_t), amt_t, 0))
        executed = executed.at[:, t].set(exec_t)
        invalid = invalid.at[:, t].set(invalid_t)
        token_valid = token_valid.at[:, t].set(tv)
        old_logp_end = old_logp_end.at[:, t].set(jnp.where(tv, _bernoulli_log_prob_local(end_logits_t, end_t), 0.0))
        old_logp_src = old_logp_src.at[:, t].set(jnp.where(exec_t, _categorical_log_prob_local(src_logits_t, src_t), 0.0))
        old_logp_tgt = old_logp_tgt.at[:, t].set(jnp.where(exec_t, _categorical_log_prob_local(tgt_logits_t, tgt_t), 0.0))
        old_logp_amt = old_logp_amt.at[:, t].set(jnp.where(exec_t, _categorical_log_prob_local(amt_logits_t, amt_t), 0.0))
        ended = ended | end_t
        return (key, ended, h_new, prev_tok, remaining, end_action, src_action, tgt_action, amt_action, executed, invalid,
                token_valid, old_logp_end, old_logp_src, old_logp_tgt, old_logp_amt, actions)

    init = (rng, ended, h, prev_tok, remaining, end_action, src_action, tgt_action, amt_action, executed, invalid,
            token_valid, old_logp_end, old_logp_src, old_logp_tgt, old_logp_amt, actions)
    (_, _ended, _h, _prev_tok, _remaining, end_action, src_action, tgt_action, amt_action, executed, invalid,
     token_valid, old_logp_end, old_logp_src, old_logp_tgt, old_logp_amt, actions) = jax.lax.fori_loop(0, k, body, init)
    return {
        "actions": actions,
        "initial_ships": initial_ships,
        "end_action": end_action,
        "src_action": src_action,
        "tgt_action": tgt_action,
        "amt_action": amt_action,
        "executed": executed,
        "invalid": invalid,
        "token_valid": token_valid,
        "old_logp_end": old_logp_end,
        "old_logp_src": old_logp_src,
        "old_logp_tgt": old_logp_tgt,
        "old_logp_amt": old_logp_amt,
        "value": value,
    }


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
    """局面结束时，用 reset 状态替换已结束的环境。"""
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
