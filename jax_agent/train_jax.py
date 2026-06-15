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
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .config import load_config  # noqa: E402
from .state import parse_observation  # noqa: E402

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
    _WANDB_IMPORT_ERROR = None
except ImportError as exc:
    _WANDB_AVAILABLE = False
    _WANDB_IMPORT_ERROR = exc

from .build_initial_state_bank import build_bank  # noqa: E402
from .eval_opponents import EvalContext, ExternalScriptOpponent, default_eval_opponents, eval_metric_fields as build_eval_metric_fields  # noqa: E402
from .fast_env import (  # noqa: E402
    EnvParams,
    EnvState,
    comet_arrays_from_observation,
    load_state_bank,
    scores,
    select_from_bank,
    step_batch,
)
from .features_jax import (  # noqa: E402
    ATTACK_REACHABILITY_HORIZON,
    FLEET_DIM,
    FeatureAux,
    FeatureParams,
    encode_batch,
    encode_batch_with_aux,
    planet_positions_over_horizon,
    reachable_full_send_eta_jax,
)
from .garrison_jax import DEFAULT_GARRISON_HORIZON, garrison_status_jax  # noqa: E402
from .model_jax import AutoregressiveActorCritic  # noqa: E402
from .normalization_jax import NormalizerParams, init_normalizer, normalize_encoded, update_and_normalize  # noqa: E402
from .physics import aim_with_prediction_eta_jax  # noqa: E402
from .pressure_jax import pressure_by_owner_jax  # noqa: E402
from .ppo_jax import (
    AutoRollout,
    PPOParams,
    compute_gae,
    compute_gae_state,
    make_auto_train_state,
    ppo_update_auto,
    safe_logits,
)  # noqa: E402
from .reward_jax import RewardParams, reward_from_transition  # noqa: E402
from .train_opponents import nearest_opponent_actions  # noqa: E402

NEG_INF = -1.0e9
MIN_LAUNCH_SHIPS = 10.0


def _stack_auto_rollout_rows(rows_by_time: list[list[AutoRollout]]) -> AutoRollout:
    """Stack Python-collected single-env rollout rows into [T, B, ...]."""
    if not rows_by_time or not rows_by_time[0]:
        raise ValueError("external rollout produced no rows")
    stacked = {}
    for field in AutoRollout._fields:
        per_t = []
        for rows_at_t in rows_by_time:
            per_t.append(jnp.stack([getattr(row, field)[0] for row in rows_at_t], axis=0))
        stacked[field] = jnp.stack(per_t, axis=0)
    return AutoRollout(**stacked)


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


def _flip_owner_int(owner):
    return jnp.where(owner == 0, 1, jnp.where(owner == 1, 0, owner)).astype(owner.dtype)


def flip_garrison_status(status, planet_count: int):
    """Flip a GarrisonStatus into the same canonical view as ``flip_state``."""
    planet_perm = _mirror_planet_indices(planet_count)
    arrivals = status.arrivals_by_owner[..., planet_perm, :, :]
    return status._replace(
        owner=_flip_owner_int(status.owner[..., planet_perm, :]),
        ships=status.ships[..., planet_perm, :],
        pre_combat_owner=_flip_owner_int(status.pre_combat_owner[..., planet_perm, :]),
        pre_combat_ships=status.pre_combat_ships[..., planet_perm, :],
        arrivals_by_owner=jnp.flip(arrivals, axis=-1),
        alive=status.alive[..., planet_perm, :],
    )


def shared_selfplay_feature_aux(state: EnvState) -> tuple[FeatureAux, FeatureAux]:
    """Build learner/opponent feature aux once for a self-play rollout step."""
    status = garrison_status_jax(state, horizon=DEFAULT_GARRISON_HORIZON)
    pressure_by_owner = pressure_by_owner_jax(state, horizon=float(DEFAULT_GARRISON_HORIZON))
    planet_perm = _mirror_planet_indices(state.planets.shape[-2])
    learner_aux = FeatureAux(
        garrison_status=status,
        enemy_pressure=pressure_by_owner[..., 1, :],
    )
    opponent_aux = FeatureAux(
        garrison_status=flip_garrison_status(status, state.planets.shape[-2]),
        enemy_pressure=pressure_by_owner[..., 0, planet_perm],
    )
    return learner_aux, opponent_aux


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


def full_send_amount(ships: jnp.ndarray) -> jnp.ndarray:
    """Full-send branch: launch all currently remaining ships from a selected source."""
    s = jnp.floor(ships)
    return jnp.where(s >= MIN_LAUNCH_SHIPS, s, 0.0)


def has_valid_logits(logits: jnp.ndarray) -> jnp.ndarray:
    """Whether a masked categorical row still has at least one real action."""
    return (jnp.isfinite(logits) & (logits > NEG_INF * 0.5)).any(axis=-1)


def _valid_logit_mask(logits: jnp.ndarray) -> jnp.ndarray:
    return jnp.isfinite(logits) & (logits > NEG_INF * 0.5)


def selected_coarse_target_mask(encoded, src_t: jnp.ndarray, feat_params: FeatureParams) -> jnp.ndarray:
    """Return the coarse candidate mask for a selected source.

    This is intentionally only the feature-stage target mask. Exact physics
    validation and tactical capture feasibility are handled after target
    sampling so that the target head is not forced onto the next-best target.
    """
    b = encoded.planet_features.shape[0]
    batch_idx = jnp.arange(b)
    src_slot = jnp.clip(src_t - 1, 0, feat_params.max_planets - 1)
    target_mask = encoded.candidate_mask[batch_idx, src_slot].astype(bool)
    target_mask = target_mask & (jnp.arange(target_mask.shape[-1])[None, :] != 0)
    return jnp.where((src_t > 0)[:, None], target_mask, jnp.zeros_like(target_mask))


def select_source_with_coarse_targets(
    src_logits: jnp.ndarray,
    key_src,
    active: jnp.ndarray,
    encoded,
    feat_params: FeatureParams,
    *,
    deterministic: bool = False,
):
    """Pick a source using only source legality and coarse target availability."""
    source_mask = _valid_logit_mask(src_logits)
    found = active & source_mask.any(axis=-1)
    safe_src_logits = safe_logits(src_logits)
    if deterministic:
        src_raw = jnp.argmax(safe_src_logits, axis=-1).astype(jnp.int32)
    else:
        src_raw = jax.random.categorical(key_src, safe_src_logits, axis=-1).astype(jnp.int32)
    src_t = jnp.where(found, src_raw, 0)
    target_mask = selected_coarse_target_mask(encoded, src_t, feat_params)
    return src_t, safe_src_logits, source_mask, target_mask, found


def post_discard_target_check(
    state: EnvState,
    src_t: jnp.ndarray,
    tgt_t: jnp.ndarray,
    ships: jnp.ndarray,
    feat_params: FeatureParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Validate selected src/tgt after sampling and return (execute_ok, angle).

    Exact hit validation is always post-sampling. Optional capture feasibility
    additionally rejects non-friendly targets where full-send is unlikely to
    beat current defenders plus enemy production through the exact ETA.
    """
    b = state.planets.shape[0]
    batch_idx = jnp.arange(b)
    src_slot = jnp.clip(src_t - 1, 0, feat_params.max_planets - 1)
    tgt_slot = jnp.clip(tgt_t - 1, 0, feat_params.max_planets - 1)
    src_planets = state.planets[batch_idx, src_slot]
    tgt_planets = state.planets[batch_idx, tgt_slot]
    angle, hit, eta_turns = aim_with_prediction_eta_jax(
        src_planets[:, None, :],
        tgt_planets[:, None, :],
        tgt_slot[:, None],
        state,
        ships[:, None],
    )
    exact_hit = hit[:, 0]
    target_owner = tgt_planets[:, 1].astype(jnp.int32)
    enemy_target = target_owner == 1
    non_friendly = target_owner != 0
    enemy_growth = jnp.where(enemy_target, eta_turns[:, 0] * tgt_planets[:, 6], 0.0)
    required = tgt_planets[:, 5] + enemy_growth
    feasible_attack = ships > required
    capture_ok = jnp.where(
        bool(feat_params.post_discard_capture_check) & non_friendly,
        feasible_attack,
        jnp.ones_like(exact_hit, dtype=bool),
    )

    def one_game(st, src_idx, tgt_idx, amount):
        future_px, future_py = planet_positions_over_horizon(st, ATTACK_REACHABILITY_HORIZON)
        src_planet = st.planets[src_idx]
        reachable, _eta = reachable_full_send_eta_jax(src_planet, st.planets, future_px, future_py, amount)
        return reachable[tgt_idx]

    cheap_reachable = jax.vmap(one_game)(state, src_slot, tgt_slot, ships)
    return exact_hit & capture_ok & cheap_reachable, angle[:, 0]


def selected_target_is_valid_from_mask(target_mask: jnp.ndarray, batch_idx, tgt_t) -> jnp.ndarray:
    tgt_safe = jnp.clip(tgt_t, 0, target_mask.shape[-1] - 1)
    return (tgt_t > 0) & target_mask[batch_idx, tgt_safe]


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
        no_valid_source = ~has_valid_logits(src_logits_t)
        end_t = tv & ((jax.nn.sigmoid(end_logits_t) > 0.5) | no_valid_source)
        src_t, _src_logits_final, _source_action_mask, target_action_mask, found_src = select_source_with_coarse_targets(
            src_logits_t,
            jax.random.PRNGKey(0),
            tv & (~end_t),
            encoded,
            feat_params,
            deterministic=True,
        )
        end_t = end_t | (tv & (~end_t) & (~found_src))
        src_for_heads = jnp.where(tv & (~end_t), src_t, 0)
        tgt_logits_t = train_state.apply_fn(
            {"params": train_state.params},
            h_new, src_for_heads, planet_out, relative_h, key_proj,
            target_action_mask,
            method=model.auto_tgt_logits_with_mask,
        )
        tgt_t = jnp.argmax(safe_logits(tgt_logits_t), axis=-1)

        src_slot = jnp.clip(src_t - 1, 0, feat_params.max_planets - 1)
        tgt_slot = jnp.clip(tgt_t - 1, 0, feat_params.max_planets - 1)
        ships = full_send_amount(remaining[batch_idx, jnp.clip(src_t, 0, feat_params.max_planets)])
        src_valid = (src_t > 0) & encoded.source_mask[batch_idx, src_slot] & (ships >= MIN_LAUNCH_SHIPS)
        tgt_valid = selected_target_is_valid_from_mask(target_action_mask, batch_idx, tgt_t)
        post_ok, angle = post_discard_target_check(state, src_t, tgt_t, ships, feat_params)
        exec_t = tv & (~end_t) & src_valid & tgt_valid & post_ok

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
            src_for_heads, jnp.where(tv & (~end_t), tgt_t, 0),
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
    feature_aux=None,
):
    """自博弈训练用的冻结参数对手。

    它先把棋盘翻转到 player 1 视角，再用冻结的 opponent_params 生成动作。
    与 learner 的 ``auto_policy_sample`` 保持一致：end / src / tgt
    三个 head 都按当前 logits 采样（categorical / bernoulli），不再 argmax，
    用于训练时给学习者提供有探索的对手。评估路径仍使用
    ``auto_deterministic_policy_actions`` 走 argmax。
    """
    flipped = flip_state_fn(state)
    raw_enc = encode_batch_with_aux(flipped, opp_fp, feature_aux) if feature_aux is not None else encode_batch(flipped, opp_fp)
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
        rng, key_end, key_src, key_tgt = jax.random.split(rng, 4)
        tv = ~ended
        sampled_end = jax.random.uniform(key_end, (b,)) < jax.nn.sigmoid(end_logits_t)
        no_valid_source = ~has_valid_logits(src_logits_t)
        end_t = tv & (sampled_end | no_valid_source)
        src_t, _src_logits_final, _source_action_mask, target_action_mask, found_src = select_source_with_coarse_targets(
            src_logits_t,
            key_src,
            tv & (~end_t),
            encoded,
            opp_fp,
            deterministic=False,
        )
        end_t = end_t | (tv & (~end_t) & (~found_src))
        src_for_heads = jnp.where(tv & (~end_t), src_t, 0)

        tgt_logits_t = model.apply(
            {"params": opponent_params},
            h_new, src_for_heads, planet_out, relative_h, key_proj, target_action_mask,
            method=model.auto_tgt_logits_with_mask,
        )
        tgt_t = jax.random.categorical(key_tgt, safe_logits(tgt_logits_t), axis=-1)
        src_slot = jnp.clip(src_t - 1, 0, opp_fp.max_planets - 1)
        tgt_slot = jnp.clip(tgt_t - 1, 0, opp_fp.max_planets - 1)
        ships = full_send_amount(remaining[batch_idx, jnp.clip(src_t, 0, opp_fp.max_planets)])
        src_valid = (src_t > 0) & encoded.source_mask[batch_idx, src_slot] & (ships >= MIN_LAUNCH_SHIPS)
        tgt_valid = selected_target_is_valid_from_mask(target_action_mask, batch_idx, tgt_t)
        post_ok, angle = post_discard_target_check(flipped, src_t, tgt_t, ships, opp_fp)
        exec_t = tv & (~end_t) & src_valid & tgt_valid & post_ok
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
            src_for_heads, jnp.where(tv & (~end_t), tgt_t, 0),
            end_t, exec_t, t, planet_out,
            method=model.auto_token_embed,
        )
        ended = ended | end_t
        h = h_new

    return flip_actions(actions)


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
    ppo_params = PPOParams(
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_coef=cfg.ppo.clip_coef,
        ent_coef=cfg.ppo.ent_coef,
        ent_coef_fire=ent_coef_fire,
        ent_coef_target=ent_coef_target,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        epochs=cfg.ppo.epochs,
        minibatch_size=min(cfg.ppo.minibatch_size, cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.env.max_planets),
        norm_adv=cfg.ppo.norm_adv,
        clip_vloss=getattr(cfg.ppo, "clip_vloss", True),
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
        hidden_size=cfg.model.hidden_size,
        use_cnn=cfg.model.use_cnn,
        fleet_dim=FLEET_DIM,
        max_fleet_tokens=feat_params.max_fleet_tokens,
        num_transformer_layers=getattr(cfg.model, "num_transformer_layers", 2),
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
    )
    normalizer_state = init_normalizer()

    def deterministic_policy_actions(ts, st, fp, ns, np):  # noqa: E501
        return auto_deterministic_policy_actions(ts, model, st, fp, ns, np)

    start_update = 1

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
        train_state = train_state.replace(params=jax.tree_util.tree_map(jnp.asarray, ckpt_params))
        if "normalizer_state" in ckpt:
            normalizer_state = jax.tree_util.tree_map(jnp.asarray, ckpt["normalizer_state"])
        if getattr(cfg, "start_update_from_checkpoint", True):
            start_update = int(ckpt.get("update", 0)) + 1
        print(
            f"[init] loaded checkpoint {ckpt_path} "
            f"(ckpt_update={ckpt.get('update', 'unknown')}, start_update={start_update}); "
            "optimizer state is freshly initialized"
        )

    run_dir = Path(cfg.save_dir) / f"{cfg.run_name}_jax"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    eval_opponents = default_eval_opponents(
        REPO_ROOT,
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
            "entropy_end",
            "entropy_src",
            "entropy_tgt",
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
    elif not _WANDB_AVAILABLE:
        print(f"wandb unavailable: {_WANDB_IMPORT_ERROR} (install wandb in the training environment to enable logging)")
    else:
        print("wandb disabled by config (set wandb.enabled: true to enable)")

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
    # - mixed: 按权重在 JAX self-play / JAX checkpoint / Python 外部脚本对手间采样。
    raw_pool_mode = str(getattr(cfg, "pool_mode", "pfsp") or "pfsp").lower()
    pool_mode_aliases = {
        "current": "current",
        "lag_checkpoint": "lag_checkpoint",
        "fixed_lag_checkpoint": "lag_checkpoint",
        "fixed": "fixed_pool",
        "fixed_pool": "fixed_pool",
        "pfsp": "pfsp",
        "snapshot": "snapshot",
        "mixed": "mixed",
        "mixed_external": "mixed",
    }
    if raw_pool_mode not in pool_mode_aliases:
        raise ValueError(
            f"Unknown pool_mode={raw_pool_mode!r}; expected one of "
            "current, lag_checkpoint, fixed_pool, pfsp, snapshot, mixed"
        )
    pool_mode = pool_mode_aliases[raw_pool_mode]
    current_pool = pool_mode == "current"
    lag_checkpoint_pool = pool_mode == "lag_checkpoint"
    fixed_pool = pool_mode == "fixed_pool"
    pfsp_pool = pool_mode == "pfsp"
    snapshot_pool = pool_mode == "snapshot"
    mixed_pool = pool_mode == "mixed"
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
        elif mixed_pool:
            print("opponent=pool mixed (JAX self-play/checkpoints + optional Python external opponents)")
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

    def _copy_tree(tree):
        return jax.tree_util.tree_map(lambda x: x, tree)

    # 冻结对手权重（初始与学习器相同，每 opponent_update_freq 步同步一次）
    opponent_params = _copy_tree(train_state.params)
    # lag_snapshot 保存“上一个 interval 起点的策略”；新 lag_checkpoint/PFSP 接口到下一个
    # interval 时才发布，因此 lag 对手是真正间隔 pool_lag_interval 的 checkpoint。
    lag_snapshot_params = _copy_tree(train_state.params)
    lag_snapshot_normalizer_state = _copy_tree(normalizer_state)
    lagged_params = _copy_tree(train_state.params)
    lagged_normalizer_state = _copy_tree(normalizer_state)
    snapshot_pool_entries: list[dict] = []  # snapshot pool 快照列表
    pool_entries: list[dict] = []
    if use_pool and pfsp_pool:
        pool_entries = [{"id": "selfplay-current", "type": "current", "params": None, "winrate": 0.5}]
        if pool_include_nearest:
            pool_entries.insert(0, {"id": "nearest", "type": "nearest", "params": None, "winrate": 0.5})
    mixed_entries: list[dict[str, Any]] = []
    if use_pool and mixed_pool:
        raw_pool = list(getattr(cfg, "opponent_pool", []) or [])
        if not raw_pool:
            raise ValueError("pool_mode='mixed' requires opponent_pool entries")
        for idx, raw_entry in enumerate(raw_pool):
            if isinstance(raw_entry, str):
                entry = {"name": raw_entry, "type": raw_entry, "weight": 1.0}
            elif isinstance(raw_entry, dict):
                entry = dict(raw_entry)
            else:
                raise ValueError(f"opponent_pool[{idx}] must be a mapping or string")
            kind = str(entry.get("type", "external") or "external").lower().replace("-", "_")
            weight = float(entry.get("weight", 1.0))
            if weight <= 0.0:
                continue
            name = str(entry.get("name") or kind)
            if kind in {"self", "self_play", "selfplay", "current", "selfplay_current"}:
                mixed_entries.append({"id": name, "type": "selfplay-current", "weight": weight})
            elif kind in {"nearest"}:
                mixed_entries.append({"id": name, "type": "nearest", "weight": weight})
            elif kind in {"jax", "jax_checkpoint", "jax_ckpt", "native", "native_checkpoint"} or (
                kind in {"checkpoint", "old"} and (entry.get("checkpoint") or entry.get("checkpoint_path"))
            ):
                ckpt_value = entry.get("checkpoint") or entry.get("checkpoint_path") or entry.get("path")
                if not ckpt_value:
                    raise ValueError(f"opponent_pool[{idx}] jax_checkpoint entry requires checkpoint")
                ckpt, ckpt_path = _load_checkpoint(str(ckpt_value))
                ckpt_params = ckpt["params"]
                _assert_param_compatible(ckpt_params, ckpt_path)
                mixed_entries.append({
                    "id": name,
                    "type": "jax-checkpoint",
                    "weight": weight,
                    "params": jax.tree_util.tree_map(jnp.asarray, ckpt_params),
                    "normalizer_state": jax.tree_util.tree_map(
                        jnp.asarray,
                        ckpt.get("normalizer_state", normalizer_state),
                    ),
                    "checkpoint": ckpt_path,
                })
            elif kind in {"external", "script", "onnx"} or kind in {"checkpoint", "old"}:
                agent_dir_value = entry.get("agent_dir") or entry.get("path")
                if not agent_dir_value:
                    raise ValueError(f"opponent_pool[{idx}] external entry requires agent_dir")
                agent_dir = _resolve_config_path(str(agent_dir_value))
                mixed_entries.append({
                    "id": name,
                    "type": "external",
                    "weight": weight,
                    "opponent": ExternalScriptOpponent(agent_dir),
                    "agent_dir": agent_dir,
                })
            else:
                raise ValueError(f"opponent_pool[{idx}] contains unknown mixed opponent type: {kind}")
        if not mixed_entries:
            raise ValueError("pool_mode='mixed' opponent_pool has no positive-weight entries")
        total_weight = sum(float(entry["weight"]) for entry in mixed_entries)
        summary = ", ".join(f"{entry['id']}={float(entry['weight']) / total_weight:.2f}" for entry in mixed_entries)
        print(f"[pool] mixed entries: {summary}")

    def _copy_params(params):
        return _copy_tree(params)

    def _pfsp_sample(entries: list[dict]) -> dict:
        weights = [(1.0 - float(entry["winrate"]) + 0.1) ** 2 for entry in entries]
        return pool_rng.choices(entries, weights=weights, k=1)[0]

    def _mixed_sample(entries: list[dict[str, Any]]) -> dict[str, Any]:
        weights = [float(entry["weight"]) for entry in entries]
        return pool_rng.choices(entries, weights=weights, k=1)[0]

    def _pool_summary(entries: list[dict]) -> str:
        return ", ".join(f"{entry['id']}:{float(entry['winrate']):.2f}" for entry in entries)

    def _add_lag_to_pool(update_idx: int, params, norm_state) -> None:
        pool_entries.append({
            "id": f"selfplay-lag-{max(update_idx - pool_lag_interval, 0):06d}",
            "type": "lag",
            "params": _copy_params(params),
            "normalizer_state": _copy_tree(norm_state),
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

    fixed_phase1_opponents = list(getattr(
        cfg,
        "fixed_pool_phase1_opponents",
        ["nearest", "selfplay-current", "selfplay-lag"],
    ))
    fixed_phase2_opponents = list(getattr(
        cfg,
        "fixed_pool_phase2_opponents",
        ["selfplay-current", "selfplay-lag"],
    ))
    current_opponents = ["selfplay-current"]
    lag_checkpoint_phase1_opponents = list(getattr(cfg, "pool_lag_checkpoint_phase1_opponents", ["selfplay-lag"]))
    lag_checkpoint_phase2_opponents = list(getattr(cfg, "pool_lag_checkpoint_phase2_opponents", ["selfplay-current", "selfplay-lag"]))
    valid_fixed_opponents = {"nearest", "selfplay-current", "selfplay-lag"}
    if fixed_pool:
        for phase_name, phase_opponents in (
            ("fixed_pool_phase1_opponents", fixed_phase1_opponents),
            ("fixed_pool_phase2_opponents", fixed_phase2_opponents),
        ):
            if not phase_opponents:
                raise ValueError(f"{phase_name} must contain at least one opponent")
            invalid = sorted(set(phase_opponents) - valid_fixed_opponents)
            if invalid:
                raise ValueError(f"{phase_name} contains unknown opponents: {invalid}")
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
    _jit_nearest = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout_auto(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, cfg.ppo.rollout_steps, "nearest",
            max_action_tokens=max_action_tokens,
        )
    )
    _jit_selfplay = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout_auto(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, cfg.ppo.rollout_steps, "selfplay",
            max_action_tokens=max_action_tokens,
        )
    )
    _jit_step = jax.jit(
        lambda ts, opp_p, opp_ns, st, ns, key: collect_rollout_auto(
            ts, opp_p, opp_ns, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, cfg.ppo.rollout_steps,
            "selfplay" if self_play else "nearest",
            max_action_tokens=max_action_tokens,
        )
    )

    def _run_step(jit_fn, ts, opp_p, opp_ns, st, ns, key, ent_scale):
        """JIT 收集 rollout，然后在 Python 层执行 PPO 更新（含 KL 早停）。"""
        ts, st, ns, rollout, returns, advantages, ppo_rng, ep_metrics = jit_fn(ts, opp_p, opp_ns, st, ns, key)
        ts, ppo_metrics = ppo_update_auto(
            ts,
            rollout,
            returns,
            advantages,
            ppo_params,
            ppo_rng,
            max_action_tokens,
            ent_scale,
        )
        return ts, st, ns, {**ppo_metrics, **ep_metrics}

    def _run_named_opponent(opponent_id: str, ts, st, ns, key, ent_scale):
        if opponent_id == "nearest":
            return _run_step(_jit_nearest, ts, opponent_params, ns, st, ns, key, ent_scale)
        if opponent_id == "selfplay-current":
            return _run_step(_jit_selfplay, ts, opponent_params, ns, st, ns, key, ent_scale)
        if opponent_id == "selfplay-lag":
            return _run_step(_jit_selfplay, ts, lagged_params, lagged_normalizer_state, st, ns, key, ent_scale)
        raise ValueError(f"Unknown training opponent: {opponent_id}")

    @jax.jit
    def _jit_external_sample(ts, st, ns, key):
        raw_encoded = encode_batch(st, feat_params)
        encoded, ns = update_and_normalize(raw_encoded, ns, norm_params)
        sample = auto_policy_sample(ts, model, st, encoded, key, feat_params, max_action_tokens)
        b = encoded.planet_features.shape[0]
        row = AutoRollout(
            encoded.planet_features,
            encoded.relative_features,
            encoded.global_features,
            encoded.map_features,
            encoded.fleet_features,
            encoded.fleet_mask,
            encoded.candidate_mask,
            encoded.source_mask,
            sample["source_action_mask"],
            sample["target_action_mask"],
            sample["initial_ships"],
            sample["end_action"],
            sample["src_action"],
            sample["tgt_action"],
            sample["executed"],
            sample["invalid"],
            sample["old_logp_end"],
            sample["old_logp_src"],
            sample["old_logp_tgt"],
            sample["token_valid"],
            sample["value"],
            jnp.zeros((b,), dtype=jnp.float32),
            jnp.zeros((b,), dtype=bool),
        )
        return ns, sample["actions"], row

    @jax.jit
    def _jit_external_value(ts, st, ns):
        raw_encoded = encode_batch(st, feat_params)
        encoded = normalize_encoded(raw_encoded, ns, norm_params)
        b = encoded.planet_features.shape[0]
        k = max_action_tokens
        initial_ships = jnp.concatenate(
            [jnp.zeros((b, 1), dtype=encoded.source_ships.dtype), encoded.source_ships],
            axis=1,
        )
        out = ts.apply_fn(
            {"params": ts.params},
            encoded.planet_features,
            encoded.relative_features,
            encoded.global_features,
            encoded.candidate_mask,
            encoded.source_mask,
            jnp.zeros((b, k), dtype=bool),
            jnp.zeros((b, k), dtype=jnp.int32),
            jnp.zeros((b, k), dtype=jnp.int32),
            jnp.zeros((b, k), dtype=bool),
            initial_ships,
            map_features=encoded.map_features,
            fleet_features=encoded.fleet_features,
            fleet_mask=encoded.fleet_mask,
        )
        return out.value

    external_next_seed = int(cfg.seed) + 3_000_000

    def _canonical_external_state(obs, learner_idx: int):
        st = kaggle_obs_to_jax_state(obs, env_params)
        return flip_state(st) if learner_idx == 1 else st

    def _reset_external_env(env_slot: int, opponent: ExternalScriptOpponent, seed: int):
        from kaggle_environments import make as kag_make

        learner_idx = (env_slot + seed) % 2 if bool(getattr(cfg, "alternate_player_sides", True)) else 0
        opponent_idx = 1 - learner_idx
        opponent.reset_env(env_slot, seed)
        env = kag_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        return {
            "env": env,
            "states": states,
            "learner_idx": learner_idx,
            "opponent_idx": opponent_idx,
            "episode_steps": 0,
        }

    def _run_external_opponent(entry: dict[str, Any], ts, st, ns, key, ent_scale):
        nonlocal external_next_seed
        opponent = entry["opponent"]
        if not isinstance(opponent, ExternalScriptOpponent):
            raise TypeError(f"mixed external entry {entry.get('id')} has invalid opponent object")
        _rng_in, rollout_key, ppo_key = jax.random.split(key, 3)
        num_envs = int(cfg.ppo.num_envs)
        envs = []
        for env_slot in range(num_envs):
            envs.append(_reset_external_env(env_slot, opponent, external_next_seed))
            external_next_seed += 1

        rows_by_time: list[list[AutoRollout]] = []
        wins = losses = draws = 0
        finished_rewards: list[float] = []
        finished_steps: list[int] = []
        for t in range(int(cfg.ppo.rollout_steps)):
            rows_at_t: list[AutoRollout] = []
            for env_slot, env_rec in enumerate(envs):
                states = env_rec["states"]
                learner_idx = int(env_rec["learner_idx"])
                opponent_idx = int(env_rec["opponent_idx"])
                obs = states[learner_idx]["observation"]
                opp_obs = states[opponent_idx]["observation"]
                jax_state = _canonical_external_state(obs, learner_idx)
                sample_key = jax.random.fold_in(rollout_key, t * num_envs + env_slot)
                ns, learner_actions, row = _jit_external_sample(ts, jax_state, ns, sample_key)
                kaggle_actions = flip_actions(learner_actions) if learner_idx == 1 else learner_actions
                learner_action = jax_actions_to_kaggle(kaggle_actions)
                opp_action = opponent.act_for_env(env_slot, opp_obs)
                joint_actions = [None, None]
                joint_actions[learner_idx] = learner_action
                joint_actions[opponent_idx] = opp_action
                next_states = env_rec["env"].step(joint_actions)
                env_rec["states"] = next_states
                env_rec["episode_steps"] = int(env_rec["episode_steps"]) + 1
                done = next_states[0]["status"] != "ACTIVE"
                reward = float(next_states[learner_idx].get("reward", 0.0) or 0.0) if done else 0.0
                row = row._replace(
                    reward=jnp.asarray([reward], dtype=jnp.float32),
                    done=jnp.asarray([done], dtype=bool),
                )
                rows_at_t.append(row)
                if done:
                    finished_rewards.append(reward)
                    finished_steps.append(int(env_rec["episode_steps"]))
                    if reward > 0.0:
                        wins += 1
                    elif reward < 0.0:
                        losses += 1
                    else:
                        draws += 1
                    envs[env_slot] = _reset_external_env(env_slot, opponent, external_next_seed)
                    external_next_seed += 1
            rows_by_time.append(rows_at_t)

        rollout = _stack_auto_rollout_rows(rows_by_time)
        last_values = []
        for env_rec in envs:
            learner_idx = int(env_rec["learner_idx"])
            obs = env_rec["states"][learner_idx]["observation"]
            jax_state = _canonical_external_state(obs, learner_idx)
            last_values.append(_jit_external_value(ts, jax_state, ns)[0])
        last_values = jnp.stack(last_values, axis=0)
        returns, advantages = compute_gae_state(rollout.reward, rollout.done, rollout.old_value, last_values, ppo_params)
        ts, ppo_metrics = ppo_update_auto(ts, rollout, returns, advantages, ppo_params, ppo_key, max_action_tokens, ent_scale)

        finished = rollout.done
        episodes = jnp.maximum(finished.sum(), 1)
        ep_reward_sum = jnp.where(finished, rollout.reward, 0.0).sum()
        ep_metrics = {
            "episode_reward": ep_reward_sum / episodes,
            "win_rate": jnp.asarray(wins / max(wins + losses + draws, 1), dtype=jnp.float32),
            "mean_return": returns.mean(),
            "episodes_per_rollout": finished.sum().astype(jnp.float32),
            "external_mean_steps": jnp.asarray(float(np.mean(finished_steps) if finished_steps else 0.0), dtype=jnp.float32),
            "external_mean_reward": jnp.asarray(float(np.mean(finished_rewards) if finished_rewards else 0.0), dtype=jnp.float32),
        }
        return ts, st, ns, {**ppo_metrics, **ep_metrics}

    for update in range(start_update, cfg.ppo.total_updates + 1):
        selected_pool_entry = None
        selected_pool_id = ""

        # 每隔 opponent_update_freq 步把当前权重同步给 self-play/current 对手
        if (use_pool and (current_pool or mixed_pool)) or ((update - 1) % max(opponent_update_freq, 1) == 0):
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
            elif mixed_pool:
                selected_pool_entry = _mixed_sample(mixed_entries)
                selected_pool_id = str(selected_pool_entry["id"])
                selected_type = str(selected_pool_entry["type"])
                if selected_type in {"selfplay-current", "nearest", "selfplay-lag"}:
                    train_state, state, normalizer_state, metrics = _run_named_opponent(
                        selected_type,
                        train_state,
                        state,
                        normalizer_state,
                        step_key,
                        ent_scale,
                    )
                elif selected_type == "external":
                    train_state, state, normalizer_state, metrics = _run_external_opponent(
                        selected_pool_entry,
                        train_state,
                        state,
                        normalizer_state,
                        step_key,
                        ent_scale,
                    )
                elif selected_type == "jax-checkpoint":
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay,
                        train_state,
                        selected_pool_entry["params"],
                        selected_pool_entry["normalizer_state"],
                        state,
                        normalizer_state,
                        step_key,
                        ent_scale,
                    )
                else:
                    raise ValueError(f"Unknown mixed pool entry type: {selected_type}")
            elif snapshot_pool:
                # 20% 打当前策略，80% 打历史快照；若快照池为空则全打当前
                if snapshot_pool_entries and pool_rng.random() >= snapshot_current_ratio:
                    snap = pool_rng.choice(snapshot_pool_entries)
                    selected_pool_id = snap["id"]
                    train_state, state, normalizer_state, metrics = _run_step(
                        _jit_selfplay, train_state, snap["params"], snap["normalizer_state"], state, normalizer_state, step_key, ent_scale)
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
                        _jit_selfplay,
                        train_state,
                        selected_pool_entry["params"],
                        selected_pool_entry.get("normalizer_state", normalizer_state),
                        state,
                        normalizer_state,
                        step_key,
                        ent_scale,
                    )
        else:
            if self_play and update % opponent_update_freq == 1:
                pass  # 上面已经同步过 current 对手。
            train_state, state, normalizer_state, metrics = _run_step(
                _jit_step, train_state, opponent_params, normalizer_state, state, normalizer_state, step_key, ent_scale)
        metrics = jax.device_get(metrics)
        current_lr = _current_lr(cfg, int(jax.device_get(train_state.step)), total_optimizer_steps, warmup_steps)
        metrics["lr"] = current_lr
        metrics["ent_scale"] = ent_scale
        if use_pool and pfsp_pool and selected_pool_entry is not None and float(metrics["episodes_per_rollout"]) > 0.0:
            old_wr = float(selected_pool_entry["winrate"])
            selected_pool_entry["winrate"] = 0.95 * old_wr + 0.05 * float(metrics["win_rate"])
            metrics["pool_selected_winrate"] = selected_pool_entry["winrate"]
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
            else:
                line += f" opp={selected_pool_id}"
        line += (
            f" invalid={metrics.get('invalid_action_rate', 0.0):.3f}"
            f" exec={metrics.get('executed_action_rate', 0.0):.3f}"
            f" seq={metrics.get('seq_len_mean', 0.0):.2f}"
            f" logr_heads="
            f"{metrics.get('max_abs_log_ratio_end', 0.0):.1f}/"
            f"{metrics.get('max_abs_log_ratio_src', 0.0):.1f}/"
            f"{metrics.get('max_abs_log_ratio_tgt', 0.0):.1f}"
        )
        if update % log_every == 0:
            print(line, flush=True)

        # snapshot pool：每 snapshot_interval 步存一个快照
        if use_pool and snapshot_pool and snapshot_interval > 0 and update % snapshot_interval == 0:
            snap_id = f"snap-{update:06d}"
            snapshot_pool_entries.append({
                "id": snap_id,
                "params": _copy_params(train_state.params),
                "normalizer_state": _copy_tree(normalizer_state),
            })
            print(f"[pool] update={update} 存入快照 {snap_id}，快照池大小={len(snapshot_pool_entries)}")

        if use_pool and lagged_pool and pool_lag_interval > 0 and update % pool_lag_interval == 0:
            if fixed_pool:
                lagged_params = _copy_params(train_state.params)
                lagged_normalizer_state = _copy_tree(normalizer_state)
                print(f"[pool] update={update} 同步 fixed_pool lagged 对手权重 (lag={pool_lag_interval})")
            elif lag_checkpoint_pool:
                lagged_params = _copy_params(lag_snapshot_params)
                lagged_normalizer_state = _copy_tree(lag_snapshot_normalizer_state)
                lag_snapshot_params = _copy_params(train_state.params)
                lag_snapshot_normalizer_state = _copy_tree(normalizer_state)
                print(f"[pool] update={update} 发布 lag_checkpoint 对手权重 (lag={pool_lag_interval})")
            elif pfsp_pool:
                _add_lag_to_pool(update, lag_snapshot_params, lag_snapshot_normalizer_state)
                lag_snapshot_params = _copy_params(train_state.params)
                lag_snapshot_normalizer_state = _copy_tree(normalizer_state)

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
                    elif mixed_pool:
                        fixed_size = len(mixed_entries)
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
                f"{metrics.get('entropy_end', 0.0):.8f}",
                f"{metrics.get('entropy_src', 0.0):.8f}",
                f"{metrics.get('entropy_tgt', 0.0):.8f}",
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
                    "losses/max_abs_log_ratio_end": float(metrics.get("max_abs_log_ratio_end", 0.0)),
                    "losses/max_abs_log_ratio_src": float(metrics.get("max_abs_log_ratio_src", 0.0)),
                    "losses/max_abs_log_ratio_tgt": float(metrics.get("max_abs_log_ratio_tgt", 0.0)),
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
                elif mixed_pool:
                    fixed_size = len(mixed_entries)
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
        if opponent_kind == "selfplay":
            learner_aux, opponent_aux = shared_selfplay_feature_aux(st)
            raw_encoded = encode_batch_with_aux(st, feat_params, learner_aux)
        else:
            opponent_aux = None
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
                feature_aux=opponent_aux,
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
            encoded.fleet_features,
            encoded.fleet_mask,
            encoded.candidate_mask,
            encoded.source_mask,
            sample["source_action_mask"],
            sample["target_action_mask"],
            sample["initial_ships"],
            sample["end_action"],
            sample["src_action"],
            sample["tgt_action"],
            sample["executed"],
            sample["invalid"],
            sample["old_logp_end"],
            sample["old_logp_src"],
            sample["old_logp_tgt"],
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
        last_encoded.source_mask,
        jnp.zeros((b, k), dtype=bool),
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


def auto_policy_sample(train_state, model, state, encoded, rng, feat_params: FeatureParams, max_action_tokens: int):
    """自回归 rollout 采样；选定 source 后动态计算该 source 的 exact target mask。"""
    b = encoded.planet_features.shape[0]
    k = max_action_tokens
    initial_ships = jnp.concatenate([jnp.zeros((b, 1), dtype=encoded.source_ships.dtype), encoded.source_ships], axis=1)
    end_action = jnp.zeros((b, k), dtype=bool)
    src_action = jnp.zeros((b, k), dtype=jnp.int32)
    tgt_action = jnp.zeros((b, k), dtype=jnp.int32)
    executed = jnp.zeros((b, k), dtype=bool)
    invalid = jnp.zeros((b, k), dtype=bool)
    token_valid = jnp.zeros((b, k), dtype=bool)
    source_action_mask = jnp.zeros((b, k, feat_params.max_planets + 1), dtype=bool)
    target_action_mask = jnp.zeros((b, k, feat_params.max_planets + 1), dtype=bool)
    old_logp_end = jnp.zeros((b, k), dtype=jnp.float32)
    old_logp_src = jnp.zeros((b, k), dtype=jnp.float32)
    old_logp_tgt = jnp.zeros((b, k), dtype=jnp.float32)
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
        (key, ended, h, prev_tok, remaining, end_action, src_action, tgt_action, executed, invalid,
         token_valid, source_action_mask, target_action_mask, old_logp_end, old_logp_src, old_logp_tgt, actions) = carry
        key, key_end, key_src, key_tgt = jax.random.split(key, 4)
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
        no_valid_source = ~has_valid_logits(src_logits_t)
        end_t = tv & (sampled_end | no_valid_source)
        src_t, src_logits_final, source_mask_t, target_mask_t, found_src = select_source_with_coarse_targets(
            src_logits_t,
            key_src,
            tv & (~end_t),
            encoded,
            feat_params,
            deterministic=False,
        )
        end_t = end_t | (tv & (~end_t) & (~found_src))

        src_for_heads = jnp.where(tv & (~end_t), src_t, 0)
        tgt_logits_t = train_state.apply_fn(
            {"params": train_state.params},
            h_new,
            src_for_heads,
            planet_out,
            relative_h,
            key_proj,
            target_mask_t,
            method=model.auto_tgt_logits_with_mask,
        )
        tgt_logits_t = safe_logits(tgt_logits_t)
        tgt_t = jax.random.categorical(key_tgt, tgt_logits_t, axis=-1)

        src_slot = jnp.clip(src_t - 1, 0, feat_params.max_planets - 1)
        tgt_slot = jnp.clip(tgt_t - 1, 0, feat_params.max_planets - 1)
        ships = full_send_amount(remaining[batch_idx, jnp.clip(src_t, 0, feat_params.max_planets)])
        src_valid = (src_t > 0) & encoded.source_mask[batch_idx, src_slot] & (ships >= MIN_LAUNCH_SHIPS)
        tgt_valid = selected_target_is_valid_from_mask(target_mask_t, batch_idx, tgt_t)
        post_ok, angle = post_discard_target_check(state, src_t, tgt_t, ships, feat_params)
        exec_t = tv & (~end_t) & src_valid & tgt_valid & post_ok
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
            jnp.where(tv & (~end_t), tgt_t, 0),
            end_t,
            exec_t,
            t,
            planet_out,
            method=model.auto_token_embed,
        )
        end_action = end_action.at[:, t].set(end_t)
        src_action = src_action.at[:, t].set(src_for_heads)
        tgt_action = tgt_action.at[:, t].set(jnp.where(tv & (~end_t), tgt_t, 0))
        executed = executed.at[:, t].set(exec_t)
        invalid = invalid.at[:, t].set(invalid_t)
        token_valid = token_valid.at[:, t].set(tv)
        source_action_mask = source_action_mask.at[:, t].set(source_mask_t)
        target_action_mask = target_action_mask.at[:, t].set(target_mask_t)
        old_logp_end = old_logp_end.at[:, t].set(jnp.where(tv, _bernoulli_log_prob_local(end_logits_t, end_t), 0.0))
        old_logp_src = old_logp_src.at[:, t].set(jnp.where(exec_t, _categorical_log_prob_local(src_logits_final, src_t), 0.0))
        old_logp_tgt = old_logp_tgt.at[:, t].set(jnp.where(exec_t, _categorical_log_prob_local(tgt_logits_t, tgt_t), 0.0))
        ended = ended | end_t
        return (key, ended, h_new, prev_tok, remaining, end_action, src_action, tgt_action, executed, invalid,
                token_valid, source_action_mask, target_action_mask, old_logp_end, old_logp_src, old_logp_tgt, actions)

    init = (rng, ended, h, prev_tok, remaining, end_action, src_action, tgt_action, executed, invalid,
            token_valid, source_action_mask, target_action_mask, old_logp_end, old_logp_src, old_logp_tgt, actions)
    (_, _ended, _h, _prev_tok, _remaining, end_action, src_action, tgt_action, executed, invalid,
     token_valid, source_action_mask, target_action_mask, old_logp_end, old_logp_src, old_logp_tgt, actions) = jax.lax.fori_loop(0, k, body, init)
    return {
        "actions": actions,
        "initial_ships": initial_ships,
        "end_action": end_action,
        "src_action": src_action,
        "tgt_action": tgt_action,
        "executed": executed,
        "invalid": invalid,
        "token_valid": token_valid,
        "source_action_mask": source_action_mask,
        "target_action_mask": target_action_mask,
        "old_logp_end": old_logp_end,
        "old_logp_src": old_logp_src,
        "old_logp_tgt": old_logp_tgt,
        "value": value,
    }


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
