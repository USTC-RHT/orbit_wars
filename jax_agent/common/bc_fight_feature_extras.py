"""行为克隆 fight 分支的共享增强特征。

这些特征不替换基础 BC 编码器，而是在 feature-row 上追加更接近实战判断的信号：

- 来源星球安全余量和局部拓扑
- 来源到目标的可达性与 ETA
- 不新增发射时的未来驻军状态
- 敌方压力梯度与防守紧急度

训练、JAX runtime、ONNX 打包前的 feature-row 生成都应复用这里，避免特征含义漂移。
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .fast_env import BOARD_SIZE, CENTER, EnvState, comet_planet_mask, fleet_speed, point_segment_distance_sq
from .features_jax import (
    INBOUND_HORIZON,
    MIN_LAUNCH,
    FeatureParams,
    build_ship_buckets,
    clip01,
    distance_to_sun,
    is_rotating,
    planet_positions_alive_over_horizon,
    proto_score,
)
from .garrison_jax import DEFAULT_GARRISON_HORIZON, garrison_status_jax, safe_drain_all_sources_jax, target_flip_status_from_garrison_jax
from .pressure_jax import enemy_pressure_jax


FIGHT_PLANET_DIM = 28              # 来源星球增强特征维度
FIGHT_RELATIVE_DIM = 22            # source-target 相对增强特征维度
FIGHT_SOURCE_DIM = FIGHT_PLANET_DIM
FIGHT_CANDIDATE_DIM = FIGHT_RELATIVE_DIM
LOCAL_PLANET_RADIUS = 25.0         # 局部拓扑统计半径
LAUNCH_SURFACE_OFFSET = 0.1        # 发射时离开来源星球表面的安全偏移
TARGET_HIT_SURFACE_OFFSET = 0.0    # 命中目标表面判定偏移
REACHABILITY_EPS = 1e-4            # ETA 二分/可达性判定容差
FRIENDLY_REGROUP_MAX_TIME = 7.0    # 友方短途调兵上限回合
FRIENDLY_REGROUP_PRESSURE_DELTA_MIN = 0.25  # 认为值得调兵的压力差阈值
CAPTURE_OVERHEAD = 1.0             # 攻击占领需要超过驻军的额外船数


class FightFeatureExtras(NamedTuple):
    """fight 增强特征输出。"""

    planet: jnp.ndarray    # [P, 28] 或 [S, 28]：来源星球特征
    relative: jnp.ndarray  # [P, P+1, 22] 或 [S, P+1, 22]：目标相对特征


def build_fight_feature_extras(state: EnvState, params: FeatureParams) -> FightFeatureExtras:
    """为所有星球槽位构建 fight 增强特征。"""

    # 先做一次“无新增动作”的未来驻军预测，后续安全余量/翻盘风险共用它。
    status = garrison_status_jax(state, horizon=DEFAULT_GARRISON_HORIZON)
    safe_drain = safe_drain_all_sources_jax(
        status,
        state.planets[:, 5],
        h_eff=float(DEFAULT_GARRISON_HORIZON),
        player_id=0,
    )
    enemy_pressure = enemy_pressure_jax(state, horizon=float(DEFAULT_GARRISON_HORIZON), player_id=0)
    flip_status = target_flip_status_from_garrison_jax(state, status, player_id=0)
    inbound = precompute_inbound_from_garrison(state, status, params)
    future_px, future_py, future_alive = planet_positions_alive_over_horizon(state, int(INBOUND_HORIZON))

    def source_row(src_idx):
        return build_source_extras(
            state,
            src_idx,
            status,
            inbound,
            safe_drain,
            enemy_pressure,
            flip_status.will_flip,
            flip_status.flip_turn,
            flip_status.urgency,
            future_px,
            future_py,
            future_alive,
            params,
        )

    rows = jax.vmap(source_row)(jnp.arange(params.max_planets))
    return FightFeatureExtras(
        planet=rows.planet,
        relative=rows.relative,
    )


def build_fight_feature_extras_for_sources(state: EnvState, source_slots: jnp.ndarray, params: FeatureParams) -> FightFeatureExtras:
    """只为指定来源槽位构建 fight 增强特征，用于预处理分桶加速。"""

    status = garrison_status_jax(state, horizon=DEFAULT_GARRISON_HORIZON)
    safe_drain = safe_drain_all_sources_jax(
        status,
        state.planets[:, 5],
        h_eff=float(DEFAULT_GARRISON_HORIZON),
        player_id=0,
    )
    enemy_pressure = enemy_pressure_jax(state, horizon=float(DEFAULT_GARRISON_HORIZON), player_id=0)
    flip_status = target_flip_status_from_garrison_jax(state, status, player_id=0)
    inbound = precompute_inbound_from_garrison(state, status, params)
    future_px, future_py, future_alive = planet_positions_alive_over_horizon(state, int(INBOUND_HORIZON))

    def source_row(src_idx):
        return build_source_extras(
            state,
            src_idx,
            status,
            inbound,
            safe_drain,
            enemy_pressure,
            flip_status.will_flip,
            flip_status.flip_turn,
            flip_status.urgency,
            future_px,
            future_py,
            future_alive,
            params,
        )

    rows = jax.vmap(source_row)(source_slots.astype(jnp.int32))
    return FightFeatureExtras(
        planet=rows.planet,
        relative=rows.relative,
    )


def build_source_extras(
    state: EnvState,
    src_idx: jnp.ndarray,
    status,
    inbound: jnp.ndarray,
    safe_drain: jnp.ndarray,
    enemy_pressure: jnp.ndarray,
    target_will_flip: jnp.ndarray,
    target_flip_turn: jnp.ndarray,
    target_flip_urgency: jnp.ndarray,
    future_px: jnp.ndarray,
    future_py: jnp.ndarray,
    future_alive: jnp.ndarray,
    params: FeatureParams,
) -> FightFeatureExtras:
    """构建单个来源星球对应的 planet/relative 两组增强特征。"""

    planet = build_planet_features(state, src_idx, inbound, safe_drain, enemy_pressure, params)
    relative = build_relative_features(
        state,
        src_idx,
        status,
        enemy_pressure,
        target_will_flip,
        target_flip_turn,
        target_flip_urgency,
        future_px,
        future_py,
        future_alive,
        params,
    )
    return FightFeatureExtras(planet=planet, relative=relative)


def build_planet_features(
    state: EnvState,
    src_idx: jnp.ndarray,
    inbound: jnp.ndarray,
    safe_drain: jnp.ndarray,
    enemy_pressure: jnp.ndarray,
    params: FeatureParams,
) -> jnp.ndarray:
    """来源星球增强特征（FIGHT_PLANET_DIM=28）。"""

    src = state.planets[src_idx]
    owner = src[1].astype(jnp.int32)
    alive = state.planet_alive[src_idx]
    is_comet = comet_planet_mask(state)[src_idx]
    is_rotating_planet = (is_rotating(src) > 0.5) & (~is_comet)
    is_static_planet = alive & (~is_rotating_planet) & (~is_comet)
    topology = local_topology_features(state, src_idx, params)
    inb = inbound[src_idx]
    safe_drain_ratio = clip01(safe_drain[src_idx] / jnp.maximum(src[5], 1.0))
    full_send_risk_ratio = clip01((src[5] - safe_drain[src_idx]) / jnp.maximum(src[5], 1.0))
    del enemy_pressure
    return jnp.asarray(
        [
            1.0,
            alive.astype(jnp.float32),
            (owner == -1).astype(jnp.float32),
            (owner == 0).astype(jnp.float32),
            (owner == 1).astype(jnp.float32),
            src[2] / BOARD_SIZE,
            src[3] / BOARD_SIZE,
            src[4] / 5.0,
            clip01(src[5] / params.max_ships),
            src[6] / params.max_production,
            is_static_planet.astype(jnp.float32),
            is_rotating_planet.astype(jnp.float32),
            is_comet.astype(jnp.float32),
            distance_to_sun(src) / BOARD_SIZE,
            topology[0],
            topology[1],
            topology[2],
            topology[3],
            topology[4],
            topology[5],
            clip01(inb[0] / params.max_ships),
            clip01(inb[1] / params.max_ships),
            inb[2],
            inb[3],
            inb[4],
            inb[5],
            safe_drain_ratio,
            full_send_risk_ratio,
        ],
        dtype=jnp.float32,
    )


def local_topology_features(state: EnvState, src_idx: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    """统计来源星球附近的友方、敌方、中立、彗星和生产力密度。"""

    planets = state.planets
    src = planets[src_idx]
    alive = state.planet_alive
    owner = planets[:, 1].astype(jnp.int32)
    not_self = jnp.arange(params.max_planets) != src_idx
    valid = alive & not_self
    dx = planets[:, 2] - src[2]
    dy = planets[:, 3] - src[3]
    dist = jnp.sqrt(dx * dx + dy * dy)
    my_mask = valid & (owner == 0)
    enemy_mask = valid & (owner == 1)
    neutral_mask = valid & (owner == -1)

    def nearest(mask):
        any_mask = mask.any()
        nearest_dist = jnp.where(mask, dist, 1.0e9).min()
        return jnp.where(any_mask, nearest_dist / BOARD_SIZE, 1.0)

    def count_in_radius(mask):
        return (mask & (dist <= LOCAL_PLANET_RADIUS)).sum() / params.max_planets

    return jnp.asarray(
        [
            nearest(my_mask),
            nearest(enemy_mask),
            nearest(neutral_mask),
            count_in_radius(my_mask),
            count_in_radius(enemy_mask),
            count_in_radius(neutral_mask),
        ],
        dtype=jnp.float32,
    )


def build_relative_features(
    state: EnvState,
    src_idx: jnp.ndarray,
    status,
    enemy_pressure: jnp.ndarray,
    target_will_flip: jnp.ndarray,
    target_flip_turn: jnp.ndarray,
    target_flip_urgency: jnp.ndarray,
    future_px: jnp.ndarray,
    future_py: jnp.ndarray,
    future_alive: jnp.ndarray,
    params: FeatureParams,
) -> jnp.ndarray:
    src = state.planets[src_idx]
    planets = state.planets
    alive = state.planet_alive
    dx = planets[:, 2] - src[2]
    dy = planets[:, 3] - src[3]
    dist = jnp.sqrt(dx * dx + dy * dy)
    owner = planets[:, 1].astype(jnp.int32)
    is_source = jnp.arange(params.max_planets) == src_idx
    is_neutral = owner == -1
    is_friend = (owner == 0) & (~is_source)
    is_enemy = owner == 1
    target_valid = alive & (~is_source) & (is_friend | is_enemy | is_neutral)
    ship_buckets = build_ship_buckets(src[5], params)
    max_bucket = jnp.maximum(ship_buckets.max(), 1.0)
    reachable, eta_turns_raw = reachable_full_send(src, planets, future_px, future_py, future_alive, max_bucket)
    eta_turns = jnp.where(target_valid, eta_turns_raw, INBOUND_HORIZON + 1.0)
    eta_norm = eta_turns / INBOUND_HORIZON
    eta_garrison = relative_eta_garrison_features(status, eta_turns, params)

    target_ships_at_eta = eta_garrison[:, 3] * params.max_ships
    my_arrivals = eta_garrison[:, 4] * params.max_ships
    enemy_arrivals = eta_garrison[:, 5] * params.max_ships
    capture_margin_at_eta = jnp.clip((max_bucket - target_ships_at_eta) / params.max_ships, -1.0, 1.0)
    pressure_delta = enemy_pressure - enemy_pressure[jnp.clip(src_idx, 0, enemy_pressure.shape[0] - 1)]
    pressure_delta_norm = jnp.clip(pressure_delta / params.max_ships, -1.0, 1.0)
    target_enemy_pressure_norm = clip01(enemy_pressure / params.max_ships)
    arrival_balance_at_eta = jnp.clip((my_arrivals - enemy_arrivals) / params.max_ships, -1.0, 1.0)
    contestedness_at_eta = clip01((my_arrivals + enemy_arrivals) / params.max_ships)
    capture_floor_at_eta, capture_surplus_at_eta = relative_capture_floor_features(status, eta_turns, max_bucket, params, player_id=0)
    defense_score = friendly_defense_score(
        target_will_flip,
        target_flip_turn,
        target_flip_urgency,
        enemy_pressure,
        src_idx,
        eta_turns,
    )
    flip_turn_norm = clip01(target_flip_turn.astype(jnp.float32) / float(DEFAULT_GARRISON_HORIZON))
    flip_urgency_norm = clip01(target_flip_urgency / params.max_ships)
    defense_score_norm = clip01(defense_score / params.max_ships)
    del reachable, flip_turn_norm, flip_urgency_norm, defense_score_norm

    per_planet = jnp.stack(
        [
            is_source.astype(jnp.float32),
            dx / BOARD_SIZE,
            dy / BOARD_SIZE,
            dist / BOARD_SIZE,
            clip01((planets[:, 5] + 1.0) / jnp.maximum(src[5], 1.0)),
            clip01(src[5] / params.max_ships) * jnp.ones_like(dist),
            proto_score(src, planets) / 100.0,
            eta_norm,
            eta_garrison[:, 0],
            eta_garrison[:, 1],
            eta_garrison[:, 2],
            eta_garrison[:, 3],
            eta_garrison[:, 4],
            eta_garrison[:, 5],
            eta_garrison[:, 6],
            pressure_delta_norm,
            target_enemy_pressure_norm,
            capture_margin_at_eta,
            arrival_balance_at_eta,
            contestedness_at_eta,
            capture_floor_at_eta,
            capture_surplus_at_eta,
        ],
        axis=-1,
    )
    per_planet = jnp.where(target_valid[:, None], per_planet, 0.0)
    relative = jnp.concatenate([jnp.zeros((1, FIGHT_RELATIVE_DIM), dtype=jnp.float32), per_planet], axis=0)
    return relative


def precompute_inbound_from_garrison(state: EnvState, status, params: FeatureParams) -> jnp.ndarray:
    arrivals = status.arrivals_by_owner.astype(jnp.float32)
    horizon = int(arrivals.shape[-2]) - 1
    if horizon <= 0:
        default = jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        return jnp.broadcast_to(default, (state.planets.shape[0], 6))

    future_arrivals = arrivals[:, 1:, :]
    my_by_step = future_arrivals[:, :, 0]
    enemy_by_step = future_arrivals[:, :, 1]
    turns = jnp.arange(1, horizon + 1, dtype=jnp.float32)
    norm = jnp.asarray(float(horizon), dtype=jnp.float32)

    def owner_features(by_step: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ships = by_step.sum(axis=-1)
        has_step = by_step > 0.0
        has_any = has_step.any(axis=-1)
        eta_min = jnp.where(has_any, jnp.where(has_step, turns[None, :], 1.0e9).min(axis=-1) / norm, 1.0)
        eta_mean = jnp.where(
            ships > 0.0,
            (by_step * turns[None, :]).sum(axis=-1) / jnp.maximum(ships, 1.0) / norm,
            1.0,
        )
        return ships, eta_min, eta_mean

    my_ships, my_min, my_mean = owner_features(my_by_step)
    enemy_ships, enemy_min, enemy_mean = owner_features(enemy_by_step)
    inbound = jnp.stack([my_ships, enemy_ships, my_min, my_mean, enemy_min, enemy_mean], axis=-1)
    default = jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    return jnp.where(state.planet_alive[:, None], inbound, default)


def reachable_full_send(
    src: jnp.ndarray,
    planets: jnp.ndarray,
    future_px: jnp.ndarray,
    future_py: jnp.ndarray,
    future_alive: jnp.ndarray,
    ships: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    horizon = int(future_px.shape[-1]) - 1
    ax = future_px[:, :-1]
    ay = future_py[:, :-1]
    bx = future_px[:, 1:]
    by = future_py[:, 1:]
    segment_alive = future_alive[:, :-1] & future_alive[:, 1:]
    d2 = point_segment_distance_sq(src[2], src[3], ax, ay, bx, by)
    seg_dist = jnp.sqrt(jnp.maximum(d2, 0.0))
    gap = src[4] + planets[:, 4] + LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET
    surface_dist = jnp.maximum(seg_dist - gap[:, None], 0.0)
    turns = jnp.arange(1, horizon + 1, dtype=jnp.float32)
    speed = fleet_speed(jnp.maximum(ships, 1.0))
    reachable_by_turn = (surface_dist <= (speed * turns[None, :] * (1.0 + REACHABILITY_EPS))) & segment_alive
    can_reach = (ships >= MIN_LAUNCH) & reachable_by_turn.any(axis=-1)
    eta = jnp.min(jnp.where(reachable_by_turn, turns[None, :], INBOUND_HORIZON + 1.0), axis=-1)
    return can_reach, jnp.where(can_reach, eta, INBOUND_HORIZON + 1.0)


def relative_eta_garrison_features(status, eta_turns: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    owner = status.pre_combat_owner
    ships = status.pre_combat_ships
    arrivals = status.arrivals_by_owner.astype(jnp.float32)
    horizon = int(owner.shape[-1]) - 1
    eta_ceil = jnp.ceil(eta_turns)
    eta_out = eta_ceil > float(horizon)
    k = jnp.clip(eta_ceil.astype(jnp.int32), 0, horizon)
    idx = jnp.arange(owner.shape[0], dtype=jnp.int32)
    owner_at_eta = owner[idx, k]
    ships_at_eta = ships[idx, k]
    step_mask = jnp.arange(horizon + 1, dtype=jnp.int32)[None, :] <= k[:, None]
    my_arrivals = jnp.where(step_mask, arrivals[:, :, 0], 0.0).sum(axis=-1)
    enemy_arrivals = jnp.where(step_mask, arrivals[:, :, 1], 0.0).sum(axis=-1)
    return jnp.stack(
        [
            (owner_at_eta == -1).astype(jnp.float32),
            (owner_at_eta == 0).astype(jnp.float32),
            (owner_at_eta == 1).astype(jnp.float32),
            clip01(ships_at_eta / params.max_ships),
            clip01(my_arrivals / params.max_ships),
            clip01(enemy_arrivals / params.max_ships),
            eta_out.astype(jnp.float32),
        ],
        axis=-1,
    )


def relative_capture_floor_features(status, eta_turns: jnp.ndarray, send_ships: jnp.ndarray, params: FeatureParams, player_id: int = 0):
    owner = status.owner
    ships = status.ships
    horizon = int(owner.shape[-1]) - 1
    eta_ceil = jnp.ceil(eta_turns)
    k = jnp.clip(eta_ceil.astype(jnp.int32), 0, horizon)
    idx = jnp.arange(owner.shape[0], dtype=jnp.int32)
    owner_at_eta = owner[idx, k]
    defenders_at_eta = ships[idx, k]
    capture_floor = jnp.ceil(jnp.maximum(defenders_at_eta + CAPTURE_OVERHEAD, 1.0))
    floor = jnp.where(owner_at_eta == int(player_id), 1.0, capture_floor)
    floor_norm = clip01(floor / params.max_ships)
    surplus_norm = jnp.clip((send_ships - floor) / params.max_ships, -1.0, 1.0)
    return floor_norm, surplus_norm


def friendly_defense_score(
    target_will_flip: jnp.ndarray,
    target_flip_turn: jnp.ndarray,
    target_flip_urgency: jnp.ndarray,
    enemy_pressure: jnp.ndarray,
    src_idx: jnp.ndarray,
    eta_turns: jnp.ndarray,
) -> jnp.ndarray:
    flip_window = jnp.maximum(target_flip_turn.astype(jnp.float32), 1.0)
    flip_eta_decay = jnp.clip(1.0 - eta_turns / flip_window, 0.0, 1.0)
    flip_score = jnp.where(target_will_flip, target_flip_urgency * flip_eta_decay, 0.0)
    src_pressure = enemy_pressure[jnp.clip(src_idx, 0, enemy_pressure.shape[0] - 1)]
    pressure_delta = enemy_pressure - src_pressure
    regroup_eta_decay = jnp.clip(1.0 - eta_turns / FRIENDLY_REGROUP_MAX_TIME, 0.0, 1.0)
    regroup_score = jnp.maximum(pressure_delta - FRIENDLY_REGROUP_PRESSURE_DELTA_MIN, 0.0) * regroup_eta_decay
    return flip_score + regroup_score
