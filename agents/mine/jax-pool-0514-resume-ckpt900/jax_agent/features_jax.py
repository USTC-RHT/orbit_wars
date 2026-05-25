"""JAX 特征编码器。

把 EnvState 转换为神经网络固定形状输入。

设计原则：
- 每颗我方星球独立对应一行决策（source）。
- 每行决策包含：self_features（自身）、candidate_features（全图目标 no-op + P 颗星球）、
  global_features（全局态势）、map_features（CNN 小地图）。
- 特征维度与 PyTorch 版严格对齐：SELF=18 / CANDIDATE=26 / GLOBAL=11。
- precompute_inbound 使用 60 步轨迹预测，支持旋转星球和彗星路径。
- encode_batch: vmap over 单局；encode_single: vmap over 来源星球。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .fast_env import (
    BOARD_SIZE,
    CENTER,
    COMETS_PER_SPAWN,
    MAX_COMET_PATH,
    MAX_COMET_SPAWNS,
    ROTATION_RADIUS_LIMIT,
    EnvParams,
    EnvState,
    comet_planet_mask,
    fleet_speed,
    point_segment_distance_sq,
)


# 特征向量维度（与 PyTorch 版 src/features.py 严格对齐，修改时必须同步更新）
SELF_DIM = 18         # 来源星球自身特征维度
CANDIDATE_DIM = 26    # 单个目标候选星球特征维度
GLOBAL_DIM = 11       # 全局局势特征维度
MAP_CHANNELS = 6      # CNN 小地图通道数（来源星/友方/敌方/中立/友方舰队/敌方舰队）
MAP_GRID_SIZE = 8     # CNN 小地图分辨率（8x8）
INBOUND_HORIZON = 60.0  # 预测舰队第局 N 回合内贫心星球
MIN_LAUNCH = 10.0     # 最小发射船数：驻军不足此数时禁止全部档位
SHIP_RATIOS = jnp.asarray([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=jnp.float32)  # 派兵比例档位


@dataclass(frozen=True)
class FeatureParams:
    """特征编码参数（编译时固定，安全传入 jit/vmap）。"""
    max_planets: int = 64         # 星球槽位数（应等于 EnvParams.max_planets）
    ship_bucket_count: int = 8    # 派兵树位数量
    max_ships: float = 500.0      # 归一化基准船数上限
    max_production: float = 5.0   # 归一化基准生产力上限
    episode_steps: int = 500      # 归一化基准回合数
    target_filter_enabled: bool = False  # 是否启用 Proto-guided 目标过滤
    attack_top_k: int = 6         # 最多保留 K 个进攻候选
    defense_top_k: int = 3        # 最多保留 K 个防守/输送候选

    @classmethod
    def from_train_config(cls, cfg) -> "FeatureParams":
        return cls(
            max_planets=int(cfg.env.max_planets),
            ship_bucket_count=int(cfg.env.ship_bucket_count),
            max_ships=float(cfg.env.max_ships),
            max_production=float(cfg.env.max_production),
            episode_steps=int(cfg.env.episode_steps),
            target_filter_enabled=bool(cfg.env.target_filter_enabled),
            attack_top_k=int(cfg.env.attack_top_k),
            defense_top_k=int(cfg.env.defense_top_k),
        )


class EncodedBatch(NamedTuple):
    self_features: jnp.ndarray
    candidate_features: jnp.ndarray
    global_features: jnp.ndarray
    map_features: jnp.ndarray
    candidate_mask: jnp.ndarray
    amount_mask: jnp.ndarray
    source_ids: jnp.ndarray
    source_ships: jnp.ndarray
    candidate_ids: jnp.ndarray
    ship_buckets: jnp.ndarray
    source_mask: jnp.ndarray


def target_count(params: FeatureParams) -> int:
    return params.max_planets + 1


def encode_batch(state: EnvState, params: FeatureParams) -> EncodedBatch:
    return jax.vmap(lambda s: encode_single(s, params))(state)


def encode_single(state: EnvState, params: FeatureParams) -> EncodedBatch:
    planets = state.planets
    alive = state.planet_alive
    owner = planets[:, 1].astype(jnp.int32)
    is_mine = alive & (owner == 0)
    source_mask = is_mine
    source_ids = planets[:, 0].astype(jnp.int32)
    source_ships = planets[:, 5]
    global_f = build_global_features(state, params)
    inbound = precompute_inbound(state, params)

    def source_row(src_idx):
        return build_source_decision(state, src_idx, global_f, inbound, params)

    rows = jax.vmap(source_row)(jnp.arange(params.max_planets))
    return EncodedBatch(
        self_features=rows.self_features,
        candidate_features=rows.candidate_features,
        global_features=rows.global_features,
        map_features=rows.map_features,
        candidate_mask=rows.candidate_mask,
        amount_mask=rows.amount_mask,
        source_ids=source_ids,
        source_ships=source_ships,
        candidate_ids=rows.candidate_ids,
        ship_buckets=rows.ship_buckets,
        source_mask=source_mask,
    )


def build_source_decision(state: EnvState, src_idx: jnp.ndarray, global_f: jnp.ndarray, inbound: jnp.ndarray, params: FeatureParams) -> EncodedBatch:
    src = state.planets[src_idx]
    self_f = build_self_features(state, src_idx, inbound, params)
    cand_f, cand_mask, amount_mask, candidate_ids, ship_buckets = build_candidate_features(state, src_idx, inbound, params)
    map_f = build_map_features(state, src_idx, params)
    return EncodedBatch(
        self_features=self_f,
        candidate_features=cand_f,
        global_features=global_f,
        map_features=map_f,
        candidate_mask=cand_mask,
        amount_mask=amount_mask,
        source_ids=src[0].astype(jnp.int32),
        source_ships=src[5],
        candidate_ids=candidate_ids,
        ship_buckets=ship_buckets,
        source_mask=(src[1].astype(jnp.int32) == 0) & state.planet_alive[src_idx],
    )


def build_self_features(state: EnvState, src_idx: jnp.ndarray, inbound: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    """构建来源星球的自身特征（SELF_DIM=18）。

    各维语义（与 FEATURES.md 对应）：
      0: is_planet=1.0（占位）
      1-2: x/y 归一化坐标
      3: radius / 5
      4: ships / max_ships (clamp)
      5: production / max_production
      6: is_rotating (0/1)
      7: 到太阳距离 / board_size
      8-9: 我方/敌方星球数 / max_planets
      10: 我方星球总船归一化
      11: （敌方舰队船 - 我方舰队船）归一化
      12-17: inbound[src] = (my_ships, enemy_ships, my_eta_min, my_eta_mean, enemy_eta_min, enemy_eta_mean)
    """
    src = state.planets[src_idx]
    planets = state.planets
    alive = state.planet_alive
    owner = planets[:, 1].astype(jnp.int32)
    my_planets = alive & (owner == 0)
    enemy_planets = alive & (owner == 1)
    my_fleets = state.fleet_alive & (state.fleets[:, 1].astype(jnp.int32) == 0)
    enemy_fleets = state.fleet_alive & (state.fleets[:, 1].astype(jnp.int32) == 1)
    # 敌方舰队假威减我方：正表示我方适应默防压力
    fleet_delta = jnp.where(enemy_fleets, state.fleets[:, 6], 0.0).sum() - jnp.where(my_fleets, state.fleets[:, 6], 0.0).sum()
    inb = inbound[src_idx]  # 预测的到达该星球的舰队信息
    return jnp.asarray(
        [
            1.0,
            src[2] / BOARD_SIZE,
            src[3] / BOARD_SIZE,
            src[4] / 5.0,
            clip01(src[5] / params.max_ships),
            src[6] / params.max_production,
            is_rotating(src),
            distance_to_sun(src) / BOARD_SIZE,
            my_planets.sum() / params.max_planets,
            enemy_planets.sum() / params.max_planets,
            jnp.where(my_planets, planets[:, 5], 0.0).sum() / (params.max_planets * params.max_ships),
            fleet_delta / (params.max_planets * params.max_ships),
            clip01(inb[0] / params.max_ships),
            clip01(inb[1] / params.max_ships),
            inb[2],
            inb[3],
            inb[4],
            inb[5],
        ],
        dtype=jnp.float32,
    )


def build_candidate_features(state: EnvState, src_idx: jnp.ndarray, inbound: jnp.ndarray, params: FeatureParams):
    """构建全图目标候选特征、mask 和派兵档位（CANDIDATE_DIM=26）。

    返回：(features[P+1, 26], candidate_mask[P+1], amount_mask[P+1, B], candidate_ids[P+1], ship_buckets[B])
    - slot 0 固定为 no-op（不发射）
    - slot 1..P 对应所有星球（包括来源星球自身，不过 mask=False）
    攻击/中立必须发超过驻军的船，进攻组 amount mask 只开放足够多的档位。
    """
    src = state.planets[src_idx]
    planets = state.planets
    alive = state.planet_alive
    n_targets = params.max_planets + 1
    candidate_ids = jnp.concatenate([jnp.asarray([-1], dtype=jnp.int32), planets[:, 0].astype(jnp.int32)])
    ship_buckets = build_ship_buckets(src[5], params)

    valid_buckets = ship_buckets > 0
    max_bucket = jnp.maximum(ship_buckets.max(), 1.0)
    spd_max = fleet_speed(max_bucket)  # 最大档位对应的舰队速度，用于 eta 归一化

    # slot 0：no-op，dim0=1 表示 is_noop，其余继续置 0
    noop_features = jnp.zeros((1, CANDIDATE_DIM), dtype=jnp.float32).at[0, 0].set(1.0)
    src_x = src[2]
    src_y = src[3]
    dx = planets[:, 2] - src_x
    dy = planets[:, 3] - src_y
    dist = jnp.sqrt(dx * dx + dy * dy)
    owner = planets[:, 1].astype(jnp.int32)
    is_source = jnp.arange(params.max_planets) == src_idx
    is_neutral = owner == -1
    is_friend = (owner == 0) & (~is_source)
    is_enemy = owner == 1
    is_comet = comet_planet_mask(state)
    proto = proto_score(src, planets) / 100.0          # Proto-V15 目标评分
    eta_max = dist / jnp.maximum(spd_max, 1e-6) / INBOUND_HORIZON  # 最快到达时间归一化
    defense_pressure = compute_defense_pressure(planets, inbound)  # 是否需要增援
    cand = jnp.stack(
        [
            jnp.zeros_like(dist),
            alive.astype(jnp.float32),
            is_source.astype(jnp.float32),
            is_neutral.astype(jnp.float32),
            is_friend.astype(jnp.float32),
            is_enemy.astype(jnp.float32),
            planets[:, 2] / BOARD_SIZE,
            planets[:, 3] / BOARD_SIZE,
            dx / BOARD_SIZE,
            dy / BOARD_SIZE,
            dist / BOARD_SIZE,
            planets[:, 4] / 5.0,
            clip01(planets[:, 5] / params.max_ships),
            planets[:, 6] / params.max_production,
            jax.vmap(is_rotating)(planets),
            is_comet.astype(jnp.float32),
            clip01((planets[:, 5] + 1.0) / jnp.maximum(src[5], 1.0)),
            clip01(src[5] / params.max_ships) * jnp.ones_like(dist),
            clip01(inbound[:, 0] / params.max_ships),
            clip01(inbound[:, 1] / params.max_ships),
            inbound[:, 2],
            inbound[:, 3],
            inbound[:, 4],
            inbound[:, 5],
            proto,
            eta_max,
        ],
        axis=-1,
    )
    features = jnp.concatenate([noop_features, cand], axis=0)

    # 候选星球合法性 mask：必须存活、不是自身、来源有足够船、存在合法档位
    can_send = alive & (~is_source) & (src[5] >= MIN_LAUNCH) & valid_buckets.any()
    positive_amount = valid_buckets
    # 进攻明星时只开放能压制驻军的派兵档位；增援友方时任意正数均可
    enough = ship_buckets[None, :] > planets[:, 5:6]
    per_planet_amount = jnp.where(is_friend[:, None], positive_amount[None, :], enough & positive_amount[None, :])
    per_planet_mask = can_send & per_planet_amount.any(axis=-1)
    # slot 0 永远可选（no-op）
    mask = jnp.concatenate([jnp.asarray([True]), per_planet_mask])
    amount_mask = jnp.concatenate([jnp.ones((1, params.ship_bucket_count), dtype=bool), per_planet_amount], axis=0)
    amount_mask = amount_mask & mask[:, None]
    amount_mask = amount_mask.at[0, :].set(True)
    if params.target_filter_enabled:
        mask = apply_target_filter(mask, is_friend, is_enemy | is_neutral, proto, defense_pressure, params)
        amount_mask = amount_mask & mask[:, None]
        amount_mask = amount_mask.at[0, :].set(True)
    return features, mask, amount_mask, candidate_ids, ship_buckets


def apply_target_filter(
    mask: jnp.ndarray,
    is_friend: jnp.ndarray,
    is_attack: jnp.ndarray,
    proto: jnp.ndarray,
    defense_pressure: jnp.ndarray,
    params: FeatureParams,
) -> jnp.ndarray:
    """Mirror Python target_filter: no-op + top proto attacks + top defense friends."""

    planet_mask = mask[1:]
    keep_planets = jnp.zeros((params.max_planets,), dtype=bool)

    attack_score = jnp.where(planet_mask & is_attack, proto, -1e9)
    attack_order = jnp.argsort(attack_score)[::-1]
    attack_k = min(params.attack_top_k, params.max_planets)
    keep_planets = keep_planets.at[attack_order[:attack_k]].set(attack_score[attack_order[:attack_k]] > -1e8)

    defense_score = jnp.where(planet_mask & is_friend & (defense_pressure > 0.0), defense_pressure, -1e9)
    defense_order = jnp.argsort(defense_score)[::-1]
    defense_k = min(params.defense_top_k, params.max_planets)
    keep_planets = keep_planets.at[defense_order[:defense_k]].set(defense_score[defense_order[:defense_k]] > -1e8)
    return jnp.concatenate([jnp.asarray([mask[0]]), keep_planets])


def build_global_features(state: EnvState, params: FeatureParams) -> jnp.ndarray:
    planets = state.planets
    alive = state.planet_alive
    owner = planets[:, 1].astype(jnp.int32)
    my = alive & (owner == 0)
    enemy = alive & (owner == 1)
    neutral = alive & (owner == -1)
    fowner = state.fleets[:, 1].astype(jnp.int32)
    my_f = state.fleet_alive & (fowner == 0)
    enemy_f = state.fleet_alive & (fowner == 1)
    return jnp.asarray(
        [
            state.step / params.episode_steps,
            state.angular_velocity / 0.05,
            my.sum() / params.max_planets,
            enemy.sum() / params.max_planets,
            neutral.sum() / params.max_planets,
            jnp.where(my, planets[:, 5], 0.0).sum() / (params.max_planets * params.max_ships),
            jnp.where(enemy, planets[:, 5], 0.0).sum() / (params.max_planets * params.max_ships),
            jnp.where(my_f, state.fleets[:, 6], 0.0).sum() / (params.max_planets * params.max_ships),
            jnp.where(enemy_f, state.fleets[:, 6], 0.0).sum() / (params.max_planets * params.max_ships),
            jnp.where(my, planets[:, 6], 0.0).sum() / (params.max_planets * params.max_production),
            jnp.where(enemy, planets[:, 6], 0.0).sum() / (params.max_planets * params.max_production),
        ],
        dtype=jnp.float32,
    )


def precompute_inbound(state: EnvState, params: FeatureParams) -> jnp.ndarray:
    """JAX port of src.features.precompute_inbound for planet/fleet arrays.

    Returns [P, 6]:
      my_ships, enemy_ships, my_eta_min, my_eta_mean, enemy_eta_min, enemy_eta_mean

    This mirrors the Python feature path for regular/rotating planets and
    comet path lookup by path_index + T.
    """

    horizon = int(INBOUND_HORIZON)
    fleets = state.fleets
    planets = state.planets
    f_alive = state.fleet_alive
    p_alive = state.planet_alive
    fleet_count = fleets.shape[0]
    planet_count = planets.shape[0]

    ts = jnp.arange(horizon + 1, dtype=jnp.float32)
    ships = jnp.maximum(fleets[:, 6], 1.0)
    speed = fleet_speed(ships)
    vx = jnp.cos(fleets[:, 4]) * speed
    vy = jnp.sin(fleets[:, 4]) * speed
    fx_t = fleets[:, 2:3] + vx[:, None] * ts[None, :]
    fy_t = fleets[:, 3:4] + vy[:, None] * ts[None, :]
    in_bounds = (fx_t >= 0.0) & (fx_t <= BOARD_SIZE) & (fy_t >= 0.0) & (fy_t <= BOARD_SIZE)

    px_t, py_t = planet_positions_over_horizon(state, horizon)
    radii = planets[:, 4]

    fx_prev = fx_t[:, :-1]
    fy_prev = fy_t[:, :-1]
    fx_cur = fx_t[:, 1:]
    fy_cur = fy_t[:, 1:]
    px_old = px_t[:, :-1]
    py_old = py_t[:, :-1]
    px_new = px_t[:, 1:]
    py_new = py_t[:, 1:]

    move_d2 = point_segment_distance_sq(
        px_old[None, :, :],
        py_old[None, :, :],
        fx_prev[:, None, :],
        fy_prev[:, None, :],
        fx_cur[:, None, :],
        fy_cur[:, None, :],
    )
    cur_in_bounds = in_bounds[:, 1:][:, None, :]
    move_hit = move_d2 < (radii[None, :, None] ** 2)
    move_hit = move_hit & cur_in_bounds & f_alive[:, None, None] & p_alive[None, :, None]

    sweep_x = px_new - px_old
    sweep_y = py_new - py_old
    sweep_active = (sweep_x * sweep_x + sweep_y * sweep_y) > 1e-9
    sweep_d2 = point_segment_distance_sq(
        fx_cur[:, None, :],
        fy_cur[:, None, :],
        px_old[None, :, :],
        py_old[None, :, :],
        px_new[None, :, :],
        py_new[None, :, :],
    )
    sweep_hit = sweep_d2 < (radii[None, :, None] ** 2)
    sweep_hit = sweep_hit & sweep_active[None, :, :] & cur_in_bounds & f_alive[:, None, None] & p_alive[None, :, None]

    large_key = jnp.int32(2 * (horizon + 1) + 1)
    steps = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    move_keys = (2 * steps)[None, None, :]
    sweep_keys = (2 * steps + 1)[None, None, :]
    first_move_key = jnp.where(move_hit, move_keys, large_key).min(axis=2)
    first_sweep_key = jnp.where(sweep_hit, sweep_keys, large_key).min(axis=2)
    first_key_fp = jnp.minimum(first_move_key, first_sweep_key)

    fleet_first_key = first_key_fp.min(axis=1)
    fleet_target = first_key_fp.argmin(axis=1)
    fleet_valid = fleet_first_key < large_key
    fleet_first_t = (fleet_first_key // 2).astype(jnp.float32)
    target_onehot = jax.nn.one_hot(fleet_target, planet_count, dtype=jnp.float32)
    selected = target_onehot * fleet_valid[:, None].astype(jnp.float32)
    owner = fleets[:, 1].astype(jnp.int32)
    my_sel = selected * (owner[:, None] == 0).astype(jnp.float32)
    enemy_sel = selected * (owner[:, None] != 0).astype(jnp.float32)

    my_ships = (my_sel * fleets[:, 6:7]).sum(axis=0)
    enemy_ships = (enemy_sel * fleets[:, 6:7]).sum(axis=0)
    norm = float(horizon)
    my_count = my_sel.sum(axis=0)
    enemy_count = enemy_sel.sum(axis=0)
    t = fleet_first_t[:, None]
    my_min = jnp.where(my_count > 0, jnp.where(my_sel > 0, t, 1e9).min(axis=0) / norm, 1.0)
    enemy_min = jnp.where(enemy_count > 0, jnp.where(enemy_sel > 0, t, 1e9).min(axis=0) / norm, 1.0)
    my_mean = jnp.where(my_count > 0, (my_sel * t).sum(axis=0) / jnp.maximum(my_count, 1.0) / norm, 1.0)
    enemy_mean = jnp.where(enemy_count > 0, (enemy_sel * t).sum(axis=0) / jnp.maximum(enemy_count, 1.0) / norm, 1.0)
    inbound = jnp.stack([my_ships, enemy_ships, my_min, my_mean, enemy_min, enemy_mean], axis=-1)
    return jnp.where(p_alive[:, None], inbound, jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32))


def planet_positions_over_horizon(state: EnvState, horizon: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    planets = state.planets
    init = state.initial_planets
    ts = jnp.arange(horizon + 1, dtype=jnp.float32)
    comet_mask = comet_planet_mask(state)
    dx = init[:, 2] - CENTER
    dy = init[:, 3] - CENTER
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    rotating = (orbit_r + init[:, 4] < ROTATION_RADIUS_LIMIT) & state.planet_alive & (~comet_mask)
    phi_cur = jnp.arctan2(planets[:, 3] - CENTER, planets[:, 2] - CENTER)
    phi = phi_cur[:, None] + state.angular_velocity * ts[None, :]
    px = CENTER + orbit_r[:, None] * jnp.cos(phi)
    py = CENTER + orbit_r[:, None] * jnp.sin(phi)
    px = jnp.where(rotating[:, None], px, planets[:, 2:3])
    py = jnp.where(rotating[:, None], py, planets[:, 3:4])

    comet_slots = state.comet_slots.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    safe_slots = jnp.where(comet_slots >= 0, comet_slots, 0)
    group_active = state.comet_path_index[:, None] >= -1
    active = (state.comet_slots >= 0) & group_active
    active_flat = active.reshape(-1)
    path_index = state.comet_path_index[:, None, None] + jnp.arange(horizon + 1, dtype=jnp.int32)[None, None, :]
    path_index = jnp.broadcast_to(path_index, (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1))
    lengths = state.comet_lengths[:, :, None]
    valid_t = active[:, :, None] & (path_index >= 0) & (path_index < lengths)
    path_index = jnp.clip(path_index, 0, MAX_COMET_PATH - 1)
    gather_index = jnp.broadcast_to(path_index[..., None], (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1, 2))
    path_pos = jnp.take_along_axis(state.comet_paths, gather_index, axis=2)
    path_pos = jnp.where(valid_t[..., None], path_pos, -1000.0)
    flat_pos = path_pos.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN, horizon + 1, 2)
    slot_onehot = jax.nn.one_hot(safe_slots, planets.shape[0], dtype=jnp.float32) * active_flat[:, None].astype(jnp.float32)
    comet_has = slot_onehot.sum(axis=0) > 0.0
    comet_px = jnp.einsum("kp,kt->pt", slot_onehot, flat_pos[:, :, 0])
    comet_py = jnp.einsum("kp,kt->pt", slot_onehot, flat_pos[:, :, 1])
    px = jnp.where(comet_has[:, None], comet_px, px)
    py = jnp.where(comet_has[:, None], comet_py, py)
    return px, py


def compute_defense_pressure(planets: jnp.ndarray, inbound: jnp.ndarray) -> jnp.ndarray:
    my_inbound = inbound[:, 0]
    enemy_inbound = inbound[:, 1]
    enemy_eta = inbound[:, 4] * INBOUND_HORIZON
    projected_defense = planets[:, 5] + my_inbound + enemy_eta * planets[:, 6]
    return jnp.where(enemy_inbound > 0.0, enemy_inbound - projected_defense, 0.0)


def build_map_features(state: EnvState, src_idx: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    grid = jnp.zeros((MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE), dtype=jnp.float32)
    planets = state.planets
    owner = planets[:, 1].astype(jnp.int32)
    idx = jnp.arange(params.max_planets)
    channels = jnp.where(idx == src_idx, 0, jnp.where(owner == 0, 1, jnp.where(owner == -1, 3, 2)))
    values = state.planet_alive.astype(jnp.float32)
    grid = _accum_many(grid, channels, planets[:, 2], planets[:, 3], values)
    f_channels = jnp.where(state.fleets[:, 1].astype(jnp.int32) == 0, 4, 5)
    f_values = clip01(state.fleets[:, 6] / params.max_ships) * state.fleet_alive.astype(jnp.float32)
    return _accum_many(grid, f_channels, state.fleets[:, 2], state.fleets[:, 3], f_values)


def _accum_many(grid, channels, x, y, values):
    in_bounds = (x >= 0.0) & (x <= BOARD_SIZE) & (y >= 0.0) & (y <= BOARD_SIZE) & (values > 0.0)
    ix = jnp.clip((x / BOARD_SIZE * MAP_GRID_SIZE).astype(jnp.int32), 0, MAP_GRID_SIZE - 1)
    iy = jnp.clip((y / BOARD_SIZE * MAP_GRID_SIZE).astype(jnp.int32), 0, MAP_GRID_SIZE - 1)
    grid = grid.at[channels, iy, ix].add(jnp.where(in_bounds, values, 0.0))
    return jnp.clip(grid, 0.0, 1.0)


def build_ship_buckets(src_ships: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    ratios = SHIP_RATIOS[: params.ship_bucket_count]
    s = jnp.floor(src_ships)
    raw = jnp.clip(jnp.round(s * ratios), MIN_LAUNCH, s)
    return jnp.where(s >= MIN_LAUNCH, raw, jnp.zeros_like(raw))


def proto_score(src: jnp.ndarray, tgt: jnp.ndarray) -> jnp.ndarray:
    dist = jnp.sqrt((tgt[:, 2] - src[2]) ** 2 + (tgt[:, 3] - src[3]) ** 2)
    min_ships = tgt[:, 5] + 1.0
    speed = fleet_speed(min_ships)
    eta = dist / jnp.maximum(speed, 1e-6)
    has_owner = tgt[:, 1] != -1.0
    enemy_produced = jnp.where(has_owner, eta * tgt[:, 6], 0.0)
    enemy_bonus = jnp.where(has_owner, tgt[:, 6], 0.0)
    total_ships = min_ships + enemy_produced
    return (100.0 - dist) + 15.0 * tgt[:, 6] + 10.0 * enemy_bonus - 0.7 * total_ships - 2.0 * eta


def is_rotating(planet: jnp.ndarray) -> jnp.ndarray:
    return (distance_to_sun(planet) + planet[4] < ROTATION_RADIUS_LIMIT).astype(jnp.float32)


def distance_to_sun(planet: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt((planet[2] - CENTER) ** 2 + (planet[3] - CENTER) ** 2)


def clip01(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(x, 0.0, 1.0)


def flatten_sources(encoded: EncodedBatch) -> EncodedBatch:
    b, p = encoded.self_features.shape[:2]

    def flat(x):
        return x.reshape((b * p,) + x.shape[2:])

    return EncodedBatch(
        self_features=flat(encoded.self_features),
        candidate_features=flat(encoded.candidate_features),
        global_features=flat(encoded.global_features),
        map_features=flat(encoded.map_features),
        candidate_mask=flat(encoded.candidate_mask),
        amount_mask=flat(encoded.amount_mask),
        source_ids=flat(encoded.source_ids),
        source_ships=flat(encoded.source_ships),
        candidate_ids=flat(encoded.candidate_ids),
        ship_buckets=flat(encoded.ship_buckets),
        source_mask=flat(encoded.source_mask),
    )


def unflatten_actions(flat_actions: jnp.ndarray, batch_size: int, max_planets: int) -> jnp.ndarray:
    return flat_actions.reshape((batch_size, max_planets) + flat_actions.shape[1:])
