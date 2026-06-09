"""JAX 特征编码器。

把 EnvState 转换为神经网络固定形状输入。

设计原则：
- 每颗我方星球独立对应一行决策（source）。
- 每行决策包含：planet_features（星球自身）、relative_features（source-target 相对特征）、
  global_features（全局态势）、map_features（CNN 小地图）。
- 特征维度：PLANET=26 / RELATIVE=8 / GLOBAL=11。
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


# 特征向量维度（JAX 主训练路径使用；修改时需要同步 model/normalizer/docs）
PLANET_DIM = 26         # 星球局部特征维度
RELATIVE_DIM = 8     # source-target 相对特征维度
GLOBAL_DIM = 11       # 全局局势特征维度
FLEET_DIM = 28        # 单支舰队 token 特征维度
MAX_FLEET_TOKENS = 32 # 每个 source 选择进入 Transformer 的舰队 token 数
MAP_CHANNELS = 6      # CNN 小地图通道数（来源星/友方/敌方/中立/我方飞船强度/敌方飞船强度）
MAP_GRID_SIZE = 8     # CNN 小地图分辨率（8x8）
INBOUND_HORIZON = 60.0  # 预测舰队未来 N 回合内命中的星球
MIN_LAUNCH = 10.0     # 最小发射船数：驻军不足此数时禁止全部档位
LOCAL_PLANET_RADIUS = 25.0  # 局部星球密度统计半径
SHIP_RATIOS = jnp.asarray([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=jnp.float32)  # 派兵比例档位


@dataclass(frozen=True)
class FeatureParams:
    """特征编码参数（编译时固定，安全传入 jit/vmap）。"""
    max_planets: int = 64         # 星球槽位数（应等于 EnvParams.max_planets）
    ship_bucket_count: int = 8    # 派兵档位数量
    max_ships: float = 500.0      # 归一化基准船数上限
    max_production: float = 5.0   # 归一化基准生产力上限
    episode_steps: int = 500      # 归一化基准回合数
    target_filter_enabled: bool = False  # 是否启用 Proto-guided 目标过滤
    attack_top_k: int = 6         # 最多保留 K 个进攻候选
    defense_top_k: int = 3        # 最多保留 K 个防守/输送候选
    max_fleet_tokens: int = MAX_FLEET_TOKENS
    fleet_selection: str = "size"  # size=大舰队优先；impact=2P-Improve 威胁/机会优先

    @classmethod
    def from_train_config(cls, cfg) -> "FeatureParams":
        model_cfg = getattr(cfg, "model", None)
        return cls(
            max_planets=int(cfg.env.max_planets),
            ship_bucket_count=int(cfg.env.ship_bucket_count),
            max_ships=float(cfg.env.max_ships),
            max_production=float(cfg.env.max_production),
            episode_steps=int(cfg.env.episode_steps),
            target_filter_enabled=bool(cfg.env.target_filter_enabled),
            attack_top_k=int(cfg.env.attack_top_k),
            defense_top_k=int(cfg.env.defense_top_k),
            max_fleet_tokens=int(getattr(model_cfg, "max_fleet_tokens", MAX_FLEET_TOKENS)),
            fleet_selection=str(getattr(model_cfg, "fleet_selection", "size")),
        )


class EncodedBatch(NamedTuple):
    """网络输入张量集合。

    candidate_mask 控制目标选择，amount_mask 控制派兵档位，
    source_mask 控制哪些星球可以作为发射来源。
    """
    planet_features: jnp.ndarray
    relative_features: jnp.ndarray
    global_features: jnp.ndarray
    map_features: jnp.ndarray
    candidate_mask: jnp.ndarray
    amount_mask: jnp.ndarray
    source_ids: jnp.ndarray
    source_ships: jnp.ndarray
    candidate_ids: jnp.ndarray
    ship_buckets: jnp.ndarray
    source_mask: jnp.ndarray
    fleet_features: jnp.ndarray
    fleet_mask: jnp.ndarray


def target_count(params: FeatureParams) -> int:
    """目标槽位数量：slot 0 是 no-op，后面是所有星球。"""
    return params.max_planets + 1


def encode_batch(state: EnvState, params: FeatureParams) -> EncodedBatch:
    """批量编码 EnvState，外层 batch 通过 vmap 并行。"""
    return jax.vmap(lambda s: encode_single(s, params))(state)


def encode_single(state: EnvState, params: FeatureParams) -> EncodedBatch:
    """编码单局棋盘，把每个星球位置都组织成一个候选 source 行。"""
    planets = state.planets
    alive = state.planet_alive
    owner = planets[:, 1].astype(jnp.int32)
    is_mine = alive & (owner == 0)
    source_mask = is_mine
    source_ids = planets[:, 0].astype(jnp.int32)
    source_ships = planets[:, 5]
    global_f = build_global_features(state, params)
    inbound = precompute_inbound(state, params)
    fleet_f, fleet_m = build_fleet_features(state, params)

    def source_row(src_idx):
        return build_source_decision(state, src_idx, global_f, inbound, fleet_f, fleet_m, params)

    rows = jax.vmap(source_row)(jnp.arange(params.max_planets))
    return EncodedBatch(
        planet_features=rows.planet_features,
        relative_features=rows.relative_features,
        global_features=rows.global_features,
        map_features=rows.map_features,
        candidate_mask=rows.candidate_mask,
        amount_mask=rows.amount_mask,
        source_ids=source_ids,
        source_ships=source_ships,
        candidate_ids=rows.candidate_ids,
        ship_buckets=rows.ship_buckets,
        source_mask=source_mask,
        fleet_features=rows.fleet_features,
        fleet_mask=rows.fleet_mask,
    )


def build_source_decision(
    state: EnvState,
    src_idx: jnp.ndarray,
    global_f: jnp.ndarray,
    inbound: jnp.ndarray,
    fleet_f: jnp.ndarray,
    fleet_m: jnp.ndarray,
    params: FeatureParams,
) -> EncodedBatch:
    """构造某个 source 星球对应的一行决策输入。"""
    src = state.planets[src_idx]
    planet_f = build_planet_features(state, src_idx, inbound, params)
    rel_f, cand_mask, amount_mask, candidate_ids, ship_buckets = build_relative_features(state, src_idx, inbound, params)
    map_f = build_map_features(state, src_idx, params)
    return EncodedBatch(
        planet_features=planet_f,
        relative_features=rel_f,
        global_features=global_f,
        map_features=map_f,
        candidate_mask=cand_mask,
        amount_mask=amount_mask,
        source_ids=src[0].astype(jnp.int32),
        source_ships=src[5],
        candidate_ids=candidate_ids,
        ship_buckets=ship_buckets,
        source_mask=(src[1].astype(jnp.int32) == 0) & state.planet_alive[src_idx],
        fleet_features=fleet_f,
        fleet_mask=fleet_m,
    )


def build_planet_features(state: EnvState, src_idx: jnp.ndarray, inbound: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    """构建星球局部特征（PLANET_DIM=26）。

    各维语义（与 FEATURES.md 对应）：
      0: is_planet=1.0（占位）
      1: is_alive
      2-4: is_neutral / is_mine / is_enemy
      5-6: x/y 归一化坐标
      7: radius / 5
      8: ships / max_ships (clamp)
      9: production / max_production
      10-12: is_static / is_rotating_planet / is_comet_planet
      13: 到太阳距离 / board_size
      14-16: 最近我方/敌方/中立星球距离 / board_size
      17-19: R=25 内我方/敌方/中立星球数量 / max_planets
      20-25: inbound[src] = (my_ships, enemy_ships, my_eta_min, my_eta_mean, enemy_eta_min, enemy_eta_mean)
    """
    src = state.planets[src_idx]
    alive = state.planet_alive[src_idx]
    owner = src[1].astype(jnp.int32)
    is_comet = comet_planet_mask(state)[src_idx]
    is_rotating_planet = (is_rotating(src) > 0.5) & (~is_comet)
    is_static_planet = alive & (~is_rotating_planet) & (~is_comet)
    topo = planet_local_topology_features(state, src_idx, params)
    inb = inbound[src_idx]  # 预测的到达该星球的舰队信息
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
            topo[0],
            topo[1],
            topo[2],
            topo[3],
            topo[4],
            topo[5],
            clip01(inb[0] / params.max_ships),
            clip01(inb[1] / params.max_ships),
            inb[2],
            inb[3],
            inb[4],
            inb[5],
        ],
        dtype=jnp.float32,
    )


def planet_local_topology_features(state: EnvState, src_idx: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    """统计 source 周围的最近距离和局部星球密度。"""
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


def build_relative_features(state: EnvState, src_idx: jnp.ndarray, inbound: jnp.ndarray, params: FeatureParams):
    """构建 source-target 相对特征、mask 和派兵档位（RELATIVE_DIM=8）。

    返回：(relative_features[P+1, 8], candidate_mask[P+1], amount_mask[P+1, B], candidate_ids[P+1], ship_buckets[B])
    - slot 0 固定为 no-op（不发射）
    - slot 1..P 对应所有星球（包括来源星球自身，不过 mask=False）
    amount mask 只负责约束正数派兵；是否值得以少打多交给策略和奖励学习。
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

    noop_features = jnp.zeros((1, RELATIVE_DIM), dtype=jnp.float32)
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
    proto = proto_score(src, planets) / 100.0          # Proto-V15 目标评分
    eta_max = dist / jnp.maximum(spd_max, 1e-6) / INBOUND_HORIZON  # 最快到达时间归一化
    defense_pressure = compute_defense_pressure(planets, inbound)  # 是否需要增援
    rel = jnp.stack(
        [
            is_source.astype(jnp.float32),
            dx / BOARD_SIZE,
            dy / BOARD_SIZE,
            dist / BOARD_SIZE,
            clip01((planets[:, 5] + 1.0) / jnp.maximum(src[5], 1.0)),
            clip01(src[5] / params.max_ships) * jnp.ones_like(dist),
            proto,
            eta_max,
        ],
        axis=-1,
    )
    relative_features = jnp.concatenate([noop_features, rel], axis=0)

    # target head 阶段先做静态目标合法性剪枝：
    # - 目标必须存活且不能是 source 自身；
    # - 增援友方不要求船数优势；
    # - 攻击敌方/中立时要求 source 当前船数大于目标驻军。
    attackable_by_ships = src[5] > planets[:, 5]
    target_valid = alive & (~is_source) & (is_friend | ((is_enemy | is_neutral) & attackable_by_ships))
    positive_amount = valid_buckets
    per_planet_amount = jnp.broadcast_to(positive_amount[None, :], (params.max_planets, params.ship_bucket_count))
    per_planet_mask = target_valid
    # slot 0 永远可选（no-op）
    mask = jnp.concatenate([jnp.asarray([True]), per_planet_mask])
    amount_mask = jnp.concatenate([jnp.ones((1, params.ship_bucket_count), dtype=bool), per_planet_amount], axis=0)
    amount_mask = amount_mask & mask[:, None]
    amount_mask = amount_mask.at[0, :].set(True)
    if params.target_filter_enabled:
        mask = apply_target_filter(mask, is_friend, is_enemy | is_neutral, proto, defense_pressure, params)
        amount_mask = amount_mask & mask[:, None]
        amount_mask = amount_mask.at[0, :].set(True)
    return relative_features, mask, amount_mask, candidate_ids, ship_buckets


def apply_target_filter(
    mask: jnp.ndarray,
    is_friend: jnp.ndarray,
    is_attack: jnp.ndarray,
    proto: jnp.ndarray,
    defense_pressure: jnp.ndarray,
    params: FeatureParams,
) -> jnp.ndarray:
    """可选目标过滤：保留 no-op、评分最高的进攻目标和最需要防守的友方目标。"""

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
    """构建全局态势特征，给 decoder/critic 提供局面摘要。"""
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


def predict_fleet_targets(state: EnvState, horizon: int = int(INBOUND_HORIZON)) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """预测每支舰队在前瞻窗口内第一个会撞到的星球。

    返回：
      target_idx: [F] 星球槽位下标，只有 target_valid=True 时有意义。
      target_valid: [F] 该舰队是否会在 horizon 内撞到存活星球。
      target_t: [F] 第一次命中的整数回合，用浮点数表示。
    """
    fleets = state.fleets
    planets = state.planets
    f_alive = state.fleet_alive
    p_alive = state.planet_alive
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
    fleet_target = jnp.clip(fleet_target, 0, planet_count - 1)
    return fleet_target, fleet_valid, fleet_first_t


def precompute_inbound(state: EnvState, params: FeatureParams) -> jnp.ndarray:
    """预估未来一段时间内每个星球会收到的我方/敌方舰队压力。

    返回 [P, 6]：
      my_ships, enemy_ships, my_eta_min, my_eta_mean, enemy_eta_min, enemy_eta_mean。
    普通星球、旋转星球和彗星都按 fast_env 的轨迹规则查表或预测。
    """

    fleets = state.fleets
    planets = state.planets
    p_alive = state.planet_alive
    planet_count = planets.shape[0]
    fleet_target, fleet_valid, fleet_first_t = predict_fleet_targets(state)
    target_onehot = jax.nn.one_hot(fleet_target, planet_count, dtype=jnp.float32)
    selected = target_onehot * fleet_valid[:, None].astype(jnp.float32)
    owner = fleets[:, 1].astype(jnp.int32)
    my_sel = selected * (owner[:, None] == 0).astype(jnp.float32)
    enemy_sel = selected * (owner[:, None] != 0).astype(jnp.float32)

    my_ships = (my_sel * fleets[:, 6:7]).sum(axis=0)
    enemy_ships = (enemy_sel * fleets[:, 6:7]).sum(axis=0)
    norm = INBOUND_HORIZON
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


def build_fleet_features(state: EnvState, params: FeatureParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    """构建单局 top-K 舰队 token，用于 Transformer 编码。"""
    fleets = state.fleets
    alive = state.fleet_alive
    fleet_target, fleet_target_valid, fleet_target_t = predict_fleet_targets(state)
    dist_center = jnp.sqrt((fleets[:, 2] - CENTER) ** 2 + (fleets[:, 3] - CENTER) ** 2)
    if params.fleet_selection == "impact":
        score = fleet_impact_score(state, params, fleet_target, fleet_target_valid, fleet_target_t, dist_center)
    else:
        score = jnp.where(alive, -fleets[:, 6] + 0.01 * dist_center, jnp.inf)
    order = jnp.argsort(score)[: params.max_fleet_tokens]
    sel = fleets[order]
    sel_alive = alive[order]
    owner = sel[:, 1].astype(jnp.int32)
    from_id = sel[:, 5].astype(jnp.int32)
    planet_match = (state.planets[:, None, 0].astype(jnp.int32) == from_id[None, :]) & state.planet_alive[:, None]
    from_slot = jnp.argmax(planet_match, axis=0)
    has_from = planet_match.any(axis=0)
    from_planets = state.planets[from_slot]
    speed = fleet_speed(jnp.maximum(sel[:, 6], 1.0))
    target_idx = fleet_target[order]
    target_valid = sel_alive & fleet_target_valid[order]
    target_planets = state.planets[target_idx]
    target_owner = target_planets[:, 1].astype(jnp.int32)
    target_dx = target_planets[:, 2] - sel[:, 2]
    target_dy = target_planets[:, 3] - sel[:, 3]
    target_dist = jnp.sqrt(target_dx * target_dx + target_dy * target_dy)
    target_eta = fleet_target_t[order] / INBOUND_HORIZON

    planet_owner = state.planets[:, 1].astype(jnp.int32)
    planet_alive = state.planet_alive
    dx_planet = state.planets[:, None, 2] - sel[None, :, 2]
    dy_planet = state.planets[:, None, 3] - sel[None, :, 3]
    dist_planet = jnp.sqrt(dx_planet * dx_planet + dy_planet * dy_planet)

    def nearest_dist(mask):
        d = jnp.where(mask[:, None], dist_planet, jnp.inf).min(axis=0)
        return jnp.where(jnp.isfinite(d), d / BOARD_SIZE, 1.0)

    dist_from_origin = jnp.sqrt((sel[:, 2] - from_planets[:, 2]) ** 2 + (sel[:, 3] - from_planets[:, 3]) ** 2)
    launch_offset = from_planets[:, 4] + 0.1
    launch_age = jnp.maximum(dist_from_origin - launch_offset, 0.0) / jnp.maximum(speed, 1e-6)
    features = jnp.stack(
        [
            (owner == 0).astype(jnp.float32),
            sel[:, 2] / BOARD_SIZE,
            sel[:, 3] / BOARD_SIZE,
            jnp.cos(sel[:, 4]),
            jnp.sin(sel[:, 4]),
            clip01(sel[:, 6] / params.max_ships),
            speed / 6.0,
            clip01(launch_age / params.episode_steps),
            jnp.where(has_from, from_planets[:, 2] / BOARD_SIZE, 0.0),
            jnp.where(has_from, from_planets[:, 3] / BOARD_SIZE, 0.0),
            (has_from & (from_planets[:, 1].astype(jnp.int32) == 0)).astype(jnp.float32),
            jnp.where(has_from, clip01(from_planets[:, 5] / params.max_ships), 0.0),
            jnp.where(has_from, dist_from_origin / BOARD_SIZE, 0.0),
            nearest_dist(planet_alive & (planet_owner == 1)),
            nearest_dist(planet_alive & (planet_owner == -1)),
            nearest_dist(planet_alive & (planet_owner == 0)),
            target_valid.astype(jnp.float32),
            (target_valid & (target_owner == 0)).astype(jnp.float32),
            (target_valid & (target_owner == 1)).astype(jnp.float32),
            (target_valid & (target_owner == -1)).astype(jnp.float32),
            jnp.where(target_valid, target_planets[:, 2] / BOARD_SIZE, 0.0),
            jnp.where(target_valid, target_planets[:, 3] / BOARD_SIZE, 0.0),
            jnp.where(target_valid, clip01(target_planets[:, 5] / params.max_ships), 0.0),
            jnp.where(target_valid, target_planets[:, 6] / params.max_production, 0.0),
            jnp.where(target_valid, target_dx / BOARD_SIZE, 0.0),
            jnp.where(target_valid, target_dy / BOARD_SIZE, 0.0),
            jnp.where(target_valid, target_dist / BOARD_SIZE, 0.0),
            jnp.where(target_valid, target_eta, 0.0),
        ],
        axis=-1,
    )
    return jnp.where(sel_alive[:, None], features, 0.0), sel_alive


def fleet_impact_score(
    state: EnvState,
    params: FeatureParams,
    fleet_target: jnp.ndarray,
    fleet_target_valid: jnp.ndarray,
    fleet_target_t: jnp.ndarray,
    dist_center: jnp.ndarray,
) -> jnp.ndarray:
    """2P-Improve fleet top-K：优先保留威胁、进攻机会、紧迫 ETA 和高价值目标。"""

    fleets = state.fleets
    alive = state.fleet_alive
    owner = fleets[:, 1].astype(jnp.int32)
    safe_target = jnp.clip(fleet_target, 0, state.planets.shape[0] - 1)
    target_planets = state.planets[safe_target]
    target_owner = target_planets[:, 1].astype(jnp.int32)
    target_valid = alive & fleet_target_valid

    target_is_mine = target_valid & (target_owner == 0)
    target_is_enemy = target_valid & (target_owner == 1)
    target_is_neutral = target_valid & (target_owner == -1)
    is_my_fleet = owner == 0
    is_enemy_fleet = owner == 1

    eta_norm = jnp.clip(fleet_target_t / INBOUND_HORIZON, 0.0, 1.0)
    urgency = jnp.where(target_valid, 1.0 - eta_norm, 0.0)
    ships_norm = clip01(fleets[:, 6] / params.max_ships)
    target_prod_norm = jnp.where(target_valid, target_planets[:, 6] / params.max_production, 0.0)
    target_ships_norm = jnp.where(target_valid, clip01(target_planets[:, 5] / params.max_ships), 0.0)

    enemy_to_my = is_enemy_fleet & target_is_mine
    my_to_attack = is_my_fleet & (target_is_enemy | target_is_neutral)
    enemy_to_neutral = is_enemy_fleet & target_is_neutral
    my_reinforce = is_my_fleet & target_is_mine
    importance = (
        3.0 * enemy_to_my.astype(jnp.float32)
        + 1.6 * my_to_attack.astype(jnp.float32)
        + 0.8 * enemy_to_neutral.astype(jnp.float32)
        + 0.7 * my_reinforce.astype(jnp.float32)
        + 1.2 * urgency
        + 1.0 * ships_norm
        + 0.8 * target_prod_norm
        + 0.4 * target_ships_norm
    )
    return jnp.where(alive, -importance + 1.0e-4 * dist_center / BOARD_SIZE, jnp.inf)


def build_map_features(state: EnvState, src_idx: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    """构建 source 周围的 8x8 小地图，作为 src head 的局部空间提示。"""
    grid = jnp.zeros((MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE), dtype=jnp.float32)
    planets = state.planets
    owner = planets[:, 1].astype(jnp.int32)
    idx = jnp.arange(params.max_planets)
    channels = jnp.where(idx == src_idx, 0, jnp.where(owner == 0, 1, jnp.where(owner == -1, 3, 2)))
    values = state.planet_alive.astype(jnp.float32)
    grid = _accum_many(grid, channels, planets[:, 2], planets[:, 3], values)
    ship_channels = jnp.where(state.fleets[:, 1].astype(jnp.int32) == 0, 4, 5)
    ship_values = clip01(state.fleets[:, 6] / params.max_ships) * state.fleet_alive.astype(jnp.float32)
    return _accum_many(grid, ship_channels, state.fleets[:, 2], state.fleets[:, 3], ship_values)


def _accum_many(grid, channels, x, y, values):
    in_bounds = (x >= 0.0) & (x <= BOARD_SIZE) & (y >= 0.0) & (y <= BOARD_SIZE) & (values > 0.0)
    ix = jnp.clip((x / BOARD_SIZE * MAP_GRID_SIZE).astype(jnp.int32), 0, MAP_GRID_SIZE - 1)
    iy = jnp.clip((y / BOARD_SIZE * MAP_GRID_SIZE).astype(jnp.int32), 0, MAP_GRID_SIZE - 1)
    grid = grid.at[channels, iy, ix].add(jnp.where(in_bounds, values, 0.0))
    return jnp.clip(grid, 0.0, 1.0)


def build_ship_buckets(src_ships: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    """按当前 source 船数生成可选派兵档位；不足最小发射数则全为 0。"""
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
    b, p = encoded.planet_features.shape[:2]

    def flat(x):
        return x.reshape((b * p,) + x.shape[2:])

    return EncodedBatch(
        planet_features=flat(encoded.planet_features),
        relative_features=flat(encoded.relative_features),
        global_features=flat(encoded.global_features),
        map_features=flat(encoded.map_features),
        candidate_mask=flat(encoded.candidate_mask),
        amount_mask=flat(encoded.amount_mask),
        source_ids=flat(encoded.source_ids),
        source_ships=flat(encoded.source_ships),
        candidate_ids=flat(encoded.candidate_ids),
        ship_buckets=flat(encoded.ship_buckets),
        source_mask=flat(encoded.source_mask),
        fleet_features=flat(encoded.fleet_features),
        fleet_mask=flat(encoded.fleet_mask),
    )


def unflatten_actions(flat_actions: jnp.ndarray, batch_size: int, max_planets: int) -> jnp.ndarray:
    return flat_actions.reshape((batch_size, max_planets) + flat_actions.shape[1:])
