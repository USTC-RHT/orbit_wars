"""JAX 特征编码器。

把 EnvState 转换为神经网络固定形状输入。

设计原则：
- 每颗我方星球独立对应一行决策（source）。
- 每行决策包含：planet_features（星球自身）、relative_features（source-target 相对特征）、
  global_features（全局态势）、map_features（CNN 小地图）。
- 特征维度：PLANET=28 / RELATIVE=22 / GLOBAL=11。
- inbound 特征使用 garrison_jax 的短期未来驻军/到达表。
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
from .garrison_jax import (
    DEFAULT_GARRISON_HORIZON,
    GarrisonStatus,
    garrison_status_jax,
    safe_drain_all_sources_jax,
    target_flip_status_from_garrison_jax,
)
from .pressure_jax import enemy_pressure_jax


# 特征向量维度（JAX 主训练路径使用；修改时需要同步 model/normalizer/docs）
PLANET_DIM = 28         # 星球局部特征维度
RELATIVE_DIM = 22    # source-target 相对特征维度
GLOBAL_DIM = 11       # 全局局势特征维度
FLEET_DIM = 28        # 单支舰队 token 特征维度
MAX_FLEET_TOKENS = 40 # 每个 source 选择进入 Transformer 的舰队 token 数
MAP_CHANNELS = 6      # CNN 小地图通道数（来源星/友方/敌方/中立/我方飞船强度/敌方飞船强度）
MAP_GRID_SIZE = 8     # CNN 小地图分辨率（8x8）
INBOUND_HORIZON = 60.0  # 预测舰队未来 N 回合内命中的星球
ATTACK_REACHABILITY_HORIZON = DEFAULT_GARRISON_HORIZON
MIN_LAUNCH = 10.0     # 最小发射船数：驻军不足此数时禁止全部档位
LOCAL_PLANET_RADIUS = 25.0  # 局部星球密度统计半径
SHIP_RATIOS = jnp.asarray([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=jnp.float32)  # 派兵比例档位
LAUNCH_SURFACE_OFFSET = 0.1
TARGET_HIT_SURFACE_OFFSET = 0.0
REACHABILITY_EPS = 1e-4
FRIENDLY_REGROUP_MAX_TIME = 7.0
FRIENDLY_REGROUP_PRESSURE_DELTA_MIN = 0.25
FRIENDLY_REGROUP_PRESSURE_WEIGHT = 1.0
CAPTURE_OVERHEAD = 1.0


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
    owned_mask: jnp.ndarray
    fleet_features: jnp.ndarray
    fleet_mask: jnp.ndarray


class FeatureAux(NamedTuple):
    """Expensive per-state feature context that can be reused across views."""

    garrison_status: GarrisonStatus
    enemy_pressure: jnp.ndarray


def target_count(params: FeatureParams) -> int:
    """目标槽位数量：slot 0 是 no-op，后面是所有星球。"""
    return params.max_planets + 1


def encode_batch(state: EnvState, params: FeatureParams) -> EncodedBatch:
    """批量编码 EnvState，外层 batch 通过 vmap 并行。"""
    return jax.vmap(lambda s: encode_single(s, params))(state)


def encode_batch_with_aux(state: EnvState, params: FeatureParams, aux: FeatureAux) -> EncodedBatch:
    """Encode a batch while reusing precomputed per-state feature context."""
    return jax.vmap(lambda s, a: encode_single_with_aux(s, params, a))(state, aux)


def encode_single(state: EnvState, params: FeatureParams) -> EncodedBatch:
    """编码单局棋盘，把每个星球位置都组织成一个候选 source 行。"""
    return encode_single_with_aux(state, params, build_feature_aux(state, params))


def build_feature_aux(state: EnvState, params: FeatureParams) -> FeatureAux:
    """Compute expensive per-state feature context for one game."""
    del params
    garrison_status = garrison_status_jax(state, horizon=DEFAULT_GARRISON_HORIZON)
    enemy_pressure = enemy_pressure_jax(state, horizon=float(DEFAULT_GARRISON_HORIZON), player_id=0)
    return FeatureAux(garrison_status=garrison_status, enemy_pressure=enemy_pressure)


def encode_single_with_aux(state: EnvState, params: FeatureParams, aux: FeatureAux) -> EncodedBatch:
    """编码单局棋盘，把每个星球位置都组织成一个候选 source 行。"""
    planets = state.planets
    alive = state.planet_alive
    owner = planets[:, 1].astype(jnp.int32)
    owned_mask = alive & (owner == 0)
    source_ids = planets[:, 0].astype(jnp.int32)
    source_ships = planets[:, 5]
    global_f = build_global_features(state, params)
    garrison_status = aux.garrison_status
    safe_drain = safe_drain_all_sources_jax(
        garrison_status,
        source_ships,
        h_eff=float(DEFAULT_GARRISON_HORIZON),
        player_id=0,
    )
    inbound = precompute_inbound_from_garrison(state, garrison_status, params)
    fleet_f, fleet_m = build_fleet_features(state, params)
    future_px, future_py = planet_positions_over_horizon(state, ATTACK_REACHABILITY_HORIZON)
    enemy_pressure = aux.enemy_pressure
    if params.target_filter_enabled:
        flip_status = target_flip_status_from_garrison_jax(state, garrison_status, player_id=0)
        target_will_flip = flip_status.will_flip
        target_flip_turn = flip_status.flip_turn
        target_flip_urgency = flip_status.urgency
    else:
        target_will_flip = jnp.zeros((params.max_planets,), dtype=bool)
        target_flip_turn = jnp.full((params.max_planets,), DEFAULT_GARRISON_HORIZON + 1, dtype=jnp.int32)
        target_flip_urgency = jnp.zeros((params.max_planets,), dtype=jnp.float32)

    def source_row(src_idx):
        return build_source_decision(
            state, src_idx, global_f, inbound, safe_drain, fleet_f, fleet_m, future_px, future_py,
            garrison_status, target_will_flip, target_flip_turn, target_flip_urgency, enemy_pressure, params,
        )

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
        source_mask=rows.source_mask,
        owned_mask=owned_mask,
        fleet_features=rows.fleet_features,
        fleet_mask=rows.fleet_mask,
    )


def build_source_decision(
    state: EnvState,
    src_idx: jnp.ndarray,
    global_f: jnp.ndarray,
    inbound: jnp.ndarray,
    safe_drain: jnp.ndarray,
    fleet_f: jnp.ndarray,
    fleet_m: jnp.ndarray,
    future_px: jnp.ndarray,
    future_py: jnp.ndarray,
    garrison_status,
    target_will_flip: jnp.ndarray,
    target_flip_turn: jnp.ndarray,
    target_flip_urgency: jnp.ndarray,
    enemy_pressure: jnp.ndarray,
    params: FeatureParams,
) -> EncodedBatch:
    """构造某个 source 星球对应的一行决策输入。"""
    src = state.planets[src_idx]
    planet_f = build_planet_features(state, src_idx, inbound, safe_drain, params)
    rel_f, cand_mask, amount_mask, candidate_ids, ship_buckets = build_relative_features(
        state, src_idx, inbound, future_px, future_py,
        garrison_status, target_will_flip, target_flip_turn, target_flip_urgency, enemy_pressure, params
    )
    has_target = cand_mask[1:].any()
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
        source_mask=(src[1].astype(jnp.int32) == 0) & state.planet_alive[src_idx] & (src[5] >= MIN_LAUNCH) & has_target,
        owned_mask=(src[1].astype(jnp.int32) == 0) & state.planet_alive[src_idx],
        fleet_features=fleet_f,
        fleet_mask=fleet_m,
    )


def build_planet_features(
    state: EnvState,
    src_idx: jnp.ndarray,
    inbound: jnp.ndarray,
    safe_drain: jnp.ndarray,
    params: FeatureParams,
) -> jnp.ndarray:
    """构建星球局部特征（PLANET_DIM=28）。

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
      26: safe_drain_ratio = producer-style safe removable ships / current ships
      27: full_send_risk_ratio = full-send exceeds safe_drain by this source fraction
    """
    src = state.planets[src_idx]
    alive = state.planet_alive[src_idx]
    owner = src[1].astype(jnp.int32)
    is_comet = comet_planet_mask(state)[src_idx]
    is_rotating_planet = (is_rotating(src) > 0.5) & (~is_comet)
    is_static_planet = alive & (~is_rotating_planet) & (~is_comet)
    topo = planet_local_topology_features(state, src_idx, params)
    inb = inbound[src_idx]  # 预测的到达该星球的舰队信息
    safe_drain_ratio = clip01(safe_drain[src_idx] / jnp.maximum(src[5], 1.0))
    full_send_risk_ratio = clip01((src[5] - safe_drain[src_idx]) / jnp.maximum(src[5], 1.0))
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
            safe_drain_ratio,
            full_send_risk_ratio,
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


def build_relative_features(
    state: EnvState,
    src_idx: jnp.ndarray,
    inbound: jnp.ndarray,
    future_px: jnp.ndarray,
    future_py: jnp.ndarray,
    garrison_status,
    target_will_flip: jnp.ndarray,
    target_flip_turn: jnp.ndarray,
    target_flip_urgency: jnp.ndarray,
    enemy_pressure: jnp.ndarray,
    params: FeatureParams,
):
    """构建 source-target 相对特征、mask 和派兵档位（RELATIVE_DIM=22）。

    返回：(relative_features[P+1, 22], candidate_mask[P+1], amount_mask[P+1, B], candidate_ids[P+1], ship_buckets[B])
    - slot 0 固定为 no-op（不发射）
    - slot 1..P 对应所有星球（包括来源星球自身，不过 mask=False）
    amount mask 只负责约束正数派兵；是否值得以少打多交给策略和奖励学习。
    """
    src = state.planets[src_idx]
    planets = state.planets
    alive = state.planet_alive
    candidate_ids = jnp.concatenate([jnp.asarray([-1], dtype=jnp.int32), planets[:, 0].astype(jnp.int32)])
    ship_buckets = build_ship_buckets(src[5], params)

    valid_buckets = ship_buckets > 0
    max_bucket = jnp.maximum(ship_buckets.max(), 1.0)

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
    reachable_amount = reachable_amount_mask_jax(src, planets, future_px, future_py, ship_buckets)
    reachable_positive = reachable_amount & valid_buckets[None, :]
    target_reachable = reachable_positive.any(axis=-1)
    send_ships = max_bucket
    _full_reachable, cheap_eta_turns = reachable_full_send_eta_jax(src, planets, future_px, future_py, send_ships)
    attackable_by_ships = send_ships > planets[:, 5]
    target_valid = alive & (~is_source) & target_reachable & (is_friend | ((is_enemy | is_neutral) & attackable_by_ships))
    eta_turns = jnp.where(target_valid, cheap_eta_turns, INBOUND_HORIZON + 1.0)
    eta_max = eta_turns / INBOUND_HORIZON  # 最快到达时间归一化
    eta_garrison = relative_eta_garrison_features(garrison_status, eta_turns, params)
    target_ships_at_eta = eta_garrison[:, 3] * params.max_ships
    my_arrivals = eta_garrison[:, 4] * params.max_ships
    enemy_arrivals = eta_garrison[:, 5] * params.max_ships
    capture_margin_at_eta = jnp.clip((send_ships - target_ships_at_eta) / params.max_ships, -1.0, 1.0)
    pressure_delta = enemy_pressure - enemy_pressure[jnp.clip(src_idx, 0, enemy_pressure.shape[0] - 1)]
    pressure_delta_norm = jnp.clip(pressure_delta / params.max_ships, -1.0, 1.0)
    target_enemy_pressure_norm = clip01(enemy_pressure / params.max_ships)
    arrival_balance_at_eta = jnp.clip((my_arrivals - enemy_arrivals) / params.max_ships, -1.0, 1.0)
    contestedness_at_eta = clip01((my_arrivals + enemy_arrivals) / params.max_ships)
    capture_floor_at_eta, capture_surplus_at_eta = relative_capture_floor_features(
        garrison_status, eta_turns, send_ships, params, player_id=0
    )
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
    relative_features = jnp.concatenate([noop_features, rel], axis=0)

    # target head 阶段先做静态目标合法性剪枝：
    # - 目标必须存活且不能是 source 自身；
    # - 增援友方不要求船数优势；
    # - 攻击敌方/中立时要求 source 当前船数大于目标驻军。
    per_planet_amount = reachable_positive
    per_planet_mask = target_valid
    # slot 0 永远可选（no-op）
    mask = jnp.concatenate([jnp.asarray([True]), per_planet_mask])
    amount_mask = jnp.concatenate([jnp.ones((1, params.ship_bucket_count), dtype=bool), per_planet_amount], axis=0)
    amount_mask = amount_mask & mask[:, None]
    amount_mask = amount_mask.at[0, :].set(True)
    if params.target_filter_enabled:
        mask = apply_target_filter(
            mask, is_friend, is_enemy | is_neutral, proto,
            target_will_flip, target_flip_turn, target_flip_urgency,
            enemy_pressure, src_idx, eta_turns, params,
        )
        amount_mask = amount_mask & mask[:, None]
        amount_mask = amount_mask.at[0, :].set(True)
    return relative_features, mask, amount_mask, candidate_ids, ship_buckets


def apply_target_filter(
    mask: jnp.ndarray,
    is_friend: jnp.ndarray,
    is_attack: jnp.ndarray,
    proto: jnp.ndarray,
    target_will_flip: jnp.ndarray,
    target_flip_turn: jnp.ndarray,
    target_flip_urgency: jnp.ndarray,
    enemy_pressure: jnp.ndarray,
    src_idx: jnp.ndarray,
    eta_turns: jnp.ndarray,
    params: FeatureParams,
) -> jnp.ndarray:
    """可选目标过滤：先按类别分数取全量 top-k，再与粗合法 mask 取交集。"""

    planet_mask = mask[1:]
    keep_planets = jnp.zeros((params.max_planets,), dtype=bool)

    attack_score = jnp.where(is_attack, proto, -1e9)
    attack_order = jnp.argsort(attack_score)[::-1]
    attack_k = min(params.attack_top_k, params.max_planets)
    attack_keep = jnp.zeros((params.max_planets,), dtype=bool)
    attack_keep = attack_keep.at[attack_order[:attack_k]].set(attack_score[attack_order[:attack_k]] > -1e8)
    keep_planets = keep_planets | (planet_mask & attack_keep)

    defense_score_raw = friendly_defense_score_jax(
        target_will_flip,
        target_flip_turn,
        target_flip_urgency,
        enemy_pressure,
        src_idx,
        eta_turns,
    )
    defense_score = jnp.where(is_friend, defense_score_raw, -1e9)
    defense_order = jnp.argsort(defense_score)[::-1]
    defense_k = min(params.defense_top_k, params.max_planets)
    defense_keep = jnp.zeros((params.max_planets,), dtype=bool)
    defense_keep = defense_keep.at[defense_order[:defense_k]].set(defense_score[defense_order[:defense_k]] > 1e-6)
    keep_planets = keep_planets | (planet_mask & defense_keep)
    return jnp.concatenate([jnp.asarray([mask[0]]), keep_planets])


def friendly_defense_score_jax(
    target_will_flip: jnp.ndarray,
    target_flip_turn: jnp.ndarray,
    target_flip_urgency: jnp.ndarray,
    enemy_pressure: jnp.ndarray,
    src_idx: jnp.ndarray,
    eta_turns: jnp.ndarray,
) -> jnp.ndarray:
    """Source-conditioned friendly target score for rescue and pressure regroup."""

    flip_window = jnp.maximum(target_flip_turn.astype(jnp.float32), 1.0)
    flip_eta_decay = jnp.clip(1.0 - eta_turns / flip_window, 0.0, 1.0)
    flip_score = jnp.where(target_will_flip, target_flip_urgency * flip_eta_decay, 0.0)

    src_pressure = enemy_pressure[jnp.clip(src_idx, 0, enemy_pressure.shape[0] - 1)]
    pressure_delta = enemy_pressure - src_pressure
    regroup_eta_decay = jnp.clip(1.0 - eta_turns / FRIENDLY_REGROUP_MAX_TIME, 0.0, 1.0)
    regroup_score = (
        FRIENDLY_REGROUP_PRESSURE_WEIGHT
        * jnp.maximum(pressure_delta - FRIENDLY_REGROUP_PRESSURE_DELTA_MIN, 0.0)
        * regroup_eta_decay
    )
    return flip_score + regroup_score


def relative_eta_garrison_features(status, eta_turns: jnp.ndarray, params: FeatureParams) -> jnp.ndarray:
    """Target future-state features sampled at ``ceil(physics_eta_turns)``.

    Returned columns:
      0-2: target pre-combat owner at eta as neutral/mine/enemy one-hot.
      3: target pre-combat ships at eta / max_ships.
      4-5: cumulative my/enemy arrivals through eta / max_ships.
      6: eta is beyond the garrison horizon.
    """

    owner = status.pre_combat_owner
    ships = status.pre_combat_ships
    arrivals = status.arrivals_by_owner.astype(jnp.float32)
    horizon = int(owner.shape[-1]) - 1
    if horizon <= 0:
        return jnp.zeros((owner.shape[0], 7), dtype=jnp.float32)

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


def relative_capture_floor_features(
    status,
    eta_turns: jnp.ndarray,
    send_ships: jnp.ndarray,
    params: FeatureParams,
    player_id: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Producer-style owner-aware capture floor sampled at ``ceil(physics_eta)``."""

    owner = status.owner
    ships = status.ships
    horizon = int(owner.shape[-1]) - 1
    if horizon <= 0:
        floor = jnp.ones((owner.shape[0],), dtype=jnp.float32)
        return clip01(floor / params.max_ships), jnp.clip((send_ships - floor) / params.max_ships, -1.0, 1.0)

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


def reachable_full_send_eta_jax(
    src: jnp.ndarray,
    planets: jnp.ndarray,
    future_px: jnp.ndarray,
    future_py: jnp.ndarray,
    ships: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cheap full-send reachability and earliest approximate ETA, returned as ``[target]``."""

    horizon = int(future_px.shape[-1]) - 1
    if horizon <= 0:
        return (
            jnp.zeros((planets.shape[0],), dtype=bool),
            jnp.full((planets.shape[0],), INBOUND_HORIZON + 1.0, dtype=jnp.float32),
        )

    ax = future_px[:, :-1]
    ay = future_py[:, :-1]
    bx = future_px[:, 1:]
    by = future_py[:, 1:]
    d2 = point_segment_distance_sq(src[2], src[3], ax, ay, bx, by)
    seg_dist = jnp.sqrt(jnp.maximum(d2, 0.0))
    gap = src[4] + planets[:, 4] + LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET
    surface_dist = jnp.maximum(seg_dist - gap[:, None], 0.0)
    turns = jnp.arange(1, horizon + 1, dtype=jnp.float32)
    speed = fleet_speed(jnp.maximum(ships, 1.0))
    reachable_by_turn = surface_dist <= (speed * turns[None, :] * (1.0 + REACHABILITY_EPS))
    can_reach = (ships >= MIN_LAUNCH) & reachable_by_turn.any(axis=-1)
    eta = jnp.min(jnp.where(reachable_by_turn, turns[None, :], INBOUND_HORIZON + 1.0), axis=-1)
    return can_reach, jnp.where(can_reach, eta, INBOUND_HORIZON + 1.0)


def reachable_amount_mask_jax(
    src: jnp.ndarray,
    planets: jnp.ndarray,
    future_px: jnp.ndarray,
    future_py: jnp.ndarray,
    ship_buckets: jnp.ndarray,
) -> jnp.ndarray:
    """Producer-style cheap reachability mask, returned as ``[target, bucket]``."""

    horizon = int(future_px.shape[-1]) - 1
    if horizon <= 0:
        return jnp.zeros((planets.shape[0], ship_buckets.shape[0]), dtype=bool)

    ax = future_px[:, :-1]
    ay = future_py[:, :-1]
    bx = future_px[:, 1:]
    by = future_py[:, 1:]
    d2 = point_segment_distance_sq(src[2], src[3], ax, ay, bx, by)
    seg_dist = jnp.sqrt(jnp.maximum(d2, 0.0))
    gap = src[4] + planets[:, 4] + LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET
    surface_dist = jnp.maximum(seg_dist - gap[:, None], 0.0)
    turns = jnp.arange(1, horizon + 1, dtype=jnp.float32)
    min_required_speed = (surface_dist / turns[None, :]).min(axis=-1)
    bucket_speed = fleet_speed(jnp.maximum(ship_buckets, 1.0))
    return min_required_speed[:, None] <= bucket_speed[None, :] * (1.0 + REACHABILITY_EPS)


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


def precompute_inbound_from_garrison(state: EnvState, status, params: FeatureParams) -> jnp.ndarray:
    """Build the 6 inbound planet features from ``GarrisonStatus.arrivals_by_owner``.

    The shape and feature order match ``precompute_inbound``. Since garrison
    arrivals are already grouped by owner and step, mean ETA is ship-weighted.
    """

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
        owned_mask=flat(encoded.owned_mask),
        fleet_features=flat(encoded.fleet_features),
        fleet_mask=flat(encoded.fleet_mask),
    )


def unflatten_actions(flat_actions: jnp.ndarray, batch_size: int, max_planets: int) -> jnp.ndarray:
    return flat_actions.reshape((batch_size, max_planets) + flat_actions.shape[1:])
