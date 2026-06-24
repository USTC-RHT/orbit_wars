"""JAX 未来驻军投影。

这个模块复刻 producer agent 中最有用的 ``PlanetGarrisonStatus`` 思路：
在“不新增发射”的假设下，预测短窗口内每颗星球的 owner/ships，并按 owner 汇总
每步抵达的舰队。BC fight 特征用它判断防守风险和安全抽兵量。
"""

from __future__ import annotations

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
    SUN_RADIUS,
    EnvState,
    circle_collision_hit,
    comet_planet_mask,
    fleet_speed,
    point_segment_distance_sq,
)


DEFAULT_GARRISON_HORIZON = 18
PLAYER_COUNT = 2


class GarrisonStatus(NamedTuple):
    """未来星球 owner 与驻军船数。

    时间轴长度为 ``H + 1``：``k=0`` 是当前观测，``k=1..H`` 是无新增发射假设下
    生产和战斗结算后的未来状态。``pre_combat_*`` 保存同回合到达结算前的状态。
    ``arrivals_by_owner`` 左侧补了 ``k=0`` 的零帧，方便和 ``owner`` / ``ships``
    对齐。
    """

    owner: jnp.ndarray
    ships: jnp.ndarray
    pre_combat_owner: jnp.ndarray
    pre_combat_ships: jnp.ndarray
    arrivals_by_owner: jnp.ndarray
    alive: jnp.ndarray


class FlipStatus(NamedTuple):
    """预测窗口内可能不再由我方持有的星球。"""

    will_flip: jnp.ndarray
    flip_turn: jnp.ndarray
    urgency: jnp.ndarray


def garrison_status_jax(
    state: EnvState,
    horizon: int = DEFAULT_GARRISON_HORIZON,
    player_count: int = PLAYER_COUNT,
) -> GarrisonStatus:
    """返回 producer 风格的 no-op 驻军预测。

    支持单局 ``EnvState`` 或批量 ``EnvState``。``horizon`` 和 ``player_count``
    预期作为 JIT 静态参数。
    """

    if state.planets.ndim == 3:
        return jax.vmap(lambda s: garrison_status_jax(s, horizon, player_count))(state)
    if int(player_count) != PLAYER_COUNT:
        raise ValueError("garrison_status_jax currently supports the 2-player JAX env")

    horizon = int(horizon)
    positions = planet_positions_and_alive_over_horizon(state, horizon)
    arrivals = fleet_arrivals_by_owner_jax(state, horizon, player_count, positions)
    return garrison_status_from_arrivals_jax(state, arrivals, positions.alive)


class FuturePlanetPositions(NamedTuple):
    """现有星球/彗星槽位的未来位置和有效性。"""

    x: jnp.ndarray      # [P, H+1]：未来 x 坐标
    y: jnp.ndarray      # [P, H+1]：未来 y 坐标
    alive: jnp.ndarray  # [P, H+1]：该时刻槽位是否有效
    comet: jnp.ndarray  # [P]：槽位是否为彗星


def planet_positions_and_alive_over_horizon(state: EnvState, horizon: int) -> FuturePlanetPositions:
    """预测现有槽位在 ``0..horizon`` 的位置和 alive mask。

    这里刻意不合成未来新刷出的彗星；只对当前已有星球和活跃彗星槽位投影，
    移动规则与 ``fast_env`` 和 aiming 代码保持一致。
    """

    planets = state.planets
    init = state.initial_planets
    turns = jnp.arange(horizon + 1, dtype=jnp.float32)
    turns_i = jnp.arange(horizon + 1, dtype=jnp.int32)
    comet_mask = comet_planet_mask(state)

    dx = init[:, 2] - CENTER
    dy = init[:, 3] - CENTER
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    rotating = (orbit_r + init[:, 4] < ROTATION_RADIUS_LIMIT) & state.planet_alive & (~comet_mask)
    init_angle = jnp.arctan2(dy, dx)
    future_step = state.step.astype(jnp.float32) + turns - 1.0
    phi = init_angle[:, None] + state.angular_velocity * future_step[None, :]
    orbit_px = CENTER + orbit_r[:, None] * jnp.cos(phi)
    orbit_py = CENTER + orbit_r[:, None] * jnp.sin(phi)
    orbit_px = orbit_px.at[:, 0].set(planets[:, 2])
    orbit_py = orbit_py.at[:, 0].set(planets[:, 3])
    px = orbit_px
    py = orbit_py
    px = jnp.where(rotating[:, None], px, planets[:, 2:3])
    py = jnp.where(rotating[:, None], py, planets[:, 3:4])
    alive = jnp.broadcast_to(state.planet_alive[:, None], (planets.shape[0], horizon + 1))

    comet_slots = state.comet_slots.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    safe_slots = jnp.where(comet_slots >= 0, comet_slots, 0)
    group_active = state.comet_path_index[:, None] >= -1
    active = (state.comet_slots >= 0) & group_active
    active_flat = active.reshape(-1)

    path_index = state.comet_path_index[:, None, None] + turns_i[None, None, :]
    path_index = jnp.broadcast_to(path_index, (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1))
    lengths = state.comet_lengths[:, :, None]
    valid_t = active[:, :, None] & (path_index >= 0) & (path_index < lengths)
    path_index = jnp.clip(path_index, 0, MAX_COMET_PATH - 1)
    gather_index = jnp.broadcast_to(path_index[..., None], (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1, 2))
    path_pos = jnp.take_along_axis(state.comet_paths, gather_index, axis=2)

    current_xy = state.planets[safe_slots, 2:4].reshape(MAX_COMET_SPAWNS, COMETS_PER_SPAWN, 1, 2)
    path_pos = jnp.where(valid_t[..., None], path_pos, current_xy)
    flat_pos = path_pos.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN, horizon + 1, 2)
    flat_valid = valid_t.reshape(MAX_COMET_SPAWNS * COMETS_PER_SPAWN, horizon + 1)

    slot_onehot = jax.nn.one_hot(safe_slots, planets.shape[0], dtype=jnp.float32) * active_flat[:, None].astype(jnp.float32)
    comet_has = slot_onehot.sum(axis=0) > 0.0
    comet_px = jnp.einsum("kp,kt->pt", slot_onehot, flat_pos[:, :, 0])
    comet_py = jnp.einsum("kp,kt->pt", slot_onehot, flat_pos[:, :, 1])
    comet_alive = jnp.einsum("kp,kt->pt", slot_onehot, flat_valid.astype(jnp.float32)) > 0.0
    comet_alive = comet_alive.at[:, 0].set(jnp.where(comet_has, state.planet_alive, comet_alive[:, 0]))

    px = jnp.where(comet_has[:, None], comet_px, px)
    py = jnp.where(comet_has[:, None], comet_py, py)
    alive = jnp.where(comet_has[:, None], comet_alive, alive)
    return FuturePlanetPositions(x=px, y=py, alive=alive, comet=comet_has)


def fleet_arrivals_by_owner_jax(
    state: EnvState,
    horizon: int,
    player_count: int = PLAYER_COUNT,
    positions: FuturePlanetPositions | None = None,
) -> jnp.ndarray:
    """把当前舰队的首次未来命中按 ``[P, H+1, A]`` 汇总。"""

    if state.planets.ndim == 3:
        return jax.vmap(lambda s: fleet_arrivals_by_owner_jax(s, horizon, player_count, None))(state)
    if int(player_count) != PLAYER_COUNT:
        raise ValueError("fleet_arrivals_by_owner_jax currently supports the 2-player JAX env")

    positions = positions if positions is not None else planet_positions_and_alive_over_horizon(state, horizon)
    target, valid, first_t = predict_fleet_first_contacts_jax(state, horizon, positions)
    planets = state.planets
    fleets = state.fleets
    owner = fleets[:, 1].astype(jnp.int32)
    owner_valid = (owner >= 0) & (owner < player_count)
    valid = valid & owner_valid

    target_oh = jax.nn.one_hot(jnp.clip(target, 0, planets.shape[0] - 1), planets.shape[0], dtype=jnp.float32)
    step_oh = jax.nn.one_hot(jnp.clip(first_t, 0, horizon), horizon + 1, dtype=jnp.float32)
    owner_oh = jax.nn.one_hot(jnp.clip(owner, 0, player_count - 1), player_count, dtype=jnp.float32)
    weights = jnp.where(valid, fleets[:, 6], 0.0)
    arrivals = jnp.einsum("fp,fh,fa,f->pha", target_oh, step_oh, owner_oh, weights)
    return arrivals.at[:, 0, :].set(0.0)


def predict_fleet_first_contacts_jax(
    state: EnvState,
    horizon: int,
    positions: FuturePlanetPositions | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """预测无新增发射假设下，每支舰队首次接触的星球。

    返回 ``(target_idx, target_valid, first_step)``。同一回合内的目标接触顺序与
    ``fast_env`` 一致：舰队移动命中、旋转星球 sweep、彗星 sweep。
    出界和撞太阳是致命但无目标的事件。
    """

    positions = positions if positions is not None else planet_positions_and_alive_over_horizon(state, horizon)
    fleets = state.fleets
    planets = state.planets
    planet_count = planets.shape[0]

    steps = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    ts = steps.astype(jnp.float32)
    ships = jnp.maximum(fleets[:, 6], 1.0)
    speed = fleet_speed(ships)
    vx = jnp.cos(fleets[:, 4]) * speed
    vy = jnp.sin(fleets[:, 4]) * speed
    fx_prev = fleets[:, 2:3] + vx[:, None] * (ts[None, :] - 1.0)
    fy_prev = fleets[:, 3:4] + vy[:, None] * (ts[None, :] - 1.0)
    fx_cur = fleets[:, 2:3] + vx[:, None] * ts[None, :]
    fy_cur = fleets[:, 3:4] + vy[:, None] * ts[None, :]
    in_bounds = (fx_cur >= 0.0) & (fx_cur <= BOARD_SIZE) & (fy_cur >= 0.0) & (fy_cur <= BOARD_SIZE)
    sun_hit = circle_collision_hit(
        point_segment_distance_sq(CENTER, CENTER, fx_prev, fy_prev, fx_cur, fy_cur),
        SUN_RADIUS,
    )

    px_old = positions.x[:, :-1]
    py_old = positions.y[:, :-1]
    px_new = positions.x[:, 1:]
    py_new = positions.y[:, 1:]
    alive_start = positions.alive[:, :-1]
    alive_now = positions.alive[:, 1:]

    move_d2 = point_segment_distance_sq(
        px_old[None, :, :],
        py_old[None, :, :],
        fx_prev[:, None, :],
        fy_prev[:, None, :],
        fx_cur[:, None, :],
        fy_cur[:, None, :],
    )
    move_hit = circle_collision_hit(move_d2, planets[None, :, None, 4])
    move_hit = move_hit & alive_start[None, :, :] & state.fleet_alive[:, None, None]
    move_hit = move_hit & in_bounds[:, None, :] & (~sun_hit[:, None, :])
    any_move_hit = move_hit.any(axis=1)

    survived_move = state.fleet_alive[:, None] & in_bounds & (~sun_hit) & (~any_move_hit)

    sweep_x = px_new - px_old
    sweep_y = py_new - py_old
    sweep_active = (sweep_x * sweep_x + sweep_y * sweep_y) > 1e-9
    rotating_sweep_active = sweep_active & alive_now & (~positions.comet[:, None])
    sweep_d2 = point_segment_distance_sq(
        fx_cur[:, None, :],
        fy_cur[:, None, :],
        px_old[None, :, :],
        py_old[None, :, :],
        px_new[None, :, :],
        py_new[None, :, :],
    )
    rotating_hit = circle_collision_hit(sweep_d2, planets[None, :, None, 4])
    rotating_hit = rotating_hit & rotating_sweep_active[None, :, :] & survived_move[:, None, :]
    any_rotating_hit = rotating_hit.any(axis=1)

    survived_rotating = survived_move & (~any_rotating_hit)
    comet_sweep_active = sweep_active & alive_now & positions.comet[:, None]
    comet_hit = circle_collision_hit(sweep_d2, planets[None, :, None, 4])
    comet_hit = comet_hit & comet_sweep_active[None, :, :] & survived_rotating[:, None, :]

    large = jnp.int32(4 * (horizon + 1) + 1)
    lethal_keys = jnp.where(
        state.fleet_alive[:, None] & ((~in_bounds) | sun_hit),
        4 * steps[None, :],
        large,
    )
    first_lethal_key = lethal_keys.min(axis=1)

    move_keys = (4 * steps).reshape(1, 1, horizon)
    rotating_keys = (4 * steps + 1).reshape(1, 1, horizon)
    comet_keys = (4 * steps + 2).reshape(1, 1, horizon)
    target_keys = jnp.minimum(
        jnp.where(move_hit, move_keys, large),
        jnp.minimum(
            jnp.where(rotating_hit, rotating_keys, large),
            jnp.where(comet_hit, comet_keys, large),
        ),
    )
    first_per_planet = target_keys.min(axis=2)
    first_target_key = first_per_planet.min(axis=1)
    target_idx = first_per_planet.argmin(axis=1)
    valid = (first_target_key < large) & (first_target_key < first_lethal_key)
    first_t = first_target_key // 4
    return jnp.clip(target_idx, 0, planet_count - 1), valid, jnp.clip(first_t, 0, horizon)


def garrison_status_from_arrivals_jax(
    state: EnvState,
    arrivals_by_owner: jnp.ndarray,
    alive_by_step: jnp.ndarray,
) -> GarrisonStatus:
    """基于预先计算的到达量，执行生产 -> 战斗的递推结算。"""

    horizon = int(arrivals_by_owner.shape[-2]) - 1
    planets = state.planets
    owner0 = planets[:, 1].astype(jnp.int32)
    ships0 = planets[:, 5].astype(jnp.float32)
    alive0 = alive_by_step[:, 0]
    owner0 = jnp.where(alive0, owner0, -1)
    ships0 = jnp.where(alive0, ships0, 0.0)
    prod = planets[:, 6].astype(jnp.float32)

    def body(carry, k):
        owner, ships = carry
        alive_before = alive_by_step[:, k - 1]
        alive_now = alive_by_step[:, k]
        produces = alive_before & (owner >= 0)
        ships = ships + jnp.where(produces, prod, 0.0)

        pre_owner = jnp.where(alive_now, owner, -1)
        pre_ships = jnp.where(alive_now, ships, 0.0)

        combat0 = arrivals_by_owner[:, k, 0]
        combat1 = arrivals_by_owner[:, k, 1]
        has0 = combat0 > 0.0
        has1 = combat1 > 0.0
        top_owner = jnp.where(combat0 >= combat1, 0, 1)
        top_ships = jnp.maximum(combat0, combat1)
        second_ships = jnp.minimum(combat0, combat1)
        survivor = jnp.where(has0 & has1, top_ships - second_ships, top_ships)
        survivor_positive = (has0 | has1) & (survivor > 0.0) & alive_now

        same_owner = pre_owner == top_owner
        new_ships_same = pre_ships + survivor
        new_ships_attack = pre_ships - survivor
        captured = new_ships_attack < 0.0
        next_owner = jnp.where(survivor_positive & (~same_owner) & captured, top_owner, pre_owner)
        next_ships = jnp.where(
            survivor_positive,
            jnp.where(same_owner, new_ships_same, jnp.where(captured, -new_ships_attack, new_ships_attack)),
            pre_ships,
        )
        next_owner = jnp.where(alive_now, next_owner, -1)
        next_ships = jnp.where(alive_now, next_ships, 0.0)
        return (next_owner.astype(jnp.int32), next_ships), (next_owner.astype(jnp.int32), next_ships, pre_owner, pre_ships)

    ks = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    (_owner, _ships), (owners, ships, pre_owners, pre_ships) = jax.lax.scan(body, (owner0, ships0), ks)
    owner_hist = jnp.concatenate([owner0[:, None], owners.T], axis=1)
    ships_hist = jnp.concatenate([ships0[:, None], ships.T], axis=1)
    pre_owner_hist = jnp.concatenate([owner0[:, None], pre_owners.T], axis=1)
    pre_ships_hist = jnp.concatenate([ships0[:, None], pre_ships.T], axis=1)
    return GarrisonStatus(
        owner=owner_hist,
        ships=ships_hist,
        pre_combat_owner=pre_owner_hist,
        pre_combat_ships=pre_ships_hist,
        arrivals_by_owner=arrivals_by_owner,
        alive=alive_by_step,
    )


def safe_drain_jax(
    status: GarrisonStatus,
    source_idx: jnp.ndarray,
    source_ships: jnp.ndarray | None = None,
    h_eff: float | jnp.ndarray | None = None,
    player_id: int = 0,
) -> jnp.ndarray:
    """在 ``h_eff`` 窗口内仍能守住时，来源最多可抽走的船数。"""

    ships = status.ships
    owner = status.owner
    horizon = int(ships.shape[-1]) - 1
    src = jnp.clip(source_idx.astype(jnp.int32), 0, ships.shape[-2] - 1)
    src_ships = ships[src, 0] if source_ships is None else source_ships.astype(jnp.float32)
    if horizon == 0:
        return jnp.zeros_like(src_ships)

    traj_ships = ships[src, 1:]
    traj_owner = owner[src, 1:]
    turns = jnp.arange(1, horizon + 1, dtype=jnp.float32)
    h_limit = jnp.asarray(horizon if h_eff is None else h_eff, dtype=jnp.float32)
    held = (traj_owner == int(player_id)) & (traj_ships > 0.0) & (turns <= h_limit)
    min_slack = jnp.where(held, traj_ships, jnp.inf).min(axis=-1)
    cap = jnp.where(jnp.isfinite(min_slack), min_slack, src_ships)
    return jnp.minimum(cap, src_ships).clip(min=0.0)


def safe_drain_all_sources_jax(
    status: GarrisonStatus,
    source_ships: jnp.ndarray | None = None,
    h_eff: float | jnp.ndarray | None = None,
    player_id: int = 0,
) -> jnp.ndarray:
    """每个星球槽位的安全抽兵预算。

    单局返回 ``[P]``，批量返回 ``[B, P]``。非来源星球是否可用交给调用方的
    source mask 判断；这个 helper 只计算 producer 风格的最大可移除驻军。
    """

    if status.ships.ndim == 3:
        if source_ships is None:
            return jax.vmap(lambda s: safe_drain_all_sources_jax(s, None, h_eff, player_id))(status)
        return jax.vmap(lambda s, ships: safe_drain_all_sources_jax(s, ships, h_eff, player_id))(status, source_ships)

    source_idx = jnp.arange(status.ships.shape[0], dtype=jnp.int32)
    return safe_drain_jax(status, source_idx, source_ships, h_eff=h_eff, player_id=player_id)


def target_flip_status_from_garrison_jax(
    state: EnvState,
    status: GarrisonStatus,
    player_id: int = 0,
) -> FlipStatus:
    """从 no-op 驻军预测中提取 producer 风格防守目标信号。"""

    if status.owner.ndim == 3:
        return jax.vmap(lambda s, st: target_flip_status_from_garrison_jax(s, st, player_id))(state, status)

    horizon = int(status.owner.shape[-1]) - 1
    current_owner = state.planets[:, 1].astype(jnp.int32)
    owned_now = state.planet_alive & (current_owner == int(player_id))
    if horizon == 0:
        return FlipStatus(
            will_flip=jnp.zeros_like(owned_now),
            flip_turn=jnp.ones_like(current_owner),
            urgency=jnp.zeros_like(state.planets[:, 5]),
        )

    future_owner = status.owner[:, 1:]
    flips = owned_now[:, None] & (future_owner != int(player_id))
    will_flip = flips.any(axis=-1)
    first_flip = jnp.argmax(flips.astype(jnp.int32), axis=-1).astype(jnp.int32) + 1
    flip_turn = jnp.where(will_flip, first_flip, horizon + 1)
    remaining = jnp.maximum(jnp.asarray(horizon, dtype=jnp.float32) - flip_turn.astype(jnp.float32), 0.0)
    urgency_raw = state.planets[:, 5].astype(jnp.float32) + state.planets[:, 6].astype(jnp.float32) * remaining
    urgency = jnp.where(will_flip, urgency_raw, 0.0)
    return FlipStatus(will_flip=will_flip, flip_turn=flip_turn, urgency=urgency)


def target_flip_status_jax(
    state: EnvState,
    horizon: int = DEFAULT_GARRISON_HORIZON,
    player_id: int = 0,
) -> FlipStatus:
    """找出 ``horizon`` 内预计会从我方手里丢失的当前友方星球。"""

    if state.planets.ndim == 3:
        return jax.vmap(lambda s: target_flip_status_jax(s, horizon, player_id))(state)
    status = garrison_status_jax(state, horizon=horizon)
    return target_flip_status_from_garrison_jax(state, status, player_id=player_id)
