"""Orbit Wars 固定形状 JAX 模拟器。

设计原则：
- 所有数组形状在编译时固定，支持 jax.vmap 批量并行和 jax.lax.scan 展开 rollout。
- 星球/舰队槽位预分配，通过 alive 掩码区分有效元素，避免动态列表。
- 支持预生成 state bank（.npz），复位时直接从 bank 随机抽取，消除 Python 初始化开销。
- 彗星使用预计算路径存储在 EnvState 中，step 时按 path_index 查表，无需调用 Python RNG。
- V1 范围：双人、无 wandb、无评估；彗星已支持（路径需通过 build_initial_state_bank 预计算）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


BOARD_SIZE = 100.0           # 地图边长（坐标范围 [0, 100]）
CENTER = 50.0                # 地图中心 = 太阳位置
SUN_RADIUS = 10.0            # 太阳碰撞半径：舰队轨迹进入此范围即被摧毁
ROTATION_RADIUS_LIMIT = 50.0 # 轨道半径 + 星球半径 < 此值则为旋转星球
DEFAULT_MAX_PLANETS = 64     # 星球槽位上限（含彗星占用的临时槽位）
DEFAULT_MAX_FLEETS = 512     # 舰队槽位上限
DEFAULT_MAX_ACTIONS = 64     # 每方每回合最多发射动作数
COMET_SPAWN_STEPS = jnp.asarray([50, 150, 250, 350, 450], dtype=jnp.int32)  # 彗星触发回合
COMET_RADIUS = 1.0           # 彗星星球半径
COMET_PRODUCTION = 1.0       # 彗星每回合生产力
MAX_COMET_SPAWNS = 5         # 最多 5 批次彗星（对应 5 个触发回合）
COMETS_PER_SPAWN = 4         # 每批次 4 颗彗星（4 折对称）
MAX_COMET_PATH = 40          # 彗星路径最长 40 步


class EnvState(NamedTuple):
    """单局完整状态。所有字段形状固定，可被 jax.vmap / jax.lax.scan 批量处理。"""
    planets: jnp.ndarray         # [P, 7]：id / owner / x / y / radius / ships / production
    initial_planets: jnp.ndarray # [P, 7]：初始星球位置，用于计算旋转轨道角度
    planet_alive: jnp.ndarray    # [P] bool：槽位是否存活（彗星过期后变 False）
    fleets: jnp.ndarray          # [F, 7]：id / owner / x / y / angle / from_id / ships
    fleet_alive: jnp.ndarray     # [F] bool：槽位是否存活
    step: jnp.ndarray            # 当前回合编号（标量）
    angular_velocity: jnp.ndarray  # 全图行星公转角速度（rad/回合）
    next_fleet_id: jnp.ndarray   # 下一支舰队的 ID（自增）
    done: jnp.ndarray            # 游戏是否已结束（标量 bool）
    winner: jnp.ndarray          # 胜者：0/1=某方胜，2=平局，-1=未结束
    overflow_fleets: jnp.ndarray # 因 fleet slot 满而丢弃的发射次数（调试用）
    comet_paths: jnp.ndarray     # [5, 4, 40, 2]：每批次每颗彗星的预计算路径坐标
    comet_lengths: jnp.ndarray   # [5, 4]：每颗彗星路径的实际有效步数
    comet_ships: jnp.ndarray     # [5]：每批次彗星的初始船数
    comet_valid: jnp.ndarray     # [5] bool：该批次是否有有效路径数据
    comet_path_index: jnp.ndarray  # [5]：当前路径索引；-2=未激活，-1=已生成未移动
    comet_slots: jnp.ndarray     # [5, 4]：每颗彗星占用的行星槽位索引；-1=无


@dataclass(frozen=True)
class EnvParams:
    """环境超参数（编译时固定，不随 state 变化）。"""
    episode_steps: int = 500           # 最大回合数
    board_size: float = BOARD_SIZE     # 地图边长
    ship_speed: float = 6.0            # 舰队最大速度（ships→∞ 时趋近此值）
    max_planets: int = DEFAULT_MAX_PLANETS  # 星球槽位数（含彗星临时槽位）
    max_fleets: int = DEFAULT_MAX_FLEETS    # 舰队槽位数
    max_actions: int = DEFAULT_MAX_ACTIONS  # 每方每回合动作槽位数


def fleet_speed(ships: jnp.ndarray, max_speed: float = 6.0) -> jnp.ndarray:
    ships_f = jnp.maximum(ships.astype(jnp.float32), 1.0)
    ratio = jnp.clip(jnp.log(ships_f) / jnp.log(1000.0), 0.0, 1.0)
    speed = 1.0 + (max_speed - 1.0) * (ratio**1.5)
    return jnp.minimum(speed, max_speed)


def point_segment_distance_sq(
    px: jnp.ndarray,
    py: jnp.ndarray,
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
) -> jnp.ndarray:
    dx = x1 - x0
    dy = y1 - y0
    denom = jnp.maximum(dx * dx + dy * dy, 1e-9)
    t = jnp.clip(((px - x0) * dx + (py - y0) * dy) / denom, 0.0, 1.0)
    cx = x0 + t * dx
    cy = y0 + t * dy
    ddx = px - cx
    ddy = py - cy
    return ddx * ddx + ddy * ddy


def empty_state(params: EnvParams = EnvParams()) -> EnvState:
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    return EnvState(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.zeros((params.max_planets,), dtype=bool),
        fleets=fleets,
        fleet_alive=jnp.zeros((params.max_fleets,), dtype=bool),
        step=jnp.array(0, dtype=jnp.int32),
        angular_velocity=jnp.array(0.0, dtype=jnp.float32),
        next_fleet_id=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False),
        winner=jnp.array(-1, dtype=jnp.int32),
        overflow_fleets=jnp.array(0, dtype=jnp.int32),
        comet_paths=jnp.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=jnp.float32),
        comet_lengths=jnp.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=jnp.int32),
        comet_ships=jnp.zeros((MAX_COMET_SPAWNS,), dtype=jnp.float32),
        comet_valid=jnp.zeros((MAX_COMET_SPAWNS,), dtype=bool),
        comet_path_index=jnp.full((MAX_COMET_SPAWNS,), -2, dtype=jnp.int32),
        comet_slots=jnp.full((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), -1, dtype=jnp.int32),
    )


def state_from_observation(observation: Any, params: EnvParams = EnvParams()) -> EnvState:
    """Convert a Kaggle observation object/dict into padded JAX arrays."""

    def read(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    state = empty_state(params)
    planets_np = np.zeros((params.max_planets, 7), dtype=np.float32)
    initial_np = np.zeros((params.max_planets, 7), dtype=np.float32)
    fleets_np = np.zeros((params.max_fleets, 7), dtype=np.float32)
    planet_alive = np.zeros((params.max_planets,), dtype=bool)
    fleet_alive = np.zeros((params.max_fleets,), dtype=bool)
    planet_id_to_slot: dict[int, int] = {}

    planets = list(read(observation, "planets", []) or [])[: params.max_planets]
    initial = list(read(observation, "initial_planets", planets) or planets)[: params.max_planets]
    fleets = list(read(observation, "fleets", []) or [])[: params.max_fleets]
    if planets:
        planets_np[: len(planets)] = np.asarray(planets, dtype=np.float32)
        planet_alive[: len(planets)] = True
        planet_id_to_slot = {int(row[0]): i for i, row in enumerate(planets)}
    if initial:
        initial_np[: len(initial)] = np.asarray(initial, dtype=np.float32)
    else:
        initial_np[:] = planets_np
    if fleets:
        fleets_np[: len(fleets)] = np.asarray(fleets, dtype=np.float32)
        fleet_alive[: len(fleets)] = True
    comet_paths, comet_lengths, comet_ships, comet_valid, comet_path_index, comet_slots = comet_arrays_from_observation(
        observation,
        planet_id_to_slot,
        planets_np,
    )

    return state._replace(
        planets=jnp.asarray(planets_np),
        initial_planets=jnp.asarray(initial_np),
        planet_alive=jnp.asarray(planet_alive),
        fleets=jnp.asarray(fleets_np),
        fleet_alive=jnp.asarray(fleet_alive),
        step=jnp.asarray(int(read(observation, "step", 0) or 0), dtype=jnp.int32),
        angular_velocity=jnp.asarray(float(read(observation, "angular_velocity", 0.0) or 0.0), dtype=jnp.float32),
        next_fleet_id=jnp.asarray(int(read(observation, "next_fleet_id", len(fleets)) or 0), dtype=jnp.int32),
        comet_paths=jnp.asarray(comet_paths),
        comet_lengths=jnp.asarray(comet_lengths),
        comet_ships=jnp.asarray(comet_ships),
        comet_valid=jnp.asarray(comet_valid),
        comet_path_index=jnp.asarray(comet_path_index),
        comet_slots=jnp.asarray(comet_slots),
    )


def comet_arrays_from_observation(
    observation: Any,
    planet_id_to_slot: dict[int, int],
    planets_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert live Kaggle comet groups into fixed JAX arrays.

    Kaggle observations keep comets as dynamic groups:
      {"planet_ids": [...], "paths": [[[x,y], ...], ...], "path_index": k}
    JAX features/aiming need the same data in fixed slots, keyed by the
    planet slot currently occupied by each comet.
    """

    def read(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    comet_paths = np.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=np.float32)
    comet_lengths = np.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=np.int32)
    comet_ships = np.zeros((MAX_COMET_SPAWNS,), dtype=np.float32)
    comet_valid = np.zeros((MAX_COMET_SPAWNS,), dtype=bool)
    comet_path_index = np.full((MAX_COMET_SPAWNS,), -2, dtype=np.int32)
    comet_slots = np.full((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), -1, dtype=np.int32)

    groups = list(read(observation, "comets", []) or [])[:MAX_COMET_SPAWNS]
    for group_idx, group in enumerate(groups):
        planet_ids = list(read(group, "planet_ids", []) or [])[:COMETS_PER_SPAWN]
        paths = list(read(group, "paths", []) or [])[:COMETS_PER_SPAWN]
        path_index = int(read(group, "path_index", -2) or 0)
        any_valid = False
        first_ships = 0.0
        for comet_idx, pid_raw in enumerate(planet_ids):
            pid = int(pid_raw)
            slot = planet_id_to_slot.get(pid, -1)
            if slot < 0:
                continue
            path = paths[comet_idx] if comet_idx < len(paths) else []
            arr = np.asarray(path[:MAX_COMET_PATH], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[-1] != 2 or arr.shape[0] == 0:
                continue
            length = min(arr.shape[0], MAX_COMET_PATH)
            comet_paths[group_idx, comet_idx, :length] = arr[:length]
            comet_lengths[group_idx, comet_idx] = length
            comet_slots[group_idx, comet_idx] = slot
            any_valid = True
            if first_ships <= 0.0:
                first_ships = float(planets_np[slot, 5])
        if any_valid:
            comet_valid[group_idx] = True
            comet_path_index[group_idx] = path_index
            comet_ships[group_idx] = first_ships
    return comet_paths, comet_lengths, comet_ships, comet_valid, comet_path_index, comet_slots


def load_state_bank(path: str | Path, params: EnvParams = EnvParams()) -> EnvState:
    data = np.load(path)
    n = data["planets"].shape[0]

    def optional(name: str, default: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(data[name] if name in data else default)

    return EnvState(
        planets=jnp.asarray(data["planets"], dtype=jnp.float32),
        initial_planets=jnp.asarray(data["initial_planets"], dtype=jnp.float32),
        planet_alive=jnp.asarray(data["planet_alive"], dtype=bool),
        fleets=jnp.asarray(data["fleets"], dtype=jnp.float32),
        fleet_alive=jnp.asarray(data["fleet_alive"], dtype=bool),
        step=jnp.asarray(data["step"], dtype=jnp.int32),
        angular_velocity=jnp.asarray(data["angular_velocity"], dtype=jnp.float32),
        next_fleet_id=jnp.asarray(data["next_fleet_id"], dtype=jnp.int32),
        done=jnp.zeros((data["planets"].shape[0],), dtype=bool),
        winner=jnp.full((data["planets"].shape[0],), -1, dtype=jnp.int32),
        overflow_fleets=jnp.zeros((data["planets"].shape[0],), dtype=jnp.int32),
        comet_paths=optional("comet_paths", np.zeros((n, MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=np.float32)).astype(jnp.float32),
        comet_lengths=optional("comet_lengths", np.zeros((n, MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=np.int32)).astype(jnp.int32),
        comet_ships=optional("comet_ships", np.zeros((n, MAX_COMET_SPAWNS), dtype=np.float32)).astype(jnp.float32),
        comet_valid=optional("comet_valid", np.zeros((n, MAX_COMET_SPAWNS), dtype=bool)).astype(bool),
        comet_path_index=jnp.full((n, MAX_COMET_SPAWNS), -2, dtype=jnp.int32),
        comet_slots=jnp.full((n, MAX_COMET_SPAWNS, COMETS_PER_SPAWN), -1, dtype=jnp.int32),
    )


def select_from_bank(bank: EnvState, indices: jnp.ndarray) -> EnvState:
    return EnvState(
        planets=bank.planets[indices],
        initial_planets=bank.initial_planets[indices],
        planet_alive=bank.planet_alive[indices],
        fleets=bank.fleets[indices],
        fleet_alive=bank.fleet_alive[indices],
        step=bank.step[indices],
        angular_velocity=bank.angular_velocity[indices],
        next_fleet_id=bank.next_fleet_id[indices],
        done=bank.done[indices],
        winner=bank.winner[indices],
        overflow_fleets=bank.overflow_fleets[indices],
        comet_paths=bank.comet_paths[indices],
        comet_lengths=bank.comet_lengths[indices],
        comet_ships=bank.comet_ships[indices],
        comet_valid=bank.comet_valid[indices],
        comet_path_index=bank.comet_path_index[indices],
        comet_slots=bank.comet_slots[indices],
    )


def step_batch(state: EnvState, actions: jnp.ndarray, params: EnvParams) -> EnvState:
    """Batched step.

    actions shape: [B, 2, max_actions, 3], where each action is
    [from_planet_id, angle_rad, ships].
    """

    return jax.vmap(lambda s, a: step_single(s, a, params))(state, actions)


def step_single(state: EnvState, actions: jnp.ndarray, params: EnvParams) -> EnvState:
    actions = actions.astype(jnp.float32)
    state = jax.lax.cond(state.done, lambda x: x, lambda x: _active_step(x, actions, params), state)
    return state


def _active_step(state: EnvState, actions: jnp.ndarray, params: EnvParams) -> EnvState:
    """单步模拟，严格对应 orbit_wars.py interpreter() 的回合顺序。"""
    # 0. 彗星过期预检（路径走完的彗星提前清除，避免占用星球槽位影响发射）
    state = _expire_comets(state)
    # 1. 彗星生成（在 COMET_SPAWN_STEPS 对应回合触发，step+1 判断）
    state = _spawn_comets(state)
    # 2. 双方发射舰队（先扣除来源星球船数，再写入舰队槽位）
    state = _launch_all(state, actions, params)
    # 3. 生产：所有有主星球 +production 船（在舰队移动之前）
    owned = (state.planets[:, 1] >= 0.0) & state.planet_alive
    planets = state.planets.at[:, 5].add(jnp.where(owned, state.planets[:, 6], 0.0))
    state = state._replace(planets=planets)
    # 4. 舰队移动 + 连续碰撞检测（超界 / 穿太阳 / 撞星球）
    state, combat0, combat1 = _move_fleets(state, params)
    # 5. 行星旋转 + sweep（行星从旧位置扫向新位置，捕获路径上的舰队）
    state, sweep0, sweep1 = _rotate_planets_and_sweep(state, params)
    # 6. 彗星移动 + sweep（沿预计算路径前进一步）
    state, comet0, comet1 = _move_comets_and_sweep(state)
    # 7. 过期检查（移动后路径走完的彗星立即清除）
    state = _expire_comets(state)
    # 8. 战斗解算（合并三类来源的战斗贡献：舰队碰撞 / 行星 sweep / 彗星 sweep）
    state = _resolve_combat(state, combat0 + sweep0 + comet0, combat1 + sweep1 + comet1)
    state = state._replace(step=state.step + 1)
    state = _update_terminal(state, params)
    return state


def _launch_all(state: EnvState, actions: jnp.ndarray, params: EnvParams) -> EnvState:
    """依次处理双方所有发射动作。

    用 fori_loop 串行处理是为了保证每次发射后星球船数立即更新，
    后续动作能看到最新的余量，与 orbit_wars.py 的 Python 循环语义一致。
    性能瓶颈在此；如需加速可改为预扣兵力后并行写入，但会引入竞争写。
    """
    # 将 [2, max_actions, 3] 展开为 [2*max_actions, 3]，player 标记同步展开
    flat_players = jnp.repeat(jnp.arange(2, dtype=jnp.int32), params.max_actions)
    flat_actions = actions.reshape((2 * params.max_actions, 3))

    def body(i: jnp.ndarray, carry: EnvState) -> EnvState:
        return _launch_one(carry, flat_players[i], flat_actions[i], params)

    return jax.lax.fori_loop(0, 2 * params.max_actions, body, state)


def _launch_one(state: EnvState, player: jnp.ndarray, action: jnp.ndarray, params: EnvParams) -> EnvState:
    """处理单条发射动作，失败时静默跳过（JAX 无分支，通过 can_launch 掩码控制）。"""
    from_id = action[0]
    angle = action[1]
    ships = jnp.floor(action[2])  # 取整，与原版 int(ships) 一致
    # 找到 from_id 对应的星球槽位（argmax 在 has_planet=False 时返回 0，被 can_launch 过滤）
    planet_match = (state.planets[:, 0] == from_id) & state.planet_alive
    idx = jnp.argmax(planet_match)
    has_planet = planet_match.any()
    owner_ok = state.planets[idx, 1].astype(jnp.int32) == player  # 必须是自己的星球
    ships_ok = (ships > 0.0) & (state.planets[idx, 5] >= ships)   # 船数充足
    # 找到第一个空闲舰队槽位（slot 满时 overflow_fleets 计数，动作被丢弃）
    free = ~state.fleet_alive
    fleet_idx = jnp.argmax(free)
    has_slot = free.any()
    can_launch = has_planet & owner_ok & ships_ok & has_slot

    radius = state.planets[idx, 4]
    # 起始位置略微偏出星球边缘（+0.1），避免舰队在第一步立即与出发星球碰撞
    start_x = state.planets[idx, 2] + jnp.cos(angle) * (radius + 0.1)
    start_y = state.planets[idx, 3] + jnp.sin(angle) * (radius + 0.1)
    new_fleet = jnp.array(
        [
            state.next_fleet_id.astype(jnp.float32),
            player.astype(jnp.float32),
            start_x,
            start_y,
            angle,
            from_id,
            ships,
        ],
        dtype=jnp.float32,
    )
    planets = state.planets.at[idx, 5].add(jnp.where(can_launch, -ships, 0.0))
    fleets = state.fleets.at[fleet_idx].set(jnp.where(can_launch, new_fleet, state.fleets[fleet_idx]))
    fleet_alive = state.fleet_alive.at[fleet_idx].set(state.fleet_alive[fleet_idx] | can_launch)
    return state._replace(
        planets=planets,
        fleets=fleets,
        fleet_alive=fleet_alive,
        next_fleet_id=state.next_fleet_id + can_launch.astype(jnp.int32),
        overflow_fleets=state.overflow_fleets + ((has_planet & owner_ok & ships_ok & ~has_slot).astype(jnp.int32)),
    )


def comet_planet_mask(state: EnvState) -> jnp.ndarray:
    valid_slots = state.comet_slots.reshape(-1)
    valid_slots = jnp.where(valid_slots >= 0, valid_slots, 0)
    active = (state.comet_path_index[:, None] >= -1) & (state.comet_slots >= 0)
    active_flat = active.reshape(-1)
    return jnp.zeros((state.planets.shape[0],), dtype=bool).at[valid_slots].set(active_flat)


def _expire_comets(state: EnvState) -> EnvState:
    idx = state.comet_path_index
    min_len = jnp.where(state.comet_lengths > 0, state.comet_lengths, MAX_COMET_PATH + 1).min(axis=1)
    expired = (idx >= min_len) & (idx >= -1)
    valid_slots = state.comet_slots >= 0
    slots = jnp.where(valid_slots, state.comet_slots, 0)
    expired_slots = valid_slots & expired[:, None]
    slot_expired = jnp.zeros_like(state.planet_alive).at[slots.reshape(-1)].max(expired_slots.reshape(-1))
    planet_alive = state.planet_alive & (~slot_expired)
    planets = jnp.where(slot_expired[:, None], jnp.zeros_like(state.planets), state.planets)
    initial_planets = jnp.where(slot_expired[:, None], jnp.zeros_like(state.initial_planets), state.initial_planets)
    comet_slots = jnp.where(expired[:, None], -1, state.comet_slots)
    comet_path_index = jnp.where(expired, -2, state.comet_path_index)
    return state._replace(planets=planets, initial_planets=initial_planets, planet_alive=planet_alive, comet_slots=comet_slots, comet_path_index=comet_path_index)


def _spawn_comets(state: EnvState) -> EnvState:
    spawn_match = COMET_SPAWN_STEPS == (state.step + 1)
    spawn_idx = jnp.argmax(spawn_match)
    should_spawn = spawn_match.any() & state.comet_valid[spawn_idx] & (state.comet_path_index[spawn_idx] == -2)
    return jax.lax.cond(should_spawn, lambda s: _spawn_comet_group(s, spawn_idx), lambda s: s, state)


def _spawn_comet_group(state: EnvState, spawn_idx: jnp.ndarray) -> EnvState:
    free_order = jnp.argsort(state.planet_alive.astype(jnp.int32))
    slots = free_order[:COMETS_PER_SPAWN]
    max_id = jnp.where(state.planet_alive, state.planets[:, 0], -1.0).max()
    ids = max_id + 1.0 + jnp.arange(COMETS_PER_SPAWN, dtype=jnp.float32)
    ships = state.comet_ships[spawn_idx]
    rows = jnp.stack(
        [
            ids,
            -jnp.ones((COMETS_PER_SPAWN,), dtype=jnp.float32),
            -99.0 * jnp.ones((COMETS_PER_SPAWN,), dtype=jnp.float32),
            -99.0 * jnp.ones((COMETS_PER_SPAWN,), dtype=jnp.float32),
            COMET_RADIUS * jnp.ones((COMETS_PER_SPAWN,), dtype=jnp.float32),
            ships * jnp.ones((COMETS_PER_SPAWN,), dtype=jnp.float32),
            COMET_PRODUCTION * jnp.ones((COMETS_PER_SPAWN,), dtype=jnp.float32),
        ],
        axis=-1,
    )
    planets = state.planets.at[slots].set(rows)
    initial_planets = state.initial_planets.at[slots].set(rows)
    planet_alive = state.planet_alive.at[slots].set(True)
    comet_slots = state.comet_slots.at[spawn_idx].set(slots.astype(jnp.int32))
    comet_path_index = state.comet_path_index.at[spawn_idx].set(-1)
    return state._replace(planets=planets, initial_planets=initial_planets, planet_alive=planet_alive, comet_slots=comet_slots, comet_path_index=comet_path_index)


def _move_fleets(state: EnvState, params: EnvParams) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    fleets = state.fleets
    old_x = fleets[:, 2]
    old_y = fleets[:, 3]
    angle = fleets[:, 4]
    speed = fleet_speed(fleets[:, 6], params.ship_speed)
    new_x = old_x + jnp.cos(angle) * speed
    new_y = old_y + jnp.sin(angle) * speed
    moved = fleets.at[:, 2].set(new_x).at[:, 3].set(new_y)

    alive = state.fleet_alive
    in_bounds = (new_x >= 0.0) & (new_x <= params.board_size) & (new_y >= 0.0) & (new_y <= params.board_size)
    sun_hit = point_segment_distance_sq(CENTER, CENTER, old_x, old_y, new_x, new_y) < SUN_RADIUS * SUN_RADIUS

    px = state.planets[:, 2]
    py = state.planets[:, 3]
    pr = state.planets[:, 4]
    d2 = point_segment_distance_sq(
        px[None, :],
        py[None, :],
        old_x[:, None],
        old_y[:, None],
        new_x[:, None],
        new_y[:, None],
    )
    planet_hit = (d2 < (pr[None, :] ** 2)) & state.planet_alive[None, :] & alive[:, None] & in_bounds[:, None] & (~sun_hit[:, None])
    any_planet_hit = planet_hit.any(axis=1)
    hit_idx = jnp.argmax(planet_hit, axis=1)
    remove = alive & ((~in_bounds) | sun_hit | any_planet_hit)

    owner = fleets[:, 1].astype(jnp.int32)
    ships = fleets[:, 6]
    valid_combat = alive & any_planet_hit & in_bounds & (~sun_hit)
    combat0 = jnp.zeros((state.planets.shape[0],), dtype=jnp.float32).at[hit_idx].add(
        jnp.where(valid_combat & (owner == 0), ships, 0.0)
    )
    combat1 = jnp.zeros((state.planets.shape[0],), dtype=jnp.float32).at[hit_idx].add(
        jnp.where(valid_combat & (owner == 1), ships, 0.0)
    )
    return state._replace(fleets=moved, fleet_alive=alive & (~remove)), combat0, combat1


def _rotate_planets_and_sweep(state: EnvState, params: EnvParams) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    old_x = state.planets[:, 2]
    old_y = state.planets[:, 3]
    init = state.initial_planets
    dx = init[:, 2] - CENTER
    dy = init[:, 3] - CENTER
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    rotating = (orbit_r + init[:, 4] < ROTATION_RADIUS_LIMIT) & state.planet_alive & (~comet_planet_mask(state))
    init_angle = jnp.arctan2(dy, dx)
    current_angle = init_angle + state.angular_velocity * state.step.astype(jnp.float32)
    new_x = CENTER + orbit_r * jnp.cos(current_angle)
    new_y = CENTER + orbit_r * jnp.sin(current_angle)
    planets = state.planets.at[:, 2].set(jnp.where(rotating, new_x, old_x))
    planets = planets.at[:, 3].set(jnp.where(rotating, new_y, old_y))

    fx = state.fleets[:, 2]
    fy = state.fleets[:, 3]
    d2 = point_segment_distance_sq(
        fx[:, None],
        fy[:, None],
        old_x[None, :],
        old_y[None, :],
        planets[None, :, 2],
        planets[None, :, 3],
    )
    moved_planet = ((old_x != planets[:, 2]) | (old_y != planets[:, 3])) & state.planet_alive
    hit = (d2 < (planets[None, :, 4] ** 2)) & state.fleet_alive[:, None] & moved_planet[None, :]
    any_hit = hit.any(axis=1)
    hit_idx = jnp.argmax(hit, axis=1)
    owner = state.fleets[:, 1].astype(jnp.int32)
    ships = state.fleets[:, 6]
    combat0 = jnp.zeros((state.planets.shape[0],), dtype=jnp.float32).at[hit_idx].add(
        jnp.where(any_hit & (owner == 0), ships, 0.0)
    )
    combat1 = jnp.zeros((state.planets.shape[0],), dtype=jnp.float32).at[hit_idx].add(
        jnp.where(any_hit & (owner == 1), ships, 0.0)
    )
    return state._replace(planets=planets, fleet_alive=state.fleet_alive & (~any_hit)), combat0, combat1


def _move_comets_and_sweep(state: EnvState) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    active = state.comet_path_index >= -1

    def body(i: jnp.ndarray, carry):
        st, c0, c1 = carry
        return jax.lax.cond(active[i], lambda x: _move_one_comet_group(i, *x), lambda x: x, (st, c0, c1))

    zeros = jnp.zeros((state.planets.shape[0],), dtype=jnp.float32)
    state, combat0, combat1 = jax.lax.fori_loop(0, MAX_COMET_SPAWNS, body, (state, zeros, zeros))
    return state, combat0, combat1


def _move_one_comet_group(i: jnp.ndarray, state: EnvState, combat0: jnp.ndarray, combat1: jnp.ndarray):
    new_index = state.comet_path_index[i] + 1
    slots = state.comet_slots[i]
    safe_slots = jnp.where(slots >= 0, slots, 0)
    old_pos = state.planets[safe_slots, 2:4]
    path_idx = jnp.clip(new_index, 0, MAX_COMET_PATH - 1)
    new_pos = state.comet_paths[i, jnp.arange(COMETS_PER_SPAWN), path_idx]
    live = (slots >= 0) & (new_index < state.comet_lengths[i])
    planets = state.planets.at[safe_slots, 2:4].set(jnp.where(live[:, None], new_pos, state.planets[safe_slots, 2:4]))
    state = state._replace(planets=planets, comet_path_index=state.comet_path_index.at[i].set(new_index))

    sweep_active = live & (old_pos[:, 0] >= 0.0)
    d2 = point_segment_distance_sq(
        state.fleets[:, None, 2],
        state.fleets[:, None, 3],
        old_pos[None, :, 0],
        old_pos[None, :, 1],
        new_pos[None, :, 0],
        new_pos[None, :, 1],
    )
    hit = (d2 < COMET_RADIUS * COMET_RADIUS) & state.fleet_alive[:, None] & sweep_active[None, :]
    any_hit = hit.any(axis=1)
    hit_group_idx = jnp.argmax(hit, axis=1)
    hit_slots = safe_slots[hit_group_idx]
    owner = state.fleets[:, 1].astype(jnp.int32)
    ships = state.fleets[:, 6]
    combat0 = combat0.at[hit_slots].add(jnp.where(any_hit & (owner == 0), ships, 0.0))
    combat1 = combat1.at[hit_slots].add(jnp.where(any_hit & (owner == 1), ships, 0.0))
    state = state._replace(fleet_alive=state.fleet_alive & (~any_hit))
    return state, combat0, combat1


def _resolve_combat(state: EnvState, combat0: jnp.ndarray, combat1: jnp.ndarray) -> EnvState:
    """战斗解算：向量化版本，对所有星球同时处理。

    规则（对应 orbit_wars.py）：
    - 单方到达：直接加入/攻击驻军。
    - 双方到达：top - second = survivor；平局时 survivor = 0，驻军不变。
    - survivor > 0 且与驻军方相同：驻军增加。
    - survivor > 0 且与驻军方不同：驻军减少；减到负数则易主，新船数取绝对值。
    """
    planets = state.planets
    has0 = combat0 > 0.0
    has1 = combat1 > 0.0
    top_owner = jnp.where(combat0 >= combat1, 0.0, 1.0)   # 船多的一方
    top_ships = jnp.maximum(combat0, combat1)
    second_ships = jnp.minimum(combat0, combat1)
    # 双方都有：survivor = 差值；单方：survivor = 全部
    survivor = jnp.where(has0 & has1, top_ships - second_ships, top_ships)
    has_combat = has0 | has1
    survivor_positive = has_combat & (survivor > 0.0)  # survivor=0 时平局，不改变驻军
    same_owner = planets[:, 1] == top_owner            # 胜方是否与驻军同属一方
    new_ships_same = planets[:, 5] + survivor          # 友方增援
    new_ships_attack = planets[:, 5] - survivor        # 攻方削减
    captured = new_ships_attack < 0.0                  # 驻军被清零 → 易主
    new_owner = jnp.where(survivor_positive & (~same_owner) & captured, top_owner, planets[:, 1])
    new_ships = jnp.where(
        survivor_positive,
        jnp.where(same_owner, new_ships_same, jnp.where(captured, -new_ships_attack, new_ships_attack)),
        planets[:, 5],
    )
    planets = planets.at[:, 1].set(jnp.where(state.planet_alive, new_owner, planets[:, 1]))
    planets = planets.at[:, 5].set(jnp.where(state.planet_alive, new_ships, planets[:, 5]))
    return state._replace(planets=planets)


def _update_terminal(state: EnvState, params: EnvParams) -> EnvState:
    """检查终局条件，计算胜者。

    终止条件（二选一）：
    1. step >= episode_steps - 2（对应原版 episodeSteps - 2 判断）
    2. 仅剩 0 或 1 个玩家在场（星球 + 飞行舰队均无则视为不在场）

    胜负规则：终局分数 = 拥有星球船数 + 飞行舰队船数之和。
    winner: 0/1 = 得分高的一方，2 = 平局（同分），-1 = 游戏进行中。
    """
    owner = state.planets[:, 1].astype(jnp.int32)
    p_alive = state.planet_alive & (owner >= 0)   # 排除中立星球
    f_owner = state.fleets[:, 1].astype(jnp.int32)
    f_alive = state.fleet_alive
    # 判断每个玩家是否仍在场（有星球或有飞行舰队均算存活）
    alive0 = ((p_alive & (owner == 0)).any()) | ((f_alive & (f_owner == 0)).any())
    alive1 = ((p_alive & (owner == 1)).any()) | ((f_alive & (f_owner == 1)).any())
    terminal = (state.step >= params.episode_steps - 2) | ((alive0.astype(jnp.int32) + alive1.astype(jnp.int32)) <= 1)
    # 终局分数 = 己方所有星球船数 + 飞行中舰队船数
    score0 = jnp.where(state.planet_alive & (owner == 0), state.planets[:, 5], 0.0).sum()
    score1 = jnp.where(state.planet_alive & (owner == 1), state.planets[:, 5], 0.0).sum()
    score0 = score0 + jnp.where(state.fleet_alive & (f_owner == 0), state.fleets[:, 6], 0.0).sum()
    score1 = score1 + jnp.where(state.fleet_alive & (f_owner == 1), state.fleets[:, 6], 0.0).sum()
    # 2 = 平局（相同分），与 reward_jax.terminal_reward_player0 的 scores() 计算方式独立
    winner = jnp.where(score0 == score1, 2, jnp.where(score0 > score1, 0, 1))
    return state._replace(done=terminal, winner=jnp.where(terminal, winner, -1))


def scores(state: EnvState) -> jnp.ndarray:
    if state.planets.ndim == 3:
        return jax.vmap(scores)(state)
    owner = state.planets[:, 1].astype(jnp.int32)
    f_owner = state.fleets[:, 1].astype(jnp.int32)
    score0 = jnp.where(state.planet_alive & (owner == 0), state.planets[:, 5], 0.0).sum(axis=-1)
    score1 = jnp.where(state.planet_alive & (owner == 1), state.planets[:, 5], 0.0).sum(axis=-1)
    score0 = score0 + jnp.where(state.fleet_alive & (f_owner == 0), state.fleets[:, 6], 0.0).sum(axis=-1)
    score1 = score1 + jnp.where(state.fleet_alive & (f_owner == 1), state.fleets[:, 6], 0.0).sum(axis=-1)
    return jnp.stack([score0, score1], axis=-1)
