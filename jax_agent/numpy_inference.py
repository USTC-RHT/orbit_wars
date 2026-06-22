"""纯 numpy 推理层。

替代 onnx-hybrid/main.py 中两处 jax.jit 调用：
  - encode_batch(state, feat_params)       → encode_batch_np(state_np, feat_params)
  - aim_with_prediction_jax(...)           → aim_with_prediction_np(...)
  - dynamic_ship_buckets(...)              → dynamic_ship_buckets_np(...)

设计原则：
  1. 逻辑与 features_jax.py / train_jax.py 完全对齐，不改变数值。
  2. 输入输出均为 numpy ndarray，兼容直接转 jnp 喂 ONNX。
  3. 零 JAX 依赖，消除 jax.jit 第一次调用的 XLA 编译时延。

核心参考：
  - features_jax.py  → encode_batch_np
  - train_jax.py     → aim_with_prediction_np, dynamic_ship_buckets_np
  - torch_agent/physics.py & features.py → inbound/aim 实现蓝本
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# ─────────────────────────────────────────────────────────────
# 常量（与 fast_env.py / features_jax.py 完全对齐）
# ─────────────────────────────────────────────────────────────
BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0

MAX_COMET_SPAWNS: int = 5
COMETS_PER_SPAWN: int = 4
MAX_COMET_PATH: int = 40

PLANET_DIM: int = 26
RELATIVE_DIM: int = 8
GLOBAL_DIM: int = 11
FLEET_DIM: int = 28
MAP_CHANNELS: int = 6
MAP_GRID_SIZE: int = 8

INBOUND_HORIZON: int = 60
MIN_LAUNCH: float = 10.0
LOCAL_PLANET_RADIUS: float = 25.0

# 与 features_jax.py SHIP_RATIOS 对齐
SHIP_RATIOS = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=np.float32)

# 瞄准参数（与 train_jax.py aim_with_prediction_jax 对齐）
_AIM_HORIZON: int = 60
_AIM_FP_ITERS: int = 8
LAUNCH_CLEARANCE: float = 0.1
AIM_TARGET_INSET: float = 1e-3


# ─────────────────────────────────────────────────────────────
# EnvState numpy 包装（只在本模块内部使用，不暴露给外部）
# 外部传入的是 JAX EnvState，这里用 np.asarray 转换成 numpy
# ─────────────────────────────────────────────────────────────
class _StateNP(NamedTuple):
    """EnvState 的 numpy 镜像，batch=1 已 squeeze。"""
    planets: np.ndarray          # [P, 7]
    initial_planets: np.ndarray  # [P, 7]
    planet_alive: np.ndarray     # [P] bool
    fleets: np.ndarray           # [F, 7]
    fleet_alive: np.ndarray      # [F] bool
    step: int
    angular_velocity: float
    # 彗星相关
    comet_paths: np.ndarray      # [5, 4, 40, 2]
    comet_lengths: np.ndarray    # [5, 4] int
    comet_path_index: np.ndarray # [5] int
    comet_slots: np.ndarray      # [5, 4] int


class EnvStateNP(NamedTuple):
    """Batch=1 EnvState mirror used by the submission runtime."""
    planets: np.ndarray
    initial_planets: np.ndarray
    planet_alive: np.ndarray
    fleets: np.ndarray
    fleet_alive: np.ndarray
    step: np.ndarray
    angular_velocity: np.ndarray
    comet_paths: np.ndarray
    comet_lengths: np.ndarray
    comet_path_index: np.ndarray
    comet_slots: np.ndarray


class NormalizerStateNP(NamedTuple):
    planet_mean: np.ndarray
    planet_var: np.ndarray
    planet_count: np.ndarray
    relative_mean: np.ndarray
    relative_var: np.ndarray
    relative_count: np.ndarray
    global_mean: np.ndarray
    global_var: np.ndarray
    global_count: np.ndarray


PLANET_NORMALIZE_INDICES = np.array([5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=np.int32)
RELATIVE_NORMALIZE_INDICES = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
GLOBAL_NORMALIZE_INDICES = np.arange(GLOBAL_DIM, dtype=np.int32)


def _read_field(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _row_value(row, idx: int, attr: str, default=0.0):
    try:
        return row[idx]
    except Exception:
        return getattr(row, attr, default)


def _flip_owner_np(owner: np.ndarray) -> np.ndarray:
    return np.where(owner == 0.0, 1.0, np.where(owner == 1.0, 0.0, owner))


def _rotate_angle_pi_np(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(-np.sin(angle), -np.cos(angle)).astype(np.float32)


def _mirror_planet_indices_np(count: int) -> np.ndarray:
    idx = np.arange(count, dtype=np.int32)
    mirror = (idx // 4) * 4 + (3 - (idx % 4))
    return np.where(mirror < count, mirror, idx)


def _comet_arrays_from_observation_np(
    observation,
    planet_id_to_slot: dict[int, int],
    planets_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    comet_paths = np.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=np.float32)
    comet_lengths = np.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=np.int32)
    comet_path_index = np.full((MAX_COMET_SPAWNS,), -2, dtype=np.int32)
    comet_slots = np.full((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), -1, dtype=np.int32)

    groups = list(_read_field(observation, "comets", []) or [])[:MAX_COMET_SPAWNS]
    for group_idx, group in enumerate(groups):
        planet_ids = list(_read_field(group, "planet_ids", []) or [])[:COMETS_PER_SPAWN]
        paths = list(_read_field(group, "paths", []) or [])[:COMETS_PER_SPAWN]
        comet_path_index[group_idx] = int(_read_field(group, "path_index", -2) or 0)
        for comet_idx, pid_raw in enumerate(planet_ids):
            slot = planet_id_to_slot.get(int(pid_raw), -1)
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
    return comet_paths, comet_lengths, comet_path_index, comet_slots


def obs_to_np_state(observation, max_planets: int, max_fleets: int) -> EnvStateNP:
    planets = np.zeros((max_planets, 7), dtype=np.float32)
    initial_planets = np.zeros((max_planets, 7), dtype=np.float32)
    fleets = np.zeros((max_fleets, 7), dtype=np.float32)
    planet_alive = np.zeros((max_planets,), dtype=bool)
    fleet_alive = np.zeros((max_fleets,), dtype=bool)
    planet_id_to_slot: dict[int, int] = {}

    for row in list(_read_field(observation, "planets", []) or []):
        pid = int(_row_value(row, 0, "id", -1))
        if 0 <= pid < max_planets:
            planets[pid] = [
                pid,
                float(_row_value(row, 1, "owner", -1)),
                float(_row_value(row, 2, "x", 0.0)),
                float(_row_value(row, 3, "y", 0.0)),
                float(_row_value(row, 4, "radius", 0.0)),
                float(_row_value(row, 5, "ships", _row_value(row, 5, "ship_count", 0.0))),
                float(_row_value(row, 6, "production", 0.0)),
            ]
            planet_alive[pid] = True
            planet_id_to_slot[pid] = pid

    initial_rows = list(_read_field(observation, "initial_planets", []) or [])
    if initial_rows:
        for row in initial_rows:
            pid = int(_row_value(row, 0, "id", -1))
            if 0 <= pid < max_planets:
                initial_planets[pid] = [
                    pid,
                    float(_row_value(row, 1, "owner", -1)),
                    float(_row_value(row, 2, "x", 0.0)),
                    float(_row_value(row, 3, "y", 0.0)),
                    float(_row_value(row, 4, "radius", 0.0)),
                    float(_row_value(row, 5, "ships", _row_value(row, 5, "ship_count", 0.0))),
                    float(_row_value(row, 6, "production", 0.0)),
                ]
    else:
        initial_planets[:] = planets

    for i, row in enumerate(list(_read_field(observation, "fleets", []) or [])[:max_fleets]):
        fleets[i] = [
            float(_row_value(row, 0, "id", i)),
            float(_row_value(row, 1, "owner", -1)),
            float(_row_value(row, 2, "x", 0.0)),
            float(_row_value(row, 3, "y", 0.0)),
            float(_row_value(row, 4, "angle", 0.0)),
            float(_row_value(row, 5, "from_planet_id", -1)),
            float(_row_value(row, 6, "ships", _row_value(row, 6, "ship_count", 0.0))),
        ]
        fleet_alive[i] = True

    comet_paths, comet_lengths, comet_path_index, comet_slots = _comet_arrays_from_observation_np(
        observation,
        planet_id_to_slot,
        planets,
    )
    return EnvStateNP(
        planets=planets[None],
        initial_planets=initial_planets[None],
        planet_alive=planet_alive[None],
        fleets=fleets[None],
        fleet_alive=fleet_alive[None],
        step=np.array([int(_read_field(observation, "step", 0) or 0)], dtype=np.int32),
        angular_velocity=np.array([float(_read_field(observation, "angular_velocity", 0.0) or 0.0)], dtype=np.float32),
        comet_paths=comet_paths[None],
        comet_lengths=comet_lengths[None],
        comet_path_index=comet_path_index[None],
        comet_slots=comet_slots[None],
    )


def flip_state_np(state: EnvStateNP) -> EnvStateNP:
    planet_perm = _mirror_planet_indices_np(state.planets.shape[-2])
    planets = state.planets[:, planet_perm, :].copy()
    planets[..., 1] = _flip_owner_np(planets[..., 1])
    planets[..., 2] = 100.0 - planets[..., 2]
    planets[..., 3] = 100.0 - planets[..., 3]

    initial_planets = state.initial_planets[:, planet_perm, :].copy()
    initial_planets[..., 1] = _flip_owner_np(initial_planets[..., 1])
    initial_planets[..., 2] = 100.0 - initial_planets[..., 2]
    initial_planets[..., 3] = 100.0 - initial_planets[..., 3]

    fleets = state.fleets.copy()
    fleets[..., 1] = _flip_owner_np(fleets[..., 1])
    fleets[..., 2] = 100.0 - fleets[..., 2]
    fleets[..., 3] = 100.0 - fleets[..., 3]
    fleets[..., 4] = _rotate_angle_pi_np(fleets[..., 4])

    comet_paths = state.comet_paths.copy()
    comet_paths[..., 0] = 100.0 - comet_paths[..., 0]
    comet_paths[..., 1] = 100.0 - comet_paths[..., 1]
    safe_slots = np.clip(state.comet_slots, 0, state.planets.shape[-2] - 1)
    comet_slots = np.where(state.comet_slots >= 0, planet_perm[safe_slots], -1).astype(np.int32)

    return state._replace(
        planets=planets,
        initial_planets=initial_planets,
        planet_alive=state.planet_alive[:, planet_perm],
        fleets=fleets,
        comet_paths=comet_paths,
        comet_slots=comet_slots,
    )


def flip_actions_np(actions: np.ndarray) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float32).copy()
    ships = out[..., 2]
    out[..., 1] = np.where(ships > 0.0, _rotate_angle_pi_np(out[..., 1]), 0.0)
    return out


def actions_to_kaggle_np(actions_arr: np.ndarray) -> list:
    arr = np.asarray(actions_arr[0], dtype=np.float32)
    return [[float(r[0]), float(r[1]), int(r[2])] for r in arr if float(r[2]) > 0]


def load_normalizer_np(npz) -> NormalizerStateNP:
    return NormalizerStateNP(
        planet_mean=np.asarray(npz["planet_mean"], dtype=np.float32),
        planet_var=np.asarray(npz["planet_var"], dtype=np.float32),
        planet_count=np.asarray(npz["planet_count"], dtype=np.float32),
        relative_mean=np.asarray(npz["relative_mean"], dtype=np.float32),
        relative_var=np.asarray(npz["relative_var"], dtype=np.float32),
        relative_count=np.asarray(npz["relative_count"], dtype=np.float32),
        global_mean=np.asarray(npz["global_mean"], dtype=np.float32),
        global_var=np.asarray(npz["global_var"], dtype=np.float32),
        global_count=np.asarray(npz["global_count"], dtype=np.float32),
    )


def _normalize_values_np(values: np.ndarray, mean: np.ndarray, var: np.ndarray, enabled: bool, clip: float, eps: float) -> np.ndarray:
    if not enabled:
        return values
    return np.clip((values - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)


def normalize_encoded_np(encoded: EncodedBatchNP, state: NormalizerStateNP, params) -> EncodedBatchNP:
    planet_features = np.array(encoded.planet_features, copy=True)
    relative_features = np.array(encoded.relative_features, copy=True)
    global_features = np.array(encoded.global_features, copy=True)
    enabled = bool(getattr(params, "enabled", True))
    clip = float(getattr(params, "clip", 5.0))
    eps = float(getattr(params, "eps", 1.0e-8))

    planet_features[..., PLANET_NORMALIZE_INDICES] = _normalize_values_np(
        planet_features[..., PLANET_NORMALIZE_INDICES], state.planet_mean, state.planet_var, enabled, clip, eps
    )
    rel_norm = _normalize_values_np(
        relative_features[..., RELATIVE_NORMALIZE_INDICES], state.relative_mean, state.relative_var, enabled, clip, eps
    )
    real_candidates = encoded.candidate_ids >= 0
    relative_features[..., RELATIVE_NORMALIZE_INDICES] = np.where(
        real_candidates[..., None],
        rel_norm,
        relative_features[..., RELATIVE_NORMALIZE_INDICES],
    )
    global_features[..., GLOBAL_NORMALIZE_INDICES] = _normalize_values_np(
        global_features[..., GLOBAL_NORMALIZE_INDICES], state.global_mean, state.global_var, enabled, clip, eps
    )
    return EncodedBatchNP(
        planet_features=planet_features,
        relative_features=relative_features,
        global_features=global_features,
        map_features=encoded.map_features,
        candidate_mask=encoded.candidate_mask,
        amount_mask=encoded.amount_mask,
        source_ids=encoded.source_ids,
        source_ships=encoded.source_ships,
        candidate_ids=encoded.candidate_ids,
        ship_buckets=encoded.ship_buckets,
        source_mask=encoded.source_mask,
        fleet_features=encoded.fleet_features,
        fleet_mask=encoded.fleet_mask,
    )


def _to_np(jax_state) -> _StateNP:
    """把 batch=1 的 JAX EnvState 转成 numpy，squeeze 掉 batch 维。"""
    def _a(x):
        return np.asarray(x).reshape(np.asarray(x).shape[1:] if np.asarray(x).ndim > 1 and np.asarray(x).shape[0] == 1 else np.asarray(x).shape)

    # squeeze batch dim（batch=1）
    def sq(x):
        arr = np.asarray(x)
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr[0]
        return arr

    return _StateNP(
        planets=sq(jax_state.planets).astype(np.float32),
        initial_planets=sq(jax_state.initial_planets).astype(np.float32),
        planet_alive=sq(jax_state.planet_alive).astype(bool),
        fleets=sq(jax_state.fleets).astype(np.float32),
        fleet_alive=sq(jax_state.fleet_alive).astype(bool),
        step=int(np.asarray(jax_state.step).item()),
        angular_velocity=float(np.asarray(jax_state.angular_velocity).item()),
        comet_paths=sq(jax_state.comet_paths).astype(np.float32),
        comet_lengths=sq(jax_state.comet_lengths).astype(np.int32),
        comet_path_index=sq(jax_state.comet_path_index).astype(np.int32),
        comet_slots=sq(jax_state.comet_slots).astype(np.int32),
    )


# ─────────────────────────────────────────────────────────────
# 基础物理工具
# ─────────────────────────────────────────────────────────────

def _fleet_speed_np(ships: np.ndarray) -> np.ndarray:
    """与 fast_env.fleet_speed 完全一致（向量化）。"""
    ships_f = np.maximum(np.asarray(ships, dtype=np.float32), 1.0)
    ratio = np.clip(np.log(ships_f) / math.log(1000.0), 0.0, 1.0)
    return np.minimum(1.0 + 5.0 * (ratio ** 1.5), 6.0)


def _fleet_speed_scalar(ships: float) -> float:
    """标量版本，用于逐 action 瞄准。"""
    s = max(float(ships), 1.0)
    ratio = max(0.0, min(1.0, math.log(s) / math.log(1000.0)))
    return min(1.0 + 5.0 * (ratio ** 1.5), 6.0)


def _point_segment_dist_sq_np(
    px: np.ndarray, py: np.ndarray,
    x0: np.ndarray, y0: np.ndarray,
    x1: np.ndarray, y1: np.ndarray,
) -> np.ndarray:
    """点到线段距离平方（向量化），与 fast_env.point_segment_distance_sq 完全一致。"""
    dx = x1 - x0
    dy = y1 - y0
    denom = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / denom, 0.0, 1.0)
    cx = x0 + t * dx
    cy = y0 + t * dy
    ddx = px - cx
    ddy = py - cy
    return ddx * ddx + ddy * ddy


def _point_segment_dist_scalar(px, py, x0, y0, x1, y1) -> float:
    """标量版，用于逐 action 瞄准中的碰撞检测。"""
    dx, dy = x1 - x0, y1 - y0
    seg_sq = dx * dx + dy * dy
    if seg_sq <= 1e-9:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg_sq))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _comet_planet_mask_np(s: _StateNP) -> np.ndarray:
    """返回 [P] bool，哪些 slot 当前被彗星占用。对应 fast_env.comet_planet_mask。"""
    n = s.planets.shape[0]
    active = (s.comet_path_index[:, None] >= -1) & (s.comet_slots >= 0)  # [5,4]
    active_flat = active.reshape(-1)                                       # [20]
    slots_flat = s.comet_slots.reshape(-1)                                 # [20]
    safe_slots = np.where(slots_flat >= 0, slots_flat, 0)
    mask = np.zeros(n, dtype=bool)
    mask[safe_slots[active_flat]] = True
    return mask


# ─────────────────────────────────────────────────────────────
# Part 2: 行星轨迹预测 + precompute_inbound
# 对应 features_jax.py: planet_positions_over_horizon,
#                        predict_fleet_targets, precompute_inbound
# ─────────────────────────────────────────────────────────────

def _planet_positions_over_horizon(s: _StateNP, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """计算所有星球未来 horizon 步的位置 [P, horizon+1]。

    t=0 为当前位置；静止星球固定，旋转星球按角速度追踪，彗星使用预存路径。
    对应 features_jax.planet_positions_over_horizon（以及 torch_agent/features.py
    的 px_t/py_t 构建段）。
    """
    P = s.planets.shape[0]
    ts = np.arange(horizon + 1, dtype=np.float32)  # [T+1]

    # ── 旋转星球 ────────────────────────────────────────────────
    init = s.initial_planets  # [P, 7]
    dx = init[:, 2] - CENTER   # [P]
    dy = init[:, 3] - CENTER
    orbit_r = np.sqrt(dx * dx + dy * dy)  # [P]

    comet_mask = _comet_planet_mask_np(s)  # [P]
    rotating = (orbit_r + init[:, 4] < ROTATION_RADIUS_LIMIT) & s.planet_alive & (~comet_mask)  # [P]

    phi_cur = np.arctan2(s.planets[:, 3] - CENTER, s.planets[:, 2] - CENTER)  # [P]
    # phi[p, t] = phi_cur[p] + omega * t
    phi = phi_cur[:, None] + s.angular_velocity * ts[None, :]  # [P, T+1]
    px = CENTER + orbit_r[:, None] * np.cos(phi)  # [P, T+1]
    py = CENTER + orbit_r[:, None] * np.sin(phi)

    # 静止星球用当前位置覆盖
    px = np.where(rotating[:, None], px, s.planets[:, 2:3])
    py = np.where(rotating[:, None], py, s.planets[:, 3:4])

    # ── 彗星路径覆盖 ────────────────────────────────────────────
    # comet_slots [5,4], comet_path_index [5], comet_paths [5,4,40,2]
    # 对应 features_jax.planet_positions_over_horizon 的彗星段
    for si in range(MAX_COMET_SPAWNS):
        pi = int(s.comet_path_index[si])
        if pi < -1:
            continue  # 未激活
        for ci in range(COMETS_PER_SPAWN):
            slot = int(s.comet_slots[si, ci])
            if slot < 0 or slot >= P:
                continue
            path_len = int(s.comet_lengths[si, ci])
            path = s.comet_paths[si, ci]  # [40, 2]
            for t_step in range(horizon + 1):
                idx = pi + t_step
                if 0 <= idx < path_len and idx < MAX_COMET_PATH:
                    px[slot, t_step] = path[idx, 0]
                    py[slot, t_step] = path[idx, 1]
                else:
                    px[slot, t_step] = -1000.0
                    py[slot, t_step] = -1000.0

    return px, py


def _predict_fleet_targets(
    s: _StateNP, horizon: int = INBOUND_HORIZON
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """预测每支舰队在 horizon 内第一个撞到的星球槽位。

    返回 (target_idx[F], target_valid[F], target_t[F])。
    对应 features_jax.predict_fleet_targets，使用 torch_agent/features.py
    的相同两阶段碰撞逻辑（move_hit + sweep_hit）。
    """
    F = s.fleets.shape[0]
    P = s.planets.shape[0]
    T = horizon

    if F == 0 or not s.fleet_alive.any():
        return (
            np.zeros(F, dtype=np.int32),
            np.zeros(F, dtype=bool),
            np.zeros(F, dtype=np.float32),
        )

    # 舰队参数
    ships = np.maximum(s.fleets[:, 6], 1.0)
    speed = _fleet_speed_np(ships)               # [F]
    vx = np.cos(s.fleets[:, 4]) * speed         # [F]
    vy = np.sin(s.fleets[:, 4]) * speed

    ts = np.arange(T + 1, dtype=np.float32)     # [T+1]
    fx_t = s.fleets[:, 2:3] + vx[:, None] * ts[None, :]  # [F, T+1]
    fy_t = s.fleets[:, 3:4] + vy[:, None] * ts[None, :]
    in_bounds = (
        (fx_t >= 0.0) & (fx_t <= BOARD_SIZE) &
        (fy_t >= 0.0) & (fy_t <= BOARD_SIZE)
    )  # [F, T+1]

    # 所有星球位置 [P, T+1]
    px_t, py_t = _planet_positions_over_horizon(s, T)

    radii = s.planets[:, 4]  # [P]
    eps = np.float32(1e-9)
    radius_sq = radii[None, :, None] ** 2   # [1, P, 1]
    cur_in_bounds = in_bounds[:, 1:][:, None, :]  # [F, 1, T]

    fx_prev = fx_t[:, :-1]   # [F, T]
    fy_prev = fy_t[:, :-1]
    fx_cur  = fx_t[:, 1:]
    fy_cur  = fy_t[:, 1:]

    px_old = px_t[:, :-1]    # [P, T]
    py_old = py_t[:, :-1]
    px_new = px_t[:, 1:]
    py_new = py_t[:, 1:]

    # move_hit: fleet 线段 vs planet 位置（旧位置）
    seg_x = (fx_cur - fx_prev)[:, None, :]   # [F, 1, T]
    seg_y = (fy_cur - fy_prev)[:, None, :]
    den = np.maximum(seg_x * seg_x + seg_y * seg_y, eps)
    rel_x = px_old[None, :, :] - fx_prev[:, None, :]  # [F, P, T]
    rel_y = py_old[None, :, :] - fy_prev[:, None, :]
    proj = np.clip((rel_x * seg_x + rel_y * seg_y) / den, 0.0, 1.0)
    cx = fx_prev[:, None, :] + proj * seg_x
    cy = fy_prev[:, None, :] + proj * seg_y
    move_d2 = (px_old[None, :, :] - cx) ** 2 + (py_old[None, :, :] - cy) ** 2
    move_hit = (move_d2 < radius_sq) & cur_in_bounds & s.fleet_alive[:, None, None] & s.planet_alive[None, :, None]

    # sweep_hit: planet sweep 线段 vs fleet 当前位置
    sweep_x = (px_new - px_old)[None, :, :]  # [1, P, T]
    sweep_y = (py_new - py_old)[None, :, :]
    sweep_den = np.maximum(sweep_x * sweep_x + sweep_y * sweep_y, eps)
    sweep_active = (sweep_x * sweep_x + sweep_y * sweep_y) > eps
    rel_x2 = fx_cur[:, None, :] - px_old[None, :, :]
    rel_y2 = fy_cur[:, None, :] - py_old[None, :, :]
    proj2 = np.clip((rel_x2 * sweep_x + rel_y2 * sweep_y) / sweep_den, 0.0, 1.0)
    cx2 = px_old[None, :, :] + proj2 * sweep_x
    cy2 = py_old[None, :, :] + proj2 * sweep_y
    sweep_d2 = (fx_cur[:, None, :] - cx2) ** 2 + (fy_cur[:, None, :] - cy2) ** 2
    sweep_hit = (sweep_d2 < radius_sq) & sweep_active & cur_in_bounds & s.fleet_alive[:, None, None] & s.planet_alive[None, :, None]

    # phase key：同一 tick move < sweep，保持引擎顺序
    large_key = np.int32(2 * (T + 1) + 1)
    steps = np.arange(1, T + 1, dtype=np.int32)
    move_keys = (2 * steps)[None, None, :]   # [1, 1, T]
    sweep_keys = (2 * steps + 1)[None, None, :]
    first_move_key = np.where(move_hit, move_keys, large_key).min(axis=2)   # [F, P]
    first_sweep_key = np.where(sweep_hit, sweep_keys, large_key).min(axis=2)
    first_key_fp = np.minimum(first_move_key, first_sweep_key)              # [F, P]

    fleet_first_key = first_key_fp.min(axis=1)    # [F]
    fleet_target = first_key_fp.argmin(axis=1).astype(np.int32)  # [F]
    fleet_valid = fleet_first_key < large_key
    fleet_first_t = (fleet_first_key // 2).astype(np.float32)

    fleet_target = np.clip(fleet_target, 0, P - 1)
    return fleet_target, fleet_valid, fleet_first_t


def _precompute_inbound(s: _StateNP) -> np.ndarray:
    """预估每个星球收到的我方/敌方舰队压力。

    返回 [P, 6]：(my_ships, enemy_ships, my_eta_min, my_eta_mean,
                  enemy_eta_min, enemy_eta_mean)
    对应 features_jax.precompute_inbound。
    """
    P = s.planets.shape[0]
    norm = float(INBOUND_HORIZON)
    empty_row = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    inbound = np.tile(empty_row, (P, 1))  # [P, 6]

    fleet_target, fleet_valid, fleet_first_t = _predict_fleet_targets(s)
    F = s.fleets.shape[0]
    if F == 0:
        # 死星球行保持 empty_row
        inbound[~s.planet_alive] = empty_row
        return inbound

    fleet_owner = s.fleets[:, 1].astype(np.int32)  # [F]

    for p_idx in range(P):
        if not s.planet_alive[p_idx]:
            continue
        sel = fleet_valid & (fleet_target == p_idx)  # [F]
        if not sel.any():
            continue
        my  = sel & (fleet_owner == 0)
        en  = sel & (fleet_owner != 0)
        my_ships  = float(s.fleets[my,  6].sum()) if my.any()  else 0.0
        en_ships  = float(s.fleets[en,  6].sum()) if en.any()  else 0.0
        t_arr = fleet_first_t  # [F]
        my_eta = t_arr[my]
        en_eta = t_arr[en]
        my_min  = float(my_eta.min())  / norm if my.any()  else 1.0
        my_mean = float(my_eta.mean()) / norm if my.any()  else 1.0
        en_min  = float(en_eta.min())  / norm if en.any()  else 1.0
        en_mean = float(en_eta.mean()) / norm if en.any()  else 1.0
        inbound[p_idx] = [my_ships, en_ships, my_min, my_mean, en_min, en_mean]

    return inbound  # [P, 6]


# ─────────────────────────────────────────────────────────────
# Part 3: encode_batch 各子函数
# 对应 features_jax.py 全部子函数，完全用 numpy 重写
# ─────────────────────────────────────────────────────────────

def _proto_score_np(src_row: np.ndarray, planets: np.ndarray) -> np.ndarray:
    """Proto-V15 目标评分，对应 features_jax.proto_score（向量化）。

    src_row: [7]  (单颗 source 星球行)
    planets: [P, 7]
    返回: [P]
    """
    dist = np.sqrt((planets[:, 2] - src_row[2]) ** 2 + (planets[:, 3] - src_row[3]) ** 2)
    min_ships = planets[:, 5] + 1.0
    speed = _fleet_speed_np(min_ships)
    eta = dist / np.maximum(speed, 1e-6)
    has_owner = planets[:, 1] != -1.0
    enemy_produced = np.where(has_owner, eta * planets[:, 6], 0.0)
    enemy_bonus    = np.where(has_owner, planets[:, 6], 0.0)
    total_ships    = min_ships + enemy_produced
    return (100.0 - dist) + 15.0 * planets[:, 6] + 10.0 * enemy_bonus - 0.7 * total_ships - 2.0 * eta


def _build_global_features_np(s: _StateNP, max_planets: int, max_ships: float,
                               max_production: float, episode_steps: int) -> np.ndarray:
    """全局局势特征 [GLOBAL_DIM=11]，对应 features_jax.build_global_features。"""
    planets = s.planets
    alive   = s.planet_alive
    owner   = planets[:, 1].astype(np.int32)
    my      = alive & (owner == 0)
    enemy   = alive & (owner == 1)
    neutral = alive & (owner == -1)
    fowner  = s.fleets[:, 1].astype(np.int32)
    my_f    = s.fleet_alive & (fowner == 0)
    en_f    = s.fleet_alive & (fowner == 1)
    return np.array([
        s.step / episode_steps,
        s.angular_velocity / 0.05,
        my.sum()      / max_planets,
        enemy.sum()   / max_planets,
        neutral.sum() / max_planets,
        np.where(my,   planets[:, 5], 0.0).sum() / (max_planets * max_ships),
        np.where(enemy, planets[:, 5], 0.0).sum() / (max_planets * max_ships),
        np.where(my_f,  s.fleets[:, 6], 0.0).sum() / (max_planets * max_ships),
        np.where(en_f,  s.fleets[:, 6], 0.0).sum() / (max_planets * max_ships),
        np.where(my,   planets[:, 6], 0.0).sum() / (max_planets * max_production),
        np.where(enemy, planets[:, 6], 0.0).sum() / (max_planets * max_production),
    ], dtype=np.float32)


def _planet_local_topology_np(s: _StateNP, src_idx: int, max_planets: int) -> np.ndarray:
    """[6]：最近我方/敌方/中立距离 + 局部密度。对应 planet_local_topology_features。"""
    planets = s.planets
    alive   = s.planet_alive
    owner   = planets[:, 1].astype(np.int32)
    not_self = np.arange(max_planets) != src_idx
    valid    = alive & not_self
    dx = planets[:, 2] - planets[src_idx, 2]
    dy = planets[:, 3] - planets[src_idx, 3]
    dist = np.sqrt(dx * dx + dy * dy)

    my_mask      = valid & (owner == 0)
    enemy_mask   = valid & (owner == 1)
    neutral_mask = valid & (owner == -1)

    def nearest(mask):
        if not mask.any():
            return 1.0
        return float(np.where(mask, dist, 1e9).min()) / BOARD_SIZE

    def count_in_r(mask):
        return float((mask & (dist <= LOCAL_PLANET_RADIUS)).sum()) / max_planets

    return np.array([
        nearest(my_mask),
        nearest(enemy_mask),
        nearest(neutral_mask),
        count_in_r(my_mask),
        count_in_r(enemy_mask),
        count_in_r(neutral_mask),
    ], dtype=np.float32)


def _build_planet_features_np(s: _StateNP, src_idx: int, inbound: np.ndarray,
                               max_planets: int, max_ships: float,
                               max_production: float, comet_mask: np.ndarray) -> np.ndarray:
    """单颗星球局部特征 [PLANET_DIM=26]，对应 features_jax.build_planet_features。"""
    src   = s.planets[src_idx]
    alive = s.planet_alive[src_idx]
    owner = int(src[1])
    is_comet     = comet_mask[src_idx]
    init_x, init_y = s.initial_planets[src_idx, 2], s.initial_planets[src_idx, 3]
    rotating = (not is_comet) and bool(
        (math.sqrt((init_x - CENTER) ** 2 + (init_y - CENTER) ** 2) + src[4]) < ROTATION_RADIUS_LIMIT
    ) and alive
    is_static = alive and (not rotating) and (not is_comet)
    topo = _planet_local_topology_np(s, src_idx, max_planets)
    inb  = inbound[src_idx]   # [6]
    return np.array([
        1.0,
        float(alive),
        float(owner == -1),
        float(owner == 0),
        float(owner == 1),
        src[2] / BOARD_SIZE,
        src[3] / BOARD_SIZE,
        src[4] / 5.0,
        float(np.clip(src[5] / max_ships, 0.0, 1.0)),
        src[6] / max_production,
        float(is_static),
        float(rotating),
        float(is_comet),
        math.sqrt((src[2] - CENTER) ** 2 + (src[3] - CENTER) ** 2) / BOARD_SIZE,
        topo[0], topo[1], topo[2], topo[3], topo[4], topo[5],
        float(np.clip(inb[0] / max_ships, 0.0, 1.0)),
        float(np.clip(inb[1] / max_ships, 0.0, 1.0)),
        float(inb[2]), float(inb[3]), float(inb[4]), float(inb[5]),
    ], dtype=np.float32)


def _build_ship_buckets_np(src_ships: float, ship_bucket_count: int) -> np.ndarray:
    """源星球派兵档位 [B]，对应 features_jax.build_ship_buckets。"""
    ratios = SHIP_RATIOS[:ship_bucket_count]
    s = math.floor(src_ships)
    raw = np.clip(np.round(s * ratios), MIN_LAUNCH, s)
    return np.where(s >= MIN_LAUNCH, raw, np.zeros(ship_bucket_count, dtype=np.float32)).astype(np.float32)


def _build_relative_features_np(
    s: _StateNP, src_idx: int, inbound: np.ndarray,
    max_planets: int, max_ships: float, ship_bucket_count: int,
    target_filter_enabled: bool, attack_top_k: int, defense_top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """source-target 相对特征 + mask + 档位，对应 features_jax.build_relative_features。

    返回: (rel_feat[P+1, 8], cand_mask[P+1], amount_mask[P+1, B],
           candidate_ids[P+1], ship_buckets[B])
    """
    src     = s.planets[src_idx]
    planets = s.planets
    alive   = s.planet_alive
    n_tgt   = max_planets + 1

    candidate_ids = np.concatenate([np.array([-1], dtype=np.int32),
                                    planets[:, 0].astype(np.int32)])
    ship_buckets = _build_ship_buckets_np(float(src[5]), ship_bucket_count)
    valid_buckets = ship_buckets > 0
    max_bucket = float(max(ship_buckets.max(), 1.0))
    spd_max = _fleet_speed_scalar(max_bucket)

    # slot 0 = no-op
    noop_feat = np.zeros((1, RELATIVE_DIM), dtype=np.float32)
    dx = planets[:, 2] - src[2]
    dy = planets[:, 3] - src[3]
    dist = np.sqrt(dx * dx + dy * dy)
    owner    = planets[:, 1].astype(np.int32)
    is_source = np.arange(max_planets) == src_idx
    is_neutral = owner == -1
    is_friend  = (owner == 0) & (~is_source)
    is_enemy   = owner == 1
    proto = _proto_score_np(src, planets) / 100.0
    eta_max = dist / max(spd_max, 1e-6) / INBOUND_HORIZON

    # defense_pressure（对应 features_jax.compute_defense_pressure）
    my_inbound   = inbound[:, 0]
    en_inbound   = inbound[:, 1]
    en_eta_turns = inbound[:, 4] * INBOUND_HORIZON
    proj_def  = planets[:, 5] + my_inbound + en_eta_turns * planets[:, 6]
    def_press = np.where(en_inbound > 0.0, en_inbound - proj_def, 0.0)

    rel = np.stack([
        is_source.astype(np.float32),
        dx / BOARD_SIZE,
        dy / BOARD_SIZE,
        dist / BOARD_SIZE,
        np.clip((planets[:, 5] + 1.0) / max(float(src[5]), 1.0), 0.0, 1.0),
        np.full(max_planets, np.clip(src[5] / max_ships, 0.0, 1.0), dtype=np.float32),
        proto,
        eta_max,
    ], axis=-1)  # [P, 8]
    relative_features = np.concatenate([noop_feat, rel], axis=0)  # [P+1, 8]

    # target mask（对应 features_jax.build_relative_features 的 target_valid）
    attackable = src[5] > planets[:, 5]
    target_valid = alive & (~is_source) & (is_friend | ((is_enemy | is_neutral) & attackable))
    per_planet_amount = np.broadcast_to(
        valid_buckets[None, :], (max_planets, ship_bucket_count)
    ).copy()

    mask = np.concatenate([[True], target_valid])          # [P+1]
    amount_mask = np.concatenate([
        np.ones((1, ship_bucket_count), dtype=bool),
        per_planet_amount,
    ], axis=0)                                              # [P+1, B]
    amount_mask = amount_mask & mask[:, None]
    amount_mask[0, :] = True

    if target_filter_enabled:
        # apply_target_filter 对应 features_jax.apply_target_filter
        planet_mask = mask[1:]
        keep = np.zeros(max_planets, dtype=bool)

        atk_score = np.where(planet_mask & (is_enemy | is_neutral), proto, -1e9)
        atk_order = np.argsort(atk_score)[::-1]
        top_atk = atk_order[:attack_top_k]
        keep[top_atk[atk_score[top_atk] > -1e8]] = True

        def_score = np.where(planet_mask & is_friend & (def_press > 0.0), def_press, -1e9)
        def_order = np.argsort(def_score)[::-1]
        top_def = def_order[:defense_top_k]
        keep[top_def[def_score[top_def] > -1e8]] = True

        mask = np.concatenate([[mask[0]], keep])
        amount_mask = amount_mask & mask[:, None]
        amount_mask[0, :] = True

    return relative_features, mask, amount_mask, candidate_ids, ship_buckets


def _build_fleet_features_np(
    s: _StateNP, max_planets: int, max_ships: float,
    max_production: float, episode_steps: int, max_fleet_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """舰队 token 特征 [max_fleet_tokens, FLEET_DIM=28]，对应 features_jax.build_fleet_features。"""
    fleets = s.fleets
    alive  = s.fleet_alive
    F = fleets.shape[0]

    dist_center = np.sqrt((fleets[:, 2] - CENTER) ** 2 + (fleets[:, 3] - CENTER) ** 2)
    score = np.where(alive, -fleets[:, 6] + 0.01 * dist_center, np.inf)
    order = np.argsort(score)[:max_fleet_tokens]

    sel       = fleets[order]          # [K, 7]
    sel_alive = alive[order]           # [K]
    sel_owner = sel[:, 1].astype(np.int32)
    from_id   = sel[:, 5].astype(np.int32)

    # 找 from_planet slot
    planet_ids_arr = s.planets[:, 0].astype(np.int32)
    from_slot = np.zeros(max_fleet_tokens, dtype=np.int32)
    has_from  = np.zeros(max_fleet_tokens, dtype=bool)
    for ki in range(max_fleet_tokens):
        if not sel_alive[ki]:
            continue
        matches = (planet_ids_arr == from_id[ki]) & s.planet_alive
        if matches.any():
            from_slot[ki] = int(np.argmax(matches))
            has_from[ki]  = True

    from_planets = s.planets[from_slot]  # [K, 7]

    speed = _fleet_speed_np(np.maximum(sel[:, 6], 1.0))

    # 预测每支舰队目标（重用已有函数）
    ft, fv, ftt = _predict_fleet_targets(s)
    target_idx   = ft[order]   # [K]
    target_valid = sel_alive & fv[order]
    target_planets = s.planets[np.clip(target_idx, 0, max_planets - 1)]
    target_owner = target_planets[:, 1].astype(np.int32)
    target_dx = target_planets[:, 2] - sel[:, 2]
    target_dy = target_planets[:, 3] - sel[:, 3]
    target_dist = np.sqrt(target_dx ** 2 + target_dy ** 2)
    target_eta  = ftt[order] / INBOUND_HORIZON

    # nearest planet distances for fleet tokens
    planet_owner = s.planets[:, 1].astype(np.int32)
    p_alive      = s.planet_alive
    dx_p = s.planets[:, None, 2] - sel[None, :, 2]   # [P, K]
    dy_p = s.planets[:, None, 3] - sel[None, :, 3]
    dist_p = np.sqrt(dx_p ** 2 + dy_p ** 2)           # [P, K]

    def nearest_dist(mask):  # mask [P]
        d = np.where(mask[:, None], dist_p, np.inf).min(axis=0)  # [K]
        return np.where(np.isfinite(d), d / BOARD_SIZE, 1.0)

    dist_from_origin = np.sqrt(
        (sel[:, 2] - from_planets[:, 2]) ** 2 +
        (sel[:, 3] - from_planets[:, 3]) ** 2
    )
    launch_offset = from_planets[:, 4] + 0.1
    launch_age = np.maximum(dist_from_origin - launch_offset, 0.0) / np.maximum(speed, 1e-6)

    features = np.stack([
        (sel_owner == 0).astype(np.float32),
        sel[:, 2] / BOARD_SIZE,
        sel[:, 3] / BOARD_SIZE,
        np.cos(sel[:, 4]),
        np.sin(sel[:, 4]),
        np.clip(sel[:, 6] / max_ships, 0.0, 1.0),
        speed / 6.0,
        np.clip(launch_age / episode_steps, 0.0, 1.0),
        np.where(has_from, from_planets[:, 2] / BOARD_SIZE, 0.0),
        np.where(has_from, from_planets[:, 3] / BOARD_SIZE, 0.0),
        np.where(has_from, (from_planets[:, 1].astype(np.int32) == 0).astype(np.float32), 0.0),
        np.where(has_from, np.clip(from_planets[:, 5] / max_ships, 0.0, 1.0), 0.0),
        np.where(has_from, dist_from_origin / BOARD_SIZE, 0.0),
        nearest_dist(p_alive & (planet_owner == 1)),
        nearest_dist(p_alive & (planet_owner == -1)),
        nearest_dist(p_alive & (planet_owner == 0)),
        target_valid.astype(np.float32),
        (target_valid & (target_owner == 0)).astype(np.float32),
        (target_valid & (target_owner == 1)).astype(np.float32),
        (target_valid & (target_owner == -1)).astype(np.float32),
        np.where(target_valid, target_planets[:, 2] / BOARD_SIZE, 0.0),
        np.where(target_valid, target_planets[:, 3] / BOARD_SIZE, 0.0),
        np.where(target_valid, np.clip(target_planets[:, 5] / max_ships, 0.0, 1.0), 0.0),
        np.where(target_valid, target_planets[:, 6] / max_production, 0.0),
        np.where(target_valid, target_dx / BOARD_SIZE, 0.0),
        np.where(target_valid, target_dy / BOARD_SIZE, 0.0),
        np.where(target_valid, target_dist / BOARD_SIZE, 0.0),
        np.where(target_valid, target_eta, 0.0),
    ], axis=-1)  # [K, 28]

    features = np.where(sel_alive[:, None], features, 0.0)
    return features.astype(np.float32), sel_alive.astype(bool)


def _build_map_features_np(s: _StateNP, src_idx: int, max_ships: float) -> np.ndarray:
    """小地图 [6, 8, 8]，对应 features_jax.build_map_features。"""
    grid = np.zeros((MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE), dtype=np.float32)
    planets = s.planets
    owner   = planets[:, 1].astype(np.int32)
    idx_arr = np.arange(planets.shape[0])
    channels = np.where(
        idx_arr == src_idx, 0,
        np.where(owner == 0, 1,
        np.where(owner == -1, 3, 2))
    )
    values = s.planet_alive.astype(np.float32)

    in_bounds_p = (
        (planets[:, 2] >= 0.0) & (planets[:, 2] <= BOARD_SIZE) &
        (planets[:, 3] >= 0.0) & (planets[:, 3] <= BOARD_SIZE) &
        (values > 0.0)
    )
    ix_p = np.clip((planets[:, 2] / BOARD_SIZE * MAP_GRID_SIZE).astype(np.int32), 0, MAP_GRID_SIZE - 1)
    iy_p = np.clip((planets[:, 3] / BOARD_SIZE * MAP_GRID_SIZE).astype(np.int32), 0, MAP_GRID_SIZE - 1)
    for i in range(planets.shape[0]):
        if in_bounds_p[i]:
            grid[channels[i], iy_p[i], ix_p[i]] = min(1.0, grid[channels[i], iy_p[i], ix_p[i]] + values[i])

    ship_channels = np.where(s.fleets[:, 1].astype(np.int32) == 0, 4, 5)
    ship_values = np.clip(s.fleets[:, 6] / max_ships, 0.0, 1.0) * s.fleet_alive.astype(np.float32)
    in_bounds_f = (
        (s.fleets[:, 2] >= 0.0) & (s.fleets[:, 2] <= BOARD_SIZE) &
        (s.fleets[:, 3] >= 0.0) & (s.fleets[:, 3] <= BOARD_SIZE) &
        (ship_values > 0.0)
    )
    ix_f = np.clip((s.fleets[:, 2] / BOARD_SIZE * MAP_GRID_SIZE).astype(np.int32), 0, MAP_GRID_SIZE - 1)
    iy_f = np.clip((s.fleets[:, 3] / BOARD_SIZE * MAP_GRID_SIZE).astype(np.int32), 0, MAP_GRID_SIZE - 1)
    for i in range(s.fleets.shape[0]):
        if in_bounds_f[i]:
            grid[ship_channels[i], iy_f[i], ix_f[i]] = min(1.0, grid[ship_channels[i], iy_f[i], ix_f[i]] + ship_values[i])

    return grid


# ─────────────────────────────────────────────────────────────
# Part 4a: encode_batch_np —— 主接口（替换 jax.jit(encode_batch)）
# 对应 features_jax.encode_batch / encode_single
# ─────────────────────────────────────────────────────────────

class EncodedBatchNP:
    """encode_batch_np 的输出容器，字段与 JAX EncodedBatch 对齐（numpy 版）。

    所有数组均带 batch=1 维，方便直接转 jnp 喂 ONNX 输入。
    """
    __slots__ = (
        "planet_features", "relative_features", "global_features",
        "map_features", "candidate_mask", "amount_mask",
        "source_ids", "source_ships", "candidate_ids",
        "ship_buckets", "source_mask", "fleet_features", "fleet_mask",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def encode_batch_np(jax_state, feat_params) -> EncodedBatchNP:
    """纯 numpy encode_batch，零 JAX JIT 依赖。

    参数：
      jax_state   — batch=1 的 JAX EnvState（直接从 main.py 传入）
      feat_params — FeatureParams dataclass（与 JAX 版相同）

    返回 EncodedBatchNP，字段形状带 batch=1，与 JAX EncodedBatch 一致：
      planet_features   [1, P, P, 26]
      relative_features [1, P, P+1, 8]
      global_features   [1, P, 11]
      map_features      [1, P, 6, 8, 8]
      candidate_mask    [1, P, P+1]
      amount_mask       [1, P, P+1, B]
      source_ids        [1, P]
      source_ships      [1, P]
      candidate_ids     [1, P, P+1]
      ship_buckets      [1, P, B]
      source_mask       [1, P]
      fleet_features    [1, P, K, 28]
      fleet_mask        [1, P, K]
    """
    s = _to_np(jax_state)
    P  = feat_params.max_planets
    B  = feat_params.ship_bucket_count
    K  = feat_params.max_fleet_tokens

    comet_mask = _comet_planet_mask_np(s)
    inbound    = _precompute_inbound(s)       # [P, 6]
    global_f   = _build_global_features_np(
        s, P, feat_params.max_ships,
        feat_params.max_production, feat_params.episode_steps,
    )  # [11]

    # 舰队 token（每颗 source 共享同一组 fleet tokens）
    fleet_f, fleet_m = _build_fleet_features_np(
        s, P, feat_params.max_ships,
        feat_params.max_production, feat_params.episode_steps, K,
    )  # [K, 28], [K]

    # 逐 source 星球构建行
    planet_features_list   = []
    relative_features_list = []
    global_features_list   = []
    map_features_list      = []
    candidate_mask_list    = []
    amount_mask_list       = []
    candidate_ids_list     = []
    ship_buckets_list      = []
    source_mask_list       = []

    owner_arr = s.planets[:, 1].astype(np.int32)
    is_mine   = s.planet_alive & (owner_arr == 0)

    for src_idx in range(P):
        pf = _build_planet_features_np(
            s, src_idx, inbound, P,
            feat_params.max_ships, feat_params.max_production, comet_mask,
        )  # [26]
        rf, cm, am, cids, sb = _build_relative_features_np(
            s, src_idx, inbound, P,
            feat_params.max_ships, B,
            feat_params.target_filter_enabled,
            feat_params.attack_top_k, feat_params.defense_top_k,
        )  # [P+1,8], [P+1], [P+1,B], [P+1], [B]
        mf = _build_map_features_np(s, src_idx, feat_params.max_ships)  # [6,8,8]

        planet_features_list.append(pf)
        relative_features_list.append(rf)
        global_features_list.append(global_f)
        map_features_list.append(mf)
        candidate_mask_list.append(cm)
        amount_mask_list.append(am)
        candidate_ids_list.append(cids)
        ship_buckets_list.append(sb)
        source_mask_list.append(bool(is_mine[src_idx]))

    # stack → [P, ...]，再加 batch=1 维
    def b(arr):
        return np.asarray(arr)[None]  # [1, ...]

    return EncodedBatchNP(
        planet_features   = b(np.stack(planet_features_list,   axis=0)),   # [1,P,26]
        relative_features = b(np.stack(relative_features_list, axis=0)),   # [1,P,P+1,8]
        global_features   = b(np.stack(global_features_list,   axis=0)),   # [1,P,11]
        map_features      = b(np.stack(map_features_list,       axis=0)),   # [1,P,6,8,8]
        candidate_mask    = b(np.stack(candidate_mask_list,     axis=0)),   # [1,P,P+1]
        amount_mask       = b(np.stack(amount_mask_list,        axis=0)),   # [1,P,P+1,B]
        source_ids        = b(s.planets[:, 0].astype(np.int32)),            # [1,P]
        source_ships      = b(s.planets[:, 5].astype(np.float32)),          # [1,P]
        candidate_ids     = b(np.stack(candidate_ids_list,      axis=0)),   # [1,P,P+1]
        ship_buckets      = b(np.stack(ship_buckets_list,       axis=0)),   # [1,P,B]
        source_mask       = b(np.array(source_mask_list, dtype=bool)),      # [1,P]
        fleet_features    = b(np.stack([fleet_f] * P,           axis=0)),   # [1,P,K,28]
        fleet_mask        = b(np.stack([fleet_m] * P,           axis=0)),   # [1,P,K]
    )


# ─────────────────────────────────────────────────────────────
# Part 4b: dynamic_ship_buckets_np
# 对应 train_jax.dynamic_ship_buckets（向量化 numpy）
# ─────────────────────────────────────────────────────────────

def dynamic_ship_buckets_np(ships: np.ndarray, bucket_count: int) -> np.ndarray:
    """根据剩余兵力动态生成派兵档位。

    ships: [N] 各位置剩余船数
    返回: [N, bucket_count]

    对应 train_jax.dynamic_ship_buckets，逻辑完全一致。
    """
    ratios = SHIP_RATIOS[:bucket_count]           # [B]
    s = np.floor(np.asarray(ships, dtype=np.float32))   # [N]
    raw = np.clip(
        np.round(s[:, None] * ratios[None, :]),
        10.0,
        s[:, None],
    )  # [N, B]
    return np.where(s[:, None] >= 10.0, raw, np.zeros_like(raw))


# ─────────────────────────────────────────────────────────────
# Part 4c: aim_with_prediction_np
# 对应 train_jax.aim_with_prediction_jax
# 策略：从 torch_agent/physics.py 搬运标量版逻辑，外层 vectorize
# ─────────────────────────────────────────────────────────────

def _target_pos_np(
    s: _StateNP,
    target_slot: int,
    T: float,
) -> tuple[float, float] | None:
    """目标星球在浮点 T 回合后的位置。

    支持静止星球、旋转星球、彗星（线性插值）。
    对应 train_jax 的 target_pos_at_time_jax（标量版）。
    """
    if target_slot < 0 or target_slot >= s.planets.shape[0]:
        return None
    if not s.planet_alive[target_slot]:
        return None

    # ── 彗星 ────────────────────────────────────────────────
    for si in range(MAX_COMET_SPAWNS):
        pi = int(s.comet_path_index[si])
        if pi < -1:
            continue
        for ci in range(COMETS_PER_SPAWN):
            if int(s.comet_slots[si, ci]) != target_slot:
                continue
            path_len = int(s.comet_lengths[si, ci])
            path     = s.comet_paths[si, ci]  # [40, 2]
            fi = pi + T
            i0 = int(fi)
            frac = fi - i0
            if i0 < 0 or i0 >= path_len or i0 >= MAX_COMET_PATH:
                return None
            x0, y0 = float(path[i0, 0]), float(path[i0, 1])
            if frac == 0.0 or i0 + 1 >= path_len or i0 + 1 >= MAX_COMET_PATH:
                return x0, y0
            return (x0 + frac * (float(path[i0 + 1, 0]) - x0),
                    y0 + frac * (float(path[i0 + 1, 1]) - y0))

    # ── 旋转 / 静止星球 ──────────────────────────────────────
    p    = s.planets[target_slot]
    init = s.initial_planets[target_slot]
    dx   = float(init[2]) - CENTER
    dy   = float(init[3]) - CENTER
    r    = math.sqrt(dx * dx + dy * dy)
    if r + float(init[4]) < ROTATION_RADIUS_LIMIT:
        phi_cur = math.atan2(float(p[3]) - CENTER, float(p[2]) - CENTER)
        phi     = phi_cur + s.angular_velocity * T
        return CENTER + r * math.cos(phi), CENTER + r * math.sin(phi)
    return float(p[2]), float(p[3])


def _target_max_turns_np(s: _StateNP, target_slot: int, horizon: int) -> int:
    """目标可搜索最大回合数：普通星球=horizon；彗星=min(horizon, 剩余路径-1)。

    对应 train_jax.target_max_turns_jax（标量版）。
    """
    for si in range(MAX_COMET_SPAWNS):
        pi = int(s.comet_path_index[si])
        if pi < -1:
            continue
        for ci in range(COMETS_PER_SPAWN):
            if int(s.comet_slots[si, ci]) == target_slot:
                remaining = int(s.comet_lengths[si, ci]) - pi
                return min(horizon, max(0, remaining - 1))
    return horizon


def _aim_f_np(
    src_x: float, src_y: float, src_r: float,
    tgt_slot: int, s: _StateNP, speed: float, T: float,
) -> float:
    """瞄准函数 f(T)，对应 train_jax.aim_f_jax（标量版）。"""
    pos = _target_pos_np(s, tgt_slot, T)
    if pos is None:
        return math.inf
    tgt_r = float(s.planets[tgt_slot, 4])
    dist  = math.hypot(pos[0] - src_x, pos[1] - src_y)
    return dist - (src_r + LAUNCH_CLEARANCE + T * speed) - tgt_r


def _target_is_first_collision_np(
    s: _StateNP,
    tgt_slot: int,
    fx0: float, fy0: float,
    speed: float, cos_a: float, sin_a: float,
    hit_turn: int,
) -> bool:
    """验证此轨迹在 hit_turn 前不先撞其他星球。

    对应 train_jax.target_is_first_collision_jax（标量版）。
    """
    P = s.planets.shape[0]
    for step in range(1, hit_turn + 1):
        fx_prev = fx0 + (step - 1) * speed * cos_a
        fy_prev = fy0 + (step - 1) * speed * sin_a
        fx_cur  = fx0 +  step      * speed * cos_a
        fy_cur  = fy0 +  step      * speed * sin_a
        # 出界即失效
        if not (0.0 <= fx_cur <= BOARD_SIZE and 0.0 <= fy_cur <= BOARD_SIZE):
            return False

        # 移动阶段：用 step-1 的星球位置
        moving_hit_target = False
        for p_idx in range(P):
            if not s.planet_alive[p_idx]:
                continue
            pos = _target_pos_np(s, p_idx, step - 1)
            if pos is None:
                continue
            d = _point_segment_dist_scalar(pos[0], pos[1], fx_prev, fy_prev, fx_cur, fy_cur)
            if d < float(s.planets[p_idx, 4]):
                if p_idx == tgt_slot and step == hit_turn:
                    moving_hit_target = True
                    break
                return False
        if moving_hit_target:
            return True

        # 扫过阶段
        for p_idx in range(P):
            if not s.planet_alive[p_idx]:
                continue
            old_pos = _target_pos_np(s, p_idx, step - 1)
            new_pos = _target_pos_np(s, p_idx, step)
            if old_pos is None or new_pos is None:
                continue
            if old_pos == new_pos:
                continue
            d = _point_segment_dist_scalar(
                fx_cur, fy_cur,
                old_pos[0], old_pos[1], new_pos[0], new_pos[1],
            )
            if d < float(s.planets[p_idx, 4]):
                if p_idx == tgt_slot and step == hit_turn:
                    return True
                return False
    return False


def _aim_single_np(
    s: _StateNP,
    src_x: float, src_y: float, src_r: float,
    tgt_slot: int, ships: float,
) -> tuple[float, bool]:
    """单对目标的瞄准求解，返回 (angle, hit_valid)。

    对应 train_jax.aim_with_prediction_jax 的单元素逻辑：
      fixed-point 求连续 T* → 3 个整数回合 x 3 个瞄准角
    """
    speed     = _fleet_speed_scalar(max(ships, 1.0))
    max_turns = _target_max_turns_np(s, tgt_slot, _AIM_HORIZON)

    if max_turns < 1 or not s.planet_alive[tgt_slot]:
        tgt_x = float(s.planets[tgt_slot, 2])
        tgt_y = float(s.planets[tgt_slot, 3])
        return math.atan2(tgt_y - src_y, tgt_x - src_x), False

    pos0 = _target_pos_np(s, tgt_slot, 0.0)
    if pos0 is None:
        tgt_x = float(s.planets[tgt_slot, 2])
        tgt_y = float(s.planets[tgt_slot, 3])
        return math.atan2(tgt_y - src_y, tgt_x - src_x), False

    gap = src_r + LAUNCH_CLEARANCE + float(s.planets[tgt_slot, 4])
    t_star = min(max((math.hypot(pos0[0] - src_x, pos0[1] - src_y) - gap) / speed, 0.0), float(max_turns))
    for _ in range(_AIM_FP_ITERS):
        pos = _target_pos_np(s, tgt_slot, t_star)
        if pos is None:
            break
        dist = math.hypot(pos[0] - src_x, pos[1] - src_y)
        t_star = min(max((dist - gap) / speed, 0.0), float(max_turns))

    # ── 整数精挑 3 候选 ──────────────────────────────────────
    T_floor     = int(t_star)
    candidates  = sorted({max(1, T_floor - 1), max(1, T_floor), T_floor + 1})
    candidates  = [T for T in candidates if 1 <= T <= max_turns]

    for T in candidates:
        pos_star = _target_pos_np(s, tgt_slot, t_star)
        pos_old = _target_pos_np(s, tgt_slot, T - 1)
        pos_next = _target_pos_np(s, tgt_slot, T)
        if pos_old is None:
            continue
        aim_positions = []
        for pos in (pos_star, pos_old, pos_next):
            if pos is None:
                continue
            if not any(abs(pos[0] - old[0]) < 1e-7 and abs(pos[1] - old[1]) < 1e-7 for old in aim_positions):
                aim_positions.append(pos)
        for aim_pos in aim_positions:
            px, py   = aim_pos
            theta    = math.atan2(py - src_y, px - src_x)
            cos_a    = math.cos(theta)
            sin_a    = math.sin(theta)
            fx0      = src_x + cos_a * (src_r + LAUNCH_CLEARANCE)
            fy0      = src_y + sin_a * (src_r + LAUNCH_CLEARANCE)
            fx_T     = fx0 + T * speed * cos_a
            fy_T     = fy0 + T * speed * sin_a

            # 太阳碰撞
            if _point_segment_dist_scalar(CENTER, CENTER, fx0, fy0, fx_T, fy_T) < SUN_RADIUS:
                continue

            # 第一碰撞验证
            if not _target_is_first_collision_np(s, tgt_slot, fx0, fy0, speed, cos_a, sin_a, T):
                continue

            # 最终命中检测。提交版保守内缩一点，避开原环境 strict < radius 的贴边误差。
            fx_prev = fx0 + (T - 1) * speed * cos_a
            fy_prev = fy0 + (T - 1) * speed * sin_a
            tgt_r   = max(float(s.planets[tgt_slot, 4]) - AIM_TARGET_INSET, 0.0)
            if _point_segment_dist_scalar(pos_old[0], pos_old[1], fx_prev, fy_prev, fx_T, fy_T) < tgt_r:
                return theta, True
            # 扫过检测
            if pos_next is not None and pos_next != pos_old:
                if _point_segment_dist_scalar(fx_T, fy_T, pos_old[0], pos_old[1], pos_next[0], pos_next[1]) < tgt_r:
                    return theta, True

    # 全部失败：返回直连角度
    tgt_x = float(s.planets[tgt_slot, 2])
    tgt_y = float(s.planets[tgt_slot, 3])
    return math.atan2(tgt_y - src_y, tgt_x - src_x), False


def aim_with_prediction_np(
    src_planets: np.ndarray,   # [B, 1, 7]  (batch, n_actions, planet_cols)
    tgt_planets: np.ndarray,   # [B, 1, 7]
    tgt_slot: np.ndarray,      # [B, 1]  int
    jax_state,                 # batch=1 JAX EnvState
    ships: np.ndarray,         # [B, 1]
) -> tuple[np.ndarray, np.ndarray]:
    """纯 numpy 版 aim_with_prediction_jax，接口与 JAX 版完全兼容。

    对应 train_jax.aim_with_prediction_jax（外部调用签名相同）。
    返回 (angle[B,1], hit_valid[B,1])，dtype=float32/bool。
    """
    s = _to_np(jax_state)
    B = src_planets.shape[0]
    N = src_planets.shape[1]   # 通常=1

    angles    = np.zeros((B, N), dtype=np.float32)
    hit_valid = np.zeros((B, N), dtype=bool)

    for b in range(B):
        for n in range(N):
            slot = int(tgt_slot[b, n])
            src  = src_planets[b, n]
            ang, hit = _aim_single_np(
                s,
                float(src[2]), float(src[3]), float(src[4]),
                slot, float(ships[b, n]),
            )
            angles[b, n]    = ang
            hit_valid[b, n] = hit

    return angles, hit_valid
