"""物理和瞄准模块。

从 rule_based/physics.py 适配，移除规则层依赖，仅保留特征工程和
动作生成所需的几何/预测函数。

核心是 aim_with_prediction：粗扫描 + 二分法求解旋转/彗星目标的
发射角度（theta）和预计到达时间（ETA，回合数）。
"""

from __future__ import annotations

import math

# ── 与 orbit_wars.py 引擎一致的物理常量 ─────────────────────────────────────
CENTER_X: float = 50.0
CENTER_Y: float = 50.0
BOARD_SIZE: float = 100.0
SUN_RADIUS: float = 10.0
ROTATION_LIMIT: float = 50.0   # 轨道半径 + 星球半径 < 此值时旋转
LAUNCH_CLEARANCE: float = 0.1  # 发射偏移，与引擎 process_moves 硬编码一致
MAX_SPEED: float = 6.0
ROUTE_SEARCH_HORIZON: int = 60  # 瞄准搜索最大回合数

# 训练时适当减少采样数以控制耗时；
# 60/12 与 rule_based 的 180/16 在大多数场景结果一致。
_N_COARSE: int = 60
_N_BISECT: int = 12


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────────────────────────

def fleet_speed(ships: int) -> float:
    """根据派出舰队数量计算飞行速度（与 orbit_wars.py 公式完全一致）。

    速度随舰队规模对数增长，最大为 MAX_SPEED。
    """
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def is_static_planet(planet) -> bool:
    """判断星球是否静止（不在内圈旋转轨道）。"""
    r = math.hypot(planet.x - CENTER_X, planet.y - CENTER_Y)
    return r + planet.radius >= ROTATION_LIMIT


def point_to_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """点到线段的最短距离。"""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def segment_hits_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    """检测直线路径是否穿过太阳（与引擎碰撞判定完全一致）。"""
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_RADIUS


def in_bounds(x: float, y: float) -> bool:
    """判断舰队当前位置是否仍在 0~100 地图内。"""
    return 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# 参数方程：旋转星球位置
# ─────────────────────────────────────────────────────────────────────────────

def planet_pos(
    planet,
    initial_by_id: dict,
    ang_vel: float,
    T: float,
) -> tuple[float, float]:
    """旋转星球在 T 回合后的位置（T=0 为当前）。

    参数方程（来自 orbit_wars.py）：
      x = CENTER + r * cos(φ_cur + ω * T)
      y = CENTER + r * sin(φ_cur + ω * T)

    仅当 r + init.radius < ROTATION_LIMIT 时旋转；否则返回当前坐标。
    """
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    dx = init.x - CENTER_X
    dy = init.y - CENTER_Y
    r = math.hypot(dx, dy)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    phi_cur = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    phi = phi_cur + ang_vel * T
    return CENTER_X + r * math.cos(phi), CENTER_Y + r * math.sin(phi)


# ─────────────────────────────────────────────────────────────────────────────
# 参数方程：彗星位置
# ─────────────────────────────────────────────────────────────────────────────

def comet_pos(planet_id: int, comets: list, T: float) -> tuple[float, float] | None:
    """彗星在 T 回合后的位置，支持浮点 T（线性插值）。

    整数 T 直接查表，浮点 T 在相邻路径点间线性插值，与引擎物理一致。
    """
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        pi = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        fi = pi + T
        i0 = int(fi)
        frac = fi - i0
        if i0 < 0 or i0 >= len(path):
            return None
        x0, y0 = path[i0][0], path[i0][1]
        if frac == 0.0 or i0 + 1 >= len(path):
            return x0, y0
        x1, y1 = path[i0 + 1][0], path[i0 + 1][1]
        return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
    return None


def comet_remaining_life(planet_id: int, comets: list) -> int:
    """彗星剩余路径长度（回合数）。"""
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        pi = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - pi)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 内部：按目标类型分发位置预测
# ─────────────────────────────────────────────────────────────────────────────

def _target_pos(
    target,
    initial_by_id: dict,
    ang_vel: float,
    comets: list,
    comet_ids: set,
    T: float,
) -> tuple[float, float] | None:
    if target.id in comet_ids:
        return comet_pos(target.id, comets, T)
    return planet_pos(target, initial_by_id, ang_vel, T)


# ─────────────────────────────────────────────────────────────────────────────
# 瞄准函数与主求解器
# ─────────────────────────────────────────────────────────────────────────────

def _aim_f(
    src,
    target,
    initial_by_id: dict,
    ang_vel: float,
    comets: list,
    comet_ids: set,
    speed: float,
    T: float,
) -> float:
    """瞄准函数 f(T)：正值=舰队未到，负值=已命中。

    f(T) = dist(src, P(T)) − (src.radius + CLEARANCE + T * speed) − target.radius
    """
    P = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, T)
    if P is None:
        return math.inf
    L = math.hypot(P[0] - src.x, P[1] - src.y)
    D = src.radius + LAUNCH_CLEARANCE + T * speed
    return L - D - target.radius


def aim_with_prediction(
    src,
    target,
    ships: int,
    initial_by_id: dict,
    ang_vel: float,
    comets: list,
    comet_ids: set,
    planets: list | None = None,
) -> tuple[float, int, float, float] | None:
    """粗扫描 + 二分法求发射角度与命中回合。

    步骤：
      1. 在 (0, ROUTE_SEARCH_HORIZON] 均匀采样 _N_COARSE 点，
         检测 f(T) 首个正→负符号变化。
      2. 二分精化 _N_BISECT 次，定位连续时刻 T*。
      3. 整数精挑：对 floor(T*)−1、floor(T*)、floor(T*)+1 验证
         引擎同款的线段–圆碰撞 + 星球扫过碰撞。若传入 planets，
         还会验证目标是不是第一碰撞对象，避免中途先撞到别的星球。

    返回 (theta, T, hit_x, hit_y) 或 None（无法在视野内命中）。
    """
    speed = fleet_speed(max(1, int(ships)))
    max_turns = ROUTE_SEARCH_HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))
    if max_turns < 1:
        return None

    def f(T: float) -> float:
        return _aim_f(src, target, initial_by_id, ang_vel, comets, comet_ids, speed, T)

    # ── 1. 粗扫描 ────────────────────────────────────────────────────────────
    dt = max_turns / _N_COARSE
    prev = f(1e-6)
    t_lo = t_hi = None
    for i in range(1, _N_COARSE + 1):
        t = i * dt
        cur = f(t)
        if prev > 0 and cur <= 0:
            t_lo = (i - 1) * dt
            t_hi = t
            break
        prev = cur

    if t_lo is None:
        return None

    # ── 2. 二分精化 ──────────────────────────────────────────────────────────
    for _ in range(_N_BISECT):
        mid = (t_lo + t_hi) * 0.5
        if f(t_lo) * f(mid) <= 0:
            t_hi = mid
        else:
            t_lo = mid

    t_star = (t_lo + t_hi) * 0.5

    # ── 3. 整数精挑 ──────────────────────────────────────────────────────────
    T_floor = int(t_star)
    candidates = sorted({max(1, T_floor - 1), max(1, T_floor), T_floor + 1})
    candidates = [T for T in candidates if 1 <= T <= max_turns]

    for T in candidates:
        # 引擎：用 T-1 回合的星球位置作为舰队本步的朝向目标
        P = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, T - 1)
        if P is None:
            continue
        px, py = P

        theta = math.atan2(py - src.y, px - src.x)
        cos_a = math.cos(theta)
        sin_a = math.sin(theta)

        # 舰队实际出发点（与引擎 process_moves 一致）
        fx0 = src.x + cos_a * (src.radius + LAUNCH_CLEARANCE)
        fy0 = src.y + sin_a * (src.radius + LAUNCH_CLEARANCE)

        fx_T = fx0 + T * speed * cos_a
        fy_T = fy0 + T * speed * sin_a

        if segment_hits_sun(fx0, fy0, fx_T, fy_T):
            continue

        if planets is not None and not target_is_first_collision(
            src=src,
            target=target,
            planets=planets,
            initial_by_id=initial_by_id,
            ang_vel=ang_vel,
            comets=comets,
            comet_ids=comet_ids,
            fx0=fx0,
            fy0=fy0,
            speed=speed,
            cos_a=cos_a,
            sin_a=sin_a,
            turns=T,
        ):
            continue

        fx_prev = fx0 + (T - 1) * speed * cos_a
        fy_prev = fy0 + (T - 1) * speed * sin_a

        # 舰队移动碰撞检测
        if point_to_segment_distance(px, py, fx_prev, fy_prev, fx_T, fy_T) < target.radius:
            return theta, T, px, py

        # 星球扫过碰撞检测（旋转星球主动扫过舰队）
        P_next = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, T)
        if P_next is not None and P_next != (px, py):
            pnx, pny = P_next
            if point_to_segment_distance(fx_T, fy_T, px, py, pnx, pny) < target.radius:
                return theta, T, pnx, pny

    return None


def target_is_first_collision(
    *,
    src,
    target,
    planets: list,
    initial_by_id: dict,
    ang_vel: float,
    comets: list,
    comet_ids: set,
    fx0: float,
    fy0: float,
    speed: float,
    cos_a: float,
    sin_a: float,
    turns: int,
) -> bool:
    """验证这条发射轨迹在 turns 前不会先撞到其它星球。

    Orbit Wars 引擎每回合先移动舰队，并对所有星球做连续碰撞检测；
    然后移动旋转星球/彗星，并检查星球扫过舰队。这里按同样阶段模拟：
    - 1..turns-1 回合如果撞到任何星球，都说明目标不是第一碰撞对象。
    - turns 回合必须撞到 target，且不能先撞到其它星球。
    - 星球扫过阶段也做保守检查：若非目标星球扫过舰队，则判为不可用。
    """

    for step in range(1, turns + 1):
        fx_prev = fx0 + (step - 1) * speed * cos_a
        fy_prev = fy0 + (step - 1) * speed * sin_a
        fx_cur = fx0 + step * speed * cos_a
        fy_cur = fy0 + step * speed * sin_a

        # 引擎在舰队每步移动后会先检查是否出界；出界舰队会直接消失。
        if not in_bounds(fx_cur, fy_cur):
            return False

        # 舰队移动阶段：星球位置使用本回合开始时的位置，即 T=step-1。
        moving_hit_target = False
        for planet in planets:
            pos = _target_pos(planet, initial_by_id, ang_vel, comets, comet_ids, step - 1)
            if pos is None:
                continue
            px, py = pos
            if point_to_segment_distance(px, py, fx_prev, fy_prev, fx_cur, fy_cur) < planet.radius:
                if planet.id == target.id and step == turns:
                    moving_hit_target = True
                    break
                return False
        if moving_hit_target:
            return True

        # 星球扫过阶段：移动星球从 step-1 位置扫到 step 位置。
        sweep_hit_target = False
        for planet in planets:
            old_pos = _target_pos(planet, initial_by_id, ang_vel, comets, comet_ids, step - 1)
            new_pos = _target_pos(planet, initial_by_id, ang_vel, comets, comet_ids, step)
            if old_pos is None or new_pos is None or old_pos == new_pos:
                continue
            if point_to_segment_distance(fx_cur, fy_cur, old_pos[0], old_pos[1], new_pos[0], new_pos[1]) < planet.radius:
                if planet.id == target.id and step == turns:
                    sweep_hit_target = True
                    break
                return False
        if sweep_hit_target:
            return True

    return False
