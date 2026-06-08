from __future__ import annotations

import math
from dataclasses import dataclass


CENTER_X = 50.0
CENTER_Y = 50.0
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1
MAX_SPEED = 6.0
ROUTE_SEARCH_HORIZON = 60

_N_COARSE = 60
_N_BISECT = 12


@dataclass(frozen=True)
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: float
    production: float


def fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def segment_hits_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_RADIUS


def in_bounds(x: float, y: float) -> bool:
    return 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE


def planet_pos(planet: Planet, initial_by_id: dict[int, Planet], ang_vel: float, turns: float) -> tuple[float, float]:
    initial = initial_by_id.get(planet.id)
    if initial is None:
        return planet.x, planet.y
    dx = initial.x - CENTER_X
    dy = initial.y - CENTER_Y
    orbit_r = math.hypot(dx, dy)
    if orbit_r + initial.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    phi_cur = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    phi = phi_cur + ang_vel * turns
    return CENTER_X + orbit_r * math.cos(phi), CENTER_Y + orbit_r * math.sin(phi)


def comet_pos(planet_id: int, comets: list[dict], turns: float) -> tuple[float, float] | None:
    for group in comets:
        planet_ids = group.get("planet_ids", [])
        if planet_id not in planet_ids:
            continue
        idx = planet_ids.index(planet_id)
        paths = group.get("paths", [])
        path_index = int(group.get("path_index", 0))
        if idx >= len(paths):
            return None
        path = paths[idx]
        fi = path_index + turns
        i0 = int(fi)
        frac = fi - i0
        if i0 < 0 or i0 >= len(path):
            return None
        x0, y0 = path[i0][0], path[i0][1]
        if frac == 0.0 or i0 + 1 >= len(path):
            return float(x0), float(y0)
        x1, y1 = path[i0 + 1][0], path[i0 + 1][1]
        return float(x0 + frac * (x1 - x0)), float(y0 + frac * (y1 - y0))
    return None


def comet_remaining_life(planet_id: int, comets: list[dict]) -> int:
    for group in comets:
        planet_ids = group.get("planet_ids", [])
        if planet_id not in planet_ids:
            continue
        idx = planet_ids.index(planet_id)
        paths = group.get("paths", [])
        path_index = int(group.get("path_index", 0))
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def _target_pos(
    target: Planet,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict],
    comet_ids: set[int],
    turns: float,
) -> tuple[float, float] | None:
    if target.id in comet_ids:
        return comet_pos(target.id, comets, turns)
    return planet_pos(target, initial_by_id, ang_vel, turns)


def _aim_f(
    src: Planet,
    target: Planet,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict],
    comet_ids: set[int],
    speed: float,
    turns: float,
) -> float:
    pos = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, turns)
    if pos is None:
        return math.inf
    length = math.hypot(pos[0] - src.x, pos[1] - src.y)
    travel = src.radius + LAUNCH_CLEARANCE + turns * speed
    return length - travel - target.radius


def aim_with_prediction(
    src: Planet,
    target: Planet,
    ships: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict],
    comet_ids: set[int],
    planets: list[Planet] | None = None,
) -> tuple[float, int, float, float] | None:
    speed = fleet_speed(max(1, int(ships)))
    max_turns = ROUTE_SEARCH_HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))
    if max_turns < 1:
        return None

    def f(turns: float) -> float:
        return _aim_f(src, target, initial_by_id, ang_vel, comets, comet_ids, speed, turns)

    dt = max_turns / _N_COARSE
    prev = f(1e-6)
    t_lo = t_hi = None
    for i in range(1, _N_COARSE + 1):
        turns = i * dt
        cur = f(turns)
        if prev > 0 and cur <= 0:
            t_lo = (i - 1) * dt
            t_hi = turns
            break
        prev = cur
    if t_lo is None:
        return None

    for _ in range(_N_BISECT):
        mid = (t_lo + t_hi) * 0.5
        if f(t_lo) * f(mid) <= 0:
            t_hi = mid
        else:
            t_lo = mid

    t_star = (t_lo + t_hi) * 0.5
    t_floor = int(t_star)
    candidates = sorted({max(1, t_floor - 1), max(1, t_floor), t_floor + 1})
    candidates = [turns for turns in candidates if 1 <= turns <= max_turns]

    for turns in candidates:
        pos = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, turns - 1)
        if pos is None:
            continue
        px, py = pos
        theta = math.atan2(py - src.y, px - src.x)
        cos_a = math.cos(theta)
        sin_a = math.sin(theta)

        fx0 = src.x + cos_a * (src.radius + LAUNCH_CLEARANCE)
        fy0 = src.y + sin_a * (src.radius + LAUNCH_CLEARANCE)
        fx_t = fx0 + turns * speed * cos_a
        fy_t = fy0 + turns * speed * sin_a

        if segment_hits_sun(fx0, fy0, fx_t, fy_t):
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
            turns=turns,
        ):
            continue

        fx_prev = fx0 + (turns - 1) * speed * cos_a
        fy_prev = fy0 + (turns - 1) * speed * sin_a
        if point_to_segment_distance(px, py, fx_prev, fy_prev, fx_t, fy_t) < target.radius:
            return theta, turns, px, py

        next_pos = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, turns)
        if next_pos is not None and next_pos != (px, py):
            pnx, pny = next_pos
            if point_to_segment_distance(fx_t, fy_t, px, py, pnx, pny) < target.radius:
                return theta, turns, pnx, pny

    return None


def target_is_first_collision(
    *,
    src: Planet,
    target: Planet,
    planets: list[Planet],
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict],
    comet_ids: set[int],
    fx0: float,
    fy0: float,
    speed: float,
    cos_a: float,
    sin_a: float,
    turns: int,
) -> bool:
    del src
    for step in range(1, turns + 1):
        fx_prev = fx0 + (step - 1) * speed * cos_a
        fy_prev = fy0 + (step - 1) * speed * sin_a
        fx_cur = fx0 + step * speed * cos_a
        fy_cur = fy0 + step * speed * sin_a

        if not in_bounds(fx_cur, fy_cur):
            return False

        for planet in planets:
            pos = _target_pos(planet, initial_by_id, ang_vel, comets, comet_ids, step - 1)
            if pos is None:
                continue
            px, py = pos
            if point_to_segment_distance(px, py, fx_prev, fy_prev, fx_cur, fy_cur) < planet.radius:
                return planet.id == target.id and step == turns

        for planet in planets:
            old_pos = _target_pos(planet, initial_by_id, ang_vel, comets, comet_ids, step - 1)
            new_pos = _target_pos(planet, initial_by_id, ang_vel, comets, comet_ids, step)
            if old_pos is None or new_pos is None or old_pos == new_pos:
                continue
            if point_to_segment_distance(fx_cur, fy_cur, old_pos[0], old_pos[1], new_pos[0], new_pos[1]) < planet.radius:
                return planet.id == target.id and step == turns

    return False
