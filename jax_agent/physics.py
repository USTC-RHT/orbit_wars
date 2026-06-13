"""JAX aiming and engine-order physical hit validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .fast_env import (
    COMETS_PER_SPAWN,
    MAX_COMET_PATH,
    MAX_COMET_SPAWNS,
    EnvState,
    comet_planet_mask,
    point_segment_distance_sq,
)


AIM_HORIZON = 60
AIM_NEWTON_ITERS = 5
AIM_NEWTON_MIN_DERIV = 1e-4
AIM_NEWTON_MAX_STEP = 12.0
AIM_TARGET_INSET = 1e-3


def aim_with_prediction_jax(src, target, target_idx, state: EnvState, ships):
    """Aim at moving targets and verify the shot using engine-order collisions."""
    angle, hit, _eta_turns = aim_with_prediction_eta_jax(src, target, target_idx, state, ships)
    return angle, hit


def aim_with_prediction_eta_jax(src, target, target_idx, state: EnvState, ships):
    """Aim at moving targets and return the exact engine hit tick.

    The routine solves a continuous intercept time with Newton iterations, then
    validates nearby integer engine ticks against sun, bounds, first planet hit,
    and moving-planet sweep collisions. ``eta_turns`` is the selected integer
    engine tick for a valid shot, otherwise ``AIM_HORIZON + 1``.
    """

    horizon = AIM_HORIZON
    clearance = 0.1
    speed = fleet_speed_local(jnp.maximum(ships, 1.0))
    max_turns = target_max_turns_jax(state, target_idx, horizon).astype(jnp.float32)
    can_search = max_turns >= 1.0

    px0, py0, target0_valid = target_pos_at_time_jax(state, target_idx, jnp.zeros_like(max_turns))
    gap = src[..., 4] + clearance + target[..., 4]
    d0 = jnp.sqrt((px0 - src[..., 2]) ** 2 + (py0 - src[..., 3]) ** 2)
    t_star = jnp.clip((d0 - gap) / speed, 0.0, max_turns)

    def newton_body(_, t_cur):
        px, py, vx, vy, pos_valid = target_pos_velocity_at_time_jax(state, target_idx, t_cur)
        rx = px - src[..., 2]
        ry = py - src[..., 3]
        dist = jnp.sqrt(jnp.maximum(rx * rx + ry * ry, 1.0e-9))
        f_val = dist - gap - speed * t_cur
        radial_velocity = (rx * vx + ry * vy) / dist
        deriv = radial_velocity - speed
        raw_step = f_val / deriv
        max_step = jnp.minimum(jnp.maximum(max_turns, 1.0), AIM_NEWTON_MAX_STEP)
        step = jnp.clip(raw_step, -max_step, max_step)
        t_newton = jnp.clip(t_cur - step, 0.0, max_turns)
        use_newton = (
            pos_valid
            & jnp.isfinite(t_newton)
            & jnp.isfinite(raw_step)
            & jnp.isfinite(deriv)
            & (jnp.abs(deriv) > AIM_NEWTON_MIN_DERIV)
        )
        return jnp.where(use_newton, t_newton, t_cur)

    t_star = jax.lax.fori_loop(0, AIM_NEWTON_ITERS, newton_body, t_star)
    has_solution = can_search & target0_valid
    t_floor = jnp.floor(t_star).astype(jnp.int32)
    candidates = jnp.stack(
        [
            jnp.maximum(t_floor - 1, 1),
            jnp.maximum(t_floor, 1),
            t_floor + 1,
        ],
        axis=-1,
    )
    candidate_turns = candidates.astype(jnp.float32)
    candidate_in_range = (candidate_turns >= 1.0) & (candidate_turns <= max_turns[..., None])

    t_star_for_candidates = jnp.broadcast_to(t_star[..., None], candidate_turns.shape)
    aim_turns = jnp.stack(
        [
            t_star_for_candidates,
            jnp.maximum(candidate_turns - 1.0, 0.0),
            candidate_turns,
        ],
        axis=-1,
    )
    aim_px, aim_py, aim_pos_valid = target_pos_at_time_jax(state, target_idx, aim_turns)
    angle = jnp.arctan2(aim_py - src[..., None, None, 3], aim_px - src[..., None, None, 2])
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    candidate_turns_e = candidate_turns[..., None]
    candidates_e = candidates[..., None]
    fx0 = src[..., None, None, 2] + cos_a * (src[..., None, None, 4] + clearance)
    fy0 = src[..., None, None, 3] + sin_a * (src[..., None, None, 4] + clearance)
    fx_t = fx0 + candidate_turns_e * speed[..., None, None] * cos_a
    fy_t = fy0 + candidate_turns_e * speed[..., None, None] * sin_a

    sun_hit = point_segment_distance_sq(50.0, 50.0, fx0, fy0, fx_t, fy_t) < 100.0
    first_collision_ok = target_is_first_collision_jax(
        src,
        target_idx,
        state,
        fx0,
        fy0,
        speed[..., None, None],
        cos_a,
        sin_a,
        candidates_e,
    )

    fx_prev = fx0 + (candidate_turns_e - 1.0) * speed[..., None, None] * cos_a
    fy_prev = fy0 + (candidate_turns_e - 1.0) * speed[..., None, None] * sin_a
    px_old, py_old, target_pos_valid = target_pos_at_time_jax(state, target_idx, candidate_turns_e - 1.0)
    move_d2 = point_segment_distance_sq(px_old, py_old, fx_prev, fy_prev, fx_t, fy_t)
    target_hit_radius = jnp.maximum(target[..., None, None, 4] - AIM_TARGET_INSET, 0.0)
    move_hits_target = move_d2 < target_hit_radius ** 2

    px_next, py_next, target_next_valid = target_pos_at_time_jax(state, target_idx, candidate_turns_e)
    target_moved = ((px_next - px_old) ** 2 + (py_next - py_old) ** 2) > 1e-9
    sweep_d2 = point_segment_distance_sq(fx_t, fy_t, px_old, py_old, px_next, py_next)
    sweep_hits_target = target_moved & target_next_valid & (sweep_d2 < target_hit_radius ** 2)

    candidate_valid = (
        has_solution[..., None, None]
        & candidate_in_range[..., None]
        & aim_pos_valid
        & target_pos_valid
        & (~sun_hit)
        & first_collision_ok
        & (move_hits_target | sweep_hits_target)
        & (ships[..., None, None] > 0.0)
    )
    candidate_valid_flat = candidate_valid.reshape(candidate_valid.shape[:-2] + (9,))
    angle_flat = angle.reshape(angle.shape[:-2] + (9,))
    eta_flat = jnp.broadcast_to(candidate_turns_e, angle.shape).reshape(angle.shape[:-2] + (9,))
    has_valid = candidate_valid_flat.any(axis=-1)
    chosen = jnp.argmax(candidate_valid_flat, axis=-1)
    chosen_angle = jnp.take_along_axis(angle_flat, chosen[..., None], axis=-1).squeeze(-1)
    chosen_eta = jnp.take_along_axis(eta_flat, chosen[..., None], axis=-1).squeeze(-1)
    direct_angle = jnp.arctan2(target[..., 3] - src[..., 3], target[..., 2] - src[..., 2])
    eta_turns = jnp.where(has_valid, chosen_eta, jnp.asarray(AIM_HORIZON + 1, dtype=jnp.float32))
    return jnp.where(has_valid, chosen_angle, direct_angle), has_valid, eta_turns


def target_pos_at_time_jax(state: EnvState, target_idx, turns):
    """Target planet position after floating-point ``turns``."""
    b, _p = target_idx.shape
    batch_idx = jnp.arange(b)[:, None]
    planets = state.planets[batch_idx, target_idx]
    initial = state.initial_planets[batch_idx, target_idx]
    alive = state.planet_alive[batch_idx, target_idx]

    base_x = _expand_like_time(planets[..., 2], turns)
    base_y = _expand_like_time(planets[..., 3], turns)
    dx = initial[..., 2] - 50.0
    dy = initial[..., 3] - 50.0
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    phi_cur = jnp.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    phi = _expand_like_time(phi_cur, turns) + _expand_like_time(state.angular_velocity[:, None], turns) * turns
    rot_x = 50.0 + _expand_like_time(orbit_r, turns) * jnp.cos(phi)
    rot_y = 50.0 + _expand_like_time(orbit_r, turns) * jnp.sin(phi)
    comet_mask = jax.vmap(comet_planet_mask)(state)
    is_comet = comet_mask[batch_idx, target_idx]
    rotating = (orbit_r + initial[..., 4] < 50.0) & alive & (~is_comet)
    px = jnp.where(_expand_like_time(rotating, turns), rot_x, base_x)
    py = jnp.where(_expand_like_time(rotating, turns), rot_y, base_y)
    valid = _expand_like_time(alive, turns)

    comet_x, comet_y, comet_valid = comet_target_pos_at_time_jax(state, target_idx, turns)
    is_comet_t = _expand_like_time(is_comet, turns)
    px = jnp.where(is_comet_t, comet_x, px)
    py = jnp.where(is_comet_t, comet_y, py)
    valid = jnp.where(is_comet_t, comet_valid, valid)
    return px, py, valid


def target_pos_velocity_at_time_jax(state: EnvState, target_idx, turns):
    """Target planet position and velocity after floating-point ``turns``."""
    b, _p = target_idx.shape
    batch_idx = jnp.arange(b)[:, None]
    planets = state.planets[batch_idx, target_idx]
    initial = state.initial_planets[batch_idx, target_idx]
    alive = state.planet_alive[batch_idx, target_idx]

    base_x = _expand_like_time(planets[..., 2], turns)
    base_y = _expand_like_time(planets[..., 3], turns)
    zero = jnp.zeros_like(base_x)
    dx = initial[..., 2] - 50.0
    dy = initial[..., 3] - 50.0
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    phi_cur = jnp.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    omega = _expand_like_time(state.angular_velocity[:, None], turns)
    phi = _expand_like_time(phi_cur, turns) + omega * turns
    orbit_r_t = _expand_like_time(orbit_r, turns)
    rot_x = 50.0 + orbit_r_t * jnp.cos(phi)
    rot_y = 50.0 + orbit_r_t * jnp.sin(phi)
    rot_vx = -omega * orbit_r_t * jnp.sin(phi)
    rot_vy = omega * orbit_r_t * jnp.cos(phi)

    comet_mask = jax.vmap(comet_planet_mask)(state)
    is_comet = comet_mask[batch_idx, target_idx]
    rotating = (orbit_r + initial[..., 4] < 50.0) & alive & (~is_comet)
    rotating_t = _expand_like_time(rotating, turns)
    px = jnp.where(rotating_t, rot_x, base_x)
    py = jnp.where(rotating_t, rot_y, base_y)
    vx = jnp.where(rotating_t, rot_vx, zero)
    vy = jnp.where(rotating_t, rot_vy, zero)
    valid = _expand_like_time(alive, turns)

    comet_x, comet_y, comet_vx, comet_vy, comet_valid = comet_target_pos_velocity_at_time_jax(state, target_idx, turns)
    is_comet_t = _expand_like_time(is_comet, turns)
    px = jnp.where(is_comet_t, comet_x, px)
    py = jnp.where(is_comet_t, comet_y, py)
    vx = jnp.where(is_comet_t, comet_vx, vx)
    vy = jnp.where(is_comet_t, comet_vy, vy)
    valid = jnp.where(is_comet_t, comet_valid, valid)
    return px, py, vx, vy, valid


def comet_target_pos_at_time_jax(state: EnvState, target_idx, turns):
    """Comet target position after floating-point ``turns``."""
    extra = turns.ndim - target_idx.ndim
    one_shape = (1,) * extra
    slots = state.comet_slots.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    target_e = target_idx.reshape(target_idx.shape + one_shape + (1, 1))
    active = (
        (state.comet_slots >= 0)
        & (state.comet_path_index[:, :, None] >= -1)
        & (state.comet_lengths > 0)
    ).reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    match = (slots == target_e) & active

    path_index = state.comet_path_index.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, 1))
    lengths = state.comet_lengths.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    fi = path_index + turns[..., None, None]
    i0 = jnp.floor(fi).astype(jnp.int32)
    frac = fi - i0.astype(jnp.float32)
    valid_i0 = (i0 >= 0) & (i0 < lengths)
    i1 = jnp.minimum(i0 + 1, lengths - 1)
    has_i1 = i0 + 1 < lengths
    i0_safe = jnp.clip(i0, 0, MAX_COMET_PATH - 1)
    i1_safe = jnp.clip(i1, 0, MAX_COMET_PATH - 1)
    path_shape = (state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2)
    paths = state.comet_paths.reshape(path_shape)
    one0 = jax.nn.one_hot(i0_safe, MAX_COMET_PATH, dtype=jnp.float32)
    one1 = jax.nn.one_hot(i1_safe, MAX_COMET_PATH, dtype=jnp.float32)
    pos0 = (one0[..., None] * paths).sum(axis=-2)
    pos1 = (one1[..., None] * paths).sum(axis=-2)
    pos = jnp.where(has_i1[..., None], pos0 + frac[..., None] * (pos1 - pos0), pos0)
    weights = (match & valid_i0).astype(jnp.float32)
    has = weights.sum(axis=(-2, -1)) > 0.0
    px = (pos[..., 0] * weights).sum(axis=(-2, -1))
    py = (pos[..., 1] * weights).sum(axis=(-2, -1))
    return px, py, has


def comet_target_pos_velocity_at_time_jax(state: EnvState, target_idx, turns):
    """Comet target position and segment velocity after floating-point ``turns``."""
    extra = turns.ndim - target_idx.ndim
    one_shape = (1,) * extra
    slots = state.comet_slots.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    target_e = target_idx.reshape(target_idx.shape + one_shape + (1, 1))
    active = (
        (state.comet_slots >= 0)
        & (state.comet_path_index[:, :, None] >= -1)
        & (state.comet_lengths > 0)
    ).reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    match = (slots == target_e) & active

    path_index = state.comet_path_index.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, 1))
    lengths = state.comet_lengths.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    fi = path_index + turns[..., None, None]
    i0 = jnp.floor(fi).astype(jnp.int32)
    frac = fi - i0.astype(jnp.float32)
    valid_i0 = (i0 >= 0) & (i0 < lengths)
    i1 = jnp.minimum(i0 + 1, lengths - 1)
    has_i1 = i0 + 1 < lengths
    i0_safe = jnp.clip(i0, 0, MAX_COMET_PATH - 1)
    i1_safe = jnp.clip(i1, 0, MAX_COMET_PATH - 1)
    path_shape = (state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2)
    paths = state.comet_paths.reshape(path_shape)
    one0 = jax.nn.one_hot(i0_safe, MAX_COMET_PATH, dtype=jnp.float32)
    one1 = jax.nn.one_hot(i1_safe, MAX_COMET_PATH, dtype=jnp.float32)
    pos0 = (one0[..., None] * paths).sum(axis=-2)
    pos1 = (one1[..., None] * paths).sum(axis=-2)
    pos = jnp.where(has_i1[..., None], pos0 + frac[..., None] * (pos1 - pos0), pos0)
    vel = jnp.where(has_i1[..., None], pos1 - pos0, jnp.zeros_like(pos0))
    weights = (match & valid_i0).astype(jnp.float32)
    has = weights.sum(axis=(-2, -1)) > 0.0
    px = (pos[..., 0] * weights).sum(axis=(-2, -1))
    py = (pos[..., 1] * weights).sum(axis=(-2, -1))
    vx = (vel[..., 0] * weights).sum(axis=(-2, -1))
    vy = (vel[..., 1] * weights).sum(axis=(-2, -1))
    return px, py, vx, vy, has


def target_max_turns_jax(state: EnvState, target_idx, horizon: int):
    """Normal targets search ``horizon``; comet targets stop before expiry."""
    slots = state.comet_slots[:, None, :, :]
    target_e = target_idx[:, :, None, None]
    active = (state.comet_slots[:, None, :, :] >= 0) & (state.comet_path_index[:, None, :, None] >= -1)
    match = (slots == target_e) & active
    remaining = state.comet_lengths[:, None, :, :] - state.comet_path_index[:, None, :, None]
    comet_remaining = jnp.where(match, remaining, 0).max(axis=(-2, -1))
    comet_has = match.any(axis=(-2, -1))
    comet_max = jnp.minimum(jnp.asarray(horizon, dtype=jnp.int32), jnp.maximum(0, comet_remaining - 1))
    return jnp.where(comet_has, comet_max, horizon)


def target_is_first_collision_jax(src, target_idx, state: EnvState, fx0, fy0, speed, cos_a, sin_a, hit_turn):
    """Verify that the target is the first engine collision at ``hit_turn``."""
    horizon = AIM_HORIZON
    turns_i = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    turns = turns_i.astype(jnp.float32)
    planets = state.planets
    planet_alive = state.planet_alive
    px_t, py_t = all_planet_pos_at_turns(state, horizon)
    fx_prev = fx0[..., None] + (turns - 1.0) * speed[..., None] * cos_a[..., None]
    fy_prev = fy0[..., None] + (turns - 1.0) * speed[..., None] * sin_a[..., None]
    fx_cur = fx0[..., None] + turns * speed[..., None] * cos_a[..., None]
    fy_cur = fy0[..., None] + turns * speed[..., None] * sin_a[..., None]
    active_step = turns_i.reshape((1,) * hit_turn.ndim + (horizon,)) <= hit_turn[..., None]
    in_bounds = (fx_cur >= 0.0) & (fx_cur <= 100.0) & (fy_cur >= 0.0) & (fy_cur <= 100.0)
    in_bounds_until_hit = (in_bounds | (~active_step)).all(axis=-1)

    src_axes = (None,) * (hit_turn.ndim - 1)
    px_old = px_t[(slice(None),) + src_axes + (slice(None), slice(None, -1))]
    py_old = py_t[(slice(None),) + src_axes + (slice(None), slice(None, -1))]
    px_new = px_t[(slice(None),) + src_axes + (slice(None), slice(1, None))]
    py_new = py_t[(slice(None),) + src_axes + (slice(None), slice(1, None))]
    radii = planets[(slice(None),) + src_axes + (slice(None), 4)]
    alive = planet_alive[(slice(None),) + src_axes + (slice(None), None)]

    move_d2 = point_segment_distance_sq(
        px_old,
        py_old,
        fx_prev[..., None, :],
        fy_prev[..., None, :],
        fx_cur[..., None, :],
        fy_cur[..., None, :],
    )
    move_hit = (move_d2 < radii[..., None] ** 2) & alive & active_step[..., None, :]

    sweep_active = ((px_new - px_old) ** 2 + (py_new - py_old) ** 2) > 1e-9
    sweep_d2 = point_segment_distance_sq(
        fx_cur[..., None, :],
        fy_cur[..., None, :],
        px_old,
        py_old,
        px_new,
        py_new,
    )
    sweep_hit = (sweep_d2 < radii[..., None] ** 2) & alive & sweep_active & active_step[..., None, :]

    large = jnp.int32(2 * (horizon + 1) + 1)
    move_keys = (2 * turns_i).reshape((1,) * (move_hit.ndim - 1) + (horizon,))
    sweep_keys = (2 * turns_i + 1).reshape((1,) * (sweep_hit.ndim - 1) + (horizon,))
    first_move = jnp.where(move_hit, move_keys, large).min(axis=-1)
    first_sweep = jnp.where(sweep_hit, sweep_keys, large).min(axis=-1)
    first_by_planet = jnp.minimum(first_move, first_sweep)
    first_planet = first_by_planet.argmin(axis=-1)
    first_key = first_by_planet.min(axis=-1)
    target_key_lo = 2 * hit_turn
    target_key_hi = 2 * hit_turn + 1
    return (
        in_bounds_until_hit
        & (first_key < large)
        & (first_planet == target_idx.reshape(target_idx.shape + (1,) * (hit_turn.ndim - target_idx.ndim)))
        & (first_key >= target_key_lo)
        & (first_key <= target_key_hi)
    )


def _expand_like_time(x, turns):
    """Expand a scalar/target tensor to the shape rank of ``turns``."""
    return x.reshape(x.shape + (1,) * (turns.ndim - x.ndim))


def all_planet_pos_at_turns(state: EnvState, horizon: int):
    """Future planet positions for all planets, shape ``[B, P, horizon + 1]``."""
    planets = state.planets
    initial_planets = state.initial_planets
    planet_alive = state.planet_alive
    turns = jnp.arange(horizon + 1, dtype=jnp.float32)
    dx = initial_planets[..., 2] - 50.0
    dy = initial_planets[..., 3] - 50.0
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    comet_mask = jax.vmap(comet_planet_mask)(state)
    rotating = (orbit_r + initial_planets[..., 4] < 50.0) & planet_alive & (~comet_mask)
    phi_cur = jnp.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    phi = phi_cur[..., None] + state.angular_velocity[:, None, None] * turns
    px = 50.0 + orbit_r[..., None] * jnp.cos(phi)
    py = 50.0 + orbit_r[..., None] * jnp.sin(phi)
    px = jnp.where(rotating[..., None], px, planets[..., None, 2])
    py = jnp.where(rotating[..., None], py, planets[..., None, 3])

    comet_slots = state.comet_slots.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    safe_slots = jnp.where(comet_slots >= 0, comet_slots, 0)
    group_active = state.comet_path_index[:, :, None] >= -1
    active = (state.comet_slots >= 0) & group_active
    active_flat = active.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    path_index = state.comet_path_index[:, :, None, None] + jnp.arange(horizon + 1, dtype=jnp.int32)[None, None, None, :]
    path_index = jnp.broadcast_to(path_index, (state.planets.shape[0], MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1))
    valid_t = active[..., None] & (path_index >= 0) & (path_index < state.comet_lengths[..., None])
    path_index = jnp.clip(path_index, 0, MAX_COMET_PATH - 1)
    gather_index = jnp.broadcast_to(
        path_index[..., None],
        (state.planets.shape[0], MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1, 2),
    )
    path_pos = jnp.take_along_axis(state.comet_paths, gather_index, axis=3)
    path_pos = jnp.where(valid_t[..., None], path_pos, -1000.0)
    flat_pos = path_pos.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN, horizon + 1, 2)
    slot_onehot = jax.nn.one_hot(safe_slots, state.planets.shape[1], dtype=jnp.float32) * active_flat[..., None].astype(jnp.float32)
    comet_has = slot_onehot.sum(axis=1) > 0.0
    comet_px = jnp.einsum("bkp,bkt->bpt", slot_onehot, flat_pos[..., 0])
    comet_py = jnp.einsum("bkp,bkt->bpt", slot_onehot, flat_pos[..., 1])
    px = jnp.where(comet_has[..., None], comet_px, px)
    py = jnp.where(comet_has[..., None], comet_py, py)
    return px, py


def fleet_speed_local(ships):
    """Fleet speed formula matching ``orbit_wars.py``."""
    ratio = jnp.clip(jnp.log(jnp.maximum(ships, 1.0)) / jnp.log(1000.0), 0.0, 1.0)
    return jnp.minimum(1.0 + 5.0 * (ratio**1.5), 6.0)
