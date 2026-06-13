"""Producer-style pressure-gradient helpers for regroup/frontline features."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .fast_env import EnvState, fleet_speed
from .garrison_jax import DEFAULT_GARRISON_HORIZON


class PressureStatus(NamedTuple):
    """Enemy pressure per planet and source-target pressure deltas."""

    enemy_pressure: jnp.ndarray
    pressure_delta: jnp.ndarray


def enemy_pressure_jax(
    state: EnvState,
    horizon: float = float(DEFAULT_GARRISON_HORIZON),
    player_id: int = 0,
) -> jnp.ndarray:
    """Distance-decayed enemy garrison pressure per planet.

    Mirrors producer's cheap pressure proxy: every enemy planet contributes a
    distance-decayed share of its current ships to every alive planet it can reach
    within ``horizon`` turns. It ignores production, orbital future drift, and
    in-flight fleets by design.
    """

    if state.planets.ndim == 3:
        return jax.vmap(lambda s: enemy_pressure_jax(s, horizon, player_id))(state)

    planets = state.planets
    alive = state.planet_alive
    owner = planets[:, 1].astype(jnp.int32)
    ships = planets[:, 5].astype(jnp.float32)
    xy = planets[:, 2:4]

    src_xy = xy[:, None, :]
    tgt_xy = xy[None, :, :]
    dist = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(src_xy - tgt_xy), axis=-1), 0.0))

    speeds = fleet_speed(jnp.maximum(ships, 1.0))
    reach_dist = jnp.maximum(speeds[:, None] * jnp.asarray(horizon, dtype=jnp.float32), 1e-6)
    decay = jnp.clip(1.0 - dist / reach_dist, 0.0, 1.0)

    enemy_source = alive & (owner >= 0) & (owner != int(player_id))
    not_self = ~jnp.eye(planets.shape[0], dtype=bool)
    valid = enemy_source[:, None] & alive[None, :] & not_self
    contrib = jnp.where(valid, ships[:, None] * decay, 0.0)
    return contrib.sum(axis=0)


def pressure_delta_jax(enemy_pressure: jnp.ndarray) -> jnp.ndarray:
    """Source-target pressure gradient: ``pressure[target] - pressure[source]``."""

    if enemy_pressure.ndim == 2:
        return enemy_pressure[:, None, :] - enemy_pressure[:, :, None]
    return enemy_pressure[None, :] - enemy_pressure[:, None]


def pressure_status_jax(
    state: EnvState,
    horizon: float = float(DEFAULT_GARRISON_HORIZON),
    player_id: int = 0,
) -> PressureStatus:
    """Return both per-planet pressure and source-target pressure deltas."""

    pressure = enemy_pressure_jax(state, horizon=horizon, player_id=player_id)
    return PressureStatus(
        enemy_pressure=pressure,
        pressure_delta=pressure_delta_jax(pressure),
    )
