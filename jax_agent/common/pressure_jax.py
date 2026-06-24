"""Producer 风格的压力梯度辅助特征。

用于 fight feature 中的调兵/前线判断：估算敌方驻军对每颗星球的距离衰减压力，
以及 source-target 之间的压力差。
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .fast_env import EnvState, fleet_speed
from .garrison_jax import DEFAULT_GARRISON_HORIZON


class PressureStatus(NamedTuple):
    """每颗星球的敌方压力，以及来源到目标的压力差。"""

    enemy_pressure: jnp.ndarray
    pressure_delta: jnp.ndarray


def enemy_pressure_jax(
    state: EnvState,
    horizon: float = float(DEFAULT_GARRISON_HORIZON),
    player_id: int = 0,
) -> jnp.ndarray:
    """计算每颗星球受到的距离衰减敌方驻军压力。

    这是 producer 的轻量压力近似：每颗敌方星球把当前船数按距离衰减贡献给
    ``horizon`` 回合内可达的存活星球。这里刻意忽略生产、轨道未来漂移和在途舰队。
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


def pressure_by_owner_jax(
    state: EnvState,
    horizon: float = float(DEFAULT_GARRISON_HORIZON),
) -> jnp.ndarray:
    """按来源 owner 分组计算距离衰减驻军压力。

    单局返回 ``[2, P]``，批量返回 ``[B, 2, P]``。当同时需要两边视角时很有用：
    player0 的敌方压力是 owner1 压力，player1 的敌方压力是 owner0 压力。
    """

    if state.planets.ndim == 3:
        return jax.vmap(lambda s: pressure_by_owner_jax(s, horizon))(state)

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

    not_self = ~jnp.eye(planets.shape[0], dtype=bool)
    valid = alive[:, None] & alive[None, :] & not_self & (owner[:, None] >= 0)
    contrib = jnp.where(valid, ships[:, None] * decay, 0.0)
    owner_oh = jax.nn.one_hot(jnp.clip(owner, 0, 1), 2, dtype=jnp.float32)
    owner_oh = owner_oh * ((owner >= 0) & alive).astype(jnp.float32)[:, None]
    return jnp.einsum("pa,pt->at", owner_oh, contrib)


def pressure_delta_jax(enemy_pressure: jnp.ndarray) -> jnp.ndarray:
    """来源-目标压力梯度：``pressure[target] - pressure[source]``。"""

    if enemy_pressure.ndim == 2:
        return enemy_pressure[:, None, :] - enemy_pressure[:, :, None]
    return enemy_pressure[None, :] - enemy_pressure[:, None]


def pressure_status_jax(
    state: EnvState,
    horizon: float = float(DEFAULT_GARRISON_HORIZON),
    player_id: int = 0,
) -> PressureStatus:
    """同时返回每颗星球压力和 source-target 压力差。"""

    pressure = enemy_pressure_jax(state, horizon=horizon, player_id=player_id)
    return PressureStatus(
        enemy_pressure=pressure,
        pressure_delta=pressure_delta_jax(pressure),
    )
