"""Kaggle observation 的轻量解析层。

原始 observation 是 dict/对象混合结构，训练代码直接用会很散。这里把
planets/fleets 解析成 dataclass，方便特征工程和 reward 计算读取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Planet:
    """星球状态，字段顺序对应环境里的 planet list。"""

    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(slots=True)
class Fleet:
    """飞行中舰队状态，字段顺序对应环境里的 fleet list。"""

    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(slots=True)
class GameState:
    """解析后的 observation。

    angular_velocity / initial_planets / comet_planet_ids / comets 均来自
    Kaggle 环境的原始 observation，用于 physics 模块的轨迹预测。
    """

    step: int
    player: int
    planets: list[Planet]
    fleets: list[Fleet]
    # 旋转星球的角速度（rad/step），每局固定，由引擎在 reset 时随机设定。
    angular_velocity: float = 0.0
    # 各星球的初始位置（用于旋转参数方程的基准角）。
    initial_planets: list[Planet] = field(default_factory=list)
    # 当前存活的彗星对应的 planet_id 列表。
    comet_planet_ids: list[int] = field(default_factory=list)
    # 彗星路径列表（原始 dict，含 planet_ids / paths / path_index）。
    comets: list[Any] = field(default_factory=list)


def read(obj: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 和对象属性读取。"""

    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_observation(observation: Any) -> GameState:
    """把 Kaggle 原始 observation 转成强类型 GameState。"""

    planets = [
        Planet(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )
        for row in read(observation, "planets", []) or []
    ]
    fleets = [
        Fleet(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            angle=float(row[4]),
            from_planet_id=int(row[5]),
            ships=int(row[6]),
        )
        for row in read(observation, "fleets", []) or []
    ]
    return GameState(
        step=int(read(observation, "step", 0) or 0),
        player=int(read(observation, "player", 0) or 0),
        planets=planets,
        fleets=fleets,
        angular_velocity=float(read(observation, "angular_velocity", 0.0) or 0.0),
        initial_planets=[
            Planet(
                id=int(row[0]),
                owner=int(row[1]),
                x=float(row[2]),
                y=float(row[3]),
                radius=float(row[4]),
                ships=int(row[5]),
                production=int(row[6]),
            )
            for row in read(observation, "initial_planets", []) or []
        ],
        comet_planet_ids=[int(pid) for pid in (read(observation, "comet_planet_ids", []) or [])],
        comets=list(read(observation, "comets", []) or []),
    )
