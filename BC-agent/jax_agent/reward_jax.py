"""Orbit Wars JAX 奖励塑造模块。

与 PyTorch 版 src/env_wrapper.py 的奖励组件基本对齐；终局平局按当前训练策略有意惩罚：
- terminal_reward_player0: 利用 scores() 重新计算得分（不依赖 winner 字段）。
  赢: +1，平局/输/0-0: -1。
- neutral_capture: 掌控中立星球奖励，较近且较早时加成。
- enemy_capture: 占领敌方星球奖励。
- lose_planet: 失去己方星球惩罚。
- wasted_attack: 己方舰队抵达非友方星球但未占领时的消耗惩罚。
- shaped: 船/生产力/星球数增量奖励。
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .fast_env import EnvState, fleet_speed, point_segment_distance_sq, scores


@dataclass(frozen=True)
class RewardParams:
    """奖励塑造超参（编译时固定）。"""
    terminal_weight: float = 1.0              # 终局奖励系数
    ships_weight: float = 0.0                 # 船只增量奖励系数
    production_weight: float = 0.0            # 生产力增量奖励系数
    planet_weight: float = 0.0                # 星球数增量奖励系数
    time_penalty: float = 0.0                 # 每步时间惩罚（鼓励快速结束）
    launch_penalty: float = 0.0               # 每次发射舰队惩罚（抑制过度分散）
    neutral_capture_weight: float = 0.0       # 掌控中立星球奖励系数
    neutral_capture_eta_scale: float = 12.0   # eta 指数平滑尺度
    neutral_capture_early_steps: int = 150    # 早期加成步数界限
    enemy_capture_weight: float = 0.0         # 占领敌方星球奖励系数
    lose_planet_penalty: float = 0.0          # 失去星球惩罚系数
    wasted_attack_penalty: float = 0.0        # 无效进攻惩罚系数
    step_clip: float = 0.0                    # 单步塑形奖励幅度剪裁
    failed_capture_penalty: float = 0.0       # 发射后 N 步内目标未变我方的惩罚
    failed_capture_window: int = 50           # 检测窗口（步数）

    @classmethod
    def from_train_config(cls, cfg) -> "RewardParams":
        reward = cfg.reward
        return cls(
            terminal_weight=float(getattr(reward, "terminal_weight", 1.0)),
            ships_weight=float(getattr(reward, "ships_weight", 0.0)),
            production_weight=float(getattr(reward, "production_weight", 0.0)),
            planet_weight=float(getattr(reward, "planet_weight", 0.0)),
            time_penalty=float(getattr(reward, "time_penalty", 0.0)),
            launch_penalty=float(getattr(reward, "launch_penalty", 0.0)),
            neutral_capture_weight=float(getattr(reward, "neutral_capture_weight", 0.0)),
            neutral_capture_eta_scale=float(getattr(reward, "neutral_capture_eta_scale", 12.0)),
            neutral_capture_early_steps=int(getattr(reward, "neutral_capture_early_steps", 150)),
            enemy_capture_weight=float(getattr(reward, "enemy_capture_weight", 0.0)),
            lose_planet_penalty=float(getattr(reward, "lose_planet_penalty", 0.0)),
            wasted_attack_penalty=float(getattr(reward, "wasted_attack_penalty", 0.0)),
            step_clip=float(getattr(reward, "step_clip", 0.0)),
            failed_capture_penalty=float(getattr(reward, "failed_capture_penalty", 0.0)),
            failed_capture_window=int(getattr(reward, "failed_capture_window", 50)),
        )


def reward_from_transition(prev: EnvState, cur: EnvState, learner_actions: jnp.ndarray, params: RewardParams) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """计算单步奖励并返回各分项调试信息。"""
    shaped = shaped_reward(prev, cur, params)
    neutral = neutral_capture_reward(prev, cur, params)
    enemy_capture = enemy_capture_reward(prev, cur, params)
    lose_planet = lose_planet_reward(prev, cur, params)
    wasted = wasted_attack_reward(prev, cur, params)
    time_penalty = -params.time_penalty * jnp.ones_like(shaped)
    launch_count = (learner_actions[..., 2] > 0.0).sum(axis=-1).astype(jnp.float32)
    launch_penalty = -params.launch_penalty * launch_count
    raw_terminal = terminal_reward_player0(cur)
    terminal = jnp.where(cur.done, raw_terminal, 0.0) * params.terminal_weight
    total = shaped + neutral + enemy_capture + lose_planet + wasted + time_penalty + launch_penalty + terminal
    return total, {
        "shaped_reward": shaped,
        "neutral_capture_reward": neutral,
        "enemy_capture_reward": enemy_capture,
        "lose_planet_reward": lose_planet,
        "wasted_attack_reward": wasted,
        "terminal_reward": terminal,
        "launch_count": launch_count,
}


def terminal_reward_player0(state: EnvState) -> jnp.ndarray:
    """计算学习方（player 0）终局奖励。

    - 赢（score0 > score1）: +1
    - 平局（score0 == score1）: -1（与输相同，逼迫主动进攻，防止 self-play 策略退化）
    - 输或 0-0: -1
    - 使用 scores() 重新计算，而非直接读 winner 字段（winner 仅保存存活玩家数，不等价）。
    """
    score = scores(state)
    s0 = score[..., 0]
    s1 = score[..., 1]
    # 赢: +1, 平局/输/0-0: -1
    return jnp.where(s0 > s1, 1.0, -1.0)


def score_snapshot(state: EnvState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """摄取当前局势的局部得分（不用于终局评价）。"""
    owner = state.planets[..., 1].astype(jnp.int32)
    mine = state.planet_alive & (owner == 0)
    planets = mine.sum(axis=-1).astype(jnp.float32)
    ships_on_planets = jnp.where(mine, state.planets[..., 5], 0.0).sum(axis=-1)
    production = jnp.where(mine, state.planets[..., 6], 0.0).sum(axis=-1)
    my_fleets = state.fleet_alive & (state.fleets[..., 1].astype(jnp.int32) == 0)
    ships_in_fleets = jnp.where(my_fleets, state.fleets[..., 6], 0.0).sum(axis=-1)
    return planets, ships_on_planets, ships_in_fleets, production


def shaped_reward(prev: EnvState, cur: EnvState, params: RewardParams) -> jnp.ndarray:
    """根据船/生产力/星球数的局间变化计算塑形奖励。"""
    prev_planets, prev_ships, _prev_fleet_ships, prev_prod = score_snapshot(prev)
    cur_planets, cur_ships, _cur_fleet_ships, cur_prod = score_snapshot(cur)
    reward = (
        (cur_ships - prev_ships) * params.ships_weight
        + (cur_prod - prev_prod) * params.production_weight
        + (cur_planets - prev_planets) * params.planet_weight
    )
    if params.step_clip <= 0.0:
        return reward
    return jnp.clip(reward, -params.step_clip, params.step_clip)


def neutral_capture_reward(prev: EnvState, cur: EnvState, params: RewardParams) -> jnp.ndarray:
    """掌控中立星球奖励。越早、己方线越短奖励越高。"""
    if params.neutral_capture_weight <= 0.0:
        return jnp.zeros_like(prev.step, dtype=jnp.float32)
    early_steps = max(float(params.neutral_capture_early_steps), 1.0)
    early_factor = jnp.maximum(0.0, 1.0 - prev.step.astype(jnp.float32) / early_steps)  # 早期加成因子
    prev_owner = prev.planets[..., 1].astype(jnp.int32)
    cur_owner = cur.planets[..., 1].astype(jnp.int32)
    captured = prev.planet_alive & cur.planet_alive & (prev_owner == -1) & (cur_owner == 0)  # 中立变为己方
    my_prev = prev.planet_alive & (prev_owner == 0)
    dx = prev.planets[:, :, None, 2] - prev.planets[:, None, :, 2]
    dy = prev.planets[:, :, None, 3] - prev.planets[:, None, :, 3]
    dist = jnp.sqrt(dx * dx + dy * dy)
    ships_needed = jnp.maximum(prev.planets[:, None, :, 5] + 1.0, 1.0)
    eta = dist / jnp.maximum(fleet_speed(ships_needed), 1e-6)  # 己方各星球到达时间
    eta = jnp.where(my_prev[:, :, None], eta, 1e9)
    nearest_eta = eta.min(axis=1)
    bonus = params.neutral_capture_weight * jnp.exp(-nearest_eta / max(params.neutral_capture_eta_scale, 1e-6)) * early_factor[:, None]
    return jnp.where(captured, bonus, 0.0).sum(axis=-1)


def enemy_capture_reward(prev: EnvState, cur: EnvState, params: RewardParams) -> jnp.ndarray:
    """占领敌方星球奖励：生产力越高奖励越多。"""
    if params.enemy_capture_weight <= 0.0:
        return jnp.zeros_like(prev.step, dtype=jnp.float32)
    prev_owner = prev.planets[..., 1].astype(jnp.int32)
    cur_owner = cur.planets[..., 1].astype(jnp.int32)
    captured = prev.planet_alive & cur.planet_alive & (prev_owner == 1) & (cur_owner == 0)  # 敌方变为己方
    return jnp.where(captured, params.enemy_capture_weight * (1.0 + prev.planets[..., 6]), 0.0).sum(axis=-1)


def lose_planet_reward(prev: EnvState, cur: EnvState, params: RewardParams) -> jnp.ndarray:
    """失去己方星球惩罚：生产力越高惩罚越重。"""
    if params.lose_planet_penalty <= 0.0:
        return jnp.zeros_like(prev.step, dtype=jnp.float32)
    prev_owner = prev.planets[..., 1].astype(jnp.int32)
    cur_owner = cur.planets[..., 1].astype(jnp.int32)
    lost = prev.planet_alive & (prev_owner == 0) & ((~cur.planet_alive) | (cur_owner != 0))  # 己方星球消失或易手
    return -jnp.where(lost, params.lose_planet_penalty * (1.0 + prev.planets[..., 6]), 0.0).sum(axis=-1)


def wasted_attack_reward(prev: EnvState, cur: EnvState, params: RewardParams) -> jnp.ndarray:
    """惩罚己方舰队抵达非友方星球但未成功占领的消耗。"""
    if params.wasted_attack_penalty <= 0.0:
        return jnp.zeros_like(prev.step, dtype=jnp.float32)
    prev_owner = prev.fleets[..., 1].astype(jnp.int32)
    cur_ids = jnp.where(cur.fleet_alive, cur.fleets[..., 0], -1.0)
    still_alive = (prev.fleets[..., 0, None] == cur_ids[:, None, :]).any(axis=-1)  # 己方舰队存活状态
    disappeared = prev.fleet_alive & (prev_owner == 0) & (~still_alive)
    target_owner = estimate_hit_owner(prev.fleets, cur.planets, cur.planet_alive)
    wasted = disappeared & (target_owner != 0) & (target_owner != -999)
    return -jnp.where(wasted, params.wasted_attack_penalty * prev.fleets[..., 6], 0.0).sum(axis=-1)


def estimate_hit_owner(fleets: jnp.ndarray, planets: jnp.ndarray, planet_alive: jnp.ndarray) -> jnp.ndarray:
    speed = fleet_speed(jnp.maximum(fleets[..., 6], 1.0))
    x0 = fleets[..., 2]
    y0 = fleets[..., 3]
    x1 = x0 + jnp.cos(fleets[..., 4]) * speed
    y1 = y0 + jnp.sin(fleets[..., 4]) * speed
    d2 = point_segment_distance_sq(
        planets[:, None, :, 2],
        planets[:, None, :, 3],
        x0[:, :, None],
        y0[:, :, None],
        x1[:, :, None],
        y1[:, :, None],
    )
    hit = d2 <= (planets[:, None, :, 4] ** 2 + 1e-6)
    hit = hit & planet_alive[:, None, :]
    dist = jnp.where(hit, d2, 1e9)
    idx = dist.argmin(axis=-1)
    has_hit = hit.any(axis=-1)
    owner = jnp.take_along_axis(planets[:, :, 1], idx, axis=1)
    return jnp.where(has_hit, owner.astype(jnp.int32), -999)
