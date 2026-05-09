"""训练和评估用的对手，以及策略动作解码。

包含 random、nearest 和 self-play 对手；也提供 `actions_from_policy`，
把神经网络的 target/amount 输出转成 Orbit Wars 需要的舰队列表。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .config import TrainConfig
from .features import DecisionContext, TurnBatch, encode_turn
from .model import ActorCritic
from .normalization import StateNormalizer
from .physics import aim_with_prediction
from .ppo import sample_amounts, sample_targets
from .state import GameState, Planet, parse_observation


@dataclass(slots=True)
class DecodedAction:
    """A physics-validated learner action plus reward/debug metadata."""

    move: list[float | int]
    source_id: int
    target_id: int
    target_owner: int
    target_production: int
    ships: int
    eta: float


class Opponent(Protocol):
    """训练对手统一接口。"""

    def act(self, observation: Any) -> list[list[float | int]]:
        ...


class RandomOpponent:
    """Kaggle 官方 random_agent 包装。"""

    def __init__(self) -> None:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        self.agent = random_agent

    def act(self, observation: Any) -> list[list[float | int]]:
        state = parse_observation(observation)
        payload = {"player": state.player, "planets": [planet_row(p) for p in state.planets]}
        return list(self.agent(payload))


class NearestOpponent:
    """简单最近目标对手，用作课程学习的中等难度靶子。"""

    def act(self, observation: Any) -> list[list[float | int]]:
        state = parse_observation(observation)
        sources = [p for p in state.planets if p.owner == state.player]
        targets = [p for p in state.planets if p.owner != state.player]
        moves: list[list[float | int]] = []
        for src in sources:
            if src.ships < 12 or not targets:
                continue
            target = min(targets, key=lambda p: (distance(src, p), p.id))
            ships = min(int(src.ships * 0.7), max(12, int(target.ships) + 2))
            if ships <= 0 or ships > src.ships:
                continue
            angle = math.atan2(target.y - src.y, target.x - src.x)
            moves.append([src.id, angle, ships])
        return moves


class SelfPlayOpponent:
    """冻结的旧策略对手。

    训练时每隔若干 update 调用 sync_from，把当前策略复制成新对手。
    """

    def __init__(self, cfg: TrainConfig, device: torch.device) -> None:
        from .features import CANDIDATE_DIM, GLOBAL_DIM, SELF_DIM, target_count

        self.cfg = cfg
        self.device = device
        self.state_normalizer: StateNormalizer | None = None
        self.policy = ActorCritic(
            self_dim=SELF_DIM,
            candidate_dim=CANDIDATE_DIM,
            global_dim=GLOBAL_DIM,
            candidate_count=target_count(cfg.env),
            ship_bucket_count=cfg.env.ship_bucket_count,
            hidden_size=cfg.model.hidden_size,
        ).to(device)
        self.policy.eval()

    def sync_from(self, model: ActorCritic, state_normalizer: StateNormalizer | None = None) -> None:
        """同步当前 learner 权重和输入归一化统计到 self-play opponent。"""

        self.policy.load_state_dict(model.state_dict())
        self.policy.eval()
        if state_normalizer is not None:
            self.state_normalizer = state_normalizer.copy()

    def act(self, observation: Any) -> list[list[float | int]]:
        """用冻结策略生成对手动作。"""

        batch = encode_turn(observation, self.cfg.env)
        return actions_from_policy(
            batch,
            self.policy,
            self.device,
            deterministic=self.cfg.self_play_deterministic,
            state_normalizer=self.state_normalizer,
        )


def build_opponent(name: str, cfg: TrainConfig, device: torch.device) -> Opponent:
    """按配置创建训练对手。"""

    if name == "random":
        return RandomOpponent()
    if name == "nearest":
        return NearestOpponent()
    if name == "self":
        return SelfPlayOpponent(cfg, device)
    raise ValueError(f"Unknown opponent: {name}")


def actions_from_policy(
    batch: TurnBatch,
    model: ActorCritic,
    device: torch.device,
    *,
    deterministic: bool,
    state_normalizer: StateNormalizer | None = None,
) -> list[list[float | int]]:
    """把神经网络输出转换成 Orbit Wars 动作。

    target head 决定目标，amount head 决定派兵档位；这里把二者还原成
    [source_id, angle, ships]。deterministic=True 时两步都用 argmax，
    用于正式推理/评估；deterministic=False 时两步都采样，用于训练探索。
    """

    if batch.self_features.shape[0] == 0:
        return []
    if state_normalizer is not None:
        batch = state_normalizer.normalize_batch(batch)
    with torch.inference_mode():
        target_outputs = model(
            torch.from_numpy(batch.self_features).to(device),
            torch.from_numpy(batch.candidate_features).to(device),
            torch.from_numpy(batch.global_features).to(device),
            torch.from_numpy(batch.candidate_mask).to(device).bool(),
        )
        target_sample = sample_targets(target_outputs, deterministic=deterministic)
        amount_outputs = model(
            torch.from_numpy(batch.self_features).to(device),
            torch.from_numpy(batch.candidate_features).to(device),
            torch.from_numpy(batch.global_features).to(device),
            torch.from_numpy(batch.candidate_mask).to(device).bool(),
            selected_targets=target_sample.target,
        )
        sampled = sample_amounts(
            amount_outputs,
            target_sample,
            deterministic=deterministic,
            amount_mask=torch.from_numpy(batch.amount_mask).to(device).bool(),
        )
    targets = sampled.target.detach().cpu().numpy()
    amounts = sampled.amount.detach().cpu().numpy()
    moves: list[list[float | int]] = []
    for row_idx, context in enumerate(batch.contexts):
        action_idx = int(targets[row_idx])
        amount_idx = int(amounts[row_idx])
        if action_idx == 0 or action_idx >= len(context.candidate_ids):
            continue
        if not context.candidate_mask[action_idx]:
            continue
        if amount_idx < 0 or amount_idx >= len(context.ship_buckets):
            continue
        if not context.amount_mask[action_idx, amount_idx]:
            continue
        ships = int(context.ship_buckets[amount_idx])
        if ships <= 0:
            continue
        move = action_from_indices(batch.state, context, action_idx, amount_idx)
        if move is not None:
            moves.append(move)
    return moves


def action_from_indices(
    state: GameState,
    context: DecisionContext,
    action_idx: int,
    amount_idx: int,
) -> list[float | int] | None:
    """把 target_idx + amount_idx 转成真实环境动作。

    这是模型和物理层的分界：模型只负责选目标和派兵档位；这里只对
    网络最终选中的这一条 src、target、ships 调 physics 求 angle/ETA。
    若 physics 找不到可达命中窗口，就丢弃这个动作，相当于本来源星球
    本回合不发兵。
    """

    decoded = decoded_action_from_indices(state, context, action_idx, amount_idx)
    return None if decoded is None else decoded.move


def decoded_action_from_indices(
    state: GameState,
    context: DecisionContext,
    action_idx: int,
    amount_idx: int,
) -> DecodedAction | None:
    """Return a move only when the selected source/target/ships can physically hit."""

    if action_idx == 0 or action_idx >= len(context.candidate_ids):
        return None
    if not context.candidate_mask[action_idx]:
        return None
    if amount_idx < 0 or amount_idx >= len(context.ship_buckets):
        return None
    ships = int(context.ship_buckets[amount_idx])
    if ships <= 0:
        return None
    if not context.amount_mask[action_idx, amount_idx]:
        return None

    planets_by_id = {planet.id: planet for planet in state.planets}
    src = planets_by_id.get(context.source_id)
    target = planets_by_id.get(context.candidate_ids[action_idx])
    if src is None or target is None:
        return None
    if target.owner != state.player and ships <= target.ships:
        return None

    initial_by_id = {planet.id: planet for planet in state.initial_planets}
    comet_ids = set(state.comet_planet_ids)
    result = aim_with_prediction(
        src,
        target,
        ships,
        initial_by_id,
        state.angular_velocity,
        state.comets,
        comet_ids,
        state.planets,
    )
    if result is None:
        return None
    angle, eta, _hit_x, _hit_y = result
    return DecodedAction(
        move=[context.source_id, float(angle), ships],
        source_id=context.source_id,
        target_id=target.id,
        target_owner=target.owner,
        target_production=target.production,
        ships=ships,
        eta=float(eta),
    )


def planet_row(p: Planet) -> list[float | int]:
    return [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]


def distance(a: Planet, b: Planet) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
