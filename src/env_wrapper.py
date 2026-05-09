"""Orbit Wars 环境封装和 reward 计算。

这个文件把 Kaggle 环境包装成 PPO 训练需要的接口：reset 返回网络输入，
step 同时执行 learner 和 opponent 动作，并计算终局奖励加 shaped reward。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import RewardConfig, TrainConfig
from .features import TurnBatch, encode_turn
from .opponents import Opponent
from .state import GameState, parse_observation, read


@dataclass(slots=True)
class StepResult:
    """环境执行一步后的结果。

    batch 是下一帧的网络输入；reward 是 shaped + terminal 的训练奖励；
    info 保存拆开的奖励项，方便日志分析。
    """

    batch: TurnBatch
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(slots=True)
class ScoreSnapshot:
    """用于计算 dense reward 的我方局势快照。"""

    planets: int
    ships_on_planets: float
    ships_in_fleets: float
    production: float


class OrbitWarsPPOEnv:
    """Kaggle Orbit Wars 环境的 PPO 包装器。

    负责：
    - 创建/重置环境。
    - 根据 learner_player 决定我方在 0 号还是 1 号玩家。
    - 拼接 learner 和 opponent 动作。
    - 把环境 observation 编码成 TurnBatch。
    - 计算终局奖励和局势变化 shaped reward。
    """

    def __init__(self, cfg: TrainConfig, opponent: Opponent, env_index: int) -> None:
        self.cfg = cfg
        self.opponent = opponent
        self.env_index = env_index
        self.env: Any | None = None
        self.learner_player = 0
        self.episode_index = 0
        self.last_obs: Any | None = None
        self.last_opp_obs: Any | None = None
        self.last_score: ScoreSnapshot | None = None

    def reset(self, seed: int) -> TurnBatch:
        """重置一局游戏，并返回第一帧可训练 batch。"""

        from kaggle_environments import make

        if self.cfg.alternate_player_sides:
            # [trick] 轮流扮演 0/1 号玩家，防止网络只学到"先手/后手优势"而非真正的策略泛化。
            # env_index 保证同一 update 内不同并行环境不会全部扮演同一方，覆盖两侧经验。
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0
        self.env = make("orbit_wars", configuration={"seed": int(seed), "randomSeed": int(seed)}, debug=False)
        self.env.reset(num_agents=2)
        states = self.env.step([[], []])
        player_state = states[self.learner_player]
        opp_state = states[1 - self.learner_player]
        self.last_obs = extract_observation(player_state)
        self.last_opp_obs = extract_observation(opp_state)
        self.last_score = score_snapshot(parse_observation(self.last_obs))
        self.episode_index += 1
        return encode_turn(self.last_obs, self.cfg.env)

    def step(self, action: list[list[float | int]], action_infos: list[dict[str, Any]] | None = None) -> StepResult:
        """执行 learner 动作和 opponent 动作，并返回训练需要的信息。"""

        if self.env is None or self.last_obs is None or self.last_opp_obs is None:
            raise RuntimeError("Call reset before step.")
        opponent_action = self.opponent.act(self.last_opp_obs)
        joint = [action, opponent_action] if self.learner_player == 0 else [opponent_action, action]
        states = self.env.step(joint)
        player_state = states[self.learner_player]
        opp_state = states[1 - self.learner_player]
        self.last_obs = extract_observation(player_state)
        self.last_opp_obs = extract_observation(opp_state)
        done = extract_status(player_state) != "ACTIVE"

        current_score = score_snapshot(parse_observation(self.last_obs))
        shaped = shaped_reward(self.last_score, current_score, self.cfg.reward)
        time_penalty = -float(self.cfg.reward.time_penalty)
        launch_penalty = -float(self.cfg.reward.launch_penalty) * float(len(action))
        eta_bonus = reachable_eta_bonus(action_infos or [], self.cfg.reward)
        self.last_score = current_score
        raw_terminal = terminal_reward(player_state) if done else 0.0
        terminal = raw_terminal * self.cfg.reward.terminal_weight if done else 0.0
        reward = shaped + eta_bonus + time_penalty + launch_penalty + terminal

        return StepResult(
            batch=encode_turn(self.last_obs, self.cfg.env),
            reward=float(reward),
            done=done,
            info={
                "terminal_reward": float(terminal),
                "raw_terminal_reward": float(raw_terminal),
                "shaped_reward": float(shaped),
                "eta_bonus": float(eta_bonus),
                "time_penalty": float(time_penalty),
                "launch_penalty": float(launch_penalty),
                "launch_count": len(action),
                "learner_player": self.learner_player,
            },
        )


def reachable_eta_bonus(action_infos: list[dict[str, Any]], cfg: RewardConfig) -> float:
    """Small one-shot bonus for physics-validated non-friendly launches.

    `action_infos` is produced only after aim_with_prediction succeeds, so failed
    or unreachable selected actions never receive this bonus.
    """

    if cfg.eta_bonus_weight <= 0.0:
        return 0.0
    bonus = 0.0
    for info in action_infos:
        if int(info.get("target_owner", -999)) == int(info.get("player", -998)):
            continue
        eta = max(float(info.get("eta", 0.0)), 1.0)
        production = max(float(info.get("target_production", 0.0)), 0.0)
        bonus += cfg.eta_bonus_weight * production / eta
    return bonus


def extract_observation(state: Any) -> Any:
    """兼容 dict / SimpleNamespace 状态对象。"""

    return read(state, "observation")


def extract_status(state: Any) -> str:
    return str(read(state, "status", "UNKNOWN"))


def extract_reward(state: Any) -> float:
    value = read(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def terminal_reward(player_state: Any) -> float:
    """把 Kaggle 终局 reward 规整成 -1/0/+1。

    [trick] Kaggle 的原始 reward 数值不固定，归一化后终局信号的量级
    与 shaped reward 之和可以通过 terminal_weight 统一调节，
    不受底层评分系统变动影响。
    """

    reward = extract_reward(player_state)
    if reward > 0:
        return 1.0
    if reward < 0:
        return -1.0
    return 0.0


def score_snapshot(state: GameState) -> ScoreSnapshot:
    """提取我方当前局势摘要，用于 shaped reward。"""

    mine = [p for p in state.planets if p.owner == state.player]
    my_fleets = [f for f in state.fleets if f.owner == state.player]
    return ScoreSnapshot(
        planets=len(mine),
        ships_on_planets=float(sum(p.ships for p in mine)),
        ships_in_fleets=float(sum(f.ships for f in my_fleets)),
        production=float(sum(p.production for p in mine)),
    )


def shaped_reward(prev: ScoreSnapshot | None, cur: ScoreSnapshot, cfg: RewardConfig) -> float:
    """根据局势变化给一点中间奖励。

    这不是游戏最终评分，只是帮助 PPO 更早知道扩张/增兵是否有利。
    """

    if prev is None:
        return 0.0
    reward = 0.0
    reward += (cur.ships_on_planets - prev.ships_on_planets) * cfg.ships_weight
    reward += (cur.ships_in_fleets - prev.ships_in_fleets) * cfg.fleet_weight
    reward += (cur.production - prev.production) * cfg.production_weight
    reward += (cur.planets - prev.planets) * cfg.planet_weight
    if cfg.step_clip <= 0.0:
        return reward
    # [trick] step_clip 把单步 shaping 幅度压住，防止密集奖励信号淹没终局胜负（terminal_weight）。
    # 理论上 potential-based shaping 不改变最优策略；clip 破坏了严格的 potential 性质，
    # 但换来训练数值稳定性，且 clip 值很小时对策略影响可忽略。step_clip <= 0 表示关闭裁剪。
    return max(-cfg.step_clip, min(cfg.step_clip, reward))
