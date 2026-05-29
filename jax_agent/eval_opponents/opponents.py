"""JAX 自回归 agent 的评估对手集合。"""

from __future__ import annotations

import importlib.util
import inspect
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from ..fast_env import EnvParams, EnvState, placements, scores, step_batch


EvalResult = dict[str, float | int]
EVAL_METRIC_SUFFIXES = [
    "win_rate",
    "top2_rate",
    "avg_place",
    "place1_rate",
    "place2_rate",
    "place3_rate",
    "place4_rate",
]


def ranking_result(final_state: EnvState, learner_idx: int, num_players: int, total_games: int) -> EvalResult:
    """从 final_state 统计四人局排名指标。

    win_rate 是唯一第一名率；place1_rate 使用分数名次，包含并列第一。
    """
    total = max(int(total_games), 1)
    winner = np.asarray(final_state.winner)
    place = np.asarray(placements(final_state, num_players))[:, learner_idx]
    score = np.asarray(scores(final_state, num_players))
    learner_score = score[:, learner_idx]
    max_score = score.max(axis=1)
    wins = int((winner == learner_idx).sum())
    draws = int((winner == num_players).sum())
    top2 = int(((place <= 2) & (max_score > 0.0)).sum())
    result: EvalResult = {
        "wins": wins,
        "losses": int(total_games - wins - draws),
        "draws": draws,
        "win_rate": wins / total,
        "loss_rate": (total_games - wins - draws) / total,
        "top2_rate": top2 / total,
        "avg_place": float(place.mean()) if total_games > 0 else 0.0,
        "avg_score": float(learner_score.mean()) if total_games > 0 else 0.0,
    }
    for rank in range(1, 5):
        result[f"place{rank}_rate"] = int((place == rank).sum()) / total
    return result


def reward_ranking_result(rewards: list[float], total_games: int) -> EvalResult:
    """外部环境兜底：用 Kaggle reward 正负统计，无法可靠区分 2/3/4 名。"""
    total = max(int(total_games), 1)
    arr = np.asarray(rewards, dtype=np.float32)
    wins = int((arr > 0.0).sum())
    draws = int((arr == 0.0).sum())
    losses = int((arr < 0.0).sum())
    result: EvalResult = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "top2_rate": wins / total,
        "avg_place": 0.0,
        "avg_score": float(arr.mean()) if total_games > 0 else 0.0,
    }
    for rank in range(1, 5):
        result[f"place{rank}_rate"] = 0.0
    return result


@dataclass(frozen=True)
class EvalContext:
    """评估当前 learner 时需要的运行时对象。"""

    train_state: Any
    model: Any
    bank: EnvState
    env_params: EnvParams
    feat_params: Any
    normalizer_state: Any
    norm_params: Any
    num_games: int
    base_seed: int
    rng: Any
    deterministic_policy_actions: Callable
    nearest_opponent_actions: Callable
    reset_from_bank: Callable
    flip_state: Callable
    flip_actions: Callable
    kaggle_obs_to_jax_state: Callable
    jax_actions_to_kaggle: Callable
    eval_player: int = 0


class JaxEvalOpponent(Protocol):
    """单个评估基准对手的统一接口。"""

    name: str
    metric_prefix: str

    def exists(self) -> bool:
        ...

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        ...

    def metric_fields(self) -> list[str]:
        return [f"eval_{self.metric_prefix}_{suffix}" for suffix in EVAL_METRIC_SUFFIXES]

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return {
            f"eval_{self.metric_prefix}_{suffix}": result.get(suffix, "")
            for suffix in EVAL_METRIC_SUFFIXES
        }


class ExternalScriptOpponent:
    """加载 Orbit Wars Lab agent 目录，并隔离每局的模块状态。"""

    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir.resolve()
        self.main_py = self.agent_dir / "main.py"
        if not self.main_py.exists():
            raise FileNotFoundError(f"External opponent entry not found: {self.main_py}")
        self.modules: dict[int, Any] = {}
        self.call_with_config: dict[int, bool] = {}
        self.load_counter = 0

    def reset_env(self, env_index: int, seed: int) -> None:
        self.modules[env_index], self.call_with_config[env_index] = self._load_module(env_index, seed)

    def act_for_env(self, env_index: int, observation: Any) -> list[list[float | int]]:
        if env_index not in self.modules:
            self.reset_env(env_index, seed=0)
        return self._call_agent(env_index, observation)

    def _load_module(self, env_index: int, seed: int) -> tuple[Any, bool]:
        self.load_counter += 1
        module_name = f"_orbit_external_{self.agent_dir.name.replace('-', '_')}_{id(self)}_{env_index}_{self.load_counter}"
        spec = importlib.util.spec_from_file_location(module_name, self.main_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load external opponent: {self.main_py}")
        module = importlib.util.module_from_spec(spec)
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(self.agent_dir))
            spec.loader.exec_module(module)
        finally:
            sys.path = old_path
        agent = getattr(module, "agent", None)
        if agent is None or not callable(agent):
            raise AttributeError(f"External opponent has no callable agent(obs): {self.main_py}")
        random.seed(int(seed))
        try:
            call_with_config = len(inspect.signature(agent).parameters) >= 2
        except (TypeError, ValueError):
            call_with_config = False
        return module, call_with_config

    def _call_agent(self, env_index: int, observation: Any) -> list[list[float | int]]:
        module = self.modules[env_index]
        if self.call_with_config.get(env_index, False):
            moves = module.agent(observation, {"__raw_path__": str(self.main_py)})
        else:
            moves = module.agent(observation)
        return list(moves or [])


@dataclass(frozen=True)
class NearestJaxEvalOpponent:
    """完全在 JAX fast_env 内运行的 nearest 规则对手。"""

    name: str = "nearest"
    metric_prefix: str = "nearest"

    def exists(self) -> bool:
        return True

    def metric_fields(self) -> list[str]:
        return JaxEvalOpponent.metric_fields(self)

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return JaxEvalOpponent.metric_values(self, result)

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        _rng, reset_key = jax.random.split(ctx.rng)
        state = ctx.reset_from_bank(ctx.bank, reset_key, ctx.num_games)
        num_players = int(getattr(ctx.env_params, "num_players", 2))
        learner_idx = int(getattr(ctx, "eval_player", 0)) % max(num_players, 1)

        def step_fn(st, _):
            learner_st = ctx.flip_state(st, learner_idx) if learner_idx != 0 else st
            learner_acts = ctx.deterministic_policy_actions(
                ctx.train_state,
                learner_st,
                ctx.feat_params,
                ctx.normalizer_state,
                ctx.norm_params,
            )
            if learner_idx != 0:
                learner_acts = ctx.flip_actions(learner_acts, learner_idx)
            acts = []
            for player in range(num_players):
                if player == learner_idx:
                    acts.append(learner_acts)
                else:
                    acts.append(ctx.nearest_opponent_actions(st, player=player, params=ctx.feat_params))
            joint = jnp.stack(acts, axis=1)
            return step_batch(st, joint, ctx.env_params), None

        final_state, _ = jax.lax.scan(step_fn, state, None, length=ctx.env_params.episode_steps)
        result = ranking_result(final_state, learner_idx, num_players, ctx.num_games)
        print(
            f"[eval/{self.name}]   p{learner_idx} wins={result['wins']} losses={result['losses']} "
            f"draws={result['draws']} win_rate={result['win_rate']:.3f} "
            f"top2={result['top2_rate']:.3f} avg_place={result['avg_place']:.2f}"
        )
        return result


@dataclass(frozen=True)
class ExternalScriptJaxEvalOpponent:
    """用 Kaggle 环境评估当前 JAX learner 对外部脚本 agent 的胜率。"""

    name: str
    metric_prefix: str
    agent_dir: Path
    seed_offset: int = 0

    def exists(self) -> bool:
        return (self.agent_dir / "main.py").exists()

    def metric_fields(self) -> list[str]:
        return JaxEvalOpponent.metric_fields(self)

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return JaxEvalOpponent.metric_values(self, result)

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        try:
            from kaggle_environments import make as kag_make
        except ImportError:
            print(f"[eval/{self.name}] kaggle_environments 未安装，跳过")
            return {}

        opponent = ExternalScriptOpponent(self.agent_dir)
        rewards: list[float] = []
        final_states: list[EnvState] = []
        num_players = int(getattr(ctx.env_params, "num_players", 2))
        learner_idx = int(getattr(ctx, "eval_player", 0)) % max(num_players, 1)

        @jax.jit
        def infer(st):
            return ctx.deterministic_policy_actions(
                ctx.train_state,
                st,
                ctx.feat_params,
                ctx.normalizer_state,
                ctx.norm_params,
            )

        for game_idx in range(ctx.num_games):
            seed = ctx.base_seed + self.seed_offset + game_idx
            for player in range(num_players):
                if player != learner_idx:
                    opponent.reset_env(player, seed + player * 1009)
            env = kag_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
            env.reset(num_agents=num_players)
            states = env.step([[] for _ in range(num_players)])
            obs = states[learner_idx]["observation"]
            done = states[0]["status"] != "ACTIVE"
            while not done:
                jax_state = ctx.kaggle_obs_to_jax_state(obs, ctx.env_params)
                if learner_idx != 0:
                    jax_state = ctx.flip_state(jax_state, learner_idx)
                jax_acts = infer(jax_state)
                if learner_idx != 0:
                    jax_acts = ctx.flip_actions(jax_acts, learner_idx)
                learner_action = ctx.jax_actions_to_kaggle(jax_acts)
                joint_actions = [[] for _ in range(num_players)]
                joint_actions[learner_idx] = learner_action
                for player in range(num_players):
                    if player == learner_idx:
                        continue
                    joint_actions[player] = opponent.act_for_env(player, states[player]["observation"])
                states = env.step(joint_actions)
                obs = states[learner_idx]["observation"]
                done = states[0]["status"] != "ACTIVE"

            reward = states[learner_idx].get("reward", 0.0) or 0.0
            rewards.append(float(reward))
            try:
                final_states.append(ctx.kaggle_obs_to_jax_state(states[learner_idx]["observation"], ctx.env_params))
            except Exception:
                final_states = []
            print(f"  [eval/{self.name}] game={game_idx + 1} seed={seed} p{learner_idx} reward={reward:.3f}")

        if len(final_states) == ctx.num_games:
            stacked = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *final_states)
            result = ranking_result(stacked, learner_idx, num_players, ctx.num_games)
        else:
            result = reward_ranking_result(rewards, ctx.num_games)
        print(
            f"[eval/{self.metric_prefix}]     wins={result['wins']} losses={result['losses']} "
            f"draws={result['draws']} win_rate={result['win_rate']:.3f} "
            f"top2={result['top2_rate']:.3f} avg_place={result['avg_place']:.2f}"
        )
        return result


def default_eval_opponents(root: Path, *, enabled_names=None) -> list[JaxEvalOpponent]:
    """根据配置创建默认评估对手列表。"""

    enabled = set(enabled_names or ["nearest", "sigma", "tamrazov"])
    agents_root = root / "agents"
    external_root = agents_root / "external"
    opponents: list[JaxEvalOpponent] = []
    if "nearest" in enabled:
        opponents.append(NearestJaxEvalOpponent())
    if "sigma" in enabled or "sigmaborov-reinforce" in enabled:
        opponents.append(ExternalScriptJaxEvalOpponent(
            name="sigmaborov-reinforce",
            metric_prefix="sigma",
            agent_dir=external_root / "sigmaborov-reinforce",
            seed_offset=7919,
        ))
    if "tamrazov" in enabled or "tamrazov-starwars" in enabled:
        opponents.append(ExternalScriptJaxEvalOpponent(
            name="tamrazov-starwars",
            metric_prefix="tamrazov",
            agent_dir=external_root / "tamrazov-starwars",
            seed_offset=0,
        ))
    if "pool_ckpt900" in enabled:
        opponents.append(ExternalScriptJaxEvalOpponent(
            name="jax-pool-0514-resume-ckpt900",
            metric_prefix="pool_ckpt900",
            agent_dir=agents_root / "mine" / "jax-pool-0514-resume-ckpt900",
            seed_offset=31337,
        ))
    return opponents


def eval_metric_fields(opponents: list[JaxEvalOpponent]) -> list[str]:
    fields: list[str] = []
    for opponent in opponents:
        fields.extend(opponent.metric_fields())
    return fields
