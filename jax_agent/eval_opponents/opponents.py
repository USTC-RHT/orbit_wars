"""JAX 自回归 agent 的评估对手集合。"""

from __future__ import annotations

import importlib.util
import inspect
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from ..fast_env import EnvParams, EnvState, step_batch


EvalResult = dict[str, float | int]


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
        return [f"eval_{self.metric_prefix}_win_rate"]

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return {f"eval_{self.metric_prefix}_win_rate": result["win_rate"]}


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
        learner_idx = 1 if int(getattr(ctx, "eval_player", 0)) == 1 else 0
        opponent_idx = 1 - learner_idx

        def step_fn(st, _):
            if learner_idx == 1:
                learner_st = ctx.flip_state(st)
                learner_acts = ctx.deterministic_policy_actions(
                    ctx.train_state,
                    learner_st,
                    ctx.feat_params,
                    ctx.normalizer_state,
                    ctx.norm_params,
                )
                learner_acts = ctx.flip_actions(learner_acts)
                opp_acts = ctx.nearest_opponent_actions(st, player=0, params=ctx.feat_params)
                joint = jnp.stack([opp_acts, learner_acts], axis=1)
            else:
                learner_acts = ctx.deterministic_policy_actions(
                    ctx.train_state,
                    st,
                    ctx.feat_params,
                    ctx.normalizer_state,
                    ctx.norm_params,
                )
                opp_acts = ctx.nearest_opponent_actions(st, player=1, params=ctx.feat_params)
                joint = jnp.stack([learner_acts, opp_acts], axis=1)
            return step_batch(st, joint, ctx.env_params), None

        final_state, _ = jax.lax.scan(step_fn, state, None, length=ctx.env_params.episode_steps)
        winners = np.array(final_state.winner)
        wins = int((winners == learner_idx).sum())
        losses = int((winners == opponent_idx).sum())
        draws = int((winners == 2).sum())
        total = max(ctx.num_games, 1)
        result = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total, "loss_rate": losses / total}
        print(f"[eval/{self.name}]   p{learner_idx} wins={wins} losses={losses} draws={draws} win_rate={result['win_rate']:.3f}")
        return result


class CheckpointJaxEvalOpponent:
    """直接用 JAX checkpoint 参数作为评估对手，不依赖外部脚本 agent。"""

    def __init__(
        self,
        *,
        name: str,
        metric_prefix: str,
        checkpoint_path: Path,
    ) -> None:
        self.name = name
        self.metric_prefix = metric_prefix
        self.checkpoint_path = checkpoint_path.resolve()
        self._loaded: tuple[Any, Any] | None = None

    def exists(self) -> bool:
        if self.checkpoint_path.exists():
            return True
        print(f"[eval/{self.name}] checkpoint 不存在，跳过: {self.checkpoint_path}")
        return False

    def metric_fields(self) -> list[str]:
        return JaxEvalOpponent.metric_fields(self)

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return JaxEvalOpponent.metric_values(self, result)

    def _load(self, ctx: EvalContext) -> tuple[Any, Any]:
        if self._loaded is None:
            with self.checkpoint_path.open("rb") as f:
                ckpt = pickle.load(f)
            params = ckpt["params"]
            expected_leaves, expected_tree = jax.tree_util.tree_flatten(ctx.train_state.params)
            loaded_leaves, loaded_tree = jax.tree_util.tree_flatten(params)
            if loaded_tree != expected_tree:
                raise ValueError(f"Checkpoint param tree does not match current model: {self.checkpoint_path}")
            mismatches = [
                (i, getattr(exp, "shape", None), getattr(got, "shape", None))
                for i, (exp, got) in enumerate(zip(expected_leaves, loaded_leaves, strict=True))
                if getattr(exp, "shape", None) != getattr(got, "shape", None)
            ]
            if mismatches:
                raise ValueError(
                    f"Checkpoint param shapes do not match current model: {self.checkpoint_path}; "
                    f"first mismatches={mismatches[:5]}"
                )
            norm_state = ckpt.get("normalizer_state", ctx.normalizer_state)
            self._loaded = (params, norm_state)
            print(f"[eval/{self.name}] loaded JAX checkpoint: {self.checkpoint_path}")
        return self._loaded

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        params, opponent_norm_state = self._load(ctx)
        opponent_train_state = ctx.train_state.replace(params=params)
        _rng, reset_key = jax.random.split(ctx.rng)
        state = ctx.reset_from_bank(ctx.bank, reset_key, ctx.num_games)
        learner_idx = 1 if int(getattr(ctx, "eval_player", 0)) == 1 else 0
        opponent_idx = 1 - learner_idx

        @jax.jit
        def learner_infer(st):
            return ctx.deterministic_policy_actions(
                ctx.train_state,
                st,
                ctx.feat_params,
                ctx.normalizer_state,
                ctx.norm_params,
            )

        @jax.jit
        def opponent_infer(st):
            return ctx.deterministic_policy_actions(
                opponent_train_state,
                st,
                ctx.feat_params,
                opponent_norm_state,
                ctx.norm_params,
            )

        def step_fn(st, _):
            if learner_idx == 1:
                learner_st = ctx.flip_state(st)
                learner_acts = learner_infer(learner_st)
                learner_acts = ctx.flip_actions(learner_acts)
                opp_acts = opponent_infer(st)
                joint = jnp.stack([opp_acts, learner_acts], axis=1)
            else:
                learner_acts = learner_infer(st)
                opp_st = ctx.flip_state(st)
                opp_acts = opponent_infer(opp_st)
                opp_acts = ctx.flip_actions(opp_acts)
                joint = jnp.stack([learner_acts, opp_acts], axis=1)
            return step_batch(st, joint, ctx.env_params), None

        final_state, _ = jax.lax.scan(step_fn, state, None, length=ctx.env_params.episode_steps)
        winners = np.array(final_state.winner)
        wins = int((winners == learner_idx).sum())
        losses = int((winners == opponent_idx).sum())
        draws = int((winners == 2).sum())
        total = max(ctx.num_games, 1)
        result = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total, "loss_rate": losses / total}
        print(f"[eval/{self.name}]   p{learner_idx} wins={wins} losses={losses} draws={draws} win_rate={result['win_rate']:.3f}")
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
        wins = losses = draws = 0

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
            opponent.reset_env(0, seed)
            env = kag_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
            env.reset(num_agents=2)
            states = env.step([[], []])
            learner_idx = 1 if int(getattr(ctx, "eval_player", 0)) == 1 else 0
            opponent_idx = 1 - learner_idx
            obs = states[learner_idx]["observation"]
            opp_obs = states[opponent_idx]["observation"]
            done = states[0]["status"] != "ACTIVE"
            while not done:
                jax_state = ctx.kaggle_obs_to_jax_state(obs, ctx.env_params)
                if learner_idx == 1:
                    jax_state = ctx.flip_state(jax_state)
                jax_acts = infer(jax_state)
                if learner_idx == 1:
                    jax_acts = ctx.flip_actions(jax_acts)
                learner_action = ctx.jax_actions_to_kaggle(jax_acts)
                opp_action = opponent.act_for_env(0, opp_obs)
                joint_actions = [None, None]
                joint_actions[learner_idx] = learner_action
                joint_actions[opponent_idx] = opp_action
                states = env.step(joint_actions)
                obs = states[learner_idx]["observation"]
                opp_obs = states[opponent_idx]["observation"]
                done = states[0]["status"] != "ACTIVE"

            reward = states[learner_idx].get("reward", 0.0) or 0.0
            if reward > 0:
                wins += 1
                outcome = "win"
            elif reward < 0:
                losses += 1
                outcome = "loss"
            else:
                draws += 1
                outcome = "draw"
            print(f"  [eval/{self.name}] game={game_idx + 1} seed={seed} p{learner_idx} {outcome} reward={reward:.1f}")

        total = max(ctx.num_games, 1)
        result = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total, "loss_rate": losses / total}
        print(f"[eval/{self.metric_prefix}]     wins={wins} losses={losses} draws={draws} win_rate={result['win_rate']:.3f}")
        return result


def default_eval_opponents(root: Path, *, enabled_names=None, init_checkpoint_path: Path | None = None) -> list[JaxEvalOpponent]:
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
    if {"init_checkpoint", "init_ckpt", "init_ckpt32700"} & enabled:
        if init_checkpoint_path is None:
            print("[eval/init_checkpoint] init_from_checkpoint 未配置，跳过")
        else:
            opponents.append(CheckpointJaxEvalOpponent(
                name="init_checkpoint",
                metric_prefix="init_checkpoint",
                checkpoint_path=init_checkpoint_path,
            ))
    return opponents


def eval_metric_fields(opponents: list[JaxEvalOpponent]) -> list[str]:
    fields: list[str] = []
    for opponent in opponents:
        fields.extend(opponent.metric_fields())
    return fields
