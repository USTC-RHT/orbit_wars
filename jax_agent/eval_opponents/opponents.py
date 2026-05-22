"""Evaluation opponents for JAX agents.

This module keeps evaluation opponents out of the training loop.  Each opponent
object evaluates the current JAX learner against one benchmark and returns the
same win/loss/draw metric shape used by train_jax.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from src.opponents import ExternalScriptOpponent

from ..fast_env import EnvParams, EnvState, step_batch


EvalResult = dict[str, float | int]


@dataclass(frozen=True)
class EvalContext:
    """Runtime objects needed to evaluate the current JAX learner."""

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
    deterministic_policy_actions_params: Callable
    nearest_opponent_actions: Callable
    reset_from_bank: Callable
    flip_state: Callable
    kaggle_obs_to_jax_state: Callable
    jax_actions_to_kaggle: Callable


class JaxEvalOpponent(Protocol):
    """One benchmark opponent for evaluating the current JAX learner."""

    name: str
    metric_prefix: str

    def exists(self) -> bool:
        ...

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        ...

    def metric_fields(self) -> list[str]:
        return [
            f"eval_{self.metric_prefix}_win_rate",
            f"eval_{self.metric_prefix}_loss_rate",
            f"eval_{self.metric_prefix}_wins",
            f"eval_{self.metric_prefix}_losses",
            f"eval_{self.metric_prefix}_draws",
        ]

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return {
            f"eval_{self.metric_prefix}_win_rate": result["win_rate"],
            f"eval_{self.metric_prefix}_loss_rate": result["loss_rate"],
            f"eval_{self.metric_prefix}_wins": result["wins"],
            f"eval_{self.metric_prefix}_losses": result["losses"],
            f"eval_{self.metric_prefix}_draws": result["draws"],
        }


@dataclass(frozen=True)
class NearestJaxEvalOpponent:
    """Nearest-rule opponent evaluated entirely inside JAX fast_env."""

    name: str = "nearest"
    metric_prefix: str = "nearest"

    def exists(self) -> bool:
        return True

    def metric_fields(self) -> list[str]:
        return [
            f"eval_{self.metric_prefix}_win_rate",
            f"eval_{self.metric_prefix}_loss_rate",
            f"eval_{self.metric_prefix}_wins",
            f"eval_{self.metric_prefix}_losses",
            f"eval_{self.metric_prefix}_draws",
        ]

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return {
            f"eval_{self.metric_prefix}_win_rate": result["win_rate"],
            f"eval_{self.metric_prefix}_loss_rate": result["loss_rate"],
            f"eval_{self.metric_prefix}_wins": result["wins"],
            f"eval_{self.metric_prefix}_losses": result["losses"],
            f"eval_{self.metric_prefix}_draws": result["draws"],
        }

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        rng, reset_key = jax.random.split(ctx.rng)
        state = ctx.reset_from_bank(ctx.bank, reset_key, ctx.num_games)

        def step_fn(st, _):
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
        wins = int((winners == 0).sum())
        losses = int((winners == 1).sum())
        draws = int((winners == 2).sum())
        total = max(ctx.num_games, 1)
        result = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total, "loss_rate": losses / total}
        print(f"[eval/{self.name}]   wins={wins} losses={losses} draws={draws} win_rate={result['win_rate']:.3f}")
        return result


@dataclass(frozen=True)
class ExternalScriptJaxEvalOpponent:
    """Orbit Wars Lab external script opponent against the current JAX learner."""

    name: str
    metric_prefix: str
    agent_dir: Path
    seed_offset: int = 0

    def exists(self) -> bool:
        return (self.agent_dir / "main.py").exists()

    def metric_fields(self) -> list[str]:
        return [
            f"eval_{self.metric_prefix}_win_rate",
            f"eval_{self.metric_prefix}_loss_rate",
            f"eval_{self.metric_prefix}_wins",
            f"eval_{self.metric_prefix}_losses",
            f"eval_{self.metric_prefix}_draws",
        ]

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return {
            f"eval_{self.metric_prefix}_win_rate": result["win_rate"],
            f"eval_{self.metric_prefix}_loss_rate": result["loss_rate"],
            f"eval_{self.metric_prefix}_wins": result["wins"],
            f"eval_{self.metric_prefix}_losses": result["losses"],
            f"eval_{self.metric_prefix}_draws": result["draws"],
        }

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
            obs = states[0]["observation"]
            opp_obs = states[1]["observation"]
            done = states[0]["status"] != "ACTIVE"
            while not done:
                jax_state = ctx.kaggle_obs_to_jax_state(obs, ctx.env_params)
                learner_action = ctx.jax_actions_to_kaggle(infer(jax_state))
                opp_action = opponent.act_for_env(0, opp_obs)
                load_error = getattr(opponent.modules.get(0), "_LOAD_ERROR", None)
                if load_error is not None:
                    raise RuntimeError(f"External eval opponent {self.name} failed to load: {load_error}") from load_error
                states = env.step([learner_action, opp_action])
                obs = states[0]["observation"]
                opp_obs = states[1]["observation"]
                done = states[0]["status"] != "ACTIVE"

            reward = states[0].get("reward", 0.0) or 0.0
            if reward > 0:
                wins += 1
                outcome = "win"
            elif reward < 0:
                losses += 1
                outcome = "loss"
            else:
                draws += 1
                outcome = "draw"
            print(f"  [eval/{self.name}] game={game_idx + 1} seed={seed} {outcome} reward={reward:.1f}")

        total = max(ctx.num_games, 1)
        result = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total, "loss_rate": losses / total}
        print(f"[eval/{self.metric_prefix}]     wins={wins} losses={losses} draws={draws} win_rate={result['win_rate']:.3f}")
        return result


@dataclass(frozen=True)
class CheckpointPolicyJaxEvalOpponent:
    """Fixed JAX checkpoint policy as an eval opponent in fast_env."""

    name: str
    metric_prefix: str
    params: Any
    normalizer_state: Any
    model: Any
    feat_params: Any

    def exists(self) -> bool:
        return self.params is not None

    def metric_fields(self) -> list[str]:
        return [
            f"eval_{self.metric_prefix}_win_rate",
            f"eval_{self.metric_prefix}_loss_rate",
            f"eval_{self.metric_prefix}_wins",
            f"eval_{self.metric_prefix}_losses",
            f"eval_{self.metric_prefix}_draws",
        ]

    def metric_values(self, result: EvalResult) -> dict[str, float | int]:
        return {
            f"eval_{self.metric_prefix}_win_rate": result["win_rate"],
            f"eval_{self.metric_prefix}_loss_rate": result["loss_rate"],
            f"eval_{self.metric_prefix}_wins": result["wins"],
            f"eval_{self.metric_prefix}_losses": result["losses"],
            f"eval_{self.metric_prefix}_draws": result["draws"],
        }

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        rng, reset_key = jax.random.split(ctx.rng)
        state = ctx.reset_from_bank(ctx.bank, reset_key, ctx.num_games)

        def step_fn(st, _):
            learner_acts = ctx.deterministic_policy_actions(
                ctx.train_state,
                st,
                ctx.feat_params,
                ctx.normalizer_state,
                ctx.norm_params,
            )
            flipped = ctx.flip_state(st)
            opp_acts = ctx.deterministic_policy_actions_params(
                self.params,
                self.model.apply,
                flipped,
                self.feat_params,
                self.normalizer_state,
                ctx.norm_params,
            )
            joint = jnp.stack([learner_acts, opp_acts], axis=1)
            return step_batch(st, joint, ctx.env_params), None

        final_state, _ = jax.lax.scan(step_fn, state, None, length=ctx.env_params.episode_steps)
        winners = np.array(final_state.winner)
        wins = int((winners == 0).sum())
        losses = int((winners == 1).sum())
        draws = int((winners == 2).sum())
        total = max(ctx.num_games, 1)
        result = {"wins": wins, "losses": losses, "draws": draws, "win_rate": wins / total, "loss_rate": losses / total}
        print(f"[eval/{self.name}]   wins={wins} losses={losses} draws={draws} win_rate={result['win_rate']:.3f}")
        return result


def default_eval_opponents(root: Path, *, checkpoint_opponent=None, enabled_names=None) -> list[JaxEvalOpponent]:
    """Default benchmark set for JAX training."""

    enabled = set(enabled_names or ["nearest", "sigma", "tamrazov"])
    external_root = root / "orbit-wars-lab" / "agents" / "external"
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
            agent_dir=root / "orbit-wars-lab" / "agents" / "mine" / "jax-pool-0514-resume-ckpt900",
            seed_offset=31337,
        ))
    if checkpoint_opponent is not None and ("checkpoint400" in enabled or "checkpoint" in enabled):
        opponents.append(
            CheckpointPolicyJaxEvalOpponent(
                name="checkpoint400",
                metric_prefix="checkpoint400",
                params=checkpoint_opponent.params,
                normalizer_state=checkpoint_opponent.normalizer_state,
                model=checkpoint_opponent.model,
                feat_params=checkpoint_opponent.feat_params,
            )
        )
    return opponents


def eval_metric_fields(opponents: list[JaxEvalOpponent]) -> list[str]:
    fields: list[str] = []
    for opponent in opponents:
        fields.extend(opponent.metric_fields())
    return fields
