"""用打包 Kaggle agent 对 feature-row BC policy 做 smoke/eval。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .bc_policy_runtime import BCPolicyAgent
from ..evaluation.eval_opponents.opponents import ExternalScriptOpponent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None, help="BC ckpt_best.pkl/ckpt_last.pkl；省略则跑随机初始化 smoke。")
    parser.add_argument("--config", type=Path, default=Path("cfg_jax_BC_1.yaml"), help="只用于随机 smoke 的默认配置。")
    parser.add_argument("--opponent", default="slawekbiel-producer", help="agent 目录，或 agents/external、agents/mine 下的名称。")
    parser.add_argument("--agents-root", type=Path, default=Path("agents"))
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--bc-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max-fleets", type=int, default=512)
    return parser.parse_args()


def load_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_agent_dir(root: Path, name: str) -> Path:
    direct = Path(name)
    if (direct / "main.py").exists():
        return direct
    for subdir in ("external", "mine", "baselines"):
        candidate = root / subdir / name
        if (candidate / "main.py").exists():
            return candidate
    raise FileNotFoundError(f"could not resolve opponent agent: {name}")


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None:
        agent = BCPolicyAgent.from_checkpoint(args.checkpoint, max_fleets=int(args.max_fleets))
        agent_label = str(args.checkpoint)
    else:
        agent = BCPolicyAgent.random_from_config(load_raw_config(args.config), seed=int(args.seed), max_fleets=int(args.max_fleets))
        agent_label = "random-init BC smoke"

    try:
        from kaggle_environments import make as kaggle_make
    except ImportError as exc:
        raise RuntimeError("kaggle_environments is required for online BC eval") from exc

    opponent_dir = resolve_agent_dir(args.agents_root, str(args.opponent))
    opponent = ExternalScriptOpponent(opponent_dir)
    bc_idx = int(args.bc_player)
    opp_idx = 1 - bc_idx
    wins = losses = draws = 0
    start = time.time()
    print(f"BC eval start policy={agent_label} opponent={opponent_dir} games={args.games} bc_player={bc_idx}", flush=True)

    for game_idx in range(int(args.games)):
        seed = int(args.seed) + game_idx
        opponent.reset_env(game_idx, seed)
        env = kaggle_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        while states[0]["status"] == "ACTIVE":
            bc_action = agent.act(states[bc_idx]["observation"], player=bc_idx)
            opp_action = opponent.act_for_env(game_idx, states[opp_idx]["observation"])
            joint = [None, None]
            joint[bc_idx] = bc_action
            joint[opp_idx] = opp_action
            states = env.step(joint)

        reward = states[bc_idx].get("reward", 0.0) or 0.0
        if reward > 0:
            wins += 1
            outcome = "win"
        elif reward < 0:
            losses += 1
            outcome = "loss"
        else:
            draws += 1
            outcome = "draw"
        print(
            f"game={game_idx + 1} seed={seed} outcome={outcome} reward={reward:.1f} "
            f"turns={len(env.steps) if hasattr(env, 'steps') else 'unknown'}",
            flush=True,
        )

    total = max(int(args.games), 1)
    print(
        f"BC eval done wins={wins} losses={losses} draws={draws} "
        f"win_rate={wins / total:.3f} elapsed={time.time() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
