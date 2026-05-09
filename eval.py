"""评估入口。

加载训练好的 checkpoint，让策略用确定性动作和 nearest/random 对手打若干局，
输出胜负统计。这个脚本不训练，只用于看模型当前实际战斗表现。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.config import load_config  # noqa: E402
from src.features import (  # noqa: E402
    CANDIDATE_DIM,
    GLOBAL_DIM,
    MAP_CHANNELS,
    MAP_GRID_SIZE,
    SELF_DIM,
    encode_turn,
    target_count,
)
from src.model import ActorCritic  # noqa: E402
from src.normalization import StateNormalizer  # noqa: E402
from src.opponents import NearestOpponent, RandomOpponent, actions_from_policy  # noqa: E402
from src.state import read  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "default_cfg.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--opponent", choices=["nearest", "random"], default="nearest")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = ActorCritic(
        self_dim=SELF_DIM,
        candidate_dim=CANDIDATE_DIM,
        global_dim=GLOBAL_DIM,
        candidate_count=target_count(cfg.env),
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        use_cnn=cfg.model.use_cnn,
        map_channels=MAP_CHANNELS,
        map_grid_size=MAP_GRID_SIZE,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    state_normalizer = StateNormalizer(
        enabled=cfg.state_norm.enabled,
        clip=cfg.state_norm.clip,
        eps=cfg.state_norm.eps,
    )
    if isinstance(ckpt, dict) and "state_normalizer" in ckpt:
        state_normalizer.load_state_dict(ckpt["state_normalizer"])
    else:
        state_normalizer.enabled = False
    opponent = NearestOpponent() if args.opponent == "nearest" else RandomOpponent()

    wins = losses = draws = 0
    for i in range(args.games):
        reward, steps = play_game(cfg, model, device, opponent, state_normalizer, seed=args.seed + i)
        if reward > 0:
            wins += 1
            label = "win"
        elif reward < 0:
            losses += 1
            label = "loss"
        else:
            draws += 1
            label = "draw"
        print(f"game={i + 1} seed={args.seed + i} result={label} reward={reward:.1f} steps={steps}")
    print(f"summary wins={wins} losses={losses} draws={draws} games={args.games}")
    print(f"win_rate={wins / max(args.games, 1):.4f}")


def play_game(cfg, model, device, opponent, state_normalizer: StateNormalizer, seed: int) -> tuple[float, int]:
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])
    obs = read(states[0], "observation")
    opp_obs = read(states[1], "observation")
    done = read(states[0], "status") != "ACTIVE"
    steps = 0
    while not done:
        batch = encode_turn(obs, cfg.env)
        action = actions_from_policy(
            batch,
            model,
            device,
            deterministic=True,
            state_normalizer=state_normalizer,
        )
        opp_action = opponent.act(opp_obs)
        states = env.step([action, opp_action])
        obs = read(states[0], "observation")
        opp_obs = read(states[1], "observation")
        done = read(states[0], "status") != "ACTIVE"
        steps += 1
    reward = read(states[0], "reward", 0.0)
    return (0.0 if reward is None else float(reward)), steps


if __name__ == "__main__":
    main()
