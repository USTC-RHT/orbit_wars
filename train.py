"""训练入口。

负责解析命令行参数、读取 YAML 配置，并把最终配置交给 `src.train_loop.train`。
真正的环境交互、rollout 收集和 PPO 更新都在 train_loop.py 里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.config import load_config  # noqa: E402
from src.train_loop import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "default_cfg.yaml"))
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--opponent", default=None)
    parser.add_argument("--wandb", action="store_true", help="启用 Weights & Biases 记录训练曲线")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--resume", default=None, help="从指定 .pt checkpoint 文件继续训练")
    parser.add_argument("--use-CNN", "--use_CNN", "-use_CNN", dest="use_cnn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.updates is not None:
        cfg.ppo.total_updates = args.updates
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.rollout_steps is not None:
        cfg.ppo.rollout_steps = args.rollout_steps
    if args.device is not None:
        cfg.device = args.device
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.opponent is not None:
        cfg.opponent = args.opponent
    if args.wandb:
        cfg.wandb.enabled = True
    if args.wandb_project is not None:
        cfg.wandb.project = args.wandb_project
    if args.wandb_entity is not None:
        cfg.wandb.entity = args.wandb_entity
    if args.wandb_mode is not None:
        cfg.wandb.mode = args.wandb_mode
    if args.use_cnn:
        cfg.model.use_cnn = True
    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
