"""训练曲线绘图脚本。

读取 `metrics.csv`，画出 loss、reward、entropy、KL、clipfrac 和
explained_variance，方便判断 PPO 训练是否稳定。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """解析画图参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="RL/artifacts/sb3_style_nearest_1000/metrics.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """绘制 loss、reward、entropy 以及 PPO 诊断曲线。"""

    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out) if args.out else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(df["update"], df["loss"], alpha=0.35, label="loss")
    axes[0].plot(df["update"], df["loss"].rolling(args.window, min_periods=1).mean(), label=f"loss MA{args.window}")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["update"], df["episode_reward_mean"], alpha=0.35, label="episode reward")
    axes[1].plot(
        df["update"],
        df["episode_reward_mean"].rolling(args.window, min_periods=1).mean(),
        label=f"reward MA{args.window}",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(df["update"], df["entropy"], label="entropy")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    if {"approx_kl", "clipfrac", "explained_variance"}.issubset(df.columns):
        axes[3].plot(df["update"], df["approx_kl"], label="approx_kl")
        axes[3].plot(df["update"], df["clipfrac"], label="clipfrac")
        axes[3].plot(df["update"], df["explained_variance"], label="explained_variance")
    else:
        axes[3].text(0.5, 0.5, "No PPO diagnostic columns in CSV", ha="center")
    axes[3].legend()
    axes[3].grid(alpha=0.25)
    axes[3].set_xlabel("update")

    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
