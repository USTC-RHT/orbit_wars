"""测特征工程速度。

用法（在 kaggle/ 根目录执行）：
    python RL/bench_inbound.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # /home/dministrator/kaggle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    target_dir = ROOT / "RL"
    sys.path.insert(0, str(target_dir))

    from src.config import EnvConfig, TrainConfig  # noqa: E402
    from src.env_wrapper import OrbitWarsPPOEnv  # noqa: E402
    from src.opponents import RandomOpponent  # noqa: E402
    from src import features as feat  # noqa: E402
    from src.state import parse_observation  # noqa: E402

    print(f"=== RL | dir={target_dir.name} ===")

    cfg = TrainConfig()
    cfg.opponent = "random"
    env = OrbitWarsPPOEnv(cfg, RandomOpponent(), env_index=0)
    env.reset(0)
    states = []
    rng = np.random.default_rng(0)
    rand_op = RandomOpponent()
    obs = env.last_obs
    for _ in range(args.n):
        states.append(obs)
        try:
            env.step(rand_op.act(obs))
            obs = env.last_obs
            if obs is None:
                obs = env.reset(int(rng.integers(0, 1_000_000)))
                obs = env.last_obs
        except Exception:
            env.reset(int(rng.integers(0, 1_000_000)))
            obs = env.last_obs
    print(f"  采集 {len(states)} 个 obs")
    cfg_env = EnvConfig()

    # warmup
    for s in states[:10]:
        feat.encode_turn(s, cfg_env)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for s in states:
            feat.encode_turn(s, cfg_env)
        times.append(time.perf_counter() - t0)
    best = min(times)
    per = best / len(states) * 1000
    print(f"  encode_turn:        per_obs={per:.3f}ms  total={best*1000:.1f}ms")

    if hasattr(feat, "precompute_inbound"):
        parsed = [parse_observation(s) for s in states]
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            for s in parsed:
                feat.precompute_inbound(s)
            times.append(time.perf_counter() - t0)
        best_in = min(times)
        per_in = best_in / len(states) * 1000
        print(f"  precompute_inbound: per_obs={per_in:.3f}ms  total={best_in*1000:.1f}ms")

    obs_per_update = 8 * 128
    print(f"  -> 估算每个 update 仅特征耗时: {per*obs_per_update:.0f}ms ({per*obs_per_update/1000:.2f}s)")


if __name__ == "__main__":
    main()
