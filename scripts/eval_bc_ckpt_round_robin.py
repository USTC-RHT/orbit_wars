from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kaggle_environments import make as kaggle_make

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jax_agent.bc_policy_runtime import BCPolicyAgent


def parse_candidate(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = Path(path).expanduser()
    if not name:
        raise argparse.ArgumentTypeError("candidate name is empty")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--games-per-pair", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1240000)
    parser.add_argument("--max-fleets", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/orbit_eval_ckpts_0624"))
    return parser.parse_args()


def run_game(agent_a, agent_b, name_a: str, name_b: str, seed: int, a_player: int) -> dict:
    env = kaggle_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])
    while states[0]["status"] == "ACTIVE":
        joint = [None, None]
        joint[a_player] = agent_a.act(states[a_player]["observation"], player=a_player)
        joint[1 - a_player] = agent_b.act(states[1 - a_player]["observation"], player=1 - a_player)
        states = env.step(joint)

    reward_a = float(states[a_player].get("reward", 0.0) or 0.0)
    outcome_a = "win" if reward_a > 0 else "loss" if reward_a < 0 else "draw"
    return {
        "a": name_a,
        "b": name_b,
        "seed": seed,
        "a_player": a_player,
        "reward_a": reward_a,
        "outcome_a": outcome_a,
        "turns": len(env.steps) if hasattr(env, "steps") else None,
    }


def summarize(rows: list[dict], candidates: list[tuple[str, Path]]) -> dict:
    pair = {}
    standings = {
        name: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "p0_games": 0,
            "p0_wins": 0,
            "p1_games": 0,
            "p1_wins": 0,
            "turns": [],
        }
        for name, _ in candidates
    }

    for row in rows:
        key = f"{row['a']}__vs__{row['b']}"
        pair_stats = pair.setdefault(
            key,
            {"a": row["a"], "b": row["b"], "games": 0, "a_wins": 0, "b_wins": 0, "draws": 0, "turns": []},
        )
        pair_stats["games"] += 1
        pair_stats["turns"].append(row.get("turns"))

        a = standings[row["a"]]
        b = standings[row["b"]]
        a["games"] += 1
        b["games"] += 1
        a["turns"].append(row.get("turns"))
        b["turns"].append(row.get("turns"))
        if row["a_player"] == 0:
            a["p0_games"] += 1
            b["p1_games"] += 1
        else:
            a["p1_games"] += 1
            b["p0_games"] += 1

        if row["outcome_a"] == "win":
            pair_stats["a_wins"] += 1
            a["wins"] += 1
            b["losses"] += 1
            if row["a_player"] == 0:
                a["p0_wins"] += 1
            else:
                a["p1_wins"] += 1
        elif row["outcome_a"] == "loss":
            pair_stats["b_wins"] += 1
            b["wins"] += 1
            a["losses"] += 1
            if row["a_player"] == 0:
                b["p1_wins"] += 1
            else:
                b["p0_wins"] += 1
        else:
            pair_stats["draws"] += 1
            a["draws"] += 1
            b["draws"] += 1

    for stats in pair.values():
        turns = [t for t in stats.pop("turns", []) if t is not None]
        stats["avg_turns"] = sum(turns) / len(turns) if turns else None
    for stats in standings.values():
        turns = [t for t in stats.pop("turns", []) if t is not None]
        stats["avg_turns"] = sum(turns) / len(turns) if turns else None
        stats["win_rate"] = stats["wins"] / max(stats["games"], 1)
        stats["net_wins"] = stats["wins"] - stats["losses"]

    ranking = sorted(
        (
            {"agent": name, **stats}
            for name, stats in standings.items()
        ),
        key=lambda item: (item["wins"], item["net_wins"], item["win_rate"]),
        reverse=True,
    )
    return {"pairs": pair, "standings": standings, "ranking": ranking, "rows": len(rows), "updated_at": time.time()}


def main() -> None:
    args = parse_args()
    candidates = list(args.candidate)
    if len(candidates) < 2:
        raise SystemExit("need at least two candidates")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "h2h_round_robin_10g.jsonl"
    summary_path = args.out_dir / "h2h_round_robin_10g_summary.json"
    out_path.write_text("", encoding="utf-8")

    print("loading agents...", flush=True)
    agents = {}
    for name, path in candidates:
        print(f"load {name}: {path}", flush=True)
        agents[name] = BCPolicyAgent.from_checkpoint(path, max_fleets=int(args.max_fleets))

    rows: list[dict] = []
    games_per_pair = int(args.games_per_pair)
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            name_a = candidates[i][0]
            name_b = candidates[j][0]
            for game_idx in range(games_per_pair):
                seed = int(args.base_seed) + i * 100000 + j * 1000 + game_idx
                a_player = game_idx % 2
                row = run_game(agents[name_a], agents[name_b], name_a, name_b, seed, a_player)
                rows.append(row)
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                summary_path.write_text(json.dumps(summarize(rows, candidates), indent=2, sort_keys=True), encoding="utf-8")
                print(
                    "H2H",
                    name_a,
                    "vs",
                    name_b,
                    f"{game_idx + 1}/{games_per_pair}",
                    row["outcome_a"],
                    "reward",
                    row["reward_a"],
                    "a_player",
                    a_player,
                    "turns",
                    row["turns"],
                    flush=True,
                )

    summary_path.write_text(json.dumps(summarize(rows, candidates), indent=2, sort_keys=True), encoding="utf-8")
    print("done", summary_path, flush=True)
    print(summary_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
