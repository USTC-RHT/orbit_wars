#!/usr/bin/env python3
"""Smoke-test torch-backed external eval opponent dependencies.

This checks the failure path that appears during JAX eval:
1. torch can be imported and used for a basic CPU tensor op.
2. agents/external/slawekbiel-producer/main.py can be dynamically imported.
3. Optionally, the agent can be called once from a Kaggle Orbit Wars obs.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "agents" / "external" / "slawekbiel-producer" / "main.py").exists():
            return candidate
    return script_path.parents[1]


def pass_line(message: str) -> None:
    print(f"PASS {message}")


def info_line(message: str) -> None:
    print(f"INFO {message}")


def fail_line(message: str) -> None:
    print(f"FAIL {message}", file=sys.stderr)


def check_torch(*, require_cuda: bool) -> None:
    import torch

    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    result = float((values * values).sum().item())
    if result != 5.0:
        raise RuntimeError(f"unexpected torch tensor result: {result}")

    pass_line(f"torch import ok: version={torch.__version__}")
    info_line(f"torch file: {Path(torch.__file__).resolve()}")
    info_line(f"torch compiled CUDA: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    info_line(f"torch cuda available: {cuda_available}")
    if require_cuda and not cuda_available:
        raise RuntimeError(
            "torch imported, but CUDA is not available. If the log says the "
            "NVIDIA driver is too old, reinstall a torch wheel compiled for "
            "your driver CUDA version, for example torch==2.6.0+cu124."
        )
    if require_cuda:
        device = torch.device("cuda")
        cuda_values = values.to(device)
        cuda_result = float((cuda_values * cuda_values).sum().cpu().item())
        if cuda_result != 5.0:
            raise RuntimeError(f"unexpected CUDA tensor result: {cuda_result}")
        pass_line(f"torch CUDA op ok: device={torch.cuda.get_device_name(0)}")


def load_external_agent(agent_dir: Path) -> tuple[Any, bool, Path]:
    main_py = agent_dir / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"external agent entry not found: {main_py}")

    module_name = f"_check_eval_deps_{agent_dir.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create module spec for {main_py}")

    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.modules[module_name] = module
        sys.path.insert(0, str(agent_dir))
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path = old_path

    agent = getattr(module, "agent", None)
    if agent is None or not callable(agent):
        raise AttributeError(f"external opponent has no callable agent(obs): {main_py}")

    try:
        call_with_config = len(inspect.signature(agent).parameters) >= 2
    except (TypeError, ValueError):
        call_with_config = False

    pass_line(f"external agent import ok: {main_py}")
    info_line(f"agent callable accepts config: {call_with_config}")
    return agent, call_with_config, main_py


def kaggle_smoke_call(agent: Any, call_with_config: bool, main_py: Path, seed: int, player: int) -> None:
    from kaggle_environments import make as kag_make

    env = kag_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])
    obs = states[player]["observation"]

    if call_with_config:
        moves = agent(obs, {"__raw_path__": str(main_py)})
    else:
        moves = agent(obs)

    if moves is None:
        moves = []
    if not isinstance(moves, (list, tuple)):
        raise TypeError(f"agent returned {type(moves).__name__}, expected list/tuple")

    pass_line(f"kaggle smoke call ok: player={player} seed={seed} moves={len(moves)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check torch and torch-backed external eval opponent imports."
    )
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=repo_root() / "agents" / "external" / "slawekbiel-producer",
        help="External agent directory containing main.py.",
    )
    parser.add_argument(
        "--with-kaggle",
        action="store_true",
        help="Also create a Kaggle Orbit Wars env and call the external agent once.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if torch imports but CUDA is not available.",
    )
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--player", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agent_dir = args.agent_dir.resolve()
    info_line(f"python: {sys.executable}")
    info_line(f"agent dir: {agent_dir}")

    try:
        check_torch(require_cuda=args.require_cuda)
        agent, call_with_config, main_py = load_external_agent(agent_dir)
        if args.with_kaggle:
            kaggle_smoke_call(agent, call_with_config, main_py, args.seed, args.player)
    except Exception as exc:
        fail_line(str(exc))
        traceback.print_exc()
        print()
        print("Install/update the eval torch dependency with:")
        print("  python -m pip uninstall -y torch")
        print("  python -m pip install -r requirements.eval-torch-cu124.txt")
        print()
        print("For the CUDA 12.4 wheel only:")
        print("  python -m pip install --force-reinstall torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124")
        return 1

    pass_line("torch-backed eval dependencies look usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
