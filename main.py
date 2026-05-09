"""Kaggle/web 对战入口。

Kaggle 或 orbit-wars-lab Web 端会调用 `agent(observation)`。本文件负责
懒加载配置和 checkpoint，把 observation 编码后交给神经网络生成舰队动作。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(globals().get("__file__", ".")).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.config import load_config  # noqa: E402
from src.features import CANDIDATE_DIM, GLOBAL_DIM, SELF_DIM, encode_turn, target_count  # noqa: E402
from src.model import ActorCritic  # noqa: E402
from src.normalization import StateNormalizer  # noqa: E402
from src.opponents import actions_from_policy  # noqa: E402

_CFG = None
_MODEL = None
_DEVICE = None
_STATE_NORMALIZER = None


def _checkpoint_candidates() -> list[Path]:
    cfg = load_config(_HERE / "default_cfg.yaml")
    candidates: list[Path] = []
    local = _HERE / "artifacts" / cfg.run_name / "ckpt_last.pt"
    if local.exists():
        candidates.append(local)
    artifact_root = _HERE / "artifacts"
    if artifact_root.exists():
        candidates.extend(
            path
            for path in sorted(
                artifact_root.glob("*/ckpt_last.pt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if path not in candidates
        )
    weights = _HERE / "weights" / "ckpt_last.pt"
    if weights.exists():
        candidates.append(weights)
    return candidates


def _load():
    import torch

    cfg = load_config(_HERE / "default_cfg.yaml")
    device = torch.device("cpu")
    model = ActorCritic(
        self_dim=SELF_DIM,
        candidate_dim=CANDIDATE_DIM,
        global_dim=GLOBAL_DIM,
        candidate_count=target_count(cfg.env),
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
    ).to(device)
    errors: list[str] = []
    for path in _checkpoint_candidates():
        ckpt = torch.load(path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt.get("model", ckpt))
            state_normalizer = StateNormalizer(
                enabled=cfg.state_norm.enabled,
                clip=cfg.state_norm.clip,
                eps=cfg.state_norm.eps,
            )
            if isinstance(ckpt, dict) and "state_normalizer" in ckpt:
                state_normalizer.load_state_dict(ckpt["state_normalizer"])
            else:
                state_normalizer.enabled = False
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        model.eval()
        print(f"Loaded checkpoint {path}", flush=True)
        return cfg, model, device, state_normalizer
    searched = ", ".join(str(path) for path in _checkpoint_candidates())
    raise RuntimeError(f"No compatible checkpoint found. searched=[{searched}] errors={errors}")


def agent(obs, config=None):
    global _CFG, _MODEL, _DEVICE, _STATE_NORMALIZER
    if _MODEL is None:
        _CFG, _MODEL, _DEVICE, _STATE_NORMALIZER = _load()
    batch = encode_turn(obs, _CFG.env)
    return actions_from_policy(
        batch,
        _MODEL,
        _DEVICE,
        deterministic=True,
        state_normalizer=_STATE_NORMALIZER,
    )


__all__ = ["agent"]
