"""把 feature-row BC checkpoint 导出为 ONNX 策略网络。

导出的图直接接收原始 feature-row 张量，内部固化 checkpoint 保存的 normalizer，
输出 fire/target/amount logits。动作组装仍保留在 Python runtime 中，因为它依赖
Orbit Wars observation 解析、视角映射和瞄准逻辑。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax2onnx
import numpy as np
import onnx

from ..common.model_jax import FeatureRowBCPolicy
from ..bc.train_bc_jax import (
    DEFAULT_DROPPED_RELATIVE_FEATURE_DIMS,
    FeatureRowBatch,
    identity_normalizer,
    jax_normalizer,
)


# 普通两阶段图：外部传入 selected_targets，amount 头按该目标出 logits。
INPUT_NAMES = (
    "planet",
    "relative",
    "global_features",
    "candidate_mask",
    "map_features",
    "selected_targets",
)
# 输出名会写入 meta.json，提交 runtime 按这些名字取结果。
OUTPUT_NAMES = ("fire_logits", "target_logits", "amount_logits")
# greedy 单图：ONNX 内部先 argmax target，再用该目标计算 amount。
GREEDY_INPUT_NAMES = (
    "planet",
    "relative",
    "global_features",
    "candidate_mask",
    "map_features",
)
GREEDY_OUTPUT_NAMES = ("fire_logits", "target_logits", "amount_logits", "greedy_target")


def _raw_get(raw: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = raw.get(section, {}).get(key, default)
    return default if value is None else value


def _shape_dtype(shape: tuple[int, ...], dtype) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype)


def normalize_batch_for_onnx(batch: FeatureRowBatch, norm: dict[str, jnp.ndarray]) -> FeatureRowBatch:
    """用 jax2onnx 更稳定的方式归一化 feature-row。

    禁用的 relative 维度在 Python ONNX runtime 侧先置零，再送入图中。
    这里避免 indexed update，是因为它在 jax2onnx 转换时更容易出兼容问题。
    """

    planet = (batch.planet - norm["planet_mean"]) / jnp.maximum(norm["planet_std"], 1e-6)
    relative = (batch.relative - norm["relative_mean"]) / jnp.maximum(norm["relative_std"], 1e-6)
    relative = jnp.where(batch.candidate_mask[..., None], relative, 0.0)
    global_features = (batch.global_features - norm["global_mean"]) / jnp.maximum(norm["global_std"], 1e-6)
    map_features = (batch.map_features - norm["map_mean"]) / jnp.maximum(norm["map_std"], 1e-6)
    return batch._replace(planet=planet, relative=relative, global_features=global_features, map_features=map_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None, help="模型目录名；默认使用 checkpoint 文件名。")
    parser.add_argument("--opset", type=int, default=23)
    parser.add_argument("--map-grid-size", type=int, default=8)
    parser.add_argument("--map-channels", type=int, default=6)
    parser.add_argument("--planet-dim", type=int, default=28)
    parser.add_argument("--relative-dim", type=int, default=22)
    parser.add_argument("--global-dim", type=int, default=11)
    parser.add_argument(
        "--greedy-single-pass",
        action="store_true",
        help="导出确定性单图推理：在 ONNX 内部 argmax target，并一次返回 amount logits。",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a dict: {path}")
    if "params" not in payload:
        raise KeyError(f"checkpoint is missing params: {path}")
    return payload


def export_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    name: str | None = None,
    opset: int = 23,
    map_grid_size: int = 8,
    map_channels: int = 6,
    planet_dim: int = 28,
    relative_dim: int = 22,
    global_dim: int = 11,
    greedy_single_pass: bool = False,
) -> Path:
    payload = load_payload(checkpoint)
    raw = payload.get("raw_config") or payload.get("config") or {}
    dropped_relative_feature_dims = _raw_get(
        raw,
        "bc",
        "dropped_relative_feature_dims",
        DEFAULT_DROPPED_RELATIVE_FEATURE_DIMS,
    )
    if dropped_relative_feature_dims is None:
        dropped_relative_feature_dims = ()
    max_planets = int(_raw_get(raw, "env", "max_planets", 45))
    candidate_count = max_planets + 1
    ship_bucket_count = int(_raw_get(raw, "env", "ship_bucket_count", 8))
    hidden_size = int(_raw_get(raw, "model", "hidden_size", 128))
    use_cnn = bool(_raw_get(raw, "model", "use_cnn", True))
    target_head = str(_raw_get(raw, "model", "target_head", "dot"))
    target_num_heads = int(_raw_get(raw, "model", "target_num_heads", 8))

    model = FeatureRowBCPolicy(
        ship_bucket_count=ship_bucket_count,
        hidden_size=hidden_size,
        use_cnn=use_cnn,
        target_head=target_head,
        target_num_heads=target_num_heads,
    )
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    normalizer = jax_normalizer(payload.get("normalizer") or identity_normalizer())

    def make_batch(
        planet: jnp.ndarray,
        relative: jnp.ndarray,
        global_features: jnp.ndarray,
        candidate_mask: jnp.ndarray,
        map_features: jnp.ndarray,
    ) -> FeatureRowBatch:
        # ONNX 前向只需要 feature；label 字段用 dummy 填充以复用 FeatureRowBatch contract。
        rows = planet.shape[0]
        dummy_bool = jnp.zeros((rows,), dtype=jnp.bool_)
        dummy_i32 = jnp.zeros((rows,), dtype=jnp.int32)
        dummy_amount_mask = jnp.zeros(
            (rows, relative.shape[1], ship_bucket_count),
            dtype=jnp.bool_,
        )
        batch = FeatureRowBatch(
            planet=planet,
            relative=relative,
            global_features=global_features,
            map_features=map_features,
            candidate_mask=candidate_mask,
            amount_mask=dummy_amount_mask,
            fire=dummy_bool,
            target=dummy_i32,
            amount=dummy_i32,
        )
        return normalize_batch_for_onnx(batch, normalizer)

    def forward(
        planet: jnp.ndarray,
        relative: jnp.ndarray,
        global_features: jnp.ndarray,
        candidate_mask: jnp.ndarray,
        map_features: jnp.ndarray,
        selected_targets: jnp.ndarray,
    ):
        batch = make_batch(planet, relative, global_features, candidate_mask, map_features)
        out = model.apply(
            {"params": params},
            batch.planet,
            batch.relative,
            batch.global_features,
            batch.candidate_mask,
            batch.map_features,
            selected_targets,
        )
        return out.fire_logits, out.target_logits, out.amount_logits

    def greedy_forward(
        planet: jnp.ndarray,
        relative: jnp.ndarray,
        global_features: jnp.ndarray,
        candidate_mask: jnp.ndarray,
        map_features: jnp.ndarray,
    ):
        batch = make_batch(planet, relative, global_features, candidate_mask, map_features)
        return model.apply(
            {"params": params},
            batch.planet,
            batch.relative,
            batch.global_features,
            batch.candidate_mask,
            batch.map_features,
            method=model.greedy_single_pass,
        )

    model_name = name or checkpoint.stem
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / "bc_policy.onnx"
    data_path = model_dir / "bc_policy.onnx.data"
    if onnx_path.exists():
        onnx_path.unlink()
    if data_path.exists():
        data_path.unlink()

    inputs = [
        _shape_dtype((max_planets, planet_dim), jnp.float32),
        _shape_dtype((max_planets, candidate_count, relative_dim), jnp.float32),
        _shape_dtype((max_planets, global_dim), jnp.float32),
        _shape_dtype((max_planets, candidate_count), jnp.bool_),
        _shape_dtype((max_planets, map_channels, map_grid_size, map_grid_size), jnp.float32),
        _shape_dtype((max_planets,), jnp.int32),
    ]
    export_inputs = inputs[:-1] if greedy_single_pass else inputs
    export_input_names = GREEDY_INPUT_NAMES if greedy_single_pass else INPUT_NAMES
    export_output_names = GREEDY_OUTPUT_NAMES if greedy_single_pass else OUTPUT_NAMES
    jax2onnx.to_onnx(
        greedy_forward if greedy_single_pass else forward,
        export_inputs,
        input_names=export_input_names,
        output_names=export_output_names,
        model_name="feature_row_bc_policy",
        opset=int(opset),
        return_mode="file",
        output_path=onnx_path,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    normalizer_np = payload.get("normalizer") or identity_normalizer()
    np.savez(model_dir / "normalizer.npz", **normalizer_np)
    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        "checkpoint_type": payload.get("checkpoint_type"),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "onnx_file": onnx_path.name,
        "external_data_file": data_path.name if data_path.exists() else None,
        "opset": int(opset),
        "input_names": list(export_input_names),
        "output_names": list(export_output_names),
        "greedy_single_pass": bool(greedy_single_pass),
        "max_planets": max_planets,
        "candidate_count": candidate_count,
        "ship_bucket_count": ship_bucket_count,
        "hidden_size": hidden_size,
        "use_cnn": use_cnn,
        "target_head": target_head,
        "target_num_heads": target_num_heads,
        "planet_dim": int(planet_dim),
        "relative_dim": int(relative_dim),
        "global_dim": int(global_dim),
        "map_channels": int(map_channels),
        "map_grid_size": int(map_grid_size),
        "dropped_relative_feature_dims": [int(x) for x in dropped_relative_feature_dims],
        "raw_config": raw,
        "metrics": payload.get("metrics") or {},
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "models": [
            {
                "name": model_name,
                "path": str(Path(model_name) / onnx_path.name),
                "checkpoint": str(checkpoint),
                "checkpoint_step": int(payload.get("step", -1)),
            }
        ]
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return onnx_path


def main() -> None:
    args = parse_args()
    path = export_checkpoint(
        args.checkpoint,
        args.out_dir,
        name=args.name,
        opset=args.opset,
        map_grid_size=args.map_grid_size,
        map_channels=args.map_channels,
        planet_dim=args.planet_dim,
        relative_dim=args.relative_dim,
        global_dim=args.global_dim,
        greedy_single_pass=args.greedy_single_pass,
    )
    print(f"exported {path}", flush=True)


if __name__ == "__main__":
    main()
