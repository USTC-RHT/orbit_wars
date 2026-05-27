"""Export autoregressive JAX checkpoints into ONNX Runtime subgraphs.

The exported agent keeps feature encoding and physics in Python/JAX eager mode,
but runs the neural encoder/decoder heads through ONNX Runtime. This avoids the
large first-step `jax.jit` compile cost in local matches.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax2onnx
import numpy as np

from .features_jax import FLEET_DIM, GLOBAL_DIM, MAP_CHANNELS, MAP_GRID_SIZE, PLANET_DIM, RELATIVE_DIM
from .fast_env import COMETS_PER_SPAWN, MAX_COMET_PATH, MAX_COMET_SPAWNS, EnvParams
from .model_jax import AutoregressiveActorCritic
from .normalization_jax import NormalizerParams


def _load_model(ckpt_path: Path):
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["cfg"]
    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    feat_max_fleets = EnvParams(
        episode_steps=cfg.env.episode_steps,
        max_planets=cfg.env.max_planets,
    ).max_fleets
    max_fleet_tokens = int(getattr(cfg.model, "max_fleet_tokens", 16))
    model = AutoregressiveActorCritic(
        planet_dim=PLANET_DIM,
        relative_dim=RELATIVE_DIM,
        global_dim=GLOBAL_DIM,
        relative_count=cfg.env.max_planets + 1,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        use_cnn=cfg.model.use_cnn,
        fleet_dim=FLEET_DIM,
        max_fleet_tokens=max_fleet_tokens,
        num_transformer_layers=int(getattr(cfg.model, "num_transformer_layers", 2)),
        num_heads=int(getattr(cfg.model, "num_heads", 4)),
        max_action_tokens=int(getattr(cfg.model, "max_action_tokens", 16)),
        end_bias_init=float(getattr(cfg.model, "end_bias_init", -2.0)),
    )
    meta = {
        "checkpoint": ckpt_path.name,
        "max_planets": int(cfg.env.max_planets),
        "max_fleets": int(feat_max_fleets),
        "ship_bucket_count": int(cfg.env.ship_bucket_count),
        "max_ships": float(cfg.env.max_ships),
        "max_production": float(cfg.env.max_production),
        "episode_steps": int(cfg.env.episode_steps),
        "target_filter_enabled": bool(cfg.env.target_filter_enabled),
        "attack_top_k": int(cfg.env.attack_top_k),
        "defense_top_k": int(cfg.env.defense_top_k),
        "hidden_size": int(cfg.model.hidden_size),
        "use_cnn": bool(cfg.model.use_cnn),
        "max_fleet_tokens": max_fleet_tokens,
        "num_transformer_layers": int(getattr(cfg.model, "num_transformer_layers", 2)),
        "num_heads": int(getattr(cfg.model, "num_heads", 4)),
        "max_action_tokens": int(getattr(cfg.model, "max_action_tokens", 16)),
        "end_bias_init": float(getattr(cfg.model, "end_bias_init", -2.0)),
        "state_norm_enabled": bool(getattr(cfg.state_norm, "enabled", True)),
        "state_norm_clip": float(getattr(cfg.state_norm, "clip", 5.0)),
        "state_norm_eps": float(getattr(cfg.state_norm, "eps", 1.0e-8)),
    }
    return ckpt, cfg, params, model, meta


def _shape(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def _export_one(ckpt_path: Path, out_dir: Path, opset: int) -> dict:
    ckpt, cfg, params, model, meta = _load_model(ckpt_path)
    ckpt_dir = out_dir / ckpt_path.stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    p = meta["max_planets"]
    a = meta["ship_bucket_count"]
    h = meta["hidden_size"]
    ft = meta["max_fleet_tokens"]

    common_names = {
        "opset": opset,
        "return_mode": "file",
    }

    def auto_start(planet_features, relative_features, global_features, candidate_mask, initial_ships, map_features, fleet_features, fleet_mask):
        state_emb, h0, prev_tok, remaining, planet_out, relative_h, key_proj, src_key_proj, value = model.apply(
            {"params": params},
            planet_features,
            relative_features,
            global_features,
            candidate_mask,
            initial_ships,
            map_features=map_features,
            fleet_features=fleet_features,
            fleet_mask=fleet_mask,
            method=model.auto_start,
        )
        return state_emb, h0, prev_tok, remaining + jnp.zeros_like(remaining), planet_out, relative_h, key_proj, src_key_proj, value

    jax2onnx.to_onnx(
        auto_start,
        [
            _shape((1, p, PLANET_DIM), jnp.float32),
            _shape((1, p, p + 1, RELATIVE_DIM), jnp.float32),
            _shape((1, p, GLOBAL_DIM), jnp.float32),
            _shape((1, p, p + 1), jnp.bool_),
            _shape((1, p + 1), jnp.float32),
            _shape((1, p, MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE), jnp.float32),
            _shape((1, p, ft, FLEET_DIM), jnp.float32),
            _shape((1, p, ft), jnp.bool_),
        ],
        output_path=ckpt_dir / "auto_start.onnx",
        input_names=[
            "planet_features",
            "relative_features",
            "global_features",
            "candidate_mask",
            "initial_ships",
            "map_features",
            "fleet_features",
            "fleet_mask",
        ],
        output_names=["state_emb", "h", "prev_tok", "remaining", "planet_out", "relative_h", "key_proj", "src_key_proj", "value"],
        **common_names,
    )

    def auto_step_base(state_emb, h0, prev_tok, remaining, relative_features, source_mask, src_key_proj):
        return model.apply(
            {"params": params},
            state_emb,
            h0,
            prev_tok,
            remaining,
            relative_features,
            source_mask,
            src_key_proj,
            method=model.auto_step_base,
        )

    jax2onnx.to_onnx(
        auto_step_base,
        [
            _shape((1, h), jnp.float32),
            _shape((1, h), jnp.float32),
            _shape((1, h), jnp.float32),
            _shape((1, p + 1), jnp.float32),
            _shape((1, p, p + 1, RELATIVE_DIM), jnp.float32),
            _shape((1, p), jnp.bool_),
            _shape((1, p + 1, h), jnp.float32),
        ],
        output_path=ckpt_dir / "auto_step_base.onnx",
        input_names=["state_emb", "h", "prev_tok", "remaining", "relative_features", "source_mask", "src_key_proj"],
        output_names=["h_new", "end_logits", "src_logits"],
        **common_names,
    )

    def auto_tgt_logits(h_new, src_t, planet_out, relative_h, key_proj, candidate_mask):
        return model.apply(
            {"params": params},
            h_new,
            src_t,
            planet_out,
            relative_h,
            key_proj,
            candidate_mask,
            method=model.auto_tgt_logits,
        )

    jax2onnx.to_onnx(
        auto_tgt_logits,
        [
            _shape((1, h), jnp.float32),
            _shape((1,), jnp.int32),
            _shape((1, p + 1, h), jnp.float32),
            _shape((1, p, p + 1, h), jnp.float32),
            _shape((1, p + 1, h), jnp.float32),
            _shape((1, p, p + 1), jnp.bool_),
        ],
        output_path=ckpt_dir / "auto_tgt_logits.onnx",
        input_names=["h_new", "src_t", "planet_out", "relative_h", "key_proj", "candidate_mask"],
        output_names=["tgt_logits"],
        **common_names,
    )

    def auto_amt_logits(h_new, src_t, tgt_t, planet_out, relative_h, remaining, amount_mask):
        return model.apply(
            {"params": params},
            h_new,
            src_t,
            tgt_t,
            planet_out,
            relative_h,
            remaining,
            amount_mask,
            method=model.auto_amt_logits,
        )

    jax2onnx.to_onnx(
        auto_amt_logits,
        [
            _shape((1, h), jnp.float32),
            _shape((1,), jnp.int32),
            _shape((1,), jnp.int32),
            _shape((1, p + 1, h), jnp.float32),
            _shape((1, p, p + 1, h), jnp.float32),
            _shape((1, p + 1), jnp.float32),
            _shape((1, p, p + 1, a), jnp.float32),
        ],
        output_path=ckpt_dir / "auto_amt_logits.onnx",
        input_names=["h_new", "src_t", "tgt_t", "planet_out", "relative_h", "remaining", "amount_mask"],
        output_names=["amount_logits"],
        **common_names,
    )

    def auto_token_embed(src_t, tgt_t, amt_t, end_t, executed_t, step_i, planet_out):
        return model.apply(
            {"params": params},
            src_t,
            tgt_t,
            amt_t,
            end_t,
            executed_t,
            step_i,
            planet_out,
            method=model.auto_token_embed,
        )

    jax2onnx.to_onnx(
        auto_token_embed,
        [
            _shape((1,), jnp.int32),
            _shape((1,), jnp.int32),
            _shape((1,), jnp.int32),
            _shape((1,), jnp.bool_),
            _shape((1,), jnp.bool_),
            _shape((), jnp.int32),
            _shape((1, p + 1, h), jnp.float32),
        ],
        output_path=ckpt_dir / "auto_token_embed.onnx",
        input_names=["src_t", "tgt_t", "amt_t", "end_t", "executed_t", "step_i", "planet_out"],
        output_names=["prev_tok"],
        **common_names,
    )

    norm = ckpt["normalizer_state"]
    np.savez(
        ckpt_dir / "normalizer.npz",
        planet_mean=np.asarray(norm.planet_mean, dtype=np.float32),
        planet_var=np.asarray(norm.planet_var, dtype=np.float32),
        planet_count=np.asarray(norm.planet_count, dtype=np.float32),
        relative_mean=np.asarray(norm.relative_mean, dtype=np.float32),
        relative_var=np.asarray(norm.relative_var, dtype=np.float32),
        relative_count=np.asarray(norm.relative_count, dtype=np.float32),
        global_mean=np.asarray(norm.global_mean, dtype=np.float32),
        global_var=np.asarray(norm.global_var, dtype=np.float32),
        global_count=np.asarray(norm.global_count, dtype=np.float32),
    )
    meta["normalizer_params"] = NormalizerParams.from_train_config(cfg).__dict__
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=23)
    args = parser.parse_args()

    logging.getLogger("jax2onnx.plugins.plugin_system").setLevel(logging.ERROR)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for ckpt in args.checkpoint:
        print(f"[export] {ckpt}")
        metas.append(_export_one(ckpt, args.out_dir, args.opset))
    (args.out_dir / "manifest.json").write_text(json.dumps({"checkpoints": metas}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
