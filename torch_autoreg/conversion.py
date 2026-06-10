"""Conversion helpers for Flax autoregressive checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import AutoregressiveActorCriticTorch


def cfg_to_model_kwargs(cfg: Any, *, planet_dim: int, relative_dim: int, global_dim: int, fleet_dim: int) -> dict[str, Any]:
    model_cfg = cfg.model
    return {
        "planet_dim": int(planet_dim),
        "relative_dim": int(relative_dim),
        "global_dim": int(global_dim),
        "relative_count": int(cfg.env.max_planets) + 1,
        "ship_bucket_count": int(cfg.env.ship_bucket_count),
        "hidden_size": int(model_cfg.hidden_size),
        "use_cnn": bool(model_cfg.use_cnn),
        "fleet_dim": int(fleet_dim),
        "max_fleet_tokens": int(getattr(model_cfg, "max_fleet_tokens", 32)),
        "num_transformer_layers": int(getattr(model_cfg, "num_transformer_layers", 2)),
        "num_heads": int(getattr(model_cfg, "num_heads", 4)),
        "max_action_tokens": int(getattr(model_cfg, "max_action_tokens", 16)),
        "end_bias_init": float(getattr(model_cfg, "end_bias_init", -2.0)),
    }


def checkpoint_metadata(cfg: Any, update: int | None, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    fleet_selection = getattr(cfg.model, "fleet_selection", "size") or "size"
    return {
        "update": None if update is None else int(update),
        "run_name": str(getattr(cfg, "run_name", "")),
        "env": {
            "board_size": float(getattr(cfg.env, "board_size", 100.0)),
            "episode_steps": int(cfg.env.episode_steps),
            "ship_bucket_count": int(cfg.env.ship_bucket_count),
            "max_planets": int(cfg.env.max_planets),
            "max_ships": float(cfg.env.max_ships),
            "max_production": float(cfg.env.max_production),
            "target_filter_enabled": bool(cfg.env.target_filter_enabled),
            "attack_top_k": int(cfg.env.attack_top_k),
            "defense_top_k": int(cfg.env.defense_top_k),
        },
        "model": dict(model_kwargs),
        "feature": {
            "max_fleet_tokens": int(model_kwargs["max_fleet_tokens"]),
            "fleet_selection": str(fleet_selection),
        },
        "state_norm": {
            "enabled": bool(getattr(cfg.state_norm, "enabled", True)),
            "clip": float(getattr(cfg.state_norm, "clip", 5.0)),
            "eps": float(getattr(cfg.state_norm, "eps", 1.0e-8)),
        },
    }


def normalizer_to_torch_dict(normalizer_state: Any) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(np.asarray(getattr(normalizer_state, name)), dtype=torch.float32)
        for name in [
            "planet_mean",
            "planet_var",
            "planet_count",
            "relative_mean",
            "relative_var",
            "relative_count",
            "global_mean",
            "global_var",
            "global_count",
        ]
    }


def convert_flax_params_to_torch(model: AutoregressiveActorCriticTorch, params: dict[str, Any]) -> None:
    with torch.no_grad():
        _copy_mlp(model.planet_mlp, params["MLP_0"])
        _copy_mlp(model.fleet_mlp, params["MLP_1"])
        _copy_mlp(model.relative_mlp, params["MLP_2"])
        _copy_ln(model.encoder_ln, params["LayerNorm_0"])
        for idx, layer in enumerate(model.encoder_layers):
            _copy_transformer_layer(layer, params[f"TransformerEncoderLayer_{idx}"])

        _copy_mlp(model.auto_global_mlp, params["auto_global_mlp"])
        _copy_linear(model.auto_state_emb, params["auto_state_emb"])
        _copy_linear(model.auto_init_h, params["auto_init_h"])
        _copy_gru(model.decoder_gru, params["decoder_gru"])
        _copy_linear(model.auto_key_proj, params["auto_key_proj"])
        _copy_linear(model.auto_src_key_proj, params["auto_src_key_proj"])
        if model.use_cnn:
            _copy_conv(model.auto_src_map_conv1, params["auto_src_map_conv1"])
            _copy_conv(model.auto_src_map_conv2, params["auto_src_map_conv2"])
            _copy_conv(model.auto_src_map_conv3, params["auto_src_map_conv3"])
            _copy_linear(model.auto_src_map_h, params["auto_src_map_h"])
            _copy_linear(model.auto_src_map_fuse, params["auto_src_map_fuse"])
            _copy_linear(model.auto_src_map_key, params["auto_src_map_key"])

        _copy_linear(model.auto_end_out, params["auto_end_out"])
        _copy_linear(model.auto_src_pre, params["auto_src_pre"])
        _copy_linear(model.auto_src_out, params["auto_src_out"])
        _copy_linear(model.auto_tgt_pre, params["auto_tgt_pre"])
        _copy_linear(model.auto_tgt_out, params["auto_tgt_out"])
        _copy_linear(model.auto_amt_pre, params["auto_amt_pre"])
        _copy_linear(model.auto_amt_out, params["auto_amt_out"])
        _copy_linear(model.auto_value_h, params["auto_value_h"])
        _copy_linear(model.auto_value_out, params["auto_value_out"])

        _copy_param(model.cls_token, params["cls_token"])
        _copy_param(model.type_embed, params["type_embed"])
        _copy_param(model.bos_token, params["bos_token"])
        _copy_param(model.end_token, params["end_token"])
        _copy_param(model.invalid_token, params["invalid_token"])
        _copy_param(model.amount_emb, params["amount_emb"])
        _copy_param(model.step_emb, params["step_emb"])
        _copy_param(model.end_bias, params["end_bias"])
        _copy_param(model.pointer_logit_scale, params["pointer_logit_scale"])


def save_converted_checkpoint(
    output_path: Path,
    model: AutoregressiveActorCriticTorch,
    metadata: dict[str, Any],
    normalizer: dict[str, torch.Tensor],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "normalizer": normalizer,
        },
        output_path,
    )


def load_converted_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[AutoregressiveActorCriticTorch, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = AutoregressiveActorCriticTorch(**payload["metadata"]["model"])
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def _copy_param(target: torch.Tensor, source: Any) -> None:
    target.copy_(torch.as_tensor(np.asarray(source), dtype=target.dtype, device=target.device))


def _copy_linear(module: torch.nn.Linear, source: dict[str, Any]) -> None:
    module.weight.copy_(torch.as_tensor(np.asarray(source["kernel"]).T, dtype=module.weight.dtype, device=module.weight.device))
    module.bias.copy_(torch.as_tensor(np.asarray(source["bias"]), dtype=module.bias.dtype, device=module.bias.device))


def _copy_conv(module: torch.nn.Conv2d, source: dict[str, Any]) -> None:
    weight = np.asarray(source["kernel"]).transpose(3, 2, 0, 1)
    module.weight.copy_(torch.as_tensor(weight, dtype=module.weight.dtype, device=module.weight.device))
    module.bias.copy_(torch.as_tensor(np.asarray(source["bias"]), dtype=module.bias.dtype, device=module.bias.device))


def _copy_ln(module: torch.nn.LayerNorm, source: dict[str, Any]) -> None:
    module.weight.copy_(torch.as_tensor(np.asarray(source["scale"]), dtype=module.weight.dtype, device=module.weight.device))
    module.bias.copy_(torch.as_tensor(np.asarray(source["bias"]), dtype=module.bias.dtype, device=module.bias.device))


def _copy_mlp(module: torch.nn.Module, source: dict[str, Any]) -> None:
    _copy_linear(module.dense0, source["Dense_0"])
    _copy_ln(module.ln0, source["LayerNorm_0"])
    _copy_linear(module.dense1, source["Dense_1"])


def _copy_transformer_layer(module: torch.nn.Module, source: dict[str, Any]) -> None:
    _copy_ln(module.ln0, source["LayerNorm_0"])
    _copy_attention(module.attn, source["MultiHeadDotProductAttention_0"])
    _copy_ln(module.ln1, source["LayerNorm_1"])
    _copy_linear(module.dense0, source["Dense_0"])
    _copy_linear(module.dense1, source["Dense_1"])


def _copy_attention(module: torch.nn.Module, source: dict[str, Any]) -> None:
    _copy_param(module.query_kernel, source["query"]["kernel"])
    _copy_param(module.query_bias, source["query"]["bias"])
    _copy_param(module.key_kernel, source["key"]["kernel"])
    _copy_param(module.key_bias, source["key"]["bias"])
    _copy_param(module.value_kernel, source["value"]["kernel"])
    _copy_param(module.value_bias, source["value"]["bias"])
    _copy_param(module.out_kernel, source["out"]["kernel"])
    _copy_param(module.out_bias, source["out"]["bias"])


def _copy_gru(module: torch.nn.Module, source: dict[str, Any]) -> None:
    _copy_linear(module.ir, source["ir"])
    _copy_linear(module.iz, source["iz"])
    _copy_linear(module.in_, source["in"])
    _copy_param(module.hr_kernel, source["hr"]["kernel"])
    _copy_param(module.hz_kernel, source["hz"]["kernel"])
    _copy_linear(module.hn, source["hn"])
