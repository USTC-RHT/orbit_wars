"""PyTorch port of ``jax_agent.model_jax.AutoregressiveActorCritic``.

The class mirrors the Flax autoregressive actor-critic closely enough that a
JAX checkpoint can be converted into a Torch ``state_dict`` and used for
incremental inference or PPO-style full-sequence logit recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


NEG_INF = -1.0e9
LAYER_NORM_EPS = 1.0e-6


@dataclass(slots=True)
class AutoPolicyOutputTorch:
    end_logits: torch.Tensor
    src_logits: torch.Tensor
    tgt_logits: torch.Tensor
    amount_logits: torch.Tensor
    value: torch.Tensor


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    denom = torch.sqrt(torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), min=1.0e-12))
    return x / denom


def _bucket_values(ships: torch.Tensor, ship_bucket_count: int) -> torch.Tensor:
    ratios = torch.tensor(
        [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        dtype=ships.dtype,
        device=ships.device,
    )[:ship_bucket_count]
    s = torch.floor(ships)
    raw = torch.clamp(torch.round(s[:, None] * ratios[None, :]), min=10.0)
    raw = torch.minimum(raw, s[:, None])
    return torch.where(s[:, None] >= 10.0, raw, torch.zeros_like(raw))


class FlaxMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.dense0 = nn.Linear(input_dim, hidden_size)
        self.ln0 = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)
        self.dense1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense0(x)
        x = self.ln0(x)
        x = F.relu(x)
        x = self.dense1(x)
        return F.relu(x)


class FlaxMultiHeadDotProductAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query_kernel = nn.Parameter(torch.empty(hidden_size, num_heads, self.head_dim))
        self.query_bias = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.key_kernel = nn.Parameter(torch.empty(hidden_size, num_heads, self.head_dim))
        self.key_bias = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.value_kernel = nn.Parameter(torch.empty(hidden_size, num_heads, self.head_dim))
        self.value_bias = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.out_kernel = nn.Parameter(torch.empty(num_heads, self.head_dim, hidden_size))
        self.out_bias = nn.Parameter(torch.empty(hidden_size))

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = torch.einsum("bti,ihd->bthd", query, self.query_kernel) + self.query_bias
        k = torch.einsum("bti,ihd->bthd", key_value, self.key_kernel) + self.key_bias
        v = torch.einsum("bti,ihd->bthd", key_value, self.value_kernel) + self.value_bias

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(float(self.head_dim))
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :].bool(), NEG_INF)
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", weights, v)
        return torch.einsum("bqhd,hdo->bqo", out, self.out_kernel) + self.out_bias


class TransformerEncoderLayerTorch(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln0 = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)
        self.attn = FlaxMultiHeadDotProductAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)
        self.dense0 = nn.Linear(hidden_size, hidden_size * mlp_ratio)
        self.dense1 = nn.Linear(hidden_size * mlp_ratio, hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln0(x)
        x = x + self.attn(h, h, mask=mask)
        h = self.ln1(x)
        h = self.dense0(h)
        h = F.gelu(h, approximate="tanh")
        h = self.dense1(h)
        return x + h


class FlaxGRUCellTorch(nn.Module):
    """GRU cell with the same gate equations and parameter split as Flax."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.ir = nn.Linear(input_size, hidden_size, bias=True)
        self.iz = nn.Linear(input_size, hidden_size, bias=True)
        self.in_ = nn.Linear(input_size, hidden_size, bias=True)
        self.hr_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.hz_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.hn = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, h: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.ir(inputs) + h @ self.hr_kernel)
        z = torch.sigmoid(self.iz(inputs) + h @ self.hz_kernel)
        n = torch.tanh(self.in_(inputs) + r * self.hn(h))
        return (1.0 - z) * n + z * h


class AutoregressiveActorCriticTorch(nn.Module):
    def __init__(
        self,
        planet_dim: int,
        relative_dim: int,
        global_dim: int,
        relative_count: int,
        ship_bucket_count: int,
        hidden_size: int = 128,
        use_cnn: bool = False,
        map_channels: int = 6,
        map_grid_size: int = 8,
        fleet_dim: int = 28,
        max_fleet_tokens: int = 32,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        max_action_tokens: int = 16,
        end_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.planet_dim = planet_dim
        self.relative_dim = relative_dim
        self.global_dim = global_dim
        self.relative_count = relative_count
        self.ship_bucket_count = ship_bucket_count
        self.hidden_size = hidden_size
        self.use_cnn = use_cnn
        self.map_channels = map_channels
        self.map_grid_size = map_grid_size
        self.fleet_dim = fleet_dim
        self.max_fleet_tokens = max_fleet_tokens
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.max_action_tokens = max_action_tokens

        self.planet_mlp = FlaxMLP(planet_dim, hidden_size)
        self.fleet_mlp = FlaxMLP(fleet_dim, hidden_size)
        self.relative_mlp = FlaxMLP(relative_dim, hidden_size)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayerTorch(hidden_size, num_heads) for _ in range(num_transformer_layers)]
        )
        self.encoder_ln = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)

        self.auto_global_mlp = FlaxMLP(global_dim, hidden_size)
        self.auto_state_emb = nn.Linear(hidden_size * 2, hidden_size)
        self.auto_init_h = nn.Linear(hidden_size, hidden_size)
        self.decoder_gru = FlaxGRUCellTorch(hidden_size * 2, hidden_size)

        self.auto_key_proj = nn.Linear(hidden_size, hidden_size)
        self.auto_src_key_proj = nn.Linear(hidden_size, hidden_size)
        if use_cnn:
            self.auto_src_map_conv1 = nn.Conv2d(map_channels, 16, kernel_size=3, padding=1)
            self.auto_src_map_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.auto_src_map_conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
            self.auto_src_map_h = nn.Linear(16, hidden_size)
            self.auto_src_map_fuse = nn.Linear(hidden_size * 2, hidden_size)
            self.auto_src_map_key = nn.Linear(hidden_size, hidden_size)

        self.auto_end_out = nn.Linear(hidden_size, 1)
        self.auto_src_pre = nn.Linear(hidden_size, hidden_size)
        self.auto_src_out = nn.Linear(hidden_size, hidden_size)
        self.auto_tgt_pre = nn.Linear(hidden_size, hidden_size)
        self.auto_tgt_out = nn.Linear(hidden_size, hidden_size)
        self.auto_amt_pre = nn.Linear(hidden_size * 4, hidden_size)
        self.auto_amt_out = nn.Linear(hidden_size, ship_bucket_count)
        self.auto_value_h = nn.Linear(hidden_size * 2, hidden_size)
        self.auto_value_out = nn.Linear(hidden_size, 1)

        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.type_embed = nn.Parameter(torch.empty(3, hidden_size))
        self.bos_token = nn.Parameter(torch.empty(hidden_size))
        self.end_token = nn.Parameter(torch.empty(hidden_size))
        self.invalid_token = nn.Parameter(torch.empty(hidden_size))
        self.amount_emb = nn.Parameter(torch.empty(ship_bucket_count, hidden_size))
        self.step_emb = nn.Parameter(torch.empty(max_action_tokens, hidden_size))
        self.end_bias = nn.Parameter(torch.full((1,), float(end_bias_init)))
        self.pointer_logit_scale = nn.Parameter(torch.tensor([math.log(10.0)], dtype=torch.float32))

    def forward(
        self,
        planet_features: torch.Tensor,
        relative_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        amount_mask: torch.Tensor,
        source_mask: torch.Tensor,
        end_actions: torch.Tensor,
        src_actions: torch.Tensor,
        tgt_actions: torch.Tensor,
        amt_actions: torch.Tensor,
        executed: torch.Tensor,
        initial_ships: torch.Tensor,
        map_features: torch.Tensor | None = None,
        fleet_features: torch.Tensor | None = None,
        fleet_mask: torch.Tensor | None = None,
    ) -> AutoPolicyOutputTorch:
        state_emb, h, prev_tok, remaining, planet_out, relative_h, key_proj, src_key_proj, value = self.auto_start(
            planet_features,
            relative_features,
            global_features,
            candidate_mask,
            initial_ships,
            map_features=map_features,
            fleet_features=fleet_features,
            fleet_mask=fleet_mask,
        )
        end_logs = []
        src_logs = []
        tgt_logs = []
        amt_logs = []
        for step_i in range(self.max_action_tokens):
            h, end_logits, src_logits = self.auto_step_base(
                state_emb, h, prev_tok, remaining, relative_features, source_mask, src_key_proj
            )
            src_t = src_actions[:, step_i].to(torch.long)
            tgt_t = tgt_actions[:, step_i].to(torch.long)
            amt_t = amt_actions[:, step_i].to(torch.long)
            tgt_logits = self.auto_tgt_logits(h, src_t, planet_out, relative_h, key_proj, candidate_mask)
            amount_logits = self.auto_amt_logits(h, src_t, tgt_t, planet_out, relative_h, remaining, amount_mask)
            src_safe = torch.clamp(src_t, 0, self.relative_count - 1)
            ships_sent = self._ships_for_bucket(remaining[torch.arange(remaining.shape[0], device=remaining.device), src_safe], amt_t)
            remaining = remaining.clone()
            remaining[torch.arange(remaining.shape[0], device=remaining.device), src_safe] += torch.where(
                executed[:, step_i].bool(), -ships_sent, torch.zeros_like(ships_sent)
            )
            prev_tok = self.auto_token_embed(
                src_safe,
                torch.clamp(tgt_t, 0, self.relative_count - 1),
                amt_t,
                end_actions[:, step_i].bool(),
                executed[:, step_i].bool(),
                step_i,
                planet_out,
            )
            end_logs.append(end_logits)
            src_logs.append(src_logits)
            tgt_logs.append(tgt_logits)
            amt_logs.append(amount_logits)
        return AutoPolicyOutputTorch(
            end_logits=torch.stack(end_logs, dim=1),
            src_logits=torch.stack(src_logs, dim=1),
            tgt_logits=torch.stack(tgt_logs, dim=1),
            amount_logits=torch.stack(amt_logs, dim=1),
            value=value,
        )

    def auto_start(
        self,
        planet_features: torch.Tensor,
        relative_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        initial_ships: torch.Tensor,
        map_features: torch.Tensor | None = None,
        fleet_features: torch.Tensor | None = None,
        fleet_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        cls_out, planet_out, relative_h = self._encode(
            planet_features,
            relative_features,
            global_features,
            candidate_mask,
            map_features,
            fleet_features,
            fleet_mask,
        )
        global_h = self.auto_global_mlp(global_features[:, 0])
        state_emb = F.relu(self.auto_state_emb(torch.cat([cls_out, global_h], dim=-1)))
        h = torch.tanh(self.auto_init_h(state_emb))
        prev_tok = self.bos_token[None, :].expand(planet_features.shape[0], -1)
        key_proj = _l2_normalize(self.auto_key_proj(planet_out))
        src_key_proj = self._source_key_proj(planet_out, map_features)
        value_h = F.relu(self.auto_value_h(torch.cat([cls_out, global_h], dim=-1)))
        value = self.auto_value_out(value_h).squeeze(-1)
        return state_emb, h, prev_tok, initial_ships, planet_out, relative_h, key_proj, src_key_proj, value

    def auto_step_base(
        self,
        state_emb: torch.Tensor,
        h: torch.Tensor,
        prev_tok: torch.Tensor,
        remaining: torch.Tensor,
        relative_features: torch.Tensor,
        source_mask: torch.Tensor,
        src_key_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del relative_features
        decoder_input = torch.cat([prev_tok, state_emb], dim=-1)
        h_new = self.decoder_gru(h, decoder_input)
        end_logits = self.auto_end_out(h_new).squeeze(-1) + self.end_bias[0]
        q_src = self.auto_src_out(F.relu(self.auto_src_pre(h_new)))
        q_src = _l2_normalize(q_src)
        src_logits = torch.einsum("bh,bch->bc", q_src, src_key_proj) * self._pointer_logit_scale()
        source_mine = torch.cat(
            [torch.zeros((h_new.shape[0], 1), dtype=torch.bool, device=h_new.device), source_mask.bool()],
            dim=1,
        )
        src_mask = source_mine & (remaining >= 10.0)
        src_logits = src_logits.masked_fill(~src_mask, NEG_INF)
        src_logits = src_logits.masked_fill(
            torch.arange(src_logits.shape[1], device=h_new.device)[None, :] == 0,
            NEG_INF,
        )
        return h_new, end_logits, src_logits

    def auto_tgt_logits(
        self,
        h_new: torch.Tensor,
        src_t: torch.Tensor,
        planet_out: torch.Tensor,
        relative_h: torch.Tensor,
        key_proj: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(h_new.shape[0], device=h_new.device)
        source_count = relative_h.shape[1]
        src_safe = torch.clamp(src_t.to(torch.long), 0, self.relative_count - 1)
        src_h = planet_out[batch_idx, src_safe]
        rel_for_src = relative_h[batch_idx, torch.clamp(src_safe - 1, 0, source_count - 1)]
        q_tgt = self.auto_tgt_out(F.relu(self.auto_tgt_pre(h_new + src_h)))
        q_tgt = _l2_normalize(q_tgt)
        tgt_keys = _l2_normalize(key_proj + rel_for_src)
        tgt_logits = torch.einsum("bh,bch->bc", q_tgt, tgt_keys) * self._pointer_logit_scale()
        src_candidate_mask = candidate_mask[batch_idx, torch.clamp(src_safe - 1, 0, source_count - 1)].bool()
        tgt_mask = src_candidate_mask & (torch.arange(src_candidate_mask.shape[1], device=h_new.device)[None, :] != 0)
        return tgt_logits.masked_fill(~tgt_mask, NEG_INF)

    def auto_amt_logits(
        self,
        h_new: torch.Tensor,
        src_t: torch.Tensor,
        tgt_t: torch.Tensor,
        planet_out: torch.Tensor,
        relative_h: torch.Tensor,
        remaining: torch.Tensor,
        amount_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(h_new.shape[0], device=h_new.device)
        source_count = relative_h.shape[1]
        src_safe = torch.clamp(src_t.to(torch.long), 0, self.relative_count - 1)
        tgt_safe = torch.clamp(tgt_t.to(torch.long), 0, self.relative_count - 1)
        src_h = planet_out[batch_idx, src_safe]
        tgt_h = planet_out[batch_idx, tgt_safe]
        rel_for_src = relative_h[batch_idx, torch.clamp(src_safe - 1, 0, source_count - 1)]
        selected_rel_h = rel_for_src[batch_idx, tgt_safe]
        amt_h = F.relu(self.auto_amt_pre(torch.cat([h_new, src_h, tgt_h, selected_rel_h], dim=-1)))
        amount_logits = self.auto_amt_out(amt_h)
        source_amount_mask = amount_mask[batch_idx, torch.clamp(src_safe - 1, 0, amount_mask.shape[1] - 1)]
        selected_amount_mask = source_amount_mask[batch_idx, tgt_safe].bool()
        dynamic_amount_mask = self._dynamic_amount_mask(remaining[batch_idx, src_safe]).bool()
        selected_amount_mask = selected_amount_mask & dynamic_amount_mask
        return amount_logits.masked_fill(~selected_amount_mask, NEG_INF)

    def auto_token_embed(
        self,
        src_t: torch.Tensor,
        tgt_t: torch.Tensor,
        amt_t: torch.Tensor,
        end_t: torch.Tensor,
        executed_t: torch.Tensor,
        step_i: int | torch.Tensor,
        planet_out: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(src_t.shape[0], device=planet_out.device)
        src_safe = torch.clamp(src_t.to(torch.long), 0, self.relative_count - 1)
        tgt_safe = torch.clamp(tgt_t.to(torch.long), 0, self.relative_count - 1)
        amt_safe = torch.clamp(amt_t.to(torch.long), 0, self.ship_bucket_count - 1)
        if isinstance(step_i, torch.Tensor):
            step_index = int(step_i.detach().cpu().item())
        else:
            step_index = int(step_i)
        src_h = planet_out[batch_idx, src_safe]
        tgt_h = planet_out[batch_idx, tgt_safe]
        raw = src_h + tgt_h + self.amount_emb[amt_safe] + self.step_emb[step_index]
        raw = _l2_normalize(raw)
        return torch.where(
            end_t.bool()[:, None],
            self.end_token[None, :],
            torch.where(executed_t.bool()[:, None], raw, self.invalid_token[None, :]),
        )

    def _pointer_logit_scale(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.pointer_logit_scale[0]), min=1.0, max=20.0)

    def _source_key_proj(self, planet_out: torch.Tensor, map_features: torch.Tensor | None) -> torch.Tensor:
        planet_keys = self.auto_src_key_proj(planet_out)
        if self.use_cnn:
            batch_size = planet_out.shape[0]
            if map_features is None:
                source_count = self.relative_count - 1
                map_features = torch.zeros(
                    (batch_size, source_count, self.map_channels, self.map_grid_size, self.map_grid_size),
                    dtype=planet_out.dtype,
                    device=planet_out.device,
                )
            source_count = map_features.shape[1]
            m = map_features.reshape(batch_size * source_count, *map_features.shape[2:])
            m = F.relu(self.auto_src_map_conv1(m))
            m = F.relu(self.auto_src_map_conv2(m))
            m = F.relu(self.auto_src_map_conv3(m))
            m = m.mean(dim=(2, 3))
            map_h = F.relu(self.auto_src_map_h(m)).reshape(batch_size, source_count, self.hidden_size)
            src_planet_h = planet_out[:, 1 : 1 + source_count]
            src_keys = self.auto_src_map_key(F.relu(self.auto_src_map_fuse(torch.cat([src_planet_h, map_h], dim=-1))))
            slot_idx = torch.arange(planet_keys.shape[1], device=planet_out.device)
            src_region = (slot_idx >= 1) & (slot_idx < 1 + source_count)
            src_idx = torch.clamp(slot_idx - 1, 0, source_count - 1)
            src_keys_padded = src_keys[:, src_idx, :]
            planet_keys = torch.where(src_region[None, :, None], src_keys_padded, planet_keys)
        return _l2_normalize(planet_keys)

    def _encode(
        self,
        planet_features: torch.Tensor,
        relative_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        map_features: torch.Tensor | None,
        fleet_features: torch.Tensor | None,
        fleet_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del global_features, candidate_mask, map_features
        batch_size, source_count = planet_features.shape[:2]
        transformer_planet_features = self._planet_features_with_noop(planet_features)
        planet_h = self.planet_mlp(transformer_planet_features)
        if fleet_features is None:
            fleet_features = torch.zeros(
                (batch_size, source_count, self.max_fleet_tokens, self.fleet_dim),
                dtype=planet_h.dtype,
                device=planet_h.device,
            )
        if fleet_mask is None:
            fleet_mask = torch.zeros(
                (batch_size, source_count, self.max_fleet_tokens),
                dtype=torch.bool,
                device=planet_h.device,
            )
        shared_fleet_features = fleet_features[:, 0]
        shared_fleet_mask = fleet_mask[:, 0].bool()
        fleet_h = self.fleet_mlp(shared_fleet_features)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat(
            [
                cls + self.type_embed[0],
                planet_h + self.type_embed[1],
                fleet_h + self.type_embed[2],
            ],
            dim=1,
        )
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=planet_h.device)
        planet_alive = transformer_planet_features[..., 1] > 0.5
        planet_alive = planet_alive | (torch.arange(planet_alive.shape[1], device=planet_h.device)[None, :] == 0)
        token_mask = torch.cat([cls_mask, planet_alive, shared_fleet_mask], dim=1)
        x = tokens
        for layer in self.encoder_layers:
            x = layer(x, mask=token_mask)
        x = self.encoder_ln(x)
        cls_out = x[:, 0, :]
        planet_out = x[:, 1 : 1 + self.relative_count, :]
        relative_h = self.relative_mlp(relative_features)
        return cls_out, planet_out, relative_h

    def _ships_for_bucket(self, ships: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
        buckets = _bucket_values(ships, self.ship_bucket_count)
        bucket_safe = torch.clamp(bucket.to(torch.long), 0, self.ship_bucket_count - 1)
        return torch.gather(buckets, 1, bucket_safe[:, None]).squeeze(-1)

    def _dynamic_amount_mask(self, ships: torch.Tensor) -> torch.Tensor:
        return _bucket_values(ships, self.ship_bucket_count) >= 10.0

    def _planet_features_with_noop(self, planet_features: torch.Tensor) -> torch.Tensor:
        noop = torch.zeros(
            planet_features.shape[:1] + (1, planet_features.shape[-1]),
            dtype=planet_features.dtype,
            device=planet_features.device,
        )
        return torch.cat([noop, planet_features], dim=1)
