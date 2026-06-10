"""PPO utilities for the Torch autoregressive policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .model import NEG_INF, AutoregressiveActorCriticTorch


@dataclass(slots=True)
class AutoRolloutTorch:
    planet_features: torch.Tensor
    relative_features: torch.Tensor
    global_features: torch.Tensor
    map_features: torch.Tensor
    fleet_features: torch.Tensor
    fleet_mask: torch.Tensor
    candidate_mask: torch.Tensor
    amount_mask: torch.Tensor
    source_mask: torch.Tensor
    initial_ships: torch.Tensor
    end_action: torch.Tensor
    src_action: torch.Tensor
    tgt_action: torch.Tensor
    amt_action: torch.Tensor
    executed: torch.Tensor
    invalid: torch.Tensor
    old_logp_end: torch.Tensor
    old_logp_src: torch.Tensor
    old_logp_tgt: torch.Tensor
    old_logp_amt: torch.Tensor
    token_valid: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    bootstrap_value: torch.Tensor
    rollout_boundary: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def size(self) -> int:
        return int(self.old_value.shape[0])

    def to(self, device: torch.device) -> "AutoRolloutTorch":
        values = {name: getattr(self, name).to(device) for name in self.__dataclass_fields__}
        return AutoRolloutTorch(**values)


def safe_logits(logits: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(logits) & (logits > NEG_INF * 0.5)
    invalid = ~valid.any(dim=-1)
    if not invalid.any():
        return logits
    fixed = logits.clone()
    fixed[invalid] = NEG_INF
    fixed[invalid, 0] = 0.0
    return fixed


def bernoulli_log_prob(logits: torch.Tensor, action_bool: torch.Tensor) -> torch.Tensor:
    action = action_bool.float()
    return -F.binary_cross_entropy_with_logits(logits, action, reduction="none")


def categorical_log_prob(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return torch.gather(F.log_softmax(logits, dim=-1), -1, action.long().unsqueeze(-1)).squeeze(-1)


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    return -(p * logp).sum(dim=-1)


def bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    return -(p * torch.log(torch.clamp(p, min=1.0e-8, max=1.0)) + (1.0 - p) * torch.log(torch.clamp(1.0 - p, min=1.0e-8, max=1.0)))


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_values: torch.Tensor | None = None,
    rollout_boundaries: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    next_adv = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros((), dtype=values.dtype, device=values.device)
    if bootstrap_values is None:
        bootstrap_values = torch.zeros_like(values)
    if rollout_boundaries is None:
        rollout_boundaries = torch.zeros_like(dones)
    for t in range(rewards.shape[0] - 1, -1, -1):
        next_value = torch.where(rollout_boundaries[t], bootstrap_values[t], next_value)
        next_adv = torch.where(rollout_boundaries[t], torch.zeros_like(next_adv), next_adv)
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        adv = delta + gamma * gae_lambda * not_done * next_adv
        advantages[t] = adv
        next_adv = adv
        next_value = values[t]
    returns = advantages + values
    return returns, advantages


def ppo_update(
    model: AutoregressiveActorCriticTorch,
    optimizer: torch.optim.Optimizer,
    rollout: AutoRolloutTorch,
    *,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    ent_coef_fire: float = 0.005,
    ent_coef_target: float = 0.01,
    ent_coef_amount: float = 0.005,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    epochs: int = 4,
    minibatch_size: int = 32,
    norm_adv: bool = True,
    target_kl: float = 0.03,
    ent_scale: float = 1.0,
) -> dict[str, float]:
    device = next(model.parameters()).device
    data = rollout.to(device)
    size = data.size()
    minibatch_size = min(max(int(minibatch_size), 1), size)
    num_minibatches = max(size // minibatch_size, 1)
    metric_sums: dict[str, float] = {}
    n_minibatches = 0
    n_epochs_run = 0
    for epoch in range(int(epochs)):
        order = torch.randperm(size, device=device)
        epoch_kl = 0.0
        epoch_batches = 0
        for mb_idx in range(num_minibatches):
            start = mb_idx * minibatch_size
            idx = order[start : start + minibatch_size]
            batch = _slice_rollout(data, idx)
            loss, metrics = ppo_minibatch_loss(
                model,
                batch,
                clip_coef=clip_coef,
                ent_coef=ent_coef,
                ent_coef_fire=ent_coef_fire,
                ent_coef_target=ent_coef_target,
                ent_coef_amount=ent_coef_amount,
                vf_coef=vf_coef,
                norm_adv=norm_adv,
                ent_scale=ent_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            n_minibatches += 1
            epoch_batches += 1
            epoch_kl += float(metrics["approx_kl"])
        n_epochs_run += 1
        if target_kl > 0.0 and epoch_batches > 0 and epoch_kl / epoch_batches > 1.5 * target_kl:
            break
    denom = max(n_minibatches, 1)
    out = {key: value / denom for key, value in metric_sums.items()}
    out["n_epochs_run"] = float(n_epochs_run)
    return out


def ppo_minibatch_loss(
    model: AutoregressiveActorCriticTorch,
    batch: AutoRolloutTorch,
    *,
    clip_coef: float,
    ent_coef: float,
    ent_coef_fire: float,
    ent_coef_target: float,
    ent_coef_amount: float,
    vf_coef: float,
    norm_adv: bool,
    ent_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(
        batch.planet_features,
        batch.relative_features,
        batch.global_features,
        batch.candidate_mask,
        batch.amount_mask,
        batch.source_mask,
        batch.end_action,
        batch.src_action,
        batch.tgt_action,
        batch.amt_action,
        batch.executed,
        batch.initial_ships,
        map_features=batch.map_features,
        fleet_features=batch.fleet_features,
        fleet_mask=batch.fleet_mask,
    )
    end_logp = bernoulli_log_prob(outputs.end_logits, batch.end_action)
    src_logits = safe_logits(outputs.src_logits)
    tgt_logits = safe_logits(outputs.tgt_logits)
    amt_logits = safe_logits(outputs.amount_logits)
    src_logp = categorical_log_prob(src_logits, batch.src_action)
    tgt_logp = categorical_log_prob(tgt_logits, batch.tgt_action)
    amt_logp = categorical_log_prob(amt_logits, batch.amt_action)

    end_mask = batch.token_valid.float()
    arg_mask = batch.executed.float()
    adv = batch.advantages
    if norm_adv and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1.0e-8)
    adv_k = adv[:, None]

    l_end, kl_end, cf_end, mr_end = _token_loss(end_logp, batch.old_logp_end, end_mask, adv_k, clip_coef)
    l_src, kl_src, cf_src, mr_src = _token_loss(src_logp, batch.old_logp_src, arg_mask, adv_k, clip_coef)
    l_tgt, kl_tgt, cf_tgt, mr_tgt = _token_loss(tgt_logp, batch.old_logp_tgt, arg_mask, adv_k, clip_coef)
    l_amt, kl_amt, cf_amt, mr_amt = _token_loss(amt_logp, batch.old_logp_amt, arg_mask, adv_k, clip_coef)
    policy_loss = (l_end + l_src + l_tgt + l_amt) / 4.0

    value_pred = batch.old_value + torch.clamp(outputs.value - batch.old_value, -clip_coef, clip_coef)
    value_loss = torch.square(batch.returns - value_pred).mean()

    ent_end = bernoulli_entropy(outputs.end_logits)
    ent_src = categorical_entropy(src_logits)
    ent_tgt = categorical_entropy(tgt_logits)
    ent_amt = categorical_entropy(amt_logits)
    end_denom = torch.clamp(end_mask.sum(), min=1.0)
    arg_denom = torch.clamp(arg_mask.sum(), min=1.0)
    entropy_end = (ent_end * end_mask).sum() / end_denom
    entropy_src = (ent_src * arg_mask).sum() / arg_denom
    entropy_tgt = (ent_tgt * arg_mask).sum() / arg_denom
    entropy_amt = (ent_amt * arg_mask).sum() / arg_denom
    entropy = entropy_end + entropy_src + entropy_tgt + entropy_amt
    entropy_bonus = ent_scale * (
        ent_coef_fire * entropy_end
        + ent_coef * entropy_src
        + ent_coef_target * entropy_tgt
        + ent_coef_amount * entropy_amt
    )
    loss = policy_loss + vf_coef * value_loss - entropy_bonus
    approx_kl = (kl_end + kl_src + kl_tgt + kl_amt) / 4.0
    clipfrac = (cf_end + cf_src + cf_tgt + cf_amt) / 4.0
    max_abs_log_ratio = torch.maximum(torch.maximum(mr_end, mr_src), torch.maximum(mr_tgt, mr_amt))
    non_end = (batch.token_valid.bool() & (~batch.end_action.bool())).float()
    non_end_denom = torch.clamp(non_end.sum(), min=1.0)
    invalid_action_rate = (batch.invalid.float() * non_end).sum() / non_end_denom
    executed_action_rate = arg_mask.sum() / non_end_denom
    seq_len_mean = non_end.sum(dim=-1).mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "entropy_end": float(entropy_end.detach().cpu()),
        "entropy_src": float(entropy_src.detach().cpu()),
        "entropy_tgt": float(entropy_tgt.detach().cpu()),
        "entropy_amt": float(entropy_amt.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "clipfrac": float(clipfrac.detach().cpu()),
        "max_abs_log_ratio": float(max_abs_log_ratio.detach().cpu()),
        "max_abs_log_ratio_end": float(mr_end.detach().cpu()),
        "max_abs_log_ratio_src": float(mr_src.detach().cpu()),
        "max_abs_log_ratio_tgt": float(mr_tgt.detach().cpu()),
        "max_abs_log_ratio_amt": float(mr_amt.detach().cpu()),
        "invalid_action_rate": float(invalid_action_rate.detach().cpu()),
        "executed_action_rate": float(executed_action_rate.detach().cpu()),
        "seq_len_mean": float(seq_len_mean.detach().cpu()),
    }
    return loss, metrics


def _token_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    mask: torch.Tensor,
    adv_k: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_lr = torch.where(mask > 0.0, new_logp - old_logp, torch.zeros_like(new_logp))
    lr = torch.clamp(raw_lr, min=-5.0, max=float(torch.log(torch.tensor(10.0))))
    ratio = torch.exp(lr)
    pg1 = adv_k * ratio
    pg2 = adv_k * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    denom = torch.clamp(mask.sum(), min=1.0)
    loss = -(torch.minimum(pg1, pg2) * mask).sum() / denom
    kl = (((ratio - 1.0) - lr) * mask).sum() / denom
    clipfrac = (((torch.abs(ratio - 1.0) > clip_coef).float()) * mask).sum() / denom
    max_abs = (torch.abs(raw_lr) * mask).max()
    return loss, kl, clipfrac, max_abs


def _slice_rollout(data: AutoRolloutTorch, idx: torch.Tensor) -> AutoRolloutTorch:
    values = {name: getattr(data, name)[idx] for name in data.__dataclass_fields__}
    return AutoRolloutTorch(**values)


def stack_records(records: list[dict[str, Any]], *, gamma: float, gae_lambda: float) -> AutoRolloutTorch:
    if not records:
        raise ValueError("cannot build rollout from zero records")

    def stack(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        values = [record[name] for record in records]
        tensor = torch.stack([value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in values], dim=0)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    rewards = stack("reward", torch.float32)
    dones = stack("done", torch.bool)
    bootstrap_values = stack("bootstrap_value", torch.float32)
    rollout_boundaries = stack("rollout_boundary", torch.bool)
    old_values = stack("old_value", torch.float32)
    returns, advantages = compute_gae(
        rewards,
        dones,
        old_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
        bootstrap_values=bootstrap_values,
        rollout_boundaries=rollout_boundaries,
    )
    return AutoRolloutTorch(
        planet_features=stack("planet_features", torch.float32),
        relative_features=stack("relative_features", torch.float32),
        global_features=stack("global_features", torch.float32),
        map_features=stack("map_features", torch.float32),
        fleet_features=stack("fleet_features", torch.float32),
        fleet_mask=stack("fleet_mask", torch.bool),
        candidate_mask=stack("candidate_mask", torch.bool),
        amount_mask=stack("amount_mask", torch.bool),
        source_mask=stack("source_mask", torch.bool),
        initial_ships=stack("initial_ships", torch.float32),
        end_action=stack("end_action", torch.bool),
        src_action=stack("src_action", torch.long),
        tgt_action=stack("tgt_action", torch.long),
        amt_action=stack("amt_action", torch.long),
        executed=stack("executed", torch.bool),
        invalid=stack("invalid", torch.bool),
        old_logp_end=stack("old_logp_end", torch.float32),
        old_logp_src=stack("old_logp_src", torch.float32),
        old_logp_tgt=stack("old_logp_tgt", torch.float32),
        old_logp_amt=stack("old_logp_amt", torch.float32),
        token_valid=stack("token_valid", torch.bool),
        old_value=old_values,
        reward=rewards,
        done=dones,
        bootstrap_value=bootstrap_values,
        rollout_boundary=rollout_boundaries,
        returns=returns,
        advantages=advantages,
    )
