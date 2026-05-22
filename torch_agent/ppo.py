"""PPO 动作分布和参数更新。

这里实现双动作头的采样、log_prob/entropy 计算，以及接近 SB3 写法的
PPOOptimizer。以后要改 loss、KL 早停、entropy 或 value loss，主要看这里。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

from .model import PolicyOutput


@dataclass(slots=True)
class ActionSample:
    """策略网络采样出来的一次动作。

    一个动作分成三部分：
    fire 表示这颗来源星球是否发射；
    target 表示打哪个候选目标；
    amount 表示选择哪个派兵比例档位。
    """

    fire: torch.Tensor
    target: torch.Tensor
    amount: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(slots=True)
class FireSample:
    """第一步：采样是否发射。"""

    fire: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(slots=True)
class TargetSample:
    """第二步：fire=True 时采样目标。"""

    fire: torch.Tensor
    target: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(slots=True)
class RolloutBatch:
    """PPO 更新所需的扁平化 rollout 数据。

    每一行对应一颗来源星球在某个环境步的一次决策。这里保存的是
    “旧策略采样时”的动作、log_prob、value，以及后来算出来的
    returns/advantages。PPO 更新时会比较新旧策略差异。
    """

    self_features: torch.Tensor
    candidate_features: torch.Tensor
    global_features: torch.Tensor
    map_features: torch.Tensor
    candidate_mask: torch.Tensor
    amount_mask: torch.Tensor
    fire_actions: torch.Tensor
    target_actions: torch.Tensor
    amount_actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def size(self) -> int:
        """返回 rollout 中一共有多少条来源星球决策样本。"""

        return int(self.self_features.shape[0])


def safe_logits(logits: torch.Tensor) -> torch.Tensor:
    """修正全非法动作行，避免 Categorical 崩溃。

    target mask 会把非法目标的 logit 设成极小值。如果某一行连 no-op
    都被 mask 掉，Categorical 无法采样；这里兜底让 no-op 变成唯一可选项。
    """

    invalid = ~torch.isfinite(logits).any(dim=-1)
    if not invalid.any():
        return logits
    fixed = logits.clone()
    # [游戏特殊] 索引 0 是 no-op，保证始终有一个合法动作，避免全 mask 行导致 Categorical 崩溃。
    # SB3 的连续/离散动作不存在这个问题；Orbit Wars 因为太阳遮挡等情况会出现所有目标都非法的行。
    fixed[invalid, 0] = 0.0
    return fixed


def fire_target_mask(candidate_mask: torch.Tensor) -> torch.Tensor:
    """fire=True 时 target head 可选目标。

    slot 0 仍然作为 no-op 占位保留在特征里，但 fire=True 后不允许再选择
    no-op；是否不行动由 fire_head 单独决定。
    """

    mask = candidate_mask.bool().clone()
    if mask.shape[-1] > 0:
        mask[..., 0] = False
    return mask


def masked_amount_logits(
    amount_logits: torch.Tensor,
    amount_mask: torch.Tensor | None,
    target: torch.Tensor,
) -> torch.Tensor:
    """根据已选 target 屏蔽基础非法派兵档位。

    这里不做物理可达性 mask；physics 只在动作解码阶段检查网络最终选中
    的那一个 target + ships，避免每帧全图枚举导致推理过慢。
    """

    if amount_mask is None:
        return amount_logits
    row = torch.arange(target.shape[0], device=target.device)
    selected_mask = amount_mask[row, target].bool()
    fixed_mask = selected_mask.clone()
    invalid = ~fixed_mask.any(dim=-1)
    if invalid.any():
        # 上游 target_mask 与 amount_mask 偶发不一致时的兜底：
        # 把整行 mask 全置 True 避免 Categorical 崩溃；
        # 真正的 ship<=0 / 攻击不足等非法动作会在解码阶段统一丢弃，
        # 不再固定塞 slot 0（之前会导致 PPO 记下永远不会执行的假动作）。
        fixed_mask[invalid] = True
    return amount_logits.masked_fill(~fixed_mask, torch.finfo(amount_logits.dtype).min)


def sample_fire(outputs: PolicyOutput, candidate_mask: torch.Tensor, deterministic: bool) -> FireSample:
    """自回归第一步：采样 fire/no-fire。

    如果当前行没有任何可发目标，则强制 fire=False。
    """

    has_target = fire_target_mask(candidate_mask).any(dim=-1)
    fire_logits = torch.where(
        has_target,
        outputs.fire_logits,
        torch.full_like(outputs.fire_logits, -1.0e8),
    )
    fire_dist = Bernoulli(logits=fire_logits)
    fire_float = (torch.sigmoid(fire_logits) > 0.5).float() if deterministic else fire_dist.sample()
    fire = fire_float.bool()
    return FireSample(
        fire=fire,
        log_prob=fire_dist.log_prob(fire_float),
        entropy=torch.where(has_target, fire_dist.entropy(), torch.zeros_like(fire_logits)),
    )


def sample_targets(
    outputs: PolicyOutput,
    fire_sample: FireSample,
    candidate_mask: torch.Tensor,
    deterministic: bool,
) -> TargetSample:
    """自回归第二步：fire=True 时从 target logits 采样目标。"""

    target_logits = outputs.target_logits.masked_fill(~fire_target_mask(candidate_mask), torch.finfo(outputs.target_logits.dtype).min)
    no_fire = ~fire_sample.fire
    if no_fire.any():
        target_logits = target_logits.clone()
        target_logits[no_fire] = torch.finfo(target_logits.dtype).min
        target_logits[no_fire, 0] = 0.0
    target_logits = safe_logits(target_logits)
    target_dist = Categorical(logits=target_logits)
    target = target_logits.argmax(dim=-1) if deterministic else target_dist.sample()
    return TargetSample(
        fire=fire_sample.fire,
        target=target,
        log_prob=fire_sample.log_prob + fire_sample.fire.float() * target_dist.log_prob(target),
        entropy=fire_sample.entropy + fire_sample.fire.float() * target_dist.entropy(),
    )


def sample_amounts(
    outputs: PolicyOutput,
    target_sample: TargetSample,
    deterministic: bool,
    amount_mask: torch.Tensor | None = None,
) -> ActionSample:
    """自回归第二步：目标已定，再从该目标对应的 amount logits 采样数量。

    训练时用随机采样保持探索；评估时可以用 argmax。因为动作由两个
    Categorical 组成，所以总 log_prob 和总 entropy 都是两部分相加。
    """

    fire = target_sample.fire
    target = target_sample.target
    amount_logits = masked_amount_logits(outputs.amount_logits, amount_mask, target)
    amount_dist = Categorical(logits=amount_logits)
    amount = amount_logits.argmax(dim=-1) if deterministic else amount_dist.sample()
    fire_float = fire.float()
    log_prob = target_sample.log_prob + fire_float * amount_dist.log_prob(amount)
    entropy = target_sample.entropy + fire_float * amount_dist.entropy()
    return ActionSample(fire=fire, target=target, amount=amount, log_prob=log_prob, entropy=entropy)


def log_prob_entropy(
    outputs: PolicyOutput,
    fire_actions: torch.Tensor,
    target_actions: torch.Tensor,
    amount_actions: torch.Tensor,
    candidate_mask: torch.Tensor,
    amount_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """重新计算旧动作在当前新策略下的 log_prob 和 entropy。

    PPO 的关键就是比较“采样时的旧 log_prob”和“现在的新 log_prob”，
    用二者的比值限制策略更新幅度。
    """

    fire_float = fire_actions.float()
    fire_dist = Bernoulli(logits=outputs.fire_logits)
    target_logits = outputs.target_logits.masked_fill(~fire_target_mask(candidate_mask), torch.finfo(outputs.target_logits.dtype).min)
    no_fire = ~fire_actions.bool()
    if no_fire.any():
        target_logits = target_logits.clone()
        target_logits[no_fire] = torch.finfo(target_logits.dtype).min
        target_logits[no_fire, 0] = 0.0
    target_dist = Categorical(logits=safe_logits(target_logits))
    amount_logits = masked_amount_logits(outputs.amount_logits, amount_mask, target_actions)
    amount_dist = Categorical(logits=amount_logits)
    log_prob = (
        fire_dist.log_prob(fire_float)
        + fire_float * target_dist.log_prob(target_actions)
        + fire_float * amount_dist.log_prob(amount_actions)
    )
    entropy = fire_dist.entropy() + fire_float * target_dist.entropy() + fire_float * amount_dist.entropy()
    return log_prob, entropy


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """衡量 critic 的 value 预测质量。

    越接近 1，说明 value 能解释大部分 return 变化；接近 0 或负数，
    说明 critic 基本没学明白。
    """

    y_pred = y_pred.detach().float().cpu()
    y_true = y_true.detach().float().cpu()
    var_y = torch.var(y_true, unbiased=False)
    if float(var_y) == 0.0:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y)


class PPOOptimizer:
    """负责一次 PPO 参数更新的优化器。

    它的职责和 SB3 的 `PPO.train()` 类似：不和环境互动，只吃
    RolloutBatch，然后更新 actor-critic。这样以后想改 loss、KL 早停、
    entropy 系数或 value loss，都集中改这个文件。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        epochs: int,
        minibatch_size: int,
        device: torch.device,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        target_kl: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.device = device
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.target_kl = target_kl
        self.n_updates = 0

    def train(self, batch: RolloutBatch) -> dict[str, float]:
        """用一批 rollout 训练当前策略。

        这一步会把同一批旧经历打乱成多个 minibatch，并重复训练多轮。
        target_actions/amount_actions 是 Orbit Wars 特有的双动作头。
        """

        if batch.size() == 0:
            return self.empty_metrics()

        self.model.train()
        tensors = self._to_device(batch)
        size = int(tensors.self_features.shape[0])
        minibatch_size = min(max(1, self.minibatch_size), size)
        totals = self.empty_totals()
        steps = 0

        for _ in range(self.epochs):
            order = torch.randperm(size, device=self.device)
            continue_training = True
            for start in range(0, size, minibatch_size):
                idx = order[start : start + minibatch_size]
                outputs = self.model(
                    tensors.self_features[idx],
                    tensors.candidate_features[idx],
                    tensors.global_features[idx],
                    tensors.candidate_mask[idx],
                    map_features=tensors.map_features[idx],
                    selected_targets=tensors.target_actions[idx],
                )
                log_prob, entropy = log_prob_entropy(
                    outputs,
                    tensors.fire_actions[idx],
                    tensors.target_actions[idx],
                    tensors.amount_actions[idx],
                    tensors.candidate_mask[idx],
                    tensors.amount_mask[idx],
                )
                values = outputs.value.flatten()
                advantages = tensors.advantages[idx]

                # [SB3] advantage 归一化在每个 minibatch 内单独做，而非对整个 rollout 做一次。
                # CleanRL 默认对整个 rollout 做全局归一化。
                if self.norm_adv and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

                log_ratio = log_prob - tensors.old_log_probs[idx]
                ratio = torch.exp(log_ratio)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # [SB3] value clipping: 先把新旧 value 的差值裁到 clip_coef 范围内，
                # 再用 clipped prediction 和 TD(GAE) return 计算 MSE。
                # 注意这个 loss 会受到 reward scale 影响。
                if self.clip_vloss:
                    values_pred = tensors.old_values[idx] + torch.clamp(
                        values - tensors.old_values[idx],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                else:
                    values_pred = values
                value_loss = F.mse_loss(tensors.returns[idx], values_pred)

                entropy_loss = -entropy.mean()
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                totals["loss"] += float(loss.detach().cpu())
                totals["policy_loss"] += float(policy_loss.detach().cpu())
                totals["value_loss"] += float(value_loss.detach().cpu())
                totals["entropy"] += float(entropy.mean().detach().cpu())
                totals["old_approx_kl"] += float(old_approx_kl.detach().cpu())
                totals["approx_kl"] += float(approx_kl.detach().cpu())
                totals["clipfrac"] += float(clipfrac.detach().cpu())
                steps += 1
                self.n_updates += 1

                # [SB3] KL 早停：approx_kl > 1.5 * target_kl 时中断当前 epoch。
                # CleanRL 默认不启用 KL 早停。
                if self.target_kl is not None and self.target_kl > 0:
                    if float(approx_kl.detach().cpu()) > 1.5 * self.target_kl:
                        continue_training = False
                        break
            if not continue_training:
                break

        metrics = {key: value / max(steps, 1) for key, value in totals.items()}
        metrics["explained_variance"] = explained_variance(tensors.old_values, tensors.returns)
        metrics["n_updates"] = float(self.n_updates)
        return metrics

    def _to_device(self, batch: RolloutBatch) -> RolloutBatch:
        """把 rollout batch 里的所有张量搬到 GPU/CPU 训练设备上。"""

        return RolloutBatch(
            self_features=batch.self_features.to(self.device),
            candidate_features=batch.candidate_features.to(self.device),
            global_features=batch.global_features.to(self.device),
            map_features=batch.map_features.to(self.device),
            candidate_mask=batch.candidate_mask.to(self.device).bool(),
            amount_mask=batch.amount_mask.to(self.device).bool(),
            fire_actions=batch.fire_actions.to(self.device).bool(),
            target_actions=batch.target_actions.to(self.device),
            amount_actions=batch.amount_actions.to(self.device),
            old_log_probs=batch.old_log_probs.to(self.device),
            old_values=batch.old_values.to(self.device),
            returns=batch.returns.to(self.device),
            advantages=batch.advantages.to(self.device),
        )

    @staticmethod
    def empty_totals() -> dict[str, float]:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "old_approx_kl": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
        }

    @staticmethod
    def empty_metrics() -> dict[str, float]:
        metrics = PPOOptimizer.empty_totals()
        metrics["explained_variance"] = float("nan")
        metrics["n_updates"] = 0.0
        return metrics
