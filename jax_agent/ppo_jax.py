"""JAX 自回归策略的 PPO 训练工具。

本文件只负责优化器、优势估计、PPO 损失和 rollout 张量展平，不负责采样动作。
动作采样在 train_jax.py 中完成，采样得到的 token 序列会在这里重新计算 log-prob。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

NEG_INF = -1.0e9  # mask 后使用的大负数，避免 softmax 选到非法动作。


def _assert_all_finite(name: str, value):
    """在 JIT 内检查数值是否有限，尽早暴露 NaN/Inf。"""
    ok = jnp.all(jnp.isfinite(value))

    def _check(is_ok):
        if not bool(is_ok):
            raise FloatingPointError(f"non-finite value detected in {name}")

    jax.debug.callback(_check, ok, ordered=True)
    return value


@dataclass(frozen=True)
class PPOParams:
    """训练超参（编译时固定）。"""
    gamma: float = 0.99        # 折扣因子
    gae_lambda: float = 0.95   # GAE 对未来优势的平滑系数
    clip_coef: float = 0.2     # PPO 策略比率裁剪系数 ε
    ent_coef: float = 0.01     # 熵奖励系数（鼓励探索）
    ent_coef_fire: float = 0.01
    ent_coef_target: float = 0.01
    vf_coef: float = 0.5       # 价值函数损失系数
    max_grad_norm: float = 0.5 # 梯度范数裁剪阈值
    epochs: int = 4            # 每次 rollout 训练轮数
    minibatch_size: int = 4096 # 单个 minibatch 大小
    norm_adv: bool = True      # 是否对 advantage 做 batch 内平均方差归一化
    target_kl: float = 0.05    # KL 早停阈值；epoch 平均 approx_kl > 1.5*target_kl 时停止剩余 epoch


class AutoRollout(NamedTuple):
    """自回归 rollout 数据，按 [T, B, ...] 堆叠。

    end/src/tgt 是每个 decoder token 的动作；executed 表示动作通过 pre-mask 并真实发射；
    token_valid 表示该 token 在终止前被解码过，用于 end head 的 PPO loss。
    """
    planet_features: jnp.ndarray
    relative_features: jnp.ndarray
    global_features: jnp.ndarray
    map_features: jnp.ndarray
    fleet_features: jnp.ndarray
    fleet_mask: jnp.ndarray
    candidate_mask: jnp.ndarray
    source_mask: jnp.ndarray
    initial_ships: jnp.ndarray
    end_action: jnp.ndarray
    src_action: jnp.ndarray
    tgt_action: jnp.ndarray
    executed: jnp.ndarray
    invalid: jnp.ndarray
    old_logp_end: jnp.ndarray
    old_logp_src: jnp.ndarray
    old_logp_tgt: jnp.ndarray
    token_valid: jnp.ndarray
    old_value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def make_lr_schedule(
    lr: float,
    total_steps: int = 0,
    schedule_name: str = "constant",
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
):
    """构造 Optax 学习率调度器。

    schedule_name:
    - constant: 固定 lr
    - linear: 线性衰减到 lr * min_lr_ratio
    - warmup_cosine: 先 warmup 到 lr，再 cosine 衰减到 lr * min_lr_ratio
    """
    name = (schedule_name or "constant").lower()
    end_lr = lr * max(float(min_lr_ratio), 0.0)
    if name == "constant" or total_steps <= 0:
        return lr
    if name == "linear":
        return optax.linear_schedule(init_value=lr, end_value=end_lr, transition_steps=max(total_steps, 1))
    if name in {"warmup_cosine", "cosine_warmup"}:
        warmup_steps = max(int(warmup_steps), 0)
        if warmup_steps <= 0:
            return optax.cosine_decay_schedule(init_value=lr, decay_steps=max(total_steps, 1), alpha=max(float(min_lr_ratio), 0.0))
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=max(total_steps, warmup_steps + 1),
            end_value=end_lr,
        )
    raise ValueError(f"Unknown lr_schedule: {schedule_name}")


def make_auto_train_state(
    rng,
    model,
    sample_batch,
    lr: float,
    max_grad_norm: float,
    max_action_tokens: int,
    lr_schedule: str = "constant",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> TrainState:
    """初始化自回归模型参数和优化器状态。"""
    b = sample_batch.planet_features.shape[0]
    k = max_action_tokens
    variables = model.init(
        rng,
        sample_batch.planet_features,
        sample_batch.relative_features,
        sample_batch.global_features,
        sample_batch.candidate_mask,
        sample_batch.source_mask,
        jnp.zeros((b, k), dtype=bool),
        jnp.zeros((b, k), dtype=jnp.int32),
        jnp.zeros((b, k), dtype=jnp.int32),
        jnp.zeros((b, k), dtype=bool),
        jnp.concatenate([jnp.zeros((b, 1), dtype=sample_batch.source_ships.dtype), sample_batch.source_ships], axis=1),
        map_features=sample_batch.map_features,
        fleet_features=sample_batch.fleet_features,
        fleet_mask=sample_batch.fleet_mask,
    )
    lr_schedule = make_lr_schedule(lr, total_steps, lr_schedule, warmup_steps, min_lr_ratio)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_schedule))
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def bernoulli_log_prob(logits, action_bool):
    """计算 Bernoulli 动作的 log-prob。"""
    action = action_bool.astype(jnp.float32)
    return -optax.sigmoid_binary_cross_entropy(logits, action)


def categorical_log_prob(logits, action):
    """按离散动作索引从 log_softmax 中取 log-prob。"""
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1).squeeze(-1)


def categorical_entropy(logits):
    """计算离散分布熵，用于 PPO 的探索奖励。"""
    logp = jax.nn.log_softmax(logits)
    p = jnp.exp(logp)
    return -(p * logp).sum(axis=-1)


def safe_logits(logits):
    """修复全非法 logits 行，避免 softmax/categorical 产生 NaN。"""
    _assert_all_finite("ppo/logits", logits)
    valid = jnp.isfinite(logits) & (logits > NEG_INF * 0.5)
    invalid = ~valid.any(axis=-1)
    fixed = jnp.where(invalid[..., None], NEG_INF, logits)
    return fixed.at[..., 0].set(jnp.where(invalid, 0.0, fixed[..., 0]))


def _fire_entropy(logits):
    """二分类 head 的熵。end head 和旧 fire head 都可复用这个形式。"""
    p = jax.nn.sigmoid(logits)
    return -(p * jnp.log(jnp.clip(p, 1e-8, 1.0)) + (1.0 - p) * jnp.log(jnp.clip(1.0 - p, 1e-8, 1.0)))


def compute_gae(rewards, dones, values, last_values, params: PPOParams):
    """GAE 优势估计，倒序扫描 [T, B, P] 张量。

    delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
    A_t = delta_t + gamma * lambda * (1-done) * A_{t+1}
    lax.scan 倒序遍历自动运行编译，无需 Python 循环。
    """

    def scan_body(carry, item):
        next_adv, next_value = carry
        reward, done, value = item
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + params.gamma * next_value * not_done - value
        adv = delta + params.gamma * params.gae_lambda * not_done * next_adv
        return (adv, value), adv

    init = (jnp.zeros_like(last_values), last_values)
    (_, _), adv_rev = jax.lax.scan(scan_body, init, (rewards[::-1], dones[::-1], values[::-1]))
    advantages = adv_rev[::-1]
    returns = advantages + values
    return returns, advantages


def compute_gae_state(rewards, dones, values, last_values, params: PPOParams):
    """状态级 GAE：自回归 actor 的 value 是每个环境状态一个标量，形状为 [T, B]。"""
    return compute_gae(rewards, dones, values, last_values, params)


@partial(jax.jit, static_argnames=("params", "max_action_tokens"))
def ppo_update_auto(train_state: TrainState, rollout: AutoRollout, returns, advantages, params: PPOParams, rng, max_action_tokens: int, ent_scale=1.0):
    """对自回归 rollout 执行一次 PPO 更新。

    一个 update 内包含多个 epoch，每个 epoch 随机打乱状态样本并切 minibatch。
    KL 超过阈值时会提前停止后续 epoch，防止策略单次更新过大。
    """
    flat = flatten_auto_rollout(rollout, returns, advantages)
    size = flat["source_mask"].shape[0]
    states_per_minibatch = min(max(params.minibatch_size, 1), size)
    num_minibatches = max(size // states_per_minibatch, 1)
    keys = [
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "entropy_end",
        "entropy_src",
        "entropy_tgt",
        "approx_kl",
        "clipfrac",
        "max_abs_log_ratio",
        "max_abs_log_ratio_end",
        "max_abs_log_ratio_src",
        "max_abs_log_ratio_tgt",
        "invalid_action_rate",
        "executed_action_rate",
        "seq_len_mean",
    ]

    def run_epoch(ts, rng_in):
        rng_in, rng_perm = jax.random.split(rng_in)
        order = jax.random.permutation(rng_perm, size)

        def mb_body(i, mb_carry):
            ts2, acc = mb_carry
            idx = jax.lax.dynamic_slice(order, (i * states_per_minibatch,), (states_per_minibatch,))
            batch = {k: v[idx] for k, v in flat.items()}
            ts2, metrics = update_auto_minibatch(ts2, batch, params, ent_scale)
            return ts2, {k: acc[k] + metrics[k] for k in acc}

        init_acc = {k: jnp.array(0.0, dtype=jnp.float32) for k in keys}
        ts, sums = jax.lax.fori_loop(0, num_minibatches, mb_body, (ts, init_acc))
        means = {k: sums[k] / num_minibatches for k in sums}
        return ts, rng_in, means

    def cond_fn(carry):
        completed, _ts, _rng, _acc, last_kl = carry
        under_epoch_limit = completed < params.epochs
        kl_ok = (params.target_kl <= 0.0) | (completed == 0) | (last_kl <= 1.5 * params.target_kl)
        return under_epoch_limit & kl_ok

    def body_fn(carry):
        completed, ts, rng_in, acc, _last_kl = carry
        ts, rng_in, epoch_metrics = run_epoch(ts, rng_in)
        acc = {k: acc[k] + epoch_metrics[k] for k in keys}
        completed = completed + 1
        return completed, ts, rng_in, acc, epoch_metrics["approx_kl"]

    init_acc = {k: jnp.array(0.0, dtype=jnp.float32) for k in keys}
    completed, train_state, rng, metrics_acc, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (jnp.array(0, dtype=jnp.int32), train_state, rng, init_acc, jnp.array(0.0, dtype=jnp.float32)),
    )
    denom = jnp.maximum(completed.astype(jnp.float32), 1.0)
    metrics = {k: metrics_acc[k] / denom for k in metrics_acc}
    metrics["n_epochs_run"] = completed.astype(jnp.float32)
    return train_state, metrics


def update_auto_minibatch(train_state: TrainState, batch: dict[str, jnp.ndarray], params: PPOParams, ent_scale=1.0):
    """单个 minibatch 的自回归 PPO 损失与梯度更新。"""

    def loss_fn(model_params):
        outputs = train_state.apply_fn(
            {"params": model_params},
            batch["planet_features"],
            batch["relative_features"],
            batch["global_features"],
            batch["candidate_mask"],
            batch["source_mask"],
            batch["end_action"],
            batch["src_action"],
            batch["tgt_action"],
            batch["executed"],
            batch["initial_ships"],
            map_features=batch["map_features"],
            fleet_features=batch["fleet_features"],
            fleet_mask=batch["fleet_mask"],
        )
        end_logp = bernoulli_log_prob(outputs.end_logits, batch["end_action"])
        src_logits = safe_logits(outputs.src_logits)
        tgt_logits = safe_logits(outputs.tgt_logits)
        src_logp = categorical_log_prob(src_logits, batch["src_action"])
        tgt_logp = categorical_log_prob(tgt_logits, batch["tgt_action"])

        token_valid = batch["token_valid"].astype(jnp.float32)
        end_mask = token_valid              # end head 对所有“已解码 token”学习。
        arg_mask = batch["executed"].astype(jnp.float32)  # src/tgt 只学习真实执行动作。
        adv = batch["advantages"]
        if params.norm_adv:
            adv_mean = adv.mean()
            adv = (adv - adv_mean) / (adv.std() + 1e-8)
        adv_k = adv[:, None]

        def token_loss(new_logp, old_logp, mask):
            """单个 head 的 PPO-clip 损失、KL、clipfrac 和 log-ratio 统计。"""
            raw_lr = jnp.where(mask > 0.0, new_logp - old_logp, 0.0)
            lr = jnp.clip(raw_lr, -5.0, jnp.log(jnp.array(10.0, dtype=jnp.float32)))
            ratio = jnp.exp(lr)
            pg1 = adv_k * ratio
            pg2 = adv_k * jnp.clip(ratio, 1.0 - params.clip_coef, 1.0 + params.clip_coef)
            denom = jnp.maximum(mask.sum(), 1.0)
            loss = -(jnp.minimum(pg1, pg2) * mask).sum() / denom
            kl = (((ratio - 1.0) - lr) * mask).sum() / denom
            clipfrac = (((jnp.abs(ratio - 1.0) > params.clip_coef).astype(jnp.float32)) * mask).sum() / denom
            max_abs = (jnp.abs(raw_lr) * mask).max()
            return loss, kl, clipfrac, max_abs

        l_end, kl_end, cf_end, mr_end = token_loss(end_logp, batch["old_logp_end"], end_mask)
        l_src, kl_src, cf_src, mr_src = token_loss(src_logp, batch["old_logp_src"], arg_mask)
        l_tgt, kl_tgt, cf_tgt, mr_tgt = token_loss(tgt_logp, batch["old_logp_tgt"], arg_mask)
        policy_loss = (l_end + l_src + l_tgt) / 3.0

        value = _assert_all_finite("ppo_auto/value", outputs.value)
        value_pred = batch["old_value"] + jnp.clip(value - batch["old_value"], -params.clip_coef, params.clip_coef)
        value_loss = jnp.square(batch["returns"] - value_pred).mean()

        ent_end = _fire_entropy(outputs.end_logits)
        ent_src = categorical_entropy(src_logits)
        ent_tgt = categorical_entropy(tgt_logits)
        end_denom = jnp.maximum(end_mask.sum(), 1.0)
        arg_denom = jnp.maximum(arg_mask.sum(), 1.0)
        entropy_end = (ent_end * end_mask).sum() / end_denom
        entropy_src = (ent_src * arg_mask).sum() / arg_denom
        entropy_tgt = (ent_tgt * arg_mask).sum() / arg_denom
        entropy = entropy_end + entropy_src + entropy_tgt
        entropy_bonus = ent_scale * (
            params.ent_coef_fire * entropy_end
            + params.ent_coef_target * entropy_tgt
            + params.ent_coef * entropy_src
        )
        loss = policy_loss + params.vf_coef * value_loss - entropy_bonus
        loss = _assert_all_finite("ppo_auto/loss", loss)
        approx_kl = (kl_end + kl_src + kl_tgt) / 3.0
        clipfrac = (cf_end + cf_src + cf_tgt) / 3.0
        max_abs_log_ratio = jnp.maximum(jnp.maximum(mr_end, mr_src), mr_tgt)
        non_end = (batch["token_valid"] & (~batch["end_action"])).astype(jnp.float32)
        non_end_denom = jnp.maximum(non_end.sum(), 1.0)
        invalid_action_rate = (batch["invalid"].astype(jnp.float32) * non_end).sum() / non_end_denom
        executed_action_rate = arg_mask.sum() / non_end_denom
        seq_len_mean = non_end.sum(axis=-1).mean()
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "entropy_end": entropy_end,
            "entropy_src": entropy_src,
            "entropy_tgt": entropy_tgt,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "max_abs_log_ratio": max_abs_log_ratio,
            "max_abs_log_ratio_end": mr_end,
            "max_abs_log_ratio_src": mr_src,
            "max_abs_log_ratio_tgt": mr_tgt,
            "invalid_action_rate": invalid_action_rate,
            "executed_action_rate": executed_action_rate,
            "seq_len_mean": seq_len_mean,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    del loss
    grads = jax.tree_util.tree_map(lambda g: _assert_all_finite("ppo_auto/grads", g), grads)
    return train_state.apply_gradients(grads=grads), metrics


def flatten_auto_rollout(rollout: AutoRollout, returns, advantages):
    """把 [T, B, ...] rollout 展平成 [T*B, ...]，便于 minibatch 采样。"""
    def flat(x):
        return x.reshape((-1,) + x.shape[2:]) if x.ndim >= 2 else x.reshape((-1,))

    return {
        "planet_features": flat(rollout.planet_features),
        "relative_features": flat(rollout.relative_features),
        "global_features": flat(rollout.global_features),
        "map_features": flat(rollout.map_features),
        "fleet_features": flat(rollout.fleet_features),
        "fleet_mask": flat(rollout.fleet_mask),
        "candidate_mask": flat(rollout.candidate_mask),
        "source_mask": flat(rollout.source_mask),
        "initial_ships": flat(rollout.initial_ships),
        "end_action": flat(rollout.end_action),
        "src_action": flat(rollout.src_action),
        "tgt_action": flat(rollout.tgt_action),
        "executed": flat(rollout.executed),
        "invalid": flat(rollout.invalid),
        "old_logp_end": flat(rollout.old_logp_end),
        "old_logp_src": flat(rollout.old_logp_src),
        "old_logp_tgt": flat(rollout.old_logp_tgt),
        "token_valid": flat(rollout.token_valid),
        "old_value": flat(rollout.old_value),
        "returns": flat(returns),
        "advantages": flat(advantages),
    }
