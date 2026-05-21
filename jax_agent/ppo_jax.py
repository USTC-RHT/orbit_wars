"""Orbit Wars JAX PPO 训练工具。

包含：
- PPOParams / Rollout 数据结构。
- make_train_state：初始化 Flax TrainState + Optax 优化器。
- compute_gae：lax.scan 倒序计算 GAE Advantage。
- ppo_update：multi-epoch fori_loop + minibatch fori_loop 更新参数。
- log_prob_entropy：计算 fire/target/amount 联合对数概率和熵。
- 采样逻辑直接内联在 train_jax.policy_sample 中。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

NEG_INF = -1.0e9


def _assert_all_finite(name: str, value):
    """Raise during JIT execution if a tensor contains NaN/Inf."""
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
    gae_lambda: float = 0.95   # GAE 将来奖励平滞系数
    clip_coef: float = 0.2     # PPO 策略梯度剥裁系数 ε
    ent_coef: float = 0.01     # 熵奖励系数（鼓励探索）
    ent_coef_fire: float = 0.01
    ent_coef_target: float = 0.01
    ent_coef_amount: float = 0.01
    vf_coef: float = 0.5       # 价值函数损失系数
    max_grad_norm: float = 0.5 # 梯度裁剔阈値
    epochs: int = 4            # 每次 rollout 训练轮数
    minibatch_size: int = 4096 # 单个 minibatch 大小
    norm_adv: bool = True      # 是否对 advantage 做 batch 内平均方差归一化
    target_kl: float = 0.05    # KL 早停阈值；epoch 平均 approx_kl > 1.5*target_kl 时停止剩余 epoch


class Rollout(NamedTuple):
    """单步 rollout 数据，層叠后形状为 [T, B, P, ...]。"""
    planet_features: jnp.ndarray      # [T, B, P, PLANET_DIM]
    relative_features: jnp.ndarray # [T, B, P, P+1, CAND_DIM]
    global_features: jnp.ndarray    # [T, B, P, GLOBAL_DIM]
    map_features: jnp.ndarray       # [T, B, P, C, H, W]
    fleet_features: jnp.ndarray     # [T, B, P, K, FLEET_DIM]
    fleet_mask: jnp.ndarray         # [T, B, P, K]
    candidate_mask: jnp.ndarray     # [T, B, P, P+1]
    amount_mask: jnp.ndarray        # [T, B, P, P+1, B]
    source_mask: jnp.ndarray        # [T, B, P] 我方星球标记
    policy_mask: jnp.ndarray        # [T, B, P] 实际执行/真实 no-op 的动作才参与 policy loss
    fire: jnp.ndarray               # [T, B, P] bool
    target: jnp.ndarray             # [T, B, P] 目标槽位索引
    amount: jnp.ndarray             # [T, B, P] 派兵档位索引
    old_logp: jnp.ndarray           # [T, B, P] 采样时记录的 log 概率
    old_value: jnp.ndarray          # [T, B, P] 采样时记录的价值估计
    reward: jnp.ndarray             # [T, B, P] 每步奖励
    done: jnp.ndarray               # [T, B, P] 是否终局


class AutoRollout(NamedTuple):
    """Autoregressive rollout data, stacked as [T, B, ...]."""
    planet_features: jnp.ndarray
    relative_features: jnp.ndarray
    global_features: jnp.ndarray
    map_features: jnp.ndarray
    fleet_features: jnp.ndarray
    fleet_mask: jnp.ndarray
    candidate_mask: jnp.ndarray
    amount_mask: jnp.ndarray
    source_mask: jnp.ndarray
    initial_ships: jnp.ndarray
    end_action: jnp.ndarray
    src_action: jnp.ndarray
    tgt_action: jnp.ndarray
    amt_action: jnp.ndarray
    executed: jnp.ndarray
    invalid: jnp.ndarray
    old_logp_end: jnp.ndarray
    old_logp_src: jnp.ndarray
    old_logp_tgt: jnp.ndarray
    old_logp_amt: jnp.ndarray
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
    """Build an Optax learning-rate schedule.

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


def make_train_state(
    rng,
    model,
    sample_batch,
    lr: float,
    max_grad_norm: float,
    total_steps: int = 0,
    lr_schedule: str = "constant",
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> TrainState:
    variables = model.init(
        rng,
        sample_batch.planet_features,
        sample_batch.relative_features,
        sample_batch.global_features,
        sample_batch.candidate_mask,
        sample_batch.map_features,
        jnp.zeros(sample_batch.planet_features.shape[:-1], dtype=jnp.int32),
        fleet_features=sample_batch.fleet_features,
        fleet_mask=sample_batch.fleet_mask,
    )
    lr_schedule = make_lr_schedule(lr, total_steps, lr_schedule, warmup_steps, min_lr_ratio)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_schedule))
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


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
    b = sample_batch.planet_features.shape[0]
    k = max_action_tokens
    variables = model.init(
        rng,
        sample_batch.planet_features,
        sample_batch.relative_features,
        sample_batch.global_features,
        sample_batch.candidate_mask,
        sample_batch.amount_mask,
        sample_batch.source_mask,
        jnp.zeros((b, k), dtype=bool),
        jnp.zeros((b, k), dtype=jnp.int32),
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
    action = action_bool.astype(jnp.float32)
    return -optax.sigmoid_binary_cross_entropy(logits, action)


def categorical_sample(rng, logits, deterministic: bool):
    return jnp.argmax(logits, axis=-1) if deterministic else jax.random.categorical(rng, logits, axis=-1)


def categorical_log_prob(logits, action):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1).squeeze(-1)


def categorical_entropy(logits):
    logp = jax.nn.log_softmax(logits)
    p = jnp.exp(logp)
    return -(p * logp).sum(axis=-1)


def safe_logits(logits):
    _assert_all_finite("ppo/logits", logits)
    invalid = ~jnp.isfinite(logits).any(axis=-1)
    fixed = jnp.where(invalid[..., None], NEG_INF, logits)
    return fixed.at[..., 0].set(jnp.where(invalid, 0.0, fixed[..., 0]))


def fire_target_mask(candidate_mask):
    return candidate_mask.at[..., 0].set(False)


def _fire_entropy(logits):
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
    """State-level GAE for autoregressive actor: tensors are [T, B]."""
    return compute_gae(rewards, dones, values, last_values, params)


@partial(jax.jit, static_argnames=("params",))
def ppo_update(train_state: TrainState, rollout: Rollout, returns, advantages, params: PPOParams, rng, ent_scale=1.0):
    """PPO 多 epoch 更新，带 JIT 内 KL 早停。"""
    flat = flatten_rollout(rollout, returns, advantages)
    size = flat["source_mask"].shape[0]
    source_count = flat["source_mask"].shape[1]
    states_per_minibatch = min(max(params.minibatch_size // source_count, 1), size)
    num_minibatches = max(size // states_per_minibatch, 1)
    keys = [
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "entropy_fire",
        "entropy_target",
        "entropy_amount",
        "approx_kl",
        "clipfrac",
        "max_abs_log_ratio",
    ]

    def run_epoch(ts, rng_in):
        rng_in, rng_perm = jax.random.split(rng_in)
        order = jax.random.permutation(rng_perm, size)

        def mb_body(i, mb_carry):
            ts2, acc = mb_carry
            idx = jax.lax.dynamic_slice(order, (i * states_per_minibatch,), (states_per_minibatch,))
            batch = {k: v[idx] for k, v in flat.items()}
            ts2, metrics = update_minibatch(ts2, batch, params, ent_scale)
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
    init_carry = (
        jnp.array(0, dtype=jnp.int32),
        train_state,
        rng,
        init_acc,
        jnp.array(0.0, dtype=jnp.float32),
    )
    completed, train_state, rng, metrics_acc, _last_kl = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    denom = jnp.maximum(completed.astype(jnp.float32), 1.0)
    metrics = {k: metrics_acc[k] / denom for k in metrics_acc}
    metrics["n_epochs_run"] = completed.astype(jnp.float32)
    return train_state, metrics


@partial(jax.jit, static_argnames=("params", "max_action_tokens"))
def ppo_update_auto(train_state: TrainState, rollout: AutoRollout, returns, advantages, params: PPOParams, rng, max_action_tokens: int, ent_scale=1.0):
    """PPO update for autoregressive state-level rollouts."""
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
        "entropy_amt",
        "approx_kl",
        "clipfrac",
        "max_abs_log_ratio",
        "max_abs_log_ratio_end",
        "max_abs_log_ratio_src",
        "max_abs_log_ratio_tgt",
        "max_abs_log_ratio_amt",
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


def update_minibatch(train_state: TrainState, batch: dict[str, jnp.ndarray], params: PPOParams, ent_scale=1.0):
    """单个 minibatch 梯度更新。"""
    def loss_fn(model_params):
        zero_target = jnp.zeros_like(batch["target"])
        value_outputs = train_state.apply_fn(
            {"params": model_params},
            batch["planet_features"],
            batch["relative_features"],
            batch["global_features"],
            batch["candidate_mask"],
            batch["map_features"],
            zero_target,
            fleet_features=batch["fleet_features"],
            fleet_mask=batch["fleet_mask"],
        )
        action_outputs = train_state.apply_fn(
            {"params": model_params},
            batch["planet_features"],
            batch["relative_features"],
            batch["global_features"],
            batch["candidate_mask"],
            batch["map_features"],
            batch["target"],
            fleet_features=batch["fleet_features"],
            fleet_mask=batch["fleet_mask"],
        )
        logp, entropy, entropy_fire, entropy_target, entropy_amount = log_prob_entropy(
            action_outputs,
            batch["fire"],
            batch["target"],
            batch["amount"],
            batch["candidate_mask"],
            batch["amount_mask"],
        )
        logp = _assert_all_finite("ppo/logp", logp)
        entropy = _assert_all_finite("ppo/entropy", entropy)
        adv = batch["advantages"]
        policy_mask_bool = batch["policy_mask"]  # 只让真实执行动作/真实 no-op 更新策略
        value_mask_bool = batch["source_mask"]    # value 仍学习所有我方 source 的状态回报
        policy_mask = policy_mask_bool.astype(jnp.float32)
        value_mask = value_mask_bool.astype(jnp.float32)
        if params.norm_adv:
            # batch 内平均方差归一化优势，稳定训练
            denom_adv = jnp.maximum(policy_mask.sum(), 1.0)
            adv_mean = (adv * policy_mask).sum() / denom_adv
            adv_var = (jnp.square(adv - adv_mean) * policy_mask).sum() / denom_adv
            adv = (adv - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
        raw_log_ratio = jnp.where(policy_mask_bool, logp - batch["old_logp"], 0.0)
        # [数值安全] 先截断 log_ratio 再 exp，避免 rare-action 样本把 ratio/KL 单点打爆。
        # 上界 log(10) 等价于 ratio <= 10，下界 -5 等价于 ratio >= 0.0067。
        log_ratio = jnp.clip(raw_log_ratio, -5.0, jnp.log(jnp.array(10.0, dtype=jnp.float32)))
        ratio = jnp.exp(log_ratio)  # 新旧策略概率比
        pg1 = adv * ratio
        pg2 = adv * jnp.clip(ratio, 1.0 - params.clip_coef, 1.0 + params.clip_coef)  # 弹性 clip
        policy_denom = jnp.maximum(policy_mask.sum(), 1.0)
        value_denom = jnp.maximum(value_mask.sum(), 1.0)
        policy_loss = -(jnp.minimum(pg1, pg2) * policy_mask).sum() / policy_denom  # PPO-clip 目标
        # 价值函数：clip 旧价值，限制单步更新幅度
        # rollout 里的 old_value 来自第一次前向（selected_target=0），
        # PPO 更新也用同一条件的 value，避免 value clipping 两侧条件化不一致。
        value = _assert_all_finite("ppo/value", value_outputs.value)
        value_pred = batch["old_value"] + jnp.clip(value - batch["old_value"], -params.clip_coef, params.clip_coef)
        value_loss = (jnp.square(batch["returns"] - value_pred) * value_mask).sum() / value_denom
        entropy_mean = (entropy * policy_mask).sum() / policy_denom
        fired_policy_mask = policy_mask * batch["fire"].astype(jnp.float32)
        entropy_fire_mean = (entropy_fire * policy_mask).sum() / policy_denom
        entropy_target_mean = (entropy_target * fired_policy_mask).sum() / policy_denom
        entropy_amount_mean = (entropy_amount * fired_policy_mask).sum() / policy_denom
        entropy_bonus = ent_scale * (
            params.ent_coef_fire * entropy_fire_mean
            + params.ent_coef_target * entropy_target_mean
            + params.ent_coef_amount * entropy_amount_mean
        )
        loss = policy_loss + params.vf_coef * value_loss - entropy_bonus
        loss = _assert_all_finite("ppo/loss", loss)
        approx_kl = (((ratio - 1.0) - log_ratio) * policy_mask).sum() / policy_denom
        clipfrac = (((jnp.abs(ratio - 1.0) > params.clip_coef).astype(jnp.float32)) * policy_mask).sum() / policy_denom
        max_abs_log_ratio = (jnp.abs(raw_log_ratio) * policy_mask).max()
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "entropy_fire": entropy_fire_mean,
            "entropy_target": entropy_target_mean,
            "entropy_amount": entropy_amount_mean,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "max_abs_log_ratio": max_abs_log_ratio,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    del loss
    grads = jax.tree_util.tree_map(lambda g: _assert_all_finite("ppo/grads", g), grads)
    return train_state.apply_gradients(grads=grads), metrics


def update_auto_minibatch(train_state: TrainState, batch: dict[str, jnp.ndarray], params: PPOParams, ent_scale=1.0):
    """Single minibatch PPO update for autoregressive actor."""

    def loss_fn(model_params):
        outputs = train_state.apply_fn(
            {"params": model_params},
            batch["planet_features"],
            batch["relative_features"],
            batch["global_features"],
            batch["candidate_mask"],
            batch["amount_mask"],
            batch["source_mask"],
            batch["end_action"],
            batch["src_action"],
            batch["tgt_action"],
            batch["amt_action"],
            batch["executed"],
            batch["initial_ships"],
            map_features=batch["map_features"],
            fleet_features=batch["fleet_features"],
            fleet_mask=batch["fleet_mask"],
        )
        end_logp = bernoulli_log_prob(outputs.end_logits, batch["end_action"])
        src_logits = safe_logits(outputs.src_logits)
        tgt_logits = safe_logits(outputs.tgt_logits)
        amt_logits = safe_logits(outputs.amount_logits)
        src_logp = categorical_log_prob(src_logits, batch["src_action"])
        tgt_logp = categorical_log_prob(tgt_logits, batch["tgt_action"])
        amt_logp = categorical_log_prob(amt_logits, batch["amt_action"])

        token_valid = batch["token_valid"].astype(jnp.float32)
        end_mask = token_valid
        arg_mask = batch["executed"].astype(jnp.float32)
        adv = batch["advantages"]
        if params.norm_adv:
            adv_mean = adv.mean()
            adv = (adv - adv_mean) / (adv.std() + 1e-8)
        adv_k = adv[:, None]

        def token_loss(new_logp, old_logp, mask):
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
        l_amt, kl_amt, cf_amt, mr_amt = token_loss(amt_logp, batch["old_logp_amt"], arg_mask)
        policy_loss = (l_end + l_src + l_tgt + l_amt) / 4.0

        value = _assert_all_finite("ppo_auto/value", outputs.value)
        value_pred = batch["old_value"] + jnp.clip(value - batch["old_value"], -params.clip_coef, params.clip_coef)
        value_loss = jnp.square(batch["returns"] - value_pred).mean()

        ent_end = _fire_entropy(outputs.end_logits)
        ent_src = categorical_entropy(src_logits)
        ent_tgt = categorical_entropy(tgt_logits)
        ent_amt = categorical_entropy(amt_logits)
        end_denom = jnp.maximum(end_mask.sum(), 1.0)
        arg_denom = jnp.maximum(arg_mask.sum(), 1.0)
        entropy_end = (ent_end * end_mask).sum() / end_denom
        entropy_src = (ent_src * arg_mask).sum() / arg_denom
        entropy_tgt = (ent_tgt * arg_mask).sum() / arg_denom
        entropy_amt = (ent_amt * arg_mask).sum() / arg_denom
        entropy = entropy_end + entropy_src + entropy_tgt + entropy_amt
        entropy_bonus = ent_scale * (
            params.ent_coef_fire * entropy_end
            + params.ent_coef_target * entropy_tgt
            + params.ent_coef_amount * entropy_amt
            + params.ent_coef * entropy_src
        )
        loss = policy_loss + params.vf_coef * value_loss - entropy_bonus
        loss = _assert_all_finite("ppo_auto/loss", loss)
        approx_kl = (kl_end + kl_src + kl_tgt + kl_amt) / 4.0
        clipfrac = (cf_end + cf_src + cf_tgt + cf_amt) / 4.0
        max_abs_log_ratio = jnp.maximum(jnp.maximum(mr_end, mr_src), jnp.maximum(mr_tgt, mr_amt))
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
            "entropy_amt": entropy_amt,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "max_abs_log_ratio": max_abs_log_ratio,
            "max_abs_log_ratio_end": mr_end,
            "max_abs_log_ratio_src": mr_src,
            "max_abs_log_ratio_tgt": mr_tgt,
            "max_abs_log_ratio_amt": mr_amt,
            "invalid_action_rate": invalid_action_rate,
            "executed_action_rate": executed_action_rate,
            "seq_len_mean": seq_len_mean,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    del loss
    grads = jax.tree_util.tree_map(lambda g: _assert_all_finite("ppo_auto/grads", g), grads)
    return train_state.apply_gradients(grads=grads), metrics


def log_prob_entropy(outputs, fire, target, amount, candidate_mask, amount_mask):
    """计算联合对数概率和熵。

    log p = log p(fire) + fire * [log p(target) + log p(amount)]
    不发射（fire=False）时不计 target/amount 的 logp，与采样逻辑严格一致。
    """
    fire_logp = bernoulli_log_prob(outputs.fire_logits, fire)
    # 重建采样时的 target logits（与 policy_sample 完全同步）
    target_logits = jnp.where(fire_target_mask(candidate_mask), outputs.target_logits, NEG_INF)
    target_logits = jnp.where(fire[..., None], target_logits, NEG_INF)
    target_logits = target_logits.at[..., 0].set(jnp.where(fire, target_logits[..., 0], 0.0))
    target_logits = safe_logits(target_logits)
    target_logp = categorical_log_prob(target_logits, target)
    selected_amount_mask = jnp.take_along_axis(amount_mask, target[..., None, None], axis=-2).squeeze(-2)
    amount_logits = jnp.where(selected_amount_mask, outputs.amount_logits, NEG_INF)
    amount_logits = jnp.where((~selected_amount_mask.any(axis=-1))[..., None], 0.0, amount_logits)
    amount_logp = categorical_log_prob(amount_logits, amount)
    # fire=False 时 target/amount 的 logp 不计入总 logp
    logp = fire_logp + fire.astype(jnp.float32) * (target_logp + amount_logp)
    entropy_fire = _fire_entropy(outputs.fire_logits)
    entropy_target = categorical_entropy(target_logits)
    entropy_amount = categorical_entropy(amount_logits)
    entropy = entropy_fire + fire.astype(jnp.float32) * (entropy_target + entropy_amount)
    return logp, entropy, entropy_fire, entropy_target, entropy_amount


def flatten_rollout(rollout: Rollout, returns, advantages):
    def flat(x):
        return x.reshape((-1,) + x.shape[2:]) if x.ndim >= 2 else x.reshape((-1,))

    data = {
        "planet_features": flat(rollout.planet_features),
        "relative_features": flat(rollout.relative_features),
        "global_features": flat(rollout.global_features),
        "map_features": flat(rollout.map_features),
        "fleet_features": flat(rollout.fleet_features),
        "fleet_mask": flat(rollout.fleet_mask),
        "candidate_mask": flat(rollout.candidate_mask),
        "amount_mask": flat(rollout.amount_mask),
        "source_mask": flat(rollout.source_mask),
        "policy_mask": flat(rollout.policy_mask),
        "fire": flat(rollout.fire),
        "target": flat(rollout.target),
        "amount": flat(rollout.amount),
        "old_logp": flat(rollout.old_logp),
        "old_value": flat(rollout.old_value),
        "returns": flat(returns),
        "advantages": flat(advantages),
    }
    return data


def flatten_auto_rollout(rollout: AutoRollout, returns, advantages):
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
        "amount_mask": flat(rollout.amount_mask),
        "source_mask": flat(rollout.source_mask),
        "initial_ships": flat(rollout.initial_ships),
        "end_action": flat(rollout.end_action),
        "src_action": flat(rollout.src_action),
        "tgt_action": flat(rollout.tgt_action),
        "amt_action": flat(rollout.amt_action),
        "executed": flat(rollout.executed),
        "invalid": flat(rollout.invalid),
        "old_logp_end": flat(rollout.old_logp_end),
        "old_logp_src": flat(rollout.old_logp_src),
        "old_logp_tgt": flat(rollout.old_logp_tgt),
        "old_logp_amt": flat(rollout.old_logp_amt),
        "token_valid": flat(rollout.token_valid),
        "old_value": flat(rollout.old_value),
        "returns": flat(returns),
        "advantages": flat(advantages),
    }
