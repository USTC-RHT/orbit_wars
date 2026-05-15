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
    vf_coef: float = 0.5       # 价值函数损失系数
    max_grad_norm: float = 0.5 # 梯度裁剔阈値
    epochs: int = 4            # 每次 rollout 训练轮数
    minibatch_size: int = 4096 # 单个 minibatch 大小
    norm_adv: bool = True      # 是否对 advantage 做 batch 内平均方差归一化
    target_kl: float = 0.05    # KL 早停阈值；epoch 平均 approx_kl > 1.5*target_kl 时停止剩余 epoch


class Rollout(NamedTuple):
    """单步 rollout 数据，層叠后形状为 [T, B, P, ...]。"""
    self_features: jnp.ndarray      # [T, B, P, SELF_DIM]
    candidate_features: jnp.ndarray # [T, B, P, P+1, CAND_DIM]
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


def make_train_state(rng, model, sample_batch, lr: float, max_grad_norm: float, total_steps: int = 0) -> TrainState:
    variables = model.init(
        rng,
        sample_batch.self_features,
        sample_batch.candidate_features,
        sample_batch.global_features,
        sample_batch.candidate_mask,
        sample_batch.map_features,
        jnp.zeros(sample_batch.self_features.shape[:-1], dtype=jnp.int32),
        fleet_features=sample_batch.fleet_features,
        fleet_mask=sample_batch.fleet_mask,
    )
    if total_steps > 0:
        lr_schedule = optax.linear_schedule(init_value=lr, end_value=0.0, transition_steps=total_steps)
    else:
        lr_schedule = lr
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


@partial(jax.jit, static_argnames=("params",))
def ppo_update(train_state: TrainState, rollout: Rollout, returns, advantages, params: PPOParams, rng):
    """PPO 多 epoch 更新，带 JIT 内 KL 早停。"""
    flat = flatten_rollout(rollout, returns, advantages)
    size = flat["source_mask"].shape[0]
    source_count = flat["source_mask"].shape[1]
    states_per_minibatch = min(max(params.minibatch_size // source_count, 1), size)
    num_minibatches = max(size // states_per_minibatch, 1)
    keys = ["loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"]

    def run_epoch(ts, rng_in):
        rng_in, rng_perm = jax.random.split(rng_in)
        order = jax.random.permutation(rng_perm, size)

        def mb_body(i, mb_carry):
            ts2, acc = mb_carry
            idx = jax.lax.dynamic_slice(order, (i * states_per_minibatch,), (states_per_minibatch,))
            batch = {k: v[idx] for k, v in flat.items()}
            ts2, metrics = update_minibatch(ts2, batch, params)
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


def update_minibatch(train_state: TrainState, batch: dict[str, jnp.ndarray], params: PPOParams):
    """单个 minibatch 梯度更新。"""
    def loss_fn(model_params):
        zero_target = jnp.zeros_like(batch["target"])
        value_outputs = train_state.apply_fn(
            {"params": model_params},
            batch["self_features"],
            batch["candidate_features"],
            batch["global_features"],
            batch["candidate_mask"],
            batch["map_features"],
            zero_target,
            fleet_features=batch["fleet_features"],
            fleet_mask=batch["fleet_mask"],
        )
        action_outputs = train_state.apply_fn(
            {"params": model_params},
            batch["self_features"],
            batch["candidate_features"],
            batch["global_features"],
            batch["candidate_mask"],
            batch["map_features"],
            batch["target"],
            fleet_features=batch["fleet_features"],
            fleet_mask=batch["fleet_mask"],
        )
        logp, entropy = log_prob_entropy(action_outputs, batch["fire"], batch["target"], batch["amount"], batch["candidate_mask"], batch["amount_mask"])
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
        log_ratio = jnp.where(policy_mask_bool, logp - batch["old_logp"], 0.0)
        # [数值安全] 截断 log_ratio 防止 exp() 在 float32 下溢出（exp(88) > 1e38）。
        # 训练中后期 policy 变锐利时，rare-action 样本会出现 log_ratio > 88 → ratio=inf → loss=nan。
        # 这是 0513 训练崩溃的直接原因（参考 _assert_all_finite("ppo/loss") 抛出 FloatingPointError）。
        log_ratio = jnp.clip(log_ratio, -20.0, 20.0)
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
        loss = policy_loss + params.vf_coef * value_loss - params.ent_coef * entropy_mean
        loss = _assert_all_finite("ppo/loss", loss)
        approx_kl = (((ratio - 1.0) - log_ratio) * policy_mask).sum() / policy_denom
        clipfrac = (((jnp.abs(ratio - 1.0) > params.clip_coef).astype(jnp.float32)) * policy_mask).sum() / policy_denom
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_mean,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    del loss
    grads = jax.tree_util.tree_map(lambda g: _assert_all_finite("ppo/grads", g), grads)
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
    entropy = _fire_entropy(outputs.fire_logits) + fire.astype(jnp.float32) * (categorical_entropy(target_logits) + categorical_entropy(amount_logits))
    return logp, entropy


def flatten_rollout(rollout: Rollout, returns, advantages):
    def flat(x):
        return x.reshape((-1,) + x.shape[2:]) if x.ndim >= 2 else x.reshape((-1,))

    data = {
        "self_features": flat(rollout.self_features),
        "candidate_features": flat(rollout.candidate_features),
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
