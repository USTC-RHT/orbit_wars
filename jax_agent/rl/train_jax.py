"""Orbit Wars 每来源星球 actor 的 JAX PPO 训练入口。

这个文件只保留训练循环和少量 glue：环境、特征、模型、PPO update 都在各自模块。
底部的预测瞄准 helper 暂时留在这里，因为训练动作和 BC runtime 都复用这套
与引擎一致的轨迹判定。
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ..preprocessing.build_initial_state_bank import build_bank  # noqa: E402
from ..common.config import load_config  # noqa: E402
from ..common.fast_env import (  # noqa: E402
    COMETS_PER_SPAWN,
    MAX_COMET_PATH,
    MAX_COMET_SPAWNS,
    EnvParams,
    EnvState,
    comet_planet_mask,
    load_state_bank,
    point_segment_distance_sq,
    select_from_bank,
    step_batch,
)
from ..common.features_jax import (  # noqa: E402
    CANDIDATE_DIM,
    GLOBAL_DIM,
    MAP_CHANNELS,
    MAP_GRID_SIZE,
    SELF_DIM,
    FeatureParams,
    encode_batch,
    flatten_sources,
)
from ..common.model_jax import ActorCritic  # noqa: E402
from ..common.normalization_jax import (  # noqa: E402
    NormalizerParams,
    init_normalizer,
    normalize_encoded,
    update_and_normalize,
)
from .ppo_jax import NEG_INF, PPOParams, Rollout, compute_gae, log_prob_entropy, make_train_state, ppo_update, safe_logits  # noqa: E402
from .reward_jax import RewardParams, reward_from_transition  # noqa: E402
from .train_opponents import nearest_opponent_actions  # noqa: E402

AIM_HORIZON = 60
AIM_NEWTON_ITERS = 5
AIM_NEWTON_MIN_DERIV = 1e-4
AIM_NEWTON_MAX_STEP = 12.0
AIM_TARGET_INSET = 1e-3


def build_env_params(cfg, max_fleets: int, max_actions: int | None = None) -> EnvParams:
    return EnvParams(
        episode_steps=int(cfg.env.episode_steps),
        board_size=float(cfg.env.board_size),
        max_planets=int(cfg.env.max_planets),
        max_fleets=int(max_fleets),
        max_actions=int(max_actions or cfg.env.max_planets),
    )


def build_model(cfg, feat_params: FeatureParams) -> ActorCritic:
    return ActorCritic(
        self_dim=SELF_DIM,
        candidate_dim=CANDIDATE_DIM,
        global_dim=GLOBAL_DIM,
        candidate_count=feat_params.max_planets + 1,
        ship_bucket_count=feat_params.ship_bucket_count,
        hidden_size=int(cfg.model.hidden_size),
        use_cnn=bool(cfg.model.use_cnn),
        map_channels=MAP_CHANNELS,
        map_grid_size=MAP_GRID_SIZE,
    )


def build_ppo_params(cfg, rollout_size: int) -> PPOParams:
    return PPOParams(
        gamma=float(cfg.ppo.gamma),
        gae_lambda=float(cfg.ppo.gae_lambda),
        clip_coef=float(cfg.ppo.clip_coef),
        ent_coef=float(cfg.ppo.ent_coef),
        vf_coef=float(cfg.ppo.vf_coef),
        max_grad_norm=float(cfg.ppo.max_grad_norm),
        epochs=int(cfg.ppo.epochs),
        minibatch_size=max(1, min(int(cfg.ppo.minibatch_size), int(rollout_size))),
        norm_adv=bool(cfg.ppo.norm_adv),
        target_kl=float(cfg.ppo.target_kl),
    )


def maybe_build_bank(path: Path, bank_size: int, start_seed: int, env_params: EnvParams) -> None:
    if path.exists():
        return
    print(f"state bank missing, building {bank_size} seeds -> {path}", flush=True)
    build_bank(path, seeds=bank_size, start_seed=start_seed, params=env_params)


def sample_from_bank(bank: EnvState, rng: jax.Array, num_envs: int) -> EnvState:
    n = int(bank.planets.shape[0])
    indices = jax.random.randint(rng, (num_envs,), 0, n)
    return select_from_bank(bank, indices)


def fit_action_slots(actions: jnp.ndarray, max_actions: int) -> jnp.ndarray:
    current = actions.shape[-2]
    if current == max_actions:
        return actions
    if current > max_actions:
        return actions[..., :max_actions, :]
    pad_width = [(0, 0)] * actions.ndim
    pad_width[-2] = (0, max_actions - current)
    return jnp.pad(actions, pad_width)


def gather_axis1(values: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(values[:, None, :, :], indices[:, :, None, None], axis=2).squeeze(2)


def policy_sample(model: ActorCritic, params: Any, encoded, rng: jax.Array, deterministic: bool = False):
    batch_size, max_planets = encoded.self_features.shape[:2]
    flat = flatten_sources(encoded)
    n = flat.self_features.shape[0]
    zero_target = jnp.zeros((n,), dtype=jnp.int32)
    value_outputs = model.apply(
        {"params": params},
        flat.self_features,
        flat.candidate_features,
        flat.global_features,
        flat.candidate_mask,
        flat.map_features,
        zero_target,
    )

    rng_fire, rng_target, rng_amount = jax.random.split(rng, 3)
    can_fire = flat.source_mask & flat.candidate_mask[:, 1:].any(axis=-1)
    fire_prob = jax.nn.sigmoid(value_outputs.fire_logits)
    fire = fire_prob >= 0.5 if deterministic else jax.random.bernoulli(rng_fire, p=fire_prob)
    fire = fire & can_fire

    target_mask = flat.candidate_mask.at[:, 0].set(False)
    target_logits = jnp.where(target_mask, value_outputs.target_logits, NEG_INF)
    target_logits = jnp.where(fire[:, None], target_logits, NEG_INF)
    target_logits = target_logits.at[:, 0].set(jnp.where(fire, target_logits[:, 0], 0.0))
    target_logits = safe_logits(target_logits)
    target = jnp.argmax(target_logits, axis=-1) if deterministic else jax.random.categorical(rng_target, target_logits, axis=-1)

    action_outputs = model.apply(
        {"params": params},
        flat.self_features,
        flat.candidate_features,
        flat.global_features,
        flat.candidate_mask,
        flat.map_features,
        target,
    )
    selected_amount_mask = jnp.take_along_axis(flat.amount_mask, target[:, None, None], axis=1).squeeze(1)
    amount_logits = jnp.where(selected_amount_mask, action_outputs.amount_logits, NEG_INF)
    amount_logits = jnp.where((~selected_amount_mask.any(axis=-1))[:, None], 0.0, amount_logits)
    amount = jnp.argmax(amount_logits, axis=-1) if deterministic else jax.random.categorical(rng_amount, amount_logits, axis=-1)
    amount = jnp.where(fire, amount, 0)

    old_logp, _entropy = log_prob_entropy(action_outputs, fire, target, amount, flat.candidate_mask, flat.amount_mask)
    shape = (batch_size, max_planets)
    return (
        fire.reshape(shape),
        target.reshape(shape),
        amount.reshape(shape),
        old_logp.reshape(shape),
        value_outputs.value.reshape(shape),
    )


def actions_from_decisions(
    state: EnvState,
    encoded,
    fire: jnp.ndarray,
    target: jnp.ndarray,
    amount: jnp.ndarray,
    env_params: EnvParams,
) -> jnp.ndarray:
    target_slot = jnp.clip(target - 1, 0, encoded.source_ids.shape[1] - 1).astype(jnp.int32)
    target_planets = gather_axis1(state.planets, target_slot)
    source_planets = state.planets
    ships = jnp.take_along_axis(encoded.ship_buckets, amount[..., None], axis=2).squeeze(-1)
    target_valid = jnp.take_along_axis(encoded.candidate_mask, target[..., None], axis=2).squeeze(-1)
    valid = fire & encoded.source_mask & target_valid & (target > 0) & (ships > 0.0)
    angle, _hit = aim_with_prediction_jax(source_planets, target_planets, target_slot, state, ships)
    actions = jnp.stack(
        [
            jnp.where(valid, encoded.source_ids.astype(jnp.float32), 0.0),
            jnp.where(valid, angle, 0.0),
            jnp.where(valid, ships, 0.0),
        ],
        axis=-1,
    )
    return fit_action_slots(actions, env_params.max_actions)


def stack_rollout(rows: list[Rollout]) -> Rollout:
    return Rollout(**{field: jnp.stack([getattr(row, field) for row in rows], axis=0) for field in Rollout._fields})


def last_values(model: ActorCritic, params: Any, state: EnvState, normalizer_state, feat_params: FeatureParams, norm_params: NormalizerParams):
    encoded = normalize_encoded(encode_batch(state, feat_params), normalizer_state, norm_params)
    flat = flatten_sources(encoded)
    outputs = model.apply(
        {"params": params},
        flat.self_features,
        flat.candidate_features,
        flat.global_features,
        flat.candidate_mask,
        flat.map_features,
        jnp.zeros((flat.self_features.shape[0],), dtype=jnp.int32),
    )
    return outputs.value.reshape(encoded.self_features.shape[:2])


def collect_rollout(
    train_state,
    model: ActorCritic,
    state: EnvState,
    bank: EnvState,
    normalizer_state,
    rng: jax.Array,
    cfg,
    env_params: EnvParams,
    feat_params: FeatureParams,
    norm_params: NormalizerParams,
    reward_params: RewardParams,
) -> tuple[Rollout, EnvState, Any, jax.Array, dict[str, jnp.ndarray]]:
    rows: list[Rollout] = []
    reward_sum = jnp.asarray(0.0, dtype=jnp.float32)
    done_sum = jnp.asarray(0.0, dtype=jnp.float32)
    win_sum = jnp.asarray(0.0, dtype=jnp.float32)
    launch_sum = jnp.asarray(0.0, dtype=jnp.float32)
    overflow_sum = jnp.asarray(0.0, dtype=jnp.float32)

    for _ in range(int(cfg.ppo.rollout_steps)):
        rng, action_rng, reset_rng = jax.random.split(rng, 3)
        raw_encoded = encode_batch(state, feat_params)
        encoded, normalizer_state = update_and_normalize(raw_encoded, normalizer_state, norm_params)
        fire, target, amount, old_logp, old_value = policy_sample(model, train_state.params, encoded, action_rng)
        learner_actions = actions_from_decisions(state, encoded, fire, target, amount, env_params)
        opponent_actions = fit_action_slots(nearest_opponent_actions(state, 1, feat_params), env_params.max_actions)
        joint_actions = jnp.stack([learner_actions, opponent_actions], axis=1)
        next_state = step_batch(state, joint_actions, env_params)
        reward, reward_metrics = reward_from_transition(state, next_state, learner_actions, reward_params)
        done = next_state.done
        source_mask = encoded.source_mask
        reward_per_source = reward[:, None] * source_mask.astype(jnp.float32)
        done_per_source = jnp.broadcast_to(done[:, None], source_mask.shape)
        rows.append(
            Rollout(
                self_features=encoded.self_features,
                candidate_features=encoded.candidate_features,
                global_features=encoded.global_features,
                map_features=encoded.map_features,
                candidate_mask=encoded.candidate_mask,
                amount_mask=encoded.amount_mask,
                source_mask=source_mask,
                policy_mask=source_mask,
                fire=fire,
                target=target,
                amount=amount,
                old_logp=old_logp,
                old_value=old_value,
                reward=reward_per_source,
                done=done_per_source,
            )
        )
        reward_sum = reward_sum + reward.mean()
        done_count = done.astype(jnp.float32).sum()
        done_sum = done_sum + done_count
        win_sum = win_sum + ((next_state.winner == 0) & done).astype(jnp.float32).sum()
        launch_sum = launch_sum + reward_metrics["launch_count"].mean()
        overflow_sum = overflow_sum + next_state.overflow_fleets.astype(jnp.float32).mean()
        reset_state = sample_from_bank(bank, reset_rng, int(cfg.ppo.num_envs))
        state = replace_done(next_state, reset_state)

    steps = max(int(cfg.ppo.rollout_steps), 1)
    metrics = {
        "rollout_reward": reward_sum / steps,
        "done_games": done_sum,
        "win_rate": jnp.where(done_sum > 0.0, win_sum / done_sum, 0.0),
        "launch_count": launch_sum / steps,
        "overflow_fleets": overflow_sum / steps,
    }
    return stack_rollout(rows), state, normalizer_state, rng, metrics


def save_checkpoint(path: Path, train_state, normalizer_state, cfg, update: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": int(update),
        "params": jax.device_get(train_state.params),
        "normalizer": jax.device_get(normalizer_state),
        "config": asdict(cfg),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_jax_RL_1.yaml")
    parser.add_argument("--bank", default="outputs/jax_initial_state_bank.npz")
    parser.add_argument("--bank-size", type=int, default=256)
    parser.add_argument("--bank-start-seed", type=int, default=100000)
    parser.add_argument("--max-fleets", type=int, default=512)
    parser.add_argument("--updates", type=int, default=None, help="覆盖 ppo.total_updates，便于 smoke run。")
    parser.add_argument("--num-envs", type=int, default=None, help="覆盖 ppo.num_envs。")
    parser.add_argument("--rollout-steps", type=int, default=None, help="覆盖 ppo.rollout_steps。")
    parser.add_argument("--minibatch-size", type=int, default=None, help="覆盖 ppo.minibatch_size。")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-build-bank", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg, args: argparse.Namespace) -> None:
    if args.updates is not None:
        cfg.ppo.total_updates = int(args.updates)
    if args.num_envs is not None:
        cfg.ppo.num_envs = int(args.num_envs)
    if args.rollout_steps is not None:
        cfg.ppo.rollout_steps = int(args.rollout_steps)
    if args.minibatch_size is not None:
        cfg.ppo.minibatch_size = int(args.minibatch_size)
    if args.seed is not None:
        cfg.seed = int(args.seed)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    env_params = build_env_params(cfg, max_fleets=args.max_fleets)
    feat_params = FeatureParams.from_train_config(cfg)
    norm_params = NormalizerParams.from_train_config(cfg)
    reward_params = RewardParams.from_train_config(cfg)
    rollout_size = int(cfg.ppo.rollout_steps) * int(cfg.ppo.num_envs) * int(cfg.env.max_planets)
    ppo_params = build_ppo_params(cfg, rollout_size)
    bank_path = Path(args.bank)
    if not args.no_build_bank:
        maybe_build_bank(bank_path, int(args.bank_size), int(args.bank_start_seed), env_params)
    bank = load_state_bank(bank_path, env_params)
    rng = jax.random.PRNGKey(int(cfg.seed))
    rng, reset_rng, init_rng = jax.random.split(rng, 3)
    state = sample_from_bank(bank, reset_rng, int(cfg.ppo.num_envs))
    normalizer_state = init_normalizer()
    model = build_model(cfg, feat_params)
    sample_encoded = normalize_encoded(encode_batch(state, feat_params), normalizer_state, norm_params)
    train_state = make_train_state(
        init_rng,
        model,
        flatten_sources(sample_encoded),
        lr=float(cfg.ppo.lr),
        max_grad_norm=float(cfg.ppo.max_grad_norm),
        total_steps=0,
    )

    save_dir = Path(cfg.save_dir) / str(cfg.run_name)
    print(
        "training start "
        f"updates={cfg.ppo.total_updates} envs={cfg.ppo.num_envs} rollout={cfg.ppo.rollout_steps} "
        f"planets={cfg.env.max_planets} minibatch={ppo_params.minibatch_size} save_dir={save_dir}",
        flush=True,
    )
    start = time.time()
    for update in range(1, int(cfg.ppo.total_updates) + 1):
        tic = time.time()
        rollout, state, normalizer_state, rng, rollout_metrics = collect_rollout(
            train_state,
            model,
            state,
            bank,
            normalizer_state,
            rng,
            cfg,
            env_params,
            feat_params,
            norm_params,
            reward_params,
        )
        values = rollout.old_value
        last_val = last_values(model, train_state.params, state, normalizer_state, feat_params, norm_params)
        returns, advantages = compute_gae(rollout.reward, rollout.done, values, last_val, ppo_params)
        rng, update_rng = jax.random.split(rng)
        train_state, ppo_metrics = ppo_update(train_state, rollout, returns, advantages, ppo_params, update_rng)
        jax.block_until_ready(train_state.params)
        elapsed = time.time() - tic
        total_steps = update * int(cfg.ppo.rollout_steps) * int(cfg.ppo.num_envs)
        sps = int(int(cfg.ppo.rollout_steps) * int(cfg.ppo.num_envs) / max(elapsed, 1e-6))
        if update % int(cfg.log_every) == 0:
            metrics = {**rollout_metrics, **ppo_metrics}
            print(
                f"update={update:06d} steps={total_steps} sps={sps} "
                f"loss={float(metrics['loss']):.4f} reward={float(metrics['rollout_reward']):.4f} "
                f"win={float(metrics['win_rate']):.3f} done={float(metrics['done_games']):.0f} "
                f"launch={float(metrics['launch_count']):.2f} kl={float(metrics['approx_kl']):.5f} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
        if int(cfg.checkpoint_every) > 0 and update % int(cfg.checkpoint_every) == 0:
            save_checkpoint(save_dir / f"ckpt_{update:06d}.pkl", train_state, normalizer_state, cfg, update)

    save_checkpoint(save_dir / "ckpt_last.pkl", train_state, normalizer_state, cfg, int(cfg.ppo.total_updates))
    print(f"done, wrote {save_dir / 'ckpt_last.pkl'}", flush=True)


def aim_with_prediction_jax(src, target, target_idx, state: EnvState, ships):
    """与 src/physics.py 的 aim_with_prediction 同步的 JAX 批量瞄准。

    流程保持一致：
    1. 用 Newton 迭代求连续拦截时间 T*。
    2. 验证 floor(T*)-1/floor(T*)/floor(T*)+1 三个整数回合。
    3. 每个整数回合同时尝试瞄 target(T*)、target(k-1) 和 target(k)，
       覆盖连续拦截点、舰队移动命中和旋转星球 sweep 命中的关键几何面。
    4. 每个候选角都按引擎顺序检查太阳、出界、第一碰撞和目标碰撞。
    """

    horizon = AIM_HORIZON
    clearance = 0.1
    speed = fleet_speed_local(jnp.maximum(ships, 1.0))
    max_turns = target_max_turns_jax(state, target_idx, horizon).astype(jnp.float32)
    can_search = max_turns >= 1.0

    px0, py0, target0_valid = target_pos_at_time_jax(state, target_idx, jnp.zeros_like(max_turns))
    gap = src[..., 4] + clearance + target[..., 4]
    d0 = jnp.sqrt((px0 - src[..., 2]) ** 2 + (py0 - src[..., 3]) ** 2)
    t_star = jnp.clip((d0 - gap) / speed, 0.0, max_turns)

    def newton_body(_, t_cur):
        px, py, vx, vy, pos_valid = target_pos_velocity_at_time_jax(state, target_idx, t_cur)
        rx = px - src[..., 2]
        ry = py - src[..., 3]
        dist = jnp.sqrt(jnp.maximum(rx * rx + ry * ry, 1.0e-9))
        f_val = dist - gap - speed * t_cur
        radial_velocity = (rx * vx + ry * vy) / dist
        deriv = radial_velocity - speed
        raw_step = f_val / deriv
        max_step = jnp.minimum(jnp.maximum(max_turns, 1.0), AIM_NEWTON_MAX_STEP)
        step = jnp.clip(raw_step, -max_step, max_step)
        t_newton = jnp.clip(t_cur - step, 0.0, max_turns)
        use_newton = (
            pos_valid
            & jnp.isfinite(t_newton)
            & jnp.isfinite(raw_step)
            & jnp.isfinite(deriv)
            & (jnp.abs(deriv) > AIM_NEWTON_MIN_DERIV)
        )
        return jnp.where(use_newton, t_newton, t_cur)

    t_star = jax.lax.fori_loop(0, AIM_NEWTON_ITERS, newton_body, t_star)
    has_solution = can_search & target0_valid
    t_floor = jnp.floor(t_star).astype(jnp.int32)
    candidates = jnp.stack(
        [
            jnp.maximum(t_floor - 1, 1),
            jnp.maximum(t_floor, 1),
            t_floor + 1,
        ],
        axis=-1,
    )
    candidate_turns = candidates.astype(jnp.float32)
    candidate_in_range = (candidate_turns >= 1.0) & (candidate_turns <= max_turns[..., None])

    t_star_for_candidates = jnp.broadcast_to(t_star[..., None], candidate_turns.shape)
    aim_turns = jnp.stack(
        [
            t_star_for_candidates,
            jnp.maximum(candidate_turns - 1.0, 0.0),
            candidate_turns,
        ],
        axis=-1,
    )
    aim_px, aim_py, aim_pos_valid = target_pos_at_time_jax(state, target_idx, aim_turns)
    angle = jnp.arctan2(aim_py - src[..., None, None, 3], aim_px - src[..., None, None, 2])
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    candidate_turns_e = candidate_turns[..., None]
    candidates_e = candidates[..., None]
    fx0 = src[..., None, None, 2] + cos_a * (src[..., None, None, 4] + clearance)
    fy0 = src[..., None, None, 3] + sin_a * (src[..., None, None, 4] + clearance)
    fx_t = fx0 + candidate_turns_e * speed[..., None, None] * cos_a
    fy_t = fy0 + candidate_turns_e * speed[..., None, None] * sin_a

    sun_hit = point_segment_distance_sq(50.0, 50.0, fx0, fy0, fx_t, fy_t) < 100.0
    first_collision_ok = target_is_first_collision_jax(
        src,
        target_idx,
        state,
        fx0,
        fy0,
        speed[..., None, None],
        cos_a,
        sin_a,
        candidates_e,
    )

    fx_prev = fx0 + (candidate_turns_e - 1.0) * speed[..., None, None] * cos_a
    fy_prev = fy0 + (candidate_turns_e - 1.0) * speed[..., None, None] * sin_a
    px_old, py_old, target_pos_valid = target_pos_at_time_jax(state, target_idx, candidate_turns_e - 1.0)
    move_d2 = point_segment_distance_sq(px_old, py_old, fx_prev, fy_prev, fx_t, fy_t)
    target_hit_radius = jnp.maximum(target[..., None, None, 4] - AIM_TARGET_INSET, 0.0)
    move_hits_target = move_d2 < target_hit_radius ** 2

    px_next, py_next, target_next_valid = target_pos_at_time_jax(state, target_idx, candidate_turns_e)
    target_moved = ((px_next - px_old) ** 2 + (py_next - py_old) ** 2) > 1e-9
    sweep_d2 = point_segment_distance_sq(fx_t, fy_t, px_old, py_old, px_next, py_next)
    sweep_hits_target = (
        target_moved
        & target_next_valid
        & (sweep_d2 < target_hit_radius ** 2)
    )

    candidate_valid = (
        has_solution[..., None, None]
        & candidate_in_range[..., None]
        & aim_pos_valid
        & target_pos_valid
        & (~sun_hit)
        & first_collision_ok
        & (move_hits_target | sweep_hits_target)
        & (ships[..., None, None] > 0.0)
    )
    candidate_valid_flat = candidate_valid.reshape(candidate_valid.shape[:-2] + (9,))
    angle_flat = angle.reshape(angle.shape[:-2] + (9,))
    has_valid = candidate_valid_flat.any(axis=-1)
    chosen = jnp.argmax(candidate_valid_flat, axis=-1)
    chosen_angle = jnp.take_along_axis(angle_flat, chosen[..., None], axis=-1).squeeze(-1)
    direct_angle = jnp.arctan2(target[..., 3] - src[..., 3], target[..., 2] - src[..., 2])
    return jnp.where(has_valid, chosen_angle, direct_angle), has_valid


def aim_f_jax(src, target, target_idx, state: EnvState, speed, turns):
    """f(T)=dist(src,target(T))-(src.radius+clearance+T*speed)-target.radius。"""
    px, py, valid = target_pos_at_time_jax(state, target_idx, turns)
    src_x = _expand_like_time(src[..., 2], turns)
    src_y = _expand_like_time(src[..., 3], turns)
    src_radius = _expand_like_time(src[..., 4], turns)
    speed_t = _expand_like_time(speed, turns)
    target_radius = _expand_like_time(target[..., 4], turns)
    dist = jnp.sqrt((px - src_x) ** 2 + (py - src_y) ** 2)
    value = dist - (src_radius + 0.1 + turns * speed_t) - target_radius
    return jnp.where(valid, value, jnp.inf)


def target_pos_at_time_jax(state: EnvState, target_idx, turns):
    """目标星球在浮点 turns 后的位置，支持普通星球和彗星线性插值。"""
    b, p = target_idx.shape
    batch_idx = jnp.arange(b)[:, None]
    planets = state.planets[batch_idx, target_idx]
    initial = state.initial_planets[batch_idx, target_idx]
    alive = state.planet_alive[batch_idx, target_idx]

    base_x = _expand_like_time(planets[..., 2], turns)
    base_y = _expand_like_time(planets[..., 3], turns)
    dx = initial[..., 2] - 50.0
    dy = initial[..., 3] - 50.0
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    phi_cur = jnp.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    phi = _expand_like_time(phi_cur, turns) + _expand_like_time(state.angular_velocity[:, None], turns) * turns
    rot_x = 50.0 + _expand_like_time(orbit_r, turns) * jnp.cos(phi)
    rot_y = 50.0 + _expand_like_time(orbit_r, turns) * jnp.sin(phi)
    comet_mask = jax.vmap(comet_planet_mask)(state)
    is_comet = comet_mask[batch_idx, target_idx]
    rotating = (orbit_r + initial[..., 4] < 50.0) & alive & (~is_comet)
    px = jnp.where(_expand_like_time(rotating, turns), rot_x, base_x)
    py = jnp.where(_expand_like_time(rotating, turns), rot_y, base_y)
    valid = _expand_like_time(alive, turns)

    comet_x, comet_y, comet_valid = comet_target_pos_at_time_jax(state, target_idx, turns)
    is_comet_t = _expand_like_time(is_comet, turns)
    px = jnp.where(is_comet_t, comet_x, px)
    py = jnp.where(is_comet_t, comet_y, py)
    valid = jnp.where(is_comet_t, comet_valid, valid)
    return px, py, valid


def target_pos_velocity_at_time_jax(state: EnvState, target_idx, turns):
    """目标星球在浮点 turns 后的位置和速度，用于 Newton 求解 T*。"""
    b, p = target_idx.shape
    batch_idx = jnp.arange(b)[:, None]
    planets = state.planets[batch_idx, target_idx]
    initial = state.initial_planets[batch_idx, target_idx]
    alive = state.planet_alive[batch_idx, target_idx]

    base_x = _expand_like_time(planets[..., 2], turns)
    base_y = _expand_like_time(planets[..., 3], turns)
    zero = jnp.zeros_like(base_x)
    dx = initial[..., 2] - 50.0
    dy = initial[..., 3] - 50.0
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    phi_cur = jnp.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    omega = _expand_like_time(state.angular_velocity[:, None], turns)
    phi = _expand_like_time(phi_cur, turns) + omega * turns
    orbit_r_t = _expand_like_time(orbit_r, turns)
    rot_x = 50.0 + orbit_r_t * jnp.cos(phi)
    rot_y = 50.0 + orbit_r_t * jnp.sin(phi)
    rot_vx = -omega * orbit_r_t * jnp.sin(phi)
    rot_vy = omega * orbit_r_t * jnp.cos(phi)

    comet_mask = jax.vmap(comet_planet_mask)(state)
    is_comet = comet_mask[batch_idx, target_idx]
    rotating = (orbit_r + initial[..., 4] < 50.0) & alive & (~is_comet)
    rotating_t = _expand_like_time(rotating, turns)
    px = jnp.where(rotating_t, rot_x, base_x)
    py = jnp.where(rotating_t, rot_y, base_y)
    vx = jnp.where(rotating_t, rot_vx, zero)
    vy = jnp.where(rotating_t, rot_vy, zero)
    valid = _expand_like_time(alive, turns)

    comet_x, comet_y, comet_vx, comet_vy, comet_valid = comet_target_pos_velocity_at_time_jax(state, target_idx, turns)
    is_comet_t = _expand_like_time(is_comet, turns)
    px = jnp.where(is_comet_t, comet_x, px)
    py = jnp.where(is_comet_t, comet_y, py)
    vx = jnp.where(is_comet_t, comet_vx, vx)
    vy = jnp.where(is_comet_t, comet_vy, vy)
    valid = jnp.where(is_comet_t, comet_valid, valid)
    return px, py, vx, vy, valid


def comet_target_pos_at_time_jax(state: EnvState, target_idx, turns):
    """彗星目标在浮点 turns 后的位置。非彗星返回无效占位。"""
    extra = turns.ndim - target_idx.ndim
    one_shape = (1,) * extra
    slots = state.comet_slots.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    target_e = target_idx.reshape(target_idx.shape + one_shape + (1, 1))
    active = (
        (state.comet_slots >= 0)
        & (state.comet_path_index[:, :, None] >= -1)
        & (state.comet_lengths > 0)
    ).reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    match = (slots == target_e) & active

    path_index = state.comet_path_index.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, 1))
    lengths = state.comet_lengths.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    fi = path_index + turns[..., None, None]
    i0 = jnp.floor(fi).astype(jnp.int32)
    frac = fi - i0.astype(jnp.float32)
    valid_i0 = (i0 >= 0) & (i0 < lengths)
    i1 = jnp.minimum(i0 + 1, lengths - 1)
    has_i1 = i0 + 1 < lengths
    i0_safe = jnp.clip(i0, 0, MAX_COMET_PATH - 1)
    i1_safe = jnp.clip(i1, 0, MAX_COMET_PATH - 1)
    path_shape = (state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2)
    paths = state.comet_paths.reshape(path_shape)
    one0 = jax.nn.one_hot(i0_safe, MAX_COMET_PATH, dtype=jnp.float32)
    one1 = jax.nn.one_hot(i1_safe, MAX_COMET_PATH, dtype=jnp.float32)
    pos0 = (one0[..., None] * paths).sum(axis=-2)
    pos1 = (one1[..., None] * paths).sum(axis=-2)
    pos = jnp.where(has_i1[..., None], pos0 + frac[..., None] * (pos1 - pos0), pos0)
    weights = (match & valid_i0).astype(jnp.float32)
    has = weights.sum(axis=(-2, -1)) > 0.0
    px = (pos[..., 0] * weights).sum(axis=(-2, -1))
    py = (pos[..., 1] * weights).sum(axis=(-2, -1))
    return px, py, has


def comet_target_pos_velocity_at_time_jax(state: EnvState, target_idx, turns):
    """彗星目标在浮点 turns 后的位置和路径段速度。非彗星返回无效占位。"""
    extra = turns.ndim - target_idx.ndim
    one_shape = (1,) * extra
    slots = state.comet_slots.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    target_e = target_idx.reshape(target_idx.shape + one_shape + (1, 1))
    active = (
        (state.comet_slots >= 0)
        & (state.comet_path_index[:, :, None] >= -1)
        & (state.comet_lengths > 0)
    ).reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    match = (slots == target_e) & active

    path_index = state.comet_path_index.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, 1))
    lengths = state.comet_lengths.reshape((state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN))
    fi = path_index + turns[..., None, None]
    i0 = jnp.floor(fi).astype(jnp.int32)
    frac = fi - i0.astype(jnp.float32)
    valid_i0 = (i0 >= 0) & (i0 < lengths)
    i1 = jnp.minimum(i0 + 1, lengths - 1)
    has_i1 = i0 + 1 < lengths
    i0_safe = jnp.clip(i0, 0, MAX_COMET_PATH - 1)
    i1_safe = jnp.clip(i1, 0, MAX_COMET_PATH - 1)
    path_shape = (state.planets.shape[0], 1) + one_shape + (MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2)
    paths = state.comet_paths.reshape(path_shape)
    one0 = jax.nn.one_hot(i0_safe, MAX_COMET_PATH, dtype=jnp.float32)
    one1 = jax.nn.one_hot(i1_safe, MAX_COMET_PATH, dtype=jnp.float32)
    pos0 = (one0[..., None] * paths).sum(axis=-2)
    pos1 = (one1[..., None] * paths).sum(axis=-2)
    pos = jnp.where(has_i1[..., None], pos0 + frac[..., None] * (pos1 - pos0), pos0)
    vel = jnp.where(has_i1[..., None], pos1 - pos0, jnp.zeros_like(pos0))
    weights = (match & valid_i0).astype(jnp.float32)
    has = weights.sum(axis=(-2, -1)) > 0.0
    px = (pos[..., 0] * weights).sum(axis=(-2, -1))
    py = (pos[..., 1] * weights).sum(axis=(-2, -1))
    vx = (vel[..., 0] * weights).sum(axis=(-2, -1))
    vy = (vel[..., 1] * weights).sum(axis=(-2, -1))
    return px, py, vx, vy, has


def target_max_turns_jax(state: EnvState, target_idx, horizon: int):
    """普通目标搜索 horizon；彗星目标按 src/physics.py 限制为 remaining_life-1。"""
    b, p = target_idx.shape
    slots = state.comet_slots[:, None, :, :]
    target_e = target_idx[:, :, None, None]
    active = (state.comet_slots[:, None, :, :] >= 0) & (state.comet_path_index[:, None, :, None] >= -1)
    match = (slots == target_e) & active
    remaining = state.comet_lengths[:, None, :, :] - state.comet_path_index[:, None, :, None]
    comet_remaining = jnp.where(match, remaining, 0).max(axis=(-2, -1))
    comet_has = match.any(axis=(-2, -1))
    comet_max = jnp.minimum(jnp.asarray(horizon, dtype=jnp.int32), jnp.maximum(0, comet_remaining - 1))
    return jnp.where(comet_has, comet_max, horizon)


def target_is_first_collision_jax(src, target_idx, state: EnvState, fx0, fy0, speed, cos_a, sin_a, hit_turn):
    """按引擎阶段验证候选回合的第一碰撞对象就是目标。"""
    horizon = AIM_HORIZON
    turns_i = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    turns = turns_i.astype(jnp.float32)
    planets = state.planets
    planet_alive = state.planet_alive
    px_t, py_t = all_planet_pos_at_turns(state, horizon)
    fx_prev = fx0[..., None] + (turns - 1.0) * speed[..., None] * cos_a[..., None]
    fy_prev = fy0[..., None] + (turns - 1.0) * speed[..., None] * sin_a[..., None]
    fx_cur = fx0[..., None] + turns * speed[..., None] * cos_a[..., None]
    fy_cur = fy0[..., None] + turns * speed[..., None] * sin_a[..., None]
    active_step = turns_i.reshape((1,) * hit_turn.ndim + (horizon,)) <= hit_turn[..., None]
    in_bounds = (fx_cur >= 0.0) & (fx_cur <= 100.0) & (fy_cur >= 0.0) & (fy_cur <= 100.0)
    in_bounds_until_hit = (in_bounds | (~active_step)).all(axis=-1)

    src_axes = (None,) * (hit_turn.ndim - 1)
    px_old = px_t[(slice(None),) + src_axes + (slice(None), slice(None, -1))]
    py_old = py_t[(slice(None),) + src_axes + (slice(None), slice(None, -1))]
    px_new = px_t[(slice(None),) + src_axes + (slice(None), slice(1, None))]
    py_new = py_t[(slice(None),) + src_axes + (slice(None), slice(1, None))]
    radii = planets[(slice(None),) + src_axes + (slice(None), 4)]
    alive = planet_alive[(slice(None),) + src_axes + (slice(None), None)]

    move_d2 = point_segment_distance_sq(
        px_old,
        py_old,
        fx_prev[..., None, :],
        fy_prev[..., None, :],
        fx_cur[..., None, :],
        fy_cur[..., None, :],
    )
    move_hit = (move_d2 < radii[..., None] ** 2) & alive & active_step[..., None, :]

    sweep_active = ((px_new - px_old) ** 2 + (py_new - py_old) ** 2) > 1e-9
    sweep_d2 = point_segment_distance_sq(
        fx_cur[..., None, :],
        fy_cur[..., None, :],
        px_old,
        py_old,
        px_new,
        py_new,
    )
    sweep_hit = (sweep_d2 < radii[..., None] ** 2) & alive & sweep_active & active_step[..., None, :]

    large = jnp.int32(2 * (horizon + 1) + 1)
    move_keys = (2 * turns_i).reshape((1,) * (move_hit.ndim - 1) + (horizon,))
    sweep_keys = (2 * turns_i + 1).reshape((1,) * (sweep_hit.ndim - 1) + (horizon,))
    first_move = jnp.where(move_hit, move_keys, large).min(axis=-1)
    first_sweep = jnp.where(sweep_hit, sweep_keys, large).min(axis=-1)
    first_by_planet = jnp.minimum(first_move, first_sweep)
    first_planet = first_by_planet.argmin(axis=-1)
    first_key = first_by_planet.min(axis=-1)
    target_key_lo = 2 * hit_turn
    target_key_hi = 2 * hit_turn + 1
    return (
        in_bounds_until_hit
        & (first_key < large)
        & (first_planet == target_idx.reshape(target_idx.shape + (1,) * (hit_turn.ndim - target_idx.ndim)))
        & (first_key >= target_key_lo)
        & (first_key <= target_key_hi)
    )


def _expand_like_time(x, turns):
    """把 [B,P] 标量特征扩展到 turns 的 [B,P,...] 形状。"""
    return x.reshape(x.shape + (1,) * (turns.ndim - x.ndim))


def all_planet_pos_at_turns(state: EnvState, horizon: int):
    """计算未来 horizon 步内全部星球位置 [B, P, horizon+1]。

    step 0 为当前位置；静止星球固定，旋转星球按角速度追踪，彗星使用预存路径。
    """
    planets = state.planets
    initial_planets = state.initial_planets
    planet_alive = state.planet_alive
    turns = jnp.arange(horizon + 1, dtype=jnp.float32)
    dx = initial_planets[..., 2] - 50.0
    dy = initial_planets[..., 3] - 50.0
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    comet_mask = jax.vmap(comet_planet_mask)(state)
    rotating = (orbit_r + initial_planets[..., 4] < 50.0) & planet_alive & (~comet_mask)
    phi_cur = jnp.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    phi = phi_cur[..., None] + state.angular_velocity[:, None, None] * turns
    px = 50.0 + orbit_r[..., None] * jnp.cos(phi)
    py = 50.0 + orbit_r[..., None] * jnp.sin(phi)
    px = jnp.where(rotating[..., None], px, planets[..., None, 2])
    py = jnp.where(rotating[..., None], py, planets[..., None, 3])

    comet_slots = state.comet_slots.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    safe_slots = jnp.where(comet_slots >= 0, comet_slots, 0)
    group_active = state.comet_path_index[:, :, None] >= -1
    active = (state.comet_slots >= 0) & group_active
    active_flat = active.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN)
    path_index = state.comet_path_index[:, :, None, None] + jnp.arange(horizon + 1, dtype=jnp.int32)[None, None, None, :]
    path_index = jnp.broadcast_to(path_index, (state.planets.shape[0], MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1))
    valid_t = active[..., None] & (path_index >= 0) & (path_index < state.comet_lengths[..., None])
    path_index = jnp.clip(path_index, 0, MAX_COMET_PATH - 1)
    gather_index = jnp.broadcast_to(
        path_index[..., None],
        (state.planets.shape[0], MAX_COMET_SPAWNS, COMETS_PER_SPAWN, horizon + 1, 2),
    )
    path_pos = jnp.take_along_axis(state.comet_paths, gather_index, axis=3)
    path_pos = jnp.where(valid_t[..., None], path_pos, -1000.0)
    flat_pos = path_pos.reshape(state.planets.shape[0], MAX_COMET_SPAWNS * COMETS_PER_SPAWN, horizon + 1, 2)
    slot_onehot = jax.nn.one_hot(safe_slots, state.planets.shape[1], dtype=jnp.float32) * active_flat[..., None].astype(jnp.float32)
    comet_has = slot_onehot.sum(axis=1) > 0.0
    comet_px = jnp.einsum("bkp,bkt->bpt", slot_onehot, flat_pos[..., 0])
    comet_py = jnp.einsum("bkp,bkt->bpt", slot_onehot, flat_pos[..., 1])
    px = jnp.where(comet_has[..., None], comet_px, px)
    py = jnp.where(comet_has[..., None], comet_py, py)
    return px, py


def fleet_speed_local(ships):
    """根据舰数计算舰队速度：与 orbit_wars.py fleet_speed 完全一致。"""
    ratio = jnp.clip(jnp.log(jnp.maximum(ships, 1.0)) / jnp.log(1000.0), 0.0, 1.0)
    return jnp.minimum(1.0 + 5.0 * (ratio**1.5), 6.0)


def replace_done(state: EnvState, reset: EnvState) -> EnvState:
    """局面结束时，用 reset 状态替换已结束的环境。"""
    mask = state.done

    def choose(a, b):
        """按 done mask 选择环境 a（未结局）或 b（重置）。"""
        return jnp.where(mask.reshape((-1,) + (1,) * (a.ndim - 1)), b, a) if a.ndim > 0 else jnp.where(mask, b, a)

    return EnvState(
        planets=choose(state.planets, reset.planets),
        initial_planets=choose(state.initial_planets, reset.initial_planets),
        planet_alive=choose(state.planet_alive, reset.planet_alive),
        fleets=choose(state.fleets, reset.fleets),
        fleet_alive=choose(state.fleet_alive, reset.fleet_alive),
        step=choose(state.step, reset.step),
        angular_velocity=choose(state.angular_velocity, reset.angular_velocity),
        next_fleet_id=choose(state.next_fleet_id, reset.next_fleet_id),
        done=choose(state.done, reset.done),
        winner=choose(state.winner, reset.winner),
        overflow_fleets=choose(state.overflow_fleets, reset.overflow_fleets),
        comet_paths=choose(state.comet_paths, reset.comet_paths),
        comet_lengths=choose(state.comet_lengths, reset.comet_lengths),
        comet_ships=choose(state.comet_ships, reset.comet_ships),
        comet_valid=choose(state.comet_valid, reset.comet_valid),
        comet_path_index=choose(state.comet_path_index, reset.comet_path_index),
        comet_slots=choose(state.comet_slots, reset.comet_slots),
    )


if __name__ == "__main__":
    main()
