"""Orbit Wars JAX PPO 训练入口。

核心设计：
- collect_and_update: jax.lax.scan 单一 jit 内完成整段 rollout + GAE + PPO 更新。
- policy_sample: 两次前向传播（第一次采 fire/target，第二次带 target 采 amount）。
- actions_from_decisions: 将网络决策转换为游戏动作格式 [from, angle, ships]。
- aim_with_prediction_jax: 60 步轨迹预测锁定目标（旋转星球/彗星均支持）。
- target_is_first_collision_jax: 验证目标星球是否为舰队路径上第一个碰撜体。
- 环境并行度 B = num_envs，星球并行度 P = max_planets。
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.opponents import ExternalScriptOpponent, NearestOpponent  # noqa: E402
from src.state import parse_observation  # noqa: E402

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from .build_initial_state_bank import build_bank  # noqa: E402
from .fast_env import (  # noqa: E402
    COMETS_PER_SPAWN,
    MAX_COMET_PATH,
    MAX_COMET_SPAWNS,
    EnvParams,
    EnvState,
    comet_planet_mask,
    load_state_bank,
    point_segment_distance_sq,
    scores,
    select_from_bank,
    step_batch,
)
from .features_jax import FeatureParams, encode_batch, flatten_sources  # noqa: E402
from .model_jax import ActorCritic  # noqa: E402
from .normalization_jax import NormalizerParams, init_normalizer, normalize_encoded, update_and_normalize  # noqa: E402
from .ppo_jax import PPOParams, Rollout, compute_gae, log_prob_entropy, make_train_state, ppo_update, safe_logits  # noqa: E402
from .reward_jax import RewardParams, reward_from_transition  # noqa: E402

NEG_INF = -1.0e9


def flip_state(state: EnvState) -> EnvState:
    """将棋盘视角从 player 0 翻转到 player 1（owner 0↔1 互换）。

    用于自博弈：把当前局面翻转后，网络以为自己是 player 0，实际输出的是 player 1 的动作。
    """
    # 星球 owner：0→1，1→0，中立 -1 不变
    po = state.planets[..., 1]
    new_po = jnp.where(po == 0.0, 1.0, jnp.where(po == 1.0, 0.0, po))
    new_planets = state.planets.at[..., 1].set(new_po)
    # 舰队 owner：只有 0/1，直接取反
    new_fo = 1.0 - state.fleets[..., 1]
    new_fleets = state.fleets.at[..., 1].set(new_fo)
    return state._replace(planets=new_planets, fleets=new_fleets)


def policy_sample_params(params, model, flat, rng):
    """与 policy_sample 相同，但直接接受 params pytree（用于冻结对手）。"""
    zeros = jnp.zeros((flat.self_features.shape[0],), dtype=jnp.int32)
    out1 = model.apply(
        {"params": params},
        flat.self_features, flat.candidate_features, flat.global_features,
        flat.candidate_mask, flat.map_features, zeros,
    )
    rng_fire, rng_target, rng_amount = jax.random.split(rng, 3)
    has_target = flat.candidate_mask.at[:, 0].set(False).any(axis=-1) & flat.source_mask
    fire_logits = jnp.where(has_target, out1.fire_logits, -1.0e8)
    fire = jax.random.uniform(rng_fire, fire_logits.shape) < jax.nn.sigmoid(fire_logits)
    target_logits = jnp.where(flat.candidate_mask.at[:, 0].set(False), out1.target_logits, NEG_INF)
    target_logits = jnp.where(fire[:, None], target_logits, NEG_INF).at[:, 0].set(jnp.where(fire, target_logits[:, 0], 0.0))
    target_logits = safe_logits(target_logits)
    target = jax.random.categorical(rng_target, target_logits, axis=-1)
    out2 = model.apply(
        {"params": params},
        flat.self_features, flat.candidate_features, flat.global_features,
        flat.candidate_mask, flat.map_features, target,
    )
    selected_amount_mask = jnp.take_along_axis(flat.amount_mask, target[:, None, None], axis=1).squeeze(1)
    amount_logits = jnp.where(selected_amount_mask, out2.amount_logits, NEG_INF)
    amount_logits = jnp.where((~selected_amount_mask.any(axis=-1))[:, None], 0.0, amount_logits)
    amount = jax.random.categorical(rng_amount, amount_logits, axis=-1)
    return fire, target, amount, None, out1.value


def kaggle_obs_to_jax_state(obs, env_params: EnvParams) -> EnvState:
    """将 kaggle_environments 单步 observation 转换为 batch=1 的 EnvState。

    用于在 Python 游戏循环中把 kaggle obs 喂给 JAX 模型。
    彗星字段全部清零（对特征编码影响极小）。
    """
    P = env_params.max_planets
    F = env_params.max_fleets
    state = parse_observation(obs)

    planets = np.zeros((P, 7), dtype=np.float32)
    planet_alive = np.zeros(P, dtype=bool)
    for p in state.planets:
        if p.id < P:
            planets[p.id] = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            planet_alive[p.id] = True

    initial_planets = planets.copy()
    for p in state.initial_planets:
        if p.id < P:
            initial_planets[p.id] = [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]

    fleets = np.zeros((F, 7), dtype=np.float32)
    fleet_alive = np.zeros(F, dtype=bool)
    for i, fl in enumerate(state.fleets[:F]):
        fleets[i] = [fl.id, fl.owner, fl.x, fl.y, fl.angle, fl.from_planet_id, fl.ships]
        fleet_alive[i] = True

    return EnvState(
        planets=jnp.array(planets[None]),          # [1, P, 7]
        initial_planets=jnp.array(initial_planets[None]),
        planet_alive=jnp.array(planet_alive[None]),
        fleets=jnp.array(fleets[None]),            # [1, F, 7]
        fleet_alive=jnp.array(fleet_alive[None]),
        step=jnp.array([state.step], dtype=jnp.int32),
        angular_velocity=jnp.array([state.angular_velocity], dtype=jnp.float32),
        next_fleet_id=jnp.array([len(state.fleets)], dtype=jnp.int32),
        done=jnp.array([False]),
        winner=jnp.array([-1], dtype=jnp.int32),
        overflow_fleets=jnp.zeros(1, dtype=jnp.int32),
        comet_paths=jnp.zeros((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=jnp.float32),
        comet_lengths=jnp.zeros((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=jnp.int32),
        comet_ships=jnp.zeros((1, MAX_COMET_SPAWNS), dtype=jnp.float32),
        comet_valid=jnp.zeros((1, MAX_COMET_SPAWNS), dtype=bool),
        comet_path_index=jnp.full((1, MAX_COMET_SPAWNS), -2, dtype=jnp.int32),
        comet_slots=jnp.full((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN), -1, dtype=jnp.int32),
    )


def jax_actions_to_kaggle(actions_arr) -> list:
    """将 JAX actions_from_decisions 输出 [1, P, 3] 转为 kaggle 格式 [[from, angle, ships], ...]。"""
    arr = np.array(actions_arr[0])  # [P, 3]
    return [[float(r[0]), float(r[1]), int(r[2])] for r in arr if float(r[2]) > 0]


def eval_vs_nearest_jax(
    train_state, model, bank: EnvState, env_params: EnvParams,
    feat_params, norm_state, norm_params, num_games: int, rng,
) -> dict:
    """在 JAX fast_env 中对 nearest 对手跑 num_games 局完整对局，返回胜/平/负率。"""
    rng, reset_key = jax.random.split(rng)
    state = reset_from_bank(bank, reset_key, num_games)

    def step_fn(carry, _):
        st, key = carry
        key, ak, rk = jax.random.split(key, 3)
        raw_enc = encode_batch(st, feat_params)
        enc = normalize_encoded(raw_enc, norm_state, norm_params)
        flat = flatten_sources(enc)
        fire, target, amount, _, _ = policy_sample(train_state, model, flat, ak)
        learner_acts = actions_from_decisions(
            st, enc, fire.reshape(st.planets.shape[:2]),
            target.reshape(st.planets.shape[:2]),
            amount.reshape(st.planets.shape[:2]), feat_params,
        )
        opp_acts = nearest_opponent_actions(st, player=1, params=feat_params)
        joint = jnp.stack([learner_acts, opp_acts], axis=1)
        next_st = step_batch(st, joint, env_params)
        reset_st = reset_from_bank(bank, rk, num_games)
        # 已结束的环境重置（但保留 winner 用于统计最后一局）
        next_st = jax.tree_util.tree_map(
            lambda a, b: jnp.where(
                st.done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a
            ),
            next_st, reset_st,
        )
        return (next_st, key), (st.done, st.winner)

    (final_state, _), (all_done, all_winners) = jax.lax.scan(
        step_fn, (state, rng), None, length=env_params.episode_steps
    )
    # all_done: [episode_steps, num_games], all_winners: [episode_steps, num_games]
    # 统计每次 done=True 时的 winner（游戏结束后会被重置，最终状态不可靠）
    all_done_np = np.array(all_done)    # bool [T, G]
    all_winners_np = np.array(all_winners)  # int [T, G]
    wins = int(((all_winners_np == 0) & all_done_np).sum())
    losses = int(((all_winners_np == 1) & all_done_np).sum())
    draws = int(((all_winners_np == 2) & all_done_np).sum())
    total = max(wins + losses + draws, 1)
    return {"wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / total, "loss_rate": losses / total}


def eval_vs_external(
    train_state, model, feat_params, norm_state, norm_params, env_params: EnvParams,
    opponent_dir: Path, num_games: int, base_seed: int, rng,
) -> dict:
    """通过 kaggle_environments 对战外部智能体 num_games 局，返回胜平负统计。

    JAX 模型以 player 0 身份出战，外部智能体为 player 1。
    """
    try:
        from kaggle_environments import make as kag_make
    except ImportError:
        print("[eval] kaggle_environments 未安装，跳过外部智能体评估")
        return {}

    opponent = ExternalScriptOpponent(opponent_dir)
    wins = losses = draws = 0

    # JIT 单步推理（batch=1）
    @jax.jit
    def infer(st):
        raw_enc = encode_batch(st, feat_params)
        enc = normalize_encoded(raw_enc, norm_state, norm_params)
        flat = flatten_sources(enc)
        zeros = jnp.zeros((flat.self_features.shape[0],), dtype=jnp.int32)
        out1 = train_state.apply_fn({"params": train_state.params},
                                    flat.self_features, flat.candidate_features,
                                    flat.global_features, flat.candidate_mask,
                                    flat.map_features, zeros)
        # 贪心取 argmax（确定性策略）
        fire = jax.nn.sigmoid(out1.fire_logits) > 0.5
        has_target = flat.candidate_mask.at[:, 0].set(False).any(axis=-1) & flat.source_mask
        fire = fire & has_target
        tgt_logits = jnp.where(flat.candidate_mask.at[:, 0].set(False), out1.target_logits, NEG_INF)
        tgt_logits = jnp.where(fire[:, None], tgt_logits, NEG_INF).at[:, 0].set(
            jnp.where(fire, tgt_logits[:, 0], 0.0))
        target = jnp.argmax(safe_logits(tgt_logits), axis=-1)
        out2 = train_state.apply_fn({"params": train_state.params},
                                    flat.self_features, flat.candidate_features,
                                    flat.global_features, flat.candidate_mask,
                                    flat.map_features, target)
        sel_amt_mask = jnp.take_along_axis(flat.amount_mask, target[:, None, None], axis=1).squeeze(1)
        amt_logits = jnp.where(sel_amt_mask, out2.amount_logits, NEG_INF)
        amt_logits = jnp.where((~sel_amt_mask.any(axis=-1))[:, None], 0.0, amt_logits)
        amount = jnp.argmax(amt_logits, axis=-1)
        acts = actions_from_decisions(
            st, enc,
            fire.reshape(st.planets.shape[:2]),
            target.reshape(st.planets.shape[:2]),
            amount.reshape(st.planets.shape[:2]),
            feat_params,
        )
        return acts

    for g in range(num_games):
        seed = base_seed + g
        opponent.reset_env(0, seed)
        env = kag_make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        obs = states[0]["observation"]
        opp_obs = states[1]["observation"]
        done = states[0]["status"] != "ACTIVE"
        while not done:
            jax_state = kaggle_obs_to_jax_state(obs, env_params)
            jax_acts = infer(jax_state)
            learner_action = jax_actions_to_kaggle(jax_acts)
            opp_action = opponent.act_for_env(0, opp_obs)
            states = env.step([learner_action, opp_action])
            obs = states[0]["observation"]
            opp_obs = states[1]["observation"]
            done = states[0]["status"] != "ACTIVE"
        reward = states[0].get("reward", 0.0) or 0.0
        if reward > 0:
            wins += 1
            result = "win"
        elif reward < 0:
            losses += 1
            result = "loss"
        else:
            draws += 1
            result = "draw"
        print(f"  [eval/{opponent_dir.name}] game={g+1} seed={seed} {result} reward={reward:.1f}")

    total = num_games
    return {"wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / total, "loss_rate": losses / total}


def main() -> None:
    """训练入口：解析参数、加载配置、初始化环境/模型/状态，循环执行 collect_and_update。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfg_jax_default.yaml")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--bank", default="outputs/jax_initial_state_bank.npz")
    parser.add_argument("--bank-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.updates is not None:
        cfg.ppo.total_updates = args.updates
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.rollout_steps is not None:
        cfg.ppo.rollout_steps = args.rollout_steps

    env_params = EnvParams(
        episode_steps=cfg.env.episode_steps,
        max_planets=cfg.env.max_planets,
        max_fleets=512,
        max_actions=cfg.env.max_planets,
    )
    feat_params = FeatureParams.from_train_config(cfg)
    norm_params = NormalizerParams.from_train_config(cfg)
    reward_params = RewardParams.from_train_config(cfg)
    ppo_params = PPOParams(
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_coef=cfg.ppo.clip_coef,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        epochs=cfg.ppo.epochs,
        minibatch_size=min(cfg.ppo.minibatch_size, cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.env.max_planets),
        norm_adv=cfg.ppo.norm_adv,
    )
    bank_path = Path(args.bank)
    if not bank_path.exists():
        build_bank(bank_path, args.bank_size, cfg.seed, env_params)

    bank = load_state_bank(bank_path, env_params)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, rng_reset, rng_model = jax.random.split(rng, 3)
    state = reset_from_bank(bank, rng_reset, cfg.ppo.num_envs)
    sample = flatten_sources(encode_batch(state, feat_params))
    model = ActorCritic(
        self_dim=sample.self_features.shape[-1],
        candidate_dim=sample.candidate_features.shape[-1],
        global_dim=sample.global_features.shape[-1],
        candidate_count=cfg.env.max_planets + 1,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        use_cnn=cfg.model.use_cnn,
    )
    train_state = make_train_state(rng_model, model, sample, cfg.ppo.lr, cfg.ppo.max_grad_norm)
    normalizer_state = init_normalizer()

    run_dir = Path(cfg.save_dir) / f"{cfg.run_name}_jax"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    eval_metric_fields = [
        "eval_nearest_win_rate",
        "eval_nearest_loss_rate",
        "eval_nearest_wins",
        "eval_nearest_losses",
        "eval_nearest_draws",
        "eval_sigma_win_rate",
        "eval_sigma_loss_rate",
        "eval_sigma_wins",
        "eval_sigma_losses",
        "eval_sigma_draws",
        "eval_tamrazov_win_rate",
        "eval_tamrazov_loss_rate",
        "eval_tamrazov_wins",
        "eval_tamrazov_losses",
        "eval_tamrazov_draws",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "update",
            "sps",
            "episode_reward",
            "win_rate",
            "mean_return",
            "episodes_per_rollout",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
            *eval_metric_fields,
        ])
    print(f"JAX trainer run_dir={run_dir} envs={cfg.ppo.num_envs} rollout_steps={cfg.ppo.rollout_steps} device={jax.devices()}")

    # 初始化 wandb（仅当配置开启且库可用时）
    use_wandb = _WANDB_AVAILABLE and getattr(cfg.wandb, "enabled", False)
    if use_wandb:
        _wandb.init(
            project=getattr(cfg.wandb, "project", "orbit-wars-rl"),
            entity=getattr(cfg.wandb, "entity", None) or None,
            name=cfg.run_name,
            config={
                "num_envs": cfg.ppo.num_envs,
                "rollout_steps": cfg.ppo.rollout_steps,
                "total_updates": cfg.ppo.total_updates,
                "gamma": cfg.ppo.gamma,
                "gae_lambda": cfg.ppo.gae_lambda,
                "clip_coef": cfg.ppo.clip_coef,
                "ent_coef": cfg.ppo.ent_coef,
                "vf_coef": cfg.ppo.vf_coef,
                "lr": cfg.ppo.lr,
                "max_grad_norm": cfg.ppo.max_grad_norm,
                "epochs": cfg.ppo.epochs,
                "hidden_size": cfg.model.hidden_size,
                "use_cnn": cfg.model.use_cnn,
                "max_planets": cfg.env.max_planets,
                "opponent": getattr(cfg, "opponent", "nearest"),
            },
            mode=getattr(cfg.wandb, "mode", "online"),
            group=getattr(cfg.wandb, "group", None) or None,
            tags=getattr(cfg.wandb, "tags", []) or [],
            dir=str(run_dir),
        )
        print(f"wandb run: {_wandb.run.url}")
    else:
        print("wandb disabled (set wandb.enabled: true in config to enable)")

    self_play = getattr(cfg, "opponent", "nearest") == "self"
    use_pool = getattr(cfg, "opponent", "nearest") == "pool"
    # 对手池模式：nearest / selfplay / selfplay-lag 各 1/3（Phase 1）
    # Phase 2（稳定战胜 nearest 后）：selfplay / selfplay-lag 各 1/2
    pool_phase = 1
    pool_lag_interval = getattr(cfg, "pool_lag_interval", 50)
    promote_nearest_win_rate = getattr(cfg, "promote_nearest_win_rate", 0.8)
    promote_consecutive_evals = getattr(cfg, "promote_consecutive_evals", 3)
    recent_nearest_win_rates: list[float] = []
    pool_rng = random.Random(cfg.seed + 12345)
    if use_pool:
        print(f"opponent=pool (lag_interval={pool_lag_interval}, promote_at={promote_nearest_win_rate:.2f}x{promote_consecutive_evals})")
    else:
        print(f"opponent={'self_play' if self_play else 'nearest'}")
    opponent_update_freq = cfg.self_play_update_interval  # 默认 25
    eval_enabled = getattr(cfg.eval, "enabled", True)
    eval_interval = cfg.eval.every            # 每多少 update 评估一次
    eval_games = cfg.eval.games               # 每次评估局数
    eval_seed = cfg.eval.seed
    tamrazov_dir = ROOT / "orbit-wars-lab" / "agents" / "external" / "tamrazov-starwars"
    sigma_dir = ROOT / "orbit-wars-lab" / "agents" / "external" / "sigmaborov-reinforce"
    checkpoint_every = getattr(cfg, "checkpoint_every", 100)

    # 冻结对手权重（初始与学习器相同，每 opponent_update_freq 步同步一次）
    opponent_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
    # 滞后对手权重（每 pool_lag_interval 步更新，用于对抗 50 步前的自己）
    lagged_params = jax.tree_util.tree_map(lambda x: x, train_state.params)

    # 两个 JIT 编译的训练步（nearest 对手 / self-play 对手），供 pool 模式按需调用
    train_step_nearest = jax.jit(
        lambda ts, opp_p, st, ns, key: collect_and_update(
            ts, opp_p, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, cfg.ppo.rollout_steps, False,
        )
    )
    train_step_selfplay = jax.jit(
        lambda ts, opp_p, st, ns, key: collect_and_update(
            ts, opp_p, st, ns, key, bank, env_params, feat_params, norm_params,
            reward_params, ppo_params, model, cfg.ppo.rollout_steps, True,
        )
    )
    # 兼容旧配置（opponent: nearest / self）
    train_step = jax.jit(
        lambda ts, opp_p, st, ns, key: collect_and_update(
            ts,
            opp_p,
            st,
            ns,
            key,
            bank,
            env_params,
            feat_params,
            norm_params,
            reward_params,
            ppo_params,
            model,
            cfg.ppo.rollout_steps,
            self_play,
        )
    )

    for update in range(1, cfg.ppo.total_updates + 1):
        # 每隔 opponent_update_freq 步把当前权重同步给 self-play 对手
        if update % opponent_update_freq == 1:
            opponent_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
            if self_play or use_pool:
                print(f"[pool] update={update} 同步 selfplay 对手权重")

        # 每隔 pool_lag_interval 步更新滞后对手权重（pool 模式使用）
        if use_pool and update % pool_lag_interval == 1:
            lagged_params = jax.tree_util.tree_map(lambda x: x, train_state.params)
            print(f"[pool] update={update} 同步 lagged 对手权重 (lag={pool_lag_interval})")

        start = time.perf_counter()
        rng, step_key = jax.random.split(rng)

        if use_pool:
            u = pool_rng.random()
            if pool_phase == 1:
                # Phase 1: nearest / selfplay-current / selfplay-lag 各 1/3
                if u < 1 / 3:
                    train_state, state, normalizer_state, metrics = train_step_nearest(
                        train_state, opponent_params, state, normalizer_state, step_key)
                elif u < 2 / 3:
                    train_state, state, normalizer_state, metrics = train_step_selfplay(
                        train_state, opponent_params, state, normalizer_state, step_key)
                else:
                    train_state, state, normalizer_state, metrics = train_step_selfplay(
                        train_state, lagged_params, state, normalizer_state, step_key)
            else:
                # Phase 2: selfplay-current / selfplay-lag 各 1/2
                # sigma 为外部 Python 进程，无法 JIT 编译（速度慢 ~100x），训练仍用 self-play
                # sigma 进度通过 eval 跟踪
                if u < 0.5:
                    train_state, state, normalizer_state, metrics = train_step_selfplay(
                        train_state, opponent_params, state, normalizer_state, step_key)
                else:
                    train_state, state, normalizer_state, metrics = train_step_selfplay(
                        train_state, lagged_params, state, normalizer_state, step_key)
        else:
            # 兼容旧配置（opponent: nearest / self）
            if self_play and update % opponent_update_freq == 1:
                pass  # already synced above
            train_state, state, normalizer_state, metrics = train_step(
                train_state, opponent_params, state, normalizer_state, step_key)
        metrics = jax.device_get(metrics)
        elapsed = time.perf_counter() - start
        samples = cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.env.max_planets
        sps = samples / max(elapsed, 1e-9)
        line = (
            f"update={update} sps={sps:.1f} loss={metrics['loss']:.4f} "
            f"ep_reward={metrics['episode_reward']:.3f} winrate={metrics['win_rate']:.3f} "
            f"return={metrics['mean_return']:.3f} eps={metrics['episodes_per_rollout']:.0f} "
            f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.5f}"
        )
        print(line, flush=True)

        eval_metrics = {field: "" for field in eval_metric_fields}
        if eval_enabled and eval_interval > 0 and update % eval_interval == 0:
            print(f"[eval] update={update} 开始评估 ({eval_games} 局)...")
            rng, eval_rng = jax.random.split(rng)

            # 评估 1：vs nearest（JAX 快速评估）
            near_res = eval_vs_nearest_jax(
                train_state, model, bank, env_params, feat_params,
                normalizer_state, norm_params, eval_games, eval_rng,
            )
            print(f"[eval/nearest]   wins={near_res['wins']} losses={near_res['losses']} "
                  f"draws={near_res['draws']} win_rate={near_res['win_rate']:.3f}")
            eval_metrics.update({
                "eval_nearest_win_rate": near_res["win_rate"],
                "eval_nearest_loss_rate": near_res["loss_rate"],
                "eval_nearest_wins": near_res["wins"],
                "eval_nearest_losses": near_res["losses"],
                "eval_nearest_draws": near_res["draws"],
            })

            # 评估 2：vs sigmaborov-reinforce（kaggle_environments）
            sigma_res: dict = {}
            if sigma_dir.exists():
                sigma_res = eval_vs_external(
                    train_state, model, feat_params, normalizer_state, norm_params, env_params,
                    sigma_dir, eval_games, eval_seed + update + 7919, rng,
                )
                print(f"[eval/sigma]     wins={sigma_res.get('wins',0)} "
                      f"losses={sigma_res.get('losses',0)} "
                      f"draws={sigma_res.get('draws',0)} win_rate={sigma_res.get('win_rate',0):.3f}")
                eval_metrics.update({
                    "eval_sigma_win_rate": sigma_res["win_rate"],
                    "eval_sigma_loss_rate": sigma_res["loss_rate"],
                    "eval_sigma_wins": sigma_res["wins"],
                    "eval_sigma_losses": sigma_res["losses"],
                    "eval_sigma_draws": sigma_res["draws"],
                })

            # 升阶检测（Pool Phase 1 → Phase 2）：连续 N 次 nearest win_rate >= threshold
            if use_pool and pool_phase == 1:
                recent_nearest_win_rates.append(float(near_res["win_rate"]))
                if len(recent_nearest_win_rates) > promote_consecutive_evals:
                    recent_nearest_win_rates.pop(0)
                if (len(recent_nearest_win_rates) >= promote_consecutive_evals
                        and all(r >= promote_nearest_win_rate for r in recent_nearest_win_rates)):
                    pool_phase = 2
                    print(f"[pool] update={update} ★ 升阶 Phase 2：连续 {promote_consecutive_evals} 次 "
                          f"nearest win_rate >= {promote_nearest_win_rate:.2f}，切换为 selfplay/selfplay-lag 训练 + sigma eval")

            # 评估 3：vs tamrazov-starwars（kaggle_environments）
            tamr_res = {}
            if tamrazov_dir.exists():
                tamr_res = eval_vs_external(
                    train_state, model, feat_params, normalizer_state, norm_params, env_params,
                    tamrazov_dir, eval_games, eval_seed + update, rng,
                )
                print(f"[eval/tamrazov]  wins={tamr_res.get('wins',0)} "
                      f"losses={tamr_res.get('losses',0)} "
                      f"win_rate={tamr_res.get('win_rate',0):.3f}")
                eval_metrics.update({
                    "eval_tamrazov_win_rate": tamr_res["win_rate"],
                    "eval_tamrazov_loss_rate": tamr_res["loss_rate"],
                    "eval_tamrazov_wins": tamr_res["wins"],
                    "eval_tamrazov_losses": tamr_res["losses"],
                    "eval_tamrazov_draws": tamr_res["draws"],
                })

            if use_wandb:
                log_dict: dict = {
                    "eval/nearest_win_rate": float(near_res["win_rate"]),
                    "eval/nearest_loss_rate": float(near_res["loss_rate"]),
                    "eval/nearest_wins": float(near_res["wins"]),
                    "eval/nearest_losses": float(near_res["losses"]),
                    "eval/nearest_draws": float(near_res["draws"]),
                }
                if sigma_res:
                    log_dict["eval/sigma_win_rate"] = float(sigma_res["win_rate"])
                    log_dict["eval/sigma_loss_rate"] = float(sigma_res["loss_rate"])
                    log_dict["eval/sigma_wins"] = float(sigma_res["wins"])
                    log_dict["eval/sigma_losses"] = float(sigma_res["losses"])
                    log_dict["eval/sigma_draws"] = float(sigma_res["draws"])
                if tamr_res:
                    log_dict["eval/tamrazov_win_rate"] = float(tamr_res["win_rate"])
                    log_dict["eval/tamrazov_loss_rate"] = float(tamr_res["loss_rate"])
                    log_dict["eval/tamrazov_wins"] = float(tamr_res["wins"])
                    log_dict["eval/tamrazov_losses"] = float(tamr_res["losses"])
                    log_dict["eval/tamrazov_draws"] = float(tamr_res["draws"])
                if use_pool:
                    log_dict["pool/phase"] = float(pool_phase)
                    if recent_nearest_win_rates:
                        log_dict["pool/recent_nearest_win_rate"] = float(np.mean(recent_nearest_win_rates))
                _wandb.log(log_dict, step=update)

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                update,
                f"{sps:.6f}",
                f"{metrics['episode_reward']:.8f}",
                f"{metrics['win_rate']:.8f}",
                f"{metrics['mean_return']:.8f}",
                f"{metrics['episodes_per_rollout']:.1f}",
                f"{metrics['loss']:.8f}",
                f"{metrics['policy_loss']:.8f}",
                f"{metrics['value_loss']:.8f}",
                f"{metrics['entropy']:.8f}",
                f"{metrics['approx_kl']:.8f}",
                f"{metrics['clipfrac']:.8f}",
                *[eval_metrics[field] for field in eval_metric_fields],
            ])
        if use_wandb:
            _wandb.log({
                "charts/win_rate": float(metrics["win_rate"]),
                "charts/episode_reward": float(metrics["episode_reward"]),
                "charts/mean_return": float(metrics["mean_return"]),
                "charts/episodes_per_rollout": float(metrics["episodes_per_rollout"]),
                "losses/total_loss": float(metrics["loss"]),
                "losses/policy_loss": float(metrics["policy_loss"]),
                "losses/value_loss": float(metrics["value_loss"]),
                "losses/entropy": float(metrics["entropy"]),
                "losses/approx_kl": float(metrics["approx_kl"]),
                "losses/clipfrac": float(metrics["clipfrac"]),
            }, step=update)

        # ── 定期保存 checkpoint ───────────────────────────────────
        if update % checkpoint_every == 0:
            import pickle
            ckpt_path = run_dir / f"ckpt_{update:06d}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "params": jax.device_get(train_state.params),
                    "normalizer_state": jax.device_get(normalizer_state),
                    "update": update,
                    "cfg": cfg,
                }, f)
            print(f"[ckpt] saved {ckpt_path}")


def reset_from_bank(bank: EnvState, rng, num_envs: int) -> EnvState:
    """从预建 state bank 中随机采样 num_envs 个初始状态。"""
    size = bank.planets.shape[0]
    idx = jax.random.randint(rng, (num_envs,), 0, size)
    return select_from_bank(bank, idx)


def collect_and_update(train_state, opponent_params, state, normalizer_state, rng, bank, env_params, feat_params, norm_params, reward_params, ppo_params, model, rollout_steps: int, self_play: bool = False):
    """收集 rollout 并执行 PPO 更新。

    1. lax.scan 循环 rollout_steps 步：特征展开 + 采样 + step 环境 + 奖励 + done 处理。
    2. 计算最后状态价值 + GAE 估计。
    3. ppo_update 执行多 epoch/minibatch 梯度更新。

    opponent_params: 冻结的对手权重 pytree（self_play=True 时使用）。
    """
    rng, rollout_key = jax.random.split(rng)

    def body(carry, _):
        """lax.scan 循环体：处理单步。"""
        ts, st, ns, key = carry
        key, action_key, opp_key, reset_key = jax.random.split(key, 4)
        raw_encoded = encode_batch(st, feat_params)
        encoded, ns = update_and_normalize(raw_encoded, ns, norm_params)
        flat = flatten_sources(encoded)
        fire, target, amount, logp, value = policy_sample(ts, model, flat, action_key)
        learner_actions = actions_from_decisions(st, encoded, fire.reshape(st.planets.shape[:2]), target.reshape(st.planets.shape[:2]), amount.reshape(st.planets.shape[:2]), feat_params)
        # 对手动作：自博弈时用冻结 opponent_params 跑翻转局面；否则用最近敌方启发式
        if self_play:
            flipped = flip_state(st)
            raw_enc_opp = encode_batch(flipped, feat_params)
            enc_opp = normalize_encoded(raw_enc_opp, ns, norm_params)  # 只读，不更新归一化器
            flat_opp = flatten_sources(enc_opp)
            f_opp, t_opp, a_opp, _, _ = policy_sample_params(opponent_params, model, flat_opp, opp_key)
            opponent_actions = actions_from_decisions(flipped, enc_opp, f_opp.reshape(flipped.planets.shape[:2]), t_opp.reshape(flipped.planets.shape[:2]), a_opp.reshape(flipped.planets.shape[:2]), feat_params)
        else:
            opponent_actions = nearest_opponent_actions(st, player=1, params=feat_params)  # 对手：最近敌方简单基线
        joint = jnp.stack([learner_actions, opponent_actions], axis=1)  # 双方动作合并
        next_state = step_batch(st, joint, env_params)
        reward_env, reward_parts = reward_from_transition(st, next_state, learner_actions, reward_params)
        reward = jnp.repeat(reward_env[:, None], feat_params.max_planets, axis=1)  # 扩展到 P 维
        done = jnp.repeat(next_state.done[:, None], feat_params.max_planets, axis=1)
        reset_state = reset_from_bank(bank, reset_key, st.planets.shape[0])
        next_state = replace_done(next_state, reset_state)  # 局结环境自动重置
        row = Rollout(
            encoded.self_features,
            encoded.candidate_features,
            encoded.global_features,
            encoded.map_features,
            encoded.candidate_mask,
            encoded.amount_mask,
            encoded.source_mask,
            fire.reshape(st.planets.shape[:2]),
            target.reshape(st.planets.shape[:2]),
            amount.reshape(st.planets.shape[:2]),
            logp.reshape(st.planets.shape[:2]),
            value.reshape(st.planets.shape[:2]),
            reward,
            done,
        )
        return (ts, next_state, ns, key), row

    (train_state, state, normalizer_state, rng), rollout = jax.lax.scan(body, (train_state, state, normalizer_state, rollout_key), None, length=rollout_steps)
    last_raw_encoded = encode_batch(state, feat_params)
    last_encoded = flatten_sources(normalize_encoded(last_raw_encoded, normalizer_state, norm_params))
    last_out = train_state.apply_fn(
        {"params": train_state.params},
        last_encoded.self_features,
        last_encoded.candidate_features,
        last_encoded.global_features,
        last_encoded.candidate_mask,
        last_encoded.map_features,
        jnp.zeros((last_encoded.self_features.shape[0],), dtype=jnp.int32),
    )
    last_values = last_out.value.reshape(state.planets.shape[:2])
    returns, advantages = compute_gae(rollout.reward, rollout.done, rollout.old_value, last_values, ppo_params)
    train_state, ppo_metrics = ppo_update(train_state, rollout, returns, advantages, ppo_params, rng)
    # rollout.done: [T, B, P]，每步每 env 的结束标志（P 维相同）
    # rollout.reward: [T, B, P]，只有 done=True 那步有 ±1，其余为 0
    finished = rollout.done.any(axis=-1)          # [T, B] 该步是否有局结束
    finished_any = finished.any(axis=0)           # [B] 该 env 在本 rollout 是否结束过
    ep_reward = rollout.reward.mean(axis=-1)       # [T, B] 对 P 维平均（P 维全同）
    # episode_reward：只统计已结束局的终局奖励均值（非零步），反映真实胜负
    ep_reward_sum = jnp.where(finished, ep_reward, 0.0).sum()
    episodes = jnp.maximum(finished.sum(), 1)
    episode_reward = ep_reward_sum / episodes
    # win_rate：结束局中学习方获胜（reward > 0）的比率
    wins = ((ep_reward > 0.0) & finished).sum()
    win_rate = wins / episodes
    # mean_return：rollout 内所有样本 return（GAE 回报）的均值，反映 value 估计水平
    mean_return = returns.mean()
    metrics = {
        **ppo_metrics,
        "episode_reward": episode_reward,   # 已结束局的终局奖励均值（接近 ±1）
        "win_rate": win_rate,               # 已结束局中的胜率
        "mean_return": mean_return,         # GAE 回报均值
        "episodes_per_rollout": finished.sum().astype(jnp.float32),  # 每次 rollout 完成的局数
    }
    return train_state, state, normalizer_state, metrics


def policy_sample(train_state, model, flat, rng):
    """两阶段采样策略。

    第一步：传入 zeros target 前向 -> 采 fire/target。
    第二步：传入真实 target 前向 -> 采 amount，并计算联合 logp。
    与 log_prob_entropy 训练路径完全一致。
    """
    zeros = jnp.zeros((flat.self_features.shape[0],), dtype=jnp.int32)
    out1 = train_state.apply_fn(
        {"params": train_state.params},
        flat.self_features,
        flat.candidate_features,
        flat.global_features,
        flat.candidate_mask,
        flat.map_features,
        zeros,
    )
    rng_fire, rng_target, rng_amount = jax.random.split(rng, 3)
    has_target = flat.candidate_mask.at[:, 0].set(False).any(axis=-1) & flat.source_mask
    fire_logits = jnp.where(has_target, out1.fire_logits, -1.0e8)  # 没有可用目标时强制 fire=0
    fire = jax.random.uniform(rng_fire, fire_logits.shape) < jax.nn.sigmoid(fire_logits)
    # 屏蔽 no-op slot，使 target 必须是阵中星球
    target_logits = jnp.where(flat.candidate_mask.at[:, 0].set(False), out1.target_logits, NEG_INF)
    target_logits = jnp.where(fire[:, None], target_logits, NEG_INF).at[:, 0].set(jnp.where(fire, target_logits[:, 0], 0.0))
    target_logits = safe_logits(target_logits)
    target = jax.random.categorical(rng_target, target_logits, axis=-1)
    out2 = train_state.apply_fn(
        {"params": train_state.params},
        flat.self_features,
        flat.candidate_features,
        flat.global_features,
        flat.candidate_mask,
        flat.map_features,
        target,
    )
    selected_amount_mask = jnp.take_along_axis(flat.amount_mask, target[:, None, None], axis=1).squeeze(1)
    amount_logits = jnp.where(selected_amount_mask, out2.amount_logits, NEG_INF)
    amount_logits = jnp.where((~selected_amount_mask.any(axis=-1))[:, None], 0.0, amount_logits)
    amount = jax.random.categorical(rng_amount, amount_logits, axis=-1)
    logp, _entropy = log_prob_entropy(out2, fire, target, amount, flat.candidate_mask, flat.amount_mask)
    return fire, target, amount, logp, out1.value


def actions_from_decisions(state, encoded, fire, target, amount, params: FeatureParams):
    """将网络决策转换为游戏动作 [from_id, angle_rad, ships]。

    fire=False 或 target=0 (属于 no-op) 时输出全 0。
    """
    b, p = fire.shape
    source_ids = encoded.source_ids.astype(jnp.float32)
    ships = jnp.take_along_axis(encoded.ship_buckets, amount[..., None], axis=-1).squeeze(-1)
    target_idx = jnp.clip(target - 1, 0, params.max_planets - 1)
    batch_idx = jnp.arange(b)[:, None]
    target_planets = state.planets[batch_idx, target_idx]
    src_planets = state.planets
    angle, aim_hit = aim_with_prediction_jax(
        src_planets,
        target_planets,
        target_idx,
        state,
        ships,
    )
    valid = fire & (target > 0) & encoded.source_mask & aim_hit
    return jnp.stack(
        [
            jnp.where(valid, source_ids, 0.0),
            jnp.where(valid, angle, 0.0),
            jnp.where(valid, ships, 0.0),
        ],
        axis=-1,
    )


def aim_with_prediction_jax(src, target, target_idx, state: EnvState, ships):
    """60 步轨迹预测打射角度，匹配 PyTorch 版 src/physics.aim_with_prediction。

    支持静止星球、旋转星球和彗星局态。
    验证：第一碰撜体为目标且舰队路径不经过太阳。
    """

    horizon = 60
    clearance = 0.1
    speed = fleet_speed_local(jnp.maximum(ships, 1.0))
    turns = jnp.arange(1, horizon + 1, dtype=jnp.float32)
    all_px_t, all_py_t = all_planet_pos_at_turns(state, horizon)  # [B, P, horizon+1]
    gather_idx = jnp.broadcast_to(target_idx[..., None], target_idx.shape + (horizon + 1,))
    target_px_seq = jnp.take_along_axis(all_px_t, gather_idx, axis=1)
    target_py_seq = jnp.take_along_axis(all_py_t, gather_idx, axis=1)
    px_scan = target_px_seq[..., 1:]
    py_scan = target_py_seq[..., 1:]
    # 计算舰队在每个时刻是否已碰到目标
    dist = jnp.sqrt((px_scan - src[..., None, 2]) ** 2 + (py_scan - src[..., None, 3]) ** 2)
    f = dist - (src[..., None, 4] + clearance + turns * speed[..., None]) - target[..., None, 4]
    hit_candidates = f <= 0.0
    has_hit = hit_candidates.any(axis=-1)
    first_idx = jnp.argmax(hit_candidates, axis=-1)  # 最早到达时刻
    hit_turn = first_idx.astype(jnp.float32) + 1.0
    aim_turn = jnp.maximum(hit_turn.astype(jnp.int32) - 1, 0)  # 提前一步预判目标位置
    px = jnp.take_along_axis(target_px_seq, aim_turn[..., None], axis=-1).squeeze(-1)
    py = jnp.take_along_axis(target_py_seq, aim_turn[..., None], axis=-1).squeeze(-1)
    angle = jnp.arctan2(py - src[..., 3], px - src[..., 2])
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    fx0 = src[..., 2] + cos_a * (src[..., 4] + clearance)  # 舰队起始坐标
    fy0 = src[..., 3] + sin_a * (src[..., 4] + clearance)
    fx_end = fx0 + hit_turn * speed * cos_a
    fy_end = fy0 + hit_turn * speed * sin_a
    sun_hit = point_segment_distance_sq(50.0, 50.0, fx0, fy0, fx_end, fy_end) < 100.0  # 太阳半径=10
    first_collision_ok = target_is_first_collision_jax(
        src,
        target_idx,
        state,
        fx0,
        fy0,
        speed,
        cos_a,
        sin_a,
        hit_turn,
    )
    valid = has_hit & (~sun_hit) & first_collision_ok & (ships > 0.0)
    direct_angle = jnp.arctan2(target[..., 3] - src[..., 3], target[..., 2] - src[..., 2])
    return jnp.where(valid, angle, direct_angle), valid


def target_is_first_collision_jax(src, target_idx, state: EnvState, fx0, fy0, speed, cos_a, sin_a, hit_turn):
    """验证目标星球是否为舰队路径上第一碰撜体。"""
    horizon = 60
    turns = jnp.arange(1, horizon + 1, dtype=jnp.float32)
    planets = state.planets
    planet_alive = state.planet_alive
    px_t, py_t = all_planet_pos_at_turns(state, horizon)
    fx_prev = fx0[..., None] + (turns - 1.0) * speed[..., None] * cos_a[..., None]
    fy_prev = fy0[..., None] + (turns - 1.0) * speed[..., None] * sin_a[..., None]
    fx_cur = fx0[..., None] + turns * speed[..., None] * cos_a[..., None]
    fy_cur = fy0[..., None] + turns * speed[..., None] * sin_a[..., None]

    px_old = px_t[:, None, :, :-1]
    py_old = py_t[:, None, :, :-1]
    px_new = px_t[:, None, :, 1:]
    py_new = py_t[:, None, :, 1:]
    radii = planets[:, None, :, 4]
    alive = planet_alive[:, None, :, None]
    before_or_at_hit = turns[None, None, None, :] <= hit_turn[..., None, None]

    move_d2 = point_segment_distance_sq(
        px_old,
        py_old,
        fx_prev[:, :, None, :],
        fy_prev[:, :, None, :],
        fx_cur[:, :, None, :],
        fy_cur[:, :, None, :],
    )
    move_hit = (move_d2 < radii[..., None] ** 2) & alive & before_or_at_hit

    sweep_active = ((px_new - px_old) ** 2 + (py_new - py_old) ** 2) > 1e-9
    sweep_d2 = point_segment_distance_sq(
        fx_cur[:, :, None, :],
        fy_cur[:, :, None, :],
        px_old,
        py_old,
        px_new,
        py_new,
    )
    sweep_hit = (sweep_d2 < radii[..., None] ** 2) & alive & sweep_active & before_or_at_hit

    large = jnp.int32(2 * (horizon + 1) + 1)
    step_i = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    move_keys = (2 * step_i)[None, None, None, :]
    sweep_keys = (2 * step_i + 1)[None, None, None, :]
    first_move = jnp.where(move_hit, move_keys, large).min(axis=-1)
    first_sweep = jnp.where(sweep_hit, sweep_keys, large).min(axis=-1)
    first_by_planet = jnp.minimum(first_move, first_sweep)
    first_planet = first_by_planet.argmin(axis=-1)
    first_key = first_by_planet.min(axis=-1)
    return (first_key < large) & (first_planet == target_idx)


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


def nearest_opponent_actions(state, player: int, params: FeatureParams):
    """生成对手简单基线动作：每颗 player 星球攻击最近敌方。"""
    return jax.vmap(lambda planets, alive: _nearest_opponent_single(planets, alive, player, params))(state.planets, state.planet_alive)


def _nearest_opponent_single(planets, alive, player: int, params: FeatureParams):
    owner = planets[:, 1].astype(jnp.int32)
    sources = alive & (owner == player)
    targets = alive & (owner != player)
    dx = planets[None, :, 2] - planets[:, None, 2]
    dy = planets[None, :, 3] - planets[:, None, 3]
    dist = jnp.sqrt(dx * dx + dy * dy)
    masked_dist = jnp.where(targets[None, :] & (~jnp.eye(params.max_planets, dtype=bool)), dist, 1e9)
    tgt_idx = jnp.argmin(masked_dist, axis=1)
    tgt = planets[tgt_idx]
    src = planets
    ships = jnp.minimum(src[:, 5], jnp.maximum(tgt[:, 5] + 1.0, jnp.floor(src[:, 5] * 0.5)))
    valid = sources & (src[:, 5] >= MIN_LAUNCH_LOCAL) & (masked_dist.min(axis=1) < 1e8)
    angle = jnp.arctan2(tgt[:, 3] - src[:, 3], tgt[:, 2] - src[:, 2])
    return jnp.stack([jnp.where(valid, src[:, 0], 0.0), jnp.where(valid, angle, 0.0), jnp.where(valid, ships, 0.0)], axis=-1)


MIN_LAUNCH_LOCAL = 10.0


def replace_done(state: EnvState, reset: EnvState) -> EnvState:
    """局结环境用 reset 收筞替换。"""
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
