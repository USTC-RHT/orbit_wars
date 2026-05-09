"""PPO 主训练循环。

负责并行创建环境、收集 rollout、计算 GAE、调用 PPOOptimizer 更新模型、
写 metrics 和保存 checkpoint。它串起整个 RL 训练流程。
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import TrainConfig
from .env_wrapper import OrbitWarsPPOEnv
from .features import (
    CANDIDATE_DIM,
    GLOBAL_DIM,
    MAP_CHANNELS,
    MAP_GRID_SIZE,
    SELF_DIM,
    TurnBatch,
    encode_turn,
    target_count,
)
from .model import ActorCritic
from .normalization import StateNormalizer
from .opponents import SelfPlayOpponent, actions_from_policy, build_opponent, decoded_action_from_indices
from .ppo import PPOOptimizer, RolloutBatch, sample_amounts, sample_fire, sample_targets
from .state import read


@dataclass(slots=True)
class StepGroup:
    """记录一次环境 step 对应的多条决策样本。

    Orbit Wars 一回合里，智能体可能会从多颗我方星球同时做决策；
    但环境只返回一个总 reward。这里用 indices 记住这些决策行在
    rollout 数组里的位置，后面计算 GAE 时把同一个 step reward 分给它们。
    """

    indices: list[int]
    reward: float
    done: bool


def resolve_device(name: str) -> torch.device:
    """把配置里的设备名转换成 torch.device。

    配置为 auto 时优先使用 GPU；没有 CUDA 时自动回退到 CPU。
    """

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """固定训练中会用到的随机源，方便复现实验。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig, device: torch.device) -> ActorCritic:
    """根据配置创建 Actor-Critic 网络，并放到训练设备上。"""

    return ActorCritic(
        self_dim=SELF_DIM,
        candidate_dim=CANDIDATE_DIM,
        global_dim=GLOBAL_DIM,
        candidate_count=target_count(cfg.env),
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        use_cnn=cfg.model.use_cnn,
        map_channels=MAP_CHANNELS,
        map_grid_size=MAP_GRID_SIZE,
    ).to(device)


def train(cfg: TrainConfig, resume_from: str | Path | None = None) -> None:
    """主训练循环。

    1. collect_rollout: 先让当前策略和环境互动一小段时间。
    2. compute_gae: 把每一步 reward 转成 advantage/return。
    3. PPOOptimizer.train: 用这一批 rollout 做多轮 minibatch 更新。
    4. 记录 metrics，并按间隔保存 checkpoint。
    """

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    opponent = build_opponent(cfg.opponent, cfg, device)
    envs = [OrbitWarsPPOEnv(cfg, opponent, env_index=i) for i in range(cfg.ppo.num_envs)]
    model = build_model(cfg, device)
    state_normalizer = StateNormalizer(
        enabled=cfg.state_norm.enabled,
        clip=cfg.state_norm.clip,
        eps=cfg.state_norm.eps,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr)
    ppo = PPOOptimizer(
        model,
        optimizer,
        clip_coef=cfg.ppo.clip_coef,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        epochs=cfg.ppo.epochs,
        minibatch_size=cfg.ppo.minibatch_size,
        device=device,
        norm_adv=cfg.ppo.norm_adv,
        clip_vloss=cfg.ppo.clip_vloss,
        target_kl=cfg.ppo.target_kl,
    )

    start_update = 1
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        state_normalizer.load_state_dict(ckpt["state_normalizer"])
        start_update = int(ckpt["update"]) + 1
        print(f"Resumed from {resume_from}, starting at update={start_update}", flush=True)

    if isinstance(opponent, SelfPlayOpponent):
        opponent.sync_from(model, state_normalizer)

    next_seed = cfg.seed
    batches = []
    for env in envs:
        batches.append(env.reset(next_seed))
        next_seed += 1

    run_dir = Path(cfg.save_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(cfg, run_dir, device)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        csv.writer(f).writerow(
            [
                "update",
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "old_approx_kl",
                "approx_kl",
                "clipfrac",
                "explained_variance",
                "n_updates",
                "learning_rate",
                "episode_reward_mean",
                "terminal_reward_mean",
                "neutral_capture_reward_mean",
                "enemy_capture_reward_mean",
                "lose_planet_reward_mean",
                "wasted_attack_reward_mean",
                "fire_rate",
                "episodes",
                "win_rate",
                "wins",
                "losses",
                "draws",
                "det_eval_win_rate",
                "det_eval_wins",
                "det_eval_losses",
                "det_eval_draws",
            ]
        )

    print(
        f"run_dir={run_dir} device={device} envs={cfg.ppo.num_envs} "
        f"rollout_steps={cfg.ppo.rollout_steps} updates={cfg.ppo.total_updates}",
        flush=True,
    )

    for update in range(start_update, cfg.ppo.total_updates + 1):
        if cfg.ppo.anneal_lr:
            # [SB3 trick] lr 线性退火到 0：训练后期步长收缩相当于隐式正则，防止策略在收敛后震荡。
            # SB3 在 on_step 回调里实现，这里等价地在每个 update 开始时手动更新。
            frac = 1.0 - (update - 1.0) / float(cfg.ppo.total_updates)
            lr_now = frac * cfg.ppo.lr
            optimizer.param_groups[0]["lr"] = lr_now
        else:
            lr_now = cfg.ppo.lr

        rollout, batches, next_seed, stats = collect_rollout(
            envs,
            batches,
            model,
            cfg,
            device,
            next_seed,
            state_normalizer,
        )
        metrics = ppo.train(rollout)

        if isinstance(opponent, SelfPlayOpponent) and update % cfg.self_play_update_interval == 0:
            opponent.sync_from(model, state_normalizer)

        eval_stats = maybe_eval_deterministic(
            model,
            cfg,
            device,
            state_normalizer,
            update,
        )

        if update % cfg.log_every == 0:
            eval_text = ""
            if eval_stats is not None:
                eval_text = (
                    f" det_winrate={eval_stats['det_eval_win_rate']:.3f} "
                    f"det_W/L/D={int(eval_stats['det_eval_wins'])}/"
                    f"{int(eval_stats['det_eval_losses'])}/"
                    f"{int(eval_stats['det_eval_draws'])}"
                )
            print(
                f"update={update} loss={metrics['loss']:.4f} "
                f"reward={stats['episode_reward_mean']:.4f} "
                f"terminal={stats['terminal_reward_mean']:.4f} "
                f"ent={metrics['entropy']:.3f} ev={metrics['explained_variance']:.3f} "
                f"fire={stats['fire_rate']:.3f} "
                f"episodes={int(stats['episodes'])} winrate={stats['win_rate']:.3f} "
                f"W/L/D={int(stats['wins'])}/{int(stats['losses'])}/{int(stats['draws'])}"
                f"{eval_text}",
                flush=True,
            )
            with metrics_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        update,
                        metrics["loss"],
                        metrics["policy_loss"],
                        metrics["value_loss"],
                        metrics["entropy"],
                        metrics["old_approx_kl"],
                        metrics["approx_kl"],
                        metrics["clipfrac"],
                        metrics["explained_variance"],
                        metrics["n_updates"],
                        lr_now,
                        stats["episode_reward_mean"],
                        stats["terminal_reward_mean"],
                        stats["neutral_capture_reward_mean"],
                        stats["enemy_capture_reward_mean"],
                        stats["lose_planet_reward_mean"],
                        stats["wasted_attack_reward_mean"],
                        stats["fire_rate"],
                        int(stats["episodes"]),
                        stats["win_rate"],
                        int(stats["wins"]),
                        int(stats["losses"]),
                        int(stats["draws"]),
                        "" if eval_stats is None else eval_stats["det_eval_win_rate"],
                        "" if eval_stats is None else int(eval_stats["det_eval_wins"]),
                        "" if eval_stats is None else int(eval_stats["det_eval_losses"]),
                        "" if eval_stats is None else int(eval_stats["det_eval_draws"]),
                    ]
                )
            # wandb 只保留 reward / win_rate / loss 相关的核心指标，
            # 避免面板被大量训练细节淹没。
            wandb_payload: dict[str, float] = {
                "reward": float(stats["episode_reward_mean"]),
                "neutral_capture_reward": float(stats["neutral_capture_reward_mean"]),
                "enemy_capture_reward": float(stats["enemy_capture_reward_mean"]),
                "lose_planet_reward": float(stats["lose_planet_reward_mean"]),
                "wasted_attack_reward": float(stats["wasted_attack_reward_mean"]),
                "fire_rate": float(stats["fire_rate"]),
                "win_rate": float(stats["win_rate"]),
                "loss": float(metrics["loss"]),
                "policy_loss": float(metrics["policy_loss"]),
                "value_loss": float(metrics["value_loss"]),
            }
            if eval_stats is not None:
                wandb_payload["det_eval_win_rate"] = float(eval_stats["det_eval_win_rate"])
            log_wandb(
                wandb_run,
                update,
                wandb_payload,
            )

        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(run_dir, update, model, optimizer, cfg, state_normalizer)
            if wandb_run is not None:
                wandb_run.save(str(run_dir / "ckpt_last.pt"), policy="now")

    if wandb_run is not None:
        wandb_run.finish()


def init_wandb(cfg: TrainConfig, run_dir: Path, device: torch.device) -> Any | None:
    """初始化 wandb 记录器。

    只有 cfg.wandb.enabled=True 时才会启用。这样本地调试、Kaggle 推理都
    不依赖 wandb；需要实验追踪时，用 `RL/train.py --wandb` 打开即可。
    """

    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continue without wandb logging.", flush=True)
        return None

    try:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=cfg.run_name,
            group=cfg.wandb.group or None,
            tags=list(cfg.wandb.tags),
            mode=cfg.wandb.mode,
            dir=str(run_dir),
            config={
                **asdict(cfg),
                "resolved_device": str(device),
                "run_dir": str(run_dir),
            },
        )
        print(
            f"wandb enabled project={cfg.wandb.project} mode={cfg.wandb.mode} "
            f"run={cfg.run_name}",
            flush=True,
        )
        return run
    except Exception as exc:
        print(f"wandb init failed: {exc!r}; continue without wandb logging.", flush=True)
        return None


def log_wandb(wandb_run: Any | None, update: int, values: dict[str, float]) -> None:
    """把一个 update 的训练指标写到 wandb。"""

    if wandb_run is None:
        return
    payload = {}
    for key, value in values.items():
        if isinstance(value, (int, float, np.floating)):
            payload[key] = float(value)
    wandb_run.log(payload, step=update)


def collect_rollout(
    envs: list[OrbitWarsPPOEnv],
    batches: list[TurnBatch],
    model: ActorCritic,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
    state_normalizer: StateNormalizer,
) -> tuple[RolloutBatch, list[TurnBatch], int, dict[str, float]]:
    """收集 rollout。

    rollout 可以理解成“训练前先攒的一段游戏经历”。每一行样本对应：
    一颗来源星球看到的特征、它选择的目标、派兵档位、旧策略概率和 value。
    后面 PPO 会拿这些旧经历反复训练几轮。
    """

    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    map_rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    amount_masks: list[np.ndarray] = []
    fire_actions: list[bool] = []
    target_actions: list[int] = []
    amount_actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    groups_by_env: list[list[StepGroup]] = [[] for _ in envs]
    episode_rewards = [0.0 for _ in envs]
    finished_rewards: list[float] = []
    finished_terminals: list[float] = []
    neutral_capture_rewards: list[float] = []
    enemy_capture_rewards: list[float] = []
    lose_planet_rewards: list[float] = []
    wasted_attack_rewards: list[float] = []
    wins = 0
    losses = 0
    draws = 0

    for _ in range(cfg.ppo.rollout_steps):
        raw_merged = merge_batches(batches, cfg)
        state_normalizer.update(raw_merged)
        merged = state_normalizer.normalize_batch(raw_merged)
        normalized_batches = [state_normalizer.normalize_batch(batch) for batch in batches]
        row_values = np.zeros((merged.self_features.shape[0],), dtype=np.float32)
        sampled_fires = np.zeros((merged.self_features.shape[0],), dtype=bool)
        sampled_targets = np.zeros((merged.self_features.shape[0],), dtype=np.int64)
        sampled_amounts = np.zeros((merged.self_features.shape[0],), dtype=np.int64)
        sampled_log_probs = np.zeros((merged.self_features.shape[0],), dtype=np.float32)
        offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])

        if merged.self_features.shape[0] > 0:
            # [性能] inference_mode 比 no_grad 更快：完全禁用 autograd 引擎，采样阶段无需任何梯度记录。
            # SB3 用 th.no_grad()；这里用 inference_mode 是 PyTorch 1.9+ 的更优选择。
            with torch.inference_mode():
                target_outputs = model(
                    torch.from_numpy(merged.self_features).to(device),
                    torch.from_numpy(merged.candidate_features).to(device),
                    torch.from_numpy(merged.global_features).to(device),
                    torch.from_numpy(merged.candidate_mask).to(device).bool(),
                    map_features=torch.from_numpy(merged.map_features).to(device),
                )
                # 训练 rollout 必须采样动作，PPO 才有探索；先采样 fire，再采样 target。
                candidate_mask_tensor = torch.from_numpy(merged.candidate_mask).to(device).bool()
                fire_sample = sample_fire(target_outputs, candidate_mask_tensor, deterministic=False)
                target_sample = sample_targets(
                    target_outputs,
                    fire_sample,
                    candidate_mask_tensor,
                    deterministic=False,
                )
                amount_outputs = model(
                    torch.from_numpy(merged.self_features).to(device),
                    torch.from_numpy(merged.candidate_features).to(device),
                    torch.from_numpy(merged.global_features).to(device),
                    torch.from_numpy(merged.candidate_mask).to(device).bool(),
                    map_features=torch.from_numpy(merged.map_features).to(device),
                    selected_targets=target_sample.target,
                )
                sample = sample_amounts(
                    amount_outputs,
                    target_sample,
                    # 自回归第二步同样采样：先前采样的 target 决定当前 amount 分布。
                    deterministic=False,
                    amount_mask=torch.from_numpy(merged.amount_mask).to(device).bool(),
                )
                row_values = target_outputs.value.detach().cpu().numpy()
                sampled_fires = sample.fire.detach().cpu().numpy()
                sampled_targets = sample.target.detach().cpu().numpy()
                sampled_amounts = sample.amount.detach().cpu().numpy()
                sampled_log_probs = sample.log_prob.detach().cpu().numpy()

        next_batches: list[TurnBatch] = []
        for env_idx, env in enumerate(envs):
            batch = normalized_batches[env_idx]
            start = int(offsets[env_idx])
            move_list: list[list[float | int]] = []
            action_infos: list[dict[str, Any]] = []
            group_indices: list[int] = []

            for local_idx, context in enumerate(batch.contexts):
                global_idx = start + local_idx
                self_rows.append(batch.self_features[local_idx])
                candidate_rows.append(batch.candidate_features[local_idx])
                global_rows.append(batch.global_features[local_idx])
                map_rows.append(batch.map_features[local_idx])
                masks.append(batch.candidate_mask[local_idx])
                amount_masks.append(batch.amount_mask[local_idx])
                fire_actions.append(bool(sampled_fires[global_idx]))
                target_actions.append(int(sampled_targets[global_idx]))
                amount_actions.append(int(sampled_amounts[global_idx]))
                log_probs.append(float(sampled_log_probs[global_idx]))
                values.append(float(row_values[global_idx]))
                group_indices.append(len(values) - 1)

                decoded = decoded_action_from_indices(
                    batch.state,
                    context,
                    int(sampled_targets[global_idx]),
                    int(sampled_amounts[global_idx]),
                ) if bool(sampled_fires[global_idx]) else None
                if decoded is not None:
                    move_list.append(decoded.move)
                    action_infos.append(
                        {
                            "player": batch.state.player,
                            "source_id": decoded.source_id,
                            "target_id": decoded.target_id,
                            "target_owner": decoded.target_owner,
                            "target_production": decoded.target_production,
                            "ships": decoded.ships,
                            "eta": decoded.eta,
                        }
                    )

            result = env.step(move_list, action_infos)
            episode_rewards[env_idx] += result.reward
            neutral_capture_rewards.append(float(result.info.get("neutral_capture_reward", 0.0)))
            enemy_capture_rewards.append(float(result.info.get("enemy_capture_reward", 0.0)))
            lose_planet_rewards.append(float(result.info.get("lose_planet_reward", 0.0)))
            wasted_attack_rewards.append(float(result.info.get("wasted_attack_reward", 0.0)))
            groups_by_env[env_idx].append(StepGroup(group_indices, result.reward, result.done))
            if result.done:
                finished_rewards.append(episode_rewards[env_idx])
                finished_terminals.append(float(result.info.get("terminal_reward", 0.0)))
                raw_terminal = float(result.info.get("raw_terminal_reward", 0.0))
                if raw_terminal > 0.0:
                    wins += 1
                elif raw_terminal < 0.0:
                    losses += 1
                else:
                    draws += 1
                episode_rewards[env_idx] = 0.0
                # 开启新 episode
                next_batch = env.reset(next_seed)
                next_seed += 1
            else:
                next_batch = result.batch
            next_batches.append(next_batch)

        batches = next_batches

    returns, advantages = compute_gae(
        values,
        groups_by_env,
        bootstrap_values(model, batches, device, state_normalizer),
        cfg,
    )
    n_targets = target_count(cfg.env)
    rollout = RolloutBatch(
        self_features=torch.from_numpy(np.asarray(self_rows, dtype=np.float32).reshape(-1, SELF_DIM)),
        candidate_features=torch.from_numpy(
            np.asarray(candidate_rows, dtype=np.float32).reshape(-1, n_targets, CANDIDATE_DIM)
        ),
        global_features=torch.from_numpy(np.asarray(global_rows, dtype=np.float32).reshape(-1, GLOBAL_DIM)),
        map_features=torch.from_numpy(
            np.asarray(map_rows, dtype=np.float32).reshape(-1, MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE)
        ),
        candidate_mask=torch.from_numpy(np.asarray(masks, dtype=bool).reshape(-1, n_targets)),
        amount_mask=torch.from_numpy(
            np.asarray(amount_masks, dtype=bool).reshape(-1, n_targets, cfg.env.ship_bucket_count)
        ),
        fire_actions=torch.tensor(fire_actions, dtype=torch.bool),
        target_actions=torch.tensor(target_actions, dtype=torch.long),
        amount_actions=torch.tensor(amount_actions, dtype=torch.long),
        old_log_probs=torch.tensor(log_probs, dtype=torch.float32),
        old_values=torch.tensor(values, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
    stats = {
        "episode_reward_mean": float(np.mean(finished_rewards)) if finished_rewards else 0.0,
        "terminal_reward_mean": float(np.mean(finished_terminals)) if finished_terminals else 0.0,
        "neutral_capture_reward_mean": float(np.mean(neutral_capture_rewards)) if neutral_capture_rewards else 0.0,
        "enemy_capture_reward_mean": float(np.mean(enemy_capture_rewards)) if enemy_capture_rewards else 0.0,
        "lose_planet_reward_mean": float(np.mean(lose_planet_rewards)) if lose_planet_rewards else 0.0,
        "wasted_attack_reward_mean": float(np.mean(wasted_attack_rewards)) if wasted_attack_rewards else 0.0,
        "fire_rate": float(np.mean(fire_actions)) if fire_actions else 0.0,
        "episodes": float(len(finished_rewards)),
        "win_rate": float(wins / max(len(finished_rewards), 1)),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
    }
    return rollout, batches, next_seed, stats


def maybe_eval_deterministic(
    model: ActorCritic,
    cfg: TrainConfig,
    device: torch.device,
    state_normalizer: StateNormalizer,
    update: int,
) -> dict[str, float] | None:
    """按配置周期性评估 argmax 策略的真实胜率。"""

    if not cfg.eval.enabled or cfg.eval.every <= 0 or cfg.eval.games <= 0:
        return None
    if update % cfg.eval.every != 0:
        return None
    # 固定 base_seed：跨 update 复用同一组评估 seed，
    # 这样胜率曲线只反映策略变化，不受 seed 抽样噪声影响。
    return eval_deterministic_policy(
        model,
        cfg,
        device,
        state_normalizer,
        games=cfg.eval.games,
        base_seed=cfg.eval.seed,
    )


def eval_deterministic_policy(
    model: ActorCritic,
    cfg: TrainConfig,
    device: torch.device,
    state_normalizer: StateNormalizer,
    *,
    games: int,
    base_seed: int,
) -> dict[str, float]:
    """用当前模型 argmax 策略完整打若干局。

    这个指标比 rollout win_rate 更接近 lab / 提交时的行为。评估时轮换
    learner 的 player 0/1 身份，避免只看先手或后手。
    """

    was_training = model.training
    model.eval()
    wins = 0
    losses = 0
    draws = 0
    try:
        for game_idx in range(games):
            learner_player = game_idx % 2 if cfg.alternate_player_sides else 0
            reward = play_eval_game(
                model,
                cfg,
                device,
                state_normalizer,
                seed=base_seed + game_idx,
                learner_player=learner_player,
            )
            if reward > 0.0:
                wins += 1
            elif reward < 0.0:
                losses += 1
            else:
                draws += 1
    finally:
        if was_training:
            model.train()

    total = max(wins + losses + draws, 1)
    return {
        "det_eval_win_rate": float(wins / total),
        "det_eval_wins": float(wins),
        "det_eval_losses": float(losses),
        "det_eval_draws": float(draws),
    }


def play_eval_game(
    model: ActorCritic,
    cfg: TrainConfig,
    device: torch.device,
    state_normalizer: StateNormalizer,
    *,
    seed: int,
    learner_player: int,
) -> float:
    """跑一局 deterministic eval，返回 learner 的 Kaggle 终局 reward。"""

    from kaggle_environments import make

    opponent = build_opponent(cfg.opponent, cfg, device)
    env = make("orbit_wars", configuration={"seed": int(seed), "randomSeed": int(seed)}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])

    while read(states[learner_player], "status") == "ACTIVE":
        learner_obs = read(states[learner_player], "observation")
        opponent_obs = read(states[1 - learner_player], "observation")
        action = actions_from_policy(
            encode_turn(learner_obs, cfg.env),
            model,
            device,
            deterministic=True,
            state_normalizer=state_normalizer,
        )
        opponent_action = opponent.act(opponent_obs)
        joint = [action, opponent_action] if learner_player == 0 else [opponent_action, action]
        states = env.step(joint)

    reward = read(states[learner_player], "reward", 0.0)
    return 0.0 if reward is None else float(reward)


def compute_gae(
    values: list[float],
    groups_by_env: list[list[StepGroup]],
    next_values: list[float],
    cfg: TrainConfig,
) -> tuple[list[float], list[float]]:
    """用 GAE(lambda) 从 reward/value 推出 return 和 advantage。

    advantage 表示“这次动作比 critic 预期好多少”。因为同一回合可能有
    多颗来源星球一起行动，这里先用这些决策行的平均 value 代表该回合
    的 V(s)，再把算出的 advantage 分配回同一回合的所有决策行。
    [游戏特殊 vs SB3] SB3 的 GAE 是 per-step 一个 value；这里因为每步有
    多颗星球，用 value_mean 代表 V(s) 后广播给同一步的所有星球行。
    return target 统一为 value_mean + gae，与 delta 计算时的基准一致，
    保证 critic 学习目标和 GAE 推导不矛盾。
    """

    returns = [0.0 for _ in values]
    advantages = [0.0 for _ in values]
    for env_idx, groups in enumerate(groups_by_env):
        next_value = next_values[env_idx]
        gae = 0.0
        for group in reversed(groups):
            if not group.indices:
                continue
            value_mean = float(np.mean([values[i] for i in group.indices]))
            nonterminal = 0.0 if group.done else 1.0
            delta = group.reward + cfg.ppo.gamma * next_value * nonterminal - value_mean
            gae = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nonterminal * gae
            # return_target 与 delta 的基准保持一致，均使用 value_mean。
            return_target = value_mean + gae
            for idx in group.indices:
                advantages[idx] = gae
                returns[idx] = return_target
            next_value = value_mean
    return returns, advantages


def bootstrap_values(
    model: ActorCritic,
    batches: list[TurnBatch],
    device: torch.device,
    state_normalizer: StateNormalizer,
) -> list[float]:
    """估计 rollout 截断处的 value。

    rollout 不一定刚好在一局结束时停下，所以最后一帧还要用 critic 估计
    一个 next_value，给 GAE 递推收尾。
    """

    merged = state_normalizer.normalize_batch(merge_batches(batches, None))
    if merged.self_features.shape[0] == 0:
        return [0.0 for _ in batches]
    offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
    with torch.inference_mode():
        outputs = model(
            torch.from_numpy(merged.self_features).to(device),
            torch.from_numpy(merged.candidate_features).to(device),
            torch.from_numpy(merged.global_features).to(device),
            torch.from_numpy(merged.candidate_mask).to(device).bool(),
            map_features=torch.from_numpy(merged.map_features).to(device),
        )
    values = outputs.value.detach().cpu().numpy()
    per_env = []
    for env_idx, batch in enumerate(batches):
        start = int(offsets[env_idx])
        count = batch.self_features.shape[0]
        # [游戏特殊] 一帧里可能有多颗我方星球各自输出 value，取均值代表这一帧的状态价值 V(s)。
        # SB3 每步只有一个 value；这里因为 per-planet 决策，需要聚合回单一状态价值。
        per_env.append(0.0 if count == 0 else float(values[start : start + count].mean()))
    return per_env


def merge_batches(batches: list[TurnBatch], cfg: TrainConfig | None) -> TurnBatch:
    """把多个并行环境的 TurnBatch 拼成一个大 batch。

    这样模型只需要一次 forward，就能同时给所有环境、所有来源星球采样动作。
    """

    if not batches:
        raise ValueError("No batches to merge.")
    has_rows = any(batch.self_features.shape[0] > 0 for batch in batches)
    if has_rows:
        self_features = np.concatenate([batch.self_features for batch in batches], axis=0)
        candidate_features = np.concatenate([batch.candidate_features for batch in batches], axis=0)
        global_features = np.concatenate([batch.global_features for batch in batches], axis=0)
        map_features = np.concatenate([batch.map_features for batch in batches], axis=0)
        candidate_mask = np.concatenate([batch.candidate_mask for batch in batches], axis=0)
        amount_mask = np.concatenate([batch.amount_mask for batch in batches], axis=0)
    else:
        candidate_count = target_count(cfg.env) if cfg is not None else batches[0].candidate_features.shape[1]
        ship_bucket_count = cfg.env.ship_bucket_count if cfg is not None else batches[0].amount_mask.shape[2]
        self_features = np.zeros((0, SELF_DIM), dtype=np.float32)
        candidate_features = np.zeros((0, candidate_count, CANDIDATE_DIM), dtype=np.float32)
        global_features = np.zeros((0, GLOBAL_DIM), dtype=np.float32)
        map_features = np.zeros((0, MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE), dtype=np.float32)
        candidate_mask = np.zeros((0, candidate_count), dtype=bool)
        amount_mask = np.zeros((0, candidate_count, ship_bucket_count), dtype=bool)
    return TurnBatch(
        self_features=self_features,
        candidate_features=candidate_features,
        global_features=global_features,
        map_features=map_features,
        candidate_mask=candidate_mask,
        amount_mask=amount_mask,
        contexts=[ctx for batch in batches for ctx in batch.contexts],
        state=batches[0].state,
    )


def save_checkpoint(
    run_dir: Path,
    update: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    state_normalizer: StateNormalizer,
) -> None:
    """保存模型。

    ckpt_last.pt 始终指向最近一次保存；ckpt_000100.pt 这类文件方便回看
    某个训练阶段的策略。
    """

    payload = {
        "update": update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "state_normalizer": state_normalizer.state_dict(),
    }
    torch.save(payload, run_dir / "ckpt_last.pt")
    torch.save(payload, run_dir / f"ckpt_{update:06d}.pt")
