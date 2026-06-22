"""JAX 自回归训练的配置 dataclass 和 YAML 加载工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EnvConfig:
    """环境和动作空间相关配置。

    max_planets 控制全图目标槽位数量：index 0 保留为 no-op，后面最多
    放 max_planets 个当前星球；ship_bucket_count 是网络可选择的派兵档位。
    """

    board_size: float = 100.0
    episode_steps: int = 500
    ship_bucket_count: int = 6
    max_planets: int = 45
    max_ships: float = 500.0
    max_production: float = 5.0
    target_filter_enabled: bool = False
    attack_top_k: int = 6
    defense_top_k: int = 3


@dataclass(slots=True)
class RewardConfig:
    """奖励塑形配置。

    terminal_weight 控制终局胜负的权重；其它权重用于每步局势变化奖励。
    step_clip 限制单步 shaping 幅度，避免压过最终胜负。
    """

    terminal_weight: float = 1.0
    ships_weight: float = 0.0015
    production_weight: float = 0.025
    planet_weight: float = 0.035
    time_penalty: float = 0.0
    launch_penalty: float = 0.0
    neutral_capture_weight: float = 0.0
    neutral_capture_eta_scale: float = 12.0
    neutral_capture_early_steps: int = 150
    enemy_capture_weight: float = 0.0
    lose_planet_penalty: float = 0.0
    wasted_attack_penalty: float = 0.0
    step_clip: float = 0.20
    failed_capture_penalty: float = 0.0   # 发射后 N 步内目标未变我方的惩罚（0=关闭）
    failed_capture_window: int = 50       # 检测窗口步数


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int = 128
    use_cnn: bool = False
    use_transformer: bool = False
    actor_type: str = "autoregressive"
    max_action_tokens: int = 16
    end_bias_init: float = -2.0
    num_transformer_layers: int = 2
    num_heads: int = 4
    max_fleet_tokens: int = 32


@dataclass(slots=True)
class StateNormConfig:
    """输入 state running normalization 配置。"""

    enabled: bool = True
    clip: float = 5.0
    eps: float = 1e-8


@dataclass(slots=True)
class EvalConfig:
    """训练中 deterministic 策略评估配置。"""

    enabled: bool = True
    every: int = 100
    nearest_every: int = 0    # nearest 专属间隔（0 = 跟随 every）
    slow_every: int = 0       # 外部脚本对手间隔（0 = 跟随 every）
    games: int = 20
    seed: int = 100000
    player: int = 0          # 0=learner as player0, 1=learner as player1 during eval
    opponents: list[str] = field(default_factory=lambda: ["nearest", "sigma", "tamrazov"])


@dataclass(slots=True)
class PPOConfig:
    """PPO 超参数，命名尽量贴近 CleanRL。

    anneal_lr / target_kl / norm_adv / clip_vloss 是 CleanRL 常见工程开关。
    """

    rollout_steps: int = 32
    num_envs: int = 4
    total_updates: int = 300
    epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    ent_coef_fire: float = -1.0
    ent_coef_target: float = -1.0
    ent_coef_amount: float = -1.0
    ent_schedule: str = "constant"
    ent_min_ratio: float = 1.0
    vf_coef: float = 0.5
    lr: float = 3e-4
    lr_schedule: str = "auto"
    warmup_updates: int = 0
    min_lr_ratio: float = 0.0
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: float = 0.03


@dataclass(slots=True)
class WandbConfig:
    """wandb 实验记录配置。

    enabled 控制是否启用 wandb；mode 可设为 online/offline/disabled。
    没登录 wandb 时可以先用 offline，本地会生成 wandb run，之后再同步。
    """

    enabled: bool = False
    project: str = "orbit-wars-rl"
    entity: str | None = None
    mode: str = "online"
    group: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrainConfig:
    """一次训练运行的完整配置。"""

    seed: int = 42
    run_name: str = "orbit_wars_clean_ppo"
    device: str = "auto"
    save_dir: str = "outputs"
    checkpoint_every: int = 10
    log_every: int = 1
    opponent: str = "nearest"
    opponent_pool: list[str] = field(default_factory=lambda: ["nearest"])
    opponent_pool_weights: list[float] = field(default_factory=lambda: [1.0])
    init_from_checkpoint: str | None = None
    start_update_from_checkpoint: bool = True
    self_play_update_interval: int = 25
    pool_mode: str = "pfsp"
    pool_start_phase: int = 1
    pool_include_nearest: bool = True
    pool_lag_checkpoint_phase1_opponents: list[str] = field(default_factory=lambda: ["selfplay-lag"])
    pool_lag_checkpoint_phase2_opponents: list[str] = field(default_factory=lambda: ["selfplay-current", "selfplay-lag"])
    pool_lag_interval: int = 50
    pool_max_size: int = 4
    promote_nearest_win_rate: float = 0.8
    promote_consecutive_evals: int = 3
    # 快照池：每 snapshot_interval 存一个快照，按 snapshot_current_ratio 概率选当前策略
    snapshot_interval: int = 200
    snapshot_current_ratio: float = 0.2
    # self-play 对手本质上是在“推理”旧策略，默认用 argmax 保持行为稳定。
    self_play_deterministic: bool = True
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    state_norm: StateNormConfig = field(default_factory=StateNormConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_config(path: str | Path) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping.")
    cfg = TrainConfig()
    update_dataclass(cfg, data, skip={"env", "reward", "model", "state_norm", "eval", "ppo", "wandb"})
    update_dataclass(cfg.env, data.get("env", {}))
    update_dataclass(cfg.reward, data.get("reward", {}))
    update_dataclass(cfg.model, data.get("model", {}))
    update_dataclass(cfg.state_norm, data.get("state_norm", {}))
    update_dataclass(cfg.eval, data.get("eval", {}))
    update_dataclass(cfg.ppo, data.get("ppo", {}))
    update_dataclass(cfg.wandb, data.get("wandb", {}))
    return cfg


def update_dataclass(obj: Any, values: Any, skip: set[str] | None = None) -> None:
    if not isinstance(values, dict):
        return
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        setattr(obj, key, coerce(value, current))


def coerce(value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value
