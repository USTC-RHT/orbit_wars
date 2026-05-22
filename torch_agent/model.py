"""Actor-Critic 神经网络。

网络按“每颗我方来源星球一行”做决策：全图目标槽位先编码成 target_h，
target_h 一方面做 masked pooling 形成地图上下文，一方面作为 dot-product
attention keys 让当前来源星球选择目标。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class PolicyOutput:
    """网络前向输出。

    fire_logits: 每颗来源星球是否发射的 Bernoulli logit。
    target_logits: 每颗来源星球对全图目标槽位的打分；slot 0 仍保留为 no-op 占位。
    amount_logits: 已选目标对应的派兵档位打分，形状为 [batch, ship_bucket_count]。
    value: critic 对该来源星球决策状态的价值估计。
    """

    fire_logits: torch.Tensor
    target_logits: torch.Tensor
    amount_logits: torch.Tensor
    value: torch.Tensor


def mlp(input_dim: int, hidden_size: int) -> nn.Sequential:
    """两层 MLP 编码器。

    LayerNorm 让不同特征尺度更稳一点，适合这种手写特征输入。
    """

    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        # [trick] LayerNorm 消除手工特征的量纲差异（坐标/舰船数/生产力量级差距大）。
        # [vs SB3] SB3 默认不用 LayerNorm，而是对 actor/critic 末层做 orthogonal init
        # （gain=0.01 for actor, gain=1.0 for critic）来稳定初始策略；
        # 本项目用 LayerNorm 达到类似效果，且对 batch size 小的 RL 比 BatchNorm 更稳定。
        nn.LayerNorm(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )


class ActorCritic(nn.Module):
    """按来源星球决策的 Actor-Critic 网络。

    一行输入代表一颗我方来源星球。Actor 分成两个头：
    - fire head: 先判断当前来源星球这一帧是否发射。
    - target head: fire=True 时，用 src query 与全图 target keys 点积选择目标。
    - amount head: 目标采样后，只针对被选中的目标槽位选择派兵数量档位。

    Critic 输出这行决策状态的价值 V(s)。
    """

    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        ship_bucket_count: int,
        hidden_size: int = 128,
        use_cnn: bool = False,
        map_channels: int = 6,
        map_grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.ship_bucket_count = ship_bucket_count
        self.use_cnn = use_cnn
        self.map_channels = map_channels
        self.map_grid_size = map_grid_size
        self.self_encoder = mlp(self_dim, hidden_size)
        self.candidate_encoder = mlp(candidate_dim, hidden_size)
        self.global_encoder = mlp(global_dim, hidden_size)
        cnn_out_dim = 16
        if self.use_cnn:
            self.map_encoder = nn.Sequential(
                nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, cnn_out_dim),
                nn.ReLU(),
            )
        else:
            self.map_encoder = None
        context_dim = hidden_size * 3 + (cnn_out_dim if self.use_cnn else 0)
        per_target_dim = context_dim + hidden_size
        self.query = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fire_actor = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.key = nn.Linear(hidden_size, hidden_size)
        self.noop_key = nn.Parameter(torch.zeros(hidden_size))
        self.amount_actor = nn.Sequential(
            nn.Linear(per_target_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ship_bucket_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def _masked_pool(self, cand_h: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        """对真实星球槽位做 max/mean 混合池化，得到全图上下文 map_h。"""

        planet_mask = candidate_features[..., 1] > 0.5
        if not planet_mask.any(dim=1).all():
            # 极端脏输入兜底：没有真实星球时让 no-op 槽参与池化，避免全 -inf。
            planet_mask = planet_mask.clone()
            planet_mask[~planet_mask.any(dim=1), 0] = True
        masked = cand_h.masked_fill(~planet_mask.unsqueeze(-1), torch.finfo(cand_h.dtype).min)
        max_pool = masked.max(dim=1).values
        mean_pool = (cand_h * planet_mask.unsqueeze(-1).float()).sum(dim=1)
        mean_pool = mean_pool / planet_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        return 0.5 * (max_pool + mean_pool)

    def _target_keys(self, cand_h: torch.Tensor) -> torch.Tensor:
        """把目标编码转成 attention keys，并给 no-op 一个可学习 key。"""

        keys = self.key(cand_h)
        keys = keys.clone()
        keys[:, 0, :] = self.noop_key
        return keys

    def _amount_logits(
        self,
        context_h: torch.Tensor,
        cand_h: torch.Tensor,
        selected_targets: torch.Tensor,
    ) -> torch.Tensor:
        """只为已选中的目标槽位输出派兵档位 logits。"""

        row = torch.arange(selected_targets.shape[0], device=selected_targets.device)
        selected_target_h = cand_h[row, selected_targets]
        return self.amount_actor(torch.cat([context_h, selected_target_h], dim=-1))

    def _critic_value(self, context_h: torch.Tensor) -> torch.Tensor:
        return self.critic(context_h).squeeze(-1)

    def _query(self, context_h: torch.Tensor) -> torch.Tensor:
        """来源星球 query：它带着自身、全局统计、全图 pooled 上下文去匹配目标。"""

        q = self.query(context_h)
        return F.normalize(q, dim=-1)

    def _score_targets(self, query: torch.Tensor, keys: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        keys = F.normalize(keys, dim=-1)
        logits = torch.einsum("bh,bch->bc", query, keys)
        logits = logits / max(self.key.out_features ** 0.5, 1.0)
        return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)

    def forward(
        self,
        self_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        map_features: torch.Tensor | None = None,
        selected_targets: torch.Tensor | None = None,
    ) -> PolicyOutput:
        """前向传播。

        candidate_mask 会把非法目标的 target logits 置为极小值，保证
        Categorical 不会采样到这些目标。若传入 selected_targets，则额外
        只为这些已选目标计算 amount logits；否则 amount_logits 是零占位。
        """

        self_h = self.self_encoder(self_features)
        cand_h = self.candidate_encoder(candidate_features)
        global_h = self.global_encoder(global_features)
        map_h = self._masked_pool(cand_h, candidate_features)
        context_parts = [self_h, global_h, map_h]
        if self.use_cnn:
            if map_features is None:
                map_features = torch.zeros(
                    self_h.shape[0],
                    self.map_channels,
                    self.map_grid_size,
                    self.map_grid_size,
                    dtype=self_h.dtype,
                    device=self_h.device,
                )
            context_parts.append(self.map_encoder(map_features))
        context_h = torch.cat(context_parts, dim=-1)

        # [设计] Dot-product target selector：当前来源星球形成 query，
        # 全图目标编码形成 keys，query 与每个 key 的匹配分数就是 target logits。
        target_logits = self._score_targets(
            self._query(context_h),
            self._target_keys(cand_h),
            candidate_mask,
        )
        fire_logits = self.fire_actor(context_h).squeeze(-1)

        if selected_targets is None:
            amount_logits = torch.zeros(
                self_h.shape[0],
                self.ship_bucket_count,
                dtype=self_h.dtype,
                device=self_h.device,
            )
        else:
            # 自回归第二步：目标已经确定，再根据 selected target_h 决定派多少船。
            amount_logits = self._amount_logits(context_h, cand_h, selected_targets)
        value = self._critic_value(context_h)
        return PolicyOutput(
            fire_logits=fire_logits,
            target_logits=target_logits,
            amount_logits=amount_logits,
            value=value,
        )
