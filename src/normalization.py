"""Running state normalization for PPO inputs.

训练过程中维护所有已见 state 特征的 running mean/std，并在进入网络前
对连续特征做标准化。离散标志位保持原值，因为模型和 mask 逻辑依赖它们。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import GLOBAL_DIM, TurnBatch


SELF_NORMALIZE_INDICES = np.asarray(
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    dtype=np.int64,
)
CANDIDATE_NORMALIZE_INDICES = np.asarray(
    [6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    dtype=np.int64,
)
GLOBAL_NORMALIZE_INDICES = np.arange(GLOBAL_DIM, dtype=np.int64)


@dataclass(slots=True)
class RunningMeanStd:
    """Welford-style running mean/std for a fixed feature vector."""

    shape: tuple[int, ...]
    epsilon: float = 1e-4
    mean: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    count: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        values = values.reshape(-1, *self.shape)
        batch_count = values.shape[0]
        if batch_count == 0:
            return
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = float(total_count)

    def normalize(self, values: np.ndarray, clip: float, eps: float) -> np.ndarray:
        normalized = (values - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + eps)
        return np.clip(normalized, -clip, clip)

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count, "shape": self.shape}

    def load_state_dict(self, state: dict[str, object]) -> None:
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        if mean.shape != self.shape or var.shape != self.shape:
            raise ValueError(f"RMS shape mismatch: expected {self.shape}, got mean={mean.shape}, var={var.shape}")
        self.mean = mean.copy()
        self.var = var.copy()
        self.count = float(state["count"])


class StateNormalizer:
    """Normalize TurnBatch features with running training statistics."""

    def __init__(self, *, enabled: bool = True, clip: float = 5.0, eps: float = 1e-8) -> None:
        self.enabled = enabled
        self.clip = float(clip)
        self.eps = float(eps)
        self.self_rms = RunningMeanStd((len(SELF_NORMALIZE_INDICES),))
        self.candidate_rms = RunningMeanStd((len(CANDIDATE_NORMALIZE_INDICES),))
        self.global_rms = RunningMeanStd((len(GLOBAL_NORMALIZE_INDICES),))

    def update(self, batch: TurnBatch) -> None:
        if not self.enabled or batch.self_features.shape[0] == 0:
            return
        self.self_rms.update(batch.self_features[:, SELF_NORMALIZE_INDICES])
        real_planets = batch.candidate_features[..., 1] > 0.5
        if real_planets.any():
            self.candidate_rms.update(batch.candidate_features[real_planets][:, CANDIDATE_NORMALIZE_INDICES])
        self.global_rms.update(batch.global_features[:, GLOBAL_NORMALIZE_INDICES])

    def normalize_batch(self, batch: TurnBatch) -> TurnBatch:
        if not self.enabled or batch.self_features.shape[0] == 0:
            return batch

        self_features = batch.self_features.copy()
        candidate_features = batch.candidate_features.copy()
        global_features = batch.global_features.copy()

        self_features[:, SELF_NORMALIZE_INDICES] = self.self_rms.normalize(
            self_features[:, SELF_NORMALIZE_INDICES],
            self.clip,
            self.eps,
        )
        real_planets = candidate_features[..., 1] > 0.5
        if real_planets.any():
            real_features = candidate_features[real_planets]
            real_features[:, CANDIDATE_NORMALIZE_INDICES] = self.candidate_rms.normalize(
                real_features[:, CANDIDATE_NORMALIZE_INDICES],
                self.clip,
                self.eps,
            )
            candidate_features[real_planets] = real_features
        global_features[:, GLOBAL_NORMALIZE_INDICES] = self.global_rms.normalize(
            global_features[:, GLOBAL_NORMALIZE_INDICES],
            self.clip,
            self.eps,
        )

        return TurnBatch(
            self_features=self_features.astype(np.float32, copy=False),
            candidate_features=candidate_features.astype(np.float32, copy=False),
            global_features=global_features.astype(np.float32, copy=False),
            candidate_mask=batch.candidate_mask,
            amount_mask=batch.amount_mask,
            contexts=batch.contexts,
            state=batch.state,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "clip": self.clip,
            "eps": self.eps,
            "self_rms": self.self_rms.state_dict(),
            "candidate_rms": self.candidate_rms.state_dict(),
            "global_rms": self.global_rms.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.clip = float(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))
        self.self_rms.load_state_dict(state["self_rms"])
        self.candidate_rms.load_state_dict(state["candidate_rms"])
        self.global_rms.load_state_dict(state["global_rms"])

    def copy(self) -> "StateNormalizer":
        cloned = StateNormalizer(enabled=self.enabled, clip=self.clip, eps=self.eps)
        cloned.load_state_dict(self.state_dict())
        return cloned
