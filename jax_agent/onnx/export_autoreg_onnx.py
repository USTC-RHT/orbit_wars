"""已废弃的自回归 ONNX 导出入口。

当前 jax_agent 已切回 torch-agent 风格的逐来源星球 ActorCritic：
fire + target pointer + amount 三头。旧自回归导出器依赖
AutoregressiveActorCritic 和 auto_* 方法，因此先显式禁用，避免误导。
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "export_autoreg_onnx.py is disabled: jax_agent now uses the per-planet "
        "ActorCritic model. Add a dedicated export_perplanet_onnx.py for this architecture."
    )


if __name__ == "__main__":
    main()
