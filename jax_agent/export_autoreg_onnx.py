"""Deprecated autoregressive ONNX export entrypoint.

The current jax_agent has been switched back to the torch-agent-style
per-planet ActorCritic model: fire + target pointer + amount heads.
The previous autoregressive exporter expected AutoregressiveActorCritic and
auto_* methods, so it is intentionally disabled until a per-planet exporter is
added.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "export_autoreg_onnx.py is disabled: jax_agent now uses the per-planet "
        "ActorCritic model. Add a dedicated export_perplanet_onnx.py for this architecture."
    )


if __name__ == "__main__":
    main()
