import pytest

jnp = pytest.importorskip("jax.numpy")

from jax_agent.ppo_jax import NEG_INF, safe_logits


def test_safe_logits_falls_back_when_all_actions_are_masked():
    logits = jnp.array(
        [
            [NEG_INF, NEG_INF, NEG_INF],
            [0.25, NEG_INF, -2.0],
            [NEG_INF, 0.5, NEG_INF],
        ],
        dtype=jnp.float32,
    )

    fixed = safe_logits(logits)

    assert float(fixed[0, 0]) == 0.0
    assert float(fixed[0, 1]) == NEG_INF
    assert float(fixed[0, 2]) == NEG_INF
    assert float(fixed[1, 0]) == 0.25
    assert float(fixed[1, 1]) == NEG_INF
    assert float(fixed[1, 2]) == -2.0
    assert float(fixed[2, 0]) == NEG_INF
    assert float(fixed[2, 1]) == 0.5
