import pytest

jnp = pytest.importorskip("jax.numpy")

from jax_agent.ppo_jax import safe_logits


def test_safe_logits_routes_all_masked_row_to_slot_zero():
    logits = jnp.array([[-1.0e9, -1.0e9, -1.0e9], [-1.0e9, 2.0, -1.0e9]], dtype=jnp.float32)

    fixed = safe_logits(logits)

    assert fixed[0, 0] == 0.0
    assert fixed[0, 1] < -1.0e8
    assert fixed[1, 1] == 2.0
