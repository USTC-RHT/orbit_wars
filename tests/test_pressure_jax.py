import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax_agent.fast_env import EnvParams, empty_state, fleet_speed
from jax_agent.pressure_jax import enemy_pressure_jax, pressure_delta_jax, pressure_status_jax


def test_enemy_pressure_matches_producer_formula_for_single_enemy_source():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 0.0, 0.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 0, 3.0, 4.0, 1.0, 20.0, 0.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, 1, 0.0, 10.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, True, False]),
    )

    horizon = 10.0
    pressure = enemy_pressure_jax(state, horizon=horizon, player_id=0)
    speed = fleet_speed(jnp.asarray(100.0))
    expected0 = 100.0 * jnp.clip(1.0 - 10.0 / (speed * horizon), 0.0, 1.0)
    expected1 = 100.0 * jnp.clip(1.0 - jnp.sqrt(45.0) / (speed * horizon), 0.0, 1.0)

    assert jnp.allclose(pressure[0], expected0)
    assert jnp.allclose(pressure[1], expected1)
    assert jnp.allclose(pressure[2], 0.0)
    assert jnp.allclose(pressure[3], 0.0)


def test_pressure_delta_is_target_pressure_minus_source_pressure():
    pressure = jnp.asarray([3.0, 25.0, 8.0], dtype=jnp.float32)
    delta = pressure_delta_jax(pressure)

    assert jnp.allclose(delta[0, 1], 22.0)
    assert jnp.allclose(delta[1, 0], -22.0)
    assert jnp.allclose(jnp.diag(delta), 0.0)


def test_enemy_pressure_ignores_neutral_dead_and_current_player_sources():
    params = EnvParams(max_planets=5, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 0.0, 0.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, -1, 1.0, 0.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, 1, 2.0, 0.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[3].set(jnp.array([3, 1, 3.0, 0.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[4].set(jnp.array([4, 0, 4.0, 0.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, True, False, True]),
    )

    pressure = enemy_pressure_jax(state, horizon=20.0, player_id=0)
    enemy_only = state._replace(
        planets=planets.at[0, 5].set(0.0).at[1, 5].set(0.0).at[3, 5].set(0.0).at[4, 5].set(0.0)
    )
    expected = enemy_pressure_jax(enemy_only, horizon=20.0, player_id=0)

    assert jnp.allclose(pressure, expected)
    assert jnp.allclose(pressure[2], 0.0)
    assert jnp.allclose(pressure[3], 0.0)


def test_pressure_status_supports_batched_jit():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 0.0, 0.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 0.0, 5.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x], axis=0), state)

    status = jax.jit(lambda s: pressure_status_jax(s, horizon=10.0, player_id=0))(batched)

    assert status.enemy_pressure.shape == (2, params.max_planets)
    assert status.pressure_delta.shape == (2, params.max_planets, params.max_planets)
    assert jnp.allclose(status.pressure_delta[:, 0, 1], -status.enemy_pressure[:, 0])
