import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax_agent.fast_env import EnvParams, empty_state, step_single


def _empty_actions(params: EnvParams):
    return jnp.zeros((2, params.max_actions, 3), dtype=jnp.float32)


def test_near_surface_launch_hits_under_strict_engine_collision():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 8.9, 21.0, 1.0, 1000.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 13.0, 20.001, 1.0, 0.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
    )

    actions = _empty_actions(params)
    actions = actions.at[0, 0].set(jnp.array([0.0, 0.0, 1000.0], dtype=jnp.float32))

    next_state = step_single(state, actions, params)

    assert int(next_state.planets[1, 1]) == 0
    assert not bool(next_state.fleet_alive[0])


def test_rotating_planet_sweep_hits_fleet_under_strict_engine_collision():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    omega = jnp.asarray(0.05, dtype=jnp.float32)
    old_x = 80.0
    old_y = 50.0
    new_x = 50.0 + 30.0 * jnp.cos(omega)
    new_y = 50.0 + 30.0 * jnp.sin(omega)

    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 5.0, 5.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, old_x, old_y, 1.0, 0.0, 0.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(
        jnp.array([0, 0, new_x, new_y + 1.0, -jnp.pi / 2.0, 0, 1.0], dtype=jnp.float32)
    )
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
        step=jnp.asarray(1, dtype=jnp.int32),
        angular_velocity=omega,
    )

    next_state = step_single(state, _empty_actions(params), params)

    assert int(next_state.planets[1, 1]) == 0
    assert not bool(next_state.fleet_alive[0])


def test_aim_with_prediction_hits_rotating_target_within_60_steps():
    from jax_agent.physics import aim_with_prediction_jax

    params = EnvParams(max_planets=4, max_fleets=8, max_actions=1)
    state = empty_state(params)
    omega = jnp.asarray(-0.028790285293132704, dtype=jnp.float32)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 14.509383, 85.06891, 1.2, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 29.247955, 64.29872, 1.5, 0.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
        step=jnp.asarray(1, dtype=jnp.int32),
        angular_velocity=omega,
    )
    batched = jax.tree_util.tree_map(lambda x: x[None], state)
    angle, hit = aim_with_prediction_jax(
        state.planets[0].reshape(1, 1, 7),
        state.planets[1].reshape(1, 1, 7),
        jnp.asarray([[1]], dtype=jnp.int32),
        batched,
        jnp.asarray([[80.0]], dtype=jnp.float32),
    )

    assert bool(hit[0, 0])
    actions = _empty_actions(params)
    actions = actions.at[0, 0].set(jnp.array([0.0, angle[0, 0], 80.0], dtype=jnp.float32))
    cur = step_single(state, actions, params)
    for _ in range(59):
        if int(cur.planets[1, 1]) == 0:
            break
        cur = step_single(cur, _empty_actions(params), params)

    assert int(cur.planets[1, 1]) == 0
