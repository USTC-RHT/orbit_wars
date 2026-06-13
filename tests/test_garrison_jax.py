import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax_agent.fast_env import COMETS_PER_SPAWN, MAX_COMET_PATH, MAX_COMET_SPAWNS, EnvParams, empty_state, step_single
from jax_agent.garrison_jax import (
    garrison_status_from_arrivals_jax,
    garrison_status_jax,
    safe_drain_all_sources_jax,
    safe_drain_jax,
    target_flip_status_from_garrison_jax,
)


def _empty_actions(params: EnvParams):
    return jnp.zeros((2, params.max_actions, 3), dtype=jnp.float32)


def _noop_history(state, params: EnvParams, horizon: int):
    owners = [jnp.where(state.planet_alive, state.planets[:, 1].astype(jnp.int32), -1)]
    ships = [jnp.where(state.planet_alive, state.planets[:, 5], 0.0)]
    cur = state
    for _ in range(horizon):
        cur = step_single(cur, _empty_actions(params), params)
        owners.append(jnp.where(cur.planet_alive, cur.planets[:, 1].astype(jnp.int32), -1))
        ships.append(jnp.where(cur.planet_alive, cur.planets[:, 5], 0.0))
    return jnp.stack(owners, axis=1), jnp.stack(ships, axis=1)


def test_garrison_status_matches_noop_production():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 10.0, 2.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 20.0, 10.0, 1.0, 20.0, 1.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, -1, 30.0, 10.0, 1.0, 5.0, 3.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, True, False]),
    )

    horizon = 5
    status = garrison_status_jax(state, horizon=horizon)
    owner_ref, ships_ref = _noop_history(state, params, horizon)

    assert jnp.array_equal(status.owner, owner_ref)
    assert jnp.allclose(status.ships, ships_ref)
    assert jnp.allclose(status.pre_combat_ships[0, 1], 12.0)
    assert jnp.allclose(status.pre_combat_ships[2, 1], 5.0)


def test_garrison_status_matches_noop_existing_fleet_capture():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 20.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 20.0, 10.0, 1.0, 5.0, 0.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(jnp.array([0, 0, 12.0, 10.0, 0.0, 0, 1000.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
    )

    horizon = 4
    status = garrison_status_jax(state, horizon=horizon)
    owner_ref, ships_ref = _noop_history(state, params, horizon)

    assert jnp.array_equal(status.owner, owner_ref)
    assert jnp.allclose(status.ships, ships_ref)
    assert jnp.allclose(status.arrivals_by_owner[1, 2, 0], 1000.0)


def test_garrison_status_matches_noop_rotating_sweep_capture():
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

    status = garrison_status_jax(state, horizon=1)
    owner_ref, ships_ref = _noop_history(state, params, 1)

    assert jnp.array_equal(status.owner, owner_ref)
    assert jnp.allclose(status.ships, ships_ref)
    assert jnp.allclose(status.arrivals_by_owner[1, 1, 0], 1.0)


def test_garrison_status_does_not_count_sun_killed_fleet_as_arrival():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 30.0, 50.0, 1.0, 20.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 70.0, 50.0, 1.0, 5.0, 0.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(jnp.array([0, 0, 40.0, 50.0, 0.0, 0, 1000.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
    )

    status = garrison_status_jax(state, horizon=8)
    owner_ref, ships_ref = _noop_history(state, params, 8)

    assert jnp.array_equal(status.owner, owner_ref)
    assert jnp.allclose(status.ships, ships_ref)
    assert jnp.allclose(status.arrivals_by_owner.sum(), 0.0)


def test_safe_drain_uses_projected_future_slack():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 20.0, 10.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 10.0, 10.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(jnp.array([0, 1, 18.5, 10.0, 0.0, 1, 8.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
    )

    status = garrison_status_jax(state, horizon=3)
    drain = safe_drain_jax(status, jnp.asarray([0], dtype=jnp.int32), h_eff=3.0, player_id=0)

    assert jnp.allclose(drain[0], 2.0)


def test_target_flip_status_reports_earliest_flip_and_urgency():
    params = EnvParams(max_planets=3, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 10.0, 2.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 0, 20.0, 10.0, 1.0, 30.0, 1.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, 1, 30.0, 10.0, 1.0, 50.0, 1.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, True]),
    )
    arrivals = jnp.zeros((params.max_planets, 6, 2), dtype=jnp.float32)
    arrivals = arrivals.at[0, 2, 1].set(20.0)
    alive = jnp.broadcast_to(state.planet_alive[:, None], (params.max_planets, 6))

    status = garrison_status_from_arrivals_jax(state, arrivals, alive)
    flip = target_flip_status_from_garrison_jax(state, status, player_id=0)

    assert bool(flip.will_flip[0])
    assert int(flip.flip_turn[0]) == 2
    assert jnp.allclose(flip.urgency[0], 16.0)
    assert not bool(flip.will_flip[1])
    assert jnp.allclose(flip.urgency[1], 0.0)
    assert not bool(flip.will_flip[2])


def test_safe_drain_all_sources_supports_batched_status():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 20.0, 10.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 10.0, 10.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(jnp.array([0, 1, 18.5, 10.0, 0.0, 1, 8.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x], axis=0), state)

    status = garrison_status_jax(batched, horizon=3)
    drain = safe_drain_all_sources_jax(status, h_eff=3.0, player_id=0)

    assert drain.shape == (2, params.max_planets)
    assert jnp.allclose(drain[:, 0], 2.0)


def test_garrison_status_supports_batched_jit():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 20.0, 1.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 20.0, 10.0, 1.0, 5.0, 0.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(jnp.array([0, 0, 12.0, 10.0, 0.0, 0, 1000.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x, x], axis=0), state)

    status = jax.jit(lambda s: garrison_status_jax(s, horizon=4))(batched)

    assert status.owner.shape == (2, params.max_planets, 5)
    assert status.arrivals_by_owner.shape == (2, params.max_planets, 5, 2)
    assert jnp.allclose(status.arrivals_by_owner[:, 1, 2, 0], 1000.0)


def test_producer_recurrence_tie_leaves_precombat_garrison():
    params = EnvParams(max_planets=2, max_fleets=1, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 10.0, 1.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, False]),
    )
    arrivals = jnp.zeros((params.max_planets, 3, 2), dtype=jnp.float32)
    arrivals = arrivals.at[0, 1, 0].set(5.0)
    arrivals = arrivals.at[0, 1, 1].set(5.0)
    alive = jnp.broadcast_to(state.planet_alive[:, None], (params.max_planets, 3))

    status = garrison_status_from_arrivals_jax(state, arrivals, alive)

    assert int(status.owner[0, 1]) == 0
    assert jnp.allclose(status.pre_combat_ships[0, 1], 11.0)
    assert jnp.allclose(status.ships[0, 1], 11.0)


def test_producer_recurrence_capture_after_step_production():
    params = EnvParams(max_planets=2, max_fleets=1, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 10.0, 1.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, False]),
    )
    arrivals = jnp.zeros((params.max_planets, 3, 2), dtype=jnp.float32)
    arrivals = arrivals.at[0, 1, 1].set(15.0)
    alive = jnp.broadcast_to(state.planet_alive[:, None], (params.max_planets, 3))

    status = garrison_status_from_arrivals_jax(state, arrivals, alive)

    assert int(status.pre_combat_owner[0, 1]) == 0
    assert jnp.allclose(status.pre_combat_ships[0, 1], 11.0)
    assert int(status.owner[0, 1]) == 1
    assert jnp.allclose(status.ships[0, 1], 4.0)


def test_producer_recurrence_friendly_reinforcement_and_neutral_no_production():
    params = EnvParams(max_planets=3, max_fleets=1, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 10.0, 1.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, -1, 20.0, 10.0, 1.0, 7.0, 5.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False]),
    )
    arrivals = jnp.zeros((params.max_planets, 3, 2), dtype=jnp.float32)
    arrivals = arrivals.at[0, 1, 0].set(5.0)
    alive = jnp.broadcast_to(state.planet_alive[:, None], (params.max_planets, 3))

    status = garrison_status_from_arrivals_jax(state, arrivals, alive)

    assert int(status.owner[0, 1]) == 0
    assert jnp.allclose(status.ships[0, 1], 16.0)
    assert int(status.owner[1, 2]) == -1
    assert jnp.allclose(status.ships[1, 2], 7.0)


def test_safe_drain_doomed_source_collapses_to_current_ships_like_producer():
    params = EnvParams(max_planets=3, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 20.0, 10.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 10.0, 10.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(jnp.array([0, 1, 18.5, 10.0, 0.0, 1, 20.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
    )

    status = garrison_status_jax(state, horizon=3)
    drain = safe_drain_jax(status, jnp.asarray([0], dtype=jnp.int32), h_eff=3.0, player_id=0)

    assert int(status.owner[0, 1]) == 1
    assert jnp.allclose(drain[0], 10.0)


def test_garrison_status_matches_noop_active_comet_expiry():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    path = jnp.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=jnp.float32)
    path = path.at[0, 0, 0].set(jnp.array([20.0, 10.0], dtype=jnp.float32))
    path = path.at[0, 0, 1].set(jnp.array([24.0, 10.0], dtype=jnp.float32))
    path = path.at[0, 0, 2].set(jnp.array([28.0, 10.0], dtype=jnp.float32))
    lengths = jnp.zeros((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=jnp.int32).at[0, 0].set(3)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 5.0, 5.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, -1, 20.0, 10.0, 1.0, 7.0, 1.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, 1, 90.0, 90.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    comet_slots = jnp.full((MAX_COMET_SPAWNS, COMETS_PER_SPAWN), -1, dtype=jnp.int32).at[0, 0].set(1)
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, True, False]),
        comet_paths=path,
        comet_lengths=lengths,
        comet_path_index=jnp.asarray([0, -2, -2, -2, -2], dtype=jnp.int32),
        comet_slots=comet_slots,
    )

    horizon = 4
    status = garrison_status_jax(state, horizon=horizon)
    owner_ref, ships_ref = _noop_history(state, params, horizon)

    assert jnp.array_equal(status.owner, owner_ref)
    assert jnp.allclose(status.ships, ships_ref)
    assert jnp.array_equal(status.alive[1], jnp.array([True, True, True, False, False]))
