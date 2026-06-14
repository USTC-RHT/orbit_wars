import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax_agent.fast_env import EnvParams, empty_state
from jax_agent.features_jax import (
    INBOUND_HORIZON,
    PLANET_DIM,
    RELATIVE_DIM,
    FeatureParams,
    encode_batch,
    friendly_defense_score_jax,
    relative_capture_floor_features,
    relative_eta_garrison_features,
)
from jax_agent.garrison_jax import GarrisonStatus
from jax_agent.physics import aim_with_prediction_eta_jax


def test_target_filter_defense_topk_keeps_projected_friendly_flip():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 1000.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 0, 18.0, 10.0, 1.0, 5.0, 2.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, 0, 35.0, 10.0, 1.0, 50.0, 1.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(jnp.array([0, 1, 12.0, 10.0, 0.0, 2, 20.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, True, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=4, target_filter_enabled=True, attack_top_k=0, defense_top_k=1)

    encoded = encode_batch(batched, fp)
    src0_mask = encoded.candidate_mask[0, 0]

    assert bool(src0_mask[0])
    assert bool(src0_mask[2])
    assert not bool(src0_mask[3])


def test_source_mask_requires_min_launch_ships():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 9.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 0, 20.0, 10.0, 1.0, 10.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)

    encoded = encode_batch(batched, FeatureParams(max_planets=4, target_filter_enabled=False))

    assert not bool(encoded.source_mask[0, 0])
    assert bool(encoded.source_mask[0, 1])
    assert bool(encoded.owned_mask[0, 0])
    assert bool(encoded.owned_mask[0, 1])


def test_friendly_defense_score_prefers_pressure_gradient_from_backline():
    score = friendly_defense_score_jax(
        target_will_flip=jnp.array([False, True, False]),
        target_flip_turn=jnp.array([19, 4, 19], dtype=jnp.int32),
        target_flip_urgency=jnp.array([0.0, 40.0, 0.0], dtype=jnp.float32),
        enemy_pressure=jnp.array([0.0, 10.0, 90.0], dtype=jnp.float32),
        src_idx=jnp.asarray(0, dtype=jnp.int32),
        eta_turns=jnp.array([0.0, 3.8, 3.0], dtype=jnp.float32),
    )

    assert score[2] > score[1]


def test_friendly_defense_score_pressure_eta_decay_prefers_near_target():
    score = friendly_defense_score_jax(
        target_will_flip=jnp.array([False, False, False, False]),
        target_flip_turn=jnp.array([19, 19, 19, 19], dtype=jnp.int32),
        target_flip_urgency=jnp.zeros((4,), dtype=jnp.float32),
        enemy_pressure=jnp.array([0.0, 60.0, 60.0, 60.0], dtype=jnp.float32),
        src_idx=jnp.asarray(0, dtype=jnp.int32),
        eta_turns=jnp.array([0.0, 1.0, 4.0, 8.0], dtype=jnp.float32),
    )

    assert score[1] > score[2]
    assert score[2] > score[3]
    assert jnp.allclose(score[3], 0.0)


def test_friendly_defense_score_frontline_source_focuses_flip_urgency():
    score = friendly_defense_score_jax(
        target_will_flip=jnp.array([False, True, False]),
        target_flip_turn=jnp.array([19, 4, 19], dtype=jnp.int32),
        target_flip_urgency=jnp.array([0.0, 40.0, 0.0], dtype=jnp.float32),
        enemy_pressure=jnp.array([75.0, 80.0, 78.0], dtype=jnp.float32),
        src_idx=jnp.asarray(0, dtype=jnp.int32),
        eta_turns=jnp.array([0.0, 2.0, 3.0], dtype=jnp.float32),
    )

    assert score[1] > score[2]


def test_target_filter_defense_topk_uses_pressure_gradient_as_regroup_candidate():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 1000.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 0, 30.0, 10.0, 1.0, 20.0, 0.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, 0, 20.0, 20.0, 1.0, 20.0, 0.0], dtype=jnp.float32))
    planets = planets.at[3].set(jnp.array([3, 1, 35.0, 10.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, True, True]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=4, target_filter_enabled=True, attack_top_k=0, defense_top_k=1)

    encoded = encode_batch(batched, fp)
    src0_mask = encoded.candidate_mask[0, 0]

    assert bool(src0_mask[2])
    assert not bool(src0_mask[3])


def test_target_filter_pressure_regroup_is_source_conditioned():
    params = EnvParams(max_planets=5, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 30.0, 65.0, 1.0, 1000.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 0, 65.0, 65.0, 1.0, 60.0, 0.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, 0, 30.0, 70.0, 1.0, 60.0, 0.0], dtype=jnp.float32))
    planets = planets.at[3].set(jnp.array([3, 1, 75.0, 65.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, True, True, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=5, target_filter_enabled=True, attack_top_k=0, defense_top_k=1)

    encoded = encode_batch(batched, fp)

    assert bool(encoded.candidate_mask[0, 0, 2])
    assert not bool(encoded.candidate_mask[0, 1, 1:].any())


def test_reachability_filters_unreachable_attack_target():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 5.0, 10.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 95.0, 10.0, 1.0, 1.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=4, target_filter_enabled=False)

    encoded = encode_batch(batched, fp)

    assert not bool(encoded.candidate_mask[0, 0, 2])
    assert not bool(encoded.source_mask[0, 0])
    assert bool(encoded.owned_mask[0, 0])


def test_reachability_keeps_target_when_full_send_can_reach():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 70.0, 10.0, 1.0, 1.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=4, target_filter_enabled=False)

    encoded = encode_batch(batched, fp)

    assert bool(encoded.candidate_mask[0, 0, 2])


def test_attack_topk_ranks_before_cheap_validity_intersection():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 1000.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 12.0, 10.0, 1.0, 1.0, 5.0], dtype=jnp.float32))
    planets = planets.at[2].set(jnp.array([2, 1, 25.0, 10.0, 1.0, 1.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, False, True, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)

    top1 = encode_batch(
        batched,
        FeatureParams(max_planets=4, target_filter_enabled=True, attack_top_k=1, defense_top_k=0),
    )
    top2 = encode_batch(
        batched,
        FeatureParams(max_planets=4, target_filter_enabled=True, attack_top_k=2, defense_top_k=0),
    )

    assert not bool(top1.candidate_mask[0, 0, 3])
    assert bool(top2.candidate_mask[0, 0, 3])


def test_relative_eta_uses_exact_physics_hit_turn():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 70.0, 10.0, 1.0, 1.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=4, target_filter_enabled=False)

    _angle, hit, eta_turns = aim_with_prediction_eta_jax(
        state.planets[0].reshape(1, 1, 7),
        state.planets[1].reshape(1, 1, 7),
        jnp.asarray([[1]], dtype=jnp.int32),
        batched,
        jnp.asarray([[100.0]], dtype=jnp.float32),
    )
    encoded = encode_batch(batched, fp)

    assert bool(hit[0, 0])
    assert jnp.allclose(encoded.relative_features[0, 0, 2, 7], eta_turns[0, 0] / INBOUND_HORIZON)


def test_dynamic_target_mask_filters_sun_blocked_target():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 20.0, 50.0, 1.0, 1000.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 80.0, 50.0, 1.0, 1.0, 0.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=4, target_filter_enabled=False)

    encoded = encode_batch(batched, fp)

    from jax_agent.train_jax import selected_exact_target_mask_and_angles

    target_mask, _angles = selected_exact_target_mask_and_angles(
        batched,
        encoded,
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([1000.0], dtype=jnp.float32),
    )

    assert bool(encoded.candidate_mask[0, 0, 2])
    assert not bool(target_mask[0, 2])


def test_planet_inbound_features_use_garrison_arrivals():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 50.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 0, 13.0, 10.0, 1.0, 20.0, 0.0], dtype=jnp.float32))
    fleets = jnp.zeros((params.max_fleets, 7), dtype=jnp.float32)
    fleets = fleets.at[0].set(jnp.array([0, 1, 10.5, 10.0, 0.0, 2, 30.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
        fleets=fleets,
        fleet_alive=jnp.array([True, False, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=4, max_ships=100.0, target_filter_enabled=False)

    encoded = encode_batch(batched, fp)
    target_features = encoded.planet_features[0, 0]

    assert jnp.allclose(target_features[20], 0.0)
    assert jnp.allclose(target_features[21], 0.3)
    assert jnp.allclose(target_features[24], 1.0 / 18.0)
    assert jnp.allclose(target_features[25], 1.0 / 18.0)


def test_relative_eta_garrison_features_sample_future_state_and_arrivals():
    owner = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    ships = jnp.zeros((3, 5), dtype=jnp.float32)
    pre_owner = owner
    pre_ships = jnp.zeros((3, 5), dtype=jnp.float32).at[1, 3].set(42.0)
    arrivals = jnp.zeros((3, 5, 2), dtype=jnp.float32)
    arrivals = arrivals.at[1, 1, 0].set(5.0)
    arrivals = arrivals.at[1, 3, 0].set(4.0)
    arrivals = arrivals.at[1, 2, 1].set(7.0)
    alive = jnp.ones((3, 5), dtype=bool)
    status = GarrisonStatus(
        owner=owner,
        ships=ships,
        pre_combat_owner=pre_owner,
        pre_combat_ships=pre_ships,
        arrivals_by_owner=arrivals,
        alive=alive,
    )

    features = relative_eta_garrison_features(
        status,
        eta_turns=jnp.array([0.0, 2.2, 9.0], dtype=jnp.float32),
        params=FeatureParams(max_planets=3, max_ships=100.0),
    )

    assert features.shape == (3, 7)
    assert jnp.allclose(features[1, :3], jnp.array([0.0, 0.0, 1.0]))
    assert jnp.allclose(features[1, 3], 0.42)
    assert jnp.allclose(features[1, 4], 0.09)
    assert jnp.allclose(features[1, 5], 0.07)
    assert jnp.allclose(features[1, 6], 0.0)
    assert jnp.allclose(features[2, 6], 1.0)


def test_relative_capture_floor_features_are_owner_aware():
    owner = jnp.array(
        [
            [0, 0],
            [1, 1],
            [-1, -1],
        ],
        dtype=jnp.int32,
    )
    ships = jnp.array(
        [
            [50.0, 80.0],
            [20.0, 20.0],
            [4.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    arrivals = jnp.zeros((3, 2, 2), dtype=jnp.float32)
    alive = jnp.ones((3, 2), dtype=bool)
    status = GarrisonStatus(
        owner=owner,
        ships=ships,
        pre_combat_owner=owner,
        pre_combat_ships=ships,
        arrivals_by_owner=arrivals,
        alive=alive,
    )

    floor_norm, surplus_norm = relative_capture_floor_features(
        status,
        eta_turns=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
        send_ships=jnp.asarray(30.0, dtype=jnp.float32),
        params=FeatureParams(max_planets=3, max_ships=100.0),
    )

    assert jnp.allclose(floor_norm[0], 0.01)
    assert jnp.allclose(floor_norm[1], 0.21)
    assert jnp.allclose(floor_norm[2], 0.05)
    assert jnp.allclose(surplus_norm[0], 0.29)
    assert jnp.allclose(surplus_norm[1], 0.09)
    assert jnp.allclose(surplus_norm[2], 0.25)


def test_encode_relative_features_include_eta_garrison_columns():
    params = EnvParams(max_planets=4, max_fleets=4, max_actions=1)
    state = empty_state(params)
    planets = jnp.zeros((params.max_planets, 7), dtype=jnp.float32)
    planets = planets.at[0].set(jnp.array([0, 0, 10.0, 10.0, 1.0, 100.0, 0.0], dtype=jnp.float32))
    planets = planets.at[1].set(jnp.array([1, 1, 20.0, 10.0, 1.0, 10.0, 2.0], dtype=jnp.float32))
    state = state._replace(
        planets=planets,
        initial_planets=planets,
        planet_alive=jnp.array([True, True, False, False]),
    )
    batched = jax.tree_util.tree_map(lambda x: x[None, ...] if getattr(x, "ndim", 0) > 0 else x[None], state)
    fp = FeatureParams(max_planets=4, max_ships=100.0, target_filter_enabled=False)

    encoded = encode_batch(batched, fp)

    assert encoded.planet_features.shape[-1] == PLANET_DIM
    assert encoded.relative_features.shape[-1] == RELATIVE_DIM
    assert jnp.allclose(encoded.planet_features[0, 0, 26], 1.0)
    assert jnp.allclose(encoded.planet_features[0, 0, 27], 0.0)
    assert jnp.allclose(encoded.relative_features[0, 0, 2, 10], 1.0)
    assert encoded.relative_features[0, 0, 2, 11] > 0.10
    assert encoded.relative_features[0, 0, 2, 17] > 0.5
    assert jnp.allclose(encoded.relative_features[0, 0, 2, 18], 0.0)
    assert jnp.allclose(encoded.relative_features[0, 0, 2, 19], 0.0)
    assert encoded.relative_features[0, 0, 2, 20] > encoded.relative_features[0, 0, 2, 11]
    assert encoded.relative_features[0, 0, 2, 21] > 0.0
