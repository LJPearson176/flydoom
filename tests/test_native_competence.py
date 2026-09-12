from fly_doom.doom.native_benchmark import NativeEpisodeResult
from fly_doom.doom.native_competence import doom004c_conditions, row_from_episode


def test_doom004c_is_a_separate_model_by_suppression_matrix():
    conditions = doom004c_conditions()
    assert len(conditions) == 12
    assert {condition.saccade_refractory_ticks for condition in conditions} == {0, 3, 5}
    assert {condition.model_type.name for condition in conditions} == {"MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D"}


def test_competence_row_preserves_unavailable_native_metrics():
    result = NativeEpisodeResult(
        episode_id="e", seed=1, condition="ModelD_Saccade_3", map_name="E1M1", iwad=None,
        start_timestamp="", end_timestamp="", termination_reason="timeout", steps=2,
        action_counts={"FORWARD": 1, "TURN_LEFT": 1, "TURN_RIGHT": 0, "FIRE": 0},
        health_remaining=100, ammo_remaining=50, kills=0, damage_dealt=None,
        navigation_status="unavailable_native_bridge",
        stability={"turning_fraction": 0.5}, environment_metadata={}, trajectory=[],
    )
    row = row_from_episode(result)
    assert row.suppression_ticks == 3
    assert row.navigation_progress is None
    assert row.native_world_metrics_status == "unavailable_native_bridge"


def test_competence_hierarchy_metrics():
    from fly_doom.doom.native_benchmark import NativeTrajectoryTick

    ticks = [
        NativeTrajectoryTick(
            episode_id="e", seed=1, condition="ModelD_Saccade_3", step=0, timestamp="", action="FORWARD",
            health=100, ammo=50, kills=0, frame_mean=0.5, frame_nonzero_fraction=1.0,
            retina_mean=0.2, retina_delta=0.05, t4_l_v=-65, t4_r_v=-65, norm_asymmetry=0.0,
            linear_velocity=None, angular_velocity=None, capture_valid=True, window_id=1, window_bounds=None,
        ),
        NativeTrajectoryTick(
            episode_id="e", seed=1, condition="ModelD_Saccade_3", step=1, timestamp="", action="TURN_LEFT",
            health=100, ammo=50, kills=0, frame_mean=0.6, frame_nonzero_fraction=1.0,
            retina_mean=0.25, retina_delta=0.02, t4_l_v=-70, t4_r_v=-60, norm_asymmetry=0.8,
            linear_velocity=None, angular_velocity=None, capture_valid=True, window_id=1, window_bounds=None,
        ),
        NativeTrajectoryTick(
            episode_id="e", seed=1, condition="ModelD_Saccade_3", step=2, timestamp="", action="TURN_LEFT",
            health=100, ammo=50, kills=0, frame_mean=0.6, frame_nonzero_fraction=1.0,
            retina_mean=0.26, retina_delta=0.02, t4_l_v=-70, t4_r_v=-60, norm_asymmetry=0.9,
            linear_velocity=None, angular_velocity=None, capture_valid=True, window_id=1, window_bounds=None,
        ),
        NativeTrajectoryTick(
            episode_id="e", seed=1, condition="ModelD_Saccade_3", step=3, timestamp="", action="FIRE",
            health=100, ammo=49, kills=0, frame_mean=0.55, frame_nonzero_fraction=1.0,
            retina_mean=0.22, retina_delta=0.01, t4_l_v=-65, t4_r_v=-65, norm_asymmetry=0.0,
            linear_velocity=None, angular_velocity=None, capture_valid=True, window_id=1, window_bounds=None,
        ),
    ]

    result = NativeEpisodeResult(
        episode_id="e", seed=1, condition="ModelD_Saccade_3", map_name="E1M1", iwad=None,
        start_timestamp="", end_timestamp="", termination_reason="timeout", steps=4,
        action_counts={"FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 0, "FIRE": 1},
        health_remaining=100, ammo_remaining=49, kills=0, damage_dealt=None,
        navigation_status="unavailable_native_bridge",
        stability={"turning_fraction": 0.5}, environment_metadata={}, trajectory=ticks,
    )

    row = row_from_episode(result)
    # Level 0
    assert row.capture_valid_fraction == 1.0
    # Level 1
    assert row.forward_fraction == 0.25
    assert row.fire_fraction == 0.25
    assert row.spin_lock_streak_max == 2
    # Level 2
    assert row.forward_motion_proxy == 0.05
    assert abs(row.rotational_motion_proxy - 0.85) < 1e-4
    assert row.visual_scene_variation > 0.0
    # Level 3
    assert row.fire_events == 1
    # Level 4
    assert row.navigation_progress is None


def test_competence_row_elevates_metrics_when_native_telemetry_present():
    result = NativeEpisodeResult(
        episode_id="e", seed=1, condition="ModelD_Saccade_3", map_name="E1M1", iwad=None,
        start_timestamp="", end_timestamp="", termination_reason="timeout", steps=100,
        action_counts={"FORWARD": 80, "TURN_LEFT": 10, "TURN_RIGHT": 10, "FIRE": 0},
        health_remaining=85.0, ammo_remaining=42, kills=3, damage_dealt=None,
        navigation_status="native_game_state_available",
        stability={"turning_fraction": 0.2}, environment_metadata={}, trajectory=[],
        distance_traveled=1234.5, survival=100,
    )
    row = row_from_episode(result)
    assert row.native_world_metrics_status == "native_game_state_available"
    assert row.distance_traveled == 1234.5
    assert row.kills == 3
    assert row.survival == 100


