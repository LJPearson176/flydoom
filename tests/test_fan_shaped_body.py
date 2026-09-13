import math
import numpy as np
import pytest

from fly_doom.dynamics.fan_shaped_body import (
    AMMCWallSlipReflex,
    FanShapedBodyVectorEngine,
    SEZNociceptiveReflex,
    Stage1WaypointGraph,
    WaypointNode,
)


def test_fan_shaped_body_aligned_forward_drive():
    engine = FanShapedBodyVectorEngine(deadband_deg=10.0)
    # Heading North (pi/2), Goal North at (0, 100) from (0, 0)
    steer, thrust, err = engine.compute_vector_steering(
        current_heading_rad=math.pi / 2.0,
        goal_pos=(0.0, 100.0),
        current_pos=(0.0, 0.0),
    )
    assert abs(steer) == 0.0  # Within deadband
    assert thrust > 0.95      # High forward thrust
    assert abs(err) < 1.0


def test_fan_shaped_body_right_turn_torque():
    engine = FanShapedBodyVectorEngine(deadband_deg=10.0)
    # Heading North (pi/2), Goal East at (100, 0) from (0, 0) -> Goal angle = 0.0
    # Discrepancy = 0 - pi/2 = -pi/2 (-90 deg) -> TURN_RIGHT (negative torque)
    steer, thrust, err = engine.compute_vector_steering(
        current_heading_rad=math.pi / 2.0,
        goal_pos=(100.0, 0.0),
        current_pos=(0.0, 0.0),
    )
    assert steer < -0.5
    assert err == pytest.approx(-90.0, abs=1.0)


def test_fan_shaped_body_left_turn_torque():
    engine = FanShapedBodyVectorEngine(deadband_deg=10.0)
    # Heading North (pi/2), Goal West at (-100, 0) from (0, 0) -> Goal angle = pi
    # Discrepancy = pi - pi/2 = +pi/2 (+90 deg) -> TURN_LEFT (positive torque)
    steer, thrust, err = engine.compute_vector_steering(
        current_heading_rad=math.pi / 2.0,
        goal_pos=(-100.0, 0.0),
        current_pos=(0.0, 0.0),
    )
    assert steer > 0.5
    assert err == pytest.approx(90.0, abs=1.0)


def test_sez_nociceptive_hazard_detection():
    reflex = SEZNociceptiveReflex(acid_threshold=1.35)

    # Safe floor: uniform gray image
    safe_frame = np.full((64, 64, 3), 80, dtype=np.uint8)
    detected, bias, frac = reflex.evaluate_retinal_hazard(safe_frame)
    assert not detected
    assert bias == 0.0
    assert frac < 0.05

    # Dangerous floor: lower half contains bright green nukage acid
    acid_frame = np.full((64, 64, 3), 50, dtype=np.uint8)
    # Set ventral rows to toxic green (R=20, G=220, B=20)
    acid_frame[48:, :, 0] = 20
    acid_frame[48:, :, 1] = 220
    acid_frame[48:, :, 2] = 20

    detected, bias, frac = reflex.evaluate_retinal_hazard(acid_frame)
    assert detected
    assert frac > 0.5
    # Since uniform across ventral strip, bias is non-zero (or neutral)


def test_ammc_tactile_wall_slip():
    reflex = AMMCWallSlipReflex(stall_threshold_units=1.0, stall_ticks=4)

    # Moving forward freely: displacement is 5.0 units per tick
    for i in range(10):
        active, torque = reflex.update(current_pos=(float(i * 5.0), 0.0), is_forward_commanded=True)
        assert not active

    # Stalling at a doorframe (stuck at same position)
    stuck_pos = (45.0, 0.0)  # already at 45.0 from previous loop
    for _ in range(3):
        active, _ = reflex.update(current_pos=stuck_pos, is_forward_commanded=True)
        assert not active

    # On 4th tick of zero displacement, stall threshold reached -> triggers deflection saccade
    active, torque = reflex.update(current_pos=stuck_pos, is_forward_commanded=True)
    assert active
    assert abs(torque) == 1.0


def test_stage1_waypoint_graph_traversal():
    graph = Stage1WaypointGraph()
    assert len(graph.waypoints) == 7

    # Start at Zone 1 (Spawn)
    wp = graph.get_current_waypoint()
    assert wp.zone_id == 1
    assert wp.name == "Spawn_Corridor"

    # Move to Zone 1 target (1056, -3200) -> advances to Zone 2
    wp = graph.update_progress((1056.0, -3200.0))
    assert wp.zone_id == 2
    assert wp.name == "Hallway_Elbow"

    # Move to Zone 2 target (1350, -2600) -> advances to Zone 3 (Door 151)
    wp = graph.update_progress((1350.0, -2600.0))
    assert wp.zone_id == 3
    assert wp.required_action == "USE_DOOR"
