import numpy as np
import pytest

from fly_doom.control.controllers import ControlledT4Controller, DoorSeekingController
from fly_doom.doom.interface import DoomAction, DoomObservation
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def _make_obs(
    x: float,
    y: float,
    angle_deg: float,
    kill_count: int = 0,
    ammo: int = 50,
    health: float = 100.0,
) -> DoomObservation:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    return DoomObservation(
        rgb=rgb,
        depth=np.zeros((64, 64), dtype=np.float32),
        health=health,
        ammo=ammo,
        kill_count=kill_count,
        x=x,
        y=y,
        angle_rad=np.deg2rad(angle_deg),
        info={
            "native_game_state_available": True,
            "native_game_state": {
                "x": x,
                "y": y,
                "z": 0.0,
                "angle_deg": angle_deg,
                "pitch_deg": 0.0,
                "speed": 0.0,
                "health": health,
                "ammo": ammo,
                "kills": kill_count,
            },
        },
    )


def test_zone1_spawn_corridor_perspective():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True)
    # At spawn, heading 0.0 deg (facing East), but zone 1 target is 90.0 deg (North)
    obs = _make_obs(x=1056.0, y=-3500.0, angle_deg=0.0)
    action = controller.select_action(obs)
    # diff = 90 - 0 = +90 > 15 -> TURN_LEFT
    assert action == DoomAction.TURN_LEFT
    state = controller.get_neural_state()
    assert state["target_angle_deg"] == pytest.approx(90.0)


def test_zone2_corridor_turn_perspective():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True)
    # Past turn, approaching door alcove from (1200, -2800) toward (1536, -2496)
    # dx = 336, dy = 304 -> atan2(304, 336) ~ 42.14 deg
    obs = _make_obs(x=1200.0, y=-2800.0, angle_deg=90.0)
    action = controller.select_action(obs)
    # diff = 42.14 - 90.0 = -47.86 < -15 -> TURN_RIGHT
    assert action == DoomAction.TURN_RIGHT
    state = controller.get_neural_state()
    assert state["target_angle_deg"] == pytest.approx(42.14, abs=1.0)


def test_zone3_door_approach_square_alignment():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True)
    # In door approach zone (X = 1500), target is strictly 0.0 deg (East)
    # If agent angle is 20.0 deg, diff = 0 - 20 = -20 < -10 -> TURN_RIGHT
    obs = _make_obs(x=1500.0, y=-2496.0, angle_deg=20.0)
    action = controller.select_action(obs)
    assert action == DoomAction.TURN_RIGHT
    state = controller.get_neural_state()
    assert state["target_angle_deg"] == pytest.approx(0.0)


def test_zone3_door_stall_emits_use():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, stall_ticks=3, manage_perspective=True)
    # Agent at door (X = 1520), aligned facing East (0.0 deg)
    obs = _make_obs(x=1520.0, y=-2496.0, angle_deg=0.0)
    # Stalled ticks 0, 1, 2, 3
    controller.select_action(obs)
    controller.select_action(obs)
    controller.select_action(obs)
    action4 = controller.select_action(obs)
    assert action4 == DoomAction.USE
    state = controller.get_neural_state()
    assert state["door_use_cooldown"] == pytest.approx(18.0)


def test_perspective_maintained_during_use_cooldown():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, stall_ticks=1, manage_perspective=True)
    obs = _make_obs(x=1520.0, y=-2496.0, angle_deg=0.0)
    controller.select_action(obs)
    action = controller.select_action(obs)
    assert action == DoomAction.USE
    # In cooldown, but angle is perturbed to 25.0 deg
    obs_perturbed = _make_obs(x=1520.0, y=-2496.0, angle_deg=25.0)
    next_action = controller.select_action(obs_perturbed)
    # Should correct perspective (TURN_RIGHT) despite use cooldown!
    assert next_action == DoomAction.TURN_RIGHT


def test_zone4_doorway_corridor_traversal():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True)
    # Moving through doorway (X = 1600), target is strictly 0.0 deg
    obs = _make_obs(x=1600.0, y=-2496.0, angle_deg=0.0)
    controller.select_action(obs)
    state = controller.get_neural_state()
    assert state["target_angle_deg"] == pytest.approx(0.0)


def test_zone5_enemy1_combat_engagement():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True)
    # In zigzag room at (1680, -2496), Enemy 1 is at (1696, -2688)
    # dx = 16, dy = -192 -> atan2(-192, 16) ~ -85.2 deg
    obs = _make_obs(x=1680.0, y=-2496.0, angle_deg=0.0, kill_count=0)
    action = controller.select_action(obs)
    # diff = -85.2 - 0 = -85.2 < -12 -> TURN_RIGHT
    assert action == DoomAction.TURN_RIGHT
    state = controller.get_neural_state()
    assert state["enemy_target_id"] == 1.0
    assert state["combat_active"] == 1.0

    # Once facing Enemy 1 (~ -85.0 deg), firing solution is locked!
    obs_aligned = _make_obs(x=1680.0, y=-2496.0, angle_deg=-85.0, kill_count=0, ammo=40)
    fire_action = controller.select_action(obs_aligned)
    assert fire_action == DoomAction.FIRE
    state_aligned = controller.get_neural_state()
    assert state_aligned["firing_solution_locked"] == 1.0


def test_zone5_enemy2_acquisition_after_kill():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True)
    # In zigzag room at (1680, -2496), but Enemy 1 is dead (kill_count = 1)
    # Target Enemy 2 at (1920, -2176): dx = 240, dy = 320 -> atan2(320, 240) ~ 53.13 deg
    obs = _make_obs(x=1680.0, y=-2496.0, angle_deg=0.0, kill_count=1)
    action = controller.select_action(obs)
    # diff = 53.13 - 0 = +53.13 > 12 -> TURN_LEFT
    assert action == DoomAction.TURN_LEFT
    state = controller.get_neural_state()
    assert state["enemy_target_id"] == 2.0
    assert state["target_angle_deg"] == pytest.approx(53.13, abs=1.0)

    # Once facing Enemy 2 (~ 53.0 deg), emit FIRE!
    obs_aligned = _make_obs(x=1680.0, y=-2496.0, angle_deg=53.0, kill_count=1, ammo=35)
    fire_action = controller.select_action(obs_aligned)
    assert fire_action == DoomAction.FIRE
    state_aligned = controller.get_neural_state()
    assert state_aligned["firing_solution_locked"] == 1.0


def test_health_priority_suppresses_fire_at_low_health():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True, panic_health=35.0)
    controller.select_action(_make_obs(x=1680.0, y=-2496.0, angle_deg=-85.0, health=100.0))
    action = controller.select_action(_make_obs(x=1680.0, y=-2496.0, angle_deg=-85.0, health=20.0))
    assert action != DoomAction.FIRE
    state = controller.get_neural_state()
    assert state["health_priority_active"] == 1.0
    assert state["health_delta"] == pytest.approx(-80.0)


def test_health_drop_triggers_evasive_heading():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True)
    controller.select_action(_make_obs(x=1680.0, y=-2496.0, angle_deg=-85.0, health=100.0))
    action = controller.select_action(_make_obs(x=1680.0, y=-2496.0, angle_deg=-85.0, health=88.0))
    assert action in {DoomAction.TURN_LEFT, DoomAction.TURN_RIGHT, DoomAction.FORWARD}
    assert action != DoomAction.FIRE
    assert controller.get_neural_state()["health_priority_active"] == 1.0


def test_zone5_live_target_acquisition_and_fire():
    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(base, manage_perspective=True)
    obs = _make_obs(x=1680.0, y=-2496.0, angle_deg=0.0, kill_count=0, ammo=45)
    obs.info["native_game_state"]["target_x"] = 2272.0
    obs.info["native_game_state"]["target_y"] = -2432.0
    obs.info["native_game_state"]["target_health"] = 20
    obs.info["native_game_state"]["target_visible"] = True

    # Heading 0.0 deg, target is at dx=592, dy=64 -> angle ~ +6.16 deg
    # diff = 6.16 - 0.0 = +6.16 deg <= 18.0 deg -> within firing solution!
    action = controller.select_action(obs)
    assert action == DoomAction.FIRE
    state = controller.get_neural_state()
    assert state["enemy_target_id"] == 100.0
    assert state["target_angle_deg"] == pytest.approx(6.16, abs=0.5)
    assert state["firing_solution_locked"] == 1.0
