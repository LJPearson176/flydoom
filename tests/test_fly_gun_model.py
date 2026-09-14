"""Unit tests for 3D Anatomical Fly with Shotgun and Kinematics Engine."""

import numpy as np
import pytest
from PIL import Image

from fly_doom.vis.fly_gun_model import AnatomicalFlyShotgunModel, FlyKinematicPose
from fly_doom.vis.fly_gun_renderer import FlyShotgunRenderer


def test_tripod_gait_phase_alternation():
    """Verify alternating canonical tripod gait stance vs swing phases."""
    model = AnatomicalFlyShotgunModel()

    # Step 0
    pose0 = model.compute_pose(step=0, action="FORWARD")
    assert "L2" in pose0.legs and "R2" in pose0.legs
    assert "L3" in pose0.legs and "R3" in pose0.legs

    # Canonical insect gait: Tripod A (R2, L3) vs Tripod B (L2, R3)
    # L2 and R2 must have opposing phases
    assert pose0.legs["L2"].is_stance != pose0.legs["R2"].is_stance
    assert pose0.legs["L3"].is_stance != pose0.legs["R3"].is_stance
    # L2 and R3 belong to the same Tripod B
    assert pose0.legs["L2"].is_stance == pose0.legs["R3"].is_stance


def test_shotgun_recoil_impulse_and_decay():
    """Verify shotgun kickback on FIRE and subsequent relaxation."""
    model = AnatomicalFlyShotgunModel()

    # At rest
    pose_rest = model.compute_pose(step=0, action="FORWARD", is_fire=False)
    assert pose_rest.gun_recoil == 0.0
    assert pose_rest.muzzle_flash == 0.0

    # Firing step
    pose_fire = model.compute_pose(step=1, action="FIRE", is_fire=True)
    assert pose_fire.gun_recoil >= 15.0
    assert pose_fire.muzzle_flash >= 0.90
    assert pose_fire.legs["L1"].femur > pose_rest.legs["L1"].femur  # joint flexes to absorb kick

    # Next step (recovery)
    pose_decay = model.compute_pose(step=2, action="FORWARD", is_fire=False)
    assert pose_decay.gun_recoil < pose_fire.gun_recoil
    assert pose_decay.muzzle_flash < pose_fire.muzzle_flash


def test_door_reach_gesture_on_use():
    """Verify right foreleg extension when approaching or actuating door."""
    model = AnatomicalFlyShotgunModel()

    # Approaching door with high reservoir probability
    pose_use = model.compute_pose(step=5, action="USE", door_p=0.95)
    assert pose_use.door_reach > 0.20
    # Right foreleg should extend forward compared to default grip
    pose_fwd = model.compute_pose(step=0, action="FORWARD", door_p=0.0)
    assert pose_use.legs["R1"].foot_pos[1] < pose_fwd.legs["R1"].foot_pos[1]  # forward is negative Y


def test_asymmetric_steering_response():
    """Verify body yaw and asymmetric stepping driven by lobula asymmetry."""
    model = AnatomicalFlyShotgunModel()

    pose_right = model.compute_pose(step=10, action="FORWARD", norm_asymmetry=0.50)
    assert pose_right.yaw > 0.0
    assert pose_right.head_yaw > pose_right.yaw  # head anticipates turn

    pose_left = model.compute_pose(step=10, action="FORWARD", norm_asymmetry=-0.50)
    assert pose_left.yaw < 0.0
    assert pose_left.head_yaw < pose_left.yaw


def test_fly_shotgun_renderer_pipeline():
    """Verify full rendering pipeline produces valid 680x540 RGB image."""
    renderer = FlyShotgunRenderer(width=680, height=540)

    frame = renderer.render_frame(step=20, action="FIRE", is_fire=True, health=95.0, kills=2)
    assert isinstance(frame, Image.Image)
    assert frame.size == (680, 540)
    assert frame.mode == "RGB"

    # Verify frame contains rendered elements (not blank)
    arr = np.array(frame)
    assert np.mean(arr) > 10.0


def test_fly_facing_up_projection_orientation():
    """Verify 3D model is projected facing UP to match first-person gameplay orientation.

    In screen space (Y=0 is top, Y=height is bottom):
    Muzzle Y < Head Y < Thorax Y < Abdomen Y
    """
    renderer = FlyShotgunRenderer(width=680, height=540)
    p_muz = renderer.project((0.0, -185.0, 4.0))
    p_head = renderer.project((0.0, -85.0, 8.0))
    p_th = renderer.project((0.0, 0.0, 0.0))
    p_abd = renderer.project((0.0, 228.0, -58.0))

    assert p_muz is not None and p_head is not None and p_th is not None and p_abd is not None
    assert p_muz[1] < p_head[1] < p_th[1] < p_abd[1], (
        f"Fly not facing UP! Projected Y coordinates: muzzle={p_muz[1]:.1f}, "
        f"head={p_head[1]:.1f}, thorax={p_th[1]:.1f}, abdomen={p_abd[1]:.1f}"
    )
