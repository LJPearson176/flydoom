"""3D Anatomical Drosophila Biomechanical Model with DOOM Shotgun.

Adapted from EPFL Ramdya Lab's NeuroMechFly v2 / FlyGym micro-CT anatomical datasets
(Lobato-Rios et al., Nature Methods 2022; Wang-Chen et al., Nature Methods 2024),
as featured in viral embodied Drosophila experiments (e.g. Nick Walton's Rubik's cube fly).

Provides forward kinematics for:
- Canonical insect tripod walking gait (alternating triplets L1-R2-L3 vs R1-L2-R3)
- Prothoracic (T1) foreleg shotgun grasp and recoil absorption
- Firing recoil kick, muzzle flash emission, and wing flare reflex
- Door actuation reach gesture on USE
- Threat arousal and damage flexion
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class LegJointState:
    """Joint angles (in radians) and end-effector position for an articulated insect leg."""
    coxa: float = 0.0
    femur: float = 0.0
    tibia: float = 0.0
    tarsus: float = 0.0
    foot_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    is_stance: bool = True


@dataclass
class FlyKinematicPose:
    """Full anatomical pose of the Drosophila body, limbs, and shotgun."""
    # Body root pose in world/local space
    body_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # Segment specific transforms
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    abdomen_curl: float = 0.0  # ventral curling under threat/pain
    left_wing_angle: float = 0.0   # wing flare angle
    right_wing_angle: float = 0.0

    # 6 Articulated legs (L1, R1, L2, R2, L3, R3)
    legs: Dict[str, LegJointState] = field(default_factory=dict)

    # DOOM Shotgun state
    gun_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    gun_recoil: float = 0.0      # backward kick displacement (mm)
    muzzle_flash: float = 0.0    # [0.0, 1.0] flash intensity
    door_reach: float = 0.0      # [0.0, 1.0] foreleg extension to door


class AnatomicalFlyShotgunModel:
    """Biomechanical 3D Drosophila Melanogaster model equipped with DOOM Shotgun."""

    def __init__(self):
        # Base anatomical attachment points on thorax (in model coordinates, origin at thorax center)
        # Coordinates: X = lateral (right positive), Y = anterior-posterior (anterior negative), Z = dorsal (up positive)
        self.coxa_bases = {
            "L1": np.array([-45.0, -45.0, -30.0], dtype=np.float32),
            "R1": np.array([45.0, -45.0, -30.0], dtype=np.float32),
            "L2": np.array([-55.0, 0.0, -35.0], dtype=np.float32),
            "R2": np.array([55.0, 0.0, -35.0], dtype=np.float32),
            "L3": np.array([-50.0, 50.0, -30.0], dtype=np.float32),
            "R3": np.array([50.0, 50.0, -30.0], dtype=np.float32),
        }

        # Segment lengths for each leg pair [coxa, femur, tibia, tarsus]
        self.leg_segment_lengths = {
            "L1": [20.0, 55.0, 55.0, 45.0],
            "R1": [20.0, 55.0, 55.0, 45.0],
            "L2": [25.0, 70.0, 70.0, 55.0],
            "R2": [25.0, 70.0, 70.0, 55.0],
            "L3": [30.0, 85.0, 90.0, 65.0],
            "R3": [30.0, 85.0, 90.0, 65.0],
        }

        # Default shotgun rest pose relative to thorax
        self.gun_base_pos = np.array([0.0, -110.0, -5.0], dtype=np.float32)

        # Recoil memory
        self.current_recoil = 0.0
        self.current_flash = 0.0
        self.current_door_reach = 0.0

    def compute_pose(
        self,
        step: int,
        action: str = "FORWARD",
        norm_asymmetry: float = 0.0,
        door_p: float = 0.0,
        health: float = 100.0,
        kills: int = 0,
        is_fire: bool = False,
    ) -> FlyKinematicPose:
        """Evaluate forward kinematics for the complete fly anatomy and shotgun.

        Args:
            step: Simulation tick counter.
            action: Current discrete action (FORWARD, TURN_LEFT, TURN_RIGHT, FIRE, USE).
            norm_asymmetry: Optic flow normalized directional asymmetry in [-1.0, 1.0].
            door_p: Reservoir door candidate probability in [0.0, 1.0].
            health: Player health percentage.
            kills: Cumulative neutralized enemies count.
            is_fire: Explicit weapon discharge flag.

        Returns:
            FlyKinematicPose containing all joint states and segment transforms.
        """
        pose = FlyKinematicPose()

        # 1. Base locomotion and heading
        is_forward = (action == "FORWARD")
        is_turn_left = (action == "TURN_LEFT" or norm_asymmetry < -0.15)
        is_turn_right = (action == "TURN_RIGHT" or norm_asymmetry > 0.15)
        is_use = (action == "USE" or door_p > 0.70)
        is_damage = (health < 40.0)

        # Body orientation response
        pose.yaw = float(norm_asymmetry * 0.35)
        pose.pitch = -0.06 if is_forward else 0.02  # forward tilt during advance
        pose.roll = float(norm_asymmetry * 0.10)

        # Head anticipation: head turns before body
        pose.head_yaw = float(norm_asymmetry * 0.50)
        pose.head_pitch = -0.04 if is_forward else 0.0

        # Abdomen curling on damage or threat
        pose.abdomen_curl = 0.35 if is_damage else (0.15 if is_fire else 0.0)

        # 2. Tripod Gait Kinematics (Canonical Drosophila walking)
        # Gait frequency: 1 full cycle per 8 steps during forward motion
        step_freq = 0.8
        walk_cycle = (step * step_freq) % (2.0 * math.pi) if is_forward else 0.0

        # Tripod A: L1 (if not gripping gun), R2, L3
        # Tripod B: R1 (if not gripping gun), L2, R3
        tripod_a_phase = math.sin(walk_cycle)
        tripod_b_phase = math.sin(walk_cycle + math.pi)

        # Asymmetric step amplitude based on optic flow asymmetry
        amp_left = 1.0 - max(0.0, norm_asymmetry) * 0.6
        amp_right = 1.0 + min(0.0, norm_asymmetry) * 0.6

        # Solve walking legs: L2, R2, L3, R3
        legs: Dict[str, LegJointState] = {}

        # Leg L2 (Mid-left): Tripod B
        l2_phase = tripod_b_phase
        l2_stance = (l2_phase <= 0.0)
        l2_lift = max(0.0, l2_phase) * 22.0 if is_forward else 0.0
        l2_ext = -l2_phase * 28.0 * amp_left if is_forward else 0.0
        legs["L2"] = LegJointState(
            coxa=0.1 + l2_ext * 0.01,
            femur=0.45 - l2_lift * 0.02,
            tibia=-0.85 + l2_lift * 0.03,
            tarsus=0.4,
            foot_pos=self.coxa_bases["L2"] + np.array([-55.0, l2_ext, -45.0 + l2_lift], dtype=np.float32),
            is_stance=l2_stance,
        )

        # Leg R2 (Mid-right): Tripod A
        r2_phase = tripod_a_phase
        r2_stance = (r2_phase <= 0.0)
        r2_lift = max(0.0, r2_phase) * 22.0 if is_forward else 0.0
        r2_ext = -r2_phase * 28.0 * amp_right if is_forward else 0.0
        legs["R2"] = LegJointState(
            coxa=-0.1 - r2_ext * 0.01,
            femur=0.45 - r2_lift * 0.02,
            tibia=-0.85 + r2_lift * 0.03,
            tarsus=0.4,
            foot_pos=self.coxa_bases["R2"] + np.array([55.0, r2_ext, -45.0 + r2_lift], dtype=np.float32),
            is_stance=r2_stance,
        )

        # Leg L3 (Hind-left): Tripod A
        l3_phase = tripod_a_phase
        l3_stance = (l3_phase <= 0.0)
        l3_lift = max(0.0, l3_phase) * 24.0 if is_forward else 0.0
        l3_ext = -l3_phase * 32.0 * amp_left if is_forward else 0.0
        legs["L3"] = LegJointState(
            coxa=0.15 + l3_ext * 0.008,
            femur=0.60 - l3_lift * 0.02,
            tibia=-1.05 + l3_lift * 0.025,
            tarsus=0.45,
            foot_pos=self.coxa_bases["L3"] + np.array([-65.0, 45.0 + l3_ext, -55.0 + l3_lift], dtype=np.float32),
            is_stance=l3_stance,
        )

        # Leg R3 (Hind-right): Tripod B
        r3_phase = tripod_b_phase
        r3_stance = (r3_phase <= 0.0)
        r3_lift = max(0.0, r3_phase) * 24.0 if is_forward else 0.0
        r3_ext = -r3_phase * 32.0 * amp_right if is_forward else 0.0
        legs["R3"] = LegJointState(
            coxa=-0.15 - r3_ext * 0.008,
            femur=0.60 - r3_lift * 0.02,
            tibia=-1.05 + r3_lift * 0.025,
            tarsus=0.45,
            foot_pos=self.coxa_bases["R3"] + np.array([65.0, 45.0 + r3_ext, -55.0 + r3_lift], dtype=np.float32),
            is_stance=r3_stance,
        )

        # 3. Shotgun Recoil & Foreleg Grip Dynamics (L1 and R1)
        if is_fire or action == "FIRE":
            self.current_recoil = 18.0  # 18 mm backward kick
            self.current_flash = 1.0
        else:
            self.current_recoil = max(0.0, self.current_recoil * 0.72)  # fast relaxation
            self.current_flash = max(0.0, self.current_flash * 0.45)

        pose.gun_recoil = float(self.current_recoil)
        pose.muzzle_flash = float(self.current_flash)

        # Shotgun position with recoil offset
        pose.gun_pos = self.gun_base_pos + np.array([0.0, self.current_recoil, self.current_recoil * 0.25], dtype=np.float32)

        # Door reach gesture
        if is_use:
            self.current_door_reach = min(1.0, self.current_door_reach + 0.35)
        else:
            self.current_door_reach = max(0.0, self.current_door_reach - 0.20)
        pose.door_reach = float(self.current_door_reach)

        # Foreleg L1 (Left Prothoracic): Grips shotgun pump fore-end
        # Foot position tracks pump with recoil absorption flexion
        pump_pos = pose.gun_pos + np.array([-18.0, -45.0, 0.0], dtype=np.float32)
        legs["L1"] = LegJointState(
            coxa=0.35,
            femur=0.85 + self.current_recoil * 0.015,
            tibia=-1.10 - self.current_recoil * 0.02,
            tarsus=0.50,
            foot_pos=pump_pos,
            is_stance=False,  # dedicated grasping limb
        )

        # Foreleg R1 (Right Prothoracic): Grips trigger OR reaches for door
        if self.current_door_reach > 0.05:
            # Reaches forward past shotgun toward door switch
            reach_offset = self.current_door_reach * 65.0
            r1_pos = self.coxa_bases["R1"] + np.array([25.0, -110.0 - reach_offset, 15.0], dtype=np.float32)
            r1_femur = 0.30 - self.current_door_reach * 0.40
            r1_tibia = -0.50 + self.current_door_reach * 0.30
        else:
            # Grips shotgun receiver / trigger guard
            r1_pos = pose.gun_pos + np.array([16.0, -10.0, -8.0], dtype=np.float32)
            r1_femur = 0.75 + self.current_recoil * 0.015
            r1_tibia = -0.95 - self.current_recoil * 0.02

        legs["R1"] = LegJointState(
            coxa=-0.30,
            femur=float(r1_femur),
            tibia=float(r1_tibia),
            tarsus=0.45,
            foot_pos=r1_pos,
            is_stance=False,
        )

        pose.legs = legs

        # 4. Wing Flares on Gunfire / Startle
        if self.current_flash > 0.1:
            wing_flare = 0.28 * self.current_flash
            pose.left_wing_angle = wing_flare + 0.05 * math.sin(step * 1.5)
            pose.right_wing_angle = wing_flare + 0.05 * math.cos(step * 1.5)
        elif is_forward:
            pose.left_wing_angle = 0.03 * math.sin(step * 0.8)
            pose.right_wing_angle = 0.03 * math.sin(step * 0.8 + 0.5)
        else:
            pose.left_wing_angle = 0.0
            pose.right_wing_angle = 0.0

        return pose
