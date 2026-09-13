"""Central Complex Fan-Shaped Body (FB) Vector Navigation & Sensorimotor Reflexes.

Biophysical modeling of:
1. Fan-Shaped Body (FB) Columnar Vector Steering:
   - 8-column neural matrix (C1..C8) receiving allocentric heading from the Ellipsoid Body (E-PG)
   - Integrates goal vector representations (Stone et al., Current Biology 2017; Green et al., Nature 2019)
   - Columnar P-FN, hΔB, and PFL3 comparator neurons producing left/right turning torque and forward thrust.
2. Subesophageal Zone (SEZ) Ventral Nociceptive Hazard Avoidance:
   - Tarsal/ventral ommatidial chrominance filtering detecting toxic nukage acid floors
   - Overrides forward drive and initiates urgent repulsive yaw away from acid pools.
3. Antennal Mechanosensory & Motor Center (AMMC) Wall-Slip Reflex:
   - Tactile stall detection triggering lateral deflection saccades to slip past doorjambs and corners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class WaypointNode:
    """A spatial milestone along the E1M1 level traversal."""

    zone_id: int
    name: str
    target_pos: Tuple[float, float]
    radius: float
    target_heading_deg: Optional[float] = None
    required_action: Optional[str] = None  # e.g., 'USE_DOOR', 'EXIT_SWITCH', 'COMBAT'
    description: str = ""


class FanShapedBodyVectorEngine:
    """8-Column Fan-Shaped Body (FB) Vector Navigation Engine.

    Models the biological P-FN -> hΔB -> PFL3 columnar microcircuit.
    Computes steering torque tau_steer and forward thrust v_thrust
    by calculating the vector difference between the current E-PG
    heading bump and the spatial goal vector.
    """

    NUM_COLUMNS: int = 8

    def __init__(self, deadband_deg: float = 12.0) -> None:
        self.deadband_rad = math.radians(deadband_deg)
        # 8 column preferred directions evenly spaced across [-pi, pi)
        self.column_angles = np.array([
            -math.pi + (c + 0.5) * (2.0 * math.pi / self.NUM_COLUMNS)
            for c in range(self.NUM_COLUMNS)
        ], dtype=np.float64)

        self.last_heading_rad: float = 0.0
        self.last_goal_angle_rad: float = 0.0
        self.last_steer_torque: float = 0.0
        self.last_forward_drive: float = 1.0

    def compute_vector_steering(
        self,
        current_heading_rad: float,
        goal_pos: Tuple[float, float],
        current_pos: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        """Compute steering torque and forward drive toward spatial goal.

        Args:
            current_heading_rad: Orientation in radians [-pi, pi).
            goal_pos: (gx, gy) world coordinates.
            current_pos: (px, py) world coordinates.

        Returns:
            (steer_torque, forward_thrust, angle_error_deg)
            steer_torque: [-1.0, 1.0] where positive is turn left, negative is turn right.
            forward_thrust: [0.0, 1.0] forward propulsion vigor.
            angle_error_deg: Discrepancy in degrees [-180, 180].
        """
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        dist = math.hypot(dx, dy)

        if dist < 1e-4:
            return 0.0, 0.0, 0.0

        goal_angle_rad = math.atan2(dy, dx)
        self.last_heading_rad = current_heading_rad
        self.last_goal_angle_rad = goal_angle_rad

        # Discrepancy wrapped to [-pi, pi)
        diff_rad = (goal_angle_rad - current_heading_rad + math.pi) % (2.0 * math.pi) - math.pi
        diff_deg = math.degrees(diff_rad)

        # Biological PFL3 / hΔB columnar comparator model
        # Columnar activation profile
        p_fn = np.maximum(0.0, np.cos(self.column_angles - current_heading_rad))
        goal_profile = np.maximum(0.0, np.cos(self.column_angles - goal_angle_rad))

        # Steering torque scaled by angular discrepancy
        if abs(diff_rad) <= self.deadband_rad:
            steer_torque = 0.0
        else:
            steer_torque = float(np.clip(math.sin(diff_rad) * 1.5, -1.0, 1.0))

        # Forward drive: full speed when facing goal, slows down during sharp turns
        forward_drive = float(np.clip(math.cos(diff_rad), 0.0, 1.0))
        if abs(diff_rad) > math.radians(60.0):
            forward_drive *= 0.25

        self.last_steer_torque = steer_torque
        self.last_forward_drive = forward_drive

        return steer_torque, forward_drive, diff_deg


class SEZNociceptiveReflex:
    """Subesophageal Zone (SEZ) Ventral Nukage Acid Avoidance Reflex.

    Analyzes ventral ommatidial rows of the retinal RGB frame for high
    green-to-(red+blue) chromatic saturation characteristic of toxic nukage pools.
    When acid is detected directly ahead on the floor, overrides forward movement
    and triggers repulsive turning toward safe walkway tiles.
    """

    def __init__(
        self,
        acid_threshold: float = 1.35,
        ventral_rows_ratio: float = 0.25,
    ) -> None:
        self.acid_threshold = acid_threshold
        self.ventral_rows_ratio = ventral_rows_ratio
        self.last_acid_detected: bool = False
        self.last_repulsive_bias: float = 0.0

    def evaluate_retinal_hazard(
        self, rgb_frame: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Evaluate retinal image for toxic floor hazard.

        Args:
            rgb_frame: (H, W, 3) uint8 RGB array.

        Returns:
            (acid_detected, repulsive_yaw_bias, acid_intensity)
            acid_detected: True if noxious pool is in immediate path.
            repulsive_yaw_bias: [-1.0, 1.0] steering bias away from acid.
            acid_intensity: Float measuring green saturation ratio.
        """
        if rgb_frame is None or rgb_frame.ndim != 3:
            return False, 0.0, 0.0

        h, w, _ = rgb_frame.shape
        start_row = int(h * (1.0 - self.ventral_rows_ratio))
        ventral_strip = rgb_frame[start_row:, :, :].astype(np.float32)

        r = ventral_strip[:, :, 0]
        g = ventral_strip[:, :, 1]
        b = ventral_strip[:, :, 2]

        # Chromatic green nukage ratio: G / (0.5*(R + B) + 1.0)
        acid_ratio = g / (0.5 * (r + b) + 1.0)
        acid_mask = acid_ratio > self.acid_threshold

        mid_w = w // 2
        center_strip = acid_mask[:, mid_w - w // 6 : mid_w + w // 6]
        left_strip = acid_mask[:, :mid_w]
        right_strip = acid_mask[:, mid_w:]

        center_acid_frac = float(np.mean(center_strip)) if center_strip.size > 0 else 0.0
        left_acid_frac = float(np.mean(left_strip)) if left_strip.size > 0 else 0.0
        right_acid_frac = float(np.mean(right_strip)) if right_strip.size > 0 else 0.0

        acid_detected = center_acid_frac > 0.18

        repulsive_yaw = 0.0
        if acid_detected:
            # Turn away from the side with more acid
            if left_acid_frac > right_acid_frac:
                repulsive_yaw = -1.0  # Turn right (away from left acid)
            else:
                repulsive_yaw = 1.0   # Turn left (away from right acid)

        self.last_acid_detected = acid_detected
        self.last_repulsive_bias = repulsive_yaw

        return acid_detected, repulsive_yaw, center_acid_frac


class AMMCWallSlipReflex:
    """Antennal Mechanosensory & Motor Center (AMMC) Tactile Obstacle Reflex.

    Monitors displacement velocity over time. When forward motion is active
    but physical position remains stalled, detects corner/doorframe collision
    and emits a lateral deflection saccade to slip past the obstruction.
    """

    def __init__(self, stall_threshold_units: float = 1.0, stall_ticks: int = 4) -> None:
        self.stall_threshold = stall_threshold_units
        self.stall_ticks = stall_ticks
        self._stalled_count: int = 0
        self._last_pos: Optional[Tuple[float, float]] = None
        self._saccade_ticks_remaining: int = 0
        self._saccade_direction: float = 1.0  # +1.0 = left, -1.0 = right

    def update(
        self,
        current_pos: Tuple[float, float],
        is_forward_commanded: bool,
    ) -> Tuple[bool, float]:
        """Update tactile wall slip monitor.

        Args:
            current_pos: (x, y) player coordinates.
            is_forward_commanded: True if forward locomotion is attempted.

        Returns:
            (is_active, slip_torque)
            is_active: True if wall slip reflex is currently overriding steering.
            slip_torque: [-1.0, 1.0] deflecting turn command.
        """
        if self._saccade_ticks_remaining > 0:
            self._saccade_ticks_remaining -= 1
            return True, self._saccade_direction

        if self._last_pos is not None and is_forward_commanded:
            disp = math.hypot(
                current_pos[0] - self._last_pos[0],
                current_pos[1] - self._last_pos[1],
            )
            if disp < self.stall_threshold:
                self._stalled_count += 1
            else:
                self._stalled_count = 0
        else:
            self._stalled_count = 0

        self._last_pos = current_pos

        if self._stalled_count >= self.stall_ticks:
            # Trigger a 3-tick deflection saccade
            self._saccade_ticks_remaining = 3
            self._saccade_direction = 1.0 if (self._stalled_count % 2 == 0) else -1.0
            self._stalled_count = 0
            return True, self._saccade_direction

        return False, 0.0

    def reset(self) -> None:
        self._stalled_count = 0
        self._last_pos = None
        self._saccade_ticks_remaining = 0


class Stage1WaypointGraph:
    """Topological Waypoint Graph for DOOM E1M1: Hangar (Spawn to Exit)."""

    def __init__(self) -> None:
        self.waypoints: List[WaypointNode] = [
            # Zone 1: Spawn Corridor
            WaypointNode(
                zone_id=1,
                name="Spawn_Corridor",
                target_pos=(1056.0, -3200.0),
                radius=120.0,
                target_heading_deg=90.0,
                description="Initial forward passage northward from spawn pad",
            ),
            # Zone 2: Hallway Elbow
            WaypointNode(
                zone_id=2,
                name="Hallway_Elbow",
                target_pos=(1350.0, -2600.0),
                radius=100.0,
                target_heading_deg=45.0,
                description="Diagonal rightward turn approaching Door 151 alcove",
            ),
            # Zone 3: Door 151 Alcove
            WaypointNode(
                zone_id=3,
                name="Door_151_Alcove",
                target_pos=(1536.0, -2496.0),
                radius=60.0,
                target_heading_deg=0.0,
                required_action="USE_DOOR",
                description="Door Line 151 requiring USE actuation to enter arena",
            ),
            # Zone 4: Doorway Ingress & Zigzag Walkway Entry
            WaypointNode(
                zone_id=4,
                name="Zigzag_Catwalk_Entry",
                target_pos=(1850.0, -2496.0),
                radius=120.0,
                target_heading_deg=0.0,
                required_action="COMBAT",
                description="Narrow bridge over nukage acid pool; eliminate perched zombiemen",
            ),
            # Zone 5: Zigzag Midpoint to Column Hall
            WaypointNode(
                zone_id=5,
                name="Zigzag_Walkway_Apex",
                target_pos=(2272.0, -2432.0),
                radius=140.0,
                target_heading_deg=35.0,
                description="Cross apex of acid walkway and approach staircase pillars",
            ),
            # Zone 6: Pillar Staircase to Upper Computer Hall
            WaypointNode(
                zone_id=6,
                name="Computer_Hall_Stairs",
                target_pos=(2800.0, -2800.0),
                radius=160.0,
                target_heading_deg=-45.0,
                required_action="COMBAT",
                description="Ascend stairs into computer terminal room with hostile Imps",
            ),
            # Zone 7: Exit Chamber & Final Switch
            WaypointNode(
                zone_id=7,
                name="Exit_Chamber_Switch",
                target_pos=(3100.0, -3600.0),
                radius=100.0,
                target_heading_deg=-90.0,
                required_action="EXIT_SWITCH",
                description="Final chamber linedef switch triggering STAGE CLEAR",
            ),
        ]
        self.current_idx: int = 0

    def get_current_waypoint(self) -> WaypointNode:
        return self.waypoints[min(self.current_idx, len(self.waypoints) - 1)]

    def update_progress(self, current_pos: Tuple[float, float]) -> WaypointNode:
        """Update active waypoint index based on Euclidean proximity."""
        wp = self.get_current_waypoint()
        dist = math.hypot(
            current_pos[0] - wp.target_pos[0],
            current_pos[1] - wp.target_pos[1],
        )
        # Advance if within radius and not at final switch
        if dist <= wp.radius and self.current_idx < len(self.waypoints) - 1:
            self.current_idx += 1
            wp = self.get_current_waypoint()
        return wp

    def reset(self) -> None:
        self.current_idx = 0
