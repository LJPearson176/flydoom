"""DOOM-001 Environment Protocol and Observation / Action Interfaces.

Defines the standard sensorimotor interface for FlyDoom embodied agents:
  - Observation: RGB retinal frame, Depth map, Health, Ammo, Kill count, Agent pose
  - Action space: NOOP, FORWARD, TURN_LEFT, TURN_RIGHT, FIRE, USE
  - Step info: step_count, reward, done, and cybernetic telemetry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Protocol, Tuple
import numpy as np

from fly_doom.core.provenance import Provenance


class DoomAction(IntEnum):
    """Minimal discrete action space for embodied insectoid navigation and combat."""

    NOOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    FIRE = 4
    USE = 5


@dataclass
class DoomObservation:
    """Sensory and internal physiological state returned at each simulation step."""

    # Retinal visual input (H, W, 3) in [0, 255] uint8 or float32 [0.0, 1.0]
    rgb: np.ndarray

    # Depth map (H, W) in world distance units (meters or game units)
    depth: np.ndarray

    # Internal physiological telemetry
    health: float = 100.0
    ammo: int = 50
    kill_count: int = 0
    damage_dealt: float = 0.0

    # Embodied pose
    x: float = 0.0
    y: float = 0.0
    angle_rad: float = 0.0

    # Step metadata
    step_count: int = 0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class DoomEnvironment(Protocol):
    """Abstract protocol for embodied Doom environments (MockArena or native ViZDoom)."""

    def reset(self, seed: Optional[int] = None) -> DoomObservation:
        """Reset environment to initial state with specified seed."""
        ...

    def step(self, action: DoomAction | int) -> Tuple[DoomObservation, float, bool, Dict[str, Any]]:
        """Advance simulation by one action step.

        Returns: (observation, reward, done, info)
        """
        ...

    def close(self) -> None:
        """Clean up environment resources."""
        ...
