"""Abstract base class for sensory retinal encoders with type-enforced provenance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

from fly_doom.core.provenance import Provenance


class SensoryEncoder(ABC):
    """Abstract interface transforming 2D visual frames into 1D neural input currents."""

    def __init__(self, num_input_units: int, provenance: Provenance):
        self.num_input_units = num_input_units
        self.provenance = provenance

    @abstractmethod
    def encode_frame(self, frame: np.ndarray, dt_ms: float) -> np.ndarray:
        """Transform 2D visual frame (H, W) in [0.0, 1.0] to 1D neural input array."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal temporal states (e.g. previous frame buffer)."""
        pass
