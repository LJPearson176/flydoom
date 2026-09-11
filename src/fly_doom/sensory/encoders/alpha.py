"""Encoder Alpha: Naive planar rectangular downsampling.

Converts visual frames directly to a square grid of intensity inputs without
biological optical geometry or temporal filtering.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from fly_doom.core.provenance import Provenance
from fly_doom.sensory.encoders.base import SensoryEncoder


class EncoderAlpha(SensoryEncoder):
    """Naive planar downsampling encoder."""

    def __init__(self, grid_size: int = 8, gain: float = 25.0):
        self.grid_size = grid_size
        self.gain = gain
        num_units = grid_size * grid_size
        super().__init__(
            num_input_units=num_units,
            provenance=Provenance(
                tier="engineering_scaffold",
                source="Naive_Planar_Raster_Downsampler",
                confidence=1.0,
                rationale="Naive engineering control: planar square downsampling without eye geometry or temporal diff",
            ),
        )

    def encode_frame(self, frame: np.ndarray, dt_ms: float) -> np.ndarray:
        h, w = frame.shape
        if h != self.grid_size or w != self.grid_size:
            zoom_y = self.grid_size / h
            zoom_x = self.grid_size / w
            downsampled = zoom(frame, (zoom_y, zoom_x), order=1)
        else:
            downsampled = frame

        # Direct luminance to current scaling
        currents = downsampled.flatten() * self.gain
        return currents

    def reset(self) -> None:
        pass
