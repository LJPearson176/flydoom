"""Encoder Delta: Biologically constrained minimal encoder.

Projects visual frames onto a hexagonal ommatidial lattice with Gaussian receptive
fields and simple temporal differencing (Delta I_t = I_t - I_{t-1}), separating into
rectified ON and OFF channels WITHOUT explicit handcrafted motion-energy filtering.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from fly_doom.core.provenance import Provenance
from fly_doom.sensory.encoders.base import SensoryEncoder


class EncoderDelta(SensoryEncoder):
    """Hexagonal ommatidial lattice with Gaussian receptive fields and temporal differencing."""

    def __init__(
        self,
        num_columns: int = 8,
        num_rows: int = 8,
        field_width_px: int = 64,
        field_height_px: int = 64,
        sigma_px: float = 4.0,
        gain: float = 35.0,
    ):
        self.num_ommatidia = num_columns * num_rows
        self.gain = gain
        self.field_width = field_width_px
        self.field_height = field_height_px
        self.sigma = sigma_px

        # Output units: 2 channels per ommatidium (ON channel: +Delta I, OFF channel: -Delta I)
        super().__init__(
            num_input_units=self.num_ommatidia * 2,
            provenance=Provenance(
                tier="experimental_assumption",
                source="Hexagonal_Ommatidia_Temporal_Diff_Nature2025",
                confidence=0.90,
                rationale="Hexagonal ommatidial receptive fields with simple temporal differencing; no explicit EMD filtering",
            ),
        )

        # Build hexagonal lattice centers
        self.centers: List[Tuple[float, float]] = []
        dx = field_width_px / (num_columns + 0.5)
        dy = field_height_px / (num_rows + 0.5)

        for r in range(num_rows):
            # Row offset for hexagonal packing
            x_offset = (dx * 0.5) if (r % 2 == 1) else 0.0
            y_pos = (r + 0.5) * dy
            for c in range(num_columns):
                x_pos = (c + 0.5) * dx + x_offset
                self.centers.append((x_pos, y_pos))

        # Precompute Gaussian spatial integration weights for each ommatidium
        ys, xs = np.meshgrid(
            np.arange(field_height_px, dtype=np.float64),
            np.arange(field_width_px, dtype=np.float64),
            indexing="ij",
        )
        self.receptive_fields = np.zeros((self.num_ommatidia, field_height_px, field_width_px), dtype=np.float64)

        two_sig2 = 2.0 * (sigma_px**2)
        for i, (cx, cy) in enumerate(self.centers):
            dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
            rf = np.exp(-dist2 / two_sig2)
            # Normalize so each ommatidium integrates to 1.0
            total = np.sum(rf)
            if total > 0:
                rf /= total
            self.receptive_fields[i] = rf

        # Flatten receptive fields for fast matrix-vector dot product
        self.rf_matrix = self.receptive_fields.reshape(self.num_ommatidia, -1)  # (N_ommatidia, H*W)

        self.prev_ommatidial_luminance: Optional[np.ndarray] = None

    def encode_frame(self, frame: np.ndarray, dt_ms: float) -> np.ndarray:
        """Project frame onto ommatidia and compute rectified ON and OFF temporal difference."""
        flat_frame = frame.flatten()
        # Fast spatial integration: (N_ommatidia, H*W) @ (H*W,) -> (N_ommatidia,)
        curr_luminance = self.rf_matrix @ flat_frame

        if self.prev_ommatidial_luminance is None:
            # First frame baseline
            delta_i = np.zeros_like(curr_luminance)
        else:
            delta_i = (curr_luminance - self.prev_ommatidial_luminance) / (dt_ms / 1000.0)

        self.prev_ommatidial_luminance = curr_luminance.copy()

        # Rectify into ON channel (+delta) and OFF channel (-delta)
        on_channel = np.maximum(delta_i, 0.0) * self.gain
        off_channel = np.maximum(-delta_i, 0.0) * self.gain

        # Interleave or concatenate: [ON_0..N-1, OFF_0..N-1]
        output_currents = np.concatenate([on_channel, off_channel])
        return output_currents

    def reset(self) -> None:
        self.prev_ommatidial_luminance = None
