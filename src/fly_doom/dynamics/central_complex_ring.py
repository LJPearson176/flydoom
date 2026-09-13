"""Central Complex Ellipsoid Body Continuous Attractor Network (CAN).

Models the 16-wedge toroidal ring attractor of E-PG (compass) neurons in the
Drosophila Central Complex, cross-coupled to P-EN (angular velocity shift)
and Delta7 (global inhibition) interneurons:
  - 16 wedges tile the 360-degree azimuthal heading space.
  - Recurrent cosine-profile excitation maintains a single localized activity bump.
  - Global subtractive/divisive inhibition enforces sparsity and prevents runaway excitation.
  - P-EN phase-shifted connections steer the bump based on optomotor yaw and efference copy.
  - Sustains heading persistence during straight walking, occlusions, and darkness.

References:
  - Green et al. (Nature 2017) "A neural circuit architecture for angular integration in Drosophila"
  - Turner-Evans et al. (eLife 2020) "The neuroanatomical ultrastructure and function of a heading direction circuit"
  - Hulse et al. (eLife 2021) "A connectome of the Drosophila central complex"
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from fly_doom.core.provenance import Provenance


@dataclass(frozen=True)
class RingAttractorState:
    """Snapshot of central complex compass state."""

    heading_angle_rad: float  # Decoded azimuthal heading [-pi, pi]
    heading_angle_deg: float  # Decoded azimuthal heading [-180, 180]
    bump_amplitude: float  # Peak activity of the attractor bump
    bump_coherence: float  # Circular vector strength R in [0, 1]
    wedge_activations: np.ndarray  # (16,) normalized wedge activities


class EllipsoidBodyRingAttractor:
    """16-wedge continuous attractor ring network modeling Drosophila E-PG compass neurons."""

    def __init__(
        self,
        num_wedges: int = 16,
        dt_ms: float = 16.67,
        tau_ms: float = 25.0,
        w_exc: float = 1.4,
        w_inh: float = 0.8,
        shift_gain: float = 3.2,
        baseline_drive: float = 0.25,
    ):
        self.num_wedges = num_wedges
        self.dt = dt_ms
        self.tau = tau_ms
        self.w_exc = w_exc
        self.w_inh = w_inh
        self.shift_gain = shift_gain
        self.baseline_drive = baseline_drive

        # Angular preferred directions for 16 wedges spanning [-pi, pi)
        self.angles = np.linspace(-np.pi, np.pi, num_wedges, endpoint=False)

        # Recurrent excitatory connectivity matrix: cosine topological coupling
        # W_ij = w_exc * cos(theta_i - theta_j)
        angle_diffs = self.angles[:, np.newaxis] - self.angles[np.newaxis, :]
        self.w_matrix = self.w_exc * np.cos(angle_diffs)

        # Decay factor
        self.alpha = math.exp(-self.dt / self.tau)

        # Neuron activations A_i in [0, inf)
        self.activations = np.zeros(num_wedges, dtype=np.float64)

        # Initialize with a canonical bump centered at heading 0 (facing forward)
        initial_bump = np.exp(-0.5 * (self.angles / 0.6) ** 2)
        self.activations = initial_bump / np.sum(initial_bump)

        self.provenance = Provenance(
            tier="biological_reconstruction",
            source="Central_Complex_E-PG_Ring_Attractor_Green_2017",
            confidence=0.95,
            rationale="16-wedge continuous attractor network with P-EN angular velocity shifting and Delta7 inhibition",
            doi="10.1038/nature22349",
            figure_table_ref="Figures 1-4",
            access_date="2026-09-12",
        )

    def reset(self, initial_heading_rad: float = 0.0) -> None:
        """Reset the ring attractor with bump centered at given heading angle."""
        diff = np.arctan2(np.sin(self.angles - initial_heading_rad), np.cos(self.angles - initial_heading_rad))
        initial_bump = np.exp(-0.5 * (diff / 0.6) ** 2)
        self.activations = initial_bump / np.sum(initial_bump)

    def step(
        self,
        angular_velocity: float,
        visual_cue_angle: Optional[float] = None,
        visual_cue_weight: float = 0.0,
    ) -> RingAttractorState:
        """Advance ring attractor dynamics by dt given angular velocity and optional visual cue.

        Args:
            angular_velocity: Turn velocity in rad/s (positive = turning right, negative = turning left).
            visual_cue_angle: Optional visual landmark heading angle in radians [-pi, pi].
            visual_cue_weight: Strength of visual landmark anchoring.

        Returns:
            RingAttractorState with decoded heading, bump amplitude, and wedge activations.
        """
        # Current decoded heading before update
        sin_sum = float(np.sum(self.activations * np.sin(self.angles)))
        cos_sum = float(np.sum(self.activations * np.cos(self.angles)))
        current_heading = math.atan2(sin_sum, cos_sum)

        # 1. Recurrent excitatory drive
        recurrent_drive = np.dot(self.w_matrix, self.activations)

        # 2. P-EN asymmetric shift drive:
        # P-EN phase shift moves activity in direction of angular rotation
        # Derivative of cosine profile: -sin(theta_i - current_heading)
        phase_shift_drive = -self.shift_gain * angular_velocity * np.sin(self.angles - current_heading)

        # 3. Delta7 global inhibition: uniform subtractive suppression
        global_inhibition = self.w_inh * np.sum(self.activations)

        # 4. Visual landmark anchoring drive (e.g. from anterior optic tubercle / bulb)
        visual_drive = np.zeros(self.num_wedges, dtype=np.float64)
        if visual_cue_angle is not None and visual_cue_weight > 0.0:
            cue_diff = np.arctan2(
                np.sin(self.angles - visual_cue_angle),
                np.cos(self.angles - visual_cue_angle),
            )
            visual_drive = visual_cue_weight * np.exp(-0.5 * (cue_diff / 0.5) ** 2)

        # 5. Net input current to each wedge
        i_net = recurrent_drive + phase_shift_drive + visual_drive + self.baseline_drive - global_inhibition

        # 6. Threshold-linear activation function [I]_+ and membrane decay
        target_a = np.maximum(0.0, i_net)
        self.activations = self.alpha * self.activations + (1.0 - self.alpha) * target_a

        # Soft normalization to prevent infinite growth while preserving bump shape
        total_act = np.sum(self.activations)
        if total_act > 1e-6:
            self.activations = (self.activations / total_act) * 2.0  # Normalized target sum = 2.0

        # 7. Decode Population Vector
        sin_pop = float(np.sum(self.activations * np.sin(self.angles)))
        cos_pop = float(np.sum(self.activations * np.cos(self.angles)))
        decoded_angle_rad = math.atan2(sin_pop, cos_pop)
        decoded_angle_deg = math.degrees(decoded_angle_rad)

        total = float(np.sum(self.activations))
        bump_amplitude = float(np.max(self.activations))
        bump_coherence = float(math.sqrt(sin_pop**2 + cos_pop**2) / max(1e-4, total))

        return RingAttractorState(
            heading_angle_rad=decoded_angle_rad,
            heading_angle_deg=decoded_angle_deg,
            bump_amplitude=bump_amplitude,
            bump_coherence=bump_coherence,
            wedge_activations=np.copy(self.activations),
        )
