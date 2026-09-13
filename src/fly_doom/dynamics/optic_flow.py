"""Retinotopic Columnar T4 Array & 2D Optic Flow Field Decomposition.

Implements a population-scale retinotopic array of elementary motion detectors (T4)
tiled across ommatidial cartridges with 2D Helmholtz-Hodge / optic flow decomposition:
  - T4a: Rightward preferred direction (+x)
  - T4b: Leftward preferred direction (-x)
  - T4c: Upward preferred direction (+y)
  - T4d: Downward preferred direction (-y)

Computes:
  1. Divergence / Centrifugal Looming (div V): expansion from focus of expansion (FOE).
  2. Curl / Rotational Shear (curl V): wide-field yaw motion mapped to LPTC-HS cells.
  3. Translational Slip (Tx, Ty): lateral and vertical background drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Tuple
import numpy as np

from fly_doom.core.provenance import Provenance
from fly_doom.dynamics.compartmental_t4 import CompartmentalParameters


@dataclass(frozen=True)
class OpticFlowMetrics:
    """Decomposed 2D optic flow field components."""

    divergence: float  # Centrifugal expansion (>0: looming obstacle/forward thrust)
    curl: float  # Rotational circulation / shear (>0: clockwise, <0: counter-clockwise)
    trans_x: float  # Mean horizontal slip velocity
    trans_y: float  # Mean vertical slip velocity
    lptc_hs_l: float  # Lobula Plate Tangential Cell HS membrane potential (Left)
    lptc_hs_r: float  # Lobula Plate Tangential Cell HS membrane potential (Right)
    looming_index: float  # Focus-of-expansion radial looming magnitude
    u_field: np.ndarray  # (H, W) horizontal flow field
    v_field: np.ndarray  # (H, W) vertical flow field


class RetinotopicT4ArrayEngine:
    """Vectorized population array of multi-compartmental T4 motion detectors."""

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        params: Optional[CompartmentalParameters] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.params = params or CompartmentalParameters()
        p = self.params

        # 4 directional subtypes: 'a' (+x), 'b' (-x), 'c' (+y), 'd' (-y)
        self.subtypes = ("a", "b", "c", "d")

        # State tensors of shape (4, rows, cols)
        # index 0: T4a, 1: T4b, 2: T4c, 3: T4d
        self.v_soma = np.full((4, rows, cols), p.v_rest, dtype=np.float64)
        self.v_leading = np.full((4, rows, cols), p.v_rest, dtype=np.float64)
        self.v_central = np.full((4, rows, cols), p.v_rest, dtype=np.float64)
        self.v_trailing = np.full((4, rows, cols), p.v_rest, dtype=np.float64)

        # Conductances
        self.g_mi1 = np.zeros((4, rows, cols), dtype=np.float64)
        self.g_tm3 = np.zeros((4, rows, cols), dtype=np.float64)
        self.g_mi4 = np.zeros((4, rows, cols), dtype=np.float64)
        self.g_mi9 = np.zeros((4, rows, cols), dtype=np.float64)

        # Refractory timers
        self.refractory_timer = np.zeros((4, rows, cols), dtype=np.int32)
        self.ref_steps = int(round(p.t_ref / p.dt))

        # Tm3 delay ring buffer: shape (max_delay_steps, 4, rows, cols)
        self.max_delay_steps = 64
        self.tm3_delay_steps = max(1, int(round(p.tm3_delay_ms / p.dt)))
        self.delay_buffer = np.zeros((self.max_delay_steps, 4, rows, cols), dtype=np.float64)
        self.ring_idx = 0

        # Precompute decay coefficients
        self.alpha_soma = math.exp(-p.dt / p.tau_soma)
        self.alpha_branch = math.exp(-p.dt / p.tau_branch)
        self.alpha_mi1 = math.exp(-p.dt / p.tau_syn_fast)
        self.alpha_tm3 = math.exp(-p.dt / p.tau_syn_slow)
        self.alpha_inh = math.exp(-p.dt / p.tau_syn_inh)

        # Precompute radial direction unit vectors from center for looming calculation
        y_coords = np.linspace(-1.0, 1.0, rows)
        x_coords = np.linspace(-1.0, 1.0, cols)
        xx, yy = np.meshgrid(x_coords, y_coords)
        radii = np.sqrt(xx**2 + yy**2)
        radii[radii < 1e-4] = 1e-4
        self.radial_x = xx / radii
        self.radial_y = yy / radii

        self.provenance = Provenance(
            tier="biological_reconstruction",
            source="Drosophila_Retinotopic_T4_Array_Haag_Borst_2020",
            confidence=0.92,
            rationale="2D retinotopic array of multi-compartment T4 motion detectors with 4 directional subtypes",
            doi="10.1016/j.cell.2020.03.045",
            figure_table_ref="Figure 3-5",
            access_date="2026-09-12",
        )

    def reset(self) -> None:
        p = self.params
        self.v_soma.fill(p.v_rest)
        self.v_leading.fill(p.v_rest)
        self.v_central.fill(p.v_rest)
        self.v_trailing.fill(p.v_rest)
        self.g_mi1.fill(0.0)
        self.g_tm3.fill(0.0)
        self.g_mi4.fill(0.0)
        self.g_mi9.fill(0.0)
        self.refractory_timer.fill(0)
        self.delay_buffer.fill(0.0)
        self.ring_idx = 0

    def step(
        self,
        on_currents: np.ndarray,
        knockouts: Optional[set[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance population T4 dynamics for all cartridges by dt.

        Args:
            on_currents: (rows, cols) sensory current matrix from retinal encoder.
            knockouts: Set of knocked-out cell types (e.g. {'Mi4'}, {'Mi9'}).

        Returns:
            Tuple of (u_field, v_field, spikes):
              - u_field: (rows, cols) horizontal motion vector field (+x is rightward)
              - v_field: (rows, cols) vertical motion vector field (+y is upward)
              - spikes: (4, rows, cols) boolean spike events
        """
        p = self.params
        ko = knockouts or set()

        # Spatial shifts for neighbor-mediated leading and trailing dendritic arbors:
        # Pad on_currents at edges with edge replication
        padded = np.pad(on_currents, ((1, 1), (1, 1)), mode="edge")

        # 4 directional subtypes receive presynaptic arbors from orthogonal neighbors:
        # Central is always the home cartridge (r, c)
        mi1_raw = on_currents

        # Trailing (Tm3, delayed excitation):
        # T4a (+x, rightward): trailing is to the left (c - 1)
        tm3_a = padded[1:-1, :-2]
        # T4b (-x, leftward): trailing is to the right (c + 1)
        tm3_b = padded[1:-1, 2:]
        # T4c (+y, upward): trailing is downward (r + 1 in image space, r - 1 in cartesian)
        tm3_c = padded[2:, 1:-1]
        # T4d (-y, downward): trailing is upward (r - 1 in image space)
        tm3_d = padded[:-2, 1:-1]

        # Leading (Mi4/Mi9, null-direction shunting/inhibition):
        # T4a (+x): leading is to the right (c + 1)
        mi4_a = padded[1:-1, 2:]
        # T4b (-x): leading is to the left (c - 1)
        mi4_b = padded[1:-1, :-2]
        # T4c (+y): leading is upward (r - 1)
        mi4_c = padded[:-2, 1:-1]
        # T4d (-y): leading is downward (r + 1)
        mi4_d = padded[2:, 1:-1]

        # Stack into (4, rows, cols) tensors
        in_mi1 = np.stack([mi1_raw, mi1_raw, mi1_raw, mi1_raw], axis=0) if "Mi1" not in ko else np.zeros_like(self.g_mi1)
        in_tm3 = np.stack([tm3_a, tm3_b, tm3_c, tm3_d], axis=0) if "Tm3" not in ko else np.zeros_like(self.g_tm3)
        in_mi4 = np.stack([mi4_a, mi4_b, mi4_c, mi4_d], axis=0) if "Mi4" not in ko else np.zeros_like(self.g_mi4)
        in_mi9 = in_mi4 * 0.6 if "Mi9" not in ko else np.zeros_like(self.g_mi9)

        # Scale input currents into physiological conductances
        input_scale = 0.05
        in_mi1 *= input_scale
        in_tm3 *= input_scale
        in_mi4 *= input_scale
        in_mi9 *= input_scale

        # 1. Update Tm3 delay ring buffer
        self.delay_buffer[self.ring_idx] = in_tm3
        read_idx = (self.ring_idx - self.tm3_delay_steps) % self.max_delay_steps
        delayed_tm3 = self.delay_buffer[read_idx]
        self.ring_idx = (self.ring_idx + 1) % self.max_delay_steps

        # 2. Update synaptic conductances
        self.g_mi1 = self.g_mi1 * self.alpha_mi1 + in_mi1
        self.g_tm3 = self.g_tm3 * self.alpha_tm3 + delayed_tm3
        self.g_mi4 = self.g_mi4 * self.alpha_inh + in_mi4
        self.g_mi9 = self.g_mi9 * self.alpha_inh + in_mi9

        # 3. Model D active biophysical dendritic tree (vectorized across all 128 cartridges)
        # Leading branch: shunting inhibition
        shunt = 1.0 / (1.0 + (p.shunting_factor * (self.g_mi4 + self.g_mi9)))
        inh_df = np.maximum(0.0, (self.v_leading - p.e_inh_gaba) / max(1e-3, (p.v_rest - p.e_inh_gaba)))
        i_inh = (self.g_mi4 + self.g_mi9) * inh_df
        self.v_leading = np.maximum(
            p.e_inh_gaba,
            p.v_rest + (self.v_leading - p.v_rest) * self.alpha_branch - i_inh,
        )

        # Central & Trailing branches: excitatory coincidence
        exc_cent_df = np.maximum(0.0, (p.e_exc - self.v_central) / max(1e-3, (p.e_exc - p.v_rest)))
        exc_trail_df = np.maximum(0.0, (p.e_exc - self.v_trailing) / max(1e-3, (p.e_exc - p.v_rest)))
        self.v_central = np.minimum(
            p.e_exc,
            p.v_rest + (self.v_central - p.v_rest) * self.alpha_branch + self.g_mi1 * exc_cent_df,
        )
        self.v_trailing = np.minimum(
            p.e_exc,
            p.v_rest + (self.v_trailing - p.v_rest) * self.alpha_branch + self.g_tm3 * exc_trail_df,
        )

        # Supralinear coincidence
        exc_coincidence = (
            p.coincidence_gain
            * np.maximum(0.0, self.v_central - p.v_rest)
            * np.maximum(0.0, self.v_trailing - p.v_rest)
            * 0.05
        )

        # Axial current convergence on soma
        i_axial_exc = ((self.v_central - p.v_rest) + (self.v_trailing - p.v_rest) + exc_coincidence) * shunt
        i_axial_inh = (self.v_leading - self.v_soma)
        i_axial = p.g_axial * (i_axial_exc + i_axial_inh)

        # Refractory decrement
        refractory_mask = self.refractory_timer > 0
        self.refractory_timer[refractory_mask] -= 1
        self.v_soma[refractory_mask] = p.v_reset

        non_refractory = ~refractory_mask
        self.v_soma[non_refractory] = np.clip(
            p.v_rest + (self.v_soma[non_refractory] - p.v_rest) * self.alpha_soma + i_axial[non_refractory],
            p.e_inh_gaba,
            p.e_exc + 10.0,
        )

        # Spiking
        spikes = self.v_soma >= p.v_thresh
        self.v_soma[spikes] = p.v_reset
        self.refractory_timer[spikes] = self.ref_steps

        # 4. Compute 2D Vector Flow Field
        # u(r, c) = T4a (right) - T4b (left)
        # v(r, c) = T4c (up) - T4d (down)
        depol = np.maximum(0.0, self.v_soma - p.v_rest) + (15.0 * spikes)
        u_field = depol[0] - depol[1]  # T4a - T4b
        v_field = depol[2] - depol[3]  # T4c - T4d

        return u_field, v_field, spikes


class OpticFlowDecomposer:
    """Decomposes 2D retinotopic flow into Helmholtz-Hodge and biological components."""

    def __init__(self, rows: int = 8, cols: int = 8):
        self.rows = rows
        self.cols = cols

        # Unit radial vectors for looming divergence
        y_coords = np.linspace(-1.0, 1.0, rows)
        x_coords = np.linspace(-1.0, 1.0, cols)
        xx, yy = np.meshgrid(x_coords, y_coords)
        radii = np.sqrt(xx**2 + yy**2)
        radii[radii < 1e-4] = 1e-4
        self.radial_x = xx / radii
        self.radial_y = yy / radii

        # Lobula Plate Tangential Cell (LPTC-HS) receptive field weighting:
        # HS cells preferentially pool horizontal motion in the equatorial retina
        y_gaussian = np.exp(-0.5 * (y_coords / 0.5) ** 2)[:, np.newaxis]
        self.hs_weights = np.tile(y_gaussian, (1, cols))
        self.hs_weights /= np.sum(self.hs_weights)

    def decompose(self, u_field: np.ndarray, v_field: np.ndarray) -> OpticFlowMetrics:
        """Decompose vector field into canonical aerodynamic & navigational components.

        Args:
            u_field: (rows, cols) horizontal motion field.
            v_field: (rows, cols) vertical motion field.

        Returns:
            OpticFlowMetrics with divergence, curl, slip, and LPTC-HS potentials.
        """
        # 1. Divergence (Expansion / Looming)
        # du/dx + dv/dy using centered finite differences
        dudx = np.gradient(u_field, axis=1)
        dvdy = np.gradient(v_field, axis=0)
        div_field = dudx + dvdy
        divergence = float(np.mean(div_field))

        # Radial looming index (dot product with outward radial unit vectors)
        looming_index = float(np.mean(u_field * self.radial_x + v_field * self.radial_y))

        # 2. Curl / Circulation (Rotational Shear)
        # dv/dx - du/dy
        dvdx = np.gradient(v_field, axis=1)
        dudy = np.gradient(u_field, axis=0)
        curl_field = dvdx - dudy
        curl = float(np.mean(curl_field))

        # 3. Translational Slip
        trans_x = float(np.mean(u_field))
        trans_y = float(np.mean(v_field))

        # 4. Bilateral Lobula Plate Tangential Cells (LPTC-HS)
        # Left hemisphere: cols [:cols//2], Right hemisphere: cols [cols//2:]
        mid = self.cols // 2
        u_left = u_field[:, :mid]
        u_right = u_field[:, mid:]

        # HS neurons pool horizontal motion; baseline rest is -60 mV
        # Left HS prefers rightward motion (u > 0 excites)
        # Right HS prefers leftward motion (u < 0 excites)
        hs_gain = 3.5
        v_hs_rest = -60.0
        lptc_hs_l = float(np.clip(v_hs_rest + hs_gain * np.mean(u_left), -75.0, -35.0))
        lptc_hs_r = float(np.clip(v_hs_rest - hs_gain * np.mean(u_right), -75.0, -35.0))

        return OpticFlowMetrics(
            divergence=divergence,
            curl=curl,
            trans_x=trans_x,
            trans_y=trans_y,
            lptc_hs_l=lptc_hs_l,
            lptc_hs_r=lptc_hs_r,
            looming_index=looming_index,
            u_field=u_field,
            v_field=v_field,
        )
