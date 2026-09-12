"""Multi-compartment biophysical simulation engine for reconstructed Drosophila T4 neurons.

Models spatial dendritic compartmentalization:
  - 3 dendritic compartments:
      * Leading branch (Mi4 / Mi9 null-direction inhibition)
      * Central branch (Mi1 non-delayed base excitation)
      * Trailing branch (Tm3 delayed excitation)
  - 1 soma / spike initiation zone (SIZ)

Supports 4 model paradigms:
  - Model A: Canonical point-LIF (all inputs lumped into soma, uniform parameters)
  - Model B: Point-LIF with calibrated heterogeneous delays and time constants
  - Model C: Passive multi-compartment dendritic tree (branch-level linear filtering, passive axial coupling)
  - Model D: Active / nonlinear multi-compartment tree (local shunting inhibition on leading branch,
             supralinear coincidence between central and trailing excitation, feeding SIZ)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from fly_doom.connectome.t4_anatomical import T4AnatomicalReconstruction
from fly_doom.core.provenance import Provenance


class CompartmentModelType(str, Enum):
    MODEL_A = "model_a_point_lif"
    MODEL_B = "model_b_temporal_point_lif"
    MODEL_C = "model_c_passive_compartments"
    MODEL_D = "model_d_nonlinear_compartments"


@dataclass(frozen=True)
class CompartmentalParameters:
    """Parameters for multi-compartment T4 dendritic integration and spike initiation."""

    # Soma parameters
    v_rest: float = -65.0  # mV
    v_reset: float = -70.0  # mV
    v_thresh: float = -50.0  # mV
    tau_soma: float = 15.0  # ms
    t_ref: float = 2.0  # ms
    dt: float = 1.0  # ms

    # Dendritic branch parameters
    tau_branch: float = 20.0  # ms (membrane time constant of dendritic branches)
    g_axial: float = 0.35  # Coupling conductance from each branch to soma

    # Synaptic reversal potentials & decay times
    e_exc: float = 0.0  # mV (ACh excitatory reversal potential)
    e_inh_gaba: float = -75.0  # mV (GABA_A shunting reversal potential)
    e_inh_glu: float = -80.0  # mV (GluCl hyperpolarizing reversal potential)

    tau_syn_fast: float = 15.0  # ms (Mi1 fast excitation)
    tau_syn_slow: float = 60.0  # ms (Tm3 slow excitation)
    tau_syn_inh: float = 40.0  # ms (Mi4/Mi9 inhibition)
    tm3_delay_ms: float = 20.0  # ms (synaptic delay for trailing Tm3 input)

    # Nonlinearity parameters (Model D)
    shunting_factor: float = 1.8  # Shunting gain on leading branch
    coincidence_gain: float = 0.75  # Supralinear boost between central and trailing excitation

    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            tier="computational_hypothesis",
            source="Compartmental_T4_Biophysics_Borst_Haag_2020",
            confidence=0.85,
            rationale="Multi-compartment passive cable and conductance-based dendritic tree for Drosophila T4",
            doi="10.1016/j.cell.2020.03.045",
            figure_table_ref="Figure 2-4",
            access_date="2026-09-11",
        )
    )


class CompartmentalT4Engine:
    """Simulates single T4 neuron under one of 4 model configurations."""

    def __init__(
        self,
        reconstruction: T4AnatomicalReconstruction,
        params: CompartmentalParameters,
        model_type: CompartmentModelType = CompartmentModelType.MODEL_D,
    ):
        self.recon = reconstruction
        self.params = params
        self.model_type = model_type

        # State variables
        self.v_soma = params.v_rest
        self.refractory_timer = 0

        # Branch potentials
        self.v_leading = params.v_rest
        self.v_central = params.v_rest
        self.v_trailing = params.v_rest

        # Synaptic currents / conductances
        self.g_mi1 = 0.0  # Central ACh
        self.g_tm3 = 0.0  # Trailing ACh
        self.g_mi4 = 0.0  # Leading GABA
        self.g_mi9 = 0.0  # Leading Glu

        # Delay buffer for Tm3
        self.max_delay_steps = 64
        self.tm3_delay_steps = int(round(params.tm3_delay_ms / params.dt))
        self.delay_buffer = np.zeros(self.max_delay_steps, dtype=np.float64)
        self.ring_idx = 0

        # Precompute decay factors
        self.alpha_soma = math.exp(-params.dt / params.tau_soma)
        self.alpha_branch = math.exp(-params.dt / params.tau_branch)
        self.alpha_mi1 = math.exp(-params.dt / params.tau_syn_fast)
        self.alpha_tm3 = math.exp(-params.dt / (params.tau_syn_slow if model_type != CompartmentModelType.MODEL_A else params.tau_syn_fast))
        self.alpha_inh = math.exp(-params.dt / params.tau_syn_inh)
        self.ref_steps = int(round(params.t_ref / params.dt))

    def reset(self) -> None:
        self.v_soma = self.params.v_rest
        self.refractory_timer = 0
        self.v_leading = self.params.v_rest
        self.v_central = self.params.v_rest
        self.v_trailing = self.params.v_rest
        self.g_mi1 = 0.0
        self.g_tm3 = 0.0
        self.g_mi4 = 0.0
        self.g_mi9 = 0.0
        self.delay_buffer.fill(0.0)
        self.ring_idx = 0

    def step(
        self,
        in_mi1: float,
        in_tm3: float,
        in_mi4: float,
        in_mi9: float,
    ) -> bool:
        """Advance simulation by dt_ms given input currents/rates from the 4 presynaptic types.

        Returns True if soma fired a spike this step.
        """
        p = self.params

        # 1. Update delay buffer for Tm3
        if self.model_type in (CompartmentModelType.MODEL_A,):
            # Model A has zero delay
            delayed_tm3 = in_tm3
        else:
            self.delay_buffer[self.ring_idx] = in_tm3
            read_idx = (self.ring_idx - self.tm3_delay_steps) % self.max_delay_steps
            delayed_tm3 = float(self.delay_buffer[read_idx])
            self.ring_idx = (self.ring_idx + 1) % self.max_delay_steps

        # 2. Update synaptic conductances
        self.g_mi1 = self.g_mi1 * self.alpha_mi1 + in_mi1
        self.g_tm3 = self.g_tm3 * self.alpha_tm3 + delayed_tm3
        self.g_mi4 = self.g_mi4 * self.alpha_inh + in_mi4
        self.g_mi9 = self.g_mi9 * self.alpha_inh + in_mi9

        # 3. Dynamic Integration according to Model Type
        if self.model_type == CompartmentModelType.MODEL_A:
            # Model A: Canonical Point-LIF
            # Everything sums linearly into soma current
            i_syn = (self.g_mi1 + self.g_tm3) - (self.g_mi4 + self.g_mi9)
            if self.refractory_timer > 0:
                self.refractory_timer -= 1
                self.v_soma = p.v_reset
                return False

            self.v_soma = p.v_rest + (self.v_soma - p.v_rest) * self.alpha_soma + i_syn

        elif self.model_type == CompartmentModelType.MODEL_B:
            # Model B: Point-LIF with calibrated heterogeneous delays & tau
            # Synaptic conductances have distinct decay, but sum at single point soma
            i_syn = (self.g_mi1 + self.g_tm3) - (self.g_mi4 + self.g_mi9) * 1.5
            if self.refractory_timer > 0:
                self.refractory_timer -= 1
                self.v_soma = p.v_reset
                return False

            self.v_soma = p.v_rest + (self.v_soma - p.v_rest) * self.alpha_soma + i_syn

        elif self.model_type == CompartmentModelType.MODEL_C:
            # Model C: Passive multi-compartment dendritic tree
            # Leading branch: receives Mi4 + Mi9
            i_lead = -(self.g_mi4 + self.g_mi9) * 1.2
            self.v_leading = p.v_rest + (self.v_leading - p.v_rest) * self.alpha_branch + i_lead

            # Central branch: receives Mi1
            i_cent = self.g_mi1 * 1.2
            self.v_central = p.v_rest + (self.v_central - p.v_rest) * self.alpha_branch + i_cent

            # Trailing branch: receives delayed Tm3
            i_trail = self.g_tm3 * 1.2
            self.v_trailing = p.v_rest + (self.v_trailing - p.v_rest) * self.alpha_branch + i_trail

            # Axial currents converge on soma
            i_axial = p.g_axial * (
                (self.v_leading - self.v_soma)
                + (self.v_central - self.v_soma)
                + (self.v_trailing - self.v_soma)
            )

            if self.refractory_timer > 0:
                self.refractory_timer -= 1
                self.v_soma = p.v_reset
                return False

            self.v_soma = p.v_rest + (self.v_soma - p.v_rest) * self.alpha_soma + i_axial

        elif self.model_type == CompartmentModelType.MODEL_D:
            # Model D: Active / nonlinear multi-compartment tree
            # 1. Leading branch has shunting inhibition: local divisive suppression
            shunt = 1.0 / (1.0 + (p.shunting_factor * (self.g_mi4 + self.g_mi9)))
            # Conductance-based driving force: current diminishes as V approaches reversal potential (-75 mV)
            inh_df = max(0.0, (self.v_leading - p.e_inh_gaba) / max(1e-3, (p.v_rest - p.e_inh_gaba)))
            i_inh = (self.g_mi4 + self.g_mi9) * inh_df
            self.v_leading = max(p.e_inh_gaba, p.v_rest + (self.v_leading - p.v_rest) * self.alpha_branch - i_inh)

            # 2. Central & Trailing branches integrate excitation with dendritic coincidence
            exc_cent_df = max(0.0, (p.e_exc - self.v_central) / max(1e-3, (p.e_exc - p.v_rest)))
            exc_trail_df = max(0.0, (p.e_exc - self.v_trailing) / max(1e-3, (p.e_exc - p.v_rest)))
            self.v_central = min(p.e_exc, p.v_rest + (self.v_central - p.v_rest) * self.alpha_branch + self.g_mi1 * exc_cent_df)
            self.v_trailing = min(p.e_exc, p.v_rest + (self.v_trailing - p.v_rest) * self.alpha_branch + self.g_tm3 * exc_trail_df)

            # Supralinear coincidence between central and trailing excitation
            exc_coincidence = p.coincidence_gain * max(0.0, self.v_central - p.v_rest) * max(0.0, self.v_trailing - p.v_rest) * 0.05

            # Net forward dendritic transfer is gated by leading shunting inhibition
            # (If null direction moves leading->central, shunt arrives first and quenches excitation)
            i_axial_exc = ((self.v_central - p.v_rest) + (self.v_trailing - p.v_rest) + exc_coincidence) * shunt
            i_axial_inh = (self.v_leading - self.v_soma)  # Hyperpolarizing component relative to soma

            i_axial = p.g_axial * (i_axial_exc + i_axial_inh)

            if self.refractory_timer > 0:
                self.refractory_timer -= 1
                self.v_soma = p.v_reset
                return False

            self.v_soma = max(p.e_inh_gaba, min(p.e_exc + 10.0, p.v_rest + (self.v_soma - p.v_rest) * self.alpha_soma + i_axial))

        # 4. Spike threshold detection
        if self.v_soma >= p.v_thresh:
            self.v_soma = p.v_reset
            self.refractory_timer = self.ref_steps
            return True

        return False
