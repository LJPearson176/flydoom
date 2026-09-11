"""Vectorized Python/NumPy reference Leaky Integrate-and-Fire (LIF) engine.

Acts as the golden mathematical reference for numerical equivalence gates.
Implements exact LIF integration with:
  - Exponential membrane potential leak
  - Exponential synaptic conductance decay
  - Absolute refractory periods
  - Synaptic event propagation across CSR graph
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np

from fly_doom.connectome.graph import ConnectomeGraph
from fly_doom.core.provenance import Provenance, attach_provenance


@dataclass(frozen=True)
class LIFParameters:
    """Biophysical parameters for LIF dynamics."""

    v_rest: float = -65.0  # mV
    v_reset: float = -70.0  # mV
    v_thresh: float = -50.0  # mV
    tau_m: float = 20.0  # ms
    tau_syn: float = 5.0  # ms
    t_ref: float = 2.0  # ms
    dt: float = 1.0  # ms
    r_m: float = 1.0  # MOhm
    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            tier="computational_hypothesis",
            source="LIF_Standard_Biophysics",
            confidence=0.85,
            rationale="Standard point-neuron LIF model with exponential membrane and synaptic decay",
        )
    )

    @property
    def alpha_m(self) -> float:
        """Membrane potential decay factor per timestep."""
        return math.exp(-self.dt / self.tau_m)

    @property
    def alpha_syn(self) -> float:
        """Synaptic current decay factor per timestep."""
        return math.exp(-self.dt / self.tau_syn)

    @property
    def refractory_steps(self) -> int:
        """Number of discrete steps for refractory period."""
        return int(round(self.t_ref / self.dt))


@dataclass
class LIFState:
    """Dynamic state variables for a simulated population of neurons."""

    num_neurons: int
    v: np.ndarray  # float64: membrane potential (mV)
    i_syn: np.ndarray  # float64: synaptic input current (nA)
    refractory_timer: np.ndarray  # int32: remaining refractory steps

    @classmethod
    def initialize(cls, num_neurons: int, params: LIFParameters) -> LIFState:
        return cls(
            num_neurons=num_neurons,
            v=np.full(num_neurons, params.v_rest, dtype=np.float64),
            i_syn=np.zeros(num_neurons, dtype=np.float64),
            refractory_timer=np.zeros(num_neurons, dtype=np.int32),
        )

    def copy(self) -> LIFState:
        return LIFState(
            num_neurons=self.num_neurons,
            v=self.v.copy(),
            i_syn=self.i_syn.copy(),
            refractory_timer=self.refractory_timer.copy(),
        )


class LIFReferenceEngine:
    """Golden reference simulator implemented purely in NumPy with float64 precision."""

    def __init__(self, graph: ConnectomeGraph, params: Optional[LIFParameters] = None):
        self.graph = graph
        self.params = params or LIFParameters()
        self.state = LIFState.initialize(graph.num_neurons, self.params)
        self.step_count: int = 0

    def step(self, external_current: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute one simulation step of duration dt.

        Args:
            external_current: Optional external input current per neuron (float64 array).

        Returns:
            spikes: 1D array of integer indices of neurons that fired in this step.
        """
        p = self.params
        s = self.state

        # 1. Decay synaptic currents
        s.i_syn *= p.alpha_syn

        # 2. Add external current to synaptic input buffer
        if external_current is not None:
            s.i_syn += external_current

        # 3. Identify non-refractory neurons
        non_refractory = s.refractory_timer <= 0
        refractory = ~non_refractory

        # 4. Integrate membrane potential for non-refractory neurons
        # V(t) = alpha_m * V(t-1) + (1 - alpha_m) * V_rest + dt * R_m * I_syn
        if np.any(non_refractory):
            dv = (1.0 - p.alpha_m) * p.v_rest + (p.dt * p.r_m) * s.i_syn[non_refractory]
            s.v[non_refractory] = p.alpha_m * s.v[non_refractory] + dv

        # For refractory neurons, clamp to v_reset and decrement counter
        if np.any(refractory):
            s.v[refractory] = p.v_reset
            s.refractory_timer[refractory] -= 1

        # 5. Detect threshold crossings
        spike_mask = (s.v >= p.v_thresh) & non_refractory
        spike_indices = np.flatnonzero(spike_mask).astype(np.int64)

        # 6. Apply reset and refractory period to spiking neurons
        if len(spike_indices) > 0:
            s.v[spike_indices] = p.v_reset
            s.refractory_timer[spike_indices] = p.refractory_steps

            # 7. Propagate synaptic spikes across CSR graph
            # For each pre-synaptic spiking neuron i, add outgoing weights to targets
            for pre_idx in spike_indices:
                start = self.graph.row_ptr[pre_idx]
                end = self.graph.row_ptr[pre_idx + 1]
                if end > start:
                    targets = self.graph.col_idx[start:end]
                    weights = self.graph.weights[start:end]
                    np.add.at(s.i_syn, targets, weights)

        self.step_count += 1
        return spike_indices

    def run(
        self,
        num_steps: int,
        stimulus_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run simulation for num_steps, recording voltages and spike trains.

        Returns:
            (v_trace, spike_trace): list of voltage arrays and list of spike index arrays per step.
        """
        v_trace: List[np.ndarray] = []
        spike_trace: List[np.ndarray] = []

        for t in range(num_steps):
            ext_i = stimulus_fn(t) if stimulus_fn else None
            spikes = self.step(ext_i)
            v_trace.append(self.state.v.copy())
            spike_trace.append(spikes)

        return v_trace, spike_trace
