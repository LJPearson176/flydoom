"""Vectorized Python/NumPy reference Leaky Integrate-and-Fire (LIF) engine.

Acts as the golden mathematical reference for numerical equivalence gates.
Implements exact LIF integration with:
  - Exponential membrane potential leak
  - Heterogeneous or uniform exponential synaptic conductance decay
  - Synaptic transmission delay ring-buffer queue
  - Absolute refractory periods
  - Synaptic event propagation across CSR graph
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

from fly_doom.connectome.graph import ConnectomeGraph
from fly_doom.core.provenance import Provenance

MAX_DELAY_STEPS = 128


@dataclass(frozen=True)
class LIFParameters:
    """Biophysical parameters for LIF dynamics."""

    v_rest: float = -65.0  # mV
    v_reset: float = -70.0  # mV
    v_thresh: float = -50.0  # mV
    tau_m: float = 20.0  # ms
    tau_syn: Union[float, np.ndarray] = 5.0  # ms (scalar or per-neuron array)
    t_ref: float = 2.0  # ms
    dt: float = 1.0  # ms
    r_m: float = 1.0  # MOhm
    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            tier="computational_hypothesis",
            source="LIF_Standard_Biophysics",
            confidence=0.85,
            rationale="Point-neuron LIF model with exponential membrane and synaptic decay, supporting delay queues",
        )
    )

    @property
    def alpha_m(self) -> float:
        """Membrane potential decay factor per timestep."""
        return math.exp(-self.dt / self.tau_m)

    @property
    def refractory_steps(self) -> int:
        """Number of discrete steps for refractory period."""
        return int(round(self.t_ref / self.dt))

    def get_alpha_syn(self, num_neurons: int) -> np.ndarray:
        """Return array of per-neuron synaptic decay factors."""
        if isinstance(self.tau_syn, np.ndarray):
            assert len(self.tau_syn) == num_neurons
            return np.exp(-self.dt / np.maximum(self.tau_syn, 0.1))
        return np.full(num_neurons, math.exp(-self.dt / self.tau_syn), dtype=np.float64)


@dataclass
class LIFState:
    """Dynamic state variables for a simulated population of neurons."""

    num_neurons: int
    v: np.ndarray  # float64: membrane potential (mV)
    i_syn: np.ndarray  # float64: synaptic input current (nA)
    refractory_timer: np.ndarray  # int32: remaining refractory steps
    delay_buffer: np.ndarray  # float64 shape (MAX_DELAY_STEPS, num_neurons)
    ring_idx: int = 0

    @classmethod
    def initialize(cls, num_neurons: int, params: LIFParameters) -> LIFState:
        return cls(
            num_neurons=num_neurons,
            v=np.full(num_neurons, params.v_rest, dtype=np.float64),
            i_syn=np.zeros(num_neurons, dtype=np.float64),
            refractory_timer=np.zeros(num_neurons, dtype=np.int32),
            delay_buffer=np.zeros((MAX_DELAY_STEPS, num_neurons), dtype=np.float64),
            ring_idx=0,
        )

    def copy(self) -> LIFState:
        return LIFState(
            num_neurons=self.num_neurons,
            v=self.v.copy(),
            i_syn=self.i_syn.copy(),
            refractory_timer=self.refractory_timer.copy(),
            delay_buffer=self.delay_buffer.copy(),
            ring_idx=self.ring_idx,
        )


class LIFReferenceEngine:
    """Golden reference simulator implemented purely in NumPy with float64 precision."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        params: Optional[LIFParameters] = None,
        edge_delays: Optional[np.ndarray] = None,
    ):
        self.graph = graph
        self.params = params or LIFParameters()
        self.edge_delays = (
            np.ascontiguousarray(edge_delays, dtype=np.uint32)
            if edge_delays is not None
            else None
        )
        self.alpha_syn = self.params.get_alpha_syn(graph.num_neurons)
        self.state = LIFState.initialize(graph.num_neurons, self.params)
        self.step_count: int = 0

    def step(self, external_current: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute one simulation step of duration dt."""
        p = self.params
        s = self.state
        n = self.graph.num_neurons

        # 0. Drain delayed current scheduled for this step
        curr_slot = s.ring_idx
        s.i_syn += s.delay_buffer[curr_slot]
        s.delay_buffer[curr_slot].fill(0.0)

        # 1. Decay synaptic currents
        s.i_syn *= self.alpha_syn

        # 2. Add external current to synaptic input buffer
        if external_current is not None:
            s.i_syn += external_current

        # 3. Identify non-refractory neurons
        non_refractory = s.refractory_timer <= 0
        refractory = ~non_refractory

        # 4. Integrate membrane potential for non-refractory neurons
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
            for pre_idx in spike_indices:
                start = self.graph.row_ptr[pre_idx]
                end = self.graph.row_ptr[pre_idx + 1]
                if end > start:
                    targets = self.graph.col_idx[start:end]
                    weights = self.graph.weights[start:end]
                    if self.edge_delays is None:
                        np.add.at(s.i_syn, targets, weights)
                    else:
                        delays = self.edge_delays[start:end]
                        for tgt, w, d in zip(targets, weights, delays):
                            if d == 0:
                                s.i_syn[tgt] += w
                            else:
                                slot = (curr_slot + int(d)) % MAX_DELAY_STEPS
                                s.delay_buffer[slot, tgt] += w

        s.ring_idx = (curr_slot + 1) % MAX_DELAY_STEPS
        self.step_count += 1
        return spike_indices

    def run(
        self,
        num_steps: int,
        stimulus_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        v_trace: List[np.ndarray] = []
        spike_trace: List[np.ndarray] = []

        for t in range(num_steps):
            ext_i = stimulus_fn(t) if stimulus_fn else None
            spikes = self.step(ext_i)
            v_trace.append(self.state.v.copy())
            spike_trace.append(spikes)

        return v_trace, spike_trace
