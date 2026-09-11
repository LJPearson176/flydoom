"""Numerical equivalence verification gate for Model E coincidence nonlinearity."""

import numpy as np
import pytest
from fly_doom.connectome.graph import create_synthetic_motif_graph
from fly_doom.dynamics.lif_native import LIFNativeEngine
from fly_doom.dynamics.lif_reference import LIFParameters, LIFReferenceEngine


def test_nonlinear_coincidence_numerical_parity():
    """Verify C++ vs Python numerical equivalence with Model E coincidence nonlinearity and delay queues."""
    num_neurons = 40
    graph = create_synthetic_motif_graph(num_neurons=num_neurons, motif_type="recurrent_inhibition", seed=88)

    rng = np.random.default_rng(202)
    tau_syn_vec = rng.uniform(5.0, 40.0, size=num_neurons)
    delays = rng.choice([0, 1, 3, 7], size=graph.num_edges).astype(np.uint32)

    # Enable nonlinearity on a subset of neurons
    nonlinear_mask = np.zeros(num_neurons, dtype=np.uint8)
    nonlinear_mask[:10] = 1
    coincidence_gamma = 1.5

    params = LIFParameters(dt=1.0, tau_m=15.0, tau_syn=tau_syn_vec, t_ref=2.0)

    ref_engine = LIFReferenceEngine(
        graph,
        params,
        edge_delays=delays,
        coincidence_gamma=coincidence_gamma,
        nonlinear_mask=nonlinear_mask.astype(bool),
    )
    native_engine = LIFNativeEngine(
        graph,
        params,
        edge_delays=delays,
        coincidence_gamma=coincidence_gamma,
        nonlinear_mask=nonlinear_mask,
    )

    total_steps = 3000
    max_voltage_diff = 0.0
    spike_mismatches = 0
    total_spikes = 0

    for t in range(total_steps):
        ext = np.zeros(num_neurons, dtype=np.float64)
        if t % 6 == 0:
            ext[:4] = 30.0
        if rng.random() < 0.15:
            target = int(rng.integers(0, num_neurons))
            ext[target] = 35.0

        ref_spikes = ref_engine.step(ext)
        native_spikes = native_engine.step(ext)

        total_spikes += len(ref_spikes)
        if not np.array_equal(ref_spikes, native_spikes):
            spike_mismatches += 1

        v_diff = np.max(np.abs(ref_engine.state.v - native_engine.v))
        if v_diff > max_voltage_diff:
            max_voltage_diff = v_diff

    print(f"\nNonlinear Coincidence Parity: Steps={total_steps}, Spikes={total_spikes}")
    print(f"Max Voltage Error: {max_voltage_diff:.3e}, Spike Mismatches: {spike_mismatches}")

    assert total_spikes > 100, "Test did not generate sufficient spiking activity"
    assert spike_mismatches == 0, f"Spike event sequences diverged ({spike_mismatches} mismatches)"
    assert max_voltage_diff < 1e-9, f"Voltage error exceeded numerical tolerance: {max_voltage_diff:.3e}"
