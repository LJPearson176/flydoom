"""Numerical equivalence verification gate: C++ native vs Python/NumPy reference LIF engine."""

import numpy as np
import pytest
from fly_doom.connectome.graph import create_synthetic_motif_graph
from fly_doom.dynamics.lif_native import LIFNativeEngine
from fly_doom.dynamics.lif_reference import LIFParameters, LIFReferenceEngine


@pytest.mark.parametrize("motif", ["recurrent_inhibition", "feedforward_chain"])
def test_lif_10000_step_numerical_equivalence(motif: str):
    """Primary Phase 0 Verification Gate.

    Asserts:
      1. max_abs_diff(V_cpp, V_numpy) < 1e-9 across 10,000 steps
      2. spike_events_cpp == spike_events_numpy bit-for-bit index identical
    """
    num_neurons = 50
    graph = create_synthetic_motif_graph(num_neurons=num_neurons, motif_type=motif, seed=123)
    params = LIFParameters(dt=1.0, tau_m=20.0, tau_syn=5.0, t_ref=2.0)

    ref_engine = LIFReferenceEngine(graph, params)
    native_engine = LIFNativeEngine(graph, params)

    total_steps = 10000
    max_voltage_diff = 0.0
    spike_mismatches = 0
    total_spikes = 0

    rng = np.random.default_rng(42)

    for t in range(total_steps):
        # Deterministic periodic and Poisson-like stimulus injections
        ext = np.zeros(num_neurons, dtype=np.float64)
        if t % 7 == 0:
            ext[:5] = 30.0
        if t % 23 == 0:
            ext[10:15] = 25.0
        if rng.random() < 0.05:
            rand_target = int(rng.integers(0, num_neurons))
            ext[rand_target] = 40.0

        ref_spikes = ref_engine.step(ext)
        native_spikes = native_engine.step(ext)

        # 1. Check spike sequence equality
        if not np.array_equal(ref_spikes, native_spikes):
            spike_mismatches += 1

        total_spikes += len(ref_spikes)

        # 2. Check membrane potential deviation
        v_diff = np.max(np.abs(ref_engine.state.v - native_engine.v))
        if v_diff > max_voltage_diff:
            max_voltage_diff = v_diff

    print(f"\n[Motif: {motif}] Steps: {total_steps}, Total Spikes: {total_spikes}")
    print(f"Max Voltage Error: {max_voltage_diff:.3e}, Spike Mismatches: {spike_mismatches}")

    # Gates
    assert total_spikes > 100, f"Test did not generate sufficient neural activity ({total_spikes} spikes)"
    assert spike_mismatches == 0, f"Spike event sequences diverged {spike_mismatches} times!"
    assert (
        max_voltage_diff < 1e-9
    ), f"Membrane potential error exceeded numerical tolerance: {max_voltage_diff:.3e} >= 1e-9"
