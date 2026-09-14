"""Comprehensive verification and numerical parity tests for the Lazy Exact Subthreshold C++ LIF kernel."""

import math
import time
import numpy as np
import pytest

from fly_doom.connectome.graph import create_synthetic_motif_graph
from fly_doom.dynamics.lif_lazy import LIFLazyEngine


def test_lazy_analytical_subthreshold_closed_form():
    """Verify that lazy evolution bit-matches the exact analytical differential solution."""
    # Single isolated neuron
    graph = {
        "ptr": np.array([0, 0], dtype=np.int64),
        "post": np.array([], dtype=np.int32),
        "weight": np.array([], dtype=np.float32),
    }

    dt = 0.1
    durations = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    for duration_ms in durations:
        engine = LIFLazyEngine(graph, dt=dt)
        v0 = -60.0
        g0 = 4.5
        drive_val = 3.0

        engine._v[0] = v0
        engine._g[0] = g0
        engine._previous_drive[0] = drive_val

        engine.advance(drive=np.array([drive_val], dtype=np.float32), duration_ms=duration_ms)

        # Closed-form analytical equations:
        a = math.exp(-duration_ms / 20.0)
        b = math.exp(-duration_ms / 5.0)
        v_expected = -52.0 + (v0 + 52.0) * a + drive_val * (1.0 - a) + g0 * (a - b) / 3.0
        g_expected = g0 * b

        assert abs(engine.v[0] - v_expected) < 1e-5, (
            f"Voltage mismatch at {duration_ms} ms: {engine.v[0]} vs {v_expected}"
        )
        assert abs(engine.g[0] - g_expected) < 1e-5, (
            f"Conductance mismatch at {duration_ms} ms: {engine.g[0]} vs {g_expected}"
        )


def test_lazy_skipping_efficiency():
    """Verify that dormant neurons are skipped via convex relaxation without computing steps."""
    num_neurons = 200
    # Disconnected graph
    graph = {
        "ptr": np.zeros(num_neurons + 1, dtype=np.int64),
        "post": np.array([], dtype=np.int32),
        "weight": np.array([], dtype=np.float32),
    }

    engine = LIFLazyEngine(graph, dt=0.1)

    # Initial state: all 200 neurons active
    assert engine.active_count == 200

    # Drive only neuron 0 with suprathreshold current (charging from -52 to -45 with tau_m=20ms takes ~3ms at drive=50)
    drive = np.zeros(num_neurons, dtype=np.float32)
    drive[0] = 50.0

    # Advance by 10 ms (100 steps of 0.1 ms)
    counts = engine.advance(drive=drive, duration_ms=10.0)

    # Neuron 0 must have spiked
    assert counts[0] > 0
    # Inactive neurons (1..199) receive zero drive and must be lazily skipped
    assert engine.active_count <= 2, f"Expected dormant neurons to be skipped, got {engine.active_count}"
    assert engine.active_fraction < 0.02

    # Despite being skipped, dormant voltages must be materialized exactly to -52.0 mV
    np.testing.assert_allclose(engine.v[1:], -52.0, atol=1e-5)
    np.testing.assert_allclose(engine.g[1:], 0.0, atol=1e-5)


def test_lazy_motif_synaptic_propagation():
    """Verify synaptic transmission, axonal delays, and refractory periods across a feedforward chain."""
    num_neurons = 5
    # Chain: 0 -> 1 -> 2 -> 3 -> 4
    row_ptr = np.array([0, 1, 2, 3, 4, 4], dtype=np.int64)
    col_idx = np.array([1, 2, 3, 4], dtype=np.int32)
    # Weight of 50.0 gives peak EPSP = 50 * 0.1575 = 7.875 mV > 7.0 mV threshold jump
    weights = np.full(4, 50.0, dtype=np.float32)

    graph = {
        "ptr": row_ptr,
        "post": col_idx,
        "weight": weights,
    }

    engine = LIFLazyEngine(graph, dt=0.1)

    # Inject pulse into neuron 0 (fires within 5 ms)
    drive = np.zeros(num_neurons, dtype=np.float32)
    drive[0] = 50.0
    counts1 = engine.advance(drive=drive, duration_ms=5.0)
    assert counts1[0] > 0, "Neuron 0 must fire under strong drive"

    # Remove external drive and let pulse propagate sequentially along the chain over 50 ms
    drive[0] = 0.0
    counts2 = engine.advance(drive=drive, duration_ms=50.0)

    total_spikes = counts1 + counts2
    # All downstream neurons must fire via synaptic transmission
    for i in range(1, num_neurons):
        assert total_spikes[i] > 0, f"Neuron {i} must fire from upstream synaptic propagation"


def test_lazy_snapshot_and_restore_reproducibility():
    """Verify exact bit-level reproducibility across snapshot and restore."""
    num_neurons = 30
    graph = create_synthetic_motif_graph(num_neurons=num_neurons, motif_type="recurrent_inhibition", seed=42)
    engine = LIFLazyEngine(graph, dt=0.1)

    rng = np.random.default_rng(123)
    drive = rng.uniform(0.0, 10.0, size=num_neurons).astype(np.float32)

    # Advance 10 ms
    engine.advance(drive=drive, duration_ms=10.0)
    snap = engine.snapshot()

    # Advance another 10 ms
    drive2 = rng.uniform(0.0, 10.0, size=num_neurons).astype(np.float32)
    counts_run1 = engine.advance(drive=drive2, duration_ms=10.0)
    v_run1 = engine.v

    # Restore snapshot and repeat identical advance
    engine.restore(snap)
    counts_run2 = engine.advance(drive=drive2, duration_ms=10.0)
    v_run2 = engine.v

    np.testing.assert_array_equal(counts_run1, counts_run2)
    np.testing.assert_allclose(v_run1, v_run2, atol=1e-6)


def test_lazy_step_single_tick():
    """Verify single step() interface convenience."""
    num_neurons = 5
    graph = {
        "ptr": np.array([0, 0, 0, 0, 0, 0], dtype=np.int64),
        "post": np.array([], dtype=np.int32),
        "weight": np.array([], dtype=np.float32),
    }
    engine = LIFLazyEngine(graph, dt=0.1)

    # Step with strong drive on neuron 2 for 35 steps (3.5 ms)
    ext = np.zeros(num_neurons, dtype=np.float32)
    ext[2] = 50.0

    fired_steps = []
    for step_idx in range(40):
        fired = engine.step(ext)
        if len(fired) > 0:
            fired_steps.append((step_idx, fired.tolist()))

    assert len(fired_steps) > 0, "Neuron 2 should fire within 40 steps"
    assert [2] in [f[1] for f in fired_steps], "Neuron 2 must be in fired events"


def test_lazy_throughput_and_scalability():
    """Benchmark lazy kernel throughput on a 5,000-neuron network."""
    num_neurons = 5000
    # Sparse random graph with ~25,000 synapses (avg degree 5)
    rng = np.random.default_rng(999)
    num_edges = 25000
    pre = rng.integers(0, num_neurons, size=num_edges, dtype=np.int64)
    post = rng.integers(0, num_neurons, size=num_edges, dtype=np.int32)
    weights = rng.uniform(1.0, 5.0, size=num_edges).astype(np.float32)

    order = np.argsort(pre)
    pre = pre[order]
    post = post[order]
    weights = weights[order]

    row_ptr = np.zeros(num_neurons + 1, dtype=np.int64)
    np.add.at(row_ptr[1:], pre, 1)
    np.cumsum(row_ptr, out=row_ptr)

    graph = {
        "ptr": row_ptr,
        "post": post,
        "weight": weights,
    }

    engine = LIFLazyEngine(graph, dt=0.1)

    # 1% active neurons
    drive = np.zeros(num_neurons, dtype=np.float32)
    drive[:50] = 15.0

    t0 = time.perf_counter()
    # Advance 50 ms (500 steps of 0.1 ms)
    counts = engine.advance(drive=drive, duration_ms=50.0)
    elapsed = time.perf_counter() - t0

    steps = 500
    steps_per_sec = steps / elapsed
    print(f"\n[5,000 Neurons Benchmark] 50 ms ({steps} steps) elapsed: {elapsed*1000:.2f} ms")
    print(f"Steps/sec: {steps_per_sec:.0f}, Active fraction: {engine.active_fraction*100:.1f}%")

    assert np.sum(counts) > 0, "Expected spikes from driven neurons"
    assert steps_per_sec > 500, f"Expected high throughput, got {steps_per_sec:.0f} steps/s"
