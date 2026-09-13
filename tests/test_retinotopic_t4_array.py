"""Tests for Retinotopic T4 Array Engine and 2D Optic Flow Decomposition."""

import time
import numpy as np
import pytest

from fly_doom.dynamics.compartmental_t4 import CompartmentalParameters
from fly_doom.dynamics.optic_flow import OpticFlowDecomposer, RetinotopicT4ArrayEngine


def test_retinotopic_t4_array_initialization():
    engine = RetinotopicT4ArrayEngine(rows=8, cols=8)
    assert engine.v_soma.shape == (4, 8, 8)
    assert np.allclose(engine.v_soma, -65.0)
    assert engine.provenance.tier == "biological_reconstruction"


def test_retinotopic_t4_directional_selectivity():
    """Verify that rightward-moving bar preferentially activates T4a over T4b."""
    params = CompartmentalParameters(dt=16.67, tm3_delay_ms=16.67)
    engine = RetinotopicT4ArrayEngine(rows=8, cols=8, params=params)

    # Simulate rightward-moving light bar across columns 0 -> 7
    u_sums = []
    for step in range(12):
        col = min(7, step // 2)
        frame = np.zeros((8, 8), dtype=np.float64)
        frame[:, col] = 10.0  # Bright bar at column
        u_field, v_field, spikes = engine.step(frame)
        u_sums.append(float(np.mean(u_field)))

    # Net horizontal motion over the sequence should be positive (rightward)
    total_u = sum(u_sums)
    assert total_u > 0.0, f"Expected positive rightward motion, got {total_u}"


def test_retinotopic_t4_leftward_motion():
    """Verify that leftward-moving bar activates T4b over T4a (negative u)."""
    params = CompartmentalParameters(dt=16.67, tm3_delay_ms=16.67)
    engine = RetinotopicT4ArrayEngine(rows=8, cols=8, params=params)

    # Simulate leftward-moving light bar across columns 7 -> 0
    u_sums = []
    for step in range(12):
        col = max(0, 7 - (step // 2))
        frame = np.zeros((8, 8), dtype=np.float64)
        frame[:, col] = 10.0
        u_field, v_field, spikes = engine.step(frame)
        u_sums.append(float(np.mean(u_field)))

    total_u = sum(u_sums)
    assert total_u < 0.0, f"Expected negative leftward motion, got {total_u}"


def test_optic_flow_decomposer_looming():
    """Expanding vector field must produce positive divergence and looming index."""
    decomposer = OpticFlowDecomposer(rows=8, cols=8)

    # Synthetic centrifugal expansion from center
    y_coords = np.linspace(-1.0, 1.0, 8)
    x_coords = np.linspace(-1.0, 1.0, 8)
    xx, yy = np.meshgrid(x_coords, y_coords)

    u_field = xx * 2.0  # Expanding horizontally
    v_field = yy * 2.0  # Expanding vertically

    metrics = decomposer.decompose(u_field, v_field)
    assert metrics.divergence > 0.5, f"Expected high divergence, got {metrics.divergence}"
    assert metrics.looming_index > 0.5, f"Expected high looming index, got {metrics.looming_index}"
    assert abs(metrics.curl) < 0.2, f"Expected near-zero curl for expansion, got {metrics.curl}"


def test_optic_flow_decomposer_rotation():
    """Pure rotational flow field must produce high curl and symmetric LPTC-HS divergence."""
    decomposer = OpticFlowDecomposer(rows=8, cols=8)

    y_coords = np.linspace(-1.0, 1.0, 8)
    x_coords = np.linspace(-1.0, 1.0, 8)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Clockwise circulation: u = -yy, v = xx
    u_field = -yy * 3.0
    v_field = xx * 3.0

    metrics = decomposer.decompose(u_field, v_field)
    assert abs(metrics.curl) > 1.0, f"Expected high curl, got {metrics.curl}"
    assert abs(metrics.divergence) < 0.2, f"Expected low divergence for pure curl, got {metrics.divergence}"


def test_retinotopic_t4_array_execution_speed():
    """Ensure population simulation of 128 cartridges executes at > 500 Hz (< 2 ms/tick)."""
    engine = RetinotopicT4ArrayEngine(rows=8, cols=8)
    decomposer = OpticFlowDecomposer(rows=8, cols=8)
    rng = np.random.RandomState(42)

    n_frames = 100
    frames = [rng.uniform(0.0, 5.0, (8, 8)) for _ in range(n_frames)]

    t0 = time.perf_counter()
    for f in frames:
        u, v, _ = engine.step(f)
        _ = decomposer.decompose(u, v)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    ms_per_frame = elapsed_ms / n_frames
    assert ms_per_frame < 2.0, f"Array step too slow: {ms_per_frame:.3f} ms/frame (budget: < 2.0 ms)"
