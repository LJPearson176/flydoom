"""Tests for directional vector tuning, DSI, circular variance, and polarity."""

import math
import pytest
from fly_doom.analysis.tuning import (
    DirectionalTuningResult,
    PopulationTuningSummary,
    compute_vector_tuning,
)


def test_perfect_directional_tuning():
    directions = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    # Pure response at 0° (Rightwards), zero everywhere else
    responses = [100.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]

    res = compute_vector_tuning(
        neuron_idx=1,
        cell_type="T4a",
        directions_deg=directions,
        responses=responses,
        on_response=120.0,
        off_response=10.0,
    )

    # Discrete preferred direction is exactly 0°
    assert res.discrete_pref_direction_deg == 0.0
    assert res.discrete_null_direction_deg == 180.0
    assert res.dsi > 0.9

    # Vector tuning is highly reliable
    assert res.vector_pd_reliable is True
    assert res.preferred_direction_deg is not None
    assert abs(res.preferred_direction_deg - 0.0) < 5.0 or abs(res.preferred_direction_deg - 360.0) < 5.0
    assert res.best_cardinal_direction_deg == 0.0
    assert res.cardinal_error_deg < 5.0
    assert res.vector_strength > 0.5

    # Polarity index: (120 - 10) / (120 + 10) ~ 0.846 (strong ON preference)
    assert res.polarity_index is not None and res.polarity_index > 0.8


def test_isotropic_unresponsive_tuning():
    directions = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    # Completely equal responses in all directions
    responses = [20.0] * 8

    res = compute_vector_tuning(
        neuron_idx=2,
        cell_type="NonDirectional",
        directions_deg=directions,
        responses=responses,
    )

    # Vector magnitude & strength should be near zero
    assert res.vector_magnitude < 1e-4
    assert res.vector_strength < 1e-4
    # Preferred direction must be marked unreliable (None), avoiding false precision
    assert res.vector_pd_reliable is False
    assert res.preferred_direction_deg is None
    # Discrete DSI should be ~0
    assert abs(res.dsi) < 1e-4
    # Circular variance should be 1.0
    assert abs(res.circular_variance - 1.0) < 1e-4


def test_decoupled_discrete_dsi_stability():
    """Verify that a near-isotropic baseline with a clear maximum maintains discrete DSI."""
    directions = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    # Response curve from gamma=0.25 diagnostic
    curve = [50.0, 117.0, 57.0, 80.0, 101.0, 46.0, 106.0, 81.0]

    res = compute_vector_tuning(3, "T4a", directions, curve)
    # Discrete pref is index 1 (45°: 117 spikes), null is index 5 (225°: 46 spikes)
    assert res.discrete_pref_direction_deg == 45.0
    assert res.discrete_null_direction_deg == 225.0
    expected_dsi = (117.0 - 46.0) / (117.0 + 46.0)
    assert abs(res.dsi - expected_dsi) < 1e-4
    # Vector strength is low, so PD is marked unreliable
    assert res.vector_strength < 0.10
    assert res.vector_pd_reliable is False
    assert res.preferred_direction_deg is None


def test_population_tuning_summary():
    directions = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    r1 = compute_vector_tuning(1, "T4a", directions, [100, 50, 0, 0, 0, 0, 0, 50], 100, 10)
    r2 = compute_vector_tuning(2, "T4b", directions, [0, 0, 0, 50, 100, 50, 0, 0], 90, 15)

    summary = PopulationTuningSummary.from_results("T4_Population", [r1, r2])

    assert summary.num_neurons == 2
    assert summary.mean_dsi > 0.9
    assert summary.mean_vector_strength > 0.5
    assert summary.fraction_pd_reliable == 1.0
    assert summary.mean_polarity_index is not None and summary.mean_polarity_index > 0.7
