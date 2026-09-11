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

    # Preferred direction should be very close to 0°
    assert abs(res.preferred_direction_deg - 0.0) < 5.0 or abs(res.preferred_direction_deg - 360.0) < 5.0
    assert res.best_cardinal_direction_deg == 0.0
    assert res.cardinal_error_deg < 5.0

    # High DSI: R(0)=100, R(180)=0 -> DSI ~ 1.0
    assert res.dsi > 0.9

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

    # Vector magnitude should be near zero
    assert res.vector_magnitude < 1e-4
    # DSI should be ~0
    assert abs(res.dsi) < 1e-4
    # Circular variance should be 1.0
    assert abs(res.circular_variance - 1.0) < 1e-4


def test_population_tuning_summary():
    directions = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    r1 = compute_vector_tuning(1, "T4a", directions, [100, 50, 0, 0, 0, 0, 0, 50], 100, 10)
    r2 = compute_vector_tuning(2, "T4b", directions, [0, 0, 0, 50, 100, 50, 0, 0], 90, 15)

    summary = PopulationTuningSummary.from_results("T4_Population", [r1, r2])

    assert summary.num_neurons == 2
    assert summary.mean_dsi > 0.9
    assert summary.mean_polarity_index is not None and summary.mean_polarity_index > 0.7
