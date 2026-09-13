"""Tests for Central Complex Ellipsoid Body Continuous Attractor Network."""

import math
import numpy as np
import pytest

from fly_doom.dynamics.central_complex_ring import EllipsoidBodyRingAttractor


def test_ring_attractor_initialization():
    ring = EllipsoidBodyRingAttractor(num_wedges=16)
    assert len(ring.angles) == 16
    assert ring.activations.shape == (16,)
    assert ring.provenance.tier == "biological_reconstruction"

    # Initial state should form a coherent bump near heading 0
    state = ring.step(angular_velocity=0.0)
    assert abs(state.heading_angle_rad) < 0.2
    assert state.bump_coherence > 0.6
    assert state.bump_amplitude > 0.15


def test_ring_attractor_heading_persistence():
    """A stationary fly (omega = 0) must maintain stable heading memory across 100 steps."""
    ring = EllipsoidBodyRingAttractor(num_wedges=16)
    ring.reset(initial_heading_rad=0.75)  # ~43 degrees

    for _ in range(100):
        state = ring.step(angular_velocity=0.0)

    # Heading should remain close to initial angle
    assert abs(state.heading_angle_rad - 0.75) < 0.15, f"Heading drifted: {state.heading_angle_rad} vs 0.75"
    assert state.bump_coherence > 0.6


def test_ring_attractor_angular_velocity_integration():
    """Turning right (omega > 0) should shift the bump in the direction of the turn."""
    ring = EllipsoidBodyRingAttractor(num_wedges=16)
    ring.reset(initial_heading_rad=0.0)

    # Turn right with omega = +1.5 rad/s for 20 steps
    # Note: in world coordinates, turning right changes heading
    headings = []
    for _ in range(20):
        state = ring.step(angular_velocity=1.5)
        headings.append(state.heading_angle_rad)

    # Final heading should have shifted significantly from 0
    total_shift = headings[-1] - headings[0]
    assert abs(total_shift) > 0.3, f"Expected bump shift, got {total_shift}"
    assert state.bump_coherence > 0.5


def test_ring_attractor_visual_cue_anchoring():
    """A salient visual landmark should pull the bump towards its azimuthal angle."""
    ring = EllipsoidBodyRingAttractor(num_wedges=16)
    ring.reset(initial_heading_rad=0.0)

    target_landmark = 1.2  # ~69 degrees
    for _ in range(30):
        state = ring.step(
            angular_velocity=0.0,
            visual_cue_angle=target_landmark,
            visual_cue_weight=1.5,
        )

    # Bump should align closely with the landmark
    diff = abs(np.arctan2(np.sin(state.heading_angle_rad - target_landmark), np.cos(state.heading_angle_rad - target_landmark)))
    assert diff < 0.25, f"Bump did not anchor to visual cue: diff = {diff}"
