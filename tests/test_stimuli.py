"""Tests for visual stimuli generator and motion integrity."""

import numpy as np
import pytest
from fly_doom.sensory.stimuli import (
    CARDINAL_DIRECTIONS,
    StimulusGenerator,
    StimulusParameters,
    StimulusSweep,
    StimulusType,
)


def test_stimulus_parameter_hashing():
    p1 = StimulusParameters(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        direction_deg=0.0,
        spatial_period_px=32.0,
        temporal_freq_hz=2.0,
    )
    p2 = StimulusParameters(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        direction_deg=0.0,
        spatial_period_px=32.0,
        temporal_freq_hz=2.0,
    )
    p3 = StimulusParameters(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        direction_deg=90.0,
        spatial_period_px=32.0,
        temporal_freq_hz=2.0,
    )

    assert p1.compute_hash() == p2.compute_hash()
    assert p1.compute_hash() != p3.compute_hash()


def test_drifting_grating_motion():
    params = StimulusParameters(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        direction_deg=0.0,
        width=32,
        height=32,
        duration_ms=250.0,
    )
    gen = StimulusGenerator(params)

    f0 = gen.get_frame(0.0)
    f50 = gen.get_frame(50.0)

    assert f0.shape == (32, 32)
    assert np.all(f0 >= 0.0) and np.all(f0 <= 1.0)
    # Frames must not be identical across time
    assert not np.array_equal(f0, f50)


def test_moving_edges_on_off():
    params_on = StimulusParameters(
        stimulus_type=StimulusType.MOVING_EDGE_ON,
        direction_deg=0.0,
        width=32,
        height=32,
        velocity_px_s=64.0,
    )
    params_off = StimulusParameters(
        stimulus_type=StimulusType.MOVING_EDGE_OFF,
        direction_deg=0.0,
        width=32,
        height=32,
        velocity_px_s=64.0,
    )

    gen_on = StimulusGenerator(params_on)
    gen_off = StimulusGenerator(params_off)

    f_on = gen_on.get_frame(100.0)
    f_off = gen_off.get_frame(100.0)

    # ON edge: leading area is dark, trailing is bright
    # OFF edge: leading area is bright, trailing is dark
    assert np.mean(f_on) != np.mean(f_off) or not np.allclose(f_on, f_off)


def test_sweep_integrity_verification():
    sweep = StimulusSweep(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        directions=CARDINAL_DIRECTIONS,
        width=32,
        height=32,
    )

    passed, msg = sweep.verify_motion_integrity()
    assert passed is True
    fp = sweep.compute_sweep_fingerprint()
    assert len(fp) == 64
