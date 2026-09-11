"""Tests for retinal encoders: Alpha (planar raster) and Delta (hexagonal ommatidia)."""

import numpy as np
import pytest
from fly_doom.sensory.encoders.alpha import EncoderAlpha
from fly_doom.sensory.encoders.delta import EncoderDelta


def test_encoder_alpha():
    encoder = EncoderAlpha(grid_size=8, gain=10.0)
    assert encoder.num_input_units == 64
    assert encoder.provenance.tier == "engineering_scaffold"

    frame = np.ones((32, 32), dtype=np.float64) * 0.5
    currents = encoder.encode_frame(frame, dt_ms=1.0)

    assert currents.shape == (64,)
    assert np.allclose(currents, 5.0)


def test_encoder_delta_temporal_differencing():
    encoder = EncoderDelta(num_columns=4, num_rows=4, field_width_px=32, field_height_px=32)
    assert encoder.num_input_units == 32  # 16 ommatidia * 2 (ON + OFF channels)
    assert encoder.provenance.tier == "experimental_assumption"

    # Frame 1: baseline dark
    f1 = np.full((32, 32), 0.1, dtype=np.float64)
    c1 = encoder.encode_frame(f1, dt_ms=1.0)
    # First frame has zero delta
    assert np.allclose(c1, 0.0)

    # Frame 2: sudden brightness increment (ON transition)
    f2 = np.full((32, 32), 0.9, dtype=np.float64)
    c2 = encoder.encode_frame(f2, dt_ms=1.0)

    on_channel = c2[:16]
    off_channel = c2[16:]

    # ON channel should be strongly activated, OFF channel should be 0
    assert np.all(on_channel > 0.0)
    assert np.all(off_channel == 0.0)

    # Frame 3: sudden brightness decrement (OFF transition)
    f3 = np.full((32, 32), 0.2, dtype=np.float64)
    c3 = encoder.encode_frame(f3, dt_ms=1.0)

    on_channel_3 = c3[:16]
    off_channel_3 = c3[16:]

    # ON channel should be 0, OFF channel should be strongly activated
    assert np.all(on_channel_3 == 0.0)
    assert np.all(off_channel_3 > 0.0)
