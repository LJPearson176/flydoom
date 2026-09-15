"""Unit tests for Heading & Attention Topological Map Renderer."""

import numpy as np
import pytest
from PIL import Image

from fly_doom.vis.heading_attention_map import HeadingAttentionMapRenderer


def test_heading_attention_map_dimensions_and_mode():
    """Verify HeadingAttentionMapRenderer produces exact RGB output dimensions."""
    renderer = HeadingAttentionMapRenderer(width=316, height=270)
    img = renderer.render(eb_heading_deg=15.8, coherence_r=0.67, asymmetry=0.02)

    assert isinstance(img, Image.Image)
    assert img.size == (316, 270)
    assert img.mode == "RGB"


def test_heading_attention_map_non_blank_and_structured():
    """Verify render contains structured topological features and colored hotspots."""
    renderer = HeadingAttentionMapRenderer(width=316, height=270)
    img = renderer.render(eb_heading_deg=15.8, coherence_r=0.67, asymmetry=0.0)

    arr = np.array(img)
    # Check overall brightness is within typical map range
    assert np.mean(arr) > 15.0
    # Check that red/hotspot core exists (high red values)
    assert np.max(arr[:, :, 0]) >= 200
    # Check that cyan/green lines and lobes exist
    assert np.max(arr[:, :, 1]) >= 180


def test_heading_attention_map_angular_modulation():
    """Verify distinct heading angles modulate spatial attention field distribution."""
    renderer = HeadingAttentionMapRenderer(width=316, height=270)

    img_pos = renderer.render(eb_heading_deg=45.0, coherence_r=0.8)
    img_neg = renderer.render(eb_heading_deg=-45.0, coherence_r=0.8)

    arr_pos = np.array(img_pos, dtype=np.float32)
    arr_neg = np.array(img_neg, dtype=np.float32)

    # Difference across fields should be significant
    diff = np.mean(np.abs(arr_pos - arr_neg))
    assert diff > 5.0
