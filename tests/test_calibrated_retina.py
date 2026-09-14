"""Tests for Calibrated Retinal Mapping (100% Visual Coverage) and EncoderCalibratedRetina."""

from pathlib import Path
import json
import numpy as np
import pytest

from fly_doom.sensory.encoders.calibrated_retina import (
    EncoderCalibratedRetina,
    balanced_retina,
    sample_retina,
    sampling_support,
)
from fly_doom.dynamics.fan_shaped_body import SEZNociceptiveReflex


def test_balanced_retina_synthetic_grid():
    # 100 points with 25 unique sites -> should map to 5x5 grid
    np.random.seed(42)
    unique_sites = np.random.uniform(0.1, 0.9, size=(25, 2)).astype(np.float32)
    repeats = np.random.choice(25, size=100)
    uv = unique_sites[repeats]

    calib_uv, meta = balanced_retina(uv)
    assert calib_uv.shape == (100, 2)
    assert meta["unique_sites"] == 25
    assert meta["grid_width"] == 5
    assert meta["grid_height"] == 5
    assert 0.0 <= calib_uv.min() and calib_uv.max() <= 1.0

    unique_calib = np.unique(calib_uv, axis=0)
    assert len(unique_calib) == 25


def test_coverage_diagnosis():
    atlas_path = Path("assets/retinal_atlas.json")
    calib_path = Path("assets/retina_calibrated_uv.npy")
    assert atlas_path.exists()
    assert calib_path.exists()

    with open(atlas_path) as f:
        atlas = json.load(f)
    orig_uv = np.array(atlas["original_uv"], dtype=np.float32)
    calib_uv = np.load(calib_path)

    # 48x48 pixel support
    support_orig = sampling_support(orig_uv, size=48)
    support_calib = sampling_support(calib_uv, size=48)

    orig_coverage = float(support_orig.mean())
    calib_coverage = float(support_calib.mean())

    # Original had ~52% coverage; calibrated achieves 100%
    assert orig_coverage < 0.60
    assert calib_coverage == pytest.approx(1.0, abs=1e-5)

    # 64x64 quadrant coverage comparison
    support_orig_64 = sampling_support(orig_uv, size=64)
    support_calib_64 = sampling_support(calib_uv, size=64)

    # Bottom-left quadrant: [32:, :32]
    bl_orig = float(support_orig_64[32:, :32].mean())
    bl_calib = float(support_calib_64[32:, :32].mean())

    # Original had massive blind spot (~10% support), calibrated has > 70% support
    assert bl_orig < 0.15
    assert bl_calib > 0.70


def test_encoder_calibrated_retina_subfields():
    encoder = EncoderCalibratedRetina()

    assert encoder.num_input_units == 3335
    assert encoder.num_receptors == 3335
    assert encoder.unique_sites == 825
    assert encoder.provenance.tier == "source_derived_fixture"

    # Verify anatomical subfield masks exist and partition space sensibly
    assert len(encoder.ventral_indices) > 0
    assert len(encoder.central_indices) > 0
    assert len(encoder.left_eye_indices) == 1107
    assert len(encoder.right_eye_indices) == 2228
    assert len(encoder.left_eye_indices) + len(encoder.right_eye_indices) == 3335


def test_encoder_temporal_difference_mode():
    encoder = EncoderCalibratedRetina(mode="on_off_temporal")
    assert encoder.num_input_units == 3335 * 2  # ON and OFF channels

    frame1 = np.full((64, 64), 0.2, dtype=np.float32)
    frame2 = np.full((64, 64), 0.8, dtype=np.float32)

    # First frame baseline (zero delta)
    out1 = encoder.encode_frame(frame1, dt_ms=16.67)
    assert out1.shape == (6670,)
    assert np.allclose(out1, 0.0)

    # Second frame with positive luminance step: ON channel fires, OFF channel silent
    out2 = encoder.encode_frame(frame2, dt_ms=16.67)
    on_current = out2[:3335]
    off_current = out2[3335:]
    assert np.all(on_current > 0.0)
    assert np.allclose(off_current, 0.0)


def test_nukage_hazard_detection():
    encoder = EncoderCalibratedRetina()

    # Neutral gray frame
    neutral_frame = np.full((64, 64, 3), 100, dtype=np.uint8)
    detected, yaw, frac = encoder.evaluate_nukage_hazard(neutral_frame)
    assert not detected
    assert yaw == 0.0

    # Toxic green nukage floor: bottom rows with bright green (R=20, G=220, B=20)
    toxic_frame = neutral_frame.copy()
    toxic_frame[48:, :, :] = [20, 220, 20]
    detected, yaw, frac = encoder.evaluate_nukage_hazard(toxic_frame)
    assert detected
    assert frac > 0.18

    # Left-biased toxic pool: turns right (yaw = -1.0)
    left_toxic = neutral_frame.copy()
    left_toxic[48:, :32, :] = [20, 220, 20]
    detected_l, yaw_l, _ = encoder.evaluate_nukage_hazard(left_toxic)
    assert detected_l
    assert yaw_l == -1.0

    # Right-biased toxic pool: turns left (yaw = +1.0)
    right_toxic = neutral_frame.copy()
    right_toxic[48:, 32:, :] = [20, 220, 20]
    detected_r, yaw_r, _ = encoder.evaluate_nukage_hazard(right_toxic)
    assert detected_r
    assert yaw_r == 1.0


def test_door_candidate_detection():
    encoder = EncoderCalibratedRetina()

    # Uniform wall: no door structure
    wall = np.full((64, 64, 3), 120, dtype=np.uint8)
    assert not encoder.detect_door_candidate(wall)

    # High contrast door texture with vertical bars in center
    door = wall.copy()
    pattern = np.zeros((32, 24, 3), dtype=np.uint8)
    pattern[:, ::2, :] = 30
    pattern[:, 1::2, :] = 220
    door[16:48, 20:44, :] = pattern
    assert encoder.detect_door_candidate(door)


def test_sez_reflex_integration():
    reflex = SEZNociceptiveReflex(use_calibrated_retina=True)

    # Toxic green floor
    frame = np.full((64, 64, 3), 100, dtype=np.uint8)
    frame[48:, :, :] = [20, 220, 20]

    acid_detected, rep_bias, acid_frac = reflex.evaluate_retinal_hazard(frame)
    assert acid_detected
    assert reflex.last_acid_detected
    assert reflex._calibrated_encoder is not None
