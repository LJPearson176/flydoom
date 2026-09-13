"""Tests for Mushroom Body Kenyon Cell expansion and dopamine-gated plasticity."""

from pathlib import Path
import numpy as np
import pytest

from fly_doom.dynamics.mushroom_body import MushroomBodyParameters, MushroomBodyPlasticityEngine


def test_kenyon_cell_k_wta_sparsity():
    params = MushroomBodyParameters(num_kenyon_cells=64, input_dim=16, sparsity_k=6)
    mb = MushroomBodyPlasticityEngine(params)

    rng = np.random.RandomState(123)
    for _ in range(10):
        feat = rng.normal(0, 1, 16)
        kc = mb.compute_kenyon_cells(feat)
        # Exactly 6 cells must be non-zero
        active_count = int(np.count_nonzero(kc > 0.0))
        assert active_count == 6, f"Expected 6 active KCs, got {active_count}"
        # Sparsity should be (64 - 6) / 64 = 90.6%
        assert np.isclose(np.sum(kc), 1.0)


def test_ppl1_aversive_ltd():
    """Pain/damage (PPL1 aversive burst) must depress MBON-app synapses, causing conditioned avoidance."""
    params = MushroomBodyParameters(num_kenyon_cells=64, input_dim=16, sparsity_k=6, learning_rate_ltd=0.15)
    mb = MushroomBodyPlasticityEngine(params)

    hazard_features = np.ones(16, dtype=np.float64)

    # Pre-training: initial valence should be neutral (~0)
    pre_state = mb.step(hazard_features, aversive_us=0.0)
    assert abs(pre_state.valence) < 0.05, f"Initial valence not neutral: {pre_state.valence}"

    # Conditioning: pair hazard features with strong damage (aversive US = 1.0)
    for _ in range(5):
        mb.step(hazard_features, aversive_us=1.0)

    # Post-training recall without US: valence must shift strongly negative (avoidance)
    post_state = mb.step(hazard_features, aversive_us=0.0)
    assert post_state.valence < -0.12, f"Expected conditioned avoidance, got {post_state.valence}"
    assert post_state.mbon_app_drive < post_state.mbon_av_drive


def test_pam_reward_ltd():
    """Reward (PAM dopamine burst) must depress MBON-av synapses, enhancing approach valence."""
    params = MushroomBodyParameters(num_kenyon_cells=64, input_dim=16, sparsity_k=6, learning_rate_ltd=0.15)
    mb = MushroomBodyPlasticityEngine(params)

    goal_features = np.zeros(16, dtype=np.float64)
    goal_features[::2] = 1.0

    # Conditioning with reward
    for _ in range(5):
        mb.step(goal_features, reward_us=1.0)

    # Recall without US
    post_state = mb.step(goal_features, reward_us=0.0)
    assert post_state.valence > 0.12, f"Expected conditioned approach, got {post_state.valence}"
    assert post_state.mbon_app_drive > post_state.mbon_av_drive


def test_plastic_weight_persistence(tmp_path: Path):
    """Plastic weights can be persisted and restored across sessions."""
    mb1 = MushroomBodyPlasticityEngine()
    features = np.linspace(-1, 1, 16)

    # Apply learning
    for _ in range(3):
        mb1.step(features, aversive_us=0.8)

    save_file = tmp_path / "mb_weights.npz"
    mb1.save_weights(save_file)
    assert save_file.exists()

    val1 = mb1.step(features, aversive_us=0.0).valence

    # Load in new engine
    mb2 = MushroomBodyPlasticityEngine()
    mb2.load_weights(save_file)
    val2 = mb2.step(features, aversive_us=0.0).valence

    assert np.isclose(val1, val2, atol=1e-3)
