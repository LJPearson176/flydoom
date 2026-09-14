"""Comprehensive verification tests for Connectome-Constrained Reservoir Computing."""

import numpy as np
import pytest

from fly_doom.control.controllers import ControlledT4Controller, DoorSeekingController
from fly_doom.doom.interface import DoomAction, DoomObservation
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType
from fly_doom.reservoir.datasets import (
    generate_door_stimulus_dataset,
    generate_enemy_stimulus_dataset,
    generate_threat_stimulus_dataset,
)
from fly_doom.reservoir.decoders import (
    DoorThresholdDecoder,
    EnemyCategorizationDecoder,
    VisceralThreatDecoder,
)
from fly_doom.reservoir.pipeline import ReservoirReadoutPipeline
from fly_doom.reservoir.reservoir import (
    ConnectomeReservoir,
    ReservoirConfig,
    create_canonical_doom_reservoir_graph,
)


def test_reservoir_feature_extraction_and_binning():
    """Verify temporal binning and Anscombe variance stabilization over frozen dynamics."""
    graph, features = create_canonical_doom_reservoir_graph(num_retinal_inputs=64, seed=42)
    config = ReservoirConfig(bins=4, duration_ms=28.57, use_anscombe=True)
    reservoir = ConnectomeReservoir(graph=graph, feature_indices=features, config=config)

    # Feature dimension must be exactly bins * num_features
    expected_dim = config.bins * features.total_features
    assert reservoir.feature_dim == expected_dim

    # Test with random sensory current
    rng = np.random.default_rng(123)
    samples = rng.uniform(0.0, 1.0, 64).astype(np.float32)
    feat_vec = reservoir.encode_samples(samples)

    assert feat_vec.shape == (expected_dim,)
    assert np.all(np.isfinite(feat_vec)), "Features must be finite"
    assert np.all(feat_vec >= 0.0), "Anscombe features must be non-negative"


def test_enemy_categorization_decoder():
    """Verify 4-way enemy discrimination on frozen connectome features."""
    pipeline = ReservoirReadoutPipeline()
    metrics = pipeline.train_enemy_decoder(test_split=0.25, seed=42)

    assert metrics["test_accuracy"] >= 0.70, (
        f"Enemy categorization accuracy below 70%: {metrics['test_accuracy'] * 100:.1f}%"
    )

    # Test single frame prediction
    dummy_img = np.full((64, 64, 3), 120, dtype=np.uint8)
    feat = pipeline.reservoir.encode_frame(dummy_img)
    pred = pipeline.enemy_decoder.predict(feat)

    assert pred.class_id in {0, 1, 2, 3}
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4


def test_door_threshold_decoder():
    """Verify door threshold detection on frozen connectome features."""
    pipeline = ReservoirReadoutPipeline()
    metrics = pipeline.train_door_decoder(test_split=0.25, seed=123)

    assert metrics["test_accuracy"] >= 0.85, (
        f"Door threshold accuracy below 85%: {metrics['test_accuracy'] * 100:.1f}%"
    )

    # Positive door test
    door_img, door_labels = generate_door_stimulus_dataset(num_samples=2, seed=999)
    door_feat = pipeline.reservoir.encode_frame(door_img[-1])
    door_pred = pipeline.door_decoder.predict(door_feat)
    assert 0.0 <= door_pred.probability <= 1.0


def test_visceral_threat_decoder():
    """Verify looming threat detection and weapon fire trigger gating."""
    pipeline = ReservoirReadoutPipeline()
    metrics = pipeline.train_threat_decoder(test_split=0.25, seed=456)

    assert metrics["correlation"] >= 0.90, (
        f"Threat arousal correlation below 0.90: {metrics['correlation']:.3f}"
    )

    # High looming threat should trigger should_fire
    threat_imgs, _ = generate_threat_stimulus_dataset(num_samples=10, seed=789)
    high_threat_feat = pipeline.reservoir.encode_frame(threat_imgs[-1])  # Largest looming patch
    threat_pred = pipeline.threat_decoder.predict(high_threat_feat)
    assert threat_pred.arousal > 0.60
    assert threat_pred.should_fire is True

    # Low threat background should not trigger fire
    low_threat_feat = pipeline.reservoir.encode_frame(threat_imgs[0])
    low_pred = pipeline.threat_decoder.predict(low_threat_feat)
    assert low_pred.arousal < 0.40
    assert low_pred.should_fire is False


def test_frozen_connectome_integrity():
    """Verify that connectome weights and graph structure remain 100% frozen."""
    graph, features = create_canonical_doom_reservoir_graph(num_retinal_inputs=64, seed=42)
    weights_before = graph.weights.copy()
    row_ptr_before = graph.row_ptr.copy()
    col_idx_before = graph.col_idx.copy()

    pipeline = ReservoirReadoutPipeline(ConnectomeReservoir(graph=graph, feature_indices=features))
    pipeline.train_enemy_decoder()
    pipeline.train_door_decoder()
    pipeline.train_threat_decoder()

    # Assert bit-level identity
    np.testing.assert_array_equal(graph.weights, weights_before)
    np.testing.assert_array_equal(graph.row_ptr, row_ptr_before)
    np.testing.assert_array_equal(graph.col_idx, col_idx_before)


def test_door_seeking_controller_reservoir_integration():
    """Verify that DoorSeekingController can query reservoir decoders during action selection."""
    pipeline = ReservoirReadoutPipeline()
    pipeline.train_door_decoder()
    pipeline.train_enemy_decoder()
    pipeline.train_threat_decoder()

    base = ControlledT4Controller(model_type=CompartmentModelType.MODEL_A)
    controller = DoorSeekingController(
        base,
        manage_perspective=True,
        reservoir=pipeline.reservoir,
        door_decoder=pipeline.door_decoder,
        enemy_decoder=pipeline.enemy_decoder,
        threat_decoder=pipeline.threat_decoder,
    )

    door_imgs, _ = generate_door_stimulus_dataset(num_samples=2, seed=555)
    rgb = door_imgs[-1]  # Door frame

    obs = DoomObservation(
        rgb=rgb,
        depth=np.zeros((64, 64), dtype=np.float32),
        health=100.0,
        ammo=50,
        kill_count=0,
        x=1520.0,
        y=-2496.0,
        angle_rad=0.0,
        info={
            "native_game_state_available": True,
            "native_game_state": {
                "x": 1520.0,
                "y": -2496.0,
                "z": 0.0,
                "angle_deg": 0.0,
                "pitch_deg": 0.0,
                "speed": 0.0,
                "health": 100.0,
                "ammo": 50,
                "kills": 0,
            },
        },
    )

    action = controller.select_action(obs)
    state = controller.get_neural_state()

    assert "reservoir_door_prob" in state
    assert "reservoir_enemy_class" in state
    assert "reservoir_threat_arousal" in state
    assert 0.0 <= state["reservoir_door_prob"] <= 1.0
    assert state["reservoir_enemy_class"] in {0.0, 1.0, 2.0, 3.0}
    assert 0.0 <= state["reservoir_threat_arousal"] <= 1.0
