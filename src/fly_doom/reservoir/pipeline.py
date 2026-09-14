"""Training, evaluation, and serialization pipeline for Connectome-Constrained Decoders."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

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
from fly_doom.reservoir.reservoir import ConnectomeReservoir


class ReservoirReadoutPipeline:
    """End-to-end training and inference pipeline for connectome reservoir decoders."""

    def __init__(self, reservoir: Optional[ConnectomeReservoir] = None):
        self.reservoir = reservoir or ConnectomeReservoir()
        self.enemy_decoder = EnemyCategorizationDecoder(l2_reg=1.0)
        self.door_decoder = DoorThresholdDecoder(l2_reg=1.0, threshold=0.50)
        self.threat_decoder = VisceralThreatDecoder(l2_reg=1.0, firing_threshold=0.65)

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract Anscombe-stabilized frozen reservoir features across all images.

        Args:
            images: (N, H, W, 3) or (N, H, W) array of frames.

        Returns:
            np.ndarray: (N, feature_dim) feature matrix.
        """
        N = len(images)
        features: list[np.ndarray] = []
        for i in range(N):
            feat = self.reservoir.encode_frame(images[i], reset_state=True)
            features.append(feat)
        return np.vstack(features)

    def train_enemy_decoder(
        self,
        images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        test_split: float = 0.25,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Train and evaluate the 4-way Enemy Categorization Decoder."""
        if images is None or labels is None:
            images, labels = generate_enemy_stimulus_dataset(num_samples_per_class=30, seed=seed)

        print(f"[Enemy Decoder] Extracting reservoir features for {len(images)} samples...")
        t0 = time.perf_counter()
        X = self.extract_features(images)
        print(f"[Enemy Decoder] Features extracted in {time.perf_counter() - t0:.2f}s, shape: {X.shape}")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(images))
        n_test = int(round(len(images) * test_split))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, y_train = X[train_idx], labels[train_idx]
        X_test, y_test = X[test_idx], labels[test_idx]

        self.enemy_decoder.fit(X_train, y_train)

        # Evaluate
        probs_test = self.enemy_decoder.predict_proba(X_test)
        preds_test = np.argmax(probs_test, axis=1)
        accuracy = float(np.mean(preds_test == y_test))

        metrics = {
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_accuracy": accuracy,
            "feature_dim": X.shape[1],
        }
        print(f"[Enemy Decoder] Test Accuracy: {accuracy * 100:.1f}% (N_test={len(test_idx)})")
        return metrics

    def train_door_decoder(
        self,
        images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        test_split: float = 0.25,
        seed: int = 123,
    ) -> Dict[str, Any]:
        """Train and evaluate the Door Threshold Decoder."""
        if images is None or labels is None:
            images, labels = generate_door_stimulus_dataset(num_samples=80, seed=seed)

        print(f"[Door Decoder] Extracting reservoir features for {len(images)} samples...")
        t0 = time.perf_counter()
        X = self.extract_features(images)
        print(f"[Door Decoder] Features extracted in {time.perf_counter() - t0:.2f}s, shape: {X.shape}")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(images))
        n_test = int(round(len(images) * test_split))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, y_train = X[train_idx], labels[train_idx]
        X_test, y_test = X[test_idx], labels[test_idx]

        self.door_decoder.fit(X_train, y_train)

        # Evaluate
        probs_test = self.door_decoder.predict_proba(X_test)
        preds_test = (probs_test >= self.door_decoder.threshold).astype(int)
        accuracy = float(np.mean(preds_test == y_test))

        metrics = {
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_accuracy": accuracy,
            "feature_dim": X.shape[1],
        }
        print(f"[Door Decoder] Test Accuracy: {accuracy * 100:.1f}% (N_test={len(test_idx)})")
        return metrics

    def train_threat_decoder(
        self,
        images: Optional[np.ndarray] = None,
        arousal: Optional[np.ndarray] = None,
        test_split: float = 0.25,
        seed: int = 456,
    ) -> Dict[str, Any]:
        """Train and evaluate the Visceral Threat Decoder."""
        if images is None or arousal is None:
            images, arousal = generate_threat_stimulus_dataset(num_samples=80, seed=seed)

        print(f"[Threat Decoder] Extracting reservoir features for {len(images)} samples...")
        t0 = time.perf_counter()
        X = self.extract_features(images)
        print(f"[Threat Decoder] Features extracted in {time.perf_counter() - t0:.2f}s, shape: {X.shape}")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(images))
        n_test = int(round(len(images) * test_split))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, y_train = X[train_idx], arousal[train_idx]
        X_test, y_test = X[test_idx], arousal[test_idx]

        self.threat_decoder.fit(X_train, y_train)

        pred_arousal = self.threat_decoder.predict_arousal(X_test)
        mse = float(np.mean((pred_arousal - y_test) ** 2))
        corr = float(np.corrcoef(pred_arousal, y_test)[0, 1])

        metrics = {
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_mse": mse,
            "correlation": corr,
            "feature_dim": X.shape[1],
        }
        print(f"[Threat Decoder] Test MSE: {mse:.4f}, Correlation: {corr:.3f}")
        return metrics

    def save_all(self, directory: Path) -> None:
        """Serialize all trained decoder weights to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.enemy_decoder.save(directory / "enemy_decoder.npz")
        self.door_decoder.save(directory / "door_decoder.npz")
        self.threat_decoder.save(directory / "threat_decoder.npz")
        print(f"[Pipeline] All decoders saved to {directory}")

    def load_all(self, directory: Path) -> None:
        """Load trained decoder weights from disk."""
        directory = Path(directory)
        self.enemy_decoder.load(directory / "enemy_decoder.npz")
        self.door_decoder.load(directory / "door_decoder.npz")
        self.threat_decoder.load(directory / "threat_decoder.npz")
        print(f"[Pipeline] All decoders loaded from {directory}")


def train_all_decoders(save_dir: Optional[Path] = None) -> ReservoirReadoutPipeline:
    """Train all three reservoir decoders and optionally save to disk."""
    pipeline = ReservoirReadoutPipeline()
    print("=" * 60)
    print("Training Connectome-Constrained Reservoir Decoders")
    print("=" * 60)

    m1 = pipeline.train_enemy_decoder()
    m2 = pipeline.train_door_decoder()
    m3 = pipeline.train_threat_decoder()

    if save_dir is not None:
        pipeline.save_all(save_dir)

    return pipeline
