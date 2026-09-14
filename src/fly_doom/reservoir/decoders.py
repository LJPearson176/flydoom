"""Lightweight Connectome-Constrained Readout Decoders for Doom Tasks.

Zero backpropagation through the brain. All learning is confined to convex
or lightweight linear/MLP decoders operating on frozen reservoir population vectors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.special import expit, softmax

from fly_doom.core.provenance import Provenance

ENEMY_CLASSES = ["Background", "Zombieman", "Shotgun Guy", "Imp"]


@dataclass(frozen=True)
class EnemyPrediction:
    """Prediction output from the Enemy Categorization Decoder."""

    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class DoorPrediction:
    """Prediction output from the Door Threshold Decoder."""

    should_use: bool
    probability: float
    threshold: float = 0.50


@dataclass(frozen=True)
class ThreatPrediction:
    """Prediction output from the Visceral Threat Decoder."""

    arousal: float  # [0.0, 1.0]
    should_fire: bool
    firing_threshold: float = 0.65


class EnemyCategorizationDecoder:
    """Lightweight Ridge / Multinomial Readout for 4-way Enemy Classification.

    Discriminates [Background, Zombieman, Shotgun Guy, Imp] based purely
    on frozen downstream connectome population vectors.
    """

    def __init__(self, l2_reg: float = 1.0):
        self.l2_reg = float(l2_reg)
        self.weights: Optional[np.ndarray] = None  # shape: (n_features, n_classes)
        self.bias: Optional[np.ndarray] = None  # shape: (n_classes,)
        self.n_classes = len(ENEMY_CLASSES)
        self.provenance = Provenance(
            tier="computational_hypothesis",
            source="Enemy_Categorization_Reservoir_Decoder",
            confidence=0.90,
            rationale="Multinomial ridge readout over frozen MaleCNS reservoir population vectors",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnemyCategorizationDecoder:
        """Fit linear ridge readout using closed-form regularized least squares.

        Args:
            X: Matrix of reservoir feature vectors, shape (N, D).
            y: Integer class labels in {0, 1, 2, 3}, shape (N,).
        """
        N, D = X.shape
        Y_onehot = np.eye(self.n_classes)[y]  # (N, C)

        # Standardize features (zero mean, unit variance)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.mean) / self.std

        # Ridge regression: W = (X^T X + lambda * I)^-1 X^T Y
        XT_X = X_norm.T @ X_norm
        reg = self.l2_reg * np.eye(D)
        W = np.linalg.solve(XT_X + reg, X_norm.T @ (Y_onehot - np.mean(Y_onehot, axis=0)))

        self.weights = W  # (D, C)
        self.bias = np.mean(Y_onehot, axis=0)  # (C,)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute calibrated class probabilities via softmax."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Decoder has not been fitted.")
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_norm = (X - self.mean) / self.std
        logits = X_norm @ self.weights + self.bias
        return softmax(logits, axis=1)

    def predict(self, x: np.ndarray) -> EnemyPrediction:
        """Predict enemy category for a single reservoir feature vector."""
        probs = self.predict_proba(x)[0]
        cid = int(np.argmax(probs))
        cname = ENEMY_CLASSES[cid]
        conf = float(probs[cid])
        prob_dict = {name: float(probs[i]) for i, name in enumerate(ENEMY_CLASSES)}
        return EnemyPrediction(
            class_id=cid,
            class_name=cname,
            confidence=conf,
            probabilities=prob_dict,
        )

    def save(self, path: Path) -> None:
        """Save decoder parameters to compressed NPZ."""
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
            l2_reg=self.l2_reg,
        )

    def load(self, path: Path) -> EnemyCategorizationDecoder:
        """Load decoder parameters from NPZ."""
        data = np.load(path)
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.mean = data["mean"]
        self.std = data["std"]
        self.l2_reg = float(data["l2_reg"])
        return self


class DoorThresholdDecoder:
    """Lightweight Logistic Readout for Interactable Door Threshold Detection.

    Fires a learned USE decision when downstream optic flow and central contrast
    indicate an interactable door threshold (Linedef 151, 340, 324) in range.
    """

    def __init__(self, l2_reg: float = 1.0, threshold: float = 0.50):
        self.l2_reg = float(l2_reg)
        self.threshold = float(threshold)
        self.weights: Optional[np.ndarray] = None  # shape: (D,)
        self.bias: float = 0.0
        self.provenance = Provenance(
            tier="computational_hypothesis",
            source="Door_Threshold_Reservoir_Decoder",
            confidence=0.92,
            rationale="Logistic readout over central complex and optic flow population vectors",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> DoorThresholdDecoder:
        """Fit logistic / ridge classifier for binary door detection."""
        N, D = X.shape
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.mean) / self.std

        # Continuous target in [-1, +1] for ridge least-squares logistic proxy
        y_target = np.where(y > 0, 1.0, -1.0)
        XT_X = X_norm.T @ X_norm
        reg = self.l2_reg * np.eye(D)
        w = np.linalg.solve(XT_X + reg, X_norm.T @ y_target)

        self.weights = w
        self.bias = float(np.mean(y_target))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute sigmoid probability of door interactability."""
        if self.weights is None:
            raise RuntimeError("Decoder has not been fitted.")
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_norm = (X - self.mean) / self.std
        logits = X_norm @ self.weights + self.bias
        return expit(logits)

    def predict(self, x: np.ndarray) -> DoorPrediction:
        """Predict door interactability for a single reservoir feature vector."""
        prob = float(self.predict_proba(x)[0])
        should_use = prob >= self.threshold
        return DoorPrediction(
            should_use=should_use,
            probability=prob,
            threshold=self.threshold,
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
            threshold=self.threshold,
        )

    def load(self, path: Path) -> DoorThresholdDecoder:
        data = np.load(path)
        self.weights = data["weights"]
        self.bias = float(data["bias"])
        self.mean = data["mean"]
        self.std = data["std"]
        self.threshold = float(data["threshold"])
        return self


class VisceralThreatDecoder:
    """Lightweight Visceral Threat & Arousal Readout.

    Reads out looming expansion from Lobula LC4 and weapon discharge tracts
    (DNpe017) to modulate motor arousal and weapon firing.
    """

    def __init__(self, l2_reg: float = 1.0, firing_threshold: float = 0.65):
        self.l2_reg = float(l2_reg)
        self.firing_threshold = float(firing_threshold)
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.provenance = Provenance(
            tier="computational_hypothesis",
            source="Visceral_Threat_Reservoir_Decoder",
            confidence=0.90,
            rationale="Continuous threat arousal and ballistic fire gating from looming LC4 and DN tracts",
        )

    def fit(self, X: np.ndarray, y_arousal: np.ndarray) -> VisceralThreatDecoder:
        """Fit continuous threat arousal readout."""
        N, D = X.shape
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.mean) / self.std

        y_clamped = np.clip(y_arousal, 0.0, 1.0)
        # Logit space proxy
        y_logit = np.log(np.clip(y_clamped, 0.01, 0.99) / (1.0 - np.clip(y_clamped, 0.01, 0.99)))

        XT_X = X_norm.T @ X_norm
        reg = self.l2_reg * np.eye(D)
        w = np.linalg.solve(XT_X + reg, X_norm.T @ y_logit)

        self.weights = w
        self.bias = float(np.mean(y_logit))
        return self

    def predict_arousal(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Decoder has not been fitted.")
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_norm = (X - self.mean) / self.std
        logits = X_norm @ self.weights + self.bias
        return expit(logits)

    def predict(self, x: np.ndarray) -> ThreatPrediction:
        arousal = float(self.predict_arousal(x)[0])
        should_fire = arousal >= self.firing_threshold
        return ThreatPrediction(
            arousal=arousal,
            should_fire=should_fire,
            firing_threshold=self.firing_threshold,
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
            firing_threshold=self.firing_threshold,
        )

    def load(self, path: Path) -> VisceralThreatDecoder:
        data = np.load(path)
        self.weights = data["weights"]
        self.bias = float(data["bias"])
        self.mean = data["mean"]
        self.std = data["std"]
        self.firing_threshold = float(data["firing_threshold"])
        return self


def load_trained_decoders(
    assets_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[DoorThresholdDecoder], Optional[EnemyCategorizationDecoder], Optional[VisceralThreatDecoder]]:
    """Load pre-trained connectome reservoir decoders from disk.

    Args:
        assets_dir: Path to directory containing saved .npz decoders.
                    Defaults to repo assets/reservoir_decoders/.

    Returns:
        Tuple of (door_decoder, enemy_decoder, threat_decoder), where any
        missing decoder is returned as None.
    """
    if assets_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        assets_dir = repo_root / "assets" / "reservoir_decoders"
    else:
        assets_dir = Path(assets_dir)

    door_dec: Optional[DoorThresholdDecoder] = None
    enemy_dec: Optional[EnemyCategorizationDecoder] = None
    threat_dec: Optional[VisceralThreatDecoder] = None

    door_path = assets_dir / "door_decoder.npz"
    if door_path.exists():
        door_dec = DoorThresholdDecoder().load(door_path)

    enemy_path = assets_dir / "enemy_decoder.npz"
    if enemy_path.exists():
        enemy_dec = EnemyCategorizationDecoder().load(enemy_path)

    threat_path = assets_dir / "threat_decoder.npz"
    if threat_path.exists():
        threat_dec = VisceralThreatDecoder().load(threat_path)

    return door_dec, enemy_dec, threat_dec
