"""Dataset generators for training and evaluating Connectome-Constrained Reservoir Decoders."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from fly_doom.reservoir.decoders import ENEMY_CLASSES


def generate_enemy_stimulus_dataset(
    num_samples_per_class: int = 30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate visual frames with realistic enemy silhouettes and corridor backgrounds.

    Classes:
      0: Background (empty corridor)
      1: Zombieman (tan/khaki uniform, rifle silhouette)
      2: Shotgun Guy (dark armor, black uniform, high central contrast)
      3: Imp (brown spiky textured body, red highlight eyes/mouth)

    Returns:
      images: (N, 64, 64, 3) uint8 RGB array
      labels: (N,) int32 class labels
    """
    rng = np.random.default_rng(seed)
    total_samples = num_samples_per_class * len(ENEMY_CLASSES)
    images = np.zeros((total_samples, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros(total_samples, dtype=np.int32)

    idx = 0
    for class_id in range(len(ENEMY_CLASSES)):
        for _ in range(num_samples_per_class):
            frame = np.zeros((64, 64, 3), dtype=np.float32)

            # Baseline corridor: ceiling (gray-blue), wall (brown-gray), floor (dark gray)
            frame[:20, :] = [45, 45, 60]
            frame[20:48, :] = [80, 70, 65]
            frame[48:, :] = [35, 35, 35]

            # Corridor noise and perspective lines
            noise = rng.normal(0, 4.0, (64, 64, 3))
            frame = np.clip(frame + noise, 0, 255)

            if class_id == 1:
                # Zombieman (POSS): Slender humanoid, khaki uniform, green hair, tan rifle
                cx = rng.integers(28, 36)
                cy = rng.integers(26, 36)
                # Slender torso + legs (h=20, w=8)
                frame[cy - 8 : cy + 12, cx - 4 : cx + 4] = [165, 145, 95]
                # Green hair
                frame[cy - 12 : cy - 8, cx - 2 : cx + 2] = [85, 140, 65]
                # Tan rifle arm
                frame[cy - 2 : cy + 6, cx + 3 : cx + 7] = [185, 160, 110]

            elif class_id == 2:
                # Shotgun Guy (SPOS): Bulky dark armor, bald white head, black shotgun
                cx = rng.integers(28, 36)
                cy = rng.integers(26, 36)
                # Black armor torso + legs (h=18, w=10)
                frame[cy - 7 : cy + 11, cx - 5 : cx + 5] = [28, 28, 35]
                # High-contrast bald white head
                frame[cy - 12 : cy - 7, cx - 3 : cx + 3] = [215, 210, 205]
                # Blue/gray armor trim
                frame[cy - 3 : cy + 3, cx - 5 : cx + 5] = [75, 80, 105]
                # Shotgun barrel
                frame[cy : cy + 8, cx + 4 : cx + 8] = [45, 45, 50]

            elif class_id == 3:
                # Imp (TROO): Broad crouching demon, dark brown body, shoulder spikes, red eyes
                cx = rng.integers(28, 36)
                cy = rng.integers(28, 36)
                # Broad crouching body (h=14, w=14)
                frame[cy - 6 : cy + 8, cx - 7 : cx + 7] = [115, 65, 35]
                # Spikes projecting from shoulders
                frame[cy - 9 : cy - 5, cx - 8 : cx - 5] = [135, 75, 40]
                frame[cy - 9 : cy - 5, cx + 5 : cx + 8] = [135, 75, 40]
                # Glowing red eyes & mouth
                frame[cy - 4 : cy - 2, cx - 4 : cx + 4] = [230, 45, 15]

            images[idx] = np.clip(frame, 0, 255).astype(np.uint8)
            labels[idx] = class_id
            idx += 1

    return images, labels


def generate_door_stimulus_dataset(
    num_samples: int = 80,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate visual frames of doors (positive) vs open corridors / plain walls (negative).

    Returns:
      images: (N, 64, 64, 3) uint8 RGB array
      labels: (N,) int32 binary labels (1 = door threshold, 0 = non-door)
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros(num_samples, dtype=np.int32)

    for i in range(num_samples):
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        is_door = i >= (num_samples // 2)

        if is_door:
            # Door: Central vertical rectangular frame with metallic vertical seams
            frame[:, :] = [70, 65, 60]
            # Dark outer door archway
            frame[8:56, 14:50] = [110, 105, 95]
            # Vertical seam lines
            frame[8:56, 31:33] = [40, 35, 30]
            # Red/yellow door track light on sides
            frame[12:20, 12:14] = [220, 180, 40]
            frame[12:20, 50:52] = [220, 180, 40]
            # Base threshold
            frame[56:, :] = [50, 45, 45]
            labels[i] = 1
        else:
            # Non-door: Either open corridor continuation or plain brick/pipe wall
            if rng.random() < 0.5:
                # Open receding corridor
                frame[:24, :] = [40, 40, 55]
                frame[24:44, :] = [75, 75, 75]
                frame[44:, :] = [30, 30, 30]
                # Center receding opening
                frame[20:44, 24:40] = [20, 20, 25]
            else:
                # Solid wall with horizontal panels
                frame[:, :] = [90, 80, 70]
                frame[::16, :] = [45, 40, 35]
            labels[i] = 0

        noise = rng.normal(0, 3.0, (64, 64, 3))
        images[i] = np.clip(frame + noise, 0, 255).astype(np.uint8)

    return images, labels


def generate_threat_stimulus_dataset(
    num_samples: int = 80,
    seed: int = 456,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate visual frames with varying threat / looming intensity.

    Returns:
      images: (N, 64, 64, 3) uint8 RGB array
      arousal: (N,) float32 continuous threat arousal target in [0.0, 1.0]
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, 64, 64, 3), dtype=np.uint8)
    arousal = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        # Background
        frame[:, :] = [50, 50, 55]

        # Target threat level: continuous from 0.0 to 1.0
        threat = float(i) / float(num_samples - 1)
        arousal[i] = threat

        if threat > 0.2:
            # Looming target: expanding patch in center of screen
            # Radius scales with threat (from 2 to 24 pixels)
            radius = int(round(2.0 + threat * 22.0))
            cy, cx = 32, 32
            y_grid, x_grid = np.ogrid[:64, :64]
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            mask = dist <= radius

            # Threat patch: bright orange/red (fireball) or dark charging silhouette
            color = [220, 80 + 40 * (1 - threat), 20]
            frame[mask] = color

        noise = rng.normal(0, 3.0, (64, 64, 3))
        images[i] = np.clip(frame + noise, 0, 255).astype(np.uint8)

    return images, arousal
