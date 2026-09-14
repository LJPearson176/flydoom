"""Connectome-Constrained Reservoir Computing for Doom Tasks.

Frozen biological connectome dynamics with lightweight downstream readouts
for enemy classification, door threshold detection, and visceral threat assessment.
"""

from fly_doom.reservoir.reservoir import (
    ConnectomeReservoir,
    ReservoirConfig,
    ReservoirFeatureIndices,
    create_canonical_doom_reservoir_graph,
)
from fly_doom.reservoir.decoders import (
    EnemyCategorizationDecoder,
    DoorThresholdDecoder,
    VisceralThreatDecoder,
    EnemyPrediction,
    DoorPrediction,
    ThreatPrediction,
    load_trained_decoders,
)
from fly_doom.reservoir.datasets import (
    generate_enemy_stimulus_dataset,
    generate_door_stimulus_dataset,
    generate_threat_stimulus_dataset,
)
from fly_doom.reservoir.pipeline import (
    ReservoirReadoutPipeline,
    train_all_decoders,
)

__all__ = [
    "ConnectomeReservoir",
    "ReservoirConfig",
    "ReservoirFeatureIndices",
    "create_canonical_doom_reservoir_graph",
    "EnemyCategorizationDecoder",
    "DoorThresholdDecoder",
    "VisceralThreatDecoder",
    "EnemyPrediction",
    "DoorPrediction",
    "ThreatPrediction",
    "load_trained_decoders",
    "generate_enemy_stimulus_dataset",
    "generate_door_stimulus_dataset",
    "generate_threat_stimulus_dataset",
    "ReservoirReadoutPipeline",
    "train_all_decoders",
]
