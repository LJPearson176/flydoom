"""Mushroom Body Dopaminergic Plasticity & Valence Learning Engine.

Models the Drosophila Mushroom Body (MB) associative learning circuit:
  - Sparse Kenyon Cell (KC) representation in the calyx (k-Winner-Take-All, ~10% sparsity).
  - Dual antagonistic Mushroom Body Output Neurons (MBON):
      * MBON-app (e.g. MBON-gamma1pedc): promotes forward approach and exploration.
      * MBON-av  (e.g. MBON-gamma2alpha'1): promotes avoidance steering and retreat.
  - Dopaminergic Neurons (DAN):
      * PPL1 cluster: encodes aversive punishment (health loss / pain).
        Induces heterosynaptic Long-Term Depression (LTD) on KC -> MBON-app synapses.
      * PAM cluster: encodes reward (hostile neutralization / item collection).
        Induces LTD on KC -> MBON-av synapses (or potentiation of approach).
  - Output valence signal V_MB in [-1.0, 1.0] modulates central complex navigation.

References:
  - Aso et al. (eLife 2014) "The neuronal architecture of the mushroom body provides a logic for associative learning"
  - Hige (Neurobiol Learn Mem 2018) "What can the fly mushroom body teach us about associative learning?"
  - Handler et al. (Nature 2019) "Distinct dopamine receptor types define opposing functions of a single dopamine neuron"
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from fly_doom.core.provenance import Provenance


@dataclass(frozen=True)
class MushroomBodyParameters:
    """Parameters for Kenyon Cell expansion and dopamine-gated plasticity."""

    num_kenyon_cells: int = 64
    input_dim: int = 16
    sparsity_k: int = 6  # ~9.4% sparsity
    learning_rate_ltd: float = 0.08  # LTD rate per dopamine burst
    homeostatic_rate: float = 0.0005  # Slow recovery towards baseline
    baseline_weight: float = 1.0
    min_weight: float = 0.05
    max_weight: float = 2.0
    seed: int = 42

    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            tier="biological_reconstruction",
            source="Drosophila_Mushroom_Body_Aso_2014_Handler_2019",
            confidence=0.94,
            rationale="Sparse KC k-WTA expansion with PPL1/PAM dopamine-gated three-factor Hebbian plasticity",
            doi="10.7554/eLife.04577",
            figure_table_ref="Figures 4-7",
            access_date="2026-09-12",
        )
    )


@dataclass(frozen=True)
class MushroomBodyState:
    """Snapshot of Mushroom Body neural activations and plasticity readouts."""

    valence: float  # Net valence [-1.0, 1.0] (+: approach, -: avoid)
    mbon_app_drive: float  # Excitatory drive to approach MBON
    mbon_av_drive: float  # Excitatory drive to avoidance MBON
    ppl1_dopamine: float  # Aversive dopamine burst [0.0, 1.0]
    pam_dopamine: float  # Reward dopamine burst [0.0, 1.0]
    active_kc_indices: Tuple[int, ...]  # Indices of active Kenyon cells
    mean_app_weight: float  # Mean KC -> MBON_app synaptic strength
    mean_av_weight: float  # Mean KC -> MBON_av synaptic strength


class MushroomBodyPlasticityEngine:
    """Simulates Mushroom Body sparse encoding and dopamine-gated associative learning."""

    def __init__(self, params: Optional[MushroomBodyParameters] = None):
        self.params = params or MushroomBodyParameters()
        p = self.params

        # Fixed random projection matrix from sensory input to Kenyon Cells
        rng = np.random.RandomState(p.seed)
        self.proj_matrix = rng.normal(0.0, 1.0 / math.sqrt(p.input_dim), (p.num_kenyon_cells, p.input_dim))

        # Plastic synaptic weights from Kenyon Cells to MBONs
        self.w_app = np.full(p.num_kenyon_cells, p.baseline_weight, dtype=np.float64)
        self.w_av = np.full(p.num_kenyon_cells, p.baseline_weight, dtype=np.float64)

        # Activity buffer
        self.last_kc_activations = np.zeros(p.num_kenyon_cells, dtype=np.float64)

    def reset_synaptic_weights(self) -> None:
        """Reset all plastic weights back to baseline."""
        self.w_app.fill(self.params.baseline_weight)
        self.w_av.fill(self.params.baseline_weight)
        self.last_kc_activations.fill(0.0)

    def compute_kenyon_cells(self, sensory_features: np.ndarray) -> np.ndarray:
        """Compute sparse k-WTA Kenyon Cell activations from sensory context.

        Args:
            sensory_features: (input_dim,) normalized feature vector.

        Returns:
            kc_acts: (num_kenyon_cells,) sparse activation vector (only top k non-zero).
        """
        p = self.params
        # Linear projection
        raw = np.dot(self.proj_matrix, sensory_features)

        # k-Winner-Take-All selection
        top_k_indices = np.argpartition(raw, -p.sparsity_k)[-p.sparsity_k:]
        kc_acts = np.zeros(p.num_kenyon_cells, dtype=np.float64)
        # ReLU on top k
        kc_acts[top_k_indices] = np.maximum(0.0, raw[top_k_indices])

        # Normalize active energy
        sum_act = float(np.sum(kc_acts))
        if sum_act > 1e-6:
            kc_acts /= sum_act

        return kc_acts

    def step(
        self,
        sensory_features: np.ndarray,
        aversive_us: float = 0.0,
        reward_us: float = 0.0,
    ) -> MushroomBodyState:
        """Advance MB dynamics and execute dopamine-gated three-factor plasticity.

        Args:
            sensory_features: (input_dim,) sensory vector representing current spatial context.
            aversive_us: Magnitude of aversive punishment (e.g. from health loss), [0.0, 1.0].
            reward_us: Magnitude of rewarding stimulus (e.g. enemy kill), [0.0, 1.0].

        Returns:
            MushroomBodyState with valence, MBON drives, dopamine levels, and active KCs.
        """
        p = self.params

        # 1. Sparse Kenyon Cell expansion
        kc_acts = self.compute_kenyon_cells(sensory_features)
        self.last_kc_activations = kc_acts

        # 2. Output Neuron (MBON) Drives
        drive_app = float(np.dot(self.w_app, kc_acts))
        drive_av = float(np.dot(self.w_av, kc_acts))

        # Net valence [-1.0, 1.0]
        denom = max(1e-3, (drive_app + drive_av) * 0.5)
        raw_val = (drive_app - drive_av) / denom
        valence = float(np.tanh(raw_val))

        # 3. Dopaminergic Plasticity (Three-Factor Hebbian LTD)
        # Active KCs paired with aversive PPL1 burst -> LTD of MBON_app synapses
        ppl1_da = float(np.clip(aversive_us, 0.0, 1.0))
        if ppl1_da > 0.01:
            delta_w_app = p.learning_rate_ltd * ppl1_da * kc_acts
            self.w_app = np.maximum(p.min_weight, self.w_app - delta_w_app)

        # Active KCs paired with reward PAM burst -> LTD of MBON_av synapses
        pam_da = float(np.clip(reward_us, 0.0, 1.0))
        if pam_da > 0.01:
            delta_w_av = p.learning_rate_ltd * pam_da * kc_acts
            self.w_av = np.maximum(p.min_weight, self.w_av - delta_w_av)

        # 4. Slow homeostatic weight recovery
        self.w_app += p.homeostatic_rate * (p.baseline_weight - self.w_app)
        self.w_av += p.homeostatic_rate * (p.baseline_weight - self.w_av)

        active_indices = tuple(int(idx) for idx in np.where(kc_acts > 0.0)[0])

        return MushroomBodyState(
            valence=valence,
            mbon_app_drive=drive_app,
            mbon_av_drive=drive_av,
            ppl1_dopamine=ppl1_da,
            pam_dopamine=pam_da,
            active_kc_indices=active_indices,
            mean_app_weight=float(np.mean(self.w_app)),
            mean_av_weight=float(np.mean(self.w_av)),
        )

    def save_weights(self, path: Union[str, Path]) -> None:
        """Persist learned plastic weights to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, w_app=self.w_app, w_av=self.w_av)

    def load_weights(self, path: Union[str, Path]) -> None:
        """Load plastic weights from disk."""
        data = np.load(path)
        self.w_app = np.copy(data["w_app"])
        self.w_av = np.copy(data["w_av"])
