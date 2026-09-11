"""MaleCNS v1.0 Biological Connectome Extractor Scaffold (Phase 1C-A).

Defines the ingestion and query pipeline for authentic Drosophila EM connectomics
from the Janelia MaleCNS v1.0 / neuPrint dataset (Takemura et al., 2017; Nature 2025).

Provides schema for:
  - Querying T4a candidate neurons in the Right Optic Lobe / Lobula Plate
  - Extracting verified presynaptic inputs (Mi1, Tm3, Mi4, Mi9, C3, CT1)
  - Sourcing raw EM 3D synapse coordinates (x, y, z nm -> converted to microns)
  - Ingesting dendritic skeleton morphology (SWC format)
  - Generating cryptographic ConnectomeFingerprint for biological provenance
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from fly_doom.connectome.manifest import ConnectomeFingerprint, DatasetManifest
from fly_doom.core.provenance import Provenance


@dataclass(frozen=True)
class RealEMSynapse:
    """Real EM chemical synapse contact extracted directly from MaleCNS v1.0."""

    synapse_id: int
    pre_body_id: int
    pre_cell_type: str
    post_body_id: int
    post_cell_type: str
    x_nm: float
    y_nm: float
    z_nm: float
    confidence: float
    neuprint_roi: str
    dendritic_branch_id: Optional[int] = None

    @property
    def x_um(self) -> float:
        return self.x_nm / 1000.0

    @property
    def y_um(self) -> float:
        return self.y_nm / 1000.0

    @property
    def z_um(self) -> float:
        return self.z_nm / 1000.0


@dataclass
class MaleCNST4aExtraction:
    """Authentic biological reconstruction extracted from MaleCNS v1.0."""

    t4a_body_id: int
    hemisphere: str
    dataset_version: str
    synapses: List[RealEMSynapse]
    partner_counts: Dict[str, int]
    morphology_swc_sha256: Optional[str] = None
    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            tier="biological_reconstruction",
            source="MaleCNS_v1.0_neuPrint_Janelia",
            confidence=1.0,
            rationale="Raw EM synaptic contacts and 3D coordinates extracted from MaleCNS v1.0 connectome",
            doi="10.1038/s41586-025-09276-5",
            figure_table_ref="MaleCNS v1.0 Connectome Release",
            access_date="2026-09-11",
        )
    )

    @property
    def total_synapses(self) -> int:
        return len(self.synapses)

    def compute_fingerprint(self) -> str:
        """Compute SHA256 over canonical JSON of all physical synapse coordinates."""
        payload = [
            (s.synapse_id, s.pre_body_id, s.pre_cell_type, round(s.x_nm, 1), round(s.y_nm, 1), round(s.z_nm, 1))
            for s in sorted(self.synapses, key=lambda x: x.synapse_id)
        ]
        canonical_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
