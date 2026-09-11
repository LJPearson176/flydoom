"""Dataset manifest and cryptographic fingerprinting for connectome graphs.

Ensures that any loaded connectome graph (or slice) is cryptographically
fingerprinted and validated against an immutable manifest before simulation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConnectomeFingerprint:
    """Cryptographic and topological signature of a connectome graph."""

    neuron_count: int
    directed_edge_count: int
    synapse_contact_count: int
    in_degree_histogram_hash: str
    out_degree_histogram_hash: str
    cell_type_counts_hash: str
    hemisphere_counts_hash: str
    combined_hash: str

    @staticmethod
    def compute_histogram_hash(data: Dict[Any, int]) -> str:
        """Compute SHA256 over a canonical JSON serialization of a histogram."""
        canonical_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    @classmethod
    def from_topological_data(
        cls,
        neuron_count: int,
        directed_edge_count: int,
        synapse_contact_count: int,
        in_degree_counts: Dict[int, int],
        out_degree_counts: Dict[int, int],
        cell_type_counts: Dict[str, int],
        hemisphere_counts: Dict[str, int],
    ) -> ConnectomeFingerprint:
        in_hash = cls.compute_histogram_hash(in_degree_counts)
        out_hash = cls.compute_histogram_hash(out_degree_counts)
        ct_hash = cls.compute_histogram_hash(cell_type_counts)
        hemi_hash = cls.compute_histogram_hash(hemisphere_counts)

        # Combined summary hash
        summary_payload = f"{neuron_count}|{directed_edge_count}|{synapse_contact_count}|{in_hash}|{out_hash}|{ct_hash}|{hemi_hash}"
        combined = hashlib.sha256(summary_payload.encode("utf-8")).hexdigest()

        return cls(
            neuron_count=neuron_count,
            directed_edge_count=directed_edge_count,
            synapse_contact_count=synapse_contact_count,
            in_degree_histogram_hash=in_hash,
            out_degree_histogram_hash=out_hash,
            cell_type_counts_hash=ct_hash,
            hemisphere_counts_hash=hemi_hash,
            combined_hash=combined,
        )


@dataclass
class DatasetManifest:
    """Complete immutable manifest for a biological connectome dataset."""

    dataset_name: str
    dataset_version: str
    source_citation: str
    source_files_sha256: Dict[str, str]
    expected_fingerprint: ConnectomeFingerprint
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verify(self, candidate_fingerprint: ConnectomeFingerprint) -> List[str]:
        """Verify candidate fingerprint against manifest. Returns list of discrepancy messages (empty if pass)."""
        discrepancies: List[str] = []
        if self.expected_fingerprint.neuron_count != candidate_fingerprint.neuron_count:
            discrepancies.append(
                f"Neuron count mismatch: expected {self.expected_fingerprint.neuron_count}, got {candidate_fingerprint.neuron_count}"
            )
        if (
            self.expected_fingerprint.directed_edge_count
            != candidate_fingerprint.directed_edge_count
        ):
            discrepancies.append(
                f"Directed edge count mismatch: expected {self.expected_fingerprint.directed_edge_count}, got {candidate_fingerprint.directed_edge_count}"
            )
        if (
            self.expected_fingerprint.synapse_contact_count
            != candidate_fingerprint.synapse_contact_count
        ):
            discrepancies.append(
                f"Synapse contact count mismatch: expected {self.expected_fingerprint.synapse_contact_count}, got {candidate_fingerprint.synapse_contact_count}"
            )
        if (
            self.expected_fingerprint.in_degree_histogram_hash
            != candidate_fingerprint.in_degree_histogram_hash
        ):
            discrepancies.append(
                f"In-degree histogram hash mismatch: expected {self.expected_fingerprint.in_degree_histogram_hash}, got {candidate_fingerprint.in_degree_histogram_hash}"
            )
        if (
            self.expected_fingerprint.out_degree_histogram_hash
            != candidate_fingerprint.out_degree_histogram_hash
        ):
            discrepancies.append(
                f"Out-degree histogram hash mismatch: expected {self.expected_fingerprint.out_degree_histogram_hash}, got {candidate_fingerprint.out_degree_histogram_hash}"
            )
        if (
            self.expected_fingerprint.cell_type_counts_hash
            != candidate_fingerprint.cell_type_counts_hash
        ):
            discrepancies.append("Cell type counts hash mismatch")
        if (
            self.expected_fingerprint.hemisphere_counts_hash
            != candidate_fingerprint.hemisphere_counts_hash
        ):
            discrepancies.append("Hemisphere counts hash mismatch")
        if (
            self.expected_fingerprint.combined_hash
            != candidate_fingerprint.combined_hash
        ):
            discrepancies.append(
                f"Combined fingerprint hash mismatch: expected {self.expected_fingerprint.combined_hash}, got {candidate_fingerprint.combined_hash}"
            )
        return discrepancies

    def to_json(self, indent: int = 2) -> str:
        data = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_citation": self.source_citation,
            "source_files_sha256": self.source_files_sha256,
            "expected_fingerprint": asdict(self.expected_fingerprint),
            "metadata": self.metadata,
        }
        return json.dumps(data, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> DatasetManifest:
        data = json.loads(json_str)
        fp_data = data["expected_fingerprint"]
        fingerprint = ConnectomeFingerprint(**fp_data)
        return cls(
            dataset_name=data["dataset_name"],
            dataset_version=data["dataset_version"],
            source_citation=data["source_citation"],
            source_files_sha256=data["source_files_sha256"],
            expected_fingerprint=fingerprint,
            metadata=data.get("metadata", {}),
        )


# Canonical MaleCNS v1.0 reference constants
MALECNS_V1_NEURON_COUNT = 166700
MALECNS_V1_DIRECTED_EDGES = 25582938
MALECNS_V1_CONTACTS = 124177617
