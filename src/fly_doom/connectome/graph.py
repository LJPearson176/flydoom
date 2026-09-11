"""Sparse connectome graph representation with topological verification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.sparse as sp

from fly_doom.connectome.manifest import ConnectomeFingerprint, DatasetManifest
from fly_doom.core.provenance import Provenance, attach_provenance


@dataclass
class ConnectomeGraph:
    """Compressed Sparse Row (CSR) representation of a connectome graph.

    Represents pre-synaptic -> post-synaptic transmission.
    Convention:
      - row i represents pre-synaptic neuron i
      - col_indices[row_ptr[i] : row_ptr[i+1]] are the post-synaptic target neurons
      - weights[row_ptr[i] : row_ptr[i+1]] are the synaptic connection strengths
    """

    num_neurons: int
    row_ptr: np.ndarray  # shape: (num_neurons + 1,), dtype: uint32 or int64
    col_idx: np.ndarray  # shape: (num_edges,), dtype: uint32 or int64
    weights: np.ndarray  # shape: (num_edges,), dtype: float64
    contacts: np.ndarray  # shape: (num_edges,), dtype: uint32
    neuron_ids: np.ndarray  # shape: (num_neurons,), dtype: uint64 or object
    cell_types: Sequence[str]  # length: num_neurons
    hemispheres: Sequence[str]  # length: num_neurons
    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            tier="biological_reconstruction",
            source="ConnectomeGraph_CSR",
            confidence=1.0,
            rationale="Verified CSR adjacency representation of connectome reconstruction",
        )
    )

    def __post_init__(self) -> None:
        assert len(self.row_ptr) == self.num_neurons + 1
        assert len(self.col_idx) == len(self.weights) == len(self.contacts)
        assert len(self.neuron_ids) == self.num_neurons
        assert len(self.cell_types) == self.num_neurons
        assert len(self.hemispheres) == self.num_neurons

    @property
    def num_edges(self) -> int:
        return len(self.col_idx)

    @property
    def total_contacts(self) -> int:
        return int(np.sum(self.contacts))

    def compute_out_degrees(self) -> np.ndarray:
        """Compute out-degree for each pre-synaptic neuron."""
        return np.diff(self.row_ptr)

    def compute_in_degrees(self) -> np.ndarray:
        """Compute in-degree for each post-synaptic neuron."""
        in_degrees = np.zeros(self.num_neurons, dtype=np.int64)
        if len(self.col_idx) > 0:
            np.add.at(in_degrees, self.col_idx, 1)
        return in_degrees

    def compute_fingerprint(self) -> ConnectomeFingerprint:
        """Compute the full topological and cryptographic fingerprint."""
        in_degrees = self.compute_in_degrees()
        out_degrees = self.compute_out_degrees()

        in_deg_counts = dict(Counter(in_degrees.tolist()))
        out_deg_counts = dict(Counter(out_degrees.tolist()))
        cell_type_counts = dict(Counter(self.cell_types))
        hemi_counts = dict(Counter(self.hemispheres))

        return ConnectomeFingerprint.from_topological_data(
            neuron_count=self.num_neurons,
            directed_edge_count=self.num_edges,
            synapse_contact_count=self.total_contacts,
            in_degree_counts=in_deg_counts,
            out_degree_counts=out_deg_counts,
            cell_type_counts=cell_type_counts,
            hemisphere_counts=hemi_counts,
        )

    def verify_against_manifest(self, manifest: DatasetManifest) -> None:
        """Verify graph against manifest. Raises ValueError if discrepancies found."""
        fp = self.compute_fingerprint()
        discrepancies = manifest.verify(fp)
        if discrepancies:
            msg = f"Graph validation failed against manifest '{manifest.dataset_name}':\n" + "\n".join(
                f"  - {d}" for d in discrepancies
            )
            raise ValueError(msg)

    def to_scipy_csr(self) -> sp.csr_matrix:
        """Convert to scipy.sparse.csr_matrix for fast sparse linear algebra."""
        return sp.csr_matrix(
            (self.weights, self.col_idx, self.row_ptr),
            shape=(self.num_neurons, self.num_neurons),
        )

    def extract_subgraph(self, selected_indices: Sequence[int]) -> ConnectomeGraph:
        """Extract an induced subgraph on selected_indices, re-indexing nodes to 0..K-1."""
        idx_set = set(selected_indices)
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        k = len(selected_indices)

        new_row_ptr = [0]
        new_col_idx = []
        new_weights = []
        new_contacts = []

        for old_u in selected_indices:
            start = self.row_ptr[old_u]
            end = self.row_ptr[old_u + 1]
            for edge_i in range(start, end):
                old_v = self.col_idx[edge_i]
                if old_v in idx_set:
                    new_col_idx.append(old_to_new[old_v])
                    new_weights.append(self.weights[edge_i])
                    new_contacts.append(self.contacts[edge_i])
            new_row_ptr.append(len(new_col_idx))

        sub_neuron_ids = np.array([self.neuron_ids[i] for i in selected_indices])
        sub_cell_types = [self.cell_types[i] for i in selected_indices]
        sub_hemispheres = [self.hemispheres[i] for i in selected_indices]

        return ConnectomeGraph(
            num_neurons=k,
            row_ptr=np.array(new_row_ptr, dtype=np.uint32),
            col_idx=np.array(new_col_idx, dtype=np.uint32),
            weights=np.array(new_weights, dtype=np.float64),
            contacts=np.array(new_contacts, dtype=np.uint32),
            neuron_ids=sub_neuron_ids,
            cell_types=sub_cell_types,
            hemispheres=sub_hemispheres,
            provenance=Provenance(
                tier="experimental_assumption",
                source=f"InducedSubgraph({self.provenance.source})",
                confidence=self.provenance.confidence,
                rationale="Induced subgraph slice for localized testing",
            ),
        )


def create_synthetic_motif_graph(
    num_neurons: int = 50,
    motif_type: str = "recurrent_inhibition",
    seed: int = 42,
) -> ConnectomeGraph:
    """Generate a reproducible, small synthetic motif graph for Phase 0 numerical testing.

    Supported motifs:
      - 'feedforward_chain': unidirectional transmission with lateral fan-out
      - 'recurrent_inhibition': excitatory principal cells with recurrent inhibitory interneurons
      - 'lateral_inhibition': competing columns with mutual inhibition
    """
    rng = np.random.default_rng(seed)

    row_list: List[int] = []
    col_list: List[int] = []
    weight_list: List[float] = []
    contact_list: List[int] = []

    cell_types: List[str] = []
    hemispheres: List[str] = []
    neuron_ids = np.arange(num_neurons, dtype=np.uint64)

    if motif_type == "feedforward_chain":
        for i in range(num_neurons):
            cell_types.append("L1" if i < num_neurons // 2 else "Tm1")
            hemispheres.append("L" if i % 2 == 0 else "R")
            if i + 1 < num_neurons:
                row_list.append(i)
                col_list.append(i + 1)
                weight_list.append(1.5)
                contact_list.append(8)
            if i + 2 < num_neurons:
                row_list.append(i)
                col_list.append(i + 2)
                weight_list.append(0.75)
                contact_list.append(4)

    elif motif_type == "recurrent_inhibition":
        num_excitatory = int(num_neurons * 0.8)
        for i in range(num_neurons):
            is_exc = i < num_excitatory
            cell_types.append("Principal" if is_exc else "GABA_Interneuron")
            hemispheres.append("L" if i % 2 == 0 else "R")

        # Excitatory recurrent and feedforward to inhibitory
        for i in range(num_excitatory):
            # Connect to next excitatory
            next_e = (i + 1) % num_excitatory
            row_list.append(i)
            col_list.append(next_e)
            weight_list.append(rng.uniform(0.5, 1.2))
            contact_list.append(rng.integers(3, 10))

            # Connect to random inhibitory
            inh_target = num_excitatory + (i % (num_neurons - num_excitatory))
            row_list.append(i)
            col_list.append(inh_target)
            weight_list.append(rng.uniform(0.8, 1.5))
            contact_list.append(rng.integers(5, 15))

        # Inhibitory feedback to excitatory
        for j in range(num_excitatory, num_neurons):
            # Connect back to 3 excitatory cells with negative weight (conductance representation)
            targets = rng.choice(num_excitatory, size=min(3, num_excitatory), replace=False)
            for t in targets:
                row_list.append(j)
                col_list.append(int(t))
                weight_list.append(-rng.uniform(0.5, 1.0))
                contact_list.append(rng.integers(4, 12))

    else:
        raise ValueError(f"Unknown motif type: {motif_type}")

    # Build CSR arrays
    row_arr = np.array(row_list, dtype=np.int64)
    col_arr = np.array(col_list, dtype=np.int64)
    weight_arr = np.array(weight_list, dtype=np.float64)
    contact_arr = np.array(contact_list, dtype=np.uint32)

    # Sort edges by row, then col
    sort_idx = np.lexsort((col_arr, row_arr))
    row_arr = row_arr[sort_idx]
    col_arr = col_arr[sort_idx]
    weight_arr = weight_arr[sort_idx]
    contact_arr = contact_arr[sort_idx]

    # Compute row_ptr
    row_ptr = np.zeros(num_neurons + 1, dtype=np.uint32)
    if len(row_arr) > 0:
        counts = np.bincount(row_arr, minlength=num_neurons)
        row_ptr[1:] = np.cumsum(counts)

    return ConnectomeGraph(
        num_neurons=num_neurons,
        row_ptr=row_ptr,
        col_idx=col_arr.astype(np.uint32),
        weights=weight_arr,
        contacts=contact_arr,
        neuron_ids=neuron_ids,
        cell_types=cell_types,
        hemispheres=hemispheres,
        provenance=Provenance(
            tier="experimental_assumption",
            source=f"SyntheticMotif({motif_type})",
            confidence=1.0,
            rationale="Reproducible synthetic microcircuit motif for numerical and topological unit tests",
        ),
    )
