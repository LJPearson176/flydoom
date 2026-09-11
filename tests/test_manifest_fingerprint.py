"""Tests for connectome dataset fingerprinting and manifest verification."""

import pytest
from fly_doom.connectome.graph import create_synthetic_motif_graph
from fly_doom.connectome.manifest import DatasetManifest


def test_fingerprint_generation():
    graph = create_synthetic_motif_graph(50, "recurrent_inhibition", seed=42)
    fp = graph.compute_fingerprint()

    assert fp.neuron_count == 50
    assert fp.directed_edge_count == graph.num_edges
    assert fp.synapse_contact_count == graph.total_contacts
    assert len(fp.in_degree_histogram_hash) == 64
    assert len(fp.out_degree_histogram_hash) == 64
    assert len(fp.combined_hash) == 64


def test_manifest_verification_pass():
    graph = create_synthetic_motif_graph(50, "recurrent_inhibition", seed=42)
    fp = graph.compute_fingerprint()

    manifest = DatasetManifest(
        dataset_name="Test_Recurrent_Inhibition_50",
        dataset_version="v1.0",
        source_citation="Synthetic test generator",
        source_files_sha256={"test.csr": "dummy_sha256"},
        expected_fingerprint=fp,
    )

    # Should succeed without error
    graph.verify_against_manifest(manifest)


def test_manifest_verification_detects_tampering():
    graph = create_synthetic_motif_graph(50, "recurrent_inhibition", seed=42)
    fp = graph.compute_fingerprint()

    manifest = DatasetManifest(
        dataset_name="Test_Recurrent_Inhibition_50",
        dataset_version="v1.0",
        source_citation="Synthetic test generator",
        source_files_sha256={"test.csr": "dummy_sha256"},
        expected_fingerprint=fp,
    )

    # Corrupt graph by altering a weight and removing an edge
    corrupted_graph = create_synthetic_motif_graph(50, "feedforward_chain", seed=42)

    with pytest.raises(ValueError, match="Graph validation failed against manifest"):
        corrupted_graph.verify_against_manifest(manifest)
