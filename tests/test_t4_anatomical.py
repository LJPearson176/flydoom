"""Tests for MaleCNS single-unit T4 anatomical reconstruction and spatial segregation."""

import pytest
from fly_doom.connectome.t4_anatomical import (
    build_canonical_t4a_reconstruction,
    T4AnatomicalReconstruction,
)


def test_t4a_reconstruction_integrity():
    recon = build_canonical_t4a_reconstruction(seed=42)

    # 1. Integrity of target neuron
    assert recon.target_cell_id == 10001
    assert recon.target_cell_type == "T4a"
    assert recon.preferred_direction_cardinal == 0.0
    assert recon.total_synapses == 110  # 20 Mi4 + 15 Mi9 + 40 Mi1 + 35 Tm3

    # 2. Partner counts
    counts = recon.presynaptic_counts_by_type
    assert counts["Mi4"] == 20
    assert counts["Mi9"] == 15
    assert counts["Mi1"] == 40
    assert counts["Tm3"] == 35

    # 3. Compartment counts
    comp_counts = recon.presynaptic_counts_by_compartment
    assert comp_counts["leading"] == 35  # Mi4 (20) + Mi9 (15)
    assert comp_counts["central"] == 40  # Mi1 (40)
    assert comp_counts["trailing"] == 35  # Tm3 (35)


def test_t4a_spatial_segregation():
    recon = build_canonical_t4a_reconstruction(seed=42)

    # Centroids by compartment along preferred motion axis (x_um)
    comp_centroids = recon.compute_compartment_centroids()
    x_leading = comp_centroids["leading"][0]
    x_central = comp_centroids["central"][0]
    x_trailing = comp_centroids["trailing"][0]

    # Biological segregation condition: leading < central < trailing
    assert x_leading < x_central < x_trailing, f"Segregation failure: leading={x_leading}, central={x_central}, trailing={x_trailing}"
    assert x_leading < -2.5
    assert -1.0 < x_central < 1.0
    assert x_trailing > 2.5


def test_t4a_connectome_graph_conversion():
    recon = build_canonical_t4a_reconstruction(seed=42)
    graph, meta = recon.to_connectome_graph()

    # 12 presynaptic neurons (3 per type * 4 types) + 1 target T4a = 13 neurons
    assert graph.num_neurons == 13
    assert graph.num_edges == 12  # 12 incoming edges to T4a
    assert graph.total_contacts == 110
    assert graph.provenance.tier == "biological_reconstruction"
    assert meta["target_cell_type"] == "T4a"
