from pathlib import Path
import pytest

from fly_doom.connectome.high_fidelity_pathways import (
    HighFidelityPathwayBundle,
    build_high_fidelity_pathway_bundle,
)


def test_high_fidelity_pathway_bundle_generation():
    bundle = build_high_fidelity_pathway_bundle()
    assert isinstance(bundle, HighFidelityPathwayBundle)
    assert len(bundle.neurons) == 14
    assert len(bundle.synapses) >= 260
    assert len(bundle.cartridges) == 128

    # Test neuron types
    cell_types = {n.cell_type for n in bundle.neurons}
    expected_types = {
        "T4a", "Mi1", "Tm3", "Mi4", "Mi9", "E-PG", "DNpe017",
        "T5a", "LC4", "P-EN", "KC", "MBON", "PPL1", "T2_Motor",
    }
    assert expected_types <= cell_types

    # Test tree topology
    for neuron in bundle.neurons:
        assert neuron.total_arbor_length_um > 0
        node_ids = {node.node_id for node in neuron.nodes}
        for node in neuron.nodes:
            if node.parent_id != -1:
                assert node.parent_id in node_ids, f"Invalid parent {node.parent_id} in {neuron.instance}"


def test_synapse_neurotransmitter_and_coordinates():
    bundle = build_high_fidelity_pathway_bundle()
    transmitters = bundle.synapse_counts_by_transmitter
    assert transmitters["ACh"] >= 150
    assert transmitters["GABA"] >= 20
    assert transmitters["Glu"] >= 15
    assert transmitters["Dopamine"] >= 8

    for s in bundle.synapses:
        assert s.synapse_id > 0
        assert s.confidence > 0.6
        assert s.color_hex.startswith("#")
        assert s.action in {"excitatory", "inhibitory_shunting", "inhibitory_hyperpolarizing", "modulatory"}


def test_retinotopic_cartridges_bilateral_symmetry():
    bundle = build_high_fidelity_pathway_bundle()
    left_carts = [c for c in bundle.cartridges if c.eye == "L"]
    right_carts = [c for c in bundle.cartridges if c.eye == "R"]
    assert len(left_carts) == 64
    assert len(right_carts) == 64

    # Test angular ranges
    azimuths = [c.azimuth_deg for c in bundle.cartridges]
    assert min(azimuths) < -15.0
    assert max(azimuths) > 15.0


def test_bundle_serialization_dictionary():
    bundle = build_high_fidelity_pathway_bundle()
    data = bundle.to_dict()
    assert "metadata" in data
    assert "neurons" in data
    assert "synapses" in data
    assert "cartridges" in data
    assert data["metadata"]["total_neurons"] == len(bundle.neurons)
    assert data["metadata"]["total_synapses"] == len(bundle.synapses)
