"""Tests for MaleCNS biological extraction schemas and provenance validation."""

import pytest
from fly_doom.connectome.malecns_extractor import RealEMSynapse, MaleCNST4aExtraction


def test_real_em_synapse_coordinates():
    syn = RealEMSynapse(
        synapse_id=987654,
        pre_body_id=12345,
        pre_cell_type="Mi1",
        post_body_id=67890,
        post_cell_type="T4a",
        x_nm=125400.0,
        y_nm=84200.0,
        z_nm=45100.0,
        synapse_confidence=0.98,
        partner_classification_confidence=1.0,
        neuprint_roi="ME(R)",
    )

    assert syn.x_um == 125.4
    assert syn.y_um == 84.2
    assert syn.z_um == 45.1


def test_malecns_extraction_provenance():
    syn = RealEMSynapse(
        synapse_id=987654,
        pre_body_id=12345,
        pre_cell_type="Mi1",
        post_body_id=67890,
        post_cell_type="T4a",
        x_nm=125400.0,
        y_nm=84200.0,
        z_nm=45100.0,
        synapse_confidence=0.98,
        partner_classification_confidence=0.95,
        neuprint_roi="ME(R)",
    )

    extraction = MaleCNST4aExtraction(
        t4a_body_id=67890,
        hemisphere="R",
        dataset_version="v1.0",
        synapses=[syn],
        partner_counts={"Mi1": 1},
    )

    assert extraction.provenance.tier == "biological_reconstruction"
    assert extraction.provenance.confidence == 1.0
    assert extraction.average_synapse_confidence == 0.98
    assert extraction.average_classification_confidence == 0.95
    centroids = extraction.compute_spatial_centroids()
    assert "Mi1" in centroids
    assert centroids["Mi1"] == (125.4, 84.2, 45.1)
    fp = extraction.compute_fingerprint()
    assert len(fp) == 64


def test_typed_coordinate_conversions():
    from fly_doom.connectome.malecns_extractor import Coordinate

    # 1. Test 8 nm voxel to micron (125 voxels = 1.0 um)
    c_vox = Coordinate(x=125.0, y=250.0, z=500.0, unit="voxel_8nm", space="male_cns_em")
    c_um = c_vox.to_um()
    assert c_um.unit == "um"
    assert c_um.space == "male_cns_em"
    assert c_um.x == 1.0
    assert c_um.y == 2.0
    assert c_um.z == 4.0

    # 2. Test round-trip back to voxels
    c_back = c_um.to_voxel_8nm()
    assert c_back.x == 125.0
    assert c_back.y == 250.0
    assert c_back.z == 500.0

    # 3. Test nanometer conversion
    c_nm = Coordinate(x=1000.0, y=2000.0, z=4000.0, unit="nm", space="male_cns_em")
    assert c_nm.to_um().x == 1.0
    assert c_nm.to_voxel_8nm().x == 125.0


def test_swc_skeleton_morphology_and_distance():
    from fly_doom.connectome.malecns_extractor import SWCNode, SWCSkeleton

    nodes = [
        SWCNode(1, 1, 0.0, 0.0, 0.0, 100.0, -1, unit="voxel_8nm"),  # root soma (0,0,0)
        SWCNode(2, 3, 125.0, 0.0, 0.0, 50.0, 1, unit="voxel_8nm"),  # 1.0 um along X
        SWCNode(3, 3, 250.0, 0.0, 0.0, 30.0, 2, unit="voxel_8nm"),  # 2.0 um along X
        SWCNode(4, 3, 250.0, 125.0, 0.0, 20.0, 3, unit="voxel_8nm"),  # branch 1.0 um along Y
    ]
    skel = SWCSkeleton(body_id=12345, nodes=nodes)
    assert skel.total_nodes == 4
    assert skel.root_id == 1

    # Find nearest node to point (1.8, 0.1, 0.0) in um -> should be node 3 (2.0, 0, 0)
    nearest, dist = skel.find_nearest_node((1.8, 0.1, 0.0))
    assert nearest.node_id == 3
    assert round(dist, 2) == 0.22

    # Geodesic distance to soma
    # Node 1 -> 0
    assert skel.compute_geodesic_distance_to_soma(1) == 0.0
    # Node 3 -> 1.0 + 1.0 = 2.0 um
    assert skel.compute_geodesic_distance_to_soma(3) == 2.0
    # Node 4 -> 1.0 + 1.0 + 1.0 = 3.0 um
    assert skel.compute_geodesic_distance_to_soma(4) == 3.0


def test_discovery_gate_and_canonical_fixtures(tmp_path):
    from fly_doom.connectome.malecns_extractor import (
        MaleCNSDiscoveryGate,
        MaleCNSExtractionBundle,
        create_canonical_malecns_fixtures,
        derive_synapse_epistemic_triad,
    )

    candidates, skeleton, synapses = create_canonical_malecns_fixtures()
    assert len(candidates) >= 3
    assert len(synapses) == 138

    # Run Discovery Gate selection
    gate = MaleCNSDiscoveryGate()
    target, rationale = gate.select_canonical_target(candidates)
    assert target.body_id == 5813072001
    assert target.cell_type == "T4a"
    assert target.hemisphere == "R"
    assert "Selected Body ID 5813072001" in rationale

    # Derive tripartite annotations
    annotated = [derive_synapse_epistemic_triad(s, skeleton) for s in synapses]
    assert len(annotated) == 138

    # Verify tri-partite layers
    first = annotated[0]
    assert first.observed.dataset_version == "male-cns:v1.0"
    assert first.derived.distance_to_nearest_node_um >= 0.0
    assert first.derived.geodesic_distance_from_soma_um >= 0.0
    assert first.hypothesis.neurotransmitter in ("ACh", "GABA", "Glu")

    # Export bundle to temp directory
    bundle = MaleCNSExtractionBundle(
        query_spec=gate.generate_query_spec(),
        candidates=candidates,
        selected_target=target,
        selection_rationale=rationale,
        skeleton=skeleton,
        synapses=annotated,
    )
    files = bundle.export_bundle(tmp_path)

    assert "query.yaml" in files
    assert "candidates.json" in files
    assert "selected_neurons.json" in files
    assert f"skeleton/{target.body_id}.swc" in files
    assert "anatomy_manifest.json" in files
    assert "provenance.json" in files
    assert "fingerprints.json" in files
    if "synapses.parquet" in files:
        assert (tmp_path / "synapses.parquet").exists()
