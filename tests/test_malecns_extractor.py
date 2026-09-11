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
        confidence=0.98,
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
        confidence=0.98,
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
    fp = extraction.compute_fingerprint()
    assert len(fp) == 64
