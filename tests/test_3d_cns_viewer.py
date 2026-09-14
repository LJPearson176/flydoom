"""Tests for the 3D Visible Drosophila Nervous System & Exoskeleton Twin."""

from __future__ import annotations

import json
from pathlib import Path


def test_authentic_brain_mesh_structure():
    mesh_path = Path("web/authentic_fly_cns_mesh.json")
    assert mesh_path.exists(), "authentic_fly_cns_mesh.json must exist"
    data = json.loads(mesh_path.read_text())
    assert "brain_surface_edges" in data
    assert len(data["brain_surface_edges"]) == 6654, "Must contain exactly 6,654 JFRC2 surface edges"
    assert "neuropils" in data
    assert len(data["neuropils"]) >= 12, "Must contain at least 12 neuropil meshes (bilateral optic, CX, SEZ, AL)"

    codes = {np["code"] for np in data["neuropils"]}
    expected_codes = {"ME_R", "ME_L", "LOP_R", "LOP_L", "LO_R", "LO_L", "EB", "FB", "PB", "GNG_SEZ", "AL_R", "AL_L"}
    assert expected_codes <= codes, f"Missing expected neuropils: {expected_codes - codes}"


def test_fly_3d_cns_model_structure():
    model_path = Path("web/fly_3d_cns_model.json")
    assert model_path.exists(), "fly_3d_cns_model.json must exist"
    model = json.loads(model_path.read_text())
    assert "exoskeleton" in model
    assert len(model["exoskeleton"]["lines"]) >= 700, "Exoskeleton must have at least 700 cuticle lines"
    assert "connectome" in model
    assert len(model["connectome"]["fibers"]) == 3030, "Must have exactly 3,030 FlyWire connectome fibers"
    assert "high_fidelity_pathways" in model
    assert len(model["high_fidelity_pathways"]["neurons"]) == 14
    assert len(model["high_fidelity_pathways"]["synapses"]) == 262


def test_compiled_3d_viewer_html():
    html_path = Path("web/fly_3d_visible_nervous_system.html")
    assert html_path.exists(), "fly_3d_visible_nervous_system.html must exist"
    content = html_path.read_text()
    assert "FULL 3D VISIBLE CNS TWIN" in content
    assert "JFRC2 BRAIN MESH" in content
    assert "LIVE GZDOOM / TELEMETRY" in content
    assert "pollLiveTelemetry" in content
    assert "checkNeuropilHover" in content
    assert "NEUROPIL_INFO" in content
    assert "high_fidelity_pathways" in content
    assert "CONNECTOME GRANULARITY" in content
    assert "SYNAPSE & MORPHOLOGY INSPECTOR" in content
    assert "setGranularity" in content
    assert "inspectSynapse" in content
    assert "inspectNeuron" in content
    assert "toggleMatrixModal" in content
    assert "wiringMatrixModal" in content
    assert "synapticAdjacencyTable" in content
    assert "updateMatrixModal" in content
    assert "onMatrixCellClick" in content
    assert "setLesionMode" in content
    assert "stage1_waypoints" in content
    assert "stage2_waypoints" in content
    assert "STAGE 1 TOPOLOGICAL WAYPOINTS" in content
    assert "STAGE 2 TOPOLOGICAL WAYPOINTS" in content
    assert "setWaypointStage" in content
    assert "btnStage1" in content
    assert "btnStage2" in content
    assert "MUSHROOM BODY VALENCE (V_MB)" in content
    assert "btnPresetFlyGun" in content
    assert "chkFlyShotgun" in content
    assert "drawEmbodiedFlyWithShotgun" in content
    assert "joined_gameplay_3d_viewer.html" in content
    assert "3D Fly &amp; DOOM Shotgun" in content or "3D Fly & DOOM Shotgun" in content


def test_observatory_index_integration():
    index_path = Path("web/index.html")
    assert index_path.exists()
    content = index_path.read_text()
    assert 'id="cns3DContainer"' in content, "cns3DContainer must be present in index.html"
    assert "fly_3d_visible_nervous_system.html" in content
    assert 'data-mode="brain"' in content
