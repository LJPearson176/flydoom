"""Tests for experiment bundle packaging and gate reporting."""

import json
from pathlib import Path
import pytest
from fly_doom.connectome.graph import create_synthetic_motif_graph
from fly_doom.core.provenance import Provenance, ProvenanceRegistry
from fly_doom.experiment.bundle import ExperimentBundle


def test_experiment_bundle_generation(tmp_path: Path):
    reg = ProvenanceRegistry()
    reg.register(
        "synthetic_graph",
        Provenance(
            tier="biological_reconstruction",
            source="Synthetic_Generator",
            confidence=1.0,
            rationale="Unit test graph",
        ),
    )

    graph = create_synthetic_motif_graph(20, "feedforward_chain")
    fp = graph.compute_fingerprint()

    bundle = ExperimentBundle(
        run_id="run_test_001",
        experiment_name="Phase0_Verification",
        config={"dt": 1.0, "tau_m": 20.0},
        provenance_registry=reg,
        dataset_fingerprint=fp,
        metrics={"total_steps": 1000, "firing_rate_hz": 12.4},
    )

    # Add a passing gate and a failing gate (verifying negative results are recorded)
    bundle.add_gate(
        name="parity_gate",
        passed=True,
        observed_value="1.2e-14",
        threshold="1e-9",
        rationale="C++ vs Python voltage error",
    )
    bundle.add_gate(
        name="synthetic_negative_result_gate",
        passed=False,
        observed_value=0.42,
        threshold=0.80,
        rationale="Demonstrating negative result as first-class citizen",
    )

    saved_dir = bundle.save(tmp_path)

    assert (saved_dir / "config.yaml").exists()
    assert (saved_dir / "manifest.json").exists()
    assert (saved_dir / "provenance.json").exists()
    assert (saved_dir / "validation_gates.json").exists()
    assert (saved_dir / "metrics.json").exists()
    assert (saved_dir / "report.md").exists()

    with open(saved_dir / "validation_gates.json") as f:
        gates = json.load(f)
    assert len(gates) == 2
    assert gates[0]["passed"] is True
    assert gates[1]["passed"] is False

    with open(saved_dir / "report.md") as f:
        report_text = f.read()
    assert "FAILED GATES DETECTED" in report_text
    assert "synthetic_negative_result_gate" in report_text
