"""Tests for three-tier provenance enforcement."""

import pytest
from fly_doom.core.provenance import (
    Provenance,
    ProvenanceRegistry,
    attach_provenance,
    get_provenance,
)


def test_provenance_valid_tiers():
    p1 = Provenance(
        tier="biological_reconstruction",
        source="MaleCNS_v1.0",
        confidence=1.0,
        rationale="EM reconstructed wiring",
    )
    assert p1.tier == "biological_reconstruction"

    p2 = Provenance(
        tier="computational_hypothesis",
        source="LIF_dynamics",
        confidence=0.8,
        rationale="Point-neuron leaky integrate-and-fire model",
    )
    assert p2.tier == "computational_hypothesis"

    p3 = Provenance(
        tier="engineering_scaffold",
        source="DNp20_to_yaw",
        confidence=0.5,
        rationale="Engineered controller mapping",
    )
    assert p3.tier == "engineering_scaffold"

    p4 = Provenance(
        tier="source_derived_fixture",
        source="MaleCNS_v1.0_prototype_fixture",
        confidence=0.85,
        rationale="Literature-derived prototype arbor",
    )
    assert p4.tier == "source_derived_fixture"


def test_provenance_invalid_tier():
    with pytest.raises(ValueError, match="Invalid provenance tier"):
        Provenance(
            tier="metaphysical_truth",  # type: ignore
            source="speculation",
            confidence=1.0,
            rationale="invalid",
        )


def test_provenance_invalid_confidence():
    with pytest.raises(ValueError, match="Confidence must be in"):
        Provenance(
            tier="biological_reconstruction",
            source="source",
            confidence=1.5,
            rationale="too high",
        )


def test_provenance_registry():
    reg = ProvenanceRegistry()
    p_bio = Provenance(
        tier="biological_reconstruction",
        source="MaleCNS_v1.0",
        confidence=1.0,
        rationale="Anatomy",
    )
    p_scaffold = Provenance(
        tier="engineering_scaffold",
        source="Retina_proxy",
        confidence=0.6,
        rationale="Screen downsample",
    )

    reg.register("connectome", p_bio)
    reg.register("retina", p_scaffold)

    audit = reg.audit_tier_counts()
    assert audit["biological_reconstruction"] == 1
    assert audit["engineering_scaffold"] == 1
    assert audit["computational_hypothesis"] == 0

    with pytest.raises(KeyError, match="already registered"):
        reg.register("connectome", p_bio)


def test_provenance_decorator():
    @attach_provenance(
        tier="computational_hypothesis",
        source="Paper_A",
        confidence=0.9,
        rationale="Hypothesis X",
    )
    class TestModule:
        pass

    prov = get_provenance(TestModule)
    assert prov is not None
    assert prov.tier == "computational_hypothesis"
    assert prov.confidence == 0.9
