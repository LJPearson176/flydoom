"""Type-enforced provenance system for FlyDoom.

Enforces strict separation between:
1. biological_reconstruction (empirical EM observations, cell anatomy, synapses)
2. computational_hypothesis (biophysical dynamics, network hypotheses, plasticity rules)
3. engineering_scaffold (proxy retinal arrays, controller bindings, discrete ticks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, cast

ProvenanceTier = Literal[
    "biological_reconstruction",
    "biological_evidence",
    "experimental_assumption",
    "computational_hypothesis",
    "engineering_scaffold",
]

VALID_PROVENANCE_TIERS = {
    "biological_reconstruction",
    "biological_evidence",
    "experimental_assumption",
    "computational_hypothesis",
    "engineering_scaffold",
}


class ProvenanceTierEnum(str, Enum):
    BIOLOGICAL_RECONSTRUCTION = "biological_reconstruction"
    BIOLOGICAL_EVIDENCE = "biological_evidence"
    EXPERIMENTAL_ASSUMPTION = "experimental_assumption"
    COMPUTATIONAL_HYPOTHESIS = "computational_hypothesis"
    ENGINEERING_SCAFFOLD = "engineering_scaffold"


@dataclass(frozen=True)
class Provenance:
    """Immutable provenance record anchored to an explicit scientific tier."""

    tier: ProvenanceTier
    source: str
    confidence: float
    rationale: str
    doi: Optional[str] = None
    figure_table_ref: Optional[str] = None
    access_date: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tier not in VALID_PROVENANCE_TIERS:
            raise ValueError(f"Invalid provenance tier: {self.tier}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if not self.source.strip():
            raise ValueError("Source citation/identifier cannot be empty")
        if not self.rationale.strip():
            raise ValueError("Rationale justification cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "source": self.source,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "doi": self.doi,
            "figure_table_ref": self.figure_table_ref,
            "access_date": self.access_date,
        }


T = TypeVar("T")


def attach_provenance(
    tier: ProvenanceTier,
    source: str,
    confidence: float,
    rationale: str,
) -> Callable[[T], T]:
    """Decorator to attach immutable provenance to classes or functions."""
    prov = Provenance(
        tier=tier, source=source, confidence=confidence, rationale=rationale
    )

    def decorator(obj: T) -> T:
        setattr(obj, "__provenance__", prov)
        return obj

    return decorator


def get_provenance(obj: Any) -> Optional[Provenance]:
    """Retrieve provenance from an annotated object or instance."""
    if hasattr(obj, "__provenance__"):
        return cast(Provenance, getattr(obj, "__provenance__"))
    if hasattr(obj, "provenance"):
        p = getattr(obj, "provenance")
        if isinstance(p, Provenance):
            return p
    return None


@dataclass
class ProvenanceRegistry:
    """Global or per-experiment registry tracking provenance across components."""

    entries: Dict[str, Provenance] = field(default_factory=dict)

    def register(self, component_name: str, provenance: Provenance) -> None:
        if component_name in self.entries:
            raise KeyError(f"Component '{component_name}' already registered in provenance registry")
        self.entries[component_name] = provenance

    def audit_tier_counts(self) -> Dict[str, int]:
        counts = {tier: 0 for tier in VALID_PROVENANCE_TIERS}
        for entry in self.entries.values():
            counts[entry.tier] += 1
        return counts

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.to_dict() for k, v in self.entries.items()}
