"""Core infrastructure, typing, and provenance for FlyDoom."""

from fly_doom.core.provenance import (
    Provenance,
    ProvenanceRegistry,
    ProvenanceTier,
    attach_provenance,
    get_provenance,
)

__all__ = [
    "Provenance",
    "ProvenanceRegistry",
    "ProvenanceTier",
    "attach_provenance",
    "get_provenance",
]
