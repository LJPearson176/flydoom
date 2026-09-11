"""Neural dynamics engines (reference NumPy and native C++20)."""

from fly_doom.dynamics.lif_reference import (
    LIFParameters,
    LIFReferenceEngine,
    LIFState,
)
from fly_doom.dynamics.lif_native import LIFNativeEngine

__all__ = [
    "LIFParameters",
    "LIFReferenceEngine",
    "LIFState",
    "LIFNativeEngine",
]
