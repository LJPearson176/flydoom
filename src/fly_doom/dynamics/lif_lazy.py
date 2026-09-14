"""High-performance Python ctypes wrapper for the Lazy Exact Subthreshold C++ LIF simulation kernel."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from fly_doom.connectome.graph import ConnectomeGraph
from fly_doom.core.provenance import Provenance


def _find_native_lib() -> Path:
    current_dir = Path(__file__).parent
    native_dir = current_dir / "native"
    target_ext = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    target = native_dir / f"liblif_native{target_ext}"
    if target.exists():
        return target

    # Auto-compile if not present
    import shutil
    import subprocess
    import sys
    cpp_source = native_dir / "lif_native.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise FileNotFoundError(f"Native LIF library not found in {native_dir} and no C++ compiler available")
    cmd = [compiler, "-O3", "-std=c++20", "-shared", "-fPIC"]
    if sys.platform == "darwin":
        cmd.extend(["-undefined", "dynamic_lookup"])
    cmd.extend([str(cpp_source), "-o", str(target)])
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return target



_LIB_PATH = _find_native_lib()
_LIB = ctypes.CDLL(str(_LIB_PATH))

_LIB.neural_advance.restype = None
_LIB.neural_advance.argtypes = [
    ctypes.c_int,  # n
    ctypes.POINTER(ctypes.c_int64),  # ptr
    ctypes.POINTER(ctypes.c_int32),  # post
    ctypes.POINTER(ctypes.c_float),  # weight
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # g
    ctypes.POINTER(ctypes.c_int16),  # refractory
    ctypes.POINTER(ctypes.c_float),  # drive
    ctypes.POINTER(ctypes.c_float),  # previous_drive
    ctypes.POINTER(ctypes.c_int32),  # queue
    ctypes.POINTER(ctypes.c_int32),  # queue_count
    ctypes.POINTER(ctypes.c_int64),  # clock
    ctypes.c_int,  # steps
    ctypes.c_float,  # dt
    ctypes.POINTER(ctypes.c_int32),  # counts
    ctypes.POINTER(ctypes.c_int32),  # active
    ctypes.POINTER(ctypes.c_uint8),  # flags
    ctypes.POINTER(ctypes.c_int32),  # nactive
    ctypes.POINTER(ctypes.c_int64),  # last
]


class LIFLazyEngine:
    """High-performance Lazy Exact Subthreshold LIF simulation engine.

    Adapted from fly_ocr (Nature-grade exact analytical continuous LIF evolution).
    Inactive neurons are lazily skipped via convex relaxation until presynaptic
    spikes awaken them or sensory drive changes. Preserves every single connectome edge.
    """

    def __init__(
        self,
        graph: Union[ConnectomeGraph, Dict[str, np.ndarray]],
        dt: float = 0.1,
    ):
        self.dt = float(dt)
        if isinstance(graph, ConnectomeGraph):
            self.num_neurons = graph.num_neurons
            self.num_edges = graph.num_edges
            self._ptr = np.ascontiguousarray(graph.row_ptr, dtype=np.int64)
            self._post = np.ascontiguousarray(graph.col_idx, dtype=np.int32)
            self._weight = np.ascontiguousarray(graph.weights, dtype=np.float32)
        elif isinstance(graph, dict):
            self._ptr = np.ascontiguousarray(graph["ptr"], dtype=np.int64)
            self._post = np.ascontiguousarray(graph["post"], dtype=np.int32)
            self._weight = np.ascontiguousarray(graph["weight"], dtype=np.float32)
            self.num_neurons = len(self._ptr) - 1
            self.num_edges = len(self._post)
        else:
            raise TypeError(f"Unsupported graph type: {type(graph)}")

        self.delay_steps = int(round(1.8 / self.dt))
        self.slots = self.delay_steps + 1
        self.rfc_steps = int(round(2.2 / self.dt))

        self.reset()

        self.provenance = Provenance(
            tier="computational_hypothesis",
            source="LazyExactSubthresholdKernel",
            confidence=0.95,
            rationale="Exact continuous analytical subthreshold integration with event-driven lazy skipping (fly_ocr)",
        )

    def reset(self) -> None:
        """Reset all neural states to baseline rest."""
        n = self.num_neurons
        self._v = np.full(n, -52.0, dtype=np.float32)
        self._g = np.zeros(n, dtype=np.float32)
        self._refractory = np.zeros(n, dtype=np.int16)
        self._drive = np.zeros(n, dtype=np.float32)
        self._previous_drive = np.zeros(n, dtype=np.float32)
        self._queue = np.zeros((self.slots, n), dtype=np.int32)
        self._queue_count = np.zeros(self.slots, dtype=np.int32)
        self._clock = np.zeros(1, dtype=np.int64)
        self._counts = np.zeros(n, dtype=np.int32)
        self._active = np.arange(n, dtype=np.int32)
        self._flags = np.ones(n, dtype=np.uint8)
        self._nactive = np.array([n], dtype=np.int32)
        self._last = np.full(n, -1, dtype=np.int64)

    @property
    def v(self) -> np.ndarray:
        """Current membrane potentials (mV)."""
        return self._v.copy()

    @property
    def g(self) -> np.ndarray:
        """Current synaptic conductances."""
        return self._g.copy()

    @property
    def counts(self) -> np.ndarray:
        """Spike counts recorded during the most recent advance call."""
        return self._counts.copy()

    @property
    def active_count(self) -> int:
        """Number of neurons currently active in the simulation."""
        return int(self._nactive[0])

    @property
    def active_fraction(self) -> float:
        """Fraction of neurons currently active in the simulation."""
        return float(self._nactive[0]) / float(self.num_neurons) if self.num_neurons > 0 else 0.0

    @property
    def clock(self) -> int:
        """Current global simulation clock step."""
        return int(self._clock[0])

    @property
    def step_count(self) -> int:
        """Alias for clock steps."""
        return int(self._clock[0])

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Create a deep copy snapshot of the current state."""
        return {
            "v": self._v.copy(),
            "g": self._g.copy(),
            "refractory": self._refractory.copy(),
            "drive": self._drive.copy(),
            "previous_drive": self._previous_drive.copy(),
            "queue": self._queue.copy(),
            "queue_count": self._queue_count.copy(),
            "clock": self._clock.copy(),
            "counts": self._counts.copy(),
            "active": self._active.copy(),
            "flags": self._flags.copy(),
            "nactive": self._nactive.copy(),
            "last": self._last.copy(),
        }

    def restore(self, snapshot: Dict[str, np.ndarray]) -> None:
        """Restore neural state from a snapshot."""
        for key in (
            "v", "g", "refractory", "drive", "previous_drive",
            "queue", "queue_count", "clock", "counts", "active",
            "flags", "nactive", "last"
        ):
            if key not in snapshot:
                raise ValueError(f"Missing key in state snapshot: {key}")
            target = getattr(self, f"_{key}")
            np.copyto(target, snapshot[key])

    def advance(self, drive: Optional[np.ndarray] = None, duration_ms: float = 1.0) -> np.ndarray:
        """Advance simulation by duration_ms given external drive currents.

        Returns:
            np.ndarray: Vector of spike counts for each neuron generated during this window.
        """
        steps = int(round(duration_ms / self.dt))
        if steps < 1:
            return np.zeros(self.num_neurons, dtype=np.int32)

        if drive is not None:
            self._drive[:] = np.ascontiguousarray(drive, dtype=np.float32)

        self._counts.fill(0)

        _LIB.neural_advance(
            self.num_neurons,
            self._ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            self._post.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._refractory.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            self._drive.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._previous_drive.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._queue.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._queue_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._clock.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            steps,
            ctypes.c_float(self.dt),
            self._counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._active.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._nactive.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._last.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )

        return self._counts.copy()

    def step(self, external_current: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance by exactly 1 time step (dt).

        Returns:
            np.ndarray: Indices of neurons that fired during this step.
        """
        counts = self.advance(drive=external_current, duration_ms=self.dt)
        return np.where(counts > 0)[0]
