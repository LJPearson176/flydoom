"""Python ctypes wrapper for native C++20 LIF simulation engine."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np

from fly_doom.connectome.graph import ConnectomeGraph
from fly_doom.core.provenance import Provenance
from fly_doom.dynamics.lif_reference import LIFParameters


class _NativeLIFParams(ctypes.Structure):
    _fields_ = [
        ("v_rest", ctypes.c_double),
        ("v_reset", ctypes.c_double),
        ("v_thresh", ctypes.c_double),
        ("tau_m", ctypes.c_double),
        ("tau_syn", ctypes.c_double),
        ("t_ref", ctypes.c_double),
        ("dt", ctypes.c_double),
        ("r_m", ctypes.c_double),
    ]


def _find_native_lib() -> Path:
    current_dir = Path(__file__).parent
    candidate = current_dir / "native" / "liblif_native.dylib"
    if candidate.exists():
        return candidate
    candidate_so = current_dir / "native" / "liblif_native.so"
    if candidate_so.exists():
        return candidate_so
    raise FileNotFoundError(f"Native LIF library not found in {current_dir / 'native'}")


_LIB_PATH = _find_native_lib()
_LIB = ctypes.CDLL(str(_LIB_PATH))

# Function prototypes
_LIB.lif_create_engine.restype = ctypes.c_void_p
_LIB.lif_create_engine.argtypes = [
    ctypes.c_uint32,  # num_neurons
    ctypes.c_uint32,  # num_edges
    ctypes.POINTER(ctypes.c_uint32),  # row_ptr
    ctypes.POINTER(ctypes.c_uint32),  # col_idx
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.POINTER(_NativeLIFParams),  # params
]

_LIB.lif_destroy_engine.restype = None
_LIB.lif_destroy_engine.argtypes = [ctypes.c_void_p]

_LIB.lif_step.restype = ctypes.c_uint32
_LIB.lif_step.argtypes = [
    ctypes.c_void_p,  # engine
    ctypes.POINTER(ctypes.c_double),  # external_current
    ctypes.POINTER(ctypes.c_uint32),  # out_spikes
    ctypes.c_uint32,  # max_spikes
]

_LIB.lif_get_v.restype = ctypes.POINTER(ctypes.c_double)
_LIB.lif_get_v.argtypes = [ctypes.c_void_p]

_LIB.lif_get_i_syn.restype = ctypes.POINTER(ctypes.c_double)
_LIB.lif_get_i_syn.argtypes = [ctypes.c_void_p]

_LIB.lif_get_refractory.restype = ctypes.POINTER(ctypes.c_int32)
_LIB.lif_get_refractory.argtypes = [ctypes.c_void_p]

_LIB.lif_get_step_count.restype = ctypes.c_uint64
_LIB.lif_get_step_count.argtypes = [ctypes.c_void_p]

_LIB.lif_set_state.restype = None
_LIB.lif_set_state.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
]


class LIFNativeEngine:
    """High-performance native C++20 LIF simulation engine."""

    def __init__(self, graph: ConnectomeGraph, params: Optional[LIFParameters] = None):
        self.graph = graph
        self.params = params or LIFParameters()
        self.num_neurons = graph.num_neurons
        self.num_edges = graph.num_edges

        # Ensure contiguous data layouts
        self._row_ptr = np.ascontiguousarray(graph.row_ptr, dtype=np.uint32)
        self._col_idx = np.ascontiguousarray(graph.col_idx, dtype=np.uint32)
        self._weights = np.ascontiguousarray(graph.weights, dtype=np.float64)

        c_params = _NativeLIFParams(
            v_rest=self.params.v_rest,
            v_reset=self.params.v_reset,
            v_thresh=self.params.v_thresh,
            tau_m=self.params.tau_m,
            tau_syn=self.params.tau_syn,
            t_ref=self.params.t_ref,
            dt=self.params.dt,
            r_m=self.params.r_m,
        )

        self._engine_ptr = _LIB.lif_create_engine(
            self.num_neurons,
            self.num_edges,
            self._row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self._col_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self._weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(c_params),
        )
        if not self._engine_ptr:
            raise RuntimeError("Failed to create native C++ LIF engine instance")

        self._spike_buffer = np.zeros(self.num_neurons, dtype=np.uint32)
        self.provenance = Provenance(
            tier="computational_hypothesis",
            source="Native_LIF_Engine_CPP20",
            confidence=0.85,
            rationale="Native C++20 implementation of LIF dynamics with strict numerical equivalence to reference model",
        )

    def __del__(self) -> None:
        if hasattr(self, "_engine_ptr") and self._engine_ptr:
            _LIB.lif_destroy_engine(self._engine_ptr)
            self._engine_ptr = None

    @property
    def step_count(self) -> int:
        return int(_LIB.lif_get_step_count(self._engine_ptr))

    @property
    def v(self) -> np.ndarray:
        ptr = _LIB.lif_get_v(self._engine_ptr)
        arr = np.ctypeslib.as_array(ptr, shape=(self.num_neurons,))
        return arr.copy()

    @property
    def i_syn(self) -> np.ndarray:
        ptr = _LIB.lif_get_i_syn(self._engine_ptr)
        arr = np.ctypeslib.as_array(ptr, shape=(self.num_neurons,))
        return arr.copy()

    @property
    def refractory_timer(self) -> np.ndarray:
        ptr = _LIB.lif_get_refractory(self._engine_ptr)
        arr = np.ctypeslib.as_array(ptr, shape=(self.num_neurons,))
        return arr.copy()

    def step(self, external_current: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute one simulation step of duration dt."""
        ext_ptr = None
        if external_current is not None:
            ext_c = np.ascontiguousarray(external_current, dtype=np.float64)
            ext_ptr = ext_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        spikes_ptr = self._spike_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        spike_count = _LIB.lif_step(
            self._engine_ptr,
            ext_ptr,
            spikes_ptr,
            self.num_neurons,
        )

        return self._spike_buffer[:spike_count].astype(np.int64).copy()

    def run(
        self,
        num_steps: int,
        stimulus_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run simulation for num_steps, recording voltages and spike trains."""
        v_trace: List[np.ndarray] = []
        spike_trace: List[np.ndarray] = []

        for t in range(num_steps):
            ext_i = stimulus_fn(t) if stimulus_fn else None
            spikes = self.step(ext_i)
            v_trace.append(self.v)
            spike_trace.append(spikes)

        return v_trace, spike_trace
