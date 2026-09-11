"""Optic lobe motion-detection subgraph extractor and canonical Phase-1 circuit."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from fly_doom.connectome.graph import ConnectomeGraph
from fly_doom.core.provenance import Provenance


# Canonical Drosophila motion cell classes
T4_TYPES = {"T4a", "T4b", "T4c", "T4d"}  # a: Right (0°), b: Left (180°), c: Up (90°), d: Down (270°)
T5_TYPES = {"T5a", "T5b", "T5c", "T5d"}
MEDULLA_ON_PARTNERS = {"Mi1", "Tm3", "Mi4", "Mi9"}
MEDULLA_OFF_PARTNERS = {"Tm1", "Tm2", "Tm4", "Tm9"}
LOBULA_PLATE_TYPES = {"HSN", "HSE", "HSS", "VS1", "VS2", "VS3"}


def query_cell_indices(graph: ConnectomeGraph, target_types: Set[str]) -> List[int]:
    """Find all neuron indices matching any of the specified target cell types."""
    return [i for i, ct in enumerate(graph.cell_types) if ct in target_types]


def create_canonical_phase1_optic_circuit(
    num_columns: int = 8,
    seed: int = 42,
) -> ConnectomeGraph:
    """Construct a canonical Phase-1 approximation circuit for motion electrophysiology.

    Architecture per column:
      - 2 Photoreceptor/Lamina inputs: L_ON (luminance increment), L_OFF (luminance decrement)
      - Medulla ON pathway:
          Mi1 (fast, non-delayed transmission to T4)
          Tm3 (delayed transmission to adjacent column T4)
      - Medulla OFF pathway:
          Tm1 (fast, non-delayed transmission to T5)
          Tm2 (delayed transmission to adjacent column T5)
      - Lobula Plate:
          T4a (prefers rightwards: 0°), T4b (prefers leftwards: 180°)
          T5a (prefers rightwards: 0°), T5b (prefers leftwards: 180°)
          HS_cell (integrates horizontal T4/T5 outputs)

    Biophysical mechanism:
      Spatial offset + differential delay produces directional coincidence detection.
    """
    rng = np.random.default_rng(seed)

    # We arrange columns linearly along x (0 to num_columns-1)
    # Neuron indexing scheme per column:
    # 0: L_ON, 1: L_OFF, 2: Mi1, 3: Tm3, 4: Tm1, 5: Tm2, 6: T4a, 7: T4b, 8: T5a, 9: T5b
    NEURONS_PER_COL = 10
    total_neurons = num_columns * NEURONS_PER_COL + 2  # +2 LPTCs: HS_Right, HS_Left
    hs_right_idx = num_columns * NEURONS_PER_COL
    hs_left_idx = hs_right_idx + 1

    cell_types: List[str] = []
    hemispheres: List[str] = []
    neuron_ids = np.arange(total_neurons, dtype=np.uint64)

    for col in range(num_columns):
        cell_types.extend([
            "L_ON",
            "L_OFF",
            "Mi1",
            "Tm3",
            "Tm1",
            "Tm2",
            "T4a",
            "T4b",
            "T5a",
            "T5b",
        ])
        hemispheres.extend(["R"] * NEURONS_PER_COL)

    cell_types.extend(["HS_Right", "HS_Left"])
    hemispheres.extend(["R", "R"])

    row_list: List[int] = []
    col_list: List[int] = []
    weight_list: List[float] = []
    contact_list: List[int] = []

    def add_synapse(pre: int, post: int, weight: float, contacts: int):
        row_list.append(pre)
        col_list.append(post)
        weight_list.append(weight)
        contact_list.append(contacts)

    for c in range(num_columns):
        base = c * NEURONS_PER_COL
        l_on = base + 0
        l_off = base + 1
        mi1 = base + 2
        tm3 = base + 3
        tm1 = base + 4
        tm2 = base + 5
        t4a = base + 6
        t4b = base + 7
        t5a = base + 8
        t5b = base + 9

        # 1. Lamina -> Medulla
        add_synapse(l_on, mi1, 15.0, 10)
        add_synapse(l_on, tm3, 15.0, 10)
        add_synapse(l_off, tm1, 15.0, 10)
        add_synapse(l_off, tm2, 15.0, 10)

        # 2. Medulla -> T4 (ON Motion)
        # Mi1 feeds local T4
        add_synapse(mi1, t4a, 12.0, 8)
        add_synapse(mi1, t4b, 12.0, 8)

        # Tm3 feeds neighboring column T4 (spatial asymmetry for Reichardt correlation)
        # For rightwards (T4a): receives Tm3 from left column (c - 1)
        if c > 0:
            left_tm3 = (c - 1) * NEURONS_PER_COL + 3
            add_synapse(left_tm3, t4a, 14.0, 12)
        # For leftwards (T4b): receives Tm3 from right column (c + 1)
        if c + 1 < num_columns:
            right_tm3 = (c + 1) * NEURONS_PER_COL + 3
            add_synapse(right_tm3, t4b, 14.0, 12)

        # 3. Medulla -> T5 (OFF Motion)
        add_synapse(tm1, t5a, 12.0, 8)
        add_synapse(tm1, t5b, 12.0, 8)
        if c > 0:
            left_tm2 = (c - 1) * NEURONS_PER_COL + 5
            add_synapse(left_tm2, t5a, 14.0, 12)
        if c + 1 < num_columns:
            right_tm2 = (c + 1) * NEURONS_PER_COL + 5
            add_synapse(right_tm2, t5b, 14.0, 12)

        # 4. T4/T5 -> Lobula Plate HS cells
        add_synapse(t4a, hs_right_idx, 6.0, 6)
        add_synapse(t5a, hs_right_idx, 6.0, 6)
        add_synapse(t4b, hs_left_idx, 6.0, 6)
        add_synapse(t5b, hs_left_idx, 6.0, 6)

    # Build CSR arrays
    row_arr = np.array(row_list, dtype=np.int64)
    col_arr = np.array(col_list, dtype=np.int64)
    weight_arr = np.array(weight_list, dtype=np.float64)
    contact_arr = np.array(contact_list, dtype=np.uint32)

    sort_idx = np.lexsort((col_arr, row_arr))
    row_arr = row_arr[sort_idx]
    col_arr = col_arr[sort_idx]
    weight_arr = weight_arr[sort_idx]
    contact_arr = contact_arr[sort_idx]

    row_ptr = np.zeros(total_neurons + 1, dtype=np.uint32)
    if len(row_arr) > 0:
        counts = np.bincount(row_arr, minlength=total_neurons)
        row_ptr[1:] = np.cumsum(counts)

    return ConnectomeGraph(
        num_neurons=total_neurons,
        row_ptr=row_ptr,
        col_idx=col_arr.astype(np.uint32),
        weights=weight_arr,
        contacts=contact_arr,
        neuron_ids=neuron_ids,
        cell_types=cell_types,
        hemispheres=hemispheres,
        provenance=Provenance(
            tier="computational_hypothesis",
            source="Canonical_Phase1_Optic_Approximation",
            confidence=0.85,
            rationale="Canonical 1D multi-column Hassenstein-Reichardt motion circuit with Mi1/Tm3 and Tm1/Tm2",
            doi="10.7554/eLife.29044",
        ),
    )


def create_canonical_phase1b_optic_circuit(
    num_columns: int = 8,
    model_variant: str = "A",
    tau_fast: float = 15.0,
    tau_slow: float = 60.0,
    delay_ms: float = 20.0,
    inhibition_weight: float = -12.0,
    seed: int = 42,
) -> Tuple[ConnectomeGraph, np.ndarray, Optional[np.ndarray]]:
    """Construct an optic circuit parametrized across Models A, B, C, D.

    Models:
      - 'A': Baseline (homogeneous tau_syn=5.0ms, delay=0ms)
      - 'B': Heterogeneous tau (Mi1/Tm1 tau_fast=15ms, Tm3/Tm2 tau_slow=60ms, delay=0ms)
      - 'C': Heterogeneous tau + Cross-column synaptic delays (delay_ms)
      - 'D': Model C + Lateral shunting null-direction inhibition

    Returns:
      (graph, tau_syn_per_neuron, edge_delays)
    """
    NEURONS_PER_COL = 10
    total_neurons = num_columns * NEURONS_PER_COL + 2
    hs_right_idx = num_columns * NEURONS_PER_COL
    hs_left_idx = hs_right_idx + 1

    cell_types: List[str] = []
    hemispheres: List[str] = []
    neuron_ids = np.arange(total_neurons, dtype=np.uint64)

    for col in range(num_columns):
        cell_types.extend([
            "L_ON",
            "L_OFF",
            "Mi1",
            "Tm3",
            "Tm1",
            "Tm2",
            "T4a",
            "T4b",
            "T5a",
            "T5b",
        ])
        hemispheres.extend(["R"] * NEURONS_PER_COL)

    cell_types.extend(["HS_Right", "HS_Left"])
    hemispheres.extend(["R", "R"])

    # Build per-neuron tau_syn
    tau_syn_vec = np.full(total_neurons, 5.0, dtype=np.float64)
    if model_variant in ("B", "C", "D"):
        for c in range(num_columns):
            base = c * NEURONS_PER_COL
            tau_syn_vec[base + 2] = tau_fast  # Mi1
            tau_syn_vec[base + 3] = tau_slow  # Tm3
            tau_syn_vec[base + 4] = tau_fast  # Tm1
            tau_syn_vec[base + 5] = tau_slow  # Tm2
            tau_syn_vec[base + 6] = 20.0      # T4a
            tau_syn_vec[base + 7] = 20.0      # T4b
            tau_syn_vec[base + 8] = 20.0      # T5a
            tau_syn_vec[base + 9] = 20.0      # T5b
        tau_syn_vec[hs_right_idx] = 30.0
        tau_syn_vec[hs_left_idx] = 30.0

    row_list: List[int] = []
    col_list: List[int] = []
    weight_list: List[float] = []
    contact_list: List[int] = []
    delay_list: List[int] = []

    def add_edge(pre: int, post: int, weight: float, contacts: int, delay_steps: int = 0):
        row_list.append(pre)
        col_list.append(post)
        weight_list.append(weight)
        contact_list.append(contacts)
        delay_list.append(delay_steps)

    delay_steps = int(round(delay_ms)) if model_variant in ("C", "D") else 0

    for c in range(num_columns):
        base = c * NEURONS_PER_COL
        l_on = base + 0
        l_off = base + 1
        mi1 = base + 2
        tm3 = base + 3
        tm1 = base + 4
        tm2 = base + 5
        t4a = base + 6
        t4b = base + 7
        t5a = base + 8
        t5b = base + 9

        # 1. Lamina -> Medulla
        add_edge(l_on, mi1, 15.0, 10, delay_steps=0)
        add_edge(l_on, tm3, 15.0, 10, delay_steps=0)
        add_edge(l_off, tm1, 15.0, 10, delay_steps=0)
        add_edge(l_off, tm2, 15.0, 10, delay_steps=0)

        # 2. Medulla -> T4 (ON Motion)
        add_edge(mi1, t4a, 12.0, 8, delay_steps=0)
        add_edge(mi1, t4b, 12.0, 8, delay_steps=0)

        # Delayed cross-column input
        if c > 0:
            left_tm3 = (c - 1) * NEURONS_PER_COL + 3
            add_edge(left_tm3, t4a, 14.0, 12, delay_steps=delay_steps)
        if c + 1 < num_columns:
            right_tm3 = (c + 1) * NEURONS_PER_COL + 3
            add_edge(right_tm3, t4b, 14.0, 12, delay_steps=delay_steps)

        # 3. Medulla -> T5 (OFF Motion)
        add_edge(tm1, t5a, 12.0, 8, delay_steps=0)
        add_edge(tm1, t5b, 12.0, 8, delay_steps=0)
        if c > 0:
            left_tm2 = (c - 1) * NEURONS_PER_COL + 5
            add_edge(left_tm2, t5a, 14.0, 12, delay_steps=delay_steps)
        if c + 1 < num_columns:
            right_tm2 = (c + 1) * NEURONS_PER_COL + 5
            add_edge(right_tm2, t5b, 14.0, 12, delay_steps=delay_steps)

        # 4. Model D: Lateral Null-Direction Shunting Inhibition
        if model_variant == "D":
            # For T4a (rightwards), inhibit when right-hand column activates first (null direction)
            if c + 1 < num_columns:
                right_mi1 = (c + 1) * NEURONS_PER_COL + 2
                add_edge(right_mi1, t4a, inhibition_weight, 8, delay_steps=0)
            # For T4b (leftwards), inhibit when left-hand column activates first
            if c > 0:
                left_mi1 = (c - 1) * NEURONS_PER_COL + 2
                add_edge(left_mi1, t4b, inhibition_weight, 8, delay_steps=0)

            # Same for T5
            if c + 1 < num_columns:
                right_tm1 = (c + 1) * NEURONS_PER_COL + 4
                add_edge(right_tm1, t5a, inhibition_weight, 8, delay_steps=0)
            if c > 0:
                left_tm1 = (c - 1) * NEURONS_PER_COL + 4
                add_edge(left_tm1, t5b, inhibition_weight, 8, delay_steps=0)

        # 5. T4/T5 -> Lobula Plate HS cells
        add_edge(t4a, hs_right_idx, 6.0, 6, delay_steps=0)
        add_edge(t5a, hs_right_idx, 6.0, 6, delay_steps=0)
        add_edge(t4b, hs_left_idx, 6.0, 6, delay_steps=0)
        add_edge(t5b, hs_left_idx, 6.0, 6, delay_steps=0)

    # Build CSR arrays
    row_arr = np.array(row_list, dtype=np.int64)
    col_arr = np.array(col_list, dtype=np.int64)
    weight_arr = np.array(weight_list, dtype=np.float64)
    contact_arr = np.array(contact_list, dtype=np.uint32)
    delay_arr = np.array(delay_list, dtype=np.uint32)

    sort_idx = np.lexsort((col_arr, row_arr))
    row_arr = row_arr[sort_idx]
    col_arr = col_arr[sort_idx]
    weight_arr = weight_arr[sort_idx]
    contact_arr = contact_arr[sort_idx]
    delay_arr = delay_arr[sort_idx]

    row_ptr = np.zeros(total_neurons + 1, dtype=np.uint32)
    if len(row_arr) > 0:
        counts = np.bincount(row_arr, minlength=total_neurons)
        row_ptr[1:] = np.cumsum(counts)

    provenance_source = f"Canonical_Phase1B_Optic_Model_{model_variant}"
    provenance_rationale = (
        f"Canonical multi-column Reichardt circuit under Model {model_variant} with engineered null-direction inhibitory proxy"
        if model_variant in ("D", "E")
        else f"Canonical multi-column Reichardt circuit under Model {model_variant}"
    )

    graph = ConnectomeGraph(
        num_neurons=total_neurons,
        row_ptr=row_ptr,
        col_idx=col_arr.astype(np.uint32),
        weights=weight_arr,
        contacts=contact_arr,
        neuron_ids=neuron_ids,
        cell_types=cell_types,
        hemispheres=hemispheres,
        provenance=Provenance(
            tier="computational_hypothesis",
            source=provenance_source,
            confidence=0.85,
            rationale=provenance_rationale,
            doi="10.7554/eLife.29044",
        ),
    )

    edge_delays = delay_arr if model_variant in ("C", "D") else None
    return graph, tau_syn_vec, edge_delays
