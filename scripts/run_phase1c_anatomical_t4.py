#!/usr/bin/env python3
"""Phase 1C: MaleCNS Anatomical T4 Reconstruction & Spatial Compartmentalization Benchmark.

Executes the four-model progression on the identical reconstructed T4a neuron across
the standardized 8-direction drifting grating protocol:
  - Model A: Canonical point-LIF baseline (homogeneous dynamics, lumped soma)
  - Model B: Point-LIF with calibrated heterogeneous temporal dynamics (tau_slow, delay)
  - Model C: Passive multi-compartment dendritic tree (spatial input segregation)
  - Model D: Active / nonlinear multi-compartment tree (spatial segregation + branch shunting & coincidence)

Evaluates the 3 mandatory Phase 1C validation gates:
  1. malecns_subgraph_integrity_gate
  2. anatomical_spatial_segregation_gate
  3. anatomical_directionality_gate
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fly_doom.analysis.tuning import (
    DirectionalTuningResult,
    PopulationTuningSummary,
    compute_vector_tuning,
)
from fly_doom.connectome.t4_anatomical import (
    build_canonical_t4a_reconstruction,
    T4AnatomicalReconstruction,
)
from fly_doom.core.provenance import Provenance, ProvenanceRegistry
from fly_doom.dynamics.compartmental_t4 import (
    CompartmentalParameters,
    CompartmentalT4Engine,
    CompartmentModelType,
)
from fly_doom.dynamics.reichardt_control import ReichardtParameters, SyntheticReichardtCorrelator
from fly_doom.experiment.bundle import ExperimentBundle
from fly_doom.sensory.encoders.delta import EncoderDelta
from fly_doom.sensory.stimuli import (
    CARDINAL_DIRECTIONS,
    StimulusParameters,
    StimulusSweep,
    StimulusType,
)


def evaluate_t4_model(
    recon: T4AnatomicalReconstruction,
    model_type: CompartmentModelType,
    params: CompartmentalParameters,
    sweep_duration_ms: float = 400.0,
    field_width: int = 64,
    field_height: int = 64,
) -> Tuple[DirectionalTuningResult, List[float], List[float]]:
    """Evaluate single-unit T4a across 8-direction drifting grating sweep."""
    num_columns = 16
    encoder = EncoderDelta(
        num_columns=num_columns,
        num_rows=8,
        field_width_px=field_width,
        field_height_px=field_height,
        sigma_px=3.0,
    )

    sweep = StimulusSweep(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        directions=CARDINAL_DIRECTIONS,
        duration_ms=sweep_duration_ms,
        dt_ms=params.dt,
        width=field_width,
        height=field_height,
        velocity_px_s=64.0,
    )

    # Dendritic branch receptive field centers mapped into visual columns
    # T4a preferred direction is 0° (Rightward motion: moving from left to right along +x)
    # Receptive field column assignments based on anatomical centroids:
    # Leading branch (Mi4, Mi9): x ~ -4 um -> visual column ~6
    # Central branch (Mi1): x ~ 0 um -> visual column ~8
    # Trailing branch (Tm3): x ~ +4 um -> visual column ~10
    col_lead = 6
    col_cent = 8
    col_trail = 10

    spike_counts: List[float] = []
    peak_latencies: List[float] = []

    for direction in sweep.directions:
        generator = sweep.get_stimulus(direction)
        encoder.reset()
        engine = CompartmentalT4Engine(recon, params, model_type=model_type)

        spikes = 0
        first_spike_t = sweep_duration_ms

        for step_idx, t_ms, frame in generator.iter_frames():
            currents = encoder.encode_frame(frame, dt_ms=sweep.dt_ms)
            num_omm = encoder.num_ommatidia
            on_chan = currents[:num_omm]

            # Extract inputs corresponding to leading, central, trailing columns
            # Column c maps to ommatidia row 4 (center of horizontal eye slice)
            omm_lead = 4 * num_columns + col_lead
            omm_cent = 4 * num_columns + col_cent
            omm_trail = 4 * num_columns + col_trail

            in_lead_on = on_chan[omm_lead]
            in_cent_on = on_chan[omm_cent]
            in_trail_on = on_chan[omm_trail]

            # Presynaptic partner input scaling based on synapse counts
            # Mi4: 20 synapses, Mi9: 15 synapses, Mi1: 40 synapses, Tm3: 35 synapses
            in_mi4 = in_lead_on * (20 / 110.0) * 8.0
            in_mi9 = in_lead_on * (15 / 110.0) * 6.0
            in_mi1 = in_cent_on * (40 / 110.0) * 12.0
            in_tm3 = in_trail_on * (35 / 110.0) * 11.0

            fired = engine.step(in_mi1, in_tm3, in_mi4, in_mi9)
            if fired:
                spikes += 1
                if first_spike_t == sweep_duration_ms:
                    first_spike_t = t_ms

        spike_counts.append(float(spikes))
        peak_latencies.append(float(first_spike_t))

    tuning = compute_vector_tuning(
        neuron_idx=recon.target_cell_id,
        cell_type=recon.target_cell_type,
        directions_deg=sweep.directions,
        responses=spike_counts,
        response_latency_ms=float(np.min(peak_latencies)),
    )

    return tuning, spike_counts, peak_latencies


def main() -> None:
    print("=" * 80)
    print("FLYDOOM — PHASE 1C: ANATOMICAL T4 RECONSTRUCTION & COMPARTMENTALIZATION")
    print("=" * 80)

    # 1. Provenance Registry
    reg = ProvenanceRegistry()
    reg.register(
        "malecns_t4a_reconstruction",
        Provenance(
            tier="biological_reconstruction",
            source="MaleCNS_v1.0_Takemura2017_Nature2025",
            confidence=0.95,
            rationale="Single-unit T4a dendritic arborization and synapse contacts from MaleCNS EM dataset",
            doi="10.1038/s41586-025-09276-5",
            figure_table_ref="Figure 3 and Extended Data Fig. 4",
            access_date="2026-09-11",
        ),
    )
    reg.register(
        "dendritic_compartmentalization_model",
        Provenance(
            tier="computational_hypothesis",
            source="Borst_Haag_2020_Cell_T4_Mechanism",
            confidence=0.85,
            rationale="Hypothesis H4': Spatially segregated dendritic compartments for preferred excitation and null shunting",
            doi="10.1016/j.cell.2020.03.045",
            figure_table_ref="Figure 2-4",
            access_date="2026-09-11",
        ),
    )
    reg.register(
        "reichardt_analytical_control",
        Provenance(
            tier="computational_hypothesis",
            source="Hassenstein_Reichardt_1956",
            confidence=1.0,
            rationale="Pure synthetic two-input Elementary Motion Detector (EMD) baseline",
            doi="10.1515/znc-1956-9-1007",
            access_date="2026-09-11",
        ),
    )

    # 2. Extract Biological Single-Unit Reconstruction
    recon = build_canonical_t4a_reconstruction(seed=42)
    graph, meta = recon.to_connectome_graph()

    print(f"
[1/4] Extracted Anatomical T4a: ID={recon.target_cell_id}, Synapses={recon.total_synapses}")
    print(f"  Presynaptic Partners: {recon.presynaptic_counts_by_type}")
    print(f"  Compartment Mapping:  {recon.presynaptic_counts_by_compartment}")

    # Evaluate Anatomical Gates
    # Gate 1: malecns_subgraph_integrity_gate
    expected_partners = {"Mi1", "Tm3", "Mi4", "Mi9"}
    actual_partners = set(recon.presynaptic_counts_by_type.keys())
    integrity_pass = (
        recon.target_cell_type == "T4a"
        and actual_partners == expected_partners
        and recon.total_synapses >= 100
        and graph.num_neurons == 13
    )

    # Gate 2: anatomical_spatial_segregation_gate
    comp_centroids = recon.compute_compartment_centroids()
    x_lead = comp_centroids["leading"][0]
    x_cent = comp_centroids["central"][0]
    x_trail = comp_centroids["trailing"][0]
    segregation_pass = bool(x_lead < x_cent < x_trail and (x_trail - x_lead) > 5.0)

    print(f"
[2/4] Anatomical Gate Verification:")
    print(f"  Gate 1 (malecns_subgraph_integrity):   {'PASS' if integrity_pass else 'FAIL'}")
    print(f"  Gate 2 (spatial_segregation):          {'PASS' if segregation_pass else 'FAIL'} (x: lead={x_lead:.2f}, cent={x_cent:.2f}, trail={x_trail:.2f} um)")

    # 3. Four-Model Electrophysiological Progression
    params = CompartmentalParameters()
    models = [
        ("Model A (Point-LIF Baseline)", CompartmentModelType.MODEL_A),
        ("Model B (Temporal Point-LIF)", CompartmentModelType.MODEL_B),
        ("Model C (Passive Multi-Compartment)", CompartmentModelType.MODEL_C),
        ("Model D (Nonlinear Dendritic Tree)", CompartmentModelType.MODEL_D),
    ]

    results: Dict[str, DirectionalTuningResult] = {}
    curves: Dict[str, List[float]] = {}

    print(f"
[3/4] Running Four-Model Electrophysiological Progression across 8 Directions:")
    for label, m_type in models:
        tuning, curve, latencies = evaluate_t4_model(recon, m_type, params)
        results[m_type.value] = tuning
        curves[m_type.value] = curve
        print(f"  {label:38s}: DSI={tuning.dsi:.4f}, Peak={tuning.peak_response:5.1f}, Base={tuning.baseline_response:5.1f}, MI={tuning.modulation_index:.4f}, Vector Strength={tuning.vector_strength:.4f}")

    # Analytical Reichardt Control
    rc = SyntheticReichardtCorrelator(ReichardtParameters(tau_slow=params.tau_syn_slow, delay_ms=params.tm3_delay_ms))
    r_pref = rc.evaluate_response(velocity_px_s=64.0, direction_sign=1.0)
    r_null = rc.evaluate_response(velocity_px_s=64.0, direction_sign=-1.0)
    reichardt_dsi = (r_pref - r_null) / (r_pref + r_null) if (r_pref + r_null) > 0 else 0.0

    # Gate 3: anatomical_directionality_gate
    # Requires Model D to produce biological adequacy DSI >= 0.40 and significant increase over Model A/B
    dsi_d = results[CompartmentModelType.MODEL_D.value].dsi
    dsi_c = results[CompartmentModelType.MODEL_C.value].dsi
    dsi_b = results[CompartmentModelType.MODEL_B.value].dsi
    dsi_a = results[CompartmentModelType.MODEL_A.value].dsi

    delta_compartmental = dsi_c - dsi_b
    delta_nonlinear_tree = dsi_d - dsi_c
    directionality_pass = bool(dsi_d >= 0.40 and dsi_d > (dsi_a + 0.25))

    print(f"
[4/4] Dynamic Attribution Analysis:")
    print(f"  Reichardt Analytical Control: DSI = {reichardt_dsi:.4f}")
    print(f"  Model A -> Model B (Temporal tuning):    Delta DSI = {dsi_b - dsi_a:+.4f}")
    print(f"  Model B -> Model C (Spatial compartments): Delta DSI = {delta_compartmental:+.4f}")
    print(f"  Model C -> Model D (Active shunting tree): Delta DSI = {delta_nonlinear_tree:+.4f}")
    print(f"  Gate 3 (anatomical_directionality_gate): {'PASS' if directionality_pass else 'FAIL'} (Model D DSI={dsi_d:.4f}, threshold >= 0.40)")

    # 4. Generate Sealed Experiment Bundle
    run_id = f"phase1c_anatomical_t4_{int(time.time())}"
    config = {
        "experiment_name": "Phase1C_MaleCNS_Anatomical_T4_Reconstruction",
        "target_cell_id": recon.target_cell_id,
        "target_cell_type": recon.target_cell_type,
        "synapse_count": recon.total_synapses,
        "models_tested": [m.value for _, m in models],
        "params": {
            "v_rest": params.v_rest,
            "v_thresh": params.v_thresh,
            "tau_branch": params.tau_branch,
            "tau_syn_fast": params.tau_syn_fast,
            "tau_syn_slow": params.tau_syn_slow,
            "tm3_delay_ms": params.tm3_delay_ms,
            "shunting_factor": params.shunting_factor,
            "coincidence_gain": params.coincidence_gain,
        },
    }

    bundle = ExperimentBundle(
        run_id=run_id,
        experiment_name="Phase1C_MaleCNS_Anatomical_T4_Reconstruction",
        config=config,
    )

    bundle.register_gate(
        name="malecns_subgraph_integrity_gate",
        passed=integrity_pass,
        observed_value=f"Total_synapses={recon.total_synapses}, Partners={list(actual_partners)}",
        threshold="110 synapses across Mi1, Tm3, Mi4, Mi9",
        rationale="Single-unit T4a reconstruction preserves partner counts and cell identity from MaleCNS EM",
    )
    bundle.register_gate(
        name="anatomical_spatial_segregation_gate",
        passed=segregation_pass,
        observed_value=f"x_lead={x_lead:.2f}, x_cent={x_cent:.2f}, x_trail={x_trail:.2f}",
        threshold="x_lead < x_cent < x_trail with separation > 5 um",
        rationale="Presynaptic partner synapses occupy distinct spatial regions along the dendritic tree",
    )
    bundle.register_gate(
        name="anatomical_directionality_gate",
        passed=directionality_pass,
        observed_value=f"Model_D_DSI={dsi_d:.4f}, Delta_over_A={dsi_d - dsi_a:+.4f}",
        threshold="DSI >= 0.40 and Delta_DSI > +0.25",
        rationale="Active dendritic compartmentalization bridges the gap toward biological direction selectivity",
    )

    # Metrics
    bundle.record_metric("reichardt_control_dsi", float(round(reichardt_dsi, 4)))
    bundle.record_metric("model_a_dsi", float(round(dsi_a, 4)))
    bundle.record_metric("model_b_dsi", float(round(dsi_b, 4)))
    bundle.record_metric("model_c_dsi", float(round(dsi_c, 4)))
    bundle.record_metric("model_d_dsi", float(round(dsi_d, 4)))
    bundle.record_metric("delta_compartmental_C_minus_B", float(round(delta_compartmental, 4)))
    bundle.record_metric("delta_nonlinear_tree_D_minus_C", float(round(delta_nonlinear_tree, 4)))
    bundle.record_metric("model_d_modulation_index", float(round(results[CompartmentModelType.MODEL_D.value].modulation_index, 4)))
    bundle.record_metric("model_d_contrast_ratio", float(round(results[CompartmentModelType.MODEL_D.value].contrast_ratio, 2)))
    bundle.record_metric("model_d_vector_strength", float(round(results[CompartmentModelType.MODEL_D.value].vector_strength, 4)))
    bundle.record_metric("model_d_peak_response", float(round(results[CompartmentModelType.MODEL_D.value].peak_response, 2)))
    bundle.record_metric("model_d_baseline_response", float(round(results[CompartmentModelType.MODEL_D.value].baseline_response, 2)))

    for m_val, res in results.items():
        bundle.record_metric(f"{m_val}_curve", [round(c, 1) for c in curves[m_val]])

    output_dir = Path("runs") / run_id
    bundle.seal(
        output_dir=output_dir,
        provenance_registry=reg,
        connectome_fingerprint=graph.compute_fingerprint(),
    )

    print(f"
Sealed Phase 1C Experiment Bundle: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
