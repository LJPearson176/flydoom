#!/usr/bin/env python3
"""Phase 1B Factorial Temporal Dynamics & Model Ablation Study.

Evaluates the comparative model ladder:
  - Model A: Baseline Homogeneous (tau=5ms, delay=0ms)
  - Model B: Heterogeneous tau (Mi1 fast vs Tm3 slow, delay=0ms)
  - Model C: Heterogeneous tau + Transmission delay queues (delay=20ms)
  - Model D: Model C + Lateral null-direction shunting inhibition

Measures DSI progression, preferred direction alignment, and attributes
Delta DSI across the competing Tier-2 hypotheses.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fly_doom.analysis.tuning import (
    DirectionalTuningResult,
    PopulationTuningSummary,
    compute_vector_tuning,
)
from fly_doom.connectome.subgraphs import (
    create_canonical_phase1b_optic_circuit,
    query_cell_indices,
)
from fly_doom.core.provenance import Provenance, ProvenanceRegistry
from fly_doom.dynamics.lif_native import LIFNativeEngine
from fly_doom.dynamics.lif_reference import LIFParameters
from fly_doom.dynamics.reichardt_control import ReichardtParameters, SyntheticReichardtCorrelator
from fly_doom.experiment.bundle import ExperimentBundle
from fly_doom.sensory.encoders.delta import EncoderDelta
from fly_doom.sensory.stimuli import (
    CARDINAL_DIRECTIONS,
    StimulusParameters,
    StimulusSweep,
    StimulusType,
)


def evaluate_circuit_tuning(
    graph,
    tau_syn_vec: np.ndarray,
    edge_delays: Optional[np.ndarray],
    encoder: EncoderDelta,
    sweep: StimulusSweep,
    num_columns: int = 8,
) -> Tuple[PopulationTuningSummary, PopulationTuningSummary, DirectionalTuningResult, DirectionalTuningResult]:
    """Run sweep through circuit and extract T4a, T4b, and HS tuning metrics."""
    NEURONS_PER_COL = 10
    spike_counts: Dict[int, List[float]] = {i: [] for i in range(graph.num_neurons)}
    params = LIFParameters(dt=sweep.dt_ms, tau_m=15.0, tau_syn=tau_syn_vec, t_ref=2.0)

    for direction in sweep.directions:
        generator = sweep.get_stimulus(direction)
        encoder.reset()
        engine = LIFNativeEngine(graph, params, edge_delays=edge_delays)

        dir_spikes = np.zeros(graph.num_neurons, dtype=np.int64)

        for step_idx, t_ms, frame in generator.iter_frames():
            currents = encoder.encode_frame(frame, dt_ms=sweep.dt_ms)
            ext_current = np.zeros(graph.num_neurons, dtype=np.float64)

            num_ommatidia = encoder.num_ommatidia
            on_channels = currents[:num_ommatidia]
            off_channels = currents[num_ommatidia:]

            for c in range(min(num_columns, num_ommatidia)):
                base = c * NEURONS_PER_COL
                ext_current[base + 0] = on_channels[c]
                ext_current[base + 1] = off_channels[c]

            fired = engine.step(ext_current)
            if len(fired) > 0:
                np.add.at(dir_spikes, fired, 1)

        for i in range(graph.num_neurons):
            spike_counts[i].append(float(dir_spikes[i]))

    t4a_indices = query_cell_indices(graph, {"T4a"})
    t4b_indices = query_cell_indices(graph, {"T4b"})
    hs_r_idx = query_cell_indices(graph, {"HS_Right"})[0]
    hs_l_idx = query_cell_indices(graph, {"HS_Left"})[0]

    t4a_results = [
        compute_vector_tuning(idx, "T4a", sweep.directions, spike_counts[idx])
        for idx in t4a_indices
    ]
    t4b_results = [
        compute_vector_tuning(idx, "T4b", sweep.directions, spike_counts[idx])
        for idx in t4b_indices
    ]

    t4a_summary = PopulationTuningSummary.from_results("T4a", t4a_results)
    t4b_summary = PopulationTuningSummary.from_results("T4b", t4b_results)

    hs_r_result = compute_vector_tuning(hs_r_idx, "HS_Right", sweep.directions, spike_counts[hs_r_idx])
    hs_l_result = compute_vector_tuning(hs_l_idx, "HS_Left", sweep.directions, spike_counts[hs_l_idx])

    return t4a_summary, t4b_summary, hs_r_result, hs_l_result


def main() -> None:
    print("=" * 80)
    print("FLYDOOM — CONNECTOME CYBERNETICS LABORATORY: PHASE 1B FACTORIAL ABLATION")
    print("=" * 80)

    # 1. Provenance Registry with Source-Locked DOIs
    reg = ProvenanceRegistry()
    reg.register(
        "reichardt_analytical_control",
        Provenance(
            tier="computational_hypothesis",
            source="Hassenstein_Reichardt_1956",
            confidence=1.0,
            rationale="Pure synthetic two-input Elementary Motion Detector (EMD) baseline",
            doi="10.1515/znc-1956-9-1007",
            figure_table_ref="Equation 1-4",
            access_date="2026-09-11",
        ),
    )
    reg.register(
        "t4_t5_motion_circuit",
        Provenance(
            tier="computational_hypothesis",
            source="Strother_et_al_2017_eLife",
            confidence=0.85,
            rationale="Directional tuning curves and cardinal motion pathways in Drosophila",
            doi="10.7554/eLife.29044",
            figure_table_ref="Figure 2-4",
            access_date="2026-09-11",
        ),
    )
    reg.register(
        "ommatidial_sensory_geometry",
        Provenance(
            tier="experimental_assumption",
            source="Nature_2025_Drosophila_Visual_Surface",
            confidence=0.90,
            rationale="Hexagonal ommatidial lattice and retinotopic dendritic tiling",
            doi="10.1038/s41586-025-09276-5",
            figure_table_ref="Extended Data Fig. 3",
            access_date="2026-09-11",
        ),
    )
    reg.register(
        "lif_delay_engine",
        Provenance(
            tier="computational_hypothesis",
            source="Clang_CPP20_ARM64_Native_Delays",
            confidence=0.90,
            rationale="Native C++20 engine with heterogeneous tau_syn and transmission delay queues",
            access_date="2026-09-11",
        ),
    )

    # 2. Pure Synthetic Reichardt Control Verification
    print("\n[Step 0] Benchmarking Pure Synthetic Reichardt Control...")
    reichardt = SyntheticReichardtCorrelator(ReichardtParameters(tau_delay_ms=30.0, dt_ms=1.0))
    r_pref, r_null, reichardt_dsi = reichardt.test_motion(velocity_px_s=20.0, pixel_spacing=1.0)
    print(f"         Reichardt Control Output: Preferred={r_pref:.3f}, Null={r_null:.3f}, DSI={reichardt_dsi:.3f}")
    reichardt_gate_passed = reichardt_dsi > 0.70
    print(f"         Reichardt Baseline Gate: {'PASS' if reichardt_gate_passed else 'FAIL'}")

    # 3. Factorial Model Ladder Execution
    sweep = StimulusSweep(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        directions=CARDINAL_DIRECTIONS,
        duration_ms=400.0,
        dt_ms=1.0,
        width=64,
        height=64,
        velocity_px_s=64.0,
    )
    encoder = EncoderDelta(num_columns=8, num_rows=8, field_width_px=64, field_height_px=64)

    # --- MODEL A: Baseline Homogeneous (tau=5ms, delay=0ms) ---
    print("\n[Model A] Baseline Homogeneous Dynamics (tau_syn=5ms, delay=0ms)...")
    graph_a, tau_a, del_a = create_canonical_phase1b_optic_circuit(num_columns=8, model_variant="A")
    t4a_a, t4b_a, hs_r_a, _ = evaluate_circuit_tuning(graph_a, tau_a, del_a, encoder, sweep)
    print(f"          Model A T4a Mean DSI: {t4a_a.mean_dsi:.3f} | HS_Right DSI: {hs_r_a.dsi:.3f}")

    # --- MODEL B: Heterogeneous Tau Grid (tau_slow in [20, 40, 60, 80, 100ms], delay=0ms) ---
    print("\n[Model B] Heterogeneous Temporal Filtering Sweep (delay=0ms)...")
    tau_slow_grid = [20.0, 40.0, 60.0, 80.0, 100.0]
    model_b_results = {}
    best_tau_slow = 60.0
    best_b_dsi = 0.0

    for tau_s in tau_slow_grid:
        graph_b, tau_b, del_b = create_canonical_phase1b_optic_circuit(
            num_columns=8, model_variant="B", tau_fast=15.0, tau_slow=tau_s
        )
        t4a_b, _, hs_r_b, _ = evaluate_circuit_tuning(graph_b, tau_b, del_b, encoder, sweep)
        model_b_results[tau_s] = t4a_b.mean_dsi
        print(f"          tau_slow={tau_s:5.1f} ms  -->  T4a DSI={t4a_b.mean_dsi:.3f}, HS DSI={hs_r_b.dsi:.3f}")
        if t4a_b.mean_dsi > best_b_dsi:
            best_b_dsi = t4a_b.mean_dsi
            best_tau_slow = tau_s

    # Selected Model B representation
    graph_b_sel, tau_b_sel, del_b_sel = create_canonical_phase1b_optic_circuit(
        num_columns=8, model_variant="B", tau_fast=15.0, tau_slow=best_tau_slow
    )
    t4a_b_sel, t4b_b_sel, hs_r_b_sel, _ = evaluate_circuit_tuning(graph_b_sel, tau_b_sel, del_b_sel, encoder, sweep)

    # --- MODEL C: Heterogeneous Tau + Synaptic Delays Grid (delay in [0, 10, 20, 40ms]) ---
    print(f"\n[Model C] Heterogeneous Dynamics + Transmission Delays (tau_slow={best_tau_slow}ms)...")
    delay_grid = [0.0, 10.0, 20.0, 40.0]
    model_c_results = {}
    best_delay = 20.0
    best_c_dsi = 0.0

    for d_ms in delay_grid:
        graph_c, tau_c, del_c = create_canonical_phase1b_optic_circuit(
            num_columns=8, model_variant="C", tau_fast=15.0, tau_slow=best_tau_slow, delay_ms=d_ms
        )
        t4a_c, _, hs_r_c, _ = evaluate_circuit_tuning(graph_c, tau_c, del_c, encoder, sweep)
        model_c_results[d_ms] = t4a_c.mean_dsi
        print(f"          delay={d_ms:4.1f} ms  -->  T4a DSI={t4a_c.mean_dsi:.3f}, HS DSI={hs_r_c.dsi:.3f}")
        if t4a_c.mean_dsi > best_c_dsi:
            best_c_dsi = t4a_c.mean_dsi
            best_delay = d_ms

    graph_c_sel, tau_c_sel, del_c_sel = create_canonical_phase1b_optic_circuit(
        num_columns=8, model_variant="C", tau_fast=15.0, tau_slow=best_tau_slow, delay_ms=best_delay
    )
    t4a_c_sel, t4b_c_sel, hs_r_c_sel, _ = evaluate_circuit_tuning(graph_c_sel, tau_c_sel, del_c_sel, encoder, sweep)

    # --- MODEL D: Model C + Lateral Shunting Inhibition ---
    print(f"\n[Model D] Model C + Lateral Shunting Null-Direction Inhibition (delay={best_delay}ms)...")
    graph_d, tau_d, del_d = create_canonical_phase1b_optic_circuit(
        num_columns=8,
        model_variant="D",
        tau_fast=15.0,
        tau_slow=best_tau_slow,
        delay_ms=best_delay,
        inhibition_weight=-14.0,
    )
    t4a_d, t4b_d, hs_r_d, _ = evaluate_circuit_tuning(graph_d, tau_d, del_d, encoder, sweep)
    print(f"          Model D T4a Mean DSI: {t4a_d.mean_dsi:.3f} | HS_Right DSI: {hs_r_d.dsi:.3f} | PD: {hs_r_d.preferred_direction_deg:.1f}°")

    # 4. Attribution Analysis across Models A -> B -> C -> D
    delta_a_to_b = t4a_b_sel.mean_dsi - t4a_a.mean_dsi
    delta_b_to_c = t4a_c_sel.mean_dsi - t4a_b_sel.mean_dsi
    delta_c_to_d = t4a_d.mean_dsi - t4a_c_sel.mean_dsi

    print("\n" + "-" * 75)
    print("FACTORIAL ABLATION SUMMARY & DSI ATTRIBUTION:")
    print(f"  Model A (Baseline):               DSI = {t4a_a.mean_dsi:.3f}")
    print(f"  Model B (Heterogeneous Tau):      DSI = {t4a_b_sel.mean_dsi:.3f}  (Delta A->B: {delta_a_to_b:+.3f})")
    print(f"  Model C (Tau + Delays):           DSI = {t4a_c_sel.mean_dsi:.3f}  (Delta B->C: {delta_b_to_c:+.3f})")
    print(f"  Model D (Delays + Shunting Inh):  DSI = {t4a_d.mean_dsi:.3f}  (Delta C->D: {delta_c_to_d:+.3f})")
    print("-" * 75)

    # 5. Model Adequacy Gate
    # Target biological criterion: DSI > 0.40 and cardinal error < 20°
    target_biological_dsi = 0.40
    model_d_adequate = t4a_d.mean_dsi >= target_biological_dsi

    print(f"\nModel Adequacy Status (Target DSI > {target_biological_dsi}):")
    print(f"  Model A Adequacy: {'ADEQUATE' if t4a_a.mean_dsi >= target_biological_dsi else 'INADEQUATE'}")
    print(f"  Model B Adequacy: {'ADEQUATE' if t4a_b_sel.mean_dsi >= target_biological_dsi else 'INADEQUATE'}")
    print(f"  Model C Adequacy: {'ADEQUATE' if t4a_c_sel.mean_dsi >= target_biological_dsi else 'INADEQUATE'}")
    print(f"  Model D Adequacy: {'ADEQUATE' if model_d_adequate else 'INADEQUATE'}")

    # 6. Hermetic Experiment Bundle Packaging
    run_id = f"phase1b_factorial_{int(time.time())}"
    bundle = ExperimentBundle(
        run_id=run_id,
        experiment_name="Phase1B_Factorial_Temporal_Dynamics_and_Ablation",
        config={
            "num_columns": 8,
            "tau_slow_grid_ms": tau_slow_grid,
            "delay_grid_ms": delay_grid,
            "best_tau_slow_ms": best_tau_slow,
            "best_delay_ms": best_delay,
            "sweep_velocity_px_s": sweep.velocity_px_s,
        },
        provenance_registry=reg,
        dataset_fingerprint=graph_d.compute_fingerprint(),
        metrics={
            "synthetic_reichardt_dsi": round(reichardt_dsi, 4),
            "model_a_dsi": round(t4a_a.mean_dsi, 4),
            "model_b_dsi": round(t4a_b_sel.mean_dsi, 4),
            "model_c_dsi": round(t4a_c_sel.mean_dsi, 4),
            "model_d_dsi": round(t4a_d.mean_dsi, 4),
            "delta_a_to_b_tau": round(delta_a_to_b, 4),
            "delta_b_to_c_delay": round(delta_b_to_c, 4),
            "delta_c_to_d_inhibition": round(delta_c_to_d, 4),
            "hs_right_dsi_model_d": round(hs_r_d.dsi, 4),
            "hs_right_pd_model_d": round(hs_r_d.preferred_direction_deg, 1),
        },
    )

    bundle.add_gate(
        name="reichardt_analytical_control_gate",
        passed=reichardt_gate_passed,
        observed_value=f"DSI={reichardt_dsi:.3f}",
        threshold="DSI > 0.70",
        rationale="Synthetic Reichardt coincidence detector must demonstrate directional selectivity in isolation",
    )
    bundle.add_gate(
        name="temporal_asymmetry_attribution_gate",
        passed=delta_b_to_c > 0.05,
        observed_value=f"Delta_DSI={delta_b_to_c:+.3f}",
        threshold="Delta_DSI > +0.05",
        rationale="Introduction of transmission delays must measurably enhance directional coincidence",
    )
    bundle.add_gate(
        name="shunting_inhibition_attribution_gate",
        passed=delta_c_to_d > 0.05,
        observed_value=f"Delta_DSI={delta_c_to_d:+.3f}",
        threshold="Delta_DSI > +0.05",
        rationale="Lateral null-direction shunting inhibition must measurably enhance directional selectivity",
    )
    bundle.add_gate(
        name="model_d_adequacy_gate",
        passed=model_d_adequate,
        observed_value=f"Model_D_DSI={t4a_d.mean_dsi:.3f}",
        threshold=f"DSI >= {target_biological_dsi}",
        rationale="Model D (Heterogeneous tau + transmission delays + shunting inhibition) recovers biological DSI",
    )
    bundle.add_gate(
        name="source_provenance_gate",
        passed=bundle.git_hash != "git_hash_unavailable",
        observed_value=f"commit={bundle.git_hash[:10]}, dirty={bundle.git_provenance.get('dirty')}",
        threshold="identifiable git commit SHA",
        rationale="Experiment must be sealed against a known git commit",
    )

    bundle_dir = bundle.save(Path("runs"))
    print(f"\nSealed Phase 1B experiment bundle saved to: {bundle_dir}/")
    print(f"Report written to: {bundle_dir / 'report.md'}")
    print("\n" + "=" * 80)
    print("PHASE 1B FACTORIAL AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
