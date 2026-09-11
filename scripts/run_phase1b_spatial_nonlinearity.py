#!/usr/bin/env python3
"""Phase 1B Spatial Resolution & Nonlinear Coincidence Study (phase1b_spatial_nonlinearity).

Factorial evaluation across:
  - Spatial Column Resolution: N_columns in {8, 16, 32, 64}
  - Integration Regime: Linear point-LIF (gamma=0.0) vs Model E nonlinear coincidence (gamma in {0.0, 0.25, 0.5, 1.0, 2.0})
  - Frozen temporal baseline: tau_slow = 60ms, delay = 20ms, engineered null-direction inhibitory proxy enabled.

Generates full 2D response surface DSI(N_columns, gamma) and rich electrophysiological observables:
  - DSI
  - Preferred direction error
  - Circular variance
  - ON/OFF polarity index
  - Response latency (ms)
  - Peak response (spikes)
  - Baseline response (spikes)
  - Trial-to-trial variance
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def evaluate_circuit_condition(
    num_columns: int,
    coincidence_gamma: float = 0.0,
    tau_fast: float = 15.0,
    tau_slow: float = 60.0,
    delay_ms: float = 20.0,
    inhibition_weight: float = -14.0,
    sweep_duration_ms: float = 400.0,
    sweep_velocity_px_s: float = 64.0,
) -> Tuple[PopulationTuningSummary, PopulationTuningSummary, DirectionalTuningResult, DirectionalTuningResult, Any]:
    """Evaluate one experimental condition (num_columns, gamma)."""
    NEURONS_PER_COL = 10
    field_width = 64
    field_height = 64

    # Construct circuit
    graph, tau_syn_vec, edge_delays = create_canonical_phase1b_optic_circuit(
        num_columns=num_columns,
        model_variant="D",
        tau_fast=tau_fast,
        tau_slow=tau_slow,
        delay_ms=delay_ms,
        inhibition_weight=inhibition_weight,
    )

    # Nonlinear mask for T4 and T5 motion detectors
    nonlinear_mask = np.zeros(graph.num_neurons, dtype=np.uint8)
    for c in range(num_columns):
        base = c * NEURONS_PER_COL
        nonlinear_mask[base + 6] = 1  # T4a
        nonlinear_mask[base + 7] = 1  # T4b
        nonlinear_mask[base + 8] = 1  # T5a
        nonlinear_mask[base + 9] = 1  # T5b

    # Encoder with matching column count
    encoder = EncoderDelta(
        num_columns=num_columns,
        num_rows=8,
        field_width_px=field_width,
        field_height_px=field_height,
    )

    # 8-direction drifting grating sweep
    sweep = StimulusSweep(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        directions=CARDINAL_DIRECTIONS,
        duration_ms=sweep_duration_ms,
        dt_ms=1.0,
        width=field_width,
        height=field_height,
        velocity_px_s=sweep_velocity_px_s,
    )

    params = LIFParameters(dt=sweep.dt_ms, tau_m=15.0, tau_syn=tau_syn_vec, t_ref=2.0)
    spike_counts: Dict[int, List[float]] = {i: [] for i in range(graph.num_neurons)}
    peak_latencies: Dict[int, List[float]] = {i: [] for i in range(graph.num_neurons)}

    for direction in sweep.directions:
        generator = sweep.get_stimulus(direction)
        encoder.reset()
        engine = LIFNativeEngine(
            graph,
            params,
            edge_delays=edge_delays,
            coincidence_gamma=coincidence_gamma,
            nonlinear_mask=nonlinear_mask,
        )

        dir_spikes = np.zeros(graph.num_neurons, dtype=np.int64)
        time_first_spike = np.full(graph.num_neurons, sweep_duration_ms, dtype=np.float64)

        for step_idx, t_ms, frame in generator.iter_frames():
            currents = encoder.encode_frame(frame, dt_ms=sweep.dt_ms)
            ext_current = np.zeros(graph.num_neurons, dtype=np.float64)

            num_ommatidia = encoder.num_ommatidia
            on_channels = currents[:num_ommatidia]
            off_channels = currents[num_ommatidia:]

            # Map ommatidia columns into neural columns
            cols_to_map = min(num_columns, encoder.num_ommatidia)
            for c in range(cols_to_map):
                base = c * NEURONS_PER_COL
                ext_current[base + 0] = on_channels[c]
                ext_current[base + 1] = off_channels[c]

            fired = engine.step(ext_current)
            if len(fired) > 0:
                np.add.at(dir_spikes, fired, 1)
                for f_idx in fired:
                    if time_first_spike[f_idx] == sweep_duration_ms:
                        time_first_spike[f_idx] = t_ms

        for i in range(graph.num_neurons):
            spike_counts[i].append(float(dir_spikes[i]))
            peak_latencies[i].append(float(time_first_spike[i]))

    t4a_indices = query_cell_indices(graph, {"T4a"})
    t4b_indices = query_cell_indices(graph, {"T4b"})
    hs_r_idx = query_cell_indices(graph, {"HS_Right"})[0]
    hs_l_idx = query_cell_indices(graph, {"HS_Left"})[0]

    t4a_results = [
        compute_vector_tuning(
            idx,
            "T4a",
            sweep.directions,
            spike_counts[idx],
            response_latency_ms=float(np.min(peak_latencies[idx])),
        )
        for idx in t4a_indices
    ]
    t4b_results = [
        compute_vector_tuning(
            idx,
            "T4b",
            sweep.directions,
            spike_counts[idx],
            response_latency_ms=float(np.min(peak_latencies[idx])),
        )
        for idx in t4b_indices
    ]

    t4a_summary = PopulationTuningSummary.from_results("T4a", t4a_results)
    t4b_summary = PopulationTuningSummary.from_results("T4b", t4b_results)

    hs_r_result = compute_vector_tuning(
        hs_r_idx,
        "HS_Right",
        sweep.directions,
        spike_counts[hs_r_idx],
        response_latency_ms=float(np.min(peak_latencies[hs_r_idx])),
    )
    hs_l_result = compute_vector_tuning(
        hs_l_idx,
        "HS_Left",
        sweep.directions,
        spike_counts[hs_l_idx],
        response_latency_ms=float(np.min(peak_latencies[hs_l_idx])),
    )

    return t4a_summary, t4b_summary, hs_r_result, hs_l_result, graph


def main() -> None:
    print("=" * 80)
    print("FLYDOOM — CONNECTOME CYBERNETICS LABORATORY: PHASE 1B SPATIAL NONLINEARITY")
    print("=" * 80)

    # 1. Provenance Registry
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
            rationale="Hexagonal ommatidial lattice and retinotopic dendritic tiling across variable column densities",
            doi="10.1038/s41586-025-09276-5",
            figure_table_ref="Extended Data Fig. 3",
            access_date="2026-09-11",
        ),
    )
    reg.register(
        "model_e_nonlinear_coincidence",
        Provenance(
            tier="computational_hypothesis",
            source="Minimal_Nonlinear_Coincidence_Hypothesis",
            confidence=0.80,
            rationale="Model E: Minimal supralinear coincidence integration proxy for converging visual pathways",
            access_date="2026-09-11",
        ),
    )
    reg.register(
        "engineered_null_direction_inhibition",
        Provenance(
            tier="computational_hypothesis",
            source="Engineered_Null_Direction_Proxy",
            confidence=0.80,
            rationale="Engineered null-direction inhibitory proxy pending full MaleCNS EM anatomical sourcing",
            access_date="2026-09-11",
        ),
    )

    # 2. Pure Synthetic Reichardt Control Verification
    print("\n[Step 0] Benchmarking Pure Synthetic Reichardt Control...")
    reichardt = SyntheticReichardtCorrelator(ReichardtParameters(tau_delay_ms=30.0, dt_ms=1.0))
    r_pref, r_null, reichardt_dsi = reichardt.test_motion(velocity_px_s=20.0, pixel_spacing=1.0)
    print(f"         Reichardt Control Output: Preferred={r_pref:.3f}, Null={r_null:.3f}, DSI={reichardt_dsi:.3f}")
    reichardt_gate_passed = reichardt_dsi > 0.70

    # 3. Factorial Matrix Parameters
    column_grid = [8, 16, 32, 64]
    gamma_grid = [0.0, 0.25, 0.5, 1.0, 2.0]

    # Fixed Phase 1B optimal parameters
    FROZEN_TAU_SLOW = 60.0
    FROZEN_DELAY_MS = 20.0
    FROZEN_INH_WEIGHT = -14.0

    print(f"\n[Experimental Design] 2x4 Factorial + Continuous Response Surface")
    print(f"  Column Resolutions: {column_grid}")
    print(f"  Coincidence Gamma:  {gamma_grid}")
    print(f"  Frozen Dynamics:    tau_slow={FROZEN_TAU_SLOW}ms, delay={FROZEN_DELAY_MS}ms, inh_weight={FROZEN_INH_WEIGHT}")

    surface_results: Dict[Tuple[int, float], Dict[str, Any]] = {}
    primary_2x4: Dict[str, float] = {}

    last_graph = None

    for n_col in column_grid:
        for gamma in gamma_grid:
            print(f"  Running condition: N_cols={n_col:2d}, gamma={gamma:4.2f} ...", end="", flush=True)
            t4a_sum, t4b_sum, hs_r, hs_l, graph = evaluate_circuit_condition(
                num_columns=n_col,
                coincidence_gamma=gamma,
                tau_slow=FROZEN_TAU_SLOW,
                delay_ms=FROZEN_DELAY_MS,
                inhibition_weight=FROZEN_INH_WEIGHT,
            )
            last_graph = graph
            res = {
                "t4a_mean_dsi": round(t4a_sum.mean_dsi, 4),
                "t4a_median_dsi": round(t4a_sum.median_dsi, 4),
                "t4a_vector_strength": round(t4a_sum.mean_vector_strength, 4),
                "t4a_fraction_pd_reliable": round(t4a_sum.fraction_pd_reliable, 4),
                "t4a_circ_var": round(t4a_sum.mean_circular_variance, 4),
                "t4a_card_err": round(t4a_sum.mean_cardinal_error_deg, 2),
                "t4a_peak": round(t4a_sum.mean_peak_response, 2),
                "t4a_baseline": round(t4a_sum.mean_baseline_response, 2),
                "t4a_variance": round(t4a_sum.mean_trial_variance, 2),
                "t4a_latency_ms": round(t4a_sum.mean_latency_ms, 2) if t4a_sum.mean_latency_ms else None,
                "hs_r_dsi": round(hs_r.dsi, 4),
                "hs_r_pd": round(hs_r.preferred_direction_deg, 1) if hs_r.preferred_direction_deg is not None else None,
                "hs_r_vector_strength": round(hs_r.vector_strength, 4),
                "hs_r_pd_reliable": hs_r.vector_pd_reliable,
            }
            surface_results[(n_col, gamma)] = res
            print(f" -> T4a Discrete DSI={t4a_sum.mean_dsi:.3f}, V_Strength={t4a_sum.mean_vector_strength:.3f}, Reliable={t4a_sum.fraction_pd_reliable*100:.0f}% | HS DSI={hs_r.dsi:.3f}")

            if gamma == 0.0:
                primary_2x4[f"A{n_col}"] = round(t4a_sum.mean_dsi, 4)
            elif gamma == 1.0:
                primary_2x4[f"E{n_col}"] = round(t4a_sum.mean_dsi, 4)

    # 4. Attribution & Factorial Effects
    # Main effect of spatial resolution under linear regime
    delta_res_linear = primary_2x4["A64"] - primary_2x4["A8"]
    # Main effect of spatial resolution under nonlinear regime
    delta_res_nonlin = primary_2x4["E64"] - primary_2x4["E8"]
    # Main effect of nonlinearity at 8 columns
    delta_nonlin_8 = primary_2x4["E8"] - primary_2x4["A8"]
    # Main effect of nonlinearity at 64 columns
    delta_nonlin_64 = primary_2x4["E64"] - primary_2x4["A64"]

    # Maximum DSI achieved across all tested conditions
    max_dsi_condition = max(surface_results.items(), key=lambda item: item[1]["t4a_mean_dsi"])
    max_dsi_coords = max_dsi_condition[0]
    max_dsi_val = max_dsi_condition[1]["t4a_mean_dsi"]

    print("\n" + "=" * 80)
    print("PRIMARY 2x4 FACTORIAL MATRIX (T4a Mean DSI):")
    print(f"  {'Resolution':<12} | {'Linear (A)':<14} | {'Nonlinear E (gamma=1.0)':<24}")
    print("  " + "-" * 56)
    for n_col in column_grid:
        a_val = primary_2x4[f"A{n_col}"]
        e_val = primary_2x4[f"E{n_col}"]
        print(f"  {n_col:>2} columns   | {a_val:<14.4f} | {e_val:<24.4f}")
    print("=" * 80)

    print("\nFACTORIAL ATTRIBUTION SUMMARY:")
    print(f"  Delta Spatial (A8 -> A64 Linear):     {delta_res_linear:+.4f}")
    print(f"  Delta Spatial (E8 -> E64 Nonlinear):  {delta_res_nonlin:+.4f}")
    print(f"  Delta Nonlinearity (A8 -> E8):        {delta_nonlin_8:+.4f}")
    print(f"  Delta Nonlinearity (A64 -> E64):      {delta_nonlin_64:+.4f}")
    print(f"  Global Maximum DSI:                   {max_dsi_val:.4f} at N={max_dsi_coords[0]}, gamma={max_dsi_coords[1]}")

    # 5. Determine Epistemic Outcome Category (A, B, C, or D)
    target_adequacy = 0.40
    outcome_category = ""
    outcome_narrative = ""

    spatial_effect_substantial = delta_res_linear > 0.15 or delta_res_nonlin > 0.15
    nonlin_effect_substantial = delta_nonlin_8 > 0.15 or delta_nonlin_64 > 0.15

    if max_dsi_val >= target_adequacy:
        if spatial_effect_substantial and not nonlin_effect_substantial:
            outcome_category = "Outcome A (Resolution dominant / H3 supported)"
            outcome_narrative = "Spatial column density directly recovers directional selectivity; nonlinearity is largely redundant."
        elif nonlin_effect_substantial and not spatial_effect_substantial:
            outcome_category = "Outcome B (Nonlinearity dominant / H4 minimal supported)"
            outcome_narrative = "Nonlinear coincidence integration recovers directional selectivity; spatial density has minimal effect."
        else:
            outcome_category = "Outcome C (Interaction: Spatial Resolution & Nonlinear Integration)"
            outcome_narrative = "Both spatial column density and nonlinear coincidence integration interact to produce high directional selectivity."
    else:
        if spatial_effect_substantial or nonlin_effect_substantial:
            outcome_category = "Sub-adequate Gain"
            outcome_narrative = f"Modest improvements observed (max DSI={max_dsi_val:.3f}), but biological threshold {target_adequacy} was not reached."
        else:
            outcome_category = "Outcome D (Neither Sufficient / Generic Abstraction Failure)"
            outcome_narrative = (
                "Neither spatial resolution nor minimal nonlinear coincidence recovers biological DSI. "
                "Generic point-neuron approximations fail; real MaleCNS anatomical synapse geometry is required."
            )

    print(f"\nEpistemic Diagnosis: {outcome_category}")
    print(f"  {outcome_narrative}")

    # 6. Hermetic Bundle Packaging
    run_id = f"phase1b_spatial_nonlinearity_{int(time.time())}"
    
    # Flatten surface results for metrics.json
    metrics_dict: Dict[str, Any] = {
        "synthetic_reichardt_dsi": round(reichardt_dsi, 4),
        "delta_res_linear": round(delta_res_linear, 4),
        "delta_res_nonlin": round(delta_res_nonlin, 4),
        "delta_nonlin_8": round(delta_nonlin_8, 4),
        "delta_nonlin_64": round(delta_nonlin_64, 4),
        "max_dsi": round(max_dsi_val, 4),
        "max_dsi_columns": max_dsi_coords[0],
        "max_dsi_gamma": max_dsi_coords[1],
        "epistemic_outcome": outcome_category,
    }
    for (ncol, gam), data in surface_results.items():
        prefix = f"N{ncol}_g{str(gam).replace('.', '_')}"
        for k, v in data.items():
            metrics_dict[f"{prefix}_{k}"] = v

    bundle = ExperimentBundle(
        run_id=run_id,
        experiment_name="Phase1B_Spatial_Resolution_and_Nonlinear_Coincidence_Study",
        config={
            "column_grid": column_grid,
            "gamma_grid": gamma_grid,
            "frozen_tau_slow_ms": FROZEN_TAU_SLOW,
            "frozen_delay_ms": FROZEN_DELAY_MS,
            "frozen_inh_weight": FROZEN_INH_WEIGHT,
            "sweep_velocity_px_s": 64.0,
            "target_biological_dsi": target_adequacy,
        },
        provenance_registry=reg,
        dataset_fingerprint=last_graph.compute_fingerprint(),
        metrics=metrics_dict,
    )

    bundle.add_gate(
        name="analytical_reichardt_control_gate",
        passed=reichardt_gate_passed,
        observed_value=f"DSI={reichardt_dsi:.3f}",
        threshold="DSI > 0.70",
        rationale="Synthetic Reichardt coincidence detector must demonstrate directional selectivity in isolation",
    )
    bundle.add_gate(
        name="spatial_resolution_attribution_gate",
        passed=delta_res_linear > 0.05 or delta_res_nonlin > 0.05,
        observed_value=f"Delta_Linear={delta_res_linear:+.3f}, Delta_Nonlin={delta_res_nonlin:+.3f}",
        threshold="Delta_DSI > +0.05",
        rationale="Increasing spatial column resolution from 8 to 64 enhances directional coincidence",
    )
    bundle.add_gate(
        name="nonlinearity_attribution_gate",
        passed=delta_nonlin_8 > 0.05 or delta_nonlin_64 > 0.05,
        observed_value=f"Delta_8col={delta_nonlin_8:+.3f}, Delta_64col={delta_nonlin_64:+.3f}",
        threshold="Delta_DSI > +0.05",
        rationale="Model E coincidence nonlinearity enhances directional coincidence",
    )
    bundle.add_gate(
        name="biological_adequacy_gate",
        passed=max_dsi_val >= target_adequacy,
        observed_value=f"Max_DSI={max_dsi_val:.3f} (N={max_dsi_coords[0]}, gamma={max_dsi_coords[1]})",
        threshold=f"DSI >= {target_adequacy}",
        rationale="Spatial scaling and/or Model E coincidence nonlinearity recovers biological DSI",
    )
    bundle.add_gate(
        name="source_provenance_gate",
        passed=bundle.git_hash != "git_hash_unavailable",
        observed_value=f"commit={bundle.git_hash[:10]}, dirty={bundle.git_provenance.get('dirty')}",
        threshold="identifiable git commit SHA",
        rationale="Experiment must be sealed against a known git commit",
    )

    bundle_dir = bundle.save(Path("runs"))
    print(f"\nSealed experiment bundle saved to: {bundle_dir}/")
    print(f"Report written to: {bundle_dir / 'report.md'}")
    print("\n" + "=" * 80)
    print("PHASE 1B SPATIAL & NONLINEARITY STUDY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
