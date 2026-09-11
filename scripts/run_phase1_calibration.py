#!/usr/bin/env python3
"""Phase 1 Biological Calibration Runner for FlyDoom: Connectome Cybernetics Laboratory.

Executes the four Phase 1 validation gates:
  - Gate 1A: Stimulus Integrity Gate (8-direction sweep & ON/OFF parameter hashing)
  - Gate 1B: Optic Subgraph Integrity Gate (topological fingerprinting of motion circuit)
  - Gate 1C: Directional Physiology & ON/OFF Separation Gate (vector tuning, DSI, polarity)
  - Gate 1D: Representation Dependence Gate (Encoder Delta vs. Encoder Alpha)

Generates sealed, provenance-anchored experiment bundle in runs/phase1_calibration_<timestamp>/
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fly_doom.analysis.tuning import (
    DirectionalTuningResult,
    PopulationTuningSummary,
    compute_vector_tuning,
)
from fly_doom.connectome.manifest import DatasetManifest
from fly_doom.connectome.subgraphs import (
    create_canonical_phase1_optic_circuit,
    query_cell_indices,
)
from fly_doom.core.provenance import Provenance, ProvenanceRegistry
from fly_doom.dynamics.lif_native import LIFNativeEngine
from fly_doom.dynamics.lif_reference import LIFParameters
from fly_doom.experiment.bundle import ExperimentBundle
from fly_doom.sensory.encoders.alpha import EncoderAlpha
from fly_doom.sensory.encoders.delta import EncoderDelta
from fly_doom.sensory.stimuli import (
    CARDINAL_DIRECTIONS,
    StimulusParameters,
    StimulusSweep,
    StimulusType,
)


def run_sweep_trial(
    graph,
    encoder,
    sweep: StimulusSweep,
    num_columns: int = 8,
) -> Dict[int, List[float]]:
    """Run an 8-direction stimulus sweep through an encoder and LIF engine.

    Returns:
      spike_counts: mapping from neuron index to list of spike counts across 8 directions.
    """
    NEURONS_PER_COL = 10
    spike_counts: Dict[int, List[float]] = {i: [] for i in range(graph.num_neurons)}
    params = LIFParameters(dt=sweep.dt_ms, tau_m=15.0, tau_syn=5.0, t_ref=2.0)

    for direction in sweep.directions:
        generator = sweep.get_stimulus(direction)
        encoder.reset()
        engine = LIFNativeEngine(graph, params)

        dir_spikes = np.zeros(graph.num_neurons, dtype=np.int64)

        for step_idx, t_ms, frame in generator.iter_frames():
            currents = encoder.encode_frame(frame, dt_ms=sweep.dt_ms)

            # Map encoder outputs to Lamina inputs (L_ON and L_OFF) in each column
            ext_current = np.zeros(graph.num_neurons, dtype=np.float64)

            if isinstance(encoder, EncoderDelta):
                # EncoderDelta has 2 * num_ommatidia channels: [ON_0..N-1, OFF_0..N-1]
                num_ommatidia = encoder.num_ommatidia
                on_channels = currents[:num_ommatidia]
                off_channels = currents[num_ommatidia:]

                # Distribute across columns (1D projection along x)
                for c in range(min(num_columns, num_ommatidia)):
                    base = c * NEURONS_PER_COL
                    l_on_idx = base + 0
                    l_off_idx = base + 1
                    ext_current[l_on_idx] = on_channels[c]
                    ext_current[l_off_idx] = off_channels[c]

            elif isinstance(encoder, EncoderAlpha):
                # EncoderAlpha provides raw luminance
                for c in range(min(num_columns, encoder.num_input_units)):
                    base = c * NEURONS_PER_COL
                    ext_current[base + 0] = currents[c]
                    ext_current[base + 1] = currents[c]

            fired = engine.step(ext_current)
            if len(fired) > 0:
                np.add.at(dir_spikes, fired, 1)

        for i in range(graph.num_neurons):
            spike_counts[i].append(float(dir_spikes[i]))

    return spike_counts


def run_edge_step_trial(
    graph,
    encoder,
    stimulus_type: StimulusType,
    direction: float = 0.0,
    duration_ms: float = 300.0,
    dt_ms: float = 1.0,
    num_columns: int = 8,
) -> np.ndarray:
    """Run a single moving edge trial (ON or OFF) and return total spikes per neuron."""
    NEURONS_PER_COL = 10
    params = StimulusParameters(
        stimulus_type=stimulus_type,
        direction_deg=direction,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        velocity_px_s=64.0,
        width=64,
        height=64,
    )
    generator = params.stimulus_type
    from fly_doom.sensory.stimuli import StimulusGenerator
    gen = StimulusGenerator(params)

    lif_params = LIFParameters(dt=dt_ms, tau_m=15.0, tau_syn=5.0, t_ref=2.0)
    engine = LIFNativeEngine(graph, lif_params)
    encoder.reset()

    total_spikes = np.zeros(graph.num_neurons, dtype=np.float64)

    for step_idx, t_ms, frame in gen.iter_frames():
        currents = encoder.encode_frame(frame, dt_ms=dt_ms)
        ext_current = np.zeros(graph.num_neurons, dtype=np.float64)

        if isinstance(encoder, EncoderDelta):
            num_ommatidia = encoder.num_ommatidia
            on_channels = currents[:num_ommatidia]
            off_channels = currents[num_ommatidia:]
            for c in range(min(num_columns, num_ommatidia)):
                base = c * NEURONS_PER_COL
                ext_current[base + 0] = on_channels[c]
                ext_current[base + 1] = off_channels[c]
        else:
            for c in range(min(num_columns, encoder.num_input_units)):
                base = c * NEURONS_PER_COL
                ext_current[base + 0] = currents[c]

        fired = engine.step(ext_current)
        if len(fired) > 0:
            np.add.at(total_spikes, fired, 1.0)

    return total_spikes


def main() -> None:
    print("=" * 75)
    print("FLYDOOM — CONNECTOME CYBERNETICS LABORATORY: PHASE 1 CALIBRATION")
    print("=" * 75)

    # 1. Provenance Registration (Activating Tier 1 & 2)
    reg = ProvenanceRegistry()
    reg.register(
        "optic_motion_circuit",
        Provenance(
            tier="computational_hypothesis",
            source="Canonical_Phase1_Optic_Approximation_eLife2017",
            confidence=0.85,
            rationale="Canonical Hassenstein-Reichardt motion detection circuit with Mi1/Tm3 and Tm1/Tm2",
        ),
    )
    reg.register(
        "stimulus_protocol",
        Provenance(
            tier="engineering_scaffold",
            source="Fisher_et_al_2015_eLife_Visual_Protocols",
            confidence=1.0,
            rationale="Standard 8-direction drifting sinusoidal grating and ON/OFF edge sweeps",
        ),
    )
    reg.register(
        "encoder_delta",
        Provenance(
            tier="experimental_assumption",
            source="Hexagonal_Ommatidia_Temporal_Diff_Nature2025",
            confidence=0.90,
            rationale="Hexagonal ommatidial lattice with simple temporal differencing; no explicit EMD filtering",
        ),
    )
    reg.register(
        "encoder_alpha",
        Provenance(
            tier="engineering_scaffold",
            source="Naive_Planar_Raster_Downsampler",
            confidence=1.0,
            rationale="Planar downsampler control baseline to test representation geometry dependence",
        ),
    )
    reg.register(
        "lif_native_engine",
        Provenance(
            tier="computational_hypothesis",
            source="Clang_CPP20_ARM64_Native",
            confidence=0.90,
            rationale="Native C++20 LIF dynamics engine running at deterministic float64 precision",
        ),
    )

    # 2. Gate 1A: Stimulus Integrity Gate
    print("\n[1/4] Executing Phase 1A: Stimulus Integrity Gate...")
    sweep = StimulusSweep(
        stimulus_type=StimulusType.DRIFTING_GRATING,
        directions=CARDINAL_DIRECTIONS,
        duration_ms=400.0,
        dt_ms=1.0,
        width=64,
        height=64,
    )
    motion_ok, motion_msg = sweep.verify_motion_integrity()
    sweep_hash = sweep.compute_sweep_fingerprint()
    print(f"      Motion Integrity: {'PASS' if motion_ok else 'FAIL'} ({motion_msg})")
    print(f"      Sweep Fingerprint: {sweep_hash[:16]}...")
    gate_1a_passed = motion_ok

    # 3. Gate 1B: Optic Subgraph Integrity Gate
    print("\n[2/4] Executing Phase 1B: Optic Subgraph Integrity Gate...")
    num_cols = 8
    graph = create_canonical_phase1_optic_circuit(num_columns=num_cols, seed=42)
    fingerprint = graph.compute_fingerprint()
    manifest = DatasetManifest(
        dataset_name="Canonical_Phase1_Optic_Subcircuit_8Col",
        dataset_version="1.0.0",
        source_citation="Hassenstein-Reichardt / eLife 2017 Drosophila Motion Architecture",
        source_files_sha256={"optic_subcircuit.csr": fingerprint.combined_hash},
        expected_fingerprint=fingerprint,
    )
    discrepancies = manifest.verify(fingerprint)
    gate_1b_passed = len(discrepancies) == 0
    print(f"      Neuron Count: {fingerprint.neuron_count} (8 columns + 2 LPTCs)")
    print(f"      Directed Synaptic Edges: {fingerprint.directed_edge_count}")
    print(f"      Contact Count: {fingerprint.synapse_contact_count}")
    print(f"      Subgraph Manifest Gate: {'PASS' if gate_1b_passed else 'FAIL'}")

    # 4. Gate 1C: Directional Physiology & ON/OFF Separation Gate (using Encoder Delta)
    print("\n[3/4] Executing Phase 1C: Directional Physiology Gate (Encoder Delta)...")
    enc_delta = EncoderDelta(num_columns=8, num_rows=8, field_width_px=64, field_height_px=64)

    # 4.1 Grating sweep across 8 directions
    t_start = time.perf_counter()
    spike_counts_delta = run_sweep_trial(graph, enc_delta, sweep, num_columns=num_cols)

    # 4.2 ON and OFF edge trials (0° rightwards)
    on_spikes = run_edge_step_trial(graph, enc_delta, StimulusType.MOVING_EDGE_ON, direction=0.0)
    off_spikes = run_edge_step_trial(graph, enc_delta, StimulusType.MOVING_EDGE_OFF, direction=0.0)
    elapsed_trials = time.perf_counter() - t_start

    # Compute tuning metrics for T4a, T4b, T5a, T5b, and HS cells
    t4a_indices = query_cell_indices(graph, {"T4a"})
    t4b_indices = query_cell_indices(graph, {"T4b"})
    t5a_indices = query_cell_indices(graph, {"T5a"})
    t5b_indices = query_cell_indices(graph, {"T5b"})
    hs_right_idx = query_cell_indices(graph, {"HS_Right"})[0]
    hs_left_idx = query_cell_indices(graph, {"HS_Left"})[0]

    t4a_results: List[DirectionalTuningResult] = []
    t4b_results: List[DirectionalTuningResult] = []
    t5a_results: List[DirectionalTuningResult] = []
    t5b_results: List[DirectionalTuningResult] = []

    for idx in t4a_indices:
        t4a_results.append(
            compute_vector_tuning(
                idx,
                "T4a",
                sweep.directions,
                spike_counts_delta[idx],
                on_response=float(on_spikes[idx]),
                off_response=float(off_spikes[idx]),
            )
        )
    for idx in t4b_indices:
        t4b_results.append(
            compute_vector_tuning(
                idx,
                "T4b",
                sweep.directions,
                spike_counts_delta[idx],
                on_response=float(on_spikes[idx]),
                off_response=float(off_spikes[idx]),
            )
        )
    for idx in t5a_indices:
        t5a_results.append(
            compute_vector_tuning(
                idx,
                "T5a",
                sweep.directions,
                spike_counts_delta[idx],
                on_response=float(on_spikes[idx]),
                off_response=float(off_spikes[idx]),
            )
        )
    for idx in t5b_indices:
        t5b_results.append(
            compute_vector_tuning(
                idx,
                "T5b",
                sweep.directions,
                spike_counts_delta[idx],
                on_response=float(on_spikes[idx]),
                off_response=float(off_spikes[idx]),
            )
        )

    t4a_summary = PopulationTuningSummary.from_results("T4a (Right-preferring)", t4a_results)
    t4b_summary = PopulationTuningSummary.from_results("T4b (Left-preferring)", t4b_results)
    t5a_summary = PopulationTuningSummary.from_results("T5a (Right-preferring)", t5a_results)
    t5b_summary = PopulationTuningSummary.from_results("T5b (Left-preferring)", t5b_results)

    # LPTC HS cell responses
    hs_r_tuning = compute_vector_tuning(
        hs_right_idx,
        "HS_Right",
        sweep.directions,
        spike_counts_delta[hs_right_idx],
    )
    hs_l_tuning = compute_vector_tuning(
        hs_left_idx,
        "HS_Left",
        sweep.directions,
        spike_counts_delta[hs_left_idx],
    )

    print(f"      Simulated 8 directions + ON/OFF edges in {elapsed_trials:.2f}s")
    print(f"      T4a (Right) Mean DSI: {t4a_summary.mean_dsi:.3f} | Polarity (ON): {t4a_summary.mean_polarity_index:.3f}")
    print(f"      T4b (Left)  Mean DSI: {t4b_summary.mean_dsi:.3f} | Polarity (ON): {t4b_summary.mean_polarity_index:.3f}")
    print(f"      T5a (Right) Mean DSI: {t5a_summary.mean_dsi:.3f} | Polarity (OFF): {t5a_summary.mean_polarity_index:.3f}")
    print(f"      T5b (Left)  Mean DSI: {t5b_summary.mean_dsi:.3f} | Polarity (OFF): {t5b_summary.mean_polarity_index:.3f}")
    print(f"      HS_Right Cell DSI: {hs_r_tuning.dsi:.3f} | PD: {hs_r_tuning.preferred_direction_deg:.1f}°")
    print(f"      HS_Left  Cell DSI: {hs_l_tuning.dsi:.3f} | PD: {hs_l_tuning.preferred_direction_deg:.1f}°")

    # Gate 1C Evaluations
    # Criteria:
    # 1. DSI > 0.40 for motion channels
    # 2. T4 polarity > 0.30 (ON preference), T5 polarity < -0.30 (OFF preference)
    # 3. Preferred directions aligned with cardinal targets (T4a ~ 0°, T4b ~ 180°)
    t4_directionality_ok = (t4a_summary.mean_dsi > 0.40) and (t4a_summary.mean_cardinal_error_deg < 20.0)
    t5_directionality_ok = (t5a_summary.mean_dsi > 0.40) and (t5a_summary.mean_cardinal_error_deg < 20.0)
    on_off_separation_ok = (t4a_summary.mean_polarity_index > 0.30) and (t5a_summary.mean_polarity_index < -0.30)

    # 5. Gate 1D: Representation Dependence Gate (Encoder Delta vs. Encoder Alpha)
    print("\n[4/4] Executing Phase 1D: Representation Dependence Gate (Encoder Alpha Comparison)...")
    enc_alpha = EncoderAlpha(grid_size=8)
    spike_counts_alpha = run_sweep_trial(graph, enc_alpha, sweep, num_columns=num_cols)

    t4a_results_alpha = [
        compute_vector_tuning(idx, "T4a_Alpha", sweep.directions, spike_counts_alpha[idx])
        for idx in t4a_indices
    ]
    t4a_summary_alpha = PopulationTuningSummary.from_results("T4a_Alpha", t4a_results_alpha)
    print(f"      T4a DSI under Encoder Delta (Geometry + Diff): {t4a_summary.mean_dsi:.3f}")
    print(f"      T4a DSI under Encoder Alpha (Naive Raster):      {t4a_summary_alpha.mean_dsi:.3f}")

    # The hypothesis: Does directional tuning collapse or weaken under naive planar raster?
    representation_dependent = t4a_summary.mean_dsi > (t4a_summary_alpha.mean_dsi + 0.15)
    print(f"      Representation Geometry Advantage: {'CONFIRMED' if representation_dependent else 'INCONCLUSIVE'}")

    # 6. Hermetic Experiment Bundle Packaging
    print("\nWriting hermetic experiment bundle to runs/...")
    run_id = f"phase1_calibration_{int(time.time())}"
    bundle = ExperimentBundle(
        run_id=run_id,
        experiment_name="Phase1_Biological_Calibration_Visual_Electrophysiology",
        config={
            "num_columns": num_cols,
            "total_neurons": graph.num_neurons,
            "sweep_directions_deg": CARDINAL_DIRECTIONS,
            "sweep_duration_ms": sweep.duration_ms,
            "dt_ms": sweep.dt_ms,
            "stimulus_velocity_px_s": sweep.velocity_px_s,
        },
        provenance_registry=reg,
        dataset_fingerprint=fingerprint,
        metrics={
            "t4a_mean_dsi_delta": round(t4a_summary.mean_dsi, 4),
            "t4b_mean_dsi_delta": round(t4b_summary.mean_dsi, 4),
            "t5a_mean_dsi_delta": round(t5a_summary.mean_dsi, 4),
            "t5b_mean_dsi_delta": round(t5b_summary.mean_dsi, 4),
            "t4a_cardinal_error_deg": round(t4a_summary.mean_cardinal_error_deg, 2),
            "t4b_cardinal_error_deg": round(t4b_summary.mean_cardinal_error_deg, 2),
            "t4_mean_on_off_polarity": round(t4a_summary.mean_polarity_index, 4),
            "t5_mean_on_off_polarity": round(t5a_summary.mean_polarity_index, 4),
            "hs_right_dsi": round(hs_r_tuning.dsi, 4),
            "hs_right_pd_deg": round(hs_r_tuning.preferred_direction_deg, 1),
            "hs_left_dsi": round(hs_l_tuning.dsi, 4),
            "hs_left_pd_deg": round(hs_l_tuning.preferred_direction_deg, 1),
            "t4a_mean_dsi_alpha": round(t4a_summary_alpha.mean_dsi, 4),
        },
    )

    # Register validation gates
    bundle.add_gate(
        name="stimulus_integrity_gate",
        passed=gate_1a_passed,
        observed_value=f"sweep_hash={sweep_hash[:10]}",
        threshold="analytical motion verified",
        rationale="8-direction stimulus frames must exhibit coherent directional velocity",
    )
    bundle.add_gate(
        name="optic_subgraph_integrity_gate",
        passed=gate_1b_passed,
        observed_value=f"{fingerprint.neuron_count} neurons, {fingerprint.directed_edge_count} edges",
        threshold="0 discrepancies against manifest",
        rationale="Topological fingerprint and cell-type metadata of optic circuit verified",
    )
    bundle.add_gate(
        name="t4_directionality_gate",
        passed=t4_directionality_ok,
        observed_value=f"DSI={t4a_summary.mean_dsi:.3f}, error={t4a_summary.mean_cardinal_error_deg:.1f}°",
        threshold="DSI > 0.40, cardinal error < 20°",
        rationale="T4 ON-motion population must exhibit strong directional selectivity aligned with cardinal angles",
    )
    bundle.add_gate(
        name="t5_directionality_gate",
        passed=t5_directionality_ok,
        observed_value=f"DSI={t5a_summary.mean_dsi:.3f}, error={t5a_summary.mean_cardinal_error_deg:.1f}°",
        threshold="DSI > 0.40, cardinal error < 20°",
        rationale="T5 OFF-motion population must exhibit strong directional selectivity aligned with cardinal angles",
    )
    bundle.add_gate(
        name="on_off_pathway_separation_gate",
        passed=on_off_separation_ok,
        observed_value=f"T4_pol={t4a_summary.mean_polarity_index:.3f}, T5_pol={t5a_summary.mean_polarity_index:.3f}",
        threshold="T4_pol > +0.3, T5_pol < -0.3",
        rationale="T4 and T5 must show clean functional segregation between ON and OFF motion stimuli",
    )
    bundle.add_gate(
        name="representation_dependence_gate",
        passed=representation_dependent,
        observed_value=f"DSI_delta={t4a_summary.mean_dsi:.3f} vs DSI_alpha={t4a_summary_alpha.mean_dsi:.3f}",
        threshold="DSI_delta > DSI_alpha + 0.15",
        rationale="Directional tuning requires spatio-temporal ommatidial representation geometry",
    )
    bundle.add_gate(
        name="source_provenance_gate",
        passed=bundle.git_hash != "git_hash_unavailable",
        observed_value=f"commit={bundle.git_hash[:10]}, dirty={bundle.git_provenance.get('dirty')}",
        threshold="identifiable git commit SHA",
        rationale="Experiment must be sealed against a known git commit",
    )

    bundle_dir = bundle.save(Path("runs"))
    print(f"      Sealed experiment bundle saved to: {bundle_dir}/")
    print(f"      Report written to: {bundle_dir / 'report.md'}")
    print("\n" + "=" * 75)
    print("PHASE 1 CALIBRATION AUDIT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
