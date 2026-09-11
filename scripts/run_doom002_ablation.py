"""DOOM-002 Architecture-Controlled Embodied Ablation Runner.

Evaluates an embodied agent across a strictly controlled ablation matrix:
Holding CONSTANT:
  - 10 deterministic seeds (1001..1010)
  - 250 steps per episode
  - Retinal encoder (8x8 ommatidia, sigma=4.0px, gain=35.0)
  - Motor decoder rule (steer_thresh=0.50, fire_thresh=0.40, depth_thresh=3.50)
  - Arena geometry & physics (MockDoomArena)
  - FlyDoom Fitness function

Varying ONLY the neural substrate & biological lesions:
  1. ControlledT4_ModelA_PointLIF (Canonical Point LIF, DSI ~0.054)
  2. ControlledT4_ModelB_TemporalPointLIF (Calibrated tau + delay, DSI ~0.126)
  3. ControlledT4_ModelC_PassiveTree (Multi-compartment passive cable, DSI ~0.144)
  4. ControlledT4_ModelD_ActiveTree (Full compartmental model with shunting & coincidence, DSI 0.5902)
  5. ControlledT4_ModelD_Mi4_KO (GABAergic shunting inhibition ablated)
  6. ControlledT4_ModelD_Mi9_KO (Glutamatergic distal inhibition ablated)
  7. ControlledT4_ModelD_Tm3_KO (Delayed excitation ablated)
  8. ControlledT4_ModelD_Mi1_KO (Fast central excitation ablated)

Outputs sealed run directory:
  - metrics.json
  - episodes.parquet
  - trajectory.parquet (Tick-by-tick neural voltages, spikes, asymmetry, and action commands)
  - report.md
  - COMPLETE lockfile
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import pyarrow as pa
import pyarrow.parquet as pq

from fly_doom.control.controllers import (
    BallisticForwardController,
    ControlledT4Controller,
    RandomController,
)
from fly_doom.doom.benchmark import BenchmarkSummary, Doom002Benchmark, EpisodeResult
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def run_doom002_suite() -> Path:
    timestamp = int(time.time())
    run_dir = Path(f"runs/doom002_ablation_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FLYDOOM DOOM-002-MOCK: ARCHITECTURE-CONTROLLED EMBODIED ABLATION")
    print(f"Output Directory: {run_dir}")
    print("=" * 80)

    benchmark_seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    max_steps = 250
    benchmark = Doom002Benchmark(seeds=benchmark_seeds, max_steps_per_episode=max_steps)

    # Standard ablation matrix
    controllers = [
        # Reference baselines
        RandomController(seed=42),
        BallisticForwardController(),

        # Controlled biophysical progression (SAME MOTOR DECODER)
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_A, custom_name="ModelA_PointLIF"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_B, custom_name="ModelB_TemporalPointLIF"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_C, custom_name="ModelC_PassiveTree"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, custom_name="ModelD_ActiveTree"),

        # Biological lesion ablations (Model D knockouts)
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, synaptic_knockouts={"Mi4"}, custom_name="ModelD_Mi4_KO"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, synaptic_knockouts={"Mi9"}, custom_name="ModelD_Mi9_KO"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, synaptic_knockouts={"Tm3"}, custom_name="ModelD_Tm3_KO"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, synaptic_knockouts={"Mi1"}, custom_name="ModelD_Mi1_KO"),
    ]

    all_summaries: List[BenchmarkSummary] = []
    all_episodes: List[EpisodeResult] = []

    print("\n[Step 1] Running Controlled Multi-Seed Evaluations Across Substrates...")
    for ctrl in controllers:
        cname = ctrl.name
        print(f"  Evaluating {cname} across {len(benchmark_seeds)} seeds...")
        summary = benchmark.evaluate(ctrl)
        all_summaries.append(summary)
        all_episodes.extend(summary.episodes)
        print(f"    -> Fitness Score: {summary.fitness_score_mean} ± {summary.fitness_score_sem}")
        print(f"    -> Kills: {summary.kills_mean} ± {summary.kills_sem} | Damage: {summary.damage_dealt_mean} ± {summary.damage_dealt_sem} | Dist: {summary.distance_traveled_mean} ± {summary.distance_traveled_sem}")

    # Export episodes.parquet
    print("\n[Step 2] Exporting Episode Summaries to episodes.parquet...")
    episodes_dict: Dict[str, List[Any]] = {
        "seed": [],
        "controller_name": [],
        "survival_steps": [],
        "kills": [],
        "damage_dealt": [],
        "distance_traveled": [],
        "health_remaining": [],
        "ammo_remaining": [],
        "total_reward": [],
        "flydoom_fitness_score": [],
    }
    for ep in all_episodes:
        episodes_dict["seed"].append(ep.seed)
        episodes_dict["controller_name"].append(ep.controller_name)
        episodes_dict["survival_steps"].append(ep.survival_steps)
        episodes_dict["kills"].append(ep.kills)
        episodes_dict["damage_dealt"].append(ep.damage_dealt)
        episodes_dict["distance_traveled"].append(ep.distance_traveled)
        episodes_dict["health_remaining"].append(ep.health_remaining)
        episodes_dict["ammo_remaining"].append(ep.ammo_remaining)
        episodes_dict["total_reward"].append(ep.total_reward)
        episodes_dict["flydoom_fitness_score"].append(ep.flydoom_fitness_score)

    episodes_table = pa.Table.from_pydict(episodes_dict)
    episodes_parquet_path = run_dir / "episodes.parquet"
    pq.write_table(episodes_table, episodes_parquet_path)

    # Export trajectory.parquet (Full tick-by-tick neural causal trace)
    print("\n[Step 3] Exporting Full Tick-by-Tick Neural Telemetry to trajectory.parquet...")
    traj_dict: Dict[str, List[Any]] = {
        "seed": [],
        "controller_name": [],
        "step": [],
        "x": [],
        "y": [],
        "angle_rad": [],
        "health": [],
        "ammo": [],
        "action": [],
        "action_name": [],
        "reward": [],
        "retina_mean": [],
        "retina_delta": [],
        "t4_l_v": [],
        "t4_r_v": [],
        "t4_l_spike": [],
        "t4_r_spike": [],
        "motion_asymmetry": [],
        "center_depth": [],
    }
    for ep in all_episodes:
        for tick in ep.trajectory:
            traj_dict["seed"].append(tick.seed)
            traj_dict["controller_name"].append(tick.controller_name)
            traj_dict["step"].append(tick.step)
            traj_dict["x"].append(tick.x)
            traj_dict["y"].append(tick.y)
            traj_dict["angle_rad"].append(tick.angle_rad)
            traj_dict["health"].append(tick.health)
            traj_dict["ammo"].append(tick.ammo)
            traj_dict["action"].append(tick.action)
            traj_dict["action_name"].append(tick.action_name)
            traj_dict["reward"].append(tick.reward)
            traj_dict["retina_mean"].append(tick.retina_mean)
            traj_dict["retina_delta"].append(tick.retina_delta)
            traj_dict["t4_l_v"].append(tick.t4_l_v)
            traj_dict["t4_r_v"].append(tick.t4_r_v)
            traj_dict["t4_l_spike"].append(tick.t4_l_spike)
            traj_dict["t4_r_spike"].append(tick.t4_r_spike)
            traj_dict["motion_asymmetry"].append(tick.motion_asymmetry)
            traj_dict["center_depth"].append(tick.center_depth)

    traj_table = pa.Table.from_pydict(traj_dict)
    traj_parquet_path = run_dir / "trajectory.parquet"
    pq.write_table(traj_table, traj_parquet_path)
    print(f"  Logged {len(traj_dict['step']):,} individual frames with full neural state.")

    # Export metrics.json
    print("\n[Step 4] Exporting Summary Metrics JSON...")
    metrics_payload = {
        "benchmark": "DOOM-002-MOCK",
        "benchmark_type": "controlled_neural_ablation",
        "protocol_version": "2.0",
        "timestamp": timestamp,
        "seeds": benchmark_seeds,
        "max_steps": max_steps,
        "controllers": {},
    }
    for s in all_summaries:
        metrics_payload["controllers"][s.controller_name] = {
            "flydoom_fitness_score_mean": s.fitness_score_mean,
            "flydoom_fitness_score_sem": s.fitness_score_sem,
            "survival_steps_mean": s.survival_steps_mean,
            "survival_steps_sem": s.survival_steps_sem,
            "kills_mean": s.kills_mean,
            "kills_sem": s.kills_sem,
            "damage_dealt_mean": s.damage_dealt_mean,
            "damage_dealt_sem": s.damage_dealt_sem,
            "distance_traveled_mean": s.distance_traveled_mean,
            "distance_traveled_sem": s.distance_traveled_sem,
        }

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Generate report.md
    print("\n[Step 5] Generating Scientific Ablation Report...")
    report_lines = [
        "# DOOM-002-MOCK Architecture-Controlled Embodied Ablation Report",
        "",
        f"**Run ID:** `{run_dir.name}`  ",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp))} UTC  ",
        "**Protocol:** DOOM-002-MOCK (Controlled Ablation; Frozen Retinal & Motor Decoders)",
        "**Environment:** MockDoomArena (`engineering_scaffold`)",
        "",
        "## Executive Summary",
        "",
        "This experiment isolates the causal contribution of neural architecture by holding the sensory encoder,",
        "motor decoder, action thresholds, environment seeds, and fitness function strictly constant.",
        "",
        "### 1. Controlled Architecture Progression",
        "| Architecture | Neural Substrate | Intrinsic DSI | FlyDoom Fitness (Mean ± SEM) | Kills | Damage | Survival | Distance |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    dsi_map = {
        "RandomController": ("Random Action", "N/A"),
        "BallisticForwardController": ("Ballistic Locomotion", "N/A"),
        "ModelA_PointLIF": ("Point LIF (Lumped Soma)", "0.0544"),
        "ModelB_TemporalPointLIF": ("Temporal Point LIF (Delays)", "0.1264"),
        "ModelC_PassiveTree": ("Passive Cable Tree", "0.1444"),
        "ModelD_ActiveTree": ("Active Nonlinear Compartments", "0.5902"),
        "ModelD_Mi4_KO": ("Active Tree (Mi4 GABA KO)", "Ablated"),
        "ModelD_Mi9_KO": ("Active Tree (Mi9 Glu KO)", "Ablated"),
        "ModelD_Tm3_KO": ("Active Tree (Tm3 Slow ACh KO)", "Ablated"),
        "ModelD_Mi1_KO": ("Active Tree (Mi1 Fast ACh KO)", "Ablated"),
    }

    for s in all_summaries[:6]:
        sub_desc, dsi_val = dsi_map.get(s.controller_name, ("Unknown", "N/A"))
        report_lines.append(
            f"| **{s.controller_name}** | {sub_desc} | {dsi_val} | **{s.fitness_score_mean:.2f} ± {s.fitness_score_sem:.2f}** | "
            f"{s.kills_mean:.2f} | {s.damage_dealt_mean:.1f} | {s.survival_steps_mean:.1f} | {s.distance_traveled_mean:.2f} |"
        )

    report_lines.extend([
        "",
        "### 2. Biological Synaptic Lesion Ablations (Model D Knockouts)",
        "| Lesion Condition | Ablated Mechanism | FlyDoom Fitness (Mean ± SEM) | Kills | Damage | Survival | Distance |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for s in all_summaries[6:]:
        sub_desc, _ = dsi_map.get(s.controller_name, ("Unknown", "N/A"))
        report_lines.append(
            f"| **{s.controller_name}** | {sub_desc} | **{s.fitness_score_mean:.2f} ± {s.fitness_score_sem:.2f}** | "
            f"{s.kills_mean:.2f} | {s.damage_dealt_mean:.1f} | {s.survival_steps_mean:.1f} | {s.distance_traveled_mean:.2f} |"
        )

    report_lines.extend([
        "",
        "## Epistemic Provenance Declarations",
        "- **Environment:** `MockDoomArena` is an explicit `engineering_scaffold`. This is not a native ViZDoom result.",
        "- **Metric:** The primary fitness index is `FlyDoom Fitness Score`, defined as `100 * kills + 1.0 * damage + 0.5 * distance + 0.1 * steps`.",
        "- **Sensorimotor Control:** Retinal encoder and motor thresholding are held invariant across all models to ensure valid causal attribution.",
        "",
        "## Checksum Fingerprints",
        "```json",
        json.dumps({
            "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "episodes_sha256": hashlib.sha256(episodes_parquet_path.read_bytes()).hexdigest(),
            "trajectory_sha256": hashlib.sha256(traj_parquet_path.read_bytes()).hexdigest(),
        }, indent=2),
        "```",
    ])

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(report_lines))

    # Lockfile COMPLETE
    complete_path = run_dir / "COMPLETE"
    complete_path.write_text(f"DOOM-002-MOCK controlled ablation completed successfully at {timestamp}\n")

    # Update Observatory web data
    web_metrics = Path("web/doom002_latest.json")
    with open(web_metrics, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"CONTROLLED ABLATION COMPLETED AND SEALED AT: {run_dir}")
    print("=" * 80)
    return run_dir


if __name__ == "__main__":
    run_doom002_suite()
