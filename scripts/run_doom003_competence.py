"""DOOM-003 Task-Competence & Disentangled Multi-Dimensional Benchmark.

Evaluates an embodied agent across a strictly controlled ablation matrix:
Holding CONSTANT:
  - 10 deterministic seeds (1001..1010)
  - 250 steps per episode
  - Retinal encoder (8x8 ommatidia, sigma=4.0px, gain=35.0)
  - Scale-invariant relative motion decoder (|hat{Delta}| > 0.15)
  - Directed corridor navigation task in MockDoomArena (Exit portal at 13.5, 12.0)

Ablation Conditions:
  1. RandomController (lower-bound anchor)
  2. BallisticForwardController (pure locomotive baseline)
  3. ModelA_PointLIF (Canonical Point LIF, DSI ~0.054)
  4. ModelB_TemporalPointLIF (Calibrated tau + delay, DSI ~0.126)
  5. ModelC_PassiveTree (Multi-compartment passive cable, DSI ~0.144)
  6. ModelD_ActiveTree (Full compartmental model with shunting & coincidence, DSI 0.5902)
  7. ModelD_Mi4_KO (GABAergic shunting inhibition ablated)
  8. ModelD_Mi9_KO (Glutamatergic distal inhibition ablated)
  9. ModelD_Tm3_KO (Delayed excitation ablated)
  10. ModelD_Mi1_KO (Fast central excitation ablated)

Outputs sealed run directory:
  - metrics.json
  - episodes.parquet
  - trajectory.parquet (Full tick-by-tick neural states, voltages, and goal distance)
  - report.md (Multi-dimensional breakdown: Navigation, Combat, Survival, Neural)
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
from fly_doom.doom.benchmark import BenchmarkSummary, Doom003Benchmark, EpisodeResult
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def run_doom003_suite() -> Path:
    timestamp = int(time.time())
    run_dir = Path(f"runs/doom003_competence_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FLYDOOM DOOM-003-MOCK: TASK-COMPETENCE & DISENTANGLED BENCHMARK")
    print(f"Output Directory: {run_dir}")
    print("=" * 80)

    benchmark_seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    max_steps = 250
    benchmark = Doom003Benchmark(seeds=benchmark_seeds, max_steps_per_episode=max_steps)

    controllers = [
        # Reference baselines
        RandomController(seed=42),
        BallisticForwardController(),

        # Controlled biophysical progression (FROZEN RELATIVE DECODER)
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_A, custom_name="ModelA_PointLIF"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_B, custom_name="ModelB_TemporalPointLIF"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_C, custom_name="ModelC_PassiveTree"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, custom_name="ModelD_ActiveTree"),

        # Model D mechanistic lesions
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, synaptic_knockouts={"Mi4"}, custom_name="ModelD_Mi4_KO"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, synaptic_knockouts={"Mi9"}, custom_name="ModelD_Mi9_KO"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, synaptic_knockouts={"Tm3"}, custom_name="ModelD_Tm3_KO"),
        ControlledT4Controller(model_type=CompartmentModelType.MODEL_D, synaptic_knockouts={"Mi1"}, custom_name="ModelD_Mi1_KO"),
    ]

    all_summaries: List[BenchmarkSummary] = []
    all_episodes: List[EpisodeResult] = []

    print("\n[Step 1] Running Task-Competence Evaluations Across Models...")
    for ctrl in controllers:
        cname = ctrl.name
        print(f"  Evaluating {cname} across {len(benchmark_seeds)} seeds...")
        summary = benchmark.evaluate(ctrl)
        all_summaries.append(summary)
        all_episodes.extend(summary.episodes)
        print(f"    -> Fitness: {summary.fitness_score_mean} ± {summary.fitness_score_sem} | Nav Progress: {summary.nav_progress_mean}%")
        print(f"    -> Kills: {summary.kills_mean} ± {summary.kills_sem} | Damage: {summary.damage_dealt_mean} ± {summary.damage_dealt_sem} | Accuracy: {summary.hit_accuracy_mean:.2f}")

    # Export episodes.parquet
    print("\n[Step 2] Exporting Disentangled Episode Metrics to episodes.parquet...")
    episodes_dict: Dict[str, List[Any]] = {
        "seed": [],
        "controller_name": [],
        "nav_progress_pct": [],
        "min_dist_to_goal": [],
        "final_dist_to_goal": [],
        "exit_reached": [],
        "distance_traveled": [],
        "wall_bumps": [],
        "kills": [],
        "damage_dealt": [],
        "damage_taken": [],
        "shots_fired": [],
        "hit_accuracy": [],
        "survival_steps": [],
        "health_remaining": [],
        "ammo_remaining": [],
        "total_reward": [],
        "std_norm_asymmetry": [],
        "flydoom_fitness_score": [],
    }
    for ep in all_episodes:
        episodes_dict["seed"].append(ep.seed)
        episodes_dict["controller_name"].append(ep.controller_name)
        episodes_dict["nav_progress_pct"].append(ep.nav_progress_pct)
        episodes_dict["min_dist_to_goal"].append(ep.min_dist_to_goal)
        episodes_dict["final_dist_to_goal"].append(ep.final_dist_to_goal)
        episodes_dict["exit_reached"].append(ep.exit_reached)
        episodes_dict["distance_traveled"].append(ep.distance_traveled)
        episodes_dict["wall_bumps"].append(ep.wall_bumps)
        episodes_dict["kills"].append(ep.kills)
        episodes_dict["damage_dealt"].append(ep.damage_dealt)
        episodes_dict["damage_taken"].append(ep.damage_taken)
        episodes_dict["shots_fired"].append(ep.shots_fired)
        episodes_dict["hit_accuracy"].append(ep.hit_accuracy)
        episodes_dict["survival_steps"].append(ep.survival_steps)
        episodes_dict["health_remaining"].append(ep.health_remaining)
        episodes_dict["ammo_remaining"].append(ep.ammo_remaining)
        episodes_dict["total_reward"].append(ep.total_reward)
        episodes_dict["std_norm_asymmetry"].append(ep.std_norm_asymmetry)
        episodes_dict["flydoom_fitness_score"].append(ep.flydoom_fitness_score)

    episodes_table = pa.Table.from_pydict(episodes_dict)
    episodes_parquet_path = run_dir / "episodes.parquet"
    pq.write_table(episodes_table, episodes_parquet_path)

    # Export trajectory.parquet
    print("\n[Step 3] Exporting Full Tick-by-Tick Telemetry to trajectory.parquet...")
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
        "norm_asymmetry": [],
        "center_depth": [],
        "dist_to_goal": [],
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
            traj_dict["norm_asymmetry"].append(tick.norm_asymmetry)
            traj_dict["center_depth"].append(tick.center_depth)
            traj_dict["dist_to_goal"].append(tick.dist_to_goal)

    traj_table = pa.Table.from_pydict(traj_dict)
    traj_parquet_path = run_dir / "trajectory.parquet"
    pq.write_table(traj_table, traj_parquet_path)
    print(f"  Logged {len(traj_dict['step']):,} individual frames with full neural state & goal tracking.")

    # Export metrics.json
    print("\n[Step 4] Exporting Multi-Dimensional Metrics JSON...")
    metrics_payload = {
        "benchmark": "DOOM-003-MOCK",
        "benchmark_type": "task_competence_and_disentangled_metrics",
        "protocol_version": "3.0",
        "timestamp": timestamp,
        "seeds": benchmark_seeds,
        "max_steps": max_steps,
        "controllers": {},
    }
    for s in all_summaries:
        metrics_payload["controllers"][s.controller_name] = {
            "flydoom_fitness_score_mean": s.fitness_score_mean,
            "flydoom_fitness_score_sem": s.fitness_score_sem,
            "navigation": {
                "nav_progress_mean_pct": s.nav_progress_mean,
                "nav_progress_sem": s.nav_progress_sem,
                "distance_traveled_mean": s.distance_traveled_mean,
                "distance_traveled_sem": s.distance_traveled_sem,
                "min_dist_to_goal_mean": s.min_dist_mean,
                "wall_bumps_mean": s.wall_bumps_mean,
                "exit_rate": s.exit_rate,
            },
            "combat": {
                "kills_mean": s.kills_mean,
                "kills_sem": s.kills_sem,
                "damage_dealt_mean": s.damage_dealt_mean,
                "damage_dealt_sem": s.damage_dealt_sem,
                "hit_accuracy_mean": s.hit_accuracy_mean,
                "hit_accuracy_sem": s.hit_accuracy_sem,
            },
            "survival": {
                "survival_steps_mean": s.survival_steps_mean,
                "survival_steps_sem": s.survival_steps_sem,
                "health_mean": s.health_mean,
            },
            "neural": {
                "std_norm_asymmetry_mean": s.std_norm_asymmetry_mean,
                "spike_rate_mean": s.spike_rate_mean,
            },
        }

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Generate comprehensive report.md
    print("\n[Step 5] Generating Scientific Report...")
    report_lines = [
        "# DOOM-003-MOCK Task-Competence & Disentangled Multi-Dimensional Benchmark Report",
        "",
        f"**Run ID:** `{run_dir.name}`  ",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp))} UTC  ",
        "**Protocol:** DOOM-003-MOCK (Disentangled 4-Dimensional Metrics + Scale-Normalized Optomotor Transduction)",
        "**Environment:** MockDoomArena (`engineering_scaffold`)",
        "",
        "## Executive Summary",
        "",
        "In DOOM-003-MOCK, we resolve the two primary confounds of DOOM-002:",
        "1. **Scale Normalization:** We replaced raw voltage difference with scale-invariant relative motion energy $\\hat{\\Delta} = \\hat{M}_R - \\hat{M}_L \\in [-1, 1]$.",
        "2. **Task Competence:** We introduced an explicit corridor navigation objective toward the exit portal `(13.5, 12.0)`, penalizing aimless wall-bumping and stationary spinning.",
        "",
        "### 1. Disentangled Multi-Dimensional Performance",
        "| Controller / Architecture | DSI | Navigation Progress (%) | Min Goal Dist (m) | Wall Bumps | Kills | Damage Dealt | Hit Accuracy | FlyDoom Fitness |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    dsi_map = {
        "RandomController": "N/A",
        "BallisticForwardController": "N/A",
        "ModelA_PointLIF": "0.0544",
        "ModelB_TemporalPointLIF": "0.1264",
        "ModelC_PassiveTree": "0.1444",
        "ModelD_ActiveTree": "0.5902",
        "ModelD_Mi4_KO": "Ablated",
        "ModelD_Mi9_KO": "Ablated",
        "ModelD_Tm3_KO": "Ablated",
        "ModelD_Mi1_KO": "Ablated",
    }

    for s in all_summaries:
        dsi_val = dsi_map.get(s.controller_name, "N/A")
        report_lines.append(
            f"| **{s.controller_name}** | {dsi_val} | **{s.nav_progress_mean:.1f} ± {s.nav_progress_sem:.1f}%** | "
            f"{s.min_dist_mean:.1f} | {s.wall_bumps_mean:.1f} | {s.kills_mean:.2f} | {s.damage_dealt_mean:.1f} | "
            f"{s.hit_accuracy_mean:.2f} | **{s.fitness_score_mean:.2f} ± {s.fitness_score_sem:.2f}** |"
        )

    report_lines.extend([
        "",
        "## Epistemic Provenance Declarations",
        "- **Environment:** `MockDoomArena` is an explicit `engineering_scaffold`. This is not a native ViZDoom result.",
        "- **Metric:** Performance is reported across four independent orthogonal dimensions (Navigation, Combat, Survival, Neural) rather than a single collapsed scalar.",
        "- **Sensorimotor Control:** Relative motion energy normalization ensures scale-invariance across biophysical cable architectures.",
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
    complete_path.write_text(f"DOOM-003-MOCK completed successfully at {timestamp}\n")

    # Update web cockpit
    web_metrics = Path("web/doom003_latest.json")
    with open(web_metrics, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED AND SEALED AT: {run_dir}")
    print("=" * 80)
    return run_dir


if __name__ == "__main__":
    run_doom003_suite()
