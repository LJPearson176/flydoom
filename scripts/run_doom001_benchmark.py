"""DOOM-001 Benchmark Evaluation Runner.

Runs the standardized DOOM-001 protocol comparing 4 controllers:
  1. RandomController (lower-bound anchor)
  2. BallisticForwardController (pure locomotive baseline)
  3. PointLIFMotionController (Point LIF EMD, DSI ~0.126)
  4. CompartmentalT4Controller (Active Compartmental T4, DSI 0.5902)

Outputs sealed bundle with:
  - metrics.json
  - episodes.parquet
  - report.md
  - COMPLETE lockfile
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import pyarrow as pa
import pyarrow.parquet as pq

from fly_doom.control.controllers import (
    BallisticForwardController,
    CompartmentalT4Controller,
    PointLIFMotionController,
    RandomController,
)
from fly_doom.doom.benchmark import BenchmarkSummary, Doom001Benchmark, EpisodeResult
from fly_doom.doom.mock_arena import MockDoomArena


def run_doom001_suite() -> Path:
    timestamp = int(time.time())
    run_dir = Path(f"runs/doom001_benchmark_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FLYDOOM EMBODIED SENSORIMOTOR BENCHMARK: DOOM-001")
    print(f"Output Directory: {run_dir}")
    print("=" * 80)

    # 10 standard evaluation seeds, 200 steps per episode for fast sealed benchmark
    benchmark_seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    max_steps = 250
    benchmark = Doom001Benchmark(seeds=benchmark_seeds, max_steps_per_episode=max_steps)

    controllers = [
        RandomController(seed=42),
        BallisticForwardController(),
        PointLIFMotionController(width=64, height=64),
        CompartmentalT4Controller(width=64, height=64),
    ]

    all_summaries: List[BenchmarkSummary] = []
    all_episodes: List[EpisodeResult] = []

    print("\n[Step 1] Running Multi-Seed Evaluations Across Controllers...")
    for ctrl in controllers:
        cname = ctrl.name
        print(f"  Evaluating {cname} across {len(benchmark_seeds)} seeds...")
        summary = benchmark.evaluate(ctrl)
        all_summaries.append(summary)
        all_episodes.extend(summary.episodes)
        print(f"    -> Doom Score: {summary.doom_score_mean} ± {summary.doom_score_sem}")
        print(f"    -> Kills: {summary.kills_mean} ± {summary.kills_sem} | Survival: {summary.survival_steps_mean} ± {summary.survival_steps_sem}")
        print(f"    -> Damage: {summary.damage_dealt_mean} ± {summary.damage_dealt_sem} | Dist: {summary.distance_traveled_mean} ± {summary.distance_traveled_sem}")

    # Export episodes.parquet
    print("\n[Step 2] Exporting Telemetry and Trajectories to Parquet...")
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
        "doom_score": [],
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
        episodes_dict["doom_score"].append(ep.doom_score)

    pa_table = pa.Table.from_pydict(episodes_dict)
    parquet_path = run_dir / "episodes.parquet"
    pq.write_table(pa_table, parquet_path)

    # Export metrics.json
    print("\n[Step 3] Exporting Summary Metrics JSON...")
    metrics_payload = {
        "benchmark": "DOOM-001",
        "protocol_version": "1.0",
        "timestamp": timestamp,
        "seeds": benchmark_seeds,
        "max_steps": max_steps,
        "controllers": {},
    }
    for s in all_summaries:
        metrics_payload["controllers"][s.controller_name] = {
            "doom_score_mean": s.doom_score_mean,
            "doom_score_sem": s.doom_score_sem,
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
    print("\n[Step 4] Generating Scientific Benchmark Report...")
    report_lines = [
        "# DOOM-001 Embodied Sensorimotor Benchmark Report",
        "",
        f"**Run ID:** `{run_dir.name}`  ",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp))} UTC  ",
        "**Protocol:** DOOM-001 (10 evaluation seeds, E1M1 visual raycaster arena, discrete 5-action space)",
        "",
        "## Executive Summary",
        "",
        "This benchmark tests the core hypothesis of FlyDoom: **does biological motion fidelity (moving from Point-LIF EMD to active dendritic compartmentalization) improve embodied task fitness in Doom?**",
        "",
        "| Controller | DSI Substrate | Doom Score (Mean ± SEM) | Kills | Damage Dealt | Survival (Steps) | Distance Traveled |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    dsi_map = {
        "RandomController": "N/A (Random)",
        "BallisticForwardController": "N/A (Ballistic)",
        "PointLIFMotionController": "DSI = 0.1264 (Point LIF)",
        "CompartmentalT4Controller": "DSI = 0.5902 (Active Dendrite)",
    }

    for s in all_summaries:
        dsi_sub = dsi_map.get(s.controller_name, "Unknown")
        report_lines.append(
            f"| **{s.controller_name}** | {dsi_sub} | **{s.doom_score_mean:.2f} ± {s.doom_score_sem:.2f}** | "
            f"{s.kills_mean:.2f} | {s.damage_dealt_mean:.1f} | {s.survival_steps_mean:.1f} | {s.distance_traveled_mean:.2f} |"
        )

    report_lines.extend([
        "",
        "## Epistemic Distinctions",
        "- **Arena Representation:** Headless 2.5D perspective raycaster simulator with moving Imp entities and spatial wall shading (`engineering_scaffold`).",
        "- **Retinal Encoding:** Hexagonal ommatidial lattice with Gaussian receptive fields and rectified ON/OFF temporal differencing (`experimental_assumption`).",
        "- **Neural Substrates:**",
        "  - `PointLIFMotionController`: Point LIF model with uniform parameters and no spatial branch segregation (`computational_hypothesis`).",
        "  - `CompartmentalT4Controller`: Reconstructed dendritic tree with leading-branch shunting and supralinear coincidence (`computational_hypothesis`).",
        "",
        "## Checksum Fingerprints",
        "```json",
        json.dumps({
            "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "episodes_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        }, indent=2),
        "```",
    ])

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(report_lines))

    # Lockfile COMPLETE
    complete_path = run_dir / "COMPLETE"
    complete_path.write_text(f"DOOM-001 Benchmark completed successfully at {timestamp}\n")

    # Update Observatory web data
    web_metrics = Path("web/doom001_latest.json")
    with open(web_metrics, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED AND SEALED AT: {run_dir}")
    print("=" * 80)
    return run_dir


if __name__ == "__main__":
    run_doom001_suite()
