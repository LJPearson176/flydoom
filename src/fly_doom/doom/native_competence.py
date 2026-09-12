"""DOOM-004C native task-competence matrix.

This is deliberately separate from DOOM-004A (architecture) and DOOM-004B
(stability). It crosses neural substrate with the two most promising control
conditions while preserving unavailable native-world endpoints as null.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from fly_doom.doom.native_benchmark import NativeCondition, NativeEpisodeResult, NativeGZDoomBenchmark
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def doom004c_conditions(door_seeking: bool = False) -> Sequence[NativeCondition]:
    return tuple(
        NativeCondition(
            name=f"Model{model}_Saccade_{suppression}",
            model_type=model_type,
            saccade_refractory_ticks=suppression,
            door_seeking=door_seeking,
        )
        for model, model_type in (
            ("A", CompartmentModelType.MODEL_A),
            ("B", CompartmentModelType.MODEL_B),
            ("C", CompartmentModelType.MODEL_C),
            ("D", CompartmentModelType.MODEL_D),
        )
        for suppression in (0, 3, 5)
    )


@dataclass
class CompetenceRow:
    condition: str
    model: str
    suppression_ticks: int
    seed: int
    steps: int
    termination_reason: str
    forward_steps: int
    turn_steps: int
    fire_steps: int
    turning_fraction: float
    capture_valid_fraction: float
    use_steps: int = 0
    door_seeking: bool = False
    # Level 1 — Control metrics
    forward_fraction: float = 0.0
    fire_fraction: float = 0.0
    spin_lock_streak_max: int = 0
    # Level 2 — Visual locomotion & scene-change progress
    forward_motion_proxy: float = 0.0
    rotational_motion_proxy: float = 0.0
    visual_scene_variation: float = 0.0
    # Level 3 — Interaction
    fire_events: int = 0
    # Level 4 — Task Competence (explicitly unavailable without world-state adapter)
    navigation_progress: Optional[float] = None
    distance_traveled: Optional[float] = None
    wall_collisions: Optional[int] = None
    kills: Optional[int] = None
    player_kills: Optional[int] = None
    friendly_fire_kills: Optional[int] = None
    damage_dealt: Optional[float] = None
    survival: Optional[int] = None
    native_world_metrics_status: str = "unavailable_native_bridge"


def row_from_episode(result: NativeEpisodeResult) -> CompetenceRow:
    action_counts = result.action_counts
    valid = sum(1 for tick in result.trajectory if tick.capture_valid)
    model, _, suppression = result.condition.partition("_Saccade_")
    total_steps = max(1, result.steps)
    fwd_count = action_counts.get("FORWARD", 0)
    turn_count = action_counts.get("TURN_LEFT", 0) + action_counts.get("TURN_RIGHT", 0)
    fire_count = action_counts.get("FIRE", 0)

    # Compute observable visual motion proxies and spin-lock streak
    fwd_deltas = [t.retina_delta for t in result.trajectory if t.action == "FORWARD"]
    turn_asyms = [abs(t.norm_asymmetry) for t in result.trajectory if t.action in {"TURN_LEFT", "TURN_RIGHT"}]
    retina_means = [t.retina_mean for t in result.trajectory]

    max_turn_streak = 0
    curr_turn_streak = 0
    for tick in result.trajectory:
        if tick.action in {"TURN_LEFT", "TURN_RIGHT"}:
            curr_turn_streak += 1
            if curr_turn_streak > max_turn_streak:
                max_turn_streak = curr_turn_streak
        else:
            curr_turn_streak = 0

    scene_variation = float(np.std(retina_means)) if len(retina_means) > 1 else 0.0
    fwd_proxy = float(np.mean(fwd_deltas)) if fwd_deltas else 0.0
    rot_proxy = float(np.mean(turn_asyms)) if turn_asyms else 0.0
    native_available = result.navigation_status == "native_game_state_available"

    return CompetenceRow(
        condition=result.condition,
        model=model,
        suppression_ticks=int(suppression or 0),
        seed=result.seed,
        steps=result.steps,
        termination_reason=result.termination_reason,
        forward_steps=fwd_count,
        turn_steps=turn_count,
        fire_steps=fire_count,
        use_steps=action_counts.get("USE", 0),
        door_seeking=result.door_seeking,
        turning_fraction=float(result.stability.get("turning_fraction") or 0.0),
        capture_valid_fraction=valid / total_steps,
        forward_fraction=fwd_count / total_steps,
        fire_fraction=fire_count / total_steps,
        spin_lock_streak_max=max_turn_streak,
        forward_motion_proxy=fwd_proxy,
        rotational_motion_proxy=rot_proxy,
        visual_scene_variation=scene_variation,
        fire_events=fire_count,
        distance_traveled=result.distance_traveled if native_available else None,
        kills=result.kills if native_available else None,
        player_kills=result.player_kills if native_available else None,
        friendly_fire_kills=result.friendly_fire_kills if native_available else None,
        damage_dealt=result.damage_dealt if native_available else None,
        survival=result.survival if native_available else None,
        native_world_metrics_status=(
            "native_game_state_available" if native_available else "unavailable_native_bridge"
        ),
    )


def run_doom004c(
    benchmark: NativeGZDoomBenchmark,
    seeds: Sequence[int],
    *,
    door_seeking: bool = False,
    conditions: Optional[Sequence[str]] = None,
) -> List[CompetenceRow]:
    rows: List[CompetenceRow] = []
    all_conditions = doom004c_conditions(door_seeking=door_seeking)
    if conditions:
        all_conditions = [c for c in all_conditions if c.name in conditions]
    for condition in all_conditions:
        for index, seed in enumerate(seeds):
            row = row_from_episode(benchmark.run_episode(condition, seed, index))
            rows.append(row)
            dist_str = f"{row.distance_traveled:.1f}" if row.distance_traveled is not None else "null"
            print(f"[{len(rows):02d}/{len(all_conditions) * len(seeds):02d}] {row.condition:26s} | fwd={row.forward_steps:3d} ({row.forward_fraction:5.1%}) | "
                  f"turn={row.turn_steps:3d} ({row.turning_fraction:5.1%}) | "
                  f"dist={dist_str:7s} | streak={row.spin_lock_streak_max:3d}", flush=True)

    has_native = any(r.native_world_metrics_status == "native_game_state_available" for r in rows)
    summary_path = benchmark.output_dir / "doom004c_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "experiment": "DOOM-004C",
        "axes": {"neural_architecture": ["A", "B", "C", "D"], "suppression_ticks": [0, 3, 5]},
        "rows": [asdict(row) for row in rows],
        "unavailable_native_world_metrics": (
            ["navigation_progress", "wall_collisions"]
            if has_native
            else ["navigation_progress", "distance_traveled", "wall_collisions", "kills", "damage_dealt", "survival"]
        ),
    }, indent=2) + "\n")
    return rows
