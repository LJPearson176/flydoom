"""Run the controlled DOOM-004 native GZDoom benchmark.

This is intentionally explicit and bounded. It launches one fresh GZDoom
process per episode and writes a sealed directory per condition/seed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

# Ensure src/ is on sys.path regardless of execution environment
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import re
from fly_doom.doom.native_benchmark import (
    DEFAULT_LESION_CONDITIONS,
    DEFAULT_NATIVE_CONDITIONS,
    DEFAULT_SUPPRESSION_ABLATION,
    NativeCondition,
    NativeGZDoomBenchmark,
)
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def parse_condition_name(name: str) -> NativeCondition:
    """Parse a condition name dynamically if not present in defaults."""
    model_type = CompartmentModelType.MODEL_D
    if name.startswith("ModelA"):
        model_type = CompartmentModelType.MODEL_A
    elif name.startswith("ModelB"):
        model_type = CompartmentModelType.MODEL_B
    elif name.startswith("ModelC"):
        model_type = CompartmentModelType.MODEL_C

    saccade_ticks = 0
    saccade_match = re.search(r"_Saccade_(\d+)", name)
    if saccade_match:
        saccade_ticks = int(saccade_match.group(1))

    knockouts: set[str] = set()
    for ko in ("Mi4", "Mi9", "Tm3", "Mi1"):
        if f"{ko}_KO" in name or f"_{ko}" in name:
            knockouts.add(ko)

    return NativeCondition(
        name=name,
        model_type=model_type,
        knockouts=frozenset(knockouts),
        saccade_refractory_ticks=saccade_ticks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--output", default="runs/doom004_native")
    parser.add_argument("--map", default="E1M1")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, action="append", help="seed; can be specified multiple times (defaults to [1001])")
    parser.add_argument("--condition", action="append", help="repeatable condition name; defaults to all in selected mode")
    parser.add_argument("--architecture", action="store_true", help="run DOOM-004A Model A-D/lesion conditions with suppression fixed at 0")
    parser.add_argument("--suppression-ablation", action="store_true", help="run Model D with 0/3/5/10 refractory ticks")
    parser.add_argument("--lesions", action="store_true", help="run Model D lesion conditions (Mi4, Mi9, Tm3, Mi1 KO) with 3-tick suppression")
    parser.add_argument("--door-seeking", action="store_true", help="enable native stalled-door USE policy")
    args = parser.parse_args()

    seeds = args.seed if args.seed else [1001]
    modes = sum([bool(args.suppression_ablation), bool(args.architecture), bool(args.lesions)])
    if modes > 1:
        parser.error("choose at most one benchmark mode: --architecture, --suppression-ablation, or --lesions")
    all_known = {c.name: c for c in DEFAULT_NATIVE_CONDITIONS + DEFAULT_SUPPRESSION_ABLATION + DEFAULT_LESION_CONDITIONS}
    if args.condition:
        selected = [all_known.get(name) or parse_condition_name(name) for name in args.condition]
    elif args.lesions:
        selected = list(DEFAULT_LESION_CONDITIONS)
    elif args.suppression_ablation:
        selected = list(DEFAULT_SUPPRESSION_ABLATION)
    else:
        selected = list(DEFAULT_NATIVE_CONDITIONS)
    if args.door_seeking:
        selected = [replace(condition, door_seeking=True) for condition in selected]
    benchmark = NativeGZDoomBenchmark(
        Path(args.output), map_name=args.map, iwad=args.iwad, max_steps=args.max_steps
    )
    for condition in selected:
        for index, seed in enumerate(seeds):
            result = benchmark.run_episode(condition, seed, index)
            dist_str = f"{result.distance_traveled:.1f}" if result.distance_traveled is not None else "null"
            target_err = result.stability.get("mean_target_angle_error")
            target_err_str = f"{target_err:.1f} deg" if target_err is not None else "N/A"
            lock_frac = result.stability.get("target_lock_fraction")
            lock_frac_str = f"{lock_frac * 100:.1f}%" if lock_frac is not None else "N/A"
            asym_std = result.stability.get("norm_asymmetry_std")
            asym_std_str = f"{asym_std:.3f}" if asym_std is not None else "N/A"
            print(f"{result.episode_id}: {result.steps} steps, termination={result.termination_reason}, "
                  f"nav={result.navigation_status}, dist={dist_str}, survival={result.survival}, "
                  f"kills={result.kills} (player={result.player_kills}, ff={result.friendly_fire_kills}, dmg={result.damage_dealt}), "
                  f"targeting=[err={target_err_str}, lock={lock_frac_str}], optic_flow_std={asym_std_str}, "
                  f"actions={result.action_counts}")





if __name__ == "__main__":
    main()
