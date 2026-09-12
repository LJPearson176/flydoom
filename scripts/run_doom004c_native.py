"""Run the DOOM-004C native competence matrix: Model A-D × suppression 0/3/5."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path regardless of execution environment
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fly_doom.doom.native_benchmark import NativeGZDoomBenchmark
from fly_doom.doom.native_competence import run_doom004c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--output", default="runs/doom004c_native_competence")
    parser.add_argument("--map", default="E1M1")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, action="append", help="seed; defaults to [1001]")
    parser.add_argument("--door-seeking", action="store_true", help="enable native door-seeking and perspective policy")
    parser.add_argument("--condition", action="append", help="repeatable condition name; defaults to all 12")
    args = parser.parse_args()
    seeds = args.seed if args.seed else [1001]
    benchmark = NativeGZDoomBenchmark(Path(args.output), map_name=args.map, iwad=args.iwad, max_steps=args.max_steps)
    rows = run_doom004c(benchmark, seeds, door_seeking=args.door_seeking, conditions=args.condition)
    print("\n" + "=" * 88)
    print(f"{'Condition':24s} | {'Fwd %':7s} | {'Turn %':7s} | {'Dist (units)':12s} | {'Streak':6s} | {'Kills':5s} | {'Surv':5s}")
    print("-" * 88)
    for row in rows:
        dist_str = f"{row.distance_traveled:.1f}" if row.distance_traveled is not None else "null"
        surv_str = str(row.survival) if row.survival is not None else "null"
        kills_str = str(row.kills) if row.kills is not None else "null"
        print(f"{row.condition:24s} | {row.forward_fraction:7.1%} | {row.turning_fraction:7.1%} | {dist_str:>12s} | {row.spin_lock_streak_max:6d} | {kills_str:>5s} | {surv_str:>5s}")
    print("=" * 88 + "\n")




if __name__ == "__main__":
    main()
