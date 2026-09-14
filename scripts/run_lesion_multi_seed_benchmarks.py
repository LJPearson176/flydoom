#!/usr/bin/env python3
"""Run multi-seed lesion benchmarking and Welch's t-test statistical validation.

Validates the neurogenetic hypothesis that Mi4 null-direction veto ablation
causes statistically significant ($p < 0.01$ Welch's $t$-test) collapse of
directional optic flow asymmetry and combat targeting lock stability across
seeds 1001, 1002, 1003.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure src/ is on sys.path
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fly_doom.analysis.lesion_benchmark import (
    load_or_generate_lesion_data,
    run_lesion_statistical_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/doom004_lesions_multi_seed"),
        help="Directory to store multi-seed episode data and reports.",
    )
    parser.add_argument(
        "--empirical-dir",
        type=Path,
        default=Path("runs/doom004_lesions_v1"),
        help="Directory containing baseline empirical runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Random seed (repeatable; defaults to [1001, 1002, 1003]).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Statistical significance threshold (default: 0.01).",
    )
    args = parser.parse_args()

    seeds = args.seed if args.seed else [1001, 1002, 1003]
    output_dir = args.output_dir.resolve()
    empirical_dir = args.empirical_dir.resolve() if args.empirical_dir else None

    print(f"\n[Multi-Seed Lesion Benchmark] Running seeds {seeds}...")
    print(f"Output Directory: {output_dir}")
    print(f"Empirical Baseline Directory: {empirical_dir}\n")

    data = load_or_generate_lesion_data(
        output_dir=output_dir,
        seeds=seeds,
        empirical_dir=empirical_dir,
    )

    report = run_lesion_statistical_suite(data, alpha=args.alpha)
    print(report["summary_table"])

    # Save JSON report
    report_file = output_dir / "lesion_multi_seed_report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved statistical report to: {report_file}\n")


if __name__ == "__main__":
    main()
