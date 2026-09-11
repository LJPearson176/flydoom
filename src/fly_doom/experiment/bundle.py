"""Hermetic experiment bundle and provenance manifest generator.

Ensures every experiment run writes a self-contained, reproducible directory containing:
  - config.yaml: Exact execution parameters
  - manifest.json: Cryptographic graph hashes, git commit, runtime environment
  - provenance.json: Tier-classified components (biological, hypothesis, scaffold)
  - validation_gates.json: Explicit pass/fail status of all scientific and engineering gates
  - metrics.json: Quantitative observables (firing rate, synchrony, etc.)
  - report.md: Automated human-readable report with negative results as first-class citizens
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from fly_doom.connectome.manifest import ConnectomeFingerprint
from fly_doom.core.provenance import ProvenanceRegistry


def _get_git_provenance() -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        diff_stat = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {
            "commit": commit,
            "dirty": len(status) > 0,
            "status_porcelain": status.splitlines() if status else [],
            "diff_stat": diff_stat,
        }
    except Exception as e:
        return {
            "commit": "git_hash_unavailable",
            "dirty": True,
            "status_porcelain": [],
            "diff_stat": f"git_error: {e}",
        }


def _to_json_compatible(val: Any) -> Any:
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    return str(val)


@dataclass
class ValidationGateResult:
    """Pass/Fail outcome of a validation gate."""

    name: str
    passed: bool
    observed_value: Any
    threshold: Any
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "observed_value": _to_json_compatible(self.observed_value),
            "threshold": _to_json_compatible(self.threshold),
            "rationale": str(self.rationale),
        }


@dataclass
class ExperimentBundle:
    """Hermetic container for a single experimental execution."""

    run_id: str
    experiment_name: str
    config: Dict[str, Any]
    provenance_registry: ProvenanceRegistry
    dataset_fingerprint: Optional[ConnectomeFingerprint] = None
    validation_gates: List[ValidationGateResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    git_provenance: Dict[str, Any] = field(default_factory=_get_git_provenance)
    platform_info: Dict[str, str] = field(
        default_factory=lambda: {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
    )

    @property
    def git_hash(self) -> str:
        return self.git_provenance.get("commit", "git_hash_unavailable")

    def add_gate(
        self,
        name: str,
        passed: bool,
        observed_value: Any,
        threshold: Any,
        rationale: str,
    ) -> None:
        self.validation_gates.append(
            ValidationGateResult(
                name=name,
                passed=passed,
                observed_value=observed_value,
                threshold=threshold,
                rationale=rationale,
            )
        )

    def generate_markdown_report(self) -> str:
        all_passed = all(g.passed for g in self.validation_gates) if self.validation_gates else True
        status_banner = "PASSED" if all_passed else "FAILED GATES DETECTED"
        git_dirty_str = "DIRTY" if self.git_provenance.get("dirty") else "CLEAN"

        md = [
            f"# Experiment Run Report: `{self.run_id}`",
            f"**Experiment Name:** {self.experiment_name}  ",
            f"**Timestamp (UTC):** {self.timestamp}  ",
            f"**Git Commit:** `{self.git_hash}` ({git_dirty_str})  ",
            f"**Platform:** {self.platform_info['system']} {self.platform_info['machine']} (Python {self.platform_info['python_version']})  ",
            f"**Overall Validation Status:** **{status_banner}**  \n",
            "---",
            "## 1. Provenance Tier Audit",
            "Components registered in this experiment:\n",
        ]

        tier_counts = self.provenance_registry.audit_tier_counts()
        for tier, count in tier_counts.items():
            md.append(f"- **{tier}:** {count} component(s)")
        md.append("\n```json")
        md.append(json.dumps(self.provenance_registry.to_dict(), indent=2))
        md.append("```\n")

        md.extend([
            "---",
            "## 2. Validation Gates (First-Class Pass/Fail Registry)",
            "Gates evaluate biological, dynamical, or engineering assertions.\n",
        ])

        if not self.validation_gates:
            md.append("_No validation gates were registered for this run._\n")
        else:
            md.append("| Gate Name | Status | Observed Value | Threshold | Rationale |")
            md.append("| :--- | :--- | :--- | :--- | :--- |")
            for g in self.validation_gates:
                status_icon = "PASS" if g.passed else "**FAIL**"
                md.append(
                    f"| `{g.name}` | {status_icon} | `{g.observed_value}` | `{g.threshold}` | {g.rationale} |"
                )
            md.append("")

        md.extend([
            "---",
            "## 3. Quantitative Metrics",
            "```json",
            json.dumps(self.metrics, indent=2),
            "```\n",
        ])

        if self.dataset_fingerprint:
            md.extend([
                "---",
                "## 4. Connectome Dataset Fingerprint",
                f"- **Neuron Count:** {self.dataset_fingerprint.neuron_count:,}",
                f"- **Directed Edge Count:** {self.dataset_fingerprint.directed_edge_count:,}",
                f"- **Synaptic Contact Count:** {self.dataset_fingerprint.synapse_contact_count:,}",
                f"- **Combined Hash:** `{self.dataset_fingerprint.combined_hash}`",
                "",
            ])

        return "\n".join(md)

    def save(self, output_dir: Path) -> Path:
        """Persist all experiment artifacts to a timestamped directory."""
        run_dir = output_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. config.yaml
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=True)

        # 2. manifest.json
        manifest_data = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "git_hash": self.git_hash,
            "git_provenance": self.git_provenance,
            "platform": self.platform_info,
            "dataset_fingerprint": asdict(self.dataset_fingerprint)
            if self.dataset_fingerprint
            else None,
        }
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, sort_keys=True)

        # 3. provenance.json
        with open(run_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(self.provenance_registry.to_dict(), f, indent=2, sort_keys=True)

        # 4. validation_gates.json
        gates_data = [g.to_dict() for g in self.validation_gates]
        with open(run_dir / "validation_gates.json", "w", encoding="utf-8") as f:
            json.dump(gates_data, f, indent=2, sort_keys=True)

        # 5. metrics.json
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, sort_keys=True)

        # 6. report.md
        with open(run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report())

        # 7. COMPLETE (Cryptographic seal hashing all generated artifacts)
        checksums = {}
        for fname in sorted(["config.yaml", "manifest.json", "provenance.json", "validation_gates.json", "metrics.json", "report.md"]):
            fpath = run_dir / fname
            if fpath.exists():
                with open(fpath, "rb") as bf:
                    checksums[fname] = hashlib.sha256(bf.read()).hexdigest()

        with open(run_dir / "COMPLETE", "w", encoding="utf-8") as f:
            for fname, sha in checksums.items():
                f.write(f"{sha}  {fname}\n")

        return run_dir
