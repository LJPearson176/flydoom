"""Build and serialize high-fidelity Drosophila neural pathways and synapse cloud.

Exports:
1. web/fly_high_fidelity_pathways.json (dedicated connectome bundle)
2. Updates web/fly_3d_cns_model.json with embedded high_fidelity_pathways
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fly_doom.connectome.high_fidelity_pathways import build_high_fidelity_pathway_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-fidelity Drosophila pathways")
    parser.add_argument("--verify", action="store_true", help="verify coordinate bounds and schemas")
    args = parser.parse_args()

    print("Building high-fidelity Drosophila pathway bundle from MaleCNS & FlyWire data...")
    bundle = build_high_fidelity_pathway_bundle()
    bundle_dict = bundle.to_dict()

    out_json = Path("web/fly_high_fidelity_pathways.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(bundle_dict, f, separators=(",", ":"))
    print(f"Saved {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")

    # Embed into web/fly_3d_cns_model.json so both viewers have instant access
    model_json = Path("web/fly_3d_cns_model.json")
    if model_json.exists():
        with open(model_json, "r") as f:
            model = json.load(f)
        model["high_fidelity_pathways"] = bundle_dict
        with open(model_json, "w") as f:
            json.dump(model, f, separators=(",", ":"))
        print(f"Embedded high-fidelity pathways into {model_json} ({model_json.stat().st_size / 1024:.1f} KB)")

    # Print summary statistics
    meta = bundle_dict["metadata"]
    print("\n" + "=" * 70)
    print("HIGH-FIDELITY NEURAL PATHWAY CONNECTOME SUMMARY")
    print("=" * 70)
    print(f"Total Morphological Neurons : {meta['total_neurons']}")
    print(f"Total Chemical Synapses     : {meta['total_synapses']}")
    print(f"Total Retinotopic Cartridges: {meta['total_cartridges']}")
    print(f"Synapses by Neurotransmitter:")
    for tx, count in meta["synapse_counts_by_transmitter"].items():
        print(f"  - {tx:15s}: {count:4d} contacts")
    print(f"Dataset & Provenance        : {meta['dataset_name']} ({meta['provenance']['doi']})")
    print("=" * 70 + "\n")

    if args.verify:
        print("Running bundle verification checks...")
        assert len(bundle_dict["neurons"]) == 14, f"Expected 14 canonical neurons, got {len(bundle_dict['neurons'])}"
        assert len(bundle_dict["synapses"]) >= 250, f"Expected >= 250 synapses, got {len(bundle_dict['synapses'])}"
        assert len(bundle_dict["cartridges"]) == 128, "Expected 128 cartridges (64 per eye)"
        # Check all node parent references
        for n in bundle_dict["neurons"]:
            node_ids = {node[0] for node in n["nodes"]}
            for node in n["nodes"]:
                pid = node[6]
                if pid != -1:
                    assert pid in node_ids, f"Neuron {n['instance']} has invalid parent {pid}"
        print("All verification checks PASSED successfully.")


if __name__ == "__main__":
    main()
