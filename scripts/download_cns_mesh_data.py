"""Download authentic Drosophila melanogaster adult brain meshes from Virtual Fly Brain (VFB).

Acquires:
1. Whole Brain JFRC2 Template Mesh (JFRCtemplate2010_simple.obj)
2. Neuropil Domain Index (Original_Index.tsv)
3. Anatomical Neuropil Meshes:
   - Medulla (ME_R / ME_L)
   - Lobula Plate (LOP_R / LOP_L)
   - Lobula (LO_R / LO_L)
   - Ellipsoid Body (EB)
   - Fan-shaped Body (FB)
   - Protocerebral Bridge (PB)
   - Gnathal Ganglion / SEZ (GNG)
   - Antennal Lobes (AL_R / AL_L)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_RAW = "https://raw.githubusercontent.com/VirtualFlyBrain/DrosAdultBRAINdomains/master"

DOWNLOAD_TARGETS = [
    ("refData/Original_Index.tsv", "Original_Index.tsv"),
    ("template/JFRCtemplate2010_simple.obj", "JFRCtemplate2010_simple.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0025.obj", "neuropils/ME_R.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0071.obj", "neuropils/ME_L.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0022.obj", "neuropils/LOP_R.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0069.obj", "neuropils/LOP_L.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0003.obj", "neuropils/LO_R.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0053.obj", "neuropils/LO_L.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0023.obj", "neuropils/EB.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0026.obj", "neuropils/FB.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0006.obj", "neuropils/PB.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0049.obj", "neuropils/GNG_SEZ.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0024.obj", "neuropils/AL_R.obj"),
    ("individualDomainFiles/101/AdultBrainDomain0070.obj", "neuropils/AL_L.obj"),
]


def main():
    dest_dir = Path("data/brain_mesh")
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "neuropils").mkdir(parents=True, exist_ok=True)

    print(f"Downloading authentic Drosophila adult brain meshes to {dest_dir}...")
    for rel_url, local_name in DOWNLOAD_TARGETS:
        url = f"{BASE_RAW}/{rel_url}"
        target_path = dest_dir / local_name
        if target_path.exists() and target_path.stat().st_size > 100:
            print(f"  [cached] {local_name} ({target_path.stat().st_size / 1024:.1f} KB)")
            continue

        print(f"  [fetch] {local_name} from {url}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"  [done]  {local_name} saved ({target_path.stat().st_size / 1024:.1f} KB)")
        except Exception as exc:
            print(f"  [error] failed to download {local_name}: {exc}")


if __name__ == "__main__":
    main()
