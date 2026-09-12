"""Process authentic JFRC2 / Virtual Fly Brain OBJ meshes into normalized 3D visual geometries.

Parses:
- JFRCtemplate2010_simple.obj (authentic whole adult brain mesh: 2,224 vertices, 4,436 faces)
- Individual domain neuropils (EB, FB, PB, ME_R, LOP_R, LO_R, AL_R, GNG_SEZ)
- Mirrors lateral neuropils to generate left counterparts (ME_L, LOP_L, LO_L, AL_L)
- Aligns and centers coordinates to [0, 0, 0] with microns scale.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def parse_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()
                # 1-indexed vertex indices in OBJ
                face = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                faces.append(face)
    return np.array(verts, dtype=np.float64), faces


def main():
    mesh_dir = Path("data/brain_mesh")
    template_obj = mesh_dir / "JFRCtemplate2010_simple.obj"
    if not template_obj.exists():
        raise FileNotFoundError(f"Missing {template_obj}")

    print("Parsing authentic JFRC2 whole-brain template...")
    verts, faces = parse_obj(template_obj)
    print(f"Loaded {len(verts)} vertices and {len(faces)} triangular faces.")

    # In JFRC2 coordinates:
    # X spans ~20 to 620 (left to right, center ~320)
    # Y spans ~0 to 300 (dorsal to ventral)
    # Z spans ~0 to 125 (anterior to posterior)
    center_x = float(np.mean(verts[:, 0]))
    center_y = float(np.mean(verts[:, 1]))
    center_z = float(np.mean(verts[:, 2]))

    # Center to (0, 0, 0)
    # Standard anatomical display:
    # X: Left (-) to Right (+)
    # Y: Ventral (-) to Dorsal (+) -> invert JFRC2 Y so dorsal is positive!
    # Z: Anterior (-) to Posterior (+)
    norm_verts = np.zeros_like(verts)
    norm_verts[:, 0] = verts[:, 0] - center_x
    norm_verts[:, 1] = -(verts[:, 1] - center_y)  # Invert Y so top is dorsal
    norm_verts[:, 2] = verts[:, 2] - center_z

    # Extract distinct mesh wireframe edges
    edges_set = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i + 1) % 3]]))
            edges_set.add(e)

    mesh_edges = []
    for e1, e2 in edges_set:
        p1 = norm_verts[e1]
        p2 = norm_verts[e2]
        mesh_edges.append([
            round(float(p1[0]), 2), round(float(p1[1]), 2), round(float(p1[2]), 2),
            round(float(p2[0]), 2), round(float(p2[1]), 2), round(float(p2[2]), 2),
        ])

    print(f"Extracted {len(mesh_edges)} authentic brain surface mesh wireframe edges.")

    # Parse key neuropil compartments
    neuropil_specs = [
        ("EB", "Ellipsoid Body", "central_complex", mesh_dir / "neuropils/EB.obj", False, "#22c55e"),
        ("FB", "Fan-shaped Body", "central_complex", mesh_dir / "neuropils/FB.obj", False, "#eab308"),
        ("PB", "Protocerebral Bridge", "central_complex", mesh_dir / "neuropils/PB.obj", False, "#a855f7"),
        ("GNG_SEZ", "Gnathal Ganglion (SEZ)", "mechanosensory", mesh_dir / "neuropils/GNG_SEZ.obj", False, "#14b8a6"),
        ("AL_R", "Antennal Lobe (Right)", "mechanosensory", mesh_dir / "neuropils/AL_R.obj", True, "#06b6d4"),
        ("ME_R", "Medulla (Right)", "optic_lobe", mesh_dir / "neuropils/ME_R.obj", True, "#38bdf8"),
        ("LOP_R", "Lobula Plate (Right)", "optic_lobe", mesh_dir / "neuropils/LOP_R.obj", True, "#10b981"),
        ("LO_R", "Lobula (Right)", "optic_lobe", mesh_dir / "neuropils/LO_R.obj", True, "#6366f1"),
    ]

    processed_neuropils = []
    for code, name, system, path, is_lateral, color in neuropil_specs:
        if not path.exists():
            continue
        np_verts, np_faces = parse_obj(path)
        # Downsample high-res neuropils (like ME_R) to ~1,200 edges for 60fps rendering
        stride = max(1, len(np_faces) // 600)
        np_edges_set = set()
        for f in np_faces[::stride]:
            for i in range(3):
                e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                np_edges_set.add(e)

        # Normalize coordinates with same center
        np_norm = np.zeros_like(np_verts)
        np_norm[:, 0] = np_verts[:, 0] - center_x
        np_norm[:, 1] = -(np_verts[:, 1] - center_y)
        np_norm[:, 2] = np_verts[:, 2] - center_z

        edges_right = []
        for e1, e2 in np_edges_set:
            p1 = np_norm[e1]
            p2 = np_norm[e2]
            edges_right.append([
                round(float(p1[0]), 2), round(float(p1[1]), 2), round(float(p1[2]), 2),
                round(float(p2[0]), 2), round(float(p2[1]), 2), round(float(p2[2]), 2),
            ])

        processed_neuropils.append({
            "code": code,
            "name": name,
            "system": system,
            "color": color,
            "edges": edges_right,
        })

        # If lateral, mirror across X to generate left hemisphere counterpart!
        if is_lateral:
            left_code = code.replace("_R", "_L")
            left_name = name.replace("(Right)", "(Left)")
            edges_left = []
            for e in edges_right:
                edges_left.append([
                    -e[0], e[1], e[2],
                    -e[3], e[4], e[5],
                ])
            processed_neuropils.append({
                "code": left_code,
                "name": left_name,
                "system": system,
                "color": color,
                "edges": edges_left,
            })

    output_path = Path("web/authentic_fly_cns_mesh.json")
    payload = {
        "metadata": {
            "source": "Janelia JFRCtemplate2010 / Virtual Fly Brain (VFB) Official Meshes",
            "reference_standard": "JFRC2 / JRC2018",
            "total_template_edges": len(mesh_edges),
            "neuropil_count": len(processed_neuropils),
            "bounds_um": {
                "min_x": round(float(np.min(norm_verts[:, 0])), 1),
                "max_x": round(float(np.max(norm_verts[:, 0])), 1),
                "min_y": round(float(np.min(norm_verts[:, 1])), 1),
                "max_y": round(float(np.max(norm_verts[:, 1])), 1),
                "min_z": round(float(np.min(norm_verts[:, 2])), 1),
                "max_z": round(float(np.max(norm_verts[:, 2])), 1),
            },
        },
        "brain_surface_edges": mesh_edges,
        "neuropils": processed_neuropils,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved authentic brain mesh data to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
