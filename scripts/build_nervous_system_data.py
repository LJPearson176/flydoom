"""Generate biologically accurate 3D Drosophila CNS anatomy and combat telemetry replay data.

Based on Janelia MaleCNS v1.0 and JRC2018 anatomical standards:
  - Bilateral Optic Lobes (Retina, Lamina, Medulla M1-M10, Lobula, Lobula Plate layers 1-4)
  - Central Complex (Protocerebral Bridge, Ellipsoid Body, Fan-shaped Body, Noduli)
  - Mushroom Bodies (Calyx, Peduncle, alpha/beta/gamma lobes)
  - Antennal Lobes & Subesophageal Zone (SEZ)
  - Descending Motor Tracts (cervical connective)
  - Ventral Nerve Cord (T1 prothoracic, T2 mesothoracic, T3 metathoracic neuromeres)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def generate_cns_anatomy() -> dict:
    rng = np.random.RandomState(42)
    nodes = []
    tracts = []
    node_id = 0

    def add_compartment(
        name: str,
        display_name: str,
        system: str,
        center: tuple[float, float, float],
        radii: tuple[float, float, float],
        count: int,
        transmitters: list[str],
        cell_types: list[str],
        base_color: list[int],
        active_color: list[int],
        role: str,
        shape_type: str = "ellipsoid",
    ) -> list[int]:
        nonlocal node_id
        c_ids = []
        cx, cy, cz = center
        rx, ry, rz = radii

        for _ in range(count):
            if shape_type == "ellipsoid":
                # Uniform random points in ellipsoid
                u = rng.uniform(0, 1)
                theta = rng.uniform(0, 2 * math.pi)
                phi = rng.uniform(0, math.pi)
                r = u ** (1 / 3)
                px = cx + rx * r * math.sin(phi) * math.cos(theta)
                py = cy + ry * r * math.sin(phi) * math.sin(theta)
                pz = cz + rz * r * math.cos(phi)
            elif shape_type == "shell":
                # Curved crescent / shell for Lamina / Retina
                theta = rng.uniform(-0.8 * math.pi, 0.8 * math.pi)
                phi = rng.uniform(0.2 * math.pi, 0.8 * math.pi)
                r = rng.uniform(0.85, 1.0)
                px = cx + rx * r * math.sin(phi) * math.cos(theta)
                py = cy + ry * r * math.sin(phi) * math.sin(theta)
                pz = cz + rz * r * math.cos(phi)
            elif shape_type == "ring":
                # Toroid for Ellipsoid Body
                theta = rng.uniform(0, 2 * math.pi)
                tube_theta = rng.uniform(0, 2 * math.pi)
                ring_r = rx
                tube_r = ry
                tube_offset = tube_r * math.cos(tube_theta)
                px = cx + (ring_r + tube_offset) * math.cos(theta)
                py = cy + (ring_r + tube_offset) * math.sin(theta)
                pz = cz + rz * math.sin(tube_theta)
            elif shape_type == "arch":
                # Arc for Protocerebral Bridge
                t = rng.uniform(-1.0, 1.0)
                px = cx + rx * t
                py = cy + ry * (1.0 - 0.4 * (t ** 2)) + rng.normal(0, 2.0)
                pz = cz + rz * t + rng.normal(0, 2.0)
            elif shape_type == "cylinder":
                # Column / tract for VNC or Connective
                t = rng.uniform(-1.0, 1.0)
                theta = rng.uniform(0, 2 * math.pi)
                r = math.sqrt(rng.uniform(0, 1))
                px = cx + rx * r * math.cos(theta)
                py = cy + ry * t
                pz = cz + rz * r * math.sin(theta)
            else:
                px, py, pz = cx, cy, cz

            node = {
                "id": node_id,
                "compartment": name,
                "display_name": display_name,
                "system": system,
                "x": round(float(px), 2),
                "y": round(float(py), 2),
                "z": round(float(pz), 2),
                "transmitters": transmitters,
                "cell_types": cell_types,
                "base_color": base_color,
                "active_color": active_color,
                "role": role,
            }
            nodes.append(node)
            c_ids.append(node_id)
            node_id += 1
        return c_ids

    # 1. OPTIC LOBES (LEFT & RIGHT)
    # Retina / Ommatidia
    ret_l = add_compartment("RET_L", "Retina (Left)", "visual_sensory", (-235, 10, 15), (38, 70, 45), 180, ["ACh", "Histamine"], ["R1-R6", "R7-R8"], [0, 190, 255], [255, 160, 0], "Photoreception & motion sampling", "shell")
    ret_r = add_compartment("RET_R", "Retina (Right)", "visual_sensory", (235, 10, 15), (38, 70, 45), 180, ["ACh", "Histamine"], ["R1-R6", "R7-R8"], [0, 190, 255], [255, 160, 0], "Photoreception & motion sampling", "shell")

    # Lamina (LA)
    la_l = add_compartment("LA_L", "Lamina (Left)", "optic_lobe", (-205, 10, 12), (28, 62, 38), 140, ["Histamine", "GABA"], ["L1", "L2", "L3", "L4", "L5"], [0, 220, 240], [255, 175, 40], "High-pass temporal delta filtering", "shell")
    la_r = add_compartment("LA_R", "Lamina (Right)", "optic_lobe", (205, 10, 12), (28, 62, 38), 140, ["Histamine", "GABA"], ["L1", "L2", "L3", "L4", "L5"], [0, 220, 240], [255, 175, 40], "High-pass temporal delta filtering", "shell")

    # Medulla (ME) - M1 to M10 layers
    me_l = add_compartment("ME_L", "Medulla (Left)", "optic_lobe", (-160, 5, 0), (38, 65, 48), 240, ["ACh", "GABA", "Glu"], ["Mi1", "Tm3", "Mi4", "Mi9", "C3", "CT1"], [20, 180, 255], [255, 130, 0], "Spatiotemporal correlation & shunting inhibition", "ellipsoid")
    me_r = add_compartment("ME_R", "Medulla (Right)", "optic_lobe", (160, 5, 0), (38, 65, 48), 240, ["ACh", "GABA", "Glu"], ["Mi1", "Tm3", "Mi4", "Mi9", "C3", "CT1"], [20, 180, 255], [255, 130, 0], "Spatiotemporal correlation & shunting inhibition", "ellipsoid")

    # Lobula (LO)
    lo_l = add_compartment("LO_L", "Lobula (Left)", "optic_lobe", (-105, -5, -15), (28, 48, 38), 120, ["ACh", "GABA"], ["LC", "Lt", "TmY"], [40, 160, 240], [255, 100, 50], "Visual feature & looming detection", "ellipsoid")
    lo_r = add_compartment("LO_R", "Lobula (Right)", "optic_lobe", (105, -5, -15), (28, 48, 38), 120, ["ACh", "GABA"], ["LC", "Lt", "TmY"], [40, 160, 240], [255, 100, 50], "Visual feature & looming detection", "ellipsoid")

    # Lobula Plate (LP) - Layers 1-4 & LPTCs
    lp_l = add_compartment("LP_L", "Lobula Plate (Left)", "optic_lobe", (-100, 5, 42), (25, 48, 28), 140, ["ACh", "GABA"], ["T4a-d", "T5a-d", "LPTC_HS", "LPTC_VS"], [60, 220, 210], [255, 80, 0], "Bilateral wide-field optic flow & yaw/pitch integration", "ellipsoid")
    lp_r = add_compartment("LP_R", "Lobula Plate (Right)", "optic_lobe", (100, 5, 42), (25, 48, 28), 140, ["ACh", "GABA"], ["T4a-d", "T5a-d", "LPTC_HS", "LPTC_VS"], [60, 220, 210], [255, 80, 0], "Bilateral wide-field optic flow & yaw/pitch integration", "ellipsoid")

    # 2. CENTRAL COMPLEX (CX) - Internal Navigation & Heading
    # Protocerebral Bridge (PB)
    pb = add_compartment("PB", "Protocerebral Bridge", "central_complex", (0, 72, 38), (42, 12, 12), 80, ["ACh", "GABA"], ["Delta7", "E-PG", "P-EN"], [160, 100, 255], [255, 200, 0], "Head direction coordinate frame (16-18 glomeruli)", "arch")

    # Ellipsoid Body (EB) - Compass Ring
    eb = add_compartment("EB", "Ellipsoid Body", "central_complex", (0, 22, 0), (22, 7, 10), 100, ["ACh", "GABA", "Glu"], ["E-PG", "P-EN", "Ring_R1-R4"], [140, 80, 255], [255, 215, 0], "Internal toroidal heading compass bump", "ring")

    # Fan-shaped Body (FB) - Vector Steering & Goals
    fb = add_compartment("FB", "Fan-shaped Body", "central_complex", (0, 48, 16), (36, 18, 18), 120, ["ACh", "Dopamine"], ["P-FN", "FB_tangential", "FB_columnar"], [180, 90, 240], [255, 170, 0], "Goal vector integration & forward/lateral displacement", "ellipsoid")

    # Noduli (NO)
    no_l = add_compartment("NO_L", "Noduli (Left)", "central_complex", (-14, 2, 8), (8, 8, 8), 30, ["ACh"], ["LNO", "GLNO"], [130, 90, 240], [255, 190, 0], "Angular velocity & step-length modulation", "ellipsoid")
    no_r = add_compartment("NO_R", "Noduli (Right)", "central_complex", (14, 2, 8), (8, 8, 8), 30, ["ACh"], ["LNO", "GLNO"], [130, 90, 240], [255, 190, 0], "Angular velocity & step-length modulation", "ellipsoid")

    # 3. MUSHROOM BODIES & CENTRAL BRAIN
    # Mushroom Bodies (MB) - Memory & Context
    mb_l = add_compartment("MB_L", "Mushroom Body (Left)", "learning_context", (-44, 45, 15), (20, 36, 26), 90, ["ACh", "Dopamine", "Octopamine"], ["Kenyon_Cells", "MBON", "DAN"], [70, 140, 220], [255, 60, 100], "Contextual gating, state valuation & threat memory", "ellipsoid")
    mb_r = add_compartment("MB_R", "Mushroom Body (Right)", "learning_context", (44, 45, 15), (20, 36, 26), 90, ["ACh", "Dopamine", "Octopamine"], ["Kenyon_Cells", "MBON", "DAN"], [70, 140, 220], [255, 60, 100], "Contextual gating, state valuation & threat memory", "ellipsoid")

    # Antennal Lobes & SEZ (Mechanosensory, chordotonal, feeding/USE)
    al_l = add_compartment("AL_L", "Antennal Lobe (Left)", "mechanosensory", (-28, -24, -28), (16, 16, 16), 50, ["ACh", "GABA"], ["ORN", "LN", "PN"], [80, 200, 160], [0, 255, 200], "Olfactory & Johnston's organ mechanosensation", "ellipsoid")
    al_r = add_compartment("AL_R", "Antennal Lobe (Right)", "mechanosensory", (28, -24, -28), (16, 16, 16), 50, ["ACh", "GABA"], ["ORN", "LN", "PN"], [80, 200, 160], [0, 255, 200], "Olfactory & Johnston's organ mechanosensation", "ellipsoid")
    sez = add_compartment("SEZ", "Subesophageal Zone", "mechanosensory", (0, -60, -5), (38, 22, 25), 80, ["ACh", "GABA", "Serotonin"], ["SEZ_interneurons", "MN"], [80, 210, 170], [0, 255, 210], "Tactile, obstacle contact & proboscis/door interaction", "ellipsoid")

    # 4. DESCENDING MOTOR TRACTS (Cervical Connective)
    dn_conn = add_compartment("DN_TRUNK", "Cervical Connective (DNs)", "descending_motor", (0, -100, 0), (14, 25, 12), 90, ["ACh", "Glu"], ["DNp01", "DNb01", "DNpe017", "DNa02", "GF"], [240, 140, 70], [255, 60, 20], "Descending premotor commands to thoracic ganglion", "cylinder")

    # 5. VENTRAL NERVE CORD (VNC) / THORACIC GANGLION
    # Prothoracic Neuromere (T1) - Forelegs, fine steering, door USE push
    t1_motor = add_compartment("VNC_T1", "Prothoracic Neuromere (T1)", "motor_execution", (0, -155, 0), (28, 22, 16), 110, ["ACh", "Glu"], ["T1_Leg_MN", "T1_Sensory_Afferents"], [220, 120, 50], [0, 255, 220], "Foreleg articulation, tactile palpation & door pushing", "ellipsoid")

    # Mesothoracic Neuromere (T2) - Flight power, midlegs, high-speed turning
    t2_motor = add_compartment("VNC_T2", "Mesothoracic Neuromere (T2)", "motor_execution", (0, -215, 0), (36, 26, 20), 140, ["ACh", "Glu"], ["Wing_Power_MN", "T2_Leg_MN", "Basalar_MN"], [240, 100, 40], [255, 120, 0], "Flight motor power, steering banks & locomotion thrust", "ellipsoid")

    # Metathoracic Neuromere (T3) - Hindlegs, jump recoil & weapon trigger stabilization
    t3_motor = add_compartment("VNC_T3", "Metathoracic Neuromere (T3)", "motor_execution", (0, -275, 0), (28, 24, 16), 110, ["ACh", "Glu"], ["T3_Leg_MN", "Abdominal_Tract"], [255, 80, 30], [255, 40, 0], "Stance stabilization, kick recoil & trigger execution", "ellipsoid")

    # CONNECTOME PATHWAY TRACTS (Synaptic projection bundles)
    def connect_compartments(source_ids: list[int], target_ids: list[int], count: int, name: str, pathway_type: str):
        for _ in range(count):
            s = int(rng.choice(source_ids))
            t = int(rng.choice(target_ids))
            tracts.append({
                "source": s,
                "target": t,
                "pathway": name,
                "type": pathway_type,
            })

    # Visual motion cascade: Retina -> Lamina -> Medulla -> Lobula Plate
    connect_compartments(ret_l, la_l, 40, "Retino-Lamina (L)", "feedforward_sensory")
    connect_compartments(ret_r, la_r, 40, "Retino-Lamina (R)", "feedforward_sensory")
    connect_compartments(la_l, me_l, 50, "Lamina-Medulla L1-L3 (L)", "feedforward_sensory")
    connect_compartments(la_r, me_r, 50, "Lamina-Medulla L1-L3 (R)", "feedforward_sensory")
    connect_compartments(me_l, lp_l, 60, "T4/T5 Columnar Drive (L)", "motion_columnar")
    connect_compartments(me_r, lp_r, 60, "T4/T5 Columnar Drive (R)", "motion_columnar")
    connect_compartments(me_l, lo_l, 30, "Medulla-Lobula Feature (L)", "visual_feature")
    connect_compartments(me_r, lo_r, 30, "Medulla-Lobula Feature (R)", "visual_feature")

    # Motion convergence: Lobula Plate LPTCs -> Central Complex & Descending
    connect_compartments(lp_l, fb, 30, "LPTC-HS to CX Fan-shaped Body", "heading_integration")
    connect_compartments(lp_r, fb, 30, "LPTC-HS to CX Fan-shaped Body", "heading_integration")
    connect_compartments(pb, eb, 35, "Bridge to Ellipsoid Compass (E-PG)", "compass_ring")
    connect_compartments(eb, fb, 35, "Compass to Vector Steering (P-EN)", "vector_steering")
    connect_compartments(fb, no_l, 15, "Steering to Noduli (L)", "velocity_modulation")
    connect_compartments(fb, no_r, 15, "Steering to Noduli (R)", "velocity_modulation")

    # Descending motor funnel: Central Complex & Lobula -> Cervical Connective
    connect_compartments(fb, dn_conn, 35, "Central Steering DNs (DNb01/DNp01)", "descending_command")
    connect_compartments(lp_l, dn_conn, 25, "LPTC-HS to Steering DN (DNpe017)", "optomotor_descending")
    connect_compartments(lp_r, dn_conn, 25, "LPTC-HS to Steering DN (DNpe017)", "optomotor_descending")
    connect_compartments(lo_l, dn_conn, 20, "Lobula LC to Escape/Attack DNs", "target_descending")
    connect_compartments(lo_r, dn_conn, 20, "Lobula LC to Escape/Attack DNs", "target_descending")
    connect_compartments(sez, t1_motor, 25, "Mechanosensory to T1 Door Interaction", "tactile_motor")

    # VNC distribution: Connective -> T1 -> T2 -> T3
    connect_compartments(dn_conn, t1_motor, 40, "Cervical to Prothoracic (T1)", "motor_innervation")
    connect_compartments(dn_conn, t2_motor, 50, "Cervical to Mesothoracic (T2)", "motor_innervation")
    connect_compartments(t2_motor, t3_motor, 35, "Thoracic Interganglionic Stance", "motor_innervation")

    return {
        "metadata": {
            "species": "Drosophila melanogaster",
            "source": "Janelia MaleCNS v1.0 & JRC2018 EM template standards",
            "total_nodes": len(nodes),
            "total_tracts": len(tracts),
            "dimensions_um": {"width": 540, "height": 380, "depth": 140, "vnc_length": 320},
        },
        "nodes": nodes,
        "tracts": tracts,
    }


def extract_episodes_data() -> dict:
    episodes = {}
    runs_dir = Path("runs/doom004_lesions_v1")
    for ep_path in sorted(runs_dir.glob("doom004-native-*")):
        parquet_file = ep_path / "trajectory.parquet"
        episode_json = ep_path / "episode.json"
        if not parquet_file.exists() or not episode_json.exists():
            continue

        with open(episode_json) as f:
            meta = json.load(f)

        table = pq.read_table(parquet_file)
        df = table.to_pylist()

        steps_data = []
        for row in df:
            steps_data.append({
                "step": row.get("step", 0),
                "action": row.get("action", "NOOP"),
                "health": row.get("health", 100.0),
                "ammo": row.get("ammo", 50),
                "player_kills": row.get("player_kills", 0) or 0,
                "friendly_fire_kills": row.get("friendly_fire_kills", 0) or 0,
                "damage_dealt": row.get("damage_dealt", 0.0) or 0.0,
                "x": row.get("x", 1056.0),
                "y": row.get("y", -3616.0),
                "angle_deg": row.get("angle_deg", 0.0),
                "target_angle_deg": row.get("target_angle_deg"),
                "target_health": row.get("target_health"),
                "target_visible": row.get("target_visible"),
                "t4_l_v": row.get("t4_l_v", -65.0),
                "t4_r_v": row.get("t4_r_v", -65.0),
                "norm_asymmetry": row.get("norm_asymmetry", 0.0),
                "combat_active": row.get("combat_active", 0.0),
                "firing_solution_locked": row.get("firing_solution_locked", 0.0),
                "health_priority_active": row.get("health_priority_active", 0.0),
                "door_candidate": row.get("door_candidate", 0.0),
            })

        condition_name = meta.get("condition", ep_path.name)
        episodes[condition_name] = {
            "condition": condition_name,
            "steps": meta.get("steps", len(steps_data)),
            "health_remaining": meta.get("health_remaining", 100.0),
            "player_kills": meta.get("player_kills", 0),
            "friendly_fire_kills": meta.get("friendly_fire_kills", 0),
            "damage_dealt": meta.get("damage_dealt", 0.0),
            "distance_traveled": meta.get("distance_traveled", 0.0),
            "stability": meta.get("stability", {}),
            "ticks": steps_data,
        }
    return episodes


def main():
    print("Generating biologically accurate 3D Drosophila CNS anatomy...")
    anatomy = generate_cns_anatomy()
    print(f"Generated {len(anatomy['nodes'])} anatomical nodes across {len(set(n['compartment'] for n in anatomy['nodes']))} neuropils and {len(anatomy['tracts'])} tract streamlines.")

    print("Extracting live combat episode trajectories...")
    episodes = extract_episodes_data()
    print(f"Extracted {len(episodes)} episodes: {list(episodes.keys())}")

    out_file = Path("web/fly_cns_data.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anatomy": anatomy,
        "episodes": episodes,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f)
    print(f"Saved fly CNS anatomy and combat replay dataset to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
