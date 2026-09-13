"""Record ACTUAL native GZDoom gameplay footage and composite with 3D isometric fly nervous system.

Executes native GZDoom on macOS, captures actual full-resolution RGB window frames,
reads real-time neural state and ZScript telemetry, and generates a side-by-side
composite video and animated GIF.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fly_doom.control.controllers import ControlledT4Controller, DoorSeekingController
from fly_doom.doom.gzdoom_target import GZDoomTarget
from fly_doom.doom.gzdoom_telemetry import GZDoomTelemetryReader
from fly_doom.doom.macos_gzdoom_bridge import MacOSGZDoomBridge
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
    return (0, 229, 255)


def render_3d_nervous_system(
    neural_state: dict,
    model_3d: dict,
    mesh_data: dict,
    width: int = 640,
    height: int = 540,
    pulse_phase: float = 0.0,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (5, 10, 14))
    draw = ImageDraw.Draw(img)

    asym = float(neural_state.get("norm_asymmetry", 0.0) or 0.0)
    action = neural_state.get("action", "FORWARD")
    is_fire = (action == "FIRE")
    is_door = (action == "USE" or neural_state.get("door_candidate"))
    is_damage = bool(neural_state.get("health_priority_active"))
    t4_l = float(neural_state.get("t4_l_v", -65.0) or -65.0)
    t4_r = float(neural_state.get("t4_r_v", -65.0) or -65.0)
    locked = bool(neural_state.get("firing_solution_locked"))

    # 3D Frontal-Isometric Projection matching the user's reference image
    yaw = 0.22
    pitch = 0.20
    dist = 680.0
    target = [0.0, -70.0, 0.0]
    focal = 600.0
    cx = width // 2
    cy = height // 2 + 10

    def project(x: float, y: float, z: float):
        tx = x - target[0]
        ty = y - target[1]
        tz = z - target[2]
        cosY, sinY = math.cos(yaw), math.sin(yaw)
        x1 = tx * cosY - tz * sinY
        z1 = tx * sinY + tz * cosY
        cosP, sinP = math.cos(pitch), math.sin(pitch)
        y2 = ty * cosP - z1 * sinP
        z2 = ty * sinP + z1 * cosP
        eyeZ = z2 + dist
        if eyeZ <= 10.0:
            return None
        scale = focal / eyeZ
        return (cx + x1 * scale, cy - y2 * scale, scale)

    # 1. 3D Cuticle Exoskeleton Wireframe
    for l in model_3d["exoskeleton"]["lines"]:
        p1 = project(l[0], l[1], l[2])
        p2 = project(l[3], l[4], l[5])
        if not p1 or not p2:
            continue
        ltype = l[6]
        col = (0, 75, 105)
        if ltype == "eye":
            col = (70, 95, 160)
        elif ltype in {"wing", "wing_vein"}:
            col = (40, 100, 140)
        elif ltype == "leg":
            col = (0, 90, 120)
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=col, width=1)

    # 2. Official JFRC2 Brain Surface Mesh (6,654 edges sampled)
    b_edges = mesh_data.get("brain_surface_edges", [])
    for i in range(0, len(b_edges), 3):
        e = b_edges[i]
        p1 = project(e[0], e[1], e[2])
        p2 = project(e[3], e[4], e[5])
        if p1 and p2:
            draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(15, 50, 80), width=1)

    # 3. Dense FlyWire Connectome Fibers (3,030 fibers)
    fibers = model_3d["connectome"]["fibers"]
    for i, f in enumerate(fibers):
        pts = f["points"]
        neuropil = f.get("neuropil", "")
        pts_proj = [project(p[0], p[1], p[2]) for p in pts]
        valid_pts = [(p[0], p[1]) for p in pts_proj if p is not None]
        if len(valid_pts) < 2:
            continue

        base_col = hex_to_rgb(f["color"])
        stroke_col = base_col
        lw = 1

        # Continuous graded biophysical illumination:
        is_right = neuropil.endswith("_R") or pts[0][0] > 0
        is_left = neuropil.endswith("_L") or pts[0][0] < 0
        drive = max(0.0, asym) if is_right else (max(0.0, -asym) if is_left else 0.0)

        if is_fire and (neuropil.startswith("LO_") or neuropil.startswith("VNC_") or neuropil == "DN_TRUNK"):
            stroke_col = (255, 50, 0)
            lw = 2
        elif is_door and (neuropil == "SEZ" or neuropil == "VNC_T1"):
            stroke_col = (0, 255, 170)
            lw = 2
        elif is_damage and neuropil.startswith("MB_"):
            stroke_col = (255, 23, 68)
            lw = 2
        elif drive > 0.02:
            # Continuous smooth color blend from resting base to active amber:
            blend = min(1.0, (drive - 0.02) / 0.35)
            r = int(base_col[0] * (1.0 - blend) + 255 * blend)
            g = int(base_col[1] * (1.0 - blend) + 160 * blend)
            b = int(base_col[2] * (1.0 - blend) + 0 * blend)
            stroke_col = (r, g, b)
            lw = 2 if blend > 0.3 else 1

        draw.line(valid_pts, fill=stroke_col, width=lw)

        # Action Potential Traveling Pulses
        if i % 8 == 0 and len(pts_proj) >= 2:
            t = (pulse_phase + i * 0.06) % 1.0
            idx = min(len(pts_proj) - 2, int(t * (len(pts_proj) - 1)))
            pA = pts_proj[idx]
            pB = pts_proj[idx + 1]
            if pA and pB:
                frac = (t * (len(pts_proj) - 1)) - idx
                px_pulse = pA[0] + (pB[0] - pA[0]) * frac
                py_pulse = pA[1] + (pB[1] - pA[1]) * frac
                pulse_col = (255, 240, 100) if lw == 2 else (0, 240, 255)
                draw.ellipse([px_pulse - 1.5, py_pulse - 1.5, px_pulse + 1.5, py_pulse + 1.5], fill=pulse_col)

    # 4. Esophageal Foramen Ring
    top_f = project(0, -15, 0)
    bot_f = project(0, -65, -10)
    if top_f and bot_f:
        draw.ellipse([top_f[0] - 8, top_f[1] - 4, top_f[0] + 8, top_f[1] + 4], outline=(0, 229, 255), width=1)
        draw.ellipse([bot_f[0] - 6, bot_f[1] - 3, bot_f[0] + 6, bot_f[1] + 3], outline=(0, 229, 255), width=1)

    # 5. Visual Retinotopic Cartridges (128 columns)
    hfp = model_3d.get("high_fidelity_pathways", {})
    cartridges = hfp.get("cartridges", [])
    for cart in cartridges:
        p1 = project(cart["ommatidium"][0], cart["ommatidium"][1], cart["ommatidium"][2])
        p2 = project(cart["medulla"][0], cart["medulla"][1], cart["medulla"][2])
        p3 = project(cart["lobula_plate"][0], cart["lobula_plate"][1], cart["lobula_plate"][2])
        cart_drive = max(0.0, asym) if cart["eye"] == "R" else max(0.0, -asym)
        if cart_drive > 0.02:
            c_blend = min(1.0, (cart_drive - 0.02) / 0.35)
            cr = int(20 * (1.0 - c_blend) + 245 * c_blend)
            cg = int(55 * (1.0 - c_blend) + 158 * c_blend)
            cb = int(70 * (1.0 - c_blend) + 11 * c_blend)
            ccol = (cr, cg, cb)
            clw = 2 if c_blend > 0.3 else 1
        else:
            ccol = (20, 55, 70)
            clw = 1

        if p1 and p2:
            draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=ccol, width=clw)
        if p2 and p3:
            draw.line([(p2[0], p2[1]), (p3[0], p3[1])], fill=ccol, width=clw)

    # 6. SWC Morphology Skeletons (7 Canonical EM Neurons)
    neurons = hfp.get("neurons", [])
    for neuron in neurons:
        ncol = hex_to_rgb(neuron.get("color_hex", "#00e5ff"))
        raw_nodes = neuron.get("nodes", [])
        nodes = {n[0]: n for n in raw_nodes}
        for n in raw_nodes:
            parent_id = n[6]
            if parent_id != -1 and parent_id in nodes:
                parent = nodes[parent_id]
                p1 = project(n[2], n[3], n[4])
                p2 = project(parent[2], parent[3], parent[4])
                if p1 and p2:
                    draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=ncol, width=2)
            if len(n) > 7 and n[7] == "soma":
                sp = project(n[2], n[3], n[4])
                if sp:
                    draw.ellipse([sp[0] - 4, sp[1] - 4, sp[0] + 4, sp[1] + 4], fill=ncol, outline=(255, 255, 255))

    # 7. Chemical Synapse Point Cloud (178 EM Active Zones)
    synapses = hfp.get("synapses", [])
    for syn in synapses:
        sp = project(syn["pos"][0], syn["pos"][1], syn["pos"][2])
        if sp:
            scol = hex_to_rgb(syn.get("color", "#00e5ff"))
            draw.ellipse([sp[0] - 1.5, sp[1] - 1.5, sp[0] + 1.5, sp[1] + 1.5], fill=scol)


    # Header in 3D panel
    draw.rectangle([0, 0, width, 38], fill=(8, 18, 24))
    draw.text((16, 11), "ISOMETRIC 3D FLY NERVOUS SYSTEM", fill=(0, 229, 255))
    draw.text((320, 11), "JFRC2 TEMPLATE · 7 SWC SKELETONS · 178 SYNAPSES", fill=(88, 223, 194))

    # Bottom Telemetry Cards in 3D panel (Detailed Activation Mapping)
    draw.rectangle([12, height - 104, width - 12, height - 6], fill=(9, 20, 27), outline=(22, 50, 62), width=1)

    div = float(neural_state.get("flow_divergence", 0.0) or 0.0)
    curl = float(neural_state.get("flow_curl", 0.0) or 0.0)
    eb_head = float(neural_state.get("eb_heading_deg", 0.0) or 0.0)
    eb_coh = float(neural_state.get("eb_bump_coherence", 0.0) or 0.0)

    draw.text((20, height - 98), f"ASYMMETRY: {asym:+.3f} (Div {div:+.2f} Curl {curl:+.2f})", fill=(240, 166, 90))
    draw.text((310, height - 98), f"EB COMPASS: {eb_head:+.1f}° (R={eb_coh:.2f})", fill=(0, 255, 170) if eb_coh > 0.5 else (180, 190, 195))
    draw.text((505, height - 98), f"LOCK: {'LOCKED' if locked else 'SCAN'}", fill=(0, 255, 170) if locked else (180, 190, 195))

    l_lead = float(neural_state.get("t4_l_leading_activation", 0.0) or 0.0)
    l_cent = float(neural_state.get("t4_l_central_activation", 0.0) or 0.0)
    l_trail = float(neural_state.get("t4_l_trailing_activation", 0.0) or 0.0)
    r_lead = float(neural_state.get("t4_r_leading_activation", 0.0) or 0.0)
    r_cent = float(neural_state.get("t4_r_central_activation", 0.0) or 0.0)
    r_trail = float(neural_state.get("t4_r_trailing_activation", 0.0) or 0.0)
    draw.text((20, height - 78), f"T4 ARBORS: L [Lead {l_lead:.2f}  Cent {l_cent:.2f}  Trail {l_trail:.2f}]   R [Lead {r_lead:.2f}  Cent {r_cent:.2f}  Trail {r_trail:.2f}]", fill=(170, 210, 220))

    circ_desc = "BILATERAL MOTION EQUILIBRIUM"
    if is_fire:
        circ_desc = "LOBULA LC COLUMNAR -> DNpe017 -> T3 TRIGGER BURST"
    elif is_door:
        circ_desc = "SEZ MECHANOSENSORY TOUCH -> T1 FORELEG DOOR USE"
    elif asym > 0.04:
        circ_desc = f"RIGHT MEDULLA M1-M10 -> LOP T4a/T5a -> RIGHT YAW (+{asym:.3f})"
    elif asym < -0.04:
        circ_desc = f"LEFT MEDULLA M1-M10 -> LOP T4a/T5a -> LEFT YAW ({asym:.3f})"

    mb_val = float(neural_state.get("mb_valence", 0.0) or 0.0)
    mb_ppl1 = float(neural_state.get("mb_ppl1_da", 0.0) or 0.0)
    fb_torque = float(neural_state.get("fb_steer_torque", 0.0) or 0.0)
    sez_acid = float(neural_state.get("sez_acid_detected", 0.0) or 0.0)
    draw.text((20, height - 38), f"MOTOR: {action} · FB STEER: {fb_torque:+.2f} · SEZ HAZARD: {'ACID' if sez_acid > 0 else 'CLEAR'} · MB VALENCE: {mb_val:+.2f}", fill=(120, 150, 160))

    return img


def run_actual_gameplay_recording(
    steps: int = 180,
    output_mp4: Path = Path("runs/doom004_actual_gameplay_3d_twin.mp4"),
    output_gif: Path = Path("runs/doom004_actual_gameplay_3d_twin.gif"),
    fps: int = 12,
) -> None:
    wad_path = Path("runs/doom004_lesions_v1/DOOM1_INSTRUMENTED.WAD")
    if not wad_path.exists():
        wad_path = Path("wads/DOOM1.WAD")

    log_path = Path("runs/doom004_actual_gameplay_3d_twin.log")
    target = GZDoomTarget(map_name="E1M1", iwad=wad_path, telemetry_logfile=log_path)
    bridge = MacOSGZDoomBridge(target=target, resolution=(64, 64))

    controller = DoorSeekingController(ControlledT4Controller(
        model_type=CompartmentModelType.MODEL_D,
        saccade_refractory_ticks=3,
    ))

    with open("web/fly_3d_cns_model.json") as f:
        model_3d = json.load(f)
    with open("web/authentic_fly_cns_mesh.json") as f:
        mesh_data = json.load(f)

    tmp_dir = Path("runs/actual_frames_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    recorded_ticks = []
    print(f"Launching native GZDoom to record {steps} ACTUAL gameplay frames with full combat...")
    try:
        win = bridge.launch()
        print(f"Connected to GZDoom window: {win.width}x{win.height}, pid={win.pid}")
        obs = bridge.reset(seed=1001)

        for step in range(steps):
            action = controller.select_action(obs)
            neural = controller.get_neural_state()

            obs, _, done, _ = bridge.step(action)
            real_rgb = bridge.last_full_rgb if bridge.last_full_rgb is not None else bridge.capture_rgb()

            native_state = obs.info.get("native_game_state") or {}

            # Combine neural state with native telemetry
            state_combined = {
                "step": step,
                "action": action.name,
                "health": float(obs.health),
                "ammo": int(obs.ammo),
                "kills": int(obs.kill_count),
                "damage_dealt": float(native_state.get("damage_dealt", 0.0) or 0.0),
                "norm_asymmetry": float(neural.get("norm_asymmetry", 0.0) or 0.0),
                "t4_l_v": float(neural.get("t4_l_v", -65.0) or -65.0),
                "t4_r_v": float(neural.get("t4_r_v", -65.0) or -65.0),
                "door_candidate": float(neural.get("door_candidate", 0.0) or 0.0),
                "firing_solution_locked": float(neural.get("firing_solution_locked", 0.0) or 0.0),
                "health_priority_active": float(neural.get("health_priority_active", 0.0) or 0.0),
            }
            state_combined.update(neural)

            recorded_ticks.append((step, action.name, float(obs.health), int(obs.ammo), int(obs.kill_count), real_rgb.copy(), state_combined))

            if step % 15 == 0 or action.name in ("FIRE", "USE"):
                print(f"  Step {step:03d}/{steps}: action={action.name:10s} kills={obs.kill_count} dmg={state_combined['damage_dealt']:.0f} asym={state_combined['norm_asymmetry']:+.3f}", flush=True)

            if done:
                print(f"Episode terminated at step {step}", flush=True)
                break

    finally:
        bridge.close()

    print(f"Rendering {len(recorded_ticks)} synchronized 3D nervous system composite frames...", flush=True)
    gif_frames = []
    for step, action_name, health_val, ammo_val, kills_val, real_rgb, state_combined in recorded_ticks:
        # 1. Left side: ACTUAL REAL GZDoom Gameplay Frame
        real_img = Image.fromarray(real_rgb)
        real_resized = real_img.resize((720, 540), Image.Resampling.BILINEAR)
        real_draw = ImageDraw.Draw(real_resized)

        # Overlay real gameplay metadata banner on top
        real_draw.rectangle([0, 0, 720, 38], fill=(8, 18, 24))
        real_draw.text((14, 11), f"ACTUAL GZDOOM GAMEPLAY  ·  STEP {step:03d}  ·  HP: {health_val:.0f}%", fill=(0, 229, 255))
        real_draw.text((430, 11), f"AMMO: {ammo_val}   KILLS: {kills_val} PK   ACTION: {action_name}", fill=(240, 166, 90))

        # Action border
        border_col = (0, 229, 255)
        if action_name == "FIRE":
            border_col = (255, 50, 0)
        elif action_name == "USE":
            border_col = (0, 255, 170)
        elif action_name.startswith("TURN"):
            border_col = (255, 160, 0)
        real_draw.rectangle([0, 0, 719, 539], outline=border_col, width=2)

        # 2. Right side: Isometric 3D Drosophila Nervous System
        pulse_phase = (step * 0.08) % 1.0
        brain_img = render_3d_nervous_system(state_combined, model_3d, mesh_data, width=640, height=540, pulse_phase=pulse_phase)

        # 3. Composite into single frame: 1360 x 540
        composite = Image.new("RGB", (1360, 540), (5, 10, 14))
        composite.paste(real_resized, (0, 0))
        composite.paste(brain_img, (720, 0))

        frame_path = tmp_dir / f"frame_{step:04d}.png"
        composite.save(frame_path)

        if step % 2 == 0:  # sample every 2nd frame to cover the entire 150-step combat run in the GIF
            gif_frames.append(composite.resize((680, 270), Image.Resampling.BILINEAR))

    # Encode MP4 using ffmpeg
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(output_mp4),
    ]
    print(f"Encoding MP4: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Saved actual gameplay MP4: {output_mp4} ({output_mp4.stat().st_size / 1024:.1f} KB)")

    # Encode GIF
    if gif_frames:
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        gif_frames[0].save(
            output_gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        print(f"Saved actual gameplay GIF: {output_gif} ({output_gif.stat().st_size / 1024:.1f} KB)")

    # Copy to artifacts directory
    artifact_dir = Path("/Users/ljp176/.gemini/antigravity/brain/fd394349-d70b-4c24-b0d6-2c95c6d775d7")
    if artifact_dir.exists():
        art_mp4 = artifact_dir / "actual_gameplay_3d_twin.mp4"
        shutil.copy(output_mp4, art_mp4)
        print(f"Copied to artifact: {art_mp4}")
        if output_gif.exists():
            art_gif = artifact_dir / "actual_gameplay_3d_twin.gif"
            shutil.copy(output_gif, art_gif)
            print(f"Copied to artifact: {art_gif}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record ACTUAL native GZDoom gameplay and 3D twin")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--output-mp4", type=Path, default=Path("runs/doom004_actual_gameplay_3d_twin.mp4"))
    parser.add_argument("--output-gif", type=Path, default=Path("runs/doom004_actual_gameplay_3d_twin.gif"))
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    run_actual_gameplay_recording(steps=args.steps, output_mp4=args.output_mp4, output_gif=args.output_gif, fps=args.fps)
