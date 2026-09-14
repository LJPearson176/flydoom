"""Export joined First-Person Doom Gameplay + Isometric 3D Fly Nervous System video and GIF.

Generates a synchronized composite clip:
- Left: 3D First-Person Doom perspective (E1M1 arena, corridor, door, enemies, crosshairs, shotgun, muzzle flash)
- Right: Isometric 3D Drosophila Nervous System Twin (JFRC2 brain mesh, 3,030 FlyWire fibers, 3D cuticle body, dynamic excitation)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq

from fly_doom.sensory.encoders.facet_atlas import CompoundEyeFacetAtlas
from fly_doom.sensory.encoders.calibrated_retina import EncoderCalibratedRetina, sample_retina


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
    return (0, 229, 255)


def create_joined_frame(
    tick: dict,
    model_3d: dict,
    mesh_data: dict,
    width: int = 1280,
    height: int = 640,
    pulse_phase: float = 0.0,
    facet_atlas: Optional[CompoundEyeFacetAtlas] = None,
    retina_encoder: Optional[EncoderCalibratedRetina] = None,
) -> Image.Image:
    half_w = width // 2
    img = Image.new("RGB", (width, height), (5, 10, 14))
    draw = ImageDraw.Draw(img)

    step = tick.get("step", 0)
    action = tick.get("action", "NOOP")
    health = float(tick.get("health", 100.0))
    ammo = int(tick.get("ammo", 50))
    kills = int(tick.get("kills", 0) or 0)
    player_kills = int(tick.get("player_kills", 0) or 0)
    ff_kills = int(tick.get("friendly_fire_kills", 0) or 0)
    damage = float(tick.get("damage_dealt", 0.0) or 0.0)
    px = float(tick.get("x", 1056.0) or 1056.0)
    py = float(tick.get("y", -3616.0) or -3616.0)
    angle_deg = float(tick.get("angle_deg", 0.0) or 0.0)
    target_deg = tick.get("target_angle_deg")
    target_x = tick.get("target_x")
    target_y = tick.get("target_y")
    target_hp = tick.get("target_health")
    target_vis = tick.get("target_visible")
    t4_l = float(tick.get("t4_l_v", -65.0) or -65.0)
    t4_r = float(tick.get("t4_r_v", -65.0) or -65.0)
    asym = float(tick.get("norm_asymmetry", 0.0) or 0.0)
    locked = bool(tick.get("firing_solution_locked", 0.0))
    is_fire = (action == "FIRE")
    is_door = (action == "USE" or tick.get("door_candidate"))
    is_damage = bool(tick.get("health_priority_active"))

    # =========================================================================
    # LEFT PANEL: FIRST-PERSON 3D DOOM COMBAT VIEWPORT
    # =========================================================================
    doom_box = [0, 0, half_w, height]
    draw.rectangle(doom_box, fill=(8, 16, 22))

    # Sky & Floor
    horizon_y = height // 2 - 10
    draw.rectangle([0, 0, half_w, horizon_y], fill=(12, 18, 26))  # dark ceiling
    draw.rectangle([0, horizon_y, half_w, height], fill=(18, 28, 32))  # floor

    # Floor grid perspective lines
    for i in range(-6, 7):
        vx_top = half_w // 2 + i * 25
        vx_bot = half_w // 2 + i * 90
        draw.line([(vx_top, horizon_y), (vx_bot, height)], fill=(28, 44, 50), width=1)

    # 3D Wall Projection (E1M1 start to courtyard)
    # Relative corridor distance based on player py progress (from -3616 to -2400)
    corridor_progress = max(0.0, min(1.0, (py - (-3616.0)) / (-2496.0 - (-3616.0))))
    door_dist = max(10.0, -2496.0 - py)

    # Corridor walls perspective
    wall_left_top = (0, int(horizon_y - 180 + corridor_progress * 40))
    wall_left_bot = (0, int(horizon_y + 180 - corridor_progress * 40))
    wall_right_top = (half_w, int(horizon_y - 180 + corridor_progress * 40))
    wall_right_bot = (half_w, int(horizon_y + 180 - corridor_progress * 40))

    vp_center = half_w // 2
    draw.polygon([wall_left_top, (vp_center - 120, horizon_y - 80), (vp_center - 120, horizon_y + 80), wall_left_bot], fill=(22, 34, 40))
    draw.polygon([wall_right_top, (vp_center + 120, horizon_y - 80), (vp_center + 120, horizon_y + 80), wall_right_bot], fill=(25, 38, 44))
    draw.line([wall_left_top, (vp_center - 120, horizon_y - 80)], fill=(40, 65, 75), width=2)
    draw.line([wall_left_bot, (vp_center - 120, horizon_y + 80)], fill=(40, 65, 75), width=2)
    draw.line([wall_right_top, (vp_center + 120, horizon_y - 80)], fill=(40, 65, 75), width=2)
    draw.line([wall_right_bot, (vp_center + 120, horizon_y + 80)], fill=(40, 65, 75), width=2)

    # Door Frame & Door (slides open if past door or interacting)
    door_open_y = 0
    if py > -2480.0 or is_door:
        door_open_y = int(min(60.0, max(0.0, (py - (-2480.0)) * 0.4 + (40.0 if is_door else 0.0))))

    door_w = int(max(40.0, min(240.0, 4800.0 / max(20.0, door_dist))))
    door_h = int(max(30.0, min(180.0, 3600.0 / max(20.0, door_dist))))
    door_rect = [vp_center - door_w // 2, horizon_y - door_h // 2 - door_open_y, vp_center + door_w // 2, horizon_y + door_h // 2 - door_open_y]
    door_frame = [vp_center - door_w // 2 - 6, horizon_y - door_h // 2 - 6, vp_center + door_w // 2 + 6, horizon_y + door_h // 2 + 6]
    draw.rectangle(door_frame, outline=(0, 229, 255), width=2)
    draw.rectangle(door_rect, fill=(35, 55, 65), outline=(0, 180, 200), width=1)
    if door_dist > 80.0:
        draw.text((vp_center - 14, horizon_y - 6), "DOOR", fill=(0, 229, 255))

    # Hostile Enemies (Imps / Zombiemen in the courtyard)
    if target_x is not None and target_y is not None:
        dx = target_x - px
        dy = target_y - py
        enemy_dist = math.hypot(dx, dy)
        enemy_bearing = math.degrees(math.atan2(dy, dx))
        rel_angle = ((enemy_bearing - angle_deg + 180.0) % 360.0) - 180.0

        # Only render if in front (FOV 90 deg)
        if -45.0 <= rel_angle <= 45.0 and enemy_dist > 5.0:
            screen_x = int(vp_center + (rel_angle / 45.0) * (half_w // 2 - 20))
            enemy_size = int(max(20.0, min(140.0, 16000.0 / enemy_dist)))
            ey = horizon_y + int(10.0 / (enemy_dist + 1.0))
            is_dead = (target_hp is not None and target_hp <= 0)

            imp_box = [screen_x - enemy_size // 2, ey - enemy_size, screen_x + enemy_size // 2, ey]
            if is_dead:
                # Dead monster on ground
                draw.ellipse([screen_x - enemy_size // 2, ey - 10, screen_x + enemy_size // 2, ey + 8], fill=(90, 20, 20), outline=(140, 30, 30))
                draw.text((screen_x - 18, ey - 22), "NEUTRALIZED", fill=(180, 80, 80))
            else:
                # Standing active Imp with glowing eyes
                draw.rectangle(imp_box, fill=(70, 45, 30), outline=(180, 90, 50), width=2)
                eye_w = max(2, enemy_size // 7)
                draw.rectangle([screen_x - enemy_size // 4, ey - int(enemy_size * 0.8), screen_x - enemy_size // 4 + eye_w, ey - int(enemy_size * 0.8) + eye_w], fill=(255, 50, 0))
                draw.rectangle([screen_x + enemy_size // 4 - eye_w, ey - int(enemy_size * 0.8), screen_x + enemy_size // 4, ey - int(enemy_size * 0.8) + eye_w], fill=(255, 50, 0))

                # Health Bar
                hp_val = target_hp if target_hp is not None else 60
                bar_w = max(30, enemy_size)
                bar_x1 = screen_x - bar_w // 2
                bar_y = ey - enemy_size - 16
                draw.rectangle([bar_x1, bar_y, bar_x1 + bar_w, bar_y + 6], fill=(30, 10, 10), outline=(100, 20, 20))
                fill_w = int(bar_w * max(0.0, min(1.0, hp_val / 60.0)))
                draw.rectangle([bar_x1, bar_y, bar_x1 + fill_w, bar_y + 6], fill=(255, 60, 60))
                draw.text((bar_x1, bar_y - 12), f"IMP  HP:{hp_val}", fill=(255, 100, 100))

                # Lock Reticle if locked
                if locked or abs(rel_angle) <= 18.0:
                    draw.ellipse([screen_x - enemy_size // 2 - 8, ey - enemy_size - 8, screen_x + enemy_size // 2 + 8, ey + 8], outline=(0, 255, 170), width=2)
                    draw.text((screen_x - 28, ey + 12), "LOCK-ON (<=18°)", fill=(0, 255, 170))

    # Crosshairs (Center)
    ch_col = (0, 255, 170) if locked else (88, 223, 194)
    draw.line([(vp_center - 12, horizon_y), (vp_center - 4, horizon_y)], fill=ch_col, width=2)
    draw.line([(vp_center + 4, horizon_y), (vp_center + 12, horizon_y)], fill=ch_col, width=2)
    draw.line([(vp_center, horizon_y - 12), (vp_center, horizon_y - 4)], fill=ch_col, width=2)
    draw.line([(vp_center, horizon_y + 4), (vp_center, horizon_y + 12)], fill=ch_col, width=2)

    # Shotgun Barrel & Muzzle Flash (Bottom)
    gun_x = vp_center
    gun_y = height - 10
    draw.polygon([(gun_x - 22, height), (gun_x - 14, height - 90), (gun_x + 14, height - 90), (gun_x + 22, height)], fill=(40, 48, 52), outline=(70, 80, 85))
    draw.rectangle([gun_x - 12, height - 100, gun_x + 12, height - 90], fill=(25, 30, 32), outline=(100, 110, 115))

    # Muzzle flash on FIRE
    if is_fire:
        draw.ellipse([gun_x - 40, height - 145, gun_x + 40, height - 80], fill=(255, 220, 80))
        draw.ellipse([gun_x - 22, height - 135, gun_x + 22, height - 90], fill=(255, 255, 255))
        for _ in range(8):
            sp_x = gun_x + np.random.randint(-35, 35)
            sp_y = height - 110 + np.random.randint(-40, 10)
            draw.line([(gun_x, height - 95), (sp_x, sp_y)], fill=(255, 160, 40), width=2)

    # Red Damage Vignette if damaged
    if is_damage:
        draw.rectangle([0, 0, half_w, 14], fill=(220, 20, 20))
        draw.rectangle([0, height - 14, half_w, height], fill=(220, 20, 20))
        draw.rectangle([0, 0, 14, height], fill=(220, 20, 20))
        draw.rectangle([half_w - 14, 0, half_w, height], fill=(220, 20, 20))

    # Top Doom Overlay Header
    draw.rectangle([0, 0, half_w, 42], fill=(8, 18, 24))
    draw.text((15, 12), f"GZDOOM E1M1  ·  STEP {step:03d}  ·  HP: {health:.0f}%", fill=(0, 229, 255))
    draw.text((280, 12), f"AMMO: {ammo}   KILLS: {player_kills} PK / {ff_kills} FF", fill=(240, 166, 90))

    # Action Badge in Doom View
    act_col = (0, 229, 255)
    act_text = action
    if is_fire:
        act_col = (255, 50, 0)
        act_text = "FIRE (STRIKE LOCK)"
    elif is_door:
        act_col = (0, 255, 170)
        act_text = "DOOR USE (TOUCH)"
    elif action == "TURN_RIGHT":
        act_col = (255, 160, 0)
        act_text = "OPTOMOTOR YAW RIGHT"
    elif action == "TURN_LEFT":
        act_col = (255, 160, 0)
        act_text = "OPTOMOTOR YAW LEFT"

    draw.rectangle([15, 52, 240, 80], fill=(10, 22, 28), outline=act_col, width=1)
    draw.text((25, 60), act_text, fill=act_col)

    # =========================================================================
    # RIGHT PANEL: ISOMETRIC 3D FLY NERVOUS SYSTEM & EXOSKELETON TWIN
    # =========================================================================
    draw.rectangle([half_w, 0, width, height], fill=(5, 10, 14))
    # Vertical divider line
    draw.line([(half_w, 0), (half_w, height)], fill=(22, 50, 62), width=2)

    # 3D Isometric Projection Parameters (Frontal-Isometric view matching user image)
    yaw = 0.22
    pitch = 0.20
    dist = 680.0
    target = [0.0, -70.0, 0.0]
    focal = 610.0
    cx_3d = half_w + half_w // 2
    cy_3d = height // 2 + 10

    def project_3d(x: float, y: float, z: float):
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
        return (cx_3d + x1 * scale, cy_3d - y2 * scale, scale)

    # 1. 3D Translucent Exoskeleton Cuticle (Head, Eyes, Thorax, Wings, Legs, Abdomen)
    for l in model_3d["exoskeleton"]["lines"]:
        p1 = project_3d(l[0], l[1], l[2])
        p2 = project_3d(l[3], l[4], l[5])
        if not p1 or not p2:
            continue
        ltype = l[6]
        col = (0, 80, 110)
        lw = 1
        if ltype == "eye":
            col = (60, 90, 150)
        elif ltype in {"wing", "wing_vein"}:
            col = (40, 100, 140)
        elif ltype == "leg":
            col = (0, 95, 125)
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=col, width=lw)

    # 2. Authentic JFRC2 Brain Surface Mesh (6,654 edges, drawn sampled for clarity)
    b_edges = mesh_data.get("brain_surface_edges", [])
    for i in range(0, len(b_edges), 3):
        e = b_edges[i]
        p1 = project_3d(e[0], e[1], e[2])
        p2 = project_3d(e[3], e[4], e[5])
        if p1 and p2:
            draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(15, 55, 85), width=1)

    # 3. Dense FlyWire Connectome Fibers (3,030 fibers) with Dynamic Activity Illumination
    fibers = model_3d["connectome"]["fibers"]
    for i, f in enumerate(fibers):
        pts = f["points"]
        neuropil = f.get("neuropil", "")
        pts_proj = [project_3d(p[0], p[1], p[2]) for p in pts]
        valid_pts = [(p[0], p[1]) for p in pts_proj if p is not None]
        if len(valid_pts) < 2:
            continue

        base_col = hex_to_rgb(f["color"])

        # Dynamic excitation coloring
        stroke_col = base_col
        lw = 1
        if is_fire and (neuropil.startswith("LO_") or neuropil.startswith("VNC_") or neuropil == "DN_TRUNK"):
            stroke_col = (255, 50, 0)
            lw = 2
        elif is_door and (neuropil == "SEZ" or neuropil == "VNC_T1"):
            stroke_col = (0, 255, 170)
            lw = 2
        elif is_damage and neuropil.startswith("MB_"):
            stroke_col = (255, 23, 68)
            lw = 2
        elif asym > 0.15 and (neuropil.endswith("_R") or pts[0][0] > 0):
            stroke_col = (255, 160, 0)
            lw = 2
        elif asym < -0.15 and (neuropil.endswith("_L") or pts[0][0] < 0):
            stroke_col = (255, 160, 0)
            lw = 2

        draw.line(valid_pts, fill=stroke_col, width=lw)

        # Traveling Action Potential pulses along axons
        if i % 8 == 0 and len(pts_proj) >= 2:
            t = (pulse_phase + i * 0.05) % 1.0
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
    top_f = project_3d(0, -15, 0)
    bot_f = project_3d(0, -65, -10)
    if top_f and bot_f:
        draw.ellipse([top_f[0] - 8, top_f[1], top_f[0] + 8, bot_f[1]], outline=(255, 60, 120), width=2)

    # Right Panel Header & Biological Callouts
    draw.rectangle([half_w, 0, width, 42], fill=(8, 18, 24))
    draw.text((half_w + 16, 12), "ISOMETRIC 3D VISIBLE FLY NERVOUS SYSTEM", fill=(0, 229, 255))
    draw.text((half_w + 390, 12), "JFRC2 TEMPLATE · 3,030 FLYWIRE FIBERS", fill=(88, 223, 194))

    # Compound-Eye Facet Atlas HUD Inset
    if facet_atlas is not None:
        card_w = 224
        card_h = 104
        card_x = width - card_w - 16
        card_y = 48
        draw.rectangle([card_x, card_y, card_x + card_w, card_y + card_h], fill=(8, 16, 22), outline=(22, 50, 62), width=1)
        draw.text((card_x + 8, card_y + 5), "COMPOUND EYE ATLAS", fill=(0, 229, 255))
        draw.text((card_x + card_w - 60, card_y + 5), "825 COLS", fill=(88, 223, 194))

        # Sample from the rendered Doom viewport
        doom_crop = img.crop((0, 0, half_w, height))
        doom_rgb = np.array(doom_crop)
        gray = np.mean(doom_rgb, axis=2).astype(np.float32)
        if retina_encoder is not None:
            retina_samples = sample_retina(gray, retina_encoder.calibrated_uv)
        else:
            retina_samples = None

        facet_atlas.render_pil(
            draw,
            x=card_x + 6,
            y=card_y + 18,
            w=card_w - 12,
            h=card_h - 22,
            values=retina_samples,
            mode="copper",
        )

    # Right Panel Bottom Telemetry Cards
    draw.rectangle([half_w + 15, height - 90, width - 15, height - 15], fill=(9, 20, 27), outline=(22, 50, 62), width=1)
    draw.text((half_w + 25, height - 82), f"ASYMMETRY Δ: {asym:+.3f}", fill=(240, 166, 90))
    draw.text((half_w + 200, height - 82), f"T4 L/R: {t4_l:.1f} / {t4_r:.1f} mV", fill=(0, 229, 255))
    draw.text((half_w + 400, height - 82), f"LOCK: {'LOCKED (<=18°)' if locked else 'SCANNING'}", fill=(0, 255, 170) if locked else (180, 190, 195))

    circ_desc = "BILATERAL MOTION EQUILIBRIUM"
    if is_fire:
        circ_desc = "LOBULA LC COLUMNAR -> DNpe017 -> T3 TRIGGER DISCHARGE"
    elif is_door:
        circ_desc = "GNG/SEZ MECHANOSENSORY PALPATION -> T1 FORELEG PUSH"
    elif asym > 0.15:
        circ_desc = "RIGHT MEDULLA M1-M10 -> LOP T4a/T5a -> RIGHT YAW THRUST"
    elif asym < -0.15:
        circ_desc = "LEFT MEDULLA M1-M10 -> LOP T4a/T5a -> LEFT YAW THRUST"

    draw.text((half_w + 25, height - 52), f"ACTIVE BIOPHYSICAL CIRCUIT: {circ_desc}", fill=(220, 235, 240))
    draw.text((half_w + 25, height - 32), f"BODY STATE: {'ENGAGING HOSTILE' if (target_hp and target_hp > 0) else 'FORWARD NAVIGATION'} · VNC DESCENDING TRACT ACTIVE", fill=(120, 150, 160))

    return img


def export_joined_clip(
    episode_dir: Path,
    output_mp4: Path,
    output_gif: Optional[Path] = None,
    fps: int = 12,
    max_frames: int = 150,
) -> None:
    parquet_path = episode_dir / "trajectory.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing trajectory.parquet in {episode_dir}")

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()[:max_frames]
    print(f"Loaded {len(rows)} ticks from {parquet_path}")

    with open("web/fly_3d_cns_model.json") as f:
        model_3d = json.load(f)
    with open("web/authentic_fly_cns_mesh.json") as f:
        mesh_data = json.load(f)

    tmp_dir = episode_dir / "joined_frames_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Rendering composite First-Person Doom + Isometric 3D Nervous System frames...")
    atlas = CompoundEyeFacetAtlas()
    retina_encoder = EncoderCalibratedRetina()
    frames_pil = []
    for i, row in enumerate(rows):
        pulse_phase = (i * 0.08) % 1.0
        frame = create_joined_frame(
            row,
            model_3d,
            mesh_data,
            width=1280,
            height=640,
            pulse_phase=pulse_phase,
            facet_atlas=atlas,
            retina_encoder=retina_encoder,
        )
        frame.save(tmp_dir / f"frame_{i:04d}.png")
        if output_gif and i < 80:  # first 80 frames for GIF to keep size compact
            frames_pil.append(frame.resize((640, 320), Image.Resampling.BILINEAR))
        if i % 30 == 0:
            print(f"  Rendered frame {i:03d} / {len(rows)}")

    # Encode MP4 using ffmpeg
    output_mp4 = Path(output_mp4)
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
    print(f"Saved MP4: {output_mp4} ({output_mp4.stat().st_size / 1024:.1f} KB)")

    # Encode GIF if requested
    if output_gif and frames_pil:
        output_gif = Path(output_gif)
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        frames_pil[0].save(
            output_gif,
            save_all=True,
            append_images=frames_pil[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        print(f"Saved GIF: {output_gif} ({output_gif.stat().st_size / 1024:.1f} KB)")

    # Cleanup temporary frame files
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Export joined First-Person Doom + Isometric 3D Fly Nervous System video")
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=Path("runs/doom004_lesions_v1/doom004-native-ModelD_ActiveTree_Saccade_3-1001-000"),
        help="Path to episode directory containing trajectory.parquet",
    )
    parser.add_argument(
        "--output-mp4",
        type=Path,
        default=Path("runs/doom004_lesions_v1/joined_isometric_3d_gameplay.mp4"),
        help="Output path for MP4 video",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=Path("runs/doom004_lesions_v1/joined_isometric_3d_gameplay.gif"),
        help="Output path for animated GIF",
    )
    parser.add_argument("--fps", type=int, default=12, help="Frames per second")
    args = parser.parse_args()

    export_joined_clip(args.episode_dir, args.output_mp4, args.output_gif, fps=args.fps)

    # Copy to artifact directory
    artifact_dir = Path("/Users/ljp176/.gemini/antigravity/brain/fd394349-d70b-4c24-b0d6-2c95c6d775d7")
    if artifact_dir.exists():
        if args.output_mp4.exists():
            art_mp4 = artifact_dir / "joined_isometric_3d_gameplay.mp4"
            shutil.copy(args.output_mp4, art_mp4)
            print(f"Copied to artifact: {art_mp4}")
        if args.output_gif and args.output_gif.exists():
            art_gif = artifact_dir / "joined_isometric_3d_gameplay.gif"
            shutil.copy(args.output_gif, art_gif)
            print(f"Copied to artifact: {art_gif}")


if __name__ == "__main__":
    main()
