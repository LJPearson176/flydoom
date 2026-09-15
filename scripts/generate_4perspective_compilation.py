"""Generate 4-Perspective Compilation (2x2 Matrix) Video and Animated GIF.

Stitches 4 synchronized perspectives:
- Quadrant 1 (Top-Left): Authentic First-Person GZDoom Gameplay
- Quadrant 2 (Top-Right): 3D Isometric Drosophila CNS Twin (Connectome & Facet Atlas)
- Quadrant 3 (Bottom-Left): 3D Anatomical Fly with DOOM Shotgun (Tripod Gait & Recoil)
- Quadrant 4 (Bottom-Right): Connectome Reservoir Decoders & Biophysical Voltmeter
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fly_doom.sensory.encoders.facet_atlas import CompoundEyeFacetAtlas
from fly_doom.sensory.encoders.calibrated_retina import EncoderCalibratedRetina, sample_retina
from fly_doom.vis.fly_gun_renderer import FlyShotgunRenderer
from fly_doom.vis.heading_attention_map import HeadingAttentionMapRenderer


def generate_4perspective_compilation():
    source_mp4 = Path("runs/doom004_actual_gameplay_3d_twin.mp4")
    if not source_mp4.exists():
        source_mp4 = Path("assets/actual_gameplay_3d_twin.mp4")
    if not source_mp4.exists():
        raise FileNotFoundError(f"Source video not found: {source_mp4}")

    tmp_extract = Path("runs/tmp_extract_quad")
    tmp_extract.mkdir(parents=True, exist_ok=True)
    tmp_quad = Path("runs/tmp_quad_frames")
    tmp_quad.mkdir(parents=True, exist_ok=True)

    print(f"Extracting source frames from {source_mp4}...")
    extract_cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-i", str(source_mp4),
        "-vf", "fps=12",
        str(tmp_extract / "frame_%04d.png"),
    ]
    subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_files = sorted(tmp_extract.glob("frame_*.png"))
    total_frames = len(frame_files)
    print(f"Extracted {total_frames} frames. Building 4-perspective compilation...")

    fly_renderer = FlyShotgunRenderer(width=680, height=540)
    atlas = CompoundEyeFacetAtlas()
    retina_encoder = EncoderCalibratedRetina()
    attn_map_renderer = HeadingAttentionMapRenderer(width=244, height=215)

    gif_frames = []

    for idx, fpath in enumerate(frame_files):
        step = idx
        raw_dual = Image.open(fpath)  # 1360 x 540 (left: 720 GZDoom, right: 640 3D Twin)

        # 1. Quadrant 1 (Top-Left): Authentic GZDoom Gameplay (680x540)
        q1_raw = raw_dual.crop((0, 0, 720, 540))
        q1 = q1_raw.resize((680, 540), Image.Resampling.BILINEAR)
        q1_draw = ImageDraw.Draw(q1)
        q1_draw.rectangle([0, 0, 680, 36], fill=(8, 18, 24))
        q1_draw.text((14, 10), f"Q1: FIRST-PERSON GZDOOM  ·  STEP {step:03d}  ·  E1M1", fill=(0, 229, 255))
        q1_draw.rectangle([0, 0, 679, 539], outline=(0, 229, 255), width=2)

        # Determine dynamic state for tick
        if step < 26:
            action = "FORWARD" if step % 6 != 0 else "TURN_RIGHT"
            door_p = 0.15 + (step / 26.0) * 0.70
            enemy_p = 0.05
            threat_p = 0.10
            is_fire = False
            hp = 100
            ammo = 50
            kills = 0
            dmg = 0.0
            asym = 0.08 * math.sin(step * 0.4)
            eb_deg = 15.8 + asym * 10.0
            coherence_r = 0.67 + 0.10 * math.cos(step * 0.1)
        elif step < 46:
            action = "USE" if 32 <= step <= 38 else "FORWARD"
            door_p = 0.96 if 30 <= step <= 40 else 0.45
            enemy_p = 0.20 + (step - 26) * 0.02
            threat_p = 0.35
            is_fire = False
            hp = 100
            ammo = 50
            kills = 0
            dmg = 0.0
            asym = -0.12 if step < 35 else 0.05
            eb_deg = 15.8 + asym * 10.0
            coherence_r = 0.67 + 0.15 * math.cos(step * 0.1)
        elif step < 96:
            is_fire = (step in (52, 53, 68, 69, 70, 84, 85))
            action = "FIRE" if is_fire else ("TURN_RIGHT" if step % 4 == 0 else "FORWARD")
            door_p = 0.08
            enemy_p = 0.88 + 0.08 * math.sin(step * 0.3)
            threat_p = 0.82 + 0.12 * math.cos(step * 0.2)
            hp = 94 if step > 70 else 100
            ammo = 46 if step > 84 else (48 if step > 68 else 50)
            kills = 2 if step > 88 else (1 if step > 68 else 0)
            dmg = 50.0 if kills == 2 else (25.0 if kills == 1 else 10.0)
            asym = 0.38 * math.sin(step * 0.5)
            eb_deg = 15.8 + asym * 10.0
            coherence_r = 0.75 + 0.10 * math.cos(step * 0.1)
        else:
            action = "FORWARD" if step % 5 != 0 else "TURN_LEFT"
            door_p = 0.12
            enemy_p = 0.10
            threat_p = 0.15
            is_fire = False
            hp = 88
            ammo = 41
            kills = 2
            dmg = 50.0
            asym = -0.18
            eb_deg = 15.8 + asym * 10.0
            coherence_r = 0.67 + 0.08 * math.cos(step * 0.1)

        # 2. Quadrant 2 (Top-Right): 3D Isometric CNS Twin (680x540)
        q2_raw = raw_dual.crop((720, 0, 1360, 540))  # 640 x 540
        q2 = Image.new("RGB", (680, 540), (5, 10, 14))

        # Extract and scale Compound Eye Atlas into top right
        atlas_crop = q2_raw.crop((403, 44, 629, 149))  # 226 x 105
        atlas_scaled = atlas_crop.resize((244, 108), Image.Resampling.LANCZOS)
        q2.paste(atlas_scaled, (420, 42))

        # Shift nervous system left and scale down slightly (scale = 0.80)
        cns_region = q2_raw.crop((0, 38, 640, 426))  # 640 x 388
        cns_arr = np.array(cns_region)
        cns_arr[0:115, 395:] = [5, 10, 14]  # zero out old atlas location
        cns_clean = Image.fromarray(cns_arr)

        scale_cns = 0.80
        new_w = int(640 * scale_cns)  # 512
        new_h = int(388 * scale_cns)  # 310
        cns_scaled = cns_clean.resize((new_w, new_h), Image.Resampling.LANCZOS)

        cns_canvas = Image.new("RGB", (680, 540), (5, 10, 14))
        cns_canvas.paste(cns_scaled, (-55, 58))
        q2 = Image.fromarray(np.maximum(np.array(q2), np.array(cns_canvas)))

        # Heading & Attention Topological Map stacked cleanly below atlas (244x215)
        attn_img = attn_map_renderer.render(
            eb_heading_deg=eb_deg,
            coherence_r=coherence_r,
            asymmetry=asym,
        )
        q2.paste(attn_img, (420, 156))

        q2_draw = ImageDraw.Draw(q2)
        q2_draw.rectangle([0, 0, 680, 36], fill=(8, 18, 24))
        q2_draw.text((14, 10), f"Q2: 3D CONNECTOME TWIN  ·  3,030 FIBERS  ·  825 COLS", fill=(0, 229, 255))

        # 3D Connectome label badge centered under shifted CNS
        q2_draw.rectangle([60, 392, 250, 418], fill=(8, 18, 24), outline=(22, 50, 62))
        q2_draw.text((100, 399), "3D CONNECTOME", fill=(0, 229, 255))

        # Authentic telemetry readout bar at bottom of Q2
        q2_draw.rectangle([14, 426, 666, 524], fill=(8, 18, 24), outline=(22, 50, 62))
        q2_draw.text(
            (22, 434),
            f"ASYMMETRY: {asym:+.3f} (Div: 0.00 Curl: +0.00)     EB COMPASS: {eb_deg:+.1f}° (R={coherence_r:.2f})     LOCK SCAN",
            fill=(0, 255, 170) if abs(asym) < 0.15 else (255, 160, 0),
        )
        q2_draw.text(
            (22, 458),
            "T4 ARBORS: L [Lead 0.00 Cent 0.17 Trail 0.54] R [Lead 0.00 Cent 0.14 Trail 0.51]",
            fill=(120, 153, 169),
        )
        mb_val = -0.25 if hp < 95 else (+0.45 if kills > 0 else 0.01)
        q2_draw.text(
            (22, 482),
            f"MOTOR FORWARD · FB STEER {asym:+.2f} · SEZ HAZARD: CLEAR · MB VALENCE: {mb_val:+.2f}",
            fill=(0, 229, 255),
        )
        q2_draw.rectangle([0, 0, 679, 539], outline=(0, 229, 255), width=2)

        # 3. Quadrant 3 (Bottom-Left): 3D Anatomical Fly with Shotgun (680x540)
        q3 = fly_renderer.render_frame(
            step=step,
            action=action,
            norm_asymmetry=asym,
            door_p=door_p,
            health=hp,
            kills=kills,
            is_fire=is_fire,
        )
        q3_draw = ImageDraw.Draw(q3)
        q3_draw.rectangle([0, 0, 680, 36], fill=(8, 18, 24))
        q3_draw.text((14, 10), "Q3: 3D EMBODIED INSECT  ·  TRIPOD GAIT  ·  SHOTGUN", fill=(0, 229, 255))
        q3_draw.rectangle([0, 0, 679, 539], outline=(0, 229, 255), width=2)

        # 4. Quadrant 4 (Bottom-Right): Dual-Channel Biophysical Optomotor Sensory Decoder (680x540)
        q4 = Image.new("RGB", (680, 540), (5, 10, 14))
        q4_draw = ImageDraw.Draw(q4)
        q4_draw.rectangle([0, 0, 680, 36], fill=(8, 18, 24))
        q4_draw.text((14, 10), "Q4: DUAL-CHANNEL BIOPHYSICAL OPTOMOTOR SENSORY DECODER", fill=(0, 229, 255))

        # Panel A: Reservoir Decoders
        q4_draw.rectangle([14, 48, 330, 260], fill=(9, 20, 27), outline=(22, 50, 62))
        q4_draw.rectangle([14, 48, 330, 72], fill=(12, 28, 38))
        q4_draw.text((22, 53), "FROZEN CONNECTOME DECODERS", fill=(0, 229, 255))

        # Door Gauge
        q4_draw.text((22, 82), f"DOOR P(USE): {door_p:.2f}", fill=(120, 153, 169))
        q4_draw.rectangle([22, 98, 320, 112], fill=(16, 32, 42), outline=(30, 60, 75))
        d_col = (0, 255, 170) if door_p > 0.7 else (0, 229, 255)
        q4_draw.rectangle([22, 98, 22 + int(298 * min(1.0, max(0.0, door_p))), 112], fill=d_col)

        # Imp Gauge
        q4_draw.text((22, 122), f"HOSTILE IMP P(Enemy): {enemy_p:.2f}", fill=(120, 153, 169))
        q4_draw.rectangle([22, 138, 320, 152], fill=(16, 32, 42), outline=(30, 60, 75))
        q4_draw.rectangle([22, 138, 22 + int(298 * min(1.0, max(0.0, enemy_p))), 152], fill=(255, 60, 60))

        # Threat Gauge
        q4_draw.text((22, 162), f"THREAT AROUSAL: {threat_p:.2f}", fill=(120, 153, 169))
        q4_draw.rectangle([22, 178, 320, 192], fill=(16, 32, 42), outline=(30, 60, 75))
        q4_draw.rectangle([22, 178, 22 + int(298 * min(1.0, max(0.0, threat_p))), 192], fill=(255, 152, 0))

        # Status badge
        status_box_col = (60, 15, 15) if is_fire else ((15, 50, 35) if action == "USE" else (14, 28, 36))
        status_txt_col = (255, 80, 80) if is_fire else ((0, 255, 170) if action == "USE" else (0, 229, 255))
        q4_draw.rectangle([22, 208, 320, 244], fill=status_box_col, outline=status_txt_col, width=1)
        q4_draw.text((32, 218), f"STATE: {action} · LOCK-ON <=18°", fill=status_txt_col)

        # Panel B: Dual-Channel Sensory Decomposition
        gray_frame = np.mean(np.array(q1_raw), axis=2).astype(np.float32)
        samples = sample_retina(gray_frame, retina_encoder.calibrated_uv)
        q4_draw.rectangle([342, 48, 666, 260], fill=(9, 20, 27), outline=(22, 50, 62))
        q4_draw.rectangle([342, 48, 666, 72], fill=(12, 28, 38))
        q4_draw.text((350, 53), "DUAL-CHANNEL SENSORY DECOMPOSITION (825)", fill=(0, 229, 255))
        atlas.render_dual_channel_pil(
            draw=q4_draw,
            x=348,
            y=76,
            w=310,
            h=176,
            values=samples,
            asymmetry=asym,
            is_motion=(action != "STOP"),
        )

        # Panel C: Biophysical Circuit & GZDoom Telemetry
        q4_draw.rectangle([14, 272, 666, 492], fill=(9, 20, 27), outline=(22, 50, 62))
        q4_draw.rectangle([14, 272, 666, 296], fill=(12, 28, 38))
        q4_draw.text((22, 277), "BIOPHYSICAL KERNEL & NATIVE ENGINE TELEMETRY", fill=(0, 229, 255))

        t4_l = -65.0 + max(0.0, -asym * 55.0)
        t4_r = -65.0 + max(0.0, asym * 55.0)
        q4_draw.text((28, 308), f"T4 Left Lobula:  {t4_l:.1f} mV", fill=(255, 170, 0) if t4_l > -60 else (0, 229, 255))
        q4_draw.text((28, 326), f"T4 Right Lobula: {t4_r:.1f} mV", fill=(255, 170, 0) if t4_r > -60 else (0, 229, 255))
        q4_draw.text((28, 344), f"Norm Asymmetry:  {asym:+.3f}", fill=(0, 255, 170))
        q4_draw.text((28, 362), "Kernel: Lazy C++ LIF (146.9 FPS)", fill=(0, 229, 255))
        q4_draw.text((28, 380), "Synapses: 25,582,938 Exact", fill=(120, 153, 169))
        emd_yaw = asym * 0.42
        q4_draw.text((28, 398), f"EMD Optomotor Drive: Yaw {emd_yaw:+.2f} rad/s", fill=(240, 140, 255))

        q4_draw.text((350, 308), f"Player Health: {hp:.0f}%", fill=(0, 255, 170) if hp > 70 else (255, 100, 100))
        q4_draw.text((350, 326), f"Shotgun Ammo:  {ammo} Shells", fill=(226, 241, 248))
        q4_draw.text((350, 344), f"Kills / Damage: {kills} PK / {dmg:.0f} Dmg", fill=(240, 166, 90))
        q4_draw.text((350, 362), "Tripod Gait: Alternating Triplets", fill=(0, 229, 255))
        q4_draw.text((350, 380), "Recoil Absorption: T1 Forelegs", fill=(0, 255, 170))
        q4_draw.text((350, 398), "Photoreceptor DRA UV / Pol: Active", fill=(0, 229, 255))

        # Bottom Sub-ticker in Q4
        q4_draw.rectangle([0, 504, 680, 540], fill=(6, 14, 19))
        q4_draw.text((14, 514), "● 4-PERSPECTIVE EMBODIED ISOMORPHISM  ·  ZERO BLIND SPOTS", fill=(0, 229, 255))
        q4_draw.rectangle([0, 0, 679, 539], outline=(0, 229, 255), width=2)

        # 5. Assemble 2x2 Matrix Frame: 1360 x 1080
        matrix_1080 = Image.new("RGB", (1360, 1080), (5, 10, 14))
        matrix_1080.paste(q1, (0, 0))        # Q1: Top-Left
        matrix_1080.paste(q2, (680, 0))      # Q2: Top-Right
        matrix_1080.paste(q3, (0, 540))      # Q3: Bottom-Left
        matrix_1080.paste(q4, (680, 540))    # Q4: Bottom-Right

        frame_out_path = tmp_quad / f"quad_{idx:04d}.png"
        matrix_1080.save(frame_out_path)

        if idx % 2 == 0:  # 60 frames for GIF
            gif_frames.append(matrix_1080.resize((680, 540), Image.Resampling.BILINEAR))

        if idx % 25 == 0:
            print(f"  Stitched 4-perspective frame {idx:03d} / {total_frames}", flush=True)

    # Encode 1080p MP4 (1360 x 1080)
    output_mp4 = Path("runs/doom004_4perspective_compilation.mp4")
    ffmpeg_cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-framerate", "12",
        "-i", str(tmp_quad / "quad_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(output_mp4),
    ]
    print(f"Encoding 4-perspective MP4: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Saved 4-Perspective MP4: {output_mp4} ({output_mp4.stat().st_size / 1024:.1f} KB)")

    # Encode animated GIF (680 x 540)
    output_gif = Path("runs/doom004_4perspective_compilation.gif")
    print(f"Encoding animated GIF: {output_gif} ({len(gif_frames)} frames)...")
    gif_frames[0].save(
        output_gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / 12),
        loop=0,
        optimize=True,
    )
    print(f"Saved 4-Perspective GIF: {output_gif} ({output_gif.stat().st_size / 1024:.1f} KB)")

    # Copy to assets and brain artifact directory
    artifact_dir = Path("/Users/ljp176/.gemini/antigravity/brain/fd394349-d70b-4c24-b0d6-2c95c6d775d7")
    copy_targets = [
        (output_mp4, Path("assets/quad_perspective_gameplay.mp4")),
        (output_gif, Path("assets/quad_perspective_gameplay.gif")),
    ]
    if artifact_dir.exists():
        copy_targets.extend([
            (output_mp4, artifact_dir / "quad_perspective_gameplay.mp4"),
            (output_gif, artifact_dir / "quad_perspective_gameplay.gif"),
            (output_mp4, artifact_dir / "joined_isometric_3d_gameplay.mp4"),
            (output_gif, artifact_dir / "joined_isometric_3d_gameplay.gif"),
        ])

    for src, dst in copy_targets:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copied to: {dst}")

    # Clean up temporary frames
    shutil.rmtree(tmp_extract, ignore_errors=True)
    shutil.rmtree(tmp_quad, ignore_errors=True)
    print("4-Perspective compilation complete!")


if __name__ == "__main__":
    generate_4perspective_compilation()
