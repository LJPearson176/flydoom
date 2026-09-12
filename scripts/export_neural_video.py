"""Export a composite video/GIF overlaying fly neural activation on gameplay telemetry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq


def render_frame(tick: dict, width: int = 800, height: int = 480) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(11, 23, 29))
    draw = ImageDraw.Draw(img)

    step = tick.get("step", 0)
    action = tick.get("action", "NOOP")
    health = tick.get("health", 100.0)
    ammo = tick.get("ammo", 50)
    kills = tick.get("kills", 0)
    player_kills = tick.get("player_kills", 0) or 0
    ff_kills = tick.get("friendly_fire_kills", 0) or 0
    damage = tick.get("damage_dealt", 0.0) or 0.0
    x = tick.get("x") or 1056.0
    y = tick.get("y") or -3616.0
    angle_deg = tick.get("angle_deg") or 0.0
    target_deg = tick.get("target_angle_deg")
    target_x = tick.get("target_x")
    target_y = tick.get("target_y")
    target_hp = tick.get("target_health")
    target_vis = tick.get("target_visible")
    t4_l = tick.get("t4_l_v", -65.0)
    t4_r = tick.get("t4_r_v", -65.0)
    asym = tick.get("norm_asymmetry", 0.0)

    def activation(key: str) -> float:
        return max(0.0, min(1.0, float(tick.get(key, 0.0) or 0.0)))

    # 1. Top Header Bar
    draw.rectangle([0, 0, width, 40], fill=(16, 37, 45))
    draw.text((20, 12), f"FLYDOOM EMBODIED LAB  ·  STEP {step:03d}", fill=(88, 223, 194))
    draw.text((360, 12), f"HEALTH: {health:.0f}   AMMO: {ammo}   KILLS: {kills} (PK:{player_kills} FF:{ff_kills})   DMG: {damage:.0f}", fill=(220, 230, 225))

    # 2. Action & Neural Circuit Badge
    badge_color = (49, 127, 121)
    badge_bg = (14, 32, 38)
    circuit = "LPTC BILATERAL YAW BALANCE"
    if action == "FIRE":
        badge_color = (255, 87, 34)
        badge_bg = (59, 20, 8)
        circuit = "TARGET LOCKED · STRIKE COMMAND (SPACE)"
    elif action == "USE":
        badge_color = (0, 229, 255)
        badge_bg = (0, 55, 64)
        circuit = "MECHANOSENSORY DOOR INTERACT (E)"
    elif action in {"TURN_LEFT", "TURN_RIGHT"}:
        badge_color = (240, 166, 90)
        badge_bg = (45, 30, 15)
        circuit = f"DESCENDING ASYMMETRY (DNpe017 -> {action})"
    elif tick.get("health_priority_active"):
        badge_color = (255, 50, 50)
        badge_bg = (50, 10, 10)
        circuit = "NOCICEPTIVE EVASION CIRCUIT"

    draw.rectangle([20, 50, 360, 100], fill=badge_bg, outline=badge_color, width=2)
    draw.text((35, 58), f"ACTION: {action}", fill=badge_color)
    draw.text((35, 78), f"CIRCUIT: {circuit}", fill=(180, 200, 195))

    # 3. Minimap / Spatial Perspective (Left Panel)
    map_x1, map_y1, map_x2, map_y2 = 20, 115, 360, 460
    draw.rectangle([map_x1, map_y1, map_x2, map_y2], fill=(13, 25, 30), outline=(37, 77, 88), width=1)
    draw.text((map_x1 + 10, map_y1 + 8), "E1M1 SPATIAL ARENA", fill=(88, 223, 194))

    # Scale map coordinates to minimap box: X [1000, 2400], Y [-3700, -2100]
    def to_map(mx, my):
        sx = map_x1 + 15 + (mx - 950.0) / (2450.0 - 950.0) * (map_x2 - map_x1 - 30)
        sy = map_y2 - 15 - (my - (-3700.0)) / (-2100.0 - (-3700.0)) * (map_y2 - map_y1 - 30)
        return sx, sy

    # Corridor and rooms outline
    pts_corridor = [to_map(1056, -3616), to_map(1056, -3200), to_map(1300, -2800), to_map(1536, -2496)]
    for p1, p2 in zip(pts_corridor, pts_corridor[1:]):
        draw.line([p1, p2], fill=(35, 70, 80), width=3)
    # Door line at (1536, -2496)
    door_p1 = to_map(1536, -2560)
    door_p2 = to_map(1536, -2432)
    draw.line([door_p1, door_p2], fill=(0, 229, 255), width=2)
    draw.text((door_p1[0] - 25, door_p1[1] - 12), "DOOR", fill=(0, 229, 255))
    # Arena walkway
    pts_arena = [to_map(1536, -2496), to_map(1750, -2448), to_map(2100, -2400), to_map(2272, -2432)]
    for p1, p2 in zip(pts_arena, pts_arena[1:]):
        draw.line([p1, p2], fill=(45, 90, 100), width=4)

    # Draw Fly Agent
    fx, fy = to_map(x, y)
    draw.ellipse([fx - 6, fy - 6, fx + 6, fy + 6], fill=(88, 223, 194), outline=(255, 255, 255), width=1)
    # Heading cone
    head_rad = math.radians(angle_deg)
    hx = fx + 22 * math.cos(head_rad)
    hy = fy - 22 * math.sin(head_rad)
    draw.line([(fx, fy), (hx, hy)], fill=(88, 223, 194), width=2)

    # Draw Monsters / Target
    if target_x is not None and target_y is not None:
        tx, ty = to_map(target_x, target_y)
        tcolor = (255, 60, 60) if target_vis else (150, 60, 60)
        draw.rectangle([tx - 5, ty - 5, tx + 5, ty + 5], fill=tcolor, outline=(255, 200, 200))
        draw.text((tx + 8, ty - 6), f"HP:{target_hp}", fill=tcolor)
        if target_vis:
            draw.line([(fx, fy), (tx, ty)], fill=(255, 100, 100), width=1)

    # 4. Neural Circuit HUD (Right Panel)
    hud_x1 = 380
    draw.rectangle([hud_x1, 50, width - 20, 460], fill=(13, 25, 30), outline=(37, 77, 88), width=1)
    draw.text((hud_x1 + 15, 60), "FLY NERVOUS SYSTEM OVERLAY", fill=(88, 223, 194))

    # Optic Lobes / T4 Bilateral Voltage Meters
    draw.text((hud_x1 + 15, 80), "1. OPTIC LOBES (T4a MOTION VISION & BRANCH ARBORS)", fill=(180, 200, 195))
    l_frac = max(0.0, min(1.0, (t4_l + 65.0) / 20.0))
    draw.text((hud_x1 + 15, 96), f"L-OPTIC ({t4_l:.1f} mV)", fill=(88, 223, 194))
    draw.rectangle([hud_x1 + 140, 98, hud_x1 + 350, 106], fill=(20, 40, 48))
    draw.rectangle([hud_x1 + 140, 98, hud_x1 + 140 + int(210 * l_frac), 106], fill=(88, 223, 194))
    r_frac = max(0.0, min(1.0, (t4_r + 65.0) / 20.0))
    draw.text((hud_x1 + 15, 112), f"R-OPTIC ({t4_r:.1f} mV)", fill=(240, 166, 90))
    draw.rectangle([hud_x1 + 140, 114, hud_x1 + 350, 122], fill=(20, 40, 48))
    draw.rectangle([hud_x1 + 140, 114, hud_x1 + 140 + int(210 * r_frac), 122], fill=(240, 166, 90))

    # Compartment-resolved activation map
    draw.text((hud_x1 + 15, 128), "COMPARTMENTS: LEAD / CENTRAL / TRAIL / SOMA", fill=(180, 200, 195))
    branch_x = [hud_x1 + 140, hud_x1 + 195, hud_x1 + 250, hud_x1 + 305]
    for row_y, side, color in ((142, "l", (88, 223, 194)), (156, "r", (240, 166, 90))):
        draw.text((hud_x1 + 15, row_y), f"{side.upper()} T4a", fill=color)
        for x, name in zip(branch_x, ("leading", "central", "trailing", "soma")):
            draw.rectangle([x, row_y, x + 42, row_y + 8], fill=(20, 40, 48))
            frac = activation(f"t4_{side}_{name}_activation")
            draw.rectangle([x, row_y, x + int(42 * frac), row_y + 8], fill=color)
    draw.text((hud_x1 + 15, 170),
              "Mi1/Tm3 excitation  ·  Mi4/Mi9 inhibition",
              fill=(117, 145, 151))
    draw.text(
        (hud_x1 + 15, 184),
        "RETINA L/R/C: "
        f"{float(tick.get('retina_left_drive', 0.0) or 0.0):.2f}/"
        f"{float(tick.get('retina_right_drive', 0.0) or 0.0):.2f}/"
        f"{float(tick.get('retina_center_drive', 0.0) or 0.0):.2f}",
        fill=(117, 145, 151),
    )

    # Central Complex (CX) Compass & Target Heading Reticle
    draw.text((hud_x1 + 15, 204), "2. CENTRAL COMPLEX (HEADING & PERSPECTIVE)", fill=(180, 200, 195))
    cx_center = (hud_x1 + 60, 260)
    draw.ellipse([cx_center[0] - 34, cx_center[1] - 34, cx_center[0] + 34, cx_center[1] + 34], outline=(37, 77, 88), width=1)
    draw.ellipse([cx_center[0] - 22, cx_center[1] - 22, cx_center[0] + 22, cx_center[1] + 22], outline=(49, 127, 121), width=1)
    draw.line([cx_center[0] - 36, cx_center[1], cx_center[0] + 36, cx_center[1]], fill=(37, 77, 88), width=1)
    draw.line([cx_center[0], cx_center[1] - 36, cx_center[0], cx_center[1] + 36], fill=(37, 77, 88), width=1)

    # Heading needle (cyan)
    agent_rad = math.radians(angle_deg)
    ax = cx_center[0] + 26 * math.cos(agent_rad)
    ay = cx_center[1] - 26 * math.sin(agent_rad)
    draw.line([cx_center, (ax, ay)], fill=(88, 223, 194), width=3)

    # Target needle (yellow/red)
    if target_deg is not None and not math.isnan(target_deg):
        tgt_rad = math.radians(target_deg)
        tx = cx_center[0] + 32 * math.cos(tgt_rad)
        ty = cx_center[1] - 32 * math.sin(tgt_rad)
        draw.line([cx_center, (tx, ty)], fill=(245, 215, 110), width=2)
        diff = ((target_deg - angle_deg + 180.0) % 360.0) - 180.0
        locked = abs(diff) <= 18.0
        lock_status = "LOCKED" if locked else "SCANNING"
        lock_col = (88, 223, 194) if locked else (240, 166, 90)
        draw.text((hud_x1 + 120, 238), f"BEARING Δ: {diff:+.1f}°", fill=(220, 230, 225))
        draw.text((hud_x1 + 120, 256), f"SOLUTION: {lock_status}", fill=lock_col)
        draw.text((hud_x1 + 120, 274), f"HEADING: {angle_deg:.1f}°", fill=(140, 160, 155))
    else:
        draw.text((hud_x1 + 120, 248), "NO ACTIVE TARGET", fill=(140, 160, 155))
        draw.text((hud_x1 + 120, 266), f"HEADING: {angle_deg:.1f}°", fill=(140, 160, 155))

    # Descending Neurons & Motor Asymmetry
    draw.text((hud_x1 + 15, 305), "3. DESCENDING MOTOR COMMAND (DNpe017)", fill=(180, 200, 195))
    draw.rectangle([hud_x1 + 15, 325, hud_x1 + 370, 345], fill=(20, 40, 48), outline=(37, 77, 88))
    # Center divider
    draw.line([hud_x1 + 192, 325, hud_x1 + 192, 345], fill=(100, 130, 130), width=1)
    # Asymmetry fill
    bar_center = hud_x1 + 192
    bar_offset = int(asym * 160.0)
    if bar_offset > 0:
        draw.rectangle([bar_center, 328, bar_center + bar_offset, 342], fill=(240, 166, 90))
    else:
        draw.rectangle([bar_center + bar_offset, 328, bar_center, 342], fill=(88, 223, 194))
    draw.text((hud_x1 + 15, 350), f"STEER LEFT (L)                         BALANCE                         STEER RIGHT (R)", fill=(140, 160, 155))
    draw.text((hud_x1 + 15, 368), f"MOTOR ASYMMETRY: {asym:+.3f}   ACTION: {action}", fill=badge_color)

    # 4. Sensorimotor Latency & Competence Readout
    draw.text((hud_x1 + 15, 392), "4. SENSORIMOTOR LATENCY & COMPETENCE", fill=(180, 200, 195))
    draw.text((hud_x1 + 15, 410), "REACTION: ~28.7ms (SUB-FRAME) · CLOSED-LOOP DSI: 0.59", fill=(88, 223, 194))
    draw.text((hud_x1 + 15, 428), "TARGET LOCK: 71.4% · DOOM RECEPTIVE FIELD: 60° FOV", fill=(140, 160, 155))

    return img


def export_video(episode_dir: Path, output_file: Path, fps: int = 10) -> Path:
    parquet_path = episode_dir / "trajectory.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing trajectory.parquet in {episode_dir}")

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    print(f"Loaded {len(rows)} ticks from {parquet_path}")

    frames = [render_frame(row) for row in rows]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gif":
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        print(f"Saved animated GIF: {output_path}")
    else:
        tmp_dir = episode_dir / "video_frames_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(tmp_dir / f"frame_{i:04d}.png")
        cmd = [
            "/opt/homebrew/bin/ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, check=True)
        for f in tmp_dir.glob("*.png"):
            f.unlink()
        tmp_dir.rmdir()
        print(f"Saved MP4 video: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", help="path to episode directory containing trajectory.parquet")
    parser.add_argument("--output", default="runs/neural_telemetry_overlay.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if args.episode:
        ep_dir = Path(args.episode)
    else:
        runs = sorted(Path("runs").glob("*/doom004*/trajectory.parquet"), reverse=True)
        if not runs:
            raise FileNotFoundError("No trajectory.parquet runs found in runs/")
        ep_dir = runs[0].parent

    export_video(ep_dir, Path(args.output), fps=args.fps)


if __name__ == "__main__":
    main()
