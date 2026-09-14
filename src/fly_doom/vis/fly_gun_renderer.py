"""3D Anatomical Fly with DOOM Shotgun Renderer.

Renders an articulated, shaded 3D Drosophila Melanogaster holding a DOOM Shotgun
from an elevated 3rd-person action camera perspective (matching Nick Walton's
viral Rubik's cube fly experiment).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fly_doom.vis.fly_gun_model import AnatomicalFlyShotgunModel, FlyKinematicPose


class FlyShotgunRenderer:
    """Renders 3D Anatomical Fly holding DOOM Shotgun to PIL Image."""

    def __init__(self, width: int = 680, height: int = 540):
        self.width = width
        self.height = height
        self.model = AnatomicalFlyShotgunModel()

        # Camera setup: 3rd-person follow action cam
        self.cam_yaw = 0.46     # ~26 degrees
        self.cam_pitch = 0.28   # ~16 degrees
        self.cam_dist = 520.0
        self.cam_target = np.array([0.0, -40.0, -10.0], dtype=np.float32)
        self.focal = 580.0

    def project(self, p: np.ndarray | Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Project 3D point (x, y, z) into 2D screen coordinates with depth."""
        x, y, z = p[0], p[1], p[2]
        tx = x - self.cam_target[0]
        ty = y - self.cam_target[1]
        tz = z - self.cam_target[2]

        cosY, sinY = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        x1 = tx * cosY - tz * sinY
        z1 = tx * sinY + tz * cosY

        cosP, sinP = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        y2 = ty * cosP - z1 * sinP
        z2 = ty * sinP + z1 * cosP

        eyeZ = z2 + self.cam_dist
        if eyeZ <= 10.0:
            return None

        scale = self.focal / eyeZ
        cx = self.width // 2
        cy = self.height // 2 + 15
        return (cx + x1 * scale, cy - y2 * scale, eyeZ)

    def render_frame(
        self,
        step: int,
        action: str = "FORWARD",
        norm_asymmetry: float = 0.0,
        door_p: float = 0.0,
        health: float = 100.0,
        kills: int = 0,
        is_fire: bool = False,
    ) -> Image.Image:
        """Render full 3D Anatomical Fly with Shotgun frame."""
        pose = self.model.compute_pose(
            step=step,
            action=action,
            norm_asymmetry=norm_asymmetry,
            door_p=door_p,
            health=health,
            kills=kills,
            is_fire=is_fire,
        )

        img = Image.new("RGB", (self.width, self.height), (5, 10, 14))
        draw = ImageDraw.Draw(img)

        # 1. Perspective Floor Grid
        ground_z = -80.0
        grid_lines = []
        for gx in range(-300, 301, 60):
            p1 = self.project((gx, -300.0, ground_z))
            p2 = self.project((gx, 300.0, ground_z))
            if p1 and p2:
                draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(12, 26, 34), width=1)
        for gy in range(-300, 301, 60):
            p1 = self.project((-300.0, gy, ground_z))
            p2 = self.project((300.0, gy, ground_z))
            if p1 and p2:
                draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(12, 26, 34), width=1)

        # 2. Ground Shadow beneath fly body
        shadow_pts = [
            (-60.0, -100.0, ground_z),
            (60.0, -100.0, ground_z),
            (80.0, 120.0, ground_z),
            (-80.0, 120.0, ground_z),
        ]
        s_proj = [self.project(sp) for sp in shadow_pts]
        if all(sp is not None for sp in s_proj):
            draw.polygon([(sp[0], sp[1]) for sp in s_proj], fill=(3, 6, 8))

        # 3. Render Segments with Depth Ordering
        # Primitive components list: (depth_z, render_func)
        render_queue: List[Tuple[float, Any]] = []

        # (a) Legs: L3, R3, L2, R2
        for leg_id, leg in pose.legs.items():
            if leg_id in ("L1", "R1"):
                continue  # prothoracic legs rendered with shotgun
            base = self.model.coxa_bases[leg_id]
            foot = leg.foot_pos
            mid = (base + foot) * 0.5 + np.array([0.0, 0.0, 25.0 if leg.is_stance else 45.0], dtype=np.float32)

            def make_leg_draw(b=base, m=mid, f=foot, lid=leg_id, is_st=leg.is_stance):
                def _draw():
                    pb = self.project(b)
                    pm = self.project(m)
                    pf = self.project(f)
                    if pb and pm and pf:
                        col = (180, 125, 60) if is_st else (220, 160, 80)
                        draw.line([(pb[0], pb[1]), (pm[0], pm[1])], fill=col, width=4)
                        draw.line([(pm[0], pm[1]), (pf[0], pf[1])], fill=col, width=3)
                        draw.ellipse([pm[0]-3, pm[1]-3, pm[0]+3, pm[1]+3], fill=(140, 90, 40))
                        # Tarsal claw
                        draw.ellipse([pf[0]-2, pf[1]-2, pf[0]+2, pf[1]+2], fill=(60, 40, 20))
                return _draw

            avg_z = (base[1] + foot[1]) * 0.5
            render_queue.append((float(avg_z), make_leg_draw()))

        # (b) Segmented Abdomen
        # 7 segments extending posteriorly
        for seg in range(7):
            sy = 60.0 + seg * 28.0
            sz = -10.0 - seg * 8.0 - pose.abdomen_curl * 35.0 * (seg / 7.0)
            sw = 52.0 - seg * 5.0
            sh = 38.0 - seg * 3.5

            def make_abd_draw(sy=sy, sz=sz, sw=sw, sh=sh, s_idx=seg):
                def _draw():
                    p_c = self.project((0.0, sy, sz))
                    p_l = self.project((-sw, sy, sz))
                    p_r = self.project((sw, sy, sz))
                    if p_c and p_l and p_r:
                        rx = abs(p_r[0] - p_l[0]) * 0.5
                        ry = rx * (sh / sw)
                        # Alternating chitin bands (dark brown / amber)
                        col = (130 - s_idx * 10, 85 - s_idx * 6, 38) if s_idx % 2 == 0 else (175 - s_idx * 10, 120 - s_idx * 6, 50)
                        draw.ellipse([p_c[0] - rx, p_c[1] - ry, p_c[0] + rx, p_c[1] + ry], fill=col, outline=(50, 30, 15), width=1)
                return _draw

            render_queue.append((float(sy), make_abd_draw()))

        # (c) Wings (Dorsal)
        def draw_wings():
            # Left wing
            lw_pts = [
                (-35.0, 10.0, 25.0),
                (-140.0, 120.0 + pose.left_wing_angle * 60.0, 40.0 + pose.left_wing_angle * 45.0),
                (-160.0, 210.0 + pose.left_wing_angle * 75.0, 35.0 + pose.left_wing_angle * 50.0),
                (-80.0, 190.0, 25.0),
                (-35.0, 30.0, 20.0),
            ]
            lw_proj = [self.project(p) for p in lw_pts]
            if all(p is not None for p in lw_proj):
                draw.polygon([(p[0], p[1]) for p in lw_proj], fill=(70, 115, 145), outline=(130, 190, 230), width=1)
                # Wing veins
                draw.line([(lw_proj[0][0], lw_proj[0][1]), (lw_proj[2][0], lw_proj[2][1])], fill=(160, 220, 255), width=1)
                draw.line([(lw_proj[0][0], lw_proj[0][1]), (lw_proj[3][0], lw_proj[3][1])], fill=(140, 200, 240), width=1)

            # Right wing
            rw_pts = [
                (35.0, 10.0, 25.0),
                (140.0, 120.0 + pose.right_wing_angle * 60.0, 40.0 + pose.right_wing_angle * 45.0),
                (160.0, 210.0 + pose.right_wing_angle * 75.0, 35.0 + pose.right_wing_angle * 50.0),
                (80.0, 190.0, 25.0),
                (35.0, 30.0, 20.0),
            ]
            rw_proj = [self.project(p) for p in rw_pts]
            if all(p is not None for p in rw_proj):
                draw.polygon([(p[0], p[1]) for p in rw_proj], fill=(70, 115, 145), outline=(130, 190, 230), width=1)
                draw.line([(rw_proj[0][0], rw_proj[0][1]), (rw_proj[2][0], rw_proj[2][1])], fill=(160, 220, 255), width=1)
                draw.line([(rw_proj[0][0], rw_proj[0][1]), (rw_proj[3][0], rw_proj[3][1])], fill=(140, 200, 240), width=1)

        render_queue.append((50.0, draw_wings))

        # (d) Thorax Capsule
        def draw_thorax():
            pt_c = self.project((0.0, 0.0, 0.0))
            pt_l = self.project((-58.0, 0.0, 0.0))
            pt_r = self.project((58.0, 0.0, 0.0))
            if pt_c and pt_l and pt_r:
                rx = abs(pt_r[0] - pt_l[0]) * 0.5
                ry = rx * 0.85
                draw.ellipse([pt_c[0] - rx, pt_c[1] - ry, pt_c[0] + rx, pt_c[1] + ry], fill=(165, 115, 55), outline=(90, 55, 25), width=2)
                # Scutellum dorsal plate
                draw.polygon([
                    (pt_c[0], pt_c[1] + ry * 0.7),
                    (pt_c[0] - rx * 0.35, pt_c[1] + ry * 0.1),
                    (pt_c[0] + rx * 0.35, pt_c[1] + ry * 0.1),
                ], fill=(195, 140, 70), outline=(110, 70, 35))

        render_queue.append((0.0, draw_thorax))

        # (e) Head, Red Ommatidial Eyes, and Antennae
        def draw_head():
            hy = -85.0
            hz = 8.0
            ph_c = self.project((0.0, hy, hz))
            ph_l = self.project((-42.0, hy, hz))
            ph_r = self.project((42.0, hy, hz))
            if ph_c and ph_l and ph_r:
                hrx = abs(ph_r[0] - ph_l[0]) * 0.5
                hry = hrx * 0.88
                draw.ellipse([ph_c[0] - hrx, ph_c[1] - hry, ph_c[0] + hrx, ph_c[1] + hry], fill=(150, 95, 45), outline=(75, 40, 20), width=1)

            # Left and Right Large Red Compound Eyes
            eye_flash_add = int(pose.muzzle_flash * 50.0)
            eye_col = (min(255, 210 + eye_flash_add), 25, 30)

            # Left Eye
            pe_l = self.project((-42.0, hy - 8.0, hz + 5.0))
            if pe_l:
                erx = 24.0 * (self.focal / pe_l[2])
                ery = 32.0 * (self.focal / pe_l[2])
                draw.ellipse([pe_l[0] - erx, pe_l[1] - ery, pe_l[0] + erx, pe_l[1] + ery], fill=eye_col, outline=(110, 15, 20), width=1)
                # Specular eye shine
                draw.ellipse([pe_l[0] - erx * 0.4, pe_l[1] - ery * 0.5, pe_l[0], pe_l[1] - ery * 0.2], fill=(255, 140, 150))

            # Right Eye
            pe_r = self.project((42.0, hy - 8.0, hz + 5.0))
            if pe_r:
                erx = 24.0 * (self.focal / pe_r[2])
                ery = 32.0 * (self.focal / pe_r[2])
                draw.ellipse([pe_r[0] - erx, pe_r[1] - ery, pe_r[0] + erx, pe_r[1] + ery], fill=eye_col, outline=(110, 15, 20), width=1)
                draw.ellipse([pe_r[0] + erx * 0.1, pe_r[1] - ery * 0.5, pe_r[0] + erx * 0.5, pe_r[1] - ery * 0.2], fill=(255, 140, 150))

            # Antennae (pointing forward)
            pa_l1 = self.project((-12.0, hy - 32.0, hz - 4.0))
            pa_l2 = self.project((-24.0, hy - 58.0, hz + 8.0))
            if pa_l1 and pa_l2:
                draw.line([(pa_l1[0], pa_l1[1]), (pa_l2[0], pa_l2[1])], fill=(90, 50, 20), width=2)
                # Arista feathering
                draw.line([(pa_l2[0], pa_l2[1]), (pa_l2[0] - 6, pa_l2[1] - 8)], fill=(120, 70, 30), width=1)

            pa_r1 = self.project((12.0, hy - 32.0, hz - 4.0))
            pa_r2 = self.project((24.0, hy - 58.0, hz + 8.0))
            if pa_r1 and pa_r2:
                draw.line([(pa_r1[0], pa_r1[1]), (pa_r2[0], pa_r2[1])], fill=(90, 50, 20), width=2)
                draw.line([(pa_r2[0], pa_r2[1]), (pa_r2[0] + 6, pa_r2[1] - 8)], fill=(120, 70, 30), width=1)

        render_queue.append((-85.0, draw_head))

        # (f) DOOM Shotgun and Prothoracic Foreleg Grasp (L1 and R1)
        def draw_shotgun_and_forelegs():
            gpos = pose.gun_pos
            # Shotgun geometry: Stock (-15), Receiver (-60), Pump (-110), Barrel (-210)
            p_stock = self.project(gpos + np.array([0.0, 35.0, -12.0], dtype=np.float32))
            p_receiver = self.project(gpos + np.array([0.0, -20.0, -6.0], dtype=np.float32))
            p_pump = self.project(gpos + np.array([0.0, -85.0, -2.0], dtype=np.float32))
            p_muzzle = self.project(gpos + np.array([0.0, -185.0, 4.0], dtype=np.float32))

            # Stock & Receiver
            if p_stock and p_receiver:
                draw.line([(p_stock[0], p_stock[1]), (p_receiver[0], p_receiver[1])], fill=(85, 55, 35), width=7)  # wooden stock
            if p_receiver and p_pump:
                draw.line([(p_receiver[0], p_receiver[1]), (p_pump[0], p_pump[1])], fill=(65, 70, 78), width=8)    # steel receiver

            # Pump Fore-end
            if p_pump and p_muzzle:
                # Barrel
                draw.line([(p_pump[0], p_pump[1]), (p_muzzle[0], p_muzzle[1])], fill=(88, 95, 108), width=5)   # dual barrel
                # Fore-end pump (ribbed)
                draw.ellipse([p_pump[0] - 8, p_pump[1] - 6, p_pump[0] + 8, p_pump[1] + 6], fill=(110, 75, 45), outline=(50, 35, 20))

            # Foreleg L1: wraps firmly around pump fore-end
            base_l1 = self.model.coxa_bases["L1"]
            foot_l1 = pose.legs["L1"].foot_pos
            mid_l1 = (base_l1 + foot_l1) * 0.5 + np.array([-18.0, 0.0, 15.0], dtype=np.float32)
            pl1_b = self.project(base_l1)
            pl1_m = self.project(mid_l1)
            pl1_f = self.project(foot_l1)
            if pl1_b and pl1_m and pl1_f:
                draw.line([(pl1_b[0], pl1_b[1]), (pl1_m[0], pl1_m[1])], fill=(200, 140, 70), width=4)
                draw.line([(pl1_m[0], pl1_m[1]), (pl1_f[0], pl1_f[1])], fill=(190, 130, 65), width=3)
                # Tarsal fingers curled around pump
                draw.ellipse([pl1_f[0] - 4, pl1_f[1] - 4, pl1_f[0] + 4, pl1_f[1] + 4], fill=(80, 50, 25))

            # Foreleg R1: trigger grasp OR forward reach for door
            base_r1 = self.model.coxa_bases["R1"]
            foot_r1 = pose.legs["R1"].foot_pos
            mid_r1 = (base_r1 + foot_r1) * 0.5 + np.array([18.0, 0.0, 15.0], dtype=np.float32)
            pr1_b = self.project(base_r1)
            pr1_m = self.project(mid_r1)
            pr1_f = self.project(foot_r1)
            if pr1_b and pr1_m and pr1_f:
                r1_col = (0, 255, 170) if pose.door_reach > 0.4 else (200, 140, 70)
                draw.line([(pr1_b[0], pr1_b[1]), (pr1_m[0], pr1_m[1])], fill=r1_col, width=4)
                draw.line([(pr1_m[0], pr1_m[1]), (pr1_f[0], pr1_f[1])], fill=r1_col, width=3)
                draw.ellipse([pr1_f[0] - 4, pr1_f[1] - 4, pr1_f[0] + 4, pr1_f[1] + 4], fill=(80, 50, 25))

            # Muzzle Flash Emission & Ejected Shell
            if pose.muzzle_flash > 0.08 and p_muzzle:
                mx, my = p_muzzle[0], p_muzzle[1]
                flash_r = int(24.0 * pose.muzzle_flash * (self.focal / p_muzzle[2]))

                # Starburst flash
                flash_poly = [
                    (mx, my - flash_r * 1.8),
                    (mx + flash_r * 0.4, my - flash_r * 0.4),
                    (mx + flash_r * 1.8, my),
                    (mx + flash_r * 0.4, my + flash_r * 0.4),
                    (mx, my + flash_r * 1.8),
                    (mx - flash_r * 0.4, my + flash_r * 0.4),
                    (mx - flash_r * 1.8, my),
                    (mx - flash_r * 0.4, my - flash_r * 0.4),
                ]
                draw.polygon(flash_poly, fill=(255, 250, 180), outline=(255, 140, 0))
                draw.ellipse([mx - flash_r, my - flash_r, mx + flash_r, my + flash_r], fill=(255, 170, 0))
                draw.ellipse([mx - flash_r * 0.5, my - flash_r * 0.5, mx + flash_r * 0.5, my + flash_r * 0.5], fill=(255, 255, 240))

                # Ejected red shotgun shell flying out to right
                shell_pos = gpos + np.array([28.0 + (1.0 - pose.muzzle_flash) * 35.0, -40.0, 15.0 + (1.0 - pose.muzzle_flash) * 20.0], dtype=np.float32)
                ps = self.project(shell_pos)
                if ps:
                    draw.rectangle([ps[0] - 4, ps[1] - 2, ps[0] + 4, ps[1] + 2], fill=(220, 40, 30), outline=(255, 215, 0))

        render_queue.append((-120.0, draw_shotgun_and_forelegs))

        # (g) Execute sorted render queue from back to front (largest depth to smallest)
        render_queue.sort(key=lambda item: item[0], reverse=True)
        for _, func in render_queue:
            func()

        # 4. Perspective HUD Overlay Banner
        draw.rectangle([0, 0, self.width, 38], fill=(8, 18, 24))
        draw.text((14, 11), "3D EMBODIED FLY · DOOM SHOTGUN · TRIPOD GAIT", fill=(0, 229, 255))
        
        status_txt = f"ACTION: {action}   HP: {health:.0f}%   RECOIL: {pose.gun_recoil:.1f}mm"
        if pose.muzzle_flash > 0.1:
            status_txt += "  ★ DISCHARGE"
        elif pose.door_reach > 0.4:
            status_txt += "  ⎔ DOOR ACTUATE"
        draw.text((380, 11), status_txt, fill=(240, 166, 90) if is_fire else (0, 255, 170))

        # Action border
        border_col = (0, 229, 255)
        if is_fire or action == "FIRE":
            border_col = (255, 50, 0)
        elif action == "USE" or pose.door_reach > 0.4:
            border_col = (0, 255, 170)
        elif action.startswith("TURN"):
            border_col = (255, 160, 0)
        draw.rectangle([0, 0, self.width - 1, self.height - 1], outline=border_col, width=2)

        return img
