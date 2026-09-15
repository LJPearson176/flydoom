"""Heading & Attention Topological Map Renderer.

Renders an allocentric, contour-mapped attention field (Heading & Attention Map)
mapping polar gain hotspots directly derived from the Central Complex Ellipsoid Body (EB)
heading angle, revealing attentional steering priorities across the agent's spatial field.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw


class HeadingAttentionMapRenderer:
    """Renders allocentric topological attention map with contour lines and polar gain hotspots."""

    def __init__(self, width: int = 316, height: int = 270):
        self.width = width
        self.height = height
        self.map_h = height - 28  # reserve bottom 28px for title banner

        # Precompute 256-color scientific attention LUT
        # Progression: Ambient Slate -> Indigo -> Purple -> Cyan -> Emerald -> Yellow -> Crimson
        self.lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.12:
                f = t / 0.12
                self.lut[i] = [int(8 + 12 * f), int(16 + 10 * f), int(22 + 25 * f)]
            elif t < 0.28:
                f = (t - 0.12) / 0.16
                self.lut[i] = [int(20 + 45 * f), int(26 + 10 * f), int(47 + 55 * f)]
            elif t < 0.45:
                f = (t - 0.28) / 0.17
                self.lut[i] = [int(65 - 55 * f), int(36 + 115 * f), int(102 + 75 * f)]
            elif t < 0.62:
                f = (t - 0.45) / 0.17
                self.lut[i] = [int(10 + 60 * f), int(151 + 65 * f), int(177 - 85 * f)]
            elif t < 0.78:
                f = (t - 0.62) / 0.16
                self.lut[i] = [int(70 + 175 * f), int(216 + 25 * f), int(92 - 75 * f)]
            elif t < 0.90:
                f = (t - 0.78) / 0.12
                self.lut[i] = [245, int(241 - 140 * f), 17]
            else:
                f = (t - 0.90) / 0.10
                self.lut[i] = [int(245 - 35 * f), int(101 - 75 * f), int(17 + 10 * f)]

        # Precompute coordinate grids at 2x resolution for anti-aliasing
        self.scale_factor = 2
        self.grid_w = self.width * self.scale_factor
        self.grid_h = self.map_h * self.scale_factor
        self.xs = np.linspace(-1.3, 1.3, self.grid_w, dtype=np.float32)
        self.ys = np.linspace(1.1, -1.1, self.grid_h, dtype=np.float32)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)

    def render(
        self,
        eb_heading_deg: float = 15.8,
        coherence_r: float = 0.67,
        asymmetry: float = 0.0,
        target_angle_deg: Optional[float] = None,
    ) -> Image.Image:
        """Render complete Heading & Attention Topological Map.

        Args:
            eb_heading_deg: Heading angle from central complex ellipsoid body.
            coherence_r: Circular vector strength R in [0, 1].
            asymmetry: Normalized lobula steering asymmetry in [-1, 1].
            target_angle_deg: Optional target bearing angle for lock-on beacon.

        Returns:
            PIL Image of size (width, height).
        """
        # Convert heading angle to radians
        theta = math.radians(eb_heading_deg)

        # 1. Primary Polar Gain Hotspot (Upper-Right based on +15.8 deg heading)
        r_gain1 = 0.55 * (0.8 + 0.3 * coherence_r)
        gx1 = float(r_gain1 * math.sin(theta) + 0.28)
        gy1 = float(r_gain1 * math.cos(theta) + 0.15)
        blob1 = 1.38 * np.exp(-((self.X - gx1) ** 2 + (self.Y - gy1) ** 2) / (2 * 0.19 ** 2))

        # 2. Secondary Lateral Hotspot (Lower-Right)
        theta2 = theta + math.radians(65.0)
        r_gain2 = 0.68
        gx2 = float(r_gain2 * math.sin(theta2) + 0.20)
        gy2 = float(r_gain2 * math.cos(theta2) - 0.40)
        blob2 = 1.25 * np.exp(-((self.X - gx2) ** 2 + (self.Y - gy2) ** 2) / (2 * 0.20 ** 2))

        # 3. Tertiary Attentional Lobe (Upper-Left)
        theta3 = theta - math.radians(52.0)
        r_gain3 = 0.58
        gx3 = float(r_gain3 * math.sin(theta3) - 0.20)
        gy3 = float(r_gain3 * math.cos(theta3) + 0.12)
        blob3 = 0.95 * np.exp(-((self.X - gx3) ** 2 + (self.Y - gy3) ** 2) / (2 * 0.22 ** 2))

        # 4. Central Agent Saddle Lobe
        blob_mid = 0.72 * np.exp(-((self.X - 0.0) ** 2 + (self.Y - 0.10) ** 2) / (2 * 0.30 ** 2))

        # 5. Far Lateral Wing Lobes
        blob_lw = 0.45 * np.exp(-((self.X - (-0.95)) ** 2 + (self.Y - (-0.08)) ** 2) / (2 * 0.20 ** 2))
        blob_rw = 0.40 * np.exp(-((self.X - 0.85) ** 2 + (self.Y - 0.12) ** 2) / (2 * 0.18 ** 2))
        blob_bot = 0.40 * np.exp(-((self.X - (-0.08)) ** 2 + (self.Y - (-0.62)) ** 2) / (2 * 0.22 ** 2))

        # Superposition Potential Field
        V = blob1 + blob2 + blob3 + blob_mid + blob_lw + blob_rw + blob_bot
        v_norm = np.clip(V / 1.40, 0.0, 1.0)

        # 6. Apply Colormap LUT
        indices = (v_norm * 255).astype(np.uint8)
        rgb_map = self.lut[indices]

        # 7. Compute Topographic Isocontour Lines
        levels = np.linspace(0.08, 1.30, 22)
        contour_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        for lvl in levels:
            diff_x = (V[:, :-1] - lvl) * (V[:, 1:] - lvl) <= 0
            diff_y = (V[:-1, :] - lvl) * (V[1:, :] - lvl) <= 0
            contour_mask[:, :-1] = np.maximum(contour_mask[:, :-1], diff_x.astype(np.float32))
            contour_mask[:-1, :] = np.maximum(contour_mask[:-1, :], diff_y.astype(np.float32))

        # Alpha blend isocontour lines in faint slate-cyan
        contour_color = np.array([40, 85, 105], dtype=np.float32)
        alpha = contour_mask[:, :, None] * 0.85
        rgb_blended = (rgb_map * (1.0 - alpha) + contour_color * alpha).astype(np.uint8)

        # Downsample with Lanczos anti-aliasing
        img = Image.fromarray(rgb_blended, "RGB").resize(
            (self.width, self.map_h), Image.Resampling.LANCZOS
        )

        # 8. Composite Final Card
        card = Image.new("RGB", (self.width, self.height), (7, 16, 22))
        card.paste(img, (0, 0))
        draw = ImageDraw.Draw(card)

        # 9. Center Agent Anchor Dot (Pink / Red)
        cx_px = int((0.0 - self.xs[0]) / (self.xs[-1] - self.xs[0]) * self.width)
        cy_px = int((self.ys[0] - 0.05) / (self.ys[0] - self.ys[-1]) * self.map_h)
        draw.ellipse([cx_px - 4, cy_px - 4, cx_px + 4, cy_px + 4], fill=(255, 60, 100), outline=(255, 180, 200))

        # 10. Polar Gain Hotspot Badges
        gain_str = f"{eb_heading_deg:+.1f}° GAIN"
        badges = [
            (gx1, gy1, f"▲ {gain_str}", 15, -12),
            (gx2, gy2, gain_str, 12, -8),
            (gx3, gy3, gain_str, -88, -8),
        ]

        for bx, by, text, off_x, off_y in badges:
            px = int((bx - self.xs[0]) / (self.xs[-1] - self.xs[0]) * self.width) + off_x
            py = int((self.ys[0] - by) / (self.ys[0] - self.ys[-1]) * self.map_h) + off_y
            px = max(4, min(self.width - 92, px))
            py = max(4, min(self.map_h - 20, py))
            draw.rectangle([px, py, px + 86, py + 16], fill=(5, 12, 16), outline=(15, 35, 45))
            draw.text((px + 5, py + 2), text, fill=(230, 245, 255))

        # 11. Bottom Title Bar
        draw.rectangle(
            [0, self.map_h, self.width - 1, self.height - 1],
            fill=(8, 18, 24),
            outline=(22, 50, 62),
        )
        draw.text(
            (self.width // 2 - 82, self.map_h + 8),
            "HEADING & ATTENTION MAP",
            fill=(0, 229, 255),
        )

        # 12. Outer Slate Frame
        draw.rectangle([0, 0, self.width - 1, self.height - 1], outline=(22, 50, 62), width=1)

        return card
