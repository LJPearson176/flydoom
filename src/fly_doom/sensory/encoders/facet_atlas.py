"""Two-Eye Compound-Eye Facet Atlas: Decoupled display geometry for Drosophila photoreceptors.

Adapted from fly_ocr/docs/retinal-atlas.md.
Cleanly separates input sampling coordinates (calibrated 33x25 regular grid) from
display coordinates (curved left/right eye ovals with 825 hexagonal facets).
Honors biological neural superposition (Langen et al., Cell 2015).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import ImageDraw

from fly_doom.core.provenance import Provenance


@dataclass
class CompoundEyeFacet:
    """A schematic compound-eye display facet pooling converging photoreceptors."""

    facet_id: int
    eye: str  # 'L' or 'R'
    u_orig: float
    v_orig: float
    x: float  # Display coordinate in [-1.0, 1.0] after square-to-oval curvature warp
    y: float  # Display coordinate in [-1.0, 1.0] after square-to-oval curvature warp
    indices: List[int]  # Indices of photoreceptors (out of 3,335) converging onto this facet
    body_ids: List[str]  # MaleCNS Body IDs of converging photoreceptors


class CompoundEyeFacetAtlas:
    """Two-Eye Compound-Eye Facet Atlas with square-to-oval display curvature warp.

    Groups the 3,335 MaleCNS R1-R6 photoreceptors (1,107 Left, 2,228 Right) into
    exactly 825 distinct ommatidial column facets, decoupling display from input sampling.
    """

    def __init__(self, atlas_path: Optional[Union[str, Path]] = None):
        if atlas_path is None:
            # Look in assets
            root = Path(__file__).resolve().parent.parent.parent.parent.parent
            candidate = root / "assets" / "retinal_atlas.json"
            if not candidate.exists():
                candidate = Path("assets/retinal_atlas.json")
            atlas_path = candidate

        self.atlas_path = Path(atlas_path)
        if not self.atlas_path.exists():
            raise FileNotFoundError(f"Retinal atlas not found at {self.atlas_path}")

        with open(self.atlas_path) as f:
            raw = json.load(f)

        self.atlas_id = raw.get("atlas_id", "retinal_atlas_v1")
        self.graph_id = raw.get("graph_id", "malecns_v1")
        self.body_ids: List[str] = raw.get("retina_body_ids", [])
        self.original_uv: np.ndarray = np.asarray(raw.get("original_uv", []), dtype=np.float32)
        self.eye_sides: List[str] = raw.get("eye", [])
        self.num_receptors = len(self.body_ids)

        # Build grouped facets with square-to-oval curvature warp
        self.facets: List[CompoundEyeFacet] = []
        self._build_facets()

        self.provenance = Provenance(
            tier="biological_reconstruction",
            source="MaleCNS_Two_Eye_Facet_Atlas",
            confidence=0.95,
            rationale="Decoupled two-eye compound-eye column atlas with square-to-oval curvature warp (fly_ocr)",
        )

    def _build_facets(self) -> None:
        """Group 3,335 receptors into distinct column facets and compute display warp."""
        grouped: Dict[str, Dict[str, Any]] = {}

        for i in range(self.num_receptors):
            u, v = float(self.original_uv[i, 0]), float(self.original_uv[i, 1])
            eye = self.eye_sides[i]
            body_id = self.body_ids[i]
            key = f"{eye}:{u:.5f}:{v:.5f}"

            if key in grouped:
                grouped[key]["indices"].append(i)
                grouped[key]["body_ids"].append(body_id)
            else:
                # 1. Undo overlapping binocular layout:
                # Left eye spans [0.0, 0.6] -> map to [-1.0, 1.0]
                # Right eye spans [0.4, 1.0] -> map to [-1.0, 1.0]
                if eye == "L":
                    nx = 2.0 * (u / 0.6) - 1.0
                else:
                    nx = 2.0 * ((u - 0.4) / 0.6) - 1.0
                ny = 2.0 * v - 1.0

                # 2. Apply square-to-oval display curvature warp (elliptical disc mapping)
                # x = nx * sqrt(1 - 0.45 * ny^2)
                # y = ny * sqrt(1 - 0.45 * nx^2)
                x_disp = nx * math.sqrt(max(0.0, 1.0 - 0.45 * ny * ny))
                y_disp = ny * math.sqrt(max(0.0, 1.0 - 0.45 * nx * nx))

                grouped[key] = {
                    "eye": eye,
                    "u_orig": u,
                    "v_orig": v,
                    "x": float(x_disp),
                    "y": float(y_disp),
                    "indices": [i],
                    "body_ids": [body_id],
                }

        self.facets = [
            CompoundEyeFacet(
                facet_id=idx,
                eye=data["eye"],
                u_orig=data["u_orig"],
                v_orig=data["v_orig"],
                x=data["x"],
                y=data["y"],
                indices=data["indices"],
                body_ids=data["body_ids"],
            )
            for idx, data in enumerate(grouped.values())
        ]

    def compute_facet_values(self, receptor_values: np.ndarray) -> np.ndarray:
        """Compute aggregated display values across all facets given 3,335 receptor signals."""
        vals = np.asarray(receptor_values, dtype=np.float32)
        if len(vals) != self.num_receptors:
            # Fallback or broadcast if dimension differs
            vals = np.resize(vals, self.num_receptors)

        facet_vals = np.zeros(len(self.facets), dtype=np.float32)
        for f in self.facets:
            facet_vals[f.facet_id] = float(np.mean(vals[f.indices]))
        return facet_vals

    def render_pil(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        w: int,
        h: int,
        values: Optional[np.ndarray] = None,
        mode: str = "copper",
    ) -> None:
        """Render the two curved compound-eye ovals with hexagonal facets onto a PIL draw context.

        Args:
            draw: PIL ImageDraw context.
            x: Top-left X coordinate of atlas bounding box.
            y: Top-left Y coordinate of atlas bounding box.
            w: Width of atlas bounding box.
            h: Height of atlas bounding box.
            values: 3,335 receptor signals or 825 facet values in [0.0, 1.0].
            mode: Color mode ('copper' for sampled luminance, 'amber_cyan' for motion/spikes).
        """
        facet_values: Optional[np.ndarray] = None
        if values is not None:
            if len(values) == self.num_receptors:
                facet_values = self.compute_facet_values(values)
            elif len(values) == len(self.facets):
                facet_values = np.asarray(values, dtype=np.float32)
            else:
                facet_values = np.resize(values, len(self.facets))

        # 1. Background eye ovals (Left and Right)
        for side in ("L", "R"):
            cx = x + w * (0.25 if side == "L" else 0.75)
            cy = y + h * 0.44
            rx = w * 0.225
            ry = h * 0.380
            bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
            draw.ellipse(bbox, fill=(33, 24, 34), outline=(120, 80, 75), width=1)

        # 2. Hexagonal facets
        radius = max(1.5, min(w / 145.0, h / 68.0))
        for f in self.facets:
            cx = x + w * (0.25 if f.eye == "L" else 0.75) + f.x * w * 0.200
            cy = y + h * 0.44 + f.y * h * 0.340

            b = float(facet_values[f.facet_id]) if facet_values is not None else 0.2
            b = max(0.0, min(1.0, b))

            if mode == "copper":
                # Monotone copper luminance scale matching fly_ocr
                cr = int(45 + 210 * b)
                cg = int(24 + 191 * b)
                cb = int(28 + 147 * b)
            elif mode == "amber_cyan":
                # Glowing biological neural illumination
                cr = int(20 + 235 * b)
                cg = int(140 + 115 * b)
                cb = int(200 * (1.0 - b))
            else:
                cr = int(255 * b)
                cg = int(255 * b)
                cb = int(255 * b)

            col = (cr, cg, cb)

            # Draw regular hexagon
            hex_pts = []
            for k in range(6):
                ang = (math.pi / 3.0) * k
                px = cx + radius * math.cos(ang)
                py = cy + radius * math.sin(ang)
                hex_pts.append((px, py))
            draw.polygon(hex_pts, fill=col)

        # 3. Anatomical hemisphere labels
        draw.text((x + int(w * 0.23), y + int(h * 0.85)), "L", fill=(148, 172, 188))
        draw.text((x + int(w * 0.73), y + int(h * 0.85)), "R", fill=(148, 172, 188))

    def render_dual_channel_pil(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        w: int,
        h: int,
        values: Optional[np.ndarray] = None,
        asymmetry: float = 0.0,
        is_motion: bool = False,
    ) -> None:
        """Render Dual-Channel Sensory Decomposition:
        - Left Hemisphere: Low-Resolution Tessellation Mask displaying raw discrete ommatidial
          array with active spatial receptive fields and localized motion cues.
        - Right Hemisphere: Polarization & UV Channel with EMD Field visualizing high-contrast
          polarization/short-wavelength spectral channel overlaid with Elementary Motion Detector
          (EMD) gradient vectors.
        """
        facet_values: Optional[np.ndarray] = None
        if values is not None:
            if len(values) == self.num_receptors:
                facet_values = self.compute_facet_values(values)
            elif len(values) == len(self.facets):
                facet_values = np.asarray(values, dtype=np.float32)
            else:
                facet_values = np.resize(values, len(self.facets))

        # 1. Background eye ovals (Left: Emerald-tinted, Right: Violet-tinted)
        for side in ("L", "R"):
            cx = x + w * (0.26 if side == "L" else 0.74)
            cy = y + h * 0.44
            rx = w * 0.220
            ry = h * 0.360
            fill_col = (18, 28, 22) if side == "L" else (28, 14, 38)
            outl_col = (45, 120, 80) if side == "L" else (150, 50, 180)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill_col, outline=outl_col, width=1)

        radius = max(1.8, min(w / 140.0, h / 66.0))

        # 2. Left Hemisphere: Low-Resolution Tessellation Mask (Discrete array & RF cues)
        for f in self.facets:
            if f.eye != "L":
                continue
            cx = x + w * 0.26 + f.x * w * 0.195
            cy = y + h * 0.44 + f.y * h * 0.320
            b = float(facet_values[f.facet_id]) if facet_values is not None else 0.35
            b = max(0.0, min(1.0, b))

            # Active spatial receptive field illumination (phosphor green palette)
            if b > 0.65:
                col = (int(40 + 190 * b), 255, int(80 + 140 * b))
            elif b > 0.35:
                col = (int(20 + 80 * b), int(120 + 90 * b), int(40 + 60 * b))
            else:
                col = (15, int(40 + 40 * b), 25)

            # Discrete ommatidial facet outline
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=col, outline=(10, 35, 20), width=1)

            # Localized motion cue markers for high-contrast receptive fields
            if b > 0.72:
                draw.ellipse(
                    [cx - radius - 1.5, cy - radius - 1.5, cx + radius + 1.5, cy + radius + 1.5],
                    outline=(0, 255, 180),
                    width=1,
                )

        # 3. Right Hemisphere: Polarization & UV Channel with EMD Field
        for f in self.facets:
            if f.eye != "R":
                continue
            cx = x + w * 0.74 + f.x * w * 0.195
            cy = y + h * 0.44 + f.y * h * 0.320
            b = float(facet_values[f.facet_id]) if facet_values is not None else 0.35
            b = max(0.0, min(1.0, b))

            # High-contrast polarization & short-wavelength UV palette (violet -> magenta -> UV cyan)
            if b > 0.70:
                col = (255, int(80 + 150 * b), 255)
            elif b > 0.40:
                col = (int(160 + 90 * b), int(20 + 60 * b), int(180 + 75 * b))
            else:
                col = (int(35 + 60 * b), int(10 + 25 * b), int(65 + 75 * b))

            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=col, outline=(45, 15, 60), width=1)

            # Overlaid Elementary Motion Detector (EMD) gradient vectors
            if b > 0.45 and f.facet_id % 3 == 0:
                dx = f.x * 7.5 - asymmetry * 8.0
                dy = f.y * 6.0
                draw.line([(cx, cy), (cx + dx, cy + dy)], fill=(255, 235, 60), width=1)
                draw.ellipse([cx + dx - 1, cy + dy - 1, cx + dx + 1, cy + dy + 1], fill=(255, 255, 180))

        # 4. Descriptive Subtitles below ovals
        draw.text((x + int(w * 0.26 - 62), y + int(h * 0.86)), "L: TESSELLATION / RF", fill=(0, 255, 170))
        draw.text((x + int(w * 0.74 - 66), y + int(h * 0.86)), "R: POLAR-UV / EMD FIELD", fill=(240, 140, 255))


    def to_dict(self) -> Dict[str, Any]:
        """Serialize display atlas for web client JSON payload."""
        return {
            "atlas_id": self.atlas_id,
            "graph_id": self.graph_id,
            "num_receptors": self.num_receptors,
            "num_facets": len(self.facets),
            "facets": [
                {
                    "id": f.facet_id,
                    "eye": f.eye,
                    "x": round(f.x, 4),
                    "y": round(f.y, 4),
                    "indices": f.indices,
                }
                for f in self.facets
            ],
        }
