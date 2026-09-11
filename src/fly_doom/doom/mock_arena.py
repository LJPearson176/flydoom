"""Pure Python / NumPy E1M1 Raycaster & Combat Simulator for Headless DOOM-001.

Provides a fully deterministic, self-contained Doom-style 2.5D visual arena:
  - Textured / shaded 2D map with corridors, walls, and pillars.
  - Moving enemies (Imps) with motion kinematics that produce authentic optical flow.
  - Perspective raycasting rendering (64x64 RGB + Depth).
  - Combat and navigation dynamics: forward movement, turning, projectile firing, imp damage/kills.
  - Health decrement when attacked, ammo consumption, navigation distance accumulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from fly_doom.core.provenance import Provenance
from fly_doom.doom.interface import DoomAction, DoomObservation


@dataclass
class ImpEntity:
    """Enemy demon roaming the corridor and attacking the agent."""

    x: float
    y: float
    vx: float
    vy: float
    health: float = 60.0
    is_alive: bool = True
    size: float = 0.4
    color_rgb: Tuple[int, int, int] = (165, 42, 42)  # Brown imp skin


class MockDoomArena:
    """High-performance 2.5D raycaster corridor simulating ViZDoom E1M1 visual arena."""

    def __init__(
        self,
        width_px: int = 64,
        height_px: int = 64,
        fov_deg: float = 90.0,
        max_steps: int = 1000,
        num_imps: int = 4,
    ):
        self.width_px = width_px
        self.height_px = height_px
        self.fov_rad = math.radians(fov_deg)
        self.max_steps = max_steps
        self.num_imps_init = num_imps
        self.provenance = Provenance(
            tier="engineering_scaffold",
            source="MockDoomArena_Raycaster_v1.0",
            confidence=0.80,
            rationale="Pure-Python/NumPy 2.5D perspective raycaster arena providing deterministic visual stimulus and combat mechanics for CI and headless testing",
        )

        # E1M1 Arena Grid: 0 = floor, 1 = tech wall, 2 = stone wall, 3 = exit gate
        self.map_grid = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.int32)
        self.map_h, self.map_w = self.map_grid.shape

        # Agent state
        self.agent_x = 2.5
        self.agent_y = 2.5
        self.agent_angle = 0.0  # radians, 0 = facing +x (east)
        self.health = 100.0
        self.ammo = 50
        self.kills = 0
        self.damage_dealt = 0.0
        self.step_count = 0
        self.distance_traveled = 0.0
        self.imps: List[ImpEntity] = []
        self.rng = np.random.RandomState(42)

    def reset(self, seed: Optional[int] = None) -> DoomObservation:
        """Reset arena and spawn imps."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agent_x = 2.5
        self.agent_y = 2.5
        self.agent_angle = 0.0
        self.health = 100.0
        self.ammo = 50
        self.kills = 0
        self.damage_dealt = 0.0
        self.step_count = 0
        self.distance_traveled = 0.0

        # Spawn imps in open areas
        self.imps = []
        spawn_locs = [
            (6.5, 3.5, 0.04, 0.02),
            (10.5, 3.5, -0.03, 0.03),
            (8.5, 7.5, 0.02, -0.04),
            (12.5, 8.5, -0.04, -0.02),
        ]
        for sx, sy, vx, vy in spawn_locs[:self.num_imps_init]:
            self.imps.append(ImpEntity(x=sx, y=sy, vx=vx, vy=vy))

        return self._get_obs(done=False)

    def step(self, action: DoomAction | int) -> Tuple[DoomObservation, float, bool, Dict[str, Any]]:
        """Advance agent and imps, check collisions, fire weapons, compute rewards."""
        self.step_count += 1
        reward = 0.0
        action_int = int(action)

        move_speed = 0.15
        turn_speed = 0.12  # ~6.8 degrees per step

        prev_x, prev_y = self.agent_x, self.agent_y

        # Execute agent action
        if action_int == DoomAction.FORWARD:
            nx = self.agent_x + math.cos(self.agent_angle) * move_speed
            ny = self.agent_y + math.sin(self.agent_angle) * move_speed
            if not self._is_wall(nx, ny):
                self.agent_x = nx
                self.agent_y = ny
                dist = math.hypot(self.agent_x - prev_x, self.agent_y - prev_y)
                self.distance_traveled += dist
                reward += dist * 0.5  # Exploration reward

        elif action_int == DoomAction.TURN_LEFT:
            self.agent_angle -= turn_speed

        elif action_int == DoomAction.TURN_RIGHT:
            self.agent_angle += turn_speed

        elif action_int == DoomAction.FIRE:
            if self.ammo > 0:
                self.ammo -= 1
                # Check hitscan line-of-sight to imps within conical targeting zone
                hit_damage = self._perform_hitscan()
                if hit_damage > 0:
                    self.damage_dealt += hit_damage
                    reward += hit_damage * 1.0

        # Keep angle in [-pi, pi]
        self.agent_angle = (self.agent_angle + math.pi) % (2.0 * math.pi) - math.pi

        # Update Imp movements and attacks
        for imp in self.imps:
            if not imp.is_alive:
                continue
            # Simple patrol / seek
            dx = self.agent_x - imp.x
            dy = self.agent_y - imp.y
            dist_to_agent = math.hypot(dx, dy)

            if dist_to_agent < 5.0:
                # Seek agent slowly
                imp.vx = (dx / dist_to_agent) * 0.05
                imp.vy = (dy / dist_to_agent) * 0.05
            else:
                # Bounce around patrol velocity
                if self.rng.rand() < 0.05:
                    imp.vx = float(self.rng.uniform(-0.04, 0.04))
                    imp.vy = float(self.rng.uniform(-0.04, 0.04))

            # Move imp with boundary check
            new_ix = imp.x + imp.vx
            new_iy = imp.y + imp.vy
            if not self._is_wall(new_ix, new_iy):
                imp.x = new_ix
                imp.y = new_iy
            else:
                imp.vx = -imp.vx
                imp.vy = -imp.vy

            # Melee attack if adjacent to agent
            if dist_to_agent < 0.8:
                dmg = 2.0
                self.health -= dmg
                reward -= dmg * 0.5

        # Check death or completion
        done = False
        if self.health <= 0.0:
            self.health = 0.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True
        elif self.map_grid[int(self.agent_y), int(self.agent_x)] == 3:
            # Reached exit portal!
            reward += 100.0
            done = True

        obs = self._get_obs(done=done)
        info = {
            "health": self.health,
            "ammo": self.ammo,
            "kills": self.kills,
            "damage_dealt": self.damage_dealt,
            "distance_traveled": self.distance_traveled,
            "step_count": self.step_count,
        }
        return obs, reward, done, info

    def _is_wall(self, x: float, y: float, margin: float = 0.25) -> bool:
        """Check if circle around (x, y) intersects any wall cell."""
        min_gx = int(math.floor(x - margin))
        max_gx = int(math.floor(x + margin))
        min_gy = int(math.floor(y - margin))
        max_gy = int(math.floor(y + margin))

        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                if 0 <= gx < self.map_w and 0 <= gy < self.map_h:
                    if self.map_grid[gy, gx] in (1, 2):
                        return True
                else:
                    return True
        return False

    def _perform_hitscan(self) -> float:
        """Check if pistol hitscan hits an alive imp within crosshair."""
        total_dmg = 0.0
        for imp in self.imps:
            if not imp.is_alive:
                continue
            dx = imp.x - self.agent_x
            dy = imp.y - self.agent_y
            dist = math.hypot(dx, dy)
            if dist > 8.0 or dist < 0.2:
                continue

            target_angle = math.atan2(dy, dx)
            angle_diff = (target_angle - self.agent_angle + math.pi) % (2.0 * math.pi) - math.pi

            # Hitscan hit cone within ±0.20 radians (~11 degrees)
            if abs(angle_diff) < 0.20:
                dmg = 25.0
                imp.health -= dmg
                total_dmg += dmg
                if imp.health <= 0.0:
                    imp.is_alive = False
                    self.kills += 1
                break  # Hit nearest imp
        return total_dmg

    def _render_scene(self) -> Tuple[np.ndarray, np.ndarray]:
        """Raycast perspective view from agent position producing (H, W, 3) RGB and (H, W) depth."""
        rgb = np.zeros((self.height_px, self.width_px, 3), dtype=np.uint8)
        depth = np.full((self.height_px, self.width_px), 20.0, dtype=np.float32)

        # Sky / ceiling (dark slate blue)
        rgb[: self.height_px // 2, :] = [20, 25, 35]
        # Floor (dark industrial grey)
        rgb[self.height_px // 2 :, :] = [45, 45, 50]

        half_fov = self.fov_rad / 2.0
        angles = np.linspace(self.agent_angle - half_fov, self.agent_angle + half_fov, self.width_px)

        wall_distances = np.zeros(self.width_px, dtype=np.float32)

        for col, ray_angle in enumerate(angles):
            cos_a = math.cos(ray_angle)
            sin_a = math.sin(ray_angle)

            # DDA raymarching
            step_size = 0.05
            dist = 0.0
            hit_wall = 0
            while dist < 15.0:
                dist += step_size
                rx = self.agent_x + cos_a * dist
                ry = self.agent_y + sin_a * dist
                gx, gy = int(rx), int(ry)

                if 0 <= gx < self.map_w and 0 <= gy < self.map_h:
                    cell = self.map_grid[gy, gx]
                    if cell in (1, 2, 3):
                        hit_wall = cell
                        break
                else:
                    hit_wall = 1
                    break

            # Fish-eye correction
            corrected_dist = dist * math.cos(ray_angle - self.agent_angle)
            wall_distances[col] = corrected_dist

            # Wall slice height in screen pixels
            wall_height = int(min(self.height_px, (self.height_px / (corrected_dist + 0.001)) * 1.2))
            top = max(0, (self.height_px - wall_height) // 2)
            bottom = min(self.height_px, (self.height_px + wall_height) // 2)

            # Wall color & shading with distance attenuation
            shade = max(0.2, 1.0 - (corrected_dist / 12.0))
            if hit_wall == 1:
                # Tech wall: cyan/grey tint
                base_c = np.array([80, 110, 130], dtype=np.float32)
            elif hit_wall == 2:
                # Stone wall: reddish brick
                base_c = np.array([120, 70, 60], dtype=np.float32)
            else:
                # Exit portal: glowing green
                base_c = np.array([50, 220, 100], dtype=np.float32)

            c_shaded = (base_c * shade).clip(0, 255).astype(np.uint8)
            rgb[top:bottom, col] = c_shaded
            depth[top:bottom, col] = corrected_dist

        # Render Imps as billboards
        for imp in self.imps:
            if not imp.is_alive:
                continue
            dx = imp.x - self.agent_x
            dy = imp.y - self.agent_y
            dist_to_imp = math.hypot(dx, dy)
            if dist_to_imp < 0.3 or dist_to_imp > 14.0:
                continue

            imp_angle = math.atan2(dy, dx)
            angle_diff = (imp_angle - self.agent_angle + math.pi) % (2.0 * math.pi) - math.pi

            if abs(angle_diff) < half_fov:
                # Screen x column
                screen_x = int(((angle_diff + half_fov) / self.fov_rad) * self.width_px)
                imp_screen_h = int(min(self.height_px, (self.height_px / dist_to_imp) * 0.9))
                imp_screen_w = max(2, int(imp_screen_h * 0.6))

                top = max(0, (self.height_px - imp_screen_h) // 2)
                bottom = min(self.height_px, (self.height_px + imp_screen_h) // 2)
                left = max(0, screen_x - imp_screen_w // 2)
                right = min(self.width_px, screen_x + imp_screen_w // 2)

                # Render if not occluded by wall
                for c in range(left, right):
                    if dist_to_imp < wall_distances[c]:
                        # Imp skin + bright demon eyes
                        imp_c = np.array(imp.color_rgb, dtype=np.uint8)
                        rgb[top:bottom, c] = imp_c
                        depth[top:bottom, c] = dist_to_imp
                        # Glowing eyes in upper third
                        eye_row = top + int((bottom - top) * 0.25)
                        if 0 <= eye_row < self.height_px:
                            rgb[eye_row, c] = [255, 60, 20]

        return rgb, depth

    def _get_obs(self, done: bool) -> DoomObservation:
        """Render frame and bundle into DoomObservation."""
        rgb, depth = self._render_scene()
        return DoomObservation(
            rgb=rgb,
            depth=depth,
            health=float(self.health),
            ammo=int(self.ammo),
            kill_count=int(self.kills),
            damage_dealt=float(self.damage_dealt),
            x=float(self.agent_x),
            y=float(self.agent_y),
            angle_rad=float(self.agent_angle),
            step_count=int(self.step_count),
            done=done,
        )

    def close(self) -> None:
        pass
