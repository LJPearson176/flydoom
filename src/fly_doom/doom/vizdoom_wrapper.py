"""ViZDoom Native Environment Wrapper.

Provides bridge to native ViZDoom if installed on the host system.
Automatically falls back to MockDoomArena if vizdoom is not available.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
import numpy as np

from fly_doom.doom.interface import DoomAction, DoomObservation
from fly_doom.doom.mock_arena import MockDoomArena


class VizDoomEnvironment:
    """Wrapper around native ViZDoom DoomGame instance."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        screen_resolution: Tuple[int, int] = (64, 64),
        max_steps: int = 1000,
        visible: bool = False,
    ):
        self.screen_res = screen_resolution
        self.max_steps = max_steps
        self.visible = visible
        self.game = None

        try:
            import vizdoom as vzd
            self.game = vzd.DoomGame()
            if config_path and os.path.exists(config_path):
                self.game.load_config(config_path)
            else:
                self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
                self.game.set_screen_format(vzd.ScreenFormat.RGB24)
                self.game.set_depth_buffer_enabled(True)
                self.game.set_window_visible(visible)
            self.game.init()
            self._has_native = True
        except ImportError:
            self._has_native = False
            self.mock_env = MockDoomArena(
                width_px=screen_resolution[0],
                height_px=screen_resolution[1],
                max_steps=max_steps,
            )

    def reset(self, seed: Optional[int] = None) -> DoomObservation:
        if not self._has_native or self.game is None:
            return self.mock_env.reset(seed=seed)

        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        state = self.game.get_state()
        rgb = state.screen_buffer
        depth = state.depth_buffer if state.depth_buffer is not None else np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        return DoomObservation(
            rgb=rgb,
            depth=depth,
            health=float(self.game.get_game_variable(self.game.GameVariable.HEALTH)),
            ammo=int(self.game.get_game_variable(self.game.GameVariable.AMMO2)),
            kill_count=int(self.game.get_game_variable(self.game.GameVariable.KILLCOUNT)),
            step_count=0,
            done=self.game.is_episode_finished(),
        )

    def step(self, action: DoomAction | int) -> Tuple[DoomObservation, float, bool, Dict[str, Any]]:
        if not self._has_native or self.game is None:
            return self.mock_env.step(action)

        # Action mapping
        actions = {
            DoomAction.NOOP: [0, 0, 0, 0],
            DoomAction.FORWARD: [1, 0, 0, 0],
            DoomAction.TURN_LEFT: [0, 1, 0, 0],
            DoomAction.TURN_RIGHT: [0, 0, 1, 0],
            DoomAction.FIRE: [0, 0, 0, 1],
        }
        raw_act = actions.get(DoomAction(int(action)), [0, 0, 0, 0])
        reward = float(self.game.make_action(raw_act))
        done = self.game.is_episode_finished()
        if not done:
            state = self.game.get_state()
            rgb = state.screen_buffer
            depth = state.depth_buffer if state.depth_buffer is not None else np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            obs = DoomObservation(
                rgb=rgb,
                depth=depth,
                health=float(self.game.get_game_variable(self.game.GameVariable.HEALTH)),
                ammo=int(self.game.get_game_variable(self.game.GameVariable.AMMO2)),
                kill_count=int(self.game.get_game_variable(self.game.GameVariable.KILLCOUNT)),
                step_count=self.game.get_episode_time(),
                done=False,
            )
        else:
            obs = DoomObservation(
                rgb=np.zeros((self.screen_res[1], self.screen_res[0], 3), dtype=np.uint8),
                depth=np.zeros(self.screen_res[::-1], dtype=np.float32),
                done=True,
            )
        info = {"native_vizdoom": True}
        return obs, reward, done, info

    def close(self) -> None:
        if self._has_native and self.game is not None:
            self.game.close()

    @staticmethod
    def check_vizdoom_available() -> Dict[str, Any]:
        """Check if native ViZDoom is importable and functional on host."""
        try:
            import vizdoom as vzd
            version = getattr(vzd, "__version__", "unknown")
            return {
                "available": True,
                "version": version,
                "status": "NATIVE_VIZDOOM_READY",
            }
        except ImportError as e:
            return {
                "available": False,
                "version": None,
                "error": str(e),
                "status": "NOT_INSTALLED",
                "recommendation": "Install via 'uv add vizdoom' or platform wheel when native build tools are present.",
            }
