"""macOS bridge between FlyDoom and a running GZDoom window.

The bridge is intentionally separate from the neural controller. It owns only
process/window lifecycle, CoreGraphics capture, and Quartz key events. macOS
Screen Recording permission is required for capture; Accessibility permission
is required for key injection.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from fly_doom.doom.gzdoom_target import GZDoomTarget
from fly_doom.doom.interface import DoomAction, DoomObservation
from fly_doom.doom.gzdoom_telemetry import GZDoomTelemetryReader, state_dict


KEYCODES = {
    DoomAction.FORWARD: 13,     # W
    DoomAction.TURN_LEFT: 0,    # A
    DoomAction.TURN_RIGHT: 2,   # D
    DoomAction.FIRE: 49,        # Space
    DoomAction.USE: 14,         # E
}


@dataclass(frozen=True)
class GZDoomWindow:
    window_id: int
    pid: int
    owner_name: str
    width: int
    height: int
    x: float = 0.0
    y: float = 0.0


class MacOSGZDoomBridge:
    """A DoomEnvironment-compatible bridge for native GZDoom on macOS."""

    def __init__(
        self,
        target: Optional[GZDoomTarget] = None,
        *,
        process: Any = None,
        quartz: Any = None,
        window_finder: Optional[Callable[[], Optional[GZDoomWindow]]] = None,
        frame_interval_s: float = 1 / 60,
        key_hold_s: float = 0.05,
        launch_timeout_s: float = 10.0,
        resolution: Optional[Tuple[int, int]] = (64, 64),
    ):
        self.target = target or GZDoomTarget()
        self.process = process
        self.quartz = quartz
        self.window_finder = window_finder
        self.frame_interval_s = frame_interval_s
        # A down/up pair posted back-to-back can be lost by GZDoom's input
        # polling. Hold each action long enough to span at least one game tic.
        self.key_hold_s = key_hold_s
        self.launch_timeout_s = launch_timeout_s
        self.resolution = resolution
        self.window: Optional[GZDoomWindow] = None
        self._owns_process = False
        self._step_count = 0
        self._last_rgb: Optional[np.ndarray] = None
        self._last_full_rgb: Optional[np.ndarray] = None
        self._stdout_handle = None
        logfile = getattr(self.target, "telemetry_logfile", None)
        self.telemetry_reader = GZDoomTelemetryReader(logfile) if logfile else None

    @staticmethod
    def _load_quartz() -> Any:
        try:
            import Quartz  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Quartz bridge unavailable. Install pyobjc-framework-Quartz "
                "and grant Screen Recording/Accessibility permissions."
            ) from exc
        return Quartz

    def discover_window(self) -> Optional[GZDoomWindow]:
        """Find the visible, non-menu GZDoom window using CoreGraphics metadata."""
        if self.window_finder is not None:
            return self.window_finder()
        q = self.quartz or self._load_quartz()
        # GZDoom may render on a secondary display whose global origin is
        # outside the active desktop viewport. Include all windows so a move
        # between displays does not silently turn the retinal frame black.
        infos = q.CGWindowListCopyWindowInfo(q.kCGWindowListOptionAll, q.kCGNullWindowID)
        wanted_pid = self.process.pid if self.process is not None else None
        candidates = []
        for info in infos or []:
            owner = str(info.get(q.kCGWindowOwnerName, ""))
            pid = int(info.get(q.kCGWindowOwnerPID, 0))
            layer = int(info.get(q.kCGWindowLayer, 1))
            bounds = info.get(q.kCGWindowBounds, {})
            width = int(bounds.get("Width", 0))
            height = int(bounds.get("Height", 0))
            x = float(bounds.get("X", 0))
            y = float(bounds.get("Y", 0))
            name = str(info.get(q.kCGWindowName, ""))
            if width < 64 or height < 64:
                continue
            # Allow standard windows (0) and fullscreen/game overlays (up to 30)
            if layer < 0 or layer > 30:
                continue
            if (wanted_pid is not None and pid == wanted_pid) or "gzdoom" in owner.lower():
                is_console = "console" in name.lower()
                priority = (0 if is_console else 1, width * height)
                candidates.append((priority, GZDoomWindow(int(info[q.kCGWindowNumber]), pid, owner, width, height, x, y)))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]
        return None

    def attach(self, pid: Optional[int] = None) -> GZDoomWindow:
        """Attach to an existing GZDoom window, optionally filtering by PID."""
        if pid is not None:
            self.process = type("AttachedProcess", (), {"pid": pid})()
        self._activate_process()
        window = self.discover_window()
        if window is None:
            raise RuntimeError("No visible GZDoom window found. Launch GZDoom or grant Screen Recording permission.")
        self.window = window
        self._activate_process()
        return window

    def launch(self) -> GZDoomWindow:
        """Launch the configured target and wait for its window; no input is sent."""
        self.target.validate()
        stdout_target = subprocess.DEVNULL
        telemetry_logfile = getattr(self.target, "telemetry_logfile", None)
        if telemetry_logfile is not None:
            Path(telemetry_logfile).parent.mkdir(parents=True, exist_ok=True)
            if "+logfile" not in self.target.command():
                self._stdout_handle = open(telemetry_logfile, "wb")
                stdout_target = self._stdout_handle
        self.process = subprocess.Popen(self.target.command(), stdout=stdout_target, stderr=subprocess.DEVNULL)
        self._owns_process = True
        self._activate_process()
        deadline = time.monotonic() + self.launch_timeout_s
        while time.monotonic() < deadline:
            window = self.discover_window()
            if window is not None:
                self.window = window
                self._activate_process()
                return window
            time.sleep(0.1)
        self.close()
        raise TimeoutError("GZDoom launched but no visible window appeared before the timeout.")

    def _activate_process(self) -> None:
        """Bring GZDoom to the active Space when AppKit is available."""
        if self.process is None:
            return
        try:
            import AppKit  # type: ignore
            app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(self.process.pid)
            if app is not None:
                app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
        except (ImportError, AttributeError):
            # Activation is helpful but not required for test doubles or
            # environments that expose Quartz without AppKit.
            return

    def capture_rgb(self) -> np.ndarray:
        """Capture the attached GZDoom window as an H×W×3 uint8 RGB array."""
        if self.window is None:
            self.attach()
        q = self.quartz or self._load_quartz()

        def _get_image(win: GZDoomWindow) -> Any:
            img = q.CGWindowListCreateImage(
                q.CGRectNull,
                q.kCGWindowListOptionIncludingWindow,
                win.window_id,
                q.kCGWindowImageBoundsIgnoreFraming,
            )
            if img is None and win.width > 0 and win.height > 0:
                rect = q.CGRectMake(win.x, win.y, win.width, win.height)
                img = q.CGWindowListCreateImage(
                    rect,
                    q.kCGWindowListOptionOnScreenOnly,
                    q.kCGNullWindowID,
                    q.kCGWindowImageDefault,
                )
            return img

        image = _get_image(self.window)

        # If window ID changed (e.g. resolution switch or internal redraw), try re-discovery
        if image is None:
            re_window = self.discover_window()
            if re_window is not None:
                self.window = re_window
                image = _get_image(self.window)

        if image is None:
            # If we have a cached prior frame, return it across transient redraw frames
            if self._last_rgb is not None:
                return self._last_rgb.copy()
            raise RuntimeError("CoreGraphics returned no frame; check Screen Recording permission.")

        width, height = q.CGImageGetWidth(image), q.CGImageGetHeight(image)
        provider = q.CGImageGetDataProvider(image)
        raw = bytes(q.CGDataProviderCopyData(provider))
        row_bytes = q.CGImageGetBytesPerRow(image)
        bitmap = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_bytes // 4, 4)[:, :width, :]
        # CGWindowListCreateImage is normally premultiplied BGRA on macOS.
        rgb = np.ascontiguousarray(bitmap[:, :, [2, 1, 0]])

        # A stale/off-display Metal surface can return a valid all-black image.
        # Re-discover once and recapture before handing black pixels to the fly.
        if not np.any(rgb):
            re_window = self.discover_window()
            if re_window is not None and re_window.window_id != self.window.window_id:
                self.window = re_window
                return self.capture_rgb()

        self._last_full_rgb = rgb.copy()

        # If resolution is specified (e.g. (64, 64)), downsample to match retinal input
        if self.resolution is not None and (rgb.shape[1], rgb.shape[0]) != self.resolution:
            target_w, target_h = self.resolution
            r_idx = np.linspace(0, height - 1, target_h).astype(np.int32)
            c_idx = np.linspace(0, width - 1, target_w).astype(np.int32)
            rgb = np.ascontiguousarray(rgb[np.ix_(r_idx, c_idx)])

        self._last_rgb = rgb.copy()
        return rgb

    @property
    def last_full_rgb(self) -> Optional[np.ndarray]:
        """The unscaled full-resolution RGB capture of the most recent frame."""
        return self._last_full_rgb

    def send_action(self, action: DoomAction | int) -> None:
        """Send a short held keypress to GZDoom's process-specific input queue."""
        if self.window is None:
            self.attach()
        action = DoomAction(int(action))
        if action == DoomAction.NOOP:
            return
        q = self.quartz or self._load_quartz()
        preflight = getattr(q, "CGPreflightPostEventAccess", None)
        if preflight is not None and not preflight():
            raise PermissionError(
                "macOS Accessibility permission is required for FlyDoom keyboard injection. "
                "Enable the active Codex/Python host in System Settings > Privacy & Security > Accessibility, "
                "then restart the host application."
            )
        self._activate_process()
        keycode = KEYCODES[action]
        down = q.CGEventCreateKeyboardEvent(None, keycode, True)
        up = q.CGEventCreateKeyboardEvent(None, keycode, False)
        post_global = getattr(q, "CGEventPost", None)
        hid_tap = getattr(q, "kCGHIDEventTap", None)
        if post_global is not None and hid_tap is not None:
            post_global(hid_tap, down)
        else:
            q.CGEventPostToPid(self.window.pid, down)
        time.sleep(self.key_hold_s)
        if post_global is not None and hid_tap is not None:
            post_global(hid_tap, up)
        else:
            q.CGEventPostToPid(self.window.pid, up)

    def reset(self, seed: Optional[int] = None) -> DoomObservation:
        del seed  # Native GZDoom episode reset is intentionally not implicit.
        self._step_count = 0
        rgb = self.capture_rgb()
        info = self._info()
        state = self.telemetry_reader.latest() if self.telemetry_reader else None
        return DoomObservation(
            rgb=rgb, depth=np.zeros(rgb.shape[:2], dtype=np.float32), step_count=0,
            health=state.health if state else 100.0,
            ammo=state.ammo if state else 50,
            kill_count=state.player_kills if state else 0,
            damage_dealt=state.damage_dealt if state else 0.0,
            x=state.x if state else 0.0,
            y=state.y if state else 0.0,
            angle_rad=np.deg2rad(state.angle_deg) if state else 0.0,
            info=info,
        )

    def step(self, action: DoomAction | int):
        self.send_action(action)
        time.sleep(self.frame_interval_s)
        self._step_count += 1
        rgb = self.capture_rgb()
        info = self._info()
        state = self.telemetry_reader.latest() if self.telemetry_reader else None
        obs = DoomObservation(
            rgb=rgb,
            depth=np.zeros(rgb.shape[:2], dtype=np.float32),
            step_count=self._step_count,
            health=state.health if state else 100.0,
            ammo=state.ammo if state else 50,
            kill_count=state.player_kills if state else 0,
            damage_dealt=state.damage_dealt if state else 0.0,
            x=state.x if state else 0.0,
            y=state.y if state else 0.0,
            angle_rad=np.deg2rad(state.angle_deg) if state else 0.0,
            info=info,
        )
        return obs, 0.0, False, info

    def _info(self) -> Dict[str, Any]:
        state = self.telemetry_reader.latest() if self.telemetry_reader else None
        return {
            "native_gzdoom": True,
            "window_id": self.window.window_id if self.window else None,
            "window_bounds": {
                "x": self.window.x,
                "y": self.window.y,
                "width": self.window.width,
                "height": self.window.height,
            } if self.window else None,
            "capture_dimensions": list(self.resolution) if self.resolution else None,
            "native_game_state": state_dict(state),
            "native_game_state_available": state is not None,
        }

    def close(self) -> None:
        if self._owns_process and self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if self._stdout_handle is not None:
            self._stdout_handle.close()
            self._stdout_handle = None
        self.process = None
        self.window = None
        self._owns_process = False
