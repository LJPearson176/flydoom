"""Small local HTTP server exposing native GZDoom telemetry to the Observatory.

The server is intentionally local-only by default. It exposes the latest state
and frame, not an unbounded event log; the experiment artifacts remain the
canonical scientific record.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import numpy as np

from fly_doom.control.controllers import ControlledT4Controller
from fly_doom.doom.gzdoom_target import GZDoomTarget
from fly_doom.doom.macos_gzdoom_bridge import MacOSGZDoomBridge


class LiveState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._telemetry: dict[str, Any] = {
            "connected": False,
            "source": "demo",
            "action": "FORWARD",
            "step": 0,
            "updated_at": None,
            "error": None,
            "neural": {"norm_asymmetry": 0.0, "t4_l_v": -65.0, "t4_r_v": -65.0},
        }

    def publish(self, obs: Any, action: str, neural: dict[str, float], source: str) -> None:
        with self._lock:
            self._frame = np.ascontiguousarray(obs.rgb).copy()
            self._telemetry.update({
                "connected": True,
                "source": source,
                "action": action,
                "step": int(obs.step_count),
                "updated_at": time.time(),
                "error": None,
                "health": obs.health,
                "ammo": obs.ammo,
                "kills": obs.kill_count,
                "damage_dealt": getattr(obs, "damage_dealt", 0.0),
                "neural": {key: float(value) for key, value in neural.items()},
                "native_game_state": obs.info.get("native_game_state") if hasattr(obs, "info") and obs.info else None,
            })

    def fail(self, message: str) -> None:
        with self._lock:
            self._telemetry.update({"connected": False, "error": message, "updated_at": time.time()})

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._telemetry)

    def frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            return None
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install the macOS extra to encode live frames: uv sync --extra macos") from exc
        output = io.BytesIO()
        Image.fromarray(frame, mode="RGB").save(output, format="JPEG", quality=82)
        return output.getvalue()


class LiveRunner(threading.Thread):
    def __init__(
        self,
        state: LiveState,
        launch: bool,
        pid: Optional[int],
        map_name: str,
        iwad: Optional[str],
        max_frames: int = 0,
        refractory_ticks: int = 3,
        door_seeking: bool = True,
    ) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.launch = launch
        self.pid = pid
        self.map_name = map_name
        self.iwad = iwad
        self.max_frames = max_frames
        self.refractory_ticks = refractory_ticks
        self.door_seeking = door_seeking

    def run(self) -> None:
        from fly_doom.control.controllers import DoorSeekingController
        bridge = MacOSGZDoomBridge(target=GZDoomTarget(map_name=self.map_name, iwad=Path(self.iwad) if self.iwad else None))
        base_controller = ControlledT4Controller(width=64, height=64, saccade_refractory_ticks=self.refractory_ticks)
        controller = DoorSeekingController(base_controller, manage_perspective=True) if self.door_seeking else base_controller
        try:
            if self.launch:
                bridge.launch()
            else:
                bridge.attach(pid=self.pid)
            obs = bridge.reset()
            self.state.publish(obs, "NOOP", {}, "native_gzdoom")
            frame = 0
            while self.max_frames <= 0 or frame < self.max_frames:
                action = controller.select_action(obs)
                obs, _, done, _ = bridge.step(action)
                self.state.publish(obs, action.name, controller.get_neural_state(), "native_gzdoom")
                frame += 1
                if done:
                    break
        except Exception as exc:  # Surface bridge/permission errors to the UI.
            self.state.fail(f"{type(exc).__name__}: {exc}")
        finally:
            bridge.close()


class ObservatoryHandler(BaseHTTPRequestHandler):
    state: LiveState
    web_root = Path(__file__).resolve().parents[3] / "web"

    def _headers(self, content_type: str, length: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        if path == "/api/telemetry":
            payload = json.dumps(self.state.snapshot()).encode()
            self._headers("application/json", len(payload))
            self.wfile.write(payload)
            return
        if path == "/api/runs":
            runs_root = Path(__file__).resolve().parents[3] / "runs"
            runs_list = []
            if runs_root.exists():
                for ep_file in sorted(runs_root.glob("*/doom004*/episode.json"), reverse=True):
                    try:
                        meta = json.loads(ep_file.read_text())
                        runs_list.append({
                            "id": str(ep_file.parent.relative_to(runs_root)),
                            "name": ep_file.parent.parent.name,
                            "steps": meta.get("steps", 0),
                            "kills": meta.get("kills", meta.get("player_kills", 0)),
                            "damage": meta.get("damage_dealt", 0),
                        })
                    except Exception:
                        pass
            payload = json.dumps(runs_list[:25]).encode()
            self._headers("application/json", len(payload))
            self.wfile.write(payload)
            return
        if path == "/api/episode":
            from urllib.parse import parse_qs
            query = parse_qs(parsed_url.query)
            ep_id = query.get("id", [None])[0]
            if not ep_id:
                self.send_error(400, "Missing id parameter")
                return
            runs_root = Path(__file__).resolve().parents[3] / "runs"
            ep_dir = (runs_root / ep_id).resolve()
            if runs_root not in ep_dir.parents and ep_dir != runs_root:
                self.send_error(403)
                return
            ep_file = ep_dir / "episode.json"
            parquet_file = ep_dir / "trajectory.parquet"
            if not ep_file.exists():
                self.send_error(404, "Episode not found")
                return
            meta = json.loads(ep_file.read_text())
            if parquet_file.exists():
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(parquet_file)
                    meta["trajectory"] = table.to_pylist()
                except Exception:
                    pass
            payload = json.dumps(meta).encode()
            self._headers("application/json", len(payload))
            self.wfile.write(payload)
            return
        if path == "/api/frame":
            try:
                payload = self.state.frame_jpeg()
            except RuntimeError as exc:
                self.send_error(503, str(exc))
                return
            if payload is None:
                self.send_error(404, "No live frame yet")
                return
            self._headers("image/jpeg", len(payload))
            self.wfile.write(payload)
            return

        relative = unquote(path.lstrip("/")) or "index.html"
        candidate = (self.web_root / relative).resolve()
        if self.web_root not in candidate.parents and candidate != self.web_root:
            self.send_error(403)
            return
        if not candidate.is_file():
            self.send_error(404)
            return
        content_type = "text/html; charset=utf-8" if candidate.suffix == ".html" else "text/plain; charset=utf-8"
        if candidate.suffix == ".css":
            content_type = "text/css"
        elif candidate.suffix == ".js":
            content_type = "text/javascript"
        payload = candidate.read_bytes()
        self._headers(content_type, len(payload))
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def make_server(host: str = "127.0.0.1", port: int = 4173, state: Optional[LiveState] = None) -> ThreadingHTTPServer:
    state = state or LiveState()
    handler = type("BoundObservatoryHandler", (ObservatoryHandler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launch", action="store_true", help="launch and control GZDoom")
    mode.add_argument("--attach", action="store_true", help="attach to an already-running GZDoom window")
    mode.add_argument("--attach-pid", type=int, help="attach to an existing GZDoom PID")
    parser.add_argument("--map", default="E1M1")
    parser.add_argument("--iwad", help="path to DOOM1.WAD or another IWAD")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--frames", type=int, default=0, help="stop native control after N frames; 0 means continuous")
    parser.add_argument("--refractory-ticks", type=int, default=3, help="post-saccadic suppression ticks to prevent spin loops")
    parser.add_argument("--door-seeking", action="store_true", default=True, help="enable native stalled-door and perspective policy")
    args = parser.parse_args()

    state = LiveState()
    should_run = args.launch or args.attach or (args.attach_pid is not None)
    if should_run:
        LiveRunner(
            state,
            launch=args.launch,
            pid=args.attach_pid,
            map_name=args.map,
            iwad=args.iwad,
            max_frames=args.frames,
            refractory_ticks=args.refractory_ticks,
            door_seeking=args.door_seeking,
        ).start()
    server = make_server(args.host, args.port, state)
    print(f"Observatory: http://{args.host}:{args.port}/")
    if not should_run:
        print("Demo telemetry only. Use --launch, --attach, or --attach-pid for native GZDoom.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
