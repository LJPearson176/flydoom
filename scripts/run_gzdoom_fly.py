"""Run the current FlyDoom controller against native GZDoom on macOS.

Examples:
  uv run --extra macos python scripts/run_gzdoom_fly.py --launch
  uv run --extra macos python scripts/run_gzdoom_fly.py --attach-pid 12345

The script is intentionally opt-in: it never launches or injects input unless
one of --launch / --attach-pid is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pathlib import Path
import sys

# Ensure src/ is on sys.path regardless of execution environment
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fly_doom.control.controllers import ControlledT4Controller
from fly_doom.doom.gzdoom_target import GZDoomTarget
from fly_doom.doom.macos_gzdoom_bridge import MacOSGZDoomBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true", help="launch /Applications/GZDoom.app")
    mode.add_argument("--attach-pid", type=int, help="attach to an already-running GZDoom PID")
    parser.add_argument("--map", default="E1M1")
    parser.add_argument("--iwad", help="path to DOOM1.WAD or another IWAD")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--refractory-ticks", type=int, default=3, help="post-saccadic suppression ticks to prevent spin loops")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = GZDoomTarget(map_name=args.map, iwad=Path(args.iwad) if args.iwad else None)
    bridge = MacOSGZDoomBridge(target=target)
    controller = ControlledT4Controller(width=64, height=64, saccade_refractory_ticks=args.refractory_ticks)

    try:
        window = bridge.launch() if args.launch else bridge.attach(pid=args.attach_pid)
        print(f"Attached to {window.owner_name} pid={window.pid} window={window.window_id}")
        obs = bridge.reset()
        for frame in range(args.frames):
            action = controller.select_action(obs)
            obs, _, _, _ = bridge.step(action)
            if frame % 30 == 0:
                state = controller.get_neural_state()
                print(f"frame={frame:04d} action={action.name:11s} norm_asym={state.get('norm_asymmetry', 0.0):+.3f}")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
