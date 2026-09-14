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
import sys

# Ensure src/ is on sys.path regardless of execution environment
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fly_doom.control.controllers import ControlledT4Controller, DoorSeekingController
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
    parser.add_argument("--no-door-seeking", action="store_false", dest="door_seeking", default=True, help="disable Stage 1 navigation wrapper")
    parser.add_argument("--reservoir", action="store_true", default=False, help="enable Connectome-Constrained Reservoir decoders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iwad_path = Path(args.iwad) if args.iwad else None
    if iwad_path is None:
        inst_wad = Path("runs/doom004_lesions_v1/DOOM1_INSTRUMENTED.WAD")
        if inst_wad.exists():
            iwad_path = inst_wad
        else:
            wads_wad = Path("wads/DOOM1.WAD")
            if wads_wad.exists():
                iwad_path = wads_wad

    log_path = Path("runs/live_gzdoom_fly.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    target = GZDoomTarget(
        map_name=args.map,
        iwad=iwad_path,
        telemetry_logfile=log_path,
    )
    bridge = MacOSGZDoomBridge(target=target, resolution=(64, 64))
    base = ControlledT4Controller(width=64, height=64, saccade_refractory_ticks=args.refractory_ticks)
    if args.door_seeking:
        if args.reservoir:
            print("Equipping DoorSeekingController with Connectome-Constrained Reservoir decoders...")
            controller = DoorSeekingController.create_with_reservoir(base)
        else:
            controller = DoorSeekingController(base)
    else:
        controller = base

    try:
        window = bridge.launch() if args.launch else bridge.attach(pid=args.attach_pid)
        print(f"Attached to {window.owner_name} pid={window.pid} window={window.window_id}")
        obs = bridge.reset()
        for frame in range(args.frames):
            action = controller.select_action(obs)
            obs, _, done, _ = bridge.step(action)
            if frame % 15 == 0 or action.name in ("USE", "FIRE"):
                state = controller.get_neural_state()
                tgt_deg = state.get("target_angle_deg", float("nan"))
                fb_t = state.get("fb_steer_torque", 0.0)
                sez_h = state.get("sez_acid_detected", 0.0)
                exit_sw = state.get("at_exit_switch", 0.0)
                res_d = state.get("reservoir_door_use", 0.0)
                res_f = state.get("reservoir_should_fire", 0.0)
                res_str = f"res=[D:{int(res_d)} F:{int(res_f)}]" if args.reservoir else ""
                native_state = obs.info.get("native_game_state", {}) if obs.info else {}
                pos_str = f"pos=({native_state.get('x', 0.0):.0f},{native_state.get('y', 0.0):.0f})" if native_state else ""
                print(
                    f"frame={frame:04d} action={action.name:10s} {pos_str:18s} kills={obs.kill_count} HP={obs.health:3.0f}% "
                    f"tgt_ang={tgt_deg:+6.1f}° fb_t={fb_t:+.2f} sez={'ACID' if sez_h > 0 else 'SAFE'} {res_str} "
                    f"{'EXIT_SWITCH!' if exit_sw > 0 else ''}"
                )
            if done:
                print(f"Stage completed or episode terminated at frame {frame}!")
                break
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
