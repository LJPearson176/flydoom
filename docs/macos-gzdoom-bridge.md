# Native GZDoom bridge

FlyDoom can connect the existing T4 controller to a local GZDoom window through
the macOS bridge in `src/fly_doom/doom/macos_gzdoom_bridge.py`.

Install the optional Quartz binding:

```bash
uv sync --extra macos
```

macOS requires:

- Screen Recording permission for the Python/Terminal host, so CoreGraphics can capture the GZDoom window.
- Accessibility permission for the Python/Terminal host, so Quartz can post keyboard events to GZDoom.

Run against a new GZDoom process:

```bash
uv run --extra macos python scripts/run_gzdoom_fly.py --launch --map E1M1 --frames 600
```

Or attach to an existing process:

```bash
uv run --extra macos python scripts/run_gzdoom_fly.py --attach-pid 12345
```

To expose the live frame and telemetry in the Observatory UI, start the local
server instead. The browser view is read-only; the runner thread is the only
component that sends actions to GZDoom.

```bash
uv run --extra macos python -m fly_doom.observatory.live_server --launch --map E1M1
open http://127.0.0.1:4173/
```

If GZDoom is not configured with an IWAD, pass it explicitly:

```bash
uv run --extra macos python -m fly_doom.observatory.live_server \
  --launch --map E1M1 \
  --iwad "/Users/ljp176/Library/Application Support/GZDoom/DOOM1.WAD"
```

For a UI-only preview with no native process or input side effects:

```bash
uv run --extra macos python -m fly_doom.observatory.live_server
```

DOOM-004 is a separate bounded benchmark runner. It launches one fresh native
episode per condition and writes `episode.json`, `provenance.json`, and
`trajectory.parquet` under the output directory:

```bash
uv run --extra macos python scripts/run_doom004_native.py \
  --iwad "/Users/ljp176/Library/Application Support/GZDoom/DOOM1.WAD" \
  --max-steps 1000 --seed 1001
```

For the engineering ablation inspired by saccadic suppression:

```bash
uv run --extra macos python scripts/run_doom004_native.py \
  --iwad "/Users/ljp176/Library/Application Support/GZDoom/DOOM1.WAD" \
  --suppression-ablation --max-steps 1000
```

DOOM-004C crosses the neural axis with the candidate control axis:

```bash
uv run --extra macos python scripts/run_doom004c_native.py \
  --iwad "/Users/ljp176/Library/Application Support/GZDoom/DOOM1.WAD" \
  --max-steps 1000 --seed 1001
```

It runs Model A/B/C/D at suppression 0/3/5 and writes a matrix summary. Native
game-state telemetry is instrumented per benchmark episode:
- For shareware `DOOM1.WAD`, which prohibits the `-file` parameter in GZDoom,
  the runner directly injects `ZSCRIPT` and `MAPINFO` binary lumps into a copied
  `DOOM1_INSTRUMENTED.WAD` passed via `-iwad`.
- For commercial IWADs, the runner supplies `flydoom_telemetry.pk3` via `-file`.
- In both cases, GZDoom is launched with `+logfile`, and the non-blocking
  `GZDoomTelemetryReader` tails structured `FLYDOOM_STATE` records for player
  position, heading, velocity, health, ammo, and kills.

Coordinates and derived distance traveled are recorded only when a valid native
state record is observed. Navigation-to-exit progress, wall collisions, and
damage attribution remain null until dedicated engine instrumentation is added.

The bridge provides RGB capture, discrete `FORWARD`, `TURN_LEFT`, `TURN_RIGHT`,
and `FIRE` key taps, and optional native state in
`DoomObservation.info["native_game_state"]`. Depth, episode reset, exit
navigation, wall-collision classification, and damage attribution remain
explicit adapters rather than being inferred from pixels.
