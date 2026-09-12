from pathlib import Path

import numpy as np

from fly_doom.doom.gzdoom_target import GZDoomTarget
from fly_doom.doom.interface import DoomAction, DoomObservation
from fly_doom.doom.macos_gzdoom_bridge import GZDoomWindow
from fly_doom.doom.native_benchmark import NativeCondition, NativeGZDoomBenchmark
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


class FakeBridge:
    def __init__(self, target: GZDoomTarget):
        self.target = target
        self.window = GZDoomWindow(7, 9, "gzdoom", 64, 64)
        self.step_count = 0

    def launch(self):
        return self.window

    def reset(self, seed=None):
        return DoomObservation(np.ones((64, 64, 3), dtype=np.uint8), np.zeros((64, 64), dtype=np.float32))

    def step(self, action):
        self.step_count += 1
        obs = DoomObservation(np.ones((64, 64, 3), dtype=np.uint8), np.zeros((64, 64), dtype=np.float32), step_count=self.step_count, info={"window_id": 7})
        return obs, 0.0, self.step_count >= 2, {}

    def close(self):
        return None


def test_native_benchmark_writes_episode_artifacts(tmp_path: Path):
    condition = NativeCondition("test_model", CompartmentModelType.MODEL_A, saccade_refractory_ticks=3)
    benchmark = NativeGZDoomBenchmark(tmp_path, max_steps=4, bridge_factory=FakeBridge)
    result = benchmark.run_episode(condition, seed=1001)
    episode_dir = tmp_path / result.episode_id
    assert result.termination_reason == "environment_done"
    assert result.steps == 2
    assert (episode_dir / "episode.json").exists()
    assert (episode_dir / "provenance.json").exists()
    assert (episode_dir / "trajectory.parquet").exists()
    assert result.environment_metadata["deterministic_replay_claim"] is False
    assert result.stability["turning_fraction"] is not None


def test_native_benchmark_shareware_iwad_injection(tmp_path: Path):
    import struct
    # Create a minimal DOOM1.WAD
    lump = b"DEMO_DATA"
    header = struct.pack("<4sII", b"IWAD", 1, 12 + len(lump))
    dir_entry = struct.pack("<II8s", 12, len(lump), b"DEMO\x00\x00\x00\x00")
    source_iwad = tmp_path / "DOOM1.WAD"
    source_iwad.write_bytes(header + lump + dir_entry)

    captured_targets = []

    class TargetCapturingBridge(FakeBridge):
        def __init__(self, target: GZDoomTarget):
            super().__init__(target)
            captured_targets.append(target)

    benchmark = NativeGZDoomBenchmark(
        tmp_path / "runs",
        iwad=source_iwad,
        max_steps=2,
        bridge_factory=TargetCapturingBridge,
    )
    condition = NativeCondition("test_shareware", CompartmentModelType.MODEL_A)
    result = benchmark.run_episode(condition, seed=1001)

    assert len(captured_targets) == 1
    target = captured_targets[0]
    # Telemetry PK3 must NOT be set for shareware (would cause fatal -file error)
    assert target.telemetry_pk3 is None
    # Target IWAD must be the instrumented WAD
    assert target.iwad == tmp_path / "runs" / "DOOM1_INSTRUMENTED.WAD"
    assert target.iwad.exists()
    assert "-file" not in target.command()
    assert "+logfile" in target.command()
    assert result.environment_metadata["original_iwad"] == str(source_iwad)
    assert result.environment_metadata["instrumented_iwad"] == str(target.iwad)


class TelemetryFakeBridge(FakeBridge):
    def step(self, action):
        self.step_count += 1
        native_state = {
            "game_tick": self.step_count,
            "x": 100.0 + self.step_count * 10.0,
            "y": 200.0,
            "z": 0.0,
            "angle_deg": 90.0,
            "pitch_deg": 0.0,
            "vx": 10.0,
            "vy": 0.0,
            "vz": 0.0,
            "health": 100.0 if self.step_count < 3 else 0.0,
            "ammo": 50,
            "kills": 2,
            "player_kills": 1,
            "friendly_fire_kills": 1,
            "damage_dealt": 25.0,
            "target_x": 2272.0,
            "target_y": -2432.0,
            "target_z": 0.0,
            "target_health": 20,
            "target_visible": True,
            "speed": 10.0,
        }
        obs = DoomObservation(
            np.ones((64, 64, 3), dtype=np.uint8),
            np.zeros((64, 64), dtype=np.float32),
            step_count=self.step_count,
            health=native_state["health"],
            ammo=native_state["ammo"],
            kill_count=native_state["player_kills"],
            damage_dealt=native_state["damage_dealt"],
            x=native_state["x"],
            y=native_state["y"],
            info={
                "native_game_state": native_state,
                "native_game_state_available": True,
            },
        )
        return obs, 0.0, self.step_count >= 3, obs.info


def test_native_benchmark_populates_native_state_and_survival(tmp_path: Path):
    benchmark = NativeGZDoomBenchmark(
        tmp_path / "runs",
        max_steps=5,
        bridge_factory=TelemetryFakeBridge,
    )
    condition = NativeCondition("test_telemetry", CompartmentModelType.MODEL_A)
    result = benchmark.run_episode(condition, seed=1001)

    assert result.navigation_status == "native_game_state_available"
    assert result.distance_traveled is not None
    # 3 steps: step 1 x=110, step 2 x=120, step 3 x=130 -> distance = 20.0
    assert abs(result.distance_traveled - 20.0) < 1e-4
    # Health dropped to 0 at step 2 (0-indexed tick 2, which is step 3)
    assert result.survival == 2
    assert result.kills == 1
    assert result.player_kills == 1
    assert result.friendly_fire_kills == 1
    assert result.damage_dealt == 25.0

    # Verify trajectory tick fields
    tick = result.trajectory[-1]
    assert tick.player_kills == 1
    assert tick.friendly_fire_kills == 1
    assert tick.damage_dealt == 25.0
    assert tick.target_x == 2272.0
    assert tick.target_y == -2432.0
    assert tick.target_health == 20
    assert tick.target_visible is True

