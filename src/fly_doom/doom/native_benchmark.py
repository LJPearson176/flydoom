"""DOOM-004 native GZDoom episodic benchmark protocol.

This runner keeps native integration separate from the Observatory server. Each
episode owns one GZDoom process, one controller reset, and one sealed artifact
directory. Native-only quantities that are not yet exposed by the bridge are
recorded as unavailable rather than inferred from pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import hashlib
import platform
import plistlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from fly_doom.control.controllers import ControlledT4Controller, DoorSeekingController
from fly_doom.doom.gzdoom_target import GZDoomTarget
from fly_doom.doom.gzdoom_telemetry import inject_telemetry_wad, write_telemetry_pk3
from fly_doom.doom.interface import DoomAction
from fly_doom.doom.macos_gzdoom_bridge import MacOSGZDoomBridge
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def is_shareware_wad(iwad: Optional[Path]) -> bool:
    if iwad is None:
        return False
    name = Path(iwad).name.lower()
    return "doom1" in name or "shareware" in name



@dataclass(frozen=True)
class NativeCondition:
    name: str
    model_type: CompartmentModelType
    knockouts: frozenset[str] = frozenset()
    saccade_refractory_ticks: int = 0
    door_seeking: bool = False

    def build_controller(self) -> ControlledT4Controller:
        controller = ControlledT4Controller(
            model_type=self.model_type,
            synaptic_knockouts=set(self.knockouts),
            custom_name=self.name,
            saccade_refractory_ticks=self.saccade_refractory_ticks,
        )
        return DoorSeekingController(controller) if self.door_seeking else controller


DEFAULT_NATIVE_CONDITIONS = (
    NativeCondition("ModelA_PointLIF", CompartmentModelType.MODEL_A),
    NativeCondition("ModelB_TemporalPointLIF", CompartmentModelType.MODEL_B),
    NativeCondition("ModelC_PassiveTree", CompartmentModelType.MODEL_C),
    NativeCondition("ModelD_ActiveTree", CompartmentModelType.MODEL_D),
    NativeCondition("ModelD_Mi4_KO", CompartmentModelType.MODEL_D, frozenset({"Mi4"})),
    NativeCondition("ModelD_Mi9_KO", CompartmentModelType.MODEL_D, frozenset({"Mi9"})),
    NativeCondition("ModelD_Tm3_KO", CompartmentModelType.MODEL_D, frozenset({"Tm3"})),
    NativeCondition("ModelD_Mi1_KO", CompartmentModelType.MODEL_D, frozenset({"Mi1"})),
)

DEFAULT_SUPPRESSION_ABLATION = tuple(
    NativeCondition(f"ModelD_ActiveTree_Saccade_{ticks}", CompartmentModelType.MODEL_D, saccade_refractory_ticks=ticks)
    for ticks in (0, 3, 5, 10)
)

DEFAULT_LESION_CONDITIONS = (
    NativeCondition("ModelD_ActiveTree_Saccade_3", CompartmentModelType.MODEL_D, saccade_refractory_ticks=3),
    NativeCondition("ModelD_Mi4_KO_Saccade_3", CompartmentModelType.MODEL_D, frozenset({"Mi4"}), saccade_refractory_ticks=3),
    NativeCondition("ModelD_Mi9_KO_Saccade_3", CompartmentModelType.MODEL_D, frozenset({"Mi9"}), saccade_refractory_ticks=3),
    NativeCondition("ModelD_Tm3_KO_Saccade_3", CompartmentModelType.MODEL_D, frozenset({"Tm3"}), saccade_refractory_ticks=3),
    NativeCondition("ModelD_Mi1_KO_Saccade_3", CompartmentModelType.MODEL_D, frozenset({"Mi1"}), saccade_refractory_ticks=3),
)


@dataclass
class NativeTrajectoryTick:
    episode_id: str
    seed: int
    condition: str
    step: int
    timestamp: str
    action: str
    health: float
    ammo: int
    kills: int
    frame_mean: float
    frame_nonzero_fraction: float
    retina_mean: float
    retina_delta: float
    t4_l_v: float
    t4_r_v: float
    norm_asymmetry: float
    linear_velocity: Optional[float]
    angular_velocity: Optional[float]
    capture_valid: bool
    window_id: Optional[int]
    window_bounds: Optional[Dict[str, float]]
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    angle_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    native_state_available: bool = False
    target_angle_deg: Optional[float] = None
    door_candidate: Optional[float] = None
    door_stalled_ticks: Optional[float] = None
    combat_active: Optional[float] = None
    firing_solution_locked: Optional[float] = None
    health_priority_active: Optional[float] = None
    health_delta: Optional[float] = None
    threat_refractory_ticks: Optional[float] = None
    player_kills: Optional[int] = None
    friendly_fire_kills: Optional[int] = None
    damage_dealt: Optional[float] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    target_z: Optional[float] = None
    target_health: Optional[int] = None
    target_visible: Optional[bool] = None
    # Compartment-level activation map. These are computational model states,
    # retained separately from native engine ground truth.
    t4_l_leading_v: Optional[float] = None
    t4_l_central_v: Optional[float] = None
    t4_l_trailing_v: Optional[float] = None
    t4_l_soma_activation: Optional[float] = None
    t4_l_leading_activation: Optional[float] = None
    t4_l_central_activation: Optional[float] = None
    t4_l_trailing_activation: Optional[float] = None
    t4_l_mi1_drive: Optional[float] = None
    t4_l_tm3_drive: Optional[float] = None
    t4_l_mi4_inhibition: Optional[float] = None
    t4_l_mi9_inhibition: Optional[float] = None
    t4_r_leading_v: Optional[float] = None
    t4_r_central_v: Optional[float] = None
    t4_r_trailing_v: Optional[float] = None
    t4_r_soma_activation: Optional[float] = None
    t4_r_leading_activation: Optional[float] = None
    t4_r_central_activation: Optional[float] = None
    t4_r_trailing_activation: Optional[float] = None
    t4_r_mi1_drive: Optional[float] = None
    t4_r_tm3_drive: Optional[float] = None
    t4_r_mi4_inhibition: Optional[float] = None
    t4_r_mi9_inhibition: Optional[float] = None
    retina_left_drive: Optional[float] = None
    retina_right_drive: Optional[float] = None
    retina_center_drive: Optional[float] = None
    t4_l_mi1_input: Optional[float] = None
    t4_l_tm3_input: Optional[float] = None
    t4_l_mi4_input: Optional[float] = None
    t4_l_mi9_input: Optional[float] = None
    t4_r_mi1_input: Optional[float] = None
    t4_r_tm3_input: Optional[float] = None
    t4_r_mi4_input: Optional[float] = None
    t4_r_mi9_input: Optional[float] = None


@dataclass
class NativeEpisodeResult:
    episode_id: str
    seed: int
    condition: str
    map_name: str
    iwad: Optional[str]
    start_timestamp: str
    end_timestamp: str
    termination_reason: str
    steps: int
    action_counts: Dict[str, int]
    health_remaining: float
    ammo_remaining: int
    kills: int
    damage_dealt: Optional[float]
    navigation_status: str
    stability: Dict[str, Optional[float]]
    environment_metadata: Dict[str, Any]
    trajectory: List[NativeTrajectoryTick] = field(default_factory=list)
    distance_traveled: Optional[float] = None
    wall_collisions: Optional[int] = None
    navigation_progress: Optional[float] = None
    survival: Optional[int] = None
    door_seeking: bool = False
    player_kills: int = 0
    friendly_fire_kills: int = 0


class NativeGZDoomBenchmark:
    """Run bounded native episodes with strict process/controller ownership."""

    def __init__(
        self,
        output_dir: Path,
        *,
        map_name: str = "E1M1",
        iwad: Optional[Path] = None,
        max_steps: int = 1000,
        bridge_factory: Callable[[GZDoomTarget], MacOSGZDoomBridge] = MacOSGZDoomBridge,
    ):
        # GZDoom is launched from outside the repository on macOS. Resolve
        # telemetry/artifact paths so +logfile and -iwad are not interpreted
        # relative to the app's working directory.
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.map_name = map_name
        self.iwad = iwad
        self.max_steps = max_steps
        self.bridge_factory = bridge_factory

    def run_episode(self, condition: NativeCondition, seed: int, episode_index: int = 0) -> NativeEpisodeResult:
        episode_id = f"doom004-native-{condition.name}-{seed}-{episode_index:03d}"
        started = datetime.now(timezone.utc).isoformat()
        telemetry_dir = self.output_dir / episode_id
        telemetry_log = telemetry_dir / "gzdoom.log"

        if is_shareware_wad(self.iwad):
            instrumented_iwad = self.output_dir / "DOOM1_INSTRUMENTED.WAD"
            if not instrumented_iwad.exists() and self.iwad is not None and Path(self.iwad).exists():
                inject_telemetry_wad(self.iwad, instrumented_iwad)
            target = GZDoomTarget(
                app_path=Path("/Applications/GZDoom.app"),
                iwad=instrumented_iwad if instrumented_iwad.exists() else self.iwad,
                map_name=self.map_name,
                telemetry_pk3=None,
                telemetry_logfile=telemetry_log,
            )
        else:
            telemetry_pk3 = write_telemetry_pk3(telemetry_dir / "flydoom_telemetry.pk3")
            target = GZDoomTarget(
                app_path=Path("/Applications/GZDoom.app"),
                iwad=self.iwad,
                map_name=self.map_name,
                telemetry_pk3=telemetry_pk3,
                telemetry_logfile=telemetry_log,
            )
        bridge = self.bridge_factory(target)
        controller = condition.build_controller()
        actions = {action.name: 0 for action in DoomAction}
        trajectory: List[NativeTrajectoryTick] = []
        termination = "timeout"
        obs = None

        try:
            bridge.launch()
            obs = bridge.reset(seed=seed)
            for step in range(self.max_steps):
                action = controller.select_action(obs)
                actions[action.name] += 1
                neural = controller.get_neural_state()
                next_obs, _, done, _ = bridge.step(action)
                frame = np.asarray(next_obs.rgb)
                native_state = next_obs.info.get("native_game_state") or {}
                trajectory.append(NativeTrajectoryTick(
                    episode_id=episode_id,
                    seed=seed,
                    condition=condition.name,
                    step=step,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    action=action.name,
                    health=float(next_obs.health),
                    ammo=int(next_obs.ammo),
                    kills=int(next_obs.kill_count),
                    frame_mean=float(frame.mean() / 255.0),
                    frame_nonzero_fraction=float(np.count_nonzero(frame) / max(1, frame.size)),
                    retina_mean=float(neural.get("retina_mean", 0.0)),
                    retina_delta=float(neural.get("retina_delta", 0.0)),
                    t4_l_v=float(neural.get("t4_l_v", -65.0)),
                    t4_r_v=float(neural.get("t4_r_v", -65.0)),
                    norm_asymmetry=float(neural.get("norm_asymmetry", 0.0)),
                    linear_velocity=(float(native_state["speed"]) if "speed" in native_state else None),
                    angular_velocity=None,
                    capture_valid=bool(frame.size and np.any(frame)),
                    window_id=next_obs.info.get("window_id"),
                    window_bounds=next_obs.info.get("window_bounds"),
                    x=float(native_state["x"]) if "x" in native_state else None,
                    y=float(native_state["y"]) if "y" in native_state else None,
                    z=float(native_state["z"]) if "z" in native_state else None,
                    angle_deg=float(native_state["angle_deg"]) if "angle_deg" in native_state else None,
                    pitch_deg=float(native_state["pitch_deg"]) if "pitch_deg" in native_state else None,
                    native_state_available=bool(next_obs.info.get("native_game_state_available")),
                    target_angle_deg=(float(neural["target_angle_deg"]) if "target_angle_deg" in neural and not np.isnan(neural["target_angle_deg"]) else None),
                    door_candidate=(float(neural["door_candidate"]) if "door_candidate" in neural else None),
                    door_stalled_ticks=(float(neural["door_stalled_ticks"]) if "door_stalled_ticks" in neural else None),
                    combat_active=(float(neural["combat_active"]) if "combat_active" in neural else None),
                    firing_solution_locked=(float(neural["firing_solution_locked"]) if "firing_solution_locked" in neural else None),
                    health_priority_active=(float(neural["health_priority_active"]) if "health_priority_active" in neural else None),
                    health_delta=(float(neural["health_delta"]) if "health_delta" in neural else None),
                    threat_refractory_ticks=(float(neural["threat_refractory_ticks"]) if "threat_refractory_ticks" in neural else None),
                    player_kills=int(native_state["player_kills"]) if "player_kills" in native_state else None,
                    friendly_fire_kills=int(native_state["friendly_fire_kills"]) if "friendly_fire_kills" in native_state else None,
                    damage_dealt=float(native_state["damage_dealt"]) if "damage_dealt" in native_state else None,
                    target_x=float(native_state["target_x"]) if native_state.get("target_x") is not None else None,
                    target_y=float(native_state["target_y"]) if native_state.get("target_y") is not None else None,
                    target_z=float(native_state["target_z"]) if native_state.get("target_z") is not None else None,
                    target_health=int(native_state["target_health"]) if native_state.get("target_health") is not None else None,
                    target_visible=bool(native_state["target_visible"]) if "target_visible" in native_state else None,
                    **{
                        key: (float(neural[key]) if key in neural else None)
                        for key in (
                            "t4_l_leading_v", "t4_l_central_v", "t4_l_trailing_v",
                            "t4_l_soma_activation", "t4_l_leading_activation",
                            "t4_l_central_activation", "t4_l_trailing_activation",
                            "t4_l_mi1_drive", "t4_l_tm3_drive", "t4_l_mi4_inhibition",
                            "t4_l_mi9_inhibition", "t4_r_leading_v", "t4_r_central_v",
                            "t4_r_trailing_v", "t4_r_soma_activation", "t4_r_leading_activation",
                            "t4_r_central_activation", "t4_r_trailing_activation",
                            "t4_r_mi1_drive", "t4_r_tm3_drive", "t4_r_mi4_inhibition",
                            "t4_r_mi9_inhibition", "retina_left_drive", "retina_right_drive",
                            "retina_center_drive", "t4_l_mi1_input", "t4_l_tm3_input",
                            "t4_l_mi4_input", "t4_l_mi9_input", "t4_r_mi1_input",
                            "t4_r_tm3_input", "t4_r_mi4_input", "t4_r_mi9_input",
                        )
                    },
                ))
                obs = next_obs
                if done:
                    termination = "environment_done"
                    break
        except KeyboardInterrupt:
            termination = "interrupted"
            raise
        except Exception:
            termination = "bridge_error"
            raise
        finally:
            bridge.close()

        turn_steps = sum(value for key, value in actions.items() if key in {"TURN_LEFT", "TURN_RIGHT"})
        sign_reversals = 0
        previous_turn = None
        for tick in trajectory:
            if tick.action not in {"TURN_LEFT", "TURN_RIGHT"}:
                continue
            if previous_turn is not None and tick.action != previous_turn:
                sign_reversals += 1
            previous_turn = tick.action

        # Combat targeting stability and optic flow tracking metrics
        combat_ticks = [t for t in trajectory if getattr(t, "combat_active", 0.0) == 1.0]
        angle_errors = []
        for t in combat_ticks:
            if t.angle_deg is not None and t.target_angle_deg is not None and not np.isnan(t.target_angle_deg):
                diff = (t.target_angle_deg - t.angle_deg + 180.0) % 360.0 - 180.0
                angle_errors.append(abs(diff))

        asymmetries = [t.norm_asymmetry for t in trajectory if t.norm_asymmetry is not None]
        asym_arr = np.array(asymmetries, dtype=np.float64) if asymmetries else np.array([], dtype=np.float64)

        target_lock_ticks = sum(1 for err in angle_errors if err <= 18.0)
        mean_angle_err = float(np.mean(angle_errors)) if angle_errors else None
        target_lock_fraction = (target_lock_ticks / len(angle_errors)) if angle_errors else None
        asym_std = float(np.std(asym_arr)) if len(asym_arr) > 1 else 0.0
        asym_abs_mean = float(np.mean(np.abs(asym_arr))) if len(asym_arr) > 0 else 0.0

        stability = {
            "turning_fraction": turn_steps / max(1, len(trajectory)),
            "direction_reversal_rate": sign_reversals / max(1, len(trajectory)),
            "angular_velocity_variance": None,
            "forward_progress_per_turn": None,
            "mean_target_angle_error": mean_angle_err,
            "target_lock_fraction": target_lock_fraction,
            "norm_asymmetry_std": asym_std,
            "norm_asymmetry_abs_mean": asym_abs_mean,
        }
        native_ticks = [tick for tick in trajectory if tick.native_state_available]
        distance_traveled = 0.0
        for previous, current in zip(native_ticks, native_ticks[1:]):
            if previous.x is not None and previous.y is not None and current.x is not None and current.y is not None:
                distance_traveled += float(np.hypot(current.x - previous.x, current.y - previous.y))
        native_available = bool(native_ticks)
        survival_steps = None
        if native_available:
            for tick in trajectory:
                if tick.health <= 0:
                    survival_steps = tick.step
                    break
            if survival_steps is None and obs is not None and obs.health > 0:
                survival_steps = len(trajectory)
        native_state_latest = (obs.info.get("native_game_state") if obs and obs.info else None) or {}
        player_kills_val = int(native_state_latest.get("player_kills", obs.kill_count if obs else 0))
        friendly_fire_val = int(native_state_latest.get("friendly_fire_kills", 0))
        damage_val = float(native_state_latest.get("damage_dealt", obs.damage_dealt if obs else 0.0))
        result = NativeEpisodeResult(
            episode_id=episode_id,
            seed=seed,
            condition=condition.name,
            map_name=self.map_name,
            iwad=str(self.iwad) if self.iwad else None,
            start_timestamp=started,
            end_timestamp=datetime.now(timezone.utc).isoformat(),
            termination_reason=termination,
            steps=len(trajectory),
            action_counts=actions,
            health_remaining=float(obs.health if obs is not None else 0.0),
            ammo_remaining=int(obs.ammo if obs is not None else 0),
            kills=player_kills_val,
            damage_dealt=damage_val,
            navigation_status=(
                "native_game_state_available"
                if any(t.native_state_available for t in trajectory)
                else "unavailable_native_bridge"
            ),
            stability=stability,
            environment_metadata=self._environment_metadata(target),
            trajectory=trajectory,
            distance_traveled=distance_traveled if native_available else None,
            survival=survival_steps,
            door_seeking=condition.door_seeking,
            player_kills=player_kills_val,
            friendly_fire_kills=friendly_fire_val,
        )
        self.write_episode(result)
        return result

    def _environment_metadata(self, target: GZDoomTarget) -> Dict[str, Any]:
        def sha256(path: Optional[Path]) -> Optional[str]:
            if path is None or not path.is_file():
                return None
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        version = None
        plist_path = target.app_path / "Contents" / "Info.plist"
        try:
            with plist_path.open("rb") as handle:
                plist = plistlib.load(handle)
            version = plist.get("CFBundleShortVersionString") or plist.get("CFBundleVersion")
        except (OSError, plistlib.InvalidFileException):
            pass
        meta = {
            "environment": "NATIVE_GZDOOM",
            "environment_tier": "ENGINEERING_ENVIRONMENT",
            "platform": platform.platform(),
            "gzdoom_version": version,
            "gzdoom_executable_sha256": sha256(target.executable),
            "iwad_sha256": sha256(target.iwad),
            "command_line": list(target.command()),
            "map": self.map_name,
            "capture_dimensions": [64, 64],
            "deterministic_replay_claim": False,
        }
        if self.iwad is not None and target.iwad != self.iwad:
            meta["original_iwad"] = str(self.iwad)
            meta["original_iwad_sha256"] = sha256(self.iwad)
            meta["instrumented_iwad"] = str(target.iwad)
        return meta

    def write_episode(self, result: NativeEpisodeResult) -> Path:
        episode_dir = self.output_dir / result.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "provenance.json").write_text(json.dumps({
            "experiment": "DOOM-004-NATIVE",
            "environment": "NATIVE_GZDOOM",
            "environment_tier": "ENGINEERING_ENVIRONMENT",
            "neural_tier": "COMPUTATIONAL_HYPOTHESIS",
            "controller": result.condition,
            "controller_config": next(({
                "model_type": condition.model_type.name,
                "synaptic_knockouts": sorted(condition.knockouts),
                "saccade_refractory_ticks": condition.saccade_refractory_ticks,
                "door_seeking": result.door_seeking,
            } for condition in DEFAULT_NATIVE_CONDITIONS + DEFAULT_SUPPRESSION_ABLATION + DEFAULT_LESION_CONDITIONS if condition.name == result.condition), {}),
            "map": result.map_name,
            "seed": result.seed,
        }, indent=2) + "\n")
        metadata = asdict(result)
        metadata.pop("trajectory")
        (episode_dir / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pylist([asdict(tick) for tick in result.trajectory])
            pq.write_table(table, episode_dir / "trajectory.parquet")
        except ImportError as exc:
            raise RuntimeError("DOOM-004 artifact writing requires pyarrow") from exc
        return episode_dir
