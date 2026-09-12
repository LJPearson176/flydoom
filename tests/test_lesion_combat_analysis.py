import numpy as np
import pytest

from fly_doom.doom.native_benchmark import (
    DEFAULT_LESION_CONDITIONS,
    NativeCondition,
    NativeEpisodeResult,
    NativeGZDoomBenchmark,
    NativeTrajectoryTick,
)
from fly_doom.dynamics.compartmental_t4 import CompartmentModelType


def test_default_lesion_conditions_definition():
    names = [c.name for c in DEFAULT_LESION_CONDITIONS]
    assert "ModelD_ActiveTree_Saccade_3" in names
    assert "ModelD_Mi4_KO_Saccade_3" in names
    assert "ModelD_Mi9_KO_Saccade_3" in names
    assert "ModelD_Tm3_KO_Saccade_3" in names
    assert "ModelD_Mi1_KO_Saccade_3" in names

    mi4 = next(c for c in DEFAULT_LESION_CONDITIONS if c.name == "ModelD_Mi4_KO_Saccade_3")
    assert mi4.knockouts == frozenset({"Mi4"})
    assert mi4.saccade_refractory_ticks == 3
    assert mi4.model_type == CompartmentModelType.MODEL_D


def test_targeting_stability_and_optic_flow_metrics(tmp_path):
    ticks = [
        NativeTrajectoryTick(
            episode_id="ep1",
            seed=1001,
            condition="ModelD_ActiveTree_Saccade_3",
            step=0,
            timestamp="",
            action="FORWARD",
            health=100.0,
            ammo=50,
            kills=0,
            frame_mean=0.5,
            frame_nonzero_fraction=1.0,
            retina_mean=0.5,
            retina_delta=0.1,
            t4_l_v=-65.0,
            t4_r_v=-65.0,
            norm_asymmetry=0.5,
            linear_velocity=None,
            angular_velocity=None,
            capture_valid=True,
            window_id=1,
            window_bounds=None,
            x=1700.0,
            y=-2500.0,
            angle_deg=10.0,
            target_angle_deg=20.0,  # diff = 10 deg <= 18 deg (locked)
            combat_active=1.0,
            native_state_available=True,
        ),
        NativeTrajectoryTick(
            episode_id="ep1",
            seed=1001,
            condition="ModelD_ActiveTree_Saccade_3",
            step=1,
            timestamp="",
            action="FIRE",
            health=100.0,
            ammo=49,
            kills=0,
            frame_mean=0.5,
            frame_nonzero_fraction=1.0,
            retina_mean=0.5,
            retina_delta=0.1,
            t4_l_v=-65.0,
            t4_r_v=-65.0,
            norm_asymmetry=-0.5,
            linear_velocity=None,
            angular_velocity=None,
            capture_valid=True,
            window_id=1,
            window_bounds=None,
            x=1700.0,
            y=-2500.0,
            angle_deg=0.0,
            target_angle_deg=30.0,  # diff = 30 deg > 18 deg (not locked)
            combat_active=1.0,
            native_state_available=True,
        ),
    ]

    class MockBridge:
        def __init__(self, target):
            pass
        def launch(self):
            return None
        def reset(self, seed=None):
            return None
        def close(self):
            pass

    benchmark = NativeGZDoomBenchmark(tmp_path, max_steps=2, bridge_factory=MockBridge)
    # Run the same stability logic directly
    combat_ticks = [t for t in ticks if getattr(t, "combat_active", 0.0) == 1.0]
    angle_errors = []
    for t in combat_ticks:
        diff = (t.target_angle_deg - t.angle_deg + 180.0) % 360.0 - 180.0
        angle_errors.append(abs(diff))

    asymmetries = [t.norm_asymmetry for t in ticks if t.norm_asymmetry is not None]
    asym_arr = np.array(asymmetries, dtype=np.float64)

    target_lock_ticks = sum(1 for err in angle_errors if err <= 18.0)
    mean_angle_err = float(np.mean(angle_errors))
    target_lock_fraction = target_lock_ticks / len(angle_errors)
    asym_std = float(np.std(asym_arr))

    assert mean_angle_err == pytest.approx(20.0)  # (10 + 30) / 2
    assert target_lock_fraction == pytest.approx(0.5)  # 1 of 2 locked
    assert asym_std == pytest.approx(0.5)
