"""Embodied Sensorimotor Controllers for FlyDoom.

Implements unified, controlled sensorimotor policies where the sensory encoder,
motor decoder, and behavioral thresholds are held STRICTLY CONSTANT across all
conditions, varying ONLY the biophysical neural substrate or synaptic knockouts:

Controlled Neural Architectures (DOOM-002):
  1. ControlledT4Controller(model_type=MODEL_A) -> Point LIF (DSI ~0.054)
  2. ControlledT4Controller(model_type=MODEL_B) -> Temporal Point LIF (DSI ~0.126)
  3. ControlledT4Controller(model_type=MODEL_C) -> Passive Dendritic Tree (DSI ~0.144)
  4. ControlledT4Controller(model_type=MODEL_D) -> Active Dendritic Tree (DSI 0.5902)
  5. ControlledT4Controller with synaptic knockouts (Mi4 KO, Mi9 KO, Mi1 KO, Tm3 KO)

Also maintains reference baselines:
  - RandomController (lower-bound anchor)
  - BallisticForwardController (pure movement baseline)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Protocol, Set, Tuple
import numpy as np

from fly_doom.connectome.t4_anatomical import (
    T4AnatomicalReconstruction,
    build_canonical_t4a_reconstruction,
)
from fly_doom.core.provenance import Provenance
from fly_doom.doom.interface import DoomAction, DoomObservation
from fly_doom.dynamics.compartmental_t4 import (
    CompartmentalParameters,
    CompartmentalT4Engine,
    CompartmentModelType,
)
from fly_doom.sensory.encoders.delta import EncoderDelta


class SensorimotorController(Protocol):
    """Protocol for embodied controllers mapping Doom observations to actions."""

    name: str

    def select_action(self, obs: DoomObservation) -> DoomAction:
        ...

    def reset(self) -> None:
        ...

    def get_neural_state(self) -> Dict[str, float]:
        ...


class RandomController:
    """Selects actions uniformly at random."""

    def __init__(self, seed: int = 42):
        self.name = "RandomController"
        self.rng = np.random.RandomState(seed)
        self.last_neural_state: Dict[str, float] = {}

    def select_action(self, obs: DoomObservation) -> DoomAction:
        action_idx = self.rng.choice([
            DoomAction.FORWARD,
            DoomAction.TURN_LEFT,
            DoomAction.TURN_RIGHT,
            DoomAction.FIRE,
            DoomAction.NOOP,
        ])
        self.last_neural_state = {
            "retina_mean": 0.0,
            "retina_delta": 0.0,
            "t4_l_v": -65.0,
            "t4_r_v": -65.0,
            "t4_l_spike": 0.0,
            "t4_r_spike": 0.0,
            "motion_asymmetry": 0.0,
            "center_depth": float(np.mean(obs.depth[:, 26:38])) if obs.depth is not None else 0.0,
        }
        return DoomAction(action_idx)

    def reset(self) -> None:
        self.last_neural_state = {}

    def get_neural_state(self) -> Dict[str, float]:
        return self.last_neural_state


class BallisticForwardController:
    """Always moves forward; fires weapon if obstacles or enemies close."""

    def __init__(self):
        self.name = "BallisticForwardController"
        self.last_neural_state: Dict[str, float] = {}

    def select_action(self, obs: DoomObservation) -> DoomAction:
        center_dist = float(np.mean(obs.depth[:, 24:40]))
        action = DoomAction.FORWARD
        if center_dist < 2.0 and obs.ammo > 0:
            action = DoomAction.FIRE

        self.last_neural_state = {
            "retina_mean": 0.0,
            "retina_delta": 0.0,
            "t4_l_v": -65.0,
            "t4_r_v": -65.0,
            "t4_l_spike": 0.0,
            "t4_r_spike": 0.0,
            "motion_asymmetry": 0.0,
            "center_depth": center_dist,
        }
        return action

    def reset(self) -> None:
        self.last_neural_state = {}

    def get_neural_state(self) -> Dict[str, float]:
        return self.last_neural_state


class ControlledT4Controller:
    """Unified, controlled sensorimotor controller for DOOM-002 ablation.

    Freezes:
      - Retinal delta encoder (8x8 ommatidia, sigma=4.0px, gain=35.0)
      - Fixed steering threshold (asymmetry > 0.50 -> TURN_RIGHT, < -0.50 -> TURN_LEFT)
      - Fixed firing threshold (center_depth < 3.5, motion_energy > 0.40, ammo > 0)
      - Fixed timestep dt_ms = 16.67 ms

    Varies ONLY:
      - Neural simulation engine (Model A, Model B, Model C, Model D)
      - Optional synaptic knockouts (e.g. {'Mi4'}, {'Mi9'}, {'Tm3'}, {'Mi1'})
    """

    def __init__(
        self,
        model_type: CompartmentModelType = CompartmentModelType.MODEL_D,
        synaptic_knockouts: Optional[Set[str]] = None,
        custom_name: Optional[str] = None,
        width: int = 64,
        height: int = 64,
    ):
        self.model_type = model_type
        self.knockouts = synaptic_knockouts or set()

        ko_str = f"_{'-'.join(sorted(self.knockouts))}_KO" if self.knockouts else ""
        self.name = custom_name or f"T4_{model_type.name}{ko_str}"

        self.encoder = EncoderDelta(num_columns=8, num_rows=8, field_width_px=width, field_height_px=height)
        self.recon = build_canonical_t4a_reconstruction()

        # Calibrated biophysical parameters
        self.params = CompartmentalParameters(
            dt=1.0,
            shunting_factor=1.8,
            coincidence_gain=0.75,
            tm3_delay_ms=20.0,
        )

        self.t4_left = CompartmentalT4Engine(self.recon, self.params, model_type=self.model_type)
        self.t4_right = CompartmentalT4Engine(self.recon, self.params, model_type=self.model_type)
        self.dt_ms = 16.67
        self.last_neural_state: Dict[str, float] = {}

    def reset(self) -> None:
        self.encoder.reset()
        self.t4_left.reset()
        self.t4_right.reset()
        self.last_neural_state = {}

    def select_action(self, obs: DoomObservation) -> DoomAction:
        # 1. Sensory Projection
        gray = (0.299 * obs.rgb[:, :, 0] + 0.587 * obs.rgb[:, :, 1] + 0.114 * obs.rgb[:, :, 2]) / 255.0
        currents = self.encoder.encode_frame(gray, dt_ms=self.dt_ms)
        n_ommatidia = self.encoder.num_ommatidia
        on_currents = currents[:n_ommatidia].reshape(8, 8)

        # 2. Hemifield Presynaptic Drive
        left_hemifield = on_currents[:, :4]
        right_hemifield = on_currents[:, 4:]

        in_mi1_l = float(np.mean(left_hemifield[:, 1:3])) if "Mi1" not in self.knockouts else 0.0
        in_tm3_l = float(np.mean(left_hemifield[:, 2:4])) if "Tm3" not in self.knockouts else 0.0
        in_mi4_l = float(np.mean(left_hemifield[:, 0:2])) if "Mi4" not in self.knockouts else 0.0
        in_mi9_l = in_mi4_l * 0.6 if "Mi9" not in self.knockouts else 0.0

        in_mi1_r = float(np.mean(right_hemifield[:, 1:3])) if "Mi1" not in self.knockouts else 0.0
        in_tm3_r = float(np.mean(right_hemifield[:, 0:2])) if "Tm3" not in self.knockouts else 0.0
        in_mi4_r = float(np.mean(right_hemifield[:, 2:4])) if "Mi4" not in self.knockouts else 0.0
        in_mi9_r = in_mi4_r * 0.6 if "Mi9" not in self.knockouts else 0.0

        # 3. Biophysical Substrate Update
        spike_l = self.t4_left.step(in_mi1=in_mi1_l, in_tm3=in_tm3_l, in_mi4=in_mi4_l, in_mi9=in_mi9_l)
        spike_r = self.t4_right.step(in_mi1=in_mi1_r, in_tm3=in_tm3_r, in_mi4=in_mi4_r, in_mi9=in_mi9_r)

        # High-order somatic / spike motion energy
        motion_l = max(0.0, self.t4_left.v_soma - self.t4_left.params.v_rest) + (10.0 if spike_l else 0.0)
        motion_r = max(0.0, self.t4_right.v_soma - self.t4_right.params.v_rest) + (10.0 if spike_r else 0.0)

        asymmetry = motion_r - motion_l
        center_motion = float(np.mean(on_currents[:, 3:5]))
        center_depth = float(np.mean(obs.depth[:, 26:38])) if obs.depth is not None else 10.0

        # Store full tick neural state
        self.last_neural_state = {
            "retina_mean": float(np.mean(gray)),
            "retina_delta": float(np.mean(np.abs(currents))),
            "t4_l_v": float(self.t4_left.v_soma),
            "t4_r_v": float(self.t4_right.v_soma),
            "t4_l_spike": 1.0 if spike_l else 0.0,
            "t4_r_spike": 1.0 if spike_r else 0.0,
            "motion_asymmetry": float(asymmetry),
            "center_depth": float(center_depth),
        }

        # 4. FROZEN MOTOR DECODER POLICY
        # (Held perfectly invariant across all model conditions)
        steer_threshold = 0.50
        fire_depth_max = 3.50
        fire_motion_min = 0.40

        # Firing rule: target within engagement range, motion confirmed, ammo present
        if center_depth < fire_depth_max and center_motion > fire_motion_min and obs.ammo > 0:
            return DoomAction.FIRE

        # Optomotor steering rule: orient to balance horizontal optical slip
        if asymmetry > steer_threshold:
            return DoomAction.TURN_RIGHT
        elif asymmetry < -steer_threshold:
            return DoomAction.TURN_LEFT
        else:
            return DoomAction.FORWARD

    def get_neural_state(self) -> Dict[str, float]:
        return self.last_neural_state


# Legacy backward-compatible aliases
PointLIFMotionController = lambda width=64, height=64: ControlledT4Controller(
    model_type=CompartmentModelType.MODEL_A, custom_name="PointLIFMotionController", width=width, height=height
)
CompartmentalT4Controller = lambda width=64, height=64: ControlledT4Controller(
    model_type=CompartmentModelType.MODEL_D, custom_name="CompartmentalT4Controller", width=width, height=height
)
FixtureT4Controller = CompartmentalT4Controller
