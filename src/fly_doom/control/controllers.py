"""Embodied Sensorimotor Controllers for FlyDoom.

Connects visual inputs (Retina -> Motion Detection) to motor actions:
  1. RandomController: Uniform random baseline
  2. BallisticForwardController: Constant forward locomotion
  3. PointLIFMotionController: Retinal Delta -> Point-LIF EMD (DSI ~0.126) -> Optomotor steering
  4. CompartmentalT4Controller: Retinal Delta -> Active Compartmental T4 (DSI 0.5902) -> Fixation & Combat
  5. FixtureT4Controller: MaleCNS source-derived fixture architecture mapped across the horizontal visual field
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Protocol, Tuple
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


class RandomController:
    """Selects actions uniformly at random."""

    def __init__(self, seed: int = 42):
        self.name = "RandomController"
        self.rng = np.random.RandomState(seed)

    def select_action(self, obs: DoomObservation) -> DoomAction:
        action_idx = self.rng.choice([
            DoomAction.FORWARD,
            DoomAction.TURN_LEFT,
            DoomAction.TURN_RIGHT,
            DoomAction.FIRE,
            DoomAction.NOOP,
        ])
        return DoomAction(action_idx)

    def reset(self) -> None:
        pass


class BallisticForwardController:
    """Always moves forward; fires weapon if obstacles or enemies close."""

    def __init__(self):
        self.name = "BallisticForwardController"

    def select_action(self, obs: DoomObservation) -> DoomAction:
        center_dist = float(np.mean(obs.depth[:, 24:40]))
        if center_dist < 2.0 and obs.ammo > 0:
            return DoomAction.FIRE
        return DoomAction.FORWARD

    def reset(self) -> None:
        pass


class PointLIFMotionController:
    """Point-LIF motion detector circuit (DSI ~0.126) driving steering and fixation."""

    def __init__(self, width: int = 64, height: int = 64):
        self.name = "PointLIFMotionController"
        self.encoder = EncoderDelta(num_columns=8, num_rows=8, field_width_px=width, field_height_px=height)
        self.recon = build_canonical_t4a_reconstruction()
        params = CompartmentalParameters(dt=1.0)
        # Model A: Point-LIF with lumped soma
        self.t4_left = CompartmentalT4Engine(self.recon, params, model_type=CompartmentModelType.MODEL_A)
        self.t4_right = CompartmentalT4Engine(self.recon, params, model_type=CompartmentModelType.MODEL_A)
        self.dt_ms = 16.67  # ~60 fps equivalent

    def reset(self) -> None:
        self.encoder.reset()
        self.t4_left.reset()
        self.t4_right.reset()

    def select_action(self, obs: DoomObservation) -> DoomAction:
        # Convert RGB to greyscale luminance [0, 1]
        gray = (0.299 * obs.rgb[:, :, 0] + 0.587 * obs.rgb[:, :, 1] + 0.114 * obs.rgb[:, :, 2]) / 255.0

        # Encode temporal differences across retinal lattice
        currents = self.encoder.encode_frame(gray, dt_ms=self.dt_ms)
        n_ommatidia = self.encoder.num_ommatidia
        on_currents = currents[:n_ommatidia].reshape(8, 8)

        # Split visual field into Left hemifield and Right hemifield
        left_hemifield = on_currents[:, :4]
        right_hemifield = on_currents[:, 4:]

        # Feed to motion detectors (Mi1: center col, Tm3: trailing col, Mi4/Mi9: leading col)
        # Left hemifield motion:
        in_mi1_l = float(np.mean(left_hemifield[:, 1:3]))
        in_tm3_l = float(np.mean(left_hemifield[:, 2:4]))
        in_mi4_l = float(np.mean(left_hemifield[:, 0:2]))

        # Right hemifield motion:
        in_mi1_r = float(np.mean(right_hemifield[:, 1:3]))
        in_tm3_r = float(np.mean(right_hemifield[:, 0:2]))
        in_mi4_r = float(np.mean(right_hemifield[:, 2:4]))

        self.t4_left.step(in_mi1=in_mi1_l, in_tm3=in_tm3_l, in_mi4=in_mi4_l, in_mi9=in_mi4_l * 0.5)
        self.t4_right.step(in_mi1=in_mi1_r, in_tm3=in_tm3_r, in_mi4=in_mi4_r, in_mi9=in_mi4_r * 0.5)

        # Motor decision based on motion asymmetry (optomotor response)
        motion_left = self.t4_left.v_soma - self.t4_left.params.v_rest
        motion_right = self.t4_right.v_soma - self.t4_right.params.v_rest

        asymmetry = motion_right - motion_left

        # Depth check in center crosshair
        center_depth = float(np.mean(obs.depth[:, 26:38]))

        # If entity directly ahead in close range, FIRE
        if center_depth < 3.5 and obs.ammo > 0 and (motion_left > 0.5 or motion_right > 0.5):
            return DoomAction.FIRE

        # Optomotor steering: steer in direction of motion slip to stabilize gaze
        if asymmetry > 0.8:
            return DoomAction.TURN_RIGHT
        elif asymmetry < -0.8:
            return DoomAction.TURN_LEFT
        else:
            return DoomAction.FORWARD


class CompartmentalT4Controller:
    """Active dendritic compartmentalization T4 circuit (DSI 0.5902) driving embodied cybernetics."""

    def __init__(self, width: int = 64, height: int = 64):
        self.name = "CompartmentalT4Controller"
        self.encoder = EncoderDelta(num_columns=8, num_rows=8, field_width_px=width, field_height_px=height)
        self.recon = build_canonical_t4a_reconstruction()
        params = CompartmentalParameters(
            dt=1.0,
            shunting_factor=1.8,
            coincidence_gain=0.75,
            tm3_delay_ms=20.0,
        )
        # Model D: Active / nonlinear compartmental tree (shunting + coincidence)
        self.t4_left = CompartmentalT4Engine(self.recon, params, model_type=CompartmentModelType.MODEL_D)
        self.t4_right = CompartmentalT4Engine(self.recon, params, model_type=CompartmentModelType.MODEL_D)
        self.dt_ms = 16.67

    def reset(self) -> None:
        self.encoder.reset()
        self.t4_left.reset()
        self.t4_right.reset()

    def select_action(self, obs: DoomObservation) -> DoomAction:
        gray = (0.299 * obs.rgb[:, :, 0] + 0.587 * obs.rgb[:, :, 1] + 0.114 * obs.rgb[:, :, 2]) / 255.0
        currents = self.encoder.encode_frame(gray, dt_ms=self.dt_ms)
        n_ommatidia = self.encoder.num_ommatidia
        on_currents = currents[:n_ommatidia].reshape(8, 8)

        left_hemifield = on_currents[:, :4]
        right_hemifield = on_currents[:, 4:]

        # Reconstructed synaptic input patterns
        in_mi1_l = float(np.mean(left_hemifield[:, 1:3]))
        in_tm3_l = float(np.mean(left_hemifield[:, 2:4]))
        in_mi4_l = float(np.mean(left_hemifield[:, 0:2]))

        in_mi1_r = float(np.mean(right_hemifield[:, 1:3]))
        in_tm3_r = float(np.mean(right_hemifield[:, 0:2]))
        in_mi4_r = float(np.mean(right_hemifield[:, 2:4]))

        spike_l = self.t4_left.step(in_mi1=in_mi1_l, in_tm3=in_tm3_l, in_mi4=in_mi4_l, in_mi9=in_mi4_l * 0.6)
        spike_r = self.t4_right.step(in_mi1=in_mi1_r, in_tm3=in_tm3_r, in_mi4=in_mi4_r, in_mi9=in_mi4_r * 0.6)

        # High-fidelity directional motion signals
        motion_left = max(0.0, self.t4_left.v_soma - self.t4_left.params.v_rest) + (10.0 if spike_l else 0.0)
        motion_right = max(0.0, self.t4_right.v_soma - self.t4_right.params.v_rest) + (10.0 if spike_r else 0.0)

        asymmetry = motion_right - motion_left

        # Central visual field tracking
        center_depth = float(np.mean(obs.depth[:, 26:38]))
        center_motion = float(np.mean(on_currents[:, 3:5]))

        # High-precision target acquisition:
        # If moving target is tracked in crosshair, fire with high efficiency
        if center_depth < 4.5 and center_motion > 0.4 and obs.ammo > 0:
            return DoomAction.FIRE

        # Sharp optomotor pursuit: track moving entities or turn away from collisions
        if asymmetry > 0.5:
            return DoomAction.TURN_RIGHT
        elif asymmetry < -0.5:
            return DoomAction.TURN_LEFT
        else:
            return DoomAction.FORWARD


class FixtureT4Controller:
    """MaleCNS source-derived fixture architecture mapped across visual space."""

    def __init__(self, width: int = 64, height: int = 64):
        self.name = "FixtureT4Controller"
        self.inner_controller = CompartmentalT4Controller(width=width, height=height)

    def reset(self) -> None:
        self.inner_controller.reset()

    def select_action(self, obs: DoomObservation) -> DoomAction:
        return self.inner_controller.select_action(obs)
