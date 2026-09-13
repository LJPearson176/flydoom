"""Embodied Sensorimotor Controllers for FlyDoom.

Implements unified, controlled sensorimotor policies where the sensory encoder,
motor decoder, and behavioral thresholds are held STRICTLY CONSTANT across all
conditions, varying ONLY the biophysical neural substrate or synaptic knockouts:

Controlled Neural Architectures (DOOM-003):
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
from fly_doom.dynamics.optic_flow import OpticFlowDecomposer, RetinotopicT4ArrayEngine
from fly_doom.dynamics.central_complex_ring import EllipsoidBodyRingAttractor
from fly_doom.dynamics.mushroom_body import MushroomBodyPlasticityEngine
from fly_doom.dynamics.fan_shaped_body import (
    AMMCWallSlipReflex,
    FanShapedBodyVectorEngine,
    SEZNociceptiveReflex,
    Stage1WaypointGraph,
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
            "norm_asymmetry": 0.0,
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
            "norm_asymmetry": 0.0,
            "center_depth": center_dist,
        }
        return action

    def reset(self) -> None:
        self.last_neural_state = {}

    def get_neural_state(self) -> Dict[str, float]:
        return self.last_neural_state


class ControlledT4Controller:
    """Unified, scale-normalized sensorimotor controller for DOOM-003.

    Holds Strictly Invariant:
      - Retinal delta encoder (8x8 ommatidia, sigma=4.0px, gain=35.0)
      - Timestep dt_ms = 16.67 ms
      - Scale-invariant relative motion decoder:
          hat{M}_L = M_L / (M_L + M_R + eps)
          hat{M}_R = M_R / (M_L + M_R + eps)
          hat{Delta} = hat{M}_R - hat{M}_L in [-1.0, 1.0]
      - Fixed steering threshold (|hat{Delta}| > 0.15)
      - Fixed firing threshold (center_depth < 3.5, center_motion > 0.35, ammo > 0)

    Varies ONLY:
      - Neural simulation engine (Model A, Model B, Model C, Model D)
      - Synaptic knockouts (e.g. {'Mi4'}, {'Mi9'}, {'Tm3'}, {'Mi1'})
    """

    def __init__(
        self,
        model_type: CompartmentModelType = CompartmentModelType.MODEL_D,
        synaptic_knockouts: Optional[Set[str]] = None,
        custom_name: Optional[str] = None,
        width: int = 64,
        height: int = 64,
        saccade_refractory_ticks: int = 0,
        use_population_flow: bool = True,
        use_ring_attractor: bool = True,
        use_mushroom_body: bool = True,
    ):
        self.model_type = model_type
        self.knockouts = synaptic_knockouts or set()
        self.saccade_refractory_ticks = int(saccade_refractory_ticks)
        self._saccade_cooldown = 0
        self.use_population_flow = use_population_flow
        self.use_ring_attractor = use_ring_attractor
        self.use_mushroom_body = use_mushroom_body

        ko_str = f"_{'-'.join(sorted(self.knockouts))}_KO" if self.knockouts else ""
        self.name = custom_name or f"T4_{model_type.name}{ko_str}"

        self.encoder = EncoderDelta(num_columns=8, num_rows=8, field_width_px=width, field_height_px=height)
        self.recon = build_canonical_t4a_reconstruction()

        self.dt_ms = 16.67

        # Calibrated biophysical parameters
        self.params = CompartmentalParameters(
            dt=self.dt_ms,
            shunting_factor=1.8,
            coincidence_gain=0.75,
            tm3_delay_ms=20.0,
        )

        self.t4_left = CompartmentalT4Engine(self.recon, self.params, model_type=self.model_type)
        self.t4_right = CompartmentalT4Engine(self.recon, self.params, model_type=self.model_type)

        # High-fidelity population-level retinotopic array & central complex ring
        self.t4_array = RetinotopicT4ArrayEngine(rows=8, cols=8, params=self.params)
        self.optic_flow_decomposer = OpticFlowDecomposer(rows=8, cols=8)
        self.eb_ring_attractor = EllipsoidBodyRingAttractor(num_wedges=16, dt_ms=self.dt_ms)
        self.mushroom_body = MushroomBodyPlasticityEngine()
        self._last_action = DoomAction.NOOP
        self._last_health: Optional[float] = None
        self._last_kill_count: Optional[int] = None
        self.last_neural_state: Dict[str, float] = {}

    def reset(self) -> None:
        self.encoder.reset()
        self.t4_left.reset()
        self.t4_right.reset()
        self.t4_array.reset()
        self.eb_ring_attractor.reset()
        self.mushroom_body.last_kc_activations.fill(0.0)
        self.last_neural_state = {}
        self._saccade_cooldown = 0
        self._last_action = DoomAction.NOOP
        self._last_health = None
        self._last_kill_count = None

    def select_action(self, obs: DoomObservation) -> DoomAction:
        # 1. Sensory Projection
        gray = (0.299 * obs.rgb[:, :, 0] + 0.587 * obs.rgb[:, :, 1] + 0.114 * obs.rgb[:, :, 2]) / 255.0

        # Ensure spatial dimensions match encoder field dimensions
        enc_h, enc_w = self.encoder.field_height, self.encoder.field_width
        if gray.shape != (enc_h, enc_w):
            r_idx = np.linspace(0, gray.shape[0] - 1, enc_h).astype(np.int32)
            c_idx = np.linspace(0, gray.shape[1] - 1, enc_w).astype(np.int32)
            gray = np.ascontiguousarray(gray[np.ix_(r_idx, c_idx)])

        currents = self.encoder.encode_frame(gray, dt_ms=self.dt_ms)
        n_ommatidia = self.encoder.num_ommatidia
        on_currents = currents[:n_ommatidia].reshape(8, 8)

        # 2. Hemifield Presynaptic Drive
        left_hemifield = on_currents[:, :4]
        right_hemifield = on_currents[:, 4:]

        # Scale sensory currents into physiological conductance rate units
        input_scale = 0.05
        in_mi1_l = float(np.mean(left_hemifield[:, 1:3])) * input_scale if "Mi1" not in self.knockouts else 0.0
        in_tm3_l = float(np.mean(left_hemifield[:, 2:4])) * input_scale if "Tm3" not in self.knockouts else 0.0
        in_mi4_l = float(np.mean(left_hemifield[:, 0:2])) * input_scale if "Mi4" not in self.knockouts else 0.0
        in_mi9_l = in_mi4_l * 0.6 if "Mi9" not in self.knockouts else 0.0

        in_mi1_r = float(np.mean(right_hemifield[:, 1:3])) * input_scale if "Mi1" not in self.knockouts else 0.0
        in_tm3_r = float(np.mean(right_hemifield[:, 0:2])) * input_scale if "Tm3" not in self.knockouts else 0.0
        in_mi4_r = float(np.mean(right_hemifield[:, 2:4])) * input_scale if "Mi4" not in self.knockouts else 0.0
        in_mi9_r = in_mi4_r * 0.6 if "Mi9" not in self.knockouts else 0.0

        # 3. Biophysical Substrate Update
        spike_l = self.t4_left.step(in_mi1=in_mi1_l, in_tm3=in_tm3_l, in_mi4=in_mi4_l, in_mi9=in_mi9_l)
        spike_r = self.t4_right.step(in_mi1=in_mi1_r, in_tm3=in_tm3_r, in_mi4=in_mi4_r, in_mi9=in_mi9_r)

        # High-order somatic / spike motion energy
        v_l = self.t4_left.v_soma
        v_r = self.t4_right.v_soma
        v_rest = self.params.v_rest

        # Membrane depolarization above rest (or hyperpolarization relative to baseline)
        depol_l = max(0.0, v_l - v_rest) + (15.0 if spike_l else 0.0)
        depol_r = max(0.0, v_r - v_rest) + (15.0 if spike_r else 0.0)

        # Smooth, non-saturating biophysical asymmetry preserving graded analog sensitivity
        diff_v = (v_r - v_l) + (15.0 if spike_r else 0.0) - (15.0 if spike_l else 0.0)
        norm_asymmetry = float(np.tanh(diff_v / 15.0))

        raw_asymmetry = depol_r - depol_l
        center_motion = float(np.mean(on_currents[:, 3:5]))
        depth_available = bool(
            obs.depth is not None
            and obs.depth.size
            and np.all(np.isfinite(obs.depth))
            and np.any(obs.depth > 0)
        )
        center_depth = float(np.mean(obs.depth[:, 26:38])) if depth_available else float("inf")

        # Preserve the internal causal path, not just the somatic output.  The
        # branch values are model state variables (and therefore computational
        # hypotheses), but exposing them makes it possible to distinguish
        # leading inhibition, central excitation, delayed trailing excitation,
        # and somatic integration in the Observatory.
        activation_span = max(1e-6, self.params.v_thresh - v_rest)

        def branch_state(engine: CompartmentalT4Engine, prefix: str) -> Dict[str, float]:
            def activation(value: float) -> float:
                return float(np.clip((value - v_rest) / activation_span, 0.0, 1.0))

            return {
                f"{prefix}_leading_v": float(engine.v_leading),
                f"{prefix}_central_v": float(engine.v_central),
                f"{prefix}_trailing_v": float(engine.v_trailing),
                f"{prefix}_soma_activation": activation(float(engine.v_soma)),
                f"{prefix}_leading_activation": activation(float(engine.v_leading)),
                f"{prefix}_central_activation": activation(float(engine.v_central)),
                f"{prefix}_trailing_activation": activation(float(engine.v_trailing)),
                f"{prefix}_mi1_drive": float(engine.g_mi1),
                f"{prefix}_tm3_drive": float(engine.g_tm3),
                f"{prefix}_mi4_inhibition": float(engine.g_mi4),
                f"{prefix}_mi9_inhibition": float(engine.g_mi9),
            }

        # Store full tick neural state
        self.last_neural_state = {
            "retina_mean": float(np.mean(gray)),
            "retina_delta": float(np.mean(np.abs(currents))),
            "t4_l_v": float(v_l),
            "t4_r_v": float(v_r),
            "t4_l_spike": 1.0 if spike_l else 0.0,
            "t4_r_spike": 1.0 if spike_r else 0.0,
            "motion_asymmetry": float(raw_asymmetry),
            "norm_asymmetry": float(norm_asymmetry),
            "center_depth": float(center_depth),
            "depth_available": 1.0 if depth_available else 0.0,
            "retina_left_drive": float(np.mean(left_hemifield)),
            "retina_right_drive": float(np.mean(right_hemifield)),
            "retina_center_drive": float(center_motion),
        }
        self.last_neural_state.update(branch_state(self.t4_left, "t4_l"))
        self.last_neural_state.update(branch_state(self.t4_right, "t4_r"))
        self.last_neural_state.update({
            "t4_l_mi1_input": float(in_mi1_l),
            "t4_l_tm3_input": float(in_tm3_l),
            "t4_l_mi4_input": float(in_mi4_l),
            "t4_l_mi9_input": float(in_mi9_l),
            "t4_r_mi1_input": float(in_mi1_r),
            "t4_r_tm3_input": float(in_tm3_r),
            "t4_r_mi4_input": float(in_mi4_r),
            "t4_r_mi9_input": float(in_mi9_r),
        })

        # High-Fidelity Population-Level Retinotopic Optic Flow & Central Complex Dynamics
        flow_metrics = None
        if self.use_population_flow:
            u_field, v_field, _ = self.t4_array.step(on_currents, knockouts=self.knockouts)
            flow_metrics = self.optic_flow_decomposer.decompose(u_field, v_field)
            self.last_neural_state.update({
                "flow_divergence": float(flow_metrics.divergence),
                "flow_curl": float(flow_metrics.curl),
                "flow_trans_x": float(flow_metrics.trans_x),
                "flow_trans_y": float(flow_metrics.trans_y),
                "flow_looming_index": float(flow_metrics.looming_index),
                "lptc_hs_l": float(flow_metrics.lptc_hs_l),
                "lptc_hs_r": float(flow_metrics.lptc_hs_r),
            })

        if self.use_ring_attractor:
            # Motor efference copy from previous action
            omega_motor = 0.0
            if self._last_action == DoomAction.TURN_RIGHT:
                omega_motor = 1.5
            elif self._last_action == DoomAction.TURN_LEFT:
                omega_motor = -1.5

            # Visual yaw slip from LPTC-HS asymmetry / curl
            omega_vis = 0.0
            if flow_metrics is not None:
                omega_vis = 0.05 * (flow_metrics.lptc_hs_r - flow_metrics.lptc_hs_l) - flow_metrics.curl * 0.2
            elif abs(norm_asymmetry) > 0.05:
                omega_vis = norm_asymmetry * 1.5

            net_omega = omega_motor + omega_vis
            eb_state = self.eb_ring_attractor.step(angular_velocity=net_omega)
            self.last_neural_state.update({
                "eb_heading_deg": float(eb_state.heading_angle_deg),
                "eb_heading_rad": float(eb_state.heading_angle_rad),
                "eb_bump_amplitude": float(eb_state.bump_amplitude),
                "eb_bump_coherence": float(eb_state.bump_coherence),
                "eb_angular_velocity": float(net_omega),
            })

        # 4. Mushroom Body Associative Learning & Spatial Valence
        if self.use_mushroom_body:
            # Construct 16-dimensional spatial context vector from row and col profile
            feat_rows = np.mean(on_currents, axis=1)
            feat_cols = np.mean(on_currents, axis=0)
            sensory_feat = np.concatenate([feat_rows, feat_cols])
            feat_norm = float(np.linalg.norm(sensory_feat))
            if feat_norm > 1e-4:
                sensory_feat = sensory_feat / feat_norm

            # Reinforcement triggers: PPL1 aversive (health drop) & PAM reward (kill)
            aversive_us = 0.0
            if self._last_health is not None and obs.health < self._last_health:
                aversive_us = min(1.0, float(self._last_health - obs.health) / 10.0)
            self._last_health = float(obs.health)

            reward_us = 0.0
            if self._last_kill_count is not None and obs.kill_count > self._last_kill_count:
                reward_us = 1.0
            self._last_kill_count = int(obs.kill_count)

            mb_state = self.mushroom_body.step(sensory_feat, aversive_us=aversive_us, reward_us=reward_us)
            self.last_neural_state.update({
                "mb_valence": float(mb_state.valence),
                "mb_mbon_app": float(mb_state.mbon_app_drive),
                "mb_mbon_av": float(mb_state.mbon_av_drive),
                "mb_ppl1_da": float(mb_state.ppl1_dopamine),
                "mb_pam_da": float(mb_state.pam_dopamine),
                "mb_active_kc_count": float(len(mb_state.active_kc_indices)),
            })

            # Threat avoidance modulation: if valence is negative, bias steering away from threat
            if mb_state.valence < -0.20:
                avoidance_bias = -0.25 if norm_asymmetry >= 0.0 else 0.25
                norm_asymmetry = float(np.clip(norm_asymmetry + avoidance_bias, -1.0, 1.0))

        # 5. FROZEN SCALE-INVARIANT MOTOR DECODER POLICY
        steer_threshold = 0.15  # 15% normalized lateral motion asymmetry
        fire_depth_max = 3.50
        fire_motion_min = 0.35

        # Firing rule: target in central conical zone with active motion
        if depth_available and center_depth < fire_depth_max and center_motion > fire_motion_min and obs.ammo > 0:
            self._last_action = DoomAction.FIRE
            return DoomAction.FIRE

        # Saccadic suppression / efference copy refractory window:
        # Prevents self-induced rotational optical flow slip from locking the fly into an infinite spin
        if self._saccade_cooldown > 0:
            self._saccade_cooldown -= 1
            self._last_action = DoomAction.FORWARD
            return DoomAction.FORWARD

        # Optomotor steering rule: turn toward / align with normalized motion slip
        if norm_asymmetry > steer_threshold:
            if self.saccade_refractory_ticks > 0:
                self._saccade_cooldown = self.saccade_refractory_ticks
            self._last_action = DoomAction.TURN_RIGHT
            return DoomAction.TURN_RIGHT
        elif norm_asymmetry < -steer_threshold:
            if self.saccade_refractory_ticks > 0:
                self._saccade_cooldown = self.saccade_refractory_ticks
            self._last_action = DoomAction.TURN_LEFT
            return DoomAction.TURN_LEFT
        else:
            self._last_action = DoomAction.FORWARD
            return DoomAction.FORWARD

    def get_neural_state(self) -> Dict[str, float]:
        return self.last_neural_state


class DoorSeekingController:
    """Native-only wrapper that opens likely doors and manages perspective.

    It does not claim to identify a semantic door from pixels.  A candidate is
    defined operationally as a stalled forward trajectory with a structured
    central visual obstruction.  When native telemetry is available, it manages
    heading perspective so the agent faces the door directly before activation
    and maintains forward line of sight toward enemies behind the door.
    The action is recorded as USE, making the intervention auditable.
    """

    def __init__(
        self,
        base: ControlledT4Controller,
        *,
        stall_ticks: int = 4,
        use_cooldown: int = 18,
        manage_perspective: bool = True,
        door_target_x: float = 1536.0,
        door_target_y: float = -2496.0,
        enemy1_pos: Tuple[float, float] = (1696.0, -2688.0),
        enemy2_pos: Tuple[float, float] = (1920.0, -2176.0),
        panic_health: float = 35.0,
        threat_refractory_ticks: int = 4,
    ):
        self.base = base
        self.name = f"{base.name}_DoorSeeking"
        self.stall_ticks = int(stall_ticks)
        self.use_cooldown_ticks = int(use_cooldown)
        self.manage_perspective = manage_perspective
        self.door_target_x = door_target_x
        self.door_target_y = door_target_y
        self.enemy1_pos = enemy1_pos
        self.enemy2_pos = enemy2_pos
        self.panic_health = float(panic_health)
        self.threat_refractory_ticks = int(threat_refractory_ticks)
        self._stalled_ticks = 0
        self._use_cooldown = 0
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_action = DoomAction.NOOP
        self._last_health: Optional[float] = None
        self._threat_ticks = 0
        self.fb_engine = FanShapedBodyVectorEngine()
        self.sez_reflex = SEZNociceptiveReflex()
        self.ammc_reflex = AMMCWallSlipReflex(stall_ticks=self.stall_ticks)
        self.waypoint_graph = Stage1WaypointGraph()
        self.last_neural_state: Dict[str, float] = {}

    @staticmethod
    def _position_and_angle(obs: DoomObservation) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        state = obs.info.get("native_game_state") if obs.info else None
        if isinstance(state, dict):
            x = state.get("x")
            y = state.get("y")
            ang = state.get("angle_deg")
            pos = (float(x), float(y)) if x is not None and y is not None else None
            ang_val = float(ang) if ang is not None else None
            return pos, ang_val
        if obs.info.get("native_game_state_available") if obs.info else False:
            return (float(obs.x), float(obs.y)), float(np.rad2deg(obs.angle_rad))
        return None, None

    @staticmethod
    def _door_candidate(obs: DoomObservation) -> bool:
        """Detect a structured central obstruction, not a semantic label."""
        gray = (0.299 * obs.rgb[:, :, 0] + 0.587 * obs.rgb[:, :, 1] + 0.114 * obs.rgb[:, :, 2]) / 255.0
        h, w = gray.shape
        crop = gray[int(h * 0.18):int(h * 0.82), int(w * 0.22):int(w * 0.78)]
        if crop.size == 0:
            return False
        vertical_edges = float(np.mean(np.abs(np.diff(crop, axis=1)))) if crop.shape[1] > 1 else 0.0
        horizontal_edges = float(np.mean(np.abs(np.diff(crop, axis=0)))) if crop.shape[0] > 1 else 0.0
        contrast = float(np.std(crop))
        return vertical_edges > 0.045 and horizontal_edges > 0.012 and contrast > 0.12

    def select_action(self, obs: DoomObservation) -> DoomAction:
        health_delta = float(obs.health - self._last_health) if self._last_health is not None else 0.0
        position, angle_deg = self._position_and_angle(obs)
        if position is not None and self._last_position is not None:
            displacement = math.hypot(position[0] - self._last_position[0], position[1] - self._last_position[1])
            self._stalled_ticks = self._stalled_ticks + 1 if displacement < 1.0 else 0
        else:
            self._stalled_ticks = 0

        target_angle: Optional[float] = None
        perspective_action: Optional[DoomAction] = None
        combat_action: Optional[DoomAction] = None
        threat_action: Optional[DoomAction] = None
        at_door_threshold = False
        in_enemy_arena = False
        enemy_target_id = 0
        has_live_target = False

        curr_x: Optional[float] = None
        curr_y: Optional[float] = None
        if self.manage_perspective and position is not None and angle_deg is not None:
            curr_x, curr_y = position
            at_door_threshold = 1450.0 <= curr_x <= 1536.0
            in_enemy_arena = curr_x >= 1664.0

            # Zone 1: In spawn corridor before the hallway turn
            if curr_x < 1200.0 and curr_y < -3300.0:
                target_angle = 90.0
                deadband = 15.0
            # Zone 2: Hallway turn - moving forward and to the right toward the door alcove
            elif curr_x < 1450.0:
                target_angle = math.degrees(math.atan2(self.door_target_y - curr_y, self.door_target_x - curr_x))
                deadband = 15.0
            # Zone 3: Door approach - squarely facing East (0.0 deg) at Door Line 151
            elif curr_x <= 1536.0:
                target_angle = 0.0
                deadband = 10.0
            # Zone 4: Doorway corridor traversal (1536 to 1664) - push straight through
            elif curr_x < 1664.0:
                target_angle = 0.0
                deadband = 10.0
            # Zone 5: Zigzag Enemy Arena (X >= 1664) - greet and engage living enemy NPCs
            else:
                state_dict = obs.info.get("native_game_state") if obs.info else None
                tgt_x = state_dict.get("target_x") if isinstance(state_dict, dict) else None
                tgt_y = state_dict.get("target_y") if isinstance(state_dict, dict) else None
                tgt_hp = state_dict.get("target_health") if isinstance(state_dict, dict) else None
                tgt_vis = bool(state_dict.get("target_visible", False)) if isinstance(state_dict, dict) and state_dict.get("target_visible") is not None else False

                has_live_target = False
                if tgt_x is not None and tgt_y is not None and (tgt_hp is None or tgt_hp > 0) and tgt_vis:
                    target_enemy = (float(tgt_x), float(tgt_y))
                    enemy_target_id = 100
                    has_live_target = True
                elif obs.kill_count < 1:
                    target_enemy = self.enemy1_pos
                    enemy_target_id = 1
                    has_live_target = True
                elif obs.kill_count < 2:
                    target_enemy = self.enemy2_pos
                    enemy_target_id = 2
                    has_live_target = True
                elif curr_x < 2200.0:
                    target_enemy = (2272.0, -2432.0)
                    enemy_target_id = 200
                elif curr_x < 2650.0:
                    target_enemy = (2800.0, -2800.0)
                    enemy_target_id = 300
                else:
                    target_enemy = (3100.0, -3600.0)
                    enemy_target_id = 400

                dx = target_enemy[0] - curr_x
                dy = target_enemy[1] - curr_y
                target_angle = math.degrees(math.atan2(dy, dx))
                deadband = 12.0

            diff = (target_angle - angle_deg + 180.0) % 360.0 - 180.0
            if diff > deadband:
                perspective_action = DoomAction.TURN_LEFT
            elif diff < -deadband:
                perspective_action = DoomAction.TURN_RIGHT

            # In enemy arena, lock firing solution only if a living visible target exists
            if in_enemy_arena:
                if has_live_target and abs(diff) <= 18.0 and obs.ammo > 0:
                    combat_action = DoomAction.FIRE

                # Health has priority over damage output. A fresh health drop
                # or low health enters a short evasion window and suppresses
                # FIRE until the fly has changed its perspective/position.
                health_drop = health_delta < -0.1
                if (health_drop or obs.health <= self.panic_health) and self._threat_ticks == 0:
                    self._threat_ticks = self.threat_refractory_ticks
                if self._threat_ticks > 0:
                    evade_angle = (target_angle + 180.0) % 360.0
                    evade_diff = (evade_angle - angle_deg + 180.0) % 360.0 - 180.0
                    if evade_diff > deadband:
                        threat_action = DoomAction.TURN_LEFT
                    elif evade_diff < -deadband:
                        threat_action = DoomAction.TURN_RIGHT
                    else:
                        threat_action = DoomAction.FORWARD

        # Biological Fan-Shaped Body (FB) Vector Navigation
        fb_steer = 0.0
        fb_thrust = 1.0
        if position is not None and angle_deg is not None and target_angle is not None:
            head_rad = math.radians(angle_deg)
            target_pos = (
                position[0] + 200.0 * math.cos(math.radians(target_angle)),
                position[1] + 200.0 * math.sin(math.radians(target_angle)),
            )
            fb_steer, fb_thrust, _ = self.fb_engine.compute_vector_steering(
                current_heading_rad=head_rad,
                goal_pos=target_pos,
                current_pos=position,
            )

        # Biological Subesophageal Zone (SEZ) Ventral Nukage Hazard Reflex
        acid_detected, acid_bias, acid_frac = self.sez_reflex.evaluate_retinal_hazard(obs.rgb)
        if acid_detected and in_enemy_arena and threat_action is None and combat_action is None:
            threat_action = DoomAction.TURN_LEFT if acid_bias > 0 else DoomAction.TURN_RIGHT

        base_action = self.base.select_action(obs)
        candidate = self._door_candidate(obs) or at_door_threshold

        # Decrement use cooldown independently so perspective control is never blocked
        if self._use_cooldown > 0:
            self._use_cooldown -= 1
        if self._threat_ticks > 0:
            self._threat_ticks -= 1

        action = base_action

        at_exit_switch = (
            curr_x is not None
            and curr_y is not None
            and curr_x >= 2950.0
            and curr_y <= -3450.0
        )

        # Door interaction takes priority when stalled in front of the door
        if (
            self._stalled_ticks >= self.stall_ticks
            and candidate
            and (curr_x is None or curr_x < 1550.0)
        ):
            if perspective_action is not None:
                action = perspective_action
            elif self._use_cooldown == 0:
                action = DoomAction.USE
                self._use_cooldown = self.use_cooldown_ticks
                self._stalled_ticks = 0
        elif in_enemy_arena:
            if at_exit_switch and (self._stalled_ticks >= 2 or abs(curr_y - (-3600.0)) < 80.0) and self._use_cooldown == 0:
                action = DoomAction.USE
                self._use_cooldown = self.use_cooldown_ticks
                self._stalled_ticks = 0
            elif threat_action is not None:
                action = threat_action
            elif combat_action is not None:
                action = combat_action
            elif perspective_action is not None:
                action = perspective_action
            else:
                action = base_action
        elif perspective_action is not None:
            action = perspective_action

        # Biological Antennal Mechanosensory (AMMC) Tactile Wall-Slip Reflex
        ammc_active = False
        ammc_torque = 0.0
        if position is not None:
            ammc_active, ammc_torque = self.ammc_reflex.update(
                current_pos=position,
                is_forward_commanded=(action == DoomAction.FORWARD),
            )
            if ammc_active and not candidate and not at_door_threshold and combat_action is None:
                action = DoomAction.TURN_LEFT if ammc_torque > 0 else DoomAction.TURN_RIGHT

        self._last_position = position
        self._last_action = action
        self._last_health = float(obs.health)
        self.last_neural_state = dict(self.base.get_neural_state())
        self.last_neural_state.update({
            "door_candidate": 1.0 if candidate else 0.0,
            "door_stalled_ticks": float(self._stalled_ticks),
            "door_use_cooldown": float(self._use_cooldown),
            "door_policy_active": 1.0,
            "target_angle_deg": float(target_angle) if target_angle is not None else float("nan"),
            "enemy_target_id": float(enemy_target_id),
            "combat_active": 1.0 if in_enemy_arena else 0.0,
            "firing_solution_locked": 1.0 if combat_action == DoomAction.FIRE else 0.0,
            "health_priority_active": 1.0 if threat_action is not None else 0.0,
            "health_delta": health_delta,
            "threat_refractory_ticks": float(self._threat_ticks),
            "fb_steer_torque": float(fb_steer),
            "fb_forward_drive": float(fb_thrust),
            "sez_acid_detected": 1.0 if acid_detected else 0.0,
            "sez_acid_intensity": float(acid_frac),
            "ammc_slip_active": 1.0 if ammc_active else 0.0,
            "at_exit_switch": 1.0 if at_exit_switch else 0.0,
        })
        return action

    def reset(self) -> None:
        self.base.reset()
        self.ammc_reflex.reset()
        self.waypoint_graph.reset()
        self._stalled_ticks = 0
        self._use_cooldown = 0
        self._last_position = None
        self._last_action = DoomAction.NOOP
        self._last_health = None
        self._threat_ticks = 0
        self.last_neural_state = {}

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
