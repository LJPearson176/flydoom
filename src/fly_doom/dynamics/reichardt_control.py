"""Pure synthetic Hassenstein-Reichardt Elementary Motion Detector (EMD) control.

Validates the mathematical coincidence-detection logic in isolation before
application to connectome-scale recurrent networks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from fly_doom.core.provenance import Provenance


@dataclass(frozen=True)
class ReichardtParameters:
    """Parameters for synthetic Reichardt correlator."""

    tau_delay_ms: float = 30.0  # Low-pass filter time constant for delayed arm
    dt_ms: float = 1.0  # Time step
    epsilon: float = 1e-6
    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            tier="computational_hypothesis",
            source="Hassenstein_Reichardt_1956",
            confidence=1.0,
            rationale="Classical two-input Elementary Motion Detector (EMD) benchmark",
            doi="10.1515/znc-1956-9-1007",
        )
    )

    @property
    def alpha_delay(self) -> float:
        return math.exp(-self.dt_ms / self.tau_delay_ms)


class SyntheticReichardtCorrelator:
    """Two-input Elementary Motion Detector (EMD) measuring rightwards vs leftwards motion."""

    def __init__(self, params: Optional[ReichardtParameters] = None):
        self.params = params or ReichardtParameters()
        self.delayed_a: float = 0.0
        self.delayed_b: float = 0.0

    def reset(self) -> None:
        self.delayed_a = 0.0
        self.delayed_b = 0.0

    def step(self, input_a: float, input_b: float) -> Tuple[float, float, float]:
        """Execute one time step.

        Input A is at spatial position x=0 (left).
        Input B is at spatial position x=1 (right).

        Returns:
          (m_right, m_left, net_motion)
        """
        p = self.params

        # Update low-pass filtered delay lines: D(t) = alpha * D(t-1) + (1 - alpha) * input
        self.delayed_a = p.alpha_delay * self.delayed_a + (1.0 - p.alpha_delay) * input_a
        self.delayed_b = p.alpha_delay * self.delayed_b + (1.0 - p.alpha_delay) * input_b

        # Half-detectors (Reichardt multiplication of delayed arm with non-delayed arm)
        # Rightward motion (A -> B): Delayed A multiplied by current B
        m_right = self.delayed_a * input_b

        # Leftward motion (B -> A): Delayed B multiplied by current A
        m_left = self.delayed_b * input_a

        # Net motion (opponent subtraction)
        net_motion = m_right - m_left

        return m_right, m_left, net_motion

    def test_motion(
        self,
        velocity_px_s: float = 20.0,
        pixel_spacing: float = 1.0,
        duration_ms: float = 200.0,
    ) -> Tuple[float, float, float]:
        """Simulate a moving pulse across inputs A and B.

        Returns:
          (total_right_response, total_left_response, dsi)
        """
        dt_s = self.params.dt_ms / 1000.0
        total_steps = int(round(duration_ms / self.params.dt_ms))
        pulse_width_s = 0.02  # 20ms pulse width

        # Transit time between A (x=0) and B (x=pixel_spacing)
        transit_time_s = pixel_spacing / velocity_px_s

        # 1. Test Rightwards Motion (A at t=0.04s, B at t=0.04s + transit_time)
        self.reset()
        t_a_on = 0.04
        t_b_on = t_a_on + transit_time_s

        total_r_under_right = 0.0
        total_l_under_right = 0.0

        for s in range(total_steps):
            t = s * dt_s
            in_a = 1.0 if (t_a_on <= t < t_a_on + pulse_width_s) else 0.0
            in_b = 1.0 if (t_b_on <= t < t_b_on + pulse_width_s) else 0.0

            m_r, m_l, _ = self.step(in_a, in_b)
            total_r_under_right += m_r
            total_l_under_right += m_l

        # 2. Test Leftwards Motion (B at t=0.04s, A at t=0.04s + transit_time)
        self.reset()
        t_b_on_left = 0.04
        t_a_on_left = t_b_on_left + transit_time_s

        total_r_under_left = 0.0
        total_l_under_left = 0.0

        for s in range(total_steps):
            t = s * dt_s
            in_b = 1.0 if (t_b_on_left <= t < t_b_on_left + pulse_width_s) else 0.0
            in_a = 1.0 if (t_a_on_left <= t < t_a_on_left + pulse_width_s) else 0.0

            m_r, m_l, _ = self.step(in_a, in_b)
            total_r_under_left += m_r
            total_l_under_left += m_l

        # DSI of the Right-tuned half-detector: (Response_Right - Response_Left) / (Response_Right + Response_Left)
        r_pref = total_r_under_right
        r_null = total_r_under_left
        dsi = (r_pref - r_null) / (r_pref + r_null + self.params.epsilon)

        return total_r_under_right, total_r_under_left, dsi
