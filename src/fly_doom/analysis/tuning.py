"""Directional tuning curve, 2D vector summation, and DSI analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class DirectionalTuningResult:
    """Tuning metrics for a single neuron tested across an angular sweep."""

    neuron_idx: int
    cell_type: str
    directions_deg: List[float]
    response_curve: List[float]  # Firing rate or spike count per direction
    preferred_direction_deg: float  # Angle from vector summation in [0, 360)
    vector_magnitude: float  # Length of mean resultant vector
    dsi: float  # Directional Selectivity Index: (R_pref - R_null) / (R_pref + R_null)
    circular_variance: float  # Circular variance in [0.0, 1.0]
    best_cardinal_direction_deg: float  # Nearest of {0, 90, 180, 270}
    cardinal_error_deg: float  # Absolute distance to nearest cardinal direction
    on_response: Optional[float] = None
    off_response: Optional[float] = None
    polarity_index: Optional[float] = None  # (R_on - R_off) / (R_on + R_off)


def compute_vector_tuning(
    neuron_idx: int,
    cell_type: str,
    directions_deg: Sequence[float],
    responses: Sequence[float],
    on_response: Optional[float] = None,
    off_response: Optional[float] = None,
    epsilon: float = 1e-6,
) -> DirectionalTuningResult:
    """Compute 2D vector summation tuning, preferred direction, and DSI.

    Formulation:
      R_vec = 1/M * sum_m R(theta_m) * [cos(theta_m), sin(theta_m)]
      PD = atan2(R_y, R_x) mod 360
      DSI = (R_pref - R_null) / (R_pref + R_null)
    """
    m = len(directions_deg)
    assert m == len(responses), "Directions and responses must have identical length"

    angles_rad = [math.radians(d) for d in directions_deg]
    r_arr = np.array(responses, dtype=np.float64)

    # 1. 2D Vector Summation
    r_x = float(np.sum(r_arr * np.cos(angles_rad))) / m
    r_y = float(np.sum(r_arr * np.sin(angles_rad))) / m
    v_mag = math.sqrt(r_x * r_x + r_y * r_y)

    total_r = float(np.sum(r_arr))

    # Preferred Direction
    if v_mag > epsilon:
        pd_deg = math.degrees(math.atan2(r_y, r_x)) % 360.0
    else:
        # Unresponsive or completely isotropic
        pd_deg = 0.0

    # 2. Circular Variance: 1 - (||sum R e^{j theta}|| / sum R)
    if total_r > epsilon:
        # Resultant vector length normalized by total response
        r_sum_x = float(np.sum(r_arr * np.cos(angles_rad)))
        r_sum_y = float(np.sum(r_arr * np.sin(angles_rad)))
        resultant_len = math.sqrt(r_sum_x**2 + r_sum_y**2)
        circular_variance = max(0.0, min(1.0, 1.0 - (resultant_len / total_r)))
    else:
        circular_variance = 1.0

    # 3. DSI using nearest tested sample angle to PD
    # Find sampled angle closest to PD
    angle_diffs = [abs((d - pd_deg + 180.0) % 360.0 - 180.0) for d in directions_deg]
    best_sample_idx = int(np.argmin(angle_diffs))
    r_pref = float(r_arr[best_sample_idx])

    # Null direction is (PD + 180) mod 360
    null_angle = (directions_deg[best_sample_idx] + 180.0) % 360.0
    null_diffs = [abs((d - null_angle + 180.0) % 360.0 - 180.0) for d in directions_deg]
    null_sample_idx = int(np.argmin(null_diffs))
    r_null = float(r_arr[null_sample_idx])

    dsi = (r_pref - r_null) / (r_pref + r_null + epsilon)

    # 4. Nearest Cardinal Direction {0, 90, 180, 270}
    cardinals = [0.0, 90.0, 180.0, 270.0]
    card_diffs = [abs((pd_deg - c + 180.0) % 360.0 - 180.0) for c in cardinals]
    best_card_idx = int(np.argmin(card_diffs))
    best_cardinal = cardinals[best_card_idx]
    card_error = card_diffs[best_card_idx]

    # 5. Polarity Index (ON vs OFF)
    polarity_idx = None
    if on_response is not None and off_response is not None:
        denom = on_response + off_response + epsilon
        polarity_idx = (on_response - off_response) / denom

    return DirectionalTuningResult(
        neuron_idx=neuron_idx,
        cell_type=cell_type,
        directions_deg=list(directions_deg),
        response_curve=list(r_arr),
        preferred_direction_deg=pd_deg,
        vector_magnitude=v_mag,
        dsi=dsi,
        circular_variance=circular_variance,
        best_cardinal_direction_deg=best_cardinal,
        cardinal_error_deg=card_error,
        on_response=on_response,
        off_response=off_response,
        polarity_index=polarity_idx,
    )


@dataclass
class PopulationTuningSummary:
    """Summary of tuning metrics across an entire neural population (e.g. T4 or T5)."""

    population_name: str
    num_neurons: int
    mean_dsi: float
    median_dsi: float
    mean_circular_variance: float
    mean_cardinal_error_deg: float
    mean_polarity_index: Optional[float]
    results: List[DirectionalTuningResult]

    @classmethod
    def from_results(
        cls, population_name: str, results: List[DirectionalTuningResult]
    ) -> PopulationTuningSummary:
        if not results:
            return cls(
                population_name=population_name,
                num_neurons=0,
                mean_dsi=0.0,
                median_dsi=0.0,
                mean_circular_variance=1.0,
                mean_cardinal_error_deg=45.0,
                mean_polarity_index=None,
                results=[],
            )

        dsi_vals = [r.dsi for r in results]
        circ_vars = [r.circular_variance for r in results]
        card_errs = [r.cardinal_error_deg for r in results]

        pol_vals = [r.polarity_index for r in results if r.polarity_index is not None]
        mean_pol = float(np.mean(pol_vals)) if pol_vals else None

        return cls(
            population_name=population_name,
            num_neurons=len(results),
            mean_dsi=float(np.mean(dsi_vals)),
            median_dsi=float(np.median(dsi_vals)),
            mean_circular_variance=float(np.mean(circ_vars)),
            mean_cardinal_error_deg=float(np.mean(card_errs)),
            mean_polarity_index=mean_pol,
            results=results,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_name": self.population_name,
            "num_neurons": self.num_neurons,
            "mean_dsi": round(self.mean_dsi, 4),
            "median_dsi": round(self.median_dsi, 4),
            "mean_circular_variance": round(self.mean_circular_variance, 4),
            "mean_cardinal_error_deg": round(self.mean_cardinal_error_deg, 2),
            "mean_polarity_index": round(self.mean_polarity_index, 4)
            if self.mean_polarity_index is not None
            else None,
        }
