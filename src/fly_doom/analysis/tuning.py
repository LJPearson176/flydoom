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
    dsi: float  # Discrete DSI: (R_max - R_null) / (R_max + R_null)
    discrete_pref_direction_deg: float  # Direction corresponding to R_max
    discrete_null_direction_deg: float  # Opposite sampled direction (discrete_pref + 180 mod 360)
    preferred_direction_deg: Optional[float]  # Vector summation angle in [0, 360) or None if unreliable
    vector_magnitude: float  # Resultant vector length ||R_vec||
    vector_strength: float  # Normalized resultant vector strength: ||R_vec|| / sum(R)
    vector_pd_reliable: bool  # Reliability gate: vector_strength >= min_vector_strength
    circular_variance: float  # Circular variance in [0.0, 1.0]
    best_cardinal_direction_deg: float  # Nearest of {0, 90, 180, 270}
    cardinal_error_deg: float  # Absolute distance to nearest cardinal direction
    peak_response: float = 0.0  # Max response across directions
    baseline_response: float = 0.0  # Min response across directions
    modulation_index: float = 0.0  # (R_peak - R_baseline) / (R_peak + R_baseline + eps)
    contrast_ratio: float = 1.0  # R_peak / (R_baseline + eps)
    response_latency_ms: Optional[float] = None  # Latency to peak response
    trial_variance: float = 0.0  # Variance across tested directional responses
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
    response_latency_ms: Optional[float] = None,
    min_vector_strength: float = 0.10,
    epsilon: float = 1e-6,
) -> DirectionalTuningResult:
    """Compute decoupled discrete DSI, 2D vector summation tuning, and reliability.

    1. Discrete DSI:
       pref_idx = argmax(R)
       null_idx = (pref_idx + M // 2) % M
       DSI = (R[pref_idx] - R[null_idx]) / (R[pref_idx] + R[null_idx] + eps)

    2. Vector Direction & Strength:
       R_x = 1/M * sum(R_m * cos(theta_m))
       R_y = 1/M * sum(R_m * sin(theta_m))
       V_mag = sqrt(R_x^2 + R_y^2)
       Vector_Strength = sqrt((sum R cos)^2 + (sum R sin)^2) / sum(R)
       PD = atan2(R_y, R_x) mod 360 if Vector_Strength >= min_vector_strength else None
    """
    m = len(directions_deg)
    assert m == len(responses), "Directions and responses must have identical length"

    angles_rad = [math.radians(d) for d in directions_deg]
    r_arr = np.array(responses, dtype=np.float64)
    total_r = float(np.sum(r_arr))

    # 1. Discrete Preferred Direction & DSI
    pref_idx = int(np.argmax(r_arr))
    null_idx = (pref_idx + m // 2) % m
    r_pref_discrete = float(r_arr[pref_idx])
    r_null_discrete = float(r_arr[null_idx])
    discrete_pref_deg = float(directions_deg[pref_idx])
    discrete_null_deg = float(directions_deg[null_idx])

    discrete_dsi = (r_pref_discrete - r_null_discrete) / (r_pref_discrete + r_null_discrete + epsilon)

    # 2. 2D Vector Summation & Vector Strength
    r_sum_x = float(np.sum(r_arr * np.cos(angles_rad)))
    r_sum_y = float(np.sum(r_arr * np.sin(angles_rad)))
    resultant_len = math.sqrt(r_sum_x**2 + r_sum_y**2)

    r_x = r_sum_x / m
    r_y = r_sum_y / m
    v_mag = math.sqrt(r_x**2 + r_y**2)

    vector_strength = (resultant_len / (total_r + epsilon)) if total_r > epsilon else 0.0
    is_reliable = vector_strength >= min_vector_strength

    if is_reliable and v_mag > epsilon:
        pd_deg: Optional[float] = math.degrees(math.atan2(r_y, r_x)) % 360.0
    else:
        pd_deg = None

    # 3. Circular Variance: 1 - (||sum R e^{j theta}|| / sum R)
    if total_r > epsilon:
        circular_variance = max(0.0, min(1.0, 1.0 - (resultant_len / total_r)))
    else:
        circular_variance = 1.0

    # 4. Nearest Cardinal Direction {0, 90, 180, 270}
    # Evaluated against discrete_pref_deg or reliable pd_deg
    eval_angle = pd_deg if pd_deg is not None else discrete_pref_deg
    cardinals = [0.0, 90.0, 180.0, 270.0]
    card_diffs = [abs((eval_angle - c + 180.0) % 360.0 - 180.0) for c in cardinals]
    best_card_idx = int(np.argmin(card_diffs))
    best_cardinal = cardinals[best_card_idx]
    card_error = card_diffs[best_card_idx]

    peak_r = float(np.max(r_arr)) if len(r_arr) > 0 else 0.0
    baseline_r = float(np.min(r_arr)) if len(r_arr) > 0 else 0.0
    trial_var = float(np.var(r_arr)) if len(r_arr) > 0 else 0.0

    # Modulation index & contrast ratio
    mod_idx = (peak_r - baseline_r) / (peak_r + baseline_r + epsilon) if (peak_r + baseline_r) > 0 else 0.0
    contrast_r = (peak_r / (baseline_r + epsilon)) if baseline_r > 0 else (peak_r / epsilon if peak_r > 0 else 1.0)

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
        dsi=discrete_dsi,
        discrete_pref_direction_deg=discrete_pref_deg,
        discrete_null_direction_deg=discrete_null_deg,
        preferred_direction_deg=pd_deg,
        vector_magnitude=v_mag,
        vector_strength=vector_strength,
        vector_pd_reliable=is_reliable,
        circular_variance=circular_variance,
        best_cardinal_direction_deg=best_cardinal,
        cardinal_error_deg=card_error,
        peak_response=peak_r,
        baseline_response=baseline_r,
        modulation_index=mod_idx,
        contrast_ratio=contrast_r,
        response_latency_ms=response_latency_ms,
        trial_variance=trial_var,
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
    mean_vector_strength: float
    fraction_pd_reliable: float
    mean_circular_variance: float
    mean_cardinal_error_deg: float
    mean_peak_response: float
    mean_baseline_response: float
    mean_modulation_index: float
    mean_contrast_ratio: float
    mean_trial_variance: float
    mean_latency_ms: Optional[float]
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
                mean_vector_strength=0.0,
                fraction_pd_reliable=0.0,
                mean_circular_variance=1.0,
                mean_cardinal_error_deg=45.0,
                mean_peak_response=0.0,
                mean_baseline_response=0.0,
                mean_modulation_index=0.0,
                mean_contrast_ratio=1.0,
                mean_trial_variance=0.0,
                mean_latency_ms=None,
                mean_polarity_index=None,
                results=[],
            )

        dsi_vals = [r.dsi for r in results]
        v_strengths = [r.vector_strength for r in results]
        reliables = [1.0 if r.vector_pd_reliable else 0.0 for r in results]
        circ_vars = [r.circular_variance for r in results]
        card_errs = [r.cardinal_error_deg for r in results]
        peaks = [r.peak_response for r in results]
        baselines = [r.baseline_response for r in results]
        mod_indices = [r.modulation_index for r in results]
        contrast_ratios = [r.contrast_ratio for r in results]
        variances = [r.trial_variance for r in results]

        latencies = [r.response_latency_ms for r in results if r.response_latency_ms is not None]
        mean_lat = float(np.mean(latencies)) if latencies else None

        pol_vals = [r.polarity_index for r in results if r.polarity_index is not None]
        mean_pol = float(np.mean(pol_vals)) if pol_vals else None

        return cls(
            population_name=population_name,
            num_neurons=len(results),
            mean_dsi=float(np.mean(dsi_vals)),
            median_dsi=float(np.median(dsi_vals)),
            mean_vector_strength=float(np.mean(v_strengths)),
            fraction_pd_reliable=float(np.mean(reliables)),
            mean_circular_variance=float(np.mean(circ_vars)),
            mean_cardinal_error_deg=float(np.mean(card_errs)),
            mean_peak_response=float(np.mean(peaks)),
            mean_baseline_response=float(np.mean(baselines)),
            mean_modulation_index=float(np.mean(mod_indices)),
            mean_contrast_ratio=float(np.mean(contrast_ratios)),
            mean_trial_variance=float(np.mean(variances)),
            mean_latency_ms=mean_lat,
            mean_polarity_index=mean_pol,
            results=results,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_name": self.population_name,
            "num_neurons": self.num_neurons,
            "mean_dsi": round(self.mean_dsi, 4),
            "median_dsi": round(self.median_dsi, 4),
            "mean_vector_strength": round(self.mean_vector_strength, 4),
            "fraction_pd_reliable": round(self.fraction_pd_reliable, 4),
            "mean_circular_variance": round(self.mean_circular_variance, 4),
            "mean_cardinal_error_deg": round(self.mean_cardinal_error_deg, 2),
            "mean_peak_response": round(self.mean_peak_response, 2),
            "mean_baseline_response": round(self.mean_baseline_response, 2),
            "mean_modulation_index": round(self.mean_modulation_index, 4),
            "mean_contrast_ratio": round(self.mean_contrast_ratio, 2),
            "mean_trial_variance": round(self.mean_trial_variance, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2)
            if self.mean_latency_ms is not None
            else None,
            "mean_polarity_index": round(self.mean_polarity_index, 4)
            if self.mean_polarity_index is not None
            else None,
        }
