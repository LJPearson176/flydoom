"""Analysis tools: directional tuning, circular variance, and population statistics."""

from fly_doom.analysis.tuning import (
    DirectionalTuningResult,
    PopulationTuningSummary,
    compute_vector_tuning,
)
from fly_doom.analysis.lesion_benchmark import (
    LesionConditionMetrics,
    WelchTestResult,
    compute_welch_ttest,
    load_or_generate_lesion_data,
    run_lesion_statistical_suite,
)

__all__ = [
    "DirectionalTuningResult",
    "PopulationTuningSummary",
    "compute_vector_tuning",
    "LesionConditionMetrics",
    "WelchTestResult",
    "compute_welch_ttest",
    "load_or_generate_lesion_data",
    "run_lesion_statistical_suite",
]
