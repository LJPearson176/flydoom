"""Unit tests for multi-seed lesion benchmarking and Welch's t-test significance."""

import pytest

from fly_doom.analysis.lesion_benchmark import (
    compute_welch_ttest,
    load_or_generate_lesion_data,
    run_lesion_statistical_suite,
)


def test_welch_ttest_computation():
    """Verify Welch's t-test returns correct t-statistic and p-value."""
    control = [0.714, 0.728, 0.709]
    lesion = [0.551, 0.542, 0.538]
    res = compute_welch_ttest(control, lesion, metric_name="target_lock", alpha=0.01)

    assert res.metric_name == "target_lock"
    assert res.t_statistic > 20.0
    assert res.p_value < 0.001
    assert res.significant is True
    assert res.control_mean == pytest.approx(0.717, abs=0.01)
    assert res.lesion_mean == pytest.approx(0.543, abs=0.01)


def test_mi4_ko_statistical_significance(tmp_path):
    """Verify statistical significance (p < 0.01) for Mi4-KO targeting stability collapse."""
    data = load_or_generate_lesion_data(
        output_dir=tmp_path,
        seeds=[1001, 1002, 1003],
    )
    assert "ModelD_ActiveTree_Saccade_3" in data
    assert "ModelD_Mi4_KO_Saccade_3" in data
    assert "ModelD_Mi1_KO_Saccade_3" in data
    assert "ModelD_Tm3_KO_Saccade_3" in data

    report = run_lesion_statistical_suite(data, alpha=0.01)
    tests = {
        (t["lesion_condition"], t["metric_name"]): t
        for t in report["welch_tests"]
    }

    # Mi4-KO target lock collapse
    mi4_lock = tests[("ModelD_Mi4_KO_Saccade_3", "target_lock_fraction")]
    assert mi4_lock["significant"] is True
    assert mi4_lock["p_value"] < 0.01
    assert mi4_lock["t_statistic"] > 0.0  # Control > Lesion

    # Mi4-KO optic flow asymmetry collapse (null-direction veto eliminated)
    mi4_asym = tests[("ModelD_Mi4_KO_Saccade_3", "norm_asymmetry_std")]
    assert mi4_asym["significant"] is True
    assert mi4_asym["p_value"] < 0.01
    assert mi4_asym["lesion_mean"] == pytest.approx(0.0, abs=1e-5)

    # Mi1-KO targeting collapse (delayed ON channel lost)
    mi1_lock = tests[("ModelD_Mi1_KO_Saccade_3", "target_lock_fraction")]
    assert mi1_lock["significant"] is True
    assert mi1_lock["p_value"] < 0.01

    assert "WELCH'S T-TEST COMPARISONS" in report["summary_table"]
