from __future__ import annotations

import numpy as np

from src.extra.dataset_experiment.run_constrained_experiment import (
    _baseline_requirement_ratios,
    _constraint_summary,
    _select_constrained_multiplier,
)


def _trace() -> dict[str, list[float]]:
    return {
        "baseline_transformer_loading": [0.10, 0.20, 0.30, 0.40],
        "baseline_maximum_branch_loading": [0.20, 0.40, 0.60, 0.80],
        "baseline_minimum_voltage_pu": [0.99, 0.98, 0.97, 0.96],
        "baseline_maximum_voltage_pu": [1.01, 1.02, 1.03, 1.04],
        "network_scale": [1.0, 0.8, 0.0, 1.0],
        "desired_control_kw": [10.0, 20.0, -30.0, -40.0],
        "accepted_control_kw": [10.0, 16.0, 0.0, -40.0],
    }


def test_requirement_ratio_combines_thermal_and_voltage_limits() -> None:
    ratios = _baseline_requirement_ratios(_trace(), voltage_limits=(0.95, 1.05))
    np.testing.assert_allclose(ratios, [0.2, 0.4, 0.6, 0.8])


def test_multiplier_uses_baseline_quantile_and_never_enlarges_feeder() -> None:
    selection = _select_constrained_multiplier(
        _trace(),
        baseline_multiplier=10.0,
        voltage_limits=(0.95, 1.05),
        quantile=0.5,
        target_loading=1.0,
    )
    assert selection["selected_multiplier"] == 5.0

    oversized = _trace()
    oversized["baseline_maximum_branch_loading"] = [2.0] * 4
    selection = _select_constrained_multiplier(
        oversized,
        baseline_multiplier=10.0,
        voltage_limits=(0.95, 1.05),
    )
    assert selection["selected_multiplier"] == 10.0


def test_constraint_summary_separates_baseline_infeasibility() -> None:
    trace = _trace()
    trace["baseline_maximum_branch_loading"][-1] = 1.2
    summary = _constraint_summary(trace, voltage_limits=(0.95, 1.05))
    assert summary["constrained_samples"] == 2
    assert summary["dispatch_limited_baseline_feasible_samples"] == 2
    assert summary["baseline_infeasible_samples"] == 1
    assert summary["fully_blocked_samples"] == 1
