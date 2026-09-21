import numpy as np
import yaml

from src.extra.nc_excel_experiments.metrics import (
    conditional_mean_pairwise_rho,
    directional_delivery,
    effective_n,
    failure_threshold,
    network_delivery_metrics,
    sign_consistency,
    wilson_interval,
)
from src.extra.nc_excel_experiments.run import DEFAULT_PROTOCOL, _legal_n_values
from src.extra.nc_excel_experiments.local_policy import ParameterizedLocalPolicy


def test_conditional_rho_removes_fixed_condition_mean():
    rng = np.random.default_rng(7)
    condition_mean = np.arange(6, dtype=float)[None, :, None]
    independent_noise = rng.normal(size=(40, 6, 20))
    rho = conditional_mean_pairwise_rho(condition_mean + independent_noise)
    assert abs(rho) < 0.03


def test_effective_n_uses_only_positive_correlation():
    assert effective_n(100, -0.2) == 100
    assert effective_n(100, 0.01) < 100


def test_directional_metrics_reject_sign_reversal():
    desired = np.array([10.0, -10.0, 10.0])
    accepted = np.array([8.0, -7.0, -5.0])
    assert np.allclose(directional_delivery(desired, accepted), [8.0, 7.0, 0.0])
    assert sign_consistency(accepted, desired) == 2 / 3


def test_network_metrics_include_safety_margins():
    diagnostics = [{
        "network_scale": 0.8,
        "maximum_branch_loading": 0.9,
        "transformer_loading": 0.7,
        "minimum_voltage_pu": 0.96,
        "maximum_voltage_pu": 1.01,
        "overloaded_branches": 0,
        "voltage_violations": 0,
    }]
    result = network_delivery_metrics(np.array([10.0]), np.array([8.0]), diagnostics)
    assert np.isclose(result["network_acceptance_ratio"], 0.8)
    assert result["line_margin"] > 0
    assert result["voltage_margin"] > 0


def test_wilson_interval_and_failure_threshold():
    lower, upper = wilson_interval(9, 10)
    assert 0 < lower < 0.9 < upper <= 1
    assert np.isclose(failure_threshold([0, 1, 2], [0.0, 0.4, 0.8]), 1.25)


def test_legal_n_values_keeps_small_source_unique_datasets():
    assert _legal_n_values([50, 100, 200], 6, include_maximum=True) == [1, 2, 4, 5, 6]
    assert _legal_n_values([50, 100, 200], 246, include_maximum=True) == [50, 100, 200, 246]
    assert _legal_n_values([1, 5, 50], 50, include_maximum=True) == [1, 5, 50]
    assert _legal_n_values([50, 100], 0, include_maximum=True) == []


def test_excel_protocol_uses_all_datasets_and_only_requested_reruns():
    protocol = yaml.safe_load(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    assert len(protocol["e1"]["datasets"]) == 15
    assert protocol["e1"]["datasets"] == protocol["e2"]["datasets"]
    requested = protocol["execution"]["experiments"]
    assert requested
    assert all(name.lower() in protocol for name in requested)


def test_local_policy_homogeneity_replaces_all_controller_parameters():
    class Record:
        def __init__(self, zone_id: str):
            self.zone_id = zone_id

    records = [Record("z0") for _ in range(8)]
    policy = ParameterizedLocalPolicy(
        records, homogeneity=1.0, delay_distribution="uniform", seed=7, steps=4
    )
    assert np.all(policy.signal_threshold == 0.30)
    assert np.all(policy.deadband == 0.05)
    assert np.all(policy.gain == 1.0)
    assert np.all(policy.delay == 2)
