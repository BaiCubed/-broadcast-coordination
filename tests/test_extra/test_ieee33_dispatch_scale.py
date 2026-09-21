from src.extra.ieee33_device_day_simulation.network.ieee33_distflow import IEEE33DistFlow


def _network() -> IEEE33DistFlow:
    return IEEE33DistFlow(
        {
            "slack_bus": 1,
            "voltage_limits_pu": [0.95, 1.05],
            "transformer": {"bus": 1, "capacity_kw": 100.0},
            "branches": [[1, 2, 0.1, 0.0, 100.0]],
        },
        {
            "network_capacity_multiplier": 1.0,
            "voltage_sensitivity": 0.0,
            "reverse_flow_sensitivity": 0.0,
            "enforce_feeder_capacity": True,
            "enforce_transformer_limit": True,
            "enforce_voltage_limits": True,
        },
    )


def test_dispatch_scale_finds_exact_largest_feasible_fraction() -> None:
    scale, evaluation = _network().constrained_dispatch_scale(
        {2: 20.0}, {}, {2: 200.0}
    )

    assert abs(scale - 0.4) < 1e-12
    assert abs(evaluation.branch_loading["1-2"] - 1.0) < 1e-12


def test_dispatch_can_repair_an_infeasible_baseline() -> None:
    scale, evaluation = _network().constrained_dispatch_scale(
        {2: 120.0}, {}, {2: -100.0}
    )

    assert scale == 1.0
    assert evaluation.branch_loading["1-2"] < 1.0


def test_dispatch_returns_zero_when_no_fraction_is_feasible() -> None:
    scale, _ = _network().constrained_dispatch_scale(
        {2: 120.0}, {}, {2: 10.0}
    )

    assert scale == 0.0


def test_capacity_multiplier_scales_per_unit_voltage_drop() -> None:
    network_config = {
        "slack_bus": 1,
        "voltage_limits_pu": [0.95, 1.05],
        "transformer": {"bus": 1, "capacity_kw": 100.0},
        "branches": [[1, 2, 1.0, 0.0, 100.0]],
    }
    base_control = {
        "voltage_sensitivity": 0.01,
        "reverse_flow_sensitivity": 0.0,
        "enforce_feeder_capacity": True,
        "enforce_transformer_limit": True,
        "enforce_voltage_limits": True,
    }
    base = IEEE33DistFlow(
        network_config,
        {**base_control, "network_capacity_multiplier": 1.0},
    )
    doubled = IEEE33DistFlow(
        network_config,
        {**base_control, "network_capacity_multiplier": 2.0},
    )

    base_voltage = base.evaluate({2: 100.0}, {}, {}, apply_limits=False).voltage_pu[2]
    doubled_voltage = doubled.evaluate({2: 100.0}, {}, {}, apply_limits=False).voltage_pu[2]

    assert abs((1.0 - doubled_voltage) - (1.0 - base_voltage) / 2.0) < 1e-12
