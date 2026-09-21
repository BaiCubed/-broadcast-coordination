import json
from types import SimpleNamespace

import numpy as np

from src.extra.dataset_experiment.run import (
    RHO_SCAN,
    _balanced_unique_fleet_limit,
    _capacity_multiplier,
    _update_fig5_protocol_metadata,
)
from src.extra.ieee33_device_day_simulation.experiments.run_experiment import (
    _run_level_response_statistics,
    _trace_curtailment,
)
from src.extra.ieee33_device_day_simulation.experiments.protocol import (
    SimulationTrace,
    _sample_original_region_perturbations,
    build_experiments_training_signal_schedule,
    build_experiments_transfer_signal_schedule,
    build_scenario_profile_signal_schedule,
    build_stratified_signal_schedule,
    trace_features,
)
from src.extra.ieee33_device_day_simulation.figures.legacy_compat import (
    _curtailment_sensitivity,
)


def test_balanced_unique_fleet_limit_uses_smallest_zone() -> None:
    records = []
    for zone, count in (("zone_1", 3), ("zone_2", 2), ("zone_3", 4)):
        records.extend(
            SimpleNamespace(zone_id=zone, source_device_id=f"{zone}-{index}")
            for index in range(count)
        )

    maximum, counts = _balanced_unique_fleet_limit(SimpleNamespace(records=records))

    assert counts == {"zone_1": 3, "zone_2": 2, "zone_3": 4}
    assert maximum == 6


def test_capacity_multiplier_accounts_for_concentrated_branch_dispatch() -> None:
    record = SimpleNamespace(
        zone_id="zone_1",
        bus_id=2,
        load_kw=np.asarray([100.0, 100.0]),
        energy_input_kw=np.asarray([0.0, 0.0]),
        peak_power_kw=100.0,
    )
    pool = SimpleNamespace(sample=lambda **_: [record])
    config = {
        "population": {
            "N_simulated_resources": 1,
            "resources_per_zone": 1,
            "with_replacement": True,
            "seed_offset": 0,
        },
        "simulation": {"random_seed": 1},
        "zones": {"zones": {"zone_1": {
            "load_multiplier": 1.0,
            "energy_input_multiplier": 1.0,
        }}},
        "network": {
            "slack_bus": 1,
            "voltage_limits_pu": [0.95, 1.05],
            "transformer": {"bus": 1, "capacity_kw": 1000.0},
            "branches": [[1, 2, 0.1, 0.0, 100.0]],
        },
        "control": {
            "network_capacity_multiplier": 1.0,
            "voltage_sensitivity": 0.0,
            "reverse_flow_sensitivity": 0.0,
        },
    }

    multiplier = _capacity_multiplier(config, pool, target_loading=0.90)

    assert abs(multiplier - (200.0 / 100.0 / 0.90)) < 1e-12


def test_experiments_scenario_profiles_preserve_order_and_direction() -> None:
    config = {
        "zones": {"zones": {"zone_1": {}, "zone_2": {}}},
        "original_model": {"time_step_seconds": 300},
    }

    peak = build_scenario_profile_signal_schedule(
        config, "peak_shaving", 576, seed=42, ratio_noise_std=0.02
    )
    valley = build_scenario_profile_signal_schedule(
        config, "valley_filling", 576, seed=42, ratio_noise_std=0.02
    )
    grid = build_scenario_profile_signal_schedule(
        config, "emergency_grid_stability", 576, seed=42, ratio_noise_std=0.02
    )
    supply = build_scenario_profile_signal_schedule(
        config, "emergency_supply_shortage", 576, seed=42, ratio_noise_std=0.02
    )

    assert all(len(step) == 2 for step in peak + valley + grid + supply)
    assert all(step[0][0] >= 8 for step in peak)
    assert all(step[0][0] <= 7 for step in valley)
    assert np.mean([step[0][0] <= 7 for step in grid]) > 0.95
    assert np.mean([step[0][0] >= 8 for step in supply]) > 0.95


def test_experiments_training_schedule_has_full_signal_mixture() -> None:
    config = {
        "zones": {"zones": {"zone_1": {}, "zone_2": {}}},
        "original_model": {"time_step_seconds": 300},
    }

    schedule = build_experiments_training_signal_schedule(config, 2000, seed=42)
    directions = np.asarray([step[0][0] for step in schedule])
    intensities = np.asarray([step[0][1] for step in schedule])

    assert len(schedule) == 2000
    assert np.any(directions <= 7)
    assert np.any(directions >= 8)
    assert intensities.min() >= 0
    assert intensities.max() <= 4095


def test_experiments_transfer_schedule_uses_70_30_source_mixture() -> None:
    config = {"zones": {"zones": {"zone_1": {}, "zone_2": {}}}}

    schedule, hours = build_experiments_transfer_signal_schedule(
        config, 1000, seed=42
    )

    assert len(schedule) == len(hours) == 1000
    assert all(len(row) == 2 and len(set(row)) == 1 for row in schedule)
    assert all(0 <= row[0][0] <= 15 and 0 <= row[0][1] <= 4095 for row in schedule)
    tail_directions = np.asarray([row[0][0] for row in schedule[700:]])
    assert np.any(tail_directions <= 7)
    assert np.any(tail_directions >= 8)


def test_run_level_response_statistics_does_not_mix_step_variation() -> None:
    repetitions = [
        np.asarray([[0.0], [20.0]]),
        np.asarray([[2.0], [22.0]]),
    ]

    statistics = _run_level_response_statistics(repetitions)

    assert statistics["run_means"].tolist() == [10.0, 12.0]
    assert statistics["mean"] == 11.0
    assert statistics["std"] == 1.0
    assert abs(statistics["cv"] - 1.0 / 11.0) < 1e-12


def test_fig5_rho_scan_uses_original_experiments_shock_parameters() -> None:
    actual = [
        (row["global_shock_std"], row["regional_shock_std"])
        for row in RHO_SCAN
    ]

    assert actual == [
        (0.0, 0.0),
        (0.10, 0.05),
        (0.15, 0.08),
        (0.20, 0.10),
        (0.30, 0.15),
        (0.45, 0.20),
    ]


def test_original_iid_correlation_does_not_consume_rng() -> None:
    rng = np.random.default_rng(1000)
    reference = np.random.default_rng(1000)

    perturbations = _sample_original_region_perturbations(rng, 5, 0.0, 0.0)

    assert perturbations is None
    assert rng.random() == reference.random()


def test_fig5_protocol_metadata_records_six_points_and_preserves_main_run(tmp_path) -> None:
    result_root = tmp_path / "network_stress"
    metadata_path = result_root / "data" / "protocol_metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps({
        "shared_settings": {
            "correlation_levels": [0.0, 0.001],
            "correlation_conditions": [{"label": "main"}],
        }
    }), encoding="utf-8")
    config = {"experiment": {
        "figure5_rho_resources": 5000,
        "figure5_rho_theoretical_regions": 5,
        "figure5_rho_replications": 10,
        "figure5_rho_steps": 100,
    }}

    _update_fig5_protocol_metadata(config, result_root)

    shared = json.loads(metadata_path.read_text(encoding="utf-8"))["shared_settings"]
    assert shared["main_experiment_correlation_levels"] == [0.0, 0.001]
    assert len(shared["correlation_levels"]) == 6
    assert len(shared["correlation_conditions"]) == 6
    assert shared["figure5_rho_scan"]["runs_per_condition"] == 10
    assert shared["figure5_rho_scan"]["network_feedback"] is False


def test_trace_curtailment_reports_mwh() -> None:
    trace = SimpleNamespace(
        energy_input_kw=[2000.0] * 12,
        load_kw=[1000.0] * 12,
        accepted_control_kw=[500.0] * 12,
    )

    baseline_mwh, eps_mwh = _trace_curtailment(trace)

    assert baseline_mwh == 1.0
    assert eps_mwh == 0.5


def test_generic_external_input_is_not_labeled_as_pv() -> None:
    trace = {
        "energy_input_kw": np.full(288, 2.0).tolist(),
        "pv_kw": np.zeros(288).tolist(),
        "load_kw": np.ones(288).tolist(),
        "accepted_control_kw": np.full(288, 0.25).tolist(),
    }

    result = _curtailment_sensitivity(trace)

    assert result["metadata"]["input_source"] == "external_energy_input"
    assert "input_ratio" in result["scenarios"][0]
    assert "solar_ratio" not in result["scenarios"][0]


def test_stratified_signal_schedule_balances_direction_and_covers_range() -> None:
    config = {
        "zones": {
            "zones": {f"zone_{index}": {} for index in range(1, 7)},
        },
    }

    schedule = build_stratified_signal_schedule(
        config,
        500,
        seed=42,
        min_abs_score=0.05,
        max_abs_score=0.95,
    )

    directions = np.asarray([row[0][0] for row in schedule])
    intensities = np.asarray([row[0][1] for row in schedule])
    assert len(schedule) == 500
    assert np.sum(directions < 8) == 250
    assert np.sum(directions >= 8) == 250
    assert intensities.min() <= round(0.052 * 4095)
    assert intensities.max() >= round(0.948 * 4095)
    assert all(len(row) == 6 and len(set(row)) == 1 for row in schedule)


def test_trace_features_include_direction_intensity_interaction() -> None:
    trace = SimulationTrace(
        time_index=[0, 1],
        load_kw=[1000.0, 1000.0],
        pv_kw=[0.0, 0.0],
        energy_input_kw=[0.0, 0.0],
        baseline_battery_kw=[0.0, 0.0],
        desired_control_kw=[0.0, 0.0],
        accepted_control_kw=[0.0, 0.0],
        net_grid_kw=[0.0, 0.0],
        intensity=[2047.5, 2047.5],
        direction=[5, 10],
        network_scale=[1.0, 1.0],
        baseline_transformer_loading=[0.1, 0.1],
        baseline_maximum_branch_loading=[0.2, 0.2],
        baseline_minimum_voltage_pu=[0.98, 0.98],
        baseline_maximum_voltage_pu=[1.01, 1.01],
        transformer_loading=[0.0, 0.0],
        minimum_voltage_pu=[1.0, 1.0],
        maximum_voltage_pu=[1.0, 1.0],
        overloaded_branches=[0, 0],
        voltage_violations=[0, 0],
        self_consumption_baseline_kwh=[0.0, 0.0],
        self_consumption_eps_kwh=[0.0, 0.0],
        zone_control_kw={},
        bus_control_kw={},
    )

    features = trace_features(trace)

    assert features.shape == (2, 19)
    assert features[0, 3] == 0.5
    assert features[1, 3] == -0.5
