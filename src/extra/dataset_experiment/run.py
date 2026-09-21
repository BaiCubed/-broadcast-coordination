from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from src.extra.ieee33_device_day_simulation.config_loader import load_config
from src.extra.ieee33_device_day_simulation.experiments.run_experiment import (
    run_default,
    run_resource_response_diagnostics,
)
from src.extra.ieee33_device_day_simulation.figures.plot_figures import plot_all
from src.extra.ieee33_device_day_simulation.network.ieee33_distflow import IEEE33DistFlow
from src.extra.ieee33_device_day_simulation.population.device_day_loader import load_device_day_pool

from .hardware_preflight import run_preflight


ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / "src/extra/ieee33_device_day_simulation/configs"
RESULTS = ROOT / "results"

def dataset_results_root(dataset: str) -> Path:
    return RESULTS / f"{dataset}_ieee33_real_load"

DATASETS: dict[str, dict[str, Any]] = {
    "bdg1_building_data_genome": {
        "input": ROOT / "data/bdg1_building_data_genome/raw/temp_open_utc_complete.csv",
        "adapter": "bdg1_building_data_genome",
        "load_scale": 0.001,
        "source_unit": "kWh_per_hour_to_scaled_kW",
        "max_sources": 507,
        "main_resources": 504,
        "rho_resources": 505,
        "input_mapping": "load_shape_counterfactual",
        "max_days_per_source": 20,
        "with_replacement": False,
        "estimator_signal_sampling": "stratified_uniform",
        "fallback_reason": "open building electricity meters; no measured EV/PV/SOC fields, so battery parameters are explicit bounded fallbacks",
    },
    "low_carbon_london": {
        "input": ROOT / "data/low_carbon_london/raw/LCL-FullData.zip",
        "adapter": "low_carbon_london",
        "load_scale": 1.0,
        "source_unit": "kWh_per_half_hour_to_kW",
        "max_sources": 5000,
        "main_resources": 3000,
        "rho_resources": 5000,
        "input_mapping": "load_shape_counterfactual",
        "cache": ROOT / "data/low_carbon_london/processed/canonical_device_days_bidirectional_v2.npz",
        "max_days_per_source": 2,
        "with_replacement": False,
        "fallback_reason": "open half-hour household smart-meter electricity; no measured EV/PV/SOC fields, so battery parameters are explicit bounded fallbacks",
    },
    "bdg2_building_data_genome": {
        "input": ROOT / "data/bdg2_building_data_genome/raw/electricity_cleaned.csv",
        "solar_input": ROOT / "data/bdg2_building_data_genome/raw/solar_cleaned.csv",
        "cache": ROOT / "data/bdg2_building_data_genome/processed/canonical_device_days.npz",
        "adapter": "bdg2_building_data_genome", "load_scale": 0.001,
        "source_unit": "hourly_meter_value_scaled_to_kw", "max_sources": 1578,
        "main_resources": 1566, "rho_resources": 1570,
        "input_mapping": "measured_or_load_shape_counterfactual",
        "max_days_per_source": 8, "with_replacement": False,
        "fallback_reason": "real building electricity and five measured solar channels; EV battery fields remain bounded fallbacks",
    },
    "danish_smart_heat_meters": {
        "input": ROOT / "data/danish_smart_heat_meters/raw/3_years_3021_smart_heat_meters_residential_denmark.zip",
        "cache": ROOT / "data/danish_smart_heat_meters/processed/canonical_device_days.npz",
        "adapter": "danish_smart_heat_meters", "load_scale": 1.0,
        "source_unit": "cumulative_heat_kwh_differenced_to_kw", "source_energy_type": "heat_proxy",
        "max_sources": 2400, "max_days_per_source": 4, "with_replacement": False,
        "main_resources": 2400, "rho_resources": 2400,
        "input_mapping": "load_shape_counterfactual",
        "fallback_reason": "real thermal demand is used only as a load proxy; no COP or synthetic conversion model is added and EV fields are fallbacks",
    },
    "smart_grid_smart_city": {
        "input": ROOT / "data/smart_grid_smart_city/raw/cdintervalreadingallnoquotes.csv.7z",
        "cache": ROOT / "data/smart_grid_smart_city/processed/canonical_device_days.npz",
        "adapter": "smart_grid_smart_city", "load_scale": 1.0,
        "source_unit": "30min_kwh_divided_by_0.5h", "max_sources": 5000,
        "main_resources": 3000, "rho_resources": 5000,
        "input_mapping": "measured_or_load_shape_counterfactual",
        "cache": ROOT / "data/smart_grid_smart_city/processed/canonical_device_days_bidirectional_v2.npz",
        "max_days_per_source": 2, "with_replacement": False,
        "fallback_reason": "real customer demand and gross generation are measured; EV SOC/capacity/power remain bounded fallbacks",
    },
    "heapo_heat_pumps": {
        "input": ROOT / "data/heapo_heat_pumps/raw/heapo_data.zip",
        "cache": ROOT / "data/heapo_heat_pumps/processed/canonical_device_days.npz",
        "adapter": "heapo_heat_pumps", "load_scale": 1.0,
        "source_unit": "15min_kwh_divided_by_0.25h", "max_sources": 1408,
        "main_resources": 1362, "rho_resources": 1360,
        "input_mapping": "load_shape_counterfactual",
        "max_days_per_source": 8, "with_replacement": False,
        "fallback_reason": "real household and heat-pump electricity is measured; PV/EV flags are not power traces and battery fields remain fallbacks",
    },
    "goiener_smart_meters": {
        "input": ROOT / "data/goiener_smart_meters/raw/imp-post.tzst",
        "cache": ROOT / "data/goiener_smart_meters/processed/canonical_device_days.npz",
        "adapter": "goiener_smart_meters", "load_scale": 1.0,
        "source_unit": "hourly_kwh_to_kw", "max_sources": 5000,
        "main_resources": 3000, "rho_resources": 5000,
        "input_mapping": "load_shape_counterfactual",
        "cache": ROOT / "data/goiener_smart_meters/processed/canonical_device_days_bidirectional_v2.npz",
        "max_days_per_source": 2, "with_replacement": False,
        "fallback_reason": "real imputed smart-meter demand is measured; self-consumption metadata is not a generation trace and EV fields remain fallbacks",
    },
    "european_lv_urban_8087": {
        "input": ROOT / "data/european_lv_urban_8087/raw/Sim_files_190128_OK_V0.zip",
        "cache": ROOT / "data/european_lv_urban_8087/processed/canonical_device_days.npz",
        "adapter": "european_lv_urban_8087", "load_scale": 1.0,
        "source_unit": "ordered_hourly_active_power", "max_sources": 5000,
        "main_resources": 3000, "rho_resources": 5000,
        "input_mapping": "load_shape_counterfactual",
        "max_days_per_source": 2, "with_replacement": False,
        "fallback_reason": "real anonymized LV customer profiles drive the unchanged IEEE33 mapping; PV and EV fields remain fallbacks",
    },
    "european_lv_rural_2731": {
        "input_dir": ROOT / "data/european_lv_rural_2731/processed/PQ_csv",
        "cache": ROOT / "data/european_lv_rural_2731/processed/canonical_device_days.npz",
        "adapter": "european_lv_rural_2731", "load_scale": 1.0,
        "source_unit": "ordered_hourly_active_power", "max_sources": 2731,
        "main_resources": 2730, "rho_resources": 2730,
        "input_mapping": "load_shape_counterfactual",
        "max_days_per_source": 4, "with_replacement": False,
        "fallback_reason": "real rural P profiles drive the unchanged IEEE33 mapping; Q is audit-only and PV/EV fields remain fallbacks",
    },
    "european_lv_urban_35297": {
        "input_dir": ROOT / "data/european_lv_urban_35297/processed/PQ_csv",
        "cache": ROOT / "data/european_lv_urban_35297/processed/canonical_device_days.npz",
        "adapter": "european_lv_urban_35297", "load_scale": 1.0,
        "source_unit": "ordered_hourly_active_power", "max_sources": 12000,
        "main_resources": 3000, "rho_resources": 5000,
        "input_mapping": "load_shape_counterfactual",
        "max_days_per_source": 1, "with_replacement": False,
        "fallback_reason": "real urban P profiles drive the unchanged IEEE33 mapping; Q is audit-only and PV/EV fields remain fallbacks",
    },
    "norway_ami_energy_distribution": {
        "input_dir": ROOT / "data/norway_ami_energy_distribution/raw/Energy distribution models with AMI smart meter sensor dataset/data/ami",
        "cache": ROOT / "data/norway_ami_energy_distribution/processed/canonical_device_days_measured_only.npz",
        "adapter": "norway_ami_energy_distribution",
        "load_scale": 1.0,
        "source_unit": "hourly_active_power_kw",
        "max_sources": 3000,
        "main_resources": 3000,
        "rho_resources": 3000,
        "input_mapping": "measured_only",
        "forbid_counterfactual_input": True,
        "energy_input_provenance": "observed_activePowerIn_net_export",
        "max_days_per_source": 2,
        "with_replacement": False,
        "fallback_reason": "observed AMI import/export drives EPS; storage capacity, power and SOC remain bounded simulator fallbacks",
    },
    "camsl_japan_smart_meters": {
        "input_dir": ROOT / "data/camsl_japan_smart_meters/extracted/public/consumption_data/consumption_data",
        "cache": ROOT / "data/camsl_japan_smart_meters/processed/canonical_device_days_measured_only.npz",
        "adapter": "camsl_japan_smart_meters",
        "load_scale": 0.002,
        "source_unit": "raw_half_hour_value_assumed_Wh_divided_by_0.5h",
        "max_sources": 1423,
        "main_resources": 1422,
        "rho_resources": 1422,
        "input_mapping": "measured_only",
        "forbid_counterfactual_input": True,
        "energy_input_provenance": "not_available_zero_interface_placeholder",
        "max_days_per_source": 2,
        "with_replacement": False,
        "fallback_reason": "observed household demand drives explicit EPS scenarios; external input is unavailable and storage fields remain bounded simulator fallbacks",
    },
    "irish_domestic_smart_meters": {
        "input_dir": ROOT / "data/irish_domestic_smart_meters/raw/SM Data 2.0",
        "cache": ROOT / "data/irish_domestic_smart_meters/processed/canonical_device_days_measured_only_kw_label.npz",
        "adapter": "irish_domestic_smart_meters",
        "load_scale": 1.0,
        "source_unit": "source_interval_kw_label_primary_interpretation",
        "max_sources": 2989,
        "main_resources": 2988,
        "rho_resources": 2988,
        "input_mapping": "measured_only",
        "forbid_counterfactual_input": True,
        "energy_input_provenance": "observed_active_export_net_return_not_gross_generation",
        "max_days_per_source": 2,
        "with_replacement": False,
        "fallback_reason": "observed import/export drives EPS under the source kW-label interpretation; storage fields remain bounded simulator fallbacks",
    },
    "opsd_household_data": {
        "input": ROOT / "data/opsd_household_data/raw/opsd-household_data-2020-04-15/household_data_15min_singleindex.csv",
        "cache": ROOT / "data/opsd_household_data/processed/canonical_device_days_measured_only.npz",
        "adapter": "opsd_household_data",
        "load_scale": 4.0,
        "source_unit": "cumulative_kwh_differenced_then_divided_by_0.25h",
        "max_sources": 11,
        "main_resources": 3000,
        "rho_resources": 6,
        "input_mapping": "measured_only",
        "forbid_counterfactual_input": True,
        "energy_input_provenance": "observed_pv_where_available_otherwise_not_available",
        "max_days_per_source": 30,
        "with_replacement": True,
        "fallback_reason": "11 observed sites and available PV/device channels are bootstrapped for mechanism testing; missing storage state and parameters remain fallbacks",
    },
    "complete_energy_community": {
        "input": ROOT / "data/complete_energy_community/EC_EV_dataset.xlsx",
        "cache": ROOT / "data/complete_energy_community/processed/canonical_device_days_measured_only.npz",
        "adapter": "complete_energy_community",
        "load_scale": 1.0,
        "source_unit": "source_constructed_15min_kw_profile",
        "max_sources": 250,
        "main_resources": 3000,
        "rho_resources": 246,
        "input_mapping": "measured_only",
        "forbid_counterfactual_input": True,
        "energy_input_provenance": "source_constructed_pv_profile_no_additional_counterfactual",
        "max_days_per_source": 1,
        "with_replacement": True,
        "fallback_reason": "250 source-constructed community players are bootstrapped; workbook BESS parameters are used where present and bounded fallbacks cover the remainder",
    },
}

BLOCKED: dict[str, str] = {}


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


RHO_SCAN = [
    {"label": "iid", "global_shock_std": 0.0, "regional_shock_std": 0.0, "nominal_rho": 0.0},
    {"label": "rho~0.003", "global_shock_std": 0.10, "regional_shock_std": 0.05, "nominal_rho": 0.00312},
    {"label": "rho~0.006", "global_shock_std": 0.15, "regional_shock_std": 0.08, "nominal_rho": 0.00721344},
    {"label": "rho~0.010", "global_shock_std": 0.20, "regional_shock_std": 0.10, "nominal_rho": 0.01248},
    {"label": "rho~0.023", "global_shock_std": 0.30, "regional_shock_std": 0.15, "nominal_rho": 0.02808},
    {"label": "rho~0.051", "global_shock_std": 0.45, "regional_shock_std": 0.20, "nominal_rho": 0.060528},
]


def _n_scaling_values(maximum: int) -> list[int]:
    values = [value for value in [50, 100, 150, 200, 500, 1000, 2000, 3000] if value <= maximum]
    if maximum not in values:
        values.append(maximum)
    return sorted(set(values))


def _balanced_unique_fleet_limit(pool: Any) -> tuple[int, dict[str, int]]:
    zone_sources: dict[str, set[str]] = {}
    for record in pool.records:
        zone_sources.setdefault(str(record.zone_id), set()).add(str(record.source_device_id))
    if not zone_sources:
        raise ValueError("canonical device-day pool has no zone assignments")
    counts = {zone: len(sources) for zone, sources in sorted(zone_sources.items())}
    return len(counts) * min(counts.values()), counts


def _fit_experiment_to_pool(config_path: Path, dataset: str, pool: Any) -> dict[str, Any]:
    maximum, zone_counts = _balanced_unique_fleet_limit(pool)
    spec = DATASETS[dataset]
    bootstrap = bool(spec.get("with_replacement", False))
    selected = int(spec["main_resources"]) if bootstrap else min(int(spec["main_resources"]), maximum)
    if selected < 500:
        raise ValueError(f"{dataset} config requests only {selected} simulated resources; need at least 500")

    population_path = config_path.parent / "population.yaml"
    population = yaml.safe_load(population_path.read_text(encoding="utf-8"))
    zone_count = len(zone_counts)
    population["N_simulated_resources"] = selected
    population["resources_per_zone"] = selected // zone_count
    _write_yaml(population_path, population)

    experiment_path = config_path.parent / "experiment_protocol.yaml"
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    experiment["n_scaling_values"] = _n_scaling_values(selected)
    experiment["figure5_rho_resources"] = min(int(spec["rho_resources"]), selected)
    _write_yaml(experiment_path, experiment)
    return {
        "configured_main_resources": int(spec["main_resources"]),
        "configured_rho_resources": int(spec["rho_resources"]),
        "balanced_unique_limit": maximum,
        "selected_main_resources": selected,
        "selected_rho_resources": int(experiment["figure5_rho_resources"]),
        "unique_sources_by_zone": zone_counts,
        "with_replacement_bootstrap": bootstrap,
    }


def _prepare_config(dataset: str, result_root: Path) -> Path:
    config_dir = result_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "original_model.yaml", "source_device_map.yaml", "zones.yaml", "network_ieee33.yaml",
        "device_constraints.yaml", "user_behavior.yaml", "control.yaml", "experiment_protocol.yaml",
        "output_compatibility.yaml",
    ):
        shutil.copy2(BASE_CONFIG / name, config_dir / name)
    _write_yaml(config_dir / "paths.yaml", {
        "data_root": str(ROOT / "data"),
        "results_root": str(result_root),
    })
    spec = DATASETS[dataset]
    main_resources = int(spec["main_resources"])
    if main_resources < 500 or main_resources % 6:
        raise ValueError(f"{dataset}: main_resources must be at least 500 and divisible by six")
    spec_input = spec.get("input")
    if spec_input is None:
        spec_input = spec.get("input_dir")
    _write_yaml(config_dir / "population.yaml", {
        "resource_unit": "canonical-device-day",
        "source_device_count_expected": 0,
        "days_per_source_device": int(spec.get("max_days_per_source", 20)),
        "device_days_expected": 0,
        "N_simulated_resources": main_resources,
        "resources_per_zone": main_resources // 6,
        "sampling": "stratified_by_zone",
        "with_replacement": bool(spec["with_replacement"]),
        "seed_offset": 101,
        "allow_pool_bootstrap": bool(spec["with_replacement"]),
        "unique_source_per_batch": not bool(spec["with_replacement"]),
        "canonical_adapter": {
            "dataset": spec["adapter"],
            "input": str(spec_input),
            "input_dir": str(spec.get("input_dir", "")),
            "solar_input": str(spec.get("solar_input", "")),
            "cache": str(spec.get("cache", "")),
            "load_scale": float(spec["load_scale"]),
            "max_sources": int(spec.get("max_sources", 507)),
            "max_days_per_source": int(spec.get("max_days_per_source", 20)),
            "max_source_days": int(spec.get("max_source_days", 30)),
            "input_mapping": str(spec.get("input_mapping", "measured_only")),
            "forbid_counterfactual_input": bool(spec.get("forbid_counterfactual_input", False)),
            "energy_input_provenance": str(spec.get("energy_input_provenance", "unspecified")),
            "counterfactual_input_scale": float(spec.get("counterfactual_input_scale", 1.0)),
            "fallback_capacity_kwh": 10.0,
            "fallback_peak_power_kw": 1.0,
            "fallback_soc_fraction": 0.50,
        },
    })
    experiment = yaml.safe_load((BASE_CONFIG / "experiment_protocol.yaml").read_text(encoding="utf-8"))
    experiment.update({
        "training_steps": 288,
        "validation_samples": 500,
        "validation_batches": 2,
        "snapshot_profile_mode": "random_aligned",
        "snapshot_profile_seed_offset": 73000,
        "training_scenarios": ["normal", "peak_shaving", "valley_filling"],
        "validation_scenarios": ["valley_filling", "peak_shaving"],
        "balanced_validation_samples": True,
        "estimator_signal_sampling": str(
            spec.get("estimator_signal_sampling", "stratified_uniform")
        ),
        "estimator_min_abs_signal": 0.05,
        "estimator_max_abs_signal": 0.95,
        "force_network_feedback": True,
        "transfer_source_region_only": True,
        "transfer_scale_by_resource_count": False,
        "n_scaling_values": _n_scaling_values(main_resources),
        "n_scaling_runs": 8,
        "scaling_steps": 24,
        "directional_n_scaling": {
            "charge": "valley_filling",
            "discharge": "peak_shaving",
        },
        "correlation_conditions": [
            {"label": "iid", "zone_correlation": 0.0, "nominal_rho": 0.0},
            {"label": "weak", "zone_correlation": 0.001, "nominal_rho": 0.0000833},
            {"label": "moderate", "zone_correlation": 0.03, "nominal_rho": 0.001},
        ],
        "experiments_rho_scan": RHO_SCAN,
        "figure5_rho_resources": int(spec["rho_resources"]),
        "figure5_rho_theoretical_regions": 5,
        "figure5_rho_replications": 10,
        "figure5_rho_steps": 100,
    })
    _write_yaml(config_dir / "experiment_protocol.yaml", experiment)
    _write_yaml(config_dir / "default.yaml", {
        "paths": "paths.yaml",
        "original_model": "original_model.yaml",
        "simulation": "simulation.yaml",
        "population": "population.yaml",
        "source_device_map": "source_device_map.yaml",
        "zones": "zones.yaml",
        "network": "network_ieee33.yaml",
        "device_constraints": "device_constraints.yaml",
        "user_behavior": "user_behavior.yaml",
        "control": "control.yaml",
        "experiment": "experiment_protocol.yaml",
        "outputs": "output_compatibility.yaml",
    })
    shutil.copy2(BASE_CONFIG / "simulation.yaml", config_dir / "simulation.yaml")
    return config_dir / "default.yaml"


def _capacity_multiplier(config: dict[str, Any], pool: Any, *, target_loading: float) -> float:
    population = config["population"]
    resources = int(population["N_simulated_resources"])
    seed = int(config["simulation"]["random_seed"]) + int(population["seed_offset"])
    if bool(population.get("with_replacement", False)):
        records = pool.sample(
            per_zone=int(population["resources_per_zone"]),
            seed=seed,
            with_replacement=True,
        )
    else:
        records = pool.sample_unique_sources(resources, seed=seed)
    zones = config["zones"]["zones"]
    steps = min(len(record.load_kw) for record in records)
    dispatch_by_bus: dict[int, float] = {}
    for record in records:
        dispatch_by_bus[record.bus_id] = (
            dispatch_by_bus.get(record.bus_id, 0.0) + float(record.peak_power_kw)
        )

    base_control = dict(config["control"])
    base_control["network_capacity_multiplier"] = 1.0
    network = IEEE33DistFlow(config["network"], base_control)
    required_scales: list[float] = []
    voltage_span = max(network.vmax - 1.0, 1.0 - network.vmin, 1e-9)
    for step in range(steps):
        load_by_bus: dict[int, float] = {}
        input_by_bus: dict[int, float] = {}
        for record in records:
            zone = zones[record.zone_id]
            load_by_bus[record.bus_id] = load_by_bus.get(record.bus_id, 0.0) + (
                float(record.load_kw[step]) * float(zone["load_multiplier"])
            )
            input_multiplier = float(
                zone.get("energy_input_multiplier", zone.get("pv_multiplier", 1.0))
            )
            input_by_bus[record.bus_id] = input_by_bus.get(record.bus_id, 0.0) + (
                float(record.energy_input_kw[step]) * input_multiplier
            )
        for sign in (0.0, 1.0, -1.0):
            control_by_bus = {
                bus: sign * value for bus, value in dispatch_by_bus.items()
            }
            evaluation = network.evaluate(
                load_by_bus, input_by_bus, control_by_bus, apply_limits=False
            )
            thermal = max(
                evaluation.transformer_loading,
                max(evaluation.branch_loading.values(), default=0.0),
            ) / max(target_loading, 1e-9)
            voltage = max(
                max(1.0 - value, value - 1.0, 0.0)
                for value in evaluation.voltage_pu.values()
            ) / max(voltage_span * target_loading, 1e-9)
            required_scales.append(max(float(thermal), float(voltage)))
    return max(float(np.percentile(required_scales, 95)), 0.05)


def _mode_config(config_path: Path, mode: str, multiplier: float) -> Path:
    config_dir = config_path.parent
    control = yaml.safe_load((config_dir / "control.yaml").read_text(encoding="utf-8"))
    control["network_capacity_multiplier"] = float(multiplier)
    control["network_equivalent_scale"] = float(multiplier)
    control["network_feedback"] = True
    control_name = f"control_{mode}.yaml"
    _write_yaml(config_dir / control_name, control)
    default = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    default["control"] = control_name
    mode_path = config_dir / f"default_{mode}.yaml"
    _write_yaml(mode_path, default)
    return mode_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _compact_result_artifacts(result_root: Path) -> None:
    for relative in (
        "data/resource_response_diagnostics.npz",
        "data/resource_response_diagnostics.json",
        "data/training_model.json",
        "data/validation_results.json",
    ):
        (result_root / relative).unlink(missing_ok=True)


def _update_fig5_protocol_metadata(
    config: dict[str, Any],
    result_root: Path,
    *,
    network_feedback: bool = False,
) -> None:
    metadata_path = result_root / "data" / "protocol_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shared = metadata.setdefault("shared_settings", {})
    shared.setdefault(
        "main_experiment_correlation_levels",
        list(shared.get("correlation_levels", [])),
    )
    shared.setdefault(
        "main_experiment_correlation_conditions",
        list(shared.get("correlation_conditions", [])),
    )
    conditions = [
        {
            "label": str(spec["label"]),
            "global_shock_std": float(spec["global_shock_std"]),
            "regional_shock_std": float(spec["regional_shock_std"]),
            "nominal_rho": float(spec["nominal_rho"]),
        }
        for spec in RHO_SCAN
    ]
    shared["correlation_levels"] = [row["nominal_rho"] for row in conditions]
    shared["correlation_conditions"] = conditions
    shared["figure5_rho_scan"] = {
        "conditions": conditions,
        "resources": int(config["experiment"]["figure5_rho_resources"]),
        "theoretical_regions": int(
            config["experiment"].get("figure5_rho_theoretical_regions", 5)
        ),
        "runs_per_condition": int(config["experiment"]["figure5_rho_replications"]),
        "steps_per_run": int(config["experiment"]["figure5_rho_steps"]),
        "aggregate_statistic": "cv_across_replication_mean_responses",
        "shock_persistence": "one_global_plus_regional_draw_per_run",
        "correlation_application": "original_EPSSimulator_persistent_power_modulation",
        "network_feedback": bool(network_feedback),
    }
    _write_json(metadata_path, metadata)


def _refresh_rho_scan(
    config_path: Path,
    mode: str,
    result_root: Path,
    *,
    network_feedback: bool = False,
) -> None:
    config = load_config(config_path)
    scan_rows: dict[str, Any] = {}
    n = int(config["experiment"]["figure5_rho_resources"])
    for spec in RHO_SCAN:
        diagnostic = run_resource_response_diagnostics(
            config_path,
            mode=mode,
            result_root=result_root,
            snapshot_sampling="aligned",
            replications=int(config["experiment"]["figure5_rho_replications"]),
            weak_rho_threshold=0.25 / max(n - 1, 1),
            zone_correlation=0.0,
            original_global_shock_std=float(spec["global_shock_std"]),
            original_regional_shock_std=float(spec["regional_shock_std"]),
            network_feedback_override=bool(network_feedback),
            condition_label=str(spec["label"]),
            diagnostic_resources=n,
            diagnostic_steps=int(config["experiment"]["figure5_rho_steps"]),
            diagnostic_scenarios=["peak_shaving"],
        )
        payload = json.loads(Path(diagnostic).read_text(encoding="utf-8"))
        rho = float(spec["nominal_rho"])
        k = int(config["experiment"].get("figure5_rho_theoretical_regions", 5))
        n_eff = n / (1.0 + (n / k - 1.0) * rho) if rho else float(n)
        cv = float(payload.get("aggregate_response_cv", 0.0))
        scan_rows[str(spec["label"])] = {
            "rho_within": rho,
            "nominal_rho": rho,
            "measured_residual_rho_signed": float(payload.get("conditional_residual_mean_pairwise_rho", 0.0)),
            "zone_correlation_parameter": 0.0,
            "global_shock_std": float(spec["global_shock_std"]),
            "regional_shock_std": float(spec["regional_shock_std"]),
            "N": n,
            "K": k,
            "N_eff": float(n_eff),
            "N_eff_over_N": float(n_eff / n),
            "mean_response_kw": float(payload.get("aggregate_response_mean_kw", 0.0)),
            "std_response_kw": float(payload.get("aggregate_response_std_kw", 0.0)),
            "cv": cv,
            "cv_at_max_N": cv,
            "cv_sqrt_neff": float(cv * np.sqrt(max(n_eff, 1.0))),
            "runs": int(payload.get("replications", 0)),
            "steps_per_run": int(payload.get("steps_per_replication", 0)),
            "run_mean_response_kw": list(payload.get("run_mean_response_kw", [])),
            "aggregate_response_statistic": str(
                payload.get("aggregate_response_statistic", "unspecified")
            ),
        }
        _write_json(result_root / "data" / f"fig5_diagnostic_{spec['label']}.json", payload)
    nscale = json.loads((result_root / "data" / "n_scaling.json").read_text(encoding="utf-8"))
    aligned = {
        "iid": scan_rows["iid"],
        "weak": scan_rows["rho~0.006"],
        "moderate": scan_rows["rho~0.023"],
        "rho_scan": scan_rows,
    }
    for key in ("iid", "weak", "moderate"):
        aligned[key]["scaling_data"] = nscale.get(key, {}).get("scaling_data", [])
        aligned[key]["loglog_slope"] = nscale.get(key, {}).get("loglog_slope", -0.5)
        aligned[key]["sigma_hat"] = nscale.get(key, {}).get("sigma_hat", 0.0)
    _write_json(result_root / "data" / "rho_sensitivity.json", aligned)
    _update_fig5_protocol_metadata(
        config,
        result_root,
        network_feedback=network_feedback,
    )


def run_dataset(dataset: str, *, results_root: Path | None = None, modes: tuple[str, ...] = ("network_stress", "weak_correlation"), preflight_only: bool = False) -> dict[str, Any]:
    result_root = results_root or dataset_results_root(dataset)
    result_root.mkdir(parents=True, exist_ok=True)
    if dataset in BLOCKED:
        payload = {"dataset": dataset, "status": "blocked", "reason": BLOCKED[dataset], "training_started": False}
        _write_json(result_root / "preflight.json", payload)
        _write_json(result_root / "run_status.json", payload)
        return payload
    config_path = _prepare_config(dataset, result_root)
    config = load_config(config_path)
    pool = load_device_day_pool(config)
    fleet_calibration = _fit_experiment_to_pool(config_path, dataset, pool)
    config = load_config(config_path)
    capacities = {
        "network_stress": _capacity_multiplier(config, pool, target_loading=0.90),
        "weak_correlation": _capacity_multiplier(config, pool, target_loading=0.45),
    }
    preflight = run_preflight(config, pool)
    preflight.update({
        "dataset": dataset,
        "input": str(DATASETS[dataset].get("input", DATASETS[dataset].get("input_dir", ""))),
        "mapping": {
            "load_scale": DATASETS[dataset]["load_scale"],
            "source_unit": DATASETS[dataset]["source_unit"],
            "source_energy_type": DATASETS[dataset].get("source_energy_type", "electricity"),
            "battery_fallback": DATASETS[dataset]["fallback_reason"],
            "input_mapping": DATASETS[dataset].get("input_mapping", "measured_only"),
            "energy_input_provenance": DATASETS[dataset].get("energy_input_provenance", "unspecified"),
            "counterfactual_energy_input_forbidden": bool(DATASETS[dataset].get("forbid_counterfactual_input", False)),
            "with_replacement_bootstrap": bool(DATASETS[dataset].get("with_replacement", False)),
            "network_and_eps": "unchanged IEEE33 DeviceDay experiment",
        },
    })
    preflight["capacity_calibration"] = capacities
    preflight["fleet_calibration"] = fleet_calibration
    _write_json(result_root / "preflight.json", preflight)
    if preflight["status"] != "pass" or preflight_only:
        return preflight
    _write_json(result_root / "run_status.json", {
        "dataset": dataset,
        "status": "running",
        "training_started": True,
        "requested_modes": list(modes),
        "completed_modes": [],
    })
    completed: list[str] = []
    for mode in modes:
        mode_root = result_root / mode
        if mode_root.exists():
            shutil.rmtree(mode_root)
        mode_config_path = _mode_config(config_path, mode, capacities[mode])
        mode_config = load_config(mode_config_path)
        mode_pool = load_device_day_pool(mode_config)
        mode_preflight = run_preflight(mode_config, mode_pool)
        if mode_preflight["status"] != "pass":
            raise RuntimeError(f"{dataset}/{mode} preflight failed: {mode_preflight}")
        run_default(
            config_path=mode_config_path,
            mode=mode,
            results_root=mode_root,
            snapshot_protocol=True,
            snapshot_sampling="aligned",
        )
        _refresh_rho_scan(mode_config_path, mode, mode_root)
        plot_all(mode_root, compact=True)
        _compact_result_artifacts(mode_root)
        completed.append(mode)
        _write_json(result_root / "run_status.json", {
            "dataset": dataset,
            "status": "running",
            "training_started": True,
            "requested_modes": list(modes),
            "completed_modes": completed,
        })
    payload = {"dataset": dataset, "status": "completed", "training_started": True, "modes": completed}
    _write_json(result_root / "run_status.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external public datasets through the unchanged IEEE33 Figure 2-5 experiment.")
    parser.add_argument("--dataset", choices=sorted(set(DATASETS) | set(BLOCKED)))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--network-stress-only", action="store_true")
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("provide --dataset or --all")
    datasets = [args.dataset] if args.dataset else sorted(DATASETS)
    modes = ("network_stress",) if args.network_stress_only else ("network_stress", "weak_correlation")
    for dataset in datasets:
        print(json.dumps(run_dataset(dataset, modes=modes, preflight_only=args.preflight_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
