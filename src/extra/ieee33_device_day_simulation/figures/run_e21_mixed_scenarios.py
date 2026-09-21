from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.estimation import EPSEstimator
from src.signal import SignalOptimizer

from ..network.ieee33_distflow import IEEE33DistFlow
from ..original_adapter.eps_adapter import OriginalEPSAdapter
from ..population.device_day_loader import DeviceDay, load_device_day_pool
from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as protocol
from . import run_e20_transfer as transfer


PROTOCOL = "E21_mixed_physical_fleet_v1"
FLEET_SIZE = 5000
AVAILABILITY_MODE = "data_driven"
NETWORK_MODES = ("aggregate", "ieee33")
SEED_COUNT = 30
TRAIN_SAMPLES_PER_SCENARIO = 128
VALIDATION_SAMPLES_PER_SCENARIO = 32
RESPONSE_SCALE_KW = 5000.0

BDG1 = "bdg1_building_data_genome"
BDG2 = "bdg2_building_data_genome"
CEC = "complete_energy_community"
DANISH = "danish_smart_heat_meters"
EU_RURAL = "european_lv_rural_2731"
EU_35297 = "european_lv_urban_35297"
GOIENER = "goiener_smart_meters"
HEAPO = "heapo_heat_pumps"
LCL = "low_carbon_london"
NORWAY = "norway_ami_energy_distribution"
CAMSL = "camsl_japan_smart_meters"
EU_8087 = "european_lv_urban_8087"
IRISH = "irish_domestic_smart_meters"
OPSD = "opsd_household_data"
SGSC = "smart_grid_smart_city"

ALL_DATASETS = (
    BDG1, BDG2, CEC, DANISH, EU_RURAL, EU_35297, GOIENER, HEAPO,
    LCL, NORWAY, CAMSL, EU_8087, IRISH, OPSD, SGSC,
)


def _equal_weights(datasets: tuple[str, ...]) -> dict[str, float]:
    return {dataset: 1.0 / len(datasets) for dataset in datasets}


SCENARIOS: dict[str, dict[str, Any]] = {
    "S1-A": {
        "label": "Similar residential",
        "weights": _equal_weights((LCL, CAMSL, IRISH)),
    },
    "S1-B": {
        "label": "Building-heat pump-residential",
        "weights": _equal_weights((BDG2, HEAPO, SGSC)),
    },
    "S2-A": {
        "label": "LV network and AMI",
        "weights": {
            EU_35297: 0.40, EU_8087: 0.25, EU_RURAL: 0.15,
            NORWAY: 0.10, GOIENER: 0.10,
        },
    },
    "S2-B": {
        "label": "Building-residential-thermal-DER",
        "weights": {
            BDG2: 0.35, LCL: 0.25, DANISH: 0.15, HEAPO: 0.15, CEC: 0.10,
        },
    },
    "S3-A": {
        "label": "Ten-source electricity mix",
        "weights": {
            LCL: 0.18, SGSC: 0.14, CAMSL: 0.12, IRISH: 0.10,
            GOIENER: 0.08, NORWAY: 0.08, EU_RURAL: 0.07,
            EU_35297: 0.10, EU_8087: 0.08, HEAPO: 0.05,
        },
    },
    "S3-B": {
        "label": "Ten-source multi-sector mix",
        "weights": {
            BDG1: 0.10, BDG2: 0.15, CEC: 0.10, DANISH: 0.10,
            HEAPO: 0.10, LCL: 0.10, SGSC: 0.10, EU_RURAL: 0.08,
            NORWAY: 0.12, OPSD: 0.05,
        },
    },
    "S4-A": {
        "label": "All datasets equally weighted",
        "weights": _equal_weights(ALL_DATASETS),
    },
    "S4-B": {
        "label": "All dataset types equally weighted",
        "weights": {
            BDG1: 0.10, BDG2: 0.10,
            LCL: 0.04, CAMSL: 0.04, IRISH: 0.04, GOIENER: 0.04, SGSC: 0.04,
            DANISH: 0.10, HEAPO: 0.10,
            EU_RURAL: 0.05, EU_35297: 0.05, EU_8087: 0.05, NORWAY: 0.05,
            CEC: 0.15, OPSD: 0.05,
        },
    },
    "S5-A": {
        "label": "Residential-dominant long tail",
        "weights": {
            LCL: 0.13, CAMSL: 0.13, IRISH: 0.13, GOIENER: 0.13, SGSC: 0.13,
            EU_RURAL: 0.05, EU_35297: 0.05, EU_8087: 0.05, NORWAY: 0.05,
            BDG1: 0.025, BDG2: 0.025, DANISH: 0.025, HEAPO: 0.025,
            CEC: 0.03, OPSD: 0.02,
        },
    },
    "S5-B": {
        "label": "Building-thermal-dominant long tail",
        "weights": {
            BDG1: 0.20, BDG2: 0.20, DANISH: 0.15, HEAPO: 0.15,
            LCL: 0.03, CAMSL: 0.03, IRISH: 0.03, GOIENER: 0.03, SGSC: 0.03,
            EU_RURAL: 0.025, EU_35297: 0.025, EU_8087: 0.025, NORWAY: 0.025,
            CEC: 0.03, OPSD: 0.02,
        },
    },
    "S5-C": {
        "label": "Network-DER-dominant long tail",
        "weights": {
            EU_RURAL: 0.1125, EU_35297: 0.1125, EU_8087: 0.1125, NORWAY: 0.1125,
            CEC: 0.20, OPSD: 0.05,
            LCL: 0.04, CAMSL: 0.04, IRISH: 0.04, GOIENER: 0.04, SGSC: 0.04,
            BDG1: 0.025, BDG2: 0.025, DANISH: 0.025, HEAPO: 0.025,
        },
    },
    "S6-A": {
        "label": "Spatially clustered LV networks",
        "weights": {EU_35297: 0.35, EU_8087: 0.25, EU_RURAL: 0.20, NORWAY: 0.20},
        "zones": {
            EU_35297: ("zone_1", "zone_2"),
            EU_8087: ("zone_3", "zone_4"),
            NORWAY: ("zone_5",),
            EU_RURAL: ("zone_6",),
        },
    },
    "S6-B": {
        "label": "Spatial cross-sector congestion",
        "weights": {BDG2: 0.35, HEAPO: 0.20, DANISH: 0.15, LCL: 0.20, SGSC: 0.10},
        "zones": {
            BDG2: ("zone_1", "zone_2"),
            LCL: ("zone_3",),
            SGSC: ("zone_4",),
            HEAPO: ("zone_5",),
            DANISH: ("zone_6",),
        },
    },
    "S6-C": {
        "label": "Spatial DER and bidirectional mix",
        "weights": {CEC: 0.30, SGSC: 0.30, OPSD: 0.05, IRISH: 0.20, NORWAY: 0.15},
        "zones": {
            NORWAY: ("zone_1",),
            IRISH: ("zone_2", "zone_3"),
            OPSD: ("zone_4",),
            SGSC: ("zone_5",),
            CEC: ("zone_6",),
        },
    },
}

BASELINE_ORDER = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
)
ALGORITHM_ORDER = (
    *BASELINE_ORDER,
    "eps_e20_pooled",
    "eps_e21_mixed",
    "centralized_optimal",
)
ALGORITHM_DEFINITIONS = {
    **{
        algorithm: legacy.STRATEGY_DEFINITIONS[algorithm]
        for algorithm in (*BASELINE_ORDER, "centralized_optimal")
    },
    "eps_e20_pooled": {
        "label": "E20 pooled EPS",
        "complexity": "O(1)",
        "description": "pooled training on the response samples of single-dataset fleets.",
    },
    "eps_e21_mixed": {
        "label": "E21 mixed EPS",
        "complexity": "O(1)",
        "description": "trained directly on the response of the physically mixed fleets.",
    },
}
ALGORITHM_COLORS = {
    **legacy.FIG4D_COLORS,
    "eps_e20_pooled": "#ff7f0e",
    "eps_e21_mixed": "#069c8f",
}

_CONFIG_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_PARTITION_CACHE: dict[str, protocol.ProfilePartitions] = {}


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _acquire_lock(root: Path) -> Any | None:
    handle = (root / ".run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _config(dataset: str, network_mode: str) -> dict[str, Any]:
    key = (dataset, network_mode)
    if key not in _CONFIG_CACHE:
        _CONFIG_CACHE[key] = transfer._config_for(dataset, network_mode)
    return _CONFIG_CACHE[key]


def _partitions(dataset: str, network_mode: str) -> protocol.ProfilePartitions:
    if dataset not in _PARTITION_CACHE:
        config = _config(dataset, network_mode)
        pool = load_device_day_pool(config)
        _PARTITION_CACHE[dataset] = protocol.partition_profile_pool(
            pool, int(config["simulation"]["random_seed"])
        )
    return _PARTITION_CACHE[dataset]


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("the mixture weights must sum to a positive value")
    return {dataset: float(value) / total for dataset, value in weights.items()}


def _validate_scenarios() -> None:
    for scenario_id, scenario in SCENARIOS.items():
        weights = _normalized_weights(scenario["weights"])
        if set(weights) - set(ALL_DATASETS):
            raise ValueError(f"{scenario_id} contains an unknown dataset")
        if any(value <= 0.0 for value in weights.values()):
            raise ValueError(f"{scenario_id} contains a non-positive weight")
        if scenario_id != "S4-A" and weights.get(OPSD, 0.0) > 0.05 + 1e-12:
            raise ValueError(f"{scenario_id}: the OPSD weight exceeds 5%")
        if "zones" in scenario and set(scenario["zones"]) != set(weights):
            raise ValueError(f"{scenario_id}: the spatial mapping is incomplete")


def _shift_weights(weights: dict[str, float], seed_index: int) -> tuple[dict[str, float], str]:
    values = _normalized_weights(weights)
    if seed_index < 10:
        return values, "nominal"
    dominant = max(values, key=values.get)
    old = values[dominant]
    if seed_index < 20:
        new = min(old + 0.15, 0.95)
        label = "dominant_plus_15pp"
    else:
        new = max(old - 0.15, 0.01)
        label = "dominant_minus_15pp"
    scale = (1.0 - new) / max(1.0 - old, 1e-12)
    shifted = {
        dataset: (new if dataset == dominant else value * scale)
        for dataset, value in values.items()
    }
    return _normalized_weights(shifted), label


def _allocate_counts(weights: dict[str, float], total: int = FLEET_SIZE) -> dict[str, int]:
    normalized = _normalized_weights(weights)
    if total < len(normalized):
        raise ValueError("the fleet is smaller than the number of datasets in the mixture")
    raw = {dataset: value * total for dataset, value in normalized.items()}
    residual_total = total - len(normalized)
    residual_weights = {
        dataset: max(value - 1.0, 0.0) for dataset, value in raw.items()
    }
    residual_weight_total = float(sum(residual_weights.values()))
    if residual_total == 0 or residual_weight_total <= 0.0:
        residual_raw = {dataset: 0.0 for dataset in normalized}
    else:
        residual_raw = {
            dataset: value * residual_total / residual_weight_total
            for dataset, value in residual_weights.items()
        }
    counts = {
        dataset: 1 + int(np.floor(residual_raw[dataset]))
        for dataset in normalized
    }
    remainder = total - sum(counts.values())
    order = sorted(
        normalized,
        key=lambda dataset: residual_raw[dataset] - np.floor(residual_raw[dataset]),
        reverse=True,
    )
    for dataset in order[:remainder]:
        counts[dataset] += 1
    if sum(counts.values()) != total:
        raise RuntimeError("the device allocation of the mixed fleet is wrong")
    return counts


def _spatial_record(
    record: DeviceDay,
    dataset: str,
    record_index: int,
    scenario: dict[str, Any],
    config: dict[str, Any],
) -> DeviceDay:
    if "zones" not in scenario:
        return record
    zones = tuple(scenario["zones"][dataset])
    zone_id = zones[record_index % len(zones)]
    buses = tuple(config["zones"]["zones"][zone_id]["buses"])
    bus_id = int(buses[(record_index // len(zones)) % len(buses)])
    return replace(record, zone_id=zone_id, bus_id=bus_id)


def _sample_mixed_fleet(
    scenario_id: str,
    network_mode: str,
    partition_name: str,
    seed: int,
    weights: dict[str, float] | None = None,
    fleet_size: int = FLEET_SIZE,
) -> tuple[list[DeviceDay], dict[str, Any], dict[str, Any]]:
    scenario = SCENARIOS[scenario_id]
    selected_weights = _normalized_weights(weights or scenario["weights"])
    counts = _allocate_counts(selected_weights, total=fleet_size)
    anchor_dataset = next(iter(selected_weights))
    config = _config(anchor_dataset, network_mode)
    records: list[DeviceDay] = []
    dataset_audit: dict[str, Any] = {}
    for dataset_index, (dataset, count) in enumerate(counts.items()):
        partition = getattr(_partitions(dataset, network_mode), partition_name)
        sampled, metadata = protocol.sample_device_fleet(
            partition,
            "fixed5000",
            count,
            seed + 10007 * (dataset_index + 1) + protocol._stable_seed(dataset) % 100000,
            dataset_id=dataset,
        )
        start = len(records)
        for local_index, record in enumerate(sampled):
            spatial = _spatial_record(
                record, dataset, local_index, scenario, config
            )
            records.append(replace(
                spatial,
                device_id=f"{dataset}__{start + local_index:05d}",
                source_device_id=f"{dataset}__{spatial.source_device_id}",
            ))
        dataset_audit[dataset] = {
            "weight": selected_weights[dataset],
            "logical_devices": count,
            "available_profile_sources": metadata["available_profile_sources"],
            "selected_unique_profile_sources": metadata["selected_unique_profile_sources"],
            "profile_bootstrap": metadata["profile_bootstrap"],
            "mean_profile_reuse": metadata["mean_profile_reuse"],
        }
    return records, config, {
        "scenario": scenario_id,
        "partition": partition_name,
        "fleet_size": len(records),
        "weights": selected_weights,
        "counts": counts,
        "datasets": dataset_audit,
        "spatial_clustering": "zones" in scenario,
    }


def _response_samples(
    records: list[DeviceDay],
    config: dict[str, Any],
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    scenario = protocol.build_pure_sim_scenario(records, config)
    availability = protocol.availability_probability(records, config, AVAILABILITY_MODE)
    network = IEEE33DistFlow(config["network"], config["control"])
    adapter = OriginalEPSAdapter(records, config, seed + 1)
    rng = np.random.default_rng(seed + 2)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    zone_count = len(config["zones"]["zones"])
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    for index in range(sample_count):
        supply_demand, intensity, hour = protocol._signal_parameters(
            index, sample_count, rng
        )
        profile_step = int(hour * 12 + rng.integers(0, 12))
        adapter.reset_to_initial_state()
        energy_kwh = 0.0
        for offset in range(3):
            step = (profile_step + offset) % legacy.STEPS
            signals_by_zone = [
                adapter.simulator._signal_generator.generate_signal(
                    zone, supply_demand, intensity, priority=10
                )
                for zone in range(zone_count)
            ]
            batch = adapter.step(signals_by_zone, step, availability[step])
            accepted, _, _ = legacy._apply_step(
                network,
                scenario,
                step,
                np.asarray(batch.desired_kw, dtype=float),
                records,
                config,
            )
            actual = adapter.apply_dispatch(accepted)
            energy_kwh += float(np.sum(actual)) * dt
        signals.append({
            "supply_demand": supply_demand,
            "intensity": intensity,
            "price": 0.0,
            "hour": hour,
            "day_of_week": 0,
            "direction": 1 if supply_demand <= 7 else -1,
        })
        responses.append(energy_kwh / max(3.0 * dt, 1e-12) / RESPONSE_SCALE_KW)
    return signals, responses, {
        "availability": legacy._availability_summary(records, availability),
        "response_range_normalized": [float(np.min(responses)), float(np.max(responses))],
    }


def _train_model(network_mode: str, model_path: Path) -> dict[str, Any]:
    train_signals: list[dict[str, Any]] = []
    train_responses: list[float] = []
    validation_signals: list[dict[str, Any]] = []
    validation_responses: list[float] = []
    scenario_metadata: list[dict[str, Any]] = []
    for scenario_index, scenario_id in enumerate(SCENARIOS):
        train_seed = 3100000 + scenario_index * 10000 + (500000 if network_mode == "ieee33" else 0)
        records, config, fleet_audit = _sample_mixed_fleet(
            scenario_id, network_mode, "train", train_seed
        )
        signals, responses, response_audit = _response_samples(
            records, config, TRAIN_SAMPLES_PER_SCENARIO, train_seed + 1
        )
        train_signals.extend(signals)
        train_responses.extend(responses)
        validation_records, validation_config, validation_fleet_audit = _sample_mixed_fleet(
            scenario_id, network_mode, "validation", train_seed + 2000
        )
        signals, responses, validation_response_audit = _response_samples(
            validation_records,
            validation_config,
            VALIDATION_SAMPLES_PER_SCENARIO,
            train_seed + 2001,
        )
        validation_signals.extend(signals)
        validation_responses.extend(responses)
        scenario_metadata.append({
            "scenario": scenario_id,
            "train_fleet": fleet_audit,
            "train_response": response_audit,
            "validation_fleet": validation_fleet_audit,
            "validation_response": validation_response_audit,
        })
        print(json.dumps({
            "stage": "training_samples",
            "network_mode": network_mode,
            "scenario": scenario_id,
            "completed_scenarios": scenario_index + 1,
            "scenario_count": len(SCENARIOS),
        }, ensure_ascii=False), flush=True)
    try:
        import torch

        torch.manual_seed(3100000 + (500000 if network_mode == "ieee33" else 0))
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = transfer._make_estimator()
    estimator.fit(train_signals, train_responses)
    legacy._save_eps_model(estimator, model_path)
    actual = np.asarray(validation_responses, dtype=float)
    predicted = np.asarray([
        estimator.estimate(signal).response_kw for signal in validation_signals
    ], dtype=float)
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    parameters = estimator._learned_params or {}
    return {
        "network_mode": network_mode,
        "model": str(model_path),
        "model_type": str(parameters.get("model_type", "unknown")),
        "training_samples": len(train_signals),
        "validation_samples": len(validation_signals),
        "scenario_count": len(SCENARIOS),
        "samples_per_scenario": TRAIN_SAMPLES_PER_SCENARIO,
        "validation_samples_per_scenario": VALIDATION_SAMPLES_PER_SCENARIO,
        "training_target": "accepted_response_kw / 5000 kW",
        "internal_training_r2": float(parameters.get("r2", float("nan"))),
        "validation_r2": float(
            1.0 - np.sum(residual ** 2) / max(denominator, 1e-12)
        ),
        "validation_rmse_normalized": float(np.sqrt(np.mean(residual ** 2))),
        "validation_mae_normalized": float(np.mean(np.abs(residual))),
        "scenario_metadata": scenario_metadata,
    }


def _no_coordination(
    scenario: dict[str, Any],
    availability: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    baseline = float(
        np.sum(np.maximum(scenario["input_total"] - scenario["load"], 0.0))
        * dt / 1000.0
    )
    return {
        "seed": int(seed),
        "mean_reduction_pct": 0.0,
        "baseline_curtailment_mwh": baseline,
        "remaining_curtailment_mwh": baseline,
        "accepted_absorption_mwh": 0.0,
        "total_charging_mwh": 0.0,
        "excess_grid_charging_mwh": 0.0,
        "total_discharge_mwh": 0.0,
        "availability_fraction": float(np.mean(availability)),
        "mean_network_scale": 1.0,
        "network_violation_steps": 0,
        "max_soc_violation": 0.0,
    }


def _evaluate_condition(task: tuple[str, str, str, str]) -> dict[str, Any]:
    logging.getLogger("src.simulation.simulator").setLevel(logging.WARNING)
    scenario_id, network_mode, e20_model_path, e21_model_path = task
    e20_estimator, _, _ = protocol.load_frozen_eps_controller(Path(e20_model_path))
    e21_estimator, _, _ = protocol.load_frozen_eps_controller(Path(e21_model_path))
    optimizers = {
        "eps_e20_pooled": SignalOptimizer(
            transfer.ScaledEstimator(e20_estimator, RESPONSE_SCALE_KW, 0.0)
        ),
        "eps_e21_mixed": SignalOptimizer(
            transfer.ScaledEstimator(e21_estimator, RESPONSE_SCALE_KW, 0.0)
        ),
    }
    per_algorithm: dict[str, list[dict[str, Any]]] = {
        algorithm: [] for algorithm in ALGORITHM_ORDER
    }
    fleet_audit: list[dict[str, Any]] = []
    for seed_index in range(SEED_COUNT):
        weights, composition_regime = _shift_weights(
            SCENARIOS[scenario_id]["weights"], seed_index
        )
        base_seed = (
            4100000
            + protocol._stable_seed(scenario_id) % 100000
            + (500000 if network_mode == "ieee33" else 0)
            + seed_index
        )
        records, config, audit = _sample_mixed_fleet(
            scenario_id, network_mode, "test", base_seed, weights
        )
        scenario = protocol.build_pure_sim_scenario(records, config)
        availability = protocol.availability_probability(
            records, config, AVAILABILITY_MODE
        )
        audit["seed"] = base_seed
        audit["composition_regime"] = composition_regime
        fleet_audit.append(audit)
        for algorithm_index, algorithm in enumerate(ALGORITHM_ORDER):
            if algorithm == "no_coordination":
                result = _no_coordination(
                    scenario, availability, config, base_seed
                )
            else:
                strategy = (
                    "eps_broadcast" if algorithm.startswith("eps_") else algorithm
                )
                result = legacy._run_seed(
                    strategy,
                    records,
                    config,
                    scenario,
                    availability,
                    base_seed + 100000 * (algorithm_index + 1),
                    base_seed + 1910000,
                    eps_optimizer=optimizers.get(algorithm),
                )
            result["composition_regime"] = composition_regime
            per_algorithm[algorithm].append(result)
    return {
        "scenario": scenario_id,
        "scenario_label": SCENARIOS[scenario_id]["label"],
        "network_mode": network_mode,
        "seed_count": SEED_COUNT,
        "nominal_weights": _normalized_weights(SCENARIOS[scenario_id]["weights"]),
        "spatial_clustering": "zones" in SCENARIOS[scenario_id],
        "fleet_audit": fleet_audit,
        "results": {
            algorithm: legacy._summarize(rows)
            for algorithm, rows in per_algorithm.items()
        },
    }


def _write_summary_csv(conditions: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scenario",
            "scenario_label",
            "network_mode",
            "algorithm",
            "algorithm_label",
            "complexity",
            "seed_count",
            "mean_reduction_pct",
            "std_reduction_pct",
            "network_violation_steps",
            "max_soc_violation",
        ])
        for condition in conditions:
            for algorithm in ALGORITHM_ORDER:
                result = condition["results"][algorithm]
                definition = ALGORITHM_DEFINITIONS[algorithm]
                writer.writerow([
                    condition["scenario"],
                    condition["scenario_label"],
                    condition["network_mode"],
                    algorithm,
                    definition["label"],
                    definition["complexity"],
                    condition["seed_count"],
                    result["mean_reduction_pct"],
                    result["std_reduction_pct"],
                    result["network_violation_steps"],
                    result["max_soc_violation"],
                ])


def _plot_heatmaps(
    conditions: list[dict[str, Any]],
    output_dir: Path,
    *,
    experiment_label: str = "E21 mixed-scenario",
    filename_prefix: str = "e21",
) -> list[str]:
    paths: list[str] = []
    labels = [
        f"{ALGORITHM_DEFINITIONS[algorithm]['label']}\n"
        f"{ALGORITHM_DEFINITIONS[algorithm]['complexity']}"
        for algorithm in ALGORITHM_ORDER
    ]
    for network_mode in NETWORK_MODES:
        values = np.asarray([
            [
                next(
                    condition["results"][algorithm]["mean_reduction_pct"]
                    for condition in conditions
                    if condition["scenario"] == scenario_id
                    and condition["network_mode"] == network_mode
                )
                for algorithm in ALGORITHM_ORDER
            ]
            for scenario_id in SCENARIOS
        ])
        fig, axis = plt.subplots(figsize=(20, 10), constrained_layout=True)
        image = axis.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        axis.set_xticks(
            np.arange(len(ALGORITHM_ORDER)), labels, rotation=30, ha="right", fontsize=8
        )
        axis.set_yticks(
            np.arange(len(SCENARIOS)),
            [f"{scenario_id}  {SCENARIOS[scenario_id]['label']}" for scenario_id in SCENARIOS],
            fontsize=8,
        )
        for row_index in range(len(SCENARIOS)):
            for column_index in range(len(ALGORITHM_ORDER)):
                value = values[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > 55 else "#202020",
                )
        fig.colorbar(image, ax=axis, label="Curtailment reduction (%)")
        axis.set_title(f"{experiment_label} algorithm comparison: {network_mode}")
        path = output_dir / f"{filename_prefix}_algorithm_heatmap_{network_mode}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _plot_eps_comparison(
    conditions: list[dict[str, Any]],
    output_dir: Path,
    *,
    experiment_label: str = "E21 physical fleet mixing",
    filename_prefix: str = "e21",
) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), constrained_layout=True)
    x = np.arange(len(SCENARIOS), dtype=float)
    width = 0.36
    for axis, network_mode in zip(axes, NETWORK_MODES):
        for offset, algorithm in ((-0.5, "eps_e20_pooled"), (0.5, "eps_e21_mixed")):
            values = [
                next(
                    condition["results"][algorithm]["mean_reduction_pct"]
                    for condition in conditions
                    if condition["scenario"] == scenario_id
                    and condition["network_mode"] == network_mode
                )
                for scenario_id in SCENARIOS
            ]
            errors = [
                next(
                    condition["results"][algorithm]["std_reduction_pct"]
                    for condition in conditions
                    if condition["scenario"] == scenario_id
                    and condition["network_mode"] == network_mode
                )
                for scenario_id in SCENARIOS
            ]
            axis.bar(
                x + offset * width,
                values,
                width,
                yerr=errors,
                capsize=2,
                color=ALGORITHM_COLORS[algorithm],
                label=ALGORITHM_DEFINITIONS[algorithm]["label"],
            )
        axis.set_xticks(x, list(SCENARIOS), rotation=30)
        axis.set_ylim(0, 110)
        axis.set_ylabel("Curtailment reduction (%)")
        axis.set_title(network_mode)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle(f"{experiment_label} versus E20 sample pooling", fontsize=16)
    path = output_dir / f"{filename_prefix}_eps_mix_vs_pool.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _plot_overall(
    conditions: list[dict[str, Any]],
    output_dir: Path,
    *,
    experiment_label: str = "E21",
    filename_prefix: str = "e21",
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    y = np.arange(len(ALGORITHM_ORDER), dtype=float)
    labels = [
        f"{ALGORITHM_DEFINITIONS[algorithm]['label']}\n"
        f"{ALGORITHM_DEFINITIONS[algorithm]['complexity']}"
        for algorithm in ALGORITHM_ORDER
    ]
    colors = [ALGORITHM_COLORS[algorithm] for algorithm in ALGORITHM_ORDER]
    for axis, network_mode in zip(axes, NETWORK_MODES):
        values = np.asarray([
            [
                condition["results"][algorithm]["mean_reduction_pct"]
                for condition in conditions
                if condition["network_mode"] == network_mode
            ]
            for algorithm in ALGORITHM_ORDER
        ])
        means = np.mean(values, axis=1)
        errors = np.std(values, axis=1)
        axis.barh(y, means, xerr=errors, color=colors, capsize=3, alpha=0.9)
        for algorithm_index, scenario_values in enumerate(values):
            axis.scatter(
                scenario_values,
                np.full(len(scenario_values), algorithm_index),
                facecolors="white",
                edgecolors="#202020",
                linewidths=0.7,
                s=20,
                zorder=3,
            )
            axis.text(
                min(means[algorithm_index] + 1.0, 108.0),
                algorithm_index,
                f"{means[algorithm_index]:.1f}",
                va="center",
                fontsize=8,
            )
        axis.set_yticks(y, labels, fontsize=8)
        axis.set_xlim(0, 112)
        axis.set_xlabel("Curtailment reduction (%)")
        axis.set_title(network_mode)
        axis.grid(axis="x", alpha=0.25)
        axis.invert_yaxis()
    fig.suptitle(
        f"{experiment_label} macro comparison across 14 complex mixed scenarios\n"
        "bars: scenario mean; error bars: between-scenario SD; dots: individual scenarios",
        fontsize=15,
    )
    path = output_dir / f"{filename_prefix}_algorithm_overall.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def main() -> None:
    _validate_scenarios()
    parser = argparse.ArgumentParser(description="Command line entry point for run e21 mixed scenarios.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=max(1, min(2, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-models", action="store_true")
    args = parser.parse_args()
    root = args.results_root / "E21"
    model_dir = root / "models"
    data_dir = root / "data"
    figure_dir = root / "figures"
    for directory in (model_dir, data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(root)
    if lock is None:
        print(f"{root} already holds a running task; this instance exits")
        return
    result_path = data_dir / "e21_results.json"
    manifest_path = root / "e21_manifest.json"
    checkpoint_path = data_dir / "e21_evaluation_checkpoint.json"
    training_path = data_dir / "e21_training.json"
    if result_path.is_file() and manifest_path.is_file() and not args.force:
        print(result_path)
        return

    if training_path.is_file() and not args.retrain_models:
        training_metadata = json.loads(training_path.read_text(encoding="utf-8"))
    else:
        training_metadata = {"protocol": PROTOCOL, "models": {}}
    for network_mode in NETWORK_MODES:
        model_path = model_dir / f"e21_mixed_{network_mode}.pt"
        reusable = (
            not args.retrain_models
            and model_path.is_file()
            and network_mode in training_metadata.get("models", {})
        )
        if reusable:
            continue
        training_metadata.setdefault("models", {})[network_mode] = _train_model(
            network_mode, model_path
        )
        _write_json(training_path, training_metadata)

    tasks = [
        (
            scenario_id,
            network_mode,
            str(args.results_root / "E20" / "models" / f"pooled_{network_mode}.pt"),
            str(model_dir / f"e21_mixed_{network_mode}.pt"),
        )
        for network_mode in NETWORK_MODES
        for scenario_id in SCENARIOS
    ]
    conditions: list[dict[str, Any]] = []
    if checkpoint_path.is_file() and not args.force:
        conditions = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed = {
        (condition["scenario"], condition["network_mode"])
        for condition in conditions
    }
    tasks = [task for task in tasks if (task[0], task[1]) not in completed]
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate_condition, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                condition = future.result()
            except Exception as exc:
                failure = {
                    "scenario": task[0],
                    "network_mode": task[1],
                    "error": repr(exc),
                }
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
                continue
            conditions.append(condition)
            _write_json(checkpoint_path, conditions)
            print(json.dumps({
                "stage": "evaluation",
                "scenario": condition["scenario"],
                "network_mode": condition["network_mode"],
                "completed_conditions": len(conditions),
                "condition_count": len(SCENARIOS) * len(NETWORK_MODES),
                "eps_e21_mean": condition["results"]["eps_e21_mixed"]["mean_reduction_pct"],
            }, ensure_ascii=False), flush=True)
    if failures:
        _write_json(manifest_path, {
            "protocol": PROTOCOL,
            "status": "failed",
            "completed_count": len(conditions),
            "failed_count": len(failures),
            "failures": failures,
            "checkpoint": str(checkpoint_path),
        })
        raise SystemExit(1)

    conditions.sort(key=lambda condition: (
        NETWORK_MODES.index(condition["network_mode"]),
        list(SCENARIOS).index(condition["scenario"]),
    ))
    summary_path = data_dir / "e21_summary.csv"
    _write_summary_csv(conditions, summary_path)
    figures = _plot_heatmaps(conditions, figure_dir)
    figures.append(_plot_eps_comparison(conditions, figure_dir))
    figures.append(_plot_overall(conditions, figure_dir))
    payload = {
        "protocol": PROTOCOL,
        "fleet_size": FLEET_SIZE,
        "availability_mode": AVAILABILITY_MODE,
        "network_modes": list(NETWORK_MODES),
        "seed_count": SEED_COUNT,
        "training_samples_per_scenario": TRAIN_SAMPLES_PER_SCENARIO,
        "validation_samples_per_scenario": VALIDATION_SAMPLES_PER_SCENARIO,
        "scenario_definitions": SCENARIOS,
        "algorithm_order": list(ALGORITHM_ORDER),
        "algorithm_definitions": ALGORITHM_DEFINITIONS,
        "online_protocol": {
            "input_features": 10,
            "controller_to_device": "one broadcast signal",
            "device_to_controller": "one aggregate scalar feedback per step",
            "per_device_ack": False,
            "per_device_state_upload": False,
            "communication_complexity": "O(1) with respect to N",
        },
        "training": training_metadata,
        "conditions": conditions,
        "summary_csv": str(summary_path),
        "figures": figures,
    }
    _write_json(result_path, payload)
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "scenario_count": len(SCENARIOS),
        "condition_count": len(conditions),
        "algorithm_count": len(ALGORITHM_ORDER),
        "seed_count": SEED_COUNT,
        "failed_count": 0,
        "result": str(result_path),
        "summary_csv": str(summary_path),
        "figures": figures,
    }
    _write_json(manifest_path, manifest)
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
