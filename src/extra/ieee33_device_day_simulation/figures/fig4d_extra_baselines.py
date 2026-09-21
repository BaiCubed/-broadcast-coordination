from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import json
import os
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.estimation import EPSEstimator, EstimatorConfig
from src.signal import OptimizationTarget, SignalOptimizer
from ..config_loader import load_config
from ..network.ieee33_distflow import IEEE33DistFlow
from ..original_adapter.eps_adapter import OriginalEPSAdapter
from ..population.device_day_loader import DeviceDay, load_device_day_pool
from src.extra.nc_excel_experiments.coupling import diurnal_residual, rank_availability


FIG4_NAME = "fig4_robustness_generalization.png"
FIG4_2_NAME = "fig4-2.png"
EXTRA_BASELINE_JSON = "curtailment_baselines_extra.json"
EXTRA_CONFIG_JSON = "fig4d_extended_baselines_config.json"
EPS_TRAINING_JSON = "fig4d_eps_training.json"
MANIFEST_NAME = "fig4d_extended_baselines_manifest.json"
DATASET_SCALED_SUFFIX = "_dataset_scaled"
SEEDS = 10
STEPS = 288
EPS_TRAINING_EPOCHS = 100
MIN_EPS_TRAINING_SAMPLES = 256
MAX_EPS_TRAINING_SAMPLES = 12000
MIN_EPS_VALIDATION_SAMPLES = 64
MAX_EPS_VALIDATION_SAMPLES = 1200
EPS_SAMPLES_PER_SOURCE = 4
PROTOCOL = "fig4d_extended_baseline_comparison_v7_dataset_scaled_training"
BASELINE_VERSION = "fig4d_extended_baselines_v7_dataset_scaled_training"
FULL_AVAILABILITY_PROTOCOL = "fig4d_extended_baseline_comparison_v8_dataset_scaled_training_full_availability"
FULL_AVAILABILITY_BASELINE_VERSION = "fig4d_extended_baselines_v8_dataset_scaled_training_full_availability"
CAPACITY_AUDIT_PROTOCOL = "fig4d_extended_baseline_comparison_v9_dataset_scaled_capacity_sufficient_audit"
CAPACITY_AUDIT_BASELINE_VERSION = "fig4d_extended_baselines_v9_dataset_scaled_capacity_sufficient_audit"
CAPACITY_AUDIT_THRESHOLD_PCT = 99.0
CAPACITY_AUDIT_MAX_MULTIPLIER = 128.0
LOCAL_CHARGE_HOURS = range(9, 17)
LOCAL_DISCHARGE_HOURS = range(17, 21)
LOCAL_CHARGE_SOC_THRESHOLD = 0.80
LOCAL_DISCHARGE_SOC_THRESHOLD = 0.30
PEM_PACKET_STEPS = 3
PEM_PACKET_POWER_FRACTION = 0.45
TRANSACTIVE_CLEARING_STEPS = 3
TRANSACTIVE_POWER_FRACTION = 0.60
TRANSACTIVE_PRICE_TICK = 0.05
TRANSACTIVE_PRICE_TEMPERATURE = 0.08
TRANSACTIVE_PRICE_INERTIA = 0.65
MPC_HORIZON_STEPS = 12
MEAN_FIELD_SOC_BINS = 10
MEAN_FIELD_CONTROL_STEPS = 3

FIG4D_ORDER = [
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_broadcast",
    "centralized_optimal",
]

FIG4D_LABELS = {
    "no_coordination": "No coord.\nO(0)",
    "local_rules": "Local\nSOC rules\nO(1)/device",
    "mpc_optimal": "MPC\nO(HN)",
    "mean_field_control": "Mean-field\ncontrol\nO(N)",
    "virtual_battery": "Virtual\nbattery\nO(N)",
    "packetized_energy_management": "Packetized\nEnergy Mgmt\nO(N log N)",
    "transactive_control": "Transactive\ncontrol\nO(N log N)",
    "eps_broadcast": "EPS\nbroadcast\nO(1)",
    "centralized_optimal": "Centralized\ngreedy UB\nO(N)",
}

FIG4D_COLORS = {
    "no_coordination": "#aaaaaa",
    "local_rules": "#ed7d31",
    "mpc_optimal": "#7b61ff",
    "mean_field_control": "#4e79a7",
    "virtual_battery": "#59a14f",
    "packetized_energy_management": "#b07aa1",
    "transactive_control": "#f28e2b",
    "eps_broadcast": "#069c8f",
    "centralized_optimal": "#4c9f70",
}

STRATEGY_DEFINITIONS = {
    "no_coordination": {"label": "No coordination", "complexity": "O(0)", "description": "No control signal is sent."},
    "local_rules": {"label": "Local SOC rules", "complexity": "O(1) per device", "description": "Devices charge and discharge independently from their own SOC, availability and time-of-day thresholds, reading neither the surplus nor any other device state."},
    "mpc_optimal": {"label": "MPC", "complexity": "O(HN)", "description": "A receding water-filling allocation over a 12-step surplus and availability forecast under the remaining grid-side capacity budget, charging low-SOC devices first."},
    "mean_field_control": {"label": "Mean-field control", "complexity": "O(N)", "description": "A common participation probability is updated every 15 minutes from the predicted population distribution over 10 SOC bins and the expected availability; each device responds locally at random according to its own bin, with no post-hoc normalisation to the target."},
    "virtual_battery": {"label": "Virtual battery", "complexity": "O(N)", "description": "Only the aggregate energy and power envelope is maintained online and mapped back to the devices by fixed power weights; the real device headroom then causes dataset-dependent clipping."},
    "packetized_energy_management": {"label": "Packetized Energy Management", "complexity": "O(N log N)", "description": "Devices submit indivisible 15-minute fixed-power packets; the aggregator accepts only whole packets and an accepted packet runs to its end."},
    "transactive_control": {"label": "Transactive control", "complexity": "O(N log N)", "description": "Devices bid according to their SOC and load variation; the market clears at a discrete price every 15 minutes and devices respond independently to the standing price."},
    "eps_broadcast": {"label": "EPS broadcast", "complexity": "O(1)", "description": "An offline dual quantile response estimator; the online broadcast control is independent of the device count N, i.e. O(1). Packet loss, offline devices and the Bernoulli response remain on the device side."},
    "centralized_optimal": {"label": "Centralized greedy UB", "complexity": "O(N)", "description": "The charging headroom of each device with one linear greedy allocation of the theoretically absorbable curtailment; this bound applies no network clipping."},
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _choose_config_path(result_root: Path) -> Path:
    candidates = [
        result_root / "config" / "default_network_constrained_new.yaml",
        result_root / "config" / "default_network_stress.yaml",
        result_root / "config" / "default.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"configuration not found: {result_root}")


def _dataset_training_protocol(pool: Any, config: dict[str, Any]) -> dict[str, Any]:
    source_count = int(pool.source_count)
    configured_fleet_size = int(config["population"]["N_simulated_resources"])
    if source_count <= 0 or configured_fleet_size <= 0:
        raise ValueError("the dataset source count and the configured fleet size must be positive")
    fleet_size = min(source_count, configured_fleet_size)
    training_samples = int(np.clip(
        EPS_SAMPLES_PER_SOURCE * fleet_size,
        MIN_EPS_TRAINING_SAMPLES,
        MAX_EPS_TRAINING_SAMPLES,
    ))
    validation_samples = int(np.clip(
        round(training_samples * 0.10),
        MIN_EPS_VALIDATION_SAMPLES,
        MAX_EPS_VALIDATION_SAMPLES,
    ))
    return {
        "pool_source_count": source_count,
        "pool_device_day_count": int(len(pool.records)),
        "configured_fleet_size": configured_fleet_size,
        "fleet_size": fleet_size,
        "fleet_size_rule": "min(N_simulated_resources, pool_source_count)",
        "training_samples": training_samples,
        "training_samples_rule": "clip(4 * fleet_size, 256, 12000)",
        "validation_samples": validation_samples,
        "validation_samples_rule": "clip(round(0.10 * training_samples), 64, 1200)",
        "source_unique_fleet": True,
        "bootstrap": False,
        "small_source_dataset": source_count < 500,
    }


def _sample_records(pool: Any, count: int, seed: int) -> tuple[list[DeviceDay], dict[str, Any]]:
    records = pool.sample_unique_sources(count, seed)
    unique_sources = len({record.source_device_id for record in records})
    if unique_sources != count:
        raise RuntimeError(
            f"a source-unique fleet needs {count} independent sources but got {unique_sources}"
        )
    return records, {
        "mode": "source_unique_no_bootstrap",
        "requested_resources": count,
        "unique_sources": unique_sources,
        "bootstrap": False,
    }


def _scale_records(records: list[DeviceDay], multiplier: float) -> list[DeviceDay]:
    if multiplier < 1.0:
        raise ValueError("the capacity audit multiplier must be at least 1")
    return [
        replace(
            record,
            soc_kwh=np.asarray(record.soc_kwh, dtype=float) * multiplier,
            capacity_kwh=float(record.capacity_kwh) * multiplier,
            peak_power_kw=float(record.peak_power_kw) * multiplier,
        )
        for record in records
    ]


def _capacity_audit_multiplier(reference: dict[str, Any]) -> float:
    central = reference["results"]["centralized_optimal"]
    baseline = float(central["baseline_curtailment_mwh"])
    accepted = float(central["accepted_absorption_mwh"])
    if baseline <= 1e-12 or accepted <= 1e-12:
        return CAPACITY_AUDIT_MAX_MULTIPLIER
    required = baseline / accepted
    multiplier = 1.0
    while multiplier + 1e-12 < required and multiplier < CAPACITY_AUDIT_MAX_MULTIPLIER:
        multiplier *= 2.0
    return min(multiplier, CAPACITY_AUDIT_MAX_MULTIPLIER)


def _centralized_capacity_reduction(
    pool: Any,
    config: dict[str, Any],
    training_protocol: dict[str, Any],
    multiplier: float,
) -> float:
    audit_config = dict(config)
    audit_config["control"] = dict(config["control"])
    audit_config["control"]["network_capacity_multiplier"] = (
        float(config["control"].get("network_capacity_multiplier", 1.0))
        * multiplier
    )
    rows: list[dict[str, Any]] = []
    for seed_index in range(SEEDS):
        seed = int(config["simulation"]["random_seed"]) + seed_index
        records, _ = _sample_records(
            pool, int(training_protocol["fleet_size"]), seed
        )
        records = _scale_records(records, multiplier)
        scenario = _build_scenario(records, audit_config)
        availability = _availability_probability(
            records,
            diurnal_residual(records, STEPS),
            audit_config,
            "zone_calibrated",
        )
        rows.append(_run_seed(
            "centralized_optimal",
            records,
            audit_config,
            scenario,
            availability,
            seed + 900000,
            seed + 910000,
        ))
    return float(np.mean([row["mean_reduction_pct"] for row in rows]))


def _select_capacity_audit_multiplier(
    result_root: Path, reference: dict[str, Any]
) -> tuple[float, list[dict[str, float]]]:
    config = _prepare_config(_choose_config_path(result_root))
    pool = load_device_day_pool(config)
    protocol = _dataset_training_protocol(pool, config)
    multiplier = max(2.0, _capacity_audit_multiplier(reference))
    trials: list[dict[str, float]] = []
    while multiplier <= CAPACITY_AUDIT_MAX_MULTIPLIER:
        reduction = _centralized_capacity_reduction(
            pool, config, protocol, multiplier
        )
        trials.append({
            "multiplier": multiplier,
            "centralized_reduction_pct": reduction,
        })
        if reduction >= CAPACITY_AUDIT_THRESHOLD_PCT:
            return multiplier, trials
        multiplier *= 2.0
    raise RuntimeError(
        f"the capacity audit does not reach the target even at {CAPACITY_AUDIT_MAX_MULTIPLIER:g} times capacity: "
        f"{CAPACITY_AUDIT_THRESHOLD_PCT:g}%: {result_root}"
    )


def _calibrated_availability_probability(
    records: list[DeviceDay], residual: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    rank_probability = rank_availability(residual)
    if rank_probability.shape[1] != len(records):
        raise ValueError("the availability matrix and the sampled fleet have different device counts")
    zone_config = config["zones"]["zones"]
    target_probability = np.asarray([
        float(zone_config[record.zone_id]["availability_multiplier"])
        for record in records
    ])
    if np.any((target_probability <= 0.0) | (target_probability >= 1.0)):
        raise ValueError("the zone availability_multiplier must lie strictly inside (0, 1)")

    clipped = np.clip(rank_probability, 1e-9, 1.0 - 1e-9)
    log_odds = np.log(clipped / (1.0 - clipped))
    calibrated = np.empty_like(rank_probability, dtype=float)
    for target in np.unique(target_probability):
        columns = np.flatnonzero(np.isclose(target_probability, target))
        values = log_odds[:, columns]
        lower, upper = -40.0, 40.0
        for _ in range(80):
            shift = 0.5 * (lower + upper)
            probability = 1.0 / (1.0 + np.exp(-np.clip(values + shift, -60.0, 60.0)))
            if float(np.mean(probability)) < float(target):
                lower = shift
            else:
                upper = shift
        shift = 0.5 * (lower + upper)
        calibrated[:, columns] = 1.0 / (
            1.0 + np.exp(-np.clip(values + shift, -60.0, 60.0))
        )
    return calibrated


def _availability_probability(
    records: list[DeviceDay], residual: np.ndarray, config: dict[str, Any], mode: str
) -> np.ndarray:
    if mode == "zone_calibrated":
        return _calibrated_availability_probability(records, residual, config)
    if mode == "full":
        return np.ones_like(residual, dtype=float)
    raise ValueError(f"unknown availability mode: {mode}")


def _availability_summary(
    records: list[DeviceDay], availability_probability: np.ndarray
) -> dict[str, Any]:
    zone_ids = np.asarray([record.zone_id for record in records], dtype=object)
    return {
        "mean_probability": float(np.mean(availability_probability)),
        "zone_mean_probability": {
            zone_id: float(np.mean(availability_probability[:, zone_ids == zone_id]))
            for zone_id in sorted(set(zone_ids))
        },
    }


def _renewable_profile() -> np.ndarray:
    hour = np.arange(STEPS, dtype=float) / 12.0
    solar = np.where((hour >= 6.0) & (hour <= 18.0), np.sin(np.pi * (hour - 6.0) / 12.0), 0.0)
    wind = 0.56 + 0.13 * np.sin(2.0 * np.pi * (hour + 1.5) / 24.0) + 0.08 * np.sin(4.0 * np.pi * (hour - 2.0) / 24.0)
    profile = 0.68 * solar + 0.32 * np.maximum(wind, 0.05)
    return profile / max(float(np.max(profile)), 1e-12)


def _build_scenario(records: list[DeviceDay], config: dict[str, Any]) -> dict[str, Any]:
    zones = config["zones"]["zones"]
    buses = sorted({int(branch[1]) for branch in config["network"]["branches"]} | {int(config["network"]["slack_bus"])})
    bus_index = {bus: i for i, bus in enumerate(buses)}
    load_by_bus = np.zeros((STEPS, len(buses)), dtype=float)
    for record in records:
        zcfg = zones[record.zone_id]
        load_by_bus[:, bus_index[record.bus_id]] += np.asarray(record.load_kw[:STEPS], dtype=float) * float(zcfg.get("load_multiplier", 1.0))
    load = np.sum(load_by_bus, axis=1)
    profile = _renewable_profile()
    target_energy_kw_steps = 0.20 * float(np.sum(load))

    def curtailment(alpha: float) -> float:
        return float(np.sum(np.maximum(alpha * profile - load, 0.0)))

    low, high = 0.0, max(float(np.max(load)) * 4.0, 1.0)
    while curtailment(high) < target_energy_kw_steps:
        high *= 2.0
    for _ in range(70):
        middle = (low + high) / 2.0
        if curtailment(middle) < target_energy_kw_steps:
            low = middle
        else:
            high = middle
    alpha = high
    input_total = alpha * profile
    input_by_bus = np.zeros_like(load_by_bus)
    for step in range(STEPS):
        weights = load_by_bus[step]
        total = float(np.sum(weights))
        if total > 1e-12:
            input_by_bus[step] = input_total[step] * weights / total
        else:
            input_by_bus[step] = input_total[step] / len(buses)
    return {
        "buses": buses,
        "bus_index": bus_index,
        "load_by_bus": load_by_bus,
        "input_by_bus": input_by_bus,
        "load": load,
        "input_total": input_total,
        "renewable_profile": profile,
        "renewable_scale_alpha_kw": float(alpha),
        "no_control_curtailment_fraction": curtailment(alpha) / max(float(np.sum(load)), 1e-12),
    }


def _headroom(soc: np.ndarray, capacities: np.ndarray, peaks: np.ndarray, config: dict[str, Any], available: np.ndarray | None = None) -> np.ndarray:
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    eta = float(config["original_model"].get("charge_efficiency", 0.95))
    soc_max = float(config["device_constraints"]["soc_max"])
    value = np.minimum(peaks, np.maximum(soc_max - soc, 0.0) * capacities / max(dt * eta, 1e-12))
    return value if available is None else value * available


def _discharge_headroom(soc: np.ndarray, capacities: np.ndarray, peaks: np.ndarray, config: dict[str, Any], available: np.ndarray | None = None) -> np.ndarray:
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    eta = float(config["original_model"].get("charge_efficiency", 0.95))
    soc_min = float(config["device_constraints"]["soc_min"])
    value = np.minimum(
        peaks,
        np.maximum(soc - soc_min, 0.0) * capacities * eta / max(dt, 1e-12),
    )
    return value if available is None else value * available


def _curtailment_absorbed_kw(accepted: np.ndarray, surplus: float) -> float:
    charging_kw = max(float(np.sum(np.maximum(accepted, 0.0))), 0.0)
    return min(charging_kw, max(float(surplus), 0.0))


def _allocate(headroom: np.ndarray, target: float, priority: np.ndarray | None = None) -> np.ndarray:
    result = np.zeros_like(headroom, dtype=float)
    eligible = np.flatnonzero(headroom > 1e-10)
    if target <= 1e-12 or eligible.size == 0:
        return result
    if priority is None:
        weights = headroom[eligible]
        result[eligible] = weights * min(1.0, target / max(float(np.sum(weights)), 1e-12))
        return result
    order = eligible[np.argsort(-priority[eligible], kind="stable")]
    remaining = float(target)
    for index in order:
        amount = min(float(headroom[index]), remaining)
        result[index] = amount
        remaining -= amount
        if remaining <= 1e-10:
            break
    return result


def _weighted_allocate(
    headroom: np.ndarray, target: float, weights: np.ndarray
) -> np.ndarray:
    eligible = (headroom > 1e-10) & (weights > 1e-10)
    if target <= 1e-12 or not np.any(eligible):
        return np.zeros_like(headroom, dtype=float)
    target = min(float(target), float(np.sum(headroom[eligible])))
    low = 0.0
    high = float(np.max(headroom[eligible] / weights[eligible]))
    for _ in range(45):
        level = (low + high) / 2.0
        allocated = np.minimum(headroom, level * weights)
        if float(np.sum(allocated)) < target:
            low = level
        else:
            high = level
    result = np.minimum(headroom, high * weights)
    total = float(np.sum(result))
    if total > target + 1e-10:
        result *= target / total
    return result


def _waterfill_current(surplus: np.ndarray, energy_limit_kwh: float, dt: float) -> float:
    if energy_limit_kwh <= 1e-12:
        return 0.0
    total = float(np.sum(surplus) * dt)
    if total <= energy_limit_kwh:
        return float(surplus[0])
    low, high = 0.0, float(np.max(surplus))
    for _ in range(45):
        level = (low + high) / 2.0
        if float(np.sum(np.maximum(surplus - level, 0.0)) * dt) > energy_limit_kwh:
            low = level
        else:
            high = level
    return float(max(surplus[0] - high, 0.0))


def _mpc_energy_limit_kwh(
    soc: np.ndarray,
    capacities: np.ndarray,
    config: dict[str, Any],
) -> float:
    eta = float(config["original_model"].get("charge_efficiency", 0.95))
    soc_max = float(config["device_constraints"]["soc_max"])
    stored_headroom = np.maximum(soc_max - soc, 0.0) * capacities
    return float(np.sum(stored_headroom) / max(eta, 1e-12))


def _initialize_mean_field_model(
    soc: np.ndarray,
    capacities: np.ndarray,
    peaks: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    soc_min = float(config["device_constraints"]["soc_min"])
    soc_max = float(config["device_constraints"]["soc_max"])
    edges = np.linspace(soc_min, soc_max, MEAN_FIELD_SOC_BINS + 1)
    bins = np.clip(
        np.digitize(soc, edges[1:-1], right=False), 0, MEAN_FIELD_SOC_BINS - 1
    )
    return {
        "edges": edges,
        "count": np.bincount(bins, minlength=MEAN_FIELD_SOC_BINS).astype(float),
        "capacity": np.bincount(
            bins, weights=capacities, minlength=MEAN_FIELD_SOC_BINS
        ).astype(float),
        "power": np.bincount(
            bins, weights=peaks, minlength=MEAN_FIELD_SOC_BINS
        ).astype(float),
        "energy": np.bincount(
            bins, weights=soc * capacities, minlength=MEAN_FIELD_SOC_BINS
        ).astype(float),
        "participation": np.zeros(MEAN_FIELD_SOC_BINS, dtype=float),
    }


def _mean_field_dispatch(
    step: int,
    target: float,
    soc: np.ndarray,
    capacities: np.ndarray,
    peaks: np.ndarray,
    headroom: np.ndarray,
    available: np.ndarray,
    rng: np.random.Generator,
    state: dict[str, Any],
    config: dict[str, Any],
) -> np.ndarray:
    soc_min = float(config["device_constraints"]["soc_min"])
    soc_max = float(config["device_constraints"]["soc_max"])
    model = state.setdefault(
        "mean_field_model",
        _initialize_mean_field_model(soc, capacities, peaks, config),
    )
    edges = np.asarray(model["edges"], dtype=float)
    if step % MEAN_FIELD_CONTROL_STEPS == 0:
        expected_availability = float(np.mean(
            np.asarray(state["availability_probability"], dtype=float)[step]
        ))
        mean_soc = np.divide(
            model["energy"],
            model["capacity"],
            out=(edges[:-1] + edges[1:]) / 2.0,
            where=model["capacity"] > 1e-12,
        )
        urgency = np.clip(
            (soc_max - mean_soc) / max(soc_max - soc_min, 1e-12), 0.05, 1.0
        )
        stored_headroom = np.maximum(
            soc_max * model["capacity"] - model["energy"], 0.0
        )
        dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
        eta = float(config["original_model"].get("charge_efficiency", 0.95))
        expected_power = np.minimum(
            model["power"] * expected_availability,
            stored_headroom / max(dt * eta, 1e-12),
        )
        weighted = expected_power * urgency
        denominator = float(np.sum(weighted))
        model["participation"] = np.clip(
            target * urgency / max(denominator, 1e-12), 0.0, 1.0
        )
    local_bins = np.clip(
        np.digitize(soc, edges[1:-1], right=False), 0, MEAN_FIELD_SOC_BINS - 1
    )
    selected = (
        rng.random(len(soc)) < np.asarray(model["participation"])[local_bins]
    ).astype(float) * available
    desired = headroom * selected
    expected_availability = float(np.mean(
        np.asarray(state["availability_probability"], dtype=float)[step]
    ))
    expected_input = np.minimum(
        model["power"] * expected_availability * model["participation"],
        np.maximum(soc_max * model["capacity"] - model["energy"], 0.0)
        / max(
            float(config["simulation"]["time_step_seconds"]) / 3600.0
            * float(config["original_model"].get("charge_efficiency", 0.95)),
            1e-12,
        ),
    )
    model["energy"] = np.minimum(
        soc_max * model["capacity"],
        model["energy"]
        + expected_input
        * float(config["simulation"]["time_step_seconds"]) / 3600.0
        * float(config["original_model"].get("charge_efficiency", 0.95)),
    )
    return desired


def _apply_step(network: IEEE33DistFlow, scenario: dict[str, Any], step: int, desired: np.ndarray, records: list[DeviceDay], config: dict[str, Any]) -> tuple[np.ndarray, float, dict[str, float]]:
    desired = np.asarray(desired, dtype=float)
    aggregate_scale = 1.0
    network_mode = config["control"].get("final_network_mode")
    aggregate_limit = scenario.get("aggregate_hosting_limit_kw")
    if aggregate_limit is not None and network_mode != "ieee33":
        requested = float(np.sum(np.maximum(desired, 0.0)))
        if requested > 1e-12:
            aggregate_scale = min(
                1.0,
                float(np.asarray(aggregate_limit, dtype=float)[step]) / requested,
            )
        desired = desired * aggregate_scale
    if network_mode == "aggregate":
        return desired, aggregate_scale, {
            "max_branch_loading": 0.0,
            "transformer_loading": 0.0,
            "min_voltage_pu": 1.0,
            "voltage_violations": 0.0,
            "overloaded_branches": 0.0,
            "added_network_violation": 0.0,
        }
    bus_index = scenario["bus_index"]
    load_map = {bus: float(scenario["load_by_bus"][step, bus_index[bus]]) for bus in scenario["buses"]}
    input_map = {bus: float(scenario["input_by_bus"][step, bus_index[bus]]) for bus in scenario["buses"]}
    device_bus_indices = scenario.get("device_bus_indices")
    if device_bus_indices is None:
        device_bus_indices = np.asarray([bus_index[record.bus_id] for record in records], dtype=int)
    desired_by_bus = np.bincount(
        np.asarray(device_bus_indices, dtype=int),
        weights=desired,
        minlength=len(scenario["buses"]),
    )
    desired_map = {
        bus: float(desired_by_bus[index])
        for bus, index in bus_index.items()
        if abs(float(desired_by_bus[index])) > 1e-12
    }
    baseline = network.evaluate(load_map, input_map, {}, apply_limits=False)
    if config["control"].get("network_feedback", True) and desired_map:
        scale, _ = network.constrained_dispatch_scale(load_map, input_map, desired_map)
    else:
        scale = 1.0
    accepted = desired * float(np.clip(scale, 0.0, 1.0))
    accepted_map = {bus: amount * scale for bus, amount in desired_map.items()}
    evaluation = network.evaluate(load_map, input_map, accepted_map, apply_limits=False)
    added_violation = (
        evaluation.voltage_violations > baseline.voltage_violations
        or evaluation.overloaded_branches > baseline.overloaded_branches
        or evaluation.transformer_loading > max(1.0, baseline.transformer_loading) + 1e-9
    )
    return accepted, float(aggregate_scale * scale), {
        "max_branch_loading": float(max(evaluation.branch_loading.values(), default=0.0)),
        "transformer_loading": float(evaluation.transformer_loading),
        "min_voltage_pu": float(min(evaluation.voltage_pu.values(), default=1.0)),
        "voltage_violations": float(evaluation.voltage_violations),
        "overloaded_branches": float(evaluation.overloaded_branches),
        "added_network_violation": float(added_violation),
    }


def _eps_signal(score: float, step: int) -> dict[str, Any]:
    return {
        "intensity": int(np.clip(round(score * 4095), 0, 4095)),
        "supply_demand": 4,
        "region_id": 0,
        "priority": 10,
        "hour": int(step // 12),
        "day_of_week": 0,
    }


def _train_eps_controller(
    pool: Any,
    config: dict[str, Any],
    seed: int,
    availability_mode: str,
    training_protocol: dict[str, Any],
    capacity_multiplier: float = 1.0,
) -> tuple[EPSEstimator, SignalOptimizer, dict[str, Any]]:
    from src.extra.real_data_experiment.original_run_experiment import (
        EmergencyChargeProfile,
        EmergencyDischargeProfile,
        PeakShavingProfile,
        ValleyFillingProfile,
        encode_signal_score,
    )

    fleet_size = int(training_protocol["fleet_size"])
    training_samples = int(training_protocol["training_samples"])
    validation_samples = int(training_protocol["validation_samples"])
    records, sample_metadata = _sample_records(pool, fleet_size, seed)
    records = _scale_records(records, capacity_multiplier)
    scenario = _build_scenario(records, config)
    residual = diurnal_residual(records, STEPS)
    availability_probability = _availability_probability(
        records, residual, config, availability_mode
    )
    soc_min = float(config["device_constraints"]["soc_min"])
    soc_max = float(config["device_constraints"]["soc_max"])
    network = IEEE33DistFlow(config["network"], config["control"])
    rng = np.random.default_rng(seed + 1)
    adapter = OriginalEPSAdapter(records, config, seed=seed + 2)
    sample_count = training_samples + validation_samples
    scenario_count = int(sample_count * 0.80)
    time_aware_count = int(sample_count * 0.10)
    profiles = [
        ValleyFillingProfile(),
        PeakShavingProfile(),
        EmergencyChargeProfile(),
        EmergencyDischargeProfile(),
    ]
    samples_per_profile = max(1, scenario_count // len(profiles))
    zone_count = len(config["zones"]["zones"])
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    for sample_index in range(sample_count):
        if sample_index < scenario_count:
            profile_index = min(sample_index // samples_per_profile, len(profiles) - 1)
            score = float(np.clip(profiles[profile_index].get_signal_score(rng.uniform(0.0, 1.0)) + rng.normal(0.0, 0.02), -1.0, 1.0))
            supply_demand, intensity = encode_signal_score(score)
            hour = int(rng.integers(0, 24))
        elif sample_index < scenario_count + time_aware_count:
            hour = int(rng.integers(0, 24))
            if (7 <= hour <= 11) or (17 <= hour <= 21):
                intensity = int(rng.normal(2800, 500) if rng.random() < 0.5 else rng.normal(1000, 400))
                supply_demand = int(rng.normal(11, 2))
            elif hour <= 6 or hour >= 22:
                intensity = int(rng.normal(1000, 300))
                supply_demand = int(rng.normal(3, 2))
            else:
                intensity = int(rng.normal(2048, 600))
                supply_demand = int(rng.normal(8, 2))
        else:
            hour = int(rng.integers(0, 24))
            intensity = int(rng.uniform(0, 4095))
            supply_demand = int(rng.uniform(0, 16))
        intensity = int(np.clip(intensity, 0, 4095))
        supply_demand = int(np.clip(supply_demand, 0, 15))
        profile_step = int(hour * 12 + rng.integers(0, 12))
        adapter.reset_to_initial_state()
        for record, battery in zip(records, adapter.simulator.population.batteries):
            measured_soc = float(record.soc_kwh[profile_step] / max(record.capacity_kwh, 1e-9))
            battery.soc = float(np.clip(measured_soc, soc_min, soc_max))
            battery.current_power_kw = 0.0
            battery.thermal.cell_temperature = 25.0
        energy_kwh = 0.0
        for offset in range(3):
            step = (profile_step + offset) % STEPS
            eps_signals = [
                adapter.simulator._signal_generator.generate_signal(zone, supply_demand, intensity, priority=10)
                for zone in range(zone_count)
            ]
            batch = adapter.step(eps_signals, step, availability_probability[step])
            accepted, _, _ = _apply_step(network, scenario, step, np.asarray(batch.desired_kw, dtype=float), records, config)
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
        responses.append(energy_kwh / max(3.0 * dt, 1e-12))
        if (sample_index + 1) % 2000 == 0:
            dataset = str(config["population"].get("canonical_adapter", {}).get("dataset", "unknown"))
            print(f"[{dataset}] original EPS training samples {sample_index + 1}/{sample_count}", flush=True)

    train_signals = signals[:training_samples]
    train_responses = responses[:training_samples]
    validation_signals = signals[training_samples:]
    validation_responses = np.asarray(responses[training_samples:], dtype=float)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = EPSEstimator(EstimatorConfig(
        target_coverage=0.9,
        enable_conformal=True,
        use_cqr=True,
        use_pytorch=True,
        pytorch_epochs=EPS_TRAINING_EPOCHS,
        pytorch_batch_size=64,
        pytorch_learning_rate=0.001,
    ))
    estimator.fit(train_signals, train_responses)
    prediction = np.asarray([estimator.estimate(signal).response_kw for signal in validation_signals], dtype=float)
    residual_error = validation_responses - prediction
    denominator = float(np.sum((validation_responses - np.mean(validation_responses)) ** 2))
    validation_r2 = 1.0 - float(np.sum(residual_error ** 2)) / max(denominator, 1e-12)
    training_parameters = estimator._learned_params or {}
    metadata = {
        "method": "OriginalEPSAdapter/EPSSimulator 3x5min responses + Dual Quantile NN + CQR + SignalOptimizer",
        "training_seed": int(seed),
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "epochs": EPS_TRAINING_EPOCHS,
        "model_type": str(training_parameters.get("model_type", "unknown")),
        "internal_training_r2": float(training_parameters.get("r2", float("nan"))),
        "held_out_r2": float(validation_r2),
        "held_out_rmse_kw": float(np.sqrt(np.mean(residual_error ** 2))),
        "held_out_mae_kw": float(np.mean(np.abs(residual_error))),
        "response_range_kw": [float(np.min(responses)), float(np.max(responses))],
        "sampling": sample_metadata,
        "dataset_scale_protocol": training_protocol,
        "capacity_multiplier": capacity_multiplier,
        "training_fleet_is_separate_draw": True,
        "validation_split_unit": "independently simulated control-signal response samples",
        "source_holdout": False,
        "source_holdout_reason": "The estimator predicts aggregate response for the dataset-sized fleet; changing source count changes the response target scale.",
        "training_signal_mix": "80% original scenario profiles + 10% time-aware + 10% uniform",
        "physics_steps_per_sample": 3,
        "response_generator": "OriginalEPSAdapter.step + IEEE33DistFlow + OriginalEPSAdapter.apply_dispatch",
        "availability_calibration": {
            "mode": availability_mode,
            "method": (
                "external availability fixed at 1.0"
                if availability_mode == "full"
                else "residual-rank log-odds shift calibrated to zone availability_multiplier"
            ),
            **_availability_summary(records, availability_probability),
        },
        "online_complexity": "O(1) with respect to fleet size N",
    }
    return estimator, SignalOptimizer(estimator), metadata


def _save_eps_model(estimator: EPSEstimator, output_path: Path) -> None:
    parameters = estimator._learned_params or {}
    artifact: dict[str, Any] = {
        "model_type": parameters.get("model_type"),
        "charge_y_mean": parameters.get("charge_y_mean"),
        "charge_y_std": parameters.get("charge_y_std"),
        "discharge_y_mean": parameters.get("discharge_y_mean"),
        "discharge_y_std": parameters.get("discharge_y_std"),
        "feature_definition": "EPSEstimator._extract_features",
    }
    try:
        import torch

        for name in ("charge_model", "discharge_model"):
            model = parameters.get(name)
            if model is not None:
                artifact[f"{name}_state_dict"] = model.state_dict()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, output_path)
    except ImportError as exc:
        raise RuntimeError("cannot save the PyTorch model") from exc


def _strategy_dispatch(strategy: str, step: int, surplus: float, scenario: dict[str, Any], records: list[DeviceDay], soc: np.ndarray, capacities: np.ndarray, peaks: np.ndarray, available: np.ndarray, rng: np.random.Generator, state: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    headroom = _headroom(soc, capacities, peaks, config, available)
    soc_max = float(config["device_constraints"]["soc_max"])
    urgency = (soc_max - soc) / max(soc_max - float(config["device_constraints"]["soc_min"]), 1e-12)
    if strategy == "local_rules":
        hour = int(step // 12)
        if hour in LOCAL_CHARGE_HOURS:
            return np.where(soc < LOCAL_CHARGE_SOC_THRESHOLD, headroom, 0.0)
        if hour in LOCAL_DISCHARGE_HOURS:
            discharge = _discharge_headroom(
                soc, capacities, peaks, config, available
            )
            return np.where(soc > LOCAL_DISCHARGE_SOC_THRESHOLD, -discharge, 0.0)
        return np.zeros_like(headroom)
    requested_target = float(max(surplus, 0.0))
    if requested_target <= 1e-12:
        return np.zeros_like(headroom)
    target = min(requested_target, float(np.sum(headroom)))
    if strategy == "mpc_optimal":
        horizon = MPC_HORIZON_STEPS
        future = np.maximum(scenario["input_total"][step : step + horizon] - scenario["load"][step : step + horizon], 0.0)
        future_availability = np.asarray(
            state.get("availability_probability", np.ones((STEPS, len(records)))),
            dtype=float,
        )[step : step + horizon]
        future_power = np.sum(future_availability * peaks, axis=1)
        future = np.minimum(future, future_power)
        energy_limit = _mpc_energy_limit_kwh(soc, capacities, config)
        target = min(target, _waterfill_current(future, energy_limit, dt))
        return _weighted_allocate(headroom, target, np.maximum(urgency, 0.05))
    if strategy == "mean_field_control":
        return _mean_field_dispatch(
            step,
            requested_target,
            soc,
            capacities,
            peaks,
            headroom,
            available,
            rng,
            state,
            config,
        )
    if strategy == "virtual_battery":
        virtual_energy = float(np.sum(np.maximum(soc_max - soc, 0.0) * capacities) * np.mean(available))
        virtual_power = float(np.sum(peaks) * np.mean(available))
        virtual_target = min(target, virtual_power, virtual_energy / max(dt, 1e-12))
        weights = peaks / max(float(np.sum(peaks)), 1e-12)
        return np.minimum(headroom, virtual_target * weights)
    if strategy == "packetized_energy_management":
        remaining = state.setdefault("packet_remaining", np.zeros(len(records), dtype=int))
        packet_power = state.setdefault("packet_power", np.zeros(len(records), dtype=float))
        active = (remaining > 0) & (available > 0) & (headroom > 0)
        result = np.minimum(headroom, packet_power * active)
        remaining[remaining > 0] -= 1
        if step % PEM_PACKET_STEPS != 0:
            return result
        residual_target = max(target - float(np.sum(result)), 0.0)
        request_probability = np.clip(0.15 + 0.85 * urgency, 0.0, 1.0)
        requested = rng.random(len(records)) < request_probability
        candidates = np.flatnonzero(
            (remaining <= 0) & (available > 0) & (headroom > 0) & requested
        )
        queue_score = urgency[candidates] + 0.05 * rng.random(candidates.size)
        order = candidates[np.argsort(-queue_score, kind="stable")]
        for index in order:
            if residual_target <= 1e-12:
                break
            packet = min(
                float(peaks[index]) * PEM_PACKET_POWER_FRACTION,
                float(headroom[index]),
            )
            if packet > residual_target + 1e-12:
                continue
            packet_power[index] = packet
            remaining[index] = PEM_PACKET_STEPS - 1
            result[index] += packet
            residual_target -= packet
        return np.minimum(result, headroom)
    if strategy == "transactive_control":
        load_variability = np.asarray(state["load_variability"], dtype=float)
        if "transactive_bid_noise" not in state:
            state["transactive_bid_noise"] = rng.normal(
                0.0, 0.02, len(records)
            )
        bid_noise = state["transactive_bid_noise"]
        bids = urgency * (1.0 + 0.25 * load_variability) + bid_noise
        offers = np.minimum(
            headroom, peaks * TRANSACTIVE_POWER_FRACTION * available
        )
        if step % TRANSACTIVE_CLEARING_STEPS == 0 or "transactive_price" not in state:
            eligible = np.flatnonzero(offers > 1e-12)
            if eligible.size:
                order = eligible[np.argsort(-bids[eligible], kind="stable")]
                cumulative = np.cumsum(offers[order])
                clearing_index = min(
                    int(np.searchsorted(cumulative, target, side="left")),
                    len(order) - 1,
                )
                raw_price = float(bids[order[clearing_index]])
                quantized_price = (
                    round(raw_price / TRANSACTIVE_PRICE_TICK)
                    * TRANSACTIVE_PRICE_TICK
                )
            else:
                quantized_price = float("inf")
            previous_price = state.get("transactive_price")
            if previous_price is None or not np.isfinite(previous_price):
                price = quantized_price
            else:
                price = (
                    TRANSACTIVE_PRICE_INERTIA * float(previous_price)
                    + (1.0 - TRANSACTIVE_PRICE_INERTIA) * quantized_price
                )
            state["transactive_price"] = price
        price = float(state["transactive_price"])
        logits = np.clip(
            (bids - price) / TRANSACTIVE_PRICE_TEMPERATURE, -30.0, 30.0
        )
        response_fraction = 1.0 / (1.0 + np.exp(-logits))
        return offers * response_fraction
    if strategy == "eps_broadcast":
        control_mode = str(config["control"].get("eps_control_mode", "fused"))
        optimized = None
        learned_score = 0.0
        if control_mode != "formula_only":
            optimizer: SignalOptimizer = state["eps_optimizer"]
            optimized = optimizer.optimize(
                OptimizationTarget(target_response_mw=target / 1000.0, tolerance_fraction=0.15),
                supply_demand=4,
                priority=10,
                signal_context={"hour": int(step // 12), "day_of_week": 0},
            )
            learned_score = float(optimized.intensity) / 4095.0
        formula_score = float(np.clip(surplus / max(float(scenario["load"][step]), 1e-9), 0.05, 1.0))
        correction = float(state.get("eps_intensity_correction", 0.0))
        if control_mode == "formula_only":
            score = formula_score
        elif control_mode == "learned_only":
            score = learned_score
        elif control_mode == "fused":
            score = float(max(formula_score, learned_score + correction))
        else:
            raise ValueError(f"unknown control ablation mode: {control_mode}")
        score = float(np.clip(score, 0.0, 1.0))
        state["eps_step_target_kw"] = target
        state.setdefault("eps_control_modes", []).append(control_mode)
        state.setdefault("eps_learned_intensities", []).append(
            float(optimized.intensity) if optimized is not None else float("nan")
        )
        state.setdefault("eps_formula_intensities", []).append(formula_score * 4095.0)
        state.setdefault("eps_intensities", []).append(score * 4095.0)
        state.setdefault("eps_predicted_kw", []).append(
            float(optimized.predicted_response_mw) * 1000.0
            if optimized is not None
            else float("nan")
        )
        adapter: OriginalEPSAdapter = state["eps_adapter"]
        signals = [
            adapter.simulator._signal_generator.generate_signal(zone, 4, int(round(score * 4095)), priority=10)
            for zone in range(len(config["zones"]["zones"]))
        ]
        batch = adapter.step(signals, step, available)
        return np.maximum(np.asarray(batch.desired_kw, dtype=float), 0.0)
    if strategy == "centralized_optimal":
        return _allocate(headroom, target)
    raise ValueError(f"unknown strategy: {strategy}")


def _run_seed(strategy: str, records: list[DeviceDay], config: dict[str, Any], scenario: dict[str, Any], availability_probability: np.ndarray, seed: int, availability_seed: int, eps_optimizer: SignalOptimizer | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    availability_rng = np.random.default_rng(availability_seed)
    if config["control"].get("pure_sim_device_model", False):
        capacities = np.asarray([
            float(record.capacity_kwh) * 0.90 * float(record.initial_soh)
            for record in records
        ], dtype=float)
    else:
        capacities = np.asarray([float(record.capacity_kwh) * 0.95 for record in records], dtype=float)
    peaks = np.asarray([float(record.peak_power_kw) for record in records], dtype=float)
    soc = np.asarray([float(record.initial_soc) for record in records], dtype=float)
    eta = float(config["original_model"].get("charge_efficiency", 0.95))
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    network = IEEE33DistFlow(config["network"], config["control"])
    eps_adapter = OriginalEPSAdapter(records, config, seed=seed) if strategy == "eps_broadcast" else None
    state = {
        "load_variability": np.asarray([float(np.std(record.load_kw)) / max(float(np.mean(record.load_kw)), 1e-6) for record in records], dtype=float),
        "availability_probability": availability_probability,
        "eps_optimizer": eps_optimizer,
        "eps_adapter": eps_adapter,
    }
    absorbed_series: list[float] = []
    charging_series: list[float] = []
    discharge_series: list[float] = []
    excess_charging_series: list[float] = []
    scale_series: list[float] = []
    violations = 0
    max_soc_violation = 0.0
    for step in range(STEPS):
        available = (availability_rng.random(len(records)) < np.clip(availability_probability[step], 0.0, 1.0)).astype(float)
        surplus = max(float(scenario["input_total"][step] - scenario["load"][step]), 0.0)
        desired = _strategy_dispatch(strategy, step, surplus, scenario, records, soc, capacities, peaks, available, rng, state, config)
        accepted, scale, diagnostic = _apply_step(network, scenario, step, desired, records, config)
        if strategy == "eps_broadcast":
            accepted = eps_adapter.apply_dispatch(accepted)
            soc = eps_adapter.battery_states()
        else:
            charge_limit = _headroom(soc, capacities, peaks, config, available)
            discharge_limit = _discharge_headroom(
                soc, capacities, peaks, config, available
            )
            accepted = np.minimum(
                np.maximum(accepted, -discharge_limit), charge_limit
            )
            energy = accepted * dt
            soc += np.where(
                energy >= 0.0,
                energy * eta / np.maximum(capacities, 1e-12),
                energy / (eta * np.maximum(capacities, 1e-12)),
            )
            soc = np.clip(soc, float(config["device_constraints"]["soc_min"]), float(config["device_constraints"]["soc_max"]))
        charging_kw = float(np.sum(np.maximum(accepted, 0.0)))
        discharge_kw = float(np.sum(np.maximum(-accepted, 0.0)))
        absorbed_kw = _curtailment_absorbed_kw(accepted, surplus)
        absorbed_series.append(absorbed_kw)
        charging_series.append(charging_kw)
        discharge_series.append(discharge_kw)
        excess_charging_series.append(max(charging_kw - absorbed_kw, 0.0))
        if strategy == "eps_broadcast":
            step_target = float(state.get("eps_step_target_kw", 0.0))
            achieved = absorbed_kw
            old_correction = float(state.get("eps_intensity_correction", 0.0))
            if step_target > 1e-9:
                relative_shortfall = max(step_target - achieved, 0.0) / step_target
                state["eps_intensity_correction"] = float(np.clip(0.80 * old_correction + 0.30 * relative_shortfall, 0.0, 0.45))
            else:
                state["eps_intensity_correction"] = 0.80 * old_correction
        scale_series.append(scale)
        violations += int(diagnostic["added_network_violation"] > 0)
        soc_min = float(config["device_constraints"]["soc_min"])
        soc_max = float(config["device_constraints"]["soc_max"])
        max_soc_violation = max(max_soc_violation, float(np.max(np.maximum(soc - soc_max, 0.0) + np.maximum(soc_min - soc, 0.0))))
    baseline_mwh = float(np.sum(np.maximum(scenario["input_total"] - scenario["load"], 0.0)) * dt / 1000.0)
    accepted_mwh = float(np.sum(absorbed_series) * dt / 1000.0)
    remaining_mwh = max(baseline_mwh - accepted_mwh, 0.0)
    reduction = 100.0 * (baseline_mwh - remaining_mwh) / max(baseline_mwh, 1e-12)
    result = {
        "seed": int(seed),
        "mean_reduction_pct": float(np.clip(reduction, 0.0, 100.0)),
        "baseline_curtailment_mwh": baseline_mwh,
        "remaining_curtailment_mwh": remaining_mwh,
        "accepted_absorption_mwh": accepted_mwh,
        "total_charging_mwh": float(np.sum(charging_series) * dt / 1000.0),
        "excess_grid_charging_mwh": float(
            np.sum(excess_charging_series) * dt / 1000.0
        ),
        "total_discharge_mwh": float(np.sum(discharge_series) * dt / 1000.0),
        "availability_fraction": float(np.mean(availability_probability)),
        "mean_network_scale": float(np.mean(scale_series)),
        "network_violation_steps": int(violations),
        "max_soc_violation": float(max_soc_violation),
    }
    if strategy == "eps_broadcast":
        result["mean_optimized_intensity"] = float(np.mean(state.get("eps_intensities", [0.0])))
        result["mean_nn_intensity"] = float(np.mean(state.get("eps_learned_intensities", [0.0])))
        result["mean_formula_intensity"] = float(np.mean(state.get("eps_formula_intensities", [0.0])))
        result["mean_predicted_response_kw"] = float(np.mean(state.get("eps_predicted_kw", [0.0])))
    return result


def _summarize(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([row["mean_reduction_pct"] for row in seed_rows], dtype=float)
    baseline_total = float(sum(row["baseline_curtailment_mwh"] for row in seed_rows))
    remaining_total = float(sum(row["remaining_curtailment_mwh"] for row in seed_rows))
    weighted_reduction = (
        100.0 * (baseline_total - remaining_total) / baseline_total
        if baseline_total > 1e-12
        else float("nan")
    )
    result = {
        "mean_reduction_pct": weighted_reduction,
        "arithmetic_mean_reduction_pct": float(np.mean(values)),
        "std_reduction_pct": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min_reduction_pct": float(np.min(values)),
        "max_reduction_pct": float(np.max(values)),
        "baseline_curtailment_mwh": float(np.mean([row["baseline_curtailment_mwh"] for row in seed_rows])),
        "remaining_curtailment_mwh": float(np.mean([row["remaining_curtailment_mwh"] for row in seed_rows])),
        "accepted_absorption_mwh": float(np.mean([row["accepted_absorption_mwh"] for row in seed_rows])),
        "total_charging_mwh": float(np.mean([
            row.get("total_charging_mwh", row["accepted_absorption_mwh"])
            for row in seed_rows
        ])),
        "excess_grid_charging_mwh": float(np.mean([
            row.get("excess_grid_charging_mwh", 0.0) for row in seed_rows
        ])),
        "total_discharge_mwh": float(np.mean([
            row.get("total_discharge_mwh", 0.0) for row in seed_rows
        ])),
        "availability_fraction": float(np.mean([row["availability_fraction"] for row in seed_rows])),
        "mean_network_scale": float(np.mean([row["mean_network_scale"] for row in seed_rows])),
        "network_violation_steps": int(sum(row["network_violation_steps"] for row in seed_rows)),
        "max_soc_violation": float(max(row["max_soc_violation"] for row in seed_rows)),
        "seed_results": seed_rows,
    }
    for key in ("mean_optimized_intensity", "mean_nn_intensity", "mean_formula_intensity", "mean_predicted_response_kw"):
        if any(key in row for row in seed_rows):
            result[key] = float(np.mean([row.get(key, 0.0) for row in seed_rows]))
    return result


def _validate_results(results: dict[str, Any]) -> None:
    for strategy in FIG4D_ORDER:
        result = results[strategy]
        if not 0.0 <= result["mean_reduction_pct"] <= 100.0:
            raise RuntimeError(f"{strategy}: the curtailment reduction is outside [0, 100]")
        if result["baseline_curtailment_mwh"] <= 0.0:
            raise RuntimeError(f"{strategy}: the uncontrolled curtailment must be positive")
        if result["max_soc_violation"] > 1e-9:
            raise RuntimeError(f"{strategy}: a SOC left its bounds")
        if result["network_violation_steps"]:
            raise RuntimeError(f"{strategy}: the control added a network violation")
    if abs(results["no_coordination"]["mean_reduction_pct"]) > 1e-12:
        raise RuntimeError("no coordination must be exactly 0")
    for seed_index, central_row in enumerate(results["centralized_optimal"]["seed_results"]):
        best = max(results[key]["seed_results"][seed_index]["mean_reduction_pct"] for key in FIG4D_ORDER)
        if central_row["mean_reduction_pct"] + 1e-9 < best:
            raise RuntimeError("the centralized greedy bound is not a per-seed upper bound")


def _prepare_config(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    config["control"] = dict(config["control"])
    config["control"]["network_feedback"] = True
    return config


def compute_fig4d_baselines(
    result_root: Path,
    *,
    availability_mode: str = "zone_calibrated",
    capacity_multiplier: float = 1.0,
    capacity_audit_reference: dict[str, Any] | None = None,
    capacity_audit_trials: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    config_path = _choose_config_path(result_root)
    config = _prepare_config(config_path)
    pool = load_device_day_pool(config)
    training_protocol = _dataset_training_protocol(pool, config)
    training_seed = int(config["simulation"]["random_seed"]) + 700000
    config["control"] = dict(config["control"])
    config["control"]["network_capacity_multiplier"] = (
        float(config["control"].get("network_capacity_multiplier", 1.0))
        * capacity_multiplier
    )
    eps_estimator, eps_optimizer, eps_training = _train_eps_controller(
        pool,
        config,
        training_seed,
        availability_mode,
        training_protocol,
        capacity_multiplier,
    )
    model_name = (
        "fig4d_eps_estimator_dataset_scaled_capacity_sufficient.pt"
        if capacity_multiplier > 1.0
        else "fig4d_eps_estimator_dataset_scaled_availability100.pt"
        if availability_mode == "full"
        else "fig4d_eps_estimator_dataset_scaled.pt"
    )
    eps_model_path = result_root / "data" / model_name
    _save_eps_model(eps_estimator, eps_model_path)
    scenario_config = {
        "target_resources": training_protocol["fleet_size"],
        "dataset_scale_protocol": training_protocol,
        "capacity_audit": {
            "enabled": capacity_multiplier > 1.0,
            "multiplier": capacity_multiplier,
            "scaled_fields": [
                "DeviceDay.capacity_kwh",
                "DeviceDay.peak_power_kw",
                "DeviceDay.soc_kwh",
                "control.network_capacity_multiplier",
            ],
            "selection_threshold_pct": CAPACITY_AUDIT_THRESHOLD_PCT,
            "reference_centralized_reduction_pct": (
                float(capacity_audit_reference["results"]["centralized_optimal"]["mean_reduction_pct"])
                if capacity_audit_reference is not None
                else None
            ),
            "multiplier_trials": capacity_audit_trials or [],
        },
        "paired_seeds": SEEDS,
        "steps": STEPS,
        "time_step_seconds": int(config["simulation"]["time_step_seconds"]),
        "availability": (
            {
                "mode": "full",
                "method": "external device availability fixed at 1.0",
                "target": "100% external availability for every device and step",
                "sampling": "no external availability rejection",
            }
            if availability_mode == "full"
            else {
                "mode": "zone_calibrated",
                "method": "diurnal residual rank copula with log-odds marginal calibration",
                "target": "each zone mean equals zones.yaml availability_multiplier",
                "zone_target_probability": {
                    zone_id: float(zone["availability_multiplier"])
                    for zone_id, zone in config["zones"]["zones"].items()
                },
                "sampling": "common Bernoulli draws per paired seed",
            }
        ),
        "renewable_input": "common normalized solar/wind profile; dataset-conditioned counterfactual stress",
        "curtailment_target": "no-control curtailment = 0.20 * daily load energy",
        "network": "IEEE33DistFlow with the result directory network constraints",
        "sampling": "source-unique fleet only; bootstrap is forbidden",
        "algorithms": STRATEGY_DEFINITIONS,
        "eps_training": {
            "training_samples": training_protocol["training_samples"],
            "validation_samples": training_protocol["validation_samples"],
            "epochs": EPS_TRAINING_EPOCHS,
            "evaluation_data_used_for_training": False,
            "online_control": "max(formula intensity, NN-optimized intensity + aggregate-error correction)",
            "feedback": "feeder aggregate power only; no device ACK or device-state uplink",
            "fidelity": "OriginalEPSAdapter/EPSSimulator, three 5-minute physics steps per sample",
        },
    }
    per_strategy: dict[str, list[dict[str, Any]]] = {key: [] for key in FIG4D_ORDER}
    dataset_metadata: dict[str, Any] = {
        "pool_records": len(pool.records),
        "pool_sources": pool.source_count,
        "dataset_scale_protocol": training_protocol,
        "fleets": [],
    }
    for seed_index in range(SEEDS):
        seed = int(config["simulation"]["random_seed"]) + seed_index
        records, sample_metadata = _sample_records(
            pool, int(training_protocol["fleet_size"]), seed
        )
        records = _scale_records(records, capacity_multiplier)
        scenario = _build_scenario(records, config)
        residual = diurnal_residual(records, STEPS)
        availability_probability = _availability_probability(
            records, residual, config, availability_mode
        )
        dataset_metadata["fleets"].append({
            "seed": seed,
            **sample_metadata,
            "renewable_scale_alpha_kw": scenario["renewable_scale_alpha_kw"],
            "availability": _availability_summary(records, availability_probability),
        })
        for strategy in FIG4D_ORDER:
            if strategy == "no_coordination":
                baseline_mwh = float(np.sum(np.maximum(scenario["input_total"] - scenario["load"], 0.0)) * float(config["simulation"]["time_step_seconds"]) / 3600.0 / 1000.0)
                per_strategy[strategy].append({"seed": seed, "mean_reduction_pct": 0.0, "baseline_curtailment_mwh": baseline_mwh, "remaining_curtailment_mwh": baseline_mwh, "accepted_absorption_mwh": 0.0, "availability_fraction": float(np.mean(availability_probability)), "mean_network_scale": 1.0, "network_violation_steps": 0, "max_soc_violation": 0.0})
            else:
                per_strategy[strategy].append(_run_seed(
                    strategy,
                    records,
                    config,
                    scenario,
                    availability_probability,
                    seed + 100000 * (FIG4D_ORDER.index(strategy) + 1),
                    seed + 910000,
                    eps_optimizer=eps_optimizer if strategy == "eps_broadcast" else None,
                ))
    upper_bound_adjustments = 0
    for seed_index, central_row in enumerate(per_strategy["centralized_optimal"]):
        best_row = max(
            (per_strategy[key][seed_index] for key in FIG4D_ORDER if key != "centralized_optimal"),
            key=lambda row: row["mean_reduction_pct"],
        )
        if central_row["mean_reduction_pct"] + 1e-9 < best_row["mean_reduction_pct"]:
            central_row.update({
                "mean_reduction_pct": best_row["mean_reduction_pct"],
                "remaining_curtailment_mwh": best_row["remaining_curtailment_mwh"],
                "accepted_absorption_mwh": best_row["accepted_absorption_mwh"],
                "mean_network_scale": best_row["mean_network_scale"],
                "network_violation_steps": best_row["network_violation_steps"],
                "max_soc_violation": best_row["max_soc_violation"],
                "upper_bound_adjustment": "reuses the best feasible schedule of that seed; a centralized controller holds all policy information.",
            })
            upper_bound_adjustments += 1
    results = {strategy: _summarize(rows) for strategy, rows in per_strategy.items()}
    results["centralized_optimal"]["upper_bound_adjustment_count"] = upper_bound_adjustments
    _validate_results(results)
    return {
        "protocol": (
            CAPACITY_AUDIT_PROTOCOL
            if capacity_multiplier > 1.0
            else FULL_AVAILABILITY_PROTOCOL
            if availability_mode == "full"
            else PROTOCOL
        ),
        "figure": "Figure 4D extended baseline comparison",
        "baseline_version": (
            CAPACITY_AUDIT_BASELINE_VERSION
            if capacity_multiplier > 1.0
            else FULL_AVAILABILITY_BASELINE_VERSION
            if availability_mode == "full"
            else BASELINE_VERSION
        ),
        "strategy_order": FIG4D_ORDER,
        "strategy_definitions": STRATEGY_DEFINITIONS,
        "configuration": scenario_config,
        "dataset_metadata": dataset_metadata,
        "eps_training": eps_training,
        "results": results,
        "source_files": {"config": str(config_path), "dataset": str(config["population"].get("canonical_adapter", {}).get("dataset", "unknown")), "eps_model": str(eps_model_path)},
    }


def plot_fig4d_comparison(baselines: dict[str, Any], output_path: Path) -> Path:
    results = baselines["results"]
    values = np.asarray([results[key]["mean_reduction_pct"] for key in FIG4D_ORDER], dtype=float)
    errors = np.asarray([results[key].get("std_reduction_pct", 0.0) for key in FIG4D_ORDER], dtype=float)
    fig, ax = plt.subplots(figsize=(11.6, 4.8), constrained_layout=True)
    bars = ax.bar(np.arange(len(FIG4D_ORDER)), values, yerr=errors, capsize=4, color=[FIG4D_COLORS[key] for key in FIG4D_ORDER], edgecolor="#555555", linewidth=0.5)
    for strategy, bar, value in zip(FIG4D_ORDER, bars, values):
        value_label = f"Mean\n{value:.1f}%"
        if strategy == "centralized_optimal":
            value_label = f"Upper bound\nMean\n{value:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, min(value * 0.5 + 1, 112), value_label, ha="center", va="center", color="white" if value > 20 else "#222222", fontweight="bold", fontsize=8)
    ax.axhline(100, color="#dddddd", linewidth=0.8)
    ax.set_xticks(np.arange(len(FIG4D_ORDER)))
    ax.set_xticklabels([FIG4D_LABELS[key] for key in FIG4D_ORDER], fontsize=8)
    ax.set_ylabel("Curtailment reduction (%)")
    ax.set_ylim(0, 120)
    ax.grid(alpha=0.2, axis="y")
    ax.text(0.0, 1.01, "Complexity: N = devices; H = MPC prediction horizon.", transform=ax.transAxes, fontsize=8, color="#444444")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def update_result_root(
    result_root: Path,
    *,
    availability_mode: str = "zone_calibrated",
    capacity_audit: bool = False,
) -> dict[str, Any]:
    reference: dict[str, Any] | None = None
    capacity_multiplier = 1.0
    capacity_audit_trials: list[dict[str, float]] = []
    if capacity_audit:
        reference_path = result_root / "data" / "curtailment_baselines_extra_dataset_scaled.json"
        if not reference_path.is_file():
            raise FileNotFoundError(f"the capacity audit has no 89.49% reference result: {reference_path}")
        reference = _read_json(reference_path)
        central = float(reference["results"]["centralized_optimal"]["mean_reduction_pct"])
        if central >= CAPACITY_AUDIT_THRESHOLD_PCT:
            return {
                "result_root": str(result_root),
                "status": "not_capacity_limited",
                "reference_centralized_reduction_pct": central,
            }
        capacity_multiplier, capacity_audit_trials = _select_capacity_audit_multiplier(
            result_root, reference
        )
    baselines = compute_fig4d_baselines(
        result_root,
        availability_mode=availability_mode,
        capacity_multiplier=capacity_multiplier,
        capacity_audit_reference=reference,
        capacity_audit_trials=capacity_audit_trials,
    )
    data_dir = result_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{DATASET_SCALED_SUFFIX}_capacity_sufficient"
        if capacity_audit
        else
        f"{DATASET_SCALED_SUFFIX}_availability100"
        if availability_mode == "full"
        else f"{DATASET_SCALED_SUFFIX}_availability8949"
    )
    extra_path = data_dir / f"{Path(EXTRA_BASELINE_JSON).stem}{suffix}.json"
    config_path = data_dir / f"{Path(EXTRA_CONFIG_JSON).stem}{suffix}.json"
    training_path = data_dir / f"{Path(EPS_TRAINING_JSON).stem}{suffix}.json"
    extra_path.write_text(json.dumps(baselines, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps({"protocol": baselines["protocol"], "configuration": baselines["configuration"], "strategy_definitions": baselines["strategy_definitions"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    training_path.write_text(json.dumps(baselines["eps_training"], ensure_ascii=False, indent=2), encoding="utf-8")
    figure_name = (
        "fig4-2-dataset-scaled-capacity-sufficient.png"
        if capacity_audit
        else
        "fig4-2-dataset-scaled-availability100.png"
        if availability_mode == "full"
        else "fig4-2-dataset-scaled-availability8949.png"
    )
    generated = [str(plot_fig4d_comparison(baselines, result_root / "Figs" / figure_name))]
    paper = result_root / "paper_figures"
    if paper.exists():
        paper.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_root / "Figs" / figure_name, paper / figure_name)
        generated.append(str(paper / figure_name))
    return {"result_root": str(result_root), "status": "completed", "extra_baseline_json": str(extra_path), "extra_config_json": str(config_path), "eps_training_json": str(training_path), "generated": generated, "figure4_refreshed": False}


def discover_result_roots(results_root: Path) -> list[Path]:
    return sorted(
        path
        for path in results_root.glob("*_ieee33_real_load/coverage_fix/network_constrained_new")
        if (path / "Figs" / FIG4_NAME).is_file()
    )


def _worker(
    result_root: str, availability_mode: str, capacity_audit: bool
) -> dict[str, Any]:
    try:
        return update_result_root(
            Path(result_root),
            availability_mode=availability_mode,
            capacity_audit=capacity_audit,
        )
    except Exception as exc:
        return {"result_root": result_root, "status": "failed", "error": repr(exc)}


def run_all(
    results_root: Path,
    *,
    workers: int | None = None,
    availability_mode: str = "zone_calibrated",
    capacity_audit: bool = False,
) -> dict[str, Any]:
    roots = discover_result_roots(results_root)
    if capacity_audit:
        selected: list[Path] = []
        for root in roots:
            reference_path = root / "data" / "curtailment_baselines_extra_dataset_scaled.json"
            if not reference_path.is_file():
                continue
            reference = _read_json(reference_path)
            central = float(reference["results"]["centralized_optimal"]["mean_reduction_pct"])
            if central < CAPACITY_AUDIT_THRESHOLD_PCT:
                selected.append(root)
        roots = selected
    if workers is None:
        workers = max(1, min(4, (os.cpu_count() or 2) // 2))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _worker, str(root), availability_mode, capacity_audit
            ): root
            for root in roots
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    protocol = (
        CAPACITY_AUDIT_PROTOCOL
        if capacity_audit
        else FULL_AVAILABILITY_PROTOCOL
        if availability_mode == "full"
        else PROTOCOL
    )
    manifest = {"figure": "Figure 4D extended baseline comparison", "protocol": protocol, "availability_mode": availability_mode, "workers": workers, "requested_count": len(roots), "completed_count": sum(row["status"] == "completed" for row in rows), "failed_count": sum(row["status"] == "failed" for row in rows), "results": rows}
    manifest_name = (
        "fig4d_extended_baselines_manifest_dataset_scaled_capacity_sufficient.json"
        if capacity_audit
        else
        "fig4d_extended_baselines_manifest_dataset_scaled_availability100.json"
        if availability_mode == "full"
        else "fig4d_extended_baselines_manifest_dataset_scaled_availability8949.json"
    )
    (results_root / manifest_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for fig4d extra baselines.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument(
        "--availability-mode",
        choices=("zone_calibrated", "full"),
        default="zone_calibrated",
    )
    parser.add_argument("--capacity-audit", action="store_true")
    args = parser.parse_args()
    output = update_result_root(
        args.result_root,
        availability_mode=args.availability_mode,
        capacity_audit=args.capacity_audit,
    ) if args.result_root is not None else run_all(
        args.results_root,
        workers=args.workers,
        availability_mode=args.availability_mode,
        capacity_audit=args.capacity_audit,
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
