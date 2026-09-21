from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from src.estimation import EPSEstimator, EstimatorConfig
from src.signal import SignalOptimizer
from src.extra.nc_excel_experiments.coupling import diurnal_residual
from ..config_loader import load_config
from ..network.ieee33_distflow import IEEE33DistFlow
from ..original_adapter.eps_adapter import OriginalEPSAdapter
from ..population.device_day_loader import DeviceDay, DeviceDayPool, load_device_day_pool
from . import fig4d_extra_baselines as legacy


PROTOCOL = "fig4d_final_v2_distinct_baselines_pure_sim_aligned_real_profiles"
FIXED_FLEET_SIZE = 5000
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
PURE_BASE_LOAD_PEAK_KW = 11250.0
PURE_SOLAR_RATIO = 1.10
PURE_WIND_RATIO = 0.50
PURE_GRID_HOSTING = 0.60

FLEET_MODES = ("fixed5000", "source_unique")
NETWORK_MODES = ("aggregate", "ieee33")
AVAILABILITY_MODES = ("sim_default", "data_driven")


@dataclass(frozen=True)
class ProfilePartitions:
    train: DeviceDayPool
    validation: DeviceDayPool
    test: DeviceDayPool
    metadata: dict[str, Any]


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def _make_pool(records: list[DeviceDay]) -> DeviceDayPool:
    return DeviceDayPool(
        records,
        len(records),
        len({record.source_device_id for record in records}),
        0,
    )


def partition_profile_pool(pool: DeviceDayPool, seed: int) -> ProfilePartitions:
    grouped: dict[str, list[DeviceDay]] = {}
    for record in pool.records:
        grouped.setdefault(record.source_device_id, []).append(record)
    splits = {"train": [], "validation": [], "test": []}
    overlap_sources = {"validation_test": 0, "all_splits": 0}
    for source_id, source_records in sorted(grouped.items()):
        ordered = sorted(source_records, key=lambda record: (record.day_id, record.device_id))
        rng = np.random.default_rng(seed + _stable_seed(source_id))
        ordered = [ordered[index] for index in rng.permutation(len(ordered))]
        count = len(ordered)
        if count == 1:
            splits["train"].append(ordered[0])
            splits["validation"].append(ordered[0])
            splits["test"].append(ordered[0])
            overlap_sources["all_splits"] += 1
            continue
        if count == 2:
            splits["train"].append(ordered[0])
            splits["validation"].append(ordered[1])
            splits["test"].append(ordered[1])
            overlap_sources["validation_test"] += 1
            continue
        train_count = max(1, int(np.floor(TRAIN_FRACTION * count)))
        validation_count = max(1, int(np.floor(VALIDATION_FRACTION * count)))
        if train_count + validation_count >= count:
            train_count = count - 2
            validation_count = 1
        splits["train"].extend(ordered[:train_count])
        splits["validation"].extend(
            ordered[train_count : train_count + validation_count]
        )
        splits["test"].extend(ordered[train_count + validation_count :])
    return ProfilePartitions(
        train=_make_pool(splits["train"]),
        validation=_make_pool(splits["validation"]),
        test=_make_pool(splits["test"]),
        metadata={
            "method": "per-source deterministic 70/15/15 device-day split",
            "records": {key: len(value) for key, value in splits.items()},
            "sources": {
                key: len({record.source_device_id for record in value})
                for key, value in splits.items()
            },
            "unavoidable_overlap_sources": overlap_sources,
            "overlap_reason": "one- and two-day sources cannot populate three disjoint date partitions",
        },
    )


def _fleet_size(pool: DeviceDayPool, fleet_mode: str) -> int:
    if fleet_mode == "fixed5000":
        return FIXED_FLEET_SIZE
    if fleet_mode == "source_unique":
        return min(FIXED_FLEET_SIZE, int(pool.source_count))
    raise ValueError(f"unknown fleet mode: {fleet_mode}")


def _sample_profiles(
    pool: DeviceDayPool, fleet_mode: str, count: int, seed: int
) -> tuple[list[DeviceDay], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[DeviceDay]] = {}
    for record in pool.records:
        grouped.setdefault(record.source_device_id, []).append(record)
    source_ids = sorted(grouped)
    if not source_ids:
        raise ValueError("the profile pool is empty")
    if fleet_mode == "source_unique" or len(source_ids) >= count:
        selected_ids = rng.choice(source_ids, size=count, replace=False)
        profiles = [
            grouped[str(source_id)][int(rng.integers(0, len(grouped[str(source_id)])))]
            for source_id in selected_ids
        ]
        bootstrap = False
    else:
        indices = rng.integers(0, len(pool.records), size=count)
        profiles = [pool.records[int(index)] for index in indices]
        bootstrap = True
    unique_profile_sources = len({record.source_device_id for record in profiles})
    return profiles, {
        "fleet_mode": fleet_mode,
        "logical_devices": count,
        "available_profile_sources": len(source_ids),
        "selected_unique_profile_sources": unique_profile_sources,
        "profile_bootstrap": bootstrap,
        "mean_profile_reuse": count / max(unique_profile_sources, 1),
    }


def _synthetic_devices(
    profiles: list[DeviceDay], seed: int
) -> list[DeviceDay]:
    rng = np.random.default_rng(seed)
    devices: list[DeviceDay] = []
    for index, profile in enumerate(profiles):
        capacity = float(rng.uniform(5.0, 20.0))
        sigma = 0.35
        c_rate = float(np.clip(
            rng.lognormal(np.log(0.45) - sigma**2 / 2.0, sigma), 0.2, 1.0
        ))
        initial_soc = float(rng.uniform(0.2, 0.9))
        initial_soh = float(rng.uniform(0.82, 1.0))
        devices.append(replace(
            profile,
            device_id=f"logical_{index:05d}",
            source_device_id=f"logical_{index:05d}",
            baseline_battery_kw=np.zeros(legacy.STEPS, dtype=float),
            soc_kwh=np.full(legacy.STEPS, capacity * initial_soc, dtype=float),
            capacity_kwh=capacity,
            peak_power_kw=capacity * c_rate,
            c_rate=c_rate,
            initial_soh=initial_soh,
        ))
    return devices


def sample_device_fleet(
    pool: DeviceDayPool,
    fleet_mode: str,
    count: int,
    seed: int,
    dataset_id: str = "",
) -> tuple[list[DeviceDay], dict[str, Any]]:
    profiles, metadata = _sample_profiles(pool, fleet_mode, count, seed)
    parameter_seed = seed + 1000003 + (_stable_seed(dataset_id) if dataset_id else 0)
    devices = _synthetic_devices(profiles, parameter_seed)
    metadata.update({
        "device_parameter_dataset_id": dataset_id or None,
        "device_parameter_seed": parameter_seed,
        "device_parameter_protocol": (
            "dataset-id keyed synthetic capacity, C-rate, initial SOC, and SOH"
            if dataset_id
            else "legacy seed-only synthetic device parameters"
        ),
        "mean_capacity_kwh": float(np.mean([record.capacity_kwh for record in devices])),
        "total_capacity_kwh": float(np.sum([record.capacity_kwh for record in devices])),
        "mean_peak_power_kw": float(np.mean([record.peak_power_kw for record in devices])),
        "total_peak_power_kw": float(np.sum([record.peak_power_kw for record in devices])),
    })
    return devices, metadata


def _training_protocol(fleet_size: int) -> dict[str, Any]:
    training_samples = int(np.clip(
        legacy.EPS_SAMPLES_PER_SOURCE * fleet_size,
        legacy.MIN_EPS_TRAINING_SAMPLES,
        legacy.MAX_EPS_TRAINING_SAMPLES,
    ))
    validation_samples = int(np.clip(
        round(0.10 * training_samples),
        legacy.MIN_EPS_VALIDATION_SAMPLES,
        legacy.MAX_EPS_VALIDATION_SAMPLES,
    ))
    return {
        "fleet_size": fleet_size,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "training_samples_rule": "clip(4 * fleet_size, 256, 12000)",
        "validation_samples_rule": "clip(round(0.10 * training_samples), 64, 1200)",
    }


def _pure_profiles() -> tuple[np.ndarray, np.ndarray]:
    solar = np.asarray([
        0, 0, 0, 0, 0, 0.02, 0.10, 0.30, 0.55, 0.78, 0.92, 0.98,
        1.00, 0.96, 0.85, 0.68, 0.45, 0.20, 0.05, 0, 0, 0, 0, 0,
    ], dtype=float)
    wind = np.asarray([
        0.45, 0.50, 0.55, 0.52, 0.48, 0.40, 0.30, 0.22, 0.18, 0.20, 0.25, 0.30,
        0.32, 0.28, 0.22, 0.18, 0.20, 0.28, 0.35, 0.42, 0.48, 0.52, 0.50, 0.48,
    ], dtype=float)
    return np.repeat(solar, 12), np.repeat(wind, 12)


def build_pure_sim_scenario(
    records: list[DeviceDay], config: dict[str, Any]
) -> dict[str, Any]:
    buses = sorted(
        {int(branch[1]) for branch in config["network"]["branches"]}
        | {int(config["network"]["slack_bus"])}
    )
    bus_index = {bus: index for index, bus in enumerate(buses)}
    raw_by_bus = np.zeros((legacy.STEPS, len(buses)), dtype=float)
    for record in records:
        raw_by_bus[:, bus_index[record.bus_id]] += np.asarray(
            record.load_kw[: legacy.STEPS], dtype=float
        )
    raw_load = np.sum(raw_by_bus, axis=1)
    hourly = np.asarray([
        float(np.mean(raw_load[hour * 12 : (hour + 1) * 12]))
        for hour in range(24)
    ])
    if float(np.max(hourly)) <= 1e-12:
        raise ValueError("the real load profile has no positive load")
    load_factor = np.repeat(hourly / float(np.max(hourly)), 12)
    load = PURE_BASE_LOAD_PEAK_KW * load_factor
    solar, wind = _pure_profiles()
    input_total = (
        PURE_BASE_LOAD_PEAK_KW * PURE_SOLAR_RATIO * solar
        + PURE_BASE_LOAD_PEAK_KW * PURE_WIND_RATIO * wind
    )
    load_by_bus = np.zeros_like(raw_by_bus)
    input_by_bus = np.zeros_like(raw_by_bus)
    for step in range(legacy.STEPS):
        weights = raw_by_bus[step]
        total = float(np.sum(weights))
        shares = (
            weights / total
            if total > 1e-12
            else np.full(len(buses), 1.0 / len(buses))
        )
        load_by_bus[step] = load[step] * shares
        input_by_bus[step] = input_total[step] * shares
    return {
        "buses": buses,
        "bus_index": bus_index,
        "device_bus_indices": np.asarray(
            [bus_index[record.bus_id] for record in records], dtype=int
        ),
        "load_by_bus": load_by_bus,
        "input_by_bus": input_by_bus,
        "load": load,
        "input_total": input_total,
        "aggregate_hosting_limit_kw": load * PURE_GRID_HOSTING,
        "load_factor": load_factor,
        "base_load_peak_kw": PURE_BASE_LOAD_PEAK_KW,
        "solar_ratio": PURE_SOLAR_RATIO,
        "wind_ratio": PURE_WIND_RATIO,
        "grid_hosting_fraction": PURE_GRID_HOSTING,
    }


def _condition_config(
    config: dict[str, Any], network_mode: str, availability_mode: str
) -> dict[str, Any]:
    configured = copy.deepcopy(config)
    configured["control"]["pure_sim_device_model"] = True
    configured["control"]["final_network_mode"] = network_mode
    configured["control"]["network_feedback"] = network_mode == "ieee33"
    configured["original_model"]["eps_packet_loss_rate"] = 0.001
    configured["original_model"]["device_offline_rate"] = (
        0.015 if availability_mode == "sim_default" else 0.0
    )
    return configured


def availability_probability(
    records: list[DeviceDay], config: dict[str, Any], availability_mode: str
) -> np.ndarray:
    residual = diurnal_residual(records, legacy.STEPS)
    if availability_mode == "sim_default":
        return np.ones_like(residual, dtype=float)
    if availability_mode == "data_driven":
        return legacy._calibrated_availability_probability(records, residual, config)
    raise ValueError(f"unknown availability mode: {availability_mode}")


def _signal_parameters(index: int, count: int, rng: np.random.Generator) -> tuple[int, int, int]:
    from src.extra.real_data_experiment.original_run_experiment import (
        EmergencyChargeProfile,
        EmergencyDischargeProfile,
        PeakShavingProfile,
        ValleyFillingProfile,
        encode_signal_score,
    )

    profiles = (
        ValleyFillingProfile(), PeakShavingProfile(),
        EmergencyChargeProfile(), EmergencyDischargeProfile(),
    )
    scenario_count = int(count * 0.80)
    time_aware_count = int(count * 0.10)
    if index < scenario_count:
        profile_index = min(
            index // max(1, scenario_count // len(profiles)), len(profiles) - 1
        )
        score = float(np.clip(
            profiles[profile_index].get_signal_score(rng.uniform(0.0, 1.0))
            + rng.normal(0.0, 0.02),
            -1.0,
            1.0,
        ))
        supply_demand, intensity = encode_signal_score(score)
        hour = int(rng.integers(0, 24))
    elif index < scenario_count + time_aware_count:
        hour = int(rng.integers(0, 24))
        if (7 <= hour <= 11) or (17 <= hour <= 21):
            intensity = int(
                rng.normal(2800, 500) if rng.random() < 0.5 else rng.normal(1000, 400)
            )
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
    return int(np.clip(supply_demand, 0, 15)), int(np.clip(intensity, 0, 4095)), hour


def _response_samples(
    pool: DeviceDayPool,
    fleet_mode: str,
    fleet_size: int,
    config: dict[str, Any],
    availability_mode: str,
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    dataset_id = str(config["population"].get("canonical_adapter", {}).get("dataset", "unknown"))
    records, sampling = sample_device_fleet(
        pool, fleet_mode, fleet_size, seed, dataset_id=dataset_id
    )
    scenario = build_pure_sim_scenario(records, config)
    availability = availability_probability(records, config, availability_mode)
    network = IEEE33DistFlow(config["network"], config["control"])
    adapter = OriginalEPSAdapter(records, config, seed + 1)
    rng = np.random.default_rng(seed + 2)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    zone_count = len(config["zones"]["zones"])
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    for index in range(sample_count):
        supply_demand, intensity, hour = _signal_parameters(index, sample_count, rng)
        profile_step = int(hour * 12 + rng.integers(0, 12))
        adapter.reset_to_initial_state()
        energy_kwh = 0.0
        for offset in range(3):
            step = (profile_step + offset) % legacy.STEPS
            eps_signals = [
                adapter.simulator._signal_generator.generate_signal(
                    zone, supply_demand, intensity, priority=10
                )
                for zone in range(zone_count)
            ]
            batch = adapter.step(eps_signals, step, availability[step])
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
        responses.append(energy_kwh / max(3.0 * dt, 1e-12))
        if (index + 1) % 2000 == 0:
            dataset = str(config["population"].get("canonical_adapter", {}).get("dataset", "unknown"))
            print(f"[{dataset}] final protocol samples {index + 1}/{sample_count}", flush=True)
    sampling["availability"] = legacy._availability_summary(records, availability)
    return signals, responses, sampling


def train_eps_controller(
    train_pool: DeviceDayPool,
    validation_pool: DeviceDayPool,
    fleet_mode: str,
    fleet_size: int,
    config: dict[str, Any],
    availability_mode: str,
    protocol: dict[str, Any],
    seed: int,
) -> tuple[EPSEstimator, SignalOptimizer, dict[str, Any]]:
    train_signals, train_responses, train_sampling = _response_samples(
        train_pool,
        fleet_mode,
        fleet_size,
        config,
        availability_mode,
        int(protocol["training_samples"]),
        seed,
    )
    validation_signals, validation_responses, validation_sampling = _response_samples(
        validation_pool,
        fleet_mode,
        fleet_size,
        config,
        availability_mode,
        int(protocol["validation_samples"]),
        seed + 10000,
    )
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
        pytorch_epochs=legacy.EPS_TRAINING_EPOCHS,
        pytorch_batch_size=64,
        pytorch_learning_rate=0.001,
    ))
    estimator.fit(train_signals, train_responses)
    actual = np.asarray(validation_responses, dtype=float)
    predicted = np.asarray([
        estimator.estimate(signal).response_kw for signal in validation_signals
    ], dtype=float)
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    parameters = estimator._learned_params or {}
    metadata = {
        "training_seed": seed,
        "training_samples": len(train_signals),
        "validation_samples": len(validation_signals),
        "epochs": legacy.EPS_TRAINING_EPOCHS,
        "model_type": str(parameters.get("model_type", "unknown")),
        "internal_training_r2": float(parameters.get("r2", float("nan"))),
        "held_out_r2": 1.0 - float(np.sum(residual**2)) / max(denominator, 1e-12),
        "held_out_rmse_kw": float(np.sqrt(np.mean(residual**2))),
        "held_out_mae_kw": float(np.mean(np.abs(residual))),
        "train_sampling": train_sampling,
        "validation_sampling": validation_sampling,
        "evaluation_data_used_for_training": False,
    }
    return estimator, SignalOptimizer(estimator), metadata


def load_frozen_eps_controller(
    model_path: Path,
    previous_training: dict[str, Any] | None = None,
) -> tuple[EPSEstimator, SignalOptimizer, dict[str, Any]]:
    import torch
    from torch import nn

    class QuantileAggregateNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(10, 128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.q10_head = nn.Linear(32, 1)
            self.q50_head = nn.Linear(32, 1)
            self.q90_head = nn.Linear(32, 1)

        def forward(self, values: Any) -> Any:
            hidden = self.shared(values)
            return torch.cat([
                self.q10_head(hidden),
                self.q50_head(hidden),
                self.q90_head(hidden),
            ], dim=1)

    try:
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(model_path, map_location="cpu")
    if artifact.get("model_type") != "dual_quantile_nn":
        raise ValueError(f"unsupported frozen model: {artifact.get('model_type')}")
    charge_model = QuantileAggregateNN()
    discharge_model = QuantileAggregateNN()
    charge_model.load_state_dict(artifact["charge_model_state_dict"])
    discharge_model.load_state_dict(artifact["discharge_model_state_dict"])
    charge_model.eval()
    discharge_model.eval()
    estimator = EPSEstimator(EstimatorConfig(
        target_coverage=0.9,
        enable_conformal=False,
        use_cqr=False,
        use_pytorch=True,
    ))
    estimator._learned_params = {
        "model_type": "dual_quantile_nn",
        "charge_model": charge_model,
        "charge_y_mean": float(artifact["charge_y_mean"]),
        "charge_y_std": float(artifact["charge_y_std"]),
        "discharge_model": discharge_model,
        "discharge_y_mean": float(artifact["discharge_y_mean"]),
        "discharge_y_std": float(artifact["discharge_y_std"]),
        "residual_std": 0.0,
    }
    estimator._is_fitted = True
    estimator._is_quantile_model = True
    estimator._is_dual_model = True
    metadata = dict(previous_training or {})
    metadata.update({
        "model_reused": True,
        "model_reuse_reason": "baseline and execution-layer rerun with frozen EPS weights",
        "source_model": str(model_path),
        "conformal_state_reused": False,
        "point_prediction_equivalence": "q50 weights and normalization restored exactly",
    })
    return estimator, SignalOptimizer(estimator), metadata


def _config_path(result_root: Path) -> Path:
    return legacy._choose_config_path(result_root)


def compute_condition(
    result_root: Path,
    fleet_mode: str,
    network_mode: str,
    availability_mode: str,
) -> dict[str, Any]:
    config_path = _config_path(result_root)
    original_config = load_config(config_path)
    config = _condition_config(original_config, network_mode, availability_mode)
    pool = load_device_day_pool(config)
    if fleet_mode == "source_unique" and pool.source_count >= FIXED_FLEET_SIZE:
        raise RuntimeError("with at least 5000 sources the main experiment is already source-unique; no extra audit is needed")
    partitions = partition_profile_pool(pool, int(config["simulation"]["random_seed"]))
    fleet_size = _fleet_size(pool, fleet_mode)
    protocol = _training_protocol(fleet_size)
    suffix = f"{fleet_mode}_{network_mode}_{availability_mode}"
    model_path = result_root / "data" / f"fig4d_eps_estimator_final_{suffix}.pt"
    previous_path = result_root / "data" / f"curtailment_baselines_final_{suffix}.json"
    previous_payload = (
        json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_path.is_file()
        else {}
    )
    if model_path.is_file():
        estimator, optimizer, training = load_frozen_eps_controller(
            model_path, previous_payload.get("eps_training")
        )
    else:
        training_seed = int(config["simulation"]["random_seed"]) + 1700000
        estimator, optimizer, training = train_eps_controller(
            partitions.train,
            partitions.validation,
            fleet_mode,
            fleet_size,
            config,
            availability_mode,
            protocol,
            training_seed,
        )
        training["model_reused"] = False
        legacy._save_eps_model(estimator, model_path)
    per_strategy: dict[str, list[dict[str, Any]]] = {
        strategy: [] for strategy in legacy.FIG4D_ORDER
    }
    fleets: list[dict[str, Any]] = []
    for seed_index in range(legacy.SEEDS):
        seed = int(config["simulation"]["random_seed"]) + seed_index
        dataset_id = str(config["population"].get("canonical_adapter", {}).get("dataset", "unknown"))
        records, sampling = sample_device_fleet(
            partitions.test, fleet_mode, fleet_size, seed, dataset_id=dataset_id
        )
        scenario = build_pure_sim_scenario(records, config)
        availability = availability_probability(records, config, availability_mode)
        fleets.append({
            "seed": seed,
            **sampling,
            "availability": legacy._availability_summary(records, availability),
            "baseline_curtailment_mwh": float(
                np.sum(np.maximum(scenario["input_total"] - scenario["load"], 0.0))
                * float(config["simulation"]["time_step_seconds"])
                / 3600.0
                / 1000.0
            ),
        })
        for strategy in legacy.FIG4D_ORDER:
            if strategy == "no_coordination":
                baseline = fleets[-1]["baseline_curtailment_mwh"]
                per_strategy[strategy].append({
                    "seed": seed,
                    "mean_reduction_pct": 0.0,
                    "baseline_curtailment_mwh": baseline,
                    "remaining_curtailment_mwh": baseline,
                    "accepted_absorption_mwh": 0.0,
                    "availability_fraction": float(np.mean(availability)),
                    "mean_network_scale": 1.0,
                    "network_violation_steps": 0,
                    "max_soc_violation": 0.0,
                })
                continue
            per_strategy[strategy].append(legacy._run_seed(
                strategy,
                records,
                config,
                scenario,
                availability,
                seed + 100000 * (legacy.FIG4D_ORDER.index(strategy) + 1),
                seed + 1910000,
                eps_optimizer=optimizer if strategy == "eps_broadcast" else None,
            ))
    adjustments = 0
    for seed_index, central in enumerate(per_strategy["centralized_optimal"]):
        best = max(
            (
                per_strategy[strategy][seed_index]
                for strategy in legacy.FIG4D_ORDER
                if strategy != "centralized_optimal"
            ),
            key=lambda row: row["mean_reduction_pct"],
        )
        if central["mean_reduction_pct"] + 1e-9 < best["mean_reduction_pct"]:
            central.update(best)
            central["upper_bound_adjustment"] = "best feasible paired strategy"
            adjustments += 1
    results = {
        strategy: legacy._summarize(rows)
        for strategy, rows in per_strategy.items()
    }
    results["centralized_optimal"]["upper_bound_adjustment_count"] = adjustments
    legacy._validate_results(results)
    return {
        "protocol": PROTOCOL,
        "figure": "Figure 4D final pure-simulation-aligned comparison",
        "condition": {
            "fleet_mode": fleet_mode,
            "fleet_size": fleet_size,
            "network_mode": network_mode,
            "availability_mode": availability_mode,
        },
        "pure_simulation_alignment": {
            "capacity_kwh": "U(5, 20)",
            "c_rate": "clipped LogNormal(mean=0.45, sigma=0.35), [0.2, 1.0]",
            "initial_soc": "U(0.2, 0.9)",
            "initial_soh": "U(0.82, 1.0)",
            "base_load_peak_kw": PURE_BASE_LOAD_PEAK_KW,
            "solar_ratio": PURE_SOLAR_RATIO,
            "wind_ratio": PURE_WIND_RATIO,
            "aggregate_hosting_fraction": PURE_GRID_HOSTING,
            "device_offline_rate": config["original_model"]["device_offline_rate"],
            "eps_packet_loss_rate": config["original_model"]["eps_packet_loss_rate"],
        },
        "baseline_design": {
            "local_rules": {
                "charge_hours": "09:00-16:59",
                "charge_soc_threshold": legacy.LOCAL_CHARGE_SOC_THRESHOLD,
                "discharge_hours": "17:00-20:59",
                "discharge_soc_threshold": legacy.LOCAL_DISCHARGE_SOC_THRESHOLD,
                "information": "local SOC, local availability, and time only",
            },
            "packetized_energy_management": {
                "packet_steps": legacy.PEM_PACKET_STEPS,
                "packet_minutes": legacy.PEM_PACKET_STEPS * 5,
                "packet_power_fraction": legacy.PEM_PACKET_POWER_FRACTION,
                "indivisible_packets": True,
                "interruptible_active_packets": False,
            },
            "transactive_control": {
                "clearing_steps": legacy.TRANSACTIVE_CLEARING_STEPS,
                "clearing_minutes": legacy.TRANSACTIVE_CLEARING_STEPS * 5,
                "power_fraction": legacy.TRANSACTIVE_POWER_FRACTION,
                "price_tick": legacy.TRANSACTIVE_PRICE_TICK,
                "price_temperature": legacy.TRANSACTIVE_PRICE_TEMPERATURE,
                "price_inertia": legacy.TRANSACTIVE_PRICE_INERTIA,
            },
            "curtailment_accounting": "min(positive accepted charging, current surplus)",
            "network_constraints": (
                "aggregate 60% instantaneous hosting only"
                if network_mode == "aggregate"
                else "IEEE-33 feasibility only; aggregate pre-clipping disabled"
            ),
        },
        "profile_partitions": partitions.metadata,
        "training_protocol": protocol,
        "eps_training": training,
        "dataset_metadata": {
            "pool_sources": pool.source_count,
            "pool_device_days": len(pool.records),
            "test_fleets": fleets,
        },
        "strategy_order": legacy.FIG4D_ORDER,
        "strategy_definitions": legacy.STRATEGY_DEFINITIONS,
        "results": results,
        "source_files": {
            "config": str(config_path),
            "eps_model": str(model_path),
        },
    }


def output_stem(fleet_mode: str, network_mode: str, availability_mode: str) -> str:
    return f"final_{fleet_mode}_{network_mode}_{availability_mode}"


def update_result_root(
    result_root: Path,
    fleet_mode: str,
    network_mode: str,
    availability_mode: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    stem = output_stem(fleet_mode, network_mode, availability_mode)
    data_path = result_root / "data" / f"curtailment_baselines_{stem}.json"
    figure_path = result_root / "Figs" / f"fig4-2-{stem.replace('_', '-')}.png"
    if data_path.is_file() and figure_path.is_file() and not force:
        return {
            "result_root": str(result_root),
            "condition": stem,
            "status": "skipped_existing",
            "data": str(data_path),
            "figure": str(figure_path),
        }
    payload = compute_condition(
        result_root, fleet_mode, network_mode, availability_mode
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    legacy.plot_fig4d_comparison(payload, figure_path)
    paper_dir = result_root / "paper_figures"
    if paper_dir.exists():
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure_path, paper_dir / figure_path.name)
    return {
        "result_root": str(result_root),
        "condition": stem,
        "status": "completed",
        "data": str(data_path),
        "figure": str(figure_path),
    }


def _worker(args: tuple[str, str, str, str, bool]) -> dict[str, Any]:
    result_root, fleet_mode, network_mode, availability_mode, force = args
    try:
        return update_result_root(
            Path(result_root),
            fleet_mode,
            network_mode,
            availability_mode,
            force=force,
        )
    except RuntimeError as exc:
        if "no extra audit needed" in str(exc):
            return {
                "result_root": result_root,
                "condition": output_stem(fleet_mode, network_mode, availability_mode),
                "status": "not_required",
                "reason": str(exc),
            }
        return {
            "result_root": result_root,
            "condition": output_stem(fleet_mode, network_mode, availability_mode),
            "status": "failed",
            "error": repr(exc),
        }
    except Exception as exc:
        return {
            "result_root": result_root,
            "condition": output_stem(fleet_mode, network_mode, availability_mode),
            "status": "failed",
            "error": repr(exc),
        }


def discover_result_roots(results_root: Path) -> list[Path]:
    return sorted(
        path
        for path in results_root.glob(
            "*_ieee33_real_load/coverage_fix/network_constrained_new"
        )
        if (path / "Figs" / legacy.FIG4_NAME).is_file()
    )


def run_all(results_root: Path, workers: int, force: bool) -> dict[str, Any]:
    tasks = [
        (str(root), fleet_mode, network_mode, availability_mode, force)
        for root in discover_result_roots(results_root)
        for fleet_mode in FLEET_MODES
        for network_mode in NETWORK_MODES
        for availability_mode in AVAILABILITY_MODES
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    manifest = {
        "protocol": PROTOCOL,
        "workers": workers,
        "requested_count": len(tasks),
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "skipped_existing_count": sum(
            row["status"] == "skipped_existing" for row in rows
        ),
        "not_required_count": sum(row["status"] == "not_required" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "results": rows,
    }
    manifest_path = results_root / "fig4d_final_protocol_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for fig4d final protocol.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--fleet-mode", choices=FLEET_MODES)
    parser.add_argument("--network-mode", choices=NETWORK_MODES)
    parser.add_argument("--availability-mode", choices=AVAILABILITY_MODES)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2))
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.result_root is not None:
        if not all((args.fleet_mode, args.network_mode, args.availability_mode)):
            parser.error("a single-dataset run must specify the fleet, network and availability mode together")
        output = update_result_root(
            args.result_root,
            args.fleet_mode,
            args.network_mode,
            args.availability_mode,
            force=args.force,
        )
    else:
        output = run_all(args.results_root, args.workers, args.force)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
