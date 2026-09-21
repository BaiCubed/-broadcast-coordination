from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.signal import SignalOptimizer

from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as protocol
from . import run_e20_transfer as transfer
from . import run_e21_mixed_scenarios as base


PROTOCOL = "E21_mixed_physical_fleet_source_unique_v1"
MAX_FLEET_SIZE = base.FLEET_SIZE
ALGORITHM_DEFINITIONS = copy.deepcopy(base.ALGORITHM_DEFINITIONS)
ALGORITHM_DEFINITIONS["eps_e21_mixed"]["label"] = "E21 source-unique mixed EPS"
ALGORITHM_DEFINITIONS["eps_e21_mixed"]["description"] = (
    "trained directly on the response of without-replacement, physically mixed fleets."
)


def _partition_capacities(
    weights: dict[str, float], network_mode: str, partition_name: str
) -> dict[str, int]:
    return {
        dataset: int(
            getattr(
                base._partitions(dataset, network_mode), partition_name
            ).source_count
        )
        for dataset in weights
    }


def max_feasible_allocation(
    weights: dict[str, float],
    capacities: dict[str, int],
    maximum: int = MAX_FLEET_SIZE,
) -> tuple[int, dict[str, int]]:
    normalized = base._normalized_weights(weights)
    if set(normalized) != set(capacities):
        raise ValueError("the capacity table and the mixture weights refer to different datasets")
    if any(capacities[dataset] < 1 for dataset in normalized):
        raise ValueError("every dataset of a source-unique mixture needs at least one real source")
    upper = min(int(maximum), sum(int(capacities[dataset]) for dataset in normalized))
    upper = min(
        upper,
        *(
            int(capacities[dataset] / weight) + len(normalized) + 2
            for dataset, weight in normalized.items()
        ),
    )
    for fleet_size in range(upper, len(normalized) - 1, -1):
        counts = base._allocate_counts(normalized, total=fleet_size)
        if all(counts[dataset] <= capacities[dataset] for dataset in counts):
            return fleet_size, counts
    raise ValueError("the available real sources cannot build this mixture")


def _sample_unique_devices(
    dataset: str,
    partition: Any,
    count: int,
    seed: int,
) -> tuple[list[Any], dict[str, Any]]:
    profiles, metadata = protocol._sample_profiles(
        partition, "source_unique", count, seed
    )
    physical_source_ids = [str(profile.source_device_id) for profile in profiles]
    if len(set(physical_source_ids)) != len(physical_source_ids):
        raise RuntimeError(f"{dataset}: the source-unique sample repeats a real source")
    parameter_seed = seed + 1000003 + protocol._stable_seed(dataset)
    synthetic = protocol._synthetic_devices(profiles, parameter_seed)
    devices = [
        replace(
            device,
            device_id=f"{dataset}__device_{index:05d}",
            source_device_id=f"{dataset}__{physical_source_ids[index]}",
        )
        for index, device in enumerate(synthetic)
    ]
    digest = hashlib.sha256(
        "\n".join(sorted(physical_source_ids)).encode("utf-8")
    ).hexdigest()
    metadata.update(
        {
            "device_parameter_dataset_id": dataset,
            "device_parameter_seed": parameter_seed,
            "device_parameter_protocol": (
                "dataset-id keyed synthetic capacity, C-rate, initial SOC, and SOH"
            ),
            "sampling_without_replacement": True,
            "duplicate_source_count": 0,
            "max_source_reuse": 1,
            "selected_source_ids_sha256": digest,
            "mean_capacity_kwh": float(
                np.mean([record.capacity_kwh for record in devices])
            ),
            "total_capacity_kwh": float(
                np.sum([record.capacity_kwh for record in devices])
            ),
            "mean_peak_power_kw": float(
                np.mean([record.peak_power_kw for record in devices])
            ),
            "total_peak_power_kw": float(
                np.sum([record.peak_power_kw for record in devices])
            ),
        }
    )
    return devices, metadata


def _sample_unique_fleet(
    scenario_id: str,
    network_mode: str,
    partition_name: str,
    seed: int,
    weights: dict[str, float] | None = None,
    maximum_fleet_size: int = MAX_FLEET_SIZE,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    scenario = base.SCENARIOS[scenario_id]
    selected_weights = base._normalized_weights(weights or scenario["weights"])
    capacities = _partition_capacities(selected_weights, network_mode, partition_name)
    fleet_size, counts = max_feasible_allocation(
        selected_weights,
        capacities,
        maximum=int(maximum_fleet_size),
    )
    anchor_dataset = next(iter(selected_weights))
    config = base._config(anchor_dataset, network_mode)
    records: list[Any] = []
    dataset_audit: dict[str, Any] = {}
    for dataset_index, (dataset, count) in enumerate(counts.items()):
        partition = getattr(base._partitions(dataset, network_mode), partition_name)
        sampled, metadata = _sample_unique_devices(
            dataset,
            partition,
            count,
            seed
            + 10007 * (dataset_index + 1)
            + protocol._stable_seed(dataset) % 100000,
        )
        start = len(records)
        for local_index, record in enumerate(sampled):
            spatial = base._spatial_record(
                record, dataset, local_index, scenario, config
            )
            records.append(
                replace(
                    spatial,
                    device_id=f"{dataset}__{start + local_index:05d}",
                )
            )
        dataset_audit[dataset] = {
            "requested_weight": selected_weights[dataset],
            "realized_weight": count / fleet_size,
            "physical_devices": count,
            "available_profile_sources": capacities[dataset],
            "selected_unique_profile_sources": metadata[
                "selected_unique_profile_sources"
            ],
            "profile_bootstrap": False,
            "mean_profile_reuse": 1.0,
            "sampling_without_replacement": True,
            "duplicate_source_count": metadata["duplicate_source_count"],
            "max_source_reuse": metadata["max_source_reuse"],
            "selected_source_ids_sha256": metadata["selected_source_ids_sha256"],
            "mean_capacity_kwh": metadata["mean_capacity_kwh"],
            "total_capacity_kwh": metadata["total_capacity_kwh"],
            "mean_peak_power_kw": metadata["mean_peak_power_kw"],
            "total_peak_power_kw": metadata["total_peak_power_kw"],
        }
    source_ids = [record.source_device_id for record in records]
    duplicate_count = len(source_ids) - len(set(source_ids))
    if duplicate_count:
        raise RuntimeError(f"{scenario_id}: the mixed fleet repeats a real source")
    return (
        records,
        config,
        {
            "scenario": scenario_id,
            "partition": partition_name,
            "fleet_size": len(records),
            "maximum_requested_fleet_size": int(maximum_fleet_size),
            "fleet_size_rule": (
                "largest integer N<=5000 whose dataset counts fit partition source capacities"
            ),
            "requested_weights": selected_weights,
            "realized_weights": {
                dataset: count / fleet_size for dataset, count in counts.items()
            },
            "counts": counts,
            "source_capacities": capacities,
            "datasets": dataset_audit,
            "sampling_without_replacement": True,
            "duplicate_source_count": 0,
            "max_source_reuse": 1,
            "spatial_clustering": "zones" in scenario,
        },
    )


def _response_samples(
    records: list[Any],
    config: dict[str, Any],
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    signals, fixed_scale_responses, metadata = base._response_samples(
        records, config, sample_count, seed
    )
    fleet_size = len(records)
    responses = (
        np.asarray(fixed_scale_responses, dtype=float)
        * base.RESPONSE_SCALE_KW
        / float(fleet_size)
    ).tolist()
    metadata.update(
        {
            "fleet_size": fleet_size,
            "training_target": "accepted_response_kw / actual source-unique fleet size N",
            "response_range_per_device_kw": [
                float(np.min(responses)),
                float(np.max(responses)),
            ],
        }
    )
    return signals, responses, metadata


def _train_model(network_mode: str, model_path: Path) -> dict[str, Any]:
    train_signals: list[dict[str, Any]] = []
    train_responses: list[float] = []
    validation_signals: list[dict[str, Any]] = []
    validation_responses: list[float] = []
    scenario_metadata: list[dict[str, Any]] = []
    for scenario_index, scenario_id in enumerate(base.SCENARIOS):
        train_seed = (
            5100000
            + scenario_index * 10000
            + (500000 if network_mode == "ieee33" else 0)
        )
        records, config, fleet_audit = _sample_unique_fleet(
            scenario_id, network_mode, "train", train_seed
        )
        signals, responses, response_audit = _response_samples(
            records, config, base.TRAIN_SAMPLES_PER_SCENARIO, train_seed + 1
        )
        train_signals.extend(signals)
        train_responses.extend(responses)
        validation_records, validation_config, validation_fleet_audit = (
            _sample_unique_fleet(
                scenario_id, network_mode, "validation", train_seed + 2000
            )
        )
        signals, responses, validation_response_audit = _response_samples(
            validation_records,
            validation_config,
            base.VALIDATION_SAMPLES_PER_SCENARIO,
            train_seed + 2001,
        )
        validation_signals.extend(signals)
        validation_responses.extend(responses)
        scenario_metadata.append(
            {
                "scenario": scenario_id,
                "train_fleet": fleet_audit,
                "train_response": response_audit,
                "validation_fleet": validation_fleet_audit,
                "validation_response": validation_response_audit,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "unique_training_samples",
                    "network_mode": network_mode,
                    "scenario": scenario_id,
                    "fleet_size": len(records),
                    "completed_scenarios": scenario_index + 1,
                    "scenario_count": len(base.SCENARIOS),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    try:
        import torch

        torch.manual_seed(5100000 + (500000 if network_mode == "ieee33" else 0))
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = transfer._make_estimator()
    estimator.fit(train_signals, train_responses)
    legacy._save_eps_model(estimator, model_path)
    actual = np.asarray(validation_responses, dtype=float)
    predicted = np.asarray(
        [estimator.estimate(signal).response_kw for signal in validation_signals],
        dtype=float,
    )
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    parameters = estimator._learned_params or {}
    fleet_sizes = [int(row["train_fleet"]["fleet_size"]) for row in scenario_metadata]
    return {
        "network_mode": network_mode,
        "model": str(model_path),
        "model_type": str(parameters.get("model_type", "unknown")),
        "training_samples": len(train_signals),
        "validation_samples": len(validation_signals),
        "scenario_count": len(base.SCENARIOS),
        "samples_per_scenario": base.TRAIN_SAMPLES_PER_SCENARIO,
        "validation_samples_per_scenario": base.VALIDATION_SAMPLES_PER_SCENARIO,
        "training_target": "accepted_response_kw / actual source-unique fleet size N",
        "training_fleet_size_range": [min(fleet_sizes), max(fleet_sizes)],
        "internal_training_r2": float(parameters.get("r2", float("nan"))),
        "validation_r2": float(1.0 - np.sum(residual**2) / max(denominator, 1e-12)),
        "validation_rmse_per_device_kw": float(np.sqrt(np.mean(residual**2))),
        "validation_mae_per_device_kw": float(np.mean(np.abs(residual))),
        "scenario_metadata": scenario_metadata,
    }


def _evaluate_condition(task: tuple[str, str, str, str]) -> dict[str, Any]:
    logging.getLogger("src.simulation.simulator").setLevel(logging.WARNING)
    scenario_id, network_mode, e20_model_path, e21_model_path = task
    e20_estimator, _, _ = protocol.load_frozen_eps_controller(Path(e20_model_path))
    e21_estimator, _, _ = protocol.load_frozen_eps_controller(Path(e21_model_path))
    per_algorithm: dict[str, list[dict[str, Any]]] = {
        algorithm: [] for algorithm in base.ALGORITHM_ORDER
    }
    fleet_audit: list[dict[str, Any]] = []
    for seed_index in range(base.SEED_COUNT):
        weights, composition_regime = base._shift_weights(
            base.SCENARIOS[scenario_id]["weights"], seed_index
        )
        base_seed = (
            6100000
            + protocol._stable_seed(scenario_id) % 100000
            + (500000 if network_mode == "ieee33" else 0)
            + seed_index
        )
        records, config, audit = _sample_unique_fleet(
            scenario_id, network_mode, "test", base_seed, weights
        )
        fleet_size = len(records)
        scenario = protocol.build_pure_sim_scenario(records, config)
        availability = protocol.availability_probability(
            records, config, base.AVAILABILITY_MODE
        )
        optimizers = {
            "eps_e20_pooled": SignalOptimizer(
                transfer.ScaledEstimator(e20_estimator, float(fleet_size), 0.0)
            ),
            "eps_e21_mixed": SignalOptimizer(
                transfer.ScaledEstimator(e21_estimator, float(fleet_size), 0.0)
            ),
        }
        audit["seed"] = base_seed
        audit["composition_regime"] = composition_regime
        fleet_audit.append(audit)
        for algorithm_index, algorithm in enumerate(base.ALGORITHM_ORDER):
            if algorithm == "no_coordination":
                result = base._no_coordination(
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
            result["fleet_size"] = fleet_size
            per_algorithm[algorithm].append(result)
    fleet_sizes = [int(row["fleet_size"]) for row in fleet_audit]
    return {
        "scenario": scenario_id,
        "scenario_label": base.SCENARIOS[scenario_id]["label"],
        "network_mode": network_mode,
        "seed_count": base.SEED_COUNT,
        "nominal_weights": base._normalized_weights(
            base.SCENARIOS[scenario_id]["weights"]
        ),
        "spatial_clustering": "zones" in base.SCENARIOS[scenario_id],
        "fleet_mode": "source_unique_without_replacement",
        "fleet_size_summary": {
            "mean": float(np.mean(fleet_sizes)),
            "std": float(np.std(fleet_sizes, ddof=1)),
            "minimum": min(fleet_sizes),
            "maximum": max(fleet_sizes),
            "nominal": fleet_sizes[0],
            "dominant_plus_15pp": fleet_sizes[10],
            "dominant_minus_15pp": fleet_sizes[20],
        },
        "fleet_audit": fleet_audit,
        "results": {
            algorithm: legacy._summarize(rows)
            for algorithm, rows in per_algorithm.items()
        },
    }


def _write_summary_csv(conditions: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "scenario_label",
                "network_mode",
                "algorithm",
                "algorithm_label",
                "complexity",
                "seed_count",
                "mean_fleet_size",
                "min_fleet_size",
                "max_fleet_size",
                "mean_reduction_pct",
                "std_reduction_pct",
                "network_violation_steps",
                "max_soc_violation",
            ]
        )
        for condition in conditions:
            fleet = condition["fleet_size_summary"]
            for algorithm in base.ALGORITHM_ORDER:
                result = condition["results"][algorithm]
                definition = ALGORITHM_DEFINITIONS[algorithm]
                writer.writerow(
                    [
                        condition["scenario"],
                        condition["scenario_label"],
                        condition["network_mode"],
                        algorithm,
                        definition["label"],
                        definition["complexity"],
                        condition["seed_count"],
                        fleet["mean"],
                        fleet["minimum"],
                        fleet["maximum"],
                        result["mean_reduction_pct"],
                        result["std_reduction_pct"],
                        result["network_violation_steps"],
                        result["max_soc_violation"],
                    ]
                )


def _plot_fleet_sizes(conditions: list[dict[str, Any]], output_dir: Path) -> str:
    import matplotlib.pyplot as plt

    aggregate = [
        condition
        for condition in conditions
        if condition["network_mode"] == "aggregate"
    ]
    aggregate.sort(key=lambda row: list(base.SCENARIOS).index(row["scenario"]))
    x = np.arange(len(aggregate), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    for offset, key, label, color in (
        (-1.0, "nominal", "Nominal composition", "#2166ac"),
        (0.0, "dominant_plus_15pp", "Dominant +15 pp", "#d95f02"),
        (1.0, "dominant_minus_15pp", "Dominant -15 pp", "#1b9e77"),
    ):
        values = [row["fleet_size_summary"][key] for row in aggregate]
        axis.bar(x + offset * width, values, width, label=label, color=color)
    axis.axhline(MAX_FLEET_SIZE, color="#222222", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, [row["scenario"] for row in aggregate], rotation=30)
    axis.set_ylabel("Source-unique physical fleet size $N$")
    axis.set_title(
        "E21 source-unique real-data support by mixed scenario\n"
        "Each physical source appears at most once in a simultaneous fleet"
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3)
    path = output_dir / "e21_unique_fleet_sizes.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def main() -> None:
    base._validate_scenarios()
    parser = argparse.ArgumentParser(description="Command line entry point for run e21 unique scenarios.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(2, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-models", action="store_true")
    args = parser.parse_args()
    root = args.results_root / "E21" / "unique"
    model_dir = root / "models"
    data_dir = root / "data"
    figure_dir = root / "figures"
    for directory in (root, model_dir, data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    lock = base._acquire_lock(root)
    if lock is None:
        print(f"{root} already holds a running task; this instance exits")
        return
    result_path = data_dir / "e21_unique_results.json"
    manifest_path = root / "e21_unique_manifest.json"
    checkpoint_path = data_dir / "e21_unique_evaluation_checkpoint.json"
    training_path = data_dir / "e21_unique_training.json"
    if result_path.is_file() and manifest_path.is_file() and not args.force:
        print(result_path)
        return

    if training_path.is_file() and not args.retrain_models:
        training_metadata = json.loads(training_path.read_text(encoding="utf-8"))
    else:
        training_metadata = {"protocol": PROTOCOL, "models": {}}
    for network_mode in base.NETWORK_MODES:
        model_path = model_dir / f"e21_unique_mixed_{network_mode}.pt"
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
        base._write_json(training_path, training_metadata)

    tasks = [
        (
            scenario_id,
            network_mode,
            str(args.results_root / "E20" / "models" / f"pooled_{network_mode}.pt"),
            str(model_dir / f"e21_unique_mixed_{network_mode}.pt"),
        )
        for network_mode in base.NETWORK_MODES
        for scenario_id in base.SCENARIOS
    ]
    conditions: list[dict[str, Any]] = []
    if checkpoint_path.is_file() and not args.force:
        conditions = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed = {
        (condition["scenario"], condition["network_mode"]) for condition in conditions
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
            base._write_json(checkpoint_path, conditions)
            print(
                json.dumps(
                    {
                        "stage": "unique_evaluation",
                        "scenario": condition["scenario"],
                        "network_mode": condition["network_mode"],
                        "fleet_size_mean": condition["fleet_size_summary"]["mean"],
                        "completed_conditions": len(conditions),
                        "condition_count": len(base.SCENARIOS)
                        * len(base.NETWORK_MODES),
                        "eps_e21_mean": condition["results"]["eps_e21_mixed"][
                            "mean_reduction_pct"
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if failures:
        base._write_json(
            manifest_path,
            {
                "protocol": PROTOCOL,
                "status": "failed",
                "completed_count": len(conditions),
                "failed_count": len(failures),
                "failures": failures,
                "checkpoint": str(checkpoint_path),
            },
        )
        raise SystemExit(1)

    conditions.sort(
        key=lambda condition: (
            base.NETWORK_MODES.index(condition["network_mode"]),
            list(base.SCENARIOS).index(condition["scenario"]),
        )
    )
    summary_path = data_dir / "e21_unique_summary.csv"
    _write_summary_csv(conditions, summary_path)
    original_label = base.ALGORITHM_DEFINITIONS["eps_e21_mixed"]["label"]
    base.ALGORITHM_DEFINITIONS["eps_e21_mixed"]["label"] = ALGORITHM_DEFINITIONS[
        "eps_e21_mixed"
    ]["label"]
    try:
        figures = base._plot_heatmaps(
            conditions,
            figure_dir,
            experiment_label="E21 source-unique mixed-scenario",
            filename_prefix="e21_unique",
        )
        figures.append(
            base._plot_eps_comparison(
                conditions,
                figure_dir,
                experiment_label="E21 source-unique physical fleet mixing",
                filename_prefix="e21_unique",
            )
        )
        figures.append(
            base._plot_overall(
                conditions,
                figure_dir,
                experiment_label="E21 source-unique",
                filename_prefix="e21_unique",
            )
        )
    finally:
        base.ALGORITHM_DEFINITIONS["eps_e21_mixed"]["label"] = original_label
    figures.append(_plot_fleet_sizes(conditions, figure_dir))
    payload = {
        "protocol": PROTOCOL,
        "fleet_mode": "source_unique_without_replacement",
        "maximum_fleet_size": MAX_FLEET_SIZE,
        "fleet_size_rule": (
            "largest integer N<=5000 supported by every component dataset in the "
            "active train/validation/test partition"
        ),
        "source_reuse_scope": (
            "a physical source appears at most once within each simultaneous fleet; "
            "independent seeds may redraw a source"
        ),
        "availability_mode": base.AVAILABILITY_MODE,
        "network_modes": list(base.NETWORK_MODES),
        "seed_count": base.SEED_COUNT,
        "training_samples_per_scenario": base.TRAIN_SAMPLES_PER_SCENARIO,
        "validation_samples_per_scenario": base.VALIDATION_SAMPLES_PER_SCENARIO,
        "scenario_definitions": base.SCENARIOS,
        "algorithm_order": list(base.ALGORITHM_ORDER),
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
    base._write_json(result_path, payload)
    base._write_json(
        manifest_path,
        {
            "protocol": PROTOCOL,
            "status": "completed",
            "scenario_count": len(base.SCENARIOS),
            "condition_count": len(conditions),
            "algorithm_count": len(base.ALGORITHM_ORDER),
            "seed_count": base.SEED_COUNT,
            "failed_count": 0,
            "result": str(result_path),
            "summary_csv": str(summary_path),
            "figures": figures,
            },
    )
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
