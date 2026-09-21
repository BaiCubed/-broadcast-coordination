from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.estimation import EPSEstimator, EstimatorConfig
from src.signal import SignalOptimizer

from ..population.data2_transaction_loader import clear_data2_transaction_pool_cache
from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as training
from . import run_e21_mixed_scenarios as mixed
from . import run_e22_ieee69_complexity as e22


PROTOCOL = "E22_ieee69_direct_training_ablation_v2"
TRAINING_TOPOLOGY = "ieee69"
OUTPUT_ROOT = Path("results/E22/trained_eps_ieee69_direct")
NATIVE_ALGORITHM = "eps_ieee69_fused"
NATIVE_ALGORITHMS = (
    "eps_ieee69_formula_only",
    "eps_ieee69_learned_only",
    NATIVE_ALGORITHM,
)
NATIVE_DEFINITIONS = {
    "eps_ieee69_formula_only": {
        "label": "EPS formula-only",
        "complexity": "O(1)",
        "description": "The analytic intensity formula only, with no learned response model.",
        "control_mode": "formula_only",
    },
    "eps_ieee69_learned_only": {
        "label": "EPS learned-only",
        "complexity": "O(1)",
        "description": "The broadcast intensity from the directly trained IEEE-69 response model only.",
        "control_mode": "learned_only",
    },
    NATIVE_ALGORITHM: {
        "label": "EPS IEEE-69 fused",
        "complexity": "O(1)",
        "description": "The analytic intensity, the learned IEEE-69 intensity and the closed-loop shortfall correction combined.",
        "control_mode": "fused",
    },
}
NATIVE_DEFINITION = {
    "label": "EPS IEEE-69 fused",
    "complexity": "O(1)",
    "description": "Each dataset is trained directly on IEEE-69 M0-M6; testing keeps O(1) broadcast control.",
}
TRANSFER_ALGORITHMS = ("eps_e20_pooled", "eps_e21_mixed")
EPS_COMPARISON_ORDER = (*TRANSFER_ALGORITHMS, *NATIVE_ALGORITHMS)
EPS_COLORS = {
    "eps_e20_pooled": "#ff9f40",
    "eps_e21_mixed": "#069c8f",
    "eps_ieee69_formula_only": "#8c8c8c",
    "eps_ieee69_learned_only": "#6f4aa8",
    NATIVE_ALGORITHM: "#1f5aa6",
}

for algorithm, definition in NATIVE_DEFINITIONS.items():
    mixed.ALGORITHM_DEFINITIONS[algorithm] = definition
    mixed.ALGORITHM_COLORS[algorithm] = EPS_COLORS[algorithm]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_bidirectional_artifact(model_path: Path) -> list[str]:
    import torch

    try:
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(model_path, map_location="cpu")
    fallbacks: list[str] = []
    charge_key = "charge_model_state_dict"
    discharge_key = "discharge_model_state_dict"
    if charge_key not in artifact and discharge_key not in artifact:
        raise ValueError(f"the model has no usable charge or discharge branch: {model_path}")
    if charge_key not in artifact:
        artifact[charge_key] = artifact[discharge_key]
        fallbacks.append("charge_from_discharge")
    if discharge_key not in artifact:
        artifact[discharge_key] = artifact[charge_key]
        fallbacks.append("discharge_from_charge")
    if fallbacks:
        artifact["branch_fallbacks"] = fallbacks
        torch.save(artifact, model_path)
    return fallbacks


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _training_config(dataset: str) -> tuple[dict[str, Any], training.ProfilePartitions]:
    source_config, partitions = e22._dataset_source(dataset)
    config = copy.deepcopy(source_config)
    config["control"] = dict(config["control"])
    config["control"]["pure_sim_device_model"] = True
    config["control"]["network_feedback"] = False
    config["original_model"] = dict(config["original_model"])
    config["original_model"]["eps_packet_loss_rate"] = 0.001
    config["original_model"]["device_offline_rate"] = 0.0
    return config, partitions


def _ieee69_response_samples(
    pool: training.DeviceDayPool,
    fleet_size: int,
    config: dict[str, Any],
    sample_count: int,
    seed: int,
    cases: dict[str, dict[str, Any]],
    stress_modes: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    dataset = str(
        config["population"].get("canonical_adapter", {}).get("dataset", "unknown")
    )
    base_records, sampling = training.sample_device_fleet(
        pool,
        "fixed5000",
        fleet_size,
        seed,
        dataset_id=dataset,
    )
    case = cases[TRAINING_TOPOLOGY]
    placement_seed = seed + 31_000
    reference_records, _ = e22._remap_records(
        base_records, case, placement_seed, "load_weighted"
    )
    contexts: dict[str, dict[str, Any]] = {}
    for stress_index, stress_mode in enumerate(stress_modes):
        placement_mode = str(e22.STRESS_MODES[stress_mode]["placement_mode"])
        records, placement = e22._remap_records(
            base_records,
            case,
            placement_seed + 1000 * (stress_index + 1),
            placement_mode,
        )
        scenario = e22._build_scenario(
            records,
            reference_records,
            case,
            TRAINING_TOPOLOGY,
            stress_mode,
            seed + 2000 * (stress_index + 1),
            placement,
        )
        contexts[stress_mode] = {
            "records": records,
            "scenario": scenario,
            "network": e22._network(case, TRAINING_TOPOLOGY, stress_mode),
            "availability": training.availability_probability(
                records, config, mixed.AVAILABILITY_MODE
            ),
            "adapter": training.OriginalEPSAdapter(
                records, config, seed + 3000 * (stress_index + 1)
            ),
        }
    rng = np.random.default_rng(seed + 71_000)
    schedule = np.resize(np.asarray(stress_modes, dtype=object), sample_count)
    rng.shuffle(schedule)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    zone_count = len(config["zones"]["zones"])
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    stress_counts = {stress_mode: 0 for stress_mode in stress_modes}
    for index, stress_value in enumerate(schedule):
        stress_mode = str(stress_value)
        context = contexts[stress_mode]
        records = context["records"]
        scenario = context["scenario"]
        network = context["network"]
        availability = context["availability"]
        adapter = context["adapter"]
        supply_demand, intensity, hour = training._signal_parameters(
            index, sample_count, rng
        )
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
            load_map, input_map = e22._maps(scenario, step)
            dispatch = network.dispatch(
                load_map,
                input_map,
                np.asarray(batch.desired_kw, dtype=float),
                scenario["device_buses"],
            )
            actual = adapter.apply_dispatch(dispatch.accepted_kw)
            energy_kwh += float(np.sum(actual)) * dt
        signals.append(
            {
                "supply_demand": supply_demand,
                "intensity": intensity,
                "price": 0.0,
                "hour": hour,
                "day_of_week": 0,
                "direction": 1 if supply_demand <= 7 else -1,
            }
        )
        responses.append(energy_kwh / max(3.0 * dt, 1e-12))
        stress_counts[stress_mode] += 1
        if (index + 1) % 2000 == 0:
            print(
                f"[{dataset}] IEEE-69 training samples {index + 1}/{sample_count}",
                flush=True,
            )
    sampling.update(
        {
            "training_topology": TRAINING_TOPOLOGY,
            "network_constrained_labels": True,
            "stress_sample_counts": stress_counts,
            "availability": legacy._availability_summary(
                base_records,
                training.availability_probability(
                    base_records, config, mixed.AVAILABILITY_MODE
                ),
            ),
        }
    )
    return signals, responses, sampling


def _train_ieee69_controller(
    partitions: training.ProfilePartitions,
    config: dict[str, Any],
    protocol: dict[str, Any],
    seed: int,
    cases: dict[str, dict[str, Any]],
    stress_modes: tuple[str, ...],
) -> tuple[EPSEstimator, dict[str, Any]]:
    train_signals, train_responses, train_sampling = _ieee69_response_samples(
        partitions.train,
        e22.FLEET_SIZE,
        config,
        int(protocol["training_samples"]),
        seed,
        cases,
        stress_modes,
    )
    validation_signals, validation_responses, validation_sampling = (
        _ieee69_response_samples(
            partitions.validation,
            e22.FLEET_SIZE,
            config,
            int(protocol["validation_samples"]),
            seed + 10_000,
            cases,
            stress_modes,
        )
    )
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = EPSEstimator(
        EstimatorConfig(
            target_coverage=0.9,
            enable_conformal=True,
            use_cqr=True,
            use_pytorch=True,
            pytorch_epochs=legacy.EPS_TRAINING_EPOCHS,
            pytorch_batch_size=64,
            pytorch_learning_rate=0.001,
        )
    )
    estimator.fit(train_signals, train_responses)
    actual = np.asarray(validation_responses, dtype=float)
    predicted = np.asarray(
        [estimator.estimate(signal).response_kw for signal in validation_signals],
        dtype=float,
    )
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
        "held_out_r2": 1.0
        - float(np.sum(residual**2)) / max(denominator, 1e-12),
        "held_out_rmse_kw": float(np.sqrt(np.mean(residual**2))),
        "held_out_mae_kw": float(np.mean(np.abs(residual))),
        "train_sampling": train_sampling,
        "validation_sampling": validation_sampling,
        "training_topology": TRAINING_TOPOLOGY,
        "training_stress_modes": list(stress_modes),
        "training_target": "IEEE-69 network-delivered aggregate response kW",
        "evaluation_data_used_for_training": False,
    }
    return estimator, metadata


def _ensure_native_model(
    dataset: str,
    output: Path,
    cases: dict[str, dict[str, Any]],
    stress_modes: tuple[str, ...],
    retrain_all: bool,
    training_samples: int | None = None,
    validation_samples: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    protocol = training._training_protocol(e22.FLEET_SIZE)
    if training_samples is not None:
        protocol["training_samples"] = training_samples
        protocol["training_samples_rule"] = "CLI override"
    if validation_samples is not None:
        protocol["validation_samples"] = validation_samples
        protocol["validation_samples_rule"] = "CLI override"
    model_path = output / "models" / f"{dataset}.pt"
    metadata_path = output / "models" / f"{dataset}.json"
    if model_path.is_file() and metadata_path.is_file() and not retrain_all:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("protocol") == PROTOCOL
            and metadata.get("training_topology") == TRAINING_TOPOLOGY
            and tuple(metadata.get("training_stress_modes", [])) == stress_modes
            and int(metadata.get("training_samples", -1))
            == int(protocol["training_samples"])
            and int(metadata.get("validation_samples", -1))
            == int(protocol["validation_samples"])
            and metadata.get("model_sha256") == _sha256(model_path)
        ):
            return model_path, metadata

    config, partitions = _training_config(dataset)
    training_seed = e22.BASE_SEED + e22._stable_seed(dataset) % 100_000 + 700_000
    started = time.perf_counter()
    estimator, metadata = _train_ieee69_controller(
        partitions,
        config,
        protocol,
        training_seed,
        cases,
        stress_modes,
    )
    legacy._save_eps_model(estimator, model_path)
    branch_fallbacks = _ensure_bidirectional_artifact(model_path)
    metadata.update(
        {
            "protocol": PROTOCOL,
            "dataset": dataset,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "model_reused": False,
            "training_regime": "dataset-specific fixed5000 IEEE-69 M0-M6 direct training",
            "training_protocol": protocol,
            "training_seconds": time.perf_counter() - started,
            "branch_fallbacks": branch_fallbacks,
            "branch_fallback_scope": (
                "unused opposite-direction loader compatibility; E22 curtailment control uses charge branch"
                if branch_fallbacks
                else None
            ),
        }
    )
    e22._write_json(metadata_path, metadata)
    return model_path, metadata


def _run_dataset(
    dataset: str,
    output: Path,
    model_path: Path,
    model_sha256: str,
    cases: dict[str, dict[str, Any]],
    seed_count: int,
    topologies: tuple[str, ...],
    stress_modes: tuple[str, ...],
    force: bool,
) -> dict[str, Any]:
    raw_path = output / "raw" / f"{dataset}.json"
    payload: dict[str, Any] = {}
    if raw_path.is_file() and not force:
        candidate = json.loads(raw_path.read_text(encoding="utf-8"))
        if (
            candidate.get("protocol") == PROTOCOL
            and candidate.get("model_sha256") == model_sha256
            and int(candidate.get("seed_count", -1)) == seed_count
            and tuple(candidate.get("topologies", [])) == topologies
            and tuple(candidate.get("stress_modes", [])) == stress_modes
            and tuple(candidate.get("algorithms", [])) == NATIVE_ALGORITHMS
        ):
            payload = candidate
    rows = list(payload.get("seed_results", []))
    completed = {
        (
            int(row["seed_index"]),
            str(row["topology"]),
            str(row["stress_mode"]),
            str(row["algorithm"]),
        )
        for row in rows
    }
    estimator, optimizer, _ = training.load_frozen_eps_controller(model_path)
    del estimator
    for seed_index in range(seed_count):
        base_seed = e22._base_seed(dataset, seed_index)
        base_records, config, _ = e22._sample_dataset(dataset, base_seed)
        availability = training.availability_probability(
            base_records, config, mixed.AVAILABILITY_MODE
        )
        for topology_index, topology in enumerate(topologies):
            placement_seed = base_seed + 10_000 * (e22.TOPOLOGIES.index(topology) + 1)
            reference_records, _ = e22._remap_records(
                base_records, cases[topology], placement_seed, "load_weighted"
            )
            placement_cache: dict[str, tuple[list[Any], dict[str, Any]]] = {}
            for stress_mode in stress_modes:
                pending_algorithms = [
                    algorithm
                    for algorithm in NATIVE_ALGORITHMS
                    if (seed_index, topology, stress_mode, algorithm) not in completed
                ]
                if not pending_algorithms:
                    continue
                placement_mode = str(e22.STRESS_MODES[stress_mode]["placement_mode"])
                if placement_mode not in placement_cache:
                    placement_cache[placement_mode] = e22._remap_records(
                        base_records, cases[topology], placement_seed, placement_mode
                    )
                records, placement = placement_cache[placement_mode]
                scenario = e22._build_scenario(
                    records,
                    reference_records,
                    cases[topology],
                    topology,
                    stress_mode,
                    base_seed,
                    placement,
                )
                for algorithm in pending_algorithms:
                    algorithm_config = copy.deepcopy(config)
                    algorithm_config["control"] = dict(algorithm_config["control"])
                    algorithm_config["control"]["eps_control_mode"] = (
                        NATIVE_DEFINITIONS[algorithm]["control_mode"]
                    )
                    network = e22._network(cases[topology], topology, stress_mode)
                    started = time.perf_counter()
                    result, _ = e22._run_seed(
                        algorithm,
                        records,
                        algorithm_config,
                        scenario,
                        availability,
                        network,
                        base_seed + 1_800_000,
                        base_seed + 1_910_000,
                        optimizer,
                        False,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": e22.DATASET_LABELS[dataset],
                            "topology": topology,
                            "stress_mode": stress_mode,
                            "stress_label": e22.STRESS_MODES[stress_mode]["label"],
                            "algorithm": algorithm,
                            "algorithm_label": NATIVE_DEFINITIONS[algorithm]["label"],
                            "complexity": NATIVE_DEFINITIONS[algorithm]["complexity"],
                            "control_ablation": NATIVE_DEFINITIONS[algorithm][
                                "control_mode"
                            ],
                            "training_regime": (
                                "dataset-specific IEEE-69 M0-M6 direct training"
                            ),
                            "seed_index": seed_index,
                            "seed": base_seed,
                            "fleet_size": len(records),
                            "placement_mode": placement["placement_mode"],
                            "target_branch": placement["target_branch"],
                            "target_bus": placement["target_bus"],
                            "target_downstream_load_share": placement[
                                "target_downstream_load_share"
                            ],
                            "configured_concentration_fraction": placement[
                                "configured_concentration_fraction"
                            ],
                            "actual_target_feeder_fraction": placement[
                                "actual_target_feeder_fraction"
                            ],
                            "actual_target_bus_fraction": placement[
                                "actual_target_bus_fraction"
                            ],
                            "maximum_bus_fraction": placement["maximum_bus_fraction"],
                            "runtime_seconds": time.perf_counter() - started,
                            **result,
                        }
                    )
                    completed.add((seed_index, topology, stress_mode, algorithm))
                    payload = {
                        "protocol": PROTOCOL,
                        "dataset": dataset,
                        "dataset_label": e22.DATASET_LABELS[dataset],
                        "seed_count": seed_count,
                        "fleet_size": e22.FLEET_SIZE,
                        "model_path": str(model_path),
                        "model_sha256": model_sha256,
                        "topologies": list(topologies),
                        "stress_modes": list(stress_modes),
                        "algorithms": list(NATIVE_ALGORITHMS),
                        "seed_results": sorted(
                            rows,
                            key=lambda row: (
                                int(row["seed_index"]),
                                e22.TOPOLOGIES.index(row["topology"]),
                                list(e22.STRESS_MODES).index(row["stress_mode"]),
                                NATIVE_ALGORITHMS.index(row["algorithm"]),
                            ),
                        ),
                        "updated_at": e22._utc_now(),
                    }
                    e22._write_json(raw_path, payload)
        print(
            json.dumps(
                {
                    "stage": "native_seed_checkpoint",
                    "dataset": dataset,
                    "completed_seeds": seed_index + 1,
                    "seed_count": seed_count,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return payload


def _run_dataset_job(
    dataset: str,
    output: str,
    seed_count: int,
    topologies: tuple[str, ...],
    stress_modes: tuple[str, ...],
    force: bool,
    retrain_all: bool,
    training_samples: int | None,
    validation_samples: int | None,
) -> dict[str, Any]:
    output_path = Path(output)
    cases = e22._network_cases()
    model_path, metadata = _ensure_native_model(
        dataset,
        output_path,
        cases,
        stress_modes,
        retrain_all,
        training_samples,
        validation_samples,
    )
    _run_dataset(
        dataset,
        output_path,
        model_path,
        str(metadata["model_sha256"]),
        cases,
        seed_count,
        topologies,
        stress_modes,
        force,
    )
    mixed._PARTITION_CACHE.clear()
    e22._ADDITIONAL_PARTITION_CACHE.clear()
    clear_data2_transaction_pool_cache()
    gc.collect()
    return {"dataset": dataset, "metadata": metadata}


def _mean_over_stress(
    rows: list[dict[str, Any]], dataset: str, topology: str, algorithm: str
) -> float:
    selected = [
        float(row["curtailment_reduction_pct"])
        for row in rows
        if row["dataset"] == dataset
        and row["topology"] == topology
        and row["algorithm"] == algorithm
    ]
    return float(np.mean(selected))


def _algorithm_label(algorithm: str) -> str:
    return mixed.ALGORITHM_DEFINITIONS[algorithm]["label"]


def _save_figure(fig: Any, figure_dir: Path, name: str) -> list[str]:
    return e22._save_figure(fig, figure_dir, name)


def _plot_topology_dumbbell(
    summaries: list[dict[str, Any]], figure_dir: Path
) -> list[str]:
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(EPS_COMPARISON_ORDER))
    for index, algorithm in enumerate(EPS_COMPARISON_ORDER):
        values = []
        for topology in e22.TOPOLOGIES:
            selected = [
                float(row["curtailment_reduction_pct"])
                for row in summaries
                if row["algorithm"] == algorithm and row["topology"] == topology
            ]
            values.append(float(np.mean(selected)))
        axis.plot(values, [index, index], color="#a0a0a0", linewidth=2)
        axis.scatter(values[0], index, color="#4e79a7", s=75, zorder=3)
        axis.scatter(values[1], index, color="#e15759", s=75, zorder=3)
        axis.text(values[0], index - 0.16, f"{values[0]:.2f}%", ha="center", fontsize=8)
        axis.text(values[1], index + 0.22, f"{values[1]:.2f}%", ha="center", fontsize=8)
    axis.set_yticks(y, [f"{_algorithm_label(a)}  O(1)" for a in EPS_COMPARISON_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Deliverable curtailment reduction (%)")
    axis.set_title("IEEE-69 direct-trained EPS ablations versus transfer models")
    axis.grid(axis="x", alpha=0.25)
    axis.scatter([], [], color="#4e79a7", label="IEEE-33")
    axis.scatter([], [], color="#e15759", label="IEEE-69")
    axis.legend(loc="lower right")
    return _save_figure(fig, figure_dir, "trained_vs_transfer_topology_dumbbell")


def _plot_dataset_heatmap(
    summaries: list[dict[str, Any]], figure_dir: Path
) -> list[str]:
    matrix = np.zeros((len(e22.E22_DATASETS), len(EPS_COMPARISON_ORDER)))
    for row_index, dataset in enumerate(e22.E22_DATASETS):
        for column_index, algorithm in enumerate(EPS_COMPARISON_ORDER):
            matrix[row_index, column_index] = _mean_over_stress(
                summaries, dataset, "ieee69", algorithm
            )
    fig, axis = plt.subplots(figsize=(9.2, 8.6))
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=100.0)
    axis.set_xticks(
        np.arange(len(EPS_COMPARISON_ORDER)),
        [_algorithm_label(algorithm) for algorithm in EPS_COMPARISON_ORDER],
        rotation=20,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(e22.E22_DATASETS)),
        [e22.DATASET_LABELS[dataset] for dataset in e22.E22_DATASETS],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if matrix[row_index, column_index] > 58 else "black",
            )
    axis.set_title("IEEE-69 mean deliverable effect across M0-M6")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Curtailment reduction (%)")
    return _save_figure(fig, figure_dir, "trained_vs_transfer_dataset_heatmap_ieee69")


def _plot_stress_profiles(
    overall: list[dict[str, Any]], figure_dir: Path
) -> list[str]:
    fig, axis = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(e22.STRESS_MODES))
    for algorithm in EPS_COMPARISON_ORDER:
        values = [
            next(
                float(row["curtailment_reduction_pct"])
                for row in overall
                if row["topology"] == "ieee69"
                and row["stress_mode"] == stress_mode
                and row["algorithm"] == algorithm
            )
            for stress_mode in e22.STRESS_MODES
        ]
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            color=EPS_COLORS[algorithm],
            label=f"{_algorithm_label(algorithm)} (O(1))",
        )
    axis.set_xticks(x, list(e22.STRESS_MODES))
    axis.set_ylim(0, 105)
    axis.set_ylabel("Deliverable curtailment reduction (%)")
    axis.set_title("IEEE-69 direct training and control ablations under M0-M6")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    return _save_figure(fig, figure_dir, "trained_vs_transfer_stress_profiles_ieee69")


def _plot_native_gain(
    summaries: list[dict[str, Any]], figure_dir: Path
) -> list[str]:
    labels = [e22.DATASET_LABELS[dataset] for dataset in e22.E22_DATASETS]
    native = np.asarray(
        [
            _mean_over_stress(summaries, dataset, "ieee69", NATIVE_ALGORITHM)
            for dataset in e22.E22_DATASETS
        ]
    )
    transfer = np.asarray(
        [
            _mean_over_stress(summaries, dataset, "ieee69", "eps_e21_mixed")
            for dataset in e22.E22_DATASETS
        ]
    )
    gain = native - transfer
    order = np.argsort(gain)
    fig, axis = plt.subplots(figsize=(10.2, 7.4))
    colors = np.where(gain[order] >= 0.0, "#2a9d8f", "#d95f5f")
    axis.barh(np.arange(len(order)), gain[order], color=colors)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(np.arange(len(order)), [labels[index] for index in order])
    for row_index, value in enumerate(gain[order]):
        axis.text(
            value + (0.25 if value >= 0 else -0.25),
            row_index,
            f"{value:+.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
        )
    axis.set_xlabel("IEEE-69 fused EPS gain over E21 mixed transfer (percentage points)")
    axis.set_title("IEEE-69 direct-training gain by dataset, averaged across M0-M6")
    axis.grid(axis="x", alpha=0.22)
    return _save_figure(fig, figure_dir, "trained_gain_by_dataset_ieee69")


def _plot_all_algorithms(
    overall: list[dict[str, Any]], figure_dir: Path
) -> list[str]:
    order = list(mixed.ALGORITHM_ORDER)
    insertion = order.index("centralized_optimal")
    for offset, algorithm in enumerate(NATIVE_ALGORITHMS):
        order.insert(insertion + offset, algorithm)
    values = []
    for algorithm in order:
        selected = [
            float(row["curtailment_reduction_pct"])
            for row in overall
            if row["topology"] == "ieee69" and row["algorithm"] == algorithm
        ]
        values.append(float(np.mean(selected)))
    fig, axis = plt.subplots(figsize=(11.2, 6.4))
    y = np.arange(len(order))
    colors = [
        EPS_COLORS.get(algorithm, mixed.ALGORITHM_COLORS.get(algorithm, "#777777"))
        for algorithm in order
    ]
    axis.barh(y, values, color=colors)
    axis.set_yticks(
        y,
        [
            f"{_algorithm_label(algorithm)}  {mixed.ALGORITHM_DEFINITIONS[algorithm]['complexity']}"
            for algorithm in order
        ],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 110)
    for index, value in enumerate(values):
        axis.text(value + 0.6, index, f"{value:.2f}%", va="center", fontsize=8)
    axis.set_xlabel("Deliverable curtailment reduction (%)")
    axis.set_title("IEEE-69 comparison after direct network-constrained EPS training")
    axis.grid(axis="x", alpha=0.22)
    return _save_figure(fig, figure_dir, "trained_eps_all_algorithms_ieee69")


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for run e22 trained eps comparison.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed-count", type=int, default=e22.SEED_COUNT)
    parser.add_argument("--bootstrap-draws", type=int, default=e22.BOOTSTRAP_DRAWS)
    parser.add_argument("--max-datasets", type=int, default=len(e22.E22_DATASETS))
    parser.add_argument("--max-stress-modes", type=int, default=len(e22.STRESS_MODES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--training-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument(
        "--topology",
        choices=(TRAINING_TOPOLOGY,),
        default=TRAINING_TOPOLOGY,
        help="Training and testing are both on IEEE-69.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-all", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_count <= e22.SEED_COUNT:
        raise ValueError("seed-count must be between 1 and 30")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if args.training_samples is not None and args.training_samples < 16:
        raise ValueError("training-samples must be at least 16")
    if args.validation_samples is not None and args.validation_samples < 8:
        raise ValueError("validation-samples must be at least 8")
    datasets = e22.E22_DATASETS[: max(1, min(args.max_datasets, len(e22.E22_DATASETS)))]
    topologies = (args.topology,)
    stress_modes = tuple(e22.STRESS_MODES)[: max(1, min(args.max_stress_modes, len(e22.STRESS_MODES)))]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_by_dataset: dict[str, dict[str, Any]] = {}
    completed_datasets = 0
    if args.workers == 1:
        for dataset in datasets:
            result = _run_dataset_job(
                dataset,
                str(args.output),
                args.seed_count,
                topologies,
                stress_modes,
                args.force,
                args.retrain_all,
                args.training_samples,
                args.validation_samples,
            )
            metadata_by_dataset[dataset] = result["metadata"]
            completed_datasets += 1
            e22._write_json(
                args.output / "data/training_metadata.json",
                [metadata_by_dataset[name] for name in datasets if name in metadata_by_dataset],
            )
            e22._write_json(
                args.output / "checkpoint.json",
                {
                    "protocol": PROTOCOL,
                    "status": "running",
                    "completed_datasets": completed_datasets,
                    "dataset_count": len(datasets),
                    "workers": args.workers,
                    "seed_count": args.seed_count,
                    "updated_at": e22._utc_now(),
                },
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(datasets)), mp_context=context
        ) as executor:
            futures = {
                executor.submit(
                    _run_dataset_job,
                    dataset,
                    str(args.output),
                    args.seed_count,
                    topologies,
                    stress_modes,
                    args.force,
                    args.retrain_all,
                    args.training_samples,
                    args.validation_samples,
                ): dataset
                for dataset in datasets
            }
            for future in as_completed(futures):
                dataset = futures[future]
                result = future.result()
                metadata_by_dataset[dataset] = result["metadata"]
                completed_datasets += 1
                e22._write_json(
                    args.output / "data/training_metadata.json",
                    [
                        metadata_by_dataset[name]
                        for name in datasets
                        if name in metadata_by_dataset
                    ],
                )
                e22._write_json(
                    args.output / "checkpoint.json",
                    {
                        "protocol": PROTOCOL,
                        "status": "running",
                        "completed_datasets": completed_datasets,
                        "dataset_count": len(datasets),
                        "last_completed_dataset": dataset,
                        "workers": args.workers,
                        "seed_count": args.seed_count,
                        "updated_at": e22._utc_now(),
                    },
                )
                print(
                    json.dumps(
                        {
                            "stage": "dataset_complete",
                            "dataset": dataset,
                            "completed_datasets": completed_datasets,
                            "dataset_count": len(datasets),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    training_metadata = [metadata_by_dataset[dataset] for dataset in datasets]
    payloads = [
        json.loads((args.output / "raw" / f"{dataset}.json").read_text(encoding="utf-8"))
        for dataset in datasets
    ]

    rows = [row for payload in payloads for row in payload["seed_results"]]
    summaries = e22._summaries(rows, args.bootstrap_draws)
    native_overall = e22._overall(summaries)
    e22._write_rows(args.output / "data/trained_eps_by_seed.csv", rows)
    e22._write_rows(args.output / "data/trained_eps_summary.csv", summaries)

    if len(datasets) == len(e22.E22_DATASETS) and stress_modes == tuple(e22.STRESS_MODES):
        existing_summary = _read_csv(args.results_root / "E22/data/e22_summary.csv")
        existing_overall = _read_csv(args.results_root / "E22/data/e22_overall_summary.csv")
        combined_summary = [*existing_summary, *summaries]
        combined_overall = [*existing_overall, *native_overall]
        e22._write_rows(args.output / "data/combined_e22_summary.csv", combined_summary)
        e22._write_rows(args.output / "data/combined_e22_overall.csv", combined_overall)
        figure_dir = args.output / "figures"
        figures = []
        if topologies == e22.TOPOLOGIES:
            figures.extend(_plot_topology_dumbbell(combined_summary, figure_dir))
        figures.extend(_plot_dataset_heatmap(combined_summary, figure_dir))
        figures.extend(_plot_stress_profiles(combined_overall, figure_dir))
        figures.extend(_plot_native_gain(combined_summary, figure_dir))
        figures.extend(_plot_all_algorithms(combined_overall, figure_dir))
    else:
        figures = []

    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "dataset_count": len(datasets),
        "seed_count": args.seed_count,
        "topology_count": len(topologies),
        "stress_mode_count": len(stress_modes),
        "workers": args.workers,
        "training_topology": TRAINING_TOPOLOGY,
        "native_algorithms": list(NATIVE_ALGORITHMS),
        "native_by_seed_rows": len(rows),
        "native_summary_rows": len(summaries),
        "models": [
            {
                "dataset": row["dataset"],
                "model_path": row["model_path"],
                "model_sha256": row["model_sha256"],
                "model_reused": row["model_reused"],
            }
            for row in training_metadata
        ],
        "figures": figures,
        "completed_at": e22._utc_now(),
    }
    e22._write_json(args.output / "manifest.json", manifest)
    e22._write_json(args.output / "checkpoint.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
