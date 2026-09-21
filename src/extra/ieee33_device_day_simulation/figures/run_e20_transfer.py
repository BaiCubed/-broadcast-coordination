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

from src.estimation import EstimationResult, EPSEstimator, EstimatorConfig
from src.signal import SignalOptimizer

from ..config_loader import load_config
from ..population.device_day_loader import load_device_day_pool
from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as protocol


PROTOCOL = "E20_pooled_source_transfer_v1_o1_online"
FLEET_MODE = "fixed5000"
AVAILABILITY_MODE = "data_driven"
NETWORK_MODES = ("aggregate", "ieee33")
METHODS = ("in_domain", "zero_shot", "target_calibrated")
BASELINE_STRATEGIES = tuple(
    strategy for strategy in legacy.FIG4D_ORDER if strategy != "eps_broadcast"
)
EPS_COMPARISON_IDS = {
    "in_domain": "eps_in_domain",
    "zero_shot": "eps_zero_shot",
    "target_calibrated": "eps_target_calibrated",
}
COMPARISON_ORDER = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_in_domain",
    "eps_zero_shot",
    "eps_target_calibrated",
    "centralized_optimal",
)
SEED_COUNT = 30
RESPONSE_SCALE_KW = 5000.0
SOURCE_TRAIN_SAMPLES_PER_DATASET = 192
TARGET_CALIBRATION_SAMPLES = 128

SOURCE_DATASETS = (
    "bdg1_building_data_genome",
    "bdg2_building_data_genome",
    "complete_energy_community",
    "danish_smart_heat_meters",
    "european_lv_rural_2731",
    "european_lv_urban_35297",
    "goiener_smart_meters",
    "heapo_heat_pumps",
    "low_carbon_london",
    "norway_ami_energy_distribution",
)
TARGET_DATASETS = (
    "camsl_japan_smart_meters",
    "european_lv_urban_8087",
    "irish_domestic_smart_meters",
    "opsd_household_data",
    "smart_grid_smart_city",
)

COMPARISON_DEFINITIONS = {
    **{
        strategy: {
            **legacy.STRATEGY_DEFINITIONS[strategy],
            "training_regime": "not_applicable",
        }
        for strategy in BASELINE_STRATEGIES
    },
    "eps_in_domain": {
        "label": "EPS in-domain",
        "complexity": "O(1)",
        "training_regime": "target_dataset_full_training",
    },
    "eps_zero_shot": {
        "label": "EPS zero-shot",
        "complexity": "O(1)",
        "training_regime": "pooled_sources_without_target_data",
    },
    "eps_target_calibrated": {
        "label": "EPS calibrated",
        "complexity": "O(1)",
        "training_regime": "pooled_sources_plus_128_target_validation_samples",
    },
}

COMPARISON_COLORS = {
    **legacy.FIG4D_COLORS,
    "eps_in_domain": "#1f77b4",
    "eps_zero_shot": "#ff7f0e",
    "eps_target_calibrated": "#2ca02c",
}


class ScaledEstimator:

    def __init__(self, estimator: EPSEstimator, scale_kw: float, margin_kw: float):
        self.estimator = estimator
        self.scale_kw = float(scale_kw)
        self.margin_kw = float(margin_kw)

    def estimate(self, signal: dict[str, Any], device_states: Any = None) -> EstimationResult:
        result = self.estimator.estimate(signal, device_states=device_states)
        return replace(
            result,
            response_kw=float(result.response_kw * self.scale_kw),
            lower_bound=float(result.lower_bound * self.scale_kw - self.margin_kw),
            upper_bound=float(result.upper_bound * self.scale_kw + self.margin_kw),
        )


def _results_root() -> Path:
    return Path("results")


def _dataset_root(dataset: str) -> Path:
    return _results_root() / f"{dataset}_ieee33_real_load" / "coverage_fix" / "network_constrained_new"


def _config_for(dataset: str, network_mode: str) -> dict[str, Any]:
    root = _dataset_root(dataset)
    original = load_config(protocol._config_path(root))
    return protocol._condition_config(original, network_mode, AVAILABILITY_MODE)


def _normalized(values: list[float] | np.ndarray) -> list[float]:
    return (np.asarray(values, dtype=float) / RESPONSE_SCALE_KW).tolist()


def _make_estimator() -> EPSEstimator:
    return EPSEstimator(EstimatorConfig(
        target_coverage=0.9,
        enable_conformal=True,
        use_cqr=True,
        use_pytorch=True,
        pytorch_epochs=legacy.EPS_TRAINING_EPOCHS,
        pytorch_batch_size=64,
        pytorch_learning_rate=0.001,
    ))


def _save_estimator(estimator: EPSEstimator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy._save_eps_model(estimator, path)


def _load_estimator(path: Path) -> EPSEstimator:
    estimator, _, _ = protocol.load_frozen_eps_controller(path)
    return estimator


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _acquire_run_lock(root: Path) -> Any | None:
    lock_path = root / ".run.lock"
    handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _sample_source_data(dataset: str, network_mode: str, sample_count: int, seed: int) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    config = _config_for(dataset, network_mode)
    pool = load_device_day_pool(config)
    partitions = protocol.partition_profile_pool(pool, int(config["simulation"]["random_seed"]))
    fleet_size = protocol._fleet_size(pool, FLEET_MODE)
    signals, responses, metadata = protocol._response_samples(
        partitions.train,
        FLEET_MODE,
        fleet_size,
        config,
        AVAILABILITY_MODE,
        sample_count,
        seed,
    )
    return signals, _normalized(responses), {
        "dataset": dataset,
        "network_mode": network_mode,
        "sample_count": sample_count,
        "pool_sources": pool.source_count,
        "pool_device_days": len(pool.records),
        "sampling": metadata,
    }


def _train_pooled(network_mode: str, model_path: Path) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    sources: list[dict[str, Any]] = []
    for index, dataset in enumerate(SOURCE_DATASETS):
        source_signals, source_responses, metadata = _sample_source_data(
            dataset,
            network_mode,
            SOURCE_TRAIN_SAMPLES_PER_DATASET,
            2200000 + index * 1000 + (0 if network_mode == "aggregate" else 500000),
        )
        signals.extend(source_signals)
        responses.extend(source_responses)
        sources.append(metadata)
    estimator = _make_estimator()
    estimator.fit(signals, responses)
    _save_estimator(estimator, model_path)
    parameters = estimator._learned_params or {}
    return {
        "model": str(model_path),
        "source_datasets": list(SOURCE_DATASETS),
        "sample_count": len(signals),
        "samples_per_source": SOURCE_TRAIN_SAMPLES_PER_DATASET,
        "model_type": str(parameters.get("model_type", "unknown")),
        "source_sampling": sources,
        "response_scale_kw": RESPONSE_SCALE_KW,
        "training_target": "accepted_response_kw / fixed_response_scale_kw",
    }


def _target_calibration(dataset: str, network_mode: str, pooled_model: Path) -> dict[str, Any]:
    config = _config_for(dataset, network_mode)
    pool = load_device_day_pool(config)
    partitions = protocol.partition_profile_pool(pool, int(config["simulation"]["random_seed"]))
    fleet_size = protocol._fleet_size(pool, FLEET_MODE)
    signals, responses, metadata = protocol._response_samples(
        partitions.validation,
        FLEET_MODE,
        fleet_size,
        config,
        AVAILABILITY_MODE,
        TARGET_CALIBRATION_SAMPLES,
        2400000 + (0 if network_mode == "aggregate" else 500000),
    )
    estimator = _load_estimator(pooled_model)
    actual = np.asarray(_normalized(responses), dtype=float)
    predicted = np.asarray([
        estimator.estimate(signal).response_kw for signal in signals
    ], dtype=float)
    denominator = float(np.dot(predicted, predicted))
    gain = float(np.dot(predicted, actual) / max(denominator, 1e-12))
    gain = float(np.clip(gain, 0.50, 1.50))
    residuals = np.abs(actual - gain * predicted)
    q90 = float(np.quantile(residuals, 0.90))
    return {
        "dataset": dataset,
        "network_mode": network_mode,
        "sample_count": len(signals),
        "target_gain": gain,
        "target_residual_q90_normalized": q90,
        "target_residual_q90_kw": q90 * RESPONSE_SCALE_KW,
        "calibration_r2": float(1.0 - np.sum((actual - gain * predicted) ** 2) / max(np.sum((actual - np.mean(actual)) ** 2), 1e-12)),
        "sampling": metadata,
    }


def _evaluate_task(task: tuple[str, str, str, str, dict[str, Any]]) -> dict[str, Any]:
    logging.getLogger("src.simulation.simulator").setLevel(logging.WARNING)
    dataset, network_mode, method, model_path, calibration = task
    config = _config_for(dataset, network_mode)
    pool = load_device_day_pool(config)
    partitions = protocol.partition_profile_pool(pool, int(config["simulation"]["random_seed"]))
    fleet_size = protocol._fleet_size(pool, FLEET_MODE)
    base = _load_estimator(Path(model_path))
    gain = float(calibration.get("target_gain", 1.0)) if method == "target_calibrated" else 1.0
    margin = float(calibration.get("target_residual_q90_kw", 0.0)) if method == "target_calibrated" else 0.0
    output_scale = 1.0 if method == "in_domain" else RESPONSE_SCALE_KW * gain
    optimizer = SignalOptimizer(ScaledEstimator(base, output_scale, margin))
    rows: list[dict[str, Any]] = []
    dataset_id = str(config["population"].get("canonical_adapter", {}).get("dataset", dataset))
    for seed_index in range(SEED_COUNT):
        seed = int(config["simulation"]["random_seed"]) + seed_index
        records, fleet_metadata = protocol.sample_device_fleet(
            partitions.test,
            FLEET_MODE,
            fleet_size,
            seed,
            dataset_id=dataset_id,
        )
        scenario = protocol.build_pure_sim_scenario(records, config)
        availability = protocol.availability_probability(records, config, AVAILABILITY_MODE)
        result = legacy._run_seed(
            "eps_broadcast",
            records,
            config,
            scenario,
            availability,
            seed + 100000 * (legacy.FIG4D_ORDER.index("eps_broadcast") + 1),
            seed + 1910000,
            eps_optimizer=optimizer,
        )
        result["device_mean_capacity_kwh"] = fleet_metadata["mean_capacity_kwh"]
        result["device_mean_peak_power_kw"] = fleet_metadata["mean_peak_power_kw"]
        rows.append(result)
    summary = legacy._summarize(rows)
    return {
        "dataset": dataset,
        "network_mode": network_mode,
        "method": method,
        "seed_count": SEED_COUNT,
        "model": model_path,
        "target_gain": gain,
        "target_interval_margin_kw": margin,
        "model_output_scale": output_scale,
        "result": summary,
    }


def _existing_in_domain_result(dataset: str, network_mode: str) -> dict[str, Any]:
    data_path = _dataset_root(dataset) / "data" / f"curtailment_baselines_final_fixed5000_{network_mode}_data_driven.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    result = payload["results"]["eps_broadcast"]
    if len(result.get("seed_results", [])) != SEED_COUNT:
        raise RuntimeError(f"{dataset}/{network_mode}: the in-domain result does not hold {SEED_COUNT} seeds")
    model_path = _dataset_root(dataset) / "data" / f"fig4d_eps_estimator_final_fixed5000_{network_mode}_data_driven.pt"
    return {
        "dataset": dataset,
        "network_mode": network_mode,
        "method": "in_domain",
        "seed_count": SEED_COUNT,
        "model": str(model_path),
        "target_gain": 1.0,
        "target_interval_margin_kw": 0.0,
        "model_output_scale": 1.0,
        "result": result,
        "result_reused": True,
        "reuse_reason": "same fixed5000, data_driven, dataset-keyed fleet and 30-seed protocol as the target test",
        "source_result": str(data_path),
    }


def _existing_baseline_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for network_mode in NETWORK_MODES:
        for dataset in TARGET_DATASETS:
            data_path = (
                _dataset_root(dataset)
                / "data"
                / f"curtailment_baselines_final_fixed5000_{network_mode}_data_driven.json"
            )
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            condition = payload.get("condition", {})
            expected = {
                "fleet_mode": FLEET_MODE,
                "network_mode": network_mode,
                "availability_mode": AVAILABILITY_MODE,
            }
            if any(condition.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"{data_path}: the experiment conditions do not match")
            for strategy in BASELINE_STRATEGIES:
                result = payload["results"][strategy]
                if len(result.get("seed_results", [])) != SEED_COUNT:
                    raise RuntimeError(
                        f"{dataset}/{network_mode}/{strategy} does not hold {SEED_COUNT} seeds"
                    )
                rows.append({
                    "dataset": dataset,
                    "network_mode": network_mode,
                    "algorithm": strategy,
                    "seed_count": SEED_COUNT,
                    "result": result,
                    "result_reused": True,
                    "reuse_reason": (
                        "same fixed5000, data_driven, dataset-keyed fleet, "
                        "test partition and 30 paired seeds"
                    ),
                    "source_result": str(data_path),
                })
    return rows


def _comparison_rows(
    transfer_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            **row,
            "training_regime": COMPARISON_DEFINITIONS[row["algorithm"]][
                "training_regime"
            ],
        }
        for row in baseline_results
    ]
    rows.extend({
        "dataset": row["dataset"],
        "network_mode": row["network_mode"],
        "algorithm": EPS_COMPARISON_IDS[row["method"]],
        "seed_count": row["seed_count"],
        "result": row["result"],
        "training_regime": COMPARISON_DEFINITIONS[
            EPS_COMPARISON_IDS[row["method"]]
        ]["training_regime"],
        "model": row["model"],
    } for row in transfer_results)
    rows.sort(key=lambda row: (
        row["network_mode"],
        TARGET_DATASETS.index(row["dataset"]),
        COMPARISON_ORDER.index(row["algorithm"]),
    ))
    return rows


def _comparison_value(
    rows: list[dict[str, Any]], dataset: str, network_mode: str, algorithm: str
) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["dataset"] == dataset
        and row["network_mode"] == network_mode
        and row["algorithm"] == algorithm
    )


def _comparison_tick_labels() -> list[str]:
    return [
        f"{COMPARISON_DEFINITIONS[algorithm]['label']}\n"
        f"{COMPARISON_DEFINITIONS[algorithm]['complexity']}"
        for algorithm in COMPARISON_ORDER
    ]


def _plot_algorithm_comparison(
    rows: list[dict[str, Any]], output_dir: Path
) -> list[str]:
    paths: list[str] = []
    tick_labels = _comparison_tick_labels()
    colors = [COMPARISON_COLORS[algorithm] for algorithm in COMPARISON_ORDER]
    y = np.arange(len(COMPARISON_ORDER), dtype=float)
    for network_mode in NETWORK_MODES:
        fig, axes = plt.subplots(3, 2, figsize=(20, 23), constrained_layout=True)
        for axis, dataset in zip(axes.flat, TARGET_DATASETS):
            selected = [
                _comparison_value(rows, dataset, network_mode, algorithm)
                for algorithm in COMPARISON_ORDER
            ]
            values = [row["result"]["mean_reduction_pct"] for row in selected]
            errors = [row["result"]["std_reduction_pct"] for row in selected]
            axis.barh(y, values, xerr=errors, color=colors, capsize=3, alpha=0.92)
            for yi, value in zip(y, values):
                axis.text(min(value + 1.0, 108.0), yi, f"{value:.1f}", va="center", fontsize=8)
            axis.set_yticks(y, tick_labels, fontsize=8)
            axis.set_xlim(0, 112)
            axis.set_xlabel("Curtailment reduction (%)")
            axis.set_title(dataset.replace("_", " ").title())
            axis.grid(axis="x", alpha=0.25)
            axis.invert_yaxis()
        axes.flat[-1].axis("off")
        fig.suptitle(
            f"E20 cross-dataset algorithm comparison: {network_mode} | 30 paired seeds",
            fontsize=18,
        )
        path = output_dir / f"e20_algorithm_comparison_{network_mode}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    for axis, network_mode in zip(axes, NETWORK_MODES):
        values_by_algorithm = np.asarray([
            [
                _comparison_value(rows, dataset, network_mode, algorithm)["result"][
                    "mean_reduction_pct"
                ]
                for dataset in TARGET_DATASETS
            ]
            for algorithm in COMPARISON_ORDER
        ])
        means = np.mean(values_by_algorithm, axis=1)
        between_dataset_std = np.std(values_by_algorithm, axis=1)
        axis.barh(
            y,
            means,
            xerr=between_dataset_std,
            color=colors,
            capsize=3,
            alpha=0.88,
        )
        for algorithm_index, dataset_values in enumerate(values_by_algorithm):
            axis.scatter(
                dataset_values,
                np.full(len(dataset_values), algorithm_index),
                color="#202020",
                facecolors="white",
                linewidths=0.8,
                s=22,
                zorder=3,
            )
            axis.text(
                min(means[algorithm_index] + 1.0, 108.0),
                algorithm_index,
                f"{means[algorithm_index]:.1f}",
                va="center",
                fontsize=8,
            )
        axis.set_yticks(y, tick_labels, fontsize=8)
        axis.set_xlim(0, 112)
        axis.set_xlabel("Curtailment reduction (%)")
        axis.set_title(network_mode)
        axis.grid(axis="x", alpha=0.25)
        axis.invert_yaxis()
    fig.suptitle(
        "E20 macro comparison across five unseen target datasets\n"
        "bars: dataset mean; error bars: between-dataset SD; dots: individual datasets",
        fontsize=16,
    )
    path = output_dir / "e20_algorithm_comparison_overall.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _plot(results: list[dict[str, Any]], output_dir: Path) -> list[str]:
    labels = {
        "in_domain": "In-domain",
        "zero_shot": "Zero-shot transfer",
        "target_calibrated": "Target calibrated",
    }
    targets = list(TARGET_DATASETS)
    target_labels = [dataset.replace("_", " ").title() for dataset in targets]
    paths: list[str] = []
    for mode in NETWORK_MODES:
        fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
        x = np.arange(len(targets), dtype=float)
        width = 0.24
        for index, method in enumerate(METHODS):
            values = [
                next(item["result"]["mean_reduction_pct"] for item in results
                     if item["dataset"] == dataset and item["network_mode"] == mode and item["method"] == method)
                for dataset in targets
            ]
            errors = [
                next(item["result"]["std_reduction_pct"] for item in results
                     if item["dataset"] == dataset and item["network_mode"] == mode and item["method"] == method)
                for dataset in targets
            ]
            ax.bar(x + (index - 1) * width, values, width, yerr=errors, capsize=3, label=labels[method])
            for xi, value in zip(x + (index - 1) * width, values):
                ax.text(xi, min(value + 2.0, 108.0), f"{value:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, target_labels, rotation=25, ha="right")
        ax.set_ylabel("Curtailment reduction (%)")
        ax.set_ylim(0, 110)
        ax.set_title(f"E20 EPS transfer: {mode} | online communication O(1)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        path = output_dir / f"e20_transfer_{mode}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    matrix = []
    for mode in NETWORK_MODES:
        matrix.append([
            [next(item["result"]["mean_reduction_pct"] for item in results
                  if item["dataset"] == dataset and item["network_mode"] == mode and item["method"] == method)
             for method in METHODS]
            for dataset in targets
        ])
    for axis, mode, values in zip(axes, NETWORK_MODES, matrix):
        image = axis.imshow(np.asarray(values), vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        axis.set_title(mode)
        axis.set_xticks(np.arange(len(METHODS)), [labels[m] for m in METHODS], rotation=25, ha="right")
        axis.set_yticks(np.arange(len(targets)), target_labels)
        for row_index in range(len(targets)):
            for col_index in range(len(METHODS)):
                value = values[row_index][col_index]
                axis.text(col_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value > 55 else "#222222")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Curtailment reduction (%)")
    path = output_dir / "e20_transfer_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset",
            "network_mode",
            "method",
            "seed_count",
            "mean_reduction_pct",
            "std_reduction_pct",
            "target_gain",
            "target_interval_margin_kw",
        ])
        for row in results:
            writer.writerow([
                row["dataset"],
                row["network_mode"],
                row["method"],
                row["seed_count"],
                row["result"]["mean_reduction_pct"],
                row["result"]["std_reduction_pct"],
                row["target_gain"],
                row["target_interval_margin_kw"],
            ])


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset",
            "network_mode",
            "algorithm",
            "algorithm_label",
            "training_regime",
            "online_complexity",
            "seed_count",
            "mean_reduction_pct",
            "std_reduction_pct",
        ])
        for row in rows:
            definition = COMPARISON_DEFINITIONS[row["algorithm"]]
            writer.writerow([
                row["dataset"],
                row["network_mode"],
                row["algorithm"],
                definition["label"],
                row["training_regime"],
                definition["complexity"],
                row["seed_count"],
                row["result"]["mean_reduction_pct"],
                row["result"]["std_reduction_pct"],
            ])


def main() -> None:
    global PROTOCOL, SOURCE_DATASETS, TARGET_DATASETS
    parser = argparse.ArgumentParser(description="Command line entry point for run e20 transfer.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("E20"))
    parser.add_argument("--split-id", default="fold_01")
    parser.add_argument("--source-datasets")
    parser.add_argument("--target-datasets")
    parser.add_argument("--workers", type=int, default=max(1, min(3, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-models", action="store_true")
    args = parser.parse_args()
    if args.source_datasets:
        SOURCE_DATASETS = tuple(
            dataset.strip()
            for dataset in args.source_datasets.split(",")
            if dataset.strip()
        )
    if args.target_datasets:
        TARGET_DATASETS = tuple(
            dataset.strip()
            for dataset in args.target_datasets.split(",")
            if dataset.strip()
        )
    if not SOURCE_DATASETS or not TARGET_DATASETS:
        raise ValueError("the source and target dataset lists must not be empty")
    overlap = sorted(set(SOURCE_DATASETS) & set(TARGET_DATASETS))
    if overlap:
        raise ValueError(f"source and target datasets overlap: {overlap}")
    PROTOCOL = f"E20_pooled_source_transfer_v2_{args.split_id}_o1_online"
    e20_root = args.results_root / args.experiment_dir
    model_dir = e20_root / "models"
    data_dir = e20_root / "data"
    figure_dir = e20_root / "figures"
    for directory in (model_dir, data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    run_lock = _acquire_run_lock(e20_root)
    if run_lock is None:
        print(f"{e20_root} already holds a running task; this instance exits")
        return
    manifest_path = e20_root / "e20_manifest.json"
    result_path = data_dir / "e20_transfer_results.json"
    training_cache_path = data_dir / "e20_training_cache.json"
    checkpoint_path = data_dir / "e20_evaluation_checkpoint.json"
    if result_path.is_file() and manifest_path.is_file() and not args.force:
        print(result_path)
        return

    if training_cache_path.is_file() and not args.retrain_models:
        training_cache = json.loads(training_cache_path.read_text(encoding="utf-8"))
        model_metadata = training_cache["model_metadata"]
        calibration_metadata = training_cache["target_calibration"]
    else:
        model_metadata: dict[str, Any] = {}
        calibration_metadata: dict[str, Any] = {}
        for mode in NETWORK_MODES:
            pooled_path = model_dir / f"pooled_{mode}.pt"
            if pooled_path.is_file() and not args.retrain_models:
                model_metadata[f"pooled_{mode}"] = {
                    "model": str(pooled_path),
                    "source_datasets": list(SOURCE_DATASETS),
                    "sample_count": len(SOURCE_DATASETS) * SOURCE_TRAIN_SAMPLES_PER_DATASET,
                    "samples_per_source": SOURCE_TRAIN_SAMPLES_PER_DATASET,
                    "model_reused": True,
                    "response_scale_kw": RESPONSE_SCALE_KW,
                    "training_target": "accepted_response_kw / fixed_response_scale_kw",
                }
            else:
                model_metadata[f"pooled_{mode}"] = _train_pooled(mode, pooled_path)
            for dataset in TARGET_DATASETS:
                model_path = _dataset_root(dataset) / "data" / f"fig4d_eps_estimator_final_fixed5000_{mode}_data_driven.pt"
                if not model_path.is_file():
                    raise FileNotFoundError(model_path)
                model_metadata[f"in_domain_{dataset}_{mode}"] = {
                    "model": str(model_path),
                    "dataset": dataset,
                    "network_mode": mode,
                    "model_reused": True,
                    "source": "existing complete in-domain Figure 4D training",
                    "model_output_unit": "kW",
                }
                calibration_metadata[f"{dataset}_{mode}"] = _target_calibration(dataset, mode, pooled_path)
        _write_json(training_cache_path, {
            "protocol": PROTOCOL,
            "model_metadata": model_metadata,
            "target_calibration": calibration_metadata,
        })

    tasks: list[tuple[str, str, str, str, dict[str, Any]]] = []
    for mode in NETWORK_MODES:
        pooled_path = str(model_dir / f"pooled_{mode}.pt")
        for dataset in TARGET_DATASETS:
            calibration = calibration_metadata[f"{dataset}_{mode}"]
            tasks.extend([
                (dataset, mode, "zero_shot", pooled_path, {}),
                (dataset, mode, "target_calibrated", pooled_path, calibration),
            ])
    results: list[dict[str, Any]] = [
        _existing_in_domain_result(dataset, mode)
        for mode in NETWORK_MODES
        for dataset in TARGET_DATASETS
    ]
    if checkpoint_path.is_file() and not args.force:
        checkpoint_rows = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        existing_keys = {
            (row["dataset"], row["network_mode"], row["method"])
            for row in results
        }
        results.extend(
            row for row in checkpoint_rows
            if (row["dataset"], row["network_mode"], row["method"]) not in existing_keys
        )
    completed_keys = {
        (row["dataset"], row["network_mode"], row["method"])
        for row in results
    }
    tasks = [
        task for task in tasks
        if (task[0], task[1], task[2]) not in completed_keys
    ]
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                failure = {
                    "dataset": task[0],
                    "network_mode": task[1],
                    "method": task[2],
                    "error": repr(exc),
                }
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
                continue
            results.append(row)
            _write_json(checkpoint_path, results)
            print(json.dumps({
                "dataset": row["dataset"],
                "network_mode": row["network_mode"],
                "method": row["method"],
                "seed_count": row["seed_count"],
                "mean_reduction_pct": row["result"]["mean_reduction_pct"],
            }, ensure_ascii=False), flush=True)
    if failures:
        _write_json(manifest_path, {
            "protocol": PROTOCOL,
            "condition_count": len(NETWORK_MODES) * len(TARGET_DATASETS) * len(METHODS),
            "completed_count": len(results),
            "failed_count": len(failures),
            "failures": failures,
            "checkpoint": str(checkpoint_path),
        })
        raise SystemExit(1)
    results.sort(key=lambda row: (row["network_mode"], TARGET_DATASETS.index(row["dataset"]), METHODS.index(row["method"])))
    summary_csv_path = data_dir / "e20_transfer_summary.csv"
    _write_summary_csv(results, summary_csv_path)
    baseline_results = _existing_baseline_results()
    comparison_results = _comparison_rows(results, baseline_results)
    comparison_csv_path = data_dir / "e20_algorithm_comparison.csv"
    _write_comparison_csv(comparison_results, comparison_csv_path)
    figures = _plot(results, figure_dir)
    figures.extend(_plot_algorithm_comparison(comparison_results, figure_dir))
    payload = {
        "protocol": PROTOCOL,
        "split_id": args.split_id,
        "experiment_dir": str(args.experiment_dir),
        "fleet_mode": FLEET_MODE,
        "availability_mode": AVAILABILITY_MODE,
        "network_modes": list(NETWORK_MODES),
        "source_datasets": list(SOURCE_DATASETS),
        "target_datasets": list(TARGET_DATASETS),
        "methods": list(METHODS),
        "seed_count": SEED_COUNT,
        "response_scale_kw": RESPONSE_SCALE_KW,
        "online_protocol": {
            "input_features": 10,
            "controller_to_device": "one broadcast signal",
            "device_to_controller": "one aggregate scalar feedback per step",
            "per_device_ack": False,
            "per_device_state_upload": False,
            "communication_complexity": "O(1) with respect to N",
        },
        "model_metadata": model_metadata,
        "target_calibration": calibration_metadata,
        "results": results,
        "baseline_results": baseline_results,
        "algorithm_comparison": {
            "order": list(COMPARISON_ORDER),
            "definitions": COMPARISON_DEFINITIONS,
            "results": comparison_results,
            "fairness": (
                "Non-EPS baselines have no offline fitted model and are evaluated once "
                "per target condition; EPS has three explicit training regimes."
            ),
        },
        "summary_csv": str(summary_csv_path),
        "comparison_csv": str(comparison_csv_path),
        "figures": figures,
    }
    _write_json(result_path, payload)
    manifest = {
        "protocol": PROTOCOL,
        "split_id": args.split_id,
        "experiment_dir": str(args.experiment_dir),
        "condition_count": len(NETWORK_MODES) * len(TARGET_DATASETS) * len(METHODS),
        "completed_count": len(results),
        "failed_count": 0,
        "seed_count": SEED_COUNT,
        "baseline_reference_count": len(baseline_results),
        "algorithm_comparison_count": len(comparison_results),
        "result": str(result_path),
        "figures": figures,
    }
    _write_json(manifest_path, manifest)
    print(result_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
