"""运行 E20 多数据集 EPS 训练、迁移测试和结果绘图。"""

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

from tools.ncstyle import SEQ

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
    """给固定维度 EPS 模型增加离线标量尺度和区间校准。"""

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


def _e20_root() -> Path:
    return _results_root() / "E20"


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
        raise RuntimeError(f"{dataset}/{network_mode} 的 in-domain 结果不是 {SEED_COUNT} seeds")
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
        "reuse_reason": "与 E20 目标测试使用相同 fixed5000、data_driven、dataset-keyed fleet 和 30-seed 协议",
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
                raise RuntimeError(f"{data_path} 的实验条件与 E20 不一致")
            for strategy in BASELINE_STRATEGIES:
                result = payload["results"][strategy]
                if len(result.get("seed_results", [])) != SEED_COUNT:
                    raise RuntimeError(
                        f"{dataset}/{network_mode}/{strategy} 不是 {SEED_COUNT} seeds"
                    )
                rows.append({
                    "dataset": dataset,
                    "network_mode": network_mode,
                    "algorithm": strategy,
                    "seed_count": SEED_COUNT,
                    "result": result,
                    "result_reused": True,
                    "reuse_reason": (
                        "与 E20 使用相同 fixed5000、data_driven、dataset-keyed fleet、"
                        "测试分区和 30 个配对 seeds"
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
        image = axis.imshow(np.asarray(values), vmin=0, vmax=100, cmap=SEQ, aspect="auto")
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


def _write_algorithm_document(path: Path) -> None:
    source_list = "、".join(SOURCE_DATASETS)
    target_list = "、".join(TARGET_DATASETS)
    strategy_rows = "\n".join(
        "|{label}|{training}|{complexity}|{description}|".format(
            label=COMPARISON_DEFINITIONS[algorithm]["label"],
            training={
                "not_applicable": "无离线训练；固定规则直接部署",
                "target_dataset_full_training": "目标数据集完整训练",
                "pooled_sources_without_target_data": "十个源数据集池化训练",
                "pooled_sources_plus_128_target_validation_samples": (
                    "源数据集池化训练，并用目标验证集 128 条样本校准"
                ),
            }[COMPARISON_DEFINITIONS[algorithm]["training_regime"]],
            complexity=COMPARISON_DEFINITIONS[algorithm]["complexity"],
            description=COMPARISON_DEFINITIONS[algorithm].get(
                "description",
                "EPS 广播协议；区别仅在离线模型的训练和校准数据。",
            ),
        )
        for algorithm in COMPARISON_ORDER
    )
    path.write_text(f"""# E20 跨数据集迁移与算法设计

## 实验目标

E20 检验 EPS 在多个数据集训练后，迁移到未参与训练的数据集时的弃电削减能力，并与 Figure 4D 的固定规则、模型驱动和集中式算法在完全相同的目标测试条件下比较。

## 数据划分

- 源数据集：{source_list}
- 未见目标数据集：{target_list}
- 每个数据集按设备源进行确定性 70%/15%/15% 训练、验证和测试划分。
- 设备规模固定为 5000，可用率使用 `data_driven`，网络分别使用 `aggregate` 和 `ieee33`。
- 每个目标数据集、网络和算法使用 30 个配对 seed。

## EPS 训练与迁移

### In-domain

在目标数据集训练分区上使用 Figure 4D 的完整训练协议训练 Dual Quantile NN，并在该目标数据集测试分区评估。该结果是目标域监督训练参照。

### Zero-shot transfer

十个源数据集各等权抽取 {SOURCE_TRAIN_SAMPLES_PER_DATASET} 条响应样本，共 {len(SOURCE_DATASETS) * SOURCE_TRAIN_SAMPLES_PER_DATASET} 条。训练目标为 `accepted_response_kw / {RESPONSE_SCALE_KW:.0f}`，避免模型仅记忆某个数据集的绝对响应尺度。目标测试时不读取任何目标训练或验证样本。

### Target calibrated

从目标验证分区抽取 {TARGET_CALIBRATION_SAMPLES} 条样本，只估计一个输出增益和绝对残差的 90% 分位区间；不更新神经网络参数。校准仍属于离线步骤。

EPS 模型使用 10 个固定输入特征：强度、供需码、区域、优先级、强度与方向交互、方向、小时正弦、小时余弦、高峰标志和周末标志。在线阶段只发送一个广播信号，并只接收一个聚合反馈标量，不上传逐设备状态，不需要逐设备 ACK，因此通信复杂度相对于设备数为 `O(1)`。

## 对比算法

|算法|训练方式|在线复杂度|设计|
|-|-|-|-|
{strategy_rows}

## 公平性说明

MPC、Mean-field、Virtual battery、PEM、Transactive、Local SOC rules 和 Centralized greedy UB 的当前实现没有离线拟合参数，因此不能人为复制为 In-domain、Zero-shot 和 Calibrated 三个版本。它们直接使用固定算法规则在目标测试集运行，作为同条件部署基线。把这些完全相同的结果重复标为三种迁移方式会制造无效比较。

所有复用基线均来自 `curtailment_baselines_final_fixed5000_<network>_data_driven.json`，与 E20 使用相同测试分区、5000 台逻辑设备、数据驱动可用率、数据集 ID 绑定的设备参数和 30 个配对 seed。EPS 三种训练方式与这些基线共享相同的目标测试协议。

## 评价指标

主指标为弃电削减率：

`100 × accepted_absorption_mwh / baseline_curtailment_mwh`

图中的误差条是同一数据集 30 个 seed 的标准差。总体图的柱高是五个未见目标数据集的宏平均，误差条是数据集间标准差，空心散点表示单个数据集。

## 输出文件

- `data/e20_transfer_results.json`：完整训练、校准、迁移和算法对比结果。
- `data/e20_transfer_summary.csv`：EPS 三种训练方式汇总。
- `data/e20_algorithm_comparison.csv`：全部算法、数据集和网络汇总。
- `figures/e20_transfer_*.png`：EPS 迁移方式对比。
- `figures/e20_algorithm_comparison_*.png`：全部算法公平对比。
""", encoding="utf-8")


def main() -> None:
    global PROTOCOL, SOURCE_DATASETS, TARGET_DATASETS
    parser = argparse.ArgumentParser(description=__doc__)
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
        raise ValueError("source 和 target 数据集均不能为空")
    overlap = sorted(set(SOURCE_DATASETS) & set(TARGET_DATASETS))
    if overlap:
        raise ValueError(f"source 与 target 数据集重叠: {overlap}")
    PROTOCOL = f"E20_pooled_source_transfer_v2_{args.split_id}_o1_online"
    e20_root = args.results_root / args.experiment_dir
    model_dir = e20_root / "models"
    data_dir = e20_root / "data"
    figure_dir = e20_root / "figures"
    for directory in (model_dir, data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    run_lock = _acquire_run_lock(e20_root)
    if run_lock is None:
        print(f"{e20_root} 已有 E20 任务运行，当前实例退出")
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
    document_path = e20_root / "ALGORITHM_DESIGN.md"
    _write_algorithm_document(document_path)
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
        "algorithm_design_document": str(document_path),
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
        "algorithm_design_document": str(document_path),
        "figures": figures,
    }
    _write_json(manifest_path, manifest)
    print(result_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
