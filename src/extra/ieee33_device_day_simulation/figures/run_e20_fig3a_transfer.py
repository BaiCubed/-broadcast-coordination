from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.linear_model import Ridge

from src.extra.ieee33_device_day_simulation.config_loader import load_config
from src.extra.ieee33_device_day_simulation.experiments.protocol import _signal_values
from src.extra.ieee33_device_day_simulation.figures.run_e20_transfer import (
    SOURCE_DATASETS,
    TARGET_DATASETS,
    _make_estimator,
    _load_estimator,
    _save_estimator,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_E1_ROOT = ROOT / "results/e1_full/E1_scale_boundary_new"
DEFAULT_E20_ROOT = ROOT / "results/E20"
DEFAULT_OUTPUT = DEFAULT_E20_ROOT / "fig3a_transfer"
DEFAULT_PROTOCOL = ROOT / "results/e1_full/config/protocol.yaml"
THRESHOLD = 0.95
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20260806
EPSILON = 1e-12
METHODS = (
    "in_domain",
    "zero_shot",
    "target_calibrated",
    "pooled_ridge",
)
METHOD_LABELS = {
    "in_domain": "E20 in-domain target model",
    "zero_shot": "E20 zero-shot transfer",
    "target_calibrated": "E20 target-calibrated transfer",
    "pooled_ridge": "Pooled Ridge transfer",
}
METHOD_COLORS = {
    "in_domain": "#2166ac",
    "zero_shot": "#d95f02",
    "target_calibrated": "#1b9e77",
    "pooled_ridge": "#7570b3",
}
SHORT_NAMES = {
    "camsl_japan_smart_meters": "CAMSL-JP",
    "european_lv_urban_8087": "EU-LV urban-8k",
    "irish_domestic_smart_meters": "Irish CER",
    "opsd_household_data": "OPSD",
    "smart_grid_smart_city": "SGSC",
}


class ResponseEstimator(Protocol):
    def estimate(self, signal: dict[str, Any]) -> Any:
        ...


class _ScalarEstimate:
    def __init__(self, response_kw: float):
        self.response_kw = float(response_kw)


class _RidgeEstimator:

    def __init__(self, model: Ridge, feature_extractor: ResponseEstimator):
        self.model = model
        self.feature_extractor = feature_extractor

    def estimate(self, signal: dict[str, Any]) -> _ScalarEstimate:
        features = self.feature_extractor._extract_features(signal)
        prediction = self.model.predict(np.asarray([features], dtype=float))[0]
        return _ScalarEstimate(float(prediction))


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator <= EPSILON:
        return None
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def log_interpolate_n95(
    left_n: float,
    left_r2: float,
    right_n: float,
    right_r2: float,
    threshold: float = THRESHOLD,
) -> float:
    if abs(right_r2 - left_r2) <= EPSILON:
        return float(right_n)
    fraction = float(np.clip(
        (threshold - left_r2) / (right_r2 - left_r2), 0.0, 1.0
    ))
    return float(10 ** (
        math.log10(left_n)
        + fraction * (math.log10(right_n) - math.log10(left_n))
    ))


def n95_from_rows(
    rows: list[dict[str, Any]], threshold: float = THRESHOLD
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["N"]))
    crossing = next(
        (index for index, row in enumerate(ordered) if float(row["r2"]) >= threshold),
        None,
    )
    if crossing is None:
        return {
            "threshold_N_95": None,
            "threshold_N_95_interpolated": None,
            "threshold_not_reached": True,
        }
    right = ordered[crossing]
    if crossing == 0:
        interpolated = float(right["N"])
    else:
        left = ordered[crossing - 1]
        interpolated = log_interpolate_n95(
            float(left["N"]),
            float(left["r2"]),
            float(right["N"]),
            float(right["r2"]),
            threshold,
        )
    return {
        "threshold_N_95": int(right["N"]),
        "threshold_N_95_interpolated": interpolated,
        "threshold_not_reached": False,
    }


def rebuild_e1_schedule(
    profile_steps: list[int], zone_count: int, random_seed: int
) -> list[list[tuple[int, int]]]:
    e1_seed = int(random_seed) + 1000
    schedule_seed = e1_seed + 101
    scenarios = ("valley_filling", "peak_shaving")
    schedule: list[list[tuple[int, int]]] = []
    for condition_index, _ in enumerate(profile_steps):
        rng = np.random.default_rng(schedule_seed + condition_index)
        scenario = scenarios[condition_index % len(scenarios)]
        schedule.append([
            _signal_values(0.0, 1.0, scenario, rng)
            for _ in range(zone_count)
        ])
    return schedule


def balanced_zone_counts(fleet_size: int, zone_count: int) -> list[int]:
    per_zone, remainder = divmod(int(fleet_size), int(zone_count))
    return [per_zone + (1 if index < remainder else 0) for index in range(zone_count)]


def transfer_prediction_by_condition(
    estimator: ResponseEstimator,
    schedule: list[list[tuple[int, int]]],
    profile_steps: list[int],
    fleet_size: int,
    zone_count: int,
    gain: float,
    output_unit: str = "normalized_per_device",
) -> np.ndarray:
    if output_unit not in {"normalized_per_device", "kw_at_5000"}:
        raise ValueError(f"unknown model output unit: {output_unit}")
    zone_counts = balanced_zone_counts(fleet_size, zone_count)
    predictions: list[float] = []
    for profile_step, zone_specs in zip(profile_steps, schedule):
        if len(zone_specs) != zone_count:
            raise ValueError("the signal and the configuration have different zone counts")
        aggregate = 0.0
        for zone_devices, (supply_demand, intensity) in zip(zone_counts, zone_specs):
            signal = {
                "supply_demand": int(supply_demand),
                "intensity": int(intensity),
                "price": 0.0,
                "hour": int(profile_step) // 12,
                "day_of_week": 0,
                "direction": 1 if int(supply_demand) <= 7 else -1,
            }
            response = float(estimator.estimate(signal).response_kw)
            per_device = response if output_unit == "normalized_per_device" else response / 5000.0
            aggregate += per_device * zone_devices
        predictions.append(float(aggregate * gain))
    return np.asarray(predictions, dtype=float)


def schedule_to_signals(
    schedule: list[list[tuple[int, int]]],
    profile_steps: list[int],
    zone_count: int,
) -> list[dict[str, Any]]:
    if len(schedule) != len(profile_steps):
        raise ValueError("the number of signal conditions does not match profile_steps")
    signals: list[dict[str, Any]] = []
    for profile_step, zone_specs in zip(profile_steps, schedule):
        if len(zone_specs) != zone_count:
            raise ValueError("the signal and the configuration have different zone counts")
        directions = np.asarray([int(direction) for direction, _ in zone_specs])
        intensities = np.asarray([int(intensity) for _, intensity in zone_specs])
        signals.append({
            "supply_demand": int(round(float(np.mean(directions)))),
            "intensity": int(round(float(np.mean(intensities)))),
            "price": 0.0,
            "hour": int(profile_step) // 12,
            "day_of_week": 0,
            "direction": 1 if float(np.mean(directions)) <= 7 else -1,
        })
    return signals


def fig3a_training_samples(
    dataset: str,
    e1_root: Path,
    zone_count: int,
    random_seed: int,
    train_replications: int = 10,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    archives = _archives(e1_root, dataset)
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    reference_steps: list[int] | None = None
    schedule: list[list[tuple[int, int]]] | None = None
    for fleet_size, archive in archives:
        with np.load(archive) as payload:
            aggregate = np.asarray(payload["aggregate_response_kw"], dtype=float)
            profile_steps = np.asarray(payload["profile_steps"], dtype=int).tolist()
        if aggregate.ndim != 2 or aggregate.shape[0] < train_replications:
            raise ValueError(f"{archive} has too few replications")
        if reference_steps is None:
            reference_steps = profile_steps
            schedule = rebuild_e1_schedule(profile_steps, zone_count, random_seed)
        elif profile_steps != reference_steps:
            raise ValueError(f"{dataset}: the conditions differ between fleet sizes")
        assert schedule is not None
        signals.extend(schedule_to_signals(schedule, profile_steps, zone_count))
        responses.extend(
            (np.mean(aggregate[:train_replications], axis=0) / float(fleet_size)).tolist()
        )
    return signals, responses, {
        "dataset": dataset,
        "archives": len(archives),
        "fleet_sizes": [fleet_size for fleet_size, _ in archives],
        "conditions_per_N": len(reference_steps or []),
        "train_replications": train_replications,
        "label": "mean(aggregate_response_kw[:train_replications], axis=0) / N",
        "test_replications_excluded": True,
    }


def _fit_fig3a_estimator(
    signals: list[dict[str, Any]],
    responses: list[float],
    seed: int,
) -> ResponseEstimator:
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = _make_estimator()
    estimator.fit(signals, responses)
    return estimator


def _fit_ridge_estimator(
    signals: list[dict[str, Any]], responses: list[float], alpha: float = 1.0
) -> _RidgeEstimator:
    feature_extractor = _make_estimator()
    features = np.asarray([
        feature_extractor._extract_features(signal) for signal in signals
    ], dtype=float)
    model = Ridge(alpha=float(alpha))
    model.fit(features, np.asarray(responses, dtype=float))
    return _RidgeEstimator(model, feature_extractor)


def _prediction_from_fleet_signal(
    estimator: ResponseEstimator,
    signal: dict[str, Any],
    fleet_size: int,
    gain: float = 1.0,
) -> float:
    return float(estimator.estimate(signal).response_kw) * float(fleet_size) * float(gain)


def cell_metrics(
    aggregate: np.ndarray,
    prediction_by_condition: np.ndarray,
    train_replications: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    test = aggregate[train_replications:]
    if test.shape[0] < 2:
        raise ValueError("at least two independent evaluation replications are required")
    if prediction_by_condition.shape != (test.shape[1],):
        raise ValueError("the predicted and the observed conditions differ in number")
    actual = test.T.reshape(-1)
    predicted = np.repeat(prediction_by_condition, test.shape[0])
    score = r2_score(actual, predicted)
    if score is None:
        raise ValueError("the R2 denominator is zero")

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, test.shape[1], size=test.shape[1])
        boot_actual = np.concatenate([test[:, index] for index in indices])
        boot_prediction = np.repeat(prediction_by_condition[indices], test.shape[0])
        value = r2_score(boot_actual, boot_prediction)
        if value is not None:
            bootstrap.append(value)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "r2": score,
        "r2_ci_lower": float(lower),
        "r2_ci_upper": float(upper),
        "rmse_kw": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mae_kw": float(np.mean(np.abs(actual - predicted))),
        "prediction_mean_kw": float(np.mean(prediction_by_condition)),
        "prediction_abs_mean_kw": float(np.mean(np.abs(prediction_by_condition))),
        "actual_mean_kw": float(np.mean(actual)),
        "actual_abs_mean_kw": float(np.mean(np.abs(actual))),
        "signal_conditions": int(test.shape[1]),
        "eval_replications": int(test.shape[0]),
    }


def _archives(e1_root: Path, dataset: str) -> list[tuple[int, Path]]:
    response_dir = e1_root / "raw/responses" / dataset
    rows: list[tuple[int, Path]] = []
    for path in response_dir.glob("responses_N*_data_coupled.npz"):
        fleet_text = path.stem.removeprefix("responses_N").removesuffix("_data_coupled")
        if fleet_text.isdigit():
            rows.append((int(fleet_text), path))
    if not rows:
        raise FileNotFoundError(f"missing response archive: {response_dir}")
    return sorted(rows)


def _zone_count(dataset: str) -> int:
    config_path = (
        ROOT
        / "results"
        / f"{dataset}_ieee33_real_load"
        / "coverage_fix/network_constrained_new/config/default.yaml"
    )
    config = load_config(config_path)
    return len(config["zones"]["zones"])


def _calibration_gains(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        dataset: float(
            payload["target_calibration"][f"{dataset}_aggregate"]["target_gain"]
        )
        for dataset in TARGET_DATASETS
    }


def evaluate_dataset(
    dataset: str,
    pooled_estimator: ResponseEstimator,
    in_domain_estimator: ResponseEstimator,
    ridge_estimator: ResponseEstimator,
    calibration_gain: float,
    e1_root: Path,
    train_replications: int,
    random_seed: int,
) -> dict[str, Any]:
    archives = _archives(e1_root, dataset)
    zone_count = _zone_count(dataset)
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    reference_steps: list[int] | None = None
    schedule: list[list[tuple[int, int]]] | None = None

    for n_index, (fleet_size, archive) in enumerate(archives):
        with np.load(archive) as payload:
            aggregate = np.asarray(payload["aggregate_response_kw"], dtype=float)
            profile_steps = np.asarray(payload["profile_steps"], dtype=int).tolist()
        if reference_steps is None:
            reference_steps = profile_steps
            schedule = rebuild_e1_schedule(profile_steps, zone_count, random_seed)
        elif profile_steps != reference_steps:
            raise ValueError(f"{dataset}: the conditions differ between fleet sizes")
        assert schedule is not None

        signals = schedule_to_signals(schedule, profile_steps, zone_count)
        in_domain_prediction = np.asarray([
            _prediction_from_fleet_signal(in_domain_estimator, signal, fleet_size)
            for signal in signals
        ])
        zero_shot_prediction = np.asarray([
            _prediction_from_fleet_signal(pooled_estimator, signal, fleet_size)
            for signal in signals
        ])
        ridge_prediction = np.asarray([
            _prediction_from_fleet_signal(ridge_estimator, signal, fleet_size)
            for signal in signals
        ])
        predictions = {
            "in_domain": in_domain_prediction,
            "zero_shot": zero_shot_prediction,
            "target_calibrated": zero_shot_prediction * calibration_gain,
            "pooled_ridge": ridge_prediction,
        }
        for method, prediction in predictions.items():
            rows_by_method[method].append({
                "N": fleet_size,
                **cell_metrics(
                    aggregate,
                    prediction,
                    train_replications,
                    BOOTSTRAP_SEED
                    + n_index
                    + 1000 * METHODS.index(method)
                    + sum(ord(character) for character in dataset),
                ),
            })

    methods = {
        method: {
            "label": METHOD_LABELS[method],
            "per_N": rows,
            **n95_from_rows(rows),
        }
        for method, rows in rows_by_method.items()
    }
    return {
        "dataset": dataset,
        "zone_count": zone_count,
        "N_values": [fleet_size for fleet_size, _ in archives],
        "target_calibration_gain": calibration_gain,
        "methods": methods,
    }


def _write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset",
            "method",
            "N",
            "R2",
            "R2_ci_lower",
            "R2_ci_upper",
            "RMSE_kw",
            "MAE_kw",
            "N95_grid",
            "N95_interpolated",
        ])
        for result in results:
            for method in METHODS:
                block = result["methods"][method]
                for row in block["per_N"]:
                    writer.writerow([
                        result["dataset"],
                        method,
                        row["N"],
                        row["r2"],
                        row["r2_ci_lower"],
                        row["r2_ci_upper"],
                        row["rmse_kw"],
                        row["mae_kw"],
                        block["threshold_N_95"],
                        block["threshold_N_95_interpolated"],
                    ])


def _plot_curves(results: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    for axis, result in zip(axes.flat, results):
        for method in METHODS:
            rows = result["methods"][method]["per_N"]
            values = np.asarray([float(row["r2"]) for row in rows])
            displayed = np.maximum(values, -0.10)
            axis.plot(
                [row["N"] for row in rows],
                displayed,
                "o-",
                color=METHOD_COLORS[method],
                linewidth=1.8,
                markersize=4,
                label=METHOD_LABELS[method],
            )
            clipped = values < -0.10
            if np.any(clipped):
                axis.scatter(
                    np.asarray([row["N"] for row in rows])[clipped],
                    np.full(int(np.sum(clipped)), -0.10),
                    marker="v",
                    color=METHOD_COLORS[method],
                    s=28,
                    zorder=4,
                )
        axis.axhline(THRESHOLD, color="#b2182b", linestyle=":", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_ylim(-0.12, 1.02)
        axis.set_xlabel("Physical fleet size $N$")
        axis.set_ylabel("$R^2$")
        axis.set_title(SHORT_NAMES.get(result["dataset"], result["dataset"]))
        axis.grid(alpha=0.22)
    if len(results) < len(axes.flat):
        for axis in axes.flat[len(results):]:
            axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Dataset-specific EPS transfer under the Figure 3A fleet-size protocol\n"
        "triangles at -0.10 denote negative R2 values clipped for readability",
        fontsize=15,
    )
    fig.savefig(output / "fig3a_transfer_r2.png", dpi=220)
    fig.savefig(output / "fig3a_transfer_r2.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    for axis, result in zip(axes.flat, results):
        for method in METHODS:
            rows = result["methods"][method]["per_N"]
            axis.plot(
                [row["N"] for row in rows],
                [row["r2"] for row in rows],
                "o-",
                color=METHOD_COLORS[method],
                linewidth=1.8,
                markersize=4,
                label=METHOD_LABELS[method],
            )
        axis.axhline(THRESHOLD, color="#b2182b", linestyle=":", linewidth=1.2)
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_ylim(-100.0, 1.05)
        axis.set_xlabel("Physical fleet size $N$")
        axis.set_ylabel("$R^2$ (symmetric log scale)")
        axis.set_title(SHORT_NAMES.get(result["dataset"], result["dataset"]))
        axis.grid(alpha=0.22)
    if len(results) < len(axes.flat):
        for axis in axes.flat[len(results):]:
            axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Full-range diagnostic: dataset-specific EPS transfer under Figure 3A",
        fontsize=15,
    )
    fig.savefig(output / "fig3a_transfer_r2_full.png", dpi=220)
    fig.savefig(output / "fig3a_transfer_r2_full.pdf")
    plt.close(fig)


def _plot_individual_curves(results: list[dict[str, Any]], output: Path) -> None:
    for result in results:
        fig, axis = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
        for method in METHODS:
            rows = result["methods"][method]["per_N"]
            n_values = np.asarray([row["N"] for row in rows], dtype=float)
            values = np.asarray([float(row["r2"]) for row in rows])
            displayed = np.maximum(values, -0.10)
            axis.plot(
                n_values,
                displayed,
                "o-",
                color=METHOD_COLORS[method],
                linewidth=2.0,
                markersize=5,
                label=METHOD_LABELS[method],
            )
            clipped = values < -0.10
            if np.any(clipped):
                axis.scatter(
                    n_values[clipped],
                    np.full(int(np.sum(clipped)), -0.10),
                    marker="v",
                    color=METHOD_COLORS[method],
                    s=34,
                    zorder=4,
                )
        axis.axhline(
            THRESHOLD,
            color="#b2182b",
            linestyle=":",
            linewidth=1.4,
            label=r"$R^2=0.95$ threshold",
        )
        axis.set_xscale("log")
        axis.set_ylim(-0.12, 1.02)
        axis.set_xlabel("Physical fleet size $N$")
        axis.set_ylabel("$R^2$")
        axis.set_title(
            f"{SHORT_NAMES.get(result['dataset'], result['dataset'])}\n"
            "Dataset-specific Figure 3A evaluation"
        )
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, loc="lower right")
        figure_stem = f"fig3a_transfer_r2_{result['dataset']}"
        fig.savefig(output / f"{figure_stem}.png", dpi=220)
        fig.savefig(output / f"{figure_stem}.pdf")
        plt.close(fig)


def _plot_n95(results: list[dict[str, Any]], output: Path) -> None:
    labels = [SHORT_NAMES.get(row["dataset"], row["dataset"]) for row in results]
    x = np.arange(len(results), dtype=float)
    offsets = np.linspace(-0.32, 0.32, len(METHODS))
    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for method_index, method in enumerate(METHODS):
        positions = x + offsets[method_index]
        reached_x: list[float] = []
        reached_y: list[float] = []
        missing_x: list[float] = []
        missing_y: list[float] = []
        for position, result in zip(positions, results):
            value = result["methods"][method]["threshold_N_95_interpolated"]
            if value is None:
                missing_x.append(float(position))
                missing_y.append(float(max(result["N_values"])))
            else:
                reached_x.append(float(position))
                reached_y.append(float(value))
        axis.scatter(
            reached_x,
            reached_y,
            marker="o",
            s=70,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        axis.scatter(
            missing_x,
            missing_y,
            marker="^",
            s=78,
            facecolors="none",
            edgecolors=METHOD_COLORS[method],
            linewidths=1.8,
        )
        for xi, maximum in zip(missing_x, missing_y):
            axis.text(
                xi,
                maximum * 1.08,
                f">{maximum:g}",
                color=METHOD_COLORS[method],
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )
    axis.set_yscale("log")
    axis.set_ylim(0.8, max(max(row["N_values"]) for row in results) * 1.65)
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("$N_{95}$ at $R^2=0.95$")
    axis.set_title(
        "Dataset-specific Figure 3A transfer threshold\n"
        "open triangles: threshold not reached by $N_{max}$"
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    fig.savefig(output / "fig3a_transfer_n95.png", dpi=220)
    fig.savefig(output / "fig3a_transfer_n95.pdf")
    plt.close(fig)


def n95_inflation_by_method(result: dict[str, Any]) -> dict[str, float | None]:
    baseline = result["methods"]["in_domain"]["threshold_N_95_interpolated"]
    ratios: dict[str, float | None] = {}
    for method in METHODS:
        current = result["methods"][method]["threshold_N_95_interpolated"]
        if baseline is None or current is None:
            ratios[method] = None
        else:
            ratios[method] = (
                1.0 if method == "in_domain" else float(current) / float(baseline)
            )
    return ratios


def _plot_n95_inflation(results: list[dict[str, Any]], output: Path) -> None:
    labels = [SHORT_NAMES.get(row["dataset"], row["dataset"]) for row in results]
    x = np.arange(len(results), dtype=float)
    offsets = np.linspace(-0.32, 0.32, len(METHODS))
    width = 0.14
    values_by_method: dict[str, list[float]] = {method: [] for method in METHODS}
    max_ratio = 1.0
    for result in results:
        ratios = n95_inflation_by_method(result)
        for method in METHODS:
            value = ratios[method]
            ratio = float("nan") if value is None else float(value)
            if np.isfinite(ratio):
                max_ratio = max(max_ratio, ratio)
            values_by_method[method].append(ratio)

    fig, axis = plt.subplots(figsize=(14, 6.8), constrained_layout=True)
    for method_index, method in enumerate(METHODS):
        positions = x + offsets[method_index]
        axis.bar(
            positions,
            values_by_method[method],
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for position, value in zip(positions, values_by_method[method]):
            if np.isfinite(value):
                axis.text(
                    position,
                    value + 0.025,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
    unavailable = [
        index
        for index, result in enumerate(results)
        if result["methods"]["in_domain"]["threshold_N_95_interpolated"] is None
    ]
    for index in unavailable:
        axis.text(
            x[index],
            0.05,
            "N95 unavailable",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
            rotation=90,
        )
    axis.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    axis.set_xticks(x, labels)
    axis.set_ylabel(r"$N_{95}$ inflation relative to in-domain EPS")
    axis.set_title(
        r"Dataset-specific transfer cost: $N_{95}$ inflation"
        "\n1.0 means no additional fleet-size cost; missing bars did not reach $R^2=0.95$"
    )
    axis.set_ylim(0.0, max(1.2, max_ratio * 1.22))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(output / "fig3a_transfer_n95_inflation.png", dpi=220)
    fig.savefig(output / "fig3a_transfer_n95_inflation.pdf")
    plt.close(fig)

    with (output / "fig3a_transfer_n95_inflation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *METHODS])
        for index, result in enumerate(results):
            writer.writerow([
                result["dataset"],
                *[values_by_method[method][index] for method in METHODS],
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for run e20 fig3a transfer.")
    parser.add_argument("--e1-root", type=Path, default=DEFAULT_E1_ROOT)
    parser.add_argument("--e20-root", type=Path, default=DEFAULT_E20_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--datasets", default=",".join(TARGET_DATASETS))
    args = parser.parse_args()

    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    unknown = sorted(set(datasets) - set(TARGET_DATASETS))
    if unknown:
        raise ValueError(f"the target datasets do not include: {unknown}")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    train_replications = int(protocol["e1"]["train_replications"])
    random_seed = int(protocol["random_seed"])

    source_signals: list[dict[str, Any]] = []
    source_responses: list[float] = []
    source_metadata: list[dict[str, Any]] = []
    for index, dataset in enumerate(SOURCE_DATASETS):
        zone_count = _zone_count(dataset)
        signals, responses, metadata = fig3a_training_samples(
            dataset,
            args.e1_root,
            zone_count,
            random_seed,
            train_replications,
        )
        source_signals.extend(signals)
        source_responses.extend(responses)
        source_metadata.append(metadata)
    pooled_estimator = _fit_fig3a_estimator(
        source_signals,
        source_responses,
        seed=3100000,
    )
    ridge_estimator = _fit_ridge_estimator(source_signals, source_responses)
    pooled_model_path = args.e20_root / "models/pooled_fig3a_sources.pt"
    _save_estimator(pooled_estimator, pooled_model_path)

    target_estimators: dict[str, ResponseEstimator] = {}
    target_model_paths: dict[str, Path] = {}
    target_metadata: dict[str, dict[str, Any]] = {}
    gains: dict[str, float] = {}
    calibration_metadata: dict[str, dict[str, Any]] = {}
    for index, dataset in enumerate(datasets):
        zone_count = _zone_count(dataset)
        signals, responses, metadata = fig3a_training_samples(
            dataset,
            args.e1_root,
            zone_count,
            random_seed,
            train_replications,
        )
        target_estimator = _fit_fig3a_estimator(
            signals,
            responses,
            seed=3200000 + index,
        )
        target_path = args.e20_root / "models" / f"in_domain_{dataset}.pt"
        _save_estimator(target_estimator, target_path)
        target_estimators[dataset] = target_estimator
        target_model_paths[dataset] = target_path
        target_metadata[dataset] = metadata

        actual = np.asarray(responses, dtype=float)
        pooled_predictions = np.asarray([
            float(pooled_estimator.estimate(signal).response_kw)
            for signal in signals
        ])
        denominator = float(np.dot(pooled_predictions, pooled_predictions))
        unclipped_gain = float(
            np.dot(pooled_predictions, actual) / max(denominator, EPSILON)
        )
        gain = float(np.clip(unclipped_gain, 0.50, 1.50))
        gains[dataset] = gain
        calibration_metadata[dataset] = {
            "gain": gain,
            "unclipped_gain": unclipped_gain,
            "clip_range": [0.50, 1.50],
            "sample_count": len(signals),
            "source": "target Figure 3A first 10 training replications",
        }

    results = [
        evaluate_dataset(
            dataset,
            pooled_estimator,
            target_estimators[dataset],
            ridge_estimator,
            gains[dataset],
            args.e1_root,
            train_replications,
            random_seed,
        )
        for dataset in datasets
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "E20_dataset_specific_fig3a_transfer_v2",
        "comparison_definition": (
            "Each target dataset is evaluated against its own 120 Figure 3A test "
            "replications; the original Figure 3A condition mean is not a comparator."
        ),
        "pooled_model": str(pooled_model_path),
        "additional_baselines": {
            "pooled_ridge": {
                "label": METHOD_LABELS["pooled_ridge"],
                "training": "all source Figure 3A training samples",
                "target": "single-device response",
                "regularization": "Ridge alpha=1.0",
            "online_complexity": "O(1) with respect to fleet size N",
            },
        },
        "model_training_target": {
            "pooled_transfer": "mean(aggregate_response_kw[:10], axis=0) / N",
            "in_domain": "mean(aggregate_response_kw[:10], axis=0) / N",
        },
        "inference_scale": (
            "predict one global broadcast condition per device and multiply by the "
            "target archive's physical fleet size N"
        ),
        "actual_response_source": str(args.e1_root / "raw/responses"),
        "network_mode": "aggregate_without_network_feedback",
        "availability_arm": "data_coupled",
        "source_datasets": list(SOURCE_DATASETS),
        "source_training": source_metadata,
        "target_training": target_metadata,
        "target_models": {dataset: str(path) for dataset, path in target_model_paths.items()},
        "fig3a_train_replications": train_replications,
        "fig3a_test_replications": 120,
        "test_replications_used_for_model_fitting": False,
        "target_calibration": calibration_metadata,
        "bootstrap_unit": "condition",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "n95_rule": "first raw N whose point R2 reaches 0.95",
        "n95_interpolation": "linear R2 between adjacent log10(N) grid points",
        "results": results,
    }
    result_path = args.output / "fig3a_transfer_results.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_summary_csv(results, args.output / "fig3a_transfer_summary.csv")
    _plot_curves(results, args.output)
    _plot_individual_curves(results, args.output)
    _plot_n95(results, args.output)
    _plot_n95_inflation(results, args.output)
    print(result_path)


if __name__ == "__main__":
    main()
