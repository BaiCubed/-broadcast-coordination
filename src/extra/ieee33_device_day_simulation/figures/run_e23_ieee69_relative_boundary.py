from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as training
from . import run_e22_ieee69_complexity as e22
from . import run_e22_trained_eps_comparison as direct


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/E23"
SOURCE = ROOT / "results/E22/trained_eps_ieee69_direct"
PROTOCOL = "E23_ieee69_continuous_relative_boundary_v1"
TOPOLOGY = "ieee69"
FLEET_SIZE = e22.FLEET_SIZE
STEPS = e22.STEPS
BOOTSTRAP_DRAWS = 2000

ALGORITHMS = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_ieee69_fused",
    "centralized_optimal",
)
BASELINES = (
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
)
O1_BASELINE = "local_rules"

REQUEST_SCALES = tuple(float(value) for value in np.round(np.arange(0.0, 1.51, 0.10), 2))
LINE_PRESSURES = tuple(float(value) for value in np.round(np.arange(0.0, 0.41, 0.05), 2))
CONCENTRATIONS = tuple(float(value) for value in np.round(np.arange(0.0, 0.81, 0.10), 2))

AXES = {
    "request_intensity": {
        "label": "broadcast request intensity q",
        "values": REQUEST_SCALES,
        "unit": "q; 1.0 is the nominal request intensity",
        "boundary_start": 1.0,
    },
    "line_derating": {
        "label": "line-capacity stress lambda_line",
        "values": LINE_PRESSURES,
        "unit": "lambda_line; 0 is the rated capacity, 0.4 reduces it to 60%",
        "boundary_start": 0.0,
    },
    "spatial_concentration": {
        "label": "spatial concentration kappa",
        "values": CONCENTRATIONS,
        "unit": "kappa; 0 is uniform deployment, 0.8 puts 80% on distal buses",
        "boundary_start": 0.0,
    },
}

COLORS = {
    "no_coordination": "#9c9c9c",
    "local_rules": "#ed7d31",
    "mpc_optimal": "#7057ff",
    "mean_field_control": "#4e79a7",
    "virtual_battery": "#59a14f",
    "packetized_energy_management": "#af7aa1",
    "transactive_control": "#d55e00",
    "eps_ieee69_fused": "#1f5aa6",
    "centralized_optimal": "#3a9d66",
}

matplotlib.rcParams["font.family"] = "SimSun"
matplotlib.rcParams["axes.unicode_minus"] = False


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no data: {path}")
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _custom_concentration_remap(
    records: list[Any], case: dict[str, Any], seed: int, concentration: float
) -> tuple[list[Any], dict[str, Any]]:
    buses = sorted({int(row[1]) for row in case["branches"]})
    target = e22._concentration_target(case)
    target_bus = int(target["target_bus"])
    concentration = float(np.clip(concentration, 0.0, 0.8))
    concentrated = int(round(concentration * len(records)))
    remaining = len(records) - concentrated
    counts = e22._even_bus_counts(buses, remaining)
    counts[target_bus] += concentrated
    assignments = np.asarray(
        [bus for bus, count in counts.items() for _ in range(count)], dtype=int
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(assignments)
    remapped = [
        replace(record, bus_id=int(assignments[index]))
        for index, record in enumerate(records)
    ]
    downstream = set(target["target_downstream_buses"])
    metadata = {
        "placement_mode": "continuous_node_concentration",
        "configured_concentration_fraction": concentration,
        "actual_target_feeder_fraction": float(sum(counts.get(bus, 0) for bus in downstream) / len(records)),
        "actual_target_bus_fraction": float(counts[target_bus] / len(records)),
        "maximum_bus_fraction": float(max(counts.values()) / len(records)),
        **target,
    }
    return remapped, metadata


def _base_stress_settings() -> None:
    e22.STRESS_MODES["E23_BASE"] = {
        "label": "E23 continuous base",
        "plain_label": "continuous stress baseline",
        "load_scale": 0.45,
        "solar_ratio": 0.75,
        "wind_ratio": 0.30,
        "input_mode": "proportional",
        "capacity_multiplier": 1.0,
        "forecast_error": 0.0,
        "placement_mode": "load_weighted",
    }


def _network(case: dict[str, Any], capacity_multiplier: float) -> Any:
    return e22.LocalRadialDistFlow(
        case,
        capacity_multiplier=float(capacity_multiplier),
        transformer_multiplier=float(capacity_multiplier),
    )


def _model_path(dataset: str) -> Path:
    path = SOURCE / "models" / f"{dataset}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing the directly trained IEEE-69 model: {path}")
    return path


def _algorithm_config(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["control"] = dict(value["control"])
    if algorithm == "eps_ieee69_fused":
        value["control"]["eps_control_mode"] = "fused"
    return value


def _context(
    dataset: str,
    seed_index: int,
    axis: str,
    pressure_value: float,
    cases: dict[str, dict[str, Any]],
) -> tuple[list[Any], dict[str, Any], np.ndarray, Any, dict[str, Any]]:
    base_seed = e22._base_seed(dataset, seed_index)
    base_records, config, _ = e22._sample_dataset(dataset, base_seed)
    availability = training.availability_probability(
        base_records, config, e22.mixed.AVAILABILITY_MODE
    )
    case = cases[TOPOLOGY]
    placement_seed = base_seed + 10_000 * (e22.TOPOLOGIES.index(TOPOLOGY) + 1)
    reference_records, _ = e22._remap_records(
        base_records, case, placement_seed, "load_weighted"
    )
    if axis == "spatial_concentration":
        records, placement = _custom_concentration_remap(
            base_records, case, placement_seed + int(pressure_value * 10_000), pressure_value
        )
        capacity = 1.0
        request_scale = 1.0
    elif axis == "line_derating":
        records, placement = e22._remap_records(
            base_records, case, placement_seed, "load_weighted"
        )
        capacity = 1.0 - pressure_value
        request_scale = 1.0
    else:
        records, placement = e22._remap_records(
            base_records, case, placement_seed, "load_weighted"
        )
        capacity = 1.0
        request_scale = pressure_value
    scenario = e22._build_scenario(
        records,
        reference_records,
        case,
        TOPOLOGY,
        "E23_BASE",
        base_seed + int(1_000_000 * pressure_value) + _stable_seed(axis),
        placement,
    )
    scenario["request_scale"] = request_scale
    network = _network(case, capacity)
    return records, config, availability, network, scenario


def _run_dataset(dataset: str, seed_count: int, workers_label: str = "") -> Path:
    _base_stress_settings()
    cases = e22._network_cases()
    model_path = _model_path(dataset)
    _, optimizer, _ = direct.training.load_frozen_eps_controller(model_path)
    output_dir = OUTPUT / "data/raw" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_seed_rows = sum(len(value["values"]) for value in AXES.values()) * len(ALGORITHMS)
    for seed_index in range(seed_count):
        output_path = output_dir / f"seed_{seed_index:02d}.csv"
        if output_path.exists():
            existing = _read_rows(output_path)
            if (
                len(existing) == expected_seed_rows
                and all(row.get("protocol") == PROTOCOL for row in existing)
            ):
                continue
            output_path.unlink()
        rows: list[dict[str, Any]] = []
        base_seed = e22._base_seed(dataset, seed_index)
        for axis, settings in AXES.items():
            for pressure_index, pressure_value in enumerate(settings["values"]):
                records, config, availability, network, scenario = _context(
                    dataset, seed_index, axis, pressure_value, cases
                )
                for algorithm_index, algorithm in enumerate(ALGORITHMS):
                    result, _ = e22._run_seed(
                        algorithm,
                        records,
                        _algorithm_config(config, algorithm),
                        scenario,
                        availability,
                        network,
                        base_seed + 1_800_000 + algorithm_index * 10_000,
                        base_seed + 1_910_000,
                        optimizer if algorithm == "eps_ieee69_fused" else None,
                        False,
                    )
                    rows.append(
                        {
                            "protocol": PROTOCOL,
                            "dataset": dataset,
                            "dataset_label": e22.DATASET_LABELS[dataset],
                            "seed_index": seed_index,
                            "seed": base_seed,
                            "axis": axis,
                            "pressure_index": pressure_index,
                            "pressure_value": pressure_value,
                            "request_scale": scenario["request_scale"],
                            "line_capacity_multiplier": 1.0 - pressure_value if axis == "line_derating" else 1.0,
                            "spatial_concentration": pressure_value if axis == "spatial_concentration" else 0.0,
                            "topology": TOPOLOGY,
                            "fleet_size": len(records),
                            "algorithm": algorithm,
                            "algorithm_label": e22.mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                            "complexity": e22.mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"],
                            "placement_mode": scenario["placement"]["placement_mode"],
                            "configured_concentration_fraction": scenario["placement"]["configured_concentration_fraction"],
                            "target_branch": scenario["placement"]["target_branch"],
                            "target_bus": scenario["placement"]["target_bus"],
                            "actual_target_feeder_fraction": scenario["placement"]["actual_target_feeder_fraction"],
                            "actual_target_bus_fraction": scenario["placement"]["actual_target_bus_fraction"],
                            **result,
                        }
                    )
        _write_rows(output_path, rows)
        print(
            json.dumps(
                {"dataset": dataset, "completed_seeds": seed_index + 1, "seed_count": seed_count, "workers": workers_label},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return output_dir


def _merge_raw(seed_count: int) -> list[dict[str, str]]:
    paths = sorted((OUTPUT / "data/raw").glob("*/seed_*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_rows(path))
    expected = len(e22.E22_DATASETS) * seed_count * sum(len(value["values"]) for value in AXES.values()) * len(ALGORITHMS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete per-seed rows: {len(rows)}, expected {expected}")
    _write_rows(OUTPUT / "data/e23_by_seed.csv", rows)
    return rows


def _bootstrap(
    values: np.ndarray,
    draws: int | None = None,
    seed: int = 20260819,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = BOOTSTRAP_DRAWS if draws is None else draws
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _aggregate(rows: list[dict[str, str]], seed_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, float, str, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["axis"], float(row["pressure_value"]), row["dataset"], int(row["seed_index"]))
        groups.setdefault(key, {})[row["algorithm"]] = row
    summary: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for axis in AXES:
        for pressure_value in AXES[axis]["values"]:
            selected = [
                (key, value)
                for key, value in groups.items()
                if key[0] == axis and abs(key[1] - pressure_value) < 1e-9
            ]
            for algorithm in ALGORITHMS:
                values = np.asarray([
                    float(value[algorithm]["mean_reduction_pct"])
                    for _, value in selected
                ])
                effect, ci_low, ci_high = _bootstrap(values, seed=_stable_seed(f"effect|{axis}|{pressure_value}|{algorithm}"))
                acceptance = np.asarray([
                    float(value[algorithm]["network_acceptance_ratio"])
                    for _, value in selected
                    if float(value[algorithm]["request_scale"]) > 0.0 and algorithm != "no_coordination"
                ])
                violation = np.asarray([
                    float(value[algorithm]["requested_added_violation_steps"]) / STEPS
                    for _, value in selected
                ])
                acc_mean, acc_low, acc_high = (float("nan"), float("nan"), float("nan"))
                if len(acceptance):
                    acc_mean, acc_low, acc_high = _bootstrap(
                        acceptance,
                        seed=_stable_seed(f"accept|{axis}|{pressure_value}|{algorithm}"),
                    )
                v_mean, v_low, v_high = _bootstrap(
                    violation,
                    seed=_stable_seed(f"violation|{axis}|{pressure_value}|{algorithm}"),
                )
                first = selected[0][1][algorithm]
                summary.append(
                    {
                        "axis": axis,
                        "pressure_value": pressure_value,
                        "algorithm": algorithm,
                        "algorithm_label": first["algorithm_label"],
                        "complexity": first["complexity"],
                        "effect_mean_pct": effect,
                        "effect_ci_low_pct": ci_low,
                        "effect_ci_high_pct": ci_high,
                        "network_acceptance_mean_pct": 100.0 * acc_mean if np.isfinite(acc_mean) else "NA",
                        "network_acceptance_ci_low_pct": 100.0 * acc_low if np.isfinite(acc_low) else "NA",
                        "network_acceptance_ci_high_pct": 100.0 * acc_high if np.isfinite(acc_high) else "NA",
                        "requested_added_violation_mean_pct": 100.0 * v_mean,
                        "requested_added_violation_ci_low_pct": 100.0 * v_low,
                        "requested_added_violation_ci_high_pct": 100.0 * v_high,
                        "dataset_seed_count": len(selected),
                        "seed_count": seed_count,
                    }
                )
            for key, value in selected:
                eps = float(value["eps_ieee69_fused"]["mean_reduction_pct"])
                best_all_algorithm = max(BASELINES, key=lambda name: float(value[name]["mean_reduction_pct"]))
                best_all = float(value[best_all_algorithm]["mean_reduction_pct"])
                o1 = float(value[O1_BASELINE]["mean_reduction_pct"])
                comparisons.append(
                    {
                        "axis": axis,
                        "pressure_value": pressure_value,
                        "dataset": key[2],
                        "seed_index": key[3],
                        "eps_effect_pct": eps,
                        "best_baseline_algorithm": best_all_algorithm,
                        "best_baseline_effect_pct": best_all,
                        "best_o1_effect_pct": o1,
                        "delta_all_pct": eps - best_all,
                        "delta_o1_pct": eps - o1,
                        "eps_requested_added_violation_pct": 100.0 * float(value["eps_ieee69_fused"]["requested_added_violation_steps"]) / STEPS,
                    }
                )
    _write_rows(OUTPUT / "data/e23_summary.csv", summary)
    _write_rows(OUTPUT / "data/e23_paired_comparisons.csv", comparisons)
    return summary, comparisons


def _comparison_curve(comparisons: list[dict[str, Any]], axis: str, field: str) -> list[dict[str, Any]]:
    values = AXES[axis]["values"]
    result = []
    for pressure_value in values:
        selected = np.asarray([
            float(row[field])
            for row in comparisons
            if row["axis"] == axis and abs(float(row["pressure_value"]) - pressure_value) < 1e-9
        ])
        mean, low, high = _bootstrap(selected, seed=_stable_seed(f"delta|{axis}|{pressure_value}|{field}"))
        result.append({"axis": axis, "pressure_value": pressure_value, "mean": mean, "ci_low": low, "ci_high": high})
    return result


def _boundary(curve: list[dict[str, Any]], threshold: float = 0.0) -> float | None:
    for index in range(len(curve) - 1):
        boundary_start = float(AXES[curve[index]["axis"]]["boundary_start"])
        if float(curve[index]["pressure_value"]) < boundary_start:
            continue
        if curve[index]["ci_high"] < threshold and curve[index + 1]["ci_high"] < threshold:
            return float(curve[index]["pressure_value"])
    return None


def _physical_boundary(summary: list[dict[str, Any]], axis: str) -> float | None:
    curve = [
        row
        for row in summary
        if row["axis"] == axis and row["algorithm"] == "eps_ieee69_fused"
    ]
    curve.sort(key=lambda row: float(row["pressure_value"]))
    for index in range(len(curve) - 1):
        if (
            float(curve[index]["pressure_value"]) >= float(AXES[axis]["boundary_start"])
            and float(curve[index]["requested_added_violation_ci_low_pct"]) > 5.0
            and float(curve[index + 1]["requested_added_violation_ci_low_pct"]) > 5.0
        ):
            return float(curve[index]["pressure_value"])
    return None


def _write_boundaries(summary: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis in AXES:
        all_curve = _comparison_curve(comparisons, axis, "delta_all_pct")
        o1_curve = _comparison_curve(comparisons, axis, "delta_o1_pct")
        rows.extend([
            {"axis": axis, "boundary_type": "relative_to_all_baselines", "threshold": "0 pp", "boundary_value": _boundary(all_curve, 0.0)},
            {"axis": axis, "boundary_type": "relative_to_all_baselines_meaningful", "threshold": "-2 pp", "boundary_value": _boundary(all_curve, -2.0)},
            {"axis": axis, "boundary_type": "relative_to_O1_baseline", "threshold": "0 pp", "boundary_value": _boundary(o1_curve, 0.0)},
            {"axis": axis, "boundary_type": "relative_to_O1_baseline_meaningful", "threshold": "-2 pp", "boundary_value": _boundary(o1_curve, -2.0)},
            {"axis": axis, "boundary_type": "physical_added_violation", "threshold": "5% lower CI", "boundary_value": _physical_boundary(summary, axis)},
        ])
    _write_rows(OUTPUT / "data/e23_boundaries.csv", rows)
    return rows


def _plot_effect_curves(summary: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.6), sharey=True)
    labels = {row["algorithm"]: row["algorithm_label"] for row in summary}
    for axis_object, axis_name in zip(axes, AXES):
        for algorithm in ALGORITHMS:
            rows = sorted([row for row in summary if row["axis"] == axis_name and row["algorithm"] == algorithm], key=lambda row: float(row["pressure_value"]))
            axis_object.plot(
                [float(row["pressure_value"]) for row in rows],
                [float(row["effect_mean_pct"]) for row in rows],
                marker="o", markersize=3.5, linewidth=1.7,
                color=COLORS[algorithm],
                label=f"{labels[algorithm]} ({rows[0]['complexity']})",
            )
        axis_object.set_title(AXES[axis_name]["label"])
        axis_object.set_xlabel(AXES[axis_name]["unit"])
        axis_object.set_ylim(-2, 105)
        axis_object.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Deliverable curtailment reduction (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.suptitle("Delivered effect under continuous IEEE-69 stress", y=1.02, fontsize=16)
    figure.tight_layout()
    _save_figure(figure, "e23_relative_effect_curves")


def _plot_advantage(comparisons: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True)
    for axis_object, axis_name in zip(axes, AXES):
        for field, label, color in (("delta_all_pct", "EPS - best baseline", "#1f5aa6"), ("delta_o1_pct", "EPS - Local SOC rules", "#d55e00")):
            curve = _comparison_curve(comparisons, axis_name, field)
            axis_object.plot([row["pressure_value"] for row in curve], [row["mean"] for row in curve], marker="o", color=color, label=label)
            axis_object.fill_between([row["pressure_value"] for row in curve], [row["ci_low"] for row in curve], [row["ci_high"] for row in curve], color=color, alpha=0.12)
        axis_object.axhline(0.0, color="black", linewidth=1)
        axis_object.axhline(-2.0, color="#d1495b", linestyle="--", linewidth=1, label="-2 pp")
        axis_object.set_title(AXES[axis_name]["label"])
        axis_object.set_xlabel(AXES[axis_name]["unit"])
        axis_object.grid(alpha=0.22)
    axes[0].set_ylabel("Relative effect of EPS (percentage points)\nband: 95% bootstrap interval")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.suptitle("Where EPS falls behind the other executable methods", y=1.02, fontsize=16)
    figure.tight_layout()
    _save_figure(figure, "e23_relative_advantage_curves")


def _plot_physics(summary: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True)
    for axis_object, axis_name in zip(axes, AXES):
        for algorithm in ALGORITHMS:
            rows = sorted([row for row in summary if row["axis"] == axis_name and row["algorithm"] == algorithm], key=lambda row: float(row["pressure_value"]))
            axis_object.plot(
                [float(row["pressure_value"]) for row in rows],
                [float(row["requested_added_violation_mean_pct"]) for row in rows],
                marker="o",
                markersize=3.2,
                linewidth=1.5,
                color=COLORS[algorithm],
                label=f"{rows[0]['algorithm_label']} ({rows[0]['complexity']})",
            )
        axis_object.axhline(5.0, color="#d1495b", linestyle="--", linewidth=1.2, label="5% risk reference")
        axis_object.set_title(AXES[axis_name]["label"])
        axis_object.set_xlabel(AXES[axis_name]["unit"])
        axis_object.grid(alpha=0.22)
    axes[0].set_ylabel("Added physical violations before safety clipping (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.suptitle("Physical constraint risk beyond the relative performance boundary", y=1.02, fontsize=16)
    figure.tight_layout()
    _save_figure(figure, "e23_physical_violation_curves")


def _plot_boundary_summary(boundaries: list[dict[str, Any]]) -> None:
    types = ["relative_to_all_baselines", "relative_to_O1_baseline", "physical_added_violation"]
    labels = {types[0]: "below the best baseline", types[1]: "below the O(1) baseline", types[2]: "added physical violations > 5%"}
    figure, axis = plt.subplots(figsize=(12.0, 5.8))
    x = np.arange(len(AXES))
    width = 0.24
    for index, boundary_type in enumerate(types):
        values = []
        for axis_name in AXES:
            row = next(row for row in boundaries if row["axis"] == axis_name and row["boundary_type"] == boundary_type)
            values.append(np.nan if row["boundary_value"] is None else float(row["boundary_value"]))
        axis.bar(x + (index - 1) * width, values, width, label=labels[boundary_type])
    axis.set_xticks(x, [AXES[name]["label"] for name in AXES])
    axis.set_ylabel("Stress value at which the boundary is first reached")
    axis.set_title("Relative performance and physical risk boundaries")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, "e23_boundary_summary")


def _plot_dataset_boundaries(comparisons: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), sharey=True)
    datasets = list(e22.E22_DATASETS)
    for axis_object, axis_name in zip(axes, AXES):
        all_values = []
        o1_values = []
        for dataset in datasets:
            dataset_rows = [row for row in comparisons if row["axis"] == axis_name and row["dataset"] == dataset]
            def first_boundary(field: str) -> float:
                curve = []
                for pressure_value in AXES[axis_name]["values"]:
                    selected = np.asarray([
                        float(row[field])
                        for row in dataset_rows
                        if abs(float(row["pressure_value"]) - pressure_value) < 1e-9
                    ])
                    _, _, high = _bootstrap(
                        selected,
                        seed=_stable_seed(
                            f"dataset-boundary|{dataset}|{axis_name}|{pressure_value}|{field}"
                        ),
                    )
                    curve.append(
                        {
                            "axis": axis_name,
                            "pressure_value": pressure_value,
                            "ci_high": high,
                        }
                    )
                boundary = _boundary(curve)
                return np.nan if boundary is None else boundary
            all_values.append(first_boundary("delta_all_pct"))
            o1_values.append(first_boundary("delta_o1_pct"))
        positions = np.arange(len(datasets))
        axis_object.scatter(all_values, positions, label="below the best baseline", color="#1f5aa6", s=22)
        axis_object.scatter(o1_values, positions, label="below the O(1) baseline", color="#d55e00", s=22)
        axis_object.set_title(AXES[axis_name]["label"])
        axis_object.set_xlabel(AXES[axis_name]["unit"])
        axis_object.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(np.arange(len(datasets)), [e22.DATASET_LABELS[name] for name in datasets])
    axes[0].set_ylabel("Dataset")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.suptitle("Relative performance boundary per dataset", y=1.02, fontsize=16)
    figure.tight_layout()
    _save_figure(figure, "e23_dataset_boundary_distribution")


def _save_figure(figure: Any, name: str) -> None:
    path = OUTPUT / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _copy_models() -> None:
    model_output = OUTPUT / "models"
    model_output.mkdir(parents=True, exist_ok=True)
    for path in (SOURCE / "models").glob("*.pt"):
        shutil.copy2(path, model_output / path.name)
    for path in (SOURCE / "models").glob("*.json"):
        shutil.copy2(path, model_output / path.name)
    metadata = json.loads((SOURCE / "data/training_metadata.json").read_text(encoding="utf-8"))
    for row in metadata:
        dataset = row.get("dataset", "")
        row["model_path"] = f"results/E23/models/{dataset}.pt"
    (OUTPUT / "data/training_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "data/training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    global BOOTSTRAP_DRAWS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    BOOTSTRAP_DRAWS = max(100, int(args.bootstrap_draws))
    if args.fresh and OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _copy_models()
    _base_stress_settings()
    completed: list[Path] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {
            executor.submit(_run_dataset, dataset, args.seed_count, f"{index + 1}/{len(e22.E22_DATASETS)}"): dataset
            for index, dataset in enumerate(e22.E22_DATASETS)
        }
        for future in as_completed(jobs):
            path = future.result()
            completed.append(path)
            print(json.dumps({"stage": "dataset_complete", "dataset": jobs[future], "completed": len(completed), "total": len(jobs)}, ensure_ascii=False), flush=True)
    raw = _merge_raw(args.seed_count)
    summary, comparisons = _aggregate(raw, args.seed_count)
    boundaries = _write_boundaries(summary, comparisons)
    _plot_effect_curves(summary, comparisons)
    _plot_advantage(comparisons)
    _plot_physics(summary)
    _plot_boundary_summary(boundaries)
    _plot_dataset_boundaries(comparisons)
    if not args.keep_raw:
        shutil.rmtree(OUTPUT / "data/raw")
    print(json.dumps({"stage": "complete", "rows": len(raw), "summary_rows": len(summary), "comparison_rows": len(comparisons), "figures": 5}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
