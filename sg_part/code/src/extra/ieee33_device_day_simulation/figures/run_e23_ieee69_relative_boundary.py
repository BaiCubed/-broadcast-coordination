"""运行E23：IEEE-69连续压力扫描下的相对性能与物理边界实验。"""

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
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches as PptInches

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
        "label": "广播请求强度 q",
        "values": REQUEST_SCALES,
        "unit": "q；1.0为正常请求强度",
        "boundary_start": 1.0,
    },
    "line_derating": {
        "label": "线路容量压力 λ_line",
        "values": LINE_PRESSURES,
        "unit": "λ_line；0为额定容量，0.4为容量降至60%",
        "boundary_start": 0.0,
    },
    "spatial_concentration": {
        "label": "空间集中比例 κ",
        "values": CONCENTRATIONS,
        "unit": "κ；0为均匀部署，0.8为80%集中在远端节点",
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
        raise ValueError(f"没有数据：{path}")
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
        "plain_label": "E23连续压力基准",
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
        raise FileNotFoundError(f"缺少IEEE-69直接训练模型：{path}")
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
        raise RuntimeError(f"E23逐seed行数不完整：{len(rows)}，期望{expected}")
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
    axes[0].set_ylabel("可交付弃电降低率 (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.tight_layout()
    _save_figure(figure, "e23_relative_effect_curves")


def _plot_advantage(comparisons: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True)
    for axis_object, axis_name in zip(axes, AXES):
        for field, label, color in (("delta_all_pct", "EPS - 最佳baseline", "#1f5aa6"), ("delta_o1_pct", "EPS - Local SOC rules", "#d55e00")):
            curve = _comparison_curve(comparisons, axis_name, field)
            axis_object.plot([row["pressure_value"] for row in curve], [row["mean"] for row in curve], marker="o", color=color, label=label)
            axis_object.fill_between([row["pressure_value"] for row in curve], [row["ci_low"] for row in curve], [row["ci_high"] for row in curve], color=color, alpha=0.12)
        axis_object.axhline(0.0, color="black", linewidth=1)
        axis_object.axhline(-2.0, color="#d1495b", linestyle="--", linewidth=1, label="-2 pp")
        axis_object.set_title(AXES[axis_name]["label"])
        axis_object.set_xlabel(AXES[axis_name]["unit"])
        axis_object.grid(alpha=0.22)
    axes[0].set_ylabel("EPS相对效果差值 (百分点)\n阴影为95% bootstrap区间")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
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
        axis_object.axhline(5.0, color="#d1495b", linestyle="--", linewidth=1.2, label="5%风险参考线")
        axis_object.set_title(AXES[axis_name]["label"])
        axis_object.set_xlabel(AXES[axis_name]["unit"])
        axis_object.grid(alpha=0.22)
    axes[0].set_ylabel("安全裁剪前新增物理违规请求比例 (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.tight_layout()
    _save_figure(figure, "e23_physical_violation_curves")


def _plot_boundary_summary(boundaries: list[dict[str, Any]]) -> None:
    types = ["relative_to_all_baselines", "relative_to_O1_baseline", "physical_added_violation"]
    labels = {types[0]: "低于最佳baseline", types[1]: "低于O(1) baseline", types[2]: "新增物理违规>5%"}
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
    axis.set_ylabel("首次达到边界的压力参数")
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
        axis_object.scatter(all_values, positions, label="低于最佳baseline", color="#1f5aa6", s=22)
        axis_object.scatter(o1_values, positions, label="低于O(1) baseline", color="#d55e00", s=22)
        axis_object.set_title(AXES[axis_name]["label"])
        axis_object.set_xlabel(AXES[axis_name]["unit"])
        axis_object.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(np.arange(len(datasets)), [e22.DATASET_LABELS[name] for name in datasets])
    axes[0].set_ylabel("数据集")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.tight_layout()
    _save_figure(figure, "e23_dataset_boundary_distribution")


def _save_figure(figure: Any, name: str) -> None:
    path = OUTPUT / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_documents(boundaries: list[dict[str, Any]], seed_count: int) -> None:
    descriptions = [
        ("e23_relative_effect_curves", "连续压力下各算法实际效果", """### 这张图回答什么
它回答：在IEEE-69中逐步增加请求强度、线路容量压力或设备空间集中时，算法最终还能交付多少弃电消纳，而不是只看算法发出了多少请求。

### 横纵坐标
- 左图横坐标是广播请求强度`q`：`q=1`为正常请求，`q>1`表示更强的同时响应需求。
- 中图横坐标是线路容量压力`λ_line`：0表示额定容量，0.4表示线路和主变容量降到60%。
- 右图横坐标是空间集中比例`κ`：0表示均匀部署，0.8表示80%的设备集中到远端节点。
- 三个面板的纵坐标都是可交付弃电降低率(%)，即通过设备约束和网络安全层后实际吸收的弃电比例。

### 图中元素
- 每条有圆点的彩色曲线是一种算法；圆点是该压力点上17个数据集、30个paired seeds的平均值。
- 灰线是No coordination，表示没有控制时的零效果；绿色线是Centralized greedy UB，只是设备层理论上界，不代表网络可执行。
- 深蓝线是IEEE-69直接训练的EPS fused；其余彩色线是Local SOC rules、MPC、Mean-field control、Virtual battery、PEM和Transactive control。
- 图例括号中的`O(1)`、`O(N)`等是在线复杂度，表示每个时间步控制决策的计算规模；不改变纵轴指标。

### 结果和含义
请求强度增大时，EPS在中等请求下保持较高效果，但强请求会受到网络裁剪；设备集中时EPS和其他可执行算法都明显下降，说明设备数量大并不能消除局部馈线瓶颈。该图证明的是IEEE-69下的经验性能曲线和压力趋势，不是对所有电网都成立的数学普适边界。"""),
        ("e23_relative_advantage_curves", "EPS相对性能差值", """### 这张图回答什么
它把EPS和竞争算法的差异直接画出来，回答EPS在什么压力范围内仍然有竞争力，以及何时可以认为它在效果上已经明显落后。

### 横纵坐标
- 三个面板的横坐标与图1相同，分别是`q`、`λ_line`和`κ`。
- 纵坐标是效果差值(百分点)：EPS效果减去最佳baseline，或EPS效果减去同为低通信复杂度的Local SOC rules。
- 0线表示两者效果相同；-2 pp虚线表示预先声明的实际意义落后阈值。

### 图中元素
- 蓝色实线和浅蓝阴影表示`EPS - 最佳baseline`及其95% paired bootstrap区间。
- 橙色实线和浅橙阴影表示`EPS - Local SOC rules`及其95% paired bootstrap区间。
- 阴影的重采样单位是同一dataset-seed的配对观测，因此比较的是相同输入下的差异，不是两个独立样本的差异。

### 结果和含义
曲线在0线上方表示EPS不低于对应参照，在0线下方表示EPS落后；只有区间上界持续低于0，才支持统计意义上的落后，低于-2 pp才同时满足实际意义判据。该图把“看起来下降”与“相对baseline确实失去优势”区分开。"""),
        ("e23_physical_violation_curves", "物理约束风险曲线", """### 这张图回答什么
它回答：算法的原始广播请求在经过安全裁剪前，是否会因为线路、节点电压或主变约束而制造额外风险。这样可以把“请求很强”与“请求本身物理上不安全”分开。

### 横纵坐标
- 三个面板的横坐标分别是`q`、`λ_line`和`κ`，定义与图1相同。
- 纵坐标是安全裁剪前新增物理违规请求比例(%)。它只统计相对No coordination基线新增的线路过载、电压越限或主变过载时间比例。
- 5%水平虚线是预先声明的工程审计参考线，不是数学定理或电网法规限值。

### 图中元素
- 每条彩色曲线对应一种算法，圆点是17个数据集和30个seed的平均风险。
- No coordination通常为0，因为它没有新增控制请求；Centralized greedy UB仍是设备层上界，不能解读为安全控制器。
- 结果是安全层介入前的请求风险；安全层介入后的实际执行由图1的可交付效果体现。

### 结果和含义
曲线超过5%并在相邻压力点持续存在时，表示该算法的请求需要安全层频繁修正，不能把其原始请求称为天然安全。该图为Reviewer #4提出的馈线容量、电压、主变和空间集中问题提供风险证据，但仍不替代完整三相潮流和现场验证。"""),
        ("e23_boundary_summary", "相对性能与物理风险边界", """### 这张图回答什么
它把前面三条压力轴上的判定结果压缩成一张边界摘要，回答“首次在哪个压力点，EPS连续两个点都不再满足竞争或风险标准”。

### 横纵坐标
- 横坐标是三个压力轴：请求强度`q`、线路容量压力`λ_line`和空间集中比例`κ`。
- 纵坐标是首次达到边界的压力参数，而不是弃电降低率百分比。

### 图中元素
- 蓝柱：EPS相对最佳baseline的95%区间上界连续两个点低于0。
- 橙柱：EPS相对Local SOC rules的同一统计边界。
- 绿色柱：EPS新增物理违规比例的95%区间下界连续两个点超过5%。
- 空柱表示在该压力轴的扫描范围内没有观察到对应边界；柱值为0表示边界从该轴的第一个扫描点就满足，不能解释为“零压力失效”。

### 结果和含义
它给出可复核的经验边界位置，并区分性能边界与物理风险边界。边界是本实验扫描范围和判据下的结果，不是由理论预先指定的固定常数。"""),
        ("e23_dataset_boundary_distribution", "不同数据集的相对性能边界", """### 这张图回答什么
它回答：总体平均得到的边界是否适用于每个数据集，还是会掩盖不同负荷类型和时间序列的差异。

### 横纵坐标
- 三个面板的横坐标分别是`q`、`λ_line`和`κ`，单位与图1相同。
- 纵坐标是数据集名称；每个点是该数据集单独计算的首次连续两点边界。

### 图中元素
- 蓝点表示该数据集相对最佳baseline的边界。
- 橙点表示该数据集相对Local SOC rules的边界。
- 空白点或缺失值表示扫描范围内没有满足连续两点判据，不表示数据集没有结果。

### 结果和含义
点的位置分散说明统一报告一个平均边界会隐藏数据集间的敏感性。该图用于报告EPS有效范围的异质性，而不是把所有数据集强行声称为同一个失效压力。"""),
    ]
    lines = [
        "# E23图片说明：IEEE-69连续压力相对性能边界",
        "",
        "E23只保留IEEE-69网络下的直接训练EPS fused和原有baseline，不包含迁移模型。每个压力点使用相同dataset、seed和设备样本进行配对比较。",
        "",
        "## 边界判定",
        "",
        "- 全baseline边界：EPS相对最佳可执行baseline的差值95% bootstrap区间上界连续两个压力点小于0。",
        "- O(1)边界：EPS相对Local SOC rules的差值95% bootstrap区间上界连续两个压力点小于0。",
        "- 实际意义边界：上述差值区间上界连续两个压力点小于-2个百分点。",
        "- 物理风险边界：EPS安全裁剪前的请求相对于无控制基线新增物理违规比例，其95% bootstrap区间下界连续两个压力点高于5%。",
        "",
    ]
    for index, (name, title, axes) in enumerate(descriptions, start=1):
        lines.extend([
            f"## {index}. {title}", "", f"![{title}](figures/{name}.png)", "",
            "文件：", "", f"- PNG：`figures/{name}.png`", f"- PDF：`figures/{name}.pdf`",
            "- 数据：`data/e23_by_seed.csv`、`data/e23_summary.csv`、`data/e23_paired_comparisons.csv`、`data/e23_boundaries.csv`",
            "- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/run_e23_ieee69_relative_boundary.py`", "",
            axes, "",
        ])
    (OUTPUT / "FIGURE_DESCRIPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    _write_docx(descriptions)
    _write_pptx(descriptions)
    (OUTPUT / "README.md").write_text(
        """# E23：IEEE-69连续压力相对性能边界

本实验只比较IEEE-69网络下直接训练的EPS fused与8个原有baseline。连续扫描三个独立压力轴：广播请求强度q、线路容量压力λ_line和空间集中比例κ。

边界使用配对dataset-seed bootstrap判定，而不是直接规定一个效果百分比。EPS相对最佳baseline或Local SOC rules的差值在连续两个压力点的95%区间上界小于0，分别定义相对性能边界；EPS安全裁剪前新增物理违规请求比例的95%区间下界连续超过5%，定义物理风险边界。

主要数据：`data/e23_by_seed.csv`、`data/e23_summary.csv`、`data/e23_paired_comparisons.csv`、`data/e23_boundaries.csv`。
""", encoding="utf-8")
    (OUTPUT / "EXPERIMENT_DESIGN.md").write_text(
        f"""# E23实验设计：IEEE-69连续压力相对性能边界

## 研究问题

在IEEE-69更严格的网络和空间条件下，EPS fused何时开始显著落后于其他可执行算法？该实验不研究迁移模型，只研究IEEE-69直接训练EPS与baseline的公平对比。

## 固定条件

- IEEE-69 radial feeder；17个数据集；5000台设备；{seed_count}个paired seeds。
- 每个dataset-seed在所有算法和压力点复用相同的设备样本、随机种子、输入场景和可用率。
- EPS使用IEEE-69网络约束数据直接训练的fused模型，在线广播复杂度为O(1)。
- Centralized greedy UB保留作设备层理论参考，不进入最佳baseline竞争集合。

## 算法与在线复杂度

|算法|在线复杂度|实验角色|
|-|-|-|
|No coordination|O(0)|不执行控制，仅作为效果和物理违规基准。|
|Local SOC rules|O(1) per device|只使用设备本地SOC和时段规则，也是EPS的同量级参照。|
|MPC|O(HN)|使用H步预测进行滚动分配。|
|Mean-field control|O(N)|按群体SOC分布计算统一参与概率。|
|Virtual battery|O(N)|将设备群聚合为等效电池后分配。|
|Packetized Energy Management|O(N log N)|按不可拆分功率包排序和接纳。|
|Transactive control|O(N log N)|按设备报价排序并进行价格清算。|
|EPS IEEE-69 fused|O(1)|使用IEEE-69直接训练模型、解析公式和闭环修正生成广播强度。|
|Centralized greedy UB|O(N)|计算设备层理论最大消纳，仅作上界参考，不进入可执行baseline竞争。|

## 三个连续压力轴

|压力轴|扫描范围|含义|
|-|-|-|
|请求强度q|0.0到1.5，步长0.1|将算法请求强度乘以q；q=1为正常请求，q>1表示更强的同时响应需求。|
|线路容量压力λ_line|0.0到0.4，步长0.05|线路和主变容量乘以1-λ_line；λ_line=0.4表示容量降至60%。|
|空间集中κ|0.0到0.8，步长0.1|κ比例的设备放到IEEE-69远端节点，其余设备均匀放置。|

三个轴分别扫描，其余条件保持基准值，不把M0-M6混成一个非连续压力轴。
请求强度轴的边界只从q=1.0开始向更高请求搜索；q<1表示降载响应，不属于“压力增加后失效”的方向。

## 主要指标

实际效果为经过设备和网络约束后的可交付弃电降低率：

`R_a = 100 × (原始弃电 - 执行后剩余弃电) / 原始弃电`

全baseline比较：`Δ_all = R_EPS - max(R_MPC, R_mean-field, R_virtual-battery, R_PEM, R_transactive, R_local-SOC)`。

同复杂度比较：`Δ_O1 = R_EPS - R_Local-SOC`。由于当前实验中EPS以外的O(1)控制baseline是Local SOC rules，因此采用它作为O(1)参照。

物理风险指标为安全裁剪前的控制请求相对无控制基线新增线路、电压或主变违规的时间比例，避免把基础负荷本身的违规误记为控制造成的违规。执行后的效果指标仍使用经过安全层裁剪的实际可交付结果。

## 边界判定

- 统计相对边界：`upper_CI95(Δ) < 0`，并要求连续两个压力点成立。
- 实际意义边界：`upper_CI95(Δ) < -2`个百分点，并要求连续两个压力点成立。
- 物理风险边界：EPS新增物理违规比例的`lower_CI95 > 5%`，并要求连续两个压力点成立。

bootstrap的重采样单位是同一压力点下的`dataset-seed`配对观测，共{BOOTSTRAP_DRAWS}次。0和-2个百分点分别表示统计上开始落后和具有实际意义的落后；5%是预先声明的风险审计标准，不是数学定理。

## 输出

- `data/e23_by_seed.csv`：全部逐dataset-seed-algorithm-pressure原始结果。
- `data/e23_summary.csv`：均值和95% bootstrap区间。
- `data/e23_paired_comparisons.csv`：EPS与最佳baseline、O(1) baseline的配对差值。
- `data/e23_boundaries.csv`：三条压力轴上的边界汇总。
- `figures/`：效果、相对优势、物理风险、边界汇总和逐数据集边界图。
""", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({"protocol": PROTOCOL, "topology": TOPOLOGY, "dataset_count": len(e22.E22_DATASETS), "seed_count": seed_count, "algorithms": list(ALGORITHMS), "axes": {key: list(value["values"]) for key, value in AXES.items()}, "bootstrap_draws": BOOTSTRAP_DRAWS, "migration_comparison": False}, indent=2), encoding="utf-8")


def _write_docx(descriptions: list[tuple[str, str, str]]) -> None:
    document = Document()
    title = document.add_heading("E23：IEEE-69连续压力相对性能边界", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph("每页对应一张最终图片。实验只比较IEEE-69直接训练EPS fused与原有baseline。")
    for index, (name, title, body) in enumerate(descriptions, start=1):
        document.add_heading(f"{index}. {title}", level=1)
        document.add_picture(str(OUTPUT / "figures" / f"{name}.png"), width=Inches(6.7))
        for paragraph in body.split("\n\n"):
            document.add_paragraph(paragraph.replace("### ", ""))
    document.save(OUTPUT / "FIGURE_DESCRIPTIONS.docx")


def _write_pptx(descriptions: list[tuple[str, str, str]]) -> None:
    presentation = Presentation()
    presentation.slide_width = PptInches(13.333)
    presentation.slide_height = PptInches(7.5)
    presentation.core_properties.title = "E23 IEEE-69连续压力相对性能边界"
    for index, (name, title, body) in enumerate(descriptions, start=1):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, PptInches(0), PptInches(0), PptInches(13.333), PptInches(0.7))
        header.fill.solid(); header.fill.fore_color.rgb = RGBColor(31, 50, 72); header.line.fill.background()
        text_box = slide.shapes.add_textbox(PptInches(0.45), PptInches(0.12), PptInches(12.0), PptInches(0.4))
        paragraph = text_box.text_frame.paragraphs[0]; paragraph.text = title; paragraph.font.size = Pt(20); paragraph.font.color.rgb = RGBColor(255, 255, 255)
        image_path = OUTPUT / "figures" / f"{name}.png"
        with Image.open(image_path) as image:
            ratio = image.width / image.height
        area_w, area_h = 12.4, 5.75
        if ratio >= area_w / area_h:
            image_w, image_h = area_w, area_w / ratio; left, top = 0.45, 0.88 + (area_h - image_h) / 2
        else:
            image_h, image_w = area_h, area_h * ratio; left, top = 0.45 + (area_w - image_w) / 2, 0.88
        slide.shapes.add_picture(str(image_path), PptInches(left), PptInches(top), width=PptInches(image_w), height=PptInches(image_h))
        note = slide.shapes.add_textbox(PptInches(0.55), PptInches(6.78), PptInches(12.0), PptInches(0.45))
        note.text_frame.paragraphs[0].text = f"E23｜{body}｜第{index}/{len(descriptions)}张"
        note.text_frame.paragraphs[0].font.size = Pt(9)
        note.text_frame.paragraphs[0].font.color.rgb = RGBColor(90, 90, 90)
    presentation.save(OUTPUT / "E23_figures.pptx")


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
    _write_documents(boundaries, args.seed_count)
    if not args.keep_raw:
        shutil.rmtree(OUTPUT / "data/raw")
    print(json.dumps({"stage": "complete", "rows": len(raw), "summary_rows": len(summary), "comparison_rows": len(comparisons), "figures": 5}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
