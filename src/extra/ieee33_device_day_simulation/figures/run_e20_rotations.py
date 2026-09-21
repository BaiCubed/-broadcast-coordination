from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import run_e20_transfer as transfer


ALL_DATASETS = (
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
    "camsl_japan_smart_meters",
    "european_lv_urban_8087",
    "irish_domestic_smart_meters",
    "opsd_household_data",
    "smart_grid_smart_city",
)

FOLD_TARGETS = {
    "fold_01": ALL_DATASETS[10:15],
    "fold_02": ALL_DATASETS[0:5],
    "fold_03": ALL_DATASETS[5:10],
}

FOLD_DIRS = {
    "fold_01": Path("E20"),
    "fold_02": Path("E20/rotations/fold_02"),
    "fold_03": Path("E20/rotations/fold_03"),
}

METHOD_LABELS = {
    "in_domain": "In-domain",
    "zero_shot": "Zero-shot",
    "target_calibrated": "Target calibrated",
}


def _fold_sources(fold_id: str) -> tuple[str, ...]:
    targets = set(FOLD_TARGETS[fold_id])
    return tuple(dataset for dataset in ALL_DATASETS if dataset not in targets)


def _run_fold(
    fold_id: str,
    results_root: Path,
    workers: int,
    force: bool,
    retrain_models: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "src.extra.ieee33_device_day_simulation.figures.run_e20_transfer",
        "--results-root",
        str(results_root),
        "--experiment-dir",
        str(FOLD_DIRS[fold_id]),
        "--split-id",
        fold_id,
        "--source-datasets",
        ",".join(_fold_sources(fold_id)),
        "--target-datasets",
        ",".join(FOLD_TARGETS[fold_id]),
        "--workers",
        str(workers),
    ]
    if force:
        command.append("--force")
    if retrain_models:
        command.append("--retrain-models")
    print(json.dumps({"fold": fold_id, "command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


def _load_fold(results_root: Path, fold_id: str) -> dict[str, Any]:
    path = results_root / FOLD_DIRS[fold_id] / "data" / "e20_transfer_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload["target_datasets"]) != FOLD_TARGETS[fold_id]:
        raise RuntimeError(f"{fold_id}: the target datasets do not match the rotation design")
    if tuple(payload["source_datasets"]) != _fold_sources(fold_id):
        raise RuntimeError(f"{fold_id}: the source datasets do not match the rotation design")
    return payload


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "fold_id",
            "dataset",
            "network_mode",
            "method",
            "mean_reduction_pct",
            "std_reduction_pct",
            "seed_count",
        ])
        for row in rows:
            writer.writerow([
                row["fold_id"],
                row["dataset"],
                row["network_mode"],
                row["method"],
                row["result"]["mean_reduction_pct"],
                row["result"]["std_reduction_pct"],
                row["seed_count"],
            ])


def _plot_transfer_heatmap(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 11), constrained_layout=True)
    for axis, network_mode in zip(axes, transfer.NETWORK_MODES):
        values = np.asarray([
            [
                next(
                    row["result"]["mean_reduction_pct"]
                    for row in rows
                    if row["dataset"] == dataset
                    and row["network_mode"] == network_mode
                    and row["method"] == method
                )
                for method in transfer.METHODS
            ]
            for dataset in ALL_DATASETS
        ])
        image = axis.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        axis.set_title(network_mode)
        axis.set_xticks(
            np.arange(len(transfer.METHODS)),
            [METHOD_LABELS[method] for method in transfer.METHODS],
            rotation=25,
            ha="right",
        )
        axis.set_yticks(
            np.arange(len(ALL_DATASETS)),
            [dataset.replace("_", " ").title() for dataset in ALL_DATASETS],
            fontsize=8,
        )
        for row_index in range(len(ALL_DATASETS)):
            for column_index in range(len(transfer.METHODS)):
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
    fig.colorbar(
        image,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label="Curtailment reduction (%)",
    )
    fig.suptitle("E20 three-fold transfer: every dataset is unseen target once", fontsize=16)
    path = output_dir / "e20_multisplit_transfer_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_transfer_macro(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    x = np.arange(len(transfer.METHODS), dtype=float)
    for axis, network_mode in zip(axes, transfer.NETWORK_MODES):
        values = np.asarray([
            [
                row["result"]["mean_reduction_pct"]
                for row in rows
                if row["network_mode"] == network_mode and row["method"] == method
            ]
            for method in transfer.METHODS
        ])
        means = np.mean(values, axis=1)
        errors = np.std(values, axis=1)
        axis.bar(x, means, yerr=errors, color=colors, capsize=4, alpha=0.9)
        for method_index, method_values in enumerate(values):
            axis.scatter(
                np.full(len(method_values), method_index),
                method_values,
                facecolors="white",
                edgecolors="#202020",
                linewidths=0.7,
                s=24,
                zorder=3,
            )
            axis.text(method_index, means[method_index] + 1.0, f"{means[method_index]:.1f}", ha="center")
        axis.set_xticks(x, [METHOD_LABELS[method] for method in transfer.METHODS])
        axis.set_ylim(0, 110)
        axis.set_ylabel("Curtailment reduction (%)")
        axis.set_title(network_mode)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "E20 three-fold macro transfer across 15 unseen targets\n"
        "bars: dataset mean; error bars: between-dataset SD; dots: individual datasets",
        fontsize=15,
    )
    path = output_dir / "e20_multisplit_transfer_overall.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_algorithm_macro(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    y = np.arange(len(transfer.COMPARISON_ORDER), dtype=float)
    labels = transfer._comparison_tick_labels()
    colors = [
        transfer.COMPARISON_COLORS[algorithm]
        for algorithm in transfer.COMPARISON_ORDER
    ]
    for axis, network_mode in zip(axes, transfer.NETWORK_MODES):
        values = np.asarray([
            [
                row["result"]["mean_reduction_pct"]
                for row in rows
                if row["network_mode"] == network_mode
                and row["algorithm"] == algorithm
            ]
            for algorithm in transfer.COMPARISON_ORDER
        ])
        means = np.mean(values, axis=1)
        errors = np.std(values, axis=1)
        axis.barh(y, means, xerr=errors, color=colors, capsize=3, alpha=0.88)
        for algorithm_index, dataset_values in enumerate(values):
            axis.scatter(
                dataset_values,
                np.full(len(dataset_values), algorithm_index),
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
        "E20 algorithm comparison across all 15 rotated target datasets",
        fontsize=16,
    )
    path = output_dir / "e20_multisplit_algorithm_overall.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _aggregate(results_root: Path) -> None:
    rotation_root = results_root / "E20" / "rotations"
    data_dir = rotation_root / "data"
    figure_dir = rotation_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fold_payloads = {
        fold_id: _load_fold(results_root, fold_id)
        for fold_id in FOLD_TARGETS
    }
    transfer_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for fold_id, payload in fold_payloads.items():
        transfer_rows.extend({**row, "fold_id": fold_id} for row in payload["results"])
        comparison_rows.extend(
            {**row, "fold_id": fold_id}
            for row in payload["algorithm_comparison"]["results"]
        )
    expected_transfer = len(ALL_DATASETS) * len(transfer.NETWORK_MODES) * len(transfer.METHODS)
    expected_comparison = len(ALL_DATASETS) * len(transfer.NETWORK_MODES) * len(transfer.COMPARISON_ORDER)
    if len(transfer_rows) != expected_transfer:
        raise RuntimeError(f"wrong number of transfer blocks: {len(transfer_rows)} != {expected_transfer}")
    if len(comparison_rows) != expected_comparison:
        raise RuntimeError(f"wrong number of algorithm blocks: {len(comparison_rows)} != {expected_comparison}")
    csv_path = data_dir / "e20_multisplit_transfer_summary.csv"
    _write_csv(transfer_rows, csv_path)
    figures = [
        _plot_transfer_heatmap(transfer_rows, figure_dir),
        _plot_transfer_macro(transfer_rows, figure_dir),
        _plot_algorithm_macro(comparison_rows, figure_dir),
    ]
    payload = {
        "protocol": "E20_three_fold_dataset_rotation_v1",
        "folds": {
            fold_id: {
                "source_datasets": list(_fold_sources(fold_id)),
                "target_datasets": list(FOLD_TARGETS[fold_id]),
                "result": str(results_root / FOLD_DIRS[fold_id] / "data" / "e20_transfer_results.json"),
            }
            for fold_id in FOLD_TARGETS
        },
        "dataset_count": len(ALL_DATASETS),
        "transfer_condition_count": len(transfer_rows),
        "algorithm_comparison_count": len(comparison_rows),
        "seed_count_per_condition": transfer.SEED_COUNT,
        "transfer_results": transfer_rows,
        "algorithm_comparison_results": comparison_rows,
        "summary_csv": str(csv_path),
        "figures": [str(path) for path in figures],
    }
    result_path = data_dir / "e20_multisplit_results.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "completed",
        "protocol": payload["protocol"],
        "completed_folds": list(FOLD_TARGETS),
        "result": str(result_path),
        "figures": payload["figures"],
    }
    (rotation_root / "e20_multisplit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result_path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for run e20 rotations.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=tuple(FOLD_TARGETS),
        default=("fold_02", "fold_03"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-models", action="store_true")
    args = parser.parse_args()
    for fold_id in args.folds:
        _run_fold(
            fold_id,
            args.results_root,
            args.workers,
            args.force,
            args.retrain_models,
        )
    _aggregate(args.results_root)


if __name__ == "__main__":
    main()
