from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import run_e20_transfer as transfer
from . import run_e21_mixed_scenarios as mixed
from . import run_e22_ieee69_complexity as e22
from . import run_e22_trained_eps_comparison as direct


ROOT = Path("results/E22")
LEGACY_TRAINED = ROOT / "trained_eps_ieee69"
DIRECT_TRAINED = ROOT / "trained_eps_ieee69_direct"
GREEDY_PROTOCOL = "E22_recompute_centralized_greedy_ub_on_device_headroom_v1"
LEGACY_ORDER = ("eps_e20_pooled", "eps_e21_mixed", "eps_in_domain")
LEGACY_COLORS = {
    "eps_e20_pooled": mixed.ALGORITHM_COLORS["eps_e20_pooled"],
    "eps_e21_mixed": mixed.ALGORITHM_COLORS["eps_e21_mixed"],
    "eps_in_domain": transfer.COMPARISON_COLORS["eps_in_domain"],
}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _label(algorithm: str) -> str:
    if algorithm == "eps_in_domain":
        return transfer.COMPARISON_DEFINITIONS[algorithm]["label"]
    return mixed.ALGORITHM_DEFINITIONS[algorithm]["label"]


def _complexity(algorithm: str) -> str:
    if algorithm == "eps_in_domain":
        return transfer.COMPARISON_DEFINITIONS[algorithm]["complexity"]
    return mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"]


def _mean_over_stress(rows, dataset, algorithm):
    values = [
        float(row["curtailment_reduction_pct"])
        for row in rows
        if row["dataset"] == dataset
        and row["topology"] == "ieee69"
        and row["algorithm"] == algorithm
    ]
    return float(np.mean(values))


def _plot_legacy_dataset_heatmap(rows, figure_dir):
    matrix = np.asarray(
        [
            [_mean_over_stress(rows, dataset, algorithm) for algorithm in LEGACY_ORDER]
            for dataset in e22.E22_DATASETS
        ]
    )
    fig, axis = plt.subplots(figsize=(8.6, 8.6))
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=100.0)
    axis.set_xticks(np.arange(len(LEGACY_ORDER)), [_label(a) for a in LEGACY_ORDER])
    axis.set_yticks(
        np.arange(len(e22.E22_DATASETS)),
        [e22.DATASET_LABELS[dataset] for dataset in e22.E22_DATASETS],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value > 58 else "black",
            )
    axis.set_title("IEEE-69 transfer and in-domain training by dataset")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Curtailment reduction (%)")
    return e22._save_figure(fig, figure_dir, "trained_vs_transfer_dataset_heatmap_ieee69")


def _plot_legacy_stress(overall, figure_dir):
    fig, axis = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(e22.STRESS_MODES))
    for algorithm in LEGACY_ORDER:
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
            color=LEGACY_COLORS[algorithm],
            label=f"{_label(algorithm)} ({_complexity(algorithm)})",
        )
    axis.set_xticks(x, list(e22.STRESS_MODES))
    axis.set_ylim(0, 105)
    axis.set_ylabel("Deliverable curtailment reduction (%)")
    axis.set_title("IEEE-69 transfer and in-domain training under M0-M6")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    return e22._save_figure(fig, figure_dir, "trained_vs_transfer_stress_profiles_ieee69")


def _plot_legacy_gain(rows, figure_dir):
    labels = [e22.DATASET_LABELS[dataset] for dataset in e22.E22_DATASETS]
    gain = np.asarray(
        [
            _mean_over_stress(rows, dataset, "eps_in_domain")
            - _mean_over_stress(rows, dataset, "eps_e21_mixed")
            for dataset in e22.E22_DATASETS
        ]
    )
    order = np.argsort(gain)
    fig, axis = plt.subplots(figsize=(10.2, 7.4))
    axis.barh(
        np.arange(len(order)),
        gain[order],
        color=np.where(gain[order] >= 0.0, "#2a9d8f", "#d95f5f"),
    )
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
    axis.set_xlabel("In-domain EPS gain over E21 mixed transfer (percentage points)")
    axis.set_title("IEEE-69 in-domain training gain by dataset")
    axis.grid(axis="x", alpha=0.22)
    return e22._save_figure(fig, figure_dir, "trained_gain_by_dataset_ieee69")


def _plot_legacy_all_algorithms(overall, figure_dir):
    order = list(e22.ALGORITHM_ORDER)
    order.insert(order.index("centralized_optimal"), "eps_in_domain")
    values = [
        float(
            np.mean(
                [
                    float(row["curtailment_reduction_pct"])
                    for row in overall
                    if row["topology"] == "ieee69" and row["algorithm"] == algorithm
                ]
            )
        )
        for algorithm in order
    ]
    colors = [
        LEGACY_COLORS.get(algorithm, mixed.ALGORITHM_COLORS.get(algorithm, "#777777"))
        for algorithm in order
    ]
    fig, axis = plt.subplots(figsize=(11.2, 6.4))
    y = np.arange(len(order))
    axis.barh(y, values, color=colors)
    axis.set_yticks(y, [f"{_label(a)}  {_complexity(a)}" for a in order])
    axis.invert_yaxis()
    axis.set_xlim(0, 110)
    for index, value in enumerate(values):
        axis.text(value + 0.6, index, f"{value:.2f}%", va="center", fontsize=8)
    axis.set_xlabel("Deliverable curtailment reduction (%)")
    axis.set_title("IEEE-69 algorithms with refreshed O(N) greedy upper bound")
    axis.grid(axis="x", alpha=0.22)
    return e22._save_figure(fig, figure_dir, "trained_eps_all_algorithms_ieee69")


def _refresh_legacy(base_summary, base_overall):
    native_summary = _read_rows(LEGACY_TRAINED / "data/trained_eps_summary.csv")
    previous_overall = _read_rows(LEGACY_TRAINED / "data/combined_e22_overall.csv")
    native_overall = [row for row in previous_overall if row["algorithm"] == "eps_in_domain"]
    combined_summary = [*base_summary, *native_summary]
    combined_overall = [*base_overall, *native_overall]
    e22._write_rows(LEGACY_TRAINED / "data/combined_e22_summary.csv", combined_summary)
    e22._write_rows(LEGACY_TRAINED / "data/combined_e22_overall.csv", combined_overall)
    figure_dir = LEGACY_TRAINED / "figures"
    figures = []
    figures.extend(_plot_legacy_dataset_heatmap(combined_summary, figure_dir))
    figures.extend(_plot_legacy_stress(combined_overall, figure_dir))
    figures.extend(_plot_legacy_gain(combined_summary, figure_dir))
    figures.extend(_plot_legacy_all_algorithms(combined_overall, figure_dir))
    manifest_path = LEGACY_TRAINED / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figures"] = figures
    manifest["greedy_ub_protocol"] = GREEDY_PROTOCOL
    manifest["greedy_ub_definition"] = "O(N) theoretical device headroom upper bound; network clipping audit-only"
    manifest["greedy_ub_refreshed"] = True
    manifest["greedy_ub_complexity"] = "O(N)"
    manifest["refreshed_at"] = e22._utc_now()
    e22._write_json(manifest_path, manifest)
    e22._write_json(LEGACY_TRAINED / "checkpoint.json", manifest)


def _refresh_direct(base_summary, base_overall):
    native_summary = _read_rows(DIRECT_TRAINED / "data/trained_eps_summary.csv")
    previous_overall = _read_rows(DIRECT_TRAINED / "data/combined_e22_overall.csv")
    native_overall = [
        row for row in previous_overall if row["algorithm"] in direct.NATIVE_ALGORITHMS
    ]
    combined_summary = [*base_summary, *native_summary]
    combined_overall = [*base_overall, *native_overall]
    e22._write_rows(DIRECT_TRAINED / "data/combined_e22_summary.csv", combined_summary)
    e22._write_rows(DIRECT_TRAINED / "data/combined_e22_overall.csv", combined_overall)
    figure_dir = DIRECT_TRAINED / "figures"
    figures = []
    figures.extend(direct._plot_dataset_heatmap(combined_summary, figure_dir))
    figures.extend(direct._plot_stress_profiles(combined_overall, figure_dir))
    figures.extend(direct._plot_native_gain(combined_summary, figure_dir))
    figures.extend(direct._plot_all_algorithms(combined_overall, figure_dir))
    manifest_path = DIRECT_TRAINED / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figures"] = figures
    manifest["greedy_ub_protocol"] = GREEDY_PROTOCOL
    manifest["greedy_ub_definition"] = "O(N) theoretical device headroom upper bound; network clipping audit-only"
    manifest["greedy_ub_refreshed"] = True
    manifest["greedy_ub_complexity"] = "O(N)"
    manifest["refreshed_at"] = e22._utc_now()
    e22._write_json(manifest_path, manifest)
    e22._write_json(DIRECT_TRAINED / "checkpoint.json", manifest)


def main():
    payloads = [
        json.loads((ROOT / "raw" / f"{dataset}.json").read_text(encoding="utf-8"))
        for dataset in e22.E22_DATASETS
    ]
    manifest = e22._finalize(ROOT, payloads, e22.BOOTSTRAP_DRAWS, e22.SEED_COUNT)
    manifest["greedy_ub_protocol"] = GREEDY_PROTOCOL
    manifest["greedy_ub_definition"] = "O(N) theoretical device headroom upper bound; network clipping audit-only"
    manifest["greedy_ub_refreshed"] = True
    manifest["greedy_ub_complexity"] = "O(N)"
    e22._write_json(ROOT / "manifest.json", manifest)
    e22._write_json(ROOT / "checkpoint.json", manifest)
    base_summary = _read_rows(ROOT / "data/e22_summary.csv")
    base_overall = _read_rows(ROOT / "data/e22_overall_summary.csv")
    _refresh_legacy(base_summary, base_overall)
    _refresh_direct(base_summary, base_overall)
    print("refreshed the main, transfer-comparison and in-domain-training figures")


if __name__ == "__main__":
    main()
