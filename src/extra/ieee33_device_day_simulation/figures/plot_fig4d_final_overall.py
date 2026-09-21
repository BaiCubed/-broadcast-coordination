from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .fig4d_extra_baselines import FIG4D_COLORS, FIG4D_LABELS
from .fig4d_final_protocol import AVAILABILITY_MODES, FLEET_MODES, NETWORK_MODES, output_stem


DATASET_SUFFIX = "_ieee33_real_load"


def _dataset_label(result_root: Path) -> str:
    name = result_root.parents[1].name
    return name.removesuffix(DATASET_SUFFIX)


def _condition_payloads(results_root: Path, stem: str) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for result_root in sorted(results_root.glob(f"*{DATASET_SUFFIX}/coverage_fix/network_constrained_new")):
        path = result_root / "data" / f"curtailment_baselines_{stem}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append((_dataset_label(result_root), payload))
    return rows


def _expected_datasets(results_root: Path, stem: str) -> set[str]:
    expected = set()
    for result_root in sorted(results_root.glob(f"*{DATASET_SUFFIX}/coverage_fix/network_constrained_new")):
        dataset = _dataset_label(result_root)
        if "_source_unique_" in stem:
            population_path = result_root / "data/population_summary.json"
            population = json.loads(population_path.read_text(encoding="utf-8"))
            if int(population["source_devices"]) >= 5000:
                continue
        expected.add(dataset)
    return expected


def plot_condition(results_root: Path, stem: str) -> tuple[Path, dict[str, Any]]:
    payloads = _condition_payloads(results_root, stem)
    if not payloads:
        raise FileNotFoundError(f"no final data found for condition {stem}")
    actual = {dataset for dataset, _ in payloads}
    expected = _expected_datasets(results_root, stem)
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"condition {stem} is missing dataset results: {', '.join(missing)}")
    order = payloads[0][1]["strategy_order"]
    definitions = payloads[0][1]["strategy_definitions"]
    matrix = np.asarray([
        [float(payload["results"][strategy]["mean_reduction_pct"]) for strategy in order]
        for _, payload in payloads
    ])
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    labels = [FIG4D_LABELS[strategy] for strategy in order]
    colors = [FIG4D_COLORS[strategy] for strategy in order]

    fig, ax = plt.subplots(figsize=(15.5, 8.2), constrained_layout=True)
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6, zorder=2)
    for dataset_index in range(matrix.shape[0]):
        jitter = (dataset_index - (matrix.shape[0] - 1) / 2.0) * 0.012
        ax.scatter(x + jitter, matrix[dataset_index], s=18, color="#1f1f1f", alpha=0.48, zorder=3)
    for strategy, bar, mean in zip(order, bars, means):
        value_label = f"Mean\n{mean:.1f}%"
        if strategy == "centralized_optimal":
            value_label = f"Upper bound\nMean\n{mean:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(2.0, mean * 0.5),
            value_label,
            ha="center",
            va="center",
            color="white" if mean > 20 else "#222222",
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("Curtailment reduction (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    condition = payloads[0][1]["condition"]
    ax.set_title(
        "Overall Figure 4D: "
        f"{condition['fleet_mode']} | {condition['network_mode']} | {condition['availability_mode']}"
    )
    ax.text(
        0.01,
        0.99,
        f"Bars: mean across {len(payloads)} applicable datasets; error bars: dataset SD; dots: individual datasets",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.text(
        0.01,
        0.94,
        "Complexity: N = devices; H = MPC prediction horizon.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    output = results_root / f"overall_fig4d_{stem}.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    summary = {
        "condition": condition,
        "dataset_count": len(payloads),
        "datasets": [dataset for dataset, _ in payloads],
        "strategy_order": order,
        "strategy_definitions": definitions,
        "mean_reduction_pct": {strategy: float(value) for strategy, value in zip(order, means)},
        "std_across_datasets_pct": {strategy: float(value) for strategy, value in zip(order, stds)},
        "per_dataset_reduction_pct": {
            dataset: {strategy: float(value) for strategy, value in zip(order, row)}
            for (dataset, _), row in zip(payloads, matrix)
        },
        "figure": str(output),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for plot fig4d final overall.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    summaries = []
    for fleet_mode in FLEET_MODES:
        for network_mode in NETWORK_MODES:
            for availability_mode in AVAILABILITY_MODES:
                stem = output_stem(fleet_mode, network_mode, availability_mode)
                figure, summary = plot_condition(args.results_root, stem)
                summaries.append(summary)
                print(figure)
    manifest = {
        "protocol": "fig4d_final_overall_v1",
        "condition_count": len(summaries),
        "conditions": summaries,
    }
    path = args.results_root / "overall_fig4d_final_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
