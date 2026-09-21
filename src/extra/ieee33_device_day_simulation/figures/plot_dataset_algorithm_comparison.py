from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DATASET_LABELS = {
    "bdg1_building_data_genome": "BDG1",
    "bdg2_building_data_genome": "BDG2",
    "low_carbon_london": "LCL",
    "danish_smart_heat_meters": "Danish",
    "smart_grid_smart_city": "SGSC",
    "heapo_heat_pumps": "HEAPO",
    "goiener_smart_meters": "GoiEner",
    "european_lv_urban_8087": "EU LV 8087",
    "european_lv_rural_2731": "EU Rural 2731",
    "european_lv_urban_35297": "EU Urban 35297",
    "norway_ami_energy_distribution": "Norway AMI",
    "camsl_japan_smart_meters": "CAMSL",
    "irish_domestic_smart_meters": "Irish",
    "opsd_household_data": "OPSD",
    "complete_energy_community": "CEC",
}

ALGORITHMS = [
    ("no_coordination", "No\ncoord."),
    ("local_rules", "Local\nSOC rules"),
    ("mpc_optimal", "MPC"),
    ("mean_field_control", "Mean-field\ncontrol"),
    ("virtual_battery", "Virtual\nbattery"),
    ("packetized_energy_management", "PEM"),
    ("transactive_control", "Transactive\ncontrol"),
    ("eps_broadcast", "EPS\nbroadcast"),
    ("centralized_optimal", "Centralized\ngreedy UB"),
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _baseline_path(
    results_root: Path, dataset: str, baseline_name: str
) -> Path | None:
    root = results_root / f"{dataset}_ieee33_real_load"
    if baseline_name != "auto":
        path = root / "coverage_fix" / "network_constrained_new" / "data" / baseline_name
        return path if path.is_file() else None
    candidates = [
        root / "network_stress" / "data" / "curtailment_baselines_extra.json",
        root / "coverage_fix" / "network_stress" / "data" / "curtailment_baselines_extra.json",
        root / "coverage_fix" / "network_constrained_new" / "data" / "curtailment_baselines_extra.json",
        root / "network_stress" / "data" / "curtailment_baselines.json",
        root / "coverage_fix" / "network_stress" / "data" / "curtailment_baselines.json",
        root / "coverage_fix" / "network_constrained_new" / "data" / "curtailment_baselines.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.glob("**/curtailment_baselines_extra.json"))
    if matches:
        return matches[0]
    matches = sorted(root.glob("**/curtailment_baselines.json"))
    return matches[0] if matches else None


def _collect_rows(
    results_root: Path, baseline_name: str = "auto", include_missing: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, label in DATASET_LABELS.items():
        path = _baseline_path(results_root, dataset, baseline_name)
        if path is None:
            if include_missing:
                rows.append({
                    "dataset": dataset,
                    "label": label,
                    "source_file": None,
                    "no_surplus": False,
                    "values": {key: None for key, _ in ALGORITHMS},
                })
            continue
        payload = _load_json(path)
        results = payload.get("results", {})
        values: dict[str, float | None] = {}
        baseline_values: list[float] = []
        for key, _ in ALGORITHMS:
            entry = results.get(key)
            if not isinstance(entry, dict):
                values[key] = None
                continue
            values[key] = float(entry.get("mean_reduction_pct", 0.0))
            baseline_values.append(float(entry.get("baseline_curtailment_mwh", 0.0)))
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "source_file": str(path),
                "no_surplus": max(baseline_values or [0.0]) <= 1e-12,
                "values": values,
            }
        )
    return rows


def _write_outputs(rows: list[dict[str, Any]], output_path: Path) -> None:
    json_path = output_path.with_suffix(".json")
    csv_path = output_path.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "label", "no_surplus", *[key for key, _ in ALGORITHMS], "source_file"])
        for row in rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["label"],
                    row["no_surplus"],
                    *[row["values"].get(key) for key, _ in ALGORITHMS],
                    row["source_file"],
                ]
            )


def plot_heatmap(
    results_root: Path,
    output_path: Path,
    *,
    baseline_name: str = "auto",
    include_missing: bool = False,
    title: str = "Curtailment reduction by dataset and algorithm",
) -> Path:
    rows = _collect_rows(results_root, baseline_name, include_missing)
    if not rows:
        raise FileNotFoundError(f"no curtailment baseline files found under {results_root}")

    matrix = _matrix(rows)
    _plot_heatmap_matrix(rows, matrix, output_path, title)
    _write_outputs(rows, output_path)
    return output_path


def _matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.full((len(rows), len(ALGORITHMS)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for col_index, (key, _) in enumerate(ALGORITHMS):
            value = row["values"].get(key)
            if value is not None:
                matrix[row_index, col_index] = value
    return matrix


def _plot_heatmap_matrix(
    rows: list[dict[str, Any]], matrix: np.ndarray, output_path: Path, title: str
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")

    fig_height = max(6.5, 0.42 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(13.6, fig_height))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0.0, vmax=100.0, cmap=cmap)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.018)
    colorbar.set_label("Curtailment reduction (%)")

    ax.set_xticks(np.arange(len(ALGORITHMS)))
    ax.set_xticklabels([label for _, label in ALGORITHMS], fontsize=9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{row['label']}*" if row["no_surplus"] else row["label"] for row in rows], fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Dataset")

    for row_index, row in enumerate(rows):
        for col_index, _ in enumerate(ALGORITHMS):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                text = "n/a"
                color = "#666666"
            elif row["no_surplus"]:
                text = "0"
                color = "#555555"
            else:
                text = f"{value:.1f}"
                color = "white" if value >= 55 else "#222222"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=7.5, color=color)

    ax.set_xticks(np.arange(-0.5, len(ALGORITHMS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", labelrotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.99,
        0.012,
        "*No positive curtailment baseline in selected result file; reductions are structurally 0.",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_grouped_bar(
    results_root: Path,
    output_path: Path,
    *,
    baseline_name: str = "auto",
    include_missing: bool = False,
    title: str = "Curtailment reduction by dataset and algorithm",
) -> Path:
    rows = _collect_rows(results_root, baseline_name, include_missing)
    if not rows:
        raise FileNotFoundError(f"no curtailment baseline files found under {results_root}")

    matrix = _matrix(rows)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    colors = [
        "#b3b3b3",
        "#ed7d31",
        "#7b61ff",
        "#4e79a7",
        "#59a14f",
        "#b07aa1",
        "#f28e2b",
        "#069c8f",
        "#4c9f70",
    ]
    x = np.arange(len(rows), dtype=float)
    width = 0.082
    offsets = (np.arange(len(ALGORITHMS), dtype=float) - (len(ALGORITHMS) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(18.5, 7.2))
    for alg_index, ((key, label), color) in enumerate(zip(ALGORITHMS, colors)):
        values = matrix[:, alg_index]
        ax.bar(
            x + offsets[alg_index],
            np.nan_to_num(values, nan=0.0),
            width,
            color=color,
            edgecolor="#555555",
            linewidth=0.35,
            label=label.replace("\n", " "),
            alpha=0.95,
        )

    for row_index, row in enumerate(rows):
        if row["no_surplus"]:
            ax.text(
                x[row_index],
                3.0,
                "no surplus",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
                color="#a33a3a",
            )

    ax.set_title(title)
    ax.set_ylabel("Curtailment reduction (%)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['label']}*" if row["no_surplus"] else row["label"] for row in rows], rotation=32, ha="right")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
    fig.text(
        0.99,
        0.012,
        "*No positive curtailment baseline in selected result file; reductions are structurally 0.",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dataset-by-algorithm curtailment heatmap")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/overall_algorithm_curtailment_heatmap.png"),
    )
    parser.add_argument("--baseline-name", default="auto")
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument(
        "--title", default="Curtailment reduction by dataset and algorithm"
    )
    parser.add_argument(
        "--bar-output",
        type=Path,
        default=Path("results/overall_algorithm_curtailment_grouped_bar.png"),
    )
    args = parser.parse_args()
    print(plot_heatmap(
        args.results_root,
        args.output,
        baseline_name=args.baseline_name,
        include_missing=args.include_missing,
        title=args.title,
    ))
    print(plot_grouped_bar(
        args.results_root,
        args.bar_output,
        baseline_name=args.baseline_name,
        include_missing=args.include_missing,
        title=args.title,
    ))


if __name__ == "__main__":
    main()
