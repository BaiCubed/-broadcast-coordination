from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/E22/trained_eps_ieee69_direct"
OUTPUT = ROOT / "results/E23"
OVERALL_SOURCE = SOURCE / "data/combined_e22_overall.csv"
SUMMARY_SOURCE = SOURCE / "data/combined_e22_summary.csv"

SCENARIO_ORDER = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
ALGORITHM_ORDER = (
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
ALGORITHM_COLORS = {
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
ALGORITHM_NAMES: dict[str, str] = {}

matplotlib.rcParams["font.family"] = "SimSun"
matplotlib.rcParams["axes.unicode_minus"] = False

FIGURES = (
    ("ieee69_effect_under_stress", "Algorithm effect under the IEEE-69 stress scenarios"),
    ("ieee69_failure_boundary_dashboard", "IEEE-69 failure boundary overview"),
    ("ieee69_constraint_margins", "IEEE-69 network constraint margin"),
    ("ieee69_mean_effect", "Mean algorithm effect on IEEE-69"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"nothing to write: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _filter_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    return [
        row
        for row in rows
        if row.get("topology") == "ieee69" and row.get("algorithm") in ALGORITHM_ORDER
    ]


def _save_figure(figure: Any, name: str) -> None:
    output = OUTPUT / "figures" / f"{name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _labels(overall: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    return {
        algorithm: next(
            (row["algorithm_label"], row["complexity"])
            for row in overall
            if row["algorithm"] == algorithm
        )
        for algorithm in ALGORITHM_ORDER
    }


def _matrix(rows: list[dict[str, Any]], field: str, scale: float = 1.0) -> np.ndarray:
    return np.asarray(
        [
            [
                scale
                * float(
                    next(
                        row[field]
                        for row in rows
                        if row["algorithm"] == algorithm and row["stress_mode"] == stress_mode
                    )
                )
                for stress_mode in SCENARIO_ORDER
            ]
            for algorithm in ALGORITHM_ORDER
        ]
    )


def _plot_effect(overall: list[dict[str, str]]) -> None:
    labels = _labels(overall)
    figure, axis = plt.subplots(figsize=(14.5, 7.2))
    x = np.arange(len(SCENARIO_ORDER))
    markers = ("o", "s", "^", "D", "v", "P", "X", "*", "h")
    for index, algorithm in enumerate(ALGORITHM_ORDER):
        values = [
            float(
                next(
                    row["curtailment_reduction_pct"]
                    for row in overall
                    if row["algorithm"] == algorithm and row["stress_mode"] == stress_mode
                )
            )
            for stress_mode in SCENARIO_ORDER
        ]
        axis.plot(
            x,
            values,
            marker=markers[index],
            linewidth=2.2,
            markersize=6,
            color=ALGORITHM_COLORS[algorithm],
            label=f"{labels[algorithm][0]} ({labels[algorithm][1]})",
        )
    axis.set_xticks(x, SCENARIO_ORDER)
    axis.set_ylim(0, 110)
    axis.set_xlabel("IEEE-69 network and spatial stress scenarios")
    axis.set_ylabel("Deliverable curtailment reduction (%)")
    axis.set_title("Effect boundary under the IEEE-69 stress conditions")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, "ieee69_effect_under_stress")


def _heatmap(
    figure: Any,
    axis: Any,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    threshold: float | None = None,
) -> None:
    image = axis.imshow(values, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=12)
    axis.set_xticks(np.arange(len(SCENARIO_ORDER)), SCENARIO_ORDER)
    axis.set_yticks(np.arange(len(ALGORITHM_ORDER)), [ALGORITHM_NAMES[a] for a in ALGORITHM_ORDER])
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > (vmax - vmin) * 0.58 + vmin else "black",
            )
    if threshold is not None:
        axis.contour(values, levels=[threshold], colors="#d1495b", linewidths=1.6)
    colorbar = figure.colorbar(image, ax=axis, pad=0.015, fraction=0.035)
    colorbar.set_label(colorbar_label)


def _plot_boundary_dashboard(boundary_rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14.0, 13.0), sharex=True)
    _heatmap(
        figure,
        axes[0],
        _matrix(boundary_rows, "curtailment_reduction_pct"),
        "Delivered effect",
        "Curtailment reduction (%)",
        0.0,
        100.0,
    )
    _heatmap(
        figure,
        axes[1],
        _matrix(boundary_rows, "network_acceptance_pct"),
        "Network acceptance",
        "Acceptance (%)",
        0.0,
        100.0,
        threshold=95.0,
    )
    _heatmap(
        figure,
        axes[2],
        _matrix(boundary_rows, "requested_violation_free_pct"),
        "Violation-free fraction of the raw request, before the safety layer",
        "Violation-free fraction (%)",
        0.0,
        100.0,
        threshold=95.0,
    )
    axes[-1].set_xlabel("IEEE-69 stress scenarios; the red contour is the 95% reference")
    figure.suptitle("IEEE-69 failure boundary overview", y=0.995, fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    _save_figure(figure, "ieee69_failure_boundary_dashboard")


def _plot_constraint_margins(boundary_rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14.0, 13.0), sharex=True)
    _heatmap(
        figure,
        axes[0],
        _matrix(boundary_rows, "mean_branch_loading"),
        "Mean maximum branch loading",
        "Loading; 1.0 is the thermal limit",
        0.0,
        2.0,
        threshold=1.0,
    )
    _heatmap(
        figure,
        axes[1],
        _matrix(boundary_rows, "mean_minimum_voltage_pu"),
        "Mean minimum bus voltage",
        "Voltage (p.u.); 0.95 is the lower bound",
        0.90,
        1.05,
        threshold=0.95,
    )
    _heatmap(
        figure,
        axes[2],
        _matrix(boundary_rows, "mean_transformer_loading"),
        "Mean maximum transformer loading",
        "Loading; 1.0 is the capacity limit",
        0.0,
        1.2,
        threshold=1.0,
    )
    axes[-1].set_xlabel("IEEE-69 stress scenarios; the red contour is the physical constraint threshold")
    figure.suptitle("Failure boundary of the physical network constraints", y=0.995, fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    _save_figure(figure, "ieee69_constraint_margins")


def _plot_mean_effect(overall: list[dict[str, str]]) -> None:
    labels = _labels(overall)
    values = []
    for algorithm in ALGORITHM_ORDER:
        values.append(
            float(
                np.mean(
                    [
                        float(row["curtailment_reduction_pct"])
                        for row in overall
                        if row["algorithm"] == algorithm
                    ]
                )
            )
        )
    y = np.arange(len(ALGORITHM_ORDER))
    figure, axis = plt.subplots(figsize=(12.0, 7.6))
    axis.barh(y, values, color=[ALGORITHM_COLORS[a] for a in ALGORITHM_ORDER])
    axis.set_yticks(y, [f"{labels[a][0]}  {labels[a][1]}" for a in ALGORITHM_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0, 110)
    for index, value in enumerate(values):
        axis.text(value + 0.6, index, f"{value:.2f}%", va="center", fontsize=8)
    axis.set_xlabel("Mean deliverable curtailment reduction over M0-M6 (%)")
    axis.set_title("Mean effect of the baselines and the fused EPS model on IEEE-69")
    axis.grid(axis="x", alpha=0.22)
    figure.tight_layout()
    _save_figure(figure, "ieee69_mean_effect")


def _copy_direct_training_artifacts() -> None:
    (OUTPUT / "models").mkdir(parents=True, exist_ok=True)
    for source_path in (SOURCE / "models").glob("*.pt"):
        shutil.copy2(source_path, OUTPUT / "models" / source_path.name)
    for source_path in (SOURCE / "models").glob("*.json"):
        shutil.copy2(source_path, OUTPUT / "models" / source_path.name)
    metadata = json.loads((SOURCE / "data/training_metadata.json").read_text(encoding="utf-8"))
    for record in metadata:
        dataset = str(record.get("dataset", record.get("dataset_id", "")))
        if dataset:
            record["model_path"] = f"results/E23/models/{dataset}.pt"
    (OUTPUT / "data/training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


    _write_csv(OUTPUT / "data/ieee69_overall.csv", overall)
    _write_csv(OUTPUT / "data/ieee69_by_dataset.csv", summary)
    _write_csv(OUTPUT / "data/ieee69_boundary_metrics.csv", boundary_rows)
    _copy_direct_training_artifacts()
    labels = _labels(overall)
    global ALGORITHM_NAMES
    ALGORITHM_NAMES = {algorithm: labels[algorithm][0] for algorithm in ALGORITHM_ORDER}
    _plot_effect(overall)
    _plot_boundary_dashboard(boundary_rows)
    _plot_constraint_margins(boundary_rows)
    _plot_mean_effect(overall)
    manifest = {
        "protocol": "E23_ieee69_failure_boundary_direct_only",
        "topology": "ieee69",
        "algorithms": list(ALGORITHM_ORDER),
        "scenario_modes": list(SCENARIO_ORDER),
        "dataset_count": 17,
        "seed_count": 30,
        "source_protocol": "E22_ieee69_direct_training_ablation_v2",
        "comparison_scope": "ieee69_direct_only",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"written: {OUTPUT}")
    print(f"pooled rows: {len(overall)}; per-dataset rows: {len(summary)}; figures: {len(FIGURES)}")


if __name__ == "__main__":
    main()
