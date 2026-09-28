#!/usr/bin/env python3
"""Build the English IEEE-69 constraint-component heatmap for the appendix."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_appendix import REF_CMAP


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/figs/e22_ieee69_appendix/ieee69_constraint_component_audit_appendix.png"
ALGORITHM_ORDER = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_e20_pooled",
    "eps_e21_mixed",
    "centralized_optimal",
)
STRESS_MODES = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
PANELS = (
    ("maximum_branch_loading_mean", "Executed maximum branch loading", 0.0, 1.5, "{:.2f}"),
    ("minimum_voltage_pu_mean", "Executed minimum voltage (p.u.)", 0.94, 1.0, "{:.3f}"),
    ("maximum_transformer_loading_mean", "Executed maximum transformer loading", 0.0, 1.0, "{:.2f}"),
    ("safety_layer_gain", "Safety-layer gain in violation-free steps (pp)", 0.0, 100.0, "{:.1f}"),
)


def _maximum_tie_mask(values: np.ndarray, formatter: str) -> np.ndarray:
    """Mark displayed ties at the maximum value within each stress-mode column."""
    displayed = np.asarray(
        [[formatter.format(value) for value in row] for row in values],
        dtype=object,
    )
    mask = np.zeros(values.shape, dtype=bool)
    for column_index in range(values.shape[1]):
        maximum = np.nanmax(values[:, column_index])
        maximum_display = formatter.format(maximum)
        ties = displayed[:, column_index] == maximum_display
        if int(np.count_nonzero(ties)) > 1:
            mask[:, column_index] = ties
    return mask


def _matrix(data: pd.DataFrame, metric: str) -> tuple[np.ndarray, list[str]]:
    if metric == "safety_layer_gain":
        data = data.assign(
            safety_layer_gain=(
                data["executed_violation_free_pct_mean"]
                - data["requested_violation_free_pct_mean"]
            )
        )
    labels = (
        data[["algorithm", "algorithm_label"]]
        .drop_duplicates("algorithm")
        .set_index("algorithm")
        .reindex(ALGORITHM_ORDER)["algorithm_label"]
        .tolist()
    )
    values = (
        data.pivot_table(index="algorithm", columns="stress_mode", values=metric, aggfunc="mean")
        .reindex(index=ALGORITHM_ORDER, columns=STRESS_MODES)
        .to_numpy(dtype=float)
    )
    if np.isnan(values).any():
        raise ValueError(f"Missing values in {metric} heatmap matrix")
    return values, labels


def build() -> None:
    data = pd.read_csv(ROOT / "results/E22/data/e22_summary.csv")
    data = data[data["topology"].eq("ieee69")].copy()
    if data["dataset"].nunique() != 17 or data["seed_count"].nunique() != 1 or int(data["seed_count"].iloc[0]) != 30:
        raise ValueError("Expected IEEE-69 summaries for 17 datasets and 30 paired seeds")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 9.2), sharex=True, sharey=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for panel_index, (metric, title, vmin, vmax, formatter) in enumerate(PANELS):
        axis = axes.flat[panel_index]
        values, labels = _matrix(data, metric)
        maximum_ties = _maximum_tie_mask(values, formatter)
        image = axis.imshow(values, cmap=REF_CMAP, aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(f"{panel_labels[panel_index]} {title}", fontsize=13, fontweight="bold", loc="left", pad=8)
        axis.set_xticks(np.arange(len(STRESS_MODES)), STRESS_MODES, fontsize=10, fontweight="bold")
        if panel_index % 2 == 0:
            axis.set_yticks(np.arange(len(labels)), labels, fontsize=9.5, fontweight="bold")
        else:
            axis.tick_params(labelleft=False)
        axis.tick_params(length=3, width=0.7)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                rgba = REF_CMAP((values[row_index, column_index] - vmin) / (vmax - vmin))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column_index,
                    row_index,
                    formatter.format(values[row_index, column_index])
                    + ("†" if maximum_ties[row_index, column_index] else ""),
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    fontweight="bold",
                    color="white" if luminance < 0.53 else "black",
                )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.ax.tick_params(labelsize=8)
    figure.supxlabel("Network and spatial stress mode", fontsize=12, fontweight="bold", y=0.035)
    figure.supylabel("Algorithm", fontsize=12, fontweight="bold", x=0.025)
    figure.text(
        0.5,
        0.012,
        "† Repeated displayed values at the current stress-mode maximum; "
        "the shared value is caused by that mode's maximum.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#404040",
    )
    figure.subplots_adjust(left=0.18, right=0.965, bottom=0.15, top=0.90, wspace=0.34, hspace=0.30)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"created={OUTPUT}")


if __name__ == "__main__":
    build()
