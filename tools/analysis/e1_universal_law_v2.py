from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLD = 0.95
COHORT_GRID = [2, 3, 5, 8, 12, 20, 30, 50, 100, 150, 250, 400, 650, 1000]
INK, OURS, FAINT = "#111111", "#1b4f72", "#9aa4ad"


def fit_a(sizes: np.ndarray, r2: np.ndarray) -> float:
    keep = np.isfinite(sizes) & np.isfinite(r2) & (r2 < 1.0) & (sizes > 0)
    x, y = sizes[keep], np.log10(1.0 - r2[keep])
    grid = np.linspace(0.5, 20.0, 20000)
    losses = [np.mean((y - np.log10(a / (x + a))) ** 2) for a in grid]
    return float(grid[int(np.argmin(losses))])


def crossing(sizes: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    envelope = np.maximum.accumulate(values)
    hit = np.flatnonzero(envelope >= threshold)
    if not hit.size:
        return None
    index = int(hit[0])
    if index == 0:
        return float(sizes[0])
    x0, x1 = np.log10(sizes[index - 1]), np.log10(sizes[index])
    y0, y1 = values[index - 1], values[index]
    if abs(y1 - y0) < 1e-12:
        return float(sizes[index])
    fraction = float(np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0))
    return float(10 ** (x0 + fraction * (x1 - x0)))


from matplotlib.ticker import FuncFormatter, NullFormatter


def _plain_tick(value, _pos=None):
    if value == 0:
        return "0"
    return f"{value:,.0f}" if abs(value) >= 10000 else f"{value:g}"


def plain_log_ticks(fig):
    formatter = FuncFormatter(_plain_tick)
    for axes in fig.axes:
        for scale, axis, limits in (
            (axes.get_xscale(), axes.xaxis, axes.get_xlim()),
            (axes.get_yscale(), axes.yaxis, axes.get_ylim()),
        ):
            if scale not in {"log", "symlog"}:
                continue
            axis.set_major_formatter(formatter)
            low, high = abs(limits[0]), abs(limits[1])
            decades = np.log10(max(high, 1e-12) / max(low, 1e-12)) if low > 0 else 99.0
            axis.set_minor_formatter(formatter if decades < 1.6 else NullFormatter())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.summary)
    frame = frame[frame["arm"] == "data_coupled"]
    column = "R2" if "R2" in frame.columns else "R2_signal"

    fit_block = frame[frame["N"] >= 8]
    a_hat = fit_a(fit_block["N"].to_numpy(float), fit_block[column].to_numpy(float))
    a_median = float((fit_block["N"] * (1 - fit_block[column]) / fit_block[column]).median())
    n95_law = 19.0 * a_hat

    cover = frame.groupby("dataset")["N"].apply(lambda v: set(COHORT_GRID) <= set(v))
    cohort = sorted(cover[cover].index)
    median_curve = (frame[frame["dataset"].isin(cohort) & frame["N"].isin(COHORT_GRID)]
                    .groupby("N")[column].median())
    n95_ours = crossing(median_curve.index.to_numpy(float), median_curve.to_numpy(), THRESHOLD)

    smooth = np.logspace(np.log10(1.6), np.log10(3600), 400)
    law = smooth / (smooth + a_hat)

    fig, axes = plt.subplots(1, 4, figsize=(25.5, 6.4), constrained_layout=True)

    for axis, log_y in zip(axes[:3], (False, False, True)):
        first = True
        for _, block in frame.groupby("dataset"):
            block = block.sort_values("N")
            y = block[column] if not log_y else np.clip(1 - block[column], 1e-4, None)
            axis.plot(block["N"], y, "-", linewidth=0.7, alpha=0.5, color=FAINT, zorder=1,
                      label="the 15 datasets (thin lines, no marker)" if first else None)
            first = False
        axis.plot(smooth, law if not log_y else 1 - law, "--", linewidth=2.6, color=INK, zorder=4,
                  label=f"universal law $R^2=N/(N+a)$, $a={a_hat:.2f}$ (thick dash, no marker)")
        y = median_curve if not log_y else np.clip(1 - median_curve, 1e-4, None)
        axis.plot(median_curve.index, y, "o-", markersize=6.5, linewidth=1.8,
                  markerfacecolor="white", markeredgewidth=1.7, color=OURS, zorder=5,
                  label=f"this work, median of {len(cohort)} datasets (circles)")

    axis = axes[0]
    axis.axhline(THRESHOLD, linestyle=":", color=INK, linewidth=1.2)
    for value, colour in ((n95_law, INK), (n95_ours, OURS)):
        if value:
            axis.axvline(value, linestyle="-", linewidth=0.9, alpha=0.45, color=colour, zorder=0)
    axis.set(xscale="log", ylim=(0.0, 1.04), xlim=(1.6, 3600),
             xlabel="number of devices $N$   [LOG axis]",
             ylabel="$R^2$: share of aggregate variance explained by the broadcast   "
                    "[LINEAR axis, 0 to 1]",
             title="a  One parameter reproduces all 15 datasets")
    axis.legend(fontsize=8, loc="lower right", framealpha=0.96)
    axis.text(0.025, 0.985, "dotted horizontal line: the $R^2=0.95$ bar",
              transform=axis.transAxes, fontsize=8.5, va="top")
    axis.text(0.025, 0.925, "series are separated by SYMBOL, not colour",
              transform=axis.transAxes, fontsize=8.5, va="top")

    axis = axes[1]
    axis.axhline(THRESHOLD, linestyle=":", color=INK, linewidth=1.2)
    for value, colour in ((n95_law, INK), (n95_ours, OURS)):
        if value:
            axis.axvline(value, linestyle="--", linewidth=1.2, alpha=0.85, color=colour, zorder=2)
    axis.set(xscale="log", ylim=(0.85, 1.002), xlim=(20, 3600),
             xlabel="number of devices $N$   [LOG axis]",
             ylabel="$R^2$   [LINEAR axis, zoomed to 0.85-1.00]",
             title="b  Zoom on the $R^2=0.95$ crossing")
    axis.text(0.03, 0.955,
              f"$N_{{95}}$:  one-parameter law $19a$ = {n95_law:.0f}   |   "
              f"measured median curve = {n95_ours:.0f}",
              transform=axis.transAxes, fontsize=9.5, va="top")

    axis = axes[2]
    axis.axhline(1 - THRESHOLD, linestyle=":", color=INK, linewidth=1.2)
    axis.text(1.75, (1 - THRESHOLD) * 1.15, "$1-R^2=0.05$", fontsize=8.5, color=INK)
    axis.set(xscale="log", yscale="log", xlim=(1.6, 3600), ylim=(3e-4, 1.2),
             xlabel="number of devices $N$   [LOG axis]",
             ylabel="unexplained fraction $1-R^2$   [LOG axis]",
             title="c  Same data, log-log: the law is a straight line of slope $-1$")
    axis.text(0.03, 0.07,
              "$1-R^2 = a/(N+a) \\rightarrow a/N$ for large $N$;\n"
              "slope $-1$ on this axis is the signature of\n"
              "device noise averaging out as $1/N$",
              transform=axis.transAxes, fontsize=8.5)

    axis = axes[3]
    residual = frame[column] - frame["N"] / (frame["N"] + a_hat)
    axis.axhspan(-0.02, 0.02, color=FAINT, alpha=0.25, label="$\\pm0.02$ band")
    axis.plot(frame["N"], residual, "o", markersize=3.6, alpha=0.5, color=OURS,
              markerfacecolor="none", markeredgewidth=1.0, label="one cell (dataset x fleet)")
    binned = residual.groupby(pd.cut(np.log10(frame["N"]), 10)).median()
    centres = [10 ** interval.mid for interval in binned.index]
    axis.plot(centres, binned.to_numpy(), "-", linewidth=2.0, color=INK,
              label="median per decade bin (line, no marker)")
    axis.axhline(0.0, color=INK, linewidth=1.1)
    axis.axvline(8, linestyle=":", color="#b2182b", linewidth=1.4)
    axis.text(8.6, 0.20, "fit uses $N\\geq8$ only", fontsize=8.5, color="#b2182b")
    axis.set(xscale="log", ylim=(-0.30, 0.30),
             xlabel="number of devices $N$   [LOG axis]",
             ylabel="measured $R^2$ $-$ law   [LINEAR axis]",
             title="d  Residual: where the one-parameter law is allowed to fail")
    axis.legend(fontsize=8, loc="lower right")
    inside = float((residual.abs() <= 0.02).mean())
    axis.text(0.03, 0.06, f"mean |residual| = {residual.abs().mean():.4f},  RMS = "
                          f"{np.sqrt((residual ** 2).mean()):.4f}\n"
                          f"{inside:.0%} of cells sit inside the $\\pm0.02$ band",
              transform=axis.transAxes, fontsize=8.5)

    fig.suptitle("E1: a one-parameter law for how much of the aggregate the broadcast explains",
                 fontsize=14)
    plain_log_ticks(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=190)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)

    fitted_residual = fit_block[column] - fit_block["N"] / (fit_block["N"] + a_hat)
    report = {
        "law": "R^2 = N / (N + a)",
        "a_fitted_on_log_misfit": a_hat,
        "a_median_of_per_cell_solve": a_median,
        "cells_used_for_fit": int(len(fit_block)),
        "cells_total": int(len(frame)),
        "N95_from_law_19a": n95_law,
        "N95_measured_cohort_median_curve": n95_ours,
        "residual_fit_population": {"mean_abs": float(fitted_residual.abs().mean()),
                                    "rms": float(np.sqrt((fitted_residual ** 2).mean())),
                                    "max_abs": float(fitted_residual.abs().max())},
        "residual_all_cells": {"mean_abs": float(residual.abs().mean()),
                               "rms": float(np.sqrt((residual ** 2).mean())),
                               "fraction_within_0.02": inside},
        "cohort_datasets": cohort,
    }
    args.output.with_name(args.output.name + "_law.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
