from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

C_COUPLED = "#1b4f72"
C_BASE = "#c0562a"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d4"
SURF = "#fcfcfb"

THRESHOLD = 0.90
plt.rcParams.update({
    "font.size": 11.5, "axes.titlesize": 13.0, "axes.labelsize": 11.5,
    "axes.edgecolor": "#9a9a96", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "savefig.facecolor": SURF,
})


def _plain(value, _pos=None):
    if value == 0:
        return "0"
    return f"{value:,.0f}" if abs(value) >= 10000 else f"{value:g}"


def plain_log_ticks(fig):
    fmt = FuncFormatter(_plain)
    for ax in fig.axes:
        for scale, axis, lim in ((ax.get_xscale(), ax.xaxis, ax.get_xlim()),
                                 (ax.get_yscale(), ax.yaxis, ax.get_ylim())):
            if scale not in {"log", "symlog"}:
                continue
            axis.set_major_formatter(fmt)
            lo, hi = abs(lim[0]), abs(lim[1])
            dec = np.log10(max(hi, 1e-12) / max(lo, 1e-12)) if lo > 0 else 99.0
            axis.set_minor_formatter(fmt if dec < 1.6 else NullFormatter())


def tidy(ax):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def crossing_span(block: pd.DataFrame, xkey: str) -> float:
    crossings = []
    for _, rows in block.groupby("dataset"):
        rows = rows.sort_values(xkey)
        x = rows[xkey].to_numpy(dtype=float)
        y = rows["p_controllable"].to_numpy(dtype=float)
        hit = np.nonzero(y >= THRESHOLD)[0]
        if hit.size == 0 or hit[0] == 0:
            continue
        i = hit[0]
        x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
        if not (np.isfinite(x0) and x0 > 0 and np.isfinite(x1) and x1 > 0) or y1 == y0:
            continue
        lx = np.log10(x0) + (THRESHOLD - y0) / (y1 - y0) * (np.log10(x1) - np.log10(x0))
        crossings.append(10.0 ** lx)
    if len(crossings) < 2:
        return float("nan")
    return float(max(crossings) / min(crossings))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--baseline", type=Path, default=Path("results/e1_full_v2"))
    parser.add_argument("--scatter-seed", default="e1_mix_s00")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    root = args.results / args.scatter_seed / "E1_scale_boundary_new"
    mix_summary = pd.read_csv(root / "summary.csv")
    quality = json.loads((root / "collapse_quality.json").read_text())
    var_margin = quality["margin_axis"]["mean_between_dataset_variance"]
    var_N = quality["physical_N_axis"]["mean_between_dataset_variance"]
    ratio = quality["variance_ratio_margin_over_N"]

    base_summary = pd.read_csv(args.baseline / "E1_scale_boundary_new" / "summary.csv")
    base_block = base_summary[base_summary["arm"] == "data_coupled"].dropna(
        subset=["margin", "N", "p_controllable"])
    base_span_N = crossing_span(base_block, "N")

    usable = set(mix_summary[mix_summary["arm"] == "data_coupled"].dropna(
        subset=["margin", "N", "p_controllable"])["dataset"])
    block_all = mix_summary[(mix_summary["arm"] == "data_coupled")
                            & mix_summary["dataset"].isin(usable)]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), constrained_layout=True)
    panels = (
        (axes[0], "margin",
         "a  Effective-margin axis  $N_{eff}/N^{*}_{eff}$",
         "Effective margin  $N_{eff}/N^{*}_{eff}$", var_margin),
        (axes[1], "N",
         "b  The same curves against physical fleet size  $N$",
         "Physical fleet size  $N$", var_N),
    )

    n_curves = 0
    spans = {}
    for ax, xkey, title, xlabel, var in panels:
        block = block_all.dropna(subset=[xkey, "p_controllable"])
        groups = list(block.groupby("dataset"))
        n_curves = len(groups)
        spans[xkey] = crossing_span(block, xkey)
        for _, rows in groups:
            rows = rows.sort_values(xkey)
            ax.plot(rows[xkey], rows["p_controllable"], color=C_COUPLED,
                    alpha=0.22, linewidth=0.9, solid_capstyle="round")
        ax.axhline(THRESHOLD, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(0.015, THRESHOLD - 0.02, f"controllability threshold {THRESHOLD:g}",
                transform=ax.get_yaxis_transform(), ha="left", va="top",
                fontsize=9.4, color=MUTED)
        if xkey == "margin":
            ax.axvline(1.0, color=INK, linewidth=0.9, linestyle=(0, (1, 2)))
            ax.text(1.07, 0.035, "$N_{eff}=N^{*}_{eff}$", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=9.4, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel + "   [log]")
        ax.set_ylabel("Controllable fraction  $p_{ctrl}$")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(title, loc="left")
        tidy(ax)
        ax.text(0.015, 0.975, f"{n_curves} curves · same set on both axes",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.6, color=MUTED)
        ax.text(0.985, 0.30,
                f"between-dataset variance of $p_{{ctrl}}$\n{var:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10.0, color=C_COUPLED, linespacing=1.5)

    axes[1].text(0.985, 0.15,
                 f"threshold crossings span {spans['N']:.1f}×\n"
                 f"15-dataset baseline here: {base_span_N:.1f}×",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=10.0, color=C_BASE, linespacing=1.5)

    fig.suptitle(
        "Mixing the datasets removes the between-dataset spread on the physical axis\n"
        f"— nothing left for the collapse coordinate to absorb   (variance ratio {ratio:.2f})",
        fontsize=13.2, fontweight="bold", ha="left", x=0.008, linespacing=1.45)

    plain_log_ticks(fig)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out / 'figure_E1_mix_de'}.{ext}", dpi=300)
    plt.close(fig)

    info = {
        "scatter_seed": args.scatter_seed,
        "n_curves": n_curves,
        "variance_margin_axis": var_margin,
        "variance_physical_N_axis": var_N,
        "variance_ratio": ratio,
        "matched": quality["comparison_is_matched"],
        "threshold_crossing_span_margin": spans["margin"],
        "threshold_crossing_span_N": spans["N"],
        "baseline_threshold_crossing_span_N": base_span_N,
    }
    (args.out / "figure_E1_mix_de_stats.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
