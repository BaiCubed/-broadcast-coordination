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
INK, MUTED, GRID, SURF = "#1a1a1a", "#6b6b6b", "#d8d8d4", "#fcfcfb"
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
        hit = np.flatnonzero(y >= THRESHOLD)
        if hit.size == 0:
            continue
        index = int(hit[0])
        if index == 0:
            crossings.append(x[0])
            continue
        x0, x1, y0, y1 = x[index - 1], x[index], y[index - 1], y[index]
        if y1 == y0:
            crossings.append(x1)
        else:
            weight = (THRESHOLD - y0) / (y1 - y0)
            crossings.append(float(np.exp(np.log(x0) + weight * (np.log(x1) - np.log(x0)))))
    if len(crossings) < 2:
        return float("nan")
    return float(max(crossings) / min(crossings))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--arms", nargs="+", required=True,
                        help="e.g. ieee33=e1_topo_ieee33_fb")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="E1 controllability boundary on three standard feeders (network feedback ON)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arms = [tuple(item.split("=", 1)) for item in args.arms]
    fig, axes = plt.subplots(2, len(arms), figsize=(6.6 * len(arms), 11.4),
                             constrained_layout=True)
    info = {}
    for column, (label, folder) in enumerate(arms):
        root = args.results / folder / "E1_scale_boundary_new"
        summary = pd.read_csv(root / "summary.csv")
        quality = json.loads((root / "collapse_quality.json").read_text(encoding="utf-8"))
        usable = set(summary[summary["arm"] == "data_coupled"].dropna(
            subset=["margin", "N", "p_controllable"])["dataset"])
        block_all = summary[(summary["arm"] == "data_coupled")
                            & summary["dataset"].isin(usable)]
        boundaries = pd.read_csv(root / "neff_boundaries.csv")
        reached = boundaries.dropna(subset=["N_star_eff"])
        panels = (
            (axes[0, column], "margin",
             f"{'abcdefgh'[column]}  {label} — effective-margin axis  $N_{{eff}}/N^{{*}}_{{eff}}$",
             "Effective margin  $N_{eff}/N^{*}_{eff}$",
             quality["margin_axis"]["mean_between_dataset_variance"]),
            (axes[1, column], "N",
             f"{'ijklmnop'[column]}  {label} — the same curves against physical fleet size $N$",
             "Physical fleet size  $N$",
             quality["physical_N_axis"]["mean_between_dataset_variance"]),
        )
        spans = {}
        for ax, xkey, title, xlabel, variance in panels:
            block = block_all.dropna(subset=[xkey, "p_controllable"])
            groups = list(block.groupby("dataset"))
            spans[xkey] = crossing_span(block, xkey)
            for _, rows in groups:
                rows = rows.sort_values(xkey)
                ax.plot(rows[xkey], rows["p_controllable"], color=C_COUPLED,
                        alpha=0.30, linewidth=1.1, solid_capstyle="round")
            ax.axhline(THRESHOLD, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
            ax.text(0.015, THRESHOLD - 0.02, f"controllability threshold {THRESHOLD:g}",
                    transform=ax.get_yaxis_transform(), ha="left", va="top",
                    fontsize=9.4, color=MUTED)
            if xkey == "margin":
                ax.axvline(1.0, color=INK, linewidth=0.9, linestyle=(0, (1, 2)))
                ax.text(1.07, 0.035, "$N_{eff}=N^{*}_{eff}$",
                        transform=ax.get_xaxis_transform(), ha="left", va="bottom",
                        fontsize=9.4, color=MUTED)
            ax.set_xscale("log")
            ax.set_xlabel(xlabel + "   [log]")
            ax.set_ylabel("Controllable fraction  $p_{ctrl}$")
            ax.set_ylim(-0.03, 1.05)
            ax.set_title(title, loc="left")
            tidy(ax)
            ax.text(0.015, 0.975, f"{len(groups)} curves · same set on both axes",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9.6, color=MUTED)
            ax.text(0.985, 0.30,
                    f"between-dataset variance of $p_{{ctrl}}$\n{variance:.4f}"
                    if variance is not None else "between-dataset variance NA",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10.0, color=C_COUPLED, linespacing=1.5)
            ax.text(0.985, 0.14, f"threshold crossings span {spans[xkey]:.1f}\u00d7",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10.0, color=C_BASE)
        info[label] = {
            "results_folder": folder,
            "datasets_usable": len(usable),
            "variance_margin_axis": quality["margin_axis"]["mean_between_dataset_variance"],
            "variance_physical_N_axis": quality["physical_N_axis"]["mean_between_dataset_variance"],
            "variance_ratio_margin_over_N": quality["variance_ratio_margin_over_N"],
            "comparison_is_matched": quality["comparison_is_matched"],
            "threshold_crossing_span_margin": spans["margin"],
            "threshold_crossing_span_N": spans["N"],
            "N_star_eff_median": float(reached["N_star_eff"].median()) if len(reached) else None,
            "N_star_eff_min": float(reached["N_star_eff"].min()) if len(reached) else None,
            "N_star_eff_max": float(reached["N_star_eff"].max()) if len(reached) else None,
            "datasets_reached": int(len(reached)),
            "datasets_total": int(len(boundaries)),
        }
    ratios = " | ".join(
        f"{label} {info[label]['variance_ratio_margin_over_N']:.2f}"
        if info[label]["variance_ratio_margin_over_N"] is not None else f"{label} NA"
        for label, _ in arms
    )
    fig.suptitle(f"{args.title}\nvariance ratio (collapse coordinate / physical N):   {ratios}",
                 fontsize=13.6, fontweight="bold", ha="left", x=0.008, linespacing=1.45)
    plain_log_ticks(fig)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"figure_E1_topology_compare.{ext}", dpi=300)
    plt.close(fig)
    (args.out / "figure_E1_topology_compare_stats.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
