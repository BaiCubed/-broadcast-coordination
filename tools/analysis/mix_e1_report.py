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
C_DECOUP = "#7f8c8d"
C_BASE = "#c0562a"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d4"
SURF = "#fcfcfb"

THRESHOLD = 0.90
plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
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


def median(values):
    values = sorted(v for v in values if np.isfinite(v))
    return float(np.median(values)) if values else float("nan")


def boot_ci(values, n=10000, seed=20260813):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_batch(root: Path) -> dict | None:
    out = root / "E1_scale_boundary_new"
    status_path = root / "run_status.json"
    if not (out / "collapse_quality.json").is_file() or not status_path.is_file():
        return None
    status = json.loads(status_path.read_text())["experiments"]["E1"]
    if status.get("status") != "completed":
        return None
    quality = json.loads((out / "collapse_quality.json").read_text())
    datasets = pd.read_csv(out / "dataset_summary.csv")
    neff = pd.read_csv(out / "neff_boundaries.csv")
    summary = pd.read_csv(out / "summary.csv")

    reached = neff[(neff["arm"] == "data_coupled") & (neff["status"] == "reached")]
    ratios = {}
    for arm in ("data_coupled", "decoupled"):
        block = summary[(summary["arm"] == arm) & (summary["N"] > 0)]
        ratios[arm] = median((block["N_eff"] / block["N"]).tolist())

    return {
        "name": root.name,
        "n_datasets": int(status.get("completed_datasets", 0)),
        "ratio": quality["variance_ratio_margin_over_N"],
        "margin_var": quality["margin_axis"]["mean_between_dataset_variance"],
        "n_var": quality["physical_N_axis"]["mean_between_dataset_variance"],
        "cells": quality["cells"],
        "collapses": quality["collapses"],
        "matched": quality["comparison_is_matched"],
        "n_reached": int(len(reached)),
        "n_star_med": median(reached["N_star_eff"].tolist()),
        "beta_coupled": median(datasets["beta_data_coupled"].dropna().tolist()),
        "beta_decoupled": median(datasets["beta_decoupled"].dropna().tolist()),
        "neff_coupled": ratios["data_coupled"],
        "neff_decoupled": ratios["decoupled"],
    }


def seed_group(name: str) -> str:
    idx = int(name.replace("e1_mix_s", ""))
    if idx < 10:
        return "same mixture (0–9)"
    if idx < 20:
        return "largest share +15 pp (10–19)"
    return "largest share −15 pp (20–29)"


def figure_collapse(mix_summary, base_summary, batches, base, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18.4, 10.2), constrained_layout=True)

    panels = (
        (axes[0, 0], base_summary, "margin", C_BASE,
         "a  Baseline · 15 public datasets, effective-margin axis",
         "Effective margin  $N_{eff}/N^{*}_{eff}$"),
        (axes[0, 1], base_summary, "N", C_BASE,
         "b  Baseline · the same points against physical fleet size",
         "Physical fleet size  $N$"),
        (axes[1, 0], mix_summary, "margin", C_COUPLED,
         "d  Mixtures · 119 combinations, effective-margin axis",
         "Effective margin  $N_{eff}/N^{*}_{eff}$"),
        (axes[1, 1], mix_summary, "N", C_COUPLED,
         "e  Mixtures · the same points against physical fleet size",
         "Physical fleet size  $N$"),
    )
    matched = {}
    for table in (base_summary, mix_summary):
        block = table[table["arm"] == "data_coupled"].dropna(subset=["margin", "N", "p_controllable"])
        matched[id(table)] = set(block["dataset"])

    for ax, table, xkey, colour, title, xlabel in panels:
        block = table[(table["arm"] == "data_coupled")
                      & table["dataset"].isin(matched[id(table)])].dropna(subset=[xkey, "p_controllable"])
        groups = list(block.groupby("dataset"))
        alpha = 0.75 if len(groups) < 30 else 0.22
        width = 1.4 if len(groups) < 30 else 0.7
        for _, rows in groups:
            rows = rows.sort_values(xkey)
            ax.plot(rows[xkey], rows["p_controllable"], color=colour,
                    alpha=alpha, linewidth=width, solid_capstyle="round")
        ax.axhline(THRESHOLD, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(0.985, THRESHOLD - 0.045, f"controllability threshold {THRESHOLD:g}",
                transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=8.6, color=MUTED)
        if xkey == "margin":
            ax.axvline(1.0, color=INK, linewidth=0.9, linestyle=(0, (1, 2)))
            ax.text(1.06, 0.045, "$N_{eff}=N^{*}_{eff}$", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=8.6, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel + "   [log]")
        ax.set_ylabel("Controllable fraction  $p_{ctrl}$")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(title, loc="left")
        tidy(ax)
        n_curves = len(groups)
        ax.text(0.015, 0.965, f"{n_curves} curves · same set on both axes",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.8, color=MUTED)

    ax = axes[0, 2]
    order = sorted(batches, key=lambda b: int(b["name"].replace("e1_mix_s", "")))
    idx = np.arange(len(order))
    vals = np.array([b["ratio"] for b in order])
    top = max(vals.max() * 1.32, 1.6)
    ax.axhspan(0, 1.0, color=C_BASE, alpha=0.07)
    ax.bar(idx, vals, color=C_COUPLED, width=0.66, zorder=3,
           label="mixtures · 119 combinations")
    ax.axhline(1.0, color=INK, linewidth=1.0, zorder=5)
    ax.text(len(order) - 0.4, 1.04, "collapse holds only below 1", ha="right", va="bottom",
            fontsize=8.8, color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc=SURF, ec="none", alpha=0.85))
    ax.axhline(base["ratio"], color=C_BASE, linewidth=1.8, linestyle=(0, (5, 3)), zorder=5,
               label=f"baseline · 15 public datasets  ({base['ratio']:.2f})")
    ax.legend(loc="upper left", fontsize=9.0, bbox_to_anchor=(0.0, 0.90))
    lo, hi = boot_ci(vals)
    ax.set_title("c  Collapse criterion, all 30 mixture seeds", loc="left")
    ax.set_xlabel("mixture seed")
    ax.set_ylabel("Variance ratio   margin axis / $N$ axis")
    ax.set_xticks(idx[::3])
    ax.set_xticklabels([b["name"].replace("e1_mix_s", "") for b in order][::3])
    ax.set_ylim(0, top * 1.12)
    ax.text(0.5, 0.985,
            f"30 of 30 seeds fail  ·  median {np.median(vals):.2f}  "
            f"·  range {vals.min():.2f}–{vals.max():.2f}\nmean 95% CI [{lo:.2f}, {hi:.2f}]",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.4, color=INK)
    tidy(ax)

    ax = axes[1, 2]
    labels = ["margin axis\n(numerator)", "physical $N$ axis\n(denominator)"]
    pos = np.arange(2)
    w = 0.34
    base_vals = [base["margin_var"], base["n_var"]]
    mix_m = [b["margin_var"] for b in order]
    mix_n = [b["n_var"] for b in order]
    mix_vals = [np.mean(mix_m), np.mean(mix_n)]
    mix_err = np.array([
        [mix_vals[0] - boot_ci(mix_m)[0], mix_vals[1] - boot_ci(mix_n)[0]],
        [boot_ci(mix_m)[1] - mix_vals[0], boot_ci(mix_n)[1] - mix_vals[1]],
    ])
    ax.bar(pos - w / 2, base_vals, width=w, color=C_BASE, label="baseline · 15 public datasets", zorder=3)
    ax.bar(pos + w / 2, mix_vals, width=w, color=C_COUPLED, label="mixtures · 119 combinations (30 seeds)",
           yerr=mix_err, capsize=4, error_kw={"lw": 1.1, "ecolor": INK}, zorder=3)
    for x, v in zip(pos - w / 2, base_vals):
        ax.text(x, v * 1.06, f"{v:.4f}", ha="center", va="bottom", fontsize=8.8, color=C_BASE)
    for x, v, e in zip(pos + w / 2, mix_vals, mix_err[1]):
        ax.text(x + w * 0.62, v + e, f"{v:.4f}", ha="left", va="center",
                fontsize=8.8, color=C_COUPLED)
    ax.annotate("", xy=(pos[0] + w / 2, mix_vals[0] * 1.02), xytext=(pos[0] - w / 2, base_vals[0] * 1.02),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(pos[0], max(base_vals[0], mix_vals[0]) * 1.55,
            f"×{mix_vals[0] / base_vals[0]:.1f}  noise added\nby estimated $N^{{*}}$",
            ha="center", va="bottom", fontsize=9.0, color=INK)
    ax.annotate("", xy=(pos[1] + w / 2, mix_vals[1] * 1.02), xytext=(pos[1] - w / 2, base_vals[1] * 1.02),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1,
                                connectionstyle="arc3,rad=0.28"))
    ax.text(pos[1], base_vals[1] * 1.25,
            f"÷{base_vals[1] / mix_vals[1]:.1f}  between-dataset\nspread averaged away",
            ha="center", va="bottom", fontsize=9.0, color=INK)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean between-dataset variance of $p_{ctrl}$")
    ax.set_title("f  Why the ratio flips: numerator up, denominator gone", loc="left")
    ax.set_ylim(0, max(base_vals + mix_vals) * 2.0)
    ax.legend(loc="upper center", fontsize=9.0, bbox_to_anchor=(0.5, 0.88))
    tidy(ax)

    fig.suptitle(
        "Mixing the datasets removes the heterogeneity the collapse claim rests on — "
        "the physics is unchanged, the criterion is not",
        fontsize=13.5, x=0.006, ha="left", fontweight="bold")
    plain_log_ticks(fig)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def figure_physics(batches, base, mix_summary, base_summary, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18.4, 9.8), constrained_layout=True)
    order = sorted(batches, key=lambda b: int(b["name"].replace("e1_mix_s", "")))
    idx = np.arange(len(order))

    def seed_panel(ax, key_c, key_d, base_c, base_d, title, ylabel, ylim=None):
        vc = np.array([b[key_c] for b in order])
        vd = np.array([b[key_d] for b in order]) if key_d else None
        lo, hi = boot_ci(vc)
        ax.axhspan(lo, hi, color=C_COUPLED, alpha=0.12, zorder=1)
        ax.axhline(base_c, color=C_BASE, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
        ax.scatter(idx, vc, s=34, color=C_COUPLED, marker="^", zorder=4,
                   label="data-coupled (real alignment kept)")
        if vd is not None:
            ax.axhline(base_d, color=C_BASE, linewidth=1.1, linestyle=(0, (2, 3)), zorder=2)
            ax.scatter(idx, vd, s=30, color=C_DECOUP, marker="o", zorder=3,
                       label="decoupled (marginals preserved)")
        ax.set_title(title, loc="left")
        ax.set_xlabel("mixture seed")
        ax.set_ylabel(ylabel)
        ax.set_xticks(idx[::5])
        ax.set_xticklabels([b["name"].replace("e1_mix_s", "") for b in order][::5])
        ax.set_xlim(-1.2, len(order))
        if ylim:
            ax.set_ylim(*ylim)
        tidy(ax)
        return vc, vd, (lo, hi)

    vc, vd, ci = seed_panel(
        axes[0, 0], "beta_coupled", "beta_decoupled", base["beta_coupled"], base["beta_decoupled"],
        "a  Scaling exponent $\\beta$ of the aggregate error",
        "$\\beta$  (median over combinations)", ylim=(-0.52, -0.48))
    axes[0, 0].axhline(-0.5, color=INK, linewidth=0.9, linestyle=(0, (1, 2)), zorder=2)
    axes[0, 0].text(len(order) - 0.5, -0.4975, "CLT  $-1/2$", ha="right", va="bottom",
                    fontsize=9.0, color=INK)
    axes[0, 0].text(0.5, 0.055,
                    f"mixtures {np.median(vc):.4f}    baseline {base['beta_coupled']:.4f}\n"
                    f"all 30 seeds within {np.abs(vc + 0.5).max() * 1e3:.2f}×10$^{{-3}}$ of $-1/2$",
                    transform=axes[0, 0].transAxes, ha="center", va="bottom", fontsize=9.2, color=INK)
    axes[0, 0].legend(loc="upper center", fontsize=8.8, ncol=2)

    vc, vd, ci = seed_panel(
        axes[0, 1], "neff_coupled", "neff_decoupled", base["neff_coupled"], base["neff_decoupled"],
        "b  Effective independent fraction $N_{eff}/N$",
        "$N_{eff}/N$  (median over cells)", ylim=(0.62, 1.06))
    axes[0, 1].text(0.5, 0.52,
                    f"mixtures {np.median(vc):.3f}    baseline {base['neff_coupled']:.3f}\n"
                    f"mean 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
                    transform=axes[0, 1].transAxes, ha="center", va="bottom", fontsize=9.2, color=INK)
    axes[0, 1].legend(loc="lower center", fontsize=8.8, ncol=2)

    vc, _, ci = seed_panel(
        axes[0, 2], "n_star_med", None, base["n_star_med"], None,
        "c  Effective critical size $N^{*}_{eff}$", "$N^{*}_{eff}$  (median over combinations)")
    axes[0, 2].set_ylim(455, 540)
    axes[0, 2].text(0.025, 0.03,
                    f"mixtures {np.median(vc):.0f}    baseline {base['n_star_med']:.0f}"
                    f"    ({np.median(vc) / base['n_star_med'] - 1:+.1%})\n"
                    f"mean 95% CI [{ci[0]:.0f}, {ci[1]:.0f}]",
                    transform=axes[0, 2].transAxes, ha="left", va="bottom", fontsize=9.2, color=INK)
    axes[0, 2].legend(loc="upper center", fontsize=8.8)

    ax = axes[1, 0]
    groups = ["same mixture (0–9)", "largest share +15 pp (10–19)", "largest share −15 pp (20–29)"]
    keys = [("beta_coupled", "$\\beta$"), ("neff_coupled", "$N_{eff}/N$"),
            ("n_star_med", "$N^{*}_{eff}$"), ("ratio", "variance ratio\n(the criterion)")]
    members = {grp: [b for b in order if seed_group(b["name"]) == grp] for grp in groups}
    ref = {k: np.mean([abs(b[k]) for b in members[groups[0]]]) for k, _ in keys}
    width = 0.26
    shades = ["#1b4f72", "#5187b0", "#9dc0d8"]
    pos = np.arange(len(keys))
    for gi, (grp, shade) in enumerate(zip(groups, shades)):
        vals_by_key = [[abs(b[k]) / ref[k] * 100 - 100 for b in members[grp]] for k, _ in keys]
        means = [float(np.mean(v)) for v in vals_by_key]
        cis = [boot_ci(v) for v in vals_by_key]
        errs = np.array([[m - c[0] for m, c in zip(means, cis)],
                         [c[1] - m for m, c in zip(means, cis)]])
        ax.bar(pos + (gi - 1) * width, means, width=width * 0.9, color=shade,
               yerr=errs, capsize=3, error_kw={"lw": 1.0, "ecolor": INK}, label=grp, zorder=3)
    ax.axhline(0, color=INK, linewidth=1.0, zorder=4)
    ax.axhspan(-1, 1, color=MUTED, alpha=0.10, zorder=1)
    ax.text(-0.42, 1.6, "±1 %", fontsize=8.6, color=MUTED, va="bottom")
    ax.set_xticks(pos)
    ax.set_xticklabels([lab for _, lab in keys])
    ax.set_ylabel("change vs the same-mixture group  (%)")
    ax.set_title("d  Mixture proportion moves the criterion, not the physics", loc="left")
    ax.legend(fontsize=8.6, loc="upper left", ncol=1)
    ax.set_ylim(-22, 50)
    tidy(ax)

    ax = axes[1, 1]
    frac = np.array([b["n_reached"] / b["n_datasets"] for b in order])
    ax.bar(idx, frac, color=C_COUPLED, width=0.66, zorder=3)
    ax.axhline(base["n_reached"] / base["n_datasets"], color=C_BASE, linewidth=1.5,
               linestyle=(0, (5, 3)), zorder=4)
    ax.text(len(order) - 0.4, base["n_reached"] / base["n_datasets"] + 0.015,
            f"baseline {base['n_reached']}/{base['n_datasets']}", ha="right", va="bottom",
            fontsize=8.8, color=C_BASE, fontweight="bold")
    ax.set_title("e  Combinations that reach the controllability boundary", loc="left")
    ax.set_xlabel("mixture seed")
    ax.set_ylabel("fraction reaching $N^{*}_{eff}$")
    ax.set_xticks(idx[::3])
    ax.set_xticklabels([b["name"].replace("e1_mix_s", "") for b in order][::3])
    ax.set_ylim(0, 1.0)
    mean_reached = np.mean([b["n_reached"] for b in order])
    ax.text(0.5, 0.93,
            f"mixtures {mean_reached:.0f} of 119 on average  ·  right-censored above $N$=3000",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.0, color=INK)
    tidy(ax)

    ax = axes[1, 2]
    dec = mix_summary[(mix_summary["arm"] == "decoupled") & (mix_summary["N"] > 0)].dropna(subset=["N_eff"])
    ax.scatter(dec["N"], dec["N_eff"] / dec["N"], s=9, color=C_DECOUP, marker="s",
               alpha=0.16, linewidths=0, label="mixtures · decoupled arm", zorder=2)
    for table, colour, marker, label, alpha, size, z in (
        (mix_summary, C_COUPLED, "^", "mixtures · 119 combinations (seed 00)", 0.28, 14, 3),
        (base_summary, C_BASE, "o", "baseline · 15 public datasets", 0.85, 22, 4),
    ):
        block = table[(table["arm"] == "data_coupled") & (table["N"] > 0)].dropna(subset=["N_eff"])
        ax.scatter(block["N"], block["N_eff"] / block["N"], s=size, color=colour, marker=marker,
                   alpha=alpha, linewidths=0, label=label, zorder=z)
    ax.set_xscale("log")
    ax.set_xlabel("Physical fleet size  $N$   [log]")
    ax.set_ylabel("$N_{eff}/N$")
    ax.set_title("f  The price real alignment charges, in both populations", loc="left")
    ax.set_ylim(0, 1.06)
    ax.legend(fontsize=8.6, loc="lower left", markerscale=1.6)
    tidy(ax)

    fig.suptitle(
        "Across 119 mixtures and 30 seeds the three physical read-outs reproduce the "
        "15-dataset baseline and are insensitive to mixture proportion",
        fontsize=13.5, x=0.006, ha="left", fontweight="bold")
    plain_log_ticks(fig)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--baseline", type=Path, default=Path("results/e1_full_v2"))
    parser.add_argument("--scatter-seed", default="e1_mix_s00")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    batches = [b for b in (load_batch(d) for d in sorted(args.results.glob("e1_mix_s*"))) if b]
    if not batches:
        raise SystemExit("no completed e1_mix_s* batch found")
    base = load_batch(args.baseline)
    if base is None:
        raise SystemExit(f"cannot read the baseline: {args.baseline}")

    mix_summary = pd.read_csv(args.results / args.scatter_seed / "E1_scale_boundary_new" / "summary.csv")
    base_summary = pd.read_csv(args.baseline / "E1_scale_boundary_new" / "summary.csv")

    figure_collapse(mix_summary, base_summary, batches, base, args.out / "figure_E1_mix_collapse")
    figure_physics(batches, base, mix_summary, base_summary, args.out / "figure_E1_mix_physics")

    frame = pd.DataFrame(batches)
    frame["seed"] = frame["name"].str.replace("e1_mix_s", "", regex=False).astype(int)
    frame["seed_group"] = frame["name"].map(seed_group)
    frame = frame.sort_values("seed")
    frame.to_csv(args.out / "source_data_E1_mix_seeds.csv", index=False)

    ratios = frame["ratio"].to_numpy()
    stats = {
        "n_seeds": int(len(frame)),
        "n_combinations": int(frame["n_datasets"].iloc[0]),
        "all_completed": bool((frame["n_datasets"] == frame["n_datasets"].iloc[0]).all()),
        "collapse": {
            "seeds_failing": int((~frame["collapses"]).sum()),
            "ratio_median": float(np.median(ratios)),
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "ratio_mean_ci95": boot_ci(ratios),
            "baseline_ratio": base["ratio"],
            "all_matched": bool(frame["matched"].all()),
        },
        "physics": {
            key: {
                "mix_median": float(np.median(frame[key])),
                "mix_mean_ci95": boot_ci(frame[key].to_numpy()),
                "baseline": base[key],
            }
            for key in ("beta_coupled", "beta_decoupled", "neff_coupled", "neff_decoupled", "n_star_med")
        },
        "mechanism": {
            "margin_var_mix_mean": float(frame["margin_var"].mean()),
            "margin_var_baseline": base["margin_var"],
            "n_var_mix_mean": float(frame["n_var"].mean()),
            "n_var_baseline": base["n_var"],
        },
        "by_seed_group": {
            grp: {
                key: float(np.median(frame[frame["seed_group"] == grp][key]))
                for key in ("ratio", "beta_coupled", "neff_coupled", "n_star_med")
            }
            for grp in frame["seed_group"].unique()
        },
    }
    (args.out / "mix_e1_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
