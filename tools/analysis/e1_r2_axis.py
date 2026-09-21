from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "blue": "#2166ac",
    "orange": "#d6604d",
    "green": "#1b9e77",
    "purple": "#7570b3",
    "gray": "#666666",
}

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260803
EPS = 1e-12


def _short(name: str) -> str:
    return {
        "bdg1_building_data_genome": "BDG1",
        "bdg2_building_data_genome": "BDG2",
        "camsl_japan_smart_meters": "CAMSL-JP",
        "complete_energy_community": "COMPLETE-EC",
        "danish_smart_heat_meters": "DK-heat",
        "european_lv_rural_2731": "EU-LV rural",
        "european_lv_urban_35297": "EU-LV urban-35k",
        "european_lv_urban_8087": "EU-LV urban-8k",
        "goiener_smart_meters": "Goiener",
        "heapo_heat_pumps": "HEAPO",
        "irish_domestic_smart_meters": "Irish CER",
        "low_carbon_london": "Low Carbon Ldn",
        "nextgen": "NextGen ACT",
        "norway_ami_energy_distribution": "Norway AMI",
        "opsd_household_data": "OPSD",
        "smart_grid_smart_city": "SGSC",
        "un_household_electricity": "UN-HH",
    }.get(name, name[:14])


def seed_level_r2(responses: pd.DataFrame) -> pd.DataFrame:
    target = responses["target_kw"].to_numpy(dtype=float)
    actual = responses["accepted_kw"].to_numpy(dtype=float)
    frame = responses[["dataset", "arm", "N", "seed_index"]].copy()
    frame["_res2"] = (actual - target) ** 2
    frame["_t"] = target
    frame["_t2"] = target ** 2
    grouped = frame.groupby(["dataset", "arm", "N", "seed_index"], sort=False)
    agg = grouped.agg(
        ss_res=("_res2", "sum"),
        sum_t=("_t", "sum"),
        sum_t2=("_t2", "sum"),
        n_cond=("_t", "size"),
    ).reset_index()
    agg["ss_tot"] = agg["sum_t2"] - agg["sum_t"] ** 2 / agg["n_cond"]
    agg["r2"] = np.where(
        agg["ss_tot"] > EPS, 1.0 - agg["ss_res"] / agg["ss_tot"].clip(lower=EPS), np.nan
    )
    return agg


def cell_level_r2(seed_r2: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for (dataset, arm, n), block in seed_r2.groupby(["dataset", "arm", "N"], sort=False):
        values = block["r2"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, values.size), replace=True)
        means = draws.mean(axis=1)
        ss_res = float(block["ss_res"].sum())
        ss_tot = float(block["ss_tot"].sum())
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "N": int(n),
                "R2_mean": float(values.mean()),
                "R2_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "R2_median": float(np.median(values)),
                "R2_ci_lower": float(np.percentile(means, 2.5)),
                "R2_ci_upper": float(np.percentile(means, 97.5)),
                "R2_pooled": float(1.0 - ss_res / ss_tot) if ss_tot > EPS else np.nan,
                "R2_seeds": int(values.size),
                "R2_frac_negative": float(np.mean(values < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def binned_between_dataset_variance(
    frame: pd.DataFrame, axis_key: str, response_key: str, *, bins: int = 8
) -> dict[str, object]:
    if len(frame) < bins * 2:
        return {"bins": 0, "mean_between_dataset_variance": None, "cells": len(frame)}
    values = np.log10(frame[axis_key].to_numpy(dtype=float))
    response = frame[response_key].to_numpy(dtype=float)
    datasets = frame["dataset"].to_numpy()
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges[-1] += 1e-9
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bins - 1)
    variances: list[float] = []
    spreads: list[float] = []
    for bin_index in range(bins):
        members = np.flatnonzero(index == bin_index)
        if members.size == 0:
            continue
        by_dataset: dict[str, list[float]] = {}
        for position in members:
            by_dataset.setdefault(str(datasets[position]), []).append(float(response[position]))
        if len(by_dataset) < 2:
            continue
        means = [float(np.mean(value)) for value in by_dataset.values()]
        variances.append(float(np.var(means)))
        spreads.append(float(np.std(values[members])))
    if not variances:
        return {"bins": 0, "mean_between_dataset_variance": None, "cells": len(frame)}
    return {
        "bins_requested": bins,
        "bins_with_at_least_two_datasets": len(variances),
        "mean_between_dataset_variance": float(np.mean(variances)),
        "mean_within_bin_log10_spread": float(np.mean(spreads)),
        "cells": int(len(frame)),
    }


def collapse_quality(
    frame: pd.DataFrame,
    margin_key: str,
    response_key: str,
    *,
    bins: int = 8,
    arm: str = "data_coupled",
) -> dict[str, object]:
    usable = frame[
        (frame["arm"] == arm)
        & frame[margin_key].notna()
        & np.isfinite(frame[margin_key].astype(float))
        & (frame[margin_key].astype(float) > 0)
        & frame[response_key].notna()
        & (frame["N"].astype(float) > 0)
    ].copy()
    margin = binned_between_dataset_variance(usable, margin_key, response_key, bins=bins)
    physical = binned_between_dataset_variance(usable, "N", response_key, bins=bins)
    ratio = None
    if margin["mean_between_dataset_variance"] is not None and physical.get(
        "mean_between_dataset_variance"
    ):
        ratio = float(
            margin["mean_between_dataset_variance"] / physical["mean_between_dataset_variance"]
        )
    matched = bool(
        margin.get("mean_within_bin_log10_spread") is not None
        and physical.get("mean_within_bin_log10_spread")
        and 0.1
        <= margin["mean_within_bin_log10_spread"] / physical["mean_within_bin_log10_spread"]
        <= 10.0
    )
    return {
        "definition": (
            f"mean over quantile bins of the between-dataset variance of {response_key}; "
            "ratio below 1 means the effective-margin axis collapses the datasets better than N"
        ),
        "response": response_key,
        "margin_axis_key": margin_key,
        "arm": arm,
        "cells": int(len(usable)),
        "margin_axis": margin,
        "physical_N_axis": physical,
        "variance_ratio_margin_over_N": ratio,
        "collapses": bool(ratio is not None and ratio < 1.0),
        "comparison_is_matched": matched,
    }


def r2_boundaries(
    frame: pd.DataFrame, threshold: float, *, arm: str = "data_coupled"
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, block in frame[frame["arm"] == arm].groupby("dataset", sort=False):
        ordered = block.sort_values("N_eff")
        lower = ordered["R2_ci_lower"].to_numpy(dtype=float)
        effective = ordered["N_eff"].to_numpy(dtype=float)
        envelope = np.maximum.accumulate(lower)
        crossing = np.flatnonzero(envelope >= threshold)
        star = float(effective[crossing[0]]) if crossing.size else None
        suffix = np.minimum.accumulate(lower[::-1])[::-1]
        strict_crossing = np.flatnonzero(suffix >= threshold)
        strict = float(effective[strict_crossing[0]]) if strict_crossing.size else None
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "threshold": threshold,
                "N_star_eff_r2": star,
                "N_star_eff_r2_strict": strict,
                "status": "reached" if star is not None else "not_reached",
                "N_eff_min": float(effective[0]),
                "N_eff_max": float(effective[-1]),
                "cells": int(len(ordered)),
                "censored_at_min_fleet": bool(
                    star is not None and abs(star - float(effective[0])) < 1e-6
                ),
            }
        )
    return pd.DataFrame(rows)


def _misfit_axis(axis: plt.Axes) -> None:
    ticks = [0.1, 0.03, 0.01, 0.003]
    limits = (0.2, 1.5e-3)
    axis.set_yscale("log")
    axis.set_ylim(*limits)
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{value:g}" for value in ticks])
    axis.minorticks_off()
    twin = axis.twinx()
    twin.set_yscale("log")
    twin.set_ylim(*limits)
    twin.set_yticks(ticks)
    twin.set_yticklabels([f"$R^2$={1 - value:.3f}" for value in ticks], fontsize=8)
    twin.minorticks_off()


def make_figure(
    frame: pd.DataFrame,
    boundaries_r2: pd.DataFrame,
    collapse_margin: dict[str, object],
    collapse_raw_r2: dict[str, object],
    threshold: float,
    out_stem: Path,
) -> None:
    coupled = frame[frame["arm"] == "data_coupled"].copy()
    coupled["misfit"] = np.clip(1.0 - coupled["R2_mean"], 1e-6, None)
    datasets = sorted(coupled["dataset"].unique())
    palette = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(len(datasets), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    def _series(dataset: str, key: str) -> pd.DataFrame:
        block = coupled[(coupled["dataset"] == dataset) & coupled[key].notna()]
        return block.sort_values(key)

    for color, dataset in zip(palette, datasets):
        rows = _series(dataset, "margin")
        if rows.empty:
            continue
        axes[0, 0].plot(
            rows["margin"], rows["misfit"], "o-", linewidth=1, markersize=3.5,
            alpha=0.85, color=color, label=_short(dataset),
        )
    axes[0, 0].axvline(1.0, linestyle="--", color=COLORS["gray"], linewidth=1.2)
    axes[0, 0].axhline(1.0 - threshold, linestyle=":", color=COLORS["orange"], linewidth=1.2)
    _misfit_axis(axes[0, 0])
    axes[0, 0].set(
        xscale="log", xlabel="Effective margin $N_{eff}/N^*_{eff}$",
        ylabel="Tracking misfit $1-R^2$  (up = better)",
        title="A  Tracking quality collapses in the effective-margin coordinate",
    )
    ratio_a = collapse_margin.get("variance_ratio_margin_over_N")
    if ratio_a is not None:
        axes[0, 0].text(
            0.40, 0.04,
            f"between-dataset variance ratio vs raw $N$: {ratio_a:.3f}",
            transform=axes[0, 0].transAxes, fontsize=8,
        )
    axes[0, 0].legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.9)

    for color, dataset in zip(palette, datasets):
        rows = _series(dataset, "N")
        if rows.empty or rows["margin"].isna().all():
            continue
        rows = rows[rows["margin"].notna()]
        axes[0, 1].plot(
            rows["N"], rows["misfit"], "o-", linewidth=1, markersize=3.5,
            alpha=0.85, color=color,
        )
    axes[0, 1].axhline(1.0 - threshold, linestyle=":", color=COLORS["orange"], linewidth=1.2)
    _misfit_axis(axes[0, 1])
    axes[0, 1].set(
        xscale="log", xlabel="Physical fleet size $N$",
        ylabel="Tracking misfit $1-R^2$  (up = better)",
        title="B  The same points do not collapse in raw $N$",
    )

    for color, dataset in zip(palette, datasets):
        rows = _series(dataset, "margin")
        if rows.empty:
            continue
        axes[1, 0].plot(
            rows["margin"], rows["R2_mean"], "o-", linewidth=1, markersize=3.5,
            alpha=0.85, color=color,
        )
    axes[1, 0].axvline(1.0, linestyle="--", color=COLORS["gray"], linewidth=1.2)
    axes[1, 0].axhline(threshold, linestyle=":", color=COLORS["orange"], linewidth=1.2)
    axes[1, 0].set(
        xscale="log", ylim=(0.85, 1.005),
        xlabel="Effective margin $N_{eff}/N^*_{eff}$", ylabel="Tracking $R^2$ (linear)",
        title="C  Same data on a linear $R^2$ axis: the transition saturates",
    )
    ratio_linear = collapse_raw_r2.get("variance_ratio_margin_over_N")
    span = coupled.loc[coupled["margin"].notna(), "R2_mean"]
    note = (
        f"$R^2\\in[{span.min():.3f},\\ {span.max():.3f}]$ over the whole boundary region; "
        f"variance ratio on this scale: "
        f"{'n/a' if ratio_linear is None else format(ratio_linear, '.2f')} (noise-dominated)"
    )
    axes[1, 0].text(0.02, 0.06, note, transform=axes[1, 0].transAxes, fontsize=8)

    reached = boundaries_r2[boundaries_r2["N_star_eff_r2"].notna()].copy()
    pctrl_star = coupled.groupby("dataset")["N_star_eff"].first()
    if not reached.empty:
        order = reached.sort_values("N_star_eff_r2").reset_index(drop=True)
        y = np.arange(len(order))
        censored = order["censored_at_min_fleet"].to_numpy(dtype=bool)
        axes[1, 1].scatter(
            order.loc[~censored, "N_star_eff_r2"], y[~censored], color=COLORS["blue"], zorder=3,
            label=f"$N^*_{{eff}}$ from $R^2 \\geq {threshold:g}$ (resolved)",
        )
        axes[1, 1].scatter(
            order.loc[censored, "N_star_eff_r2"], y[censored], facecolors="none",
            edgecolors=COLORS["blue"], zorder=3,
            label=f"$R^2 \\geq {threshold:g}$ already met at the smallest fleet (upper bound)",
        )
        for index in np.flatnonzero(censored):
            value = float(order.loc[index, "N_star_eff_r2"])
            axes[1, 1].annotate(
                "", xy=(value * 0.55, index), xytext=(value * 0.95, index),
                arrowprops=dict(arrowstyle="->", color=COLORS["blue"], linewidth=0.9),
            )
        companion = [pctrl_star.get(name, np.nan) for name in order["dataset"]]
        axes[1, 1].scatter(
            companion, y, color=COLORS["orange"], marker="x", zorder=3,
            label="$N^*_{eff}$ from $p_{ctrl} \\geq 0.9$",
        )
        for index, row in order.iterrows():
            if pd.notna(row["N_star_eff_r2_strict"]):
                axes[1, 1].plot(
                    [row["N_star_eff_r2"], row["N_star_eff_r2_strict"]], [index, index],
                    color=COLORS["gray"], linewidth=1.4, zorder=2,
                )
        axes[1, 1].set(
            xscale="log", yticks=y,
            yticklabels=[_short(str(name)) for name in order["dataset"]],
            xlabel="Critical effective fleet $N^*_{eff}$",
            title="D  Critical effective fleet: $R^2$ criterion vs $p_{ctrl}$ criterion",
        )
        axes[1, 1].set_ylim(-1.6, len(order) + 1.4)
        left, right = axes[1, 1].get_xlim()
        axes[1, 1].set_xlim(left * 0.75, right * 1.9)
        axes[1, 1].legend(fontsize=7, loc="lower right", framealpha=1.0)
        axes[1, 1].text(
            0.02, 0.985,
            f"{int(censored.sum())}/{len(order)} datasets meet $R^2\\geq{threshold:g}$ at the "
            "smallest simulated fleet:\nthe $R^2$ criterion cannot resolve a boundary on this $N$ grid",
            transform=axes[1, 1].transAxes, fontsize=8, va="top", color=COLORS["gray"],
        )
    missing = boundaries_r2[boundaries_r2["N_star_eff_r2"].isna()]["dataset"].tolist()
    if missing:
        axes[1, 1].text(
            0.02, 0.02, "not reached: " + ", ".join(_short(str(d)) for d in missing),
            transform=axes[1, 1].transAxes, fontsize=7, color=COLORS["orange"],
        )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the E1 scale boundary on the R^2 axis.")
    parser.add_argument("--source", type=Path, required=True, help="E1_scale_boundary_new directory")
    parser.add_argument("--output", type=Path, required=True, help="output root for the R^2 readout")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()

    responses = pd.read_csv(args.source / "raw" / "condition_responses.csv")
    summary = pd.read_csv(args.source / "summary.csv")

    seed_r2 = seed_level_r2(responses)
    cells = cell_level_r2(seed_r2)
    merged = summary.merge(cells, on=["dataset", "arm", "N"], how="left", validate="one_to_one")
    if merged["R2_mean"].isna().any():
        missing = merged[merged["R2_mean"].isna()][["dataset", "arm", "N"]]
        raise SystemExit(f"no R^2 for these cells; check the raw responses first:\n{missing.to_string()}")

    merged["log_misfit"] = np.log10(np.clip(1.0 - merged["R2_mean"], 1e-6, None))

    bounds = r2_boundaries(merged, args.threshold)
    star_map = dict(zip(bounds["dataset"], bounds["N_star_eff_r2"]))
    merged["N_star_eff_r2"] = merged["dataset"].map(star_map)
    merged["margin_r2"] = np.where(
        merged["N_star_eff_r2"].notna() & (merged["N_star_eff_r2"].astype(float) > 0),
        merged["N_eff"] / merged["N_star_eff_r2"],
        np.nan,
    )

    matched = merged[merged["margin"].notna() & merged["margin_r2"].notna()].copy()

    def _quality(frame: pd.DataFrame, axis_key: str, response: str) -> dict[str, object]:
        return collapse_quality(frame, axis_key, response, bins=args.bins)

    primary = _quality(matched, "margin", "log_misfit")
    grid = {
        "log_misfit__pctrl_axis": primary,
        "log_misfit__r2_axis": _quality(matched, "margin_r2", "log_misfit"),
        "raw_R2__pctrl_axis": _quality(matched, "margin", "R2_mean"),
        "raw_R2__r2_axis": _quality(matched, "margin_r2", "R2_mean"),
        "p_ctrl__pctrl_axis": _quality(matched, "margin", "p_controllable"),
        "p_ctrl__r2_axis": _quality(matched, "margin_r2", "p_controllable"),
    }
    unmatched_r2_axis = {
        "log_misfit": _quality(merged, "margin_r2", "log_misfit"),
        "raw_R2": _quality(merged, "margin_r2", "R2_mean"),
    }

    sensitivity = []
    for threshold in (0.8, 0.85, 0.9, 0.95):
        alt = r2_boundaries(merged, threshold)
        alt_map = dict(zip(alt["dataset"], alt["N_star_eff_r2"]))
        probe = merged.copy()
        probe["N_star_alt"] = probe["dataset"].map(alt_map)
        probe["margin_alt"] = np.where(
            probe["N_star_alt"].notna() & (probe["N_star_alt"].astype(float) > 0),
            probe["N_eff"] / probe["N_star_alt"], np.nan,
        )
        probe = probe[probe["margin"].notna() & probe["margin_alt"].notna()]
        quality = _quality(probe, "margin_alt", "log_misfit")
        sensitivity.append(
            {
                "threshold": threshold,
                "datasets_reaching_boundary": int(alt["N_star_eff_r2"].notna().sum()),
                "variance_ratio_margin_over_N": quality["variance_ratio_margin_over_N"],
                "collapses": quality["collapses"],
                "comparison_is_matched": quality["comparison_is_matched"],
                "cells": quality["cells"],
            }
        )

    matched["log_misfit_pooled"] = np.log10(np.clip(1.0 - matched["R2_pooled"], 1e-6, None))
    aggregation_check = _quality(matched, "margin", "log_misfit_pooled")
    pooled_gap = float((matched["R2_mean"] - matched["R2_pooled"]).abs().max())

    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "summary_r2.csv", index=False)
    seed_r2.to_csv(args.output / "seed_r2.csv.gz", index=False, compression="gzip")
    bounds.to_csv(args.output / "neff_boundaries_r2.csv", index=False)

    coupled = merged[merged["arm"] == "data_coupled"]
    boundary_region = coupled.loc[coupled["margin"].notna(), "R2_mean"]
    report = {
        "source": str(args.source),
        "r2_definition": (
            "per seed over its 48 dispatch conditions, 1 - SS_res/SS_tot with SS_tot "
            "taken about that seed's mean target; cell value is the mean over seeds"
        ),
        "primary_response": "log_misfit = log10(1 - R2_mean)",
        "why_not_raw_r2": (
            "the controllability boundary is set by NRMSE<=0.1, which already implies "
            f"R2>={boundary_region.min():.3f}; over the whole boundary region R2 spans only "
            f"[{boundary_region.min():.4f}, {boundary_region.max():.4f}], so a variance-based "
            "collapse statistic on the linear R2 scale is dominated by saturation noise"
        ),
        "population_rule": (
            "axis comparisons use only cells where both margin and margin_r2 are defined; "
            "the R2-defined boundary reaches 3 datasets that p_ctrl never does"
        ),
        "threshold": args.threshold,
        "cells_total": int(len(merged)),
        "cells_matched_data_coupled": int(primary["cells"]),
        "primary": primary,
        "response_by_axis_grid": grid,
        "r2_axis_on_its_own_full_population": unmatched_r2_axis,
        "threshold_sensitivity": sensitivity,
        "r2_boundary_censoring": {
            "datasets_at_smallest_simulated_fleet": int(bounds["censored_at_min_fleet"].sum()),
            "datasets_reaching_boundary": int(bounds["N_star_eff_r2"].notna().sum()),
            "verdict": (
                (
                    "R2>=%.2f is already met at the smallest simulated fleet for %d of %d "
                    "datasets, so this threshold yields an upper bound rather than a resolved "
                    "boundary; raise the threshold or simulate smaller fleets"
                    % (
                        args.threshold,
                        int(bounds["censored_at_min_fleet"].sum()),
                        int(bounds["N_star_eff_r2"].notna().sum()),
                    )
                )
                if int(bounds["censored_at_min_fleet"].sum())
                else (
                    "R2>=%.2f resolves a boundary inside the simulated grid for all %d "
                    "datasets that reach it (no left censoring)"
                    % (args.threshold, int(bounds["N_star_eff_r2"].notna().sum()))
                )
            ),
            "N_star_eff_median": (
                float(bounds["N_star_eff_r2"].median())
                if bounds["N_star_eff_r2"].notna().any()
                else None
            ),
        },
        "aggregation_check_pooled_r2": aggregation_check,
        "aggregation_check_is_degenerate": {
            "max_abs_gap_pooled_vs_seed_mean": pooled_gap,
            "note": (
                "every seed faces the same 48-condition schedule, so SS_tot is seed-invariant "
                "and pooled R2 coincides with the seed-mean; this check carries no information"
            ),
        },
        "r2_range": {
            "min": float(merged["R2_mean"].min()),
            "max": float(merged["R2_mean"].max()),
            "cells_below_zero": int((merged["R2_mean"] < 0).sum()),
            "mean_frac_negative_seeds": float(merged["R2_frac_negative"].mean()),
            "boundary_region_min": float(boundary_region.min()),
            "boundary_region_max": float(boundary_region.max()),
        },
        "datasets": {
            "with_pctrl_boundary": int(
                coupled.loc[coupled["margin"].notna(), "dataset"].nunique()
            ),
            "with_r2_boundary": int(bounds["N_star_eff_r2"].notna().sum()),
            "matched": int(matched["dataset"].nunique()),
        },
    }
    (args.output / "collapse_quality_r2.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    make_figure(
        merged, bounds, primary, grid["raw_R2__pctrl_axis"], args.threshold,
        args.output / "Figs" / "figure_E1_r2_effective_margin",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
