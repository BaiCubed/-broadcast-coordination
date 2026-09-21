from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"ours": "#2166ac", "paper": "#b2182b", "grey": "#666666"}
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20260803
EPS = 1e-12
PAPER_N95 = 150.0

SHORT = {
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
    "norway_ami_energy_distribution": "Norway AMI",
    "opsd_household_data": "OPSD",
    "smart_grid_smart_city": "SGSC",
}


def _short(name: str) -> str:
    return SHORT.get(name, name[:14])


def _cell_matrices(responses: pd.DataFrame):
    ordered = responses.sort_values(["dataset", "arm", "N", "condition_id", "seed_index"])
    for (dataset, arm, n), block in ordered.groupby(["dataset", "arm", "N"], sort=False):
        conditions = block["condition_id"].unique()
        seeds = block["seed_index"].unique()
        shape = (len(conditions), len(seeds))
        if len(block) != shape[0] * shape[1]:
            raise SystemExit(f"{dataset}/{arm}/N={n}: the condition x seed matrix is not full, check the raw responses first")
        actual = block["accepted_kw"].to_numpy(dtype=float).reshape(shape)
        yield dataset, arm, int(n), actual


def variance_explained(actual: np.ndarray) -> float:
    within = actual - actual.mean(axis=1, keepdims=True)
    total = actual - actual.mean()
    ss_within = float(np.sum(within ** 2))
    ss_total = float(np.sum(total ** 2))
    return 1.0 - ss_within / ss_total if ss_total > EPS else np.nan


def compute_cells(responses: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for dataset, arm, n, actual in _cell_matrices(responses):
        seeds = actual.shape[1]
        draws = rng.integers(0, seeds, size=(BOOTSTRAP_DRAWS, seeds))
        boot = np.empty(BOOTSTRAP_DRAWS, dtype=float)
        for index in range(BOOTSTRAP_DRAWS):
            boot[index] = variance_explained(actual[:, draws[index]])
        rows.append({
            "dataset": dataset,
            "arm": arm,
            "N": n,
            "R2_signal": variance_explained(actual),
            "R2_signal_ci_lower": float(np.percentile(boot, 2.5)),
            "R2_signal_ci_upper": float(np.percentile(boot, 97.5)),
            "conditions": int(actual.shape[0]),
            "seeds": int(seeds),
        })
    return pd.DataFrame(rows)


def _log_interpolate(scale: np.ndarray, values: np.ndarray, index: int, threshold: float) -> float:
    if index == 0:
        return float(scale[0])
    x0, x1 = np.log10(scale[index - 1]), np.log10(scale[index])
    y0, y1 = values[index - 1], values[index]
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) < EPS:
        return float(scale[index])
    fraction = float(np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0))
    return float(10 ** (x0 + fraction * (x1 - x0)))


def n95_table(frame: pd.DataFrame, threshold: float, *, arm: str = "data_coupled") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, block in frame[frame["arm"] == arm].groupby("dataset", sort=False):
        ordered = block.sort_values("N")
        lower = ordered["R2_signal_ci_lower"].to_numpy(dtype=float)
        point = ordered["R2_signal"].to_numpy(dtype=float)
        sizes = ordered["N"].to_numpy(dtype=float)
        effective = (
            ordered["N_eff"].to_numpy(dtype=float)
            if "N_eff" in ordered else np.full(len(ordered), np.nan)
        )
        envelope = np.maximum.accumulate(lower)
        crossing = np.flatnonzero(envelope >= threshold)
        index = int(crossing[0]) if crossing.size else None
        point_crossing = np.flatnonzero(np.maximum.accumulate(point) >= threshold)
        point_index = int(point_crossing[0]) if point_crossing.size else None
        rows.append({
            "dataset": dataset,
            "N95_grid": float(sizes[index]) if index is not None else None,
            "N95": (_log_interpolate(sizes, point, point_index, threshold)
                    if point_index is not None else None),
            "N95_eff": (_log_interpolate(effective, point, point_index, threshold)
                        if point_index is not None and np.isfinite(effective).all() else None),
            "R2_at_grid_crossing": float(point[index]) if index is not None else None,
            "N_min": float(sizes.min()),
            "N_max": float(sizes.max()),
            "status": "reached" if index is not None else "not_reached",
            "censored_at_min_fleet": bool(index is not None and sizes[index] == sizes.min()),
        })
    return pd.DataFrame(rows)


def make_figure(frame: pd.DataFrame, table: pd.DataFrame, threshold: float, out_stem: Path) -> None:
    coupled = frame[frame["arm"] == "data_coupled"]
    datasets = sorted(coupled["dataset"].unique())
    palette = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(len(datasets), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), constrained_layout=True)

    for color, dataset in zip(palette, datasets):
        block = coupled[coupled["dataset"] == dataset].sort_values("N")
        if block.empty:
            continue
        axes[0].plot(block["N"], block["R2_signal"], "o-", linewidth=1, markersize=3.5,
                     alpha=0.85, color=color, label=_short(dataset))
    axes[0].axhline(threshold, linestyle=":", color=COLORS["paper"], linewidth=1.2)
    axes[0].axvline(PAPER_N95, linestyle="--", color=COLORS["grey"], linewidth=1.2)
    axes[0].text(PAPER_N95 * 1.08, 0.28, f"paper $N_{{95}}\\approx{PAPER_N95:g}$",
                 fontsize=8, color=COLORS["grey"], rotation=90, va="bottom")
    axes[0].set(xscale="log", ylim=(0.0, 1.02), xlabel="Physical fleet size $N$",
                ylabel="$R^2$: aggregate variance explained by the broadcast signal",
                title="a  Signal determinism vs fleet size (paper Fig. 2b)")
    axes[0].legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.9)

    reached = table[table["N95"].notna()].sort_values("N95").reset_index(drop=True)
    if not reached.empty:
        y = np.arange(len(reached))
        axes[1].scatter(reached["N95"], y, color=COLORS["ours"], zorder=3,
                        label="$N_{95}$ (log-interpolated)")
        axes[1].scatter(reached["N95_grid"], y, facecolors="none", edgecolors=COLORS["grey"],
                        zorder=2, label="grid crossing (simulated $N$ values)")
        axes[1].axvline(PAPER_N95, linestyle="--", color=COLORS["paper"], linewidth=1.4,
                        label=f"paper $N_{{95}}\\approx{PAPER_N95:g}$ (IEEE33 simulation)")
        axes[1].set(xscale="log", yticks=y,
                    yticklabels=[_short(str(name)) for name in reached["dataset"]],
                    xlabel=f"$N_{{95}}$: fleet size at which $R^2$ crosses {threshold:g}",
                    title="b  Determinism threshold per dataset vs the paper's value")
        axes[1].legend(fontsize=7, loc="lower right")
        axes[1].text(0.02, 0.96, "median $N_{{95}}$ = {:.0f}  (paper: {:g})".format(
            reached["N95"].median(), PAPER_N95), transform=axes[1].transAxes, fontsize=8, va="top")
    missing = table[table["N95"].isna()]["dataset"].tolist()
    if missing:
        axes[1].text(0.02, 0.02, "not reached: " + ", ".join(_short(str(m)) for m in missing),
                     transform=axes[1].transAxes, fontsize=7, color=COLORS["paper"])

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute R^2 against the broadcast signal basis for every (dataset, arm, N) cell.")
    parser.add_argument("--source", type=Path, required=True, nargs="+",
                        help="one or more E1_scale_boundary_new directories, de-duplicated in the order given; "
                             "overlapping (dataset, arm, N) cells are used to check run-to-run agreement")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    frames = []
    for source in args.source:
        responses = pd.read_csv(source / "raw" / "condition_responses.csv")
        block = compute_cells(responses)
        summary_path = source / "summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            keep = [c for c in ("dataset", "arm", "N", "N_eff", "margin", "N_star_eff",
                                "p_controllable", "mean_nrmse") if c in summary.columns]
            block = block.merge(summary[keep], on=["dataset", "arm", "N"],
                                how="left", validate="one_to_one")
        block["source"] = str(source)
        frames.append(block)

    overlap = None
    if len(frames) > 1:
        joined = pd.concat(frames, ignore_index=True)
        counts = joined.groupby(["dataset", "arm", "N"])["R2_signal"].agg(["size", "min", "max"])
        shared = counts[counts["size"] > 1]
        overlap = {
            "overlapping_cells": int(len(shared)),
            "max_abs_gap": float((shared["max"] - shared["min"]).max()) if len(shared) else None,
            "median_abs_gap": float((shared["max"] - shared["min"]).median()) if len(shared) else None,
        }
        print("overlapping cell check:", json.dumps(overlap, ensure_ascii=False))

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["dataset", "arm", "N"], keep="first"
    )
    if "N_star_eff" in merged:
        star = (merged.loc[merged["N_star_eff"].notna(), ["dataset", "N_star_eff"]]
                .drop_duplicates("dataset").set_index("dataset")["N_star_eff"])
        merged["N_star_eff"] = merged["dataset"].map(star)
        merged["margin"] = np.where(
            merged["N_star_eff"].notna() & (merged["N_star_eff"].astype(float) > 0),
            merged["N_eff"] / merged["N_star_eff"], np.nan,
        )

    table = n95_table(merged, args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "summary_r2_signal.csv", index=False)
    table.to_csv(args.output / "n95_table.csv", index=False)

    coupled = merged[merged["arm"] == "data_coupled"]
    reached = table[table["N95"].notna()]
    report = {
        "quantity": "R2 = fraction of aggregate response variance explained by the broadcast signal",
        "estimator": "V1 variance decomposition: 1 - SS(within-condition, across seeds) / SS(total)",
        "uses_target": False,
        "source": [str(s) for s in args.source],
        "overlap_consistency_check": overlap,
        "threshold": args.threshold,
        "paper_N95": PAPER_N95,
        "cells": int(len(merged)),
        "datasets_reaching_threshold": int(len(reached)),
        "datasets_total": int(len(table)),
        "N95_median": float(reached["N95"].median()) if len(reached) else None,
        "N95_range": ([float(reached["N95"].min()), float(reached["N95"].max())]
                      if len(reached) else None),
        "N95_grid_median": float(reached["N95_grid"].median()) if len(reached) else None,
        "censored_at_min_fleet": int(reached["censored_at_min_fleet"].sum()) if len(reached) else 0,
        "R2_median_by_N": {str(int(n)): round(float(v), 4)
                           for n, v in coupled.groupby("N")["R2_signal"].median().items()},
    }
    (args.output / "n95_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figure(merged, table, args.threshold, args.output / "Figs" / "figure_R2_signal")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
