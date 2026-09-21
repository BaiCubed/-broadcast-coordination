from __future__ import annotations

import argparse
import json
import re
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
ARCHIVE = re.compile(r"^responses_N(\d+)_(.+)\.npz$")

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


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator <= EPS:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def cell_r2(aggregate: np.ndarray, train_replications: int) -> dict[str, float]:
    train = aggregate[:train_replications]
    test = aggregate[train_replications:]
    if train.shape[0] < 1 or test.shape[0] < 2:
        raise SystemExit(f"not enough replications: train={train.shape[0]} test={test.shape[0]}")

    prediction_by_condition = train.mean(axis=0)
    actual = test.reshape(-1)
    prediction = np.tile(prediction_by_condition, test.shape[0])
    r2 = r2_score(actual, prediction)

    insample = r2_score(actual, np.tile(test.mean(axis=0), test.shape[0]))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    conditions = test.shape[1]
    draws = rng.integers(0, conditions, size=(BOOTSTRAP_DRAWS, conditions))
    boot = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for index in range(BOOTSTRAP_DRAWS):
        picked = draws[index]
        boot[index] = r2_score(
            test[:, picked].reshape(-1), np.tile(prediction_by_condition[picked], test.shape[0])
        )
    return {
        "R2": r2,
        "R2_ci_lower": float(np.percentile(boot, 2.5)),
        "R2_ci_upper": float(np.percentile(boot, 97.5)),
        "R2_insample": insample,
        "insample_optimism": insample - r2,
        "conditions": int(conditions),
        "train_replications": int(train.shape[0]),
        "test_replications": int(test.shape[0]),
    }


def load_cells(source: Path, train_replications: int) -> pd.DataFrame:
    root = source / "raw" / "responses"
    if not root.is_dir():
        raise SystemExit(f"{root} not found; this readout needs the npz with stored training replications, the csv is not enough")
    rows: list[dict[str, object]] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for archive in sorted(dataset_dir.iterdir()):
            match = ARCHIVE.match(archive.name)
            if not match:
                continue
            with np.load(archive) as payload:
                aggregate = np.asarray(payload["aggregate_response_kw"], dtype=float)
            rows.append({
                "dataset": dataset_dir.name,
                "arm": match.group(2),
                "N": int(match.group(1)),
                **cell_r2(aggregate, train_replications),
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
        sizes = ordered["N"].to_numpy(dtype=float)
        point = ordered["R2"].to_numpy(dtype=float)
        crossing = np.flatnonzero(point >= threshold)
        index = int(crossing[0]) if crossing.size else None
        rows.append({
            "dataset": dataset,
            "N95_grid": float(sizes[index]) if index is not None else None,
            "N95": _log_interpolate(sizes, point, index, threshold) if index is not None else None,
            "R2_at_crossing": float(point[index]) if index is not None else None,
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
        axes[0].plot(block["N"], block["R2"], "o-", linewidth=1, markersize=3.5,
                     alpha=0.85, color=color, label=_short(dataset))
    axes[0].axhline(threshold, linestyle=":", color=COLORS["paper"], linewidth=1.2)
    axes[0].axvline(PAPER_N95, linestyle="--", color=COLORS["grey"], linewidth=1.2)
    axes[0].text(PAPER_N95 * 1.08, 0.28, f"paper $N_{{95}}\\approx{PAPER_N95:g}$",
                 fontsize=8, color=COLORS["grey"], rotation=90, va="bottom")
    axes[0].set(xscale="log", ylim=(0.0, 1.02), xlabel="Physical fleet size $N$",
                ylabel="$R^2$: aggregate variance explained by the broadcast signal",
                title="a  Paper's Fig. 2b estimator, recomputed on real datasets")
    axes[0].legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.9)

    reached = table[table["N95"].notna()].sort_values("N95").reset_index(drop=True)
    if not reached.empty:
        y = np.arange(len(reached))
        axes[1].scatter(reached["N95"], y, color=COLORS["ours"], zorder=3,
                        label="$N_{95}$ (log-interpolated)")
        axes[1].scatter(reached["N95_grid"], y, facecolors="none", edgecolors=COLORS["grey"],
                        zorder=2, label="grid crossing (paper's own rule)")
        axes[1].axvline(PAPER_N95, linestyle="--", color=COLORS["paper"], linewidth=1.4,
                        label=f"paper $N_{{95}}\\approx{PAPER_N95:g}$ (IEEE33)")
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
    parser = argparse.ArgumentParser(description="Recompute the critical fleet size of each dataset under the published R^2 >= 0.95 rule.")
    parser.add_argument("--source", type=Path, required=True, nargs="+",
                        help="one or more E1_scale_boundary_new directories; overlapping cells are used to check run-to-run agreement")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--train-replications", type=int, default=10)
    args = parser.parse_args()

    frames = []
    for source in args.source:
        block = load_cells(source, args.train_replications)
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
        counts = joined.groupby(["dataset", "arm", "N"])["R2"].agg(["size", "min", "max"])
        shared = counts[counts["size"] > 1]
        overlap = {
            "overlapping_cells": int(len(shared)),
            "max_abs_gap": float((shared["max"] - shared["min"]).max()) if len(shared) else None,
            "median_abs_gap": float((shared["max"] - shared["min"]).median()) if len(shared) else None,
        }
        print("overlapping cell check:", json.dumps(overlap, ensure_ascii=False))

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["dataset", "arm", "N"], keep="first"
    ).sort_values(["dataset", "arm", "N"])

    table = n95_table(merged, args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "summary_r2_paper.csv", index=False)
    table.to_csv(args.output / "n95_table_paper.csv", index=False)

    coupled = merged[merged["arm"] == "data_coupled"]
    reached = table[table["N95"].notna()]
    report = {
        "estimator": "paper Fig. 2b: 1 - SS(actual - train-set condition mean) / SS(actual - mean actual)",
        "matches_repo_file": "src/extra/ieee33_device_day_simulation/experiments/"
                             "ieee33_real_snapshot/threshold_only.py",
        "n95_rule": "first grid N whose point estimate reaches the threshold (paper's own rule); "
                    "N95 additionally log-interpolated for like-for-like comparison with '≈150'",
        "bootstrap": "resample conditions with replacement (as in the paper), 400 draws",
        "threshold": args.threshold,
        "paper_N95": PAPER_N95,
        "cells": int(len(merged)),
        "overlap_consistency_check": overlap,
        "datasets_reaching_threshold": int(len(reached)),
        "datasets_total": int(len(table)),
        "N95_median": float(reached["N95"].median()) if len(reached) else None,
        "N95_range": [float(reached["N95"].min()), float(reached["N95"].max())] if len(reached) else None,
        "N95_grid_median": float(reached["N95_grid"].median()) if len(reached) else None,
        "censored_at_min_fleet": int(reached["censored_at_min_fleet"].sum()) if len(reached) else 0,
        "R2_median_by_N": {
            str(int(n)): round(float(v), 4)
            for n, v in coupled.groupby("N")["R2"].median().items()
        },
        "insample_optimism": {
            "definition": "R2 with the in-sample condition mean minus R2 with the training-set "
                          "condition mean; this is exactly how much the old V1 reading was optimistic",
            "median": float(coupled["insample_optimism"].median()),
            "max": float(coupled["insample_optimism"].max()),
        },
    }
    (args.output / "n95_report_paper.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figure(merged, table, args.threshold, args.output / "Figs" / "figure_R2_paper_estimator")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
