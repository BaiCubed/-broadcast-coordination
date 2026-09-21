from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_E1_ROOT = ROOT / "results/e1_full/E1_scale_boundary_new"
DEFAULT_OUTPUT = ROOT / "results/E21/gamma_universal_boundary"
ARCHIVE_RE = re.compile(r"^responses_N(\d+)_(data_coupled|decoupled)\.npz$")
R2_THRESHOLD = 0.95
EPS = 1e-12

SHORT_NAMES = {
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


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator <= EPS:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def effective_n(n: float, rho: float) -> float:
    return float(n / (1.0 + max(n - 1.0, 0.0) * max(float(rho), 0.0)))


def log_interpolate_crossing(
    left_x: float,
    left_y: float,
    right_x: float,
    right_y: float,
    threshold: float,
) -> float:
    if right_y == left_y or min(left_x, right_x) <= 0:
        return float(right_x)
    fraction = float(np.clip((threshold - left_y) / (right_y - left_y), 0.0, 1.0))
    return float(10 ** (math.log10(left_x) + fraction * (math.log10(right_x) - math.log10(left_x))))


def bootstrap_r2(
    test: np.ndarray,
    prediction_by_condition: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    test = np.asarray(test, dtype=float)
    prediction_by_condition = np.asarray(prediction_by_condition, dtype=float)
    actual = test.reshape(-1)
    predicted = np.tile(prediction_by_condition, test.shape[0])
    point = r2_score(actual, predicted)

    condition_sse = np.sum((test - prediction_by_condition[None, :]) ** 2, axis=0)
    condition_sum = np.sum(test, axis=0)
    condition_sum_sq = np.sum(test * test, axis=0)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, test.shape[1], size=(draws, test.shape[1]))
    boot = np.empty(draws, dtype=float)
    for index, selected in enumerate(sampled):
        counts = np.bincount(selected, minlength=test.shape[1]).astype(float)
        total = float(test.shape[0] * test.shape[1])
        total_sum = float(counts @ condition_sum)
        total_sum_sq = float(counts @ condition_sum_sq)
        denominator = total_sum_sq - total_sum * total_sum / total
        boot[index] = 1.0 - float(counts @ condition_sse) / max(denominator, EPS)
    return point, boot


def _archive_paths(e1_root: Path, dataset: str, arm: str) -> list[tuple[int, Path]]:
    directory = e1_root / "raw" / "responses" / dataset
    rows: list[tuple[int, Path]] = []
    for path in directory.glob(f"responses_N*_{arm}.npz"):
        match = ARCHIVE_RE.match(path.name)
        if match and match.group(2) == arm:
            rows.append((int(match.group(1)), path))
    return sorted(rows)


def _rho_reference(summary_path: Path) -> dict[str, float]:
    frame = pd.read_csv(summary_path)
    frame = frame[frame["arm"].astype(str) == "data_coupled"]
    references: dict[str, float] = {}
    for dataset, block in frame.groupby("dataset"):
        values = np.maximum(block["rho_measured"].astype(float).to_numpy(), 0.0)
        references[str(dataset)] = float(np.median(values)) if len(values) else 0.0
    return references


def _load_dataset(
    e1_root: Path,
    dataset: str,
    rho_reference: float,
    train_replications: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coupled = _archive_paths(e1_root, dataset, "data_coupled")
    if not coupled:
        raise FileNotFoundError(f"missing data_coupled archive: {dataset}")
    reference_path = next((path for n, path in coupled if n == 1), None)
    if reference_path is None:
        raise FileNotFoundError(f"{dataset} has no N=1 training reference archive")
    with np.load(reference_path) as payload:
        reference = np.asarray(payload["aggregate_response_kw"], dtype=float)
    train_reference = reference[:train_replications]
    if train_reference.shape[0] < 2:
        raise ValueError(f"{dataset} has too few training replications")
    within_reference = float(np.mean(np.var(train_reference, axis=0, ddof=1)))
    between_observed = float(np.var(np.mean(train_reference, axis=0), ddof=1))
    between_debiased = max(between_observed - within_reference / train_replications, EPS)
    sigma_squared = within_reference / between_debiased
    finite_sample_factor = (R2_THRESHOLD + 1.0 / train_replications) / (1.0 - R2_THRESHOLD)
    n_star_95 = finite_sample_factor * sigma_squared

    rows: list[dict[str, Any]] = []
    bootstrap_by_cell: dict[tuple[str, int], np.ndarray] = {}
    for arm in ("data_coupled", "decoupled"):
        for n, path in _archive_paths(e1_root, dataset, arm):
            with np.load(path) as payload:
                aggregate = np.asarray(payload["aggregate_response_kw"], dtype=float)
            if aggregate.shape[0] <= train_replications:
                raise ValueError(f"{path} has too few test replications")
            train = aggregate[:train_replications]
            test = aggregate[train_replications:]
            prediction = np.mean(train, axis=0)
            seed = bootstrap_seed + n * 17 + sum(ord(c) for c in dataset) + (0 if arm == "data_coupled" else 1_000_000)
            point, boot = bootstrap_r2(test, prediction, bootstrap_draws, seed)
            rho = rho_reference if arm == "data_coupled" else 0.0
            n_eff = effective_n(n, rho)
            gamma = n_eff / max(n_star_95, EPS)
            key = (arm, n)
            bootstrap_by_cell[key] = boot
            rows.append({
                "dataset": dataset,
                "dataset_label": SHORT_NAMES.get(dataset, dataset),
                "arm": arm,
                "N": int(n),
                "rho_reference": rho,
                "N_eff": n_eff,
                "within_variance_N1_train": within_reference,
                "between_variance_N1_observed": between_observed,
                "between_variance_N1_debiased": between_debiased,
                "sigma_squared": sigma_squared,
                "sigma": math.sqrt(max(sigma_squared, 0.0)),
                "N_star_95": n_star_95,
                "N_star_95_formula_factor": finite_sample_factor,
                "Gamma": gamma,
                "R2": point,
                "R2_ci_lower": float(np.percentile(boot, 2.5)),
                "R2_ci_upper": float(np.percentile(boot, 97.5)),
                "train_replications": int(train.shape[0]),
                "test_replications": int(test.shape[0]),
                "bootstrap_key": key,
            })

    metadata = {
        "dataset": dataset,
        "dataset_label": SHORT_NAMES.get(dataset, dataset),
        "rho_reference": rho_reference,
        "rho_reference_source": "median(max(rho_measured, 0)) across E1 data_coupled summary cells",
        "calibration_method": "training-only de-biased variance ratio with finite-training R2 threshold formula",
        "within_variance_N1_train": within_reference,
        "between_variance_N1_observed": between_observed,
        "between_variance_N1_debiased": between_debiased,
        "sigma_squared": sigma_squared,
        "sigma": math.sqrt(max(sigma_squared, 0.0)),
        "N_star_95": n_star_95,
        "N_star_95_formula_factor": finite_sample_factor,
        "bootstrap_draws": bootstrap_draws,
        "train_replications": train_replications,
        "bootstrap_by_cell": bootstrap_by_cell,
    }
    return rows, metadata


def _critical(rows: list[dict[str, Any]], bootstrap_by_cell: dict[tuple[str, int], np.ndarray], threshold: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["N"]))
    crossing = next((index for index, row in enumerate(ordered) if float(row["R2"]) >= threshold), None)
    if crossing is None:
        value = None
    elif crossing == 0:
        value = float(ordered[0]["Gamma"])
    else:
        left, right = ordered[crossing - 1], ordered[crossing]
        value = log_interpolate_crossing(float(left["Gamma"]), float(left["R2"]), float(right["Gamma"]), float(right["R2"]), threshold)
    draws = min(len(next(iter(bootstrap_by_cell.values()))), 400) if bootstrap_by_cell else 0
    critical_draws: list[float] = []
    for draw_index in range(draws):
        boot_crossing = next((index for index, row in enumerate(ordered) if bootstrap_by_cell[(row["arm"], int(row["N"]))][draw_index] >= threshold), None)
        if boot_crossing is None:
            continue
        if boot_crossing == 0:
            critical_draws.append(float(ordered[0]["Gamma"]))
        else:
            left, right = ordered[boot_crossing - 1], ordered[boot_crossing]
            critical_draws.append(log_interpolate_crossing(float(left["Gamma"]), float(bootstrap_by_cell[(left["arm"], int(left["N"]))][draw_index]), float(right["Gamma"]), float(bootstrap_by_cell[(right["arm"], int(right["N"]))][draw_index]), threshold))
    return {
        "dataset": rows[0]["dataset"] if rows else None,
        "arm": rows[0]["arm"] if rows else None,
        "Gamma_c": value,
        "threshold": threshold,
        "N_grid_at_crossing": int(ordered[crossing]["N"]) if crossing is not None else None,
        "Gamma_max": float(max(row["Gamma"] for row in ordered)) if ordered else None,
        "status": "reached" if value is not None else "right_censored",
        "Gamma_c_ci_lower": float(np.percentile(critical_draws, 2.5)) if critical_draws else None,
        "Gamma_c_ci_upper": float(np.percentile(critical_draws, 97.5)) if critical_draws else None,
    }


def _binned_variance(frame: pd.DataFrame, key: str, bins: int = 8, seed_values: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(seed_values if seed_values is not None else frame[key], dtype=float)
    valid = np.isfinite(values) & (values > 0)
    values = values[valid]
    block = frame.loc[valid].copy()
    if len(values) < bins * 2:
        return {"bins_requested": bins, "bins_with_at_least_two_datasets": 0, "mean_between_dataset_variance": None, "mean_within_bin_log10_spread": None, "cells": int(len(values))}
    log_values = np.log10(values)
    edges = np.unique(np.quantile(log_values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return {"bins_requested": bins, "bins_with_at_least_two_datasets": 0, "mean_between_dataset_variance": None, "mean_within_bin_log10_spread": None, "cells": int(len(values))}
    labels = np.clip(np.searchsorted(edges, log_values, side="right") - 1, 0, len(edges) - 2)
    variances: list[float] = []
    spreads: list[float] = []
    for label in sorted(set(labels)):
        members = np.flatnonzero(labels == label)
        by_dataset: dict[str, list[float]] = {}
        for index in members:
            by_dataset.setdefault(str(block.iloc[index]["dataset"]), []).append(float(block.iloc[index]["R2"]))
        if len(by_dataset) >= 2:
            variances.append(float(np.var([np.mean(v) for v in by_dataset.values()])))
            spreads.append(float(np.std(log_values[members])))
    return {
        "bins_requested": bins,
        "bins_with_at_least_two_datasets": len(variances),
        "mean_between_dataset_variance": float(np.mean(variances)) if variances else None,
        "mean_within_bin_log10_spread": float(np.mean(spreads)) if spreads else None,
        "cells": int(len(values)),
    }


def _fit_hierarchical(frame: pd.DataFrame, output_dir: Path, train_replications: int) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    fit_frame = frame.copy()
    fit_frame["x"] = np.log(fit_frame["Gamma"].astype(float))
    fit_frame["y"] = np.log(np.clip(fit_frame["R2"].astype(float), 1e-5, 1 - 1e-5) / np.clip(1 - fit_frame["R2"].astype(float), 1e-5, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = smf.mixedlm("y ~ x", fit_frame, groups=fit_frame["dataset"]).fit(reml=False, method="powell", maxiter=1000, disp=False)
    fixed = fitted.fe_params
    covariance = fitted.cov_params().loc[["Intercept", "x"], ["Intercept", "x"]].to_numpy(dtype=float)
    grid = np.logspace(np.log10(max(float(fit_frame["Gamma"].min()) * 0.8, 1e-3)), np.log10(float(fit_frame["Gamma"].max()) * 1.2), 300)
    design = np.column_stack([np.ones_like(grid), np.log(grid)])
    linear = design @ fixed.to_numpy(dtype=float)
    standard_error = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design, covariance, design), 0.0))
    def logistic(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))
    finite_sample_factor = (R2_THRESHOLD + 1.0 / train_replications) / (1.0 - R2_THRESHOLD)
    x_ratio = 1.0 / np.maximum(finite_sample_factor * grid, EPS)
    predictions = pd.DataFrame({
        "Gamma": grid,
        "R2_fit": logistic(linear),
        "R2_fit_lower": logistic(linear - 1.96 * standard_error),
        "R2_fit_upper": logistic(linear + 1.96 * standard_error),
        "R2_theory": 1.0 - (1.0 + 1.0 / train_replications) * x_ratio / (1.0 + x_ratio),
    })
    predictions.to_csv(output_dir / "gamma_fit_predictions.csv", index=False)

    lodo_rows: list[dict[str, Any]] = []
    for dataset in sorted(fit_frame["dataset"].unique()):
        train = fit_frame[fit_frame["dataset"] != dataset]
        test = fit_frame[fit_frame["dataset"] == dataset]
        coefficients = np.polyfit(train["x"], train["y"], 1)
        predicted = logistic(np.polyval(coefficients, test["x"].to_numpy()))
        lodo_rows.append({
            "dataset": dataset,
            "dataset_label": SHORT_NAMES.get(dataset, dataset),
            "n_test_cells": int(len(test)),
            "rmse_R2": float(np.sqrt(np.mean((test["R2"].to_numpy() - predicted) ** 2))),
            "mae_R2": float(np.mean(np.abs(test["R2"].to_numpy() - predicted))),
            "slope_logit_R2_log_Gamma": float(coefficients[0]),
            "intercept": float(coefficients[1]),
        })
    pd.DataFrame(lodo_rows).to_csv(output_dir / "leave_one_dataset_out.csv", index=False)
    return {
        "model": "logit(R2) = fixed_intercept + fixed_slope*log(Gamma) + dataset_random_intercept",
        "optimizer": "Powell",
        "fixed_intercept": float(fixed["Intercept"]),
        "fixed_slope": float(fixed["x"]),
        "random_intercept_variance": float(np.asarray(fitted.cov_re)[0, 0]),
        "residual_variance": float(fitted.scale),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "lodo_mean_rmse_R2": float(np.mean([row["rmse_R2"] for row in lodo_rows])),
        "lodo_max_rmse_R2": float(np.max([row["rmse_R2"] for row in lodo_rows])),
    }


def _save_scatter(frame: pd.DataFrame, output: Path) -> None:
    datasets = sorted(frame["dataset"].unique())
    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, len(datasets)))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*", "p", "8", "d", "H"]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), constrained_layout=True, sharey=True)
    for color, marker, dataset in zip(colors, markers, datasets):
        block = frame[frame["dataset"] == dataset]
        block_gamma = block[np.isfinite(block["Gamma"].astype(float))]
        sizes = 22.0 + 14.0 * np.log10(np.maximum(block["N"].astype(float), 1.0))
        gamma_sizes = sizes.loc[block_gamma.index]
        yerr_gamma = [block_gamma["R2"] - block_gamma["R2_ci_lower"], block_gamma["R2_ci_upper"] - block_gamma["R2"]]
        axes[0].errorbar(block_gamma["Gamma"], block_gamma["R2"], yerr=yerr_gamma, fmt=marker, ms=5, lw=0.5, capsize=1.5, alpha=0.68, color=color, label=SHORT_NAMES.get(dataset, dataset))
        axes[0].scatter(block_gamma["Gamma"], block_gamma["R2"], s=gamma_sizes, color=color, alpha=0.18, edgecolors="none")
        yerr_n = [block["R2"] - block["R2_ci_lower"], block["R2_ci_upper"] - block["R2"]]
        axes[1].errorbar(block["N"], block["R2"], yerr=yerr_n, fmt=marker, ms=5, lw=0.5, capsize=1.5, alpha=0.68, color=color, label=SHORT_NAMES.get(dataset, dataset))
        axes[1].scatter(block["N"], block["R2"], s=sizes, color=color, alpha=0.18, edgecolors="none")
    axes[0].axvline(1.0, color="#222222", ls="--", lw=1.0, label=r"$\Gamma=1$")
    for ax in axes:
        ax.axhline(R2_THRESHOLD, color="#b2182b", ls=":", lw=1.2, label=r"$R^2=0.95$")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.2)
    axes[0].set_xlabel(r"Finite-training normalized scale $\Gamma=N_{\mathrm{eff}}/N^*_{95}$")
    axes[1].set_xlabel(r"Physical fleet size $N$")
    axes[0].set_ylabel(r"Independent evaluation-set $R^2$")
    axes[0].set_title("a  Normalized scale")
    axes[1].set_title("b  Physical scale reference")
    dataset_legend = axes[0].legend(fontsize=7, ncol=3, frameon=False, loc="lower right")
    axes[1].legend(fontsize=7, ncol=2, frameon=False, loc="lower right")
    size_handles = [axes[0].scatter([], [], s=22 + 14 * np.log10(n), color="#444444", alpha=0.5, label=f"N={n:g}") for n in (1, 50, 1000, 3000)]
    axes[0].legend(handles=size_handles, title="Marker area shows N", fontsize=7, title_fontsize=7, frameon=False, loc="upper left")
    axes[0].add_artist(dataset_legend)
    fig.suptitle("E21a  Cross-dataset R² with physical scale made explicit", fontsize=14)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _save_fit(frame: pd.DataFrame, predictions: pd.DataFrame, fit: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.8), constrained_layout=True)
    ax.scatter(frame["Gamma"], frame["R2"], s=15, color="#808080", alpha=0.24, edgecolors="none", label="15 datasets")
    ax.plot(predictions["Gamma"], predictions["R2_fit"], color="#2166ac", lw=2.4, label="Hierarchical fixed trend")
    ax.fill_between(predictions["Gamma"], predictions["R2_fit_lower"], predictions["R2_fit_upper"], color="#2166ac", alpha=0.16, label="95% fixed-trend CI")
    ax.plot(predictions["Gamma"], predictions["R2_theory"], color="#b2182b", ls="--", lw=1.6, label="Finite-training variance reference")
    ax.axvline(1.0, color="#222222", ls=":", lw=1.0)
    ax.axhline(R2_THRESHOLD, color="#222222", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"Dimensionless control scale $\Gamma$")
    ax.set_ylabel(r"Test-set $R^2$")
    ax.set_title("E21b  Hierarchical universal transition")
    ax.text(0.03, 0.06, f"fixed slope={fit['fixed_slope']:.2f}\nLODO RMSE={fit['lodo_mean_rmse_R2']:.3f}", transform=ax.transAxes, fontsize=9, va="bottom")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _save_critical(table: pd.DataFrame, output: Path) -> None:
    table = table.copy().sort_values("Gamma_c", na_position="last")
    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    x = np.arange(len(table), dtype=float)
    values = table["Gamma_c"].to_numpy(dtype=float)
    reached = np.isfinite(values)
    ax.scatter(x[reached], values[reached], color="#2166ac", s=58, zorder=3, label=r"$\Gamma_c$: first $R^2\geq0.95$")
    lower = table["Gamma_c_ci_lower"].to_numpy(dtype=float)
    upper = table["Gamma_c_ci_upper"].to_numpy(dtype=float)
    ci = reached & np.isfinite(lower) & np.isfinite(upper)
    ax.vlines(x[ci], lower[ci], upper[ci], color="#2166ac", lw=1.5, zorder=2)
    censored = ~reached
    if np.any(censored):
        ax.scatter(x[censored], table.loc[censored, "Gamma_max"], marker="^", facecolors="none", edgecolors="#b2182b", s=70, label="Right-censored")
    ax.axhline(1.0, color="#b2182b", ls="--", lw=1.2, label=r"Reference $\Gamma=1$")
    ax.set_yscale("log")
    ax.set_ylim(max(0.25, float(np.nanmin(np.where(np.isfinite(values), values, 0.5))) * 0.65), max(3.0, float(np.nanmax(np.where(np.isfinite(values), values, 1.0))) * 1.45))
    ax.set_xticks(x, table["dataset_label"], rotation=28, ha="right")
    ax.set_ylabel(r"Critical normalized scale $\Gamma_c$")
    ax.set_title(r"E21c  Stability of the $R^2=0.95$ boundary")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _save_combined(frame: pd.DataFrame, predictions: pd.DataFrame, critical: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    axes[0].scatter(frame["Gamma"], frame["R2"], s=9, color="#4c78a8", alpha=0.35)
    axes[0].axvline(1, color="#222", ls="--", lw=0.8)
    axes[0].axhline(.95, color="#b2182b", ls=":", lw=1)
    axes[0].set(xscale="log", ylim=(0, 1.02), xlabel=r"$\Gamma$", ylabel=r"Test $R^2$", title="a  Dataset observations")
    axes[1].scatter(frame["Gamma"], frame["R2"], s=8, color="#777", alpha=.18)
    axes[1].plot(predictions["Gamma"], predictions["R2_fit"], color="#2166ac", lw=2)
    axes[1].fill_between(predictions["Gamma"], predictions["R2_fit_lower"], predictions["R2_fit_upper"], color="#2166ac", alpha=.15)
    axes[1].plot(predictions["Gamma"], predictions["R2_theory"], color="#b2182b", ls="--", lw=1.2)
    axes[1].axvline(1, color="#222", ls=":", lw=.8)
    axes[1].axhline(.95, color="#222", ls=":", lw=.8)
    axes[1].set(xscale="log", ylim=(0, 1.02), xlabel=r"$\Gamma$", ylabel=r"Test $R^2$", title="b  Common hierarchical trend")
    ordered = critical.sort_values("Gamma_c", na_position="last")
    axes[2].scatter(np.arange(len(ordered)), ordered["Gamma_c"], color="#2166ac", s=24)
    censored = ordered["Gamma_c"].isna()
    if censored.any():
        axes[2].scatter(
            np.flatnonzero(censored),
            ordered.loc[censored, "Gamma_max"],
            marker="^",
            facecolors="none",
            edgecolors="#b2182b",
            s=32,
        )
    axes[2].axhline(1, color="#b2182b", ls="--", lw=1)
    axes[2].set(yscale="log", xlabel="Dataset", ylabel=r"$\Gamma_c$", title="c  Boundary stability")
    axes[2].set_xticks(np.arange(len(ordered)), ordered["dataset_label"], rotation=65, ha="right", fontsize=7)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("E21  Dimensionless controllability transition across 15 datasets", fontsize=14)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for run e21 gamma universal boundary.")
    parser.add_argument("--e1-root", type=Path, default=DEFAULT_E1_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-replications", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    output = args.output
    data_dir = output / "data"
    figures_dir = output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads((args.e1_root.parent.parent / "protocol_manifest.json").read_text(encoding="utf-8")) if (args.e1_root.parent.parent / "protocol_manifest.json").exists() else {}
    datasets = list(protocol.get("datasets", [])) or sorted(p.name for p in (args.e1_root / "raw" / "responses").iterdir() if p.is_dir())
    rho_refs = _rho_reference(args.e1_root / "summary.csv")
    all_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    bootstrap_map: dict[tuple[str, str, int], np.ndarray] = {}
    for dataset in datasets:
        rows, info = _load_dataset(args.e1_root, dataset, rho_refs.get(dataset, 0.0), args.train_replications, args.bootstrap_draws, args.seed)
        bootstrap_by_cell = info.pop("bootstrap_by_cell")
        for row in rows:
            bootstrap_map[(row["dataset"], row["arm"], row["N"])] = bootstrap_by_cell[(row["arm"], row["N"])]
            row.pop("bootstrap_key", None)
        all_rows.extend(rows)
        metadata.append(info)

    main_frame = pd.DataFrame([row for row in all_rows if row["arm"] == "data_coupled"])
    negative_frame = pd.DataFrame([row for row in all_rows if row["arm"] == "decoupled"])
    main_frame.to_csv(data_dir / "gamma_points.csv", index=False)
    negative_frame.to_csv(data_dir / "decoupled_negative_control.csv", index=False)
    pd.DataFrame(metadata).to_csv(data_dir / "dataset_calibration.csv", index=False)

    critical_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        block = [row for row in all_rows if row["dataset"] == dataset and row["arm"] == "data_coupled"]
        cell_boot = {(row["arm"], row["N"]): bootstrap_map[(row["dataset"], row["arm"], row["N"])] for row in block}
        critical_rows.append(_critical(block, cell_boot, R2_THRESHOLD))
    critical_frame = pd.DataFrame(critical_rows)
    critical_frame["dataset_label"] = critical_frame["dataset"].map(lambda value: SHORT_NAMES.get(value, value))
    critical_frame.to_csv(data_dir / "gamma_critical.csv", index=False)

    analysis_frame = main_frame[np.isfinite(main_frame["Gamma"].astype(float))].copy()
    fit = _fit_hierarchical(analysis_frame, data_dir, args.train_replications)
    predictions = pd.read_csv(data_dir / "gamma_fit_predictions.csv")
    gamma_axis = _binned_variance(analysis_frame, "Gamma", bins=8)
    physical_axis = _binned_variance(analysis_frame, "N", bins=8)
    null_rng = np.random.default_rng(args.seed + 99)
    observed = float(gamma_axis["mean_between_dataset_variance"] or np.inf)
    null_values = []
    gamma_values = analysis_frame["Gamma"].to_numpy(dtype=float)
    for _ in range(args.permutations):
        null_values.append(float(_binned_variance(analysis_frame, "Gamma", bins=8, seed_values=null_rng.permutation(gamma_values))["mean_between_dataset_variance"] or np.inf))
    null_values_array = np.asarray(null_values, dtype=float)
    permutation = {
        "statistic": "mean between-dataset variance within equal-occupancy Gamma bins; lower is better",
        "observed": observed,
        "null_mean": float(np.mean(null_values_array)),
        "null_ci_95": [float(np.percentile(null_values_array, 2.5)), float(np.percentile(null_values_array, 97.5))],
        "p_lower_tail": float((1 + np.sum(null_values_array <= observed)) / (len(null_values_array) + 1)),
        "permutations": args.permutations,
        "seed": args.seed + 99,
    }
    (data_dir / "permutation_test.json").write_text(json.dumps(permutation, indent=2, ensure_ascii=False), encoding="utf-8")
    collapse = {
        "definition": "matched equal-occupancy bins of the cross-dataset R2 variance",
        "main_arm": "data_coupled",
        "gamma_axis": gamma_axis,
        "physical_N_axis": physical_axis,
        "variance_ratio_gamma_over_N": (gamma_axis["mean_between_dataset_variance"] / physical_axis["mean_between_dataset_variance"] if gamma_axis["mean_between_dataset_variance"] is not None and physical_axis["mean_between_dataset_variance"] else None),
        "collapse_improves_over_physical_N": bool(gamma_axis["mean_between_dataset_variance"] is not None and physical_axis["mean_between_dataset_variance"] is not None and gamma_axis["mean_between_dataset_variance"] < physical_axis["mean_between_dataset_variance"]),
        "comparison_is_matched": True,
        "dataset_count": int(main_frame["dataset"].nunique()),
        "cells": int(len(analysis_frame)),
        "fit": fit,
    }
    (data_dir / "collapse_quality.json").write_text(json.dumps(collapse, indent=2, ensure_ascii=False), encoding="utf-8")

    _save_scatter(main_frame, figures_dir / "e21_gamma_scatter")
    _save_fit(analysis_frame, predictions, fit, figures_dir / "e21_gamma_fit")
    _save_critical(critical_frame, figures_dir / "e21_gamma_critical_distribution")
    _save_combined(analysis_frame, predictions, critical_frame, figures_dir / "e21_gamma_universal_boundary")

    manifest = {
        "experiment": "E21_gamma_universal_boundary",
        "status": "completed",
        "source": str(args.e1_root),
        "datasets": datasets,
        "dataset_count": len(datasets),
        "main_arm": "data_coupled",
        "negative_control": "decoupled",
        "r2_definition": "1 - SS(test - train_condition_mean) / SS(test - mean(test))",
        "gamma_definition": "N_eff / N_star_95",
        "n_eff_definition": "N / (1 + (N - 1) * max(rho_reference, 0))",
        "rho_reference_definition": "dataset-level median(max(E1 rho_measured, 0))",
        "n_star_95_definition": "empirical first R2=0.95 crossing on the independent calibration fold, log-interpolated on N_eff",
        "finite_training_correction": "N_star_95 = sigma_squared * (q + 1/T) / (1 - q), with q=0.95 and T=train_replications",
        "bootstrap_draws": args.bootstrap_draws,
        "permutations": args.permutations,
        "seed": args.seed,
        "generated_files": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["generated_files"].append({"path": str(path.relative_to(output)), "sha256": digest})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "completed", "datasets": len(datasets), "cells": len(analysis_frame), "raw_cells": len(main_frame), "output": str(output), "collapse": collapse, "permutation": permutation}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
