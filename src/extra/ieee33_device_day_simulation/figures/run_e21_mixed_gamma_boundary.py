from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.extra.nc_excel_experiments.coupling import (
    availability_at_steps,
    decoupled_copy,
    diurnal_residual,
    rank_availability,
)
from src.extra.nc_excel_experiments.run import _profile_and_schedule, _run_snapshots
from src.extra.ieee33_device_day_simulation.experiments.protocol import r2_score
from . import run_e21_mixed_scenarios as mixed


PROTOCOL = "E21_mixed_gamma_physical_fleet_v1"
DEFAULT_OUTPUT = Path("results/E21/gamma_mixed_boundary")
DEFAULT_N_VALUES = (30, 50, 100, 150, 250, 400, 650, 1000, 1600, 2500, 5000)
R2_THRESHOLD = 0.95
EPS = 1e-12
SHORT_NAMES = {
    mixed.BDG1: "BDG1",
    mixed.BDG2: "BDG2",
    mixed.CEC: "CEC",
    mixed.DANISH: "Danish",
    mixed.EU_RURAL: "EU-Rural",
    mixed.EU_35297: "EU-35k",
    mixed.GOIENER: "GoiEner",
    mixed.HEAPO: "HEAPO",
    mixed.LCL: "LCL",
    mixed.NORWAY: "Norway",
    mixed.CAMSL: "CAMSL",
    mixed.EU_8087: "EU-8k",
    mixed.IRISH: "Irish",
    mixed.OPSD: "OPSD",
    mixed.SGSC: "SGSC",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_r2(
    test: np.ndarray,
    prediction: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    actual = np.asarray(test, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    point = r2_score(actual.reshape(-1), np.tile(predicted, actual.shape[0]))
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, actual.shape[0], size=actual.shape[0])
        sample = actual[selected]
        values[draw] = r2_score(
            sample.reshape(-1), np.tile(predicted, sample.shape[0])
        )
    return point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _mixture_metrics(weights: dict[str, float]) -> dict[str, float | int]:
    values = np.asarray(list(weights.values()), dtype=float)
    values = values / np.sum(values)
    component_count = int(len(values))
    entropy = -float(np.sum(values * np.log(np.maximum(values, EPS))))
    normalized_entropy = entropy / math.log(component_count) if component_count > 1 else 0.0
    return {
        "component_count": component_count,
        "dominant_weight": float(np.max(values)),
        "mixture_entropy": entropy,
        "normalized_mixture_entropy": normalized_entropy,
        "effective_dataset_count": float(1.0 / np.sum(values**2)),
    }


def _run_cell(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        scenario_id,
        n,
        train_replications,
        test_replications,
        conditions,
        bootstrap_draws,
        seed,
        response_path_text,
    ) = task
    response_path = Path(response_path_text)
    scenario_index = list(mixed.SCENARIOS).index(scenario_id)
    cell_seed = seed + scenario_index * 1_000_000 + int(n) * 1009
    records, config, audit = mixed._sample_mixed_fleet(
        scenario_id,
        "aggregate",
        "test",
        cell_seed,
        fleet_size=int(n),
    )
    config["control"]["network_feedback"] = False
    profile_steps, schedule = _profile_and_schedule(
        records,
        config,
        int(conditions),
        seed + scenario_index * 10000,
    )
    steps_per_day = int(config["simulation"]["steps_per_day"])
    coupled = availability_at_steps(
        rank_availability(diurnal_residual(records, steps_per_day)),
        profile_steps,
    )
    decoupled = decoupled_copy(coupled, cell_seed + 700_000)
    total_replications = int(train_replications + test_replications)
    rows: list[dict[str, Any]] = []
    archive: dict[str, Any] = {
        "profile_steps": np.asarray(profile_steps, dtype=int),
        "N": np.asarray([n], dtype=int),
    }
    for arm_index, (arm, availability) in enumerate(
        (("data_coupled", coupled), ("decoupled", decoupled))
    ):
        aggregate, _, _, _ = _run_snapshots(
            records,
            config,
            profile_steps,
            schedule,
            replications=total_replications,
            seed=cell_seed,
            availability_factory=lambda _replication, table=availability: table,
        )
        normalized = aggregate / float(n)
        train = normalized[:train_replications]
        test = normalized[train_replications:]
        prediction = np.mean(train, axis=0)
        r2, r2_lower, r2_upper = _bootstrap_r2(
            test,
            prediction,
            int(bootstrap_draws),
            cell_seed + arm_index * 100_000,
        )
        train_within = float(np.mean(np.var(train, axis=0, ddof=1)))
        test_within = float(np.mean(np.var(test, axis=0, ddof=1)))
        between_observed = float(np.var(prediction, ddof=1))
        between_debiased = max(
            between_observed - train_within / max(train_replications, 1),
            EPS,
        )
        archive[f"aggregate_response_kw_{arm}"] = aggregate
        archive[f"availability_{arm}"] = availability.astype(np.float32)
        rows.append({
            "scenario": scenario_id,
            "scenario_label": mixed.SCENARIOS[scenario_id]["label"],
            "arm": arm,
            "N": int(n),
            "R2": r2,
            "R2_ci_lower": r2_lower,
            "R2_ci_upper": r2_upper,
            "train_within_variance_per_device": train_within,
            "test_within_variance_per_device": test_within,
            "between_variance_observed": between_observed,
            "between_variance_debiased": between_debiased,
            "mean_response_kw_per_device": float(np.mean(test)),
            "std_response_kw_per_device": float(np.std(test)),
            "train_replications": int(train_replications),
            "test_replications": int(test_replications),
            "conditions": int(conditions),
        })
    response_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(response_path, **archive)
    return {
        "scenario": scenario_id,
        "N": int(n),
        "rows": rows,
        "audit": audit,
        "response_archive": str(response_path),
    }


def _log_interpolate(
    left_x: float,
    left_y: float,
    right_x: float,
    right_y: float,
    threshold: float,
) -> float:
    if left_x <= 0 or right_x <= 0 or abs(right_y - left_y) <= EPS:
        return float(right_x)
    fraction = float(np.clip((threshold - left_y) / (right_y - left_y), 0.0, 1.0))
    return float(10 ** (
        math.log10(left_x)
        + fraction * (math.log10(right_x) - math.log10(left_x))
    ))


def _first_crossing(
    rows: list[dict[str, Any]],
    x_key: str,
    threshold: float = R2_THRESHOLD,
) -> float | None:
    ordered = sorted(rows, key=lambda row: float(row[x_key]))
    crossing = next(
        (index for index, row in enumerate(ordered) if float(row["R2"]) >= threshold),
        None,
    )
    if crossing is None:
        return None
    if crossing == 0:
        return float(ordered[0][x_key])
    left, right = ordered[crossing - 1], ordered[crossing]
    return _log_interpolate(
        float(left[x_key]),
        float(left["R2"]),
        float(right[x_key]),
        float(right["R2"]),
        threshold,
    )


def _annotate_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    for scenario_id in mixed.SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario_id]
        coupled = {
            int(row["N"]): row for row in scenario_rows if row["arm"] == "data_coupled"
        }
        decoupled = {
            int(row["N"]): row for row in scenario_rows if row["arm"] == "decoupled"
        }
        n_values = sorted(set(coupled) & set(decoupled))
        if not n_values:
            continue
        unit_noise = np.asarray([
            n * float(decoupled[n]["test_within_variance_per_device"])
            for n in n_values
        ])
        signal_variance = np.asarray([
            float(coupled[n]["between_variance_debiased"])
            for n in n_values
        ])
        w_mix = float(np.median(unit_noise))
        b_mix = float(np.median(signal_variance))
        sigma_squared = w_mix / max(b_mix, EPS)
        train_count = int(coupled[n_values[0]]["train_replications"])
        finite_factor = (R2_THRESHOLD + 1.0 / train_count) / (1.0 - R2_THRESHOLD)
        n_star_95 = finite_factor * sigma_squared
        mixture = _mixture_metrics(
            mixed._normalized_weights(mixed.SCENARIOS[scenario_id]["weights"])
        )
        coupled_rows: list[dict[str, Any]] = []
        for n in n_values:
            coupled_row = coupled[n]
            coupled_variance = float(coupled_row["test_within_variance_per_device"])
            decoupled_variance = float(decoupled[n]["test_within_variance_per_device"])
            n_eff = n * decoupled_variance / max(coupled_variance, EPS)
            equivalent_rho = (
                (n / max(n_eff, EPS) - 1.0) / (n - 1.0)
                if n > 1 else 0.0
            )
            coupled_row.update({
                **mixture,
                "W_mix": w_mix,
                "B_mix": b_mix,
                "sigma_squared": sigma_squared,
                "N_star_95": n_star_95,
                "N_eff": n_eff,
                "N_eff_over_N": n_eff / n,
                "equivalent_rho": equivalent_rho,
                "Gamma": n_eff / max(n_star_95, EPS),
                "R2_decoupled": float(decoupled[n]["R2"]),
                "coupling_R2_penalty": float(decoupled[n]["R2"]) - float(coupled_row["R2"]),
                "variance_ratio_coupled_over_decoupled": coupled_variance / max(decoupled_variance, EPS),
            })
            coupled_rows.append(coupled_row)
        n95 = _first_crossing(coupled_rows, "N")
        gamma_c = _first_crossing(coupled_rows, "Gamma")
        summaries.append({
            "scenario": scenario_id,
            "scenario_label": mixed.SCENARIOS[scenario_id]["label"],
            **mixture,
            "W_mix": w_mix,
            "W_mix_cv_across_N": float(np.std(unit_noise) / max(np.mean(unit_noise), EPS)),
            "B_mix": b_mix,
            "B_mix_cv_across_N": float(np.std(signal_variance) / max(np.mean(signal_variance), EPS)),
            "sigma_squared": sigma_squared,
            "N_star_95": n_star_95,
            "N95_test": n95,
            "Gamma_c_test": gamma_c,
            "boundary_status": "reached" if n95 is not None else "right_censored",
            "R2_at_max_N": float(coupled[max(n_values)]["R2"]),
            "R2_decoupled_at_max_N": float(decoupled[max(n_values)]["R2"]),
            "N_eff_at_max_N": float(coupled[max(n_values)]["N_eff"]),
            "Gamma_at_max_N": float(coupled[max(n_values)]["Gamma"]),
            "mean_coupling_R2_penalty": float(np.mean([
                float(row["coupling_R2_penalty"]) for row in coupled_rows
            ])),
        })
    main_rows = [row for row in rows if row["arm"] == "data_coupled"]
    return main_rows, summaries


def _binned_variance(rows: list[dict[str, Any]], key: str, bins: int = 7) -> dict[str, Any]:
    valid = [
        row for row in rows
        if np.isfinite(float(row[key])) and float(row[key]) > 0
    ]
    if len(valid) < bins * 2:
        return {"bins": 0, "mean_between_scenario_variance": None}
    values = np.log10(np.asarray([float(row[key]) for row in valid]))
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return {"bins": 0, "mean_between_scenario_variance": None}
    labels = np.clip(
        np.searchsorted(edges, values, side="right") - 1,
        0,
        len(edges) - 2,
    )
    variances: list[float] = []
    for label in sorted(set(labels)):
        members = np.flatnonzero(labels == label)
        by_scenario: dict[str, list[float]] = {}
        for index in members:
            by_scenario.setdefault(str(valid[index]["scenario"]), []).append(
                float(valid[index]["R2"])
            )
        if len(by_scenario) >= 2:
            variances.append(float(np.var([np.mean(values_) for values_ in by_scenario.values()])))
    return {
        "bins": len(variances),
        "mean_between_scenario_variance": float(np.mean(variances)) if variances else None,
    }


def _leave_one_scenario_out(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    audit_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for key in ("N", "Gamma"):
        errors: list[float] = []
        for held_out in scenarios:
            train = [row for row in rows if row["scenario"] != held_out]
            test = [row for row in rows if row["scenario"] == held_out]
            train_x = np.log(np.asarray([float(row[key]) for row in train]))
            train_y = np.asarray([float(row["R2"]) for row in train])
            degree = min(3, len(np.unique(train_x)) - 1)
            coefficients = np.polyfit(train_x, train_y, degree)
            test_x = np.log(np.asarray([float(row[key]) for row in test]))
            test_y = np.asarray([float(row["R2"]) for row in test])
            predicted = np.polyval(coefficients, test_x)
            residual = test_y - predicted
            errors.extend(residual.tolist())
            audit_rows.append({
                "coordinate": key,
                "held_out_scenario": held_out,
                "cells": len(test),
                "polynomial_degree_log_coordinate": degree,
                "RMSE_R2": float(np.sqrt(np.mean(residual**2))),
                "MAE_R2": float(np.mean(np.abs(residual))),
            })
        error_array = np.asarray(errors)
        summaries[key] = {
            "RMSE_R2": float(np.sqrt(np.mean(error_array**2))),
            "MAE_R2": float(np.mean(np.abs(error_array))),
            "held_out_scenarios": len(scenarios),
        }
    ratio = summaries["Gamma"]["RMSE_R2"] / max(summaries["N"]["RMSE_R2"], EPS)
    return {
        "method": "leave one mixture out; a cubic polynomial fit of the common R2 curve in log coordinates",
        "physical_N": summaries["N"],
        "Gamma": summaries["Gamma"],
        "RMSE_ratio_Gamma_over_N": ratio,
        "gamma_improves_out_of_scenario_prediction": bool(ratio < 1.0),
    }, audit_rows


def _collapse_metrics(
    rows: list[dict[str, Any]],
    leave_one_out: dict[str, Any],
) -> dict[str, Any]:
    gamma = _binned_variance(rows, "Gamma")
    physical = _binned_variance(rows, "N")
    gamma_value = gamma["mean_between_scenario_variance"]
    physical_value = physical["mean_between_scenario_variance"]
    ratio = (
        gamma_value / physical_value
        if gamma_value is not None and physical_value not in (None, 0.0)
        else None
    )
    return {
        "definition": "between-mixture variance of R2 within equal-occupancy bins; lower is better",
        "gamma_axis": gamma,
        "physical_N_axis": physical,
        "variance_ratio_gamma_over_N": ratio,
        "gamma_improves_collapse": bool(ratio is not None and ratio < 1.0),
        "leave_one_scenario_out": leave_one_out,
        "primary_conclusion": (
            "Gamma improves cross-scenario prediction"
            if leave_one_out["gamma_improves_out_of_scenario_prediction"]
            else "Gamma does not improve cross-scenario prediction over physical N"
        ),
    }


def _scenario_family(scenario_id: str) -> str:
    return scenario_id.split("-")[0]


def _plot_r2_panel(
    axis: Any,
    scenario_rows: list[dict[str, Any]],
    compact: bool = False,
) -> None:
    styles = {
        "data_coupled": {
            "label": "Data-coupled mixed fleet",
            "color": "#2166ac",
            "marker": "o",
            "linestyle": "-",
        },
        "decoupled": {
            "label": "Decoupled negative control",
            "color": "#d6604d",
            "marker": "s",
            "linestyle": "--",
        },
    }
    for arm, style in styles.items():
        block = sorted(
            [row for row in scenario_rows if row["arm"] == arm],
            key=lambda row: int(row["N"]),
        )
        n_values = np.asarray([int(row["N"]) for row in block], dtype=float)
        r2_values = np.asarray([float(row["R2"]) for row in block])
        lower = np.asarray([float(row["R2_ci_lower"]) for row in block])
        upper = np.asarray([float(row["R2_ci_upper"]) for row in block])
        axis.fill_between(
            n_values,
            lower,
            upper,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(
            n_values,
            r2_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.7 if compact else 2.1,
            markersize=3.5 if compact else 5.0,
            label=style["label"],
        )
    axis.axhline(
        R2_THRESHOLD,
        color="#8e1b1b",
        linestyle=":",
        linewidth=1.3,
        label=r"$R^2=0.95$ threshold",
    )
    axis.set_xscale("log")
    axis.set_xlim(25, 6200)
    axis.set_ylim(0.80, 1.005)
    axis.grid(alpha=0.22)


def _save_r2_curves(
    arm_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> list[str]:
    scenario_ids = list(mixed.SCENARIOS)
    generated: list[str] = []
    individual_dir = figure_dir / "r2_curves"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id in scenario_ids:
        scenario_rows = [row for row in arm_rows if row["scenario"] == scenario_id]
        label = mixed.SCENARIOS[scenario_id]["label"]
        fig, axis = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
        _plot_r2_panel(axis, scenario_rows)
        axis.set_xlabel(r"Physical fleet size $N$")
        axis.set_ylabel(r"Independent test $R^2$")
        axis.set_title(f"{scenario_id}: {label}\nE21 mixed-scenario fleet-size evaluation")
        axis.legend(frameon=False, loc="lower right")
        path = individual_dir / f"e21_mixed_r2_{scenario_id}.png"
        fig.savefig(path, dpi=240)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        generated.extend([str(path), str(path.with_suffix(".pdf"))])

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(17, 14.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for axis, scenario_id in zip(axes.flat, scenario_ids):
        scenario_rows = [row for row in arm_rows if row["scenario"] == scenario_id]
        _plot_r2_panel(axis, scenario_rows, compact=True)
        axis.set_title(
            f"{scenario_id}\n{mixed.SCENARIOS[scenario_id]['label']}",
            fontsize=9.5,
        )
    for axis in axes.flat[len(scenario_ids):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[len(scenario_ids)].legend(
        handles,
        labels,
        loc="center",
        ncol=1,
        frameon=False,
    )
    fig.supxlabel(r"Physical fleet size $N$")
    fig.supylabel(r"Independent test $R^2$")
    fig.suptitle(
        "E21 dataset-mixed scenarios under the fleet-size protocol",
        fontsize=15,
    )
    path = figure_dir / "e21_mixed_r2_group.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    generated.extend([str(path), str(path.with_suffix(".pdf"))])
    return generated


def _save_figures(
    rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    figure_dir: Path,
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    scenario_ids = list(mixed.SCENARIOS)
    dataset_ids = list(mixed.ALL_DATASETS)
    weights = np.asarray([
        [
            mixed._normalized_weights(mixed.SCENARIOS[scenario]["weights"]).get(dataset, 0.0)
            for dataset in dataset_ids
        ]
        for scenario in scenario_ids
    ])
    generated: list[str] = []

    fig, axis = plt.subplots(figsize=(15, 8), constrained_layout=True)
    image = axis.imshow(weights, cmap="YlGnBu", vmin=0.0, vmax=float(np.max(weights)), aspect="auto")
    axis.set_xticks(
        np.arange(len(dataset_ids)),
        [SHORT_NAMES[dataset] for dataset in dataset_ids],
        rotation=55,
        ha="right",
    )
    axis.set_yticks(np.arange(len(scenario_ids)), scenario_ids)
    axis.set_xlabel("Source dataset")
    axis.set_ylabel("E21 mixed scenario")
    axis.set_title("E21 mixed-fleet composition used by the Gamma experiment")
    fig.colorbar(image, ax=axis, label="Device fraction")
    path = figure_dir / "mixed_composition_heatmap.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    generated.extend([str(path), str(path.with_suffix(".pdf"))])

    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, len(scenario_ids)))
    markers = {"S1": "o", "S2": "s", "S3": "^", "S4": "D", "S5": "P", "S6": "X"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), constrained_layout=True, sharey=True)
    for color, scenario in zip(colors, scenario_ids):
        block = sorted([row for row in rows if row["scenario"] == scenario], key=lambda row: int(row["N"]))
        marker = markers[_scenario_family(scenario)]
        axes[0].plot([row["N"] for row in block], [row["R2"] for row in block], marker=marker, ms=4, lw=1.0, color=color, alpha=0.82, label=scenario)
        axes[1].plot([row["Gamma"] for row in block], [row["R2"] for row in block], marker=marker, ms=4, lw=1.0, color=color, alpha=0.82, label=scenario)
        axes[1].scatter([row["Gamma"] for row in block], [row["R2"] for row in block], s=[18 + 11 * math.log10(float(row["N"])) for row in block], color=color, alpha=0.25, edgecolors="none")
    r2_min = min(float(row["R2_ci_lower"]) for row in rows)
    y_min = min(-0.05, math.floor((r2_min - 0.03) * 10.0) / 10.0)
    for axis in axes:
        axis.set_xscale("log")
        axis.axhline(R2_THRESHOLD, color="#b2182b", ls=":", lw=1.2)
        axis.set_ylim(y_min, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_xlabel(r"Physical fleet size $N$")
    axes[0].set_ylabel(r"Independent test $R^2$")
    axes[0].set_title("a  Physical-scale response predictability")
    axes[1].axvline(1.0, color="#222222", ls="--", lw=1.0)
    axes[1].set_xlabel(r"Mixed-fleet normalized scale $\Gamma$")
    axes[1].set_title("b  Mixed-fleet Gamma collapse")
    axes[1].legend(ncol=2, fontsize=7, frameon=False, loc="lower right")
    path = figure_dir / "mixed_gamma_scaling.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    generated.extend([str(path), str(path.with_suffix(".pdf"))])

    ordered = sorted(summaries, key=lambda row: (row["N95_test"] is None, row["N95_test"] or math.inf))
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), constrained_layout=True)
    reached = [row["N95_test"] is not None for row in ordered]
    n95_values = [row["N95_test"] if row["N95_test"] is not None else max(DEFAULT_N_VALUES) for row in ordered]
    axes[0].scatter(n95_values, y, c=["#2166ac" if value else "#b2182b" for value in reached], s=48)
    axes[0].set_xscale("log")
    axes[0].set_yticks(y, [row["scenario"] for row in ordered])
    axes[0].set_xlabel(r"Physical threshold $N_{95}$")
    axes[0].set_title("a  Test R2 threshold by mixed scenario")
    gamma_values = [row["Gamma_c_test"] if row["Gamma_c_test"] is not None else row["Gamma_at_max_N"] for row in ordered]
    axes[1].scatter(gamma_values, y, c=["#2166ac" if value else "#b2182b" for value in reached], s=48)
    axes[1].axvline(1.0, color="#222222", ls="--", lw=1.0)
    axes[1].set_xscale("log")
    axes[1].set_yticks(y, [row["scenario"] for row in ordered])
    axes[1].set_xlabel(r"Critical normalized scale $\Gamma_c$")
    axes[1].set_title("b  Normalized boundary stability")
    for axis in axes:
        axis.grid(alpha=0.2)
    path = figure_dir / "mixed_gamma_boundaries.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    generated.extend([str(path), str(path.with_suffix(".pdf"))])
    generated.extend(_save_r2_curves(arm_rows, figure_dir))
    return generated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    mixed._validate_scenarios()
    parser = argparse.ArgumentParser(description="Command line entry point for run e21 mixed gamma boundary.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    parser.add_argument("--train-replications", type=int, default=10)
    parser.add_argument("--test-replications", type=int, default=30)
    parser.add_argument("--conditions", type=int, default=48)
    parser.add_argument("--bootstrap-draws", type=int, default=400)
    parser.add_argument("--workers", type=int, default=max(1, min(3, (os.cpu_count() or 2) // 4)))
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.n_values = sorted({int(value) for value in args.n_values if int(value) >= 2})
    output = args.output
    raw_dir = output / "raw" / "responses"
    data_dir = output / "data"
    figure_dir = output / "figures"
    checkpoint_path = data_dir / "checkpoint.json"
    for directory in (raw_dir, data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    if checkpoint_path.is_file() and not args.force:
        completed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed_keys = {(row["scenario"], int(row["N"])) for row in completed}
    tasks: list[tuple[Any, ...]] = []
    for scenario in mixed.SCENARIOS:
        for n in args.n_values:
            if (scenario, n) in completed_keys:
                continue
            tasks.append((
                scenario,
                n,
                args.train_replications,
                args.test_replications,
                args.conditions,
                args.bootstrap_draws,
                args.seed,
                str(raw_dir / scenario / f"responses_N{n}.npz"),
            ))

    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_cell, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failure = {"scenario": task[0], "N": task[1], "error": repr(exc)}
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
                continue
            completed.append(result)
            completed.sort(key=lambda row: (list(mixed.SCENARIOS).index(row["scenario"]), int(row["N"])))
            _write_json(checkpoint_path, completed)
            print(json.dumps({
                "stage": "mixed_gamma_cell",
                "scenario": result["scenario"],
                "N": result["N"],
                "completed": len(completed),
                "total": len(mixed.SCENARIOS) * len(args.n_values),
            }, ensure_ascii=False), flush=True)
    if failures:
        _write_json(output / "failures.json", failures)
        raise SystemExit(1)

    arm_rows = [row for result in completed for row in result["rows"]]
    main_rows, summaries = _annotate_metrics(arm_rows)
    leave_one_out, leave_one_out_rows = _leave_one_scenario_out(main_rows)
    collapse = _collapse_metrics(main_rows, leave_one_out)
    _write_rows(data_dir / "mixed_gamma_all_arms.csv", arm_rows)
    _write_rows(data_dir / "mixed_gamma_points.csv", main_rows)
    _write_rows(data_dir / "mixed_gamma_scenario_summary.csv", summaries)
    _write_rows(data_dir / "leave_one_scenario_out.csv", leave_one_out_rows)
    _write_json(data_dir / "collapse_quality.json", collapse)
    _write_json(
        data_dir / "fleet_audit.json",
        [{"scenario": row["scenario"], "N": row["N"], "audit": row["audit"]} for row in completed],
    )
    figures = _save_figures(main_rows, arm_rows, summaries, figure_dir)
    manifest = {
        "experiment": PROTOCOL,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(mixed.SCENARIOS),
        "cell_count": len(completed),
        "n_values": args.n_values,
        "train_replications": args.train_replications,
        "test_replications": args.test_replications,
        "conditions": args.conditions,
        "bootstrap_draws": args.bootstrap_draws,
        "network_mode": "aggregate",
        "availability_arms": ["data_coupled", "decoupled"],
        "r2_threshold": R2_THRESHOLD,
        "collapse_quality": collapse,
        "figures": figures,
        "generated_files": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["generated_files"].append({
                "path": str(path.relative_to(output)),
                "sha256": _sha256(path),
            })
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "status": "completed",
        "output": str(output),
        "scenarios": len(summaries),
        "cells": len(main_rows),
        "collapse": collapse,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
