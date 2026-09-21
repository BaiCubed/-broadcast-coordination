from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .run import (
    DATASETS,
    _compact_result_artifacts,
    _refresh_rho_scan,
    _write_json,
    _write_yaml,
    dataset_results_root,
)
from ..ieee33_device_day_simulation.config_loader import load_config
from ..ieee33_device_day_simulation.experiments.run_experiment import (
    _split_zone_batches,
    run_cross_region_transfer_experiments_protocol,
    run_default,
)
from ..ieee33_device_day_simulation.experiments.protocol import (
    build_stratified_signal_schedule,
    simulate_day,
)
from ..ieee33_device_day_simulation.figures.plot_figures import plot_all, plot_figure4
from ..ieee33_device_day_simulation.population.device_day_loader import load_device_day_pool


EXPERIMENT_DIR = "network_constrained_new"
STATUS_PATH = Path("results") / "network_constrained_new_status.json"
BASELINE_QUANTILE = 0.95
BASELINE_TARGET_LOADING = 0.95
EXPANDED_BASELINE_TARGET_LOADING = 1.0
TARGET_CONSTRAINED_FRACTION = 0.25
MIN_CONSERVATIVE_CONSTRAINED_FRACTION = 0.10
MAX_BASELINE_INFEASIBLE_FRACTION = 0.05
MAX_FULLY_BLOCKED_FRACTION = 0.05


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_requirement_ratios(
    trace: dict[str, Any],
    *,
    voltage_limits: tuple[float, float],
) -> np.ndarray:
    vmin, vmax = voltage_limits
    lower_span = max(1.0 - float(vmin), 1e-9)
    upper_span = max(float(vmax) - 1.0, 1e-9)
    transformer = np.asarray(trace["baseline_transformer_loading"], dtype=float)
    branches = np.asarray(trace["baseline_maximum_branch_loading"], dtype=float)
    low_voltage = (1.0 - np.asarray(trace["baseline_minimum_voltage_pu"], dtype=float)) / lower_span
    high_voltage = (np.asarray(trace["baseline_maximum_voltage_pu"], dtype=float) - 1.0) / upper_span
    return np.maximum.reduce((transformer, branches, low_voltage, high_voltage))


def _select_constrained_multiplier(
    baseline_trace: dict[str, Any],
    *,
    baseline_multiplier: float,
    voltage_limits: tuple[float, float],
    quantile: float = BASELINE_QUANTILE,
    target_loading: float = BASELINE_TARGET_LOADING,
) -> dict[str, float]:
    requirements = _baseline_requirement_ratios(
        baseline_trace,
        voltage_limits=voltage_limits,
    )
    requirement_quantile = float(np.quantile(requirements, quantile))
    relative_multiplier = min(
        requirement_quantile / max(float(target_loading), 1e-9),
        1.0,
    )
    relative_multiplier = max(relative_multiplier, 0.005)
    return {
        "baseline_multiplier": float(baseline_multiplier),
        "baseline_requirement_quantile": requirement_quantile,
        "quantile": float(quantile),
        "target_loading": float(target_loading),
        "relative_multiplier": float(relative_multiplier),
        "selected_multiplier": float(baseline_multiplier * relative_multiplier),
    }


def _pilot_constraint_summary(config: dict[str, Any]) -> dict[str, Any]:
    pool = load_device_day_pool(config)
    population = config["population"]
    experiment = config["experiment"]
    per_zone = int(population["resources_per_zone"])
    validation_batches = int(experiment["validation_batches"])
    seed = int(config["simulation"]["random_seed"])
    sample_seed = seed + int(population["seed_offset"])
    if bool(population.get("unique_source_per_batch", False)):
        batches = pool.sample_source_unique_batches(
            per_zone=per_zone,
            batch_count=1 + validation_batches,
            seed=sample_seed,
        )
    else:
        sampled = pool.sample(
            per_zone=per_zone * (1 + validation_batches),
            seed=sample_seed,
            with_replacement=bool(population["with_replacement"]),
        )
        batches = _split_zone_batches(
            sampled,
            per_zone=per_zone,
            batch_count=1 + validation_batches,
        )
    validation_records = batches[1]
    validation_samples = int(experiment["validation_samples"])
    if bool(experiment.get("balanced_validation_samples", False)):
        steps = validation_samples // max(validation_batches, 1)
        if validation_samples % max(validation_batches, 1):
            steps += 1
    else:
        steps = validation_samples
    steps = min(int(config["simulation"]["steps_per_day"]), steps)

    horizon = max(
        int(experiment["training_steps"]),
        validation_samples,
        int(experiment["scenario_steps"]),
    )
    profile_rng = np.random.default_rng(
        seed + int(experiment.get("snapshot_profile_seed_offset", 0))
    )
    profile_steps = profile_rng.integers(
        0,
        int(config["simulation"]["steps_per_day"]),
        size=horizon,
    ).astype(int).tolist()
    schedule = build_stratified_signal_schedule(
        config,
        validation_samples,
        seed=seed + 71002,
        min_abs_score=float(experiment.get("estimator_min_abs_signal", 0.05)),
        max_abs_score=float(experiment.get("estimator_max_abs_signal", 0.95)),
    )
    trace = simulate_day(
        validation_records,
        config,
        steps=steps,
        seed=seed + 2,
        profile_steps=profile_steps[:steps],
        signal_overrides=schedule[:steps],
        reset_each_step=True,
    )
    return _constraint_summary(
        trace.as_dict(),
        voltage_limits=tuple(float(value) for value in config["network"]["voltage_limits_pu"]),
    )


def _calibrate_paired_validation(
    config: dict[str, Any],
    initial: dict[str, float],
) -> dict[str, Any]:
    lower = float(initial["selected_multiplier"])
    upper = float(initial["baseline_multiplier"])
    candidates = np.unique(np.geomspace(lower, upper, 7))
    evaluations: list[dict[str, Any]] = []

    def evaluate(multiplier: float) -> dict[str, Any]:
        candidate_config = copy.deepcopy(config)
        candidate_config["control"]["network_capacity_multiplier"] = float(multiplier)
        candidate_config["control"]["network_equivalent_scale"] = float(multiplier)
        candidate_config["control"]["network_feedback"] = True
        summary = _pilot_constraint_summary(candidate_config)
        row = {"multiplier": float(multiplier), **summary}
        evaluations.append(row)
        return row

    for multiplier in candidates:
        evaluate(float(multiplier))

    ordered = sorted(evaluations, key=lambda row: float(row["multiplier"]))
    bracket = next(
        (
            (left, right)
            for left, right in zip(ordered, ordered[1:])
            if float(left["dispatch_limited_baseline_feasible_fraction"])
            >= TARGET_CONSTRAINED_FRACTION
            >= float(right["dispatch_limited_baseline_feasible_fraction"])
        ),
        None,
    )
    if bracket is not None:
        high_clipping, low_clipping = bracket
        for _ in range(4):
            midpoint = float(np.sqrt(
                float(high_clipping["multiplier"])
                * float(low_clipping["multiplier"])
            ))
            row = evaluate(midpoint)
            if float(row["dispatch_limited_baseline_feasible_fraction"]) >= TARGET_CONSTRAINED_FRACTION:
                high_clipping = row
            else:
                low_clipping = row

    eligible = [
        row
        for row in evaluations
        if row["baseline_infeasible_fraction"] <= MAX_BASELINE_INFEASIBLE_FRACTION
        and row["fully_blocked_fraction"] <= MAX_FULLY_BLOCKED_FRACTION
    ]
    choices = eligible or evaluations
    selected = min(
        choices,
        key=lambda row: (
            abs(
                float(row["dispatch_limited_baseline_feasible_fraction"])
                - TARGET_CONSTRAINED_FRACTION
            ),
            float(row["baseline_infeasible_fraction"]),
            float(row["fully_blocked_fraction"]),
        ),
    )
    return {
        **initial,
        "lower_bound_multiplier": lower,
        "selection_target": {
            "dispatch_limited_baseline_feasible_fraction": TARGET_CONSTRAINED_FRACTION,
            "maximum_baseline_infeasible_fraction": MAX_BASELINE_INFEASIBLE_FRACTION,
            "maximum_fully_blocked_fraction": MAX_FULLY_BLOCKED_FRACTION,
        },
        "candidate_evaluations": evaluations,
        "selected_multiplier": float(selected["multiplier"]),
        "selected_pilot_summary": selected,
    }


def _prepare_paired_config(dataset_root: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    baseline_config_dir = dataset_root / "coverage_fix" / "config"
    baseline_default = baseline_config_dir / "default_network_stress.yaml"
    baseline_trace_path = dataset_root / "coverage_fix" / "network_stress" / "data" / "network_timeseries.json"
    if not baseline_default.is_file() or not baseline_trace_path.is_file():
        raise FileNotFoundError(f"missing coverage_fix baseline under {dataset_root}")

    output_config_dir = output_root / "config"
    if output_config_dir.exists():
        shutil.rmtree(output_config_dir)
    shutil.copytree(baseline_config_dir, output_config_dir)

    default_payload = yaml.safe_load(baseline_default.read_text(encoding="utf-8"))
    baseline_control_name = str(default_payload["control"])
    baseline_control = yaml.safe_load(
        (baseline_config_dir / baseline_control_name).read_text(encoding="utf-8")
    )
    network = yaml.safe_load(
        (baseline_config_dir / str(default_payload["network"])).read_text(encoding="utf-8")
    )
    baseline_trace = _read_json(baseline_trace_path)
    voltage_limits = tuple(float(value) for value in network["voltage_limits_pu"])
    initial_calibration = _select_constrained_multiplier(
        baseline_trace,
        baseline_multiplier=float(baseline_control["network_capacity_multiplier"]),
        voltage_limits=voltage_limits,
    )

    constrained_control = dict(baseline_control)
    constrained_control["network_capacity_multiplier"] = initial_calibration["selected_multiplier"]
    constrained_control["network_equivalent_scale"] = initial_calibration["selected_multiplier"]
    constrained_control["network_feedback"] = True
    control_name = "control_network_constrained_new.yaml"
    _write_yaml(output_config_dir / control_name, constrained_control)

    experiment_name = str(default_payload["experiment"])
    experiment = yaml.safe_load(
        (output_config_dir / experiment_name).read_text(encoding="utf-8")
    )
    experiment["force_network_feedback"] = True
    experiment["experiment_variant"] = "paired_network_constrained_new"
    experiment["constraint_calibration"] = initial_calibration
    _write_yaml(output_config_dir / experiment_name, experiment)

    default_payload["control"] = control_name
    constrained_default = output_config_dir / "default_network_constrained_new.yaml"
    _write_yaml(constrained_default, default_payload)
    pilot_config = load_config(constrained_default)
    calibration = _calibrate_paired_validation(pilot_config, initial_calibration)
    calibration["calibration_strategy"] = "conservative_baseline_headroom"
    if (
        float(calibration["selected_pilot_summary"]["dispatch_limited_baseline_feasible_fraction"])
        < MIN_CONSERVATIVE_CONSTRAINED_FRACTION
    ):
        expanded_initial = _select_constrained_multiplier(
            baseline_trace,
            baseline_multiplier=float(baseline_control["network_capacity_multiplier"]),
            voltage_limits=voltage_limits,
            target_loading=EXPANDED_BASELINE_TARGET_LOADING,
        )
        expanded_config = copy.deepcopy(pilot_config)
        expanded_config["control"]["network_capacity_multiplier"] = expanded_initial[
            "selected_multiplier"
        ]
        expanded_config["control"]["network_equivalent_scale"] = expanded_initial[
            "selected_multiplier"
        ]
        expanded = _calibrate_paired_validation(expanded_config, expanded_initial)
        expanded_summary = expanded["selected_pilot_summary"]
        if (
            float(expanded_summary["dispatch_limited_baseline_feasible_fraction"])
            > float(calibration["selected_pilot_summary"]["dispatch_limited_baseline_feasible_fraction"])
            and float(expanded_summary["baseline_infeasible_fraction"])
            <= MAX_BASELINE_INFEASIBLE_FRACTION
            and float(expanded_summary["fully_blocked_fraction"])
            <= MAX_FULLY_BLOCKED_FRACTION
        ):
            expanded["calibration_strategy"] = "expanded_after_weak_conservative_activation"
            expanded["conservative_selected_pilot_summary"] = calibration[
                "selected_pilot_summary"
            ]
            calibration = expanded
    constrained_control["network_capacity_multiplier"] = calibration["selected_multiplier"]
    constrained_control["network_equivalent_scale"] = calibration["selected_multiplier"]
    _write_yaml(output_config_dir / control_name, constrained_control)

    experiment["constraint_calibration"] = calibration
    _write_yaml(output_config_dir / experiment_name, experiment)
    return constrained_default, calibration


def _constraint_summary(
    trace: dict[str, Any],
    *,
    voltage_limits: tuple[float, float],
) -> dict[str, Any]:
    scale = np.asarray(trace["network_scale"], dtype=float)
    desired = np.asarray(trace["desired_control_kw"], dtype=float)
    accepted = np.asarray(trace["accepted_control_kw"], dtype=float)
    baseline_requirement = _baseline_requirement_ratios(
        trace,
        voltage_limits=voltage_limits,
    )
    baseline_feasible = baseline_requirement <= 1.0 + 1e-9
    constrained = scale < 1.0 - 1e-9
    return {
        "samples": int(scale.size),
        "constrained_samples": int(np.sum(constrained)),
        "constrained_fraction": float(np.mean(constrained)),
        "dispatch_limited_baseline_feasible_samples": int(np.sum(constrained & baseline_feasible)),
        "dispatch_limited_baseline_feasible_fraction": float(np.mean(constrained & baseline_feasible)),
        "baseline_infeasible_samples": int(np.sum(~baseline_feasible)),
        "baseline_infeasible_fraction": float(np.mean(~baseline_feasible)),
        "fully_blocked_samples": int(np.sum(scale <= 0.05)),
        "fully_blocked_fraction": float(np.mean(scale <= 0.05)),
        "mean_network_scale": float(np.mean(scale)),
        "minimum_network_scale": float(np.min(scale)),
        "desired_abs_energy_kwh": float(np.sum(np.abs(desired)) * 5.0 / 60.0),
        "accepted_abs_energy_kwh": float(np.sum(np.abs(accepted)) * 5.0 / 60.0),
        "response_reduction_fraction": float(
            1.0 - np.sum(np.abs(accepted)) / max(np.sum(np.abs(desired)), 1e-9)
        ),
    }


def _plot_constraint_effects(
    baseline: dict[str, Any],
    constrained: dict[str, Any],
    output: Path,
) -> Path:
    desired = np.asarray(constrained["desired_control_kw"], dtype=float)
    accepted = np.asarray(constrained["accepted_control_kw"], dtype=float)
    scale = np.asarray(constrained["network_scale"], dtype=float)
    baseline_accepted = np.asarray(baseline["accepted_control_kw"], dtype=float)
    count = min(desired.size, accepted.size, baseline_accepted.size)
    desired, accepted, scale, baseline_accepted = (
        values[:count] for values in (desired, accepted, scale, baseline_accepted)
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    ax = axes[0, 0]
    ax.plot(baseline_accepted, color="#6b7280", lw=1.0, label="Baseline accepted")
    ax.plot(accepted, color="#0072b2", lw=1.0, label="Constrained accepted")
    ax.set(xlabel="Validation sample", ylabel="Response (kW)", title="A  Paired response trace")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    scatter = ax.scatter(desired, accepted, c=scale, cmap="viridis", vmin=0, vmax=1, s=15, alpha=0.8)
    limits = [float(min(np.min(desired), np.min(accepted))), float(max(np.max(desired), np.max(accepted)))]
    ax.plot(limits, limits, color="#4b5563", ls="--", lw=1)
    ax.set(xlabel="Desired response (kW)", ylabel="Accepted response (kW)", title="B  Network clipping")
    fig.colorbar(scatter, ax=ax, label="Network scale")

    ax = axes[1, 0]
    ax.hist(scale, bins=np.linspace(0, 1, 21), color="#009e73", edgecolor="white")
    ax.set(xlabel="Network scale", ylabel="Samples", title="C  Constraint severity")

    ax = axes[1, 1]
    order = np.argsort(desired)
    ax.plot(desired[order], baseline_accepted[order], color="#6b7280", lw=1.2, label="Baseline")
    ax.plot(desired[order], accepted[order], color="#d55e00", lw=1.2, label="Constrained")
    ax.set(xlabel="Desired response (kW), sorted", ylabel="Accepted response (kW)", title="D  Response envelope")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _run_one(dataset: str) -> tuple[str, bool, dict[str, Any]]:
    started = time.time()
    dataset_root = dataset_results_root(dataset)
    output_root = dataset_root / "coverage_fix" / EXPERIMENT_DIR
    item_status = output_root / "run_status.json"
    try:
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True)
        _write_json(item_status, {"dataset": dataset, "phase": "configuration"})
        config_path, calibration = _prepare_paired_config(dataset_root, output_root)

        _write_json(item_status, {"dataset": dataset, "phase": "main_experiment", "calibration": calibration})
        run_default(
            config_path=config_path,
            mode="network_stress",
            results_root=output_root,
            snapshot_protocol=True,
            snapshot_sampling="aligned",
        )

        _write_json(item_status, {"dataset": dataset, "phase": "figure4e_transfer", "calibration": calibration})
        run_cross_region_transfer_experiments_protocol(
            config_path,
            mode="network_stress",
            result_root=output_root,
        )

        _write_json(item_status, {"dataset": dataset, "phase": "figure5_rho_scan", "calibration": calibration})
        _refresh_rho_scan(
            config_path,
            "network_stress",
            output_root,
            network_feedback=True,
        )
        figures = plot_all(output_root, compact=True)

        config = load_config(config_path)
        trace = _read_json(output_root / "data" / "network_timeseries.json")
        baseline_trace = _read_json(dataset_root / "coverage_fix" / "network_stress" / "data" / "network_timeseries.json")
        summary = _constraint_summary(
            trace,
            voltage_limits=tuple(float(value) for value in config["network"]["voltage_limits_pu"]),
        )
        summary.update({"dataset": dataset, "calibration": calibration})
        _write_json(output_root / "data" / "constraint_summary.json", summary)
        constraint_figure = _plot_constraint_effects(
            baseline_trace,
            trace,
            output_root / "Figs" / "constraint_effects.png",
        )

        metadata_path = output_root / "data" / "protocol_metadata.json"
        metadata = _read_json(metadata_path)
        metadata["experiment_variant"] = "paired_network_constrained_new"
        metadata["constraint_calibration"] = calibration
        metadata["constraint_summary"] = summary
        _write_json(metadata_path, metadata)
        _compact_result_artifacts(output_root)

        estimator = _read_json(output_root / "estimation" / "estimation_validation_results.json")
        details = {
            "elapsed_seconds": round(time.time() - started, 1),
            "selected_multiplier": calibration["selected_multiplier"],
            "constrained_fraction": summary["constrained_fraction"],
            "dispatch_limited_baseline_feasible_fraction": summary["dispatch_limited_baseline_feasible_fraction"],
            "baseline_infeasible_fraction": summary["baseline_infeasible_fraction"],
            "fully_blocked_fraction": summary["fully_blocked_fraction"],
            "r2": estimator["point_metrics"]["r2"]["value"],
            "figures": [str(path) for path in [*figures, constraint_figure]],
        }
        _write_json(item_status, {"dataset": dataset, "phase": "completed", **details})
        return dataset, True, details
    except BaseException as exc:
        details = {
            "elapsed_seconds": round(time.time() - started, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_json(item_status, {"dataset": dataset, "phase": "failed", **details})
        return dataset, False, details


def _refresh_fig4e_one(dataset: str) -> tuple[str, bool, dict[str, Any]]:
    started = time.time()
    output_root = dataset_results_root(dataset) / "coverage_fix" / EXPERIMENT_DIR
    item_status = output_root / "run_status.json"
    try:
        config_path = output_root / "config" / "default_network_constrained_new.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing constrained config: {config_path}")
        _write_json(item_status, {"dataset": dataset, "phase": "figure4e_transfer_refresh"})
        transfer_path = run_cross_region_transfer_experiments_protocol(
            config_path,
            mode="network_stress",
            result_root=output_root,
        )
        figures = plot_figure4(output_root, compact=True)
        transfer = _read_json(transfer_path)
        cold = {
            key: float(value["cold_start"]["r2"])
            for key, value in transfer["regions"].items()
        }
        adapted = {
            key: float(value["adapted_nn"]["r2"])
            for key, value in transfer["regions"].items()
        }
        metadata_path = output_root / "data" / "protocol_metadata.json"
        metadata = _read_json(metadata_path)
        metadata["figure4e_transfer"] = {
            "protocol": transfer["protocol"],
            "training_samples": int(transfer["training_samples"]),
            "evaluation_samples_per_region": int(transfer["evaluation_samples_per_region"]),
            "online_samples_per_region": int(transfer["online_samples_per_region"]),
            "network_feedback": True,
        }
        _write_json(metadata_path, metadata)
        details = {
            "elapsed_seconds": round(time.time() - started, 1),
            "protocol": transfer["protocol"],
            "source_r2": float(transfer["region_A"]["r2"]),
            "cold_start_r2": cold,
            "adapted_r2": adapted,
            "figures": [str(path) for path in figures],
        }
        _write_json(item_status, {"dataset": dataset, "phase": "completed", "refresh": "figure4e_only", **details})
        return dataset, True, details
    except BaseException as exc:
        details = {
            "elapsed_seconds": round(time.time() - started, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_json(item_status, {"dataset": dataset, "phase": "failed", "refresh": "figure4e_only", **details})
        return dataset, False, details


def write_all_dataset_summary() -> Path:
    rows: list[dict[str, Any]] = []
    expected_figures = {
        "constraint_effects.png",
        "fig2_scaling_predictability.png",
        "fig3_scaling_heterogeneity.png",
        "fig4_robustness_generalization.png",
        "fig4_self_consumption.png",
        "fig5_correlation_effects.png",
    }
    for dataset in DATASETS:
        root = dataset_results_root(dataset) / "coverage_fix" / EXPERIMENT_DIR
        summary = _read_json(root / "data" / "constraint_summary.json")
        estimator = _read_json(root / "estimation" / "estimation_validation_results.json")
        metadata = _read_json(root / "data" / "protocol_metadata.json")
        rho = _read_json(root / "data" / "rho_sensitivity.json")
        transfer = _read_json(root / "data" / "cross_region_transfer.json")
        figures = {path.name for path in (root / "Figs").glob("*.png")}
        row = {
            "dataset": dataset,
            "result_directory": str(root.resolve()),
            "selected_multiplier": summary["calibration"]["selected_multiplier"],
            "constrained_fraction": summary["constrained_fraction"],
            "dispatch_limited_baseline_feasible_fraction": summary[
                "dispatch_limited_baseline_feasible_fraction"
            ],
            "baseline_infeasible_fraction": summary["baseline_infeasible_fraction"],
            "fully_blocked_fraction": summary["fully_blocked_fraction"],
            "response_reduction_fraction": summary["response_reduction_fraction"],
            "r2": estimator["point_metrics"]["r2"]["value"],
            "picp": estimator["interval_metrics"]["picp"],
            "pinaw": estimator["interval_metrics"]["pinaw"],
            "rho_points": len(rho.get("rho_scan", {})),
            "rho_network_feedback": bool(
                metadata["shared_settings"]["figure5_rho_scan"]["network_feedback"]
            ),
            "figure4e_protocol": transfer.get("protocol"),
            "figure4e_cold_start_r2": {
                key: value["cold_start"]["r2"]
                for key, value in transfer.get("regions", {}).items()
            },
            "figure4e_adapted_r2": {
                key: value["adapted_nn"]["r2"]
                for key, value in transfer.get("regions", {}).items()
            },
            "figures_complete": expected_figures <= figures,
            "run_completed": _read_json(root / "run_status.json").get("phase") == "completed",
        }
        row["audit_pass"] = bool(
            row["dispatch_limited_baseline_feasible_fraction"] > 0.0
            and row["baseline_infeasible_fraction"] <= MAX_BASELINE_INFEASIBLE_FRACTION
            and row["fully_blocked_fraction"] <= MAX_FULLY_BLOCKED_FRACTION
            and row["r2"] >= 0.95
            and row["rho_points"] == 6
            and row["rho_network_feedback"]
            and row["figure4e_protocol"]
            == "experiments_source_train_4000_target_eval_500_online_1000"
            and row["figures_complete"]
            and row["run_completed"]
        )
        rows.append(row)

    output = Path("results") / "network_constrained_new_all_datasets_summary.json"
    _write_json(output, {
        "protocol": "paired_network_constrained_new",
        "datasets": rows,
        "dataset_count": len(rows),
        "audit_pass_count": sum(bool(row["audit_pass"]) for row in rows),
        "all_passed": all(bool(row["audit_pass"]) for row in rows),
        "criteria": {
            "calibration_policy": (
                "95th-percentile baseline at 95% loading; expand to 100% only "
                "when conservative dispatch-limited activation is below 10%"
            ),
            "dispatch_limited_baseline_feasible_fraction": "> 0",
            "maximum_baseline_infeasible_fraction": MAX_BASELINE_INFEASIBLE_FRACTION,
            "maximum_fully_blocked_fraction": MAX_FULLY_BLOCKED_FRACTION,
            "minimum_r2": 0.95,
            "rho_points": 6,
            "rho_network_feedback": True,
            "figure4e_protocol": "experiments_source_train_4000_target_eval_500_online_1000",
            "required_figure_count": len(expected_figures),
        },
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired, actively constrained IEEE33 experiment for every dataset.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="*", choices=list(DATASETS))
    parser.add_argument(
        "--fig4e-only",
        action="store_true",
        help="overwrite aligned cross-region JSON and the two Figure 4 images only",
    )
    args = parser.parse_args()
    datasets = list(args.datasets or DATASETS)
    status: dict[str, Any] = {
        "protocol": (
            "experiments_source_train_4000_target_eval_500_online_1000"
            if args.fig4e_only
            else "paired_network_constrained_new"
        ),
        "output_subdirectory": EXPERIMENT_DIR,
        "pending": datasets.copy(),
        "running": [],
        "completed": {},
        "failed": {},
    }
    _write_json(STATUS_PATH, status)

    worker = _refresh_fig4e_one if args.fig4e_only else _run_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, dataset): dataset for dataset in datasets}
        status["running"] = datasets[: args.workers]
        status["pending"] = datasets[args.workers :]
        _write_json(STATUS_PATH, status)
        for future in concurrent.futures.as_completed(futures):
            dataset, succeeded, details = future.result()
            destination = status["completed"] if succeeded else status["failed"]
            destination[dataset] = details
            status["running"] = [item for item in status["running"] if item != dataset]
            if status["pending"]:
                status["running"].append(status["pending"].pop(0))
            _write_json(STATUS_PATH, status)
            print(f"=== {'DONE' if succeeded else 'FAILED'} {dataset} {details['elapsed_seconds']}s ===", flush=True)

    if not status["failed"]:
        write_all_dataset_summary()
    print(json.dumps({"completed": list(status["completed"]), "failed": status["failed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
