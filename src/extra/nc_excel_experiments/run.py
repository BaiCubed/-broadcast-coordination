from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.extra.ieee33_device_day_simulation.config_loader import load_config
from src.extra.ieee33_device_day_simulation.experiments.protocol import (
    _signal_values,
    build_condition_signal_schedule,
    build_real_signal_schedule,
    r2_score,
    simulate_day,
    trace_features,
)
from src.extra.ieee33_device_day_simulation.experiments.run_experiment import _balanced_subset
from src.extra.ieee33_device_day_simulation.population.device_day_loader import (
    DeviceDay,
    load_device_day_pool,
)

from .metrics import (
    bootstrap_slope_ci,
    conditional_mean_pairwise_rho,
    effective_n,
    failure_threshold,
    loglog_slope,
    network_delivery_metrics,
    nrmse,
    sign_consistency,
    wilson_interval,
)
from .topology import apply_topology
from .coupling import (
    availability_at_steps,
    broadcast_residual_rho,
    decoupled_copy,
    diurnal_residual,
    rank_availability,
)
from .local_policy import ParameterizedLocalPolicy
from .control_logic import CONTROL_LOGICS, build_policy
from .dominant import imbalance_diagnostics, inject_dominant
from .longhorizon import (
    ARCHITECTURES,
    SocRecorder,
    build_architecture,
    daily_fleet_sequence,
    days_available,
    draw_initial_soc,
    set_initial_soc,
    spread_initial_soc,
)
from .phase import analyse_window, phase_histogram, replication_permutation_band
from .drift import (
    DriftedLocalPolicy,
    aggregate_refit,
    detection_index,
    drift_profile,
    drift_regret,
    false_alarm_rate,
    recovered_share,
    recovery_index,
    sample_complexity,
    update_bytes,
    window_losses,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "configs" / "protocol.yaml"
DATASET_RESULTS_SUFFIX = "_ieee33_real_load"
COLORS = {
    "blue": "#2166ac",
    "orange": "#d6604d",
    "green": "#1b9e77",
    "purple": "#7570b3",
    "gray": "#666666",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _result_root(protocol: dict[str, Any]) -> Path:
    path = Path(protocol["results_root"])
    return path if path.is_absolute() else ROOT / path


def _primary_config(protocol: dict[str, Any]) -> Path:
    path = Path(protocol["primary_config"])
    return path if path.is_absolute() else ROOT / path


def _sample_master(config: dict[str, Any], count: int, seed: int) -> list[DeviceDay]:
    pool = load_device_day_pool(config)
    return pool.sample_unique_sources(count, seed=seed)


def _balanced_unique_capacity(pool: Any) -> int:
    zone_counts: list[int] = []
    for zone in sorted({record.zone_id for record in pool.records}):
        zone_counts.append(len({record.source_device_id for record in pool.by_zone(zone)}))
    return int(len(zone_counts) * min(zone_counts)) if zone_counts else 0


def _profile_and_schedule(
    records: list[DeviceDay],
    config: dict[str, Any],
    conditions: int,
    seed: int,
    *,
    direction: str | None = None,
) -> tuple[list[int], list[list[tuple[int, int]]]]:
    rng = np.random.default_rng(seed)
    profile_steps = rng.integers(0, int(config["simulation"]["steps_per_day"]), size=conditions).tolist()
    if direction == "charge":
        scenarios = ["valley_filling"]
    elif direction == "discharge":
        scenarios = ["peak_shaving"]
    else:
        scenarios = ["valley_filling", "peak_shaving"]
    schedule = build_condition_signal_schedule(records, config, profile_steps, scenarios, seed=seed + 1)
    return profile_steps, schedule


def _fit_frozen_prediction(features: np.ndarray, train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conditional_mean = np.mean(train_values, axis=0)
    coefficients, _, _, _ = np.linalg.lstsq(features, conditional_mean, rcond=None)
    return coefficients, features @ coefficients


def _run_snapshots(
    records: list[DeviceDay],
    config: dict[str, Any],
    profile_steps: list[int],
    schedule: list[list[tuple[int, int]]],
    *,
    replications: int,
    seed: int,
    zone_correlation: float = 0.0,
    availability_factory: Any = None,
    keep_resources: bool = False,
    policy_factory: Any = None,
    policy_sink: list[Any] | None = None,
) -> tuple[np.ndarray, list[Any], np.ndarray | None, list[np.ndarray]]:
    aggregates: list[np.ndarray] = []
    traces: list[Any] = []
    resources: list[np.ndarray] = []
    active_counts: list[np.ndarray] = []
    for replication in range(replications):
        availability = None
        if availability_factory is not None:
            availability = availability_factory(replication)
            zone_availability = np.asarray([
                float(config["zones"]["zones"][record.zone_id]["availability_multiplier"])
                for record in records
            ])
            active_counts.append(np.sum(availability * zone_availability[None, :], axis=1))
        sink: list[list[float]] | None = [] if keep_resources else None
        policy = policy_factory(replication) if policy_factory is not None else None
        trace = simulate_day(
            records,
            config,
            steps=len(profile_steps),
            seed=seed + replication,
            profile_steps=profile_steps,
            signal_overrides=schedule,
            zone_correlation=zone_correlation,
            resource_response_sink=sink,
            availability_overrides=availability,
            response_policy=policy,
            reset_each_step=True,
        )
        aggregates.append(np.asarray(trace.accepted_control_kw, dtype=float))
        traces.append(trace)
        if policy is not None and policy_sink is not None:
            policy_sink.append(policy)
        if sink is not None:
            resources.append(np.asarray(sink, dtype=float))
    resource_array = np.asarray(resources, dtype=float) if resources else None
    return np.asarray(aggregates, dtype=float), traces, resource_array, active_counts


def _success_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    network_scale: np.ndarray,
    common: dict[str, Any],
) -> dict[str, Any]:
    error = nrmse(actual, target)
    signs = sign_consistency(actual, target)
    blocked = float(np.mean(network_scale <= 0.05))
    success = (
        error <= float(common["nrmse_limit"])
        and blocked <= float(common["blocked_fraction_limit"])
        and signs >= float(common["sign_consistency_limit"])
    )
    return {
        "nrmse": error,
        "sign_consistency": signs,
        "blocked_fraction": blocked,
        "controllable": bool(success),
    }


def _annotate_effective_margin(
    dataset: str,
    dataset_summary: list[dict[str, Any]],
    threshold: float,
    *,
    arm: str = "data_coupled",
) -> dict[str, Any]:
    ordered = sorted(
        [row for row in dataset_summary if row.get("arm", arm) == arm],
        key=lambda row: float(row["N_eff"]),
    )
    if not ordered:
        return {"dataset": dataset, "N_star_eff": None, "status": "no_cells"}
    lower = np.asarray([float(row["p_controllable_ci_lower"]) for row in ordered])
    effective = np.asarray([float(row["N_eff"]) for row in ordered])
    envelope = np.maximum.accumulate(lower)
    crossing = np.flatnonzero(envelope >= threshold)
    star = float(effective[crossing[0]]) if crossing.size else None
    suffix = np.minimum.accumulate(lower[::-1])[::-1]
    strict_crossing = np.flatnonzero(suffix >= threshold)
    strict = float(effective[strict_crossing[0]]) if strict_crossing.size else None
    for row in dataset_summary:
        row["N_star_eff"] = star
        row["margin"] = float(row["N_eff"]) / star if star else None
        row["N_eff_over_N"] = float(row["N_eff"]) / float(row["N"]) if row["N"] else None
    return {
        "dataset": dataset,
        "N_star_eff": star,
        "N_star_eff_strict": strict,
        "status": "reached" if star is not None else "not_reached",
        "N_eff_min": float(effective[0]),
        "N_eff_max": float(effective[-1]),
        "cells": len(ordered),
        "arm": arm,
    }


def _binned_between_dataset_variance(
    rows: list[dict[str, Any]], key: str, *, bins: int
) -> dict[str, Any]:
    if len(rows) < bins * 2:
        return {"bins": 0, "mean_between_dataset_variance": None, "cells": len(rows)}
    values = np.log10(np.asarray([float(row[key]) for row in rows]))
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges[-1] += 1e-9
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bins - 1)
    variances: list[float] = []
    spreads: list[float] = []
    for bin_index in range(bins):
        members = np.flatnonzero(index == bin_index)
        by_dataset: dict[str, list[float]] = {}
        for position in members:
            by_dataset.setdefault(
                str(rows[position]["dataset"]), []
            ).append(float(rows[position]["p_controllable"]))
        if len(by_dataset) < 2:
            continue
        means = [float(np.mean(value)) for value in by_dataset.values()]
        variances.append(float(np.var(means)))
        spreads.append(float(np.std(values[members])))
    if not variances:
        return {"bins": 0, "mean_between_dataset_variance": None, "cells": len(rows)}
    return {
        "bins_requested": bins,
        "bins_with_at_least_two_datasets": len(variances),
        "mean_between_dataset_variance": float(np.mean(variances)),
        "mean_within_bin_log10_spread": float(np.mean(spreads)),
        "cells": len(rows),
    }


def _e1_collapse_quality(
    summaries: list[dict[str, Any]], *, bins: int = 8, arm: str = "data_coupled"
) -> dict[str, Any]:
    usable = [
        row for row in summaries
        if row.get("arm", arm) == arm
        and all(
            row.get(key) is not None
            and np.isfinite(float(row[key]))
            and float(row[key]) > 0
            for key in ("margin", "N")
        )
    ]
    margin = _binned_between_dataset_variance(usable, "margin", bins=bins)
    physical = _binned_between_dataset_variance(usable, "N", bins=bins)
    ratio = None
    if margin["mean_between_dataset_variance"] is not None and physical["mean_between_dataset_variance"]:
        ratio = float(
            margin["mean_between_dataset_variance"] / physical["mean_between_dataset_variance"]
        )
    return {
        "definition": (
            "mean over quantile bins of the between-dataset variance of p_ctrl; "
            "ratio below 1 means N_eff/N* collapses the datasets better than N"
        ),
        "arm": arm,
        "cells": len(usable),
        "margin_axis": margin,
        "physical_N_axis": physical,
        "variance_ratio_margin_over_N": ratio,
        "collapses": bool(ratio is not None and ratio < 1.0),
        "comparison_is_matched": bool(
            margin.get("mean_within_bin_log10_spread") is not None
            and physical.get("mean_within_bin_log10_spread") is not None
            and physical["mean_within_bin_log10_spread"] > 0
            and 0.1
            <= margin["mean_within_bin_log10_spread"]
            / physical["mean_within_bin_log10_spread"]
            <= 10.0
        ),
    }


def run_e1(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E1_scale_boundary_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    seed = int(protocol["random_seed"]) + 1000
    common = protocol["common"]
    settings = protocol["e1"]
    train_count = int(settings.get("train_replications", common["train_replications"]))
    test_count = int(settings.get("test_replications", common["test_replications"]))
    total_count = train_count + test_count
    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    neff_boundaries: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    datasets = [str(value) for value in settings["datasets"]]

    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E1", protocol, datasets)
        raw_rows = merged["raw/seed_metrics.csv"]
        condition_rows = merged["raw/condition_responses.csv"]
        summaries = merged["summary.csv"]
        boundaries = merged["boundaries.csv"]
        neff_boundaries = merged["neff_boundaries.csv"]
        dataset_rows = merged["dataset_summary.csv"]
        _write_binary_outputs(output, merged.get("__binary__", {}))
        collapse = _e1_collapse_quality(summaries)
        _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
        _write_rows(raw_dir / "condition_responses.csv", condition_rows)
        _write_rows(output / "summary.csv", summaries)
        _write_rows(output / "boundaries.csv", boundaries)
        _write_rows(output / "neff_boundaries.csv", neff_boundaries)
        _write_rows(output / "dataset_summary.csv", dataset_rows)
        _write_json(output / "collapse_quality.json", collapse)
        _plot_e1(summaries, neff_boundaries, dataset_rows, figures, collapse)
        completed = sum(row.get("status") == "completed" for row in dataset_rows)
        result = {
            "status": "completed" if completed == len(datasets) else "partial",
            "completed_datasets": completed,
            "requested_datasets": len(datasets),
            "primary_estimands": ["p_ctrl", "N_star_eff", "margin=N_eff/N_star_eff", "A_ctrl"],
            "collapse_quality": collapse,
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    threshold = float(common["controllable_probability"])
    for dataset in datasets:
        config_path, network_protocol = _dataset_config(dataset)
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        _rebase_missing_workspace_paths(config)
        config["control"]["network_feedback"] = False
        config, network_protocol = apply_topology(config, dataset, network_protocol)
        pool = load_device_day_pool(config)
        configured = int(config["population"]["N_simulated_resources"])
        maximum = min(configured, _balanced_unique_capacity(pool))
        n_values = _legal_n_values(
            [int(value) for value in settings["n_candidates"]],
            maximum,
            include_maximum=bool(settings.get("include_maximum_unique_fleet", True)),
        )
        if len(n_values) < 2:
            dataset_rows.append({
                "dataset": dataset,
                "status": "insufficient_balanced_unique_sources",
                "maximum_unique_fleet": maximum,
            })
            continue
        master = _sample_master(config, max(n_values), seed)
        profile_steps, schedule = _profile_and_schedule(
            master, config, int(settings.get("conditions", common["conditions"])), seed + 100
        )
        steps_per_day = int(config["simulation"]["steps_per_day"])
        broadcast_columns = int(settings.get("broadcast_basis_columns", 9))
        trials_by_arm: dict[str, dict[int, list[float]]] = {"decoupled": {}, "data_coupled": {}}
        for n_index, n in enumerate(n_values):
            records = _balanced_subset(master, n)
            coupled_availability = availability_at_steps(
                rank_availability(diurnal_residual(records, steps_per_day)), profile_steps
            )
            arms = [
                ("decoupled", decoupled_copy(coupled_availability, seed + n_index * 7919)),
                ("data_coupled", coupled_availability),
            ]
            for arm_label, availability in arms:
                aggregate, traces, resource, _ = _run_snapshots(
                    records,
                    config,
                    profile_steps,
                    schedule,
                    replications=total_count,
                    seed=seed + n_index * 10000,
                    zone_correlation=0.0,
                    keep_resources=True,
                    availability_factory=lambda replication, table=availability: table,
                )
                _, frozen_prediction = _fit_frozen_prediction(
                    trace_features(traces[0]), aggregate[:train_count]
                )
                trials_by_arm[arm_label][n] = np.mean(aggregate[train_count:], axis=1).tolist()
                if frozen_prediction is None or resource is None:
                    raise RuntimeError("E1 calibration did not produce a frozen prediction")
                measured_rho = broadcast_residual_rho(
                    resource[train_count:], trace_features(traces[0]), broadcast_columns
                )
                per_seed: list[dict[str, Any]] = []
                for test_index in range(test_count):
                    trace = traces[train_count + test_index]
                    metrics = _success_metrics(
                        aggregate[train_count + test_index], frozen_prediction,
                        np.asarray(trace.network_scale, dtype=float), common,
                    )
                    row = {
                        "dataset": dataset,
                        "N": n,
                        "arm": arm_label,
                        "rho_measured": measured_rho,
                        "seed_index": test_index,
                        **metrics,
                    }
                    raw_rows.append(row)
                    per_seed.append(row)
                    for condition_index, (actual, target) in enumerate(
                        zip(aggregate[train_count + test_index], frozen_prediction)
                    ):
                        condition_rows.append({
                            "dataset": dataset,
                            "N": n,
                            "arm": arm_label,
                            "seed_index": test_index,
                            "condition_id": condition_index,
                            "direction": "charge" if target > 0 else "discharge",
                            "target_kw": float(target),
                            "accepted_kw": float(actual),
                        })
                successes = sum(bool(row["controllable"]) for row in per_seed)
                lower, upper = wilson_interval(successes, test_count)
                condition_means = np.mean(aggregate[train_count:], axis=0)
                condition_stds = np.std(aggregate[train_count:], axis=0)
                summaries.append({
                    "dataset": dataset,
                    "N": n,
                    "arm": arm_label,
                    "rho_measured": measured_rho,
                    "N_eff": effective_n(n, measured_rho),
                    "successes": successes,
                    "replications": test_count,
                    "p_controllable": successes / test_count,
                    "p_controllable_ci_lower": lower,
                    "p_controllable_ci_upper": upper,
                    "mean_nrmse": float(np.mean([row["nrmse"] for row in per_seed])),
                    "condition_cv": float(np.mean(
                        condition_stds / np.maximum(np.abs(condition_means), 1e-12)
                    )),
                })
                archive = raw_dir / "responses" / dataset / f"responses_N{n}_{arm_label}.npz"
                archive.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    archive,
                    aggregate_response_kw=aggregate,
                    frozen_prediction_kw=frozen_prediction,
                    profile_steps=np.asarray(profile_steps),
                )

        dataset_summary = [row for row in summaries if row["dataset"] == dataset]
        for arm_label in ("decoupled", "data_coupled"):
            rows = sorted(
                [row for row in dataset_summary if row["arm"] == arm_label],
                key=lambda item: item["N"],
            )
            reached = next(
                (row["N"] for row in rows if row["p_controllable_ci_lower"] >= threshold), None
            )
            boundaries.append({
                "dataset": dataset,
                "arm": arm_label,
                "rho_measured_max": max((row["rho_measured"] for row in rows), default=None),
                "N_star": reached,
                "status": "reached" if reached is not None else "not_reached",
            })
        neff_boundaries.append(
            _annotate_effective_margin(dataset, dataset_summary, threshold, arm="data_coupled")
        )
        slopes: dict[str, Any] = {}
        for arm_label in ("decoupled", "data_coupled"):
            rows = sorted(
                [row for row in dataset_summary if row["arm"] == arm_label],
                key=lambda item: item["N"],
            )
            slopes[f"beta_{arm_label}"] = loglog_slope(
                [row["N"] for row in rows], [row["condition_cv"] for row in rows]
            )
            lower, upper = bootstrap_slope_ci(
                trials_by_arm[arm_label], seed=seed + 333, samples=int(common["bootstrap_samples"])
            )
            slopes[f"beta_{arm_label}_ci_lower"] = lower
            slopes[f"beta_{arm_label}_ci_upper"] = upper
        coupled_rows = [row for row in dataset_summary if row["arm"] == "data_coupled"]
        dataset_rows.append({
            "dataset": dataset,
            "status": "completed",
            "configured_resources": configured,
            "maximum_unique_fleet": maximum,
            "N_values": ";".join(str(value) for value in n_values),
            "rho_data_coupled_max": max(
                (row["rho_measured"] for row in coupled_rows), default=None
            ),
            "A_ctrl": float(np.mean([
                row["p_controllable_ci_lower"] >= threshold for row in coupled_rows
            ])) if coupled_rows else None,
            **slopes,
            "network_protocol": network_protocol,
            "network_feedback": bool(config["control"]["network_feedback"]),
        })

    collapse = _e1_collapse_quality(summaries)
    _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
    _write_rows(raw_dir / "condition_responses.csv", condition_rows)
    _write_rows(output / "summary.csv", summaries)
    _write_rows(output / "boundaries.csv", boundaries)
    _write_rows(output / "neff_boundaries.csv", neff_boundaries)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    _write_json(output / "collapse_quality.json", collapse)
    _plot_e1(summaries, neff_boundaries, dataset_rows, figures, collapse)
    completed = sum(row.get("status") == "completed" for row in dataset_rows)
    result = {
        "status": "completed" if completed == len(datasets) else "partial",
        "completed_datasets": completed,
        "requested_datasets": len(datasets),
        "primary_estimands": ["p_ctrl", "N_star_eff", "margin=N_eff/N_star_eff", "A_ctrl"],
        "collapse_quality": collapse,
        "raw_seed_rows": len(raw_rows),
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _e2_signal_schedule(
    config: dict[str, Any], steps: int, mode: str, seed: int
) -> list[list[tuple[int, int]]]:
    zone_count = len(config["zones"]["zones"])
    rng = np.random.default_rng(seed)
    schedule: list[list[tuple[int, int]]] = []
    for step in range(steps):
        phase = step / max(steps - 1, 1)
        if mode == "step":
            score = 0.20 if phase < 0.25 or phase >= 0.75 else 0.85
            discharge = phase >= 0.50
        elif mode == "periodic":
            wave = np.sin(2.0 * np.pi * step / 12.0)
            score = 0.20 + 0.70 * abs(wave)
            discharge = wave < 0.0
        elif mode == "ramp":
            local = (step % 24) / 23.0
            score = 0.12 + 0.83 * local
            discharge = (step // 24) % 2 == 1
        elif mode == "rapid_changing":
            score = float([0.25, 0.90, 0.45, 0.75][step % 4])
            discharge = bool((step + int(rng.integers(0, 2))) % 2)
        else:
            raise ValueError(f"unknown E2 broadcast mode: {mode}")
        intensity = int(np.clip(round(score * 4095), 0, 4095))
        direction = 10 if discharge else 5
        schedule.append([(direction, intensity)] * zone_count)
    return schedule


def _run_e2_trials(
    records: list[DeviceDay],
    config: dict[str, Any],
    profile_steps: list[int],
    schedule: list[list[tuple[int, int]]],
    *,
    replications: int,
    response_seed: int,
    policy_seed: int,
    homogeneity: float,
    delay_distribution: str,
    availability: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Any], np.ndarray]:
    aggregates: list[np.ndarray] = []
    traces: list[Any] = []
    events: list[np.ndarray] = []
    for replication in range(replications):
        sink: list[list[float]] = []
        policy = ParameterizedLocalPolicy(
            records,
            homogeneity=homogeneity,
            delay_distribution=delay_distribution,
            seed=policy_seed,
            steps=len(profile_steps),
        )
        trace = simulate_day(
            records,
            config,
            steps=len(profile_steps),
            seed=response_seed + replication,
            profile_steps=profile_steps,
            signal_overrides=schedule,
            resource_response_sink=sink,
            response_policy=policy,
            availability_overrides=availability,
            reset_each_step=False,
        )
        aggregates.append(np.asarray(trace.accepted_control_kw, dtype=float))
        traces.append(trace)
        events.append(np.asarray(sink, dtype=float))
    return np.asarray(aggregates), traces, np.asarray(events)


def _e2_sync_metrics(
    events: np.ndarray,
    baseline_events: np.ndarray,
    *,
    z99: float,
) -> dict[str, Any]:
    event_mask = np.abs(events) > 1e-9
    baseline_mask = np.abs(baseline_events) > 1e-9
    n = events.shape[-1]
    simultaneous = np.mean(event_mask, axis=2)
    baseline_probability = np.mean(baseline_mask, axis=(0, 2))
    scale = float(np.mean(event_mask) / max(float(np.mean(baseline_mask)), 1e-12))
    probability = np.clip(baseline_probability * scale, 0.0, 1.0)
    envelope = np.clip(
        probability + z99 * np.sqrt(probability * (1.0 - probability) / max(n, 1)),
        0.0,
        1.0,
    )
    positive = np.sum(events > 1e-9, axis=2)
    negative = np.sum(events < -1e-9, axis=2)
    responding = positive + negative
    concentration = np.divide(
        np.maximum(positive, negative),
        responding,
        out=np.zeros_like(responding, dtype=float),
        where=responding > 0,
    )
    excess = np.maximum(simultaneous - envelope[None, :], 0.0)
    exceed = excess > 0.0
    return {
        "S_mean": float(np.mean(simultaneous)),
        "S_peak": float(np.max(simultaneous)),
        "D_mean": float(np.mean(concentration[responding > 0])) if np.any(responding) else 0.0,
        "X_sync": float(np.mean(excess)),
        "sync_exceedance_fraction": float(np.mean(exceed)),
        "surrogate_q99_mean": float(np.mean(envelope)),
        "per_seed_X_sync": np.mean(excess, axis=1),
    }


def run_e2(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E2_controller_synchronization_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    settings = protocol["e2"]
    common = protocol["common"]
    seed = int(protocol["random_seed"]) + 2000
    train_count = int(settings["train_replications"])
    test_count = int(settings["test_replications"])
    steps = int(settings["steps"])
    homogeneity_values = [float(value) for value in settings["homogeneity"]]
    datasets = [str(value) for value in settings["datasets"]]
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E2", protocol, datasets)
        raw_rows = merged["raw/seed_metrics.csv"]
        summary_rows = merged["synchronization_summary.csv"]
        threshold_rows = merged["synchronization_thresholds.csv"]
        dataset_rows = merged["dataset_summary.csv"]
        _write_binary_outputs(output, merged.get("__binary__", {}))
        _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
        _write_rows(output / "synchronization_summary.csv", summary_rows)
        _write_rows(output / "synchronization_thresholds.csv", threshold_rows)
        _write_rows(output / "dataset_summary.csv", dataset_rows)
        _plot_e2(summary_rows, threshold_rows, figures)
        completed = sum(row.get("status") == "completed" for row in dataset_rows)
        result = {
            "status": "completed" if completed == len(datasets) else "partial",
            "completed_datasets": completed,
            "requested_datasets": len(datasets),
            "coupling_arms": ["decoupled", "data_coupled"],
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    for dataset in datasets:
        config_path, network_protocol = _dataset_config(dataset)
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        config["control"]["network_feedback"] = False
        pool = load_device_day_pool(config)
        configured = int(config["population"]["N_simulated_resources"])
        maximum = min(configured, _balanced_unique_capacity(pool))
        n_values = sorted({
            min(int(settings["common_resources"]), maximum),
            maximum if bool(settings.get("include_maximum_unique_fleet", True)) else min(
                int(settings["common_resources"]), maximum
            ),
        })
        n_values = [value for value in n_values if value >= 2]
        if not n_values:
            dataset_rows.append({
                "dataset": dataset,
                "status": "insufficient_balanced_unique_sources",
                "maximum_unique_fleet": maximum,
            })
            continue
        master = _sample_master(config, max(n_values), seed)
        profile_steps = [step % int(config["simulation"]["steps_per_day"]) for step in range(steps)]
        event_archive: dict[str, Any] = {
            "dataset": np.asarray(dataset),
            "steps": np.asarray(steps),
        }
        response_archive: dict[str, Any] = {
            "dataset": np.asarray(dataset),
            "train_replications": np.asarray(train_count),
            "test_replications": np.asarray(test_count),
        }
        raster_events: np.ndarray | None = None
        steps_per_day = int(config["simulation"]["steps_per_day"])
        for n in n_values:
            records = _balanced_subset(master, n)
            coupled_availability = availability_at_steps(
                rank_availability(diurnal_residual(records, steps_per_day)), profile_steps
            )
            arms = [
                ("decoupled", decoupled_copy(coupled_availability, seed + n * 7919)),
                ("data_coupled", coupled_availability),
            ]
            for arm_label, availability in arms:
                for mode_index, mode in enumerate(settings["broadcast_modes"]):
                    schedule = _e2_signal_schedule(config, steps, str(mode), seed + mode_index * 100)
                    for delay_index, delay in enumerate(settings["delay_distributions"]):
                        response_seed = seed + n * 100000 + mode_index * 10000 + delay_index * 1000
                        policy_seed = seed + n * 100 + mode_index * 10 + delay_index
                        baseline_aggregate, baseline_traces, baseline_events = _run_e2_trials(
                            records,
                            config,
                            profile_steps,
                            schedule,
                            replications=train_count + test_count,
                            response_seed=response_seed,
                            policy_seed=policy_seed,
                            homogeneity=0.0,
                            delay_distribution=str(delay),
                            availability=availability,
                        )
                        frozen_target = np.mean(baseline_aggregate[:train_count], axis=0)
                        test_baseline_events = baseline_events[train_count:]
                        for c_index, homogeneity in enumerate(homogeneity_values):
                            if c_index == 0:
                                aggregate = baseline_aggregate[train_count:]
                                traces = baseline_traces[train_count:]
                                events = test_baseline_events
                            else:
                                aggregate, traces, events = _run_e2_trials(
                                    records,
                                    config,
                                    profile_steps,
                                    schedule,
                                    replications=test_count,
                                    response_seed=response_seed + train_count,
                                    policy_seed=policy_seed,
                                    homogeneity=homogeneity,
                                    delay_distribution=str(delay),
                                    availability=availability,
                                )
                            sync = _e2_sync_metrics(
                                events, test_baseline_events, z99=float(settings["surrogate_z99"])
                            )
                            per_seed: list[dict[str, Any]] = []
                            for replication in range(test_count):
                                metrics = _success_metrics(
                                    aggregate[replication], frozen_target,
                                    np.asarray(traces[replication].network_scale, dtype=float),
                                    common,
                                )
                                row = {
                                    "dataset": dataset,
                                    "coupling_arm": arm_label,
                                    "N": n,
                                    "broadcast_mode": mode,
                                    "delay_distribution": delay,
                                    "homogeneity": homogeneity,
                                    "seed_index": replication,
                                    "X_sync": float(sync["per_seed_X_sync"][replication]),
                                    **metrics,
                                }
                                raw_rows.append(row)
                                per_seed.append(row)
                            successes = sum(bool(row["controllable"]) for row in per_seed)
                            lower, upper = wilson_interval(successes, test_count)
                            actual_mean = np.mean(aggregate, axis=0)
                            measured_rho = conditional_mean_pairwise_rho(events)
                            summary_rows.append({
                                "dataset": dataset,
                                "coupling_arm": arm_label,
                                "N": n,
                                "broadcast_mode": mode,
                                "delay_distribution": delay,
                                "homogeneity": homogeneity,
                                "S_mean": sync["S_mean"],
                                "S_peak": sync["S_peak"],
                                "D_mean": sync["D_mean"],
                                "X_sync": sync["X_sync"],
                                "sync_exceedance_fraction": sync["sync_exceedance_fraction"],
                                "surrogate_q99_mean": sync["surrogate_q99_mean"],
                                "p_controllable": successes / test_count,
                                "p_controllable_ci_lower": lower,
                                "p_controllable_ci_upper": upper,
                                "mean_nrmse": float(np.mean([row["nrmse"] for row in per_seed])),
                                "R2": r2_score(
                                    aggregate.reshape(-1),
                                    np.tile(frozen_target, aggregate.shape[0]),
                                ),
                                "R2_seed_averaged": r2_score(actual_mean, frozen_target),
                                "rho_measured": measured_rho,
                                "N_eff": effective_n(n, measured_rho),
                                "network_feedback": False,
                            })
                            key = (
                                f"N{n}__{arm_label}__{mode}__{delay}__c"
                                f"{str(homogeneity).replace('.', 'p')}"
                            )
                            representative = events[0]
                            if (
                                n == max(n_values)
                                and arm_label == "data_coupled"
                                and mode == "periodic"
                                and delay == "fixed"
                                and homogeneity == 1.0
                            ):
                                raster_events = representative.copy()
                            event_archive[f"{key}__positive"] = np.packbits(
                                representative > 1e-9, axis=1
                            )
                            event_archive[f"{key}__negative"] = np.packbits(
                                representative < -1e-9, axis=1
                            )
                            event_archive[f"{key}__shape"] = np.asarray(representative.shape)
                            response_archive[f"{key}__aggregate"] = aggregate.astype(
                                np.float32
                            )
                            response_archive[f"{key}__frozen_target"] = frozen_target.astype(
                                np.float32
                            )

        archive_path = raw_dir / "events" / f"{dataset}_device_response_events.npz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive_path, **event_archive)
        response_path = raw_dir / "responses" / f"{dataset}_aggregate_responses.npz"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(response_path, **response_archive)
        if raster_events is not None:
            _plot_event_raster(
                raster_events,
                figures / f"event_raster_{dataset}.png",
                title=f"{_short_dataset(dataset)}: periodic, fixed delay, c=1",
            )
        dataset_summary = [row for row in summary_rows if row["dataset"] == dataset]
        for n in n_values:
            for arm_label in ("decoupled", "data_coupled"):
                for mode in settings["broadcast_modes"]:
                    for delay in settings["delay_distributions"]:
                        rows = sorted([
                            row for row in dataset_summary
                            if row["N"] == n
                            and row["coupling_arm"] == arm_label
                            and row["broadcast_mode"] == mode
                            and row["delay_distribution"] == delay
                        ], key=lambda item: item["homogeneity"])
                        sync_row = next((
                            row for row in rows
                            if row["X_sync"] > float(settings["sync_excess_limit"])
                            and row["D_mean"] >= float(settings["direction_concentration_limit"])
                        ), None)
                        fail_row = next((
                            row for row in rows
                            if row["p_controllable"] < float(common["controllable_probability"])
                            or row["mean_nrmse"] > float(common["nrmse_limit"])
                        ), None)
                        threshold_rows.append({
                            "dataset": dataset,
                            "coupling_arm": arm_label,
                            "N": n,
                            "broadcast_mode": mode,
                            "delay_distribution": delay,
                            "c_sync_star": sync_row["homogeneity"] if sync_row else None,
                            "c_sync_status": "reached" if sync_row else "not_reached",
                            "c_fail_star": fail_row["homogeneity"] if fail_row else None,
                            "c_fail_status": "reached" if fail_row else "not_reached",
                        })
        largest = max(n_values)
        arm_rho = {
            arm: [
                float(row["rho_measured"]) for row in dataset_summary
                if row["coupling_arm"] == arm and row["N"] == largest
            ]
            for arm in ("decoupled", "data_coupled")
        }
        dataset_rows.append({
            "dataset": dataset,
            "status": "completed",
            "configured_resources": configured,
            "maximum_unique_fleet": maximum,
            "N_values": ";".join(str(value) for value in n_values),
            "rho_data_coupled_mean": float(np.mean(arm_rho["data_coupled"]))
            if arm_rho["data_coupled"] else None,
            "rho_decoupled_mean": float(np.mean(arm_rho["decoupled"]))
            if arm_rho["decoupled"] else None,
            "network_protocol": network_protocol,
            "network_feedback": False,
            "controller_provenance": "parameterized LocalPolicy stress; not measured firmware",
            "coupling_source": "data_residual_rank_availability",
        })

    _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
    _write_rows(output / "synchronization_summary.csv", summary_rows)
    _write_rows(output / "synchronization_thresholds.csv", threshold_rows)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    _plot_e2(summary_rows, threshold_rows, figures)
    completed = sum(row.get("status") == "completed" for row in dataset_rows)
    result = {
        "status": "completed" if completed == len(datasets) else "partial",
        "completed_datasets": completed,
        "requested_datasets": len(datasets),
        "primary_estimands": [
            "X_sync", "c_sync_star", "c_fail_star", "p_ctrl", "NRMSE", "R2",
            "R2_seed_averaged", "N_eff"
        ],
        "coupling_arms": ["decoupled", "data_coupled"],
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _e3_periodic_schedule(
    config: dict[str, Any], steps: int, *, period_steps: int, duty: float
) -> list[list[tuple[int, int]]]:
    if period_steps < 3:
        raise ValueError("a cycle shorter than three steps is below the sampling Nyquist limit")
    zone_count = len(config["zones"]["zones"])
    half = max(1, period_steps // 2)
    schedule: list[list[tuple[int, int]]] = []
    for step in range(steps):
        position = step % period_steps
        charging = position < half
        within = position if charging else position - half
        span = half if charging else period_steps - half
        active = within < max(1, int(round(duty * span)))
        score = 0.88 if active else 0.12
        intensity = int(np.clip(round(score * 4095), 0, 4095))
        schedule.append([(5 if charging else 10, intensity)] * zone_count)
    return schedule


def _e3_cells(settings: dict[str, Any]) -> list[dict[str, Any]]:
    primary = settings["primary"]
    anchor_duty = float(primary["duty"])
    anchor_period = int(primary["period_steps"])
    anchor_delay = str(primary["delay_distribution"])
    anchor_n = int(primary["resources"])
    cells: list[dict[str, Any]] = []
    seen: set[tuple[int, str, float, int]] = set()

    def add(period: int, delay: str, duty: float, resources: int, role: str) -> None:
        key = (int(period), str(delay), float(duty), int(resources))
        if key in seen:
            return
        seen.add(key)
        cells.append({
            "period_steps": int(period), "delay_distribution": str(delay),
            "duty": float(duty), "resources": int(resources), "design_role": role,
        })

    for period in [int(value) for value in settings["period_steps"]]:
        for delay in [str(value) for value in settings["delay_distributions"]]:
            add(period, delay, anchor_duty, anchor_n, "period_delay_grid")
    for duty in [float(value) for value in settings["duty_cycles"]]:
        add(anchor_period, anchor_delay, duty, anchor_n, "duty_excursion")
    for resources in [int(value) for value in settings["resources"]]:
        add(anchor_period, anchor_delay, anchor_duty, resources, "scale_excursion")
    return cells


def run_e3(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E3_phase_coherence_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    settings = protocol["e3"]
    seed = int(protocol["random_seed"]) + 3000
    datasets = [str(value) for value in settings["datasets"]]
    replications = int(settings["replications"])
    transient = int(settings["transient_periods"])
    minimum_cycles = int(settings["minimum_cycles"])
    surrogates = int(settings["surrogates"])
    quantile = float(settings["surrogate_quantile"])
    plv_pairs = int(settings["plv_pairs"])
    homogeneity_values = sorted(float(value) for value in settings["homogeneity"])
    dt_minutes = float(protocol["common"].get("step_minutes", 5.0))

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = [{
        "period_minutes": float(value),
        "reason": "cycle shorter than two simulation steps is below the Nyquist limit",
    } for value in settings.get("excluded_period_minutes", [])]

    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E3", protocol, datasets)
        raw_rows = merged["raw/window_metrics.csv"]
        summary_rows = merged["phase_summary.csv"]
        dataset_rows = merged["dataset_summary.csv"]
        _write_binary_outputs(output, merged.get("__binary__", {}))
        _write_rows(raw_dir / "window_metrics.csv", raw_rows)
        _write_rows(output / "phase_summary.csv", summary_rows)
        _write_rows(output / "dataset_summary.csv", dataset_rows)
        _write_rows(output / "excluded_conditions.csv", excluded_rows)
        _plot_e3(summary_rows, raw_rows, figures)
        completed = sum(row.get("status") == "completed" for row in dataset_rows)
        result = {
            "status": "completed" if completed == len(datasets) else "partial",
            "completed_datasets": completed,
            "requested_datasets": len(datasets),
            "primary_estimands": ["R_K_mean", "H_phase", "L_phase_minutes", "plv_mean"],
            "coupling_arms": ["decoupled", "data_coupled"],
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    cells = _e3_cells(settings)
    for dataset in datasets:
        config_path, network_protocol = _dataset_config(dataset)
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        config["control"]["network_feedback"] = False
        pool = load_device_day_pool(config)
        configured = int(config["population"]["N_simulated_resources"])
        maximum = min(configured, _balanced_unique_capacity(pool))
        usable_cells = [cell for cell in cells if cell["resources"] <= maximum]
        if not usable_cells:
            dataset_rows.append({
                "dataset": dataset,
                "status": "insufficient_balanced_unique_sources",
                "maximum_unique_fleet": maximum,
            })
            continue
        master = _sample_master(config, max(cell["resources"] for cell in usable_cells), seed)
        day_steps = int(config["simulation"]["steps_per_day"])

        for cell_index, cell in enumerate(usable_cells):
            period_steps = int(cell["period_steps"])
            steps = max(day_steps, (minimum_cycles + transient) * period_steps)
            profile_steps = [index % day_steps for index in range(steps)]
            schedule = _e3_periodic_schedule(
                config, steps, period_steps=period_steps, duty=float(cell["duty"])
            )
            records = _balanced_subset(master, int(cell["resources"]))
            discard = transient * period_steps
            coupled_availability = availability_at_steps(
                rank_availability(diurnal_residual(records, day_steps)), profile_steps
            )
            arms = [
                ("decoupled", decoupled_copy(coupled_availability, seed + cell_index * 7919)),
                ("data_coupled", coupled_availability),
            ]
            for arm_label, availability in arms:
                reference_aggregate: dict[int, np.ndarray] = {}
                for homogeneity in homogeneity_values:
                    per_cell: list[dict[str, Any]] = []
                    policy_seed = seed + cell_index * 1000 + 500
                    windows: list[np.ndarray] = []
                    for replication in range(replications):
                        trial_seed = seed + cell_index * 1000 + replication
                        sink: list[list[float]] = []
                        policy = ParameterizedLocalPolicy(
                            records,
                            homogeneity=homogeneity,
                            delay_distribution=str(cell["delay_distribution"]),
                            seed=policy_seed,
                            steps=steps,
                        )
                        simulate_day(
                            records,
                            config,
                            steps=steps,
                            seed=trial_seed,
                            profile_steps=profile_steps,
                            signal_overrides=schedule,
                            resource_response_sink=sink,
                            response_policy=policy,
                            availability_overrides=availability,
                            reset_each_step=False,
                        )
                        window = np.asarray(sink, dtype=float)[discard:].T
                        if window.shape[1] >= 8:
                            windows.append(window)
                    if len(windows) < 2:
                        continue
                    conditioned = replication_permutation_band(
                        np.asarray(windows), surrogates=surrogates, quantile=quantile,
                        seed=seed + cell_index * 1000 + 313,
                    )
                    for replication, window in enumerate(windows):
                        trial_seed = seed + cell_index * 1000 + replication
                        analysis = analyse_window(
                            window,
                            dt_minutes=dt_minutes,
                            conditioned_band=conditioned,
                            surrogates=surrogates,
                            quantile=quantile,
                            seed=trial_seed + 991,
                            plv_pairs=plv_pairs,
                            surrogate_method=str(settings.get("surrogate_method", "both")),
                        )
                        aggregate = np.sum(window, axis=0)
                        if homogeneity == homogeneity_values[0]:
                            reference_aggregate[replication] = aggregate
                        reference = reference_aggregate.get(replication)
                        row = {
                            "dataset": dataset,
                            "coupling_arm": arm_label,
                            "homogeneity": homogeneity,
                            "period_steps": period_steps,
                            "period_minutes": period_steps * dt_minutes,
                            "duty": float(cell["duty"]),
                            "delay_distribution": str(cell["delay_distribution"]),
                            "N": int(cell["resources"]),
                            "design_role": str(cell["design_role"]),
                            "seed_index": replication,
                            "analysed_steps": int(window.shape[1]),
                            "cycles_analysed": float(window.shape[1] / period_steps),
                            "nrmse_vs_heterogeneous": (
                                float(nrmse(aggregate, reference)) if reference is not None else None
                            ),
                            **{key: value for key, value in analysis["summary"].items()},
                        }
                        raw_rows.append(row)
                        per_cell.append(row)
                        if (
                            replication == 0
                            and arm_label == "data_coupled"
                            and str(cell["design_role"]) == "period_delay_grid"
                            and period_steps == int(settings["primary"]["period_steps"])
                            and str(cell["delay_distribution"])
                            == str(settings["primary"]["delay_distribution"])
                        ):
                            archive = raw_dir / "timeseries" / dataset / (
                                f"phase_c{homogeneity:.2f}_p{period_steps}.npz"
                            )
                            archive.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                archive,
                                order_parameter=analysis["order"],
                                surrogate_band=analysis["band"],
                                phase_histogram=phase_histogram(
                                    analysis["phases"][analysis["active"]]
                                ),
                                aggregate_kw=aggregate,
                            )
                    if not per_cell:
                        continue
                    summary_rows.append({
                        "dataset": dataset,
                        "coupling_arm": arm_label,
                        "homogeneity": homogeneity,
                        "period_steps": period_steps,
                        "period_minutes": period_steps * dt_minutes,
                        "duty": float(cell["duty"]),
                        "delay_distribution": str(cell["delay_distribution"]),
                        "N": int(cell["resources"]),
                        "design_role": str(cell["design_role"]),
                        "replications": len(per_cell),
                        "H_phase_mean": float(np.mean([row["H_phase"] for row in per_cell])),
                        "H_phase_timeshuffle_mean": float(np.mean(
                            [row["H_phase_timeshuffle"] for row in per_cell]
                        )),
                        "H_phase_p95": float(
                            np.percentile([row["H_phase"] for row in per_cell], 95)
                        ),
                        "H_phase_ci_lower": float(
                            np.percentile([row["H_phase"] for row in per_cell], 2.5)
                        ),
                        "H_phase_ci_upper": float(
                            np.percentile([row["H_phase"] for row in per_cell], 97.5)
                        ),
                        "L_phase_minutes_mean": float(
                            np.mean([row["L_phase_minutes"] for row in per_cell])
                        ),
                        "L_phase_minutes_max": float(
                            np.max([row["L_phase_minutes"] for row in per_cell])
                        ),
                        "R_K_mean": float(np.mean([row["R_K_mean"] for row in per_cell])),
                        "R_K_peak": float(np.max([row["R_K_peak"] for row in per_cell])),
                        "plv_mean": float(np.mean([row["plv_mean"] for row in per_cell])),
                        "active_fraction": float(
                            np.mean([row["active_fraction"] for row in per_cell])
                        ),
                        "aggregate_amplitude_kw": float(
                            np.mean([row["aggregate_amplitude_kw"] for row in per_cell])
                        ),
                        "nrmse_vs_heterogeneous": float(np.mean([
                            row["nrmse_vs_heterogeneous"] for row in per_cell
                            if row["nrmse_vs_heterogeneous"] is not None
                        ])) if any(
                            row["nrmse_vs_heterogeneous"] is not None for row in per_cell
                        ) else None,
                        "nominal_exceedance_level": round(1.0 - quantile, 6),
                    })

        rows = [row for row in summary_rows if row["dataset"] == dataset]
        coupled = [row for row in rows if row["coupling_arm"] == "data_coupled"]
        control = [row for row in rows if row["coupling_arm"] == "decoupled"]
        baseline = [row for row in coupled if row["homogeneity"] == homogeneity_values[0]]
        stress = [row for row in coupled if row["homogeneity"] == homogeneity_values[-1]]
        control_stress = [row for row in control if row["homogeneity"] == homogeneity_values[-1]]
        dataset_rows.append({
            "dataset": dataset,
            "status": "completed",
            "cells": len(rows),
            "cells_per_arm": len(coupled),
            "maximum_unique_fleet": maximum,
            "network_protocol": network_protocol,
            "nominal_exceedance_level": round(1.0 - quantile, 6),
            "H_phase_heterogeneous": float(np.mean([row["H_phase_mean"] for row in baseline]))
            if baseline else None,
            "H_phase_homogeneous": float(np.mean([row["H_phase_mean"] for row in stress]))
            if stress else None,
            "H_phase_homogeneous_decoupled": float(
                np.mean([row["H_phase_mean"] for row in control_stress])
            ) if control_stress else None,
            "R_K_heterogeneous": float(np.mean([row["R_K_mean"] for row in baseline]))
            if baseline else None,
            "R_K_homogeneous": float(np.mean([row["R_K_mean"] for row in stress]))
            if stress else None,
            "R_K_homogeneous_decoupled": float(
                np.mean([row["R_K_mean"] for row in control_stress])
            ) if control_stress else None,
            "L_phase_minutes_homogeneous": float(
                np.max([row["L_phase_minutes_max"] for row in stress])
            ) if stress else None,
            "surrogates_per_window": surrogates,
            "coupling_source": "data_residual_rank_availability",
        })

    _write_rows(raw_dir / "window_metrics.csv", raw_rows)
    _write_rows(output / "phase_summary.csv", summary_rows)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    _write_rows(output / "excluded_conditions.csv", excluded_rows)
    _plot_e3(summary_rows, raw_rows, figures)
    completed = sum(row.get("status") == "completed" for row in dataset_rows)
    result = {
        "status": "completed" if completed == len(datasets) else "partial",
        "completed_datasets": completed,
        "requested_datasets": len(datasets),
        "primary_estimands": ["R_K_mean", "H_phase", "L_phase_minutes", "plv_mean"],
        "coupling_arms": ["decoupled", "data_coupled"],
        "raw_window_rows": len(raw_rows),
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _e4_condition_samples(
    values: np.ndarray, *, conditions: int, settle: int, measure: int
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    tail = array.reshape(conditions, settle, *array.shape[1:])[:, settle - measure :]
    return tail.mean(axis=1)


def _e4_run_stream(
    records: list[DeviceDay],
    config: dict[str, Any],
    profile_steps: list[int],
    schedule: list[list[tuple[int, int]]],
    *,
    replications: int,
    seed: int,
    drift_fractions: np.ndarray,
    delay_distribution: str,
    drift_order: np.ndarray,
    drift_delay_shift: int,
    conditions: int,
    settle: int,
    measure: int,
) -> tuple[np.ndarray, np.ndarray]:
    aggregates: list[np.ndarray] = []
    features: np.ndarray | None = None
    for replication in range(replications):
        policy = DriftedLocalPolicy(
            records,
            drift_fraction=float(drift_fractions[replication]),
            delay_distribution=delay_distribution,
            seed=seed + replication + 1,
            steps=len(profile_steps),
            drift_order=drift_order,
            drift_delay_shift=drift_delay_shift,
        )
        trace = simulate_day(
            records,
            config,
            steps=len(profile_steps),
            seed=seed + replication,
            profile_steps=profile_steps,
            signal_overrides=schedule,
            response_policy=policy,
            reset_each_step=False,
        )
        aggregates.append(_e4_condition_samples(
            np.asarray(trace.accepted_control_kw, dtype=float),
            conditions=conditions, settle=settle, measure=measure,
        ))
        if features is None:
            features = _e4_condition_samples(
                trace_features(trace), conditions=conditions, settle=settle, measure=measure
            )
    if features is None:
        raise RuntimeError("E4 stream produced no replications")
    return np.asarray(aggregates, dtype=float), features


def _e4_recalibration_arm(
    actual: np.ndarray,
    features: np.ndarray,
    coefficients: np.ndarray,
    *,
    window: int,
    detected_window: int | None,
    interval: int,
    refit_samples: int,
    ridge: float,
) -> dict[str, Any]:
    replications, conditions = actual.shape
    stream_actual = actual.reshape(-1)
    stream_features = np.tile(features, (replications, 1))
    total_windows = stream_actual.size // window
    current = np.asarray(coefficients, dtype=float).copy()
    predicted = np.empty_like(stream_actual)
    checkpoints: list[dict[str, Any]] = []
    bytes_sent = 0
    rounds = 0
    for index in range(total_windows):
        start, stop = index * window, (index + 1) * window
        predicted[start:stop] = stream_features[start:stop] @ current
        if detected_window is None or index < detected_window:
            continue
        if (index - detected_window) % interval:
            continue
        history_start = max(0, stop - refit_samples)
        sample_features = stream_features[history_start:stop]
        sample_actual = stream_actual[history_start:stop]
        if sample_features.shape[0] <= sample_features.shape[1]:
            continue
        current = aggregate_refit(sample_features, sample_actual, ridge=ridge)
        rounds += 1
        bytes_sent += update_bytes(sample_features.shape[0], current.size)
        checkpoints.append({
            "window_index": index,
            "samples_used": int(sample_features.shape[0]),
            "cumulative_bytes": int(bytes_sent),
            "coefficients": current.copy(),
        })
    predicted[total_windows * window :] = (
        stream_features[total_windows * window :] @ current
    )
    return {
        "predicted": predicted,
        "checkpoints": checkpoints,
        "update_bytes": int(bytes_sent),
        "recalibration_rounds": int(rounds),
        "final_coefficients": current,
    }


def run_e4(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E4_controller_drift_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    settings = protocol["e4"]
    common = protocol["common"]
    seed = int(protocol["random_seed"]) + 4000
    datasets = [str(value) for value in settings["datasets"]]
    conditions = int(settings["conditions"])
    settle = int(settings["settle_steps"])
    measure = int(settings["measure_steps"])
    window = int(settings["window_samples"])
    consecutive = int(settings["detection_consecutive_windows"])
    interval = int(settings["recalibration_interval_windows"])
    refit_samples = int(settings["recalibration_samples"])
    ridge = float(settings["ridge"])
    tolerance = float(settings["recovery_tolerance"])
    recovery_level = float(settings["recovery_level"])
    fractions = [float(value) for value in settings["unknown_fractions"]]
    modes = [str(value) for value in settings["drift_modes"]]
    resources = int(settings["resources"])
    ramp = int(settings["gradual_ramp_replications"])
    delay_distribution = str(settings["delay_distribution"])
    drift_delay_shift = int(settings["drift_delay_shift"])
    train_reps = int(settings["train_replications"])
    validation_reps = int(settings["validation_replications"])
    test_reps = int(settings["test_replications"])
    holdout_reps = int(settings["holdout_replications"])
    injection = int(settings["injection_replication"])
    seeds = int(settings["seeds"])

    stream_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E4", protocol, datasets)
        stream_rows = merged["raw/policy_drift_stream.csv"]
        event_rows = merged["drift_events.csv"]
        recovery_rows = merged["recovery_curves.csv"]
        dataset_rows = merged["dataset_summary.csv"]
        _write_binary_outputs(output, merged.get("__binary__", {}))
        _write_rows(raw_dir / "policy_drift_stream.csv", stream_rows)
        _write_rows(output / "drift_events.csv", event_rows)
        _write_rows(output / "recovery_curves.csv", recovery_rows)
        _write_rows(output / "dataset_summary.csv", dataset_rows)
        _plot_e4(stream_rows, event_rows, recovery_rows, figures)
        completed = sum(row.get("status") == "completed" for row in dataset_rows)
        result = {
            "status": "completed" if completed == len(datasets) else "partial",
            "completed_datasets": completed,
            "requested_datasets": len(datasets),
            "primary_estimands": ["T_detect", "T_recover", "M_90", "Regret_drift"],
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    for dataset in datasets:
        config_path, network_protocol = _dataset_config(dataset)
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        config["control"]["network_feedback"] = False
        pool = load_device_day_pool(config)
        maximum = min(int(config["population"]["N_simulated_resources"]), _balanced_unique_capacity(pool))
        fleet_size = min(resources, maximum)
        if fleet_size < 100:
            dataset_rows.append({
                "dataset": dataset,
                "status": "insufficient_balanced_unique_sources",
                "maximum_unique_fleet": maximum,
            })
            continue
        records = _balanced_subset(_sample_master(config, fleet_size, seed), fleet_size)
        base_profile, base_schedule = _profile_and_schedule(records, config, conditions, seed + 100)
        profile_steps = [step for step in base_profile for _ in range(settle)]
        schedule = [signal for signal in base_schedule for _ in range(settle)]
        stream_kwargs = {
            "conditions": conditions, "settle": settle, "measure": measure,
            "delay_distribution": delay_distribution, "drift_delay_shift": drift_delay_shift,
        }
        dataset_events: list[dict[str, Any]] = []

        for seed_index in range(seeds):
            base_seed = seed + 10_000 * (seed_index + 1)
            drift_order = np.random.default_rng(base_seed + 77).permutation(fleet_size)
            no_drift_train = np.zeros(train_reps)
            train_actual, features = _e4_run_stream(
                records, config, profile_steps, schedule,
                replications=train_reps, seed=base_seed,
                drift_fractions=no_drift_train, drift_order=drift_order, **stream_kwargs,
            )
            coefficients, _ = _fit_frozen_prediction(features, train_actual)

            validation_actual, _ = _e4_run_stream(
                records, config, profile_steps, schedule,
                replications=validation_reps, seed=base_seed + 2_000,
                drift_fractions=np.zeros(validation_reps), drift_order=drift_order, **stream_kwargs,
            )
            baseline_prediction = np.tile(features @ coefficients, validation_reps)
            validation_losses = window_losses(
                validation_actual.reshape(-1), baseline_prediction, window=window
            )
            if validation_losses.size < 4:
                continue
            threshold = float(np.quantile(validation_losses, 0.99))
            loss_base = float(np.median(validation_losses))
            validation_false_alarm = false_alarm_rate(
                validation_losses, threshold=threshold, stop=validation_losses.size,
                consecutive=consecutive,
            )

            for fraction in fractions:
                holdout_actual, _ = _e4_run_stream(
                    records, config, profile_steps, schedule,
                    replications=holdout_reps, seed=base_seed + 4_000 + int(fraction * 1000),
                    drift_fractions=np.full(holdout_reps, fraction),
                    drift_order=drift_order, **stream_kwargs,
                )
                oracle_actual, _ = _e4_run_stream(
                    records, config, profile_steps, schedule,
                    replications=train_reps,
                    seed=base_seed + 8_000 + int(fraction * 1000),
                    drift_fractions=np.full(train_reps, fraction),
                    drift_order=drift_order, **stream_kwargs,
                )
                oracle_coefficients = aggregate_refit(
                    np.tile(features, (train_reps, 1)), oracle_actual.reshape(-1), ridge=ridge
                )
                score_stream = holdout_actual.reshape(-1)
                score_features = np.tile(features, (holdout_reps, 1))
                loss_failed = float(nrmse(score_stream, score_features @ coefficients))
                loss_oracle = float(nrmse(score_stream, score_features @ oracle_coefficients))

                for mode in modes:
                    fractions_profile = drift_profile(
                        test_reps, injection_index=injection,
                        target_fraction=fraction, mode=mode, ramp=ramp,
                    )
                    test_actual, _ = _e4_run_stream(
                        records, config, profile_steps, schedule,
                        replications=test_reps,
                        seed=base_seed + 6_000 + int(fraction * 1000) + (0 if mode == "abrupt" else 500),
                        drift_fractions=fractions_profile, drift_order=drift_order, **stream_kwargs,
                    )
                    test_stream = test_actual.reshape(-1)
                    test_features = np.tile(features, (test_reps, 1))
                    frozen_prediction = test_features @ coefficients
                    frozen_losses = window_losses(test_stream, frozen_prediction, window=window)
                    injection_window = int(injection * conditions / window)
                    detected = detection_index(
                        frozen_losses, threshold=threshold,
                        start=injection_window, consecutive=consecutive,
                    )
                    pre_false_alarm = false_alarm_rate(
                        frozen_losses, threshold=threshold,
                        stop=injection_window, consecutive=consecutive,
                    )
                    arm = _e4_recalibration_arm(
                        test_actual, features, coefficients,
                        window=window, detected_window=detected, interval=interval,
                        refit_samples=refit_samples, ridge=ridge,
                    )
                    recalibrated_losses = window_losses(test_stream, arm["predicted"], window=window)
                    oracle_prediction = frozen_prediction.copy()
                    oracle_replication = injection + (ramp if mode == "gradual" else 0)
                    oracle_start = min(oracle_replication * conditions, test_stream.size)
                    oracle_prediction[oracle_start:] = (
                        test_features[oracle_start:] @ oracle_coefficients
                    )
                    oracle_losses = window_losses(test_stream, oracle_prediction, window=window)

                    recovered = recovery_index(
                        recalibrated_losses, baseline=loss_base, tolerance=tolerance,
                        start=detected if detected is not None else injection_window,
                    )
                    shares: list[float] = []
                    for checkpoint in arm["checkpoints"]:
                        loss_now = float(nrmse(
                            score_stream, score_features @ checkpoint["coefficients"]
                        ))
                        share = float(recovered_share(
                            np.asarray([loss_now]), failed=loss_failed, baseline=loss_oracle
                        )[0])
                        shares.append(share)
                        recovery_rows.append({
                            "dataset": dataset, "seed_index": seed_index, "mode": mode,
                            "unknown_fraction": fraction,
                            "window_index": checkpoint["window_index"],
                            "samples_used": checkpoint["samples_used"],
                            "cumulative_bytes": checkpoint["cumulative_bytes"],
                            "holdout_nrmse": loss_now,
                            "recovered_share": share,
                        })
                    m90 = sample_complexity(np.asarray(shares), level=recovery_level)
                    event = {
                        "dataset": dataset, "seed_index": seed_index, "mode": mode,
                        "unknown_fraction": fraction,
                        "detection_threshold": threshold,
                        "validation_false_alarm_rate": validation_false_alarm,
                        "pre_injection_false_alarm_rate": pre_false_alarm,
                        "loss_base": loss_base,
                        "loss_failed_holdout": loss_failed,
                        "loss_oracle_holdout": loss_oracle,
                        "detected": detected is not None,
                        "T_detect_windows": (detected - injection_window) if detected is not None else None,
                        "T_detect_samples": (
                            (detected - injection_window) * window if detected is not None else None
                        ),
                        "detection_censored": detected is None,
                        "T_recover_windows": (
                            (recovered - detected) if (recovered is not None and detected is not None) else None
                        ),
                        "recovery_censored": recovered is None,
                        "M_90_checkpoints": m90,
                        "M_90_samples": (
                            arm["checkpoints"][m90]["samples_used"] if m90 is not None else None
                        ),
                        "recalibration_rounds": arm["recalibration_rounds"],
                        "update_bytes": arm["update_bytes"],
                        "regret_frozen": drift_regret(
                            frozen_losses, baseline=loss_base, start=injection_window
                        ),
                        "regret_recalibrated": drift_regret(
                            recalibrated_losses, baseline=loss_base, start=injection_window
                        ),
                        "regret_oracle": drift_regret(
                            oracle_losses, baseline=loss_base, start=injection_window
                        ),
                        "final_recovered_share": shares[-1] if shares else None,
                    }
                    event_rows.append(event)
                    dataset_events.append(event)
                    for index in range(len(frozen_losses)):
                        stream_rows.append({
                            "dataset": dataset, "seed_index": seed_index, "mode": mode,
                            "unknown_fraction": fraction, "window_index": index,
                            "post_injection": index >= injection_window,
                            "loss_frozen": float(frozen_losses[index]),
                            "loss_recalibrated": float(recalibrated_losses[index]),
                            "loss_oracle": float(oracle_losses[index]),
                            "threshold": threshold,
                            "injection_window": injection_window,
                        })

        detected_events = [row for row in dataset_events if row["detected"]]
        dataset_rows.append({
            "dataset": dataset,
            "status": "completed" if dataset_events else "no_events",
            "fleet_size": fleet_size,
            "maximum_unique_fleet": maximum,
            "network_protocol": network_protocol,
            "runs": len(dataset_events),
            "detected_runs": len(detected_events),
            "detection_rate": len(detected_events) / max(len(dataset_events), 1),
            "median_T_detect_windows": float(np.median(
                [row["T_detect_windows"] for row in detected_events]
            )) if detected_events else None,
            "max_validation_false_alarm_rate": float(np.max(
                [row["validation_false_alarm_rate"] for row in dataset_events]
            )) if dataset_events else None,
            "median_regret_frozen": float(np.median(
                [row["regret_frozen"] for row in dataset_events]
            )) if dataset_events else None,
            "median_regret_recalibrated": float(np.median(
                [row["regret_recalibrated"] for row in dataset_events]
            )) if dataset_events else None,
            "median_update_bytes": float(np.median(
                [row["update_bytes"] for row in dataset_events]
            )) if dataset_events else None,
        })

    _write_rows(raw_dir / "policy_drift_stream.csv", stream_rows)
    _write_rows(output / "drift_events.csv", event_rows)
    _write_rows(output / "recovery_curves.csv", recovery_rows)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    _plot_e4(stream_rows, event_rows, recovery_rows, figures)
    completed = sum(row.get("status") == "completed" for row in dataset_rows)
    result = {
        "status": "completed" if completed == len(datasets) else "partial",
        "completed_datasets": completed,
        "requested_datasets": len(datasets),
        "primary_estimands": ["T_detect", "T_recover", "M_90", "Regret_drift"],
        "censoring_policy": "undetected runs reported as censored, never dropped from the mean",
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _availability_factory(
    structure: str,
    participation: float,
    records: list[DeviceDay],
    steps: int,
    seed: int,
    persistence: float,
    shared_fraction: float,
    *,
    profile_steps: list[int] | None = None,
    steps_per_day: int | None = None,
    options: dict[str, Any] | None = None,
):
    options = dict(options or {})
    zone_ids = np.asarray([record.zone_id for record in records])
    zones = sorted(set(zone_ids.tolist()))

    behaviour_table: np.ndarray | None = None
    profile_index: np.ndarray | None = None
    if structure in {"data_behavior", "data_shift"}:
        if profile_steps is None or steps_per_day is None:
            raise ValueError("data-driven availability needs profile_steps and steps_per_day")
        behaviour_table = rank_availability(diurnal_residual(records, int(steps_per_day)))
        profile_index = np.asarray(profile_steps, dtype=int)

    def factory(replication: int) -> np.ndarray:
        rng = np.random.default_rng(seed + replication)
        if participation >= 1.0:
            return np.ones((steps, len(records)), dtype=float)
        if behaviour_table is not None and profile_index is not None:
            period, device_count = behaviour_table.shape
            columns = np.arange(device_count)
            if structure == "data_behavior":
                offsets = np.full(device_count, int(rng.integers(0, period)))
            else:
                offsets = rng.integers(0, period, size=device_count)
            rows = (profile_index[:, None] - offsets[None, :]) % period
            sampled = behaviour_table[rows, columns[None, :]]
            return (sampled > 1.0 - participation).astype(float)
        if structure == "iid":
            return (rng.random((steps, len(records))) < participation).astype(float)
        if structure == "markov":
            values = np.zeros((steps, len(records)), dtype=float)
            values[0] = rng.random(len(records)) < participation
            p11 = persistence + (1.0 - persistence) * participation
            p01 = (1.0 - persistence) * participation
            for step in range(1, steps):
                probability = np.where(values[step - 1] > 0, p11, p01)
                values[step] = rng.random(len(records)) < probability
            return values
        if structure == "community":
            independent = rng.random((steps, len(records))) < participation
            use_shared = rng.random((steps, len(records))) < shared_fraction
            result = independent.copy()
            for zone in zones:
                zone_mask = zone_ids == zone
                shared = (rng.random(steps) < participation)[:, None]
                result[:, zone_mask] = np.where(use_shared[:, zone_mask], shared, independent[:, zone_mask])
            return result.astype(float)

        if structure in {"diurnal_class", "diurnal_antiphase", "diurnal_shift"}:
            if steps_per_day is None or profile_steps is None:
                raise ValueError("diurnal availability needs profile_steps and steps_per_day")
            period = int(steps_per_day)
            hours = np.arange(period) * 24.0 / period
            classes = options.get("diurnal_classes") or [
                {"name": "ev", "share": 0.4, "peak_hour": 12.0, "depth": 0.9},
                {"name": "home_battery", "share": 0.4, "peak_hour": 10.0, "depth": 0.5},
                {"name": "flat", "share": 0.2, "peak_hour": 0.0, "depth": 0.0},
            ]
            if structure == "diurnal_antiphase":
                classes = [dict(c) for c in classes]
                classes[0]["peak_hour"] = (float(classes[0]["peak_hour"]) + 12.0) % 24.0
            device_count = len(records)
            shares = np.asarray([float(c["share"]) for c in classes], dtype=float)
            shares = shares / shares.sum()
            edges = np.cumsum(shares)
            class_of = np.asarray([
                int(np.searchsorted(edges, ((index % 100) + 0.5) / 100.0))
                for index in range(device_count)
            ])
            class_of = np.clip(class_of, 0, len(classes) - 1)
            profile = np.zeros((period, device_count), dtype=float)
            for index, cls in enumerate(classes):
                mask = class_of == index
                if not mask.any():
                    continue
                shape = 1.0 + float(cls["depth"]) * np.cos(
                    2.0 * np.pi * (hours - float(cls["peak_hour"])) / 24.0
                )
                profile[:, mask] = shape[:, None]
            profile_index_local = np.asarray(profile_steps, dtype=int) % period

            def diurnal(replication: int, profile=profile, period=period,
                        profile_index_local=profile_index_local,
                        device_count=device_count, structure=structure) -> np.ndarray:
                rng_local = np.random.default_rng(seed + 7919 * replication)
                weight = float(options.get("diurnal_noise_weight", 0.5))
                noisy = (1.0 - weight) * profile + weight * rng_local.random(
                    (period, device_count)
                ) * float(np.mean(profile))
                if structure == "diurnal_shift":
                    offsets = rng_local.integers(0, period, size=device_count)
                else:
                    offsets = np.zeros(device_count, dtype=int)
                rows = (profile_index_local[:, None] - offsets[None, :]) % period
                sampled = noisy[rows, np.arange(device_count)[None, :]]
                steps_used = rows.shape[0]
                order = np.argsort(np.argsort(sampled, axis=0), axis=0)
                uniform = (order + 0.5) / steps_used
                return (uniform > 1.0 - participation).astype(float)

            return diurnal(replication)

        if structure.startswith("group_shared"):
            _, _, tail = structure.partition(":")
            device_count = len(records)
            if tail:
                group_count = max(1, min(int(tail), device_count))
                group_of = np.arange(device_count) % group_count
            else:
                group_count = len(zones)
                lookup = {zone: index for index, zone in enumerate(zones)}
                group_of = np.asarray([lookup[zone] for zone in zone_ids.tolist()])
            draws = rng.random((steps, group_count)) < participation
            return draws[:, group_of].astype(float)

        if structure == "weather_block":
            if steps_per_day is None or profile_steps is None:
                raise ValueError("weather_block needs profile_steps and steps_per_day")
            period = int(steps_per_day)
            outage = int(round((1.0 - participation) * period))
            start = int(rng.integers(0, period))
            offline = np.zeros(period, dtype=bool)
            index = (np.arange(outage) + start) % period
            offline[index] = True
            local_steps = np.asarray(profile_steps, dtype=int) % period
            online = (~offline[local_steps]).astype(float)
            return np.repeat(online[:, None], len(records), axis=1)

        raise ValueError(f"unknown availability structure: {structure}")

    return factory


def run_e6(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    return _run_availability_experiment(
        protocol, root, "E6", "e6", "E6_behaviour_availability_new", 6000)


def run_e15(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    return _run_availability_experiment(
        protocol, root, "E15", "e15", "E15_availability_cases_new", 15000)


def _run_availability_experiment(
    protocol: dict[str, Any],
    root: Path,
    experiment: str,
    section: str,
    output_name: str,
    seed_offset: int,
) -> dict[str, Any]:
    output = root / output_name
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    settings = protocol[section]
    common = protocol["common"]
    train_count = int(common["train_replications"])
    test_count = int(common["test_replications"])
    steps = int(common["conditions"])
    seed = int(protocol["random_seed"]) + seed_offset
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    datasets = [str(dataset) for dataset in settings["datasets"]]
    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows(experiment, protocol, datasets)
        raw_rows = merged["raw/seed_metrics.csv"]
        summary_rows = merged["summary.csv"]
        dataset_rows = merged["dataset_summary.csv"]
        _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
        _write_rows(output / "summary.csv", summary_rows)
        _write_rows(output / "dataset_summary.csv", dataset_rows)
        try:
            _plot_e6(summary_rows, figures / "figure_E6_behaviour_availability.png")
        except Exception as error:
            print(f"  [{experiment}] built-in plot skipped: {error}")
        completed = sum(row["status"] == "completed" for row in dataset_rows)
        result = {
            "status": "completed" if completed == len(datasets) else "partial",
            "completed_datasets": completed,
            "requested_datasets": len(datasets),
            "primary_estimands": [
                "N_eff_behavior", "behavior_penalty_nrmse", "reserve_kw_q95"
            ],
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    for dataset_index, dataset in enumerate(datasets):
        config_path, network_protocol = _dataset_config(str(dataset))
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        config["control"]["network_feedback"] = False
        pool = load_device_day_pool(config)
        requested = int(settings["resources"])
        configured = int(config["population"]["N_simulated_resources"])
        resource_count = min(requested, configured, _balanced_unique_capacity(pool))
        if resource_count < 2:
            dataset_rows.append({
                "dataset": dataset,
                "status": "insufficient_balanced_unique_sources",
                "resource_count": resource_count,
            })
            continue

        dataset_seed = seed
        broadcast_columns = int(settings.get("broadcast_basis_columns", 9))
        records = _sample_master(config, resource_count, dataset_seed)
        profile_steps, schedule = _profile_and_schedule(records, config, steps, dataset_seed)
        baseline, baseline_traces, _, _ = _run_snapshots(
            records,
            config,
            profile_steps,
            schedule,
            replications=train_count,
            seed=dataset_seed + 100,
        )
        _, target = _fit_frozen_prediction(trace_features(baseline_traces[0]), baseline)

        test_seed = dataset_seed + 100_000
        availability_seed = dataset_seed + 200_000
        for participation in settings["participation"]:
            for structure in settings["structures"]:
                factory = _availability_factory(
                    str(structure),
                    float(participation),
                    records,
                    steps,
                    availability_seed,
                    float(settings["markov_persistence"]),
                    float(settings["community_shared_fraction"]),
                    profile_steps=profile_steps,
                    steps_per_day=int(config["simulation"]["steps_per_day"]),
                    options=settings,
                )
                aggregate, traces, resource, active_counts = _run_snapshots(
                    records,
                    config,
                    profile_steps,
                    schedule,
                    replications=test_count,
                    seed=test_seed,
                    availability_factory=factory,
                    keep_resources=True,
                )
                if resource is None:
                    raise RuntimeError("E6 resource response recording failed")
                rho = broadcast_residual_rho(
                    resource, trace_features(traces[0]), broadcast_columns
                )
                rho_replication = conditional_mean_pairwise_rho(resource)
                scale = max(float(np.percentile(np.abs(target), 95)), 1e-12)
                mean_response = np.mean(aggregate, axis=0)
                bias_nrmse = float(
                    np.sqrt(np.mean(np.square(mean_response - target))) / scale
                )
                variance_nrmse = float(
                    np.sqrt(np.mean(np.square(aggregate - mean_response))) / scale
                )
                mean_active = float(np.mean(active_counts))
                n_effective = effective_n(mean_active, rho)
                n_eff_replication = effective_n(mean_active, rho_replication)
                reserve_values: list[float] = []
                group_rows: list[dict[str, Any]] = []
                for replication in range(test_count):
                    trace = traces[replication]
                    metrics = _success_metrics(
                        aggregate[replication], target, np.asarray(trace.network_scale), common
                    )
                    metrics["r2"] = float(r2_score(target, aggregate[replication]))
                    same_direction = np.sign(aggregate[replication]) == np.sign(target)
                    delivered = np.where(same_direction, np.abs(aggregate[replication]), 0.0)
                    reserve = float(
                        np.percentile(np.maximum(np.abs(target) - delivered, 0.0), 95)
                    )
                    reserve_values.append(reserve)
                    row = {
                        "dataset": dataset,
                        "N": resource_count,
                        "participation": float(participation),
                        "structure": structure,
                        "seed_index": replication,
                        "active_count_mean": float(np.mean(active_counts[replication])),
                        "rho_behavior": rho,
                        "N_eff_behavior": n_effective,
                        "rho_replication": rho_replication,
                        "N_eff_replication": n_eff_replication,
                        "reserve_kw_q95": reserve,
                        **metrics,
                    }
                    raw_rows.append(row)
                    group_rows.append(row)
                summary_rows.append({
                    "dataset": dataset,
                    "N": resource_count,
                    "participation": float(participation),
                    "structure": structure,
                    "active_count_mean": mean_active,
                    "rho_behavior": rho,
                    "N_eff_behavior": n_effective,
                    "rho_replication": rho_replication,
                    "N_eff_replication": n_eff_replication,
                    "N_requested": resource_count,
                    "bias_nrmse": bias_nrmse,
                    "variance_nrmse": variance_nrmse,
                    "mean_nrmse": float(np.mean([row["nrmse"] for row in group_rows])),
                    "mean_r2": float(np.mean([row["r2"] for row in group_rows])),
                    "mean_sign_consistency": float(
                        np.mean([row["sign_consistency"] for row in group_rows])
                    ),
                    "mean_blocked_fraction": float(
                        np.mean([row["blocked_fraction"] for row in group_rows])
                    ),
                    "reserve_kw_q95": float(np.percentile(reserve_values, 95)),
                    "failure_probability": float(
                        np.mean([not row["controllable"] for row in group_rows])
                    ),
                })
        dataset_rows.append({
            "dataset": dataset,
            "status": "completed",
            "resource_count": resource_count,
            "configured_resources": configured,
            "balanced_unique_capacity": _balanced_unique_capacity(pool),
            "network_protocol": network_protocol,
        })
    for row in summary_rows:
        iid = next(
            candidate for candidate in summary_rows
            if candidate["dataset"] == row["dataset"]
            and candidate["participation"] == row["participation"]
            and candidate["structure"] == "iid"
        )
        row["behavior_penalty_nrmse"] = float(row["mean_nrmse"] - iid["mean_nrmse"])
    _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
    _write_rows(output / "summary.csv", summary_rows)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    try:
        _plot_e6(summary_rows, figures / "figure_E6_behaviour_availability.png")
    except Exception as error:
        print(f"  [{experiment}] built-in plot skipped: {error}")
    completed = sum(row["status"] == "completed" for row in dataset_rows)
    result = {
        "status": "completed" if completed == len(settings["datasets"]) else "partial",
        "completed_datasets": completed,
        "requested_datasets": len(settings["datasets"]),
        "primary_estimands": [
            "N_eff_behavior", "behavior_penalty_nrmse", "reserve_kw_q95"
        ],
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _dataset_config(dataset: str) -> tuple[Path, str]:
    config_root = ROOT / "results" / f"{dataset}{DATASET_RESULTS_SUFFIX}" / "config"
    corrected = (
        ROOT
        / "results"
        / f"{dataset}{DATASET_RESULTS_SUFFIX}"
        / "coverage_fix/network_constrained_new/config/default.yaml"
    )
    if corrected.exists():
        return corrected, "coverage_fix_config_network_bypass_for_scaling"
    weak = config_root / "default_weak_correlation.yaml"
    if weak.exists():
        return weak, "weak_capacity_config_network_bypass_for_scaling"
    return config_root / "default.yaml", "base_config_network_bypass_for_scaling"


def _rebase_missing_workspace_paths(config: dict[str, Any]) -> None:
    groups = [
        config.get("paths", {}),
        config.get("population", {}).get("canonical_adapter", {}),
    ]
    for group in groups:
        for key, value in list(group.items()):
            if key not in {"data_root", "results_root", "input", "input_dir", "solar_input", "cache"}:
                continue
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute() or path.exists():
                continue
            anchors = [
                index for index, part in enumerate(path.parts)
                if part in {"data", "results"}
            ]
            if anchors:
                group[key] = str(ROOT.joinpath(*path.parts[anchors[-1] :]))


def _legal_n_values(
    candidates: list[int], maximum: int, *, include_maximum: bool
) -> list[int]:
    if maximum < 1:
        return []
    limit = maximum if include_maximum else min(maximum, max(int(n) for n in candidates))
    values = {int(n) for n in candidates if 1 <= int(n) <= limit}
    if include_maximum:
        values.add(maximum)
    minimum_points = min(5, limit) if limit < 50 else 3
    if len(values) < minimum_points:
        values.update(
            int(round(value))
            for value in np.linspace(1, limit, minimum_points)
        )
    return sorted(value for value in values if 1 <= value <= limit)


def run_e7(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E7_empirical_scaling_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    settings = protocol["e7"]
    train_count = int(protocol["common"]["train_replications"])
    replications = int(settings["replications"])
    test_count = replications - train_count
    raw_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    seed = int(protocol["random_seed"]) + 7000

    datasets = [str(dataset) for dataset in settings["datasets"]]
    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E7", protocol, datasets)
        raw_rows = merged["raw/seed_metrics.csv"]
        scaling_rows = merged["scaling_summary.csv"]
        dataset_rows = merged["dataset_summary.csv"]
        _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
        _write_rows(output / "scaling_summary.csv", scaling_rows)
        _write_rows(output / "dataset_summary.csv", dataset_rows)
        _plot_e7(scaling_rows, dataset_rows, figures / "figure_E7_empirical_scaling.png")
        completed = sum(row.get("status") == "completed" for row in dataset_rows)
        requested = len(datasets) * len(settings["directions"])
        result = {
            "status": "completed" if completed == requested else ("partial" if completed else "failed"),
            "completed_dataset_direction_pairs": completed,
            "requested_dataset_direction_pairs": requested,
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    for dataset_index, dataset in enumerate(datasets):
        config_path, network_protocol = _dataset_config(str(dataset))
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        config["control"]["network_feedback"] = False
        pool = load_device_day_pool(config)
        configured_maximum = int(config["population"]["N_simulated_resources"])
        maximum = min(configured_maximum, _balanced_unique_capacity(pool))
        n_values = _legal_n_values(
            [int(n) for n in settings["n_candidates"]],
            maximum,
            include_maximum=bool(settings.get("include_maximum_unique_fleet", True)),
        )
        if len(n_values) < 2:
            dataset_rows.append({
                "dataset": dataset,
                "status": "insufficient_balanced_unique_sources",
                "configured_maximum": configured_maximum,
                "maximum_unique_fleet": maximum,
                "network_protocol": network_protocol,
            })
            continue
        try:
            master = _sample_master(config, max(n_values), seed)
        except Exception as exc:
            dataset_rows.append({"dataset": dataset, "status": "sampling_failed", "reason": repr(exc)})
            continue
        direction_slopes: list[float] = []
        direction_n90: list[int | None] = []
        direction_n95: list[int | None] = []
        for direction_index, direction in enumerate(settings["directions"]):
            trials_by_n: dict[int, list[float]] = {}
            p_rows: list[dict[str, Any]] = []
            for n_index, n in enumerate(n_values):
                records = _balanced_subset(master, n)
                profile_steps, schedule = _profile_and_schedule(
                    records,
                    config,
                    int(settings["conditions"]),
                    seed + direction_index * 10000,
                    direction=str(direction),
                )
                aggregate, traces, _, _ = _run_snapshots(
                    records,
                    config,
                    profile_steps,
                    schedule,
                    replications=replications,
                    seed=seed + 1000000 + direction_index * 10000,
                )
                target = np.mean(aggregate[:train_count], axis=0)
                trial_means = np.mean(aggregate[train_count:], axis=1)
                trials_by_n[n] = trial_means.tolist()
                successes = 0
                for replication in range(test_count):
                    trace = traces[train_count + replication]
                    metrics = _success_metrics(
                        aggregate[train_count + replication],
                        target,
                        np.asarray(trace.network_scale),
                        protocol["common"],
                    )
                    successes += int(metrics["controllable"])
                    raw_rows.append({
                        "dataset": dataset,
                        "direction": direction,
                        "N": n,
                        "seed_index": replication,
                        "mean_response_kw": float(trial_means[replication]),
                        **metrics,
                    })
                lower, upper = wilson_interval(successes, test_count)
                mean = float(np.mean(trial_means))
                std = float(np.std(trial_means))
                row = {
                    "dataset": dataset,
                    "direction": direction,
                    "N": n,
                    "mean_response_kw": mean,
                    "std_response_kw": std,
                    "cv": std / max(abs(mean), 1e-12),
                    "p_controllable": successes / test_count,
                    "p_controllable_ci_lower": lower,
                    "p_controllable_ci_upper": upper,
                    "source_unique": True,
                }
                scaling_rows.append(row)
                p_rows.append(row)
            slope = loglog_slope(n_values, [
                next(row["cv"] for row in p_rows if row["N"] == n) for n in n_values
            ])
            ci_lower, ci_upper = bootstrap_slope_ci(
                trials_by_n,
                seed=seed + direction_index,
                samples=int(protocol["common"]["bootstrap_samples"]),
            )
            direction_slopes.append(slope)
            n90 = next((row["N"] for row in p_rows if row["p_controllable_ci_lower"] >= 0.90), None)
            n95 = next((row["N"] for row in p_rows if row["p_controllable_ci_lower"] >= 0.95), None)
            direction_n90.append(n90)
            direction_n95.append(n95)
            dataset_rows.append({
                "dataset": dataset,
                "direction": direction,
                "status": "completed",
                "beta": slope,
                "beta_ci_lower": ci_lower,
                "beta_ci_upper": ci_upper,
                "N_90": n90,
                "N_95": n95,
                "maximum_unique_fleet": maximum,
            })
    _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
    _write_rows(output / "scaling_summary.csv", scaling_rows)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    _plot_e7(scaling_rows, dataset_rows, figures / "figure_E7_empirical_scaling.png")
    completed = sum(row.get("status") == "completed" for row in dataset_rows)
    requested = len(settings["datasets"]) * len(settings["directions"])
    result = {
        "status": "completed" if completed == requested else ("partial" if completed else "failed"),
        "completed_dataset_direction_pairs": completed,
        "requested_dataset_direction_pairs": requested,
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _global_schedule(records: list[DeviceDay], config: dict[str, Any], profile_steps: list[int], seed: int):
    rng = np.random.default_rng(seed)
    zones = config["zones"]["zones"]
    schedule: list[list[tuple[int, int]]] = []
    zone_count = len(zones)
    for profile_step in profile_steps:
        load = 0.0
        energy_input = 0.0
        for record in records:
            zone = zones[record.zone_id]
            load += float(record.load_kw[profile_step]) * float(zone["load_multiplier"])
            energy_input += float(record.energy_input_kw[profile_step]) * float(
                zone.get("energy_input_multiplier", zone.get("pv_multiplier", 1.0))
            )
        direction, intensity = _signal_values(energy_input - load, load, "normal", rng)
        schedule.append([(direction, intensity)] * zone_count)
    return schedule


def _layout(records: list[DeviceDay], config: dict[str, Any], layout: str, concentration: float) -> list[DeviceDay]:
    if layout == "balanced":
        return list(records)
    zones = config["zones"]["zones"]
    if layout == "end_bus_concentrated":
        return [replace(record, bus_id=max(zones[record.zone_id]["buses"])) for record in records]
    if layout == "feeder_concentrated":
        target_zone = sorted(zones)[-1]
        buses = list(zones[target_zone]["buses"])
        count = int(round(len(records) * concentration))
        result: list[DeviceDay] = []
        for index, record in enumerate(records):
            if index < count:
                result.append(replace(record, bus_id=int(buses[index % len(buses)])))
            else:
                result.append(record)
        return result
    raise ValueError(f"unknown layout: {layout}")


def run_e10(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E10_network_constraints_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    settings = protocol["e10"]
    seed = int(protocol["random_seed"]) + 10000
    steps = int(settings["steps"])
    profile_steps = list(range(steps))
    raw_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    datasets = [str(dataset) for dataset in settings["datasets"]]
    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E10", protocol, datasets)
        raw_rows = merged["raw/seed_metrics.csv"]
        timeseries_rows = merged["raw/network_timeseries.csv"]
        summary_rows = merged["summary.csv"]
        dataset_rows = merged["dataset_summary.csv"]
        _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
        _write_rows(raw_dir / "network_timeseries.csv", timeseries_rows)
        _write_rows(output / "summary.csv", summary_rows)
        _write_rows(output / "dataset_summary.csv", dataset_rows)
        _plot_e10(raw_rows, summary_rows, figures / "figure_E10_network_constraints.png")
        completed = sum(row["status"] == "completed" for row in dataset_rows)
        result = {
            "status": "completed" if completed == len(datasets) else "partial",
            "completed_datasets": completed,
            "requested_datasets": len(datasets),
            "primary_estimands": [
                "network_acceptance_ratio", "clipping_index", "voltage_margin",
                "line_margin", "transformer_margin",
            ],
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    for dataset_index, dataset in enumerate(datasets):
        base_path, weak_protocol = _dataset_config(str(dataset))
        stress_path = base_path.parent / "default_network_stress.yaml"
        if not base_path.exists() or not stress_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_network_config"})
            continue
        base_config = load_config(base_path)
        pool = load_device_day_pool(base_config)
        requested = int(settings["resources"])
        configured = int(base_config["population"]["N_simulated_resources"])
        resource_count = min(requested, configured, _balanced_unique_capacity(pool))
        if resource_count < 2:
            dataset_rows.append({
                "dataset": dataset,
                "status": "insufficient_balanced_unique_sources",
                "resource_count": resource_count,
            })
            continue
        dataset_seed = seed
        records = _sample_master(base_config, resource_count, dataset_seed)
        mode_paths = {
            "weak_correlation": base_path,
            "network_stress": stress_path,
        }
        for mode_index, mode in enumerate(settings["modes"]):
            config = load_config(mode_paths[str(mode)])
            for layout_index, layout_name in enumerate(settings["layouts"]):
                layout_records = _layout(
                    records, config, str(layout_name), float(settings["feeder_concentration"])
                )
                for control_name in settings["controls"]:
                    schedule_seed = dataset_seed + layout_index * 10_000 + 1
                    if control_name == "global":
                        schedule = _global_schedule(
                            layout_records, config, profile_steps, schedule_seed
                        )
                    else:
                        schedule = build_real_signal_schedule(
                            layout_records,
                            config,
                            profile_steps,
                            seed=schedule_seed,
                            scenario="normal",
                        )
                    group_rows: list[dict[str, Any]] = []
                    for replication in range(int(settings["replications"])):
                        diagnostics: list[dict[str, float | int]] = []
                        trace = simulate_day(
                            layout_records,
                            config,
                            steps=steps,
                            seed=dataset_seed + layout_index * 10_000 + replication,
                            profile_steps=profile_steps,
                            signal_overrides=schedule,
                            network_diagnostic_sink=diagnostics,
                            reset_each_step=False,
                        )
                        metrics = network_delivery_metrics(
                            np.asarray(trace.desired_control_kw),
                            np.asarray(trace.accepted_control_kw),
                            diagnostics,
                            voltage_min=float(config["network"]["voltage_limits_pu"][0]),
                            voltage_max=float(config["network"]["voltage_limits_pu"][1]),
                        )
                        row = {
                            "dataset": dataset,
                            "N": resource_count,
                            "mode": mode,
                            "layout": layout_name,
                            "control": control_name,
                            "seed_index": replication,
                            **metrics,
                        }
                        raw_rows.append(row)
                        group_rows.append(row)
                        for step, diagnostic in enumerate(diagnostics):
                            timeseries_rows.append({
                                "dataset": dataset,
                                "N": resource_count,
                                "mode": mode,
                                "layout": layout_name,
                                "control": control_name,
                                "seed_index": replication,
                                "step": step,
                                "desired_kw": trace.desired_control_kw[step],
                                "accepted_kw": trace.accepted_control_kw[step],
                                **diagnostic,
                            })
                    summary_rows.append({
                        "dataset": dataset,
                        "N": resource_count,
                        "mode": mode,
                        "layout": layout_name,
                        "control": control_name,
                        **{
                            key: float(np.mean([row[key] for row in group_rows]))
                            for key in (
                                "network_acceptance_ratio", "clipping_index", "voltage_margin",
                                "line_margin", "transformer_margin", "unsafe_step_fraction",
                                "mean_network_scale", "fully_blocked_fraction",
                            )
                        },
                    })
        dataset_rows.append({
            "dataset": dataset,
            "status": "completed",
            "resource_count": resource_count,
            "configured_resources": configured,
            "balanced_unique_capacity": _balanced_unique_capacity(pool),
            "weak_config_protocol": weak_protocol,
        })
    _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
    _write_rows(raw_dir / "network_timeseries.csv", timeseries_rows)
    _write_rows(output / "summary.csv", summary_rows)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    _plot_e10(raw_rows, summary_rows, figures / "figure_E10_network_constraints.png")
    completed = sum(row["status"] == "completed" for row in dataset_rows)
    result = {
        "status": "completed" if completed == len(settings["datasets"]) else "partial",
        "completed_datasets": completed,
        "requested_datasets": len(settings["datasets"]),
        "primary_estimands": [
            "network_acceptance_ratio", "clipping_index", "voltage_margin",
            "line_margin", "transformer_margin",
        ],
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def run_e11(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E11_failure_modes_new"
    figures = output / "Figs"
    figures.mkdir(parents=True, exist_ok=True)
    e6 = _read_csv(root / "E6_behaviour_availability_new" / "summary.csv")
    e10 = _read_csv(root / "E10_network_constraints_new" / "raw" / "seed_metrics.csv")
    rows: list[dict[str, Any]] = []
    for row in e6:
        if row["structure"] == "community":
            rows.append({
                "failure_mode": "behavior_unavailability",
                "dataset": row["dataset"],
                "strength": 1.0 - float(row["participation"]),
                "stress_label": f"unavailable={1.0 - float(row['participation']):.2f}",
                "failure_probability": float(row["failure_probability"]),
            })
    network_group: dict[tuple[str, str, str, str], list[bool]] = {}
    for row in e10:
        key = (row["dataset"], row["mode"], row["layout"], row["control"])
        network_group.setdefault(key, []).append(
            float(row["unsafe_step_fraction"]) > 0 or float(row["network_acceptance_ratio"]) < 0.80
        )
    for key, values in network_group.items():
        dataset, mode, layout, control = key
        rows.append({
            "failure_mode": "network_stress",
            "dataset": dataset,
            "strength": None,
            "stress_label": f"{mode}/{layout}/{control}",
            "failure_probability": float(np.mean(values)),
        })
    thresholds: list[dict[str, Any]] = []
    behavior_datasets = sorted({
        row["dataset"] for row in rows if row["failure_mode"] == "behavior_unavailability"
    })
    for dataset in behavior_datasets:
        selected = sorted(
            [
                row for row in rows
                if row["failure_mode"] == "behavior_unavailability"
                and row["dataset"] == dataset
            ],
            key=lambda row: row["strength"],
        )
        thresholds.append({
            "failure_mode": "behavior_unavailability",
            "dataset": dataset,
            "x_10": failure_threshold([row["strength"] for row in selected], [row["failure_probability"] for row in selected], 0.10),
            "x_50": failure_threshold([row["strength"] for row in selected], [row["failure_probability"] for row in selected], 0.50),
            "x_90": failure_threshold([row["strength"] for row in selected], [row["failure_probability"] for row in selected], 0.90),
        })
    pooled_strengths = sorted({
        float(row["strength"])
        for row in rows
        if row["failure_mode"] == "behavior_unavailability"
    })
    pooled_probabilities = [
        float(np.mean([
            row["failure_probability"] for row in rows
            if row["failure_mode"] == "behavior_unavailability"
            and row["strength"] == strength
        ]))
        for strength in pooled_strengths
    ]
    thresholds.append({
        "failure_mode": "behavior_unavailability",
        "dataset": "cross_dataset_mean",
        "x_10": failure_threshold(pooled_strengths, pooled_probabilities, 0.10),
        "x_50": failure_threshold(pooled_strengths, pooled_probabilities, 0.50),
        "x_90": failure_threshold(pooled_strengths, pooled_probabilities, 0.90),
    })
    _write_rows(output / "failure_probabilities.csv", rows)
    _write_rows(output / "failure_thresholds.csv", thresholds)
    _plot_e11(rows, figures / "figure_E11_failure_modes.png")
    result = {
        "status": "completed",
        "primary_estimands": thresholds,
        "recovery_fraction": "not_computed_no_registered_recovery_strategy_in_phase_1",
        "omitted_mechanisms": {
            "residual_correlation": "E1 excluded from this rerun by scope",
            "controller_synchronization": "requires E2 device-event protocol",
            "policy_drift": "requires E4 streaming detector protocol",
        },
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _figure4_energy_metrics(trace: dict[str, Any]) -> dict[str, float | bool]:
    energy_input = np.asarray(trace.get("energy_input_kw", trace.get("pv_kw", [])), dtype=float)
    load = np.asarray(trace.get("load_kw", []), dtype=float)
    control = np.asarray(trace.get("accepted_control_kw", []), dtype=float)
    if not energy_input.size or energy_input.size != load.size or load.size != control.size:
        return {
            "energy_trace_available": False,
            "baseline_curtailment_mwh": float("nan"),
            "eps_curtailment_mwh": float("nan"),
            "curtailment_reduction_pct": float("nan"),
        }
    baseline = np.maximum(energy_input - load, 0.0)
    after_eps = np.maximum(energy_input - load - np.maximum(control, 0.0), 0.0)
    baseline_mwh = float(np.sum(baseline) * 5.0 / 60.0 / 1000.0)
    eps_mwh = float(np.sum(after_eps) * 5.0 / 60.0 / 1000.0)
    reduction = (
        100.0 * (baseline_mwh - eps_mwh) / baseline_mwh
        if baseline_mwh > 1e-12
        else float("nan")
    )
    return {
        "energy_trace_available": True,
        "baseline_curtailment_mwh": baseline_mwh,
        "eps_curtailment_mwh": eps_mwh,
        "curtailment_reduction_pct": reduction,
    }


def _largest_zone_balanced_capacity(zone_counts: dict[str, Any]) -> int:
    counts = [int(value) for _, value in sorted(zone_counts.items())]
    for candidate in range(sum(counts), -1, -1):
        base, remainder = divmod(candidate, len(counts))
        if all(
            available >= base + (1 if index < remainder else 0)
            for index, available in enumerate(counts)
        ):
            return candidate
    return 0


def run_e12(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    from src.extra.dataset_experiment.run import DATASETS

    output = root / "E12_dataset_pipeline_integrity_new"
    figures = output / "Figs"
    figures.mkdir(parents=True, exist_ok=True)
    settings = protocol["e12"]
    expected_figures = [str(name) for name in settings["expected_figures"]]
    modes = [str(mode) for mode in settings["modes"]]
    datasets = [str(dataset) for dataset in protocol["e7"]["datasets"]]
    rows: list[dict[str, Any]] = []
    fleet_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        dataset_root = ROOT / "results" / f"{dataset}{DATASET_RESULTS_SUFFIX}"
        status = _json_or_empty(dataset_root / "run_status.json")
        preflight = _json_or_empty(dataset_root / "preflight.json")
        zone_counts = preflight.get("dataset_pool", {}).get("zone_source_counts", {})
        balanced_equal_limit = (
            len(zone_counts) * min(int(value) for value in zone_counts.values())
            if zone_counts
            else 0
        )
        zone_balanced_capacity = (
            _largest_zone_balanced_capacity(zone_counts) if zone_counts else 0
        )
        config_path = dataset_root / "config" / "default.yaml"
        config = load_config(config_path) if config_path.exists() else {}
        selected_main = int(config.get("population", {}).get("N_simulated_resources", 0))
        selected_rho = int(config.get("experiment", {}).get("figure5_rho_resources", 0))
        spec = DATASETS.get(dataset, {})
        fleet_rows.append({
            "dataset": dataset,
            "requested_main_resources": int(spec.get("main_resources", selected_main)),
            "requested_rho_resources": int(spec.get("rho_resources", selected_rho)),
            "balanced_equal_limit": balanced_equal_limit,
            "zone_balanced_capacity": zone_balanced_capacity,
            "selected_main_resources": selected_main,
            "selected_rho_resources": selected_rho,
            "zone_source_counts": json.dumps(zone_counts, sort_keys=True),
            "main_within_limit": bool(
                selected_main and selected_main <= balanced_equal_limit
            ),
            "rho_within_limit": bool(
                selected_rho and selected_rho <= zone_balanced_capacity
            ),
        })
        mapping = str(spec.get("input_mapping", "unknown"))
        if mapping == "load_shape_counterfactual":
            provenance = "counterfactual_load_shape"
        elif mapping == "measured_or_load_shape_counterfactual":
            provenance = "measured_or_counterfactual"
        else:
            provenance = mapping

        for mode in modes:
            mode_root = dataset_root / mode
            present = [name for name in expected_figures if (mode_root / "Figs" / name).exists()]
            directional = mode_root / "data" / "directional_n_scaling.json"
            figure4 = mode_root / "Figs" / "fig4_robustness_generalization.png"
            figure4_self = mode_root / "Figs" / "fig4_self_consumption.png"
            figure4_distinct = bool(
                figure4.exists()
                and figure4_self.exists()
                and _sha256(figure4) != _sha256(figure4_self)
            )
            estimator = _json_or_empty(
                mode_root / "estimation" / "estimation_validation_results.json"
            )
            actual = np.asarray(
                estimator.get("test_data", {}).get("actuals", []), dtype=float
            )
            tolerance = 1e-9
            charge = int(np.sum(actual > tolerance))
            discharge = int(np.sum(actual < -tolerance))
            blocked = int(np.sum(np.abs(actual) <= tolerance))
            responsive = charge + discharge
            charge_fraction = charge / responsive if responsive else float("nan")
            trace_path = mode_root / "data" / "network_timeseries_day.json"
            if not trace_path.exists():
                trace_path = mode_root / "data" / "network_timeseries.json"
            energy = _figure4_energy_metrics(_json_or_empty(trace_path))
            status_complete = status.get("status") == "completed"
            completeness_checks = (
                len(present)
                + int(directional.exists())
                + int(figure4_distinct)
                + int(status_complete)
            )
            rows.append({
                "dataset": dataset,
                "mode": mode,
                "run_status": status.get("status", "missing"),
                "figure_count": len(present),
                "expected_figure_count": len(expected_figures),
                "directional_scaling_available": directional.exists(),
                "figure4_distinct": figure4_distinct,
                "completeness_score": completeness_checks / (len(expected_figures) + 3),
                "valid_base_mode": bool(
                    status_complete
                    and len(present) == len(expected_figures)
                    and directional.exists()
                    and figure4_distinct
                ),
                "r2": estimator.get("point_metrics", {}).get("r2", {}).get("value"),
                "validation_charge_count": charge,
                "validation_discharge_count": discharge,
                "validation_blocked_count": blocked,
                "validation_charge_fraction": charge_fraction,
                "input_provenance": provenance,
                **energy,
            })

    _write_rows(output / "base_mode_audit.csv", rows)
    _write_rows(output / "fleet_capacity_audit.csv", fleet_rows)
    _plot_e12(rows, fleet_rows, figures / "figure_E12_dataset_pipeline_integrity.png")
    invalid = [
        f"{row['dataset']}/{row['mode']}" for row in rows if not row["valid_base_mode"]
    ]
    result = {
        "status": "completed",
        "audited_datasets": len(datasets),
        "audited_modes": len(rows),
        "valid_base_modes": len(rows) - len(invalid),
        "invalid_base_modes": invalid,
        "all_fleet_selections_within_balanced_limit": all(
            row["main_within_limit"] and row["rho_within_limit"] for row in fleet_rows
        ),
        "energy_unit": "MWh = sum(kW) * (5/60 h) / 1000",
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _typed_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                converted[key] = None
            elif value == "True":
                converted[key] = True
            elif value == "False":
                converted[key] = False
            else:
                try:
                    number = float(value)
                except ValueError:
                    converted[key] = value
                else:
                    converted[key] = int(number) if number.is_integer() else number
        rows.append(converted)
    return rows


_SHARD_FILES = {
    "E1": (
        "E1_scale_boundary_new",
        [
            "raw/seed_metrics.csv", "raw/condition_responses.csv", "summary.csv",
            "boundaries.csv", "neff_boundaries.csv", "dataset_summary.csv",
        ],
    ),
    "E2": (
        "E2_controller_synchronization_new",
        [
            "raw/seed_metrics.csv", "synchronization_summary.csv",
            "synchronization_thresholds.csv", "dataset_summary.csv",
        ],
    ),
    "E3": (
        "E3_phase_coherence_new",
        ["raw/window_metrics.csv", "phase_summary.csv", "dataset_summary.csv"],
    ),
    "E4": (
        "E4_controller_drift_new",
        [
            "raw/policy_drift_stream.csv", "drift_events.csv",
            "recovery_curves.csv", "dataset_summary.csv",
        ],
    ),
    "E6": (
        "E6_behaviour_availability_new",
        ["raw/seed_metrics.csv", "summary.csv", "dataset_summary.csv"],
    ),
    "E15": (
        "E15_availability_cases_new",
        ["raw/seed_metrics.csv", "summary.csv", "dataset_summary.csv"],
    ),
    "E7": (
        "E7_empirical_scaling_new",
        ["raw/seed_metrics.csv", "scaling_summary.csv", "dataset_summary.csv"],
    ),
    "E10": (
        "E10_network_constraints_new",
        ["raw/seed_metrics.csv", "raw/network_timeseries.csv", "summary.csv", "dataset_summary.csv"],
    ),
    "E13": (
        "E13_control_logic_new",
        ["raw/seed_metrics.csv", "summary.csv", "neff_boundaries.csv", "dataset_summary.csv"],
    ),
    "E14": (
        "E14_dominant_capacity_new",
        ["raw/seed_metrics.csv", "summary.csv", "dataset_summary.csv"],
    ),
    "E9": (
        "E9_long_horizon_soc_new",
        ["daily_metrics.csv", "dataset_summary.csv"],
    ),
}


def _run_dataset_shard(
    experiment: str, protocol: dict[str, Any], dataset: str
) -> dict[str, list[dict[str, Any]]]:
    section = experiment.lower()
    local = copy.deepcopy(protocol)
    local[section]["datasets"] = [dataset]
    local["execution"]["dataset_workers"] = 1
    output_name, files = _SHARD_FILES[experiment]
    with tempfile.TemporaryDirectory(prefix=f"nc_{experiment.lower()}_") as temporary:
        root = Path(temporary)
        RUNNERS[experiment](local, root)
        result: dict[str, Any] = {
            relative: _typed_csv_rows(root / output_name / relative)
            for relative in files
        }
        binary: dict[str, bytes] = {}
        for path in (root / output_name).rglob("*"):
            if path.is_file() and path.suffix.lower() in {".npz", ".png"}:
                binary[str(path.relative_to(root / output_name))] = path.read_bytes()
        result["__binary__"] = binary
        return result


def _parallel_dataset_rows(
    experiment: str, protocol: dict[str, Any], datasets: list[str]
) -> dict[str, list[dict[str, Any]]]:
    workers = min(int(protocol["execution"].get("dataset_workers", 1)), len(datasets))
    combined: dict[str, Any] = {
        relative: [] for relative in _SHARD_FILES[experiment][1]
    }
    combined["__binary__"] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_dataset_shard, experiment, protocol, dataset)
            for dataset in datasets
        ]
        for future in futures:
            shard = future.result()
            for relative, rows in shard.items():
                if relative == "__binary__":
                    combined[relative].update(rows)
                else:
                    combined[relative].extend(rows)
    return combined


def _write_binary_outputs(output: Path, files: dict[str, bytes]) -> None:
    for relative, payload in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def _plot_e1(
    summary: list[dict[str, Any]],
    neff_boundaries: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    figures: Path,
    collapse: dict[str, Any] | None = None,
) -> None:
    completed = [row for row in dataset_summary if row.get("status") == "completed"]
    datasets = [str(row["dataset"]) for row in completed]
    if not datasets:
        return
    palette = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(len(datasets), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    def _coupled(dataset: str, key: str) -> list[dict[str, Any]]:
        usable = []
        for row in summary:
            if str(row["dataset"]) != dataset or str(row.get("arm", "")) != "data_coupled":
                continue
            value = row.get(key)
            if value is None or value == "":
                continue
            usable.append(row)
        return sorted(usable, key=lambda item: float(item[key]))

    for color, dataset in zip(palette, datasets):
        rows = _coupled(dataset, "margin")
        if not rows:
            continue
        axes[0, 0].plot(
            [float(row["margin"]) for row in rows],
            [float(row["p_controllable"]) for row in rows],
            "o-", linewidth=1, markersize=3.5, alpha=0.8, color=color,
            label=_short_dataset(dataset),
        )
    axes[0, 0].axvline(1.0, linestyle="--", color=COLORS["gray"], linewidth=1.2)
    axes[0, 0].axhline(0.90, linestyle=":", color=COLORS["orange"], linewidth=1.2)
    axes[0, 0].set(
        xscale="log", xlabel="Effective margin $N_{eff}/N^*_{eff}$", ylabel="$p_{ctrl}$",
        ylim=(-0.02, 1.02), title="A  Controllability collapses in the effective-margin coordinate",
    )
    if collapse and collapse.get("variance_ratio_margin_over_N") is not None:
        axes[0, 0].text(
            0.02, 0.06,
            f"between-dataset variance ratio vs raw $N$: {collapse['variance_ratio_margin_over_N']:.3f}",
            transform=axes[0, 0].transAxes, fontsize=8,
        )
    axes[0, 0].legend(fontsize=6, ncol=3, loc="lower right")

    for color, dataset in zip(palette, datasets):
        rows = _coupled(dataset, "N")
        if not rows:
            continue
        axes[0, 1].plot(
            [float(row["N"]) for row in rows],
            [float(row["p_controllable"]) for row in rows],
            "o-", linewidth=1, markersize=3.5, alpha=0.8, color=color,
        )
    axes[0, 1].axhline(0.90, linestyle=":", color=COLORS["orange"], linewidth=1.2)
    axes[0, 1].set(
        xscale="log", xlabel="Physical fleet size $N$", ylabel="$p_{ctrl}$",
        ylim=(-0.02, 1.02), title="B  Control: the same points do not collapse in raw $N$",
    )

    reached = [row for row in neff_boundaries if row.get("N_star_eff")]
    if reached:
        order = sorted(reached, key=lambda row: float(row["N_star_eff"]))
        y = np.arange(len(order))
        axes[1, 0].scatter(
            [float(row["N_star_eff"]) for row in order], y, color=COLORS["blue"], zorder=3,
        )
        for index, row in enumerate(order):
            if row.get("N_star_eff_strict"):
                axes[1, 0].plot(
                    [float(row["N_star_eff"]), float(row["N_star_eff_strict"])],
                    [index, index], color=COLORS["gray"], linewidth=1.4, zorder=2,
                )
        axes[1, 0].set(
            xscale="log", yticks=y,
            yticklabels=[_short_dataset(str(row["dataset"])) for row in order],
            xlabel="Critical effective fleet $N^*_{eff}$",
            title="C  Critical effective fleet per dataset (bar: envelope to strict)",
        )
    not_reached = [row["dataset"] for row in neff_boundaries if not row.get("N_star_eff")]
    if not_reached:
        axes[1, 0].text(
            0.02, 0.02, "not reached: " + ", ".join(_short_dataset(str(d)) for d in not_reached),
            transform=axes[1, 0].transAxes, fontsize=7, color=COLORS["orange"],
        )

    arm_styles = {
        "decoupled": ("o", COLORS["gray"], "decoupled control (marginals preserved)"),
        "data_coupled": ("^", COLORS["blue"], "data-coupled"),
    }
    pooled_slopes: dict[str, float] = {}
    for arm_label, (marker, color, label) in arm_styles.items():
        arm_rows = [
            row for row in summary
            if str(row.get("arm", "")) == arm_label
            and float(row.get("condition_cv") or 0.0) > 0
            and float(row.get("N_eff") or 0.0) > 0
        ]
        if len(arm_rows) < 2:
            continue
        axes[1, 1].plot(
            [float(row["N_eff"]) for row in arm_rows],
            [float(row["condition_cv"]) for row in arm_rows],
            marker, markersize=3.5, alpha=0.55, color=color, linestyle="none", label=label,
        )
        pooled_slopes[arm_label] = loglog_slope(
            [float(row["N_eff"]) for row in arm_rows],
            [float(row["condition_cv"]) for row in arm_rows],
        )
    all_rows = [
        row for row in summary
        if float(row.get("condition_cv") or 0.0) > 0 and float(row.get("N_eff") or 0.0) > 0
    ]
    if len(all_rows) >= 2:
        reference = np.asarray([
            min(float(row["N_eff"]) for row in all_rows),
            max(float(row["N_eff"]) for row in all_rows),
        ])
        anchor = float(np.median([float(row["condition_cv"]) for row in all_rows]))
        centre = float(np.median([float(row["N_eff"]) for row in all_rows]))
        axes[1, 1].plot(
            reference, anchor * (reference / centre) ** -0.5,
            "--", color=COLORS["orange"], label="$-1/2$ reference",
        )
        caption = "  ".join(
            f"{name}: {value:.3f}" for name, value in sorted(pooled_slopes.items())
        )
        axes[1, 1].set_title(f"D  CV against effective fleet (pooled slope — {caption})")
        axes[1, 1].legend(fontsize=7)
    else:
        axes[1, 1].set_title("D  CV against effective fleet")
    axes[1, 1].set(xscale="log", yscale="log", xlabel="$N_{eff}$", ylabel="Condition CV")

    fig.savefig(figures / "figure_E1_effective_margin_boundary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(18, 11), constrained_layout=True, squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        rows = _coupled(dataset, "N_eff")
        if not rows:
            continue
        axis.plot(
            [float(row["N_eff"]) for row in rows],
            [float(row["p_controllable"]) for row in rows],
            "o-", markersize=3, linewidth=1, color=COLORS["blue"],
        )
        axis.fill_between(
            [float(row["N_eff"]) for row in rows],
            [float(row["p_controllable_ci_lower"]) for row in rows],
            [float(row["p_controllable_ci_upper"]) for row in rows],
            color=COLORS["blue"], alpha=0.18,
        )
        star = next((row.get("N_star_eff") for row in rows if row.get("N_star_eff")), None)
        if star:
            axis.axvline(float(star), linestyle="--", color=COLORS["orange"], linewidth=1)
        axis.axhline(0.90, linestyle=":", color=COLORS["gray"], linewidth=1)
        axis.set(
            xscale="log", ylim=(-0.02, 1.02), title=_short_dataset(dataset),
            xlabel="$N_{eff}$", ylabel="$p_{ctrl}$",
        )
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(datasets):]:
        axis.set_visible(False)
    fig.savefig(figures / "figure_E1_pctrl_all_datasets.png", dpi=180)
    plt.close(fig)


def _plot_event_raster(events: np.ndarray, path: Path, *, title: str) -> None:
    count = min(events.shape[1], 250)
    signed = np.sign(events[:, :count]).T
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    image = axis.imshow(signed, aspect="auto", interpolation="nearest", cmap="bwr", vmin=-1, vmax=1)
    axis.set(xlabel="Time step", ylabel="Resource", title=title)
    colorbar = fig.colorbar(image, ax=axis, ticks=[-1, 0, 1])
    colorbar.ax.set_yticklabels(["discharge", "idle", "charge"])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_e2_coupling_arms(summary: list[dict[str, Any]], figures: Path) -> None:
    arms = ("decoupled", "data_coupled")
    rows = [row for row in summary if row.get("coupling_arm") in arms]
    if not rows:
        return
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    c_values = sorted({float(row["homogeneity"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for axis, metric, label in [
        (axes[0], "X_sync", "$X_{sync}$"),
        (axes[1], "mean_nrmse", "NRMSE"),
        (axes[2], "rho_measured", r"$\rho$ (response)"),
    ]:
        for arm, style in zip(arms, ("--", "-")):
            axis.plot(
                c_values,
                [float(np.mean([
                    float(row[metric]) for row in rows
                    if row["coupling_arm"] == arm and float(row["homogeneity"]) == c
                ] or [np.nan])) for c in c_values],
                style, linewidth=2, label=arm,
            )
        axis.set(xlabel="Controller homogeneity c", ylabel=label, title=f"{label}: coupled vs decoupled arm")
        axis.legend()
    fig.savefig(figures / "figure_E2_coupling_arms.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, max(4.0, 0.32 * len(datasets) + 1.5)))
    y = np.arange(len(datasets))
    for offset, arm, marker in ((-0.16, "decoupled", "o"), (0.16, "data_coupled", "s")):
        values = [
            float(np.mean([
                float(row["X_sync"]) for row in rows
                if row["dataset"] == dataset and row["coupling_arm"] == arm
            ] or [np.nan]))
            for dataset in datasets
        ]
        axis.scatter(values, y + offset, marker=marker, label=arm)
    axis.set(
        yticks=y, yticklabels=[_short_dataset(value) for value in datasets],
        xlabel="$X_{sync}$ (mean over conditions)", title="Per dataset: data-coupled vs decoupled arm",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "figure_E2_coupling_arms_by_dataset.png", dpi=180)
    plt.close(fig)


def _plot_e2(
    summary: list[dict[str, Any]], thresholds: list[dict[str, Any]], figures: Path
) -> None:
    both_arms = list(summary)
    _plot_e2_coupling_arms(both_arms, figures)
    summary = [row for row in summary if row.get("coupling_arm", "data_coupled") == "data_coupled"]
    thresholds = [
        row for row in thresholds if row.get("coupling_arm", "data_coupled") == "data_coupled"
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in summary))
    c_values = sorted({float(row["homogeneity"]) for row in summary})
    if not datasets:
        return

    def aggregate_grid(metric: str) -> np.ndarray:
        return np.asarray([
            [np.mean([
                float(row[metric]) for row in summary
                if row["dataset"] == dataset and float(row["homogeneity"]) == c
            ]) for c in c_values]
            for dataset in datasets
        ])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for axis, metric, title, cmap, vmin, vmax in [
        (axes[0, 0], "X_sync", "A  Synchronization excess", "magma", 0.0, None),
        (axes[0, 1], "mean_nrmse", "B  Dispatch error", "viridis", 0.0, None),
        (axes[1, 2], "R2", "F  Frozen-estimator $R^2$", "RdYlGn", -1.0, 1.0),
    ]:
        image = axis.imshow(aggregate_grid(metric), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set(
            xticks=range(len(c_values)), xticklabels=c_values,
            yticks=range(len(datasets)), yticklabels=[_short_dataset(value) for value in datasets],
            xlabel="Controller homogeneity c", ylabel="Dataset", title=title,
        )
        fig.colorbar(image, ax=axis, shrink=0.8)

    y = np.arange(len(datasets))
    for index, dataset in enumerate(datasets):
        rows = [row for row in thresholds if row["dataset"] == dataset]
        sync = [float(row["c_sync_star"]) for row in rows if row["c_sync_star"] is not None]
        fail = [float(row["c_fail_star"]) for row in rows if row["c_fail_star"] is not None]
        if sync:
            axes[0, 2].scatter(np.median(sync), index, color=COLORS["orange"], marker="o")
        if fail:
            axes[0, 2].scatter(np.median(fail), index, color=COLORS["blue"], marker="x")
    axes[0, 2].set(
        xlim=(-0.03, 1.03), yticks=y, yticklabels=[_short_dataset(value) for value in datasets],
        xlabel="Median reached threshold c", title="C  Synchronization and failure thresholds",
    )
    axes[0, 2].scatter([], [], color=COLORS["orange"], marker="o", label="$c^*_{sync}$")
    axes[0, 2].scatter([], [], color=COLORS["blue"], marker="x", label="$c^*_{fail}$")
    axes[0, 2].legend()

    for dataset in datasets:
        axes[1, 0].plot(
            c_values,
            [np.mean([float(row["X_sync"]) for row in summary
                      if row["dataset"] == dataset and float(row["homogeneity"]) == c])
             for c in c_values],
            linewidth=1, alpha=0.75, label=_short_dataset(dataset),
        )
    axes[1, 0].set(
        xlabel="Controller homogeneity c", ylabel="$X_{sync}$",
        title="D  Dataset-specific synchronization curves",
    )

    scatter = axes[1, 1].scatter(
        [float(row["X_sync"]) for row in summary],
        [float(row["mean_nrmse"]) for row in summary],
        c=[float(row["homogeneity"]) for row in summary], cmap="plasma", s=12, alpha=0.45,
    )
    axes[1, 1].axhline(0.10, linestyle="--", color=COLORS["gray"])
    axes[1, 1].set(
        xlabel="$X_{sync}$", ylabel="NRMSE", title="E  Synchronization consequence",
    )
    fig.colorbar(scatter, ax=axes[1, 1], label="c", shrink=0.8)
    fig.savefig(figures / "figure_E2_controller_synchronization.png", dpi=180)
    plt.close(fig)

    modes = list(dict.fromkeys(str(row["broadcast_mode"]) for row in summary))
    fig, axes = plt.subplots(2, len(modes), figsize=(18, 9), constrained_layout=True, squeeze=False)
    for column, mode in enumerate(modes):
        mode_rows = [row for row in summary if row["broadcast_mode"] == mode]
        for row_index, (metric, label, cmap) in enumerate([
            ("X_sync", "$X_{sync}$", "magma"),
            ("mean_nrmse", "NRMSE", "viridis"),
        ]):
            grid = np.asarray([
                [np.mean([
                    float(row[metric]) for row in mode_rows
                    if row["dataset"] == dataset and float(row["homogeneity"]) == c
                ]) for c in c_values]
                for dataset in datasets
            ])
            image = axes[row_index, column].imshow(grid, aspect="auto", cmap=cmap, vmin=0.0)
            axes[row_index, column].set(
                xticks=range(len(c_values)), xticklabels=c_values,
                yticks=range(len(datasets)),
                yticklabels=[_short_dataset(value) for value in datasets] if column == 0 else [],
                xlabel="Controller homogeneity c",
                ylabel="Dataset" if column == 0 else "",
                title=f"{mode}: {label}",
            )
            fig.colorbar(image, ax=axes[row_index, column], shrink=0.78)
    fig.savefig(figures / "figure_E2_broadcast_mode_facets.png", dpi=180)
    plt.close(fig)


def _plot_e3_coupling_arms(summary: list[dict[str, Any]], figures: Path) -> None:
    rows = [row for row in summary if row.get("coupling_arm") in ("decoupled", "data_coupled")]
    if not rows:
        return
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    c_values = sorted({float(row["homogeneity"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for axis, metric, label in (
        (axes[0], "R_K_mean", "$R_K$ (Kuramoto order)"),
        (axes[1], "H_phase_mean", "$H_{phase}$"),
    ):
        for arm, style in (("decoupled", "--"), ("data_coupled", "-")):
            axis.plot(
                c_values,
                [float(np.mean([
                    float(row[metric]) for row in rows
                    if row["coupling_arm"] == arm and float(row["homogeneity"]) == c
                ] or [np.nan])) for c in c_values],
                style, linewidth=2, label=arm,
            )
        axis.set(xlabel="Controller homogeneity c", ylabel=label, title=f"{label}: coupled vs decoupled arm")
        axis.legend()
    fig.savefig(figures / "figure_E3_coupling_arms.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, max(4.0, 0.32 * len(datasets) + 1.5)))
    y = np.arange(len(datasets))
    for offset, arm, marker in ((-0.16, "decoupled", "o"), (0.16, "data_coupled", "s")):
        axis.scatter(
            [
                float(np.mean([
                    float(row["R_K_mean"]) for row in rows
                    if row["dataset"] == dataset and row["coupling_arm"] == arm
                ] or [np.nan]))
                for dataset in datasets
            ],
            y + offset, marker=marker, label=arm,
        )
    axis.set(
        yticks=y, yticklabels=[_short_dataset(value) for value in datasets],
        xlabel="$R_K$ (mean over cells)", title="Per dataset: data-coupled vs decoupled arm",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "figure_E3_coupling_arms_by_dataset.png", dpi=180)
    plt.close(fig)


def _plot_e3(
    summary: list[dict[str, Any]], raw: list[dict[str, Any]], figures: Path
) -> None:
    if not summary:
        return
    _plot_e3_coupling_arms(summary, figures)
    summary = [row for row in summary if row.get("coupling_arm", "data_coupled") == "data_coupled"]
    raw = [row for row in raw if row.get("coupling_arm", "data_coupled") == "data_coupled"]
    if not summary:
        return
    homogeneity_values = sorted({float(row["homogeneity"]) for row in summary})
    baseline_c, stress_c = homogeneity_values[0], homogeneity_values[-1]
    nominal = float(summary[0].get("nominal_exceedance_level") or 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    grid_rows = [row for row in summary if row["design_role"] == "period_delay_grid"]
    for axis, homogeneity, title in (
        (axes[0, 0], baseline_c, f"A  Exceedance $H_{{phase}}$, heterogeneous controllers (c={baseline_c:g})"),
        (axes[0, 1], stress_c, f"B  Exceedance $H_{{phase}}$, homogenized controllers (c={stress_c:g})"),
    ):
        subset = [row for row in grid_rows if float(row["homogeneity"]) == homogeneity]
        periods = sorted({float(row["period_minutes"]) for row in subset})
        delays = sorted({str(row["delay_distribution"]) for row in subset})
        if not periods or not delays:
            continue
        matrix = np.full((len(delays), len(periods)), np.nan)
        for y, delay in enumerate(delays):
            for x, period in enumerate(periods):
                values = [
                    float(row["H_phase_mean"]) for row in subset
                    if str(row["delay_distribution"]) == delay
                    and float(row["period_minutes"]) == period
                ]
                if values:
                    matrix[y, x] = float(np.mean(values))
        image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.05, np.nanmax(matrix)))
        axis.set(
            xticks=range(len(periods)), xticklabels=[f"{value:g}" for value in periods],
            yticks=range(len(delays)), yticklabels=delays,
            xlabel="Broadcast cycle (min)", ylabel="Response delay distribution", title=title,
        )
        for y in range(len(delays)):
            for x in range(len(periods)):
                if np.isfinite(matrix[y, x]):
                    axis.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center",
                              fontsize=7, color="white" if matrix[y, x] > 0.2 else "black")
        fig.colorbar(image, ax=axis, label="$H_{phase}$")

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in summary))
    positions = np.arange(len(datasets))
    for offset, homogeneity, color, label in (
        (-0.18, baseline_c, COLORS["blue"], f"c={baseline_c:g}"),
        (0.18, stress_c, COLORS["orange"], f"c={stress_c:g}"),
    ):
        values = [
            [float(row["H_phase_mean"]) for row in summary
             if str(row["dataset"]) == dataset and float(row["homogeneity"]) == homogeneity]
            for dataset in datasets
        ]
        axes[1, 0].scatter(
            np.repeat(positions + offset, [len(item) for item in values]),
            [value for item in values for value in item],
            s=9, alpha=0.5, color=color, label=label,
        )
        axes[1, 0].scatter(
            positions + offset,
            [float(np.mean(item)) if item else np.nan for item in values],
            marker="_", s=420, color=color, linewidths=2.2,
        )
    axes[1, 0].axhline(nominal, linestyle="--", color=COLORS["gray"],
                       label=f"nominal {nominal:g}")
    axes[1, 0].set(
        xticks=positions, xticklabels=[_short_dataset(value) for value in datasets],
        ylabel="$H_{phase}$", title="C  Exceedance against the surrogate null",
    )
    axes[1, 0].tick_params(axis="x", rotation=40, labelsize=7)
    axes[1, 0].legend(fontsize=7)

    for homogeneity, color, label in (
        (baseline_c, COLORS["blue"], f"c={baseline_c:g}"),
        (stress_c, COLORS["orange"], f"c={stress_c:g}"),
    ):
        points = [
            row for row in summary
            if float(row["homogeneity"]) == homogeneity
            and row.get("nrmse_vs_heterogeneous") is not None
        ]
        if not points:
            continue
        axes[1, 1].scatter(
            [float(row["H_phase_mean"]) for row in points],
            [float(row["nrmse_vs_heterogeneous"]) for row in points],
            s=14, alpha=0.6, color=color, label=label,
        )
    axes[1, 1].set(
        xlabel="$H_{phase}$", ylabel="Aggregate NRMSE vs heterogeneous reference",
        title="D  Locking only matters if delivery degrades with it",
    )
    axes[1, 1].legend(fontsize=8)
    fig.savefig(figures / "figure_E3_phase_coherence.png", dpi=180)
    plt.close(fig)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _plot_e4(
    stream: list[dict[str, Any]],
    events: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    figures: Path,
) -> None:
    if not events:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    if stream:
        target = str(stream[0]["dataset"])
        fraction = max(float(row["unknown_fraction"]) for row in stream)
        trace = sorted(
            [row for row in stream
             if str(row["dataset"]) == target and str(row["mode"]) == "abrupt"
             and float(row["unknown_fraction"]) == fraction
             and int(row["seed_index"]) == 0],
            key=lambda item: int(item["window_index"]),
        )
        if trace:
            x = [int(row["window_index"]) for row in trace]
            for key, color, label in (
                ("loss_frozen", COLORS["orange"], "frozen"),
                ("loss_recalibrated", COLORS["blue"], "aggregate recalibration"),
                ("loss_oracle", COLORS["green"], "full retraining oracle"),
            ):
                axes[0, 0].plot(x, [float(row[key]) for row in trace], linewidth=1.2,
                                color=color, label=label)
            axes[0, 0].axhline(float(trace[0]["threshold"]), linestyle=":",
                               color=COLORS["gray"], label="frozen detection threshold")
            axes[0, 0].axvline(int(trace[0]["injection_window"]), linestyle="--",
                               color=COLORS["purple"], label="drift injection")
            axes[0, 0].set(
                xlabel="Aggregate-sample window", ylabel="Window NRMSE",
                title=f"A  {_short_dataset(target)}: abrupt drift at unknown fraction {fraction:g}",
            )
            axes[0, 0].legend(fontsize=7)

    for mode, color in (("abrupt", COLORS["orange"]), ("gradual", COLORS["blue"])):
        subset = [row for row in events if str(row["mode"]) == mode]
        if not subset:
            continue
        detected = [
            float(row["T_detect_windows"]) for row in subset
            if row.get("T_detect_windows") is not None
        ]
        censored = sum(_as_bool(row["detection_censored"]) for row in subset)
        if not detected:
            continue
        horizon = np.arange(0, int(max(detected)) + 2)
        survival = [
            (sum(value > point for value in detected) + censored) / len(subset)
            for point in horizon
        ]
        axes[0, 1].step(horizon, survival, where="post", color=color,
                        label=f"{mode} (censored {censored}/{len(subset)})")
    axes[0, 1].set(
        xlabel="Windows since injection", ylabel="Fraction not yet detected", ylim=(-0.02, 1.02),
        title="B  Detection survival, censored runs retained",
    )
    axes[0, 1].legend(fontsize=8)

    fractions = sorted({float(row["unknown_fraction"]) for row in events})
    positions = np.arange(len(fractions))
    for offset, mode, color in ((-0.16, "abrupt", COLORS["orange"]), (0.16, "gradual", COLORS["blue"])):
        values = [
            [float(row["T_recover_windows"]) for row in events
             if float(row["unknown_fraction"]) == fraction and str(row["mode"]) == mode
             and row.get("T_recover_windows") is not None]
            for fraction in fractions
        ]
        censored = [
            sum(_as_bool(row["recovery_censored"]) for row in events
                if float(row["unknown_fraction"]) == fraction and str(row["mode"]) == mode)
            for fraction in fractions
        ]
        axes[1, 0].scatter(
            np.repeat(positions + offset, [len(item) for item in values]),
            [value for item in values for value in item],
            s=16, alpha=0.55, color=color, label=mode,
        )
        for index, item in enumerate(values):
            if item:
                axes[1, 0].scatter(positions[index] + offset, float(np.mean(item)),
                                   marker="_", s=380, color=color, linewidths=2.2)
            if censored[index]:
                axes[1, 0].annotate(f"+{censored[index]} censored",
                                    (positions[index] + offset, 0), fontsize=6,
                                    ha="center", va="bottom", color=color)
    axes[1, 0].set(
        xticks=positions, xticklabels=[f"{value:g}" for value in fractions],
        xlabel="Unknown-policy fraction", ylabel="$T_{recover}$ (windows)",
        title="C  Recovery time after aggregate recalibration",
    )
    axes[1, 0].legend(fontsize=8)

    if recovery:
        for fraction, color in zip(fractions, (COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"])):
            subset = [row for row in recovery if float(row["unknown_fraction"]) == fraction]
            if not subset:
                continue
            by_samples: dict[int, list[float]] = {}
            for row in subset:
                by_samples.setdefault(int(row["samples_used"]), []).append(float(row["recovered_share"]))
            ordered = sorted(by_samples)
            axes[1, 1].plot(
                ordered, [float(np.mean(by_samples[key])) for key in ordered],
                "o-", markersize=3, color=color, label=f"unknown fraction {fraction:g}",
            )
        axes[1, 1].axhline(0.90, linestyle="--", color=COLORS["gray"], label="$G=0.90$")
        axes[1, 1].set(
            xlabel="Aggregate samples used by recalibration", ylabel="Recovered share $G(m)$",
            title="D  Sample efficiency of aggregate-only recovery",
        )
        axes[1, 1].legend(fontsize=7)
    fig.savefig(figures / "figure_E4_controller_drift.png", dpi=180)
    plt.close(fig)


def _short_dataset(name: str) -> str:
    return (
        name.replace("_building_data_genome", "")
        .replace("_smart_meters", "")
        .replace("european_lv_", "lv_")
        .replace("smart_grid_smart_city", "sgsc")
        .replace("complete_energy_community", "energy_community")
    )


def _plot_e6(summary: list[dict[str, Any]], path: Path) -> None:
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in summary))
    participations = sorted({float(row["participation"]) for row in summary})
    structures = sorted({str(row["structure"]) for row in summary})
    community = "community" if "community" in structures else structures[-1]

    def grid(metric: str) -> np.ndarray:
        return np.asarray([
            [
                next(
                    float(row[metric]) for row in summary
                    if row["dataset"] == dataset
                    and float(row["participation"]) == participation
                    and row["structure"] == community
                )
                for participation in participations
            ]
            for dataset in datasets
        ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    labels = [_short_dataset(dataset) for dataset in datasets]
    for ax, values, title, colorbar in (
        (axes[0, 0], grid("N_eff_behavior"), "A  Community-mask effective N", "$N^{beh}_{eff}$"),
        (axes[0, 1], grid("mean_nrmse"), "B  Community-mask tracking error", "NRMSE"),
    ):
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set(
            xticks=range(len(participations)),
            xticklabels=[f"{value:.1f}" for value in participations],
            yticks=range(len(labels)),
            yticklabels=labels,
            xlabel="Participation",
            title=title,
        )
        ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(image, ax=ax, label=colorbar)

    for structure in structures:
        medians: list[float] = []
        low: list[float] = []
        high: list[float] = []
        for participation in participations:
            values = np.asarray([
                float(row["mean_nrmse"]) for row in summary
                if row["structure"] == structure
                and float(row["participation"]) == participation
            ])
            medians.append(float(np.median(values)))
            low.append(float(np.percentile(values, 25)))
            high.append(float(np.percentile(values, 75)))
        axes[1, 0].plot(participations, medians, "o-", label=structure)
        axes[1, 0].fill_between(participations, low, high, alpha=0.12)
    axes[1, 0].axhline(0.10, linestyle="--", color=COLORS["gray"], linewidth=1)
    axes[1, 0].set(
        xlabel="Participation",
        ylabel="Cross-dataset NRMSE",
        title="C  Median and interquartile range",
    )
    axes[1, 0].legend()

    selected_p = min(participations, key=lambda value: abs(value - 0.7))
    selected = [
        next(
            row for row in summary
            if row["dataset"] == dataset
            and float(row["participation"]) == selected_p
            and row["structure"] == community
        )
        for dataset in datasets
    ]
    positions = np.arange(len(datasets))
    axes[1, 1].barh(
        positions,
        [float(row["reserve_kw_q95"]) for row in selected],
        color=COLORS["orange"],
    )
    axes[1, 1].set(
        yticks=positions,
        yticklabels=[f"{label} (N={int(row['N'])})" for label, row in zip(labels, selected)],
        xlabel="Reserve requirement (kW, 95th percentile)",
        title=f"D  Community mask at participation={selected_p:.1f}",
    )
    axes[1, 1].tick_params(axis="y", labelsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_e7(scaling: list[dict[str, Any]], datasets: list[dict[str, Any]], path: Path) -> None:
    completed = [row for row in datasets if row.get("status") == "completed"]
    ordered_datasets = list(dict.fromkeys(str(row["dataset"]) for row in completed))
    labels = [_short_dataset(dataset) for dataset in ordered_datasets]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    positions = np.arange(len(completed))
    beta = np.asarray([row["beta"] for row in completed])
    lower = beta - np.asarray([row["beta_ci_lower"] for row in completed])
    upper = np.asarray([row["beta_ci_upper"] for row in completed]) - beta
    axes[0, 0].errorbar(beta, positions, xerr=np.vstack([lower, upper]), fmt="o", color=COLORS["blue"])
    axes[0, 0].axvline(-0.5, linestyle="--", color=COLORS["gray"])
    axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(
        yticks=positions,
        yticklabels=[f"{_short_dataset(str(row['dataset']))} / {row['direction']}" for row in completed],
        xlabel="Log-log slope beta",
        title="A  Source-unique CV scaling",
    )
    axes[0, 0].tick_params(axis="y", labelsize=6)

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in ordered_datasets:
        for direction in ("charge", "discharge"):
            rows = [
                row for row in scaling
                if row["dataset"] == dataset and row["direction"] == direction
            ]
            selected[(dataset, direction)] = min(
                rows, key=lambda row: (abs(int(row["N"]) - 500), -int(row["N"]))
            )
    x = np.arange(len(ordered_datasets))
    width = 0.36
    for offset, direction, color in (
        (-width / 2, "charge", COLORS["blue"]),
        (width / 2, "discharge", COLORS["orange"]),
    ):
        rows = [selected[(dataset, direction)] for dataset in ordered_datasets]
        axes[0, 1].bar(x + offset, [row["cv"] for row in rows], width, color=color, label=direction)
        axes[1, 0].bar(
            x + offset,
            [abs(float(row["mean_response_kw"])) / int(row["N"]) for row in rows],
            width,
            color=color,
            label=direction,
        )
    n_labels = [
        str(int(selected[(dataset, "charge")]["N"])) for dataset in ordered_datasets
    ]
    axes[0, 1].set(
        xticks=x,
        xticklabels=[f"{label}\nN={n}" for label, n in zip(labels, n_labels)],
        ylabel="CV",
        title="B  CV at nearest legal N to 500",
    )
    axes[1, 0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Absolute mean response (kW/resource)",
        title="C  Unit response at the same N",
    )
    for ax in (axes[0, 1], axes[1, 0]):
        ax.tick_params(axis="x", rotation=55, labelsize=7)
        ax.legend(fontsize=8)

    headroom: list[float] = []
    reached: list[bool] = []
    for dataset in ordered_datasets:
        direction_rows = [
            row for row in completed if row["dataset"] == dataset
        ]
        ratios = [
            float(row["maximum_unique_fleet"]) / float(row["N_90"])
            for row in direction_rows if row["N_90"] not in (None, "", "None")
        ]
        headroom.append(min(ratios) if ratios else 0.0)
        reached.append(bool(ratios))
    colors = [COLORS["green"] if value else COLORS["gray"] for value in reached]
    axes[1, 1].bar(x, headroom, color=colors)
    for index, value in enumerate(reached):
        if not value:
            axes[1, 1].text(index, 0.02, "NR", ha="center", va="bottom", fontsize=7)
    axes[1, 1].axhline(1.0, linestyle="--", color=COLORS["gray"], linewidth=1)
    axes[1, 1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="maximum unique fleet / N90",
        title="D  Conservative headroom across directions",
    )
    axes[1, 1].tick_params(axis="x", rotation=55, labelsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_e10(raw: list[dict[str, Any]], summary: list[dict[str, Any]], path: Path) -> None:
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in summary))
    scenarios = list(dict.fromkeys(
        (str(row["mode"]), str(row["layout"]), str(row["control"])) for row in summary
    ))
    scenario_labels = [
        f"{mode.replace('_correlation', '')[:6]}\n{layout.replace('_concentrated', '')[:7]}\n{control}"
        for mode, layout, control in scenarios
    ]

    def grid(metric: str) -> np.ndarray:
        return np.asarray([
            [
                next(
                    float(row[metric]) for row in summary
                    if row["dataset"] == dataset
                    and (row["mode"], row["layout"], row["control"]) == scenario
                )
                for scenario in scenarios
            ]
            for dataset in datasets
        ])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    labels = [_short_dataset(dataset) for dataset in datasets]
    for ax, values, title, colorbar, vmin, vmax in (
        (axes[0, 0], grid("network_acceptance_ratio"), "A  Safe response delivery", "$A_{net}$", 0.0, 1.0),
        (axes[0, 1], grid("clipping_index"), "B  Network clipping", "$I_{clip}$", 0.0, None),
    ):
        image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set(
            xticks=range(len(scenarios)),
            xticklabels=scenario_labels,
            yticks=range(len(datasets)),
            yticklabels=labels,
            title=title,
        )
        ax.tick_params(axis="x", rotation=55, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(image, ax=ax, label=colorbar)

    axes[1, 0].scatter(
        [row["clipping_index"] for row in raw],
        [
            min(float(row["voltage_margin"]), float(row["line_margin"]), float(row["transformer_margin"]))
            for row in raw
        ],
        c=[0 if row["mode"] == "weak_correlation" else 1 for row in raw],
        cmap="coolwarm",
        alpha=0.45,
    )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set(xlabel="$I_{clip}$", ylabel="Minimum safety margin", title="C  Safety-delivery tradeoff")

    gains: list[float] = []
    for dataset in datasets:
        paired: list[float] = []
        for mode in sorted({row["mode"] for row in summary}):
            for layout in sorted({row["layout"] for row in summary}):
                global_row = next(
                    row for row in summary
                    if row["dataset"] == dataset and row["mode"] == mode
                    and row["layout"] == layout and row["control"] == "global"
                )
                zonal_row = next(
                    row for row in summary
                    if row["dataset"] == dataset and row["mode"] == mode
                    and row["layout"] == layout and row["control"] == "zonal"
                )
                paired.append(
                    float(zonal_row["network_acceptance_ratio"])
                    - float(global_row["network_acceptance_ratio"])
                )
        gains.append(float(np.mean(paired)))
    axes[1, 1].bar(range(len(gains)), gains, color=COLORS["green"])
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set(
        xticks=range(len(gains)),
        xticklabels=labels,
        ylabel="$\\Delta A_{net}$",
        title="D  Mean zonal minus global delivery",
    )
    axes[1, 1].tick_params(axis="x", rotation=55, labelsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_e11(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    behavior = [row for row in rows if row["failure_mode"] == "behavior_unavailability"]
    datasets = sorted({row["dataset"] for row in behavior})
    for dataset in datasets:
        selected = sorted(
            [row for row in behavior if row["dataset"] == dataset],
            key=lambda row: row["strength"],
        )
        axes[0].plot(
            [row["strength"] for row in selected],
            [row["failure_probability"] for row in selected],
            color=COLORS["blue"], alpha=0.22, linewidth=1,
        )
    strengths = sorted({float(row["strength"]) for row in behavior})
    means = [
        float(np.mean([
            row["failure_probability"] for row in behavior if row["strength"] == strength
        ]))
        for strength in strengths
    ]
    axes[0].plot(strengths, means, "o-", color="black", linewidth=2, label="cross-dataset mean")
    axes[0].axhline(0.5, linestyle="--", color=COLORS["gray"])
    axes[0].set(
        xlabel="Unavailable fraction",
        ylabel="Failure probability",
        ylim=(-0.03, 1.03),
        title="A  Behavior unavailability (numeric dose)",
    )
    axes[0].legend()

    network = [row for row in rows if row["failure_mode"] == "network_stress"]
    categories = sorted({row["stress_label"] for row in network})
    values = [
        [float(row["failure_probability"]) for row in network if row["stress_label"] == label]
        for label in categories
    ]
    axes[1].boxplot(values, showfliers=False)
    for index, category_values in enumerate(values, 1):
        jitter = np.linspace(-0.12, 0.12, len(category_values))
        axes[1].scatter(index + jitter, category_values, s=12, alpha=0.45, color=COLORS["orange"])
    axes[1].axhline(0.5, linestyle="--", color=COLORS["gray"])
    axes[1].set(
        xticks=range(1, len(categories) + 1),
        xticklabels=[label.replace("/", "\n") for label in categories],
        ylabel="Failure probability",
        ylim=(-0.03, 1.03),
        title="B  Network scenarios (categorical, no false common axis)",
    )
    axes[1].tick_params(axis="x", rotation=45, labelsize=6)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_e12(
    rows: list[dict[str, Any]], fleet: list[dict[str, Any]], path: Path
) -> None:
    datasets = [row["dataset"] for row in fleet]
    short = [
        name.replace("_building_data_genome", "").replace("_smart_meters", "")
        .replace("european_lv_", "lv_").replace("smart_grid_smart_city", "sgsc")
        for name in datasets
    ]
    x = np.arange(len(datasets))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    width = 0.36
    for offset, mode, color in (
        (-width / 2, "network_stress", COLORS["orange"]),
        (width / 2, "weak_correlation", COLORS["blue"]),
    ):
        selected = [next(row for row in rows if row["dataset"] == dataset and row["mode"] == mode) for dataset in datasets]
        axes[0, 0].bar(
            x + offset,
            [100.0 * float(row["completeness_score"]) for row in selected],
            width,
            color=color,
            label=mode,
        )
    axes[0, 0].axhline(100, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[0, 0].set(
        xticks=x, xticklabels=short, ylabel="Completeness score (%)",
        ylim=(0, 108), title="A  Base Figure 2-5 artifact completeness",
    )
    axes[0, 0].tick_params(axis="x", rotation=55, labelsize=7)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(
        x - width / 2,
        [row["balanced_equal_limit"] for row in fleet],
        width,
        color=COLORS["green"],
        label="Balanced unique limit",
    )
    axes[0, 1].bar(
        x + width / 2,
        [row["requested_main_resources"] for row in fleet],
        width,
        color=COLORS["purple"],
        label="Requested main N",
    )
    axes[0, 1].scatter(
        x,
        [row["selected_main_resources"] for row in fleet],
        marker="_", s=130, linewidth=2.5, color="black", label="Selected main N",
    )
    axes[0, 1].set(
        xticks=x, xticklabels=short, ylabel="Independent resources",
        title="B  Fleet request versus balanced source capacity",
    )
    axes[0, 1].tick_params(axis="x", rotation=55, labelsize=7)
    axes[0, 1].legend(fontsize=7)

    for offset, mode, color, marker in (
        (-0.08, "network_stress", COLORS["orange"], "s"),
        (0.08, "weak_correlation", COLORS["blue"], "o"),
    ):
        selected = [next(row for row in rows if row["dataset"] == dataset and row["mode"] == mode) for dataset in datasets]
        valid = np.asarray([bool(row["valid_base_mode"]) for row in selected])
        values = np.asarray(
            [row["validation_charge_fraction"] for row in selected], dtype=float
        )
        axes[1, 0].scatter(
            x[valid] + offset, values[valid], marker=marker, color=color,
            s=45, label=mode,
        )
        axes[1, 0].scatter(
            x[~valid] + offset, values[~valid], marker="x", color="#b2182b", s=55,
        )
    axes[1, 0].axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[1, 0].set(
        xticks=x, xticklabels=short, ylabel="Charge fraction among nonzero responses",
        ylim=(-0.04, 1.04), title="C  Bidirectional validation coverage",
    )
    axes[1, 0].tick_params(axis="x", rotation=55, labelsize=7)
    axes[1, 0].legend(fontsize=8)

    provenance_colors = {
        "counterfactual_load_shape": COLORS["gray"],
        "measured_or_counterfactual": COLORS["green"],
    }
    for mode, marker in (("network_stress", "s"), ("weak_correlation", "o")):
        selected = [next(row for row in rows if row["dataset"] == dataset and row["mode"] == mode) for dataset in datasets]
        for index, row in enumerate(selected):
            valid = bool(row["valid_base_mode"])
            axes[1, 1].scatter(
                index,
                row["curtailment_reduction_pct"],
                marker=marker if valid else "x",
                color=(
                    provenance_colors.get(row["input_provenance"], COLORS["purple"])
                    if valid else "#b2182b"
                ),
                edgecolor="black" if valid and mode == "network_stress" else None,
                s=55,
            )
    axes[1, 1].axhline(0, color=COLORS["gray"], linewidth=1)
    axes[1, 1].set(
        xticks=x, xticklabels=short, ylabel="Curtailment reduction (%)",
        title="D  Figure 4 energy-accounting effect",
    )
    axes[1, 1].tick_params(axis="x", rotation=55, labelsize=7)
    axes[1, 1].text(
        0.02, 0.98,
        "square=edge: stress; circle: weak; red x: invalid\ngreen: measured/mixed; gray: counterfactual",
        transform=axes[1, 1].transAxes, va="top", fontsize=7,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _scaling_cell(
    dataset: str,
    config: dict[str, Any],
    records: list[DeviceDay],
    profile_steps: list[int],
    schedule: list[list[tuple[int, int]]],
    common: dict[str, Any],
    *,
    seed: int,
    train_count: int,
    test_count: int,
    broadcast_columns: int,
    steps_per_day: int,
    policy_factory: Any = None,
    tags: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    tags = tags or {}
    n = len(records)
    coupled_availability = availability_at_steps(
        rank_availability(diurnal_residual(records, steps_per_day)), profile_steps
    )
    arms = [
        ("decoupled", decoupled_copy(coupled_availability, seed + 7919)),
        ("data_coupled", coupled_availability),
    ]
    summaries: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    trials: dict[str, list[float]] = {}
    threshold = float(common["controllable_probability"])
    for arm_label, availability in arms:
        policies: list[Any] = []
        aggregate, traces, resource, _ = _run_snapshots(
            records,
            config,
            profile_steps,
            schedule,
            replications=train_count + test_count,
            seed=seed,
            zone_correlation=0.0,
            keep_resources=True,
            availability_factory=lambda replication, table=availability: table,
            policy_factory=policy_factory,
            policy_sink=policies,
        )
        _, frozen_prediction = _fit_frozen_prediction(
            trace_features(traces[0]), aggregate[:train_count]
        )
        if frozen_prediction is None or resource is None:
            raise RuntimeError(f"{dataset}: calibration did not produce a frozen prediction")
        trials[arm_label] = np.mean(aggregate[train_count:], axis=1).tolist()
        measured_rho = broadcast_residual_rho(
            resource[train_count:], trace_features(traces[0]), broadcast_columns
        )
        per_seed: list[dict[str, Any]] = []
        for test_index in range(test_count):
            trace = traces[train_count + test_index]
            metrics = _success_metrics(
                aggregate[train_count + test_index], frozen_prediction,
                np.asarray(trace.network_scale, dtype=float), common,
            )
            row = {
                "dataset": dataset, **tags, "N": n, "arm": arm_label,
                "rho_measured": measured_rho, "seed_index": test_index, **metrics,
            }
            raw_rows.append(row)
            per_seed.append(row)
        successes = sum(bool(row["controllable"]) for row in per_seed)
        lower, upper = wilson_interval(successes, test_count)
        condition_means = np.mean(aggregate[train_count:], axis=0)
        condition_stds = np.std(aggregate[train_count:], axis=0)
        summaries.append({
            "dataset": dataset, **tags, "N": n, "arm": arm_label,
            "rho_measured": measured_rho,
            "N_eff": effective_n(n, measured_rho),
            "successes": successes,
            "replications": test_count,
            "p_controllable": successes / test_count,
            "p_controllable_ci_lower": lower,
            "p_controllable_ci_upper": upper,
            "mean_nrmse": float(np.mean([row["nrmse"] for row in per_seed])),
            "condition_cv": float(np.mean(
                condition_stds / np.maximum(np.abs(condition_means), 1e-12)
            )),
            "response_fraction": (
                float(np.mean([policy.response_fraction for policy in policies]))
                if policies else None
            ),
            "meets_threshold": bool(lower >= threshold),
        })
    return summaries, raw_rows, trials


def run_e13(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E13_control_logic_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    seed = int(protocol["random_seed"]) + 13000
    common = protocol["common"]
    settings = protocol["e13"]
    train_count = int(settings.get("train_replications", common["train_replications"]))
    test_count = int(settings.get("test_replications", common["test_replications"]))
    datasets = [str(value) for value in settings["datasets"]]
    logics = [str(value) for value in settings.get("logics", CONTROL_LOGICS)]

    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E13", protocol, datasets)
        _write_rows(raw_dir / "seed_metrics.csv", merged["raw/seed_metrics.csv"])
        _write_rows(output / "summary.csv", merged["summary.csv"])
        _write_rows(output / "neff_boundaries.csv", merged["neff_boundaries.csv"])
        _write_rows(output / "dataset_summary.csv", merged["dataset_summary.csv"])
        comparison = _e13_comparison(merged["summary.csv"], merged["dataset_summary.csv"])
        _write_json(output / "logic_comparison.json", comparison)
        completed = sum(row.get("status") == "completed" for row in merged["dataset_summary.csv"])
        result = {
            "status": "completed" if completed == len(datasets) * len(logics) else "partial",
            "completed_datasets": completed,
            "requested_datasets": len(datasets) * len(logics),
            "logics": logics,
            "primary_estimands": ["p_ctrl", "N_star_eff", "beta", "response_fraction"],
            "logic_comparison": comparison,
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    threshold = float(common["controllable_probability"])
    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    neff_boundaries: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        config_path, network_protocol = _dataset_config(dataset)
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        config["control"]["network_feedback"] = False
        pool = load_device_day_pool(config)
        configured = int(config["population"]["N_simulated_resources"])
        maximum = min(configured, _balanced_unique_capacity(pool))
        n_values = _legal_n_values(
            [int(value) for value in settings["n_candidates"]], maximum,
            include_maximum=bool(settings.get("include_maximum_unique_fleet", True)),
        )
        if len(n_values) < 2:
            dataset_rows.append({
                "dataset": dataset, "status": "insufficient_balanced_unique_sources",
                "maximum_unique_fleet": maximum,
            })
            continue
        master = _sample_master(config, max(n_values), seed)
        profile_steps, schedule = _profile_and_schedule(
            master, config, int(settings.get("conditions", common["conditions"])), seed + 100
        )
        steps_per_day = int(config["simulation"]["steps_per_day"])
        broadcast_columns = int(settings.get("broadcast_basis_columns", 9))
        soc_low = float(settings.get("initial_soc_low", 0.20))
        soc_high = float(settings.get("initial_soc_high", 0.80))
        for logic in logics:
            trials_by_arm: dict[str, dict[int, list[float]]] = {"decoupled": {}, "data_coupled": {}}
            for n_index, n in enumerate(n_values):
                records, soc_diagnostics = spread_initial_soc(
                    _balanced_subset(master, n),
                    low=soc_low, high=soc_high, seed=seed + n_index * 31,
                )
                cell_seed = seed + n_index * 10000
                policy_factory = None
                if logic != "open_loop":
                    policy_factory = (
                        lambda replication, recs=records, index=n_index, name=logic: build_policy(
                            name, recs, seed=seed + index * 7919 + replication,
                            steps=len(profile_steps),
                        )
                    )
                cells, rows, trials = _scaling_cell(
                    dataset, config, records, profile_steps, schedule, common,
                    seed=cell_seed, train_count=train_count, test_count=test_count,
                    broadcast_columns=broadcast_columns, steps_per_day=steps_per_day,
                    policy_factory=policy_factory,
                    tags={"logic": logic},
                )
                for cell in cells:
                    cell.update(soc_diagnostics)
                summaries.extend(cells)
                raw_rows.extend(rows)
                for arm_label, values in trials.items():
                    trials_by_arm[arm_label][n] = values
            scoped = [
                row for row in summaries
                if row["dataset"] == dataset and row["logic"] == logic
            ]
            boundary = _annotate_effective_margin(dataset, scoped, threshold, arm="data_coupled")
            boundary["logic"] = logic
            neff_boundaries.append(boundary)
            slopes: dict[str, Any] = {}
            for arm_label in ("decoupled", "data_coupled"):
                rows = sorted(
                    [row for row in scoped if row["arm"] == arm_label],
                    key=lambda item: item["N"],
                )
                slopes[f"beta_{arm_label}"] = loglog_slope(
                    [row["N"] for row in rows], [row["condition_cv"] for row in rows]
                )
                lower, upper = bootstrap_slope_ci(
                    trials_by_arm[arm_label], seed=seed + 333,
                    samples=int(common["bootstrap_samples"]),
                )
                slopes[f"beta_{arm_label}_ci_lower"] = lower
                slopes[f"beta_{arm_label}_ci_upper"] = upper
            coupled = [row for row in scoped if row["arm"] == "data_coupled"]
            dataset_rows.append({
                "dataset": dataset, "logic": logic, "status": "completed",
                "maximum_unique_fleet": maximum,
                "N_values": ";".join(str(value) for value in n_values),
                "N_star_eff": boundary.get("N_star_eff"),
                "N_eff_over_N_median": float(np.median([
                    row["N_eff"] / row["N"] for row in coupled
                ])) if coupled else None,
                "response_fraction_mean": (
                    float(np.mean(_responses)) if (_responses := [
                        row["response_fraction"] for row in coupled
                        if row.get("response_fraction") is not None
                    ]) else None
                ),
                "A_ctrl": float(np.mean([
                    row["p_controllable_ci_lower"] >= threshold for row in coupled
                ])) if coupled else None,
                **slopes,
                "network_protocol": network_protocol,
            })

    _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
    _write_rows(output / "summary.csv", summaries)
    _write_rows(output / "neff_boundaries.csv", neff_boundaries)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    comparison = _e13_comparison(summaries, dataset_rows)
    _write_json(output / "logic_comparison.json", comparison)
    completed = sum(row.get("status") == "completed" for row in dataset_rows)
    result = {
        "status": "completed" if completed == len(datasets) * len(logics) else "partial",
        "completed_datasets": completed,
        "requested_datasets": len(datasets) * len(logics),
        "logics": logics,
        "primary_estimands": ["p_ctrl", "N_star_eff", "beta", "response_fraction"],
        "logic_comparison": comparison,
        "raw_seed_rows": len(raw_rows),
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _e13_comparison(
    summaries: list[dict[str, Any]], dataset_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    logics = sorted({str(row["logic"]) for row in summaries if row.get("logic")})
    table: dict[str, Any] = {}
    for logic in logics:
        coupled = [
            row for row in summaries
            if row.get("logic") == logic and row["arm"] == "data_coupled"
        ]
        rows = [row for row in dataset_rows if row.get("logic") == logic
                and row.get("status") == "completed"]
        stars = [row["N_star_eff"] for row in rows if row.get("N_star_eff")]
        betas = [row["beta_data_coupled"] for row in rows
                 if row.get("beta_data_coupled") is not None]
        table[logic] = {
            "datasets_completed": len(rows),
            "datasets_reaching_boundary": len(stars),
            "N_star_eff_median": float(np.median(stars)) if stars else None,
            "beta_data_coupled_median": float(np.median(betas)) if betas else None,
            "N_eff_over_N_median": float(np.median([
                row["N_eff"] / row["N"] for row in coupled
            ])) if coupled else None,
            "response_fraction_mean": float(np.mean([
                row["response_fraction"] for row in coupled
                if row.get("response_fraction") is not None
            ])) if coupled else None,
        }
    return {
        "definition": (
            "per control logic: median N*_eff, beta, N_eff/N over datasets. "
            "response_fraction is reported so a logic that simply does less work "
            "is not mistaken for a logic that scales worse."
        ),
        "by_logic": table,
    }


def run_e14(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E14_dominant_capacity_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    seed = int(protocol["random_seed"]) + 14000
    common = protocol["common"]
    settings = protocol["e14"]
    train_count = int(settings.get("train_replications", common["train_replications"]))
    test_count = int(settings.get("test_replications", common["test_replications"]))
    datasets = [str(value) for value in settings["datasets"]]
    injections = [
        (int(cell["count"]), float(cell["ratio"]))
        for cell in settings["injections"]
    ]

    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E14", protocol, datasets)
        _write_rows(raw_dir / "seed_metrics.csv", merged["raw/seed_metrics.csv"])
        _write_rows(output / "summary.csv", merged["summary.csv"])
        _write_rows(output / "dataset_summary.csv", merged["dataset_summary.csv"])
        lindeberg = _e14_lindeberg_summary(merged["dataset_summary.csv"])
        collapse = _e14_axis_collapse(merged["summary.csv"])
        _write_json(output / "lindeberg_summary.json", lindeberg)
        _write_json(output / "axis_collapse.json", collapse)
        completed = sum(row.get("status") == "completed" for row in merged["dataset_summary.csv"])
        result = {
            "status": "completed" if completed == len(datasets) * len(injections) else "partial",
            "completed_datasets": completed,
            "requested_datasets": len(datasets) * len(injections),
            "injections": [{"count": c, "ratio": r} for c, r in injections],
            "primary_estimands": [
                "beta_vs_N", "beta_vs_N_capacity_eff", "lindeberg_share", "p_ctrl"
            ],
            "lindeberg_summary": lindeberg,
            "axis_collapse": collapse,
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    threshold = float(common["controllable_probability"])
    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        config_path, network_protocol = _dataset_config(dataset)
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        config["control"]["network_feedback"] = False
        pool = load_device_day_pool(config)
        configured = int(config["population"]["N_simulated_resources"])
        maximum = min(configured, _balanced_unique_capacity(pool))
        n_values = _legal_n_values(
            [int(value) for value in settings["n_candidates"]], maximum,
            include_maximum=bool(settings.get("include_maximum_unique_fleet", True)),
        )
        if len(n_values) < 2:
            dataset_rows.append({
                "dataset": dataset, "status": "insufficient_balanced_unique_sources",
                "maximum_unique_fleet": maximum,
            })
            continue
        master = _sample_master(config, max(n_values), seed)
        profile_steps, schedule = _profile_and_schedule(
            master, config, int(settings.get("conditions", common["conditions"])), seed + 100
        )
        steps_per_day = int(config["simulation"]["steps_per_day"])
        broadcast_columns = int(settings.get("broadcast_basis_columns", 9))
        for count, ratio in injections:
            tag = {"dominant_count": count, "capacity_ratio": ratio}
            for n_index, n in enumerate(n_values):
                plain = _balanced_subset(master, n)
                if count > n:
                    continue
                records, chosen = inject_dominant(
                    plain, count=count, ratio=ratio, seed=seed + n_index * 101
                )
                diagnostics = imbalance_diagnostics(records)
                cells, rows, _ = _scaling_cell(
                    dataset, config, records, profile_steps, schedule, common,
                    seed=seed + n_index * 10000, train_count=train_count,
                    test_count=test_count, broadcast_columns=broadcast_columns,
                    steps_per_day=steps_per_day, tags=tag,
                )
                for cell in cells:
                    cell.update({
                        "N_capacity_eff": diagnostics["N_capacity_eff"],
                        "lindeberg_share_capacity": diagnostics["lindeberg_share_capacity"],
                        "capacity_count_ratio": diagnostics["capacity_count_ratio"],
                        "capacity_total_kwh": diagnostics["capacity_total_kwh"],
                        "capacity_max_kwh": diagnostics["capacity_max_kwh"],
                        "promoted_indices": ";".join(str(value) for value in chosen),
                    })
                summaries.extend(cells)
                raw_rows.extend(rows)
            scoped = [
                row for row in summaries
                if row["dataset"] == dataset
                and row["dominant_count"] == count and row["capacity_ratio"] == ratio
            ]
            entry: dict[str, Any] = {
                "dataset": dataset, "dominant_count": count, "capacity_ratio": ratio,
                "status": "completed", "maximum_unique_fleet": maximum,
                "network_protocol": network_protocol,
            }
            for arm_label in ("decoupled", "data_coupled"):
                rows = sorted(
                    [row for row in scoped if row["arm"] == arm_label],
                    key=lambda item: item["N"],
                )
                if len(rows) < 2:
                    continue
                cv = [row["condition_cv"] for row in rows]
                entry[f"beta_{arm_label}_vs_N"] = loglog_slope(
                    [row["N"] for row in rows], cv
                )
                entry[f"beta_{arm_label}_vs_capacity_eff"] = loglog_slope(
                    [row["N_capacity_eff"] for row in rows], cv
                )
                entry[f"cells_{arm_label}"] = len(rows)
            coupled = [row for row in scoped if row["arm"] == "data_coupled"]
            if coupled:
                entry.update({
                    "lindeberg_share_max": max(
                        row["lindeberg_share_capacity"] for row in coupled
                    ),
                    "capacity_count_ratio_min": min(
                        row["capacity_count_ratio"] for row in coupled
                    ),
                    "A_ctrl": float(np.mean([
                        row["p_controllable_ci_lower"] >= threshold for row in coupled
                    ])),
                    "N_eff_over_N_median": float(np.median([
                        row["N_eff"] / row["N"] for row in coupled
                    ])),
                })
            dataset_rows.append(entry)

    _write_rows(raw_dir / "seed_metrics.csv", raw_rows)
    _write_rows(output / "summary.csv", summaries)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    lindeberg = _e14_lindeberg_summary(dataset_rows)
    collapse = _e14_axis_collapse(summaries)
    _write_json(output / "lindeberg_summary.json", lindeberg)
    _write_json(output / "axis_collapse.json", collapse)
    completed = sum(row.get("status") == "completed" for row in dataset_rows)
    result = {
        "status": "completed" if completed == len(datasets) * len(injections) else "partial",
        "completed_datasets": completed,
        "requested_datasets": len(datasets) * len(injections),
        "injections": [{"count": c, "ratio": r} for c, r in injections],
        "primary_estimands": [
            "beta_vs_N", "beta_vs_N_capacity_eff", "lindeberg_share", "p_ctrl"
        ],
        "lindeberg_summary": lindeberg,
        "axis_collapse": collapse,
        "raw_seed_rows": len(raw_rows),
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _binned_between_group_variance(
    rows: list[dict[str, Any]], axis_key: str, *, group_key: str, value_key: str, bins: int
) -> dict[str, Any]:
    usable = [
        row for row in rows
        if _numeric(row, axis_key) is not None and _numeric(row, axis_key) > 0
        and _numeric(row, value_key) is not None
        and row.get(group_key) not in (None, "")
    ]
    if len(usable) < bins * 2:
        return {"bins": 0, "mean_between_group_variance": None, "cells": len(usable)}
    values = np.log10(np.asarray([_numeric(row, axis_key) for row in usable]))
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges[-1] += 1e-9
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bins - 1)
    variances: list[float] = []
    spreads: list[float] = []
    for bin_index in range(bins):
        members = np.flatnonzero(index == bin_index)
        by_group: dict[str, list[float]] = {}
        for position in members:
            by_group.setdefault(
                str(usable[position][group_key]), []
            ).append(float(_numeric(usable[position], value_key)))
        if len(by_group) < 2:
            continue
        variances.append(float(np.var([float(np.mean(v)) for v in by_group.values()])))
        spreads.append(float(np.std(values[members])))
    if not variances:
        return {"bins": 0, "mean_between_group_variance": None, "cells": len(usable)}
    return {
        "bins_requested": bins,
        "bins_with_at_least_two_groups": len(variances),
        "mean_between_group_variance": float(np.mean(variances)),
        "mean_within_bin_log10_spread": float(np.mean(spreads)),
        "cells": len(usable),
    }


def _e14_axis_collapse(summaries: list[dict[str, Any]], *, bins: int = 6) -> dict[str, Any]:
    coupled = [row for row in summaries if str(row.get("arm", "")) == "data_coupled"]
    for row in coupled:
        row["_config"] = f"{row.get('dominant_count')}|{row.get('capacity_ratio')}"
    datasets = sorted({str(row["dataset"]) for row in coupled})
    per_dataset: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = [row for row in coupled if str(row["dataset"]) == dataset]
        physical = _binned_between_group_variance(
            rows, "N", group_key="_config", value_key="condition_cv", bins=bins
        )
        capacity = _binned_between_group_variance(
            rows, "N_capacity_eff", group_key="_config", value_key="condition_cv", bins=bins
        )
        ratio = None
        if (
            capacity["mean_between_group_variance"] is not None
            and physical["mean_between_group_variance"]
        ):
            ratio = float(
                capacity["mean_between_group_variance"]
                / physical["mean_between_group_variance"]
            )
        matched = bool(
            capacity.get("mean_within_bin_log10_spread") is not None
            and physical.get("mean_within_bin_log10_spread")
            and 0.1
            <= capacity["mean_within_bin_log10_spread"]
            / physical["mean_within_bin_log10_spread"]
            <= 10.0
        )
        per_dataset.append({
            "dataset": dataset,
            "variance_ratio_capacity_over_N": ratio,
            "capacity_axis": capacity,
            "physical_N_axis": physical,
            "comparison_is_matched": matched,
            "collapses": bool(ratio is not None and ratio < 1.0),
        })
    ratios = [row["variance_ratio_capacity_over_N"] for row in per_dataset
              if row["variance_ratio_capacity_over_N"] is not None]
    return {
        "definition": (
            "Pooling every capacity composition, the mean between-composition variance "
            "of the aggregate CV within quantile bins, computed on the capacity-weighted "
            "axis and on the physical N axis. Ratio below 1 means the capacity-weighted "
            "effective size is the coordinate the scaling law actually lives on."
        ),
        "datasets_with_ratio": len(ratios),
        "variance_ratio_median": float(np.median(ratios)) if ratios else None,
        "datasets_collapsing": sum(row["collapses"] for row in per_dataset),
        "datasets_matched": sum(row["comparison_is_matched"] for row in per_dataset),
        "by_dataset": per_dataset,
    }


def _e14_lindeberg_summary(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({
        (int(row["dominant_count"]), float(row["capacity_ratio"]))
        for row in dataset_rows
        if row.get("status") == "completed" and row.get("dominant_count") is not None
    })
    table: dict[str, Any] = {}
    for count, ratio in keys:
        rows = [
            row for row in dataset_rows
            if row.get("status") == "completed"
            and int(row.get("dominant_count", -1)) == count
            and float(row.get("capacity_ratio", -1)) == ratio
        ]

        def _median(field: str) -> float | None:
            values = [row[field] for row in rows if row.get(field) is not None]
            return float(np.median(values)) if values else None

        table[f"count={count},ratio={ratio:g}"] = {
            "datasets": len(rows),
            "beta_vs_N_median": _median("beta_data_coupled_vs_N"),
            "beta_vs_capacity_eff_median": _median("beta_data_coupled_vs_capacity_eff"),
            "beta_decoupled_vs_N_median": _median("beta_decoupled_vs_N"),
            "lindeberg_share_max_median": _median("lindeberg_share_max"),
            "capacity_count_ratio_min_median": _median("capacity_count_ratio_min"),
            "A_ctrl_median": _median("A_ctrl"),
        }
    return {
        "definition": (
            "beta is the log-log slope of the aggregate CV. Reported twice on the "
            "same cells: against the physical device count N and against the "
            "capacity-weighted effective size (sum c)^2/sum c^2. Lindeberg requires "
            "max c_i^2 / sum c_i^2 -> 0; where it does not, CLT has no reason to hold. "
            "WARNING: beta_vs_capacity_eff is NOT comparable to beta_vs_N at a fixed "
            "injection level. With a fixed dominant count the capacity axis barely moves "
            "across the N sweep, so its slope is the physical-axis slope inflated by the "
            "axis compression ratio. Use axis_collapse.json, which pools every capacity "
            "composition so both axes span the full grid."
        ),
        "by_injection": table,
    }


def _e9_day_schedule(
    records: list[DeviceDay],
    config: dict[str, Any],
    steps: int,
    mode: str,
    *,
    seed: int,
) -> list[list[tuple[int, int]]]:
    if mode == "real":
        return build_real_signal_schedule(
            records, config, list(range(steps)), seed=seed, scenario="normal"
        )
    if mode == "sustained_discharge":
        zone_count = len(config["zones"]["zones"])
        intensity = int(round(0.85 * 4095))
        return [[(10, intensity)] * zone_count for _ in range(steps)]
    raise ValueError(f"unknown E9 broadcast mode: {mode}")


def run_e9(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    output = root / "E9_long_horizon_soc_new"
    raw_dir = output / "raw"
    figures = output / "Figs"
    (raw_dir / "soc_traces").mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    seed = int(protocol["random_seed"]) + 9000
    common = protocol["common"]
    settings = protocol["e9"]
    days = int(settings.get("days", 30))
    train_days = int(settings.get("train_days", 3))
    architectures = [str(value) for value in settings.get("architectures", ARCHITECTURES)]
    modes = [str(value) for value in settings.get("broadcast_modes", ["real", "sustained_discharge"])]
    requested_resources = int(settings.get("resources", 1000))
    broadcast_columns = int(settings.get("broadcast_basis_columns", 9))
    datasets = [str(value) for value in settings["datasets"]]

    if int(protocol["execution"].get("dataset_workers", 1)) > 1 and len(datasets) > 1:
        merged = _parallel_dataset_rows("E9", protocol, datasets)
        _write_rows(output / "daily_metrics.csv", merged["daily_metrics.csv"])
        _write_rows(output / "dataset_summary.csv", merged["dataset_summary.csv"])
        _write_binary_outputs(output, merged.get("__binary__", {}))
        drift = _e9_drift_summary(merged["dataset_summary.csv"])
        _write_json(output / "soc_drift_summary.json", drift)
        completed = sum(row.get("status") == "completed" for row in merged["dataset_summary.csv"])
        result = {
            "status": "completed"
            if completed == len(datasets) * len(modes) * len(architectures) else "partial",
            "completed_cells": completed,
            "requested_cells": len(datasets) * len(modes) * len(architectures),
            "days": days, "architectures": architectures, "broadcast_modes": modes,
            "primary_estimands": [
                "soc_mean_drift", "soc_std", "saturated_fraction", "N_eff_by_day",
                "dispatch_nrmse_by_day", "self_consumption_ratio",
            ],
            "soc_drift_summary": drift,
            "completed_at": _utc_now(),
        }
        _write_json(output / "statistical_tests.json", result)
        return result

    daily_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        config_path, network_protocol = _dataset_config(dataset)
        if not config_path.exists():
            dataset_rows.append({"dataset": dataset, "status": "missing_config"})
            continue
        config = load_config(config_path)
        config["control"]["network_feedback"] = False
        pool = load_device_day_pool(config)
        configured = int(config["population"]["N_simulated_resources"])
        fleet_size = min(requested_resources, configured, _balanced_unique_capacity(pool))
        if fleet_size < 2:
            dataset_rows.append({
                "dataset": dataset, "status": "insufficient_balanced_unique_sources",
                "maximum_unique_fleet": _balanced_unique_capacity(pool),
            })
            continue
        steps_per_day = int(config["simulation"]["steps_per_day"])
        constraints = config["device_constraints"]
        soc_min = float(constraints["soc_min"])
        soc_max = float(constraints["soc_max"])
        master = _sample_master(config, fleet_size, seed)
        sequence = daily_fleet_sequence(pool, master, days, seed=seed + 7)
        availability = days_available(pool, master)
        initial_soc, soc_diagnostics = draw_initial_soc(
            fleet_size,
            low=float(settings.get("initial_soc_low", 0.30)),
            high=float(settings.get("initial_soc_high", 0.70)),
            seed=seed + 13,
        )
        for mode in modes:
            for architecture in architectures:
                recorder = SocRecorder(fleet_size, soc_min=soc_min, soc_max=soc_max)
                soc = initial_soc.copy()
                aggregates: list[np.ndarray] = []
                base_features: np.ndarray | None = None
                frozen: np.ndarray | None = None
                for day in range(days):
                    day_start_soc = float(np.mean(soc))
                    records = set_initial_soc(sequence[day], soc)
                    schedule = _e9_day_schedule(
                        records, config, steps_per_day, mode, seed=seed + day * 17
                    )
                    policy = build_architecture(
                        architecture, records, recorder=recorder,
                        soc_min=soc_min, soc_max=soc_max,
                        options=settings.get("architecture_options", {}),
                    )
                    sink: list[list[float]] = []
                    adapters: list[Any] = []
                    trace = simulate_day(
                        records, config, steps=steps_per_day, seed=seed + day * 101,
                        profile_steps=list(range(steps_per_day)),
                        signal_overrides=schedule,
                        resource_response_sink=sink,
                        response_policy=policy,
                        reset_each_step=False,
                        adapter_sink=adapters,
                    )
                    soc = np.asarray(adapters[0].battery_states(), dtype=float)
                    recorder.snapshot(soc)
                    aggregate = np.asarray(trace.accepted_control_kw, dtype=float)
                    aggregates.append(aggregate)
                    features = trace_features(trace)
                    if day == train_days - 1:
                        base_features = features
                        _, frozen = _fit_frozen_prediction(
                            features, np.asarray(aggregates[:train_days], dtype=float)
                        )
                    resource = np.asarray(sink, dtype=float)
                    measured_rho = broadcast_residual_rho(
                        resource[None, ...], features, broadcast_columns
                    )
                    responding = float(np.mean(np.abs(resource) > 1e-9))
                    throughput = float(np.mean(np.abs(resource)))
                    baseline_self = float(np.sum(trace.self_consumption_baseline_kwh))
                    eps_self = float(np.sum(trace.self_consumption_eps_kwh))
                    window = slice(day * steps_per_day, (day + 1) * steps_per_day)
                    daily_rows.append({
                        "dataset": dataset, "broadcast_mode": mode,
                        "architecture": architecture, "day": day, "N": fleet_size,
                        "soc_mean_start": day_start_soc,
                        "soc_mean": float(np.mean(soc)),
                        "soc_std": float(np.std(soc)),
                        "soc_min": float(np.min(soc)),
                        "soc_max": float(np.max(soc)),
                        "floor_fraction": float(np.mean(recorder.floor_fraction[window])),
                        "ceiling_fraction": float(np.mean(recorder.ceiling_fraction[window])),
                        "saturated_fraction": float(
                            np.mean(recorder.floor_fraction[window])
                            + np.mean(recorder.ceiling_fraction[window])
                        ),
                        "rho_measured": measured_rho,
                        "N_eff": effective_n(fleet_size, measured_rho),
                        "N_eff_over_N": effective_n(fleet_size, measured_rho) / fleet_size,
                        "responding_fraction": responding,
                        "mean_abs_device_kw": throughput,
                        "dispatch_nrmse": (
                            float(nrmse(aggregate, frozen))
                            if frozen is not None and day >= train_days else None
                        ),
                        "mean_accepted_kw": float(np.mean(aggregate)),
                        "self_consumption_ratio": (
                            eps_self / baseline_self if baseline_self > 1e-9 else None
                        ),
                        "is_training_day": bool(day < train_days),
                    })
                arrays = recorder.as_arrays()
                archive = raw_dir / "soc_traces" / dataset / f"{mode}_{architecture}.npz"
                archive.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(archive, **arrays)
                scoped = [
                    row for row in daily_rows
                    if row["dataset"] == dataset and row["broadcast_mode"] == mode
                    and row["architecture"] == architecture
                ]
                evaluated = [row for row in scoped if not row["is_training_day"]]
                first, last = scoped[0], scoped[-1]
                entry = {
                    "dataset": dataset, "broadcast_mode": mode,
                    "architecture": architecture, "status": "completed",
                    "N": fleet_size, "days": days,
                    "soc_mean_initial": float(np.mean(initial_soc)),
                    "soc_mean_day0_end": first["soc_mean"],
                    "soc_mean_end": last["soc_mean"],
                    "soc_mean_drift": last["soc_mean"] - float(np.mean(initial_soc)),
                    "soc_mean_drift_day0": first["soc_mean"] - float(np.mean(initial_soc)),
                    "soc_mean_drift_after_day0": last["soc_mean"] - first["soc_mean"],
                    "soc_std_initial": float(np.std(initial_soc)),
                    "soc_std_day0_end": first["soc_std"], "soc_std_end": last["soc_std"],
                    "soc_std_ratio": (
                        last["soc_std"] / float(np.std(initial_soc))
                        if float(np.std(initial_soc)) > 1e-9 else None
                    ),
                    "saturated_fraction_end": last["saturated_fraction"],
                    "saturated_fraction_max": max(row["saturated_fraction"] for row in scoped),
                    "N_eff_over_N_day0": first["N_eff_over_N"],
                    "N_eff_over_N_end": last["N_eff_over_N"],
                    "N_eff_retention": (
                        last["N_eff_over_N"] / first["N_eff_over_N"]
                        if first["N_eff_over_N"] > 1e-9 else None
                    ),
                    "responding_fraction_day0": first["responding_fraction"],
                    "responding_fraction_end": last["responding_fraction"],
                    "throughput_retention": (
                        last["mean_abs_device_kw"] / first["mean_abs_device_kw"]
                        if first["mean_abs_device_kw"] > 1e-12 else None
                    ),
                    "n_eff_interpretable": bool(
                        first["mean_abs_device_kw"] > 1e-12
                        and last["mean_abs_device_kw"] / first["mean_abs_device_kw"] >= 0.2
                    ),
                    "maximum_unique_fleet": _balanced_unique_capacity(pool),
                    "network_protocol": network_protocol,
                    **soc_diagnostics, **availability,
                }
                errors = [row["dispatch_nrmse"] for row in evaluated
                          if row["dispatch_nrmse"] is not None]
                if errors:
                    entry.update({
                        "dispatch_nrmse_mean": float(np.mean(errors)),
                        "dispatch_nrmse_first": errors[0],
                        "dispatch_nrmse_last": errors[-1],
                        "dispatch_nrmse_growth": (
                            errors[-1] / errors[0] if errors[0] > 1e-12 else None
                        ),
                    })
                ratios = [row["self_consumption_ratio"] for row in scoped
                          if row["self_consumption_ratio"] is not None]
                if ratios:
                    entry["self_consumption_ratio_mean"] = float(np.mean(ratios))
                dataset_rows.append(entry)

    _write_rows(output / "daily_metrics.csv", daily_rows)
    _write_rows(output / "dataset_summary.csv", dataset_rows)
    drift = _e9_drift_summary(dataset_rows)
    _write_json(output / "soc_drift_summary.json", drift)
    completed = sum(row.get("status") == "completed" for row in dataset_rows)
    result = {
        "status": "completed"
        if completed == len(datasets) * len(modes) * len(architectures) else "partial",
        "completed_cells": completed,
        "requested_cells": len(datasets) * len(modes) * len(architectures),
        "days": days, "architectures": architectures, "broadcast_modes": modes,
        "primary_estimands": [
            "soc_mean_drift", "soc_std", "saturated_fraction", "N_eff_by_day",
            "dispatch_nrmse_by_day", "self_consumption_ratio",
        ],
        "soc_drift_summary": drift,
        "completed_at": _utc_now(),
    }
    _write_json(output / "statistical_tests.json", result)
    return result


def _e9_drift_summary(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in dataset_rows if row.get("status") == "completed"]
    keys = sorted({
        (str(row["broadcast_mode"]), str(row["architecture"])) for row in rows
    })
    table: dict[str, Any] = {}
    for mode, architecture in keys:
        scoped = [
            row for row in rows
            if str(row["broadcast_mode"]) == mode and str(row["architecture"]) == architecture
        ]

        def _median(field: str) -> float | None:
            values = [_numeric(row, field) for row in scoped]
            values = [value for value in values if value is not None]
            return float(np.median(values)) if values else None

        table[f"{mode}/{architecture}"] = {
            "datasets": len(scoped),
            "soc_mean_drift_median": _median("soc_mean_drift"),
            "soc_mean_drift_day0_median": _median("soc_mean_drift_day0"),
            "soc_mean_end_median": _median("soc_mean_end"),
            "soc_std_ratio_median": _median("soc_std_ratio"),
            "saturated_fraction_end_median": _median("saturated_fraction_end"),
            "N_eff_retention_median": _median("N_eff_retention"),
            "throughput_retention_median": _median("throughput_retention"),
            "responding_fraction_end_median": _median("responding_fraction_end"),
            "cells_with_interpretable_N_eff": sum(
                1 for row in scoped if row.get("n_eff_interpretable") in (True, "True")
            ),
            "dispatch_nrmse_growth_median": _median("dispatch_nrmse_growth"),
            "self_consumption_ratio_median": _median("self_consumption_ratio_mean"),
        }
    return {
        "definition": (
            "Long-horizon closed loop with SOC carried across days. Arms differ only in "
            "what the aggregator can observe: every device SOC (full_state), the fleet "
            "mean SOC alone (aggregate_feedback), or nothing (broadcast_only). "
            "soc_mean_drift is end minus start; N_eff_retention is the last day's N_eff/N "
            "over the first day's; dispatch_nrmse_growth is last over first, so a value "
            "above one means the frozen aggregate model decays as the fleet ages. "
            "WARNING: N_eff_retention is only interpretable while the fleet still moves "
            "energy. A fleet drained to its SOC floor stops responding, the residual "
            "carries no correlation, and N_eff/N rises towards one -- a dead fleet reads "
            "as perfectly independent. Always read it next to throughput_retention and "
            "cells_with_interpretable_N_eff."
        ),
        "by_mode_and_architecture": table,
    }


RUNNERS = {
    "E1": run_e1,
    "E2": run_e2,
    "E3": run_e3,
    "E4": run_e4,
    "E6": run_e6,
    "E15": run_e15,
    "E7": run_e7,
    "E10": run_e10,
    "E11": run_e11,
    "E12": run_e12,
    "E13": run_e13,
    "E14": run_e14,
    "E9": run_e9,
}


def initialize(protocol_path: Path) -> tuple[dict[str, Any], Path]:
    protocol = _load_protocol(protocol_path)
    root = _result_root(protocol)
    root.mkdir(parents=True, exist_ok=True)
    config_dir = root / "config"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(protocol_path, config_dir / "protocol.yaml")
    unavailable = {
        "E5": "superseded by E9, whose three arms are the full-state / aggregate-feedback / broadcast-only comparison",
        "E8": "mean-field, PEM and transactive-control benchmarks are not implemented",
    }
    manifest = {
        "protocol": "nc_excel_metric_driven_supplementary_experiments",
        "created_at": _utc_now(),
        "git_revision_at_launch": _git_revision(),
        "protocol_file": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "executable_phase_1": list(protocol["execution"]["experiments"]),
        "not_executed": unavailable,
        "truthfulness_note": "Unavailable experiments are not replaced with renamed existing outputs.",
        "status": "initialized",
    }
    _write_json(root / "protocol_manifest.json", manifest)
    return protocol, root


def run_selected(protocol_path: Path, experiments: list[str]) -> dict[str, Any]:
    protocol, root = initialize(protocol_path)
    status_path = root / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["status"] = "running"
        status["resumed_at"] = _utc_now()
        status.setdefault("experiments", {})
    else:
        status = {
            "status": "running",
            "started_at": _utc_now(),
            "experiments": {},
        }
    _write_json(status_path, status)
    for experiment in experiments:
        runner = RUNNERS.get(experiment)
        if runner is None:
            raise ValueError(f"experiment is not executable in phase 1: {experiment}")
        status["experiments"][experiment] = {"status": "running", "started_at": _utc_now()}
        _write_json(status_path, status)
        started = time.time()
        try:
            result = runner(protocol, root)
        except Exception as exc:
            status["experiments"][experiment] = {
                "status": "failed",
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
            }
            status["status"] = "failed"
            _write_json(status_path, status)
            raise
        status["experiments"][experiment] = {
            **result,
            "elapsed_seconds": time.time() - started,
        }
        _write_json(status_path, status)
    status["status"] = "completed"
    status["completed_at"] = _utc_now()
    _write_json(status_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the currently executable experiments from the review-workbook protocol.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--experiments", nargs="+", choices=sorted(RUNNERS), default=None)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    protocol, root = initialize(args.protocol)
    if args.initialize_only:
        print(json.dumps({"status": "initialized", "results_root": str(root)}, ensure_ascii=False))
        return
    selected = args.experiments or list(protocol["execution"]["experiments"])
    print(json.dumps(run_selected(args.protocol, selected), ensure_ascii=False))


if __name__ == "__main__":
    main()
