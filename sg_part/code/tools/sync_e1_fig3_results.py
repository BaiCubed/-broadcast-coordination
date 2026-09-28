"""Synchronize the complete E1 response experiment into legacy Figure 3 outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / "results/e1_full/E1_scale_boundary_new"
ARCHIVE_PATTERN = re.compile(r"responses_N(\d+)_data_coupled\.npz$")
THRESHOLD = 0.95
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20260803
EPSILON = 1e-12


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator <= EPSILON:
        return None
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def _log_interpolate(left_n: float, left_r2: float, right_n: float, right_r2: float) -> float:
    if abs(right_r2 - left_r2) <= EPSILON:
        return float(right_n)
    fraction = float(np.clip((THRESHOLD - left_r2) / (right_r2 - left_r2), 0.0, 1.0))
    return float(10 ** (math.log10(left_n) + fraction * (math.log10(right_n) - math.log10(left_n))))


def _cell_metrics(path: Path, train_replications: int, bootstrap_seed: int) -> dict[str, Any]:
    with np.load(path) as payload:
        aggregate = np.asarray(payload["aggregate_response_kw"], dtype=float)
        frozen = np.asarray(payload["frozen_prediction_kw"], dtype=float)
        profile_steps = np.asarray(payload["profile_steps"], dtype=int)
    train = aggregate[:train_replications]
    test = aggregate[train_replications:]
    if train.shape[0] != train_replications or test.shape[0] < 2:
        raise ValueError(f"insufficient replications in {path}: {aggregate.shape}")

    prediction_by_condition = np.mean(train, axis=0)
    actual = test.T.reshape(-1)
    predicted = np.repeat(prediction_by_condition, test.shape[0])
    score = _r2(actual, predicted)
    if score is None:
        raise ValueError(f"zero-variance R2 denominator in {path}")

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, test.shape[1], size=test.shape[1])
        boot_actual = np.concatenate([test[:, index] for index in indices])
        boot_prediction = np.repeat(prediction_by_condition[indices], test.shape[0])
        boot_score = _r2(boot_actual, boot_prediction)
        if boot_score is not None:
            bootstrap.append(boot_score)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])

    condition_means = np.mean(test, axis=0)
    condition_stds = np.std(test, axis=0)
    replication_means = np.mean(test, axis=1)
    linear_prediction = np.repeat(frozen, test.shape[0])
    return {
        "r2": score,
        "r2_ci_lower": float(lower),
        "r2_ci_upper": float(upper),
        "within_signal_cv": float(np.mean(condition_stds / np.maximum(np.abs(condition_means), EPSILON))),
        "rmse_kw": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mae_kw": float(np.mean(np.abs(actual - predicted))),
        "linear_model_r2": _r2(actual, linear_prediction),
        "signal_conditions": int(test.shape[1]),
        "train_replications": int(train.shape[0]),
        "eval_replications": int(test.shape[0]),
        "profile_steps": profile_steps.tolist(),
        "scaling_mean": float(np.mean(replication_means)),
        "scaling_std": float(np.std(replication_means)),
        "scaling_cv": float(np.std(replication_means) / max(abs(float(np.mean(replication_means))), EPSILON)),
    }


def _dataset_payload(source: Path, dataset: str, train_replications: int) -> tuple[dict[str, Any], dict[str, Any]]:
    source = source.resolve()
    response_dir = source / "raw/responses" / dataset
    archives = []
    for path in response_dir.glob("responses_N*_data_coupled.npz"):
        match = ARCHIVE_PATTERN.match(path.name)
        if match:
            archives.append((int(match.group(1)), path))
    if not archives:
        raise FileNotFoundError(f"no data-coupled response archives for {dataset}")

    per_n: dict[str, dict[str, Any]] = {}
    profile_steps: list[int] | None = None
    scaling_rows = []
    for index, (fleet_size, archive) in enumerate(sorted(archives)):
        metrics = _cell_metrics(
            archive,
            train_replications,
            BOOTSTRAP_SEED + index + sum(ord(character) for character in dataset),
        )
        current_steps = metrics.pop("profile_steps")
        if profile_steps is None:
            profile_steps = current_steps
        elif profile_steps != current_steps:
            raise ValueError(f"profile steps differ across N for {dataset}")
        scaling_rows.append({
            "N": fleet_size,
            "mean": metrics.pop("scaling_mean"),
            "std": metrics.pop("scaling_std"),
            "cv": metrics.pop("scaling_cv"),
        })
        per_n[str(fleet_size)] = {"N": fleet_size, **metrics}

    n_values = [row["N"] for row in scaling_rows]
    crossing_index = next(
        (index for index, fleet_size in enumerate(n_values) if per_n[str(fleet_size)]["r2"] >= THRESHOLD),
        None,
    )
    threshold_grid = n_values[crossing_index] if crossing_index is not None else None
    if crossing_index is None:
        threshold_interpolated = None
    elif crossing_index == 0:
        threshold_interpolated = float(threshold_grid)
    else:
        left_n = n_values[crossing_index - 1]
        right_n = n_values[crossing_index]
        threshold_interpolated = _log_interpolate(
            left_n,
            per_n[str(left_n)]["r2"],
            right_n,
            per_n[str(right_n)]["r2"],
        )

    n_array = np.asarray(n_values, dtype=float)
    cv_array = np.asarray([max(row["cv"], EPSILON) for row in scaling_rows], dtype=float)
    slope = float(np.polyfit(np.log(n_array), np.log(cv_array), 1)[0]) if len(n_array) >= 2 else None
    threshold_payload = {
        "N_values": n_values,
        "per_N": per_n,
        "threshold_N_95": threshold_grid,
        "threshold_N_95_interpolated": threshold_interpolated,
        "threshold_not_reached": crossing_index is None,
        "protocol": "e1_data_coupled_fixed_condition_multi_replication",
        "protocol_details": {
            "source": str(source.relative_to(ROOT)),
            "coupling_arm": "data_coupled",
            "train_conditions": len(profile_steps or []),
            "eval_conditions": len(profile_steps or []),
            "train_replications": train_replications,
            "eval_replications": next(iter(per_n.values()))["eval_replications"],
            "profile_steps": profile_steps,
            "profile_sampling": "seeded_random_steps_from_device_day",
            "shared_train_eval_conditions": True,
            "same_device_subset_within_N": True,
            "reset_initial_state_each_snapshot": True,
            "fixed_signal_schedule": True,
            "signal_scenarios": ["valley_filling", "peak_shaving"],
            "availability_mapping": "ranked_real_data_diurnal_residual_data_coupled",
            "network_feedback_preserved": False,
            "bootstrap_unit": "condition",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "r2_prediction": "independent_training_replication_condition_mean",
            "n95_grid_rule": "first_N_with_point_R2_at_least_0.95",
            "n95_interpolation": "linear_R2_between_adjacent_log10_N_grid_points",
        },
    }
    scaling_payload = {
        "real_snapshot": {
            "scaling_data": scaling_rows,
            "loglog_slope": slope,
            "sigma_hat": float(cv_array[0] * np.sqrt(n_array[0])),
            "protocol": "same_E1_data_coupled_test_replications_as_Figure_3A",
        }
    }
    return threshold_payload, scaling_payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _update_config(path: Path, threshold: dict[str, Any]) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    details = threshold["protocol_details"]
    config["n_scaling_values"] = threshold["N_values"]
    config["n_threshold_train_conditions"] = details["train_conditions"]
    config["n_threshold_eval_conditions"] = details["eval_conditions"]
    config["n_threshold_train_replications"] = details["train_replications"]
    config["n_threshold_eval_replications"] = details["eval_replications"]
    config["n_threshold_shared_conditions"] = True
    config["n_threshold_bootstrap"] = details["bootstrap_draws"]
    config["fig3a_synchronized_protocol"] = details
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _update_heterogeneity(path: Path, threshold: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8"))
    cv_values = existing.get("metadata", {}).get("cv_values", [0.0])
    if not cv_values:
        cv_values = [0.0]
    r2_rows = [
        [threshold["per_N"][str(fleet_size)]["r2"] for _ in cv_values]
        for fleet_size in threshold["N_values"]
    ]
    payload = {
        "metadata": {
            "protocol": "e1_data_coupled_r2_at_empirical_capacity_cv",
            "cv_values": cv_values,
            "N_values": threshold["N_values"],
            "threshold_protocol": threshold["protocol"],
            "source": threshold["protocol_details"]["source"],
        },
        "grid": {"r2": r2_rows},
    }
    _write_json(path, payload)


def _render_figure3(result_root: Path) -> None:
    from src.extra.ieee33_device_day_simulation.figures.legacy_compat import _prepare
    from src.extra.paper_figures import make_figure3

    compatibility_root = _prepare(result_root)
    figure = make_figure3(compatibility_root, result_root / "Figs")
    paper_figure = result_root / "paper_figures" / figure.name
    paper_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure, paper_figure)


def synchronize(source: Path, *, render: bool) -> list[dict[str, Any]]:
    train_replications = 10
    response_root = source / "raw/responses"
    summaries = []
    for dataset_dir in sorted(path for path in response_root.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        result_root = ROOT / "results" / f"{dataset}_ieee33_real_load/coverage_fix/network_constrained_new"
        if not result_root.is_dir():
            raise FileNotFoundError(f"legacy result directory is missing: {result_root}")
        threshold, scaling = _dataset_payload(source, dataset, train_replications)
        _write_json(result_root / "data/n_threshold.json", threshold)
        _write_json(result_root / "data/n_scaling.json", scaling)
        _update_heterogeneity(result_root / "data/heterogeneity_lookup_table.json", threshold)
        config_paths = (
            result_root / "config/experiment_protocol.yaml",
            result_root.parent / "config/experiment_protocol.yaml",
        )
        for config_path in config_paths:
            if config_path.is_file():
                _update_config(config_path, threshold)
        _write_json(result_root / "data/fig3a_experiment_settings.json", threshold["protocol_details"])
        if render:
            _render_figure3(result_root)
        summaries.append({
            "dataset": dataset,
            "N_values": threshold["N_values"],
            "threshold_N_95": threshold["threshold_N_95"],
            "threshold_N_95_interpolated": threshold["threshold_N_95_interpolated"],
            "threshold_not_reached": threshold["threshold_not_reached"],
        })
    _write_json(source / "fig3_legacy_sync_summary.json", summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    summaries = synchronize(source, render=not args.no_render)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
