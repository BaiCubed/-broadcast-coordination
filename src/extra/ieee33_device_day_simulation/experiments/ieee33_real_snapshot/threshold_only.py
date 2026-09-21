from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from ...config_loader import load_config
from ...population.device_day_loader import load_device_day_pool
from ..run_experiment import _balanced_subset, _split_zone_batches
from ..protocol import build_condition_signal_schedule, r2_score, simulate_day, trace_features, write_json


def _threshold(config: dict, mode: str, result_root: Path) -> None:
    config = copy.deepcopy(config)
    config["control"]["network_feedback"] = mode == "network_stress"
    pool = load_device_day_pool(config)
    population = config["population"]
    per_zone = int(population["resources_per_zone"])
    validation_batches = int(config["experiment"]["validation_batches"])
    sampled = pool.sample(
        per_zone=per_zone * (1 + validation_batches),
        seed=int(config["simulation"]["random_seed"]) + int(population["seed_offset"]),
        with_replacement=bool(population["with_replacement"]),
    )
    records = _split_zone_batches(sampled, per_zone, 1 + validation_batches)[0]
    experiment = config["experiment"]
    seed = int(config["simulation"]["random_seed"])
    n_values = [int(value) for value in experiment["n_scaling_values"]]
    train_conditions = int(experiment["n_threshold_train_conditions"])
    eval_conditions = int(experiment["n_threshold_eval_conditions"])
    train_reps = int(experiment["n_threshold_train_replications"])
    eval_reps = int(experiment["n_threshold_eval_replications"])
    shared = bool(experiment["n_threshold_shared_conditions"])
    total_conditions = train_conditions if shared else train_conditions + eval_conditions
    start = int(experiment["n_threshold_profile_start"])
    if start + total_conditions > int(config["simulation"]["steps_per_day"]):
        raise ValueError("threshold profile window exceeds one day")

    profile_steps = list(range(start, start + total_conditions))
    schedule = build_condition_signal_schedule(
        _balanced_subset(records, max(n_values)),
        config,
        profile_steps,
        list(experiment["n_threshold_signal_scenarios"]),
        seed=seed + 2900,
    )
    train_steps = profile_steps[:train_conditions]
    train_schedule = schedule[:train_conditions]
    if shared:
        eval_steps, eval_schedule = train_steps, train_schedule
    else:
        eval_steps, eval_schedule = profile_steps[train_conditions:], schedule[train_conditions:]

    def replicates(subset, profile_window, signal_window, count, seed_offset):
        features = None
        values = []
        for replicate in range(count):
            trace = simulate_day(
                subset,
                config,
                steps=len(profile_window),
                seed=seed_offset + replicate,
                profile_steps=profile_window,
                signal_overrides=signal_window,
                reset_each_step=True,
            )
            if features is None:
                features = trace_features(trace)
            values.append(np.asarray(trace.accepted_control_kw, dtype=float))
        return features, np.asarray(values, dtype=float)

    rows = {}
    bootstrap_count = int(experiment["n_threshold_bootstrap"])
    for index, n in enumerate(n_values):
        subset = _balanced_subset(records, n)
        train_x, train_y = replicates(subset, train_steps, train_schedule, train_reps, seed + 3000 + index * 100)
        eval_x, eval_y = replicates(subset, eval_steps, eval_schedule, eval_reps, seed + 4000 + index * 100)
        coefficients, _, _, _ = np.linalg.lstsq(train_x, np.mean(train_y, axis=0), rcond=None)
        linear_prediction_by_condition = eval_x @ coefficients
        prediction_by_condition = np.mean(train_y, axis=0)
        actual = eval_y.T.reshape(-1)
        prediction = np.repeat(prediction_by_condition, eval_reps)
        group_cv = [float(np.std(eval_y[:, j]) / max(abs(float(np.mean(eval_y[:, j]))), 1e-6)) for j in range(eval_conditions)]
        rng = np.random.default_rng(seed + 7000 + index)
        bootstrap = []
        for _ in range(bootstrap_count):
            sampled_indices = rng.integers(0, eval_conditions, size=eval_conditions)
            boot_actual = np.concatenate([eval_y[:, j] for j in sampled_indices])
            boot_prediction = np.repeat(prediction_by_condition[sampled_indices], eval_reps)
            bootstrap.append(r2_score(boot_actual, boot_prediction))
        lower, upper = np.percentile(bootstrap, [2.5, 97.5])
        rows[str(n)] = {
            "N": n,
            "r2": float(r2_score(actual, prediction)),
            "r2_ci_lower": float(lower),
            "r2_ci_upper": float(upper),
            "picp": 0.0,
            "within_signal_cv": float(np.mean(group_cv)),
            "rmse_kw": float(np.sqrt(np.mean((actual - prediction) ** 2))),
            "mae_kw": float(np.mean(np.abs(actual - prediction))),
            "signal_conditions": eval_conditions,
            "train_replications": train_reps,
            "eval_replications": eval_reps,
            "linear_model_r2": float(r2_score(actual, np.repeat(linear_prediction_by_condition, eval_reps))),
        }

    result_root.mkdir(parents=True, exist_ok=True)
    write_json(result_root / "data" / "n_threshold.json", {
        "N_values": n_values,
        "per_N": rows,
        "threshold_N_95": next((n for n in n_values if rows[str(n)]["r2"] >= 0.95), None),
        "protocol": "fixed_real_snapshot_multi_replication_shared_conditions",
        "protocol_details": {
            "train_conditions": train_conditions,
            "eval_conditions": eval_conditions,
            "train_replications": train_reps,
            "eval_replications": eval_reps,
            "profile_steps": profile_steps,
            "shared_train_eval_conditions": shared,
            "same_device_subset_within_N": True,
            "reset_initial_state_each_snapshot": True,
            "fixed_signal_schedule": True,
            "signal_scenarios": list(experiment["n_threshold_signal_scenarios"]),
            "network_feedback_preserved": mode == "network_stress",
        },
    })
    hetero_path = result_root / "data" / "heterogeneity_lookup_table.json"
    hetero = json.loads(hetero_path.read_text(encoding="utf-8")) if hetero_path.exists() else {
        "metadata": {"cv_values": [0.0], "N_values": n_values}
    }
    hetero.setdefault("metadata", {})["N_values"] = n_values
    hetero.setdefault("grid", {})["r2"] = [[rows[str(n)]["r2"]] for n in n_values]
    hetero["metadata"]["threshold_protocol"] = "fixed_real_snapshot_multi_replication_shared_conditions"
    write_json(hetero_path, hetero)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute real-snapshot Figure 3A")
    parser.add_argument("--mode", choices=["weak_correlation", "network_stress"], required=True)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    config = load_config()
    root = Path(args.results_root)
    _threshold(config, args.mode, root)
    from ...figures.plot_figures import plot_all
    for figure in plot_all(root):
        print(figure)


if __name__ == "__main__":
    main()
