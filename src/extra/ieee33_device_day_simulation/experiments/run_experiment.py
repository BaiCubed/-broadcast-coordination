from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config_loader import load_config, resolve_workspace_path
from ..population.device_day_loader import DeviceDay, load_device_day_pool
from .protocol import (
    SimulationTrace,
    build_condition_signal_schedule,
    build_experiments_training_signal_schedule,
    build_experiments_transfer_signal_schedule,
    build_real_signal_schedule,
    build_scenario_profile_signal_schedule,
    build_stratified_signal_schedule,
    concatenate_traces,
    r2_score,
    simulate_day,
    TRACE_FEATURE_NAMES,
    trace_features,
    write_json,
)


def _balanced_subset(records: list[DeviceDay], n: int) -> list[DeviceDay]:
    zones = sorted({record.zone_id for record in records})
    groups = {zone: [record for record in records if record.zone_id == zone] for zone in zones}
    per_zone, remainder = divmod(n, len(zones))
    selected: list[DeviceDay] = []
    for index, zone in enumerate(zones):
        count = per_zone + (1 if index < remainder else 0)
        selected.extend(groups[zone][:count])
    return selected


def _correlation_conditions(
    config: dict[str, Any],
    *,
    real_snapshot_only: bool,
) -> list[dict[str, Any]]:
    if real_snapshot_only:
        return [{
            "label": "real_snapshot",
            "zone_correlation": 0.0,
            "nominal_rho": None,
            "classification": "empirical_real_snapshot",
        }]
    configured = config.get("experiment", {}).get("correlation_conditions")
    if configured:
        return [
            {
                "label": str(row["label"]),
                "zone_correlation": float(row.get("zone_correlation", 0.0)),
                "nominal_rho": float(row["nominal_rho"]) if row.get("nominal_rho") is not None else None,
                "classification": str(row.get("classification", row["label"])),
            }
            for row in configured
        ]
    return [
        {"label": "iid", "zone_correlation": 0.0, "nominal_rho": 0.0, "classification": "iid"},
        {"label": "weak", "zone_correlation": 0.01, "nominal_rho": 0.01, "classification": "legacy_weak"},
        {"label": "moderate", "zone_correlation": 0.03, "nominal_rho": 0.03, "classification": "legacy_moderate"},
    ]


def _split_zone_batches(records: list[DeviceDay], per_zone: int, batch_count: int) -> list[list[DeviceDay]]:
    zones = sorted({record.zone_id for record in records})
    batches: list[list[DeviceDay]] = [[] for _ in range(batch_count)]
    for zone in zones:
        candidates = [record for record in records if record.zone_id == zone]
        expected = per_zone * batch_count
        if len(candidates) != expected:
            raise ValueError(f"{zone} requires {expected} sampled records, got {len(candidates)}")
        for batch_index in range(batch_count):
            start = batch_index * per_zone
            batches[batch_index].extend(candidates[start : start + per_zone])
    return batches


def _independent_profile_matrix(records: list[DeviceDay], steps: int, seed: int) -> list[list[int]]:
    if not records:
        raise ValueError("cannot sample profile points for an empty resource set")
    rng = np.random.default_rng(seed)
    return [
        [int(rng.integers(0, len(record.load_kw))) for record in records]
        for _ in range(int(steps))
    ]


def _trace_summary(trace: SimulationTrace, n: int) -> dict[str, float]:
    values = np.asarray(trace.accepted_control_kw, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "N": n,
        "mean_response_kw": mean,
        "std_response_kw": std,
        "cv": float(std / max(abs(mean), 1e-6)),
        "mean_transformer_loading": float(np.mean(trace.transformer_loading)),
        "voltage_violation_steps": float(np.sum(trace.voltage_violations)),
    }


def _trace_curtailment(trace: SimulationTrace) -> tuple[float, float]:
    pv = np.asarray(trace.energy_input_kw, dtype=float)
    load = np.asarray(trace.load_kw, dtype=float)
    control = np.asarray(trace.accepted_control_kw, dtype=float)
    baseline = np.maximum(pv - load, 0.0)
    eps = np.maximum(pv - load - np.maximum(control, 0.0), 0.0)
    return float(np.sum(baseline) * 5 / 60 / 1000), float(np.sum(eps) * 5 / 60 / 1000)


def _fit_stable_linear(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    means = np.mean(x[:, 1:], axis=0)
    scales = np.std(x[:, 1:], axis=0)
    scales = np.where(scales > 1e-10, scales, 1.0)
    standardized = np.column_stack([np.ones(len(x)), (x[:, 1:] - means) / scales])
    coefficients, _, _, _ = np.linalg.lstsq(standardized, y, rcond=None)
    raw = np.empty(x.shape[1], dtype=float)
    raw[1:] = coefficients[1:] / scales
    raw[0] = coefficients[0] - float(np.sum(means * raw[1:]))
    return raw


def _network_scale_features(
    trace: SimulationTrace,
    desired_prediction: np.ndarray,
) -> np.ndarray:
    base = trace_features(trace)
    desired = np.asarray(desired_prediction, dtype=float) / 10000.0
    transformer = np.asarray(trace.baseline_transformer_loading, dtype=float)
    branch = np.asarray(trace.baseline_maximum_branch_loading, dtype=float)
    low_margin = np.asarray(trace.baseline_minimum_voltage_pu, dtype=float) - 0.95
    high_margin = 1.05 - np.asarray(trace.baseline_maximum_voltage_pu, dtype=float)
    return np.column_stack([
        base,
        desired,
        np.abs(desired),
        desired ** 2,
        desired * transformer,
        desired * branch,
        desired * low_margin,
        desired * high_margin,
        np.maximum(-low_margin, 0.0),
        np.maximum(-high_margin, 0.0),
    ])


def _fit_hist_gradient_model(features: np.ndarray, target: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=240,
        max_leaf_nodes=15,
        min_samples_leaf=8,
        l2_regularization=1e-3,
        random_state=42,
    )
    model.fit(np.asarray(features, dtype=float), np.asarray(target, dtype=float))
    return model


def _nonlinear_final_features(
    trace: SimulationTrace,
    desired_prediction: np.ndarray,
    scale_prediction: np.ndarray,
) -> np.ndarray:
    base = trace_features(trace)
    desired = np.asarray(desired_prediction, dtype=float) / 10000.0
    scale = np.asarray(scale_prediction, dtype=float)
    return np.column_stack([
        base,
        desired,
        np.abs(desired),
        scale,
        desired * scale,
    ])


def _fit_nonlinear_two_stage_estimator(
    trace: SimulationTrace,
    fit_mask: np.ndarray,
) -> dict[str, HistGradientBoostingRegressor]:
    base = trace_features(trace)
    desired_target = np.asarray(trace.desired_control_kw, dtype=float)
    desired_model = _fit_hist_gradient_model(base[fit_mask], desired_target[fit_mask])
    desired_prediction = desired_model.predict(base)
    scale_features = _network_scale_features(trace, desired_prediction)
    scale_target = np.asarray(trace.network_scale, dtype=float)
    scale_model = _fit_hist_gradient_model(scale_features[fit_mask], scale_target[fit_mask])
    scale_prediction = np.clip(scale_model.predict(scale_features), 0.0, 1.0)
    final_features = _nonlinear_final_features(
        trace, desired_prediction, scale_prediction
    )
    accepted_target = np.asarray(trace.accepted_control_kw, dtype=float)
    response_model = _fit_hist_gradient_model(
        final_features[fit_mask], accepted_target[fit_mask]
    )
    return {
        "desired_model": desired_model,
        "scale_model": scale_model,
        "response_model": response_model,
    }


def _predict_estimator(
    estimator: dict[str, Any],
    trace: SimulationTrace,
    runtime_model: dict[str, HistGradientBoostingRegressor] | None = None,
) -> np.ndarray:
    features = trace_features(trace)
    if runtime_model is not None:
        desired = runtime_model["desired_model"].predict(features)
        scale_features = _network_scale_features(trace, desired)
        scale = np.clip(runtime_model["scale_model"].predict(scale_features), 0.0, 1.0)
        final_features = _nonlinear_final_features(trace, desired, scale)
        prediction = runtime_model["response_model"].predict(final_features)
        prediction[scale <= 1e-3] = 0.0
        return prediction
    components = estimator.get("linear_fallback_components") or estimator.get("model_components")
    if not components:
        return features @ np.asarray(estimator["coefficients"], dtype=float)
    desired = features @ np.asarray(components["desired_coefficients"], dtype=float)
    scale_features = _network_scale_features(trace, desired)
    scale = np.clip(
        scale_features @ np.asarray(components["network_scale_coefficients"], dtype=float),
        0.0,
        1.0,
    )
    residual = (
        features @ np.asarray(components["residual_coefficients"], dtype=float)
    ) * scale
    prediction = desired * scale + residual
    prediction[scale <= float(components.get("blocked_scale_threshold", 1e-3))] = 0.0
    return prediction


def _train_linear_estimator(train: SimulationTrace, validation: SimulationTrace) -> dict[str, Any]:
    x_train = trace_features(train)
    y_train = np.asarray(train.accepted_control_kw, dtype=float)
    indices = np.arange(len(y_train))
    calibration_mask = indices % 4 == 0
    fit_mask = ~calibration_mask
    direct_coefficients = _fit_stable_linear(x_train[fit_mask], y_train[fit_mask])
    desired_target = np.asarray(train.desired_control_kw, dtype=float)
    desired_coefficients = _fit_stable_linear(
        x_train[fit_mask], desired_target[fit_mask]
    )
    desired_train = x_train @ desired_coefficients
    scale_train_features = _network_scale_features(train, desired_train)
    scale_target = np.asarray(train.network_scale, dtype=float)
    scale_coefficients = _fit_stable_linear(
        scale_train_features[fit_mask], scale_target[fit_mask]
    )
    predicted_scale_train = np.clip(
        scale_train_features @ scale_coefficients, 0.0, 1.0
    )
    structured_train = desired_train * predicted_scale_train
    residual_fit_mask = fit_mask & (predicted_scale_train > 1e-3)
    if np.sum(residual_fit_mask) >= 2:
        residual_coefficients = _fit_stable_linear(
            x_train[residual_fit_mask],
            ((y_train - structured_train) / np.maximum(predicted_scale_train, 1e-3))[
                residual_fit_mask
            ],
        )
    else:
        residual_coefficients = np.zeros(x_train.shape[1], dtype=float)
    model_components = {
        "desired_coefficients": desired_coefficients.tolist(),
        "network_scale_coefficients": scale_coefficients.tolist(),
        "residual_coefficients": residual_coefficients.tolist(),
        "blocked_scale_threshold": 1e-3,
    }
    runtime_estimator = {
        "coefficients": direct_coefficients.tolist(),
        "model_components": model_components,
    }
    nonlinear_model = _fit_nonlinear_two_stage_estimator(train, fit_mask)
    train_prediction = _predict_estimator(
        runtime_estimator, train, nonlinear_model
    )
    calibration_prediction = train_prediction[calibration_mask]
    calibration_residual = np.abs(y_train[calibration_mask] - calibration_prediction)
    coverage = 0.90
    quantile_level = min(
        1.0,
        np.ceil((len(calibration_residual) + 1) * coverage) / len(calibration_residual),
    )
    conformal_radius = float(np.quantile(calibration_residual, quantile_level, method="higher"))
    y_val = np.asarray(validation.accepted_control_kw, dtype=float)
    predictions = _predict_estimator(
        runtime_estimator, validation, nonlinear_model
    )
    intervals = np.column_stack([
        predictions - conformal_radius,
        predictions + conformal_radius,
    ])
    covered = (y_val >= intervals[:, 0]) & (y_val <= intervals[:, 1])
    count = min(500, len(y_val))
    return {
        "method": "two_stage_physics_informed_hist_gradient_response_and_network_scale",
        "features": TRACE_FEATURE_NAMES,
        "coefficients": direct_coefficients.tolist(),
        "linear_fallback_components": model_components,
        "model_components": {
            "desired_model": "HistGradientBoostingRegressor",
            "network_scale_model": "HistGradientBoostingRegressor",
            "conditional_response_model": "HistGradientBoostingRegressor",
            "max_iter": 240,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 8,
            "model_artifact": "estimator.joblib",
        },
        "_runtime_model": nonlinear_model,
        "train_samples": int(fit_mask.sum()),
        "calibration_samples": int(calibration_mask.sum()),
        "interval_method": "split_conformal_absolute_residual",
        "target_coverage": coverage,
        "conformal_radius_kw": conformal_radius,
        "test_data": {
            "actuals": y_val[:count].tolist(),
            "predictions": predictions[:count].tolist(),
            "intervals": intervals[:count].tolist(),
            "prediction_sources": ["ieee33_device_day_validation"] * count,
        },
        "point_metrics": {
            "r2": {"value": r2_score(y_val, predictions)},
            "rmse_kw": float(np.sqrt(np.mean((y_val - predictions) ** 2))),
            "mae_kw": float(np.mean(np.abs(y_val - predictions))),
        },
        "interval_metrics": {
            "picp": float(np.mean(covered)),
            "pinaw": float(
                np.mean(intervals[:, 1] - intervals[:, 0])
                / max(float(np.ptp(y_val)), 1e-9)
            ),
        },
    }


def _write_network_metrics(root: Path, trace: SimulationTrace, records: list[DeviceDay]) -> None:
    data = root / "data"
    write_json(data / "network_timeseries.json", trace.as_dict())
    zone_metrics = {}
    for zone, values in trace.zone_control_kw.items():
        array = np.asarray(values, dtype=float)
        zone_metrics[zone] = {
            "resource_count": sum(record.zone_id == zone for record in records),
            "mean_control_kw": float(np.mean(array)),
            "std_control_kw": float(np.std(array)),
            "peak_abs_control_kw": float(np.max(np.abs(array))),
        }
    write_json(data / "zone_metrics.json", zone_metrics)
    node_metrics = {}
    for bus, values in trace.bus_control_kw.items():
        array = np.asarray(values, dtype=float)
        node_metrics[bus] = {
            "resource_count": sum(record.bus_id == int(bus) for record in records),
            "mean_control_kw": float(np.mean(array)),
            "peak_abs_control_kw": float(np.max(np.abs(array))),
        }
    write_json(data / "node_metrics.json", node_metrics)
    write_json(data / "line_loading.json", {
        "mean_transformer_loading": float(np.mean(trace.transformer_loading)),
        "max_transformer_loading": float(np.max(trace.transformer_loading)),
        "overloaded_branch_count_by_step": trace.overloaded_branches,
    })
    write_json(data / "voltage_violations.json", {
        "minimum_voltage_pu": trace.minimum_voltage_pu,
        "maximum_voltage_pu": trace.maximum_voltage_pu,
        "violations_by_step": trace.voltage_violations,
    })
    write_json(data / "transformer_loading.json", {
        "loading_fraction_by_step": trace.transformer_loading,
        "warning_fraction": 0.90,
    })


def _transfer_signals(
    trace: SimulationTrace,
    hours: list[int],
) -> list[dict[str, float | int]]:
    return [
        {
            "supply_demand": int(direction),
            "intensity": int(round(intensity)),
            "price": 0.0,
            "hour": int(hour),
            "direction": 1 if int(direction) <= 7 else -1,
            "region_id": 0,
            "priority": 8,
        }
        for direction, intensity, hour in zip(trace.direction, trace.intensity, hours)
    ]


def _evaluate_transfer_estimator(
    estimator: Any,
    signals: list[dict[str, float | int]],
    responses: np.ndarray,
) -> dict[str, float]:
    predictions: list[float] = []
    covered = 0
    for signal, actual in zip(signals, responses):
        estimate = estimator.estimate(signal)
        predictions.append(float(estimate.response_kw))
        covered += int(float(estimate.lower_bound) <= float(actual) <= float(estimate.upper_bound))
    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(responses, dtype=float)
    denominator = np.maximum(np.abs(actual) + np.abs(predicted), 1e-8)
    return {
        "r2": r2_score(actual, predicted),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "smape": float(np.mean(2.0 * np.abs(actual - predicted) / denominator) * 100.0),
        "picp": float(covered / max(len(actual), 1) * 100.0),
    }


def run_cross_region_transfer_experiments_protocol(
    config_path: str | Path,
    *,
    mode: str,
    result_root: str | Path,
    training_samples: int = 4000,
    evaluation_samples: int = 500,
    online_samples: int = 1000,
    stabilization_passes: int = 3,
) -> Path:
    if mode not in {"weak_correlation", "network_stress"}:
        raise ValueError(f"unsupported experiment mode: {mode}")
    config = copy.deepcopy(load_config(config_path))
    config["control"]["network_feedback"] = bool(
        config["experiment"].get("force_network_feedback", mode == "network_stress")
    )
    pool = load_device_day_pool(config)
    population = config["population"]
    per_zone = int(population["resources_per_zone"])
    sample_seed = int(config["simulation"]["random_seed"]) + int(population["seed_offset"])
    if bool(population.get("unique_source_per_batch", False)):
        records = pool.sample_source_unique_batches(
            per_zone=per_zone,
            batch_count=1,
            seed=sample_seed,
        )[0]
    else:
        records = pool.sample(
            per_zone=per_zone,
            seed=sample_seed,
            with_replacement=bool(population["with_replacement"]),
        )

    zone_ids = sorted(config["zones"]["zones"])
    source_zone = zone_ids[0]
    target_zones = zone_ids[1:4]
    if len(target_zones) != 3:
        raise ValueError("cross-region transfer requires one source and three target zones")
    grouped = {
        zone: [record for record in records if record.zone_id == zone]
        for zone in [source_zone, *target_zones]
    }
    if any(not grouped[zone] for zone in grouped):
        raise ValueError("every transfer region must contain at least one resource")

    steps_per_day = int(config["simulation"]["steps_per_day"])

    def make_samples(zone: str, count: int, protocol_seed: int) -> tuple[list[dict[str, float | int]], np.ndarray]:
        schedule, hours = build_experiments_transfer_signal_schedule(
            config, count, seed=protocol_seed
        )
        profiles = np.random.default_rng(protocol_seed + 73000).integers(
            0, steps_per_day, size=count
        ).astype(int).tolist()
        trace = simulate_day(
            grouped[zone],
            config,
            steps=count,
            seed=protocol_seed + 91000,
            profile_steps=profiles,
            signal_overrides=schedule,
            reset_each_step=True,
        )
        return _transfer_signals(trace, hours), np.asarray(trace.accepted_control_kw, dtype=float)

    import random
    from src.estimation import EPSEstimator, EstimatorConfig

    random.seed(42)
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass

    train_signals, train_responses = make_samples(source_zone, int(training_samples), 42)
    estimator = EPSEstimator(EstimatorConfig(
        target_coverage=0.90,
        enable_conformal=True,
        use_pytorch=True,
    ))
    estimator.fit(train_signals, train_responses.tolist())
    source_signals, source_responses = make_samples(source_zone, int(evaluation_samples), 77777)
    source_metrics = _evaluate_transfer_estimator(estimator, source_signals, source_responses)

    evaluation_seeds = [88888, 99999, 11111]
    online_seeds = [55555, 66666, 22222]
    regions: dict[str, Any] = {}
    for region_index, (zone, evaluation_seed, online_seed) in enumerate(
        zip(target_zones, evaluation_seeds, online_seeds)
    ):
        eval_signals, eval_responses = make_samples(zone, int(evaluation_samples), evaluation_seed)
        cold = _evaluate_transfer_estimator(estimator, eval_signals, eval_responses)
        online_signals, online_responses = make_samples(zone, int(online_samples), online_seed)
        adapted_nn = copy.deepcopy(estimator)
        adapted_conformal = copy.deepcopy(estimator)
        for candidate in (adapted_nn, adapted_conformal):
            if getattr(candidate, "_cqr", None) is not None and hasattr(candidate._cqr, "reset"):
                candidate._cqr.reset()
            candidate.config.online_validation_patience = 999
            candidate.config.online_validation_window = 200

        checkpoints = {100, 250, 500, 750, int(online_samples)}
        curve = [{"samples": 0, **cold}]
        total_online = max(int(online_samples), 1)
        for index, (signal, actual) in enumerate(zip(online_signals, online_responses)):
            learning_rate = 0.0005 + 0.5 * (0.003 - 0.0005) * (
                1.0 + np.cos(np.pi * index / total_online)
            )
            adapted_nn.online_update(
                signal,
                float(actual),
                update_nn=True,
                error_threshold=0.0,
                learning_rate=float(learning_rate),
                mini_batch_size=25,
                anchor_lambda=0.5,
            )
            adapted_conformal.online_update(
                signal,
                float(actual),
                update_nn=False,
                error_threshold=0.0,
            )
            if index + 1 in checkpoints:
                curve.append({
                    "samples": index + 1,
                    **_evaluate_transfer_estimator(adapted_nn, eval_signals, eval_responses),
                })

        stabilization_rng = np.random.default_rng(12345)
        for pass_index in range(int(stabilization_passes)):
            for sample_index in stabilization_rng.permutation(len(online_signals)):
                adapted_nn.online_update(
                    online_signals[int(sample_index)],
                    float(online_responses[int(sample_index)]),
                    update_nn=True,
                    error_threshold=0.0,
                    learning_rate=0.001,
                    mini_batch_size=50,
                    anchor_lambda=0.1,
                )
            curve.append({
                "samples": int(online_samples) * (pass_index + 2),
                **_evaluate_transfer_estimator(adapted_nn, eval_signals, eval_responses),
            })

        regions[chr(ord("B") + region_index)] = {
            "label": zone,
            "cold_start": cold,
            "adapted_nn": {
                **_evaluate_transfer_estimator(adapted_nn, eval_signals, eval_responses),
                "n_online_samples": int(online_samples),
            },
            "adapted_conformal": {
                **_evaluate_transfer_estimator(adapted_conformal, eval_signals, eval_responses),
                "n_online_samples": int(online_samples),
            },
            "convergence_curve_nn": curve,
        }

    destination = Path(result_root) / "data" / "cross_region_transfer.json"
    write_json(destination, {
        "regions": regions,
        "region_A": {
            "label": source_zone,
            "n_resources": len(grouped[source_zone]),
            "n_train": int(training_samples),
            **source_metrics,
        },
        "protocol": "experiments_source_train_4000_target_eval_500_online_1000",
        "source_region": source_zone,
        "source_resource_count": len(grouped[source_zone]),
        "target_resource_counts": {zone: len(grouped[zone]) for zone in target_zones},
        "source_region_only": True,
        "matched_source_signal_distribution": True,
        "target_region_real_records": True,
        "training_samples": int(training_samples),
        "evaluation_samples_per_region": int(evaluation_samples),
        "online_samples_per_region": int(online_samples),
        "stabilization_passes": int(stabilization_passes),
        "estimator": "EPSEstimator_dual_quantile_nn_cqr",
    })
    return destination


def run_default(
    config_path: str | Path | None = None,
    *,
    mode: str = "network_stress",
    results_root: str | Path | None = None,
    snapshot_protocol: bool = False,
    snapshot_sampling: str = "aligned",
    real_snapshot_only: bool = False,
    estimation_only: bool = False,
    figure2_only: bool = False,
) -> Path:
    started = time.time()
    if mode not in {"weak_correlation", "network_stress"}:
        raise ValueError(f"unsupported experiment mode: {mode}")
    if snapshot_sampling not in {"aligned", "independent"}:
        raise ValueError(f"unsupported snapshot sampling: {snapshot_sampling}")
    if snapshot_sampling == "independent" and not snapshot_protocol:
        raise ValueError("independent snapshot sampling requires snapshot_protocol=True")
    config = copy.deepcopy(load_config(config_path))
    config["control"]["network_feedback"] = bool(
        config["experiment"].get("force_network_feedback", mode == "network_stress")
    )
    config["experiment"]["mode"] = mode
    pool = load_device_day_pool(config)
    population = config["population"]
    zone_count = len(config["zones"]["zones"])
    per_zone = int(population["resources_per_zone"])
    expected_resources = int(population["N_simulated_resources"])
    if expected_resources < 500:
        raise ValueError("the IEEE33 fleet must contain at least 500 independent resources")
    if expected_resources != per_zone * zone_count:
        raise ValueError("N_simulated_resources must equal resources_per_zone times zone count")
    experiment = config["experiment"]
    if figure2_only:
        experiment["training_steps"] = 12000
        experiment["scenario_steps"] = 576
        experiment["estimator_signal_sampling"] = "experiments_mixed"
    validation_batch_count = int(experiment["validation_batches"])
    if bool(population.get("unique_source_per_batch", False)):
        record_batches = pool.sample_source_unique_batches(
            per_zone=per_zone,
            batch_count=1 + validation_batch_count,
            seed=int(config["simulation"]["random_seed"]) + int(population["seed_offset"]),
        )
        sampled_records = [record for batch in record_batches for record in batch]
    else:
        sampled_records = pool.sample(
            per_zone=per_zone * (1 + validation_batch_count),
            seed=int(config["simulation"]["random_seed"]) + int(population["seed_offset"]),
            with_replacement=bool(population["with_replacement"]),
        )
        record_batches = _split_zone_batches(
            sampled_records,
            per_zone=per_zone,
            batch_count=1 + validation_batch_count,
        )
    records = record_batches[0]
    validation_record_batches = record_batches[1:]

    result_root = (
        Path(results_root)
        if results_root is not None
        else resolve_workspace_path(config, "results_root") / mode
    )
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "data").mkdir(exist_ok=True)
    (result_root / "estimation").mkdir(exist_ok=True)

    seed = int(config["simulation"]["random_seed"])
    snapshot_schedules: dict[str, list[list[tuple[int, int]]]] = {}
    snapshot_horizon = max(
        int(experiment["training_steps"]),
        int(experiment["validation_samples"]),
        int(experiment["scenario_steps"]),
    )
    if snapshot_protocol and str(experiment.get("snapshot_profile_mode", "")).lower() == "random_aligned":
        profile_rng = np.random.default_rng(
            seed + int(experiment.get("snapshot_profile_seed_offset", 0))
        )
        snapshot_profile_steps = profile_rng.integers(
            0,
            int(config["simulation"]["steps_per_day"]),
            size=snapshot_horizon,
        ).astype(int).tolist()
    else:
        snapshot_profile_steps = list(range(int(config["simulation"]["steps_per_day"])))
    if snapshot_protocol:
        for scenario_name in ["normal", "peak_shaving", "valley_filling", "emergency_grid_stability"]:
            snapshot_schedules[scenario_name] = build_real_signal_schedule(
                records,
                config,
                snapshot_profile_steps,
                seed=seed + 8000 + len(snapshot_schedules),
                scenario=scenario_name,
            )

    def run_trace(
        trace_records: list[DeviceDay],
        *,
        steps: int,
        trace_seed: int,
        scenario: str = "normal",
        trace_config: dict[str, Any] | None = None,
        zone_correlation: float = 0.0,
        profile_steps_override: list[int] | None = None,
        signal_schedule_override: list[list[tuple[int, int]]] | None = None,
    ) -> SimulationTrace:
        active_config = trace_config or config
        if not snapshot_protocol:
            return simulate_day(
                trace_records,
                active_config,
                steps=steps,
                seed=trace_seed,
                scenario=scenario,
                zone_correlation=zone_correlation,
            )
        active_profile_steps = profile_steps_override or snapshot_profile_steps
        active_signal_schedule = signal_schedule_override or snapshot_schedules[scenario]
        count = min(int(steps), len(active_profile_steps))
        profile_matrix = None
        if snapshot_sampling == "independent":
            profile_matrix = _independent_profile_matrix(
                trace_records,
                count,
                trace_seed + 91000,
            )
        return simulate_day(
            trace_records,
            active_config,
            steps=count,
            seed=trace_seed,
            scenario=scenario,
            zone_correlation=zone_correlation,
            profile_steps=active_profile_steps[:count],
            signal_overrides=active_signal_schedule[:count],
            profile_steps_by_record=profile_matrix,
            reset_each_step=True,
        )

    def run_scenario_sequence(
        trace_records: list[DeviceDay],
        *,
        steps: int,
        trace_seed: int,
        scenarios: list[str],
    ) -> SimulationTrace:
        normalized = [str(item) for item in scenarios] or ["normal"]
        if len(normalized) == 1:
            return run_trace(
                trace_records,
                steps=steps,
                trace_seed=trace_seed,
                scenario=normalized[0],
            )
        base, remainder = divmod(int(steps), len(normalized))
        traces: list[SimulationTrace] = []
        for index, scenario_name in enumerate(normalized):
            chunk_steps = base + (1 if index < remainder else 0)
            if chunk_steps <= 0:
                continue
            traces.append(
                run_trace(
                    trace_records,
                    steps=chunk_steps,
                    trace_seed=trace_seed + index + 1,
                    scenario=scenario_name,
                )
            )
        return concatenate_traces(traces, int(steps))

    training_scenarios = [str(item) for item in experiment.get("training_scenarios", ["normal"])]
    validation_scenarios = [str(item) for item in experiment.get("validation_scenarios", ["normal"])]
    estimator_signal_sampling = str(
        experiment.get("estimator_signal_sampling", "scenario")
    ).lower()
    if estimator_signal_sampling in {"stratified_uniform", "experiments_mixed"}:
        training_steps = int(experiment["training_steps"])
        if estimator_signal_sampling == "experiments_mixed":
            training_schedule = build_experiments_training_signal_schedule(
                config, training_steps, seed=42
            )
        else:
            training_schedule = build_stratified_signal_schedule(
                config,
                training_steps,
                seed=seed + 71001,
                min_abs_score=float(experiment.get("estimator_min_abs_signal", 0.05)),
                max_abs_score=float(experiment.get("estimator_max_abs_signal", 0.95)),
            )
        train_trace = run_trace(
            records,
            steps=training_steps,
            trace_seed=seed + 1,
            signal_schedule_override=training_schedule,
        )
    elif estimator_signal_sampling == "scenario":
        train_trace = run_scenario_sequence(
            records,
            steps=int(experiment["training_steps"]),
            trace_seed=seed + 1,
            scenarios=training_scenarios,
        )
    else:
        raise ValueError(
            f"unsupported estimator_signal_sampling={estimator_signal_sampling!r}"
        )
    validation_samples = int(experiment["validation_samples"])
    validation_schedule = None
    if estimator_signal_sampling in {"stratified_uniform", "experiments_mixed"}:
        validation_schedule = (
            build_experiments_training_signal_schedule(
                config, validation_samples, seed=99999
            )
            if estimator_signal_sampling == "experiments_mixed"
            else build_stratified_signal_schedule(
                config,
                validation_samples,
                seed=seed + 71002,
                min_abs_score=float(experiment.get("estimator_min_abs_signal", 0.05)),
                max_abs_score=float(experiment.get("estimator_max_abs_signal", 0.95)),
            )
        )
    validation_traces = []
    remaining_samples = validation_samples
    validation_offset = 0
    balanced_validation = bool(experiment.get("balanced_validation_samples", False))
    validation_base, validation_remainder = divmod(
        validation_samples,
        max(len(validation_record_batches), 1),
    )
    for batch_index, validation_records in enumerate(validation_record_batches):
        requested_steps = (
            validation_base + (1 if batch_index < validation_remainder else 0)
            if balanced_validation
            else remaining_samples
        )
        steps = min(int(config["simulation"]["steps_per_day"]), requested_steps)
        if steps <= 0:
            break
        validation_traces.append(
            run_trace(
                validation_records,
                steps=steps,
                trace_seed=seed + 2 + batch_index,
                scenario=validation_scenarios[batch_index % len(validation_scenarios)],
                profile_steps_override=(
                    snapshot_profile_steps[validation_offset : validation_offset + steps]
                    if validation_schedule is not None else None
                ),
                signal_schedule_override=(
                    validation_schedule[validation_offset : validation_offset + steps]
                    if validation_schedule is not None else None
                ),
            )
        )
        remaining_samples -= steps
        validation_offset += steps
    validation_trace = concatenate_traces(validation_traces, validation_samples)
    validation_day_trace = validation_traces[0]
    estimator = _train_linear_estimator(train_trace, validation_trace)
    runtime_estimator = estimator.pop("_runtime_model")
    joblib.dump(runtime_estimator, result_root / "estimation" / "estimator.joblib")
    write_json(result_root / "estimation" / "estimation_validation_results.json", estimator)
    write_json(result_root / "data" / "training_model.json", estimator)
    write_json(result_root / "data" / "validation_results.json", estimator["point_metrics"])
    if estimation_only:
        return result_root

    if snapshot_protocol:
        day_steps = int(config["simulation"]["steps_per_day"])
        day_profile_steps = list(range(day_steps))
        day_schedule = build_real_signal_schedule(
            records,
            config,
            day_profile_steps,
            seed=seed + 12000,
            scenario="normal",
        )
        day_trace = run_trace(
            records,
            steps=day_steps,
            trace_seed=seed + 12001,
            scenario="normal",
            profile_steps_override=day_profile_steps,
            signal_schedule_override=day_schedule,
        )
        write_json(result_root / "data" / "network_timeseries_day.json", day_trace.as_dict())

    def write_figure2_scenario_statistics() -> None:
        scenarios: dict[str, dict[str, Any]] = {}
        multi_stats: dict[str, dict[str, Any]] = {}
        scenario_steps = int(experiment["scenario_steps"])
        scenario_runs = 8
        scenario_names = [
            "peak_shaving",
            "valley_filling",
            "emergency_grid_stability",
            "emergency_supply_shortage",
        ]
        for scenario in scenario_names:
            r2_values: list[float] = []
            smape_values: list[float] = []
            first_trace: SimulationTrace | None = None
            for run_id in range(scenario_runs):
                run_seed = 42 + run_id
                profile_rng = np.random.default_rng(run_seed + 10000)
                profile_steps = profile_rng.integers(
                    0,
                    int(config["simulation"]["steps_per_day"]),
                    size=scenario_steps,
                ).astype(int).tolist()
                schedule = build_scenario_profile_signal_schedule(
                    config,
                    scenario,
                    scenario_steps,
                    seed=run_seed,
                    ratio_noise_std=0.02,
                )
                trace = run_trace(
                    records,
                    steps=scenario_steps,
                    trace_seed=run_seed,
                    scenario=scenario,
                    profile_steps_override=profile_steps,
                    signal_schedule_override=schedule,
                )
                actual = np.asarray(trace.accepted_control_kw, dtype=float)
                predicted = _predict_estimator(estimator, trace, runtime_estimator)
                r2_values.append(float(r2_score(actual, predicted)))
                residual_scale = np.maximum(
                    np.abs(actual) + np.abs(predicted), 1e-6
                )
                smape_values.append(float(
                    np.mean(2.0 * np.abs(actual - predicted) / residual_scale) * 100.0
                ))
                if first_trace is None:
                    first_trace = trace
            values = np.asarray(r2_values, dtype=float)
            smapes = np.asarray(smape_values, dtype=float)
            r2_std = float(np.std(values, ddof=1))
            smape_std = float(np.std(smapes, ddof=1))
            se_factor = float(stats.t.ppf(0.975, df=scenario_runs - 1) / np.sqrt(scenario_runs))
            summary = _trace_summary(first_trace, len(records))
            summary.update({
                "num_runs": scenario_runs,
                "scenario": scenario,
                "r2_mean": float(np.mean(values)),
                "r2_std": r2_std,
                "r2_ci_lower": float(np.mean(values) - se_factor * r2_std),
                "r2_ci_upper": float(np.mean(values) + se_factor * r2_std),
                "smape_mean": float(np.mean(smapes)),
                "smape_std": smape_std,
                "all_r2s": r2_values,
                "all_smapes": smape_values,
                "simulation_hours": 48,
                "time_step_minutes": 5,
                "steps_per_run": scenario_steps,
                "confidence_level": 0.95,
                "profile_ratio_noise_std": 0.02,
                "protocol": "experiments_four_scenario_8run_48h",
                "online_learning": "not_supported_by_two_stage_hist_gradient",
            })
            scenarios[scenario] = first_trace.as_dict()
            multi_stats[scenario] = summary
        write_json(result_root / "data" / "complete_results.json", {"scenarios": scenarios})
        write_json(result_root / "data" / "multi_run_statistics.json", multi_stats)

    mismatch_specs = {
        "default": {"response_prob_bias": 1.0, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        "prob_-30%": {"response_prob_bias": 0.7, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        "prob_-20%": {"response_prob_bias": 0.8, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        "prob_-10%": {"response_prob_bias": 0.9, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        "prob_+10%": {"response_prob_bias": 1.1, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        "prob_+20%": {"response_prob_bias": 1.2, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        "prob_+30%": {"response_prob_bias": 1.3, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        "soc_10%": {"response_prob_bias": 1.0, "soc_noise_std": 0.10, "device_offline_rate": 0.015},
        "soc_20%": {"response_prob_bias": 1.0, "soc_noise_std": 0.20, "device_offline_rate": 0.015},
        "soc_30%": {"response_prob_bias": 1.0, "soc_noise_std": 0.30, "device_offline_rate": 0.015},
        "offline_5%": {"response_prob_bias": 1.0, "soc_noise_std": 0.0, "device_offline_rate": 0.05},
        "offline_10%": {"response_prob_bias": 1.0, "soc_noise_std": 0.0, "device_offline_rate": 0.10},
        "offline_15%": {"response_prob_bias": 1.0, "soc_noise_std": 0.0, "device_offline_rate": 0.15},
        "combined_mild": {"response_prob_bias": 1.1, "soc_noise_std": 0.10, "device_offline_rate": 0.05},
        "combined_moderate": {"response_prob_bias": 1.2, "soc_noise_std": 0.20, "device_offline_rate": 0.10},
        "combined_severe": {"response_prob_bias": 1.3, "soc_noise_std": 0.30, "device_offline_rate": 0.15},
        "combined_negative_mild": {"response_prob_bias": 0.9, "soc_noise_std": 0.10, "device_offline_rate": 0.05},
        "combined_negative_moderate": {"response_prob_bias": 0.8, "soc_noise_std": 0.20, "device_offline_rate": 0.10},
        "combined_negative_severe": {"response_prob_bias": 0.7, "soc_noise_std": 0.30, "device_offline_rate": 0.15},
    }
    mismatch_results = {}
    base_coefficients = np.asarray(estimator["coefficients"], dtype=float)
    mismatch_train_samples = 12000 if len(records) >= 5000 else 2000
    mismatch_train_schedule = build_experiments_training_signal_schedule(
        config, mismatch_train_samples, seed=42
    )
    mismatch_train_profiles = np.random.default_rng(42 + 73000).integers(
        0,
        int(config["simulation"]["steps_per_day"]),
        size=mismatch_train_samples,
    ).astype(int).tolist()
    mismatch_train_trace = run_trace(
        records,
        steps=mismatch_train_samples,
        trace_seed=42,
        profile_steps_override=mismatch_train_profiles,
        signal_schedule_override=mismatch_train_schedule,
    )
    mismatch_fit_mask = np.arange(mismatch_train_samples) % 4 != 0
    mismatch_runtime_estimator = _fit_nonlinear_two_stage_estimator(
        mismatch_train_trace, mismatch_fit_mask
    )
    mismatch_test_samples = 200
    mismatch_schedule = build_experiments_training_signal_schedule(
        config, mismatch_test_samples, seed=99999
    )
    mismatch_profiles = np.random.default_rng(99999 + 73000).integers(
        0,
        int(config["simulation"]["steps_per_day"]),
        size=mismatch_test_samples,
    ).astype(int).tolist()
    for variant, overrides in mismatch_specs.items():
        variant_config = copy.deepcopy(config)
        variant_config["original_model"].update(overrides)
        trace = run_trace(
            records,
            steps=mismatch_test_samples,
            trace_seed=99999,
            trace_config=variant_config,
            profile_steps_override=mismatch_profiles,
            signal_schedule_override=mismatch_schedule,
        )
        actual_variant = np.asarray(trace.accepted_control_kw, dtype=float)
        predicted_variant = _predict_estimator(
            estimator, trace, mismatch_runtime_estimator
        )
        residual = actual_variant - predicted_variant
        denominator = np.maximum(
            np.abs(actual_variant) + np.abs(predicted_variant), 1e-8
        )
        mismatch_results[variant] = {
            **overrides,
            "r2": r2_score(actual_variant, predicted_variant),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "smape": float(np.mean(2.0 * np.abs(residual) / denominator) * 100.0),
            "bias_kw": float(np.mean(residual)),
            "n_test": mismatch_test_samples,
            "n_train": mismatch_train_samples,
            "shared_test_seed": 99999,
        }
    write_json(result_root / "data" / "model_mismatch.json", mismatch_results)
    if figure2_only:
        write_figure2_scenario_statistics()
        return result_root

    zone_ids = sorted(config["zones"]["zones"])
    transfer_regions = {}
    source_zone = zone_ids[0]
    source_records = [record for record in records if record.zone_id == source_zone]
    source_only = bool(experiment.get("transfer_source_region_only", False))
    if source_only:
        source_trace = run_trace(source_records, steps=288, trace_seed=seed + 5999)
        transfer_coefficients, _, _, _ = np.linalg.lstsq(
            trace_features(source_trace),
            np.asarray(source_trace.accepted_control_kw, dtype=float),
            rcond=None,
        )
        source_prediction = trace_features(source_trace) @ transfer_coefficients
        source_entry = {
            "label": source_zone,
            "n_resources": len(source_records),
            "r2": r2_score(
                np.asarray(source_trace.accepted_control_kw, dtype=float),
                source_prediction,
            ),
        }
    else:
        transfer_coefficients = base_coefficients
        source_entry = {
            "label": "all_zones",
            "n_resources": len(records),
            "r2": r2_score(
                np.asarray(train_trace.accepted_control_kw, dtype=float),
                _predict_estimator(estimator, train_trace, runtime_estimator),
            ),
        }
    scale_transfer_by_count = bool(experiment.get("transfer_scale_by_resource_count", False))
    target_zones = [zone for zone in zone_ids if zone != source_zone][:3]
    for region_index, zone in enumerate(target_zones):
        zone_records = [record for record in records if record.zone_id == zone]
        zone_trace = run_trace(zone_records, steps=288, trace_seed=seed + 6000 + region_index)
        zone_x = trace_features(zone_trace)
        zone_y = np.asarray(zone_trace.accepted_control_kw, dtype=float)
        cold_prediction = zone_x @ transfer_coefficients
        if scale_transfer_by_count:
            cold_prediction *= len(zone_records) / max(len(records), 1)
        local_count = min(96, len(zone_y) - 1)
        adapted_coefficients, _, _, _ = np.linalg.lstsq(zone_x[:local_count], zone_y[:local_count], rcond=None)
        adapted_prediction = zone_x @ adapted_coefficients
        transfer_regions[chr(ord("B") + region_index)] = {
            "label": zone,
            "cold_start": {"r2": r2_score(zone_y, cold_prediction)},
            "adapted_nn": {"r2": r2_score(zone_y, adapted_prediction), "n_online_samples": local_count},
        }
    write_json(result_root / "data" / "cross_region_transfer.json", {
        "regions": transfer_regions,
        "region_A": source_entry,
        "protocol": "source_region_500_to_target_region_500_linear_transfer",
        "source_region": source_zone,
        "source_resource_count": len(source_records),
        "target_resource_counts": {zone: sum(record.zone_id == zone for record in records) for zone in target_zones},
        "source_region_only": source_only,
        "cold_start_count_scale": scale_transfer_by_count,
    })

    write_figure2_scenario_statistics()

    scaling: dict[str, dict[str, Any]] = {}
    correlation_conditions = _correlation_conditions(
        config,
        real_snapshot_only=real_snapshot_only,
    )
    for condition in correlation_conditions:
        label = condition["label"]
        correlation = condition["zone_correlation"]
        rows = []
        for n_index, n in enumerate(experiment["n_scaling_values"]):
            trial_values = []
            for trial in range(int(experiment["n_scaling_runs"])):
                subset = _balanced_subset(records, int(n))
                trace = run_trace(
                    subset,
                    steps=int(experiment.get("scaling_steps", 24)),
                    trace_seed=seed + 1000 + n_index * 31 + trial,
                    zone_correlation=correlation,
                )
                trial_values.append(float(np.mean(trace.accepted_control_kw)))
            values = np.asarray(trial_values, dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values))
            rows.append({"N": int(n), "mean": mean, "std": std, "cv": std / max(abs(mean), 1e-6)})
        valid_n = np.asarray([row["N"] for row in rows], dtype=float)
        valid_cv = np.asarray([max(row["cv"], 1e-9) for row in rows], dtype=float)
        slope = float(np.polyfit(np.log(valid_n), np.log(valid_cv), 1)[0])
        scaling[label] = {"scaling_data": rows, "loglog_slope": slope, "sigma_hat": float(valid_cv[0] * np.sqrt(valid_n[0]))}
    write_json(result_root / "data" / "n_scaling.json", scaling)

    directional_specs = experiment.get("directional_n_scaling", {})
    if directional_specs:
        directional: dict[str, dict[str, Any]] = {}
        for label, scenario_name in directional_specs.items():
            rows = []
            for n_index, n in enumerate(experiment["n_scaling_values"]):
                trial_values = []
                for trial in range(int(experiment["n_scaling_runs"])):
                    subset = _balanced_subset(records, int(n))
                    trace = run_trace(
                        subset,
                        steps=int(experiment.get("scaling_steps", 24)),
                        trace_seed=seed + 15000 + n_index * 31 + trial,
                        scenario=str(scenario_name),
                        zone_correlation=0.0,
                    )
                    trial_values.append(float(np.mean(trace.accepted_control_kw)))
                values = np.asarray(trial_values, dtype=float)
                mean = float(np.mean(values))
                std = float(np.std(values))
                rows.append({
                    "N": int(n),
                    "mean": mean,
                    "std": std,
                    "cv": std / max(abs(mean), 1e-6),
                    "scenario": str(scenario_name),
                })
            valid_n = np.asarray([row["N"] for row in rows], dtype=float)
            valid_cv = np.asarray([max(row["cv"], 1e-9) for row in rows], dtype=float)
            directional[str(label)] = {
                "scenario": str(scenario_name),
                "scaling_data": rows,
                "loglog_slope": float(np.polyfit(np.log(valid_n), np.log(valid_cv), 1)[0]),
                "sigma_hat": float(valid_cv[0] * np.sqrt(valid_n[0])),
            }
        write_json(result_root / "data" / "directional_n_scaling.json", directional)

    threshold_rows = {}
    n_values = [int(value) for value in experiment["n_scaling_values"]]
    train_conditions = int(experiment["n_threshold_train_conditions"])
    eval_conditions = int(experiment["n_threshold_eval_conditions"])
    train_replications = int(experiment["n_threshold_train_replications"])
    eval_replications = int(experiment["n_threshold_eval_replications"])
    profile_start = int(experiment["n_threshold_profile_start"])
    shared_conditions = bool(experiment["n_threshold_shared_conditions"])
    total_conditions = train_conditions if shared_conditions else train_conditions + eval_conditions
    max_profile_step = int(config["simulation"]["steps_per_day"]) - 1
    if profile_start < 0 or profile_start + total_conditions - 1 > max_profile_step:
        raise ValueError("N-threshold profile window exceeds one device-day")
    if min(train_conditions, eval_conditions, train_replications, eval_replications) < 2:
        raise ValueError("N-threshold conditions and replications must be at least two")

    profile_steps = list(range(profile_start, profile_start + total_conditions))
    reference_subset = _balanced_subset(records, max(n_values))
    signal_schedule = build_condition_signal_schedule(
        reference_subset,
        config,
        profile_steps,
        list(experiment["n_threshold_signal_scenarios"]),
        seed=seed + 2900,
    )
    train_profile_steps = profile_steps[:train_conditions]
    train_signal_schedule = signal_schedule[:train_conditions]
    if shared_conditions:
        eval_profile_steps = train_profile_steps
        eval_signal_schedule = train_signal_schedule
    else:
        eval_profile_steps = profile_steps[train_conditions:]
        eval_signal_schedule = signal_schedule[train_conditions:]

    def _snapshot_replicates(
        subset: list[DeviceDay],
        profile_window: list[int],
        signal_window: list[list[tuple[int, int]]],
        replications: int,
        seed_offset: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = None
        responses = []
        for replicate in range(replications):
            profile_matrix = None
            if snapshot_sampling == "independent":
                profile_matrix = _independent_profile_matrix(
                    subset,
                    len(profile_window),
                    seed_offset + replicate + 91000,
                )
            trace = simulate_day(
                subset,
                config,
                steps=len(profile_window),
                seed=seed_offset + replicate,
                profile_steps=profile_window,
                signal_overrides=signal_window,
                profile_steps_by_record=profile_matrix,
                reset_each_step=True,
            )
            if features is None:
                features = trace_features(trace)
            responses.append(np.asarray(trace.accepted_control_kw, dtype=float))
        if features is None:
            raise RuntimeError("N-threshold snapshot generation produced no features")
        return features, np.asarray(responses, dtype=float)

    n_bootstrap = int(experiment["n_threshold_bootstrap"])
    for n_index, n in enumerate(n_values):
        subset = _balanced_subset(records, int(n))
        train_x, train_y = _snapshot_replicates(
            subset,
            train_profile_steps,
            train_signal_schedule,
            train_replications,
            seed + 3000 + n_index * 100,
        )
        valid_x, valid_y = _snapshot_replicates(
            subset,
            eval_profile_steps,
            eval_signal_schedule,
            eval_replications,
            seed + 4000 + n_index * 100,
        )
        train_mean = np.mean(train_y, axis=0)
        coefficients, _, _, _ = np.linalg.lstsq(train_x, train_mean, rcond=None)
        linear_prediction_by_condition = valid_x @ coefficients
        predicted_by_condition = train_mean
        actual_n = valid_y.T.reshape(-1)
        predicted_n = np.repeat(predicted_by_condition, eval_replications)
        r2_n = r2_score(actual_n, predicted_n)

        groups = [valid_y[:, index] for index in range(eval_conditions)]
        within_cvs = [
            float(np.std(group) / max(abs(float(np.mean(group))), 1e-6))
            for group in groups
        ]
        boot_rng = np.random.default_rng(seed + 7000 + n_index)
        boot_r2 = []
        for _ in range(n_bootstrap):
            indices = boot_rng.integers(0, eval_conditions, size=eval_conditions)
            boot_actual = np.concatenate([valid_y[:, index] for index in indices])
            boot_prediction = np.repeat(predicted_by_condition[indices], eval_replications)
            boot_r2.append(r2_score(boot_actual, boot_prediction))
        r2_ci_lower, r2_ci_upper = np.percentile(boot_r2, [2.5, 97.5])
        threshold_rows[str(int(n))] = {
            "N": int(n),
            "r2": float(r2_n),
            "r2_ci_lower": float(r2_ci_lower),
            "r2_ci_upper": float(r2_ci_upper),
            "picp": 0.0,
            "within_signal_cv": float(np.mean(within_cvs)),
            "rmse_kw": float(np.sqrt(np.mean((actual_n - predicted_n) ** 2))),
            "mae_kw": float(np.mean(np.abs(actual_n - predicted_n))),
            "signal_conditions": eval_conditions,
            "train_replications": train_replications,
            "eval_replications": eval_replications,
            "linear_model_r2": float(r2_score(actual_n, np.repeat(linear_prediction_by_condition, eval_replications))),
        }
    threshold_n95 = next((int(row["N"]) for row in threshold_rows.values() if row["r2"] >= 0.95), None)
    write_json(result_root / "data" / "n_threshold.json", {
        "N_values": n_values,
        "per_N": threshold_rows,
        "threshold_N_95": threshold_n95,
        "protocol": "fixed_real_snapshot_multi_replication_shared_conditions",
        "protocol_details": {
            "train_conditions": train_conditions,
            "eval_conditions": eval_conditions,
            "train_replications": train_replications,
            "eval_replications": eval_replications,
            "profile_steps": profile_steps,
            "same_device_subset_within_N": True,
            "reset_initial_state_each_snapshot": True,
            "fixed_signal_schedule": True,
            "shared_train_eval_conditions": shared_conditions,
            "signal_scenarios": list(experiment["n_threshold_signal_scenarios"]),
            "network_feedback_preserved": bool(config["control"]["network_feedback"]),
            "snapshot_sampling": snapshot_sampling,
            "independent_profile_point_per_resource": snapshot_sampling == "independent",
        },
    })
    capacities = np.asarray([record.capacity_kwh for record in records], dtype=float)
    capacity_cv = float(np.std(capacities) / max(np.mean(capacities), 1e-9))
    write_json(result_root / "data" / "heterogeneity_lookup_table.json", {
        "metadata": {"protocol": "empirical_device_day_capacity_cv", "cv_values": [capacity_cv], "N_values": [int(n) for n in experiment["n_scaling_values"]]},
        "grid": {"r2": [[threshold_rows[str(int(n))]["r2"]] for n in n_values]},
    })

    rho = {}
    base_n = expected_resources
    for condition in correlation_conditions:
        label = condition["label"]
        rho_value = condition["nominal_rho"]
        base_cv = float(scaling[label]["scaling_data"][-1]["cv"])
        if real_snapshot_only:
            rho[label] = {
                "rho_definition": "no synthetic correlation injected; estimate empirically from recorded snapshots",
                "rho_within": None,
                "N": base_n,
                "N_eff": None,
                "cv_sqrt_neff": None,
                "cv_at_max_N": base_cv,
            }
        elif rho_value is None:
            rho[label] = {
                "rho_definition": "nominal correlation not supplied; use empirical resource diagnostic",
                "rho_within": None,
                "N": base_n,
                "N_eff": None,
                "cv_sqrt_neff": None,
                "cv_at_max_N": base_cv,
            }
        else:
            n_eff = base_n / (1.0 + rho_value * (base_n / zone_count - 1.0)) if rho_value else float(base_n)
            rho[label] = {
                "rho_within": rho_value,
                "N": base_n,
                "N_eff": n_eff,
                "cv_sqrt_neff": base_cv * np.sqrt(max(n_eff, 1.0)),
            }
    write_json(result_root / "data" / "rho_sensitivity.json", rho)

    validation_curtail = _trace_curtailment(validation_day_trace)
    baselines = {
        "real_profile_no_eps": {"curtailment_mwh": validation_curtail[0], "accepted_control_mwh": 0.0},
        "eps_network_constrained": {"curtailment_mwh": validation_curtail[1], "accepted_control_mwh": float(np.sum(np.maximum(validation_day_trace.accepted_control_kw, 0)) * 5 / 60 / 1000)},
    }
    write_json(result_root / "data" / "curtailment_baselines.json", baselines)
    write_json(result_root / "data" / "curtailment_sensitivity.json", {
        "rows": [{"pv_multiplier": value, "baseline_curtailment_mwh": validation_curtail[0] * value, "eps_curtailment_mwh": validation_curtail[1] * value} for value in [0.8, 1.0, 1.2]],
        "protocol": "real_nextgen_pv_load_plus_original_eps",
    })
    _write_network_metrics(result_root, validation_day_trace, validation_record_batches[0])
    write_json(result_root / "data" / "population_summary.json", {
        "source_devices": pool.source_count,
        "device_days": pool.metadata_count,
        "interpolated_missing_values": pool.filled_values,
        "simulated_resources": len(records),
        "sampled_device_days_total": len(sampled_records),
        "validation_samples": validation_samples,
        "validation_resource_batches": len(validation_traces),
        "resources_per_validation_batch": len(validation_record_batches[0]),
        "resources_by_zone": {zone: sum(record.zone_id == zone for record in records) for zone in sorted({r.zone_id for r in records})},
        "samples_per_device_day": 288,
        "sampling_interval_minutes": 5,
    })
    write_json(result_root / "data" / "protocol_metadata.json", {
        "mode": mode,
        "snapshot_protocol": snapshot_protocol,
        "snapshot_sampling": snapshot_sampling,
        "real_snapshot_only": real_snapshot_only,
        "network_feedback": bool(config["control"]["network_feedback"]),
        "same_protocol_group": "ieee33_weak_vs_stress_20260714",
        "network_layer": "diagnostic_only" if mode == "weak_correlation" else "active_dispatch_constraint",
        "shared_settings": {
            "random_seed": seed,
            "resources_per_zone": per_zone,
            "simulated_resources": expected_resources,
            "validation_samples": validation_samples,
            "n_scaling_values": [int(value) for value in experiment["n_scaling_values"]],
            "n_scaling_runs": int(experiment["n_scaling_runs"]),
            "correlation_levels": [
                condition["nominal_rho"]
                for condition in correlation_conditions
                if condition["nominal_rho"] is not None
            ],
            "correlation_conditions": correlation_conditions,
            "training_scenarios": training_scenarios,
            "validation_scenarios": validation_scenarios,
            "estimator_signal_sampling": estimator_signal_sampling,
            "scenario_steps": int(experiment["scenario_steps"]),
            "snapshot_state_reset": snapshot_protocol,
            "snapshot_profile_mode": str(experiment.get("snapshot_profile_mode", "sequential_day")),
            "snapshot_profile_seed_offset": int(experiment.get("snapshot_profile_seed_offset", 0)),
            "snapshot_profile_steps": snapshot_profile_steps if snapshot_protocol else [],
            "independent_profile_point_per_resource": snapshot_sampling == "independent",
            "synthetic_zone_correlation_injected": not real_snapshot_only,
        },
    })

    print(f"completed mode={mode} {len(records)} resources from {pool.metadata_count} device-days in {time.time() - started:.1f}s")
    return result_root


def _run_level_response_statistics(
    response_repetitions: list[np.ndarray],
) -> dict[str, Any]:
    run_means = np.asarray([
        float(np.mean(np.sum(repetition, axis=1)))
        for repetition in response_repetitions
    ])
    mean = float(np.mean(run_means))
    std = float(np.std(run_means))
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / max(abs(mean), 1e-12)),
        "run_means": run_means,
    }


def run_resource_response_diagnostics(
    config_path: str | Path | None = None,
    *,
    mode: str,
    result_root: str | Path,
    snapshot_sampling: str,
    replications: int = 8,
    weak_rho_threshold: float = 0.01,
    zone_correlation: float = 0.0,
    original_global_shock_std: float | None = None,
    original_regional_shock_std: float | None = None,
    network_feedback_override: bool | None = None,
    condition_label: str = "uncontrolled",
    diagnostic_resources: int | None = None,
    diagnostic_steps: int | None = None,
    diagnostic_scenarios: list[str] | None = None,
) -> Path:
    if mode not in {"weak_correlation", "network_stress"}:
        raise ValueError(f"unsupported experiment mode: {mode}")
    if snapshot_sampling not in {"aligned", "independent"}:
        raise ValueError(f"unsupported snapshot sampling: {snapshot_sampling}")
    config = copy.deepcopy(load_config(config_path))
    config["control"]["network_feedback"] = (
        bool(network_feedback_override)
        if network_feedback_override is not None
        else bool(config["experiment"].get("force_network_feedback", mode == "network_stress"))
    )
    pool = load_device_day_pool(config)
    population = config["population"]
    configured_per_zone = int(population["resources_per_zone"])
    zone_count = len(config["zones"]["zones"])
    per_zone = (
        int(np.ceil(int(diagnostic_resources) / zone_count))
        if diagnostic_resources is not None
        else configured_per_zone
    )
    validation_batch_count = int(config["experiment"]["validation_batches"])
    if bool(population.get("unique_source_per_batch", False)):
        requested_resources = int(diagnostic_resources or per_zone * zone_count)
        records = pool.sample_unique_sources(
            requested_resources,
            seed=int(config["simulation"]["random_seed"]) + int(population["seed_offset"]),
        )
    else:
        sampled_records = pool.sample(
            per_zone=per_zone * (1 + validation_batch_count),
            seed=int(config["simulation"]["random_seed"]) + int(population["seed_offset"]),
            with_replacement=bool(population["with_replacement"]),
        )
        records = _split_zone_batches(
            sampled_records,
            per_zone=per_zone,
            batch_count=1 + validation_batch_count,
        )[0]
        if diagnostic_resources is not None:
            records = _balanced_subset(records, int(diagnostic_resources))
    steps = int(diagnostic_steps or config["experiment"]["n_threshold_eval_conditions"])
    seed = int(config["simulation"]["random_seed"])
    if str(config["experiment"].get("snapshot_profile_mode", "")).lower() == "random_aligned":
        profile_rng = np.random.default_rng(
            seed + int(config["experiment"].get("snapshot_profile_seed_offset", 0)) + 13000
        )
        profile_steps = profile_rng.integers(
            0,
            int(config["simulation"]["steps_per_day"]),
            size=steps,
        ).astype(int).tolist()
    else:
        profile_steps = list(range(steps))
    scenarios = list(diagnostic_scenarios or config["experiment"]["n_threshold_signal_scenarios"])
    signal_schedule = build_condition_signal_schedule(
        records,
        config,
        profile_steps,
        scenarios,
        seed=seed + 12900,
    )
    response_repetitions: list[np.ndarray] = []
    for replicate in range(int(replications)):
        sink: list[list[float]] = []
        profile_matrix = None
        if snapshot_sampling == "independent":
            profile_matrix = _independent_profile_matrix(
                records,
                steps,
                seed + 13000 + replicate * 101,
            )
        simulate_day(
            records,
            config,
            steps=steps,
            seed=seed + 13100 + replicate,
            profile_steps=profile_steps,
            signal_overrides=signal_schedule,
            profile_steps_by_record=profile_matrix,
            zone_correlation=zone_correlation,
            original_global_shock_std=original_global_shock_std,
            original_regional_shock_std=original_regional_shock_std,
            original_response_seed=(
                1000 + replicate
                if original_global_shock_std is not None
                or original_regional_shock_std is not None
                else None
            ),
            resource_response_sink=sink,
            reset_each_step=True,
        )
        response_repetitions.append(np.asarray(sink, dtype=float))

    response_matrix = np.concatenate(response_repetitions, axis=0)
    zone_array = np.asarray([record.zone_id for record in records])
    step_aggregate_response = np.sum(response_matrix, axis=1)
    run_statistics = _run_level_response_statistics(response_repetitions)
    run_mean_responses = run_statistics["run_means"]
    aggregate_mean = float(run_statistics["mean"])
    aggregate_std = float(run_statistics["std"])
    aggregate_cv = float(run_statistics["cv"])
    step_aggregate_mean = float(np.mean(step_aggregate_response))
    step_aggregate_std = float(np.std(step_aggregate_response))
    step_aggregate_cv = step_aggregate_std / max(abs(step_aggregate_mean), 1e-12)

    def _correlation_values(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        resource_std = np.std(matrix, axis=0)
        valid = resource_std > 1e-12
        if np.sum(valid) < 2:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float), int(np.sum(valid))
        correlation = np.corrcoef(matrix[:, valid], rowvar=False)
        upper_i, upper_j = np.triu_indices(correlation.shape[0], k=1)
        pair_values = correlation[upper_i, upper_j]
        valid_zone_ids = zone_array[valid]
        same_zone = valid_zone_ids[upper_i] == valid_zone_ids[upper_j]
        return pair_values, pair_values[same_zone], pair_values[~same_zone], int(np.sum(valid))

    n_resources = len(records)
    raw_pair_values, raw_within_values, raw_between_values, valid_count = _correlation_values(response_matrix)
    repeated_matrix = response_matrix.reshape(int(replications), steps, n_resources)
    conditional_residual_matrix = repeated_matrix - np.mean(repeated_matrix, axis=0, keepdims=True)
    residual_pair_values, residual_within_values, residual_between_values, residual_valid_count = _correlation_values(
        conditional_residual_matrix.reshape(-1, n_resources)
    )

    def _summary(values: np.ndarray) -> dict[str, float | int | None]:
        if values.size == 0:
            return {"count": 0, "mean": None, "median": None, "q05": None, "q95": None}
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q05": float(np.percentile(values, 5)),
            "q95": float(np.percentile(values, 95)),
        }

    mean_rho = float(np.mean(raw_pair_values)) if raw_pair_values.size else 0.0
    residual_mean_rho = float(np.mean(residual_pair_values)) if residual_pair_values.size else 0.0
    n_eff = n_resources / (1.0 + max(n_resources - 1, 0) * mean_rho) if mean_rho > -1.0 / max(n_resources - 1, 1) else None
    residual_n_eff = (
        n_resources / (1.0 + max(n_resources - 1, 0) * residual_mean_rho)
        if residual_mean_rho > -1.0 / max(n_resources - 1, 1)
        else None
    )
    output = Path(result_root) / "data"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "resource_response_diagnostics.npz",
        accepted_response_kw=response_matrix,
        resource_zone_ids=np.asarray([record.zone_id for record in records]),
        resource_device_day_ids=np.asarray([record.device_id for record in records]),
    )
    write_json(output / "resource_response_diagnostics.json", {
        "protocol": "fixed_real_snapshot_empirical_resource_correlation",
        "mode": mode,
        "condition_label": condition_label,
        "zone_correlation_parameter": float(zone_correlation),
        "original_global_shock_std": (
            None if original_global_shock_std is None else float(original_global_shock_std)
        ),
        "original_regional_shock_std": (
            None if original_regional_shock_std is None else float(original_regional_shock_std)
        ),
        "correlation_application": (
            "original_EPSSimulator_persistent_power_modulation"
            if original_global_shock_std is not None or original_regional_shock_std is not None
            else "legacy_post_response_zone_multiplier"
        ),
        "snapshot_sampling": snapshot_sampling,
        "network_feedback": bool(config["control"]["network_feedback"]),
        "synthetic_zone_correlation_injected": bool(
            zone_correlation > 0
            or float(original_global_shock_std or 0.0) > 0
            or float(original_regional_shock_std or 0.0) > 0
        ),
        "resources": n_resources,
        "diagnostic_resource_target": diagnostic_resources,
        "observations": int(response_matrix.shape[0]),
        "steps_per_replication": steps,
        "replications": int(replications),
        "response_field": "accepted_control_kw",
        "aggregate_response_mean_kw": aggregate_mean,
        "aggregate_response_std_kw": aggregate_std,
        "aggregate_response_cv": float(aggregate_cv),
        "aggregate_response_statistic": "cv_across_replication_mean_responses",
        "run_mean_response_kw": run_mean_responses.tolist(),
        "step_aggregate_response_mean_kw": step_aggregate_mean,
        "step_aggregate_response_std_kw": step_aggregate_std,
        "step_aggregate_response_cv": float(step_aggregate_cv),
        "resource_variance_nonzero_count": valid_count,
        "all_pairwise_correlation": _summary(raw_pair_values),
        "within_zone_correlation": _summary(raw_within_values),
        "between_zone_correlation": _summary(raw_between_values),
        "mean_pairwise_rho": mean_rho,
        "pairwise_N_eff": float(n_eff) if n_eff is not None else None,
        "conditional_residual_definition": "subtract each resource's mean response for the same fixed EPS condition across repetitions",
        "conditional_residual_variance_nonzero_count": residual_valid_count,
        "conditional_residual_all_pairwise_correlation": _summary(residual_pair_values),
        "conditional_residual_within_zone_correlation": _summary(residual_within_values),
        "conditional_residual_between_zone_correlation": _summary(residual_between_values),
        "conditional_residual_mean_pairwise_rho": residual_mean_rho,
        "conditional_residual_pairwise_N_eff": float(residual_n_eff) if residual_n_eff is not None else None,
        "weak_correlation_definition": {
            "metric": "absolute mean conditional residual pairwise rho",
            "threshold": float(weak_rho_threshold),
            "status": bool(abs(residual_mean_rho) <= weak_rho_threshold),
        },
        "data_file": "resource_response_diagnostics.npz",
    })
    return output / "resource_response_diagnostics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="IEEE 33 NextGen device-day experiment")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["weak_correlation", "network_stress"], default="network_stress")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--snapshot-protocol", action="store_true")
    parser.add_argument("--snapshot-sampling", choices=["aligned", "independent"], default="aligned")
    parser.add_argument("--real-snapshot-only", action="store_true")
    parser.add_argument("--resource-response-diagnostics", action="store_true")
    args = parser.parse_args()
    result = run_default(
        args.config,
        mode=args.mode,
        results_root=args.results_root,
        snapshot_protocol=args.snapshot_protocol,
        snapshot_sampling=args.snapshot_sampling,
        real_snapshot_only=args.real_snapshot_only,
    )
    if args.resource_response_diagnostics:
        run_resource_response_diagnostics(
            args.config,
            mode=args.mode,
            result_root=result,
            snapshot_sampling=args.snapshot_sampling,
        )


if __name__ == "__main__":
    main()
