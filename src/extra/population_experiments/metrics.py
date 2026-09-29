from __future__ import annotations

from typing import Iterable

import numpy as np


EPSILON = 1e-12


def nrmse(actual: np.ndarray, target: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    scale = float(np.percentile(np.abs(target), 95)) if target.size else 0.0
    return float(np.sqrt(np.mean((actual - target) ** 2)) / max(scale, EPSILON))


def sign_consistency(actual: np.ndarray, target: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.abs(target) > EPSILON
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.sign(actual[valid]) == np.sign(target[valid])))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def conditional_mean_pairwise_rho(response: np.ndarray) -> float:
    values = np.asarray(response, dtype=float)
    if values.ndim != 3:
        raise ValueError("response must have replication, condition and resource dimensions")
    residual = values - np.mean(values, axis=0, keepdims=True)
    matrix = residual.reshape(-1, residual.shape[-1])
    standard_deviation = np.std(matrix, axis=0)
    valid = standard_deviation > EPSILON
    matrix = matrix[:, valid]
    resource_count = matrix.shape[1]
    if resource_count < 2:
        return 0.0
    standardized = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
    row_sums = np.sum(standardized, axis=1)
    numerator = float(np.mean(row_sums * row_sums) - resource_count)
    return numerator / (resource_count * (resource_count - 1))


def effective_n(n: float, rho: float) -> float:
    return float(n / (1.0 + max(n - 1.0, 0.0) * max(float(rho), 0.0)))


def loglog_slope(n_values: Iterable[float], values: Iterable[float]) -> float:
    n_array = np.asarray(list(n_values), dtype=float)
    value_array = np.asarray(list(values), dtype=float)
    valid = (n_array > 0) & (value_array > 0) & np.isfinite(value_array)
    if np.sum(valid) < 2:
        return float("nan")
    return float(np.polyfit(np.log(n_array[valid]), np.log(value_array[valid]), 1)[0])


def bootstrap_slope_ci(
    trials_by_n: dict[int, list[float]],
    *,
    seed: int,
    samples: int = 1000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_values = sorted(trials_by_n)
    slopes: list[float] = []
    for _ in range(samples):
        cvs: list[float] = []
        for n in n_values:
            values = np.asarray(trials_by_n[n], dtype=float)
            draw = values[rng.integers(0, len(values), size=len(values))]
            cvs.append(float(np.std(draw) / max(abs(float(np.mean(draw))), EPSILON)))
        slopes.append(loglog_slope(n_values, cvs))
    return tuple(float(value) for value in np.nanpercentile(slopes, [2.5, 97.5]))


def directional_delivery(desired: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    desired = np.asarray(desired, dtype=float)
    accepted = np.asarray(accepted, dtype=float)
    same_direction = np.sign(desired) == np.sign(accepted)
    return np.where(same_direction, np.minimum(np.abs(accepted), np.abs(desired)), 0.0)


def network_delivery_metrics(
    desired: np.ndarray,
    accepted: np.ndarray,
    diagnostics: list[dict[str, float | int]],
    *,
    voltage_min: float = 0.95,
    voltage_max: float = 1.05,
) -> dict[str, float]:
    desired = np.asarray(desired, dtype=float)
    accepted = np.asarray(accepted, dtype=float)
    delivered = directional_delivery(desired, accepted)
    denominator = float(np.sum(np.abs(desired)))
    acceptance = float(np.sum(delivered) / max(denominator, EPSILON))
    minimum_voltage = min(float(row["minimum_voltage_pu"]) for row in diagnostics)
    maximum_voltage = max(float(row["maximum_voltage_pu"]) for row in diagnostics)
    maximum_branch = max(float(row["maximum_branch_loading"]) for row in diagnostics)
    maximum_transformer = max(float(row["transformer_loading"]) for row in diagnostics)
    return {
        "network_acceptance_ratio": acceptance,
        "clipping_index": float(max(0.0, 1.0 - acceptance)),
        "voltage_margin": float(min(minimum_voltage - voltage_min, voltage_max - maximum_voltage)),
        "line_margin": float(1.0 - maximum_branch),
        "transformer_margin": float(1.0 - maximum_transformer),
        "unsafe_step_fraction": float(np.mean([
            int(row["overloaded_branches"]) > 0 or int(row["voltage_violations"]) > 0
            for row in diagnostics
        ])),
        "mean_network_scale": float(np.mean([float(row["network_scale"]) for row in diagnostics])),
        "fully_blocked_fraction": float(np.mean([float(row["network_scale"]) <= 0.05 for row in diagnostics])),
    }


def failure_threshold(strengths: Iterable[float], probabilities: Iterable[float], level: float = 0.5) -> float | None:
    x = np.asarray(list(strengths), dtype=float)
    probability = np.maximum.accumulate(np.asarray(list(probabilities), dtype=float))
    order = np.argsort(x)
    x = x[order]
    probability = probability[order]
    crossing = np.flatnonzero(probability >= level)
    if not len(crossing):
        return None
    index = int(crossing[0])
    if index == 0 or probability[index] == probability[index - 1]:
        return float(x[index])
    weight = (level - probability[index - 1]) / (probability[index] - probability[index - 1])
    return float(x[index - 1] + weight * (x[index] - x[index - 1]))
