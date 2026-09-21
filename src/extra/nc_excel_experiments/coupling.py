from __future__ import annotations

from typing import Any, Sequence

import numpy as np

DIURNAL_HOURS = 24


def _hour_of_day_design(step_indices: np.ndarray, steps_per_day: int) -> np.ndarray:
    hours = (step_indices * DIURNAL_HOURS // max(steps_per_day, 1)) % DIURNAL_HOURS
    columns = [np.ones(len(step_indices))]
    for hour in range(DIURNAL_HOURS):
        indicator = (hours == hour).astype(float)
        if indicator.sum() > 0:
            columns.append(indicator)
    return np.column_stack(columns)


def device_load_matrix(records: Sequence[Any]) -> np.ndarray:
    steps = min(len(record.load_kw) for record in records)
    return np.column_stack([
        np.asarray(record.load_kw, dtype=float)[:steps] for record in records
    ])


def diurnal_residual(records: Sequence[Any], steps_per_day: int) -> np.ndarray:
    power = device_load_matrix(records)
    design = _hour_of_day_design(np.arange(power.shape[0]), steps_per_day)
    coefficients, _, _, _ = np.linalg.lstsq(design, power, rcond=None)
    return power - design @ coefficients


def rank_availability(residual: np.ndarray) -> np.ndarray:
    steps = residual.shape[0]
    order = np.argsort(np.argsort(residual, axis=0), axis=0)
    percentile = (order + 0.5) / steps
    return 1.0 - percentile


def availability_at_steps(availability: np.ndarray, profile_steps: Sequence[int]) -> np.ndarray:
    indices = np.clip(np.asarray(profile_steps, dtype=int), 0, availability.shape[0] - 1)
    return availability[indices]


def decoupled_copy(availability: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(availability)
    for device in range(availability.shape[1]):
        shuffled[:, device] = rng.permutation(availability[:, device])
    return shuffled


def mean_pairwise_rho(matrix: np.ndarray) -> float:
    deviation = np.std(matrix, axis=0)
    keep = deviation > 1e-12
    matrix = matrix[:, keep]
    device_count = matrix.shape[1]
    if device_count < 2:
        return 0.0
    standardized = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
    row_sums = np.sum(standardized, axis=1)
    numerator = float(np.mean(row_sums * row_sums) - device_count)
    return numerator / (device_count * (device_count - 1))


def broadcast_residual_rho(
    resource: np.ndarray, basis: np.ndarray, broadcast_columns: int
) -> float:
    design = basis[:, :broadcast_columns]
    values = []
    for replication in range(resource.shape[0]):
        response = resource[replication]
        coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        values.append(mean_pairwise_rho(response - design @ coefficients))
    return float(np.mean(values)) if values else 0.0
