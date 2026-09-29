from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np


def capacity_vector(records: Sequence[Any]) -> np.ndarray:
    return np.asarray([float(record.capacity_kwh) for record in records], dtype=float)


def power_vector(records: Sequence[Any]) -> np.ndarray:
    return np.asarray([float(record.peak_power_kw) for record in records], dtype=float)


def effective_capacity_count(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    squared = float(np.sum(weights * weights))
    if squared <= 0.0:
        return 0.0
    return total * total / squared


def lindeberg_share(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    squared = weights * weights
    total = float(np.sum(squared))
    if total <= 0.0:
        return 0.0
    return float(np.max(squared) / total)


def inject_dominant(
    records: list[Any],
    *,
    count: int,
    ratio: float,
    seed: int,
) -> tuple[list[Any], np.ndarray]:
    if count <= 0 or ratio <= 1.0:
        return list(records), np.asarray([], dtype=int)
    if count > len(records):
        raise ValueError(f"cannot promote {count} of {len(records)} records")
    chosen = np.random.default_rng(seed).choice(len(records), size=count, replace=False)
    promoted = list(records)
    for index in chosen:
        record = records[int(index)]
        promoted[int(index)] = replace(
            record,
            capacity_kwh=float(record.capacity_kwh) * ratio,
            peak_power_kw=float(record.peak_power_kw) * ratio,
            soc_kwh=np.asarray(record.soc_kwh, dtype=float) * ratio,
        )
    return promoted, np.asarray(sorted(int(value) for value in chosen), dtype=int)


def imbalance_diagnostics(records: Sequence[Any]) -> dict[str, Any]:
    capacity = capacity_vector(records)
    power = power_vector(records)
    return {
        "N": len(records),
        "capacity_total_kwh": float(np.sum(capacity)),
        "capacity_max_kwh": float(np.max(capacity)) if capacity.size else 0.0,
        "capacity_median_kwh": float(np.median(capacity)) if capacity.size else 0.0,
        "N_capacity_eff": effective_capacity_count(capacity),
        "N_power_eff": effective_capacity_count(power),
        "lindeberg_share_capacity": lindeberg_share(capacity),
        "lindeberg_share_power": lindeberg_share(power),
        "capacity_count_ratio": (
            effective_capacity_count(capacity) / len(records) if records else 0.0
        ),
    }
