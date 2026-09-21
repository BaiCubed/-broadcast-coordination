from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

ARCHITECTURES = ("full_state", "aggregate_feedback", "broadcast_only")

SOC_BINS = 20


class SocRecorder:

    def __init__(self, count: int, *, soc_min: float, soc_max: float, bins: int = SOC_BINS):
        self.count = int(count)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.edges = np.linspace(0.0, 1.0, bins + 1)
        self.mean: list[float] = []
        self.std: list[float] = []
        self.minimum: list[float] = []
        self.maximum: list[float] = []
        self.floor_fraction: list[float] = []
        self.ceiling_fraction: list[float] = []
        self.histogram: list[np.ndarray] = []
        self.daily_snapshots: list[np.ndarray] = []

    def observe(self, soc: np.ndarray) -> None:
        soc = np.asarray(soc, dtype=float)
        self.mean.append(float(np.mean(soc)))
        self.std.append(float(np.std(soc)))
        self.minimum.append(float(np.min(soc)))
        self.maximum.append(float(np.max(soc)))
        self.floor_fraction.append(float(np.mean(soc <= self.soc_min + 0.01)))
        self.ceiling_fraction.append(float(np.mean(soc >= self.soc_max - 0.01)))
        counts, _ = np.histogram(soc, bins=self.edges)
        self.histogram.append(counts.astype(np.int32))

    def snapshot(self, soc: np.ndarray) -> None:
        self.daily_snapshots.append(np.asarray(soc, dtype=float).copy())

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "soc_mean": np.asarray(self.mean, dtype=float),
            "soc_std": np.asarray(self.std, dtype=float),
            "soc_min": np.asarray(self.minimum, dtype=float),
            "soc_max": np.asarray(self.maximum, dtype=float),
            "floor_fraction": np.asarray(self.floor_fraction, dtype=float),
            "ceiling_fraction": np.asarray(self.ceiling_fraction, dtype=float),
            "soc_histogram": np.asarray(self.histogram, dtype=np.int32),
            "soc_bin_edges": self.edges,
            "daily_soc_snapshots": (
                np.asarray(self.daily_snapshots, dtype=float)
                if self.daily_snapshots else np.zeros((0, self.count))
            ),
        }


class _ArchitecturePolicy:

    def __init__(
        self,
        records: Sequence[Any],
        *,
        recorder: SocRecorder,
        soc_min: float,
        soc_max: float,
    ) -> None:
        self.count = len(records)
        self.recorder = recorder
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.zone_order = sorted({record.zone_id for record in records})
        self.zone_index = {zone: index for index, zone in enumerate(self.zone_order)}
        self.record_zone_index = np.asarray(
            [self.zone_index[record.zone_id] for record in records], dtype=int
        )

    def _discharge(self, signals: list[Any]) -> np.ndarray:
        by_zone = np.asarray([signal.supply_demand >= 8 for signal in signals])
        return by_zone[self.record_zone_index]

    def __call__(self, step: int, signals, desired_kw: np.ndarray, adapter) -> np.ndarray:
        soc = adapter.battery_states()
        self.recorder.observe(soc)
        return self._decide(step, signals, np.asarray(desired_kw, dtype=float), soc)

    def _decide(self, step, signals, desired_kw: np.ndarray, soc: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BroadcastOnlyPolicy(_ArchitecturePolicy):

    def _decide(self, step, signals, desired_kw, soc):
        return desired_kw


class AggregateFeedbackPolicy(_ArchitecturePolicy):

    def __init__(self, *args, setpoint: float = 0.5, gain: float = 2.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setpoint = float(setpoint)
        self.gain = float(gain)

    def _decide(self, step, signals, desired_kw, soc):
        error = float(np.mean(soc)) - self.setpoint
        discharge = self._discharge(signals)
        discharge_gain = float(np.clip(1.0 + self.gain * error, 0.0, 1.5))
        charge_gain = float(np.clip(1.0 - self.gain * error, 0.0, 1.5))
        return desired_kw * np.where(discharge, discharge_gain, charge_gain)


class FullStatePolicy(_ArchitecturePolicy):

    def __init__(self, *args, floor: float = 0.15, boost_cap: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.floor = float(floor)
        self.boost_cap = float(boost_cap)

    def _decide(self, step, signals, desired_kw, soc):
        discharge = self._discharge(signals)
        headroom = np.where(discharge, soc - self.soc_min, self.soc_max - soc)
        eligible = headroom > self.floor
        requested = float(np.sum(np.abs(desired_kw)))
        gated = desired_kw * eligible
        delivered = float(np.sum(np.abs(gated)))
        if delivered <= 1e-12 or requested <= 1e-12:
            return gated
        boost = min(requested / delivered, self.boost_cap)
        return gated * boost


_POLICY_TYPES = {
    "full_state": FullStatePolicy,
    "aggregate_feedback": AggregateFeedbackPolicy,
    "broadcast_only": BroadcastOnlyPolicy,
}


def build_architecture(
    architecture: str,
    records: Sequence[Any],
    *,
    recorder: SocRecorder,
    soc_min: float,
    soc_max: float,
    options: dict[str, Any] | None = None,
) -> _ArchitecturePolicy:
    if architecture not in _POLICY_TYPES:
        raise ValueError(f"unknown architecture: {architecture}")
    accepted = {
        "full_state": ("floor", "boost_cap"),
        "aggregate_feedback": ("setpoint", "gain"),
        "broadcast_only": (),
    }[architecture]
    extra = {key: value for key, value in (options or {}).items() if key in accepted}
    return _POLICY_TYPES[architecture](
        records, recorder=recorder, soc_min=soc_min, soc_max=soc_max, **extra
    )


def daily_fleet_sequence(
    pool: Any, fleet: Sequence[Any], days: int, *, seed: int
) -> list[list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for record in pool.records:
        grouped.setdefault(record.source_device_id, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: int(item.day_id))
    rng = np.random.default_rng(seed)
    offsets = {
        record.source_device_id: int(rng.integers(0, max(len(grouped[record.source_device_id]), 1)))
        for record in fleet
    }
    sequence: list[list[Any]] = []
    for day in range(days):
        batch: list[Any] = []
        for record in fleet:
            candidates = grouped[record.source_device_id]
            index = (offsets[record.source_device_id] + day) % len(candidates)
            batch.append(candidates[index])
        sequence.append(batch)
    return sequence


def set_initial_soc(records: Sequence[Any], soc: np.ndarray) -> list[Any]:
    updated: list[Any] = []
    for index, record in enumerate(records):
        profile = np.asarray(record.soc_kwh, dtype=float).copy()
        profile[0] = float(soc[index]) * float(record.capacity_kwh)
        updated.append(replace(record, soc_kwh=profile))
    return updated


def carry_soc(records: Sequence[Any], soc: np.ndarray) -> list[Any]:
    return set_initial_soc(records, soc)


def spread_initial_soc(
    records: Sequence[Any], *, low: float, high: float, seed: int
) -> tuple[list[Any], dict[str, float]]:
    soc, diagnostics = draw_initial_soc(len(records), low=low, high=high, seed=seed)
    return set_initial_soc(records, soc), diagnostics


def draw_initial_soc(
    count: int, *, low: float, high: float, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    soc = rng.uniform(float(low), float(high), int(count))
    return soc, {
        "initial_soc_low": float(low),
        "initial_soc_high": float(high),
        "initial_soc_mean": float(np.mean(soc)),
        "initial_soc_std": float(np.std(soc)),
    }


def days_available(pool: Any, fleet: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in pool.records:
        counts[record.source_device_id] = counts.get(record.source_device_id, 0) + 1
    values = [counts.get(record.source_device_id, 0) for record in fleet]
    return {
        "days_available_min": int(min(values)) if values else 0,
        "days_available_median": int(np.median(values)) if values else 0,
        "days_available_max": int(max(values)) if values else 0,
    }
