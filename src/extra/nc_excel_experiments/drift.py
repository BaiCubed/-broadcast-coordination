from __future__ import annotations

from typing import Any

import numpy as np

from src.extra.ieee33_device_day_simulation.population.device_day_loader import DeviceDay

from .metrics import EPSILON, nrmse


class DriftedLocalPolicy:

    BASELINE = {
        "signal_threshold": (0.12, 0.48),
        "deadband": (0.015, 0.10),
        "gain": (0.85, 1.15),
        "soc_low": (0.10, 0.24),
        "soc_high": (0.76, 0.92),
        "response_probability": (0.86, 1.00),
        "power_cap_scale": (0.90, 1.10),
    }
    DRIFTED = {
        "signal_threshold": (0.34, 0.72),
        "deadband": (0.06, 0.20),
        "gain": (0.60, 0.95),
        "soc_low": (0.22, 0.38),
        "soc_high": (0.62, 0.80),
        "response_probability": (0.55, 0.85),
        "power_cap_scale": (0.55, 0.90),
    }

    def __init__(
        self,
        records: list[DeviceDay],
        *,
        drift_fraction: float,
        delay_distribution: str,
        seed: int,
        steps: int,
        drift_order: np.ndarray | None = None,
        drift_delay_shift: int = 3,
    ) -> None:
        if not 0.0 <= drift_fraction <= 1.0:
            raise ValueError("drift_fraction must be in [0, 1]")
        if delay_distribution not in {"fixed", "uniform", "lognormal"}:
            raise ValueError(f"unknown delay distribution: {delay_distribution}")
        self.records = records
        self.zone_order = sorted({record.zone_id for record in records})
        self.zone_index = {zone: index for index, zone in enumerate(self.zone_order)}
        self.record_zone_index = np.asarray(
            [self.zone_index[record.zone_id] for record in records], dtype=int
        )
        self.steps = int(steps)
        count = len(records)
        rng = np.random.default_rng(seed)

        for name, (low, high) in self.BASELINE.items():
            setattr(self, name, rng.uniform(low, high, count))
        if delay_distribution == "fixed":
            self.delay = np.full(count, 2, dtype=int)
        elif delay_distribution == "uniform":
            self.delay = rng.integers(0, 5, count, dtype=int)
        else:
            self.delay = np.clip(
                np.rint(rng.lognormal(mean=np.log(2.0), sigma=0.55, size=count)), 0, 8
            ).astype(int)

        if drift_order is None:
            drift_order = np.random.default_rng(seed + 1).permutation(count)
        self.drift_order = np.asarray(drift_order, dtype=int)
        drifted = self.drift_order[: int(round(drift_fraction * count))]
        self.drifted_mask = np.zeros(count, dtype=bool)
        self.drifted_mask[drifted] = True
        if drifted.size:
            drift_rng = np.random.default_rng(seed + 2)
            for name, (low, high) in self.DRIFTED.items():
                getattr(self, name)[drifted] = drift_rng.uniform(low, high, drifted.size)
            self.delay[drifted] = np.clip(
                self.delay[drifted] + int(drift_delay_shift), 0, 12
            )

        self.response_draw = rng.random((self.steps, count))
        self.active = np.zeros(count, dtype=bool)
        self.queue = np.zeros((self.steps + int(np.max(self.delay)) + 1, count), dtype=float)

    def __call__(self, step: int, signals: list[Any], desired_kw: np.ndarray, adapter: Any) -> np.ndarray:
        if step < 0 or step >= self.steps:
            raise ValueError("policy step is outside the configured horizon")
        scores_by_zone = np.asarray([signal.intensity / 4095.0 for signal in signals])
        discharge_by_zone = np.asarray([signal.supply_demand >= 8 for signal in signals])
        score = scores_by_zone[self.record_zone_index]
        discharge = discharge_by_zone[self.record_zone_index]

        turn_on = score >= self.signal_threshold + self.deadband
        turn_off = score <= self.signal_threshold - self.deadband
        self.active = np.where(turn_on, True, np.where(turn_off, False, self.active))

        soc = adapter.battery_states()
        soc_eligible = np.where(discharge, soc > self.soc_low, soc < self.soc_high)
        responding = self.response_draw[step] <= self.response_probability
        filtered = np.where(
            self.active & soc_eligible & responding,
            desired_kw * self.gain * self.power_cap_scale,
            0.0,
        )

        indices = np.arange(len(filtered))
        destinations = step + self.delay
        valid = destinations < self.queue.shape[0]
        np.add.at(self.queue, (destinations[valid], indices[valid]), filtered[valid])
        return self.queue[step].copy()


def drift_profile(
    replications: int,
    *,
    injection_index: int,
    target_fraction: float,
    mode: str,
    ramp: int,
) -> np.ndarray:
    profile = np.zeros(replications, dtype=float)
    if mode == "abrupt":
        profile[injection_index:] = target_fraction
    elif mode == "gradual":
        for index in range(injection_index, replications):
            progress = min(1.0, (index - injection_index + 1) / max(ramp, 1))
            profile[index] = target_fraction * progress
    else:
        raise ValueError(f"unknown drift mode: {mode}")
    return profile


def window_losses(actual: np.ndarray, predicted: np.ndarray, *, window: int) -> np.ndarray:
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted streams must align")
    count = actual.size // window
    if count == 0:
        return np.empty(0)
    trimmed_actual = actual[: count * window].reshape(count, window)
    trimmed_predicted = predicted[: count * window].reshape(count, window)
    return np.asarray([
        nrmse(trimmed_actual[index], trimmed_predicted[index]) for index in range(count)
    ])


def detection_index(
    losses: np.ndarray, *, threshold: float, start: int, consecutive: int
) -> int | None:
    values = np.asarray(losses, dtype=float)
    run = 0
    for index in range(len(values)):
        if values[index] > threshold:
            run += 1
            if run >= consecutive and index - consecutive + 1 >= start:
                return index - consecutive + 1
        else:
            run = 0
    return None


def false_alarm_rate(losses: np.ndarray, *, threshold: float, stop: int, consecutive: int) -> float:
    values = np.asarray(losses, dtype=float)[:stop]
    if values.size == 0:
        return 0.0
    alarms = 0
    run = 0
    for value in values:
        run = run + 1 if value > threshold else 0
        if run >= consecutive:
            alarms += 1
            run = 0
    return float(alarms / max(values.size, 1))


def recovery_index(
    losses: np.ndarray,
    *,
    baseline: float,
    tolerance: float,
    start: int,
    consecutive: int = 3,
) -> int | None:
    values = np.asarray(losses, dtype=float)
    run = 0
    for index in range(start, len(values)):
        if values[index] <= baseline + tolerance:
            run += 1
            if run >= consecutive:
                return index - consecutive + 1
        else:
            run = 0
    return None


def recovered_share(holdout: np.ndarray, *, failed: float, baseline: float) -> np.ndarray:
    values = np.asarray(holdout, dtype=float)
    span = failed - baseline
    return (failed - values) / (span if abs(span) > EPSILON else EPSILON)


def sample_complexity(share: np.ndarray, *, level: float, consecutive: int = 3) -> int | None:
    values = np.asarray(share, dtype=float)
    run = 0
    for index, value in enumerate(values):
        if value >= level:
            run += 1
            if run >= consecutive:
                return index - consecutive + 1
        else:
            run = 0
    return None


def drift_regret(losses: np.ndarray, *, baseline: float, start: int) -> float:
    values = np.asarray(losses, dtype=float)[start:]
    return float(np.sum(np.maximum(values - baseline, 0.0)))


def aggregate_refit(
    features: np.ndarray,
    observations: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    design = np.asarray(features, dtype=float)
    target = np.asarray(observations, dtype=float)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def update_bytes(samples: int, coefficients: int) -> int:
    return int(8 * (samples + coefficients))
