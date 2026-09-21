from __future__ import annotations

from typing import Any

import numpy as np

from src.extra.ieee33_device_day_simulation.population.device_day_loader import DeviceDay


class ParameterizedLocalPolicy:

    def __init__(
        self,
        records: list[DeviceDay],
        *,
        homogeneity: float,
        delay_distribution: str,
        seed: int,
        steps: int,
    ) -> None:
        if not 0.0 <= homogeneity <= 1.0:
            raise ValueError("homogeneity must be in [0, 1]")
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

        self.signal_threshold = rng.uniform(0.12, 0.48, count)
        self.deadband = rng.uniform(0.015, 0.10, count)
        self.gain = rng.uniform(0.85, 1.15, count)
        self.soc_low = rng.uniform(0.10, 0.24, count)
        self.soc_high = rng.uniform(0.76, 0.92, count)
        if delay_distribution == "fixed":
            self.delay = np.full(count, 2, dtype=int)
            shared_delay = 2
        elif delay_distribution == "uniform":
            self.delay = rng.integers(0, 5, count, dtype=int)
            shared_delay = 2
        else:
            self.delay = np.clip(
                np.rint(rng.lognormal(mean=np.log(2.0), sigma=0.55, size=count)),
                0,
                8,
            ).astype(int)
            shared_delay = 2

        shared_count = int(round(homogeneity * count))
        shared_order = np.random.default_rng(seed + 1).permutation(count)
        shared = shared_order[:shared_count]
        self.signal_threshold[shared] = 0.30
        self.deadband[shared] = 0.05
        self.gain[shared] = 1.0
        self.soc_low[shared] = 0.18
        self.soc_high[shared] = 0.86
        self.delay[shared] = shared_delay

        self.active = np.zeros(count, dtype=bool)
        self.queue = np.zeros((self.steps + int(np.max(self.delay)) + 1, count), dtype=float)

    def __call__(
        self,
        step: int,
        signals: list[Any],
        desired_kw: np.ndarray,
        adapter: Any,
    ) -> np.ndarray:
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
        filtered = np.where(self.active & soc_eligible, desired_kw * self.gain, 0.0)

        indices = np.arange(len(filtered))
        destinations = step + self.delay
        valid = destinations < self.queue.shape[0]
        np.add.at(self.queue, (destinations[valid], indices[valid]), filtered[valid])
        return self.queue[step].copy()

