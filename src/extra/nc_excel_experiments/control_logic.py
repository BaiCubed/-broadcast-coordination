from __future__ import annotations

from typing import Any

import numpy as np

CONTROL_LOGICS = (
    "soc_threshold",
    "probabilistic",
    "price_response",
    "random_delay",
    "hybrid",
)


class _BasePolicy:

    def __init__(self, records: list[Any], *, seed: int, steps: int) -> None:
        self.records = records
        self.count = len(records)
        self.steps = int(steps)
        self.zone_order = sorted({record.zone_id for record in records})
        self.zone_index = {zone: index for index, zone in enumerate(self.zone_order)}
        self.record_zone_index = np.asarray(
            [self.zone_index[record.zone_id] for record in records], dtype=int
        )
        self.rng = np.random.default_rng(seed)
        self._requested = 0
        self._responded = 0

    def _decode(self, signals: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        scores_by_zone = np.asarray([signal.intensity / 4095.0 for signal in signals])
        discharge_by_zone = np.asarray([signal.supply_demand >= 8 for signal in signals])
        return (
            scores_by_zone[self.record_zone_index],
            discharge_by_zone[self.record_zone_index],
        )

    def _account(self, desired_kw: np.ndarray, emitted: np.ndarray) -> np.ndarray:
        self._requested += int(np.sum(np.abs(desired_kw) > 1e-12))
        self._responded += int(np.sum(np.abs(emitted) > 1e-12))
        return emitted

    @property
    def response_fraction(self) -> float:
        return float(self._responded) / max(self._requested, 1)


class SocThresholdPolicy(_BasePolicy):

    def __init__(self, records, *, seed: int, steps: int) -> None:
        super().__init__(records, seed=seed, steps=steps)
        self.soc_low = self.rng.uniform(0.10, 0.24, self.count)
        self.soc_high = self.rng.uniform(0.76, 0.92, self.count)

    def __call__(self, step: int, signals, desired_kw: np.ndarray, adapter) -> np.ndarray:
        _, discharge = self._decode(signals)
        soc = adapter.battery_states()
        eligible = np.where(discharge, soc > self.soc_low, soc < self.soc_high)
        return self._account(desired_kw, np.where(eligible, desired_kw, 0.0))


class ProbabilisticPolicy(_BasePolicy):

    def __init__(self, records, *, seed: int, steps: int) -> None:
        super().__init__(records, seed=seed, steps=steps)
        self.probability = self.rng.beta(8.0, 2.0, self.count)

    def __call__(self, step: int, signals, desired_kw: np.ndarray, adapter) -> np.ndarray:
        draw = self.rng.random(self.count)
        return self._account(desired_kw, np.where(draw < self.probability, desired_kw, 0.0))


class PriceResponsePolicy(_BasePolicy):

    def __init__(self, records, *, seed: int, steps: int, elasticity: float = 2.5) -> None:
        super().__init__(records, seed=seed, steps=steps)
        self.reservation = self.rng.uniform(0.25, 0.65, self.count)
        self.elasticity = float(elasticity)

    def __call__(self, step: int, signals, desired_kw: np.ndarray, adapter) -> np.ndarray:
        price, discharge = self._decode(signals)
        excess = np.where(
            discharge,
            price - self.reservation,
            (1.0 - self.reservation) - price,
        )
        fraction = np.clip(self.elasticity * excess, 0.0, 1.0)
        return self._account(desired_kw, desired_kw * fraction)


class RandomDelayPolicy(_BasePolicy):

    def __init__(self, records, *, seed: int, steps: int, max_delay: int = 6) -> None:
        super().__init__(records, seed=seed, steps=steps)
        self.max_delay = int(max_delay)
        self.queue = np.zeros((self.steps + self.max_delay + 1, self.count), dtype=float)

    def __call__(self, step: int, signals, desired_kw: np.ndarray, adapter) -> np.ndarray:
        active = np.abs(desired_kw) > 1e-12
        delay = self.rng.integers(0, self.max_delay + 1, self.count)
        destinations = step + delay
        indices = np.arange(self.count)
        valid = active & (destinations < self.queue.shape[0])
        np.add.at(self.queue, (destinations[valid], indices[valid]), desired_kw[valid])
        return self._account(desired_kw, self.queue[step].copy())


class HybridPolicy(_BasePolicy):

    def __init__(self, records, *, seed: int, steps: int) -> None:
        super().__init__(records, seed=seed, steps=steps)
        self.members = [
            SocThresholdPolicy(records, seed=seed + 11, steps=steps),
            ProbabilisticPolicy(records, seed=seed + 22, steps=steps),
            PriceResponsePolicy(records, seed=seed + 33, steps=steps),
            RandomDelayPolicy(records, seed=seed + 44, steps=steps),
        ]
        order = np.random.default_rng(seed + 55).permutation(self.count)
        self.group = np.zeros(self.count, dtype=int)
        for group_index, chunk in enumerate(np.array_split(order, len(self.members))):
            self.group[chunk] = group_index

    def __call__(self, step: int, signals, desired_kw: np.ndarray, adapter) -> np.ndarray:
        emitted = np.zeros(self.count, dtype=float)
        for group_index, member in enumerate(self.members):
            mask = self.group == group_index
            emitted[mask] = member(step, signals, desired_kw, adapter)[mask]
        return self._account(desired_kw, emitted)


_POLICY_TYPES = {
    "soc_threshold": SocThresholdPolicy,
    "probabilistic": ProbabilisticPolicy,
    "price_response": PriceResponsePolicy,
    "random_delay": RandomDelayPolicy,
    "hybrid": HybridPolicy,
}


def build_policy(logic: str, records: list[Any], *, seed: int, steps: int) -> _BasePolicy:
    if logic not in _POLICY_TYPES:
        raise ValueError(f"unknown control logic: {logic}")
    return _POLICY_TYPES[logic](records, seed=seed, steps=steps)
