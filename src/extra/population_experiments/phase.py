from __future__ import annotations

import numpy as np


EPSILON = 1e-12


def analytic_phase(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError("series must have device and time dimensions")
    steps = values.shape[1]
    if steps < 4:
        raise ValueError("phase analysis needs at least four time steps")
    centred = values - np.mean(values, axis=1, keepdims=True)
    active = np.std(centred, axis=1) > EPSILON
    multiplier = np.zeros(steps)
    multiplier[0] = 1.0
    if steps % 2 == 0:
        multiplier[steps // 2] = 1.0
        multiplier[1 : steps // 2] = 2.0
    else:
        multiplier[1 : (steps + 1) // 2] = 2.0
    spectrum = np.fft.fft(centred, axis=1)
    analytic = np.fft.ifft(spectrum * multiplier[None, :], axis=1)
    return np.angle(analytic), active


def kuramoto_order(phases: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(phases, dtype=float)
    if active is not None:
        values = values[np.asarray(active, dtype=bool)]
    if values.shape[0] == 0:
        return np.zeros(np.asarray(phases).shape[1])
    return np.abs(np.mean(np.exp(1j * values), axis=0))


def pairwise_plv(phases: np.ndarray, *, pairs: int, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(phases, dtype=float)
    count = values.shape[0]
    if count < 2:
        return {"plv_mean": 0.0, "plv_p95": 0.0, "plv_pairs": 0}
    budget = int(min(pairs, count * (count - 1) // 2))
    left = rng.integers(0, count, size=budget)
    right = rng.integers(0, count, size=budget)
    keep = left != right
    left, right = left[keep], right[keep]
    if left.size == 0:
        return {"plv_mean": 0.0, "plv_p95": 0.0, "plv_pairs": 0}
    difference = values[left] - values[right]
    locking = np.abs(np.mean(np.exp(1j * difference), axis=1))
    return {
        "plv_mean": float(np.mean(locking)),
        "plv_p95": float(np.percentile(locking, 95)),
        "plv_pairs": int(left.size),
    }


def circular_shift_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    steps = values.shape[1]
    lags = rng.integers(0, steps, size=values.shape[0])
    index = (np.arange(steps)[None, :] + lags[:, None]) % steps
    return np.take_along_axis(values, index, axis=1)


def fourier_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    devices, steps = values.shape
    spectrum = np.fft.rfft(values, axis=1)
    random_phase = rng.uniform(0.0, 2.0 * np.pi, size=(devices, spectrum.shape[1]))
    random_phase[:, 0] = 0.0
    if steps % 2 == 0:
        random_phase[:, -1] = 0.0
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * random_phase), n=steps, axis=1)


def surrogate_band(
    series: np.ndarray,
    *,
    surrogates: int,
    quantile: float,
    seed: int,
    method: str = "both",
) -> dict[str, np.ndarray]:
    values = np.asarray(series, dtype=float)
    rng = np.random.default_rng(seed)
    builders = []
    if method in {"circular", "both"}:
        builders.append(circular_shift_surrogate)
    if method in {"fourier", "both"}:
        builders.append(fourier_surrogate)
    if not builders:
        raise ValueError(f"unknown surrogate method: {method}")
    bands = []
    for builder in builders:
        draws = np.empty((surrogates, values.shape[1]))
        for index in range(surrogates):
            phases, active = analytic_phase(builder(values, rng))
            draws[index] = kuramoto_order(phases, active)
        bands.append(np.quantile(draws, quantile, axis=0))
    stacked = np.max(np.vstack(bands), axis=0)
    return {"band": stacked, "surrogates": np.asarray([surrogates * len(builders)])}


def replication_permutation_band(
    windows: np.ndarray, *, surrogates: int, quantile: float, seed: int
) -> np.ndarray:
    stack = np.asarray(windows, dtype=float)
    if stack.ndim != 3:
        raise ValueError("windows must have replication, device and time dimensions")
    replications, devices, steps = stack.shape
    if replications < 2:
        raise ValueError("the permutation null needs at least two replications")
    rng = np.random.default_rng(seed)
    draws = np.empty((surrogates, steps))
    index = np.arange(devices)
    for draw in range(surrogates):
        mixed = stack[rng.integers(0, replications, size=devices), index]
        phases, active = analytic_phase(mixed)
        draws[draw] = kuramoto_order(phases, active)
    return np.quantile(draws, quantile, axis=0)


def exceedance_statistics(order: np.ndarray, band: np.ndarray, *, dt_minutes: float) -> dict[str, float]:
    above = np.asarray(order, dtype=float) > np.asarray(band, dtype=float)
    if not above.size:
        return {"H_phase": 0.0, "L_phase_minutes": 0.0, "R_K_mean": 0.0, "R_K_peak": 0.0}
    longest = current = 0
    for flag in above:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return {
        "H_phase": float(np.mean(above)),
        "L_phase_minutes": float(longest * dt_minutes),
        "R_K_mean": float(np.mean(order)),
        "R_K_peak": float(np.max(order)),
    }


def phase_histogram(phases: np.ndarray, *, bins: int = 36) -> np.ndarray:
    values = np.mod(np.asarray(phases, dtype=float).ravel(), 2.0 * np.pi)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 2.0 * np.pi))
    return counts.astype(float)


def analyse_window(
    series: np.ndarray,
    *,
    dt_minutes: float,
    conditioned_band: np.ndarray,
    surrogates: int,
    quantile: float,
    seed: int,
    plv_pairs: int,
    surrogate_method: str = "both",
) -> dict[str, object]:
    values = np.asarray(series, dtype=float)
    phases, active = analytic_phase(values)
    order = kuramoto_order(phases, active)
    band = np.asarray(conditioned_band, dtype=float)
    stats = exceedance_statistics(order, band, dt_minutes=dt_minutes)
    shuffled = surrogate_band(
        values, surrogates=surrogates, quantile=quantile, seed=seed, method=surrogate_method,
    )["band"]
    reference = exceedance_statistics(order, shuffled, dt_minutes=dt_minutes)
    stats.update({
        "H_phase_timeshuffle": reference["H_phase"],
        "L_phase_minutes_timeshuffle": reference["L_phase_minutes"],
        "timeshuffle_band_mean": float(np.mean(shuffled)),
    })
    rng = np.random.default_rng(seed + 7919)
    stats.update(pairwise_plv(phases[active], pairs=plv_pairs, rng=rng))
    aggregate = np.sum(values, axis=0)
    stats.update({
        "active_fraction": float(np.mean(active)),
        "inactive_devices": int(np.sum(~active)),
        "aggregate_amplitude_kw": float(np.std(aggregate)),
        "aggregate_peak_kw": float(np.max(np.abs(aggregate))),
        "conditioned_band_mean": float(np.mean(band)),
    })
    return {"summary": stats, "order": order, "band": band, "phases": phases, "active": active}
