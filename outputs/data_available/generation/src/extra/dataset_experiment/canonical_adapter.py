"""Map load-only public datasets to the existing DeviceDay contract.

The adapter deliberately keeps missing battery/PV fields explicit.  Battery
parameters are bounded fallbacks for exercising the unchanged simulator; they
are never presented as measured EV parameters in the manifest.
"""

from __future__ import annotations

from pathlib import Path
import io
import re
import subprocess
import tarfile
from zipfile import ZipFile
from typing import Any

import numpy as np
import pandas as pd

from src.extra.ieee33_device_day_simulation.population.device_day_loader import (
    DeviceDay,
    DeviceDayPool,
)


ZONE_ORDER = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5", "zone_6"]


def _spec(config: dict[str, Any]) -> dict[str, Any]:
    value = config["population"].get("canonical_adapter")
    if not isinstance(value, dict):
        raise ValueError("population.canonical_adapter must be a mapping")
    return value


def _fallback(spec: dict[str, Any]) -> tuple[float, float, float]:
    capacity = float(spec.get("fallback_capacity_kwh", 10.0))
    peak = float(spec.get("fallback_peak_power_kw", 1.0))
    soc_fraction = float(spec.get("fallback_soc_fraction", 0.50))
    if not (5.0 <= capacity <= 20.0):
        raise ValueError(f"fallback capacity {capacity} is outside original 5-20 kWh range")
    if not (0.01 <= peak <= capacity):
        raise ValueError(f"fallback peak power {peak} is invalid for capacity {capacity}")
    if not (0.10 <= soc_fraction <= 0.95):
        raise ValueError(f"fallback SOC fraction {soc_fraction} violates device limits")
    return capacity, peak, soc_fraction


def _zone_bus(index: int, config: dict[str, Any]) -> tuple[str, int]:
    zone = ZONE_ORDER[index % len(ZONE_ORDER)]
    buses = list(config["zones"]["zones"][zone]["buses"])
    return zone, int(buses[(index // len(ZONE_ORDER)) % len(buses)])


def _record(
    profile: np.ndarray,
    source_id: str,
    day_id: int,
    index: int,
    config: dict[str, Any],
    *,
    timestamp: np.ndarray,
    source_unit: str,
    energy_input: np.ndarray | None = None,
    pv: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
    capacity_kwh: float | None = None,
    peak_power_kw: float | None = None,
    soc_fraction: float | None = None,
) -> DeviceDay:
    fallback_capacity, fallback_peak, fallback_soc = _fallback(_spec(config))
    capacity = float(capacity_kwh) if capacity_kwh is not None else fallback_capacity
    peak = float(peak_power_kw) if peak_power_kw is not None else fallback_peak
    initial_soc_fraction = float(soc_fraction) if soc_fraction is not None else fallback_soc
    if not (5.0 <= capacity <= 20.0):
        capacity = fallback_capacity
    if not (0.01 <= peak <= capacity * 2.0):
        peak = fallback_peak
    if not (0.10 <= initial_soc_fraction <= 0.95):
        initial_soc_fraction = fallback_soc
    zone, bus = _zone_bus(index, config)
    profile = np.asarray(profile, dtype=float)
    external = np.zeros(288, dtype=float) if energy_input is None else np.asarray(energy_input, dtype=float)
    legacy_pv = np.zeros(288, dtype=float) if pv is None else np.asarray(pv, dtype=float)
    baseline_profile = np.zeros(288, dtype=float) if baseline is None else np.asarray(baseline, dtype=float)
    if (
        profile.shape != (288,)
        or external.shape != (288,)
        or legacy_pv.shape != (288,)
        or baseline_profile.shape != (288,)
        or not np.isfinite(profile).all()
        or not np.isfinite(external).all()
        or not np.isfinite(legacy_pv).all()
        or not np.isfinite(baseline_profile).all()
        or (profile < 0).any()
        or (external < 0).any()
        or (legacy_pv < 0).any()
    ):
        raise ValueError(f"invalid 288-point load profile for {source_id} day {day_id}")
    soc = np.full(288, capacity * initial_soc_fraction, dtype=float)
    return DeviceDay(
        device_id=f"{source_id}_day_{day_id:03d}",
        source_device_id=source_id,
        day_id=day_id,
        zone_id=zone,
        bus_id=bus,
        load_kw=profile,
        pv_kw=legacy_pv,
        energy_input_kw=external,
        baseline_battery_kw=baseline_profile,
        soc_kwh=soc,
        capacity_kwh=capacity,
        peak_power_kw=peak,
        c_rate=float(peak / capacity),
        timestamp=np.asarray(timestamp, dtype=np.int64),
    )


def _full_day_profiles(frame: pd.DataFrame, value_column: str, scale: float, max_days: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", value_column]).sort_values("timestamp")
    series = frame.set_index("timestamp")[value_column].astype(float) * float(scale)
    target_days: list[tuple[str, np.ndarray, np.ndarray]] = []
    for day in sorted(series.index.normalize().unique()):
        start = pd.Timestamp(day)
        target = pd.date_range(start, periods=288, freq="5min")
        if series.index.min() > start or series.index.max() < target[-1]:
            continue
        merged = series.reindex(series.index.union(target)).sort_index().interpolate(limit_direction="both")
        values = merged.reindex(target).to_numpy(dtype=float)
        target_days.append((start.strftime("%Y-%m-%d"), np.maximum(values, 0.0), target.view("int64")))
        if len(target_days) >= int(max_days):
            break
    return target_days


def _resample_series_days(
    series: pd.Series,
    *,
    scale: float,
    max_days: int,
    source_unit: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Convert a single real time series into complete five-minute days."""
    series = pd.to_numeric(series, errors="coerce").dropna().groupby(level=0).mean().sort_index()
    if series.empty:
        return []
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for day in sorted(series.index.normalize().unique()):
        start = pd.Timestamp(day)
        target = pd.date_range(start, periods=288, freq="5min")
        if series.index.min() > start or series.index.max() < target[-1]:
            continue
        joined = series.reindex(series.index.union(target)).sort_index()
        values = joined.interpolate(limit_direction="both").reindex(target).to_numpy(dtype=float)
        values = np.maximum(values * float(scale), 0.0)
        if np.isfinite(values).all() and values.size == 288:
            result.append((start.strftime("%Y-%m-%d"), values, target.view("int64")))
        if len(result) >= int(max_days):
            break
    return result


def _regular_profile(values: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    """Interpolate an ordered daily profile to the unchanged 288-step grid."""
    values = pd.to_numeric(pd.Series(np.asarray(values).reshape(-1)), errors="coerce")
    values = values.interpolate(limit_direction="both").to_numpy(dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("a profile needs at least two finite values")
    source_x = np.linspace(0.0, 288.0, values.size, endpoint=False)
    target_x = np.arange(288, dtype=float)
    return np.maximum(np.interp(target_x, source_x, values, left=values[0], right=values[-1]) * scale, 0.0)


def _records_from_ordered_files(
    paths: list[Path], config: dict[str, Any], *, max_days: int, value_scale: float = 1.0,
    source_limit: int | None = None,
) -> DeviceDayPool:
    """Read P/Q-style ordered profiles without inventing calendar timestamps."""
    records: list[DeviceDay] = []
    selected = paths[:source_limit] if source_limit else paths
    for source_index, path in enumerate(selected):
        values = pd.read_csv(path, header=None, usecols=[0]).iloc[:, 0].to_numpy(dtype=float)
        day_count = min(max_days, values.size // 24)
        for day_id in range(day_count):
            profile = _regular_profile(values[day_id * 24 : (day_id + 1) * 24], scale=value_scale)
            timestamp = pd.date_range(
                pd.Timestamp("2011-01-01") + pd.Timedelta(days=day_id), periods=288, freq="5min"
            ).view("int64")
            records.append(_record(
                profile, path.stem, day_id, source_index, config,
                timestamp=timestamp, source_unit="ordered_hourly_profile",
            ))
    source_count = len({record.source_device_id for record in records})
    return DeviceDayPool(records, len(records), source_count, 0)


def _cache_path(spec: dict[str, Any]) -> Path | None:
    value = spec.get("cache")
    return Path(value).resolve() if value else None


def _load_cache(path: Path) -> DeviceDayPool | None:
    if not path.exists():
        return None
    try:
        # NPZ members are lazy. Materialize each member once; repeatedly using
        # data["load"] inside the record loop would decompress the complete
        # matrix once per record and retain it through each row view.
        with np.load(path, allow_pickle=True) as archive:
            data = {name: archive[name] for name in archive.files}
        records = []
        for i in range(len(data["device_id"])):
            records.append(DeviceDay(
                device_id=str(data["device_id"][i]), source_device_id=str(data["source_id"][i]),
                day_id=int(data["day_id"][i]), zone_id=str(data["zone_id"][i]), bus_id=int(data["bus_id"][i]),
                load_kw=data["load"][i], pv_kw=data["pv"][i], energy_input_kw=data["energy_input"][i],
                baseline_battery_kw=data["baseline"][i], soc_kwh=data["soc"][i],
                capacity_kwh=float(data["capacity"][i]), peak_power_kw=float(data["peak"][i]),
                c_rate=float(data["c_rate"][i]), timestamp=data["timestamp"][i],
            ))
        return DeviceDayPool(records, len(records), int(data["source_count"]), int(data["filled_values"]))
    except (OSError, KeyError, ValueError, EOFError):
        return None


def _save_cache(path: Path, pool: DeviceDayPool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        device_id=np.asarray([r.device_id for r in pool.records], dtype=object),
        source_id=np.asarray([r.source_device_id for r in pool.records], dtype=object),
        day_id=np.asarray([r.day_id for r in pool.records], dtype=np.int32),
        zone_id=np.asarray([r.zone_id for r in pool.records], dtype=object),
        bus_id=np.asarray([r.bus_id for r in pool.records], dtype=np.int16),
        load=np.asarray([r.load_kw for r in pool.records], dtype=np.float32),
        pv=np.asarray([r.pv_kw for r in pool.records], dtype=np.float32),
        energy_input=np.asarray([r.energy_input_kw for r in pool.records], dtype=np.float32),
        baseline=np.asarray([r.baseline_battery_kw for r in pool.records], dtype=np.float32),
        soc=np.asarray([r.soc_kwh for r in pool.records], dtype=np.float32),
        capacity=np.asarray([r.capacity_kwh for r in pool.records], dtype=np.float32),
        peak=np.asarray([r.peak_power_kw for r in pool.records], dtype=np.float32),
        c_rate=np.asarray([r.c_rate for r in pool.records], dtype=np.float32),
        timestamp=np.asarray([r.timestamp for r in pool.records], dtype=np.int64),
        source_count=np.asarray(pool.source_count), filled_values=np.asarray(pool.filled_values),
    )


def _counterfactual_input(load: np.ndarray, scale: float) -> np.ndarray:
    """Create an explicitly labeled surplus/shortage proxy from real load shape.

    This is an input adapter transform for load-only datasets, not a PV or
    generation model. Low-load snapshots receive more available input and
    high-load snapshots receive less, so the unchanged normal signal can
    observe both directions over one real day.
    """
    load = np.asarray(load, dtype=float)
    low, high = np.percentile(load, [10.0, 90.0])
    normalized = np.clip((load - low) / max(high - low, 1e-9), 0.0, 1.0)
    factor = float(scale) * (1.60 - 1.20 * normalized)
    return np.maximum(load * factor, 0.0)


def _apply_input_mapping(pool: DeviceDayPool, spec: dict[str, Any]) -> DeviceDayPool:
    mapping = str(spec.get("input_mapping", "measured_only"))
    if bool(spec.get("forbid_counterfactual_input", False)) and mapping != "measured_only":
        raise ValueError("this dataset forbids counterfactual energy input mapping")
    scale = float(spec.get("counterfactual_input_scale", 1.0))
    if mapping not in {"measured_only", "load_shape_counterfactual", "measured_or_load_shape_counterfactual"}:
        raise ValueError(f"unsupported canonical input_mapping={mapping!r}")
    if scale <= 0:
        raise ValueError("counterfactual_input_scale must be positive")
    if mapping == "measured_only":
        return pool
    for record in pool.records:
        has_measured_input = bool(np.max(np.asarray(record.energy_input_kw, dtype=float)) > 1e-9)
        if mapping == "load_shape_counterfactual" or not has_measured_input:
            record.energy_input_kw = _counterfactual_input(record.load_kw, scale)
            record.pv_kw = np.zeros_like(record.energy_input_kw)
    return pool


def _load_bdg1(config: dict[str, Any]) -> DeviceDayPool:
    """Load the official 507-building BDG1 wide hourly CSV."""
    spec = _spec(config)
    path = Path(spec["input"]).resolve()
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    timestamp = pd.to_datetime(frame.pop("timestamp"), errors="coerce")
    records: list[DeviceDay] = []
    max_sources = int(spec.get("max_sources", 507))
    max_days = int(spec.get("max_days_per_source", 3))
    for source_index, column in enumerate(frame.columns[:max_sources]):
        values = pd.to_numeric(frame[column], errors="coerce")
        profiles = _resample_series_days(
            pd.Series(values.to_numpy(dtype=float), index=timestamp),
            scale=float(spec.get("load_scale", 0.001)),
            max_days=max_days,
            source_unit="kWh_per_hour",
        )
        for day_index, (_, profile, target) in enumerate(profiles):
            records.append(_record(
                profile,
                str(column),
                day_index,
                source_index,
                config,
                timestamp=target,
                source_unit="kWh_per_hour",
            ))
    if len({record.source_device_id for record in records}) < 500:
        raise ValueError("BDG1 produced fewer than 500 independent building units")
    return DeviceDayPool(records, len(records), len({r.source_device_id for r in records}), 0)


def _load_lcl(config: dict[str, Any]) -> DeviceDayPool:
    """Stream the official Low Carbon London long CSV without extracting 8.5 GB."""
    spec = _spec(config)
    path = Path(spec["input"]).resolve()
    max_sources = int(spec.get("max_sources", 600))
    max_days = int(spec.get("max_days_per_source", 2))
    source_frames: list[tuple[str, pd.DataFrame]] = []
    # The official archive uses a ZIP method unsupported by Python's zipfile
    # module here. The system unzip utility can stream it without extraction.
    listing = subprocess.run(["unzip", "-Z1", str(path)], check=True, capture_output=True, text=True)
    names = [name for name in listing.stdout.splitlines() if name.endswith("CC_LCL-FullData.csv")]
    if not names:
        raise FileNotFoundError("CC_LCL-FullData.csv is missing from Low Carbon London archive")
    process = subprocess.Popen(["unzip", "-p", str(path), names[0]], stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("bsdtar did not provide a data stream")
    try:
        reader = pd.read_csv(
            process.stdout,
            skipinitialspace=True,
            chunksize=250_000,
        )
        current_id: str | None = None
        rows: list[tuple[str, float]] = []

        def finish() -> None:
            nonlocal current_id, rows
            if current_id is None or not rows:
                return
            source_frames.append((current_id, pd.DataFrame(rows, columns=["timestamp", "value"])))
            rows = []

        for chunk in reader:
            chunk.columns = [str(column).strip() for column in chunk.columns]
            value_column = next(
                (column for column in chunk.columns if str(column).lower().startswith("kwh/")),
                None,
            )
            if value_column is None:
                raise ValueError("Low Carbon London archive has no KWH half-hour column")
            chunk["LCLid"] = chunk["LCLid"].astype(str)
            for source_id, group in chunk.groupby("LCLid", sort=False):
                source_id = str(source_id)
                if current_id is not None and source_id != current_id:
                    finish()
                    if len(source_frames) >= max_sources:
                        break
                current_id = source_id
                for timestamp, value in zip(group["DateTime"], group[value_column]):
                    numeric_value = pd.to_numeric(value, errors="coerce")
                    if pd.notna(timestamp) and pd.notna(numeric_value):
                        rows.append((str(timestamp), float(numeric_value)))
            if len(source_frames) >= max_sources:
                break
        finish()
    finally:
        process.stdout.close()
        process.wait()
    records: list[DeviceDay] = []
    for source_index, (source_id, source_frame) in enumerate(source_frames[:max_sources]):
        timestamps = pd.to_datetime(source_frame["timestamp"], errors="coerce")
        series = pd.Series(
            pd.to_numeric(source_frame["value"], errors="coerce").to_numpy(dtype=float) * 2.0,
            index=timestamps,
        )
        profiles = _resample_series_days(
            series,
            scale=float(spec.get("load_scale", 1.0)),
            max_days=max_days,
            source_unit="kWh_per_half_hour_to_kW",
        )
        for day_index, (_, profile, target) in enumerate(profiles):
            records.append(_record(
                profile,
                source_id,
                day_index,
                source_index,
                config,
                timestamp=target,
                source_unit="kWh_per_half_hour_to_kW",
            ))
    source_count = len({record.source_device_id for record in records})
    if source_count < 500:
        raise ValueError(f"Low Carbon London produced only {source_count} independent households")
    return DeviceDayPool(records, len(records), source_count, 0)


def _load_bdg2(config: dict[str, Any]) -> DeviceDayPool:
    """Load BDG2 electricity and the overlapping measured solar columns."""
    spec = _spec(config)
    electricity_path = Path(spec["input"]).resolve()
    solar_path = Path(spec["solar_input"]).resolve()
    electricity = pd.read_csv(electricity_path, parse_dates=["timestamp"])
    solar = pd.read_csv(solar_path, parse_dates=["timestamp"])
    timestamps = pd.to_datetime(electricity.pop("timestamp"), errors="coerce")
    solar_timestamp = pd.to_datetime(solar.pop("timestamp"), errors="coerce")
    solar_by_source = {str(column): pd.Series(pd.to_numeric(solar[column], errors="coerce").to_numpy(), index=solar_timestamp)
                       for column in solar.columns}
    records: list[DeviceDay] = []
    max_sources = int(spec.get("max_sources", 1636))
    max_days = int(spec.get("max_days_per_source", 8))
    scale = float(spec.get("load_scale", 0.001))
    for source_index, column in enumerate(electricity.columns[:max_sources]):
        series = pd.Series(pd.to_numeric(electricity[column], errors="coerce").to_numpy(dtype=float), index=timestamps)
        profiles = _resample_series_days(series, scale=scale, max_days=max_days, source_unit="hourly_wh")
        input_profiles = _resample_series_days(
            solar_by_source[column], scale=scale, max_days=max_days, source_unit="hourly_wh"
        ) if str(column) in solar_by_source else []
        input_by_day = {day: values for day, values, _ in input_profiles}
        for day_id, (day, profile, target) in enumerate(profiles):
            energy = input_by_day.get(day, np.zeros(288, dtype=float))
            records.append(_record(
                profile, str(column), day_id, source_index, config,
                timestamp=target, source_unit="hourly_wh_to_kw", energy_input=energy, pv=energy,
            ))
    source_count = len({r.source_device_id for r in records})
    if source_count < 500:
        raise ValueError(f"BDG2 produced only {source_count} independent building units")
    return DeviceDayPool(records, len(records), source_count, 0)


def _load_danish(config: dict[str, Any]) -> DeviceDayPool:
    """Convert cumulative heat-meter energy to a transparent heat-load proxy."""
    spec = _spec(config)
    archive_path = Path(spec["input"]).resolve()
    names = sorted(
        name for name in ZipFile(archive_path).namelist()
        if re.search(r"02_processed_final_data/\d+\.csv$", name)
    )[: int(spec.get("max_sources", 2400))]
    records: list[DeviceDay] = []
    max_days = int(spec.get("max_days_per_source", 4))
    with ZipFile(archive_path) as archive:
        for source_index, name in enumerate(names):
            with archive.open(name) as handle:
                frame = pd.read_csv(handle, nrows=max_days * 24 * 3 + 48)
            frame["timestamp"] = pd.to_datetime(frame["time_rounded"], errors="coerce", utc=True)
            frame["cumulative"] = pd.to_numeric(frame["energy_heat_kwh"], errors="coerce")
            frame = frame.dropna(subset=["timestamp", "cumulative"]).sort_values("timestamp")
            frame = frame.drop_duplicates("timestamp").set_index("timestamp")
            # The processed field is a cumulative register, not interval heat.
            increments = frame["cumulative"].diff().clip(lower=0.0)
            profiles = _resample_series_days(
                increments, scale=float(spec.get("load_scale", 1.0)),
                max_days=max_days, source_unit="cumulative_kwh_difference_per_hour",
            )
            for day_id, (_, profile, target) in enumerate(profiles):
                records.append(_record(
                    profile, Path(name).stem, day_id, source_index, config,
                    timestamp=target, source_unit="heat_kwh_per_hour_to_kw",
                ))
    source_count = len({r.source_device_id for r in records})
    if source_count < 500:
        raise ValueError(f"Danish heat meter adapter produced only {source_count} independent meters")
    return DeviceDayPool(records, len(records), source_count, 0)


def _load_heapo(config: dict[str, Any]) -> DeviceDayPool:
    """Load measured 15-minute whole-house electricity from HEAPO."""
    spec = _spec(config)
    archive_path = Path(spec["input"]).resolve()
    max_sources = int(spec.get("max_sources", 1408))
    max_days = int(spec.get("max_days_per_source", 8))
    with ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if re.search(r"smart_meter_data/15min/.*\.csv$", name))[:max_sources]
        records: list[DeviceDay] = []
        for source_index, name in enumerate(names):
            with archive.open(name) as handle:
                frame = pd.read_csv(handle, sep=";", nrows=max_days * 96 + 192)
            frame["timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
            frame["value"] = pd.to_numeric(frame["kWh_received_Total"], errors="coerce")
            series = frame.dropna(subset=["timestamp", "value"]).set_index("timestamp")["value"]
            profiles = _resample_series_days(series, scale=4.0, max_days=max_days, source_unit="15min_kwh_to_kw")
            for day_id, (_, profile, target) in enumerate(profiles):
                records.append(_record(
                    profile, Path(name).stem, day_id, source_index, config,
                    timestamp=target, source_unit="15min_kwh_to_kw",
                ))
    source_count = len({r.source_device_id for r in records})
    if source_count < 500:
        raise ValueError(f"HEAPO produced only {source_count} independent households")
    return DeviceDayPool(records, len(records), source_count, 0)


def _resample_pair_days(load: pd.Series, external: pd.Series, *, max_days: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    load = pd.to_numeric(load, errors="coerce").groupby(level=0).mean().sort_index()
    external = pd.to_numeric(external, errors="coerce").groupby(level=0).mean().sort_index()
    result = []
    if load.empty:
        return result
    for day in sorted(load.index.normalize().unique()):
        start = pd.Timestamp(day)
        target = pd.date_range(start, periods=288, freq="5min")
        if load.index.min() > start:
            continue
        union = load.index.union(target)
        load_values = load.reindex(union).interpolate(limit_direction="both").reindex(target).to_numpy(float)
        input_union = external.index.union(target)
        input_values = external.reindex(input_union).interpolate(limit_direction="both").reindex(target).fillna(0.0).to_numpy(float)
        if np.isfinite(load_values).all() and np.isfinite(input_values).all():
            result.append((start.strftime("%Y-%m-%d"), np.maximum(load_values, 0.0), np.maximum(input_values, 0.0), target.view("int64")))
        if len(result) >= max_days:
            break
    return result


def _load_sgsc(config: dict[str, Any]) -> DeviceDayPool:
    """Stream SGSC's 1.7 GB 7z interval table until enough customers are collected."""
    spec = _spec(config)
    path = Path(spec["input"]).resolve()
    max_sources = int(spec.get("max_sources", 600))
    max_days = int(spec.get("max_days_per_source", 20))
    source_frames: list[tuple[str, pd.DataFrame]] = []
    process = subprocess.Popen(["bsdtar", "-xOf", str(path), "CD_INTERVAL_READING_ALL_NO_QUOTES.csv"], stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("cannot stream SGSC 7z archive")
    try:
        reader = pd.read_csv(process.stdout, usecols=[0, 1, 4, 5, 6, 8], skipinitialspace=True, chunksize=200_000)
        current_id: str | None = None
        rows: list[tuple[str, float, float]] = []
        def finish() -> None:
            nonlocal current_id, rows
            if current_id is not None and rows:
                source_frames.append((current_id, pd.DataFrame(rows, columns=["timestamp", "load", "input"])))
            rows = []
        for chunk in reader:
            chunk.columns = ["customer", "timestamp", "general", "controlled", "gross", "other"]
            chunk["customer"] = chunk["customer"].astype(str)
            for source_id, group in chunk.groupby("customer", sort=False):
                source_id = str(source_id)
                if current_id is not None and source_id != current_id:
                    finish()
                    if len(source_frames) >= max_sources:
                        break
                current_id = source_id
                general = pd.to_numeric(group["general"], errors="coerce").fillna(0.0)
                controlled = pd.to_numeric(group["controlled"], errors="coerce").fillna(0.0)
                gross = pd.to_numeric(group["gross"], errors="coerce").fillna(0.0)
                other = pd.to_numeric(group["other"], errors="coerce").fillna(0.0)
                for timestamp, load, external in zip(group["timestamp"], general + controlled + other, gross):
                    rows.append((str(timestamp), float(load) * 2.0, float(external) * 2.0))
            if len(source_frames) >= max_sources:
                break
        finish()
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()
    records: list[DeviceDay] = []
    for source_index, (source_id, frame) in enumerate(source_frames):
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.assign(timestamp=timestamps).dropna(subset=["timestamp"]).set_index("timestamp")
        for day_id, (_, load, external, target) in enumerate(_resample_pair_days(frame["load"], frame["input"], max_days=max_days)):
            records.append(_record(
                load, source_id, day_id, source_index, config,
                timestamp=target, source_unit="30min_kwh_to_kw", energy_input=external,
            ))
    source_count = len({r.source_device_id for r in records})
    if source_count < 500:
        raise ValueError(f"SGSC produced only {source_count} independent customers")
    return DeviceDayPool(records, len(records), source_count, 0)


def _load_goiener(config: dict[str, Any]) -> DeviceDayPool:
    """Stream selected one-year user files from the official Zstandard tar."""
    spec = _spec(config)
    path = Path(spec["input"]).resolve()
    max_sources = int(spec.get("max_sources", 600))
    max_days = int(spec.get("max_days_per_source", 20))
    process = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("cannot stream GoiEner archive")
    records: list[DeviceDay] = []
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            source_index = 0
            for member in archive:
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                frame = pd.read_csv(io.BytesIO(handle.read()))
                if "kWh" not in frame.columns:
                    continue
                timestamp = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
                series = pd.Series(pd.to_numeric(frame["kWh"], errors="coerce").to_numpy(float), index=timestamp)
                for day_id, (_, profile, target) in enumerate(_resample_series_days(series, scale=1.0, max_days=max_days, source_unit="hourly_kwh_to_kw")):
                    records.append(_record(
                        profile, Path(member.name).stem, day_id, source_index, config,
                        timestamp=target, source_unit="hourly_kwh_to_kw",
                    ))
                source_index += 1
                if source_index >= max_sources:
                    break
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()
    source_count = len({r.source_device_id for r in records})
    if source_count < 500:
        raise ValueError(f"GoiEner produced only {source_count} independent users")
    return DeviceDayPool(records, len(records), source_count, 0)


def _load_european_lv(config: dict[str, Any], *, urban_zip: bool = False) -> DeviceDayPool:
    spec = _spec(config)
    if "input_dir" in spec and spec.get("input_dir"):
        paths = sorted(Path(spec["input_dir"]).resolve().glob("P*.csv"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
        return _records_from_ordered_files(paths, config, max_days=int(spec.get("max_days_per_source", 4)), value_scale=float(spec.get("load_scale", 1.0)), source_limit=int(spec.get("max_sources", len(paths))))
    archive_path = Path(spec["input"]).resolve()
    with ZipFile(archive_path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if re.search(r"day_20_profile/shape_\d+\.csv$", name)
        )[: int(spec.get("max_sources", 5000))]
        records: list[DeviceDay] = []
        for source_index, name in enumerate(names):
            with archive.open(name) as handle:
                values = np.fromstring(handle.read().decode("utf-8"), sep="\n")
            for day_id in range(min(int(spec.get("max_days_per_source", 2)), values.size // 24)):
                profile = _regular_profile(values[day_id * 24 : (day_id + 1) * 24], scale=float(spec.get("load_scale", 1.0)))
                timestamp = pd.date_range(pd.Timestamp("2011-01-01") + pd.Timedelta(days=day_id), periods=288, freq="5min").view("int64")
                records.append(_record(profile, Path(name).stem, day_id, source_index, config, timestamp=timestamp, source_unit="ordered_hourly_profile"))
    source_count = len({r.source_device_id for r in records})
    if source_count < 500:
        raise ValueError(f"European LV produced only {source_count} independent customers")
    return DeviceDayPool(records, len(records), source_count, 0)


def _profile_on_target(series: pd.Series, target: pd.DatetimeIndex, *, default: float = 0.0) -> np.ndarray:
    """Interpolate one measured day onto the canonical five-minute grid."""
    numeric = pd.to_numeric(series, errors="coerce").dropna().groupby(level=0).mean().sort_index()
    if numeric.empty:
        return np.full(len(target), float(default), dtype=float)
    joined = numeric.reindex(numeric.index.union(target)).sort_index().interpolate(
        method="time", limit_direction="both"
    )
    return joined.reindex(target).fillna(float(default)).to_numpy(dtype=float)


def _measured_pair_days(
    load: pd.Series,
    external: pd.Series,
    *,
    max_days: int,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Build complete days without inventing an external-energy curve."""
    load = pd.to_numeric(load, errors="coerce").dropna().groupby(level=0).mean().sort_index()
    external = pd.to_numeric(external, errors="coerce").dropna().groupby(level=0).mean().sort_index()
    if load.empty:
        return []
    differences = load.index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    positive = differences[differences > 0]
    cadence_seconds = float(np.median(positive)) if positive.size else 300.0
    expected_points = max(2, int(round(86400.0 / max(cadence_seconds, 1.0))))
    minimum_points = max(2, int(np.ceil(expected_points * 0.75)))
    tolerance = pd.Timedelta(seconds=max(cadence_seconds * 1.5, 300.0))
    results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for day in sorted(load.index.normalize().unique()):
        start = pd.Timestamp(day)
        end = start + pd.Timedelta(days=1)
        load_day = load[(load.index >= start) & (load.index < end)]
        if (
            len(load_day) < minimum_points
            or load_day.index.min() > start + tolerance
            or load_day.index.max() < end - tolerance
        ):
            continue
        input_day = (
            external[(external.index >= start) & (external.index < end)]
            if not external.empty
            else external
        )
        target = pd.date_range(start, periods=288, freq="5min")
        load_values = np.maximum(_profile_on_target(load_day, target), 0.0)
        input_values = np.maximum(_profile_on_target(input_day, target), 0.0)
        if np.isfinite(load_values).all() and np.isfinite(input_values).all():
            results.append((start.strftime("%Y-%m-%d"), load_values, input_values, target.view("int64")))
        if len(results) >= int(max_days):
            break
    return results


def _load_norway_ami(config: dict[str, Any]) -> DeviceDayPool:
    """Load observed hourly AMI import and net-return channels from Parquet."""
    import duckdb

    spec = _spec(config)
    root = Path(spec["input_dir"]).resolve()
    paths = sorted(root.glob("*.parquet"))[: int(spec.get("max_sources", 3000))]
    if not paths:
        raise FileNotFoundError(f"no Norway AMI Parquet files under {root}")
    max_days = int(spec.get("max_days_per_source", 2))
    row_limit = (max_days + 3) * 24
    records: list[DeviceDay] = []
    connection = duckdb.connect()
    try:
        for batch_start in range(0, len(paths), 128):
            batch = paths[batch_start : batch_start + 128]
            frame = connection.execute(
                """
                SELECT filename, dateTime, activePowerOut, activePowerIn
                FROM (
                    SELECT filename, dateTime, activePowerOut, activePowerIn,
                           row_number() OVER (PARTITION BY filename ORDER BY dateTime) AS source_row
                    FROM read_parquet(?, filename=true)
                )
                WHERE source_row <= ?
                ORDER BY filename, dateTime
                """,
                [[str(path) for path in batch], row_limit],
            ).fetchdf()
            for filename, source_frame in frame.groupby("filename", sort=True):
                source_index = batch_start + batch.index(Path(str(filename)))
                timestamp = pd.to_datetime(source_frame["dateTime"], errors="coerce")
                load = pd.Series(
                    pd.to_numeric(source_frame["activePowerOut"], errors="coerce").to_numpy(dtype=float),
                    index=timestamp,
                )
                external = pd.Series(
                    pd.to_numeric(source_frame["activePowerIn"], errors="coerce").to_numpy(dtype=float),
                    index=timestamp,
                )
                source_id = Path(str(filename)).stem
                for day_id, (_, profile, measured_input, target) in enumerate(
                    _measured_pair_days(load, external, max_days=max_days)
                ):
                    records.append(_record(
                        profile,
                        source_id,
                        day_id,
                        source_index,
                        config,
                        timestamp=target,
                        source_unit="hourly_active_power_kw",
                        energy_input=measured_input,
                    ))
    finally:
        connection.close()
    source_count = len({record.source_device_id for record in records})
    if source_count < 500:
        raise ValueError(f"Norway AMI produced only {source_count} independent meters")
    return DeviceDayPool(records, len(records), source_count, 0)


def _load_camsl(config: dict[str, Any]) -> DeviceDayPool:
    """Load CAMSL trial households without constructing an energy input."""
    spec = _spec(config)
    root = Path(spec["input_dir"]).resolve()
    source_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: int(path.name))
    source_dirs = source_dirs[: int(spec.get("max_sources", 1423))]
    max_days = int(spec.get("max_days_per_source", 2))
    scale = float(spec.get("load_scale", 0.002))
    records: list[DeviceDay] = []
    for source_index, source_dir in enumerate(source_dirs):
        frames: list[pd.DataFrame] = []
        for path in sorted(source_dir.glob("*.csv"))[:2]:
            frames.append(pd.read_csv(
                path,
                header=None,
                names=["timestamp", "slot", "value"],
                usecols=[0, 2],
            ))
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        series = pd.Series(
            pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float),
            index=timestamp,
        )
        profiles = _resample_series_days(
            series,
            scale=scale,
            max_days=max_days,
            source_unit="raw_half_hour_value_assumed_Wh_to_kW",
        )
        for day_id, (_, profile, target) in enumerate(profiles):
            records.append(_record(
                profile,
                source_dir.name,
                day_id,
                source_index,
                config,
                timestamp=target,
                source_unit="raw_half_hour_value_assumed_Wh_to_kW",
            ))
    source_count = len({record.source_device_id for record in records})
    if source_count < 500:
        raise ValueError(f"CAMSL produced only {source_count} independent households")
    return DeviceDayPool(records, len(records), source_count, 0)


def _parse_irish_timestamps(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.strip()
    dayfirst = True
    for value in raw.head(200):
        match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", value)
        if match and int(match.group(2)) > 12:
            dayfirst = False
            break
        if match and int(match.group(1)) > 12:
            dayfirst = True
            break
    return pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=dayfirst)


def _load_irish(config: dict[str, Any]) -> DeviceDayPool:
    """Load measured Irish import/export under the source kW-label interpretation."""
    spec = _spec(config)
    root = Path(spec["input_dir"]).resolve()
    paths = sorted(
        root.glob("MPRN_*.csv"),
        key=lambda path: int(re.search(r"\d+", path.stem).group()),
    )[: int(spec.get("max_sources", 2989))]
    max_days = int(spec.get("max_days_per_source", 2))
    scale = float(spec.get("load_scale", 1.0))
    records: list[DeviceDay] = []
    rows_to_read = (max_days + 4) * 96
    for source_index, path in enumerate(paths):
        frame = pd.read_csv(path, nrows=rows_to_read)
        timestamp = _parse_irish_timestamps(frame["Read Date and End Time"])
        value = pd.to_numeric(frame["Read Value"], errors="coerce") * scale
        kind = frame["Read Type"].astype(str)
        valid = timestamp.notna() & value.notna()
        indexed = pd.DataFrame({
            "timestamp": timestamp[valid],
            "value": value[valid],
            "kind": kind[valid],
        })
        imported = indexed[indexed["kind"].str.contains("Import", na=False)].groupby("timestamp")["value"].sum()
        exported = indexed[indexed["kind"].str.contains("Export", na=False)].groupby("timestamp")["value"].sum()
        for day_id, (_, profile, measured_export, target) in enumerate(
            _measured_pair_days(imported, exported, max_days=max_days)
        ):
            records.append(_record(
                profile,
                path.stem,
                day_id,
                source_index,
                config,
                timestamp=target,
                source_unit="source_interval_kw_label_primary_interpretation",
                energy_input=measured_export,
            ))
    source_count = len({record.source_device_id for record in records})
    if source_count < 500:
        raise ValueError(f"Irish domestic adapter produced only {source_count} independent MPRNs")
    return DeviceDayPool(records, len(records), source_count, 0)


def _load_opsd(config: dict[str, Any]) -> DeviceDayPool:
    """Load OPSD site import, PV and available controllable-device channels."""
    spec = _spec(config)
    path = Path(spec["input"]).resolve()
    columns = list(pd.read_csv(path, nrows=0).columns)
    sites = sorted({column[: -len("_grid_import")] for column in columns if column.endswith("_grid_import")})
    selected = ["utc_timestamp"] + [
        column for column in columns if any(column.startswith(f"{site}_") for site in sites)
    ]
    frame = pd.read_csv(path, usecols=selected)
    timestamp = pd.to_datetime(frame.pop("utc_timestamp"), errors="coerce", utc=True)
    frame.index = timestamp
    scale = float(spec.get("load_scale", 4.0))
    max_days = int(spec.get("max_days_per_source", 30))
    records: list[DeviceDay] = []

    def cumulative_to_power(column: pd.Series) -> pd.Series:
        cumulative = pd.to_numeric(column, errors="coerce")
        increments = cumulative.diff()
        # Negative jumps are meter resets or invalid joins, not generation.
        return increments.where(increments >= 0.0) * scale

    for source_index, site in enumerate(sites):
        load = cumulative_to_power(frame[f"{site}_grid_import"])
        pv_columns = [
            column for column in frame.columns
            if column.startswith(f"{site}_pv")
        ]
        external = (
            pd.concat([cumulative_to_power(frame[column]) for column in pv_columns], axis=1)
            .sum(axis=1, min_count=1).fillna(0.0)
            if pv_columns else pd.Series(0.0, index=frame.index)
        )
        positive_baseline_columns = [
            column for column in frame.columns
            if column in {f"{site}_storage_charge", f"{site}_ev", f"{site}_heat_pump"}
        ]
        baseline = (
            pd.concat([cumulative_to_power(frame[column]) for column in positive_baseline_columns], axis=1)
            .sum(axis=1, min_count=1).fillna(0.0)
            if positive_baseline_columns else pd.Series(0.0, index=frame.index)
        )
        discharge_column = f"{site}_storage_decharge"
        if discharge_column in frame.columns:
            baseline = baseline - cumulative_to_power(frame[discharge_column]).fillna(0.0)
        for day_id, (_, profile, measured_pv, target_values) in enumerate(
            _measured_pair_days(load, external, max_days=max_days)
        ):
            target = pd.to_datetime(target_values, utc=True)
            day_start = target[0].normalize()
            baseline_day = baseline[
                (baseline.index >= day_start) & (baseline.index < day_start + pd.Timedelta(days=1))
            ]
            baseline_profile = _profile_on_target(baseline_day, target)
            records.append(_record(
                profile,
                site,
                day_id,
                source_index,
                config,
                timestamp=target_values,
                source_unit="15min_kwh_to_kw",
                energy_input=measured_pv,
                pv=measured_pv,
                baseline=baseline_profile,
            ))
    source_count = len({record.source_device_id for record in records})
    if source_count != len(sites):
        raise ValueError(f"OPSD produced {source_count} sources from {len(sites)} sites")
    return DeviceDayPool(records, len(records), source_count, 0)


def _load_complete_energy_community(config: dict[str, Any]) -> DeviceDayPool:
    """Load the downloaded source-constructed PV/BESS/EV community workbook."""
    spec = _spec(config)
    path = Path(spec["input"]).resolve()
    load = pd.read_excel(path, sheet_name="Load", header=0, index_col=0)
    pv = pd.read_excel(path, sheet_name="PV", header=0, index_col=0)
    ev1 = pd.read_excel(path, sheet_name="EV1 Load", header=0, index_col=0)
    ev2 = pd.read_excel(path, sheet_name="EV2 Load", header=0, index_col=0)
    bess = pd.read_excel(path, sheet_name="BESS", header=0, index_col=0)
    column_by_player = {str(column): column for column in load.columns}
    bess_column_by_player = {str(column): column for column in bess.columns}
    players = sorted(column_by_player, key=int)[: int(spec.get("max_sources", 250))]
    timestamp = pd.date_range("2023-01-01", periods=288, freq="5min").view("int64")
    records: list[DeviceDay] = []
    for source_index, player in enumerate(players):
        column = column_by_player[player]
        profile = _regular_profile(load[column].to_numpy(dtype=float))
        measured_pv = _regular_profile(pv[column].to_numpy(dtype=float))
        baseline = _regular_profile(
            ev1[column].to_numpy(dtype=float) + ev2[column].to_numpy(dtype=float)
        )
        capacity = peak = initial_fraction = None
        bess_column = bess_column_by_player.get(player)
        if bess_column is not None:
            capacity_value = pd.to_numeric(bess.at["Capacity (kW)", bess_column], errors="coerce")
            charge_value = pd.to_numeric(bess.at["Charge (kW)", bess_column], errors="coerce")
            discharge_value = pd.to_numeric(bess.at["Discharge (kW)", bess_column], errors="coerce")
            initial_value = pd.to_numeric(bess.at["Initial (kW)", bess_column], errors="coerce")
            if pd.notna(capacity_value) and float(capacity_value) > 0:
                capacity = float(capacity_value)
                peak = max(float(charge_value or 0.0), float(discharge_value or 0.0))
                if pd.notna(initial_value):
                    initial_fraction = float(initial_value) / max(capacity, 1e-9)
        records.append(_record(
            profile,
            f"player_{player}",
            0,
            source_index,
            config,
            timestamp=timestamp,
            source_unit="source_constructed_15min_kw_profile",
            energy_input=measured_pv,
            pv=measured_pv,
            baseline=baseline,
            capacity_kwh=capacity,
            peak_power_kw=peak,
            soc_fraction=initial_fraction,
        ))
    return DeviceDayPool(records, len(records), len(records), 0)


def _load_uci_appliances(config: dict[str, Any]) -> DeviceDayPool:
    spec = _spec(config)
    path = Path(spec["input"]).resolve()
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("energydata_complete.csv")]
        if not names:
            raise FileNotFoundError("energydata_complete.csv is missing from UCI Appliances archive")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle)
    profiles = _full_day_profiles(
        frame.rename(columns={"date": "timestamp"}),
        "Appliances",
        float(spec.get("load_scale", 0.1)),
        int(spec.get("max_days_per_source", 180)),
    )
    records = [
        _record(values, "uci_appliances_home", day_id, day_id, config, timestamp=timestamp, source_unit="Wh/10min")
        for day_id, (_, values, timestamp) in enumerate(profiles)
    ]
    if len(records) < 6:
        raise ValueError("UCI Appliances has fewer than six complete days after 5-minute mapping")
    return DeviceDayPool(records, len(records), 1, 0)


def _electricity_profile(values: np.ndarray, scale: float) -> np.ndarray:
    values = pd.to_numeric(pd.Series(values), errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    if values.size != 96:
        raise ValueError(f"expected 96 quarter-hour values, got {values.size}")
    source_x = np.arange(96, dtype=float) / 4.0 + 0.25
    target_x = np.arange(288, dtype=float) / 12.0
    return np.maximum(np.interp(target_x, source_x, values, left=values[0], right=values[-1]) * float(scale), 0.0)


def _load_uci_electricity(config: dict[str, Any]) -> DeviceDayPool:
    spec = _spec(config)
    path = Path(spec["input"]).resolve()
    max_days = int(spec.get("max_source_days", 30))
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("LD2011_2014.txt")]
        if not names:
            raise FileNotFoundError("LD2011_2014.txt is missing from UCI load archive")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, sep=";", decimal=",", nrows=96 * max_days)
    frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    timestamps = pd.to_datetime(frame.pop("timestamp"), errors="coerce")
    frame.index = timestamps
    records: list[DeviceDay] = []
    for source_index, column in enumerate(frame.columns):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        complete_days = len(values) // 96
        for day_id in range(min(complete_days, max_days)):
            block = values[day_id * 96 : (day_id + 1) * 96]
            profile = _electricity_profile(block, float(spec.get("load_scale", 0.001)))
            target = pd.date_range(pd.Timestamp("2011-01-01") + pd.Timedelta(days=day_id), periods=288, freq="5min")
            records.append(_record(
                profile,
                str(column),
                day_id,
                source_index * max_days + day_id,
                config,
                timestamp=target.view("int64"),
                source_unit="source_value_scaled_to_kw",
            ))
    if len(records) < 9 * 500:
        raise ValueError(f"UCI Electricity adapter produced only {len(records)} records; need 4500")
    return DeviceDayPool(records, len(records), len(frame.columns), 0)


def load_canonical_device_day_pool(config: dict[str, Any]) -> DeviceDayPool:
    spec = _spec(config)
    dataset = str(spec.get("dataset", ""))
    cache = _cache_path(spec)
    if cache is not None:
        cached = _load_cache(cache)
        if cached is not None:
            return _apply_input_mapping(cached, spec)
    loaders = {
        "uci_appliances_energy": _load_uci_appliances,
        "uci_electricity_load_diagrams": _load_uci_electricity,
        "bdg1_building_data_genome": _load_bdg1,
        "bdg2_building_data_genome": _load_bdg2,
        "low_carbon_london": _load_lcl,
        "danish_smart_heat_meters": _load_danish,
        "smart_grid_smart_city": _load_sgsc,
        "heapo_heat_pumps": _load_heapo,
        "goiener_smart_meters": _load_goiener,
        "european_lv_urban_8087": _load_european_lv,
        "european_lv_rural_2731": _load_european_lv,
        "european_lv_urban_35297": _load_european_lv,
        "norway_ami_energy_distribution": _load_norway_ami,
        "camsl_japan_smart_meters": _load_camsl,
        "irish_domestic_smart_meters": _load_irish,
        "opsd_household_data": _load_opsd,
        "complete_energy_community": _load_complete_energy_community,
    }
    if dataset not in loaders:
        raise ValueError(f"no canonical adapter is implemented for {dataset!r}")
    pool = loaders[dataset](config)
    if cache is not None:
        _save_cache(cache, pool)
    return _apply_input_mapping(pool, spec)
