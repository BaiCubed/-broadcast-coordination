"""NextGen source-device to device-day population adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


CSV_COLUMNS = {
    "timestamp": "original index",
    "load": "load power (kW)",
    "solar": "solar power (kW)",
    "battery": "battery power (kW)",
    "soc_kwh": "battery SoC (kWh)",
    "solar_capacity": "solar capacity (kW)",
    "capacity": "battery capacity (kWh)",
    "peak_power": "battery peak power (kW)",
}


@dataclass
class DeviceDay:
    """One independent resource with one complete 288-point day profile."""

    device_id: str
    source_device_id: str
    day_id: int
    zone_id: str
    bus_id: int
    load_kw: np.ndarray
    pv_kw: np.ndarray
    # Generic positive external energy input used by the EPS signal logic.
    # ``pv_kw`` remains as a legacy diagnostic/source field.
    energy_input_kw: np.ndarray
    baseline_battery_kw: np.ndarray
    soc_kwh: np.ndarray
    capacity_kwh: float
    peak_power_kw: float
    c_rate: float
    timestamp: np.ndarray
    initial_soh: float = 0.95

    @property
    def initial_soc(self) -> float:
        return float(np.clip(self.soc_kwh[0] / max(self.capacity_kwh, 1e-6), 0.05, 0.98))

    @property
    def input_components_kw(self) -> dict[str, np.ndarray]:
        """Return named input components without changing the old data contract."""
        return {"pv": self.pv_kw, "energy_input": self.energy_input_kw}


def _source_id(path: Path) -> str:
    match = re.search(r"_id_([^./]+)", path.name)
    return match.group(1) if match else path.stem


class DeviceDayPool:
    """Lazy, deterministic pool of all 36,500 device-day records."""

    def __init__(self, records: list[DeviceDay], metadata_count: int, source_count: int, filled_values: int = 0):
        self.records = records
        self.metadata_count = metadata_count
        self.source_count = source_count
        self.filled_values = filled_values

    def by_zone(self, zone_id: str) -> list[DeviceDay]:
        return [record for record in self.records if record.zone_id == zone_id]

    def sample(self, per_zone: int, seed: int, with_replacement: bool = False) -> list[DeviceDay]:
        rng = np.random.default_rng(seed)
        selected: list[DeviceDay] = []
        zones = sorted({record.zone_id for record in self.records})
        for zone_id in zones:
            candidates = self.by_zone(zone_id)
            if not candidates:
                raise ValueError(f"No device-day records mapped to {zone_id}")
            if not with_replacement and per_zone > len(candidates):
                raise ValueError(f"{zone_id} has only {len(candidates)} records; requested {per_zone}")
            indices = rng.choice(len(candidates), size=per_zone, replace=with_replacement)
            selected.extend(candidates[int(index)] for index in indices)
        return selected

    def sample_source_unique_batches(
        self,
        per_zone: int,
        batch_count: int,
        seed: int,
    ) -> list[list[DeviceDay]]:
        """Sample one record per source in each independently drawn batch.

        A physical source may contribute a different day to another batch, but
        can never appear twice in one simultaneous fleet snapshot.
        """
        rng = np.random.default_rng(seed)
        zones = sorted({record.zone_id for record in self.records})
        by_zone_source: dict[str, dict[str, list[DeviceDay]]] = {}
        for zone in zones:
            grouped: dict[str, list[DeviceDay]] = {}
            for record in self.by_zone(zone):
                grouped.setdefault(record.source_device_id, []).append(record)
            if len(grouped) < per_zone:
                raise ValueError(
                    f"{zone} has only {len(grouped)} independent sources; requested {per_zone}"
                )
            by_zone_source[zone] = grouped
        batches: list[list[DeviceDay]] = []
        for _ in range(batch_count):
            batch: list[DeviceDay] = []
            for zone in zones:
                grouped = by_zone_source[zone]
                source_ids = sorted(grouped)
                indices = rng.choice(len(source_ids), size=per_zone, replace=False)
                for index in indices:
                    candidates = grouped[source_ids[int(index)]]
                    batch.append(candidates[int(rng.integers(0, len(candidates)))])
            batches.append(batch)
        return batches

    def sample_unique_sources(self, count: int, seed: int) -> list[DeviceDay]:
        """Sample a zone-balanced fleet with globally unique source IDs."""
        rng = np.random.default_rng(seed)
        zones = sorted({record.zone_id for record in self.records})
        base, remainder = divmod(int(count), len(zones))
        selected: list[DeviceDay] = []
        for zone_index, zone in enumerate(zones):
            requested = base + (1 if zone_index < remainder else 0)
            grouped: dict[str, list[DeviceDay]] = {}
            for record in self.by_zone(zone):
                grouped.setdefault(record.source_device_id, []).append(record)
            if len(grouped) < requested:
                raise ValueError(
                    f"{zone} has only {len(grouped)} independent sources; requested {requested}"
                )
            source_ids = sorted(grouped)
            indices = rng.choice(len(source_ids), size=requested, replace=False)
            for index in indices:
                candidates = grouped[source_ids[int(index)]]
                selected.append(candidates[int(rng.integers(0, len(candidates)))])
        if len({record.source_device_id for record in selected}) != len(selected):
            raise RuntimeError("source-unique sampling produced a duplicate source ID")
        return selected


def _assignment(source_index: int, source_id: str, config: dict[str, Any]) -> tuple[str, int]:
    mapping = config["source_device_map"]
    zones_cfg = config["zones"]["zones"]
    zone_order = list(mapping["zone_order"])
    zone_id = zone_order[source_index % len(zone_order)]
    buses = list(zones_cfg[zone_id]["buses"])
    local_index = source_index // len(zone_order)
    bus_id = int(buses[local_index % len(buses)])
    return zone_id, bus_id


def load_device_day_pool(config: dict[str, Any]) -> DeviceDayPool:
    """Read source CSVs once and materialize the complete device-day pool.

    Each source file is a year of 5-minute records.  A source device's days
    retain one fixed zone and bus, so the same household is never split across
    network regions during resampling.
    """
    if config["population"].get("resource_unit") == "transaction-session":
        from .data2_transaction_loader import load_data2_transaction_pool

        return load_data2_transaction_pool(config)
    if config["population"].get("resource_unit") == "canonical-device-day":
        from src.extra.dataset_experiment.canonical_adapter import load_canonical_device_day_pool

        return load_canonical_device_day_pool(config)
    data_root = Path(config["paths"]["data_root"])
    if not data_root.is_absolute():
        data_root = Path.cwd() / data_root
    files = sorted(data_root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No NextGen CSV files found under {data_root}")
    expected = int(config["population"]["source_device_count_expected"])
    if len(files) != expected:
        raise ValueError(f"Expected {expected} source files, found {len(files)}")

    records: list[DeviceDay] = []
    mapping = config["source_device_map"]
    assignment_mode = mapping.get("assignment_mode", "round_robin")
    random_assignments: list[tuple[str, int]] = []
    if assignment_mode == "random_device_day":
        total_days = int(config["population"]["device_days_expected"])
        zone_order = list(mapping["zone_order"])
        rng = np.random.default_rng(int(mapping.get("assignment_seed", 0)))
        zone_indices = np.tile(np.arange(len(zone_order)), int(np.ceil(total_days / len(zone_order))))[:total_days]
        rng.shuffle(zone_indices)
        bus_counters = {zone: 0 for zone in zone_order}
        for zone_index in zone_indices:
            zone = zone_order[int(zone_index)]
            buses = list(config["zones"]["zones"][zone]["buses"])
            bus_offset = int(rng.integers(0, len(buses)))
            bus = int(buses[(bus_counters[zone] + bus_offset) % len(buses)])
            bus_counters[zone] += 1
            random_assignments.append((zone, bus))
    elif assignment_mode != "round_robin":
        raise ValueError(f"unsupported device-day assignment_mode: {assignment_mode}")
    assignment_cursor = 0
    filled_values = 0
    for source_index, path in enumerate(files):
        source_id = _source_id(path)
        source_zone_id, source_bus_id = _assignment(source_index, source_id, config)
        frame = pd.read_csv(path, usecols=list(CSV_COLUMNS.values()))
        numeric_columns = [value for key, value in CSV_COLUMNS.items() if key != "timestamp"]
        missing_before_file = int(frame[numeric_columns].isna().sum().sum())
        if missing_before_file:
            # Interpolate over the complete source year so a fully missing day
            # can use the measured days immediately before and after it.
            frame[numeric_columns] = frame[numeric_columns].interpolate(limit_direction="both")
            if frame[numeric_columns].isna().any().any():
                raise ValueError(f"Unrecoverable missing values in {path}")
            filled_values += missing_before_file
        step_count = int(config["simulation"]["steps_per_day"])
        if len(frame) % step_count != 0:
            raise ValueError(f"{path} has {len(frame)} rows, not a whole number of days")
        day_count = len(frame) // step_count
        for day_id in range(day_count):
            block = frame.iloc[day_id * step_count : (day_id + 1) * step_count]
            if random_assignments:
                zone_id, bus_id = random_assignments[assignment_cursor]
                assignment_cursor += 1
            else:
                zone_id, bus_id = source_zone_id, source_bus_id
            capacity = float(block[CSV_COLUMNS["capacity"]].iloc[0])
            peak = float(block[CSV_COLUMNS["peak_power"]].iloc[0])
            if capacity <= 0 or peak <= 0:
                raise ValueError(f"Invalid battery parameters in {path}, day {day_id}")
            records.append(
                DeviceDay(
                    device_id=f"{source_id}_day_{day_id:03d}",
                    source_device_id=source_id,
                    day_id=day_id,
                    zone_id=zone_id,
                    bus_id=bus_id,
                    load_kw=block[CSV_COLUMNS["load"]].to_numpy(dtype=float),
                    # NextGen stores PV as a negative injection.
                    pv_kw=-block[CSV_COLUMNS["solar"]].to_numpy(dtype=float),
                    energy_input_kw=-block[CSV_COLUMNS["solar"]].to_numpy(dtype=float),
                    baseline_battery_kw=block[CSV_COLUMNS["battery"]].to_numpy(dtype=float),
                    soc_kwh=block[CSV_COLUMNS["soc_kwh"]].to_numpy(dtype=float),
                    capacity_kwh=capacity,
                    peak_power_kw=peak,
                    c_rate=float(np.clip(peak / capacity, 0.05, 2.0)),
                    timestamp=block[CSV_COLUMNS["timestamp"]].to_numpy(dtype=np.int64),
                )
            )
    return DeviceDayPool(
        records=records,
        metadata_count=len(records),
        source_count=len(files),
        filled_values=filled_values,
    )
