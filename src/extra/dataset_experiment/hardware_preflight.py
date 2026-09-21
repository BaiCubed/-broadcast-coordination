from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.extra.ieee33_device_day_simulation.population.device_day_loader import DeviceDayPool


def run_preflight(config: dict[str, Any], pool: DeviceDayPool) -> dict[str, Any]:
    population = config["population"]
    zones = config["zones"]["zones"]
    resources = int(population["N_simulated_resources"])
    validation_batches = int(config["experiment"]["validation_batches"])
    source_unique = bool(population.get("unique_source_per_batch", False))
    required_per_zone = int(population["resources_per_zone"]) * (
        1 if source_unique else (1 + validation_batches)
    )
    zone_counts = {zone: len(pool.by_zone(zone)) for zone in sorted(zones)}
    zone_source_counts = {
        zone: len({record.source_device_id for record in pool.by_zone(zone)})
        for zone in sorted(zones)
    }
    records = pool.records
    finite = all(
        np.isfinite(np.asarray(record.load_kw, dtype=float)).all()
        and np.isfinite(np.asarray(record.soc_kwh, dtype=float)).all()
        and len(record.load_kw) == int(config["simulation"]["steps_per_day"])
        for record in records
    )
    physical_errors: list[str] = []
    for record in records:
        if not (5.0 <= float(record.capacity_kwh) <= 20.0):
            physical_errors.append(f"{record.device_id}: capacity outside original range")
        if float(record.peak_power_kw) <= 0 or float(record.peak_power_kw) > float(record.capacity_kwh) * 2.0:
            physical_errors.append(f"{record.device_id}: peak power exceeds 2C device bound")
        if not (0.10 <= record.initial_soc <= 0.95):
            physical_errors.append(f"{record.device_id}: initial SOC outside configured device range")
        if (np.asarray(record.load_kw) < 0).any():
            physical_errors.append(f"{record.device_id}: negative load")
    available_bytes = None
    try:
        available_bytes = int(os.statvfs(Path.cwd()).f_bavail * os.statvfs(Path.cwd()).f_frsize)
    except OSError:
        pass
    estimated_bytes = resources * int(config["simulation"]["steps_per_day"]) * 16 * 8
    memory_ok = available_bytes is None or estimated_bytes < available_bytes * 0.50
    warnings: list[str] = []
    if pool.source_count < 500:
        message = (
            f"dataset has only {pool.source_count} independent source units; "
            "simulated resources reuse source profiles and cannot support source-unique scaling claims"
        )
        if population.get("allow_pool_bootstrap", False):
            warnings.append(message)
        else:
            physical_errors.append(message)
    available_counts = zone_source_counts if source_unique else zone_counts
    if any(count < required_per_zone for count in available_counts.values()) and not population.get("with_replacement", False):
        physical_errors.append(f"pool does not contain {required_per_zone} records per zone without replacement")
    transformer_capacity = float(config["network"]["transformer"]["capacity_kw"])
    fleet_peak = resources * max(float(record.peak_power_kw) for record in records)
    if fleet_peak > transformer_capacity:
        warnings.append(
            f"theoretical fleet peak {fleet_peak:.1f} kW exceeds transformer {transformer_capacity:.1f} kW; "
            "IEEE33 network feedback must curtail accepted dispatch"
        )
    if not finite:
        physical_errors.append("non-finite or non-288-point profile found")
    if not memory_ok:
        physical_errors.append("estimated trace memory exceeds half of available filesystem space")
    return {
        "status": "pass" if not physical_errors else "blocked",
        "dataset_pool": {
            "records": len(records),
            "source_devices": pool.source_count,
            "filled_values": pool.filled_values,
            "zone_counts": zone_counts,
            "zone_source_counts": zone_source_counts,
        },
        "experiment_request": {
            "resources_per_zone": int(population["resources_per_zone"]),
            "N_simulated_resources": resources,
            "validation_batches": validation_batches,
            "required_records_per_zone": required_per_zone,
        },
        "hardware": {
            "cpu_threads": os.cpu_count(),
            "available_filesystem_bytes": available_bytes,
            "estimated_trace_bytes": estimated_bytes,
            "memory_check_passed": memory_ok,
        },
        "physical_range_check": {
            "max_device_peak_kw": max(float(record.peak_power_kw) for record in records),
            "max_device_capacity_kwh": max(float(record.capacity_kwh) for record in records),
            "transformer_capacity_kw": transformer_capacity,
            "fleet_theoretical_peak_kw": fleet_peak,
            "errors": physical_errors,
            "warnings": warnings,
        },
    }
