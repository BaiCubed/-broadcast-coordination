from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.extra.dataset_combinations.constants import BDG1, DATASET_SOURCES
from src.extra.dataset_experiment.canonical_adapter import load_canonical_device_day_pool
from src.extra.ieee33_device_day_simulation.config_loader import load_config
from src.extra.ieee33_device_day_simulation.population.device_day_loader import load_device_day_pool

ROOT = Path.cwd()
RESULTS = ROOT / "results"
N_CANDIDATES = [50, 100, 150, 250, 400, 650, 1000, 1600, 2000, 2500, 3000]

names = sys.argv[1:] or ["mix_S1-A_s00", "mix_S4-A_s00", "mix_P001_s00"]

source_pools: dict[str, list] = {}
bdg1_index: dict[tuple[str, int], int] = {}


def source_records(dataset: str) -> list:
    if dataset not in source_pools:
        config = load_config(RESULTS / ("%s_ieee33_real_load" % dataset) / "config" / "default_weak_correlation.yaml")
        spec = config["population"]["canonical_adapter"]
        if dataset == BDG1 and not spec.get("cache"):
            spec["cache"] = str((ROOT / "data/bdg1_building_data_genome/processed/canonical_device_days.npz").resolve())
        source_pools[dataset] = load_canonical_device_day_pool(config).records
        if dataset == BDG1:
            for i, r in enumerate(source_pools[dataset]):
                bdg1_index[(r.source_device_id, int(r.day_id))] = i
    return source_pools[dataset]


def legal_n_values(candidates, maximum, include_maximum=True):
    if maximum < 2:
        return []
    limit = maximum if include_maximum else min(maximum, max(candidates))
    values = {int(n) for n in candidates if 2 <= int(n) <= limit}
    if include_maximum:
        values.add(maximum)
    minimum_points = min(5, limit - 1) if limit < 50 else 3
    if len(values) < minimum_points:
        values.update(int(round(v)) for v in np.linspace(2, limit, minimum_points))
    return sorted(v for v in values if 2 <= v <= limit)


report = []
for name in names:
    config_path = RESULTS / ("%s_ieee33_real_load" % name) / "config" / "default_weak_correlation.yaml"
    config = load_config(config_path)
    pool = load_device_day_pool(config)
    records = pool.records

    manifest = json.loads(Path("/tmp/pools_smoke.json").read_text())
    entry = next(e for e in manifest["entries"] if e["name"] == name)
    with (Path(entry["fleet_dir"]) / "devices.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(records) == len(rows), (name, len(records), len(rows))

    step = max(1, len(rows) // 200)
    checked = mismatch = 0
    for i in range(0, len(rows), step):
        row, rec = rows[i], records[i]
        dataset = row["dataset_id"]
        src = source_records(dataset)
        if dataset == BDG1:
            origin = src[bdg1_index[(row["source_device_id"].split("__", 1)[-1], int(row["day_id"]))]]
        else:
            origin = src[int(row["source_row_index"])]
        ok = (np.array_equal(np.asarray(rec.load_kw, dtype=np.float32), np.asarray(origin.load_kw, dtype=np.float32))
              and np.array_equal(np.asarray(rec.energy_input_kw, dtype=np.float32),
                                 np.asarray(origin.energy_input_kw, dtype=np.float32))
              and rec.zone_id == row["zone_id"] and int(rec.bus_id) == int(row["bus_id"])
              and rec.source_device_id == row["source_device_id"])
        checked += 1
        mismatch += 0 if ok else 1

    unique_sources = len({r.source_device_id for r in records})
    zone_sources: dict[str, set] = {}
    for r in records:
        zone_sources.setdefault(r.zone_id, set()).add(r.source_device_id)
    balanced = len(zone_sources) * min(len(v) for v in zone_sources.values())
    configured = int(config["population"]["N_simulated_resources"])
    maximum = min(configured, balanced)
    grid = legal_n_values(N_CANDIDATES, maximum)

    caps = {float(r.capacity_kwh) for r in records}
    peaks = {float(r.peak_power_kw) for r in records}

    row_out = {
        "name": name, "records": len(records), "unique_sources": unique_sources,
        "duplicate_sources": len(records) - unique_sources,
        "zone_unique": {k: len(v) for k, v in sorted(zone_sources.items())},
        "balanced_unique_limit": balanced, "N_simulated_resources": configured,
        "e1_n_grid": grid, "profiles_checked": checked, "profile_mismatch": mismatch,
        "capacity_kwh_distinct": sorted(caps), "peak_kw_distinct": sorted(peaks),
        "network_capacity_multiplier": float(config["control"]["network_capacity_multiplier"]),
        "datasets_in_fleet": sorted({r["dataset_id"] for r in rows}),
    }
    report.append(row_out)
    print(json.dumps(row_out, ensure_ascii=False))

bad = [r for r in report if r["profile_mismatch"] or r["duplicate_sources"]]
print()
print("point-by-point comparison passed" if not bad else "mismatches: %s" % [r["name"] for r in bad])
