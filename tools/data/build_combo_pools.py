from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from src.extra.dataset_combinations.constants import BDG1, DATASET_SOURCES, SCENARIOS
from src.extra.dataset_combinations.generator import pair_specs
from src.extra.dataset_experiment.canonical_adapter import load_canonical_device_day_pool
from src.extra.dataset_experiment.run import _capacity_multiplier, _mode_config
from src.extra.ieee33_device_day_simulation.config_loader import load_config

ROOT = Path.cwd()
COMBO_ROOT = ROOT / "data/dataset_combinations"
POOL_ROOT = ROOT / "data/dataset_combination_pools"
RESULTS = ROOT / "results"

BASE_CAPACITY_KWH = 10.0
BASE_PEAK_KW = 1.0
BASE_C_RATE = 0.1
BASE_SOC_KWH = 5.0

MAX_SIMULATED_RESOURCES = 3000

TEMPLATE_DATASET = "low_carbon_london"

_SOURCE_POOLS: dict[str, list] = {}
_BDG1_INDEX: dict[tuple[str, int], int] = {}


def combo_id_to_name(kind: str, cid: str, seed_index: int) -> str:
    return "mix_%s_s%02d" % (cid, seed_index)


def combo_fleet_dir(kind: str, cid: str, seed_index: int) -> Path:
    if kind == "scenario":
        return COMBO_ROOT / "unique" / "ieee33" / cid / "test" / ("seed_%02d" % seed_index)
    return COMBO_ROOT / "unique" / "ieee33" / "pairwise" / cid / ("seed_%02d" % seed_index)


def all_combos(seed_indices: list[int]) -> list[tuple[str, str, int]]:
    items = []
    for seed_index in seed_indices:
        for cid in SCENARIOS:
            items.append(("scenario", cid, seed_index))
        for pair in pair_specs():
            items.append(("pair", pair["pair_id"], seed_index))
    return items


def _dataset_config_path(dataset: str) -> Path:
    return RESULTS / ("%s_ieee33_real_load" % dataset) / "config" / "default_weak_correlation.yaml"


def load_source_pools() -> None:
    for name in DATASET_SOURCES:
        config = load_config(_dataset_config_path(name))
        spec = config["population"]["canonical_adapter"]
        if name == BDG1 and not spec.get("cache"):
            spec["cache"] = str((ROOT / "data/bdg1_building_data_genome/processed/canonical_device_days.npz").resolve())
        pool = load_canonical_device_day_pool(config)
        _SOURCE_POOLS[name] = pool.records
        print("  source pool %-34s records=%6d" % (name, len(pool.records)), flush=True)
    for index, record in enumerate(_SOURCE_POOLS[BDG1]):
        _BDG1_INDEX[(record.source_device_id, int(record.day_id))] = index


def _resolve(row: dict) -> object:
    dataset = row["dataset_id"]
    records = _SOURCE_POOLS[dataset]
    if dataset == BDG1:
        bare = row["source_device_id"].split("__", 1)[-1]
        key = (bare, int(row["day_id"]))
        return records[_BDG1_INDEX[key]]
    return records[int(row["source_row_index"])]


def build_pool_npz(kind: str, cid: str, seed_index: int) -> dict:
    name = combo_id_to_name(kind, cid, seed_index)
    fleet_dir = combo_fleet_dir(kind, cid, seed_index)
    with (fleet_dir / "devices.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    count = len(rows)
    load = np.empty((count, 288), dtype=np.float32)
    pv = np.empty((count, 288), dtype=np.float32)
    energy_input = np.empty((count, 288), dtype=np.float32)
    baseline = np.empty((count, 288), dtype=np.float32)
    timestamp = np.empty((count, 288), dtype=np.int64)
    device_id = np.empty(count, dtype=object)
    source_id = np.empty(count, dtype=object)
    zone_id = np.empty(count, dtype=object)
    day_id = np.empty(count, dtype=np.int32)
    bus_id = np.empty(count, dtype=np.int16)

    for index, row in enumerate(rows):
        record = _resolve(row)
        load[index] = record.load_kw
        pv[index] = record.pv_kw
        energy_input[index] = record.energy_input_kw
        baseline[index] = record.baseline_battery_kw
        timestamp[index] = record.timestamp
        device_id[index] = row["device_id"]
        source_id[index] = row["source_device_id"]
        zone_id[index] = row["zone_id"]
        day_id[index] = int(row["day_id"])
        bus_id[index] = int(row["bus_id"])

    soc = np.full((count, 288), BASE_SOC_KWH, dtype=np.float32)
    capacity = np.full(count, BASE_CAPACITY_KWH, dtype=np.float32)
    peak = np.full(count, BASE_PEAK_KW, dtype=np.float32)
    c_rate = np.full(count, BASE_C_RATE, dtype=np.float32)

    POOL_ROOT.mkdir(parents=True, exist_ok=True)
    npz_path = POOL_ROOT / ("%s.npz" % name)
    temporary = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        device_id=device_id, source_id=source_id, day_id=day_id, zone_id=zone_id, bus_id=bus_id,
        load=load, pv=pv, energy_input=energy_input, baseline=baseline, soc=soc,
        capacity=capacity, peak=peak, c_rate=c_rate, timestamp=timestamp,
        source_count=np.asarray(len(set(source_id.tolist()))), filled_values=np.asarray(0),
    )
    temporary.replace(npz_path)

    zone_sources: dict[str, set] = {}
    for zone, source in zip(zone_id.tolist(), source_id.tolist()):
        zone_sources.setdefault(zone, set()).add(source)
    zone_counts = {zone: len(sources) for zone, sources in sorted(zone_sources.items())}
    balanced = len(zone_counts) * min(zone_counts.values())
    return {
        "name": name, "kind": kind, "combination_id": cid, "seed_index": seed_index,
        "fleet_dir": str(fleet_dir), "npz": str(npz_path), "fleet_size": count,
        "unique_sources": len(set(source_id.tolist())),
        "zone_unique_sources": zone_counts, "balanced_unique_limit": balanced,
        "datasets": sorted({row["dataset_id"] for row in rows}),
    }


def build_config(info: dict) -> dict:
    name = info["name"]
    config_dir = RESULTS / ("%s_ieee33_real_load" % name) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    template_dir = RESULTS / ("%s_ieee33_real_load" % TEMPLATE_DATASET) / "config"
    for item in ("original_model.yaml", "source_device_map.yaml", "zones.yaml",
                 "network_ieee33.yaml", "device_constraints.yaml", "user_behavior.yaml",
                 "control.yaml", "experiment_protocol.yaml", "output_compatibility.yaml",
                 "simulation.yaml", "population.yaml", "default.yaml"):
        (config_dir / item).write_text((template_dir / item).read_text(encoding="utf-8"), encoding="utf-8")

    (config_dir / "paths.yaml").write_text(yaml.safe_dump({
        "data_root": str((ROOT / "data").resolve()),
        "results_root": str((RESULTS / ("%s_ieee33_real_load" % name)).resolve()),
    }, sort_keys=False), encoding="utf-8")

    selected = min(info["balanced_unique_limit"], MAX_SIMULATED_RESOURCES)
    zone_count = len(info["zone_unique_sources"])
    population = yaml.safe_load((config_dir / "population.yaml").read_text(encoding="utf-8"))
    population["N_simulated_resources"] = int(selected)
    population["resources_per_zone"] = int(selected // zone_count)
    population["days_per_source_device"] = 1
    population["canonical_adapter"] = {
        "dataset": name,
        "input": "",
        "input_dir": "",
        "solar_input": "",
        "cache": str(Path(info["npz"]).resolve()),
        "load_scale": 1.0,
        "max_sources": int(info["fleet_size"]),
        "max_days_per_source": 1,
        "max_source_days": 30,
        "input_mapping": "measured_only",
        "forbid_counterfactual_input": False,
        "energy_input_provenance": "inherited_per_source_dataset",
        "counterfactual_input_scale": 1.0,
        "fallback_capacity_kwh": BASE_CAPACITY_KWH,
        "fallback_peak_power_kw": BASE_PEAK_KW,
        "fallback_soc_fraction": 0.5,
    }
    (config_dir / "population.yaml").write_text(
        yaml.safe_dump(population, sort_keys=False, allow_unicode=True), encoding="utf-8")

    default = yaml.safe_load((config_dir / "default.yaml").read_text(encoding="utf-8"))
    default["control"] = "control.yaml"
    (config_dir / "default.yaml").write_text(
        yaml.safe_dump(default, sort_keys=False), encoding="utf-8")

    config = load_config(config_dir / "default.yaml")
    from src.extra.ieee33_device_day_simulation.population.device_day_loader import load_device_day_pool
    pool = load_device_day_pool(config)
    multiplier = _capacity_multiplier(config, pool, target_loading=0.45)
    _mode_config(config_dir / "default.yaml", "weak_correlation", multiplier)
    info["network_capacity_multiplier"] = float(multiplier)
    info["N_simulated_resources"] = int(selected)
    info["config"] = str(config_dir / "default_weak_correlation.yaml")
    return info


def worker(item: tuple[str, str, int]) -> dict:
    kind, cid, seed_index = item
    try:
        info = build_pool_npz(kind, cid, seed_index)
        return build_config(info)
    except Exception as exc:
        return {"name": combo_id_to_name(kind, cid, seed_index), "status": "error",
                "error": repr(exc), "traceback": traceback.format_exc(limit=6)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Turn mixture fleet manifests into canonical device-day pools for E1 and E2.")
    parser.add_argument("--seeds", default="0", help="comma separated list or an a-b range, e.g. 0 or 0-29")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--manifest", default="data/dataset_combination_pools/pools_manifest.json")
    parser.add_argument("--limit", type=int, default=0, help="build only the first N pools (smoke test)")
    args = parser.parse_args()

    if "-" in args.seeds:
        low, high = args.seeds.split("-", 1)
        seed_indices = list(range(int(low), int(high) + 1))
    else:
        seed_indices = [int(value) for value in args.seeds.split(",")]

    print("loading the 15 source pools", flush=True)
    load_source_pools()
    items = all_combos(seed_indices)
    if args.limit:
        items = items[:args.limit]
    print("building %d mixture pools (seed %s)" % (len(items), args.seeds), flush=True)

    done, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for finished, future in enumerate(as_completed(futures), 1):
            payload = future.result()
            if payload.get("status") == "error":
                failed.append(payload)
                print("  [%4d/%d] failed %s: %s" % (finished, len(items), payload["name"], payload["error"]), flush=True)
            else:
                done.append(payload)
                if finished % 25 == 0 or finished == len(items):
                    print("  [%4d/%d] %s fleet=%d N=%d" % (
                        finished, len(items), payload["name"],
                        payload["fleet_size"], payload["N_simulated_resources"]), flush=True)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "protocol": "dataset_combination_pools_v1",
        "sampling_mode": "unique",
        "device_hardware": "baseline_uniform_10kWh_1kW_soc0.5",
        "seeds": seed_indices,
        "built": len(done), "failed": len(failed),
        "entries": sorted(done, key=lambda r: r["name"]),
        "failures": failed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"built": len(done), "failed": len(failed), "manifest": str(manifest)},
                     ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
