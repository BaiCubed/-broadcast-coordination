"""Incrementally add small-fleet E1 response archives used by Figure 3."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extra.ieee33_device_day_simulation.config_loader import load_config
from src.extra.ieee33_device_day_simulation.experiments.protocol import trace_features
from src.extra.ieee33_device_day_simulation.experiments.run_experiment import _balanced_subset
from src.extra.ieee33_device_day_simulation.population.device_day_loader import (
    DeviceDay,
    load_device_day_pool,
)
from src.extra.nc_excel_experiments.coupling import (
    availability_at_steps,
    diurnal_residual,
    rank_availability,
)
from src.extra.nc_excel_experiments.run import (
    _balanced_unique_capacity,
    _dataset_config,
    _fit_frozen_prediction,
    _profile_and_schedule,
    _run_snapshots,
    _sample_master,
)


DEFAULT_PROTOCOL = ROOT / "src/extra/nc_excel_experiments/configs/e1_full.yaml"
DEFAULT_SOURCE = ROOT / "results/e1_full/E1_scale_boundary_new"
ARCHIVE_PATTERN = re.compile(r"responses_N(\d+)_data_coupled\.npz$")


def _logical_fleet(records: list[DeviceDay], count: int) -> list[DeviceDay]:
    """Build a zone-balanced logical fleet while retaining source provenance."""
    if count <= len(records):
        return _balanced_subset(records, count)
    zones = sorted({record.zone_id for record in records})
    groups = {zone: [record for record in records if record.zone_id == zone] for zone in zones}
    per_zone, remainder = divmod(count, len(zones))
    selected: list[DeviceDay] = []
    for zone_index, zone in enumerate(zones):
        requested = per_zone + (1 if zone_index < remainder else 0)
        group = groups[zone]
        for logical_index in range(requested):
            source = group[logical_index % len(group)]
            selected.append(
                replace(source, device_id=f"{source.device_id}_logical_{logical_index:04d}")
            )
    return selected


def _existing_grid(response_dir: Path) -> list[int]:
    values = []
    for path in response_dir.glob("responses_N*_data_coupled.npz"):
        match = ARCHIVE_PATTERN.match(path.name)
        if match:
            values.append(int(match.group(1)))
    return sorted(values)


def _run_dataset(
    dataset: str,
    source: str,
    requested_n: tuple[int, ...],
    train_replications: int,
    test_replications: int,
    conditions: int,
    random_seed: int,
    force: bool,
) -> dict[str, Any]:
    source_root = Path(source)
    response_dir = source_root / "raw/responses" / dataset
    response_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_grid(response_dir)
    missing = [
        fleet_size
        for fleet_size in requested_n
        if force or not (response_dir / f"responses_N{fleet_size}_data_coupled.npz").exists()
    ]
    if not missing:
        return {"dataset": dataset, "status": "skipped_existing", "N_values": list(requested_n)}

    config_path, _ = _dataset_config(dataset)
    config = load_config(config_path)
    config["control"]["network_feedback"] = False
    pool = load_device_day_pool(config)
    configured = int(config["population"]["N_simulated_resources"])
    maximum_unique = min(configured, _balanced_unique_capacity(pool))
    if maximum_unique < 1:
        raise ValueError(f"{dataset} has no balanced source fleet")

    reference_count = min(max(existing or [maximum_unique]), maximum_unique)
    seed = random_seed + 1000
    master = _sample_master(config, reference_count, seed)
    profile_steps, schedule = _profile_and_schedule(master, config, conditions, seed + 100)
    steps_per_day = int(config["simulation"]["steps_per_day"])
    completed = []
    for fleet_size in missing:
        records = _logical_fleet(master, fleet_size)
        availability = availability_at_steps(
            rank_availability(diurnal_residual(records, steps_per_day)), profile_steps
        )
        n_index = sorted(set(existing + list(requested_n))).index(fleet_size)
        aggregate, traces, _, _ = _run_snapshots(
            records,
            config,
            profile_steps,
            schedule,
            replications=train_replications + test_replications,
            seed=seed + n_index * 10000,
            zone_correlation=0.0,
            availability_factory=lambda replication, table=availability: table,
        )
        _, frozen_prediction = _fit_frozen_prediction(
            trace_features(traces[0]), aggregate[:train_replications]
        )
        archive = response_dir / f"responses_N{fleet_size}_data_coupled.npz"
        np.savez_compressed(
            archive,
            aggregate_response_kw=aggregate,
            frozen_prediction_kw=frozen_prediction,
            profile_steps=np.asarray(profile_steps),
        )
        try:
            archive_label = str(archive.relative_to(ROOT))
        except ValueError:
            archive_label = str(archive)
        completed.append({
            "N": fleet_size,
            "archive": archive_label,
            "sampling": "source_unique" if fleet_size <= maximum_unique else "logical_profile_reuse",
            "maximum_unique_fleet": maximum_unique,
        })
    return {"dataset": dataset, "status": "completed", "points": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--n-values", nargs="+", type=int, default=[1, 5, 15, 30])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20260720)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    common = protocol["common"]
    settings = protocol["e1"]
    rows = []
    tasks = [
        (
            str(dataset),
            str(args.source.resolve()),
            tuple(sorted(set(args.n_values))),
            int(settings.get("train_replications", common["train_replications"])),
            int(settings.get("test_replications", common["test_replications"])),
            int(settings.get("conditions", common["conditions"])),
            args.random_seed,
            args.force,
        )
        for dataset in settings["datasets"]
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = {executor.submit(_run_dataset, *task): task[0] for task in tasks}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"dataset": dataset, "status": "failed", "error": repr(exc)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    rows.sort(key=lambda row: row["dataset"])
    manifest = {
        "protocol": "fig3_small_fleet_incremental_e1_data_coupled",
        "requested_N_values": sorted(set(args.n_values)),
        "train_replications": int(settings.get("train_replications", common["train_replications"])),
        "test_replications": int(settings.get("test_replications", common["test_replications"])),
        "conditions": int(settings.get("conditions", common["conditions"])),
        "random_seed": args.random_seed,
        "opsd_profile_reuse_note": (
            "OPSD has six balanced independent sources; N=15 and N=30 reuse measured source "
            "profiles as distinct logical devices and preserve source_device_id provenance."
        ),
        "results": rows,
    }
    manifest_path = args.source / "fig3_small_fleet_extension_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)
    print(json.dumps({"manifest": str(manifest_path), "status": "completed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
