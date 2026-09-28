"""增量计算 Figure 3 的小规模 N 采样点。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
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
from src.extra.ieee33_device_day_simulation.population.device_day_loader import DeviceDay, load_device_day_pool
from src.extra.nc_excel_experiments.coupling import availability_at_steps, diurnal_residual, rank_availability
from src.extra.nc_excel_experiments.run import (
    _balanced_unique_capacity,
    _dataset_config,
    _fit_frozen_prediction,
    _profile_and_schedule,
    _run_snapshots,
    _sample_master,
)


DEFAULT_PROTOCOL = ROOT / "src/extra/nc_excel_experiments/configs/e1_full.yaml"
DEFAULT_OUTPUT = ROOT / "results/e1_full/E1_scale_boundary_new"
SMALL_N_VALUES = (1, 5, 15, 30)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bootstrap_balanced(master: list[DeviceDay], count: int) -> tuple[list[DeviceDay], dict[str, Any]]:
    if count <= len(master):
        selected = _balanced_subset(master, count)
        return selected, {
            "sampling": "source_unique",
            "logical_devices": count,
            "unique_profile_sources": len({row.source_device_id for row in selected}),
            "profile_bootstrap": False,
        }

    zones = sorted({row.zone_id for row in master})
    by_zone = {zone: [row for row in master if row.zone_id == zone] for zone in zones}
    per_zone, remainder = divmod(count, len(zones))
    selected: list[DeviceDay] = []
    for zone_index, zone in enumerate(zones):
        requested = per_zone + (1 if zone_index < remainder else 0)
        candidates = by_zone[zone]
        for index in range(requested):
            source = candidates[index % len(candidates)]
            selected.append(replace(
                source,
                device_id=f"{source.device_id}__profile_copy_{zone_index}_{index}",
            ))
    return selected, {
        "sampling": "balanced_profile_bootstrap",
        "logical_devices": count,
        "unique_profile_sources": len({row.source_device_id for row in selected}),
        "profile_bootstrap": True,
        "reason": "数据集独立源数量不足该 N，仅复用真实曲线，不宣称新增独立测量源。",
    }


def _reference_profile_steps(response_dir: Path) -> list[int]:
    candidates = sorted(response_dir.glob("responses_N*_data_coupled.npz"))
    if not candidates:
        raise FileNotFoundError(f"缺少已有 E1 响应档案: {response_dir}")
    with np.load(candidates[0]) as payload:
        return np.asarray(payload["profile_steps"], dtype=int).tolist()


def _run_dataset(
    dataset: str,
    settings: dict[str, Any],
    common: dict[str, Any],
    random_seed: int,
    output: str,
    force: bool,
) -> dict[str, Any]:
    output_root = Path(output)
    response_dir = output_root / "raw/responses" / dataset
    requested = [
        value for value in SMALL_N_VALUES
        if force or not (response_dir / f"responses_N{value}_data_coupled.npz").is_file()
    ]
    config_path, _ = _dataset_config(dataset)
    config = load_config(config_path)
    config["control"]["network_feedback"] = False
    pool = load_device_day_pool(config)
    maximum = min(
        int(config["population"]["N_simulated_resources"]),
        _balanced_unique_capacity(pool),
    )
    if maximum < 1:
        raise RuntimeError(f"{dataset} 没有可用独立源")
    if not requested:
        audits = []
        for fleet_size in SMALL_N_VALUES:
            bootstrap = fleet_size > maximum
            audits.append({
                "N": fleet_size,
                "archive": str(
                    (response_dir / f"responses_N{fleet_size}_data_coupled.npz").relative_to(ROOT)
                ),
                "sampling": "balanced_profile_bootstrap" if bootstrap else "source_unique",
                "logical_devices": fleet_size,
                "unique_profile_sources": min(fleet_size, maximum),
                "profile_bootstrap": bootstrap,
            })
        return {
            "dataset": dataset,
            "status": "skipped_existing",
            "maximum_unique_fleet": maximum,
            "results": audits,
        }

    seed = int(random_seed) + 1000
    master = _sample_master(config, maximum, seed)
    profile_steps, schedule = _profile_and_schedule(
        master,
        config,
        int(settings.get("conditions", common["conditions"])),
        seed + 100,
    )
    reference_steps = _reference_profile_steps(response_dir)
    if profile_steps != reference_steps:
        raise RuntimeError(f"{dataset} 的条件采样与已有 E1 档案不一致")

    train_count = int(settings.get("train_replications", common["train_replications"]))
    test_count = int(settings.get("test_replications", common["test_replications"]))
    total_count = train_count + test_count
    audits = []
    for fleet_size in requested:
        records, sampling = _bootstrap_balanced(master, fleet_size)
        availability = availability_at_steps(
            rank_availability(diurnal_residual(records, int(config["simulation"]["steps_per_day"]))),
            profile_steps,
        )
        aggregate, traces, _, _ = _run_snapshots(
            records,
            config,
            profile_steps,
            schedule,
            replications=total_count,
            seed=seed + fleet_size * 10000,
            zone_correlation=0.0,
            keep_resources=False,
            availability_factory=lambda replication, table=availability: table,
        )
        _, frozen_prediction = _fit_frozen_prediction(
            trace_features(traces[0]), aggregate[:train_count]
        )
        archive = response_dir / f"responses_N{fleet_size}_data_coupled.npz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            archive,
            aggregate_response_kw=aggregate,
            frozen_prediction_kw=frozen_prediction,
            profile_steps=np.asarray(profile_steps, dtype=int),
        )
        audit = {
            "N": fleet_size,
            "archive": str(archive.relative_to(ROOT)),
            "train_replications": train_count,
            "eval_replications": test_count,
            "conditions": len(profile_steps),
            **sampling,
        }
        audits.append(audit)
        print(json.dumps({"dataset": dataset, **audit}, ensure_ascii=False), flush=True)

    return {
        "dataset": dataset,
        "status": "completed",
        "maximum_unique_fleet": maximum,
        "results": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    settings = protocol["e1"]
    datasets = [str(value) for value in settings["datasets"]]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(datasets))) as executor:
        futures = {
            executor.submit(
                _run_dataset,
                dataset,
                settings,
                protocol["common"],
                int(protocol.get("random_seed", 20260720)),
                str(args.output.resolve()),
                args.force,
            ): dataset
            for dataset in datasets
        }
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"dataset": dataset, "status": "failed", "error": repr(exc)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    manifest = {
        "protocol": "fig3_small_n_increment_v1",
        "N_values": list(SMALL_N_VALUES),
        "source_protocol": str(args.protocol),
        "completed": sum(row["status"] in {"completed", "skipped_existing"} for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "datasets": sorted(rows, key=lambda row: row["dataset"]),
    }
    _write_json(args.output / "fig3_small_n_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    if manifest["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
