from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import multiprocessing as mp

from . import run_e22_ieee69_complexity as e22
from . import run_e21_mixed_scenarios as mixed
from ..population.data2_transaction_loader import clear_data2_transaction_pool_cache


TARGET_ALGORITHM = "centralized_optimal"
PROTOCOL = "E22_recompute_centralized_greedy_ub_on_device_headroom_v1"


def _recompute_dataset(dataset: str, output: str, seed_count: int) -> str:
    output_path = Path(output)
    raw_path = output_path / "raw" / f"{dataset}.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    original_rows = list(payload.get("seed_results", []))
    rows = [row for row in original_rows if row["algorithm"] != TARGET_ALGORITHM]
    cases = e22._network_cases()
    for seed_index in range(seed_count):
        base_seed = e22._base_seed(dataset, seed_index)
        base_records, config, audit = e22._sample_dataset(dataset, base_seed)
        availability = e22.protocol.availability_probability(
            base_records, config, mixed.AVAILABILITY_MODE
        )
        for topology_index, topology in enumerate(e22.TOPOLOGIES):
            placement_seed = base_seed + 10_000 * (topology_index + 1)
            reference_records, _ = e22._remap_records(
                base_records, cases[topology], placement_seed, "load_weighted"
            )
            placement_cache: dict[str, tuple[list[object], dict[str, object]]] = {}
            for stress_mode in e22.STRESS_MODES:
                placement_mode = str(e22.STRESS_MODES[stress_mode]["placement_mode"])
                if placement_mode not in placement_cache:
                    placement_cache[placement_mode] = e22._remap_records(
                        base_records,
                        cases[topology],
                        placement_seed,
                        placement_mode,
                    )
                records, placement = placement_cache[placement_mode]
                scenario = e22._build_scenario(
                    records,
                    reference_records,
                    cases[topology],
                    topology,
                    stress_mode,
                    base_seed,
                    placement,
                )
                network = e22._network(cases[topology], topology, stress_mode)
                algorithm_index = e22.ALGORITHM_ORDER.index(TARGET_ALGORITHM)
                result, _ = e22._run_seed(
                    TARGET_ALGORITHM,
                    records,
                    config,
                    scenario,
                    availability,
                    network,
                    base_seed + 100_000 * (algorithm_index + 1),
                    base_seed + 1_910_000,
                    None,
                    False,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": e22.DATASET_LABELS[dataset],
                        "topology": topology,
                        "stress_mode": stress_mode,
                        "stress_label": e22.STRESS_MODES[stress_mode]["label"],
                        "algorithm": TARGET_ALGORITHM,
                        "algorithm_label": mixed.ALGORITHM_DEFINITIONS[TARGET_ALGORITHM]["label"],
                        "complexity": mixed.ALGORITHM_DEFINITIONS[TARGET_ALGORITHM]["complexity"],
                        "seed_index": seed_index,
                        "seed": base_seed,
                        "fleet_size": len(records),
                        "placement_mode": placement["placement_mode"],
                        "target_branch": placement["target_branch"],
                        "target_bus": placement["target_bus"],
                        "target_downstream_load_share": placement[
                            "target_downstream_load_share"
                        ],
                        "configured_concentration_fraction": placement[
                            "configured_concentration_fraction"
                        ],
                        "actual_target_feeder_fraction": placement[
                            "actual_target_feeder_fraction"
                        ],
                        "actual_target_bus_fraction": placement["actual_target_bus_fraction"],
                        "maximum_bus_fraction": placement["maximum_bus_fraction"],
                        "runtime_seconds": 0.0,
                        **result,
                    }
                )
        print(
            f"[{dataset}] greedy UB seeds {seed_index + 1}/{seed_count}",
            flush=True,
        )
    rows.sort(
        key=lambda row: (
            int(row["seed_index"]),
            e22.TOPOLOGIES.index(row["topology"]),
            list(e22.STRESS_MODES).index(row["stress_mode"]),
            e22.ALGORITHM_ORDER.index(row["algorithm"]),
        )
    )
    payload["seed_results"] = rows
    payload["greedy_ub_protocol"] = PROTOCOL
    payload["greedy_ub_definition"] = (
        "O(N) per-step theoretical device headroom upper bound; network clipping is audit-only"
    )
    payload["updated_at"] = e22._utc_now()
    e22._write_json(raw_path, payload)
    mixed._PARTITION_CACHE.clear()
    e22._ADDITIONAL_PARTITION_CACHE.clear()
    clear_data2_transaction_pool_cache()
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for recompute e22 greedy ub.")
    parser.add_argument("--output", type=Path, default=Path("results/E22"))
    parser.add_argument("--seed-count", type=int, default=e22.SEED_COUNT)
    parser.add_argument("--bootstrap-draws", type=int, default=e22.BOOTSTRAP_DRAWS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.seed_count <= e22.SEED_COUNT:
        raise ValueError("seed-count must be between 1 and 30")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = e22._acquire_lock(output)
    if lock is None:
        raise RuntimeError(f"{output} already holds a running task")
    datasets = tuple(e22.E22_DATASETS)
    context = mp.get_context("spawn")
    completed: list[str] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(datasets)), mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                _recompute_dataset, dataset, str(output), args.seed_count
            ): dataset
            for dataset in datasets
        }
        for future in as_completed(futures):
            dataset = future.result()
            completed.append(dataset)
            e22._write_json(
                output / "greedy_ub_checkpoint.json",
                {
                    "protocol": PROTOCOL,
                    "status": "running",
                    "completed_datasets": len(completed),
                    "dataset_count": len(datasets),
                    "workers": args.workers,
                    "seed_count": args.seed_count,
                    "last_completed_dataset": dataset,
                    "updated_at": e22._utc_now(),
                },
            )
    payloads = [
        json.loads(
            (output / "raw" / f"{dataset}.json").read_text(encoding="utf-8")
        )
        for dataset in datasets
    ]
    manifest = e22._finalize(
        output, payloads, args.bootstrap_draws, args.seed_count
    )
    manifest["greedy_ub_protocol"] = PROTOCOL
    manifest["greedy_ub_definition"] = (
        "O(N) theoretical device headroom upper bound; network clipping audit-only"
    )
    e22._write_json(output / "manifest.json", manifest)
    e22._write_json(output / "checkpoint.json", manifest)
    e22._write_json(
        output / "greedy_ub_checkpoint.json",
        {
            "protocol": PROTOCOL,
            "status": "completed",
            "completed_datasets": len(datasets),
            "dataset_count": len(datasets),
            "workers": args.workers,
            "seed_count": args.seed_count,
            "updated_at": e22._utc_now(),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
