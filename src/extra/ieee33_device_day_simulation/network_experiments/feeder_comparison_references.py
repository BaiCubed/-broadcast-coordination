#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(ROOT))

from src.extra.ieee33_device_day_simulation.network_experiments import ieee69_network_implementation as implementation
from src.extra.ieee33_device_day_simulation.network_experiments import train_mixed_model as mixed
from src.extra.ieee33_device_day_simulation.network_experiments import pairwise_curtailment as pairwise
import src.extra.ieee33_device_day_simulation.network_experiments.feeder_comparison_pairwise as supplement


OUTPUT = ROOT / "results/ieee69_network_implementation/feeder_comparison"
RAW = OUTPUT / "baseline_completion_raw"
PROTOCOL = "feeder_comparison_references_v1"
SEED_COUNT = 30
BASE_SEED = 31_000_000
ALGORITHMS = tuple(
    algorithm
    for algorithm in mixed.ALGORITHM_ORDER
    if not algorithm.startswith("eps_")
)
ORIGINAL_STRESS = ("M0", "M4", "M5", "M6")
PAIRWISE_STRESS = ("M0",)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _configure_ieee123() -> dict[str, Any]:
    case = supplement._configure_ieee123()
    implementation.TOPOLOGIES = ("ieee33", "ieee69", "ieee123")
    return case


def _network_case(topology: str) -> dict[str, Any]:
    return supplement._network_case(topology)


def _config_for_records(config: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    value["control"] = dict(value.get("control", {}))
    value["control"]["pure_sim_device_model"] = True
    value["control"]["network_feedback"] = False
    value["control"]["eps_control_mode"] = "fused"
    return value


def _common_row(
    *,
    topology: str,
    stress_mode: str,
    algorithm: str,
    seed_index: int,
    seed: int,
    dataset: str | None = None,
    pair: dict[str, Any] | None = None,
    audit: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "protocol": PROTOCOL,
        "topology": topology,
        "network_mode": topology,
        "stress_mode": stress_mode,
        "stress_label": implementation.STRESS_MODES[stress_mode]["label"],
        "algorithm": algorithm,
        "algorithm_label": mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
        "complexity": mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"],
        "training_regime": "target-network direct evaluation; non-EPS baseline",
        "seed_index": seed_index,
        "seed": seed,
        "fleet_size": int(audit.get("fleet_size", 5000)),
        **result,
    }
    if dataset is not None:
        row["dataset"] = dataset
        row["dataset_label"] = implementation.DATASET_LABELS.get(dataset, dataset)
        row["source"] = "original"
    if pair is not None:
        row.update(
            {
                "composition_id": pair["pair_id"],
                "pair_id": pair["pair_id"],
                "pair_index": int(pair["pair_index"]),
                "pair_label": pair["pair_label"],
                "dataset_a": pair["dataset_a"],
                "dataset_b": pair["dataset_b"],
                "source": "pairwise_mixed",
            }
        )
    return row


def _run_original_task(task: dict[str, Any]) -> dict[str, Any]:
    topology = "ieee123"
    _configure_ieee123()
    dataset = str(task["dataset"])
    output_path = RAW / "original" / f"{dataset}.csv"
    expected = int(task["seed_count"]) * len(ORIGINAL_STRESS) * len(ALGORITHMS)
    if output_path.is_file() and not task["force"]:
        rows = _read_rows(output_path)
        if len(rows) == expected and all(row.get("protocol") == PROTOCOL for row in rows):
            return {"kind": "original", "dataset": dataset, "rows": len(rows), "skipped": True}
    case = _network_case(topology)
    rows: list[dict[str, Any]] = []
    for seed_index in range(int(task["seed_count"])):
        base_seed = implementation._base_seed(dataset, seed_index) + 17_000_000
        records, config, audit = implementation._sample_dataset(dataset, base_seed)
        config = _config_for_records(config)
        audit = {**audit, "fleet_size": len(records)}
        placement_seed = base_seed + 31_000
        reference_records, _ = implementation._remap_records(records, case, placement_seed, "load_weighted")
        cache: dict[str, tuple[list[Any], dict[str, Any]]] = {}
        for stress_index, stress_mode in enumerate(ORIGINAL_STRESS):
            placement_mode = str(implementation.STRESS_MODES[stress_mode]["placement_mode"])
            cache.setdefault(placement_mode, implementation._remap_records(records, case, placement_seed, placement_mode))
            placed_records, placement = cache[placement_mode]
            scenario = implementation._build_scenario(
                placed_records,
                reference_records,
                case,
                topology,
                stress_mode,
                base_seed + stress_index * 10_000,
                placement,
            )
            availability = supplement.protocol.availability_probability(
                placed_records, config, mixed.AVAILABILITY_MODE
            )
            network = implementation._network(case, topology, stress_mode)
            for algorithm_index, algorithm in enumerate(ALGORITHMS):
                result, _ = implementation._run_seed(
                    algorithm,
                    placed_records,
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
                    _common_row(
                        topology=topology,
                        stress_mode=stress_mode,
                        algorithm=algorithm,
                        seed_index=seed_index,
                        seed=base_seed,
                        dataset=dataset,
                        audit=audit,
                        result={**result, "placement_mode": placement["placement_mode"]},
                    )
                )
    _write_rows(output_path, rows)
    return {"kind": "original", "dataset": dataset, "rows": len(rows), "skipped": False}


def _run_pair_task(task: dict[str, Any]) -> dict[str, Any]:
    topology = str(task["topology"])
    pair = task
    _configure_ieee123()
    output_path = RAW / "pairwise" / topology / f"{pair['pair_id']}.csv"
    expected = int(task["seed_count"]) * len(PAIRWISE_STRESS) * len(ALGORITHMS)
    if output_path.is_file() and not task["force"]:
        rows = _read_rows(output_path)
        if len(rows) == expected and all(row.get("protocol") == PROTOCOL for row in rows):
            return {"kind": "pairwise", "topology": topology, "pair_id": pair["pair_id"], "rows": len(rows), "skipped": True}
    case = _network_case(topology)
    rows: list[dict[str, Any]] = []
    for seed_index in range(int(task["seed_count"])):
        topology_offset = {
            "ieee33": 0,
            "ieee69": 10_000_000,
            "ieee123": 20_000_000,
        }[topology]
        base_seed = BASE_SEED + int(pair["pair_index"]) * 100_000 + topology_offset + seed_index
        records, config, audit = supplement._pair_records(
            str(pair["dataset_a"]), str(pair["dataset_b"]), "test", base_seed
        )
        config = _config_for_records(config)
        audit = {**audit, "fleet_size": len(records)}
        reference_records, _ = implementation._remap_records(records, case, base_seed + 31_000, "load_weighted")
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            stress_mode = "M0"
            placed_records, placement = implementation._remap_records(
                records, case, base_seed + 31_001, implementation.STRESS_MODES[stress_mode]["placement_mode"]
            )
            scenario = implementation._build_scenario(
                placed_records,
                reference_records,
                case,
                topology,
                stress_mode,
                base_seed + 2_000,
                placement,
            )
            availability = supplement.protocol.availability_probability(
                placed_records, config, mixed.AVAILABILITY_MODE
            )
            network = implementation._network(case, topology, stress_mode)
            result, _ = implementation._run_seed(
                algorithm,
                placed_records,
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
                _common_row(
                    topology=topology,
                    stress_mode=stress_mode,
                    algorithm=algorithm,
                    seed_index=seed_index,
                    seed=base_seed,
                    pair=pair,
                    audit=audit,
                    result={**result, "placement_mode": placement["placement_mode"]},
                )
            )
    _write_rows(output_path, rows)
    return {"kind": "pairwise", "topology": topology, "pair_id": pair["pair_id"], "rows": len(rows), "skipped": False}


def _tasks(seed_count: int, force: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for dataset in implementation.NETWORK_DATASETS:
        tasks.append({"kind": "original", "dataset": dataset, "seed_count": seed_count, "force": force})
    for pair in pairwise._pair_specs():
        for topology in ("ieee33", "ieee69", "ieee123"):
            tasks.append({**pair, "kind": "pairwise", "topology": topology, "seed_count": seed_count, "force": force})
    return tasks


def _merge(seed_count: int) -> dict[str, int]:
    original_rows = [row for path in sorted((RAW / "original").glob("*.csv")) for row in _read_rows(path)]
    pairwise_rows = [row for path in sorted((RAW / "pairwise").glob("*/*.csv")) for row in _read_rows(path)]
    expected_original = len(implementation.NETWORK_DATASETS) * seed_count * len(ORIGINAL_STRESS) * len(ALGORITHMS)
    expected_pairwise = len(pairwise._pair_specs()) * 3 * seed_count * len(PAIRWISE_STRESS) * len(ALGORITHMS)
    if len(original_rows) != expected_original:
        raise RuntimeError(f"incomplete IEEE-123 original baseline results: {len(original_rows)}, expected {expected_original}")
    if len(pairwise_rows) != expected_pairwise:
        raise RuntimeError(f"incomplete pairwise baseline results: {len(pairwise_rows)}, expected {expected_pairwise}")
    _write_rows(OUTPUT / "data/ieee123_original_baselines_m0_m4_m5_m6_by_seed.csv", original_rows)
    _write_rows(OUTPUT / "data/pairwise_baselines_target_by_seed.csv", pairwise_rows)
    return {"original_rows": len(original_rows), "pairwise_rows": len(pairwise_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the non-EPS algorithm results missing on the three target networks.")
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_count <= 30:
        raise ValueError("seed-count must be between 1 and 30")
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    tasks = _tasks(args.seed_count, args.force)
    RAW.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT / "baseline_completion_checkpoint.json"
    checkpoint = {"protocol": PROTOCOL, "status": "running", "task_count": len(tasks), "completed": 0, "seed_count": args.seed_count, "workers": args.workers}
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context = mp.get_context("spawn")
    runner = _run_original_task
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(_run_original_task if task["kind"] == "original" else _run_pair_task, task): task
            for task in tasks
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            checkpoint.update({"completed": completed, "last": result})
            checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"stage": "task_complete", "completed": completed, "total": len(tasks), **result}, ensure_ascii=False), flush=True)
    counts = _merge(args.seed_count)
    checkpoint.update({"status": "completed", **counts})
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", **counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
