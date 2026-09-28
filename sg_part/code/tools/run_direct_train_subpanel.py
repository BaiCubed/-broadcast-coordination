#!/usr/bin/env python3
"""为 E22 子图生成 EPS 全直接训练版本。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from src.estimation import EPSEstimator
from src.extra.ieee33_device_day_simulation.figures import (
    fig4d_extra_baselines as legacy,
    fig4d_final_protocol as protocol,
    run_e20_transfer as transfer,
    run_e21_mixed_scenarios as mixed,
    run_e21_pairwise_curtailment as pairwise,
    run_e22_ieee69_complexity as e22,
    run_e22_trained_eps_comparison as direct69,
)
from src.signal import SignalOptimizer


OUTPUT = ROOT / "results/E21/pairwise_curtailment/direct_training"
MODEL_ROOT = OUTPUT / "models"
DATA_ROOT = OUTPUT / "data"
ORIGINAL_33_ROOT = DATA_ROOT / "original_ieee33_direct"
PUBLISH_ROOT = ROOT / "outputs/figs/direct_train"
SEED_COUNT = 30
TRAIN_SAMPLES = 128
VALIDATION_SAMPLES = 32
BASE_SEED = 27_000_000
NETWORK_MODES = ("aggregate", "ieee33")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def pair_specs() -> list[dict[str, Any]]:
    return pairwise._pair_specs()


def model_path(network_mode: str, pair_id: str) -> Path:
    return MODEL_ROOT / network_mode / f"{pair_id}.pt"


def metadata_path(network_mode: str, pair_id: str) -> Path:
    return MODEL_ROOT / network_mode / f"{pair_id}.json"


def original_33_model_path(dataset: str) -> Path:
    existing = ROOT / f"results/{dataset}_ieee33_real_load/coverage_fix/network_constrained_new/data/fig4d_eps_estimator_final_fixed5000_ieee33_data_driven.pt"
    if existing.is_file():
        return existing
    return MODEL_ROOT / "ieee33_original" / f"{dataset}.pt"


def train_missing_original_33(dataset: str, force: bool) -> dict[str, Any]:
    path = original_33_model_path(dataset)
    metadata_file = MODEL_ROOT / "ieee33_original" / f"{dataset}.json"
    if path.is_file() and not metadata_file.is_file() and path.parent != (MODEL_ROOT / "ieee33_original"):
        return {
            "dataset": dataset,
            "model_path": str(path),
            "model_sha256": sha256(path),
            "protocol": "existing_dataset_specific_ieee33_direct_training",
            "training_topology": "ieee33",
            "model_reused": True,
        }
    if path.is_file() and metadata_file.is_file() and not force:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    direct69.TRAINING_TOPOLOGY = "ieee33"
    config, partitions = direct69._training_config(dataset)
    cases = e22._network_cases()
    training_protocol = protocol._training_protocol(e22.FLEET_SIZE)
    started = time.perf_counter()
    estimator, training_metadata = direct69._train_ieee69_controller(
        partitions,
        config,
        training_protocol,
        e22.BASE_SEED + e22._stable_seed(dataset) % 100_000 + 810_000,
        cases,
        tuple(e22.STRESS_MODES),
    )
    legacy._save_eps_model(estimator, path)
    metadata = {
        "protocol": "E22_subpanel_ieee33_original_direct_training_v1",
        "dataset": dataset,
        "training_topology": "ieee33",
        "training_stress_modes": list(e22.STRESS_MODES),
        "training_regime": "dataset-specific IEEE-33 M0-M6 direct training",
        "model_path": str(path),
        "model_sha256": sha256(path),
        "model_reused": False,
        "training_seconds": time.perf_counter() - started,
        **training_metadata,
    }
    write_json(metadata_file, metadata)
    return metadata


def original_33_task(dataset: str, force: bool) -> dict[str, Any]:
    path = original_33_model_path(dataset)
    return {
        "dataset": dataset,
        "model_path": str(path),
        "output_path": str(ORIGINAL_33_ROOT / f"{dataset}.csv"),
        "force": force,
    }


def evaluate_original_33(task: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(task["output_path"])
    if output_path.is_file() and not task["force"]:
        rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
        if len(rows) == SEED_COUNT * len(e22.STRESS_MODES):
            return {"dataset": task["dataset"], "rows": len(rows)}
    estimator, _, _ = protocol.load_frozen_eps_controller(Path(task["model_path"]))
    optimizer = SignalOptimizer(transfer.ScaledEstimator(estimator, mixed.RESPONSE_SCALE_KW, 0.0))
    case = e22._network_cases()["ieee33"]
    rows: list[dict[str, Any]] = []
    for seed_index in range(SEED_COUNT):
        dataset = task["dataset"]
        base_seed = e22._base_seed(dataset, seed_index)
        base_records, config, _ = e22._sample_dataset(dataset, base_seed)
        availability = protocol.availability_probability(base_records, config, mixed.AVAILABILITY_MODE)
        placement_seed = base_seed + 10_000 * (e22.TOPOLOGIES.index("ieee33") + 1)
        reference_records, _ = e22._remap_records(base_records, case, placement_seed, "load_weighted")
        placement_cache: dict[str, tuple[list[Any], dict[str, Any]]] = {}
        for stress_mode in e22.STRESS_MODES:
            placement_mode = str(e22.STRESS_MODES[stress_mode]["placement_mode"])
            if placement_mode not in placement_cache:
                placement_cache[placement_mode] = e22._remap_records(
                    base_records, case, placement_seed, placement_mode
                )
            records, placement = placement_cache[placement_mode]
            scenario = e22._build_scenario(
                records, reference_records, case, "ieee33", stress_mode, base_seed, placement
            )
            algorithm_config = copy.deepcopy(config)
            algorithm_config["control"] = dict(algorithm_config["control"])
            algorithm_config["control"]["eps_control_mode"] = "fused"
            result, _ = e22._run_seed(
                "eps_ieee69_fused", records, algorithm_config, scenario, availability,
                e22._network(case, "ieee33", stress_mode), base_seed + 1_800_000,
                base_seed + 1_910_000, optimizer, False,
            )
            rows.append({
                "dataset": dataset,
                "dataset_label": e22.DATASET_LABELS[dataset],
                "topology": "ieee33",
                "stress_mode": stress_mode,
                "stress_label": e22.STRESS_MODES[stress_mode]["label"],
                "algorithm": "eps_ieee69_fused",
                "algorithm_label": "EPS",
                "complexity": "O(1)",
                "training_regime": "dataset-specific IEEE-33 M0-M6 direct training",
                "seed_index": seed_index,
                "seed": base_seed,
                "fleet_size": len(records),
                "placement_mode": placement["placement_mode"],
                **result,
            })
    write_rows(output_path, rows)
    return {"dataset": task["dataset"], "rows": len(rows)}


def merge_original_33() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(ORIGINAL_33_ROOT.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != len(original_datasets()) * SEED_COUNT * len(e22.STRESS_MODES):
        raise ValueError(f"IEEE-33直接训练结果数量异常：{len(rows)}")
    write_rows(DATA_ROOT / "original_ieee33_direct_by_seed.csv", rows)


def original_datasets() -> list[str]:
    return list(e22.E22_DATASETS)


def train_model(task: dict[str, Any]) -> dict[str, Any]:
    pair_id = task["pair_id"]
    network_mode = task["network_mode"]
    path = Path(task["model_path"])
    metadata_file = Path(task["metadata_path"])
    if path.is_file() and metadata_file.is_file() and not task["force"]:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("protocol") == task["protocol"] and metadata.get("model_sha256") == sha256(path):
            return metadata

    weights = {task["dataset_a"]: 0.5, task["dataset_b"]: 0.5}
    train_records, train_config, train_audit = mixed._sample_mixed_fleet(
        "S4-A", network_mode, "train", int(task["seed"]), weights=weights
    )
    train_signals, train_responses, train_response_audit = mixed._response_samples(
        train_records, train_config, TRAIN_SAMPLES, int(task["seed"]) + 1
    )
    validation_records, validation_config, validation_audit = mixed._sample_mixed_fleet(
        "S4-A", network_mode, "validation", int(task["seed"]) + 2000, weights=weights
    )
    validation_signals, validation_responses, validation_response_audit = mixed._response_samples(
        validation_records, validation_config, VALIDATION_SAMPLES, int(task["seed"]) + 2001
    )
    try:
        import torch

        torch.manual_seed(int(task["seed"]))
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = transfer._make_estimator()
    estimator.fit(train_signals, train_responses)
    predicted = np.asarray([estimator.estimate(signal).response_kw for signal in validation_signals])
    actual = np.asarray(validation_responses)
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    legacy._save_eps_model(estimator, path)
    metadata = {
        "protocol": task["protocol"],
        "pair_id": pair_id,
        "dataset_a": task["dataset_a"],
        "dataset_b": task["dataset_b"],
        "network_mode": network_mode,
        "training_seed": int(task["seed"]),
        "training_samples": TRAIN_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "training_fleet": train_audit,
        "training_response": train_response_audit,
        "validation_fleet": validation_audit,
        "validation_response": validation_response_audit,
        "validation_r2": float(1.0 - np.sum((actual - predicted) ** 2) / max(denominator, 1e-12)),
        "model_path": str(path),
        "model_sha256": sha256(path),
        "training_regime": "pair-specific direct training on the same two-source mixed fleet",
        "communication_complexity": "O(1)",
    }
    write_json(metadata_file, metadata)
    return metadata


def evaluate_model(task: dict[str, Any]) -> dict[str, Any]:
    pair_id = task["pair_id"]
    network_mode = task["network_mode"]
    path = Path(task["model_path"])
    raw_path = DATA_ROOT / "raw" / network_mode / f"{pair_id}.json"
    if raw_path.is_file() and not task["force"]:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if payload.get("protocol") == task["protocol"] and len(payload.get("seed_results", [])) == SEED_COUNT:
            return payload
    estimator, _, _ = protocol.load_frozen_eps_controller(path)
    optimizer = SignalOptimizer(transfer.ScaledEstimator(estimator, mixed.RESPONSE_SCALE_KW, 0.0))
    weights = {task["dataset_a"]: 0.5, task["dataset_b"]: 0.5}
    rows: list[dict[str, Any]] = []
    for seed_index in range(SEED_COUNT):
        base_seed = BASE_SEED + int(task["pair_index"]) * 100_000 + seed_index
        records, config, audit = mixed._sample_mixed_fleet(
            "S4-A", network_mode, "test", base_seed, weights=weights
        )
        config = copy.deepcopy(config)
        config["control"] = dict(config["control"])
        config["control"]["network_feedback"] = False
        scenario = protocol.build_pure_sim_scenario(records, config)
        availability = protocol.availability_probability(records, config, mixed.AVAILABILITY_MODE)
        result = legacy._run_seed(
            "eps_broadcast", records, config, scenario, availability,
            base_seed + 800_000, base_seed + 1_910_000, eps_optimizer=optimizer
        )
        rows.append({
            "pair_id": pair_id,
            "pair_index": int(task["pair_index"]),
            "pair_label": task["pair_label"],
            "dataset_a": task["dataset_a"],
            "dataset_b": task["dataset_b"],
            "network_mode": network_mode,
            "seed_index": seed_index,
            "seed": base_seed,
            "fleet_size": len(records),
            "devices_a": audit["counts"][task["dataset_a"]],
            "devices_b": audit["counts"][task["dataset_b"]],
            "algorithm": "eps_direct_pairwise",
            "algorithm_label": "EPS direct pairwise",
            "complexity": "O(1)",
            "model_path": str(path),
            "model_sha256": sha256(path),
            "curtailment_reduction_pct": float(result["mean_reduction_pct"]),
            "mean_reduction_pct": float(result["mean_reduction_pct"]),
            **{key: value for key, value in result.items() if key != "mean_reduction_pct"},
        })
    payload = {
        "protocol": task["protocol"],
        "pair_id": pair_id,
        "pair_index": int(task["pair_index"]),
        "pair_label": task["pair_label"],
        "dataset_a": task["dataset_a"],
        "dataset_b": task["dataset_b"],
        "network_mode": network_mode,
        "seed_count": SEED_COUNT,
        "model_path": str(path),
        "model_sha256": sha256(path),
        "seed_results": rows,
    }
    write_json(raw_path, payload)
    return payload


def task_list(force: bool) -> list[dict[str, Any]]:
    tasks = []
    for pair in pair_specs():
        for network_mode in NETWORK_MODES:
            tasks.append({
                **pair,
                "network_mode": network_mode,
                "seed": BASE_SEED + int(pair["pair_index"]) * 100_000 + (500_000 if network_mode == "ieee33" else 0),
                "protocol": "E22_subpanel_all_eps_direct_training_v1",
                "model_path": str(model_path(network_mode, pair["pair_id"])),
                "metadata_path": str(metadata_path(network_mode, pair["pair_id"])),
                "force": force,
            })
    return tasks


def merge_outputs() -> None:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for task in task_list(False):
        raw_path = DATA_ROOT / "raw" / task["network_mode"] / f"{task['pair_id']}.json"
        metadata_file = Path(task["metadata_path"])
        if not raw_path.is_file() or not metadata_file.is_file():
            raise FileNotFoundError(f"组合任务尚未完成：{task['network_mode']}/{task['pair_id']}")
        rows.extend(json.loads(raw_path.read_text(encoding="utf-8"))["seed_results"])
        metadata.append(json.loads(metadata_file.read_text(encoding="utf-8")))
    write_rows(DATA_ROOT / "pairwise_eps_direct_by_seed.csv", rows)
    write_rows(DATA_ROOT / "pairwise_eps_direct_summary.csv", [
        {
            "pair_id": row["pair_id"],
            "pair_label": row["pair_label"],
            "dataset_a": row["dataset_a"],
            "dataset_b": row["dataset_b"],
            "network_mode": row["network_mode"],
            "algorithm": row["algorithm"],
            "mean_reduction_pct": float(np.mean([float(item["mean_reduction_pct"]) for item in rows if item["pair_id"] == row["pair_id"] and item["network_mode"] == row["network_mode"]])),
            "seed_count": SEED_COUNT,
        }
        for row in rows
        if row["seed_index"] == 0
    ])
    write_json(DATA_ROOT / "training_metadata.json", metadata)
    write_json(OUTPUT / "manifest.json", {
        "protocol": "E22_subpanel_all_eps_direct_training_v1",
        "status": "completed",
        "pair_count": len(pair_specs()),
        "network_modes": list(NETWORK_MODES),
        "seed_count": SEED_COUNT,
        "model_count": len(metadata),
        "data": ["data/pairwise_eps_direct_by_seed.csv", "data/pairwise_eps_direct_summary.csv"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-original-datasets", type=int, default=None)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers 必须在 1 到 4 之间")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    selected_original_datasets = original_datasets()
    if args.max_original_datasets is not None:
        selected_original_datasets = selected_original_datasets[: max(1, int(args.max_original_datasets))]
    original_tasks = [original_33_task(dataset, args.force) for dataset in selected_original_datasets]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(original_tasks)), mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(train_missing_original_33, dataset, args.force): dataset for dataset in selected_original_datasets}
        for future in as_completed(futures):
            dataset = futures[future]
            future.result()
            print(json.dumps({"stage": "original_ieee33_model", "dataset": dataset}, ensure_ascii=False), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(evaluate_original_33, task): task for task in original_tasks}
        for future in as_completed(futures):
            task = futures[future]
            future.result()
            print(json.dumps({"stage": "original_ieee33_evaluation", "dataset": task["dataset"]}, ensure_ascii=False), flush=True)
    merge_original_33()
    tasks = task_list(args.force)
    if args.max_tasks is not None:
        tasks = tasks[: max(1, int(args.max_tasks))]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {executor.submit(train_model, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            future.result()
            print(json.dumps({"stage": "model", "pair": task["pair_id"], "network": task["network_mode"]}, ensure_ascii=False), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {executor.submit(evaluate_model, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            future.result()
            print(json.dumps({"stage": "evaluation", "pair": task["pair_id"], "network": task["network_mode"]}, ensure_ascii=False), flush=True)
    if len(tasks) == len(pair_specs()) * len(NETWORK_MODES):
        merge_outputs()


if __name__ == "__main__":
    main()
