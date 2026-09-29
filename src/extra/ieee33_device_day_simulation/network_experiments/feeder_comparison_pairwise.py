#!/usr/bin/env python3
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

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.estimation import EPSEstimator, EstimatorConfig
from src.extra.ieee33_device_day_simulation.network_experiments import (
    reference_controllers as controllers,
    network_dispatch_protocol as protocol,
    train_pooled_model as transfer,
    train_mixed_model as mixed,
    pairwise_curtailment as pairwise,
    ieee69_network_implementation as implementation,
)
from src.signal import SignalOptimizer


OUTPUT = ROOT / "results/ieee69_network_implementation/feeder_comparison"
PROTOCOL = "feeder_comparison_pairwise_v3"
TOPOLOGIES = ("ieee33", "ieee69", "ieee123")
DIRECT_STRESS_MODES = ("M0",)
SEED_COUNT = 30
TRAINING_SAMPLES = 256
VALIDATION_SAMPLES = 64
BASE_SEED = 29_000_000
IEEE123_NETWORK = ROOT / "src/extra/ieee33_device_day_simulation/configs/network_ieee123.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _configure_ieee123() -> dict[str, Any]:
    case = implementation.load_network_case(IEEE123_NETWORK)
    layout = implementation._radial_layout(case)
    buses = sorted(int(bus) for bus in layout["buses"])
    ordered = sorted(buses, key=lambda bus: (layout["distance"].get(bus, 0.0), bus))
    implementation.DISTAL_GROUPS["ieee123"] = (
        tuple(ordered[: max(1, len(ordered) // 2)]),
        tuple(ordered[max(1, len(ordered) // 2) :]),
    )
    implementation.ABSORPTION_ZONES["ieee123"] = tuple(
        tuple(int(bus) for bus in group) for group in np.array_split(ordered, 6)
    )
    branch_keys = [f"{int(row[0])}-{int(row[1])}" for row in case["branches"]]
    implementation.DERATINGS["ieee123"] = {
        key: value for key, value in zip(branch_keys, (0.65, 0.55, 0.60))
    }
    return case


def _network_case(topology: str) -> dict[str, Any]:
    if topology == "ieee33":
        return implementation._network_cases()["ieee33"]
    if topology == "ieee69":
        return implementation._network_cases()["ieee69"]
    if topology == "ieee123":
        return _configure_ieee123()
    raise ValueError(f"unknown target network: {topology}")


def _pair_records(
    dataset_a: str,
    dataset_b: str,
    partition: str,
    seed: int,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    weights = {dataset_a: 0.5, dataset_b: 0.5}
    records, config, audit = mixed._sample_mixed_fleet(
        "S4-A", "aggregate", partition, seed, weights=weights
    )
    config = copy.deepcopy(config)
    config["control"] = dict(config["control"])
    config["control"]["pure_sim_device_model"] = True
    config["control"]["network_feedback"] = False
    config["control"]["eps_control_mode"] = "fused"
    config["original_model"] = dict(config["original_model"])
    config["original_model"]["eps_packet_loss_rate"] = 0.001
    config["original_model"]["device_offline_rate"] = 0.0
    return records, config, audit


def _network_response_samples(
    records: list[Any],
    config: dict[str, Any],
    case: dict[str, Any],
    topology: str,
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    placement_seed = seed + 31_000
    reference_records, _ = implementation._remap_records(
        records, case, placement_seed, "load_weighted"
    )
    contexts: dict[str, dict[str, Any]] = {}
    for stress_index, stress_mode in enumerate(DIRECT_STRESS_MODES):
        placement_mode = str(implementation.STRESS_MODES[stress_mode]["placement_mode"])
        placed_records, placement = implementation._remap_records(
            records, case, placement_seed + 1_000 * (stress_index + 1), placement_mode
        )
        contexts[stress_mode] = {
            "scenario": implementation._build_scenario(
                placed_records,
                reference_records,
                case,
                topology,
                stress_mode,
                seed + 2_000 * (stress_index + 1),
                placement,
            ),
            "network": implementation._network(case, topology, stress_mode),
            "availability": protocol.availability_probability(
                placed_records, config, mixed.AVAILABILITY_MODE
            ),
            "adapter": protocol.OriginalEPSAdapter(
                placed_records, config, seed + 3_000 * (stress_index + 1)
            ),
        }
    rng = np.random.default_rng(seed + 71_000)
    schedule = np.resize(np.asarray(DIRECT_STRESS_MODES, dtype=object), sample_count)
    rng.shuffle(schedule)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    zone_count = len(config["zones"]["zones"])
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    stress_counts = {stress_mode: 0 for stress_mode in DIRECT_STRESS_MODES}
    for index, stress_value in enumerate(schedule):
        stress_mode = str(stress_value)
        context = contexts[stress_mode]
        scenario = context["scenario"]
        network = context["network"]
        availability = context["availability"]
        adapter = context["adapter"]
        supply_demand, intensity, hour = protocol._signal_parameters(
            index, sample_count, rng
        )
        profile_step = int(hour * 12 + rng.integers(0, 12))
        adapter.reset_to_initial_state()
        energy_kwh = 0.0
        for offset in range(3):
            step = (profile_step + offset) % controllers.STEPS
            zone_signals = [
                adapter.simulator._signal_generator.generate_signal(
                    zone, supply_demand, intensity, priority=10
                )
                for zone in range(zone_count)
            ]
            batch = adapter.step(zone_signals, step, availability[step])
            load_map, input_map = implementation._maps(scenario, step)
            dispatch = network.dispatch(
                load_map,
                input_map,
                np.asarray(batch.desired_kw, dtype=float),
                scenario["device_buses"],
            )
            accepted = adapter.apply_dispatch(dispatch.accepted_kw)
            energy_kwh += float(np.sum(accepted)) * dt
        signals.append(
            {
                "supply_demand": supply_demand,
                "intensity": intensity,
                "price": 0.0,
                "hour": hour,
                "day_of_week": 0,
                "direction": 1 if supply_demand <= 7 else -1,
            }
        )
        responses.append(energy_kwh / max(3.0 * dt, 1e-12))
        stress_counts[stress_mode] += 1
    return signals, responses, {
        "network_constrained_labels": True,
        "training_topology": topology,
        "training_stress_modes": list(DIRECT_STRESS_MODES),
        "stress_sample_counts": stress_counts,
        "availability_mean": float(
            np.mean([np.mean(value["availability"]) for value in contexts.values()])
        ),
    }


def _model_paths(task: dict[str, Any]) -> tuple[Path, Path]:
    topology = str(task["topology"])
    pair_id = str(task["pair_id"])
    return (
        OUTPUT / "models" / topology / f"{pair_id}.pt",
        OUTPUT / "models" / topology / f"{pair_id}.json",
    )


def _train_pair(task: dict[str, Any]) -> dict[str, Any]:
    model_path, metadata_path = _model_paths(task)
    if model_path.is_file() and metadata_path.is_file() and not task["force"]:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("protocol") == PROTOCOL
            and metadata.get("model_sha256") == _sha256(model_path)
            and int(metadata.get("training_samples", -1)) == int(task["training_samples"])
            and int(metadata.get("validation_samples", -1)) == int(task["validation_samples"])
        ):
            return metadata
    topology = str(task["topology"])
    case = _network_case(topology)
    train_records, config, train_audit = _pair_records(
        str(task["dataset_a"]), str(task["dataset_b"]), "train", int(task["seed"])
    )
    validation_records, _, validation_audit = _pair_records(
        str(task["dataset_a"]), str(task["dataset_b"]), "validation", int(task["seed"]) + 10_000
    )
    train_signals, train_responses, train_response_audit = _network_response_samples(
        train_records, config, case, topology, int(task["training_samples"]), int(task["seed"]) + 100
    )
    validation_signals, validation_responses, validation_response_audit = _network_response_samples(
        validation_records, config, case, topology, int(task["validation_samples"]), int(task["seed"]) + 20_000
    )
    try:
        import torch

        torch.manual_seed(int(task["seed"]))
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = EPSEstimator(
        EstimatorConfig(
            target_coverage=0.9,
            enable_conformal=True,
            use_cqr=True,
            use_pytorch=True,
            pytorch_epochs=controllers.EPS_TRAINING_EPOCHS,
            pytorch_batch_size=64,
            pytorch_learning_rate=0.001,
        )
    )
    started = time.perf_counter()
    estimator.fit(train_signals, train_responses)
    predicted = np.asarray(
        [estimator.estimate(signal).response_kw for signal in validation_signals], dtype=float
    )
    actual = np.asarray(validation_responses, dtype=float)
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    controllers._save_eps_model(estimator, model_path)
    metadata = {
        "protocol": PROTOCOL,
        "pair_id": task["pair_id"],
        "pair_label": task["pair_label"],
        "dataset_a": task["dataset_a"],
        "dataset_b": task["dataset_b"],
        "topology": topology,
        "training_regime": "pair-specific target-network direct training",
        "training_target": f"{topology.upper()} M0 network-delivered aggregate response kW",
        "training_samples": len(train_signals),
        "validation_samples": len(validation_signals),
        "held_out_r2": float(1.0 - np.sum((actual - predicted) ** 2) / max(denominator, 1e-12)),
        "training_fleet": train_audit,
        "validation_fleet": validation_audit,
        "training_response": train_response_audit,
        "validation_response": validation_response_audit,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "training_seconds": time.perf_counter() - started,
    }
    _write_json(metadata_path, metadata)
    return metadata


def _evaluate_pair(task: dict[str, Any]) -> dict[str, Any]:
    topology = str(task["topology"])
    raw_path = OUTPUT / "raw" / topology / f"{task['pair_id']}.json"
    model_path, _ = _model_paths(task)
    if raw_path.is_file() and not task["force"]:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") == PROTOCOL
            and payload.get("model_sha256") == _sha256(model_path)
            and len(payload.get("seed_results", []))
            == int(task["seed_count"]) * len(DIRECT_STRESS_MODES)
        ):
            return payload
    case = _network_case(topology)
    estimator, _, _ = protocol.load_frozen_eps_controller(model_path)
    optimizer = SignalOptimizer(transfer.ScaledEstimator(estimator, mixed.RESPONSE_SCALE_KW, 0.0))
    rows: list[dict[str, Any]] = []
    for seed_index in range(int(task["seed_count"])):
        base_seed = int(task["seed"]) + seed_index
        base_records, config, audit = _pair_records(
            str(task["dataset_a"]), str(task["dataset_b"]), "test", base_seed
        )
        reference_records, _ = implementation._remap_records(
            base_records, case, base_seed + 31_000, "load_weighted"
        )
        for stress_index, stress_mode in enumerate(DIRECT_STRESS_MODES):
            placement_mode = str(implementation.STRESS_MODES[stress_mode]["placement_mode"])
            records, placement = implementation._remap_records(
                base_records,
                case,
                base_seed + 31_001 + 1_000 * stress_index,
                placement_mode,
            )
            scenario = implementation._build_scenario(
                records,
                reference_records,
                case,
                topology,
                stress_mode,
                base_seed + 2_000 * (stress_index + 1),
                placement,
            )
            availability = protocol.availability_probability(
                records, config, mixed.AVAILABILITY_MODE
            )
            result, _ = implementation._run_seed(
                "eps_ieee69_fused",
                records,
                config,
                scenario,
                availability,
                implementation._network(case, topology, stress_mode),
                base_seed + 1_800_000 + stress_index,
                base_seed + 1_910_000,
                optimizer,
                False,
            )
            rows.append({
                "protocol": PROTOCOL,
                "pair_id": task["pair_id"],
                "composition_id": task["pair_id"],
                "pair_index": int(task["pair_index"]),
                "pair_label": task["pair_label"],
                "dataset_a": task["dataset_a"],
                "dataset_b": task["dataset_b"],
                "topology": topology,
                "network_mode": topology,
                "stress_mode": stress_mode,
                "stress_label": implementation.STRESS_MODES[stress_mode]["label"],
                "seed_index": seed_index,
                "seed": base_seed,
                "fleet_size": len(records),
                "devices_a": audit["counts"][task["dataset_a"]],
                "devices_b": audit["counts"][task["dataset_b"]],
                "algorithm": "eps_ieee69_fused",
                "algorithm_label": "EPS",
                "complexity": "O(1)",
                "training_regime": "pair-specific target-network direct training",
                "model_path": str(model_path),
                "model_sha256": _sha256(model_path),
                **result,
            })
    payload = {
        "protocol": PROTOCOL,
        "pair_id": task["pair_id"],
        "topology": topology,
        "seed_count": int(task["seed_count"]),
        "model_sha256": _sha256(model_path),
        "seed_results": rows,
    }
    _write_json(raw_path, payload)
    return payload


def _tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for pair in pairwise._pair_specs():
        for topology_index, topology in enumerate(TOPOLOGIES):
            values.append({
                **pair,
                "topology": topology,
                "seed": BASE_SEED + int(pair["pair_index"]) * 100_000 + topology_index * 10_000_000,
                "seed_count": args.seed_count,
                "training_samples": args.training_samples,
                "validation_samples": args.validation_samples,
                "force": args.force,
            })
    return values


def _write_ieee123_spatial_source() -> None:
    source = ROOT / "results/ieee123_safety_audit/trained_eps_ieee123_direct/data/trained_eps_by_seed.csv"
    if not source.is_file():
        raise FileNotFoundError(f"missing IEEE-123 direct-training results: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("algorithm") == "eps_ieee69_fused" and row.get("stress_mode") in {"M4", "M5", "M6"}
        ]
    expected = len(implementation.NETWORK_DATASETS) * 3 * SEED_COUNT
    if len(rows) != expected:
        raise RuntimeError(f"incomplete IEEE-123 M4-M6 direct-training results: {len(rows)}, expected {expected}")
    _write_rows(OUTPUT / "data/ieee123_original_direct_m4_m6_by_seed.csv", rows)


def _merge(tasks: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for task in tasks:
        topology = str(task["topology"])
        raw_path = OUTPUT / "raw" / topology / f"{task['pair_id']}.json"
        _, metadata_path = _model_paths(task)
        if not raw_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"supplementary task not finished: {topology}/{task['pair_id']}")
        rows.extend(json.loads(raw_path.read_text(encoding="utf-8"))["seed_results"])
        metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    _write_rows(OUTPUT / "data/pairwise_eps_target_direct_by_seed.csv", rows)
    summaries = []
    for task in tasks:
        selected = [
            row for row in rows
            if row["pair_id"] == task["pair_id"] and row["topology"] == task["topology"]
        ]
        summaries.append({
            "pair_id": task["pair_id"],
            "pair_label": task["pair_label"],
            "dataset_a": task["dataset_a"],
            "dataset_b": task["dataset_b"],
            "topology": task["topology"],
            "algorithm": "eps_ieee69_fused",
            "mean_reduction_pct": float(np.mean([float(row["mean_reduction_pct"]) for row in selected])),
            "seed_count": len(selected),
        })
    _write_rows(OUTPUT / "data/pairwise_eps_target_direct_summary.csv", summaries)
    _write_json(OUTPUT / "data/training_metadata.json", metadata)
    _write_json(OUTPUT / "manifest.json", {
        "protocol": PROTOCOL,
        "status": "completed",
        "pair_count": len(pairwise._pair_specs()),
        "topologies": list(TOPOLOGIES),
        "seed_count": SEED_COUNT,
        "training_samples": TRAINING_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "evaluation_stress_modes": list(DIRECT_STRESS_MODES),
        "direct_training": "pair-specific target-network direct training",
        "data": [
            "data/ieee123_original_direct_m4_m6_by_seed.csv",
            "data/pairwise_eps_target_direct_by_seed.csv",
            "data/pairwise_eps_target_direct_summary.csv",
        ],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate pair-specific EPS models directly on each target network.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--training-samples", type=int, default=TRAINING_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=VALIDATION_SAMPLES)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("workers must be between 1 and 2 to limit background hardware load")
    if not 1 <= args.seed_count <= SEED_COUNT:
        raise ValueError("seed-count must be between 1 and 30")
    if args.training_samples < 128 or args.validation_samples < 32:
        raise ValueError("training/validation samples must be at least 128/32")
    tasks = _tasks(args)
    if args.max_tasks is not None:
        tasks = tasks[: max(1, int(args.max_tasks))]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT / "manifest.json", {
        "protocol": PROTOCOL,
        "status": "running",
        "selected_task_count": len(tasks),
        "pair_count": len(pairwise._pair_specs()),
        "topologies": list(TOPOLOGIES),
        "seed_count": args.seed_count,
        "training_samples": args.training_samples,
        "validation_samples": args.validation_samples,
        "evaluation_stress_modes": list(DIRECT_STRESS_MODES),
    })
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {executor.submit(_train_pair, task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            future.result()
            _write_json(OUTPUT / "checkpoint.json", {"stage": "training", "completed": count, "total": len(tasks), "last": f"{task['topology']}/{task['pair_id']}"})
            print(json.dumps({"stage": "training", "completed": count, "total": len(tasks), "topology": task["topology"], "pair": task["pair_id"]}, ensure_ascii=False), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {executor.submit(_evaluate_pair, task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            future.result()
            _write_json(OUTPUT / "checkpoint.json", {"stage": "evaluation", "completed": count, "total": len(tasks), "last": f"{task['topology']}/{task['pair_id']}"})
            print(json.dumps({"stage": "evaluation", "completed": count, "total": len(tasks), "topology": task["topology"], "pair": task["pair_id"]}, ensure_ascii=False), flush=True)
    if len(tasks) == len(pairwise._pair_specs()) * len(TOPOLOGIES) and args.seed_count == SEED_COUNT:
        _write_ieee123_spatial_source()
        _merge(tasks)
        print(json.dumps({"status": "complete", "tasks": len(tasks)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
