from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from . import network_dispatch_protocol as training
from . import ieee69_network_implementation as implementation
from . import ieee123_safety_audit as safety_audit


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/ieee69_network_implementation/constraint_sweep"
PROTOCOL = "constraint_sweep_v2_extended"
TOPOLOGIES = ("ieee33", "ieee69", "ieee123")
CONSTRAINTS = (
    "Reference",
    "Line",
    "Transformer",
    "Minimum voltage",
    "Maximum voltage",
)
CAPACITY_VALUES = (
    1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70,
    0.65, 0.60, 0.55, 0.50,
)
MINIMUM_VOLTAGE_VALUES = tuple(np.round(np.arange(0.9500, 1.0000 + 0.0001, 0.0025), 4))
MAXIMUM_VOLTAGE_VALUES = tuple(np.round(np.arange(1.0500, 1.0000 - 0.0001, -0.0025), 4))
FLEET_SIZE = 5000
DEFAULT_SEED_COUNT = 30


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no data to write: {path}")
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_ieee123() -> dict[str, Any]:
    case = implementation.load_network_case(safety_audit.NETWORK_FILE)
    layout = implementation._radial_layout(case)
    buses = sorted(int(bus) for bus in layout["buses"])
    ordered = sorted(buses, key=lambda bus: (layout["distance"].get(bus, 0.0), bus))
    implementation.ABSORPTION_ZONES["ieee123"] = tuple(
        tuple(int(bus) for bus in chunk) for chunk in np.array_split(ordered, 6)
    )
    midpoint = max(1, len(ordered) // 2)
    implementation.DISTAL_GROUPS["ieee123"] = (
        tuple(ordered[:midpoint]),
        tuple(ordered[midpoint:]),
    )
    return case


def _case(topology: str) -> dict[str, Any]:
    if topology == "ieee123":
        return _configure_ieee123()
    return implementation._network_cases()[topology]


def _model_path(dataset: str, topology: str) -> Path:
    if topology == "ieee69":
        path = ROOT / "results/ieee69_network_implementation/trained_eps_ieee69_direct/models" / f"{dataset}.pt"
    elif topology == "ieee123":
        path = ROOT / "results/ieee123_safety_audit/trained_eps_ieee123_direct/models" / f"{dataset}.pt"
    elif dataset in {implementation.NEXTGEN, implementation.DATA2}:
        path = ROOT / "results/ieee69_network_implementation/trained_eps_ieee33_direct/models" / f"{dataset}.pt"
    else:
        path = (
            ROOT
            / f"results/{dataset}_ieee33_real_load/network_baseline/network_constrained/data"
            / "eps_estimator_final_fixed5000_ieee33_data_driven.pt"
        )
    if not path.is_file():
        raise FileNotFoundError(f"missing the {topology} directly trained model: {path}")
    return path


def _network(
    case: dict[str, Any], constraint: str, value: float
) -> implementation.LocalRadialDistFlow:
    line_multiplier = value if constraint == "Line" else 1.0
    transformer_multiplier = value if constraint == "Transformer" else 1.0
    minimum_voltage = value if constraint == "Minimum voltage" else 0.95
    maximum_voltage = value if constraint == "Maximum voltage" else 1.05
    return implementation.LocalRadialDistFlow(
        case,
        capacity_multiplier=line_multiplier,
        transformer_multiplier=transformer_multiplier,
        enforce_line_limit=constraint == "Line",
        enforce_transformer_limit=constraint == "Transformer",
        enforce_minimum_voltage=constraint == "Minimum voltage",
        enforce_maximum_voltage=constraint == "Maximum voltage",
        minimum_voltage_pu=minimum_voltage,
        maximum_voltage_pu=maximum_voltage,
    )


def _constraint_values(constraint: str) -> tuple[float, ...]:
    if constraint in {"Line", "Transformer"}:
        return CAPACITY_VALUES
    if constraint == "Minimum voltage":
        return MINIMUM_VOLTAGE_VALUES
    if constraint == "Maximum voltage":
        return MAXIMUM_VOLTAGE_VALUES
    return (1.0,)


def _scenario(
    dataset: str,
    topology: str,
    seed_index: int,
    case: dict[str, Any],
) -> tuple[list[Any], dict[str, Any], np.ndarray, dict[str, Any]]:
    base_seed = implementation._base_seed(dataset, seed_index)
    records, config, _ = implementation._sample_dataset(dataset, base_seed)
    reference_records, _ = implementation._remap_records(
        records, case, base_seed + 31_000, "load_weighted"
    )
    records, placement = implementation._remap_records(
        records, case, base_seed + 32_000, "uniform"
    )
    scenario = implementation._build_scenario(
        records,
        reference_records,
        case,
        topology,
        "M4",
        base_seed + 33_000,
        placement,
    )
    availability = training.availability_probability(
        records, config, implementation.mixed.AVAILABILITY_MODE
    )
    return records, config, availability, scenario


def _run_one(
    dataset: str,
    topology: str,
    seed_index: int,
    records: list[Any],
    config: dict[str, Any],
    availability: np.ndarray,
    scenario: dict[str, Any],
    optimizer: Any,
    case: dict[str, Any],
    constraint: str,
    value: float,
) -> dict[str, Any]:
    base_seed = implementation._base_seed(dataset, seed_index)
    result, _ = implementation._run_seed(
        "eps_ieee69_fused",
        records,
        config,
        scenario,
        availability,
        _network(case, constraint, value),
        base_seed + 410_000,
        base_seed + 420_000,
        optimizer,
        False,
    )
    return {
        **result,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "dataset_label": implementation.DATASET_LABELS[dataset],
        "topology": topology,
        "fleet_size": FLEET_SIZE,
        "seed_index": seed_index,
        "constraint": constraint,
        "constraint_value": value,
        "line_capacity_fraction": value if constraint == "Line" else 1.0,
        "transformer_capacity_fraction": value if constraint == "Transformer" else 1.0,
        "minimum_voltage_limit_pu": value if constraint == "Minimum voltage" else 0.95,
        "maximum_voltage_limit_pu": value if constraint == "Maximum voltage" else 1.05,
        "active_constraint_count": 0 if constraint == "Reference" else 1,
        "direct_training_model": str(_model_path(dataset, topology)),
    }


def _run_dataset_topology(
    dataset: str,
    topology: str,
    seed_count: int,
) -> dict[str, Any]:
    case = _case(topology)
    model_path = _model_path(dataset, topology)
    _, optimizer, _ = training.load_frozen_eps_controller(model_path)
    output_dir = OUTPUT / "data/raw" / topology / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = 1 + sum(len(_constraint_values(name)) for name in CONSTRAINTS[1:])
    completed = 0
    for seed_index in range(seed_count):
        path = output_dir / f"seed_{seed_index:02d}.csv"
        if path.is_file():
            existing = _read_rows(path)
            if len(existing) == expected_rows and all(
                row.get("protocol") == PROTOCOL for row in existing
            ):
                completed += 1
                continue
        records, config, availability, scenario = _scenario(
            dataset, topology, seed_index, case
        )
        rows = [
            _run_one(
                dataset,
                topology,
                seed_index,
                records,
                config,
                availability,
                scenario,
                optimizer,
                case,
                "Reference",
                1.0,
            )
        ]
        for constraint in CONSTRAINTS[1:]:
            for value in _constraint_values(constraint):
                rows.append(
                    _run_one(
                        dataset,
                        topology,
                        seed_index,
                        records,
                        config,
                        availability,
                        scenario,
                        optimizer,
                        case,
                        constraint,
                        value,
                    )
                )
        _write_rows(path, rows)
        completed += 1
    return {
        "dataset": dataset,
        "topology": topology,
        "completed_seeds": completed,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
    }


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["topology"],
            row["constraint"],
            float(row["constraint_value"]),
        )
        groups.setdefault(key, []).append(row)
    output = []
    metrics = (
        "mean_reduction_pct",
        "response_r2",
        "network_acceptance_ratio",
        "maximum_branch_loading",
        "maximum_transformer_loading",
        "minimum_voltage_pu",
        "maximum_voltage_pu",
        "mean_local_clip_fraction",
        "fallback_global_steps",
        "requested_violation_steps",
        "executed_violation_steps",
    )
    for key, group in sorted(groups.items()):
        dataset, topology, constraint, value = key
        record: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": implementation.DATASET_LABELS[dataset],
            "topology": topology,
            "constraint": constraint,
            "constraint_value": value,
            "seed_count": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            record[f"{metric}_median"] = float(np.nanmedian(values))
            record[f"{metric}_mean"] = float(np.nanmean(values))
        output.append(record)
    reference = {
        (row["dataset"], row["topology"]): row
        for row in output
        if row["constraint"] == "Reference"
    }
    for row in output:
        base = reference[(row["dataset"], row["topology"])]
        denominator = float(base["mean_reduction_pct_median"])
        row["curtailment_reduction_retention_pct"] = (
            100.0 * float(row["mean_reduction_pct_median"]) / denominator
            if denominator > 1e-12
            else float("nan")
        )
        row["curtailment_reduction_loss_percentage_points"] = (
            denominator - float(row["mean_reduction_pct_median"])
        )
    return output


def _finalize(seed_count: int, jobs: list[dict[str, Any]]) -> None:
    paths = sorted((OUTPUT / "data/raw").glob("*/*/seed_*.csv"))
    rows = [row for path in paths for row in _read_rows(path)]
    summaries = _summarize(rows)
    _write_rows(OUTPUT / "data/single_constraint_effect_by_seed.csv", rows)
    _write_rows(OUTPUT / "data/single_constraint_effect_summary.csv", summaries)
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "topologies": list(TOPOLOGIES),
        "datasets": list(implementation.NETWORK_DATASETS),
        "constraints": list(CONSTRAINTS),
        "capacity_values": list(CAPACITY_VALUES),
        "minimum_voltage_values": list(MINIMUM_VOLTAGE_VALUES),
        "maximum_voltage_values": list(MAXIMUM_VOLTAGE_VALUES),
        "seed_count": seed_count,
        "row_count": len(rows),
        "jobs": jobs,
    }
    _write_json(OUTPUT / "manifest.json", manifest)
    _write_json(OUTPUT / "checkpoint.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experiment on the effect of each of the four individual network constraints on EPS.")
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dataset", action="append", choices=implementation.NETWORK_DATASETS)
    parser.add_argument("--topology", action="append", choices=TOPOLOGIES)
    args = parser.parse_args()
    if not 1 <= args.seed_count <= DEFAULT_SEED_COUNT:
        raise ValueError("seed-count must be between 1 and 30")
    if not 1 <= args.workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    datasets = tuple(args.dataset) if args.dataset else tuple(implementation.NETWORK_DATASETS)
    topologies = tuple(args.topology) if args.topology else TOPOLOGIES
    jobs = [(dataset, topology) for dataset in datasets for topology in topologies]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(jobs)), mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                _run_dataset_topology, dataset, topology, args.seed_count
            ): (dataset, topology)
            for dataset, topology in jobs
        }
        for future in as_completed(futures):
            completed.append(future.result())
            _write_json(
                OUTPUT / "checkpoint.json",
                {
                    "protocol": PROTOCOL,
                    "status": "running",
                    "completed_jobs": len(completed),
                    "job_count": len(jobs),
                    "jobs": completed,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "dataset_topology_complete",
                        "completed_jobs": len(completed),
                        "job_count": len(jobs),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    _finalize(args.seed_count, completed)
    print(json.dumps({"status": "complete", "output": str(OUTPUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
