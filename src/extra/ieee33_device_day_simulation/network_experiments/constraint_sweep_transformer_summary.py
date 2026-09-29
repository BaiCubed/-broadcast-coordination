from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ieee69_network_implementation as implementation
from . import ieee123_safety_audit as safety_audit
from . import constraint_sweep_transformer as replay


ROOT = Path(__file__).resolve().parents[4]
REPLAY_OUTPUT = ROOT / "results/ieee69_network_implementation/constraint_sweep_transformer"
ORIGINAL_123_OUTPUT = ROOT / "results/ieee69_network_implementation/constraint_sweep_transformer_ieee123"
SINGLE_SUMMARY = ROOT / "results/ieee69_network_implementation/constraint_sweep/data/single_constraint_effect_summary.csv"
OVERRIDE = REPLAY_OUTPUT / "data/transformer_panel_replay_summary.csv"
PROTOCOL = "constraint_sweep_transformer_summary_v1"
TOPOLOGIES = ("ieee33", "ieee69", "ieee123")


def _original_123_case() -> dict:
    case = replay.load_network_case(safety_audit.NETWORK_FILE)
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


def _case(topology: str) -> dict:
    if topology == "ieee123":
        return _original_123_case()
    return implementation._network_cases()[topology]


def _model_path(dataset: str) -> Path:
    return ROOT / "results/ieee123_safety_audit/trained_eps_ieee123_direct/models" / f"{dataset}.pt"


def _network(case: dict, value: float) -> replay.LocalRadialDistFlow:
    return replay.LocalRadialDistFlow(
        case,
        capacity_multiplier=1.0,
        transformer_multiplier=1.0,
        enforce_line_limit=False,
        enforce_transformer_limit=True,
        enforce_minimum_voltage=False,
        enforce_maximum_voltage=False,
        absolute_transformer_loading_limit=value,
    )


def _metrics_from_trace(trace: list[dict], target: np.ndarray, dt: float) -> dict[str, float]:
    actual = np.asarray([row["zone_matched_absorption_kw"] for row in trace], dtype=float)
    requested = np.asarray([row["requested_kw"] for row in trace], dtype=float)
    admitted = np.asarray([row["network_admitted_kw"] for row in trace], dtype=float)
    target = np.asarray(target, dtype=float)
    baseline_mwh = float(np.sum(target) * dt / 1000.0)
    accepted_mwh = float(np.sum(actual) * dt / 1000.0)
    requested_mwh = float(np.sum(requested) * dt / 1000.0)
    admitted_mwh = float(np.sum(admitted) * dt / 1000.0)
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(
        1.0 - np.sum((actual - target) ** 2) / denominator
        if denominator > 1e-12
        else np.nan
    )
    return {
        "mean_reduction_pct": 100.0 * admitted_mwh / max(requested_mwh, 1e-12),
        "response_r2": r2,
        "network_acceptance_ratio": admitted_mwh / max(requested_mwh, 1e-12),
        "baseline_curtailment_mwh": baseline_mwh,
        "accepted_absorption_mwh": accepted_mwh,
        "network_admitted_power_mwh": admitted_mwh,
    }


def _base_row(dataset: str, topology: str, value: float, metrics: dict[str, float]) -> dict:
    reduction = float(metrics["mean_reduction_pct"])
    r2 = float(metrics["response_r2"])
    acceptance = float(metrics["network_acceptance_ratio"])
    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "dataset_label": implementation.DATASET_LABELS[dataset],
        "topology": topology,
        "constraint": "Transformer",
        "constraint_value": value,
        "curtailment_value_pct": reduction,
        "response_r2_pct": 100.0 * r2,
        "network_acceptance_pct": 100.0 * acceptance,
        "mean_reduction_pct_mean": reduction,
        "mean_reduction_pct_median": reduction,
        "response_r2_mean": r2,
        "response_r2_median": r2,
        "network_acceptance_ratio_mean": acceptance,
        "network_acceptance_ratio_median": acceptance,
        "baseline_curtailment_mwh": metrics["baseline_curtailment_mwh"],
        "accepted_absorption_mwh": metrics["accepted_absorption_mwh"],
        "network_admitted_power_mwh": metrics["network_admitted_power_mwh"],
        "seed_count": 1,
        "fixed_desired": True,
    }


def _read_existing_replay(dataset: str, topology: str, case: dict) -> list[dict]:
    csv_path = REPLAY_OUTPUT / "data/raw" / topology / dataset / "seed_00.csv"
    trace_path = REPLAY_OUTPUT / "data/raw" / topology / dataset / "seed_00_trace.json"
    frame = pd.read_csv(csv_path)
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    base_seed = implementation._base_seed(dataset, 0)
    records, config, _, scenario = replay._scenario(dataset, topology, 0, case)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    target = np.sum(scenario["local_surplus_by_bus"], axis=1)
    rows = []
    for _, raw in frame.iterrows():
        if raw["constraint"] != "Transformer":
            continue
        key = f"Transformer_{float(raw['constraint_value']):.2f}"
        metrics = _metrics_from_trace(trace_payload["replay"][key], target, dt)
        rows.append(_base_row(dataset, topology, float(raw["constraint_value"]), metrics))
    return rows


def _run_original_123(dataset: str, case: dict) -> list[dict]:
    model_path = _model_path(dataset)
    _, optimizer, _ = replay.training.load_frozen_eps_controller(model_path)
    records, config, availability, scenario = replay._scenario(dataset, "ieee123", 0, case)
    base_seed = implementation._base_seed(dataset, 0)
    desired, availability_rows = replay._generate_fixed_desired(
        records,
        config,
        scenario,
        availability,
        optimizer,
        base_seed + 410_000,
        base_seed + 420_000,
    )
    output_dir = ORIGINAL_123_OUTPUT / "data/raw/ieee123" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    target = np.sum(scenario["local_surplus_by_bus"], axis=1)
    rows = []
    trace_payload = {"desired_kw": desired.tolist(), "availability": availability_rows.tolist(), "replay": {}}
    for constraint, value in tuple(("Transformer", v) for v in replay.CONSTRAINT_VALUES):
        result = replay._replay(
            records,
            config,
            scenario,
            desired,
            "ieee123",
            case,
            base_seed + 440_000,
            constraint,
            value,
        )
        trace_payload["replay"]["Reference" if constraint == "Reference" else f"Transformer_{value:.2f}"] = result["trace"]
        metrics = _metrics_from_trace(result["trace"], target, dt)
        rows.append(_base_row(dataset, "ieee123", value, metrics))
    replay._write_rows(output_dir / "seed_00.csv", rows)
    replay._write_json(output_dir / "seed_00_trace.json", trace_payload)
    return rows


def _write_override(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    replay._write_rows(OVERRIDE, frame.to_dict("records"))
    replay._write_rows(
        ORIGINAL_123_OUTPUT / "data/transformer_panel_replay_summary.csv",
        [row for row in rows if row["topology"] == "ieee123"],
    )


def main() -> None:
    replay._configure_topology_metadata()
    rows: list[dict] = []
    for topology in ("ieee33", "ieee69"):
        case = _case(topology)
        for dataset in implementation.NETWORK_DATASETS:
            rows.extend(_read_existing_replay(dataset, topology, case))
    case = _case("ieee123")
    for dataset in implementation.NETWORK_DATASETS:
        rows.extend(_run_original_123(dataset, case))
    _write_override(rows)
    replay._write_json(
        ORIGINAL_123_OUTPUT / "manifest.json",
        {
            "protocol": PROTOCOL,
            "status": "completed",
            "topology": "ieee123",
            "seed_count": 1,
            "datasets": list(implementation.NETWORK_DATASETS),
            "constraint_values": list(replay.CONSTRAINT_VALUES),
            "fixed_desired_replay": True,
            "synthetic_ieee123_extension_used": False,
            "metrics": [
                "curtailment_value_pct",
                "response_r2_pct",
                "network_acceptance_pct",
            ],
            "summary": str(ORIGINAL_123_OUTPUT / "data/transformer_panel_replay_summary.csv"),
        },
    )
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(OVERRIDE)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
