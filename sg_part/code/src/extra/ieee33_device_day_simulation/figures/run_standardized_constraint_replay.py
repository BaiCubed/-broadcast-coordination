"""Replay one fixed EPS request trajectory under transformer constraints.

This is a causal audit of the network layer.  For each dataset/topology/seed,
the EPS controller emits one fixed desired_kw[t, device] trajectory.  Every
transformer condition then replays that same trajectory; only network
admission changes.  The IEEE-123 case is a synthetic 69-bus expansion that
keeps the IEEE-69 core and adds six nine-bus distal chains.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..network.local_radial_distflow import LocalRadialDistFlow, load_network_case
from . import fig4d_final_protocol as training
from . import run_e22_ieee69_complexity as e22


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/E22/standardized_constraint_replay"
FIGURE = ROOT / "outputs/figs/subpanel/08_standardized_constraint_audit.png"
FIGURE_PDF = ROOT / "outputs/figs/subpanel/08_standardized_constraint_audit.pdf"
TOPOLOGIES = ("ieee33", "ieee69", "ieee123x69")
LABELS = {"ieee33": "IEEE-33", "ieee69": "IEEE-69", "ieee123x69": "IEEE-123 (69 extension)"}
COLORS = {"ieee33": "#365A7C", "ieee69": "#55A6B5", "ieee123x69": "#E5633E"}
CONSTRAINT_VALUES = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50)
PROTOCOL = "E22_standardized_constraint_replay_v1"
FLEET_SIZE = 5000
STEPS = e22.STEPS

# The six roots are leaves in the IEEE-69 core.  Each receives a nine-bus
# distal chain, preserving the 69-bus trunk while increasing radial depth.
EXTENSION_ROOTS = (25, 35, 45, 50, 65, 69)
EXTENSION_ZONE_ROOTS = {
    25: tuple(range(2, 28)),
    35: tuple(range(28, 36)),
    45: tuple(range(36, 47)),
    50: tuple(range(47, 51)),
    65: tuple(range(51, 66)),
    69: tuple(range(66, 70)),
}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_ieee69(case: dict[str, Any]) -> dict[str, Any]:
    expanded = copy.deepcopy(case)
    expanded["name"] = "IEEE_123_E22_expanded_from_69_core"
    expanded["source"] = "Synthetic IEEE-69 core with six nine-bus distal extensions"
    expanded["transformer_capacity_kva"] = float(case["transformer_capacity_kva"])
    branches = list(expanded["branches"])
    base_loads = list(expanded.get("base_loads", []))
    load_by_bus = {int(row[0]): (float(row[1]), float(row[2])) for row in base_loads}
    next_bus = 70
    for root in EXTENSION_ROOTS:
        parent = root
        template = load_by_bus.get(root, (20.0, 10.0))
        for _ in range(9):
            child = next_bus
            # Use a moderate 69-like distal branch impedance for each added edge.
            branches.append([parent, child, 0.1732, 0.0572])
            base_loads.append([child, max(template[0] * 0.50, 10.0), max(template[1] * 0.50, 5.0)])
            parent = child
            next_bus += 1
    if next_bus != 124:
        raise RuntimeError(f"IEEE-69 expansion generated bus {next_bus - 1}, expected 123")
    expanded["branches"] = branches
    expanded["base_loads"] = base_loads
    return expanded


def _configure_topology_metadata() -> None:
    zones: list[tuple[int, ...]] = []
    for root_zone in EXTENSION_ZONE_ROOTS.values():
        zones.append(tuple(root_zone))
    for root, first in zip(EXTENSION_ROOTS, range(70, 124, 9)):
        zone_index = EXTENSION_ROOTS.index(root)
        zones[zone_index] = tuple(zones[zone_index]) + tuple(range(first, first + 9))
    e22.ABSORPTION_ZONES["ieee123x69"] = tuple(zones)
    ordered = []
    for zone in zones:
        ordered.extend(zone)
    midpoint = len(ordered) // 2
    e22.DISTAL_GROUPS["ieee123x69"] = (tuple(ordered[:midpoint]), tuple(ordered[midpoint:]))


def _case(topology: str) -> dict[str, Any]:
    if topology == "ieee123x69":
        return _expand_ieee69(e22._network_cases()["ieee69"])
    return e22._network_cases()[topology]


def _model_path(dataset: str, topology: str) -> Path:
    if topology == "ieee123x69":
        return ROOT / "results/E24/trained_eps_ieee123_direct/models" / f"{dataset}.pt"
    if topology == "ieee69":
        return ROOT / "results/E22/trained_eps_ieee69_direct/models" / f"{dataset}.pt"
    if dataset in {e22.NEXTGEN, e22.DATA2}:
        return ROOT / "results/E22/trained_eps_ieee33_direct/models" / f"{dataset}.pt"
    return ROOT / f"results/{dataset}_ieee33_real_load/coverage_fix/network_constrained_new/data/fig4d_eps_estimator_final_fixed5000_ieee33_data_driven.pt"


def _scenario(dataset: str, topology: str, seed_index: int, case: dict[str, Any]):
    base_seed = e22._base_seed(dataset, seed_index)
    records, config, _ = e22._sample_dataset(dataset, base_seed)
    reference_records, _ = e22._remap_records(records, case, base_seed + 31_000, "load_weighted")
    records, placement = e22._remap_records(records, case, base_seed + 32_000, "uniform")
    scenario = e22._build_scenario(records, reference_records, case, topology, "M4", base_seed + 33_000, placement)
    availability = training.availability_probability(records, config, e22.mixed.AVAILABILITY_MODE)
    return records, config, availability, scenario


def _network(case: dict[str, Any], topology: str, constraint: str, value: float) -> LocalRadialDistFlow:
    line_multiplier = value if constraint == "Line" else 1.0
    return LocalRadialDistFlow(
        case,
        capacity_multiplier=line_multiplier,
        # Keep the physical transformer fixed; only the allowed loading
        # fraction changes across the replay conditions.
        transformer_multiplier=1.0,
        enforce_line_limit=constraint == "Line",
        enforce_transformer_limit=constraint == "Transformer",
        enforce_minimum_voltage=constraint == "Minimum voltage",
        enforce_maximum_voltage=constraint == "Maximum voltage",
        minimum_voltage_pu=value if constraint == "Minimum voltage" else 0.95,
        maximum_voltage_pu=value if constraint == "Maximum voltage" else 1.05,
        absolute_transformer_loading_limit=value if constraint == "Transformer" else None,
    )


def _generate_fixed_desired(records, config, scenario, availability_probability, optimizer, seed, availability_seed):
    capacities = np.asarray([float(record.capacity_kwh) * 0.90 * float(record.initial_soh) for record in records])
    peaks = np.asarray([float(record.peak_power_kw) for record in records])
    soc = np.asarray([float(record.initial_soc) for record in records])
    adapter = e22.OriginalEPSAdapter(records, config, seed=seed)
    state = {
        "load_variability": np.asarray([float(np.std(record.load_kw)) / max(float(np.mean(record.load_kw)), 1e-6) for record in records]),
        "availability_probability": availability_probability,
        "eps_optimizer": optimizer,
        "eps_adapter": adapter,
    }
    rng = np.random.default_rng(seed)
    availability_rng = np.random.default_rng(availability_seed)
    desired_rows: list[np.ndarray] = []
    availability_rows: list[np.ndarray] = []
    for step in range(STEPS):
        available = (availability_rng.random(len(records)) < np.clip(availability_probability[step], 0.0, 1.0)).astype(float)
        surplus = float(np.sum(scenario["local_surplus_by_bus"][step]))
        desired = e22.legacy._strategy_dispatch(
            "eps_broadcast", step, surplus, scenario, records, soc, capacities, peaks,
            available, rng, state, config,
        )
        desired = np.maximum(np.asarray(desired, dtype=float), 0.0)
        actual = adapter.apply_dispatch(desired)
        soc = adapter.battery_states()
        target = float(state.get("eps_step_target_kw", 0.0))
        absorbed = float(np.sum(np.maximum(actual, 0.0)))
        shortfall = max(target - absorbed, 0.0) / target if target > 1e-12 else 0.0
        state["eps_intensity_correction"] = float(np.clip(0.80 * float(state.get("eps_intensity_correction", 0.0)) + 0.30 * shortfall, 0.0, 0.45))
        desired_rows.append(desired)
        availability_rows.append(available)
    return np.asarray(desired_rows), np.asarray(availability_rows)


def _maps(scenario: dict[str, Any], step: int):
    return e22._maps(scenario, step)


def _replay(records, config, scenario, desired, topology, case, seed, constraint, value):
    network = _network(case, topology, constraint, value)
    adapter = e22.OriginalEPSAdapter(records, config, seed=seed)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    admitted = []
    actual = []
    matched = []
    requested = []
    max_transformer = 0.0
    max_branch = 0.0
    min_voltage = float("inf")
    max_voltage = float("-inf")
    local_clip = []
    fallback_steps = 0
    requested_violations = 0
    executed_violations = 0
    added_violations = 0
    trace_rows = []
    for step in range(STEPS):
        load_map, input_map = _maps(scenario, step)
        dispatch = network.dispatch(load_map, input_map, desired[step], scenario["device_buses"])
        network_kw = np.maximum(dispatch.accepted_kw, 0.0)
        actual_kw = np.maximum(adapter.apply_dispatch(network_kw), 0.0)
        by_zone = np.bincount(scenario["device_absorption_zones"], weights=actual_kw, minlength=scenario["surplus_by_zone"].shape[1])
        matched_kw = float(np.sum(np.minimum(by_zone, scenario["surplus_by_zone"][step])))
        admitted_kw = float(np.sum(network_kw))
        actual_total_kw = float(np.sum(actual_kw))
        requested_kw = float(np.sum(np.maximum(desired[step], 0.0)))
        admitted.append(admitted_kw)
        actual.append(actual_total_kw)
        matched.append(matched_kw)
        requested.append(requested_kw)
        max_transformer = max(max_transformer, float(dispatch.executed.transformer_loading))
        max_branch = max(max_branch, max(dispatch.executed.branch_loading.values(), default=0.0))
        min_voltage = min(min_voltage, min(dispatch.executed.voltage_pu.values()))
        max_voltage = max(max_voltage, max(dispatch.executed.voltage_pu.values()))
        local_clip.append(float(dispatch.local_clip_fraction))
        fallback_steps += int(dispatch.fallback_global)
        requested_violations += int(e22._violates(dispatch.requested, network))
        executed_violations += int(e22._violates(dispatch.executed, network))
        added_violations += int(e22._adds_violation(dispatch.executed, dispatch.baseline, network))
        trace_rows.append({"step": step, "requested_kw": requested_kw, "network_admitted_kw": admitted_kw, "actual_absorption_kw": actual_total_kw, "zone_matched_absorption_kw": matched_kw, "transformer_loading": float(dispatch.executed.transformer_loading), "branch_loading": max(dispatch.executed.branch_loading.values(), default=0.0), "minimum_voltage_pu": min(dispatch.executed.voltage_pu.values()), "maximum_voltage_pu": max(dispatch.executed.voltage_pu.values())})
    requested_mwh = float(np.sum(requested) * dt / 1000.0)
    admitted_mwh = float(np.sum(admitted) * dt / 1000.0)
    actual_mwh = float(np.sum(actual) * dt / 1000.0)
    matched_mwh = float(np.sum(matched) * dt / 1000.0)
    return {
        "requested_power_mwh": requested_mwh,
        "network_admitted_power_mwh": admitted_mwh,
        "actual_absorption_power_mwh": actual_mwh,
        "zone_matched_absorption_mwh": matched_mwh,
        "network_acceptance_ratio": admitted_mwh / requested_mwh if requested_mwh > 1e-12 else 1.0,
        "actual_to_network_ratio": actual_mwh / admitted_mwh if admitted_mwh > 1e-12 else 1.0,
        "mean_step_acceptance_ratio": float(np.mean([row["network_admitted_kw"] / row["requested_kw"] if row["requested_kw"] > 1e-12 else 1.0 for row in trace_rows])),
        "mean_local_clip_fraction": float(np.mean(local_clip)),
        "requested_violation_steps": requested_violations,
        "executed_violation_steps": executed_violations,
        "added_violation_steps": added_violations,
        "requested_added_violation_steps": added_violations,
        "requested_violation_free_pct": 100.0 * (STEPS - requested_violations) / STEPS,
        "requested_added_violation_free_pct": 100.0 * (STEPS - added_violations) / STEPS,
        "executed_violation_free_pct": 100.0 * (STEPS - executed_violations) / STEPS,
        "fallback_global_steps": fallback_steps,
        "minimum_voltage_pu": min_voltage,
        "maximum_voltage_pu": max_voltage,
        "maximum_branch_loading": max_branch,
        "maximum_transformer_loading": max_transformer,
        "response_r2": e22._r2(np.asarray(matched), np.asarray([float(np.sum(row)) for row in scenario["surplus_by_zone"]])),
        "response_nrmse": float(np.sqrt(np.mean((np.asarray(matched) - np.asarray([float(np.sum(row)) for row in scenario["surplus_by_zone"]])) ** 2)) / max(float(np.mean([float(np.sum(row)) for row in scenario["surplus_by_zone"] if float(np.sum(row)) > 0.0])), 1e-12)),
        "trace": trace_rows,
    }


def _run_dataset_topology(dataset: str, topology: str, seed_index: int) -> dict[str, Any]:
    case = _case(topology)
    model_path = _model_path(dataset, topology)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    _, optimizer, _ = training.load_frozen_eps_controller(model_path)
    records, config, availability, scenario = _scenario(dataset, topology, seed_index, case)
    base_seed = e22._base_seed(dataset, seed_index)
    desired, availability_rows = _generate_fixed_desired(records, config, scenario, availability, optimizer, base_seed + 410_000, base_seed + 420_000)
    output_dir = OUTPUT / "data/raw" / topology / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = {}
    reference = _replay(records, config, scenario, desired, topology, case, base_seed + 440_000, "Reference", 1.0)
    rows.append({k: v for k, v in reference.items() if k != "trace"} | {"dataset": dataset, "topology": topology, "seed_index": seed_index, "constraint": "Reference", "constraint_value": 1.0, "protocol": PROTOCOL, "fixed_desired": True})
    traces["Reference"] = reference["trace"]
    for value in CONSTRAINT_VALUES:
        result = _replay(records, config, scenario, desired, topology, case, base_seed + 440_000, "Transformer", value)
        rows.append({k: v for k, v in result.items() if k != "trace"} | {"dataset": dataset, "topology": topology, "seed_index": seed_index, "constraint": "Transformer", "constraint_value": value, "protocol": PROTOCOL, "fixed_desired": True})
        traces[f"Transformer_{value:.2f}"] = result["trace"]
    _write_rows(output_dir / f"seed_{seed_index:02d}.csv", rows)
    _write_json(output_dir / f"seed_{seed_index:02d}_trace.json", {"desired_kw": desired.tolist(), "availability": availability_rows.tolist(), "replay": traces})
    return {"dataset": dataset, "topology": topology, "seed_index": seed_index, "model": str(model_path), "model_sha256": _sha256(model_path)}


def _summarize() -> pd.DataFrame:
    paths = sorted((OUTPUT / "data/raw").glob("*/*/seed_*.csv"))
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    group_cols = ["topology", "constraint", "constraint_value"]
    metrics = ["network_admitted_power_mwh", "actual_absorption_power_mwh", "zone_matched_absorption_mwh", "network_acceptance_ratio", "actual_to_network_ratio", "maximum_transformer_loading"]
    summary = frame.groupby(group_cols, as_index=False)[metrics].mean()
    _write_rows(OUTPUT / "data/standardized_constraint_replay_summary.csv", summary.to_dict("records"))
    return summary


def _plot(summary: pd.DataFrame) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    metrics = [
        ("network_admitted_power_mwh", "Network-admitted energy (MWh)"),
        ("actual_absorption_power_mwh", "Actual battery absorption (MWh)"),
        ("zone_matched_absorption_mwh", "Zone-matched curtailment absorption (MWh)"),
    ]
    for axis, (metric, ylabel) in zip(axes, metrics):
        for topology in TOPOLOGIES:
            selected = summary[(summary.topology == topology) & (summary.constraint == "Transformer")].sort_values("constraint_value", ascending=False)
            axis.plot(selected.constraint_value, selected[metric], marker="o", linewidth=2.2, markersize=5.5, color=COLORS[topology], label=LABELS[topology])
            reference = summary[(summary.topology == topology) & (summary.constraint == "Reference")]
            if not reference.empty:
                axis.scatter([1.0], reference[metric], marker="*", s=75, color=COLORS[topology], edgecolor="white", linewidth=0.7, zorder=4)
        axis.set_xlabel("Transformer loading limit")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.68, 1.03)
        axis.set_xticks(CONSTRAINT_VALUES)
        axis.invert_xaxis()
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle("Fixed desired_kw replay, one seed per dataset; IEEE-123 is a 69-bus extension", fontsize=13)
    fig.savefig(FIGURE, dpi=220)
    fig.savefig(FIGURE_PDF)
    plt.close(fig)


def main() -> None:
    _configure_topology_metadata()
    seed_index = 0
    jobs = []
    for dataset in e22.E22_DATASETS:
        for topology in TOPOLOGIES:
            jobs.append(_run_dataset_topology(dataset, topology, seed_index))
    summary = _summarize()
    _plot(summary)
    _write_json(OUTPUT / "manifest.json", {
        "protocol": PROTOCOL,
        "status": "completed",
        "seed_count": 1,
        "seed_index": seed_index,
        "datasets": list(e22.E22_DATASETS),
        "topologies": list(TOPOLOGIES),
        "constraint_values": list(CONSTRAINT_VALUES),
        "fixed_desired_replay": True,
        "metrics": ["network_admitted_power_mwh", "actual_absorption_power_mwh", "zone_matched_absorption_mwh"],
        "transformer_rule": "same physical transformer capacity; absolute loading limit varies from 1.00 to 0.50",
        "ieee123_construction": "IEEE-69 buses 1-69 plus six nine-bus distal chains attached at 25,35,45,50,65,69",
        "jobs": jobs,
        "summary": str(OUTPUT / "data/standardized_constraint_replay_summary.csv"),
        "figure": str(FIGURE),
    })
    print(json.dumps({"status": "complete", "jobs": len(jobs), "figure": str(FIGURE), "summary": str(OUTPUT / "data/standardized_constraint_replay_summary.csv")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
