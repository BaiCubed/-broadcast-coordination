from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONFIG_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ieee33_device_day_simulation"
    / "configs"
)
CASE_FILES = {
    "ieee33": CONFIG_ROOT / "network_ieee33_e22.yaml",
    "ieee69": CONFIG_ROOT / "network_ieee69_e22.yaml",
    "ieee123": CONFIG_ROOT / "network_ieee123_e24.yaml",
}
TOPOLOGIES = tuple(CASE_FILES)
CALIBRATION_CACHE = Path("results") / "_topology_calibration.json"

BASE_ZONE_SIZES = (4, 4, 4, 5, 6, 9)


def _load_case(topology: str) -> dict[str, Any]:
    path = CASE_FILES[topology]
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _children(branches: list[list[float]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for row in branches:
        children.setdefault(int(row[0]), []).append(int(row[1]))
    return children


def _subtree_totals(
    slack: int, children: dict[int, list[int]], values: dict[int, float]
) -> dict[int, float]:
    totals: dict[int, float] = {}

    def visit(bus: int) -> float:
        total = float(values.get(bus, 0.0))
        for child in children.get(bus, []):
            total += visit(child)
        totals[bus] = total
        return total

    visit(slack)
    return totals


def _depths(slack: int, children: dict[int, list[int]]) -> dict[int, int]:
    depth = {slack: 0}
    stack = [slack]
    while stack:
        bus = stack.pop()
        for child in children.get(bus, []):
            depth[child] = depth[bus] + 1
            stack.append(child)
    return depth


def derive_network_config(topology: str) -> dict[str, Any]:
    case = _load_case(topology)
    branches = [list(row) for row in case["branches"]]
    slack = int(case["slack_bus"])
    children = _children(branches)
    base_loads = case.get("base_loads", [])
    base_p = {int(row[0]): float(row[1]) for row in base_loads}
    base_q = {int(row[0]): float(row[2]) for row in base_loads}
    p_flow = _subtree_totals(slack, children, base_p)
    q_flow = _subtree_totals(slack, children, base_q)
    calibration = case["thermal_calibration"]
    reference_loading = float(calibration["reference_loading"])
    minimum_capacity = float(calibration["minimum_branch_capacity_kva"])
    rows = []
    for row in branches:
        child = int(row[1])
        reference_flow = float(np.hypot(p_flow.get(child, 0.0), q_flow.get(child, 0.0)))
        capacity = max(reference_flow / reference_loading, minimum_capacity)
        rows.append([int(row[0]), child, float(row[2]), float(row[3]), float(capacity)])
    return {
        "name": f"{case['name']}_distflow_derived",
        "source": case.get("source", ""),
        "base_kv": float(case["base_kv"]),
        "slack_bus": slack,
        "voltage_limits_pu": [float(v) for v in case["voltage_limits_pu"]],
        "transformer": {
            "bus": slack,
            "capacity_kw": float(case["transformer_capacity_kva"]),
            "warning_fraction": 0.90,
        },
        "branches": rows,
        "capacity_rule": (
            "max(hypot(P_subtree, Q_subtree) / reference_loading, "
            "minimum_branch_capacity_kva), the same rule used by the radial DistFlow model"
        ),
    }


def zone_bus_blocks(topology: str) -> list[list[int]]:
    case = _load_case(topology)
    branches = [list(row) for row in case["branches"]]
    slack = int(case["slack_bus"])
    children = _children(branches)
    depth = _depths(slack, children)
    buses = sorted(bus for bus in depth if bus != slack)
    total = len(buses)
    weight_sum = float(sum(BASE_ZONE_SIZES))
    exact = [total * size / weight_sum for size in BASE_ZONE_SIZES]
    sizes = [int(np.floor(value)) for value in exact]
    remainder = total - sum(sizes)
    order = sorted(range(len(sizes)), key=lambda i: exact[i] - sizes[i], reverse=True)
    for index in order[:remainder]:
        sizes[index] += 1
    if sum(sizes) != total or min(sizes) < 1:
        raise ValueError(f"{topology}: zone partition failed: blocks {sizes}, total {total}")
    blocks, cursor = [], 0
    for size in sizes:
        blocks.append(buses[cursor : cursor + size])
        cursor += size
    return blocks


def derive_zones_config(base_zones: dict[str, Any], topology: str) -> dict[str, Any]:
    zones = copy.deepcopy(base_zones)
    zone_ids = sorted(zones["zones"])
    if len(zone_ids) != len(BASE_ZONE_SIZES):
        raise ValueError(f"expected 6 zones, got {len(zone_ids)}")
    for zone_id, block in zip(zone_ids, zone_bus_blocks(topology)):
        zones["zones"][zone_id]["buses"] = [int(bus) for bus in block]
    return zones


def _read_cache() -> dict[str, Any]:
    if CALIBRATION_CACHE.is_file():
        return json.loads(CALIBRATION_CACHE.read_text(encoding="utf-8"))
    return {}


def calibration_key(dataset: str, topology: str) -> str:
    return f"{dataset}|{topology}"


def settings() -> tuple[str | None, bool, float]:
    topology = (os.environ.get("NC_TOPOLOGY") or "").strip() or None
    if topology is not None and topology not in CASE_FILES:
        raise ValueError(f"unknown topology {topology}; available: {TOPOLOGIES}")
    feedback = (os.environ.get("NC_NETWORK_FEEDBACK") or "0").strip() == "1"
    derate = float((os.environ.get("NC_NETWORK_DERATE") or "1").strip())
    if derate <= 0.0:
        raise ValueError("NC_NETWORK_DERATE must be positive")
    return topology, feedback, derate


def apply_topology(
    config: dict[str, Any], dataset: str, network_protocol: str
) -> tuple[dict[str, Any], str]:
    topology, feedback, derate = settings()
    if topology is None and not feedback:
        return config, network_protocol
    if topology is not None:
        config["network"] = derive_network_config(topology)
        config["zones"] = derive_zones_config(config["zones"], topology)
    config["control"] = dict(config["control"])
    config["control"]["network_feedback"] = bool(feedback)
    label = topology or "ieee33_published_config"
    if feedback:
        cache = _read_cache()
        key = calibration_key(dataset, label)
        if key not in cache:
            raise KeyError(
                f"no capacity calibration for {key}; run tools/protocols/calibrate_topology.py first"
            )
        multiplier = float(cache[key]["network_capacity_multiplier"]) * derate
        config["control"]["network_capacity_multiplier"] = multiplier
        config["control"]["network_equivalent_scale"] = multiplier
        config["control"]["network_derate_factor"] = derate
    return config, (
        f"{network_protocol}|topology={label}"
        f"|network_feedback={'on' if feedback else 'off'}"
        + (f"|derate={derate:g}" if feedback and derate != 1.0 else "")
    )
