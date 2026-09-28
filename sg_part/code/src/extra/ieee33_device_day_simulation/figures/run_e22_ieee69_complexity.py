"""Run the paired IEEE-33/IEEE-69 E22 network-complexity experiment."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.ncstyle import SEQ

from src.signal import SignalOptimizer

from ..config_loader import load_config
from ..network.local_radial_distflow import LocalRadialDistFlow, load_network_case
from ..original_adapter.eps_adapter import OriginalEPSAdapter
from ..population.data2_transaction_loader import clear_data2_transaction_pool_cache
from ..population.device_day_loader import load_device_day_pool
from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as protocol
from . import run_e20_transfer as transfer
from . import run_e21_mixed_scenarios as mixed


PROTOCOL = "E22_ieee69_network_complexity_m0_m6_v2"
DATASET_REGISTRY_VERSION = "E22_17_datasets_v2"
OUTPUT_ROOT = Path("results/E22")
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT_ROOT = CONFIG_ROOT.parent / "experiments"
NETWORK_FILES = {
    "ieee33": CONFIG_ROOT / "network_ieee33_e22.yaml",
    "ieee69": CONFIG_ROOT / "network_ieee69_e22.yaml",
}
TOPOLOGIES = ("ieee33", "ieee69")
ALGORITHM_ORDER = mixed.ALGORITHM_ORDER
SEED_COUNT = 30
FLEET_SIZE = 5000
BOOTSTRAP_DRAWS = 4000
BASE_SEED = 22_000_000
STEPS = legacy.STEPS
EPSILON = 1e-12

NEXTGEN = "nextgen_device_days"
DATA2 = "data2_charging_sessions"
E22_DATASETS = (*mixed.ALL_DATASETS, NEXTGEN, DATA2)
ADDITIONAL_DATASET_CONFIGS = {
    NEXTGEN: CONFIG_ROOT / "default.yaml",
    DATA2: (
        EXPERIMENT_ROOT / "data2_ieee33_adapted/configs/default_network_stress_data2_adapted.yaml"
    ),
}

DATASET_LABELS = {
    mixed.BDG1: "BDG1",
    mixed.BDG2: "BDG2",
    mixed.CEC: "CEC",
    mixed.DANISH: "Danish",
    mixed.EU_RURAL: "EU-Rural",
    mixed.EU_35297: "EU-35k",
    mixed.GOIENER: "GoiEner",
    mixed.HEAPO: "HEAPO",
    mixed.LCL: "LCL",
    mixed.NORWAY: "Norway",
    mixed.CAMSL: "CAMSL",
    mixed.EU_8087: "EU-8k",
    mixed.IRISH: "Irish",
    mixed.OPSD: "OPSD",
    mixed.SGSC: "SGSC",
    NEXTGEN: "NextGen",
    DATA2: "data2",
}

DATASET_DESCRIPTIONS = {
    mixed.BDG1: ("建筑", "Building Data Genome 1 建筑负荷 device-day"),
    mixed.BDG2: ("建筑", "Building Data Genome 2 建筑负荷 device-day"),
    mixed.CEC: ("综合能源", "社区综合能源与 DER device-day"),
    mixed.DANISH: ("热负荷", "丹麦智能热表 device-day"),
    mixed.EU_RURAL: ("配网/AMI", "欧洲低压农村网络 device-day"),
    mixed.EU_35297: ("配网/AMI", "欧洲低压城市 35k device-day"),
    mixed.GOIENER: ("住宅/AMI", "GoiEner 智能电表 device-day"),
    mixed.HEAPO: ("热泵", "HEAPO 热泵 device-day"),
    mixed.LCL: ("住宅", "Low Carbon London 住宅电表 device-day"),
    mixed.NORWAY: ("配网/AMI", "挪威 AMI 能量分布 device-day"),
    mixed.CAMSL: ("住宅", "CAMSL 日本住宅智能电表 device-day"),
    mixed.EU_8087: ("配网/AMI", "欧洲低压城市 8k device-day"),
    mixed.IRISH: ("住宅", "爱尔兰住宅智能电表 device-day"),
    mixed.OPSD: ("住宅/DER", "OPSD 家庭能源 device-day"),
    mixed.SGSC: ("住宅/受控负荷", "Smart Grid Smart City device-day"),
    NEXTGEN: ("住宅光储", "100 个源设备、365 天、五分钟 NextGen device-day"),
    DATA2: ("充电交易", "18,531 个 data2 transaction-session，按结束时刻聚合为五分钟负荷"),
}

_ADDITIONAL_CONFIG_CACHE: dict[str, dict[str, Any]] = {}
_ADDITIONAL_PARTITION_CACHE: dict[str, protocol.ProfilePartitions] = {}

STRESS_MODES: dict[str, dict[str, Any]] = {
    "M0": {
        "label": "Matched operating point",
        "plain_label": "常规运行",
        "load_scale": 0.45,
        "solar_ratio": 0.75,
        "wind_ratio": 0.30,
        "input_mode": "proportional",
        "capacity_multiplier": 1.00,
        "forecast_error": 0.00,
        "placement_mode": "load_weighted",
    },
    "M1": {
        "label": "Deep-feeder high load",
        "plain_label": "深馈线高负载",
        "load_scale": 0.56,
        "solar_ratio": 0.85,
        "wind_ratio": 0.35,
        "input_mode": "proportional",
        "capacity_multiplier": 0.68,
        "forecast_error": 0.00,
        "placement_mode": "load_weighted",
    },
    "M2": {
        "label": "Dual bottleneck reverse flow",
        "plain_label": "双瓶颈反向潮流",
        "load_scale": 0.48,
        "solar_ratio": 1.00,
        "wind_ratio": 0.40,
        "input_mode": "dual_distal",
        "capacity_multiplier": 0.72,
        "forecast_error": 0.00,
        "placement_mode": "load_weighted",
    },
    "M3": {
        "label": "Compound derating and forecast error",
        "plain_label": "线路降额与预测误差复合扰动",
        "load_scale": 0.56,
        "solar_ratio": 1.05,
        "wind_ratio": 0.45,
        "input_mode": "clustered",
        "capacity_multiplier": 0.65,
        "forecast_error": 0.18,
        "placement_mode": "load_weighted",
    },
    "M4": {
        "label": "Uniform device placement",
        "plain_label": "设备均匀分布",
        "load_scale": 0.45,
        "solar_ratio": 0.75,
        "wind_ratio": 0.30,
        "input_mode": "proportional",
        "capacity_multiplier": 1.00,
        "forecast_error": 0.00,
        "placement_mode": "uniform",
    },
    "M5": {
        "label": "50% feeder concentration",
        "plain_label": "50% 设备集中于远端馈线",
        "load_scale": 0.45,
        "solar_ratio": 0.75,
        "wind_ratio": 0.30,
        "input_mode": "proportional",
        "capacity_multiplier": 1.00,
        "forecast_error": 0.00,
        "placement_mode": "feeder_50",
    },
    "M6": {
        "label": "80% distal-node concentration",
        "plain_label": "80% 设备集中于远端节点",
        "load_scale": 0.45,
        "solar_ratio": 0.75,
        "wind_ratio": 0.30,
        "input_mode": "proportional",
        "capacity_multiplier": 1.00,
        "forecast_error": 0.00,
        "placement_mode": "node_80",
    },
}

PLACEMENT_DESCRIPTIONS = {
    "load_weighted": "按节点基准负荷加权分布",
    "uniform": "在全部非平衡节点尽量等量分布",
    "feeder_50": "50% 位于固定远端支路下游，50% 位于其他节点",
    "node_80": "80% 位于固定远端节点，20% 位于其他节点",
}

DERATINGS = {
    "ieee33": {"6-7": 0.65, "14-15": 0.55, "28-29": 0.60},
    "ieee69": {"11-12": 0.65, "49-50": 0.55, "60-61": 0.60},
}

DISTAL_GROUPS = {
    "ieee33": ((16, 17, 18, 21, 22), (24, 25, 30, 31, 32, 33)),
    "ieee69": ((24, 25, 26, 27, 33, 34, 35), (45, 46, 49, 50, 61, 62, 64, 65, 68, 69)),
}

ABSORPTION_ZONES = {
    "ieee33": (
        tuple(range(2, 19)),
        tuple(range(19, 23)),
        tuple(range(23, 26)),
        tuple(range(26, 34)),
    ),
    "ieee69": (
        tuple(range(2, 28)),
        tuple(range(28, 36)),
        tuple(range(36, 47)),
        tuple(range(47, 51)),
        tuple(range(51, 53)),
        tuple(range(53, 66)),
        tuple(range(66, 68)),
        tuple(range(68, 70)),
    ),
}

EXEMPLAR = {
    "dataset": mixed.CEC,
    "topology": "ieee69",
    "stress_mode": "M2",
    "algorithm": "eps_e21_mixed",
    "seed_index": 0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


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
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _acquire_lock(output: Path) -> Any | None:
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / ".run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _network_cases() -> dict[str, dict[str, Any]]:
    return {name: load_network_case(path) for name, path in NETWORK_FILES.items()}


def _model_paths(results_root: Path) -> dict[str, Path]:
    return {
        "eps_e20_pooled": results_root / "E20/models/pooled_aggregate.pt",
        "eps_e21_mixed": results_root / "E21/models/e21_mixed_aggregate.pt",
    }


def _optimizers(paths: dict[str, Path]) -> dict[str, SignalOptimizer]:
    values: dict[str, SignalOptimizer] = {}
    for algorithm, path in paths.items():
        estimator, _, _ = protocol.load_frozen_eps_controller(path)
        values[algorithm] = SignalOptimizer(
            transfer.ScaledEstimator(estimator, mixed.RESPONSE_SCALE_KW, 0.0)
        )
    return values


def _even_bus_counts(buses: list[int], total: int) -> dict[int, int]:
    if not buses:
        raise ValueError("设备部署节点不能为空")
    quotient, remainder = divmod(total, len(buses))
    return {bus: quotient + int(index < remainder) for index, bus in enumerate(sorted(buses))}


def _radial_layout(case: dict[str, Any]) -> dict[str, Any]:
    slack = int(case["slack_bus"])
    branches = [
        {"from": int(row[0]), "to": int(row[1]), "r": float(row[2]), "x": float(row[3]),}
        for row in case["branches"]
    ]
    children: dict[int, list[int]] = {}
    branch_by_child: dict[int, dict[str, Any]] = {}
    for branch in branches:
        children.setdefault(branch["from"], []).append(branch["to"])
        branch_by_child[branch["to"]] = branch
    distance = {slack: 0.0}

    def assign_distance(bus: int) -> None:
        for child in children.get(bus, []):
            branch = branch_by_child[child]
            distance[child] = distance[bus] + float(np.hypot(branch["r"], branch["x"]))
            assign_distance(child)

    def descendants(bus: int) -> set[int]:
        values = {bus}
        for child in children.get(bus, []):
            values.update(descendants(child))
        return values

    assign_distance(slack)
    buses = sorted(branch_by_child)
    return {
        "slack": slack,
        "buses": buses,
        "branches": branches,
        "distance": distance,
        "descendants": {bus: descendants(bus) for bus in buses},
    }


def _concentration_target(case: dict[str, Any]) -> dict[str, Any]:
    layout = _radial_layout(case)
    base_load = {int(row[0]): float(row[1]) for row in case["base_loads"]}
    total_load = float(sum(max(value, 0.0) for value in base_load.values()))
    candidates = []
    for branch in layout["branches"]:
        child = int(branch["to"])
        downstream = layout["descendants"][child]
        positive_load_buses = [bus for bus in downstream if base_load.get(bus, 0.0) > 0.0]
        downstream_load = float(sum(base_load.get(bus, 0.0) for bus in downstream))
        load_share = downstream_load / max(total_load, EPSILON)
        if len(positive_load_buses) >= 2 and 0.10 <= load_share <= 0.25:
            candidates.append((layout["distance"][child], child, branch, downstream, load_share))
    if not candidates:
        raise ValueError("网络中不存在下游负荷占比为 10%-25% 的远端候选支路")
    _, child, branch, downstream, load_share = max(candidates, key=lambda item: (item[0], item[1]))
    positive_downstream = [bus for bus in downstream if base_load.get(bus, 0.0) > 0.0]
    target_bus = max(positive_downstream, key=lambda bus: (layout["distance"][bus], bus),)
    return {
        "target_branch": f"{branch['from']}-{branch['to']}",
        "target_branch_child": child,
        "target_downstream_buses": tuple(sorted(downstream)),
        "target_downstream_load_share": float(load_share),
        "target_bus": int(target_bus),
        "target_bus_distance": float(layout["distance"][target_bus]),
    }


def _allocate_bus_counts(
    case: dict[str, Any], total: int, placement_mode: str = "load_weighted"
) -> tuple[dict[int, int], dict[str, Any]]:
    buses = sorted({int(row[1]) for row in case["branches"]})
    if total < len(buses):
        raise ValueError("设备数量不能少于非平衡节点数")
    target = _concentration_target(case)
    if placement_mode == "load_weighted":
        base = {int(row[0]): float(row[1]) for row in case["base_loads"]}
        positive = [value for value in base.values() if value > 0.0]
        floor = 0.02 * float(np.mean(positive))
        weights = np.asarray([max(base.get(bus, 0.0), floor) for bus in buses], dtype=float)
        weights /= float(np.sum(weights))
        counts_array = np.ones(len(buses), dtype=int)
        residual = total - len(buses)
        raw = weights * residual
        counts_array += np.floor(raw).astype(int)
        remainder = total - int(np.sum(counts_array))
        order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        counts_array[order[:remainder]] += 1
        counts = {bus: int(count) for bus, count in zip(buses, counts_array)}
        configured_fraction = None
    elif placement_mode == "uniform":
        counts = _even_bus_counts(buses, total)
        configured_fraction = None
    elif placement_mode == "feeder_50":
        downstream = list(target["target_downstream_buses"])
        outside = [bus for bus in buses if bus not in set(downstream)]
        target_total = int(round(0.50 * total))
        counts = _even_bus_counts(downstream, target_total)
        counts.update(_even_bus_counts(outside, total - target_total))
        configured_fraction = 0.50
    elif placement_mode == "node_80":
        target_bus = int(target["target_bus"])
        target_total = int(round(0.80 * total))
        counts = _even_bus_counts([bus for bus in buses if bus != target_bus], total - target_total)
        counts[target_bus] = target_total
        configured_fraction = 0.80
    else:
        raise ValueError(f"未知设备部署方式：{placement_mode}")
    if sum(counts.values()) != total or set(counts) != set(buses):
        raise RuntimeError("设备节点分配未覆盖全部设备或全部非平衡节点")
    downstream_devices = sum(counts[bus] for bus in target["target_downstream_buses"])
    metadata = {
        "placement_mode": placement_mode,
        "configured_concentration_fraction": configured_fraction,
        "actual_target_feeder_fraction": downstream_devices / total,
        "actual_target_bus_fraction": counts[int(target["target_bus"])] / total,
        "maximum_bus_fraction": max(counts.values()) / total,
        **target,
    }
    return counts, metadata


def _remap_records(
    records: list[Any], case: dict[str, Any], seed: int, placement_mode: str
) -> tuple[list[Any], dict[str, Any]]:
    counts, metadata = _allocate_bus_counts(case, len(records), placement_mode)
    assignments = np.asarray(
        [bus for bus, count in counts.items() for _ in range(count)], dtype=int
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(assignments)
    remapped = [
        replace(record, bus_id=int(assignments[index])) for index, record in enumerate(records)
    ]
    return remapped, metadata


def _profiles() -> tuple[np.ndarray, np.ndarray]:
    return protocol._pure_profiles()


def _shares(raw_by_bus: np.ndarray) -> np.ndarray:
    totals = np.sum(raw_by_bus, axis=1, keepdims=True)
    uniform = np.full_like(raw_by_bus, 1.0 / raw_by_bus.shape[1])
    return np.divide(raw_by_bus, totals, out=uniform, where=totals > EPSILON)


def _cluster_weights(buses: list[int], selected: tuple[int, ...]) -> np.ndarray:
    values = np.asarray([1.0 if bus in selected else 0.0 for bus in buses], dtype=float)
    if float(np.sum(values)) <= EPSILON:
        raise ValueError("能源输入聚集节点不在网络中")
    return values / float(np.sum(values))


def _build_scenario(
    records: list[Any],
    background_records: list[Any],
    case: dict[str, Any],
    topology: str,
    stress_mode: str,
    seed: int,
    placement: dict[str, Any],
) -> dict[str, Any]:
    settings = STRESS_MODES[stress_mode]
    buses = sorted({int(case["slack_bus"])} | {int(row[1]) for row in case["branches"]})
    bus_index = {bus: index for index, bus in enumerate(buses)}
    raw_by_bus = np.zeros((STEPS, len(buses)), dtype=float)
    for record in background_records:
        raw_by_bus[:, bus_index[int(record.bus_id)]] += np.asarray(
            record.load_kw[:STEPS], dtype=float
        )
    raw_total = np.sum(raw_by_bus, axis=1)
    hourly = np.asarray(
        [float(np.mean(raw_total[hour * 12 : (hour + 1) * 12])) for hour in range(24)]
    )
    if float(np.max(hourly)) <= EPSILON:
        raise ValueError("数据集负荷曲线没有正值")
    load_factor = np.repeat(hourly / float(np.max(hourly)), 12)
    base_peak_kw = 3800.0
    load = base_peak_kw * float(settings["load_scale"]) * load_factor
    solar, wind = _profiles()
    solar_total = base_peak_kw * float(settings["solar_ratio"]) * solar
    wind_total = base_peak_kw * float(settings["wind_ratio"]) * wind
    potential_input_total = solar_total + wind_total
    load_by_bus = _shares(raw_by_bus) * load[:, None]
    if settings["input_mode"] == "proportional":
        input_by_bus = _shares(raw_by_bus) * potential_input_total[:, None]
    else:
        first, second = DISTAL_GROUPS[topology]
        first_weights = _cluster_weights(buses, first)
        second_weights = _cluster_weights(buses, second)
        if settings["input_mode"] == "dual_distal":
            input_by_bus = (
                solar_total[:, None] * first_weights[None, :]
                + wind_total[:, None] * second_weights[None, :]
            )
        else:
            proportional = _shares(raw_by_bus)
            clustered = (
                solar_total[:, None] * second_weights[None, :]
                + wind_total[:, None] * first_weights[None, :]
            )
            input_by_bus = 0.90 * clustered + 0.10 * proportional * potential_input_total[:, None]
    local_surplus_by_bus = np.maximum(input_by_bus - load_by_bus, 0.0)
    zone_by_bus: dict[int, int] = {}
    for zone_index, zone_buses in enumerate(ABSORPTION_ZONES[topology]):
        for bus in zone_buses:
            zone_by_bus[int(bus)] = zone_index
    if set(buses) - {int(case["slack_bus"])} - set(zone_by_bus):
        raise ValueError(f"{topology} 的弃电吸收分区没有覆盖所有非平衡节点")
    absorption_zone_indices = np.asarray([zone_by_bus.get(bus, -1) for bus in buses], dtype=int)
    device_absorption_zones = np.asarray(
        [zone_by_bus[int(record.bus_id)] for record in records], dtype=int
    )
    zone_count = len(ABSORPTION_ZONES[topology])
    surplus_by_zone = np.zeros((STEPS, zone_count), dtype=float)
    for zone_index in range(zone_count):
        bus_mask = absorption_zone_indices == zone_index
        surplus_by_zone[:, zone_index] = np.sum(local_surplus_by_bus[:, bus_mask], axis=1)
    curtailment_total = np.sum(surplus_by_zone, axis=1)
    input_total = load + curtailment_total
    rng = np.random.default_rng(seed + 404)
    forecast_error = float(settings["forecast_error"])
    if forecast_error > 0.0:
        hourly_error = rng.normal(0.0, forecast_error, 24)
        error = np.repeat(hourly_error, 12)
        forecast_curtailment = np.maximum(curtailment_total * (1.0 + error), 0.0)
        forecast_input = load + forecast_curtailment
    else:
        forecast_input = input_total.copy()
    return {
        "buses": buses,
        "bus_index": bus_index,
        "device_buses": np.asarray([int(record.bus_id) for record in records], dtype=int),
        "load_by_bus": load_by_bus,
        "input_by_bus": input_by_bus,
        "load": load,
        "input_total": input_total,
        "potential_input_total": potential_input_total,
        "local_surplus_by_bus": local_surplus_by_bus,
        "surplus_by_zone": surplus_by_zone,
        "device_absorption_zones": device_absorption_zones,
        "forecast_input_total": forecast_input,
        "load_factor": load_factor,
        "base_load_peak_kw": base_peak_kw,
        "stress_mode": stress_mode,
        "placement": placement,
    }


def _network(case: dict[str, Any], topology: str, stress_mode: str) -> LocalRadialDistFlow:
    settings = STRESS_MODES[stress_mode]
    return LocalRadialDistFlow(
        case,
        capacity_multiplier=float(settings["capacity_multiplier"]),
        transformer_multiplier=float(settings["capacity_multiplier"]),
        branch_deratings=(DERATINGS[topology] if stress_mode == "M3" else None),
    )


def _maps(scenario: dict[str, Any], step: int) -> tuple[dict[int, float], dict[int, float]]:
    return (
        {
            bus: float(scenario["load_by_bus"][step, index])
            for bus, index in scenario["bus_index"].items()
        },
        {
            bus: float(scenario["input_by_bus"][step, index])
            for bus, index in scenario["bus_index"].items()
        },
    )


def _violates(evaluation: Any, network: LocalRadialDistFlow) -> bool:
    return bool(
        evaluation.overloaded_branches
        or evaluation.voltage_violations
        or evaluation.transformer_loading > 1.0 + 1e-9
    )


def _adds_violation(
    evaluation: Any,
    baseline: Any,
    network: LocalRadialDistFlow,
) -> bool:
    if any(
        evaluation.branch_loading[key]
        > max(1.0, baseline.branch_loading[key]) + 1e-9
        for key in evaluation.branch_loading
    ):
        return True
    if evaluation.transformer_loading > max(1.0, baseline.transformer_loading) + 1e-9:
        return True
    return any(
        evaluation.voltage_pu[bus]
        < min(network.vmin, baseline.voltage_pu[bus]) - 1e-9
        or evaluation.voltage_pu[bus]
        > max(network.vmax, baseline.voltage_pu[bus]) + 1e-9
        for bus in evaluation.voltage_pu
    )


def _r2(actual: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    return (
        float(1.0 - np.sum((actual - target) ** 2) / denominator)
        if denominator > EPSILON
        else float("nan")
    )


def _run_seed(
    algorithm: str,
    records: list[Any],
    config: dict[str, Any],
    scenario: dict[str, Any],
    availability_probability: np.ndarray,
    network: LocalRadialDistFlow,
    seed: int,
    availability_seed: int,
    optimizer: SignalOptimizer | None,
    record_trace: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    strategy = "eps_broadcast" if algorithm.startswith("eps_") else algorithm
    rng = np.random.default_rng(seed)
    availability_rng = np.random.default_rng(availability_seed)
    capacities = np.asarray(
        [float(record.capacity_kwh) * 0.90 * float(record.initial_soh) for record in records]
    )
    peaks = np.asarray([float(record.peak_power_kw) for record in records])
    soc = np.asarray([float(record.initial_soc) for record in records])
    eta = float(config["original_model"].get("charge_efficiency", 0.95))
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    eps_adapter = (
        OriginalEPSAdapter(records, config, seed=seed) if strategy == "eps_broadcast" else None
    )
    state = {
        "load_variability": np.asarray(
            [
                float(np.std(record.load_kw)) / max(float(np.mean(record.load_kw)), 1e-6)
                for record in records
            ]
        ),
        "availability_probability": availability_probability,
        "eps_optimizer": optimizer,
        "eps_adapter": eps_adapter,
    }
    absorbed: list[float] = []
    target_series: list[float] = []
    requested_abs: list[float] = []
    accepted_abs: list[float] = []
    network_accepted_abs: list[float] = []
    acceptance: list[float] = []
    local_clip: list[float] = []
    baseline_losses: list[float] = []
    executed_losses: list[float] = []
    min_voltage: list[float] = []
    max_voltage: list[float] = []
    max_loading: list[float] = []
    max_transformer_loading: list[float] = []
    raw_violations = 0
    requested_added_violations = 0
    executed_violations = 0
    added_violations = 0
    fallback_steps = 0
    max_soc_violation = 0.0
    trace = (
        {
            "branch_keys": [f"{row['from']}-{row['to']}" for row in network.branches],
            "branch_loading": [],
            "minimum_voltage_pu": [],
            "network_acceptance_ratio": [],
            "target_surplus_kw": [],
            "absorbed_kw": [],
        }
        if record_trace
        else None
    )
    forecast_scenario = dict(scenario)
    forecast_scenario["input_total"] = scenario["forecast_input_total"]
    for step in range(STEPS):
        available = (
            availability_rng.random(len(records))
            < np.clip(availability_probability[step], 0.0, 1.0)
        ).astype(float)
        surplus = float(np.sum(scenario["local_surplus_by_bus"][step]))
        request_scale = float(scenario.get("request_scale", 1.0))
        requested_surplus = max(surplus * request_scale, 0.0)
        target_series.append(surplus)
        if algorithm == "no_coordination":
            desired = np.zeros(len(records), dtype=float)
        else:
            dispatch_scenario = (
                forecast_scenario if stress_mode_from(scenario) == "M3" else scenario
            )
            desired = legacy._strategy_dispatch(
                strategy,
                step,
                requested_surplus,
                dispatch_scenario,
                records,
                soc,
                capacities,
                peaks,
                available,
                rng,
                state,
                config,
            )
        load_map, input_map = _maps(scenario, step)
        dispatch = network.dispatch(load_map, input_map, desired, scenario["device_buses"])
        network_accepted = dispatch.accepted_kw
        theoretical_upper_bound = strategy == "centralized_optimal"
        accepted = desired if theoretical_upper_bound else network_accepted
        if (
            strategy == "eps_broadcast"
            and str(config["control"].get("eps_control_mode", "fused")) == "fused"
        ):
            accepted = eps_adapter.apply_dispatch(accepted)
            soc = eps_adapter.battery_states()
        else:
            charge_limit = legacy._headroom(soc, capacities, peaks, config, available)
            discharge_limit = legacy._discharge_headroom(soc, capacities, peaks, config, available)
            accepted = np.minimum(np.maximum(accepted, -discharge_limit), charge_limit)
            energy = accepted * dt
            soc += np.where(
                energy >= 0.0,
                energy * eta / np.maximum(capacities, EPSILON),
                energy / (eta * np.maximum(capacities, EPSILON)),
            )
            soc = np.clip(
                soc,
                float(config["device_constraints"]["soc_min"]),
                float(config["device_constraints"]["soc_max"]),
            )
        if theoretical_upper_bound:
            absorbed_kw = float(
                min(surplus, float(np.sum(np.maximum(accepted, 0.0))))
            )
        else:
            charge_by_zone = np.bincount(
                scenario["device_absorption_zones"],
                weights=np.maximum(accepted, 0.0),
                minlength=scenario["surplus_by_zone"].shape[1],
            )
            absorbed_kw = float(
                np.sum(np.minimum(charge_by_zone, scenario["surplus_by_zone"][step]))
            )
        absorbed.append(absorbed_kw)
        requested_abs.append(float(np.sum(np.abs(desired))))
        accepted_abs.append(float(np.sum(np.abs(accepted))))
        network_accepted_abs.append(float(np.sum(np.abs(network_accepted))))
        acceptance.append(dispatch.acceptance_ratio)
        local_clip.append(dispatch.local_clip_fraction)
        baseline_losses.append(dispatch.baseline.loss_kw)
        executed_losses.append(dispatch.executed.loss_kw)
        min_voltage.append(min(dispatch.executed.voltage_pu.values()))
        max_voltage.append(max(dispatch.executed.voltage_pu.values()))
        max_loading.append(max(dispatch.executed.branch_loading.values(), default=0.0))
        max_transformer_loading.append(dispatch.executed.transformer_loading)
        raw_violations += int(_violates(dispatch.requested, network))
        requested_added_violations += int(
            _adds_violation(dispatch.requested, dispatch.baseline, network)
        )
        executed_violations += int(_violates(dispatch.executed, network))
        added_violations += int(
            _adds_violation(dispatch.executed, dispatch.baseline, network)
        )
        fallback_steps += int(dispatch.fallback_global)
        soc_min = float(config["device_constraints"]["soc_min"])
        soc_max = float(config["device_constraints"]["soc_max"])
        max_soc_violation = max(
            max_soc_violation,
            float(np.max(np.maximum(soc - soc_max, 0.0) + np.maximum(soc_min - soc, 0.0))),
        )
        if strategy == "eps_broadcast":
            step_target = float(state.get("eps_step_target_kw", 0.0))
            old = float(state.get("eps_intensity_correction", 0.0))
            shortfall = (
                max(step_target - absorbed_kw, 0.0) / step_target if step_target > EPSILON else 0.0
            )
            state["eps_intensity_correction"] = float(
                np.clip(0.80 * old + 0.30 * shortfall, 0.0, 0.45)
            )
        if trace is not None:
            trace["branch_loading"].append(
                [dispatch.executed.branch_loading[key] for key in trace["branch_keys"]]
            )
            trace["minimum_voltage_pu"].append(min_voltage[-1])
            trace["network_acceptance_ratio"].append(acceptance[-1])
            trace["target_surplus_kw"].append(surplus)
            trace["absorbed_kw"].append(absorbed_kw)
        target = np.asarray(target_series)
    actual = np.asarray(absorbed)
    baseline_mwh = float(np.sum(target) * dt / 1000.0)
    accepted_mwh = float(np.sum(actual) * dt / 1000.0)
    remaining_mwh = max(baseline_mwh - accepted_mwh, 0.0)
    reduction = 100.0 * accepted_mwh / max(baseline_mwh, EPSILON)
    nrmse = float(
        np.sqrt(np.mean((actual - target) ** 2))
        / max(float(np.mean(target[target > 0.0])) if np.any(target > 0.0) else 0.0, EPSILON)
    )
    result = {
        "mean_reduction_pct": float(np.clip(reduction, 0.0, 100.0)),
        "baseline_curtailment_mwh": baseline_mwh,
        "remaining_curtailment_mwh": remaining_mwh,
        "accepted_absorption_mwh": accepted_mwh,
        "network_acceptance_ratio": (
            float(np.sum(network_accepted_abs) / np.sum(requested_abs))
            if float(np.sum(requested_abs)) > EPSILON
            else 1.0
        ),
        "mean_step_acceptance_ratio": float(np.mean(acceptance)),
        "mean_local_clip_fraction": float(np.mean(local_clip)),
        "requested_violation_steps": int(raw_violations),
        "executed_violation_steps": int(executed_violations),
        "added_violation_steps": int(added_violations),
        "requested_violation_free_pct": 100.0 * (STEPS - raw_violations) / STEPS,
        "requested_added_violation_steps": int(requested_added_violations),
        "requested_added_violation_free_pct": 100.0
        * (STEPS - requested_added_violations)
        / STEPS,
        "executed_violation_free_pct": 100.0 * (STEPS - executed_violations) / STEPS,
        "fallback_global_steps": int(fallback_steps),
        "minimum_voltage_pu": float(np.min(min_voltage)),
        "maximum_voltage_pu": float(np.max(max_voltage)),
        "maximum_branch_loading": float(np.max(max_loading)),
        "maximum_transformer_loading": float(np.max(max_transformer_loading)),
        "mean_baseline_loss_kw": float(np.mean(baseline_losses)),
        "mean_executed_loss_kw": float(np.mean(executed_losses)),
        "response_r2": _r2(actual, target),
        "response_nrmse": nrmse,
        "availability_fraction": float(np.mean(availability_probability)),
        "max_soc_violation": float(max_soc_violation),
        "theoretical_upper_bound": bool(theoretical_upper_bound),
        "upper_bound_network_ignored": bool(theoretical_upper_bound),
    }
    return result, trace


def stress_mode_from(scenario: dict[str, Any]) -> str:
    return str(scenario["stress_mode"])


def _additional_config(dataset: str) -> dict[str, Any]:
    if dataset not in ADDITIONAL_DATASET_CONFIGS:
        raise KeyError(f"E22 未注册附加数据集：{dataset}")
    if dataset not in _ADDITIONAL_CONFIG_CACHE:
        original = load_config(ADDITIONAL_DATASET_CONFIGS[dataset])
        _ADDITIONAL_CONFIG_CACHE[dataset] = protocol._condition_config(
            original, "aggregate", mixed.AVAILABILITY_MODE
        )
    return _ADDITIONAL_CONFIG_CACHE[dataset]


def _additional_partition(dataset: str) -> protocol.ProfilePartitions:
    if dataset not in _ADDITIONAL_PARTITION_CACHE:
        config = _additional_config(dataset)
        pool = load_device_day_pool(config)
        _ADDITIONAL_PARTITION_CACHE[dataset] = protocol.partition_profile_pool(
            pool, int(config["simulation"]["random_seed"])
        )
    return _ADDITIONAL_PARTITION_CACHE[dataset]


def _dataset_source(dataset: str) -> tuple[dict[str, Any], protocol.ProfilePartitions]:
    if dataset in mixed.ALL_DATASETS:
        return mixed._config(dataset, "aggregate"), mixed._partitions(dataset, "aggregate")
    return _additional_config(dataset), _additional_partition(dataset)


def _sample_dataset(dataset: str, seed: int) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    source_config, partitions = _dataset_source(dataset)
    config = copy.deepcopy(source_config)
    partition = partitions.test
    records, audit = protocol.sample_device_fleet(
        partition, "fixed5000", FLEET_SIZE, seed, dataset_id=dataset,
    )
    if dataset == NEXTGEN:
        audit.update(
            {
                "source_adapter": "nextgen_device_day",
                "source_role": "measured five-minute load shape and data-driven availability",
                "external_input_role": "E22 M0-M6 common renewable input; measured PV is not injected directly",
            }
        )
    elif dataset == DATA2:
        audit.update(
            {
                "source_adapter": "data2_transaction_session_observed_demand",
                "source_role": "observed out_power binned by end_time as load shape and availability evidence",
                "external_input_role": "E22 M0-M6 common renewable input; out_power is not treated as supply",
            }
        )
    config["control"] = dict(config["control"])
    config["control"]["pure_sim_device_model"] = True
    config["control"]["network_feedback"] = False
    config["original_model"] = dict(config["original_model"])
    config["original_model"]["eps_packet_loss_rate"] = 0.001
    config["original_model"]["device_offline_rate"] = 0.0
    return records, config, audit


def _base_seed(dataset: str, seed_index: int) -> int:
    return BASE_SEED + _stable_seed(dataset) % 100_000 + seed_index


def _run_dataset(
    dataset: str,
    output: Path,
    cases: dict[str, dict[str, Any]],
    optimizers: dict[str, SignalOptimizer],
    model_hashes: dict[str, str],
    seed_count: int,
    force: bool,
    topologies: tuple[str, ...] = TOPOLOGIES,
    stress_modes: tuple[str, ...] = tuple(STRESS_MODES),
) -> dict[str, Any]:
    raw_path = output / "raw" / f"{dataset}.json"
    payload: dict[str, Any] = {}
    if raw_path.is_file() and not force:
        candidate = json.loads(raw_path.read_text(encoding="utf-8"))
        if (
            candidate.get("protocol") == PROTOCOL
            and candidate.get("model_sha256") == model_hashes
            and int(candidate.get("seed_count", -1)) == seed_count
        ):
            payload = candidate
    rows = list(payload.get("seed_results", []))
    completed = {
        (
            int(row["seed_index"]),
            str(row["topology"]),
            str(row["stress_mode"]),
            str(row["algorithm"]),
        )
        for row in rows
    }
    fleet_audits = {int(row["seed_index"]): row for row in payload.get("fleet_audit", [])}
    for seed_index in range(seed_count):
        base_seed = _base_seed(dataset, seed_index)
        base_records, config, audit = _sample_dataset(dataset, base_seed)
        fleet_audits[seed_index] = {"seed_index": seed_index, "seed": base_seed, **audit}
        availability = protocol.availability_probability(
            base_records, config, mixed.AVAILABILITY_MODE
        )
        for topology_index, topology in enumerate(topologies):
            placement_seed = base_seed + 10_000 * (topology_index + 1)
            reference_records, _ = _remap_records(
                base_records, cases[topology], placement_seed, "load_weighted",
            )
            placement_cache: dict[str, tuple[list[Any], dict[str, Any]]] = {}
            for stress_index, stress_mode in enumerate(stress_modes):
                placement_mode = str(STRESS_MODES[stress_mode]["placement_mode"])
                if placement_mode not in placement_cache:
                    placement_cache[placement_mode] = _remap_records(
                        base_records, cases[topology], placement_seed, placement_mode,
                    )
                records, placement = placement_cache[placement_mode]
                scenario = _build_scenario(
                    records,
                    reference_records,
                    cases[topology],
                    topology,
                    stress_mode,
                    base_seed,
                    placement,
                )
                network = _network(cases[topology], topology, stress_mode)
                missing = [
                    algorithm
                    for algorithm in ALGORITHM_ORDER
                    if (seed_index, topology, stress_mode, algorithm) not in completed
                ]
                for algorithm in missing:
                    algorithm_index = ALGORITHM_ORDER.index(algorithm)
                    record_trace = (
                        dataset == EXEMPLAR["dataset"]
                        and topology == EXEMPLAR["topology"]
                        and stress_mode == EXEMPLAR["stress_mode"]
                        and algorithm == EXEMPLAR["algorithm"]
                        and seed_index == EXEMPLAR["seed_index"]
                    )
                    started = time.perf_counter()
                    result, trace = _run_seed(
                        algorithm,
                        records,
                        config,
                        scenario,
                        availability,
                        network,
                        base_seed + 100_000 * (algorithm_index + 1),
                        base_seed + 1_910_000,
                        optimizers.get(algorithm),
                        record_trace,
                    )
                    runtime = time.perf_counter() - started
                    row = {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "topology": topology,
                        "stress_mode": stress_mode,
                        "stress_label": STRESS_MODES[stress_mode]["label"],
                        "algorithm": algorithm,
                        "algorithm_label": mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                        "complexity": mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"],
                        "seed_index": seed_index,
                        "seed": base_seed,
                        "fleet_size": len(records),
                        "placement_mode": placement["placement_mode"],
                        "target_branch": placement["target_branch"],
                        "target_bus": placement["target_bus"],
                        "target_downstream_load_share": placement["target_downstream_load_share"],
                        "configured_concentration_fraction": placement[
                            "configured_concentration_fraction"
                        ],
                        "actual_target_feeder_fraction": placement["actual_target_feeder_fraction"],
                        "actual_target_bus_fraction": placement["actual_target_bus_fraction"],
                        "maximum_bus_fraction": placement["maximum_bus_fraction"],
                        "runtime_seconds": runtime,
                        **result,
                    }
                    rows.append(row)
                    completed.add((seed_index, topology, stress_mode, algorithm))
                    if trace is not None:
                        _write_json(
                            output / "data" / "ieee69_m2_eps_trace.json",
                            {
                                "protocol": PROTOCOL,
                                "dataset": dataset,
                                "dataset_label": DATASET_LABELS[dataset],
                                "topology": topology,
                                "stress_mode": stress_mode,
                                "algorithm": algorithm,
                                "seed_index": seed_index,
                                **trace,
                            },
                        )
                condition_rows = [
                    row
                    for row in rows
                    if int(row["seed_index"]) == seed_index
                    and row["topology"] == topology
                    and row["stress_mode"] == stress_mode
                ]
                baselines = [float(row["baseline_curtailment_mwh"]) for row in condition_rows]
                if baselines and max(baselines) - min(baselines) > 1e-8:
                    raise RuntimeError("同一成对条件下各算法的基线弃电量不一致")
                if any(float(row["max_soc_violation"]) > 1e-9 for row in condition_rows):
                    raise RuntimeError("设备 SOC 出现越界")
                print(
                    json.dumps(
                        {
                            "stage": "condition_checkpoint",
                            "dataset": dataset,
                            "seed_index": seed_index,
                            "topology": topology,
                            "stress_mode": stress_mode,
                            "algorithm_count": len(condition_rows),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        payload = {
            "protocol": PROTOCOL,
            "dataset_registry_version": DATASET_REGISTRY_VERSION,
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "seed_count": seed_count,
            "fleet_size": FLEET_SIZE,
            "model_sha256": model_hashes,
            "seed_results": sorted(
                rows,
                key=lambda row: (
                    int(row["seed_index"]),
                    TOPOLOGIES.index(row["topology"]),
                    list(STRESS_MODES).index(row["stress_mode"]),
                    ALGORITHM_ORDER.index(row["algorithm"]),
                ),
            ),
            "fleet_audit": [fleet_audits[index] for index in sorted(fleet_audits)],
            "updated_at": _utc_now(),
        }
        _write_json(raw_path, payload)
        print(
            json.dumps(
                {
                    "stage": "seed_checkpoint",
                    "dataset": dataset,
                    "completed_seeds": seed_index + 1,
                    "seed_count": seed_count,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return payload


def _bootstrap_reduction(rows: list[dict[str, Any]], draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    baseline = np.asarray([float(row["baseline_curtailment_mwh"]) for row in rows])
    remaining = np.asarray([float(row["remaining_curtailment_mwh"]) for row in rows])
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(rows), len(rows))
        denominator = float(np.sum(baseline[indices]))
        values.append(
            100.0 * (denominator - float(np.sum(remaining[indices]))) / denominator
            if denominator > EPSILON
            else float("nan")
        )
    return tuple(np.nanpercentile(values, [2.5, 97.5]))


def _summaries(rows: list[dict[str, Any]], draws: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["topology"], row["stress_mode"], row["algorithm"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    metric_means = (
        "network_acceptance_ratio",
        "requested_violation_free_pct",
        "executed_violation_free_pct",
        "minimum_voltage_pu",
        "maximum_voltage_pu",
        "maximum_branch_loading",
        "maximum_transformer_loading",
        "mean_baseline_loss_kw",
        "mean_executed_loss_kw",
        "response_r2",
        "response_nrmse",
        "runtime_seconds",
        "fallback_global_steps",
    )
    for key, group in sorted(grouped.items()):
        dataset, topology, stress_mode, algorithm = key
        baseline = float(sum(float(row["baseline_curtailment_mwh"]) for row in group))
        remaining = float(sum(float(row["remaining_curtailment_mwh"]) for row in group))
        reduction = (
            100.0 * (baseline - remaining) / baseline if baseline > EPSILON else float("nan")
        )
        low, high = _bootstrap_reduction(
            group, draws, BASE_SEED + _stable_seed("|".join(key)) % 1_000_000
        )
        summary = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "topology": topology,
            "stress_mode": stress_mode,
            "stress_label": STRESS_MODES[stress_mode]["label"],
            "algorithm": algorithm,
            "algorithm_label": mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
            "complexity": mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"],
            "seed_count": len(group),
            "curtailment_reduction_pct": reduction,
            "curtailment_reduction_ci_low": low,
            "curtailment_reduction_ci_high": high,
            "baseline_curtailment_mwh_mean": float(
                np.mean([row["baseline_curtailment_mwh"] for row in group])
            ),
            "remaining_curtailment_mwh_mean": float(
                np.mean([row["remaining_curtailment_mwh"] for row in group])
            ),
            "placement_mode": str(group[0]["placement_mode"]),
            "target_branch": str(group[0]["target_branch"]),
            "target_bus": int(group[0]["target_bus"]),
            "target_downstream_load_share": float(group[0]["target_downstream_load_share"]),
            "configured_concentration_fraction": group[0]["configured_concentration_fraction"],
            "actual_target_feeder_fraction": float(group[0]["actual_target_feeder_fraction"]),
            "actual_target_bus_fraction": float(group[0]["actual_target_bus_fraction"]),
            "maximum_bus_fraction": float(group[0]["maximum_bus_fraction"]),
        }
        for metric in metric_means:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
        summaries.append(summary)
    return summaries


def _overall(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault((row["topology"], row["stress_mode"], row["algorithm"]), []).append(row)
    results = []
    for (topology, stress_mode, algorithm), group in sorted(grouped.items()):
        baseline = float(sum(row["baseline_curtailment_mwh_mean"] for row in group))
        remaining = float(sum(row["remaining_curtailment_mwh_mean"] for row in group))
        results.append(
            {
                "topology": topology,
                "stress_mode": stress_mode,
                "stress_label": STRESS_MODES[stress_mode]["label"],
                "algorithm": algorithm,
                "algorithm_label": mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                "complexity": mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"],
                "dataset_count": len(group),
                "curtailment_reduction_pct": (
                    100.0 * (baseline - remaining) / baseline
                    if baseline > EPSILON
                    else float("nan")
                ),
                "network_acceptance_ratio": float(
                    np.mean([row["network_acceptance_ratio_mean"] for row in group])
                ),
                "requested_violation_free_pct": float(
                    np.mean([row["requested_violation_free_pct_mean"] for row in group])
                ),
                "executed_violation_free_pct": float(
                    np.mean([row["executed_violation_free_pct_mean"] for row in group])
                ),
                "response_r2": float(np.nanmean([row["response_r2_mean"] for row in group])),
                "response_nrmse": float(np.mean([row["response_nrmse_mean"] for row in group])),
                "runtime_seconds": float(np.mean([row["runtime_seconds_mean"] for row in group])),
            }
        )
    return results


def _retention(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["topology"], row["stress_mode"], row["algorithm"]): row
        for row in summaries
    }
    rows = []
    for dataset in E22_DATASETS:
        for stress_mode in STRESS_MODES:
            for algorithm in ALGORITHM_ORDER:
                ieee33 = lookup[(dataset, "ieee33", stress_mode, algorithm)]
                ieee69 = lookup[(dataset, "ieee69", stress_mode, algorithm)]
                denominator = float(ieee33["curtailment_reduction_pct"])
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "stress_mode": stress_mode,
                        "algorithm": algorithm,
                        "algorithm_label": mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                        "ieee33_reduction_pct": denominator,
                        "ieee69_reduction_pct": float(ieee69["curtailment_reduction_pct"]),
                        "retention_ratio": (
                            float(ieee69["curtailment_reduction_pct"]) / denominator
                            if denominator > EPSILON
                            else float("nan")
                        ),
                        "change_percentage_points": float(ieee69["curtailment_reduction_pct"])
                        - denominator,
                    }
                )
    return rows


def _spatial_retention(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["topology"], row["stress_mode"], row["algorithm"]): row
        for row in summaries
    }
    rows = []
    for dataset in E22_DATASETS:
        for topology in TOPOLOGIES:
            for algorithm in ALGORITHM_ORDER:
                reference = lookup[(dataset, topology, "M4", algorithm)]
                denominator = float(reference["curtailment_reduction_pct"])
                for stress_mode in ("M4", "M5", "M6"):
                    concentrated = lookup[(dataset, topology, stress_mode, algorithm)]
                    value = float(concentrated["curtailment_reduction_pct"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": DATASET_LABELS[dataset],
                            "topology": topology,
                            "stress_mode": stress_mode,
                            "stress_label": STRESS_MODES[stress_mode]["label"],
                            "algorithm": algorithm,
                            "algorithm_label": mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                            "uniform_m4_reduction_pct": denominator,
                            "concentrated_reduction_pct": value,
                            "spatial_retention_ratio": (
                                value / denominator if denominator > EPSILON else float("nan")
                            ),
                            "change_percentage_points": value - denominator,
                        }
                    )
    return rows


def _save_figure(fig: Any, figure_dir: Path, name: str) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = figure_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _plot_dumbbell(overall: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    values: dict[tuple[str, str], float] = {}
    for topology in TOPOLOGIES:
        for algorithm in ALGORITHM_ORDER:
            selected = [
                row
                for row in overall
                if row["topology"] == topology and row["algorithm"] == algorithm
            ]
            values[(topology, algorithm)] = float(
                np.mean([row["curtailment_reduction_pct"] for row in selected])
            )
    fig, axis = plt.subplots(figsize=(9.2, 6.8))
    y = np.arange(len(ALGORITHM_ORDER))
    for index, algorithm in enumerate(ALGORITHM_ORDER):
        first = values[("ieee33", algorithm)]
        second = values[("ieee69", algorithm)]
        axis.plot([first, second], [index, index], color="#a0a0a0", linewidth=2)
        axis.scatter(first, index, color="#4e79a7", s=65, zorder=3)
        axis.scatter(second, index, color="#e15759", s=65, zorder=3)
    axis.set_yticks(
        y,
        [
            f"{mixed.ALGORITHM_DEFINITIONS[a]['label']}  {mixed.ALGORITHM_DEFINITIONS[a]['complexity']}"
            for a in ALGORITHM_ORDER
        ],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Deliverable curtailment reduction (%)")
    axis.set_title("Paired effect of feeder complexity")
    axis.grid(axis="x", alpha=0.25)
    axis.scatter([], [], color="#4e79a7", label="IEEE-33")
    axis.scatter([], [], color="#e15759", label="IEEE-69")
    axis.legend(loc="lower right")
    return _save_figure(fig, figure_dir, "topology_effect_dumbbell")


def _plot_pareto(overall: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    selected = [row for row in overall if row["topology"] == "ieee69"]
    column_count = 4
    row_count = int(np.ceil(len(STRESS_MODES) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(16.0, 4.5 * row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, stress_mode in zip(axes.flat, STRESS_MODES):
        rows = [row for row in selected if row["stress_mode"] == stress_mode]
        for row in rows:
            algorithm = row["algorithm"]
            axis.scatter(
                row["curtailment_reduction_pct"],
                row["requested_violation_free_pct"],
                s=55,
                color=mixed.ALGORITHM_COLORS.get(algorithm, "#777777"),
            )
        axis.set_title(f"{stress_mode}: {STRESS_MODES[stress_mode]['label']}")
        axis.grid(alpha=0.22)
    legend_axis = axes.flat[len(STRESS_MODES)]
    legend_axis.axis("off")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=7,
            color=mixed.ALGORITHM_COLORS.get(algorithm, "#777777"),
            label=(
                f"{mixed.ALGORITHM_DEFINITIONS[algorithm]['label']} "
                f"({mixed.ALGORITHM_DEFINITIONS[algorithm]['complexity']})"
            ),
        )
        for algorithm in ALGORITHM_ORDER
    ]
    legend_axis.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=7.4,
        title="Algorithms",
        title_fontsize=8.5,
    )
    fig.supxlabel("Deliverable curtailment reduction (%)")
    fig.supylabel("Violation-free requested dispatch (%)")
    return _save_figure(fig, figure_dir, "ieee69_effect_safety_pareto")


def _plot_retention(retention: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    matrix = np.full((len(E22_DATASETS), len(ALGORITHM_ORDER)), np.nan)
    for row_index, dataset in enumerate(E22_DATASETS):
        for column_index, algorithm in enumerate(ALGORITHM_ORDER):
            values = [
                row["retention_ratio"]
                for row in retention
                if row["dataset"] == dataset and row["algorithm"] == algorithm
            ]
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            matrix[row_index, column_index] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
    fig, axis = plt.subplots(figsize=(12.5, 7.0))
    image = axis.imshow(matrix, cmap=SEQ, aspect="auto", vmin=0.0, vmax=1.2)
    axis.set_xticks(
        np.arange(len(ALGORITHM_ORDER)),
        [mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHM_ORDER],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(np.arange(len(E22_DATASETS)), [DATASET_LABELS[d] for d in E22_DATASETS])
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Retention ratio = IEEE-69 effect / IEEE-33 effect")
    return _save_figure(fig, figure_dir, "retention_by_dataset_algorithm")


def _plot_stress(overall: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    fig, axis = plt.subplots(figsize=(15.0, 7.0))
    x = np.arange(len(STRESS_MODES))
    for algorithm in ALGORITHM_ORDER:
        values = [
            next(
                row["curtailment_reduction_pct"]
                for row in overall
                if row["topology"] == "ieee69"
                and row["stress_mode"] == stress_mode
                and row["algorithm"] == algorithm
            )
            for stress_mode in STRESS_MODES
        ]
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=1.7,
            color=mixed.ALGORITHM_COLORS.get(algorithm, "#777777"),
            label=f"{mixed.ALGORITHM_DEFINITIONS[algorithm]['label']} ({mixed.ALGORITHM_DEFINITIONS[algorithm]['complexity']})",
        )
    axis.set_xticks(x, [f"{key}\n{value['label']}" for key, value in STRESS_MODES.items()])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Deliverable curtailment reduction (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7, ncol=2, loc="upper right")
    return _save_figure(fig, figure_dir, "ieee69_stress_profiles")


def _plot_acceptance(summaries: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    matrix = np.zeros((len(ALGORITHM_ORDER), len(STRESS_MODES)))
    for row_index, algorithm in enumerate(ALGORITHM_ORDER):
        for column_index, stress_mode in enumerate(STRESS_MODES):
            selected = [
                row["network_acceptance_ratio_mean"]
                for row in summaries
                if row["topology"] == "ieee69"
                and row["algorithm"] == algorithm
                and row["stress_mode"] == stress_mode
            ]
            matrix[row_index, column_index] = float(np.mean(selected))
    fig, axis = plt.subplots(figsize=(11.0, 6.4))
    image = axis.imshow(matrix, cmap=SEQ, aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(STRESS_MODES)), list(STRESS_MODES))
    axis.set_yticks(
        np.arange(len(ALGORITHM_ORDER)),
        [mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHM_ORDER],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{100 * matrix[row, column]:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Network acceptance ratio")
    return _save_figure(fig, figure_dir, "ieee69_network_acceptance")


def _plot_constraint_audit(summaries: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    selected = [row for row in summaries if row["topology"] == "ieee69"]
    specifications = (
        (
            "maximum_branch_loading_mean",
            "Executed maximum branch loading",
            "YlOrRd",
            0.0,
            1.5,
            "{:.2f}",
        ),
        (
            "minimum_voltage_pu_mean",
            "Executed minimum voltage (p.u.)",
            "RdYlGn",
            0.94,
            1.00,
            "{:.3f}",
        ),
        (
            "maximum_transformer_loading_mean",
            "Executed maximum transformer loading",
            "YlOrRd",
            0.0,
            1.0,
            "{:.2f}",
        ),
        (
            "safety_layer_gain",
            "Safety-layer gain in violation-free steps (pp)",
            "YlGnBu",
            0.0,
            100.0,
            "{:.1f}",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.8), sharex=True, sharey=True)
    for panel_index, (metric, title, cmap, vmin, vmax, formatter) in enumerate(specifications):
        axis = axes.flat[panel_index]
        matrix = np.zeros((len(ALGORITHM_ORDER), len(STRESS_MODES)))
        for row_index, algorithm in enumerate(ALGORITHM_ORDER):
            for column_index, stress_mode in enumerate(STRESS_MODES):
                rows = [
                    row
                    for row in selected
                    if row["algorithm"] == algorithm and row["stress_mode"] == stress_mode
                ]
                if metric == "safety_layer_gain":
                    values = [
                        float(row["executed_violation_free_pct_mean"])
                        - float(row["requested_violation_free_pct_mean"])
                        for row in rows
                    ]
                else:
                    values = [float(row[metric]) for row in rows]
                matrix[row_index, column_index] = float(np.nanmean(values))
        image = axis.imshow(matrix, cmap=SEQ, aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=10.5)
        axis.set_xticks(np.arange(len(STRESS_MODES)), list(STRESS_MODES))
        if panel_index % 2 == 0:
            axis.set_yticks(
                np.arange(len(ALGORITHM_ORDER)),
                [mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHM_ORDER],
            )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                normalized = (value - vmin) / max(vmax - vmin, EPSILON)
                axis.text(
                    column_index,
                    row_index,
                    formatter.format(value),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if normalized > 0.62 else "black",
                )
        fig.colorbar(image, ax=axis, pad=0.015, fraction=0.035)
    fig.supxlabel("Network and spatial stress mode")
    fig.supylabel("Algorithm")
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.95))
    return _save_figure(fig, figure_dir, "ieee69_constraint_component_audit")


def _plot_spatial_retention(spatial_retention: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), sharey=True)
    for axis, topology in zip(axes, TOPOLOGIES):
        matrix = np.full((len(ALGORITHM_ORDER), 2), np.nan)
        for row_index, algorithm in enumerate(ALGORITHM_ORDER):
            for column_index, stress_mode in enumerate(("M5", "M6")):
                values = np.asarray(
                    [
                        row["spatial_retention_ratio"]
                        for row in spatial_retention
                        if row["topology"] == topology
                        and row["algorithm"] == algorithm
                        and row["stress_mode"] == stress_mode
                    ],
                    dtype=float,
                )
                matrix[row_index, column_index] = float(np.nanmean(values))
        image = axis.imshow(matrix, cmap=SEQ, aspect="auto", vmin=0.0, vmax=1.2)
        axis.set_xticks((0, 1), ("M5: 50% feeder", "M6: 80% node"))
        axis.set_yticks(
            np.arange(len(ALGORITHM_ORDER)),
            [mixed.ALGORITHM_DEFINITIONS[algorithm]["label"] for algorithm in ALGORITHM_ORDER],
        )
        axis.set_title(topology.upper())
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7,
                )
    colorbar = fig.colorbar(image, ax=axes, pad=0.02, shrink=0.9)
    colorbar.set_label("Spatial retention ratio relative to M4")
    return _save_figure(fig, figure_dir, "spatial_concentration_retention")


def _plot_trace(output: Path, figure_dir: Path) -> list[str]:
    path = output / "data" / "ieee69_m2_eps_trace.json"
    if not path.is_file():
        return []
    trace = json.loads(path.read_text(encoding="utf-8"))
    loading = np.asarray(trace["branch_loading"], dtype=float).T
    fig, axis = plt.subplots(figsize=(12.0, 8.5))
    image = axis.imshow(
        loading,
        aspect="auto",
        cmap=SEQ,
        vmin=0.0,
        vmax=max(1.0, float(np.nanpercentile(loading, 99))),
    )
    axis.set_xlabel("Five-minute time step")
    axis.set_ylabel("IEEE-69 branch index")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Branch loading (1.0 = thermal limit)")
    return _save_figure(fig, figure_dir, "ieee69_branch_time_heatmap")


def _write_design(output: Path, seed_count: int, model_paths: dict[str, Path]) -> None:
    algorithms = "\n".join(
        f"- {mixed.ALGORITHM_DEFINITIONS[a]['label']}：{mixed.ALGORITHM_DEFINITIONS[a]['complexity']}。{mixed.ALGORITHM_DEFINITIONS[a]['description']}"
        for a in ALGORITHM_ORDER
    )
    stress = "\n".join(
        f"- **{key} {value['plain_label']}**：负荷尺度 {value['load_scale']:.2f}，线路容量尺度 {value['capacity_multiplier']:.2f}，能源输入方式 `{value['input_mode']}`；设备部署为{PLACEMENT_DESCRIPTIONS[value['placement_mode']]}。"
        for key, value in STRESS_MODES.items()
    )
    datasets = "\n".join(
        f"| {DATASET_LABELS[dataset]} | {DATASET_DESCRIPTIONS[dataset][0]} | {DATASET_DESCRIPTIONS[dataset][1]} |"
        for dataset in E22_DATASETS
    )
    result_rows = (
        len(E22_DATASETS) * len(TOPOLOGIES) * len(STRESS_MODES) * seed_count * len(ALGORITHM_ORDER)
    )
    targets = "\n".join(
        f"| {topology.upper()} | `{target['target_branch']}` | {target['target_bus']} | {100.0 * target['target_downstream_load_share']:.2f}% |"
        for topology, target in (
            (name, _concentration_target(case)) for name, case in _network_cases().items()
        )
    )
    text = f"""# E22 IEEE-69 复杂配电网实验设计

## 实验要回答的问题

IEEE-69 不是简单把节点数从 33 增加到 69。本实验保持设备、数据、时间和随机种子不变，分别改变馈线拓扑、网络压力和设备空间部署，检验算法提出的响应有多少能够在配网约束下安全交付。

## 与 Reviewer #4 意见的对应关系

Reviewer #4 指出，原聚合仿真没有充分考虑馈线功率上限、电压约束、变压器负载以及设备空间集中响应，因此统计上可控的聚合响应不一定能在配电网中安全执行。E22 不把聚合响应直接等同于可交付响应，而是在算法输出之后增加显式径向配电网校核，并同时保存“算法原始请求”和“网络约束后执行”两套结果。

| Reviewer #4 关注点 | E22 实验处理 | 主要判定指标 |
|---|---|---|
| 馈线功率上限 | 为 IEEE-33 和 IEEE-69 的每条支路设置冻结热限；M1-M3 施加容量收紧和固定支路降额 | 最大支路负载率、过载支路数、请求/执行无违规比例 |
| 电压约束 | 使用含有功和无功的 LinDistFlow，在每个五分钟步检查全部节点的 0.95-1.05 pu 电压范围 | 最低节点电压、电压违规数、请求/执行无违规比例 |
| 变压器负载 | 在馈线入口设置冻结的 6 MVA 主变容量，并在原始请求和安全执行后分别检查负载率 | 主变负载率、请求/执行无违规比例 |
| 空间集中响应 | M4 将 5000 台设备均匀分配到全部非平衡节点；M5 将 50% 集中到预先固定的远端支路；M6 将 80% 集中到该支路最远负荷节点 | 相对 M4 的空间保持率、网络接受率、请求无违规比例、可交付弃电降低率 |
| 同一广播下大量设备同步响应 | 所有算法在同一设备、可用率、能源输入和随机扰动下提出响应，先检查未经修正的原始请求，再按违规支路下游范围局部限幅 | 原始请求违规率与执行违规率之差、网络接受率、R²、NRMSE |
| 结论仍来自仿真 | 用 {len(E22_DATASETS)} 个真实数据集驱动设备时序和可用性，并在两个标准测试馈线上进行 {seed_count} 个 paired seeds；不把结果表述为现场验证 | 跨数据集置信区间、IEEE-33/69 成对保持率、结果适用边界 |

该设计区分两个问题：聚合层算法“希望设备响应多少”，以及配电网“允许设备安全交付多少”。只有通过电池约束、支路热限、电压范围和主变容量校核后的响应，才计入可交付弃电降低率。这样可以避免用安全层删除违规请求后，仍把最终安全性归因于算法本身。

## 针对审稿意见的检验假设

- **H1，网络复杂度效应**：在设备和随机种子相同的成对条件下，IEEE-69 的更长供电路径和更多局部瓶颈会改变网络接受率及可交付效果；差异由成对保持率衡量。
- **H2，空间集中效应**：M4-M6 的背景负荷、能源输入、设备参数、可用率、广播信号和 seed 完全成对固定；若 M5/M6 相对 M4 的网络接受率、请求无违规比例或可交付效果下降，该差异才可归因于设备空间集中。
- **H3，安全交付效应**：局部网络安全层应显著降低执行违规，但可能牺牲网络接受率和弃电降低率；两者必须同时报告。
- **H4，跨数据集稳健性**：相同网络条件下，不同数据集产生的时序、可用性和设备异质性应形成不同结果，结论以 {len(E22_DATASETS)} 个数据集和 {seed_count} 个 paired seeds 的分布为依据，而不是依赖单一案例。

## 回应范围与结论边界

E22 是针对 Reviewer #4 的网络约束补充仿真，不是现场实验。它比原聚合模型进一步考虑了径向拓扑、P/Q 潮流、支路热限、节点电压、主变容量、设备节点位置和空间集中响应，但仍采用平衡单相 LinDistFlow 与标准 IEEE 测试馈线。实验未覆盖三相不平衡、完整非线性 AC 潮流、保护动作、谐波、通信故障、参数辨识误差和真实控制器硬件。因此 E22 支持的结论是“在所声明网络模型与压力范围内具备可交付性和安全层有效性”，不能声称已经证明真实配电网现场安全。

## 数据集范围与输入角色

| 数据集 | 类型 | E22 输入表示 |
|---|---|---|
{datasets}

前 15 个数据集沿用 E20/E21 已验证的 canonical device-day 适配器。E22 额外加入项目原有的 NextGen 与 data2，使数据范围从 15 个扩大到 {len(E22_DATASETS)} 个，但不把两者并入 E21 mixed EPS 的训练集；两个冻结 EPS 模型保持不变，因此新增结果检验的是跨数据来源泛化，而不是重新训练收益。

### NextGen 适配

- 原始输入：`data/nextgen/*.csv`，100 个源设备，每个设备 365 天，每天 288 个五分钟点。
- 保留信息：实测负荷时序及其数据驱动可用率证据；源设备按确定性 70/15/15 划分，E22 只从 test partition 抽样。
- 统一设备层：每个 seed 构造 5000 台逻辑设备；容量、C-rate、初始 SOC 和 SOH 使用与其他数据集相同的 dataset-ID keyed 纯仿真协议。
- 能源输入：NextGen 的实测 PV 不直接替代 M0-M6 的公共能源输入，否则无法保持 IEEE-33/69 和跨数据集成对条件一致；其负荷时序决定网络负荷空间份额和日内形状。

### data2 适配

- 原始输入：`data/data2/processed_data_longer_than_30.xlsx`，18,531 个去重 transaction-session。
- 时间转换：按 `end_time` 将 `out_power` 聚合到 288 个五分钟槽；采用原 `observed_demand` 适配，`out_power` 表示充电负荷，不解释为发电或外部能源输入。
- 数据划分：交易 ID 是源 ID；单会话无法形成互斥日期划分时，按现有 partition 协议记录不可避免的 split overlap，E22 只从 test partition 抽样并在 audit 中保留复用率。
- 统一设备层与能源输入：同样构造 5000 台逻辑设备，使用公共纯仿真电池参数和 M0-M6 能源输入。data2 仅提供充电负荷日内形状及数据驱动可用率证据。

## 公平比较与数据审计

- 每个数据集、seed、拓扑和场景下，10 种算法共享同一批逻辑设备、可用率、节点映射、能源输入和扰动。
- IEEE-33/69 成对比较只改变拓扑；两个拓扑使用相同基础设备记录和 seed，并分别按冻结节点负荷权重映射到全部非平衡节点。
- M4-M6 使用相同的 M0 网络参数和同一份按负荷加权构造的背景负荷及能源输入；只替换设备节点映射。固定目标支路由网络拓扑和标准节点负荷预先决定：选择阻抗距离最远、下游负荷占全网 10%-25% 且至少包含两个负荷节点的支路，再以该支路内阻抗距离最远的负荷节点作为 M6 目标。该规则不读取算法结果。
- 5000 台超过真实独立源数量时允许有放回 profile bootstrap，但重新生成独立的纯仿真设备容量、功率和初始状态；audit 保存可用源数、唯一源数、是否 bootstrap 和平均复用次数。
- NextGen/data2 不参与冻结模型训练。E20 pooled EPS 与 E21 mixed EPS 仍使用原模型哈希，避免因新增数据重新训练而破坏成对归因。

### 冻结的空间集中目标

| 网络 | M5 目标支路 | M6 目标节点 | 目标支路下游基准有功负荷占比 |
|---|---:|---:|---:|
{targets}

目标位置仅由网络 case 决定，对全部数据集、算法和 seed 固定，不依据结果重新选择。

## 成对设计

- 数据集：{len(E22_DATASETS)} 个，每个数据集独立测试；包括原 15 个 canonical 数据集、NextGen 和 data2。
- 设备规模：每次 5000 台设备；容量不足时沿用当前固定规模 bootstrap 规则。
- 网络：IEEE-33 和 IEEE-69。
- 场景：M0 至 M6；M0-M3 检查网络压力，M4-M6 检查设备空间部署。
- 重复次数：每个条件 {seed_count} 个 paired seeds。
- 算法：10 种；同一条件共享设备、可用率、外部能源输入和随机扰动。
- EPS 模型：IEEE-33 和 IEEE-69 都使用同一冻结 aggregate 模型，避免把重新训练收益误认为拓扑收益。
- 总规模：{len(E22_DATASETS)} 数据集 × {len(TOPOLOGIES)} 拓扑 × {len(STRESS_MODES)} 压力场景 × {seed_count} seeds × {len(ALGORITHM_ORDER)} 算法 = {result_rows:,} 行逐 seed/算法结果。

## 网络模型

阻抗和标准节点负荷来自公开的 IEEE 33-bus 与 69-bus 径向测试系统。标准 case 没有统一可用于本实验的线路热限，因此先用标准节点负荷计算每条支路的基准下游潮流，再以 72% 基准负载率一次性确定热限；热限随后对全部数据集、算法和 seed 冻结。M1-M3 只使用预先声明的容量倍率和固定支路降额；M4-M6 恢复 M0 的网络参数，使其差异只来自设备位置。

潮流采用有功/无功 LinDistFlow。输入曲线表示“如果不受限时可获得的潜在能源”，未被负荷或设备吸收的部分直接记为弃电，不会错误地注入馈线。弃电吸收按径向网络的电气分区匹配：同一分区内允许通过支路传输，不允许把一个末端支路的富余无约束地记到另一个支路的充电上。

网络安全层不再把全系统乘同一个比例；发生拥塞时只压缩违规支路下游设备，只有局部处理仍不能恢复可行性时才启用全局兜底，并单独记录兜底步数。

## 七种连续场景

{stress}

## 算法与在线复杂度

{algorithms}

`Centralized greedy UB` 是逐时间步的理论设备吸收上界：按全部设备 headroom 做 O(N) 线性贪心分配并计算最多可吸收弃电；该列不代表经过 IEEE 网络约束后的物理可执行调度，网络可执行性单独由审计字段记录。

## 冻结模型

{os.linesep.join(f'- `{key}`：`{path}`' for key, path in model_paths.items())}

## 公共网络来源

- MATPOWER case data：<https://github.com/MATPOWER/matpower/tree/master/data>
- MATPOWER 许可证：BSD-3-Clause。
"""
    (output / "EXPERIMENT_DESIGN.md").write_text(text, encoding="utf-8")


def _write_metric_definitions(output: Path) -> None:
    text = """# E22 指标说明

## 可交付弃电降低率

先计算每个时刻“外部能源输入减去本地负荷”的正富余量。算法提出充电响应后，还必须经过电池和配电网约束。真正被设备安全吸收的能量才算减少弃电。

潜在能源没有被吸收时直接作为弃电，不会被当成已经注入网络的功率。IEEE-33 划分为 4 个径向电气分区，IEEE-69 划分为 8 个电气分区；一个分区内的设备可以吸收该分区内其他节点的富余，但不能跨越无关瓶颈“借用”富余。

`可交付弃电降低率 = 实际安全吸收的富余能量 / 无控制时的富余能量 × 100%`

该值越高越好。它不是算法请求了多少，而是网络最终允许交付多少。

## 网络接受率

`网络接受率 = 网络约束后接受的响应绝对值 / 算法请求的响应绝对值`

100% 表示算法请求全部可以安全执行；较低表示请求位置或功率与网络承载能力不匹配。

## 请求无违规比例

先不经过安全层，直接检查算法原始请求是否会造成线路过载、电压越界或主变过载。一天 288 个五分钟步中，无违规步数所占百分比就是请求无违规比例。该指标越高，说明算法原始决策越符合网络条件。

## 执行无违规比例

算法请求经过局部网络安全层之后，再检查实际执行结果。该值用于确认安全层是否有效，不能替代请求无违规比例。

## 网络性能保持率

`保持率 = IEEE-69 可交付弃电降低率 / IEEE-33 可交付弃电降低率`

1 表示换到更复杂网络后效果不变；小于 1 表示有效效果下降；大于 1 可能来自两个网络的空间富余位置和约束差异，不能简单解释为 IEEE-69 更容易。

## 空间保持率

`空间保持率 = M5 或 M6 的可交付弃电降低率 / 同网络 M4 的可交付弃电降低率`

M4-M6 除设备位置外全部成对固定。1 表示设备集中后效果保持不变；小于 1 表示空间集中削弱可交付效果。M4 自身固定为 1，仅作为审计参考。

## R² 与 NRMSE

R² 比较每个五分钟步的目标富余功率和实际吸收功率是否具有一致的时间变化。R² 可以为负，负值表示其误差比始终使用目标均值还大。NRMSE 是均方根误差除以正富余时段的平均目标功率，越小越好。

## 电压、支路负载、主变负载与损耗

- 最低电压使用标幺值 pu，正常范围为 0.95 至 1.05 pu。
- 同时保存全天最高电压，用于识别集中吸收或反向潮流造成的过电压风险。
- 支路负载率 1.0 表示达到热限，大于 1.0 表示过载。
- 主变负载率 1.0 表示达到 6 MVA 容量上限。
- 网络损耗使用 kW，来自 LinDistFlow 的电阻损耗近似。
"""
    (output / "METRIC_DEFINITIONS.md").write_text(text, encoding="utf-8")


def _write_figure_descriptions(
    output: Path, overall: list[dict[str, Any]], retention: list[dict[str, Any]],
) -> None:
    lookup = {(row["topology"], row["stress_mode"], row["algorithm"]): row for row in overall}
    eps33 = float(
        np.mean(
            [
                lookup[("ieee33", mode, "eps_e21_mixed")]["curtailment_reduction_pct"]
                for mode in STRESS_MODES
            ]
        )
    )
    eps69 = float(
        np.mean(
            [
                lookup[("ieee69", mode, "eps_e21_mixed")]["curtailment_reduction_pct"]
                for mode in STRESS_MODES
            ]
        )
    )
    eps_retention = float(
        np.nanmean(
            [row["retention_ratio"] for row in retention if row["algorithm"] == "eps_e21_mixed"]
        )
    )
    text = f"""# E22 图片说明

# 1. topology_effect_dumbbell

![IEEE-33 与 IEEE-69 成对效果](figures/topology_effect_dumbbell.png)

文件：

- `figures/topology_effect_dumbbell.png`
- `figures/topology_effect_dumbbell.pdf`
- 数据：`data/e22_overall_summary.csv`
- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/run_e22_ieee69_complexity.py`

### 实验内容

每一行是一种算法。蓝点和红点使用相同数据集、设备、seed 和网络场景，只改变 IEEE-33/IEEE-69 拓扑。该图直接回答 Reviewer #4 提出的“更复杂馈线是否削弱聚合控制效果”。连线长度表示拓扑改变带来的效果差异，不等同于现场网络安全证明。

### 横纵坐标

- 横坐标：除 Centralized greedy UB 外，为经过电池和网络约束后的可交付弃电降低率；Greedy UB 是不经过网络裁剪的设备层理论参考，不能解释为物理可执行效果。
- 纵坐标：算法名称及在线复杂度。
- 蓝点：IEEE-33；红点：IEEE-69。

### 结果

- E21 mixed EPS 在 IEEE-33 上的七场景平均效果：{eps33:.2f}%。
- E21 mixed EPS 在 IEEE-69 上的七场景平均效果：{eps69:.2f}%。
- 跨数据集和场景的平均网络保持率：{eps_retention:.3f}。

# 2. ieee69_effect_safety_pareto

![IEEE-69 效果与安全性](figures/ieee69_effect_safety_pareto.png)

文件：

- `figures/ieee69_effect_safety_pareto.png`
- `figures/ieee69_effect_safety_pareto.pdf`
- 数据：`data/e22_overall_summary.csv`

### 实验内容

七个子图分别对应 M0-M6。它将“最终效果”和“未经安全层修正的原始请求安全性”分开，直接检验大量设备同时响应广播时是否会先提出违反线路、电压或主变约束的请求。右上角表示效果高且原始请求本身较少触发约束。

### 横纵坐标

- 横坐标：除 Centralized greedy UB 外，为可交付弃电降低率；Greedy UB 仅表示设备层可吸收上界。
- 纵坐标：原始请求无网络违规的时间比例，越高越好。
- 理想算法位于右上角。

# 3. retention_by_dataset_algorithm

![不同数据集和算法的网络保持率](figures/retention_by_dataset_algorithm.png)

文件：

- `figures/retention_by_dataset_algorithm.png`
- `figures/retention_by_dataset_algorithm.pdf`
- 数据：`data/e22_retention.csv`

### 实验内容

该图不把 {len(E22_DATASETS)} 个数据集合并成一个数，而是逐数据集检查从 IEEE-33 切换到 IEEE-69 后还保留多少效果，用于确认结论不是由少数数据集平均得到。Centralized greedy UB 列只作为设备层参考，不用于支持网络可实施性结论。

### 横纵坐标

- 横坐标：不同算法。
- 纵坐标：{len(E22_DATASETS)} 个真实数据集。
- 颜色：七个场景平均后的保持率；绿色接近或高于 1，红色表示下降明显。

# 4. ieee69_stress_profiles

![IEEE-69 压力场景曲线](figures/ieee69_stress_profiles.png)

文件：

- `figures/ieee69_stress_profiles.png`
- `figures/ieee69_stress_profiles.pdf`
- 数据：`data/e22_overall_summary.csv`

### 横纵坐标

- 横坐标：M0-M3 为网络压力条件，M4-M6 为均匀、远端支路 50% 集中和远端节点 80% 集中。
- 纵坐标：{len(E22_DATASETS)} 个数据集汇总后的可交付弃电降低率。
- 每条线：一种算法，图例同时写出在线复杂度。

### 对 Reviewer #4 的回答

M0-M3 逐步加入深馈线高负载、双瓶颈反向潮流、线路降额和预测误差；M4-M6 只改变设备空间分布。曲线下降说明配网约束或空间集中会削弱聚合层效果，因此原聚合仿真的高弃电降低率不能无条件外推到真实馈线。

# 5. ieee69_network_acceptance

![IEEE-69 网络接受率](figures/ieee69_network_acceptance.png)

文件：

- `figures/ieee69_network_acceptance.png`
- `figures/ieee69_network_acceptance.pdf`
- 数据：`data/e22_summary.csv`

### 横纵坐标

- 横坐标：M0-M6 连续场景。
- 纵坐标：算法。
- 颜色和单元格数字：网络最终接受的响应占算法请求响应的比例。

### 对 Reviewer #4 的回答

该图量化线路热限、0.95-1.05 p.u. 电压范围和 6 MVA 主变容量共同作用后，算法请求有多少能够执行。接受率低表示聚合层请求与设备空间位置或网络余量不匹配。Centralized greedy UB 行表示其理论请求中可通过网络校核的比例，而不是把其上界值视为可交付结果。

# 6. ieee69_branch_time_heatmap

![IEEE-69 支路时空负载](figures/ieee69_branch_time_heatmap.png)

文件：

- `figures/ieee69_branch_time_heatmap.png`
- `figures/ieee69_branch_time_heatmap.pdf`
- 数据：`data/ieee69_m2_eps_trace.json`

### 实验内容

选取 CEC 数据集、M2 双瓶颈场景、E21 mixed EPS 和第一个 seed，展示复杂网络约束具体发生在什么支路、什么时间，而不是只给一天平均值。该图直接提供馈线热限的时空证据，但它是单数据集、单 seed 案例，不能替代跨数据集统计，也不单独证明电压和主变安全。

### 横纵坐标

- 横坐标：一天中的 288 个五分钟时间步。
- 纵坐标：IEEE-69 的 68 条支路，按配置顺序排列。
- 颜色：支路负载率；1.0 表示达到热限，越亮表示越接近或超过约束。

# 7. spatial_concentration_retention

![空间集中效果保持率](figures/spatial_concentration_retention.png)

文件：

- `figures/spatial_concentration_retention.png`
- `figures/spatial_concentration_retention.pdf`
- 数据：`data/e22_spatial_retention.csv`

### 实验内容

M4-M6 固定背景负荷、能源输入、设备参数、可用率、控制随机数和网络参数，只改变设备母线位置。该图直接检验同一广播响应在空间集中后还能保留多少可交付效果。

### 横纵坐标

- 横坐标：M5 的 50% 远端支路集中和 M6 的 80% 远端节点集中。
- 纵坐标：10 种算法。
- 左图：IEEE-33；右图：IEEE-69。
- 颜色和数字：相对同网络 M4 均匀部署的空间保持率；1 表示效果不变，小于 1 表示集中部署导致效果下降。

### 对 Reviewer #4 的回答

M5 模拟社区或馈线级集中，M6 模拟充电站或工业园式单节点集中。若保持率低于 1，说明设备数量大并不能消除局部网络瓶颈，空间位置必须进入可实施性判断。Centralized greedy UB 仅作设备层参考，不用于网络安全结论。

# 8. ieee69_constraint_component_audit

![IEEE-69 约束分项审计](figures/ieee69_constraint_component_audit.png)

文件：

- `figures/ieee69_constraint_component_audit.png`
- `figures/ieee69_constraint_component_audit.pdf`
- 数据：`data/e22_summary.csv`

### 实验内容

该图把 Reviewer #4 指出的四个问题拆开报告：支路热限、节点最低电压、主变负载，以及网络安全层将违规请求修正为无违规执行的幅度。每个单元格汇总 {len(E22_DATASETS)} 个数据集和 30 个 paired seeds。

### 横纵坐标

- 横坐标：M0-M6 网络与空间压力场景。
- 纵坐标：10 种算法。
- 左上：执行后的最大支路负载率，1.0 为热限。
- 右上：执行后的最低节点电压，允许下限为 0.95 p.u.。
- 左下：执行后的最大主变负载率，1.0 为 6 MVA 容量上限。
- 右下：安全层使无违规时间比例提升的百分点。

### 结论边界

该图证明这些约束在仿真中被显式检查，并展示安全层修正请求的代价和作用；它仍基于平衡单相 LinDistFlow 和标准测试馈线，不能替代三相 AC 潮流、保护校核或现场试验。
"""
    (output / "FIGURE_DESCRIPTIONS.md").write_text(text, encoding="utf-8")


def _write_readme(output: Path, manifest: dict[str, Any]) -> None:
    text = f"""# E22 IEEE-69 复杂网络实验

本目录比较相同设备和算法在 IEEE-33 与 IEEE-69 上的可交付控制效果。完整设计见 `EXPERIMENT_DESIGN.md`，指标的通俗定义见 `METRIC_DEFINITIONS.md`，每张图的读法见 `FIGURE_DESCRIPTIONS.md`。

## 完成状态

- 状态：`{manifest['status']}`
- 数据集：{manifest['dataset_count']} 个
- 网络：2 种
- 场景：{manifest['stress_mode_count']} 种（M0-M6）
- 算法：{manifest['algorithm_count']} 种
- seeds：{manifest['seed_count']}
- 逐 seed 结果：{manifest['by_seed_rows']} 行

## 主要文件

- `data/e22_by_seed.csv`：最细粒度的逐数据集、逐 seed、逐算法结果。
- `data/e22_summary.csv`：每个条件的 30-seed 汇总和置信区间。
- `data/e22_overall_summary.csv`：跨 {manifest['dataset_count']} 数据集汇总结果。
- `data/e22_retention.csv`：IEEE-69 相对 IEEE-33 的效果保持率。
- `data/e22_spatial_retention.csv`：M5/M6 相对均匀部署 M4 的空间保持率。
- `figures/`：PNG 和 PDF 图片。

## 复现

```bash
python -m src.extra.ieee33_device_day_simulation.figures.run_e22_ieee69_complexity
```

程序按数据集保存 checkpoint。任务中断后再次运行会跳过已经完成的 seed。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _finalize(
    output: Path, payloads: list[dict[str, Any]], bootstrap_draws: int, seed_count: int,
) -> dict[str, Any]:
    rows = [row for payload in payloads for row in payload["seed_results"]]
    rows.sort(
        key=lambda row: (
            E22_DATASETS.index(row["dataset"]),
            TOPOLOGIES.index(row["topology"]),
            list(STRESS_MODES).index(row["stress_mode"]),
            int(row["seed_index"]),
            ALGORITHM_ORDER.index(row["algorithm"]),
        )
    )
    summaries = _summaries(rows, bootstrap_draws)
    overall = _overall(summaries)
    retention = _retention(summaries)
    spatial_retention = _spatial_retention(summaries)
    data_dir = output / "data"
    _write_rows(data_dir / "e22_by_seed.csv", rows)
    _write_rows(data_dir / "e22_summary.csv", summaries)
    _write_rows(data_dir / "e22_overall_summary.csv", overall)
    _write_rows(data_dir / "e22_retention.csv", retention)
    _write_rows(data_dir / "e22_spatial_retention.csv", spatial_retention)
    figure_dir = output / "figures"
    figures = []
    figures.extend(_plot_dumbbell(overall, figure_dir))
    figures.extend(_plot_pareto(overall, figure_dir))
    figures.extend(_plot_retention(retention, figure_dir))
    figures.extend(_plot_stress(overall, figure_dir))
    figures.extend(_plot_acceptance(summaries, figure_dir))
    figures.extend(_plot_constraint_audit(summaries, figure_dir))
    figures.extend(_plot_spatial_retention(spatial_retention, figure_dir))
    figures.extend(_plot_trace(output, figure_dir))
    _write_figure_descriptions(output, overall, retention)
    manifest = {
        "protocol": PROTOCOL,
        "dataset_registry_version": DATASET_REGISTRY_VERSION,
        "status": "completed",
        "dataset_count": len(payloads),
        "dataset_ids": [payload["dataset"] for payload in payloads],
        "topology_count": len(TOPOLOGIES),
        "stress_mode_count": len(STRESS_MODES),
        "algorithm_count": len(ALGORITHM_ORDER),
        "seed_count": seed_count,
        "by_seed_rows": len(rows),
        "summary_rows": len(summaries),
        "overall_rows": len(overall),
        "retention_rows": len(retention),
        "spatial_retention_rows": len(spatial_retention),
        "figures": figures,
        "completed_at": _utc_now(),
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "checkpoint.json", manifest)
    _write_readme(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--max-datasets", type=int, default=len(E22_DATASETS))
    parser.add_argument("--max-topologies", type=int, default=len(TOPOLOGIES))
    parser.add_argument("--max-stress-modes", type=int, default=len(STRESS_MODES))
    parser.add_argument("--topology", action="append", choices=TOPOLOGIES)
    parser.add_argument("--stress-mode", action="append", choices=tuple(STRESS_MODES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_count <= SEED_COUNT:
        raise ValueError("seed-count 必须在 1 至 30 之间")
    args.max_datasets = max(1, min(args.max_datasets, len(E22_DATASETS)))
    args.max_topologies = max(1, min(args.max_topologies, len(TOPOLOGIES)))
    args.max_stress_modes = max(1, min(args.max_stress_modes, len(STRESS_MODES)))
    lock = _acquire_lock(args.output)
    if lock is None:
        print(f"{args.output} 已有 E22 实验运行，当前实例退出")
        return
    cases = _network_cases()
    model_paths = _model_paths(args.results_root)
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少冻结模型：{missing}")
    model_hashes = {key: _sha256(path) for key, path in model_paths.items()}
    network_hashes = {key: _sha256(NETWORK_FILES[key]) for key in TOPOLOGIES}
    optimizers = _optimizers(model_paths)
    _write_design(args.output, args.seed_count, model_paths)
    _write_metric_definitions(args.output)
    selected_datasets = E22_DATASETS[: args.max_datasets]
    selected_topologies = (
        tuple(args.topology) if args.topology else TOPOLOGIES[: args.max_topologies]
    )
    selected_stress_modes = (
        tuple(args.stress_mode)
        if args.stress_mode
        else tuple(STRESS_MODES)[: args.max_stress_modes]
    )
    payloads = []
    for dataset_index, dataset in enumerate(selected_datasets):
        payload = _run_dataset(
            dataset,
            args.output,
            cases,
            optimizers,
            model_hashes,
            args.seed_count,
            args.force,
            selected_topologies,
            selected_stress_modes,
        )
        payloads.append(payload)
        mixed._PARTITION_CACHE.clear()
        _ADDITIONAL_PARTITION_CACHE.clear()
        clear_data2_transaction_pool_cache()
        gc.collect()
        checkpoint = {
            "protocol": PROTOCOL,
            "dataset_registry_version": DATASET_REGISTRY_VERSION,
            "status": "running",
            "completed_datasets": len(payloads),
            "dataset_count": len(selected_datasets),
            "dataset_ids": list(selected_datasets),
            "seed_count": args.seed_count,
            "model_sha256": model_hashes,
            "network_sha256": network_hashes,
            "updated_at": _utc_now(),
        }
        _write_json(args.output / "checkpoint.json", checkpoint)
        print(
            json.dumps(
                {
                    "stage": "dataset_checkpoint",
                    "dataset": dataset,
                    "completed_datasets": dataset_index + 1,
                    "dataset_count": len(selected_datasets),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if (
        len(selected_datasets) == len(E22_DATASETS)
        and len(selected_topologies) == len(TOPOLOGIES)
        and len(selected_stress_modes) == len(STRESS_MODES)
    ):
        manifest = _finalize(args.output, payloads, args.bootstrap_draws, args.seed_count)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    else:
        manifest = {
            "protocol": PROTOCOL,
            "dataset_registry_version": DATASET_REGISTRY_VERSION,
            "status": "partial",
            "dataset_count": len(payloads),
            "dataset_ids": [payload["dataset"] for payload in payloads],
            "seed_count": args.seed_count,
            "completed_at": _utc_now(),
        }
        _write_json(args.output / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
