"""运行 E23：IEEE-69 空间数据类型与设备密度异质性实验。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import gc
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.signal import SignalOptimizer

from ..population.data2_transaction_loader import clear_data2_transaction_pool_cache
from . import fig4d_final_protocol as protocol
from . import run_e20_transfer as transfer
from . import run_e21_mixed_scenarios as mixed
from . import run_e22_ieee69_complexity as e22


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/E23"
CACHE_ROOT = OUTPUT / "data/.fleet_cache"
TOPOLOGY = "ieee69"
FLEET_SIZE = 5000
DEFAULT_SEED_COUNT = 30
STEPS = e22.STEPS
EPSILON = 1e-12
GLOBAL_MODEL = ROOT / "results/E21/models/e21_mixed_aggregate.pt"

DATASETS = tuple(e22.E22_DATASETS)
DATASET_LABELS = e22.DATASET_LABELS

REGIONS = {
    "R1": tuple(range(2, 28)),
    "R2": tuple(range(28, 36)),
    "R3": tuple(range(36, 47)),
    "R4": tuple(range(47, 51)),
    "R5": tuple(range(51, 53)),
    "R6": tuple(range(53, 66)),
    "R7": tuple(range(66, 68)),
    "R8": tuple(range(68, 70)),
}
REGION_ORDER = tuple(REGIONS)

DATASET_TYPES = {
    "building": (mixed.BDG1, mixed.BDG2),
    "residential": (mixed.LCL, mixed.CAMSL, mixed.IRISH, mixed.GOIENER, mixed.SGSC, mixed.OPSD),
    "thermal": (mixed.DANISH, mixed.HEAPO),
    "network_ami": (mixed.EU_RURAL, mixed.EU_35297, mixed.EU_8087, mixed.NORWAY),
    "der": (mixed.CEC, e22.NEXTGEN),
    "charging": (e22.DATA2,),
}
DATASET_TYPE = {dataset: kind for kind, values in DATASET_TYPES.items() for dataset in values}
TYPE_LABELS = {
    "building": "建筑",
    "residential": "住宅",
    "thermal": "热负荷/热泵",
    "network_ami": "配网/AMI",
    "der": "综合能源/DER",
    "charging": "充电",
}

ALGORITHMS = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_global_mixed",
    "centralized_optimal",
)
RUN_ALGORITHMS = ALGORITHMS
ALGORITHM_LABELS = {
    **{algorithm: mixed.ALGORITHM_DEFINITIONS[algorithm]["label"] for algorithm in ALGORITHMS if algorithm in mixed.ALGORITHM_DEFINITIONS},
    "eps_global_mixed": "EPS global mixed",
}
ALGORITHM_COMPLEXITY = {
    **{algorithm: mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"] for algorithm in ALGORITHMS if algorithm in mixed.ALGORITHM_DEFINITIONS},
    "eps_global_mixed": "O(1)",
}
COLORS = {
    "no_coordination": "#9c9c9c",
    "local_rules": "#ed7d31",
    "mpc_optimal": "#7057ff",
    "mean_field_control": "#4e79a7",
    "virtual_battery": "#59a14f",
    "packetized_energy_management": "#af7aa1",
    "transactive_control": "#d55e00",
    "eps_global_mixed": "#1f5aa6",
    "centralized_optimal": "#3a9d66",
}

CONDITIONS = {
    "H0": {"label": "H0 区域完全混合", "kind": "balanced"},
    "H1": {"label": "H1 类型分区", "kind": "type_zoned"},
    "H2_25": {"label": "H2 密度分区 25%", "kind": "density", "strength": 0.25},
    "H2_50": {"label": "H2 密度分区 50%", "kind": "density", "strength": 0.50},
    "H2_75": {"label": "H2 密度分区 75%", "kind": "density", "strength": 0.75},
    "H3": {"label": "H3 类型与密度联合不均衡", "kind": "joint", "strength": 0.50},
    "H4": {"label": "H4 不利网络耦合", "kind": "coupling", "coupling": "adverse", "strength": 0.50},
    "H5": {"label": "H5 有利网络耦合", "kind": "coupling", "coupling": "favorable", "strength": 0.50},
}

CONDITION_MARKERS = {
    "H0": "o",
    "H1": "s",
    "H2_25": "^",
    "H2_50": "D",
    "H2_75": "P",
    "H3": "X",
    "H4": "v",
    "H5": "<",
}

matplotlib.rcParams["font.family"] = "SimSun"
matplotlib.rcParams["axes.unicode_minus"] = False


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _allocate_counts(weights: np.ndarray, total: int) -> np.ndarray:
    raw = np.asarray(weights, dtype=float) / max(float(np.sum(weights)), EPSILON) * total
    values = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - values))[: total - int(np.sum(values))]:
        values[int(index)] += 1
    return values


def _network_cases() -> dict[str, dict[str, Any]]:
    return {TOPOLOGY: e22._network_cases()[TOPOLOGY]}


def _config_and_partitions(dataset: str) -> tuple[dict[str, Any], Any]:
    source_config, partitions = e22._dataset_source(dataset)
    return source_config, partitions


def _global_config() -> dict[str, Any]:
    """读取一个与全部适配器共用的配置，不加载任何完整 profile pool。"""
    source_config = mixed._config(mixed.BDG1, "aggregate")
    config = source_config.copy()
    config["control"] = dict(config["control"])
    config["original_model"] = dict(config["original_model"])
    config["control"]["pure_sim_device_model"] = True
    config["control"]["network_feedback"] = False
    config["control"]["final_network_mode"] = "ieee69"
    config["original_model"]["eps_packet_loss_rate"] = 0.001
    config["original_model"]["device_offline_rate"] = 0.0
    return config


def _prepare_fleet_cache(seed_count: int, force: bool = False) -> None:
    """每个数据集只加载一次，缓存所有 seed 的已抽样设备日记录。"""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    counts = _allocate_counts(np.ones(len(DATASETS)), FLEET_SIZE)
    for dataset_index, (dataset, count) in enumerate(zip(DATASETS, counts)):
        cache_path = CACHE_ROOT / f"{dataset}.pkl"
        if cache_path.is_file() and not force:
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                if cached.get("fleet_size") == FLEET_SIZE and cached.get("seed_count", 0) >= seed_count:
                    continue
            except (EOFError, OSError, pickle.UnpicklingError):
                cache_path.unlink(missing_ok=True)
        _, partitions = _config_and_partitions(dataset)
        seed_samples: dict[int, list[Any]] = {}
        audits: dict[int, dict[str, Any]] = {}
        for seed_index in range(seed_count):
            base_seed = 900000 + seed_index
            sampled, audit = protocol.sample_device_fleet(
                partitions.test,
                "fixed5000",
                int(count),
                base_seed + 10007 * (dataset_index + 1) + _stable_seed(dataset) % 100000,
                dataset_id=dataset,
            )
            start = 0
            seed_samples[seed_index] = [
                replace(
                    record,
                    device_id=f"{dataset}__{start + local_index:05d}",
                    source_device_id=f"{dataset}__{record.source_device_id}",
                    bus_id=2,
                )
                for local_index, record in enumerate(sampled)
            ]
            audits[seed_index] = audit
        temporary = cache_path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"fleet_size": FLEET_SIZE, "seed_count": seed_count, "dataset": dataset, "count": int(count), "samples": seed_samples, "audits": audits}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache_path)
        e22.mixed._PARTITION_CACHE.clear()
        e22._ADDITIONAL_PARTITION_CACHE.clear()
        clear_data2_transaction_pool_cache()
        gc.collect()
        print(json.dumps({"stage": "fleet_cache_checkpoint", "dataset": dataset, "dataset_index": dataset_index + 1, "dataset_count": len(DATASETS), "seed_count": seed_count}, ensure_ascii=False), flush=True)


def _sample_global_fleet(seed: int) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """按17个源等权抽样一次全局 fleet，并释放原始 pool 缓存。"""
    counts = _allocate_counts(np.ones(len(DATASETS)), FLEET_SIZE)
    records: list[Any] = []
    audits: dict[str, Any] = {}
    config = _global_config()
    for dataset_index, (dataset, count) in enumerate(zip(DATASETS, counts)):
        cache_path = CACHE_ROOT / f"{dataset}.pkl"
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        sampled = cached["samples"][int(seed - 900000)]
        records.extend(sampled)
        audits[dataset] = {"type": DATASET_TYPE[dataset], "count": int(count), **cached["audits"][int(seed - 900000)]}
    if len(records) != FLEET_SIZE:
        raise RuntimeError("全局 fleet 构造失败")
    return records, config, {"counts": {dataset: int(count) for dataset, count in zip(DATASETS, counts)}, "datasets": audits}


def _region_bus_counts(case: dict[str, Any], region_weights: np.ndarray, total: int) -> dict[str, dict[int, int]]:
    buses = sorted({int(row[1]) for row in case["branches"]})
    region_counts = _allocate_counts(region_weights, total)
    result: dict[str, dict[int, int]] = {}
    for region, count in zip(REGION_ORDER, region_counts):
        region_buses = [bus for bus in buses if bus in REGIONS[region]]
        if not region_buses:
            raise ValueError(f"区域 {region} 没有可部署节点")
        bus_counts = _allocate_counts(np.ones(len(region_buses)), int(count))
        result[region] = {bus: int(value) for bus, value in zip(region_buses, bus_counts)}
    return result


def _type_preferences() -> np.ndarray:
    types = tuple(DATASET_TYPES)
    values = np.ones((len(REGION_ORDER), len(types)), dtype=float)
    dominant = {
        "R1": "residential", "R2": "residential", "R3": "building",
        "R4": "thermal", "R5": "network_ami", "R6": "network_ami",
        "R7": "der", "R8": "charging",
    }
    for row, region in enumerate(REGION_ORDER):
        values[row, types.index(dominant[region])] = 8.0
    return np.asarray(
        [[values[row, types.index(DATASET_TYPE[dataset])] for dataset in DATASETS] for row in range(len(REGION_ORDER))],
        dtype=float,
    )


def _joint_preferences(headroom: np.ndarray) -> np.ndarray:
    """构造 H3 的类型与密度联合分区偏好。"""
    types = tuple(DATASET_TYPES)
    values = np.ones((len(REGION_ORDER), len(types)), dtype=float)
    ranked = np.argsort(np.asarray(headroom, dtype=float))
    low_regions = set(ranked[: max(2, len(ranked) // 3)])
    high_regions = set(ranked[-max(2, len(ranked) // 3):])
    for row in range(len(REGION_ORDER)):
        if row in low_regions:
            weights = {"residential": 5.0, "thermal": 7.0, "charging": 7.0, "der": 2.0, "building": 1.0, "network_ami": 1.0}
        elif row in high_regions:
            weights = {"residential": 5.0, "network_ami": 5.0, "building": 2.0, "thermal": 1.0, "der": 1.0, "charging": 1.0}
        else:
            weights = {"building": 6.0, "der": 6.0, "network_ami": 3.0, "residential": 2.0, "thermal": 1.0, "charging": 1.0}
        for index, kind in enumerate(types):
            values[row, index] = weights[kind]
    return np.asarray(
        [[values[row, types.index(DATASET_TYPE[dataset])] for dataset in DATASETS] for row in range(len(REGION_ORDER))],
        dtype=float,
    )


def _coupling_preferences(headroom: np.ndarray, mode: str) -> np.ndarray:
    """让 H4/H5 的高功率类型偏向低裕度或高裕度区域。"""
    power_score = {
        "building": 0.65,
        "residential": 0.35,
        "thermal": 0.80,
        "network_ami": 0.45,
        "der": 0.75,
        "charging": 1.00,
    }
    margin = np.asarray(headroom, dtype=float)
    region_score = 1.0 - margin if mode == "adverse" else margin
    region_score = (region_score - np.min(region_score)) / max(float(np.ptp(region_score)), EPSILON)
    return np.exp(
        3.0
        * region_score[:, None]
        * np.asarray([power_score[DATASET_TYPE[dataset]] for dataset in DATASETS])[None, :]
    )


def _ipf(global_dataset_counts: np.ndarray, region_counts: np.ndarray, preference: np.ndarray) -> np.ndarray:
    target_dataset = global_dataset_counts / max(float(np.sum(global_dataset_counts)), EPSILON)
    target_region = region_counts / max(float(np.sum(region_counts)), EPSILON)
    matrix = np.asarray(preference, dtype=float) * target_region[:, None] * target_dataset[None, :]
    for _ in range(100):
        matrix *= target_region[:, None] / np.maximum(matrix.sum(axis=1, keepdims=True), EPSILON)
        matrix *= target_dataset[None, :] / np.maximum(matrix.sum(axis=0, keepdims=True), EPSILON)
    raw = matrix * float(np.sum(region_counts))
    result = np.floor(raw).astype(int)
    row_need = np.asarray(region_counts, dtype=int) - result.sum(axis=1)
    column_need = np.asarray(global_dataset_counts, dtype=int) - result.sum(axis=0)
    while int(np.sum(row_need)) > 0:
        candidates = np.argwhere((row_need[:, None] > 0) & (column_need[None, :] > 0))
        if candidates.size == 0:
            raise RuntimeError("IPF整数化时没有可满足的区域和数据集余量")
        scores = np.asarray([raw[int(row), int(column)] - result[int(row), int(column)] for row, column in candidates])
        row, column = candidates[int(np.argmax(scores))]
        result[int(row), int(column)] += 1
        row_need[int(row)] -= 1
        column_need[int(column)] -= 1
    if not np.all(result.sum(axis=1) == region_counts) or not np.all(result.sum(axis=0) == global_dataset_counts):
        raise RuntimeError("区域与数据集配额没有同时守恒")
    return result


def _region_weights(case: dict[str, Any]) -> np.ndarray:
    base_load = {int(row[0]): float(row[1]) for row in case.get("base_loads", [])}
    values = np.asarray([sum(base_load.get(bus, 0.0) for bus in REGIONS[region]) for region in REGION_ORDER], dtype=float)
    if float(np.sum(values)) <= EPSILON:
        values = np.ones(len(REGION_ORDER), dtype=float)
    return values / float(np.sum(values))


def _concentrated_weights(base: np.ndarray, strength: float, headroom: np.ndarray) -> np.ndarray:
    low = np.argsort(headroom)[: max(2, len(headroom) // 3)]
    target = np.zeros_like(base)
    target[low] = base[low]
    if float(np.sum(target)) <= EPSILON:
        target[:] = base
    target /= float(np.sum(target))
    return (1.0 - strength) * base + strength * target


def _condition_matrix(condition: str, global_counts: np.ndarray, base_weights: np.ndarray, headroom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = CONDITIONS[condition]
    if spec["kind"] == "balanced":
        region_weights = base_weights
        preferences = np.ones((len(REGION_ORDER), len(DATASETS)))
    elif spec["kind"] == "type_zoned":
        region_weights = base_weights
        preferences = _type_preferences()
    elif spec["kind"] == "density":
        region_weights = _concentrated_weights(base_weights, float(spec["strength"]), headroom)
        preferences = np.ones((len(REGION_ORDER), len(DATASETS)))
    elif spec["kind"] == "joint":
        region_weights = _concentrated_weights(base_weights, float(spec["strength"]), headroom)
        preferences = _joint_preferences(headroom)
    else:
        region_weights = _concentrated_weights(base_weights, float(spec["strength"]), headroom)
        preferences = _coupling_preferences(headroom, str(spec["coupling"]))
    region_counts = _allocate_counts(region_weights, FLEET_SIZE)
    matrix = _ipf(global_counts, region_counts, preferences)
    return matrix, region_counts


def _assign_records(records: list[Any], matrix: np.ndarray, region_counts: np.ndarray, case: dict[str, Any], seed: int) -> list[Any]:
    bus_counts = _region_bus_counts(case, region_counts / FLEET_SIZE, FLEET_SIZE)
    by_dataset: dict[str, list[Any]] = {dataset: [] for dataset in DATASETS}
    for record in records:
        by_dataset[str(record.source_device_id).split("__", 1)[0]].append(record)
    rng = np.random.default_rng(seed)
    region_assignments: dict[str, list[int]] = {}
    region_cursors = {region: 0 for region in REGION_ORDER}
    for region in REGION_ORDER:
        buses = [bus for bus, value in bus_counts[region].items() for _ in range(value)]
        rng.shuffle(buses)
        region_assignments[region] = buses
    assigned_bus: dict[str, int] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        candidates = sorted(by_dataset[dataset], key=lambda record: record.device_id)
        cursor = 0
        for region_index, region in enumerate(REGION_ORDER):
            count = int(matrix[region_index, dataset_index])
            buses = region_assignments[region]
            if len(buses) != int(region_counts[region_index]):
                raise RuntimeError("区域节点配额错误")
            start = region_cursors[region]
            local = buses[start : start + count]
            region_cursors[region] += count
            for bus in local:
                record = candidates[cursor]
                cursor += 1
                assigned_bus[record.device_id] = int(bus)
        if cursor != len(candidates):
            raise RuntimeError(f"数据集 {dataset} 的区域分配没有用完记录")
    if any(region_cursors[region] != len(region_assignments[region]) for region in REGION_ORDER):
        raise RuntimeError("区域节点分配没有完整消耗")
    return [replace(record, bus_id=assigned_bus[record.device_id]) for record in records]


def _headroom_by_region(case: dict[str, Any], network: Any) -> np.ndarray:
    result = []
    base_load = {int(row[0]): float(row[1]) for row in case.get("base_loads", [])}
    base_q = {int(row[0]): float(row[2]) for row in case.get("base_loads", [])}
    flow: dict[int, float] = {}
    def visit(bus: int) -> float:
        value = np.hypot(base_load.get(bus, 0.0), base_q.get(bus, 0.0))
        for child in network.children.get(bus, []):
            value += visit(child)
        flow[bus] = value
        return value
    visit(network.slack)
    branch_margin = {f"{row['from']}-{row['to']}": 1.0 - flow[int(row["to"])] / max(network.branch_capacity[f"{row['from']}-{row['to']}"], EPSILON) for row in network.branches}
    evaluation = network.evaluate(base_load, {}, {})
    for region in REGION_ORDER:
        margins = []
        for bus in REGIONS[region]:
            current = bus
            while current != network.slack:
                branch = network.branch_by_child[current]
                margins.append(branch_margin[network._key(branch)])
                current = int(branch["from"])
        thermal_margin = float(np.clip(np.min(margins) if margins else 0.0, 0.0, 1.0))
        voltage_margin = float(np.clip(
            (min(evaluation.voltage_pu[bus] for bus in REGIONS[region]) - network.vmin)
            / max(1.0 - network.vmin, EPSILON),
            0.0,
            1.0,
        ))
        result.append(min(thermal_margin, voltage_margin))
    return np.asarray(result, dtype=float)


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / max(float(np.sum(p)), EPSILON); q = q / max(float(np.sum(q)), EPSILON)
    midpoint = 0.5 * (p + q)
    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > EPSILON
        return float(np.sum(left[mask] * np.log2(left[mask] / np.maximum(right[mask], EPSILON))))
    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def _imbalance_metrics(records: list[Any], condition: str, matrix: np.ndarray, base_weights: np.ndarray, headroom: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    region_counts = np.asarray([sum(int(record.bus_id) in REGIONS[region] for record in records) for region in REGION_ORDER], dtype=float)
    density = region_counts / FLEET_SIZE / np.maximum(base_weights, EPSILON)
    capacities = np.asarray([sum(float(record.capacity_kwh) for record in records if int(record.bus_id) in REGIONS[region]) for region in REGION_ORDER])
    capacity_share = capacities / max(float(np.sum(capacities)), EPSILON)
    type_names = tuple(DATASET_TYPES)
    type_global = np.asarray([sum(DATASET_TYPE[str(record.source_device_id).split("__", 1)[0]] == kind for record in records) for kind in type_names], dtype=float)
    type_rows = []
    jsd_values = []
    for region in REGION_ORDER:
        selected = [record for record in records if int(record.bus_id) in REGIONS[region]]
        counts = np.asarray([sum(DATASET_TYPE[str(record.source_device_id).split("__", 1)[0]] == kind for record in selected) for kind in type_names], dtype=float)
        jsd_value = _jsd(counts, type_global)
        jsd_values.append(jsd_value)
        type_rows.append({"condition": condition, "region": region, "device_count": len(selected), "device_share": len(selected) / FLEET_SIZE, "base_load_share": float(base_weights[len(type_rows)]), "density_ratio": float(density[len(type_rows)]), "capacity_share": float(capacity_share[len(type_rows)]), "headroom": float(headroom[len(type_rows)]), "type_jsd": jsd_value, "type_entropy": float(-np.sum((counts / max(np.sum(counts), EPSILON)) * np.log2(np.maximum(counts / max(np.sum(counts), EPSILON), EPSILON))))})
    if np.std(density) <= EPSILON or np.std(1.0 - headroom) <= EPSILON:
        i_h = 0.0
    else:
        i_h = float(np.corrcoef(density, 1.0 - headroom)[0, 1])
    capacity_density = capacity_share / np.maximum(base_weights, EPSILON)
    i_ch = float(np.corrcoef(capacity_density, 1.0 - headroom)[0, 1]) if np.std(capacity_density) > EPSILON else 0.0
    metrics = {"condition": condition, "condition_label": CONDITIONS[condition]["label"], "I_N": float(np.sqrt(np.sum(base_weights * (density - 1.0) ** 2))), "I_C": float(np.sqrt(np.sum(base_weights * (capacity_density - 1.0) ** 2))), "I_T": float(np.sum(region_counts / FLEET_SIZE * np.asarray(jsd_values)) / np.log2(2.0)), "I_H": i_h, "I_CH": i_ch, "fleet_size": FLEET_SIZE, "headroom_min": float(np.min(headroom)), "headroom_median": float(np.median(headroom))}
    return metrics, type_rows


def _run_condition(records: list[Any], config: dict[str, Any], condition: str, matrix: np.ndarray, region_counts: np.ndarray, case: dict[str, Any], base_weights: np.ndarray, headroom: np.ndarray, optimizer: SignalOptimizer, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    mapped = _assign_records(records, matrix, region_counts, case, seed + _stable_seed(condition))
    scenario = e22._build_scenario(mapped, mapped, case, TOPOLOGY, "M0", seed, {"placement_mode": "spatial_heterogeneity", "configured_concentration_fraction": 0.0})
    scenario["stress_mode"] = "M0"
    network = e22._network(case, TOPOLOGY, "M0")
    availability = protocol.availability_probability(mapped, config, mixed.AVAILABILITY_MODE)
    metric, region_rows = _imbalance_metrics(mapped, condition, matrix, base_weights, headroom)
    rows: list[dict[str, Any]] = []
    for algorithm_index, algorithm in enumerate(RUN_ALGORITHMS):
        result, _ = e22._run_seed(algorithm, mapped, config, scenario, availability, network, seed + 100000 * (algorithm_index + 1), seed + 1910000, optimizer if algorithm == "eps_global_mixed" else None, False)
        rows.append({"condition": condition, "condition_label": CONDITIONS[condition]["label"], "algorithm": algorithm, "algorithm_label": ALGORITHM_LABELS[algorithm], "complexity": ALGORITHM_COMPLEXITY[algorithm], "seed_index": int(seed - 900000), "fleet_seed": seed, "fleet_size": len(mapped), "I_N": metric["I_N"], "I_C": metric["I_C"], "I_T": metric["I_T"], "I_H": metric["I_H"], **result})
    return rows, metric, region_rows


def _bootstrap_mean(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return float(np.mean(array)), float(np.mean(array)), float(np.mean(array))
    rng = np.random.default_rng(seed)
    samples = np.asarray([np.mean(array[rng.integers(0, len(array), len(array))]) for _ in range(1000)])
    return float(np.mean(array)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _plot_outputs(rows: list[dict[str, Any]], metrics: list[dict[str, Any]], region_rows: list[dict[str, Any]]) -> list[str]:
    figure_dir = OUTPUT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    available_conditions = tuple(condition for condition in CONDITIONS if any(row["condition"] == condition for row in rows))
    available_algorithms = tuple(algorithm for algorithm in ALGORITHMS if any(row["algorithm"] == algorithm for row in rows))
    metric_names = ("I_N", "I_C", "I_T", "I_H")
    fig, ax = plt.subplots(figsize=(12, 5.5)); x = np.arange(len(available_conditions)); width = 0.18
    for index, name in enumerate(metric_names):
        values = [np.mean([float(row[name]) for row in metrics if row["condition"] == condition]) for condition in available_conditions]
        ax.bar(x + (index - 1.5) * width, values, width, label=name)
    ax.set_xticks(x, [f"{key}\n{CONDITIONS[key]['label']}" for key in available_conditions], rotation=20, ha="right")
    ax.set_ylabel("不平衡指标值"); ax.set_title("E23 空间异质性：四个不平衡指标"); ax.grid(axis="y", alpha=0.25); ax.legend(ncol=4)
    path = figure_dir / "imbalance_metric_summary.png"; fig.tight_layout(); fig.savefig(path, dpi=220); fig.savefig(path.with_suffix(".pdf")); plt.close(fig); paths.append(str(path))
    for metric_name, xlabel, filename in (("I_N", "设备密度不平衡 I_N", "effect_vs_density_imbalance"), ("I_T", "数据类型异质性 I_T", "effect_vs_type_heterogeneity"), ("I_H", "密度与低裕度耦合 I_H", "safety_vs_headroom_coupling")):
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        for algorithm in available_algorithms:
            for condition in available_conditions:
                selected = [row for row in rows if row["algorithm"] == algorithm and row["condition"] == condition]
                if not selected:
                    continue
                ax.scatter(
                    [float(row[metric_name]) for row in selected],
                    [float(row["mean_reduction_pct"]) for row in selected],
                    s=28,
                    alpha=0.42,
                    color=COLORS[algorithm],
                    marker=CONDITION_MARKERS[condition],
                    edgecolors="white",
                    linewidths=0.25,
                    label=ALGORITHM_LABELS[algorithm] if condition == available_conditions[0] else None,
                )
        algorithm_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[algorithm], markeredgecolor="white", markersize=7, label=ALGORITHM_LABELS[algorithm])
            for algorithm in available_algorithms
        ]
        condition_handles = [
            Line2D([0], [0], marker=CONDITION_MARKERS[condition], color="#555555", linestyle="None", markersize=7, label=CONDITIONS[condition]["label"])
            for condition in available_conditions
        ]
        algorithm_legend = ax.legend(handles=algorithm_handles, title="颜色：算法", fontsize=7, title_fontsize=8, loc="upper left", ncol=2)
        ax.add_artist(algorithm_legend)
        ax.legend(handles=condition_handles, title="点形：空间条件", fontsize=7, title_fontsize=8, loc="lower right", ncol=2)
        ax.text(0.01, 1.015, "颜色表示算法；点形和图例直接标出 H 条件", transform=ax.transAxes, fontsize=8, va="bottom")
        ax.set_xlabel(xlabel); ax.set_ylabel("可交付弃电降低率 (%)"); ax.set_title(f"E23：{xlabel}与算法效果"); ax.grid(alpha=0.25)
        path = figure_dir / f"{filename}.png"; fig.tight_layout(); fig.savefig(path, dpi=220); fig.savefig(path.with_suffix(".pdf")); plt.close(fig); paths.append(str(path))
    fig, ax = plt.subplots(figsize=(13, 6)); positions = np.arange(len(available_conditions));
    for index, algorithm in enumerate(available_algorithms):
        means = [np.mean([float(row["mean_reduction_pct"]) for row in rows if row["condition"] == condition and row["algorithm"] == algorithm]) for condition in available_conditions]
        ax.plot(positions, means, marker="o", linewidth=1.7, color=COLORS[algorithm], label=f"{ALGORITHM_LABELS[algorithm]} ({ALGORITHM_COMPLEXITY[algorithm]})")
    ax.set_xticks(positions, list(available_conditions)); ax.set_ylabel("可交付弃电降低率 (%)"); ax.set_title("E23：IEEE-69空间异质性下各算法效果"); ax.grid(axis="y", alpha=0.25); ax.legend(ncol=2, fontsize=8)
    path = figure_dir / "spatial_heterogeneity_algorithm_comparison.png"; fig.tight_layout(); fig.savefig(path, dpi=220); fig.savefig(path.with_suffix(".pdf")); plt.close(fig); paths.append(str(path))
    fig, ax = plt.subplots(figsize=(12, 6));
    for region in REGION_ORDER:
        values = [np.mean([float(row["device_share"]) for row in region_rows if row["condition"] == condition and row["region"] == region]) for condition in available_conditions]
        ax.plot(list(available_conditions), values, marker="o", label=region)
    ax.set_ylabel("区域设备比例"); ax.set_xlabel("实验条件"); ax.set_title("E23：区域设备密度和条件构造"); ax.grid(axis="y", alpha=0.25); ax.legend(ncol=4)
    path = figure_dir / "region_density_assignments.png"; fig.tight_layout(); fig.savefig(path, dpi=220); fig.savefig(path.with_suffix(".pdf")); plt.close(fig); paths.append(str(path))
    return paths


def _write_docs(seed_count: int, output_paths: list[str]) -> None:
    spatial_design = f'''# E23 空间异质性实验设计

## 目的

本实验在 IEEE-69 的不同电气区域中分配不同数据集类型，并逐步增加设备密度不均衡，回答：**当数据类型、设备数量和网络裕度不再空间均匀时，EPS 和其他算法的实际弃电消纳与网络安全如何变化？**

BDG、住宅、热泵、AMI、DER 和充电数据作为行为类型，区域分配用于构造可复现的 IEEE-69 空间压力测试。

## 固定协议

- IEEE-69 标准径向馈线，8 个区域：R1=2--27、R2=28--35、R3=36--46、R4=47--50、R5=51--52、R6=53--65、R7=66--67、R8=68--69。
- 使用 17 个数据集，每个数据集等权抽样；总 fleet 固定为 5000 台设备。
- 每个 seed 的所有条件和算法复用相同的设备 profile、容量、功率、SOC 和可用率；只改变数据类型到区域的分配和设备密度。
- EPS 使用 `results/E21/models/e21_mixed_aggregate.pt` 的全局 mixed controller。该模型输入只有广播信号特征，不读取 region ID 或 dataset ID；IEEE-69 潮流安全层在执行时统一约束所有算法。
- 共 {seed_count} 个固定 seed，单进程运行以控制内存；结果可以从 checkpoint 断点续跑。

## 六种正式条件

|条件|构造|保持不变的内容|
|-|-|-|
|H0|区域完全混合：每个区域采用相同的数据类型比例，设备数量按区域基础负荷比例分配|全局数据集比例、设备总量|
|H1|类型分区：住宅、建筑、热负荷、配网/AMI、DER、充电分别偏向不同区域|区域设备总量仍按网络负荷权重|
|H2|各区域类型比例保持全局一致，设备密度分别设置为25%、50%、75%集中强度|数据类型组成、总设备数、全局数据集比例|
|H3|在 H2 50%密度集中基础上叠加类型分区：低裕度远端区域偏住宅、热负荷和充电，中部偏建筑和DER，高裕度近端偏低功率住宅和AMI|总设备量、全局数据集比例、设备参数|
|H4|保持 H3 的区域设备密度和全局类型比例，将高功率类型匹配到最低网络裕度区域|设备总量、区域密度和全局数据集比例|
|H5|保持 H3 的区域设备密度和全局类型比例，将高功率类型匹配到较高网络裕度区域|设备总量、区域密度和全局数据集比例|

## 指标

- `I_N`：设备数量密度不平衡，按区域网络负荷权重加权的密度比离散程度；0 表示均衡。
- `I_C`：可控电池容量不平衡，计算方式与 `I_N` 相同，但把设备数量换成容量。
- `I_T`：区域数据类型异质性，使用区域类型分布与全局类型分布的 Jensen--Shannon divergence，并按设备比例加权；0 表示每个区域类型组成与全局相同。
- `I_H`：密度与低网络裕度的 Pearson 相关系数。正值表示设备越集中在低裕度区域，负值表示设备更多位于高裕度区域。
- `I_CH`：可控容量密度与低网络裕度的相关系数，专门审计 H4/H5 在设备数量相同但高功率类型位置互换时的耦合方向。
- 效果：IEEE-69 潮流和设备 SOC 约束之后的可交付弃电降低率。
- 安全：新增线路过载、电压越限和变压器过载由既有 E22 执行器逐步记录。

## 产出

原始逐 seed/算法结果在 `data/spatial_heterogeneity_by_seed.csv`，区域指标在 `data/region_assignments.csv` 和 `data/imbalance_metrics.csv`。交互效应在 `data/spatial_interaction_effects.csv`。图中曲线、散点和柱状图均直接由这些 CSV 生成。

'''
    (OUTPUT / "E23_SPATIAL_EXPERIMENT_DESIGN.md").write_text(spatial_design, encoding="utf-8")
    (OUTPUT / "data/spatial_heterogeneity_README.md").write_text("""# E23 空间异质性\n\n本目录保存 E23 的 IEEE-69 空间异质性结果。正式条件为 H0、H1、H2、H3、H4、H5；H2包含25%、50%、75%三种密度强度。\n\n运行：\n\n```bash\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m src.extra.ieee33_device_day_simulation.figures.run_e23_spatial_heterogeneity --seed-count 30 --force\n```\n""", encoding="utf-8")
    descriptions = {
        "imbalance_metric_summary": ("四个不平衡指标", "横轴是 H0、H1、H2（25%、50%、75%）、H3、H4、H5；纵轴是 I_N、I_C、I_T、I_H 的数值。每种颜色对应一个指标。该图检查实验条件是否产生了设计中的数量、容量、类型和网络耦合差异。"),
        "effect_vs_density_imbalance": ("密度不均衡与效果", "横轴是 I_N，纵轴是经过设备和 IEEE-69 网络安全层后的可交付弃电降低率。每个点是一个条件、seed 和算法。点的颜色表示算法。该图回答设备集中程度变化是否对应效果变化。"),
        "effect_vs_type_heterogeneity": ("类型异质性与效果", "横轴是 I_T，纵轴是可交付弃电降低率。I_T 越大，区域类型组成越偏离全局混合。每个点对应一个条件、seed 和算法，用来区分数据类型空间差异与单纯设备数量差异。"),
        "safety_vs_headroom_coupling": ("网络裕度耦合与效果", "横轴是 I_H，纵轴是可交付弃电降低率。I_H 为正表示设备密度更多落在低裕度区域，负值表示更多落在高裕度区域。H4 与 H5 使用相同的区域设备密度和全局数据集配额，图中重点比较两种类型与网络裕度匹配方向下的效果差异。"),
        "spatial_heterogeneity_algorithm_comparison": ("不同空间条件下的算法对比", "横轴是实验条件，纵轴是可交付弃电降低率。每条线是一种算法，图例括号给出在线复杂度。它回答类型分区、密度集中和网络耦合改变后，各方法实际能消纳多少弃电。"),
        "region_density_assignments": ("区域密度构造检查", "横轴是实验条件，纵轴是各 IEEE-69 区域设备比例，每条线是 R1--R8。它是审计图，用于检查 H2 的密度强度以及 H3/H4/H5 的区域设备密度是否保持配对。"),
    }
    lines = ["# E23 空间异质性图片说明", "", "图中效果均为 IEEE-69 网络执行后的可交付效果。Centralized greedy UB 表示设备层理论可吸收上限，不承担网络安全可执行算法的含义。", ""]
    for name, (title, body) in descriptions.items():
        lines.extend([f"## {title}", "", f"![{title}](figures/{name}.png)", "", f"文件：`figures/{name}.png`、`figures/{name}.pdf`", "", body, ""])
    spatial_descriptions = "\n".join(lines)
    (OUTPUT / "E23_SPATIAL_FIGURE_DESCRIPTIONS.md").write_text(spatial_descriptions, encoding="utf-8")
    (OUTPUT / "data/spatial_heterogeneity_FIGURE_DESCRIPTIONS.md").write_text(spatial_descriptions, encoding="utf-8")
    root_descriptions = OUTPUT / "FIGURE_DESCRIPTIONS.md"
    if root_descriptions.is_file():
        root_text = root_descriptions.read_text(encoding="utf-8")
        marker = "# E23 空间异质性图片说明"
        if marker in root_text:
            root_text = root_text.split(marker, 1)[0].rstrip() + "\n\n" + spatial_descriptions
            root_descriptions.write_text(root_text, encoding="utf-8")
    _write_json(OUTPUT / "data/spatial_heterogeneity_manifest.json", {"protocol": "E23_ieee69_spatial_heterogeneity_v1", "topology": TOPOLOGY, "dataset_count": len(DATASETS), "fleet_size": FLEET_SIZE, "seed_count": seed_count, "conditions": list(CONDITIONS), "algorithms": list(ALGORITHMS), "figures": output_paths})
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["spatial_heterogeneity"] = {"protocol": "E23_ieee69_spatial_heterogeneity_v1", "topology": TOPOLOGY, "dataset_count": len(DATASETS), "fleet_size": FLEET_SIZE, "seed_count": seed_count, "conditions": list(CONDITIONS), "algorithms": list(ALGORITHMS), "data": ["data/spatial_heterogeneity_by_seed.csv", "data/imbalance_metrics.csv", "data/region_assignments.csv", "data/spatial_heterogeneity_summary.csv", "data/spatial_interaction_effects.csv"], "figures": [str(Path(path).relative_to(OUTPUT)) for path in output_paths]}
    _write_json(manifest_path, manifest)


def main() -> None:
    global FLEET_SIZE, RUN_ALGORITHMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--max-conditions", type=int, default=len(CONDITIONS))
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--fleet-size", type=int, default=FLEET_SIZE)
    parser.add_argument("--algorithm", action="append", choices=ALGORITHMS, help="smoke test时只运行指定算法；正式运行不传此参数")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_count <= DEFAULT_SEED_COUNT:
        raise ValueError("seed-count 必须位于 1 到 30 之间")
    if args.fleet_size < len(DATASETS):
        raise ValueError("fleet-size 必须至少覆盖每个数据集一个设备")
    FLEET_SIZE = int(args.fleet_size)
    RUN_ALGORITHMS = tuple(args.algorithm) if args.algorithm else ALGORITHMS
    seed_count = min(args.seed_count, args.max_seeds or args.seed_count)
    selected_conditions = tuple(list(CONDITIONS)[: max(1, min(args.max_conditions, len(CONDITIONS)))])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_dir = OUTPUT / "data"; data_dir.mkdir(exist_ok=True)
    if not GLOBAL_MODEL.is_file():
        raise FileNotFoundError(f"缺少全局 EPS 模型：{GLOBAL_MODEL}")
    _prepare_fleet_cache(seed_count, force=args.force)
    optimizer = SignalOptimizer(transfer.ScaledEstimator(protocol.load_frozen_eps_controller(GLOBAL_MODEL)[0], mixed.RESPONSE_SCALE_KW, 0.0))
    case = _network_cases()[TOPOLOGY]
    base_weights = _region_weights(case)
    headroom = _headroom_by_region(case, e22._network(case, TOPOLOGY, "M0"))
    global_counts = _allocate_counts(np.ones(len(DATASETS)), FLEET_SIZE)
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    raw_path = data_dir / "spatial_heterogeneity_by_seed.csv"
    metrics_path = data_dir / "imbalance_metrics.csv"
    region_path = data_dir / "region_assignments.csv"
    completed_keys: set[tuple[int, str, str]] = set()
    if raw_path.is_file() and not args.force:
        with raw_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row); completed_keys.add((int(row["seed_index"]), row["condition"], row["algorithm"]))
        if metrics_path.is_file():
            with metrics_path.open(newline="", encoding="utf-8") as handle: metrics.extend(csv.DictReader(handle))
        if region_path.is_file():
            with region_path.open(newline="", encoding="utf-8") as handle: region_rows.extend(csv.DictReader(handle))
    for seed_index in range(seed_count):
        records, config, audit = _sample_global_fleet(900000 + seed_index)
        h3_region_counts = None
        h3_global_matrix = None
        for condition in selected_conditions:
            matrix, region_counts = _condition_matrix(condition, global_counts, base_weights, headroom)
            if condition == "H3":
                h3_region_counts = np.asarray(region_counts, dtype=int)
                h3_global_matrix = np.asarray(matrix, dtype=int)
            if condition in {"H4", "H5"} and h3_region_counts is not None:
                if not np.array_equal(np.asarray(region_counts, dtype=int), h3_region_counts):
                    raise RuntimeError(f"{condition} 必须与 H3 使用完全相同的区域设备密度")
                if not np.array_equal(np.asarray(matrix, dtype=int).sum(axis=1), h3_global_matrix.sum(axis=1)):
                    raise RuntimeError(f"{condition} 必须与 H3 使用完全相同的区域设备总量")
            if all((seed_index, condition, algorithm) in completed_keys for algorithm in RUN_ALGORITHMS):
                continue
            condition_rows, metric, condition_regions = _run_condition(records, config, condition, matrix, region_counts, case, base_weights, headroom, optimizer, 900000 + seed_index)
            rows.extend(condition_rows); metrics.append({"seed_index": seed_index, **metric}); region_rows.extend([{**row, "seed_index": seed_index} for row in condition_regions])
            _write_rows(raw_path, rows); _write_rows(metrics_path, metrics); _write_rows(region_path, region_rows)
            print(json.dumps({"stage": "condition_checkpoint", "seed_index": seed_index, "condition": condition, "rows": len(rows)}, ensure_ascii=False), flush=True)
    # 交互效应以条件均值计算，便于审计类型和密度的联合影响。
    grouped = {(row["condition"], row["algorithm"]): float(np.mean([float(item["mean_reduction_pct"]) for item in rows if item["condition"] == row["condition"] and item["algorithm"] == row["algorithm"]])) for row in rows}
    summary_rows = []
    for condition, algorithm in sorted(grouped):
        selected = [item for item in rows if item["condition"] == condition and item["algorithm"] == algorithm]
        mean, low, high = _bootstrap_mean([float(item["mean_reduction_pct"]) for item in selected], _stable_seed(f"summary|{condition}|{algorithm}"))
        summary_rows.append({"condition": condition, "condition_label": CONDITIONS[condition]["label"], "algorithm": algorithm, "algorithm_label": ALGORITHM_LABELS[algorithm], "complexity": ALGORITHM_COMPLEXITY[algorithm], "seed_count": len(selected), "mean_reduction_pct": mean, "ci_low_pct": low, "ci_high_pct": high, "network_acceptance_mean_pct": 100.0 * np.mean([float(item["network_acceptance_ratio"]) for item in selected]), "requested_added_violation_mean_pct": 100.0 * np.mean([float(item["requested_added_violation_steps"]) / STEPS for item in selected]), "minimum_voltage_pu": np.mean([float(item["minimum_voltage_pu"]) for item in selected]), "maximum_branch_loading": np.mean([float(item["maximum_branch_loading"]) for item in selected]), "maximum_transformer_loading": np.mean([float(item["maximum_transformer_loading"]) for item in selected])})
    _write_rows(data_dir / "spatial_heterogeneity_summary.csv", summary_rows)
    interaction_rows = []
    for algorithm in RUN_ALGORITHMS:
        h0 = grouped.get(("H0", algorithm), float("nan")); h1 = grouped.get(("H1", algorithm), float("nan")); h2 = grouped.get(("H2_50", algorithm), float("nan")); h3 = grouped.get(("H3", algorithm), float("nan"))
        interaction_rows.append({"algorithm": algorithm, "algorithm_label": ALGORITHM_LABELS[algorithm], "effect_interaction_pp": h3 - h2 - h1 + h0, "adverse_minus_favorable_pp": grouped.get(("H4", algorithm), float("nan")) - grouped.get(("H5", algorithm), float("nan")), "condition_baseline": "H0", "type_condition": "H1", "density_condition": "H2_50", "joint_condition": "H3"})
    _write_rows(data_dir / "spatial_interaction_effects.csv", interaction_rows)
    figure_paths = _plot_outputs(rows, metrics, region_rows)
    _write_docs(seed_count, figure_paths)
    print(json.dumps({"status": "complete", "seed_count": seed_count, "conditions": selected_conditions, "rows": len(rows), "figures": figure_paths}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
