"""不依赖实验代码的场景混合与两两组合生成器。"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import ProfileCatalog, ProfilePartition, ProfileRef, stable_seed
from .constants import (
    FLEET_SIZE,
    PAIR_ORDER,
    PAIR_SEED_BASE,
    PAIR_SEED_STRIDE,
    PARTITION_SEED,
    SCENARIOS,
    SCENARIO_SEEDS,
    SHORT_NAMES,
    TEST_SEED_COUNT,
    ZONE_BUSES,
)


@dataclass(frozen=True)
class GeneratedDevice:
    row_index: int
    dataset_id: str
    device_id: str
    source_device_id: str
    day_id: int
    zone_id: str
    bus_id: int
    capacity_kwh: float
    peak_power_kw: float
    c_rate: float
    initial_soc: float
    initial_soh: float
    source_path: str
    source_row_index: int
    source_locator: str


def normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("组合权重总和必须为正")
    return {dataset: float(value) / total for dataset, value in weights.items()}


def allocate_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    normalized = normalized_weights(weights)
    if total < len(normalized):
        raise ValueError("fleet 规模小于组成数据集数量")
    raw = {dataset: value * total for dataset, value in normalized.items()}
    residual_total = total - len(normalized)
    residual_weights = {dataset: max(value - 1.0, 0.0) for dataset, value in raw.items()}
    residual_sum = float(sum(residual_weights.values()))
    residual_raw = {
        dataset: (value * residual_total / residual_sum if residual_sum > 0 else 0.0)
        for dataset, value in residual_weights.items()
    }
    counts = {dataset: 1 + int(np.floor(value)) for dataset, value in residual_raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        normalized,
        key=lambda dataset: residual_raw[dataset] - np.floor(residual_raw[dataset]),
        reverse=True,
    )
    for dataset in order[:remainder]:
        counts[dataset] += 1
    return counts


def shifted_weights(weights: dict[str, float], seed_index: int) -> tuple[dict[str, float], str]:
    values = normalized_weights(weights)
    if seed_index < 10:
        return values, "nominal"
    dominant = max(values, key=values.get)
    old = values[dominant]
    if seed_index < 20:
        new = min(old + 0.15, 0.95)
        regime = "dominant_plus_15pp"
    else:
        new = max(old - 0.15, 0.01)
        regime = "dominant_minus_15pp"
    scale = (1.0 - new) / max(1.0 - old, 1e-12)
    return normalized_weights({
        dataset: (new if dataset == dominant else value * scale)
        for dataset, value in values.items()
    }), regime


def pair_specs() -> list[dict[str, Any]]:
    pairs = []
    for first_index, dataset_a in enumerate(PAIR_ORDER):
        for dataset_b in PAIR_ORDER[first_index + 1:]:
            pairs.append({
                "pair_index": len(pairs),
                "pair_id": f"P{len(pairs) + 1:03d}",
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "pair_label": f"{SHORT_NAMES[dataset_a]} + {SHORT_NAMES[dataset_b]}",
            })
    return pairs


def _sample_profiles(
    partition: ProfilePartition, count: int, seed: int, unique: bool
) -> tuple[list[ProfileRef], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[ProfileRef]] = {}
    for record in partition.records:
        grouped.setdefault(record.source_device_id, []).append(record)
    source_ids = sorted(grouped)
    if unique or len(source_ids) >= count:
        selected_ids = rng.choice(source_ids, size=count, replace=False)
        profiles = [
            grouped[str(source_id)][int(rng.integers(0, len(grouped[str(source_id)])))]
            for source_id in selected_ids
        ]
        bootstrap = False
    else:
        indices = rng.integers(0, len(partition.records), size=count)
        profiles = [partition.records[int(index)] for index in indices]
        bootstrap = True
    unique_sources = len({profile.source_device_id for profile in profiles})
    return profiles, {
        "available_profile_sources": len(source_ids),
        "selected_unique_profile_sources": unique_sources,
        "profile_bootstrap": bootstrap,
        "mean_profile_reuse": count / max(unique_sources, 1),
    }


def _max_unique_allocation(
    weights: dict[str, float], capacities: dict[str, int], maximum: int
) -> tuple[int, dict[str, int]]:
    normalized = normalized_weights(weights)
    upper = min(maximum, sum(capacities.values()))
    upper = min(
        upper,
        *(int(capacities[dataset] / weight) + len(normalized) + 2
          for dataset, weight in normalized.items()),
    )
    for fleet_size in range(upper, len(normalized) - 1, -1):
        counts = allocate_counts(normalized, fleet_size)
        if all(counts[dataset] <= capacities[dataset] for dataset in counts):
            return fleet_size, counts
    raise ValueError("真实源容量不足，无法构造 source-unique fleet")


def _synthetic_devices(
    profiles: list[ProfileRef], dataset: str, parameter_seed: int,
    start: int, scenario: dict[str, Any] | None,
) -> list[GeneratedDevice]:
    rng = np.random.default_rng(parameter_seed)
    devices = []
    for local_index, profile in enumerate(profiles):
        capacity = float(rng.uniform(5.0, 20.0))
        sigma = 0.35
        c_rate = float(np.clip(
            rng.lognormal(np.log(0.45) - sigma**2 / 2.0, sigma), 0.2, 1.0
        ))
        sampled_initial_soc = float(rng.uniform(0.2, 0.9))
        initial_soc = float(np.clip(
            (capacity * sampled_initial_soc) / max(capacity, 1e-6), 0.05, 0.98
        ))
        initial_soh = float(rng.uniform(0.82, 1.0))
        zone_id = profile.zone_id
        bus_id = profile.bus_id
        if scenario is not None and "zones" in scenario:
            zones = tuple(scenario["zones"][dataset])
            zone_id = zones[local_index % len(zones)]
            buses = ZONE_BUSES[zone_id]
            bus_id = int(buses[(local_index // len(zones)) % len(buses)])
        row_index = start + local_index
        devices.append(GeneratedDevice(
            row_index=row_index,
            dataset_id=dataset,
            device_id=f"{dataset}__{row_index:05d}",
            source_device_id=f"{dataset}__{profile.source_device_id}",
            day_id=profile.day_id,
            zone_id=zone_id,
            bus_id=bus_id,
            capacity_kwh=capacity,
            peak_power_kw=capacity * c_rate,
            c_rate=c_rate,
            initial_soc=initial_soc,
            initial_soh=initial_soh,
            source_path=profile.source_path,
            source_row_index=profile.source_row_index,
            source_locator=profile.source_locator,
        ))
    return devices


class FleetGenerator:
    def __init__(self, data_root: Path):
        self.catalog = ProfileCatalog(data_root, PARTITION_SEED)

    def build_fleet(
        self,
        weights: dict[str, float],
        sampling_mode: str,
        partition_name: str,
        fleet_seed: int,
        scenario: dict[str, Any] | None = None,
    ) -> tuple[list[GeneratedDevice], dict[str, Any]]:
        selected_weights = normalized_weights(weights)
        partitions = {
            dataset: getattr(self.catalog.partitions(dataset), partition_name)
            for dataset in selected_weights
        }
        if sampling_mode == "unique":
            capacities = {dataset: partition.source_count for dataset, partition in partitions.items()}
            fleet_size, counts = _max_unique_allocation(
                selected_weights, capacities, FLEET_SIZE
            )
        elif sampling_mode == "non_unique":
            fleet_size = FLEET_SIZE
            counts = allocate_counts(selected_weights, fleet_size)
            capacities = {dataset: partition.source_count for dataset, partition in partitions.items()}
        else:
            raise ValueError(f"未知 sampling mode：{sampling_mode}")

        devices = []
        audit = {}
        for dataset_index, (dataset, count) in enumerate(counts.items()):
            profile_seed = (
                fleet_seed + 10007 * (dataset_index + 1)
                + stable_seed(dataset) % 100000
            )
            profiles, sample_audit = _sample_profiles(
                partitions[dataset], count, profile_seed,
                unique=sampling_mode == "unique",
            )
            parameter_seed = profile_seed + 1000003 + stable_seed(dataset)
            generated = _synthetic_devices(
                profiles, dataset, parameter_seed, len(devices), scenario
            )
            devices.extend(generated)
            audit[dataset] = {
                "weight": selected_weights[dataset],
                "logical_devices": count,
                "profile_seed": profile_seed,
                "device_parameter_seed": parameter_seed,
                **sample_audit,
            }
        source_ids = [device.source_device_id for device in devices]
        source_counts = Counter(source_ids)
        duplicate_count = len(source_ids) - len(source_counts)
        if sampling_mode == "unique" and duplicate_count:
            raise RuntimeError("source-unique fleet 出现重复真实源")
        return devices, {
            "sampling_mode": sampling_mode,
            "partition": partition_name,
            "seed": fleet_seed,
            "fleet_size": fleet_size,
            "weights": selected_weights,
            "counts": counts,
            "source_capacities": capacities,
            "datasets": audit,
            "profile_bootstrap": any(row["profile_bootstrap"] for row in audit.values()),
            "duplicate_source_count": duplicate_count,
            "max_source_reuse": max(source_counts.values(), default=0),
        }

    def scenario_fleet(
        self, scenario_id: str, sampling_mode: str,
        partition_name: str, seed_index: int | None = None,
    ) -> tuple[list[GeneratedDevice], dict[str, Any]]:
        scenario = SCENARIOS[scenario_id]
        train_seed, validation_seed, test_start = SCENARIO_SEEDS[sampling_mode][scenario_id]
        if partition_name == "train":
            seed = train_seed
            weights = scenario["weights"]
            regime = "fixed"
        elif partition_name == "validation":
            seed = validation_seed
            weights = scenario["weights"]
            regime = "fixed"
        elif partition_name == "test" and seed_index is not None:
            if not 0 <= seed_index < TEST_SEED_COUNT:
                raise ValueError("test seed index 必须在 0 至 29")
            seed = test_start + seed_index
            weights, regime = shifted_weights(scenario["weights"], seed_index)
        else:
            raise ValueError("test partition 必须指定 seed index")
        devices, metadata = self.build_fleet(
            weights, sampling_mode, partition_name, seed, scenario
        )
        metadata.update({
            "combination_type": "predefined_scenario",
            "scenario": scenario_id,
            "scenario_label": scenario["label"],
            "seed_index": seed_index,
            "composition_regime": regime,
        })
        return devices, metadata

    def pairwise_fleet(
        self, pair_id: str, sampling_mode: str, seed_index: int,
    ) -> tuple[list[GeneratedDevice], dict[str, Any]]:
        pair = next(row for row in pair_specs() if row["pair_id"] == pair_id)
        seed = PAIR_SEED_BASE + pair["pair_index"] * PAIR_SEED_STRIDE + seed_index
        weights = {pair["dataset_a"]: 0.5, pair["dataset_b"]: 0.5}
        devices, metadata = self.build_fleet(
            weights, sampling_mode, "test", seed, None
        )
        metadata.update({
            "combination_type": "pairwise_equal_mix",
            "pair_id": pair_id,
            "pair_index": pair["pair_index"],
            "pair_label": pair["pair_label"],
            "dataset_a": pair["dataset_a"],
            "dataset_b": pair["dataset_b"],
            "seed_index": seed_index,
        })
        return devices, metadata


def write_fleet(
    output_dir: Path, devices: list[GeneratedDevice], metadata: dict[str, Any]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "devices.csv"
    metadata_path = output_dir / "metadata.json"
    fields = list(asdict(devices[0])) if devices else []
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(device) for device in devices)
    temporary_csv.replace(csv_path)
    payload = {
        "protocol": "standalone_fixed_seed_fleet_mix_v1",
        "csv_only": True,
        "profile_storage": "source references in devices.csv",
        **metadata,
    }
    temporary_json = metadata_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(metadata_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return {
        "relative_path": str(output_dir),
        "fleet_size": len(devices),
        "seed": metadata["seed"],
        "devices_csv_sha256": digest,
    }
