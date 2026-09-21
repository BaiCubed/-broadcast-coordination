"""直接从标准数据目录建立轻量 profile 引用池。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import BDG1, DATASET_SOURCES, ZONE_BUSES, ZONE_ORDER


@dataclass(frozen=True)
class ProfileRef:
    dataset_id: str
    source_device_id: str
    day_id: int
    device_id: str
    zone_id: str
    bus_id: int
    source_path: str
    source_row_index: int
    source_locator: str


@dataclass(frozen=True)
class ProfilePartition:
    records: tuple[ProfileRef, ...]

    @property
    def source_count(self) -> int:
        return len({record.source_device_id for record in self.records})


@dataclass(frozen=True)
class ProfilePartitions:
    train: ProfilePartition
    validation: ProfilePartition
    test: ProfilePartition


def stable_seed(value: str) -> int:
    import hashlib

    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8], "little"
    )


def _zone_bus(source_index: int) -> tuple[str, int]:
    zone = ZONE_ORDER[source_index % len(ZONE_ORDER)]
    buses = ZONE_BUSES[zone]
    return zone, int(buses[(source_index // len(ZONE_ORDER)) % len(buses)])


def _load_npz_refs(dataset_id: str, path: Path, relative: str) -> list[ProfileRef]:
    with np.load(path, allow_pickle=True) as archive:
        source_ids = archive["source_id"]
        day_ids = archive["day_id"]
        device_ids = archive["device_id"]
        zone_ids = archive["zone_id"]
        bus_ids = archive["bus_id"]
    return [
        ProfileRef(
            dataset_id=dataset_id,
            source_device_id=str(source_ids[index]),
            day_id=int(day_ids[index]),
            device_id=str(device_ids[index]),
            zone_id=str(zone_ids[index]),
            bus_id=int(bus_ids[index]),
            source_path=relative,
            source_row_index=index,
            source_locator=str(index),
        )
        for index in range(len(source_ids))
    ]


def _valid_bdg1_days(series: pd.Series, maximum: int = 20) -> list[str]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series.groupby(level=0).mean().sort_index()
    if series.empty:
        return []
    days = []
    for day in sorted(series.index.normalize().unique()):
        start = pd.Timestamp(day)
        target = pd.date_range(start, periods=288, freq="5min")
        if series.index.min() > start or series.index.max() < target[-1]:
            continue
        joined = series.reindex(series.index.union(target)).sort_index()
        values = joined.interpolate(limit_direction="both").reindex(target).to_numpy()
        if values.size == 288 and np.isfinite(values).all():
            days.append(start.strftime("%Y-%m-%d"))
        if len(days) >= maximum:
            break
    return days


def _load_bdg1_refs(path: Path, relative: str) -> list[ProfileRef]:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    timestamps = pd.to_datetime(frame.pop("timestamp"), errors="coerce")
    records = []
    for source_index, column in enumerate(frame.columns[:507]):
        series = pd.Series(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(),
            index=timestamps,
        )
        for day_id, day in enumerate(_valid_bdg1_days(series)):
            zone, bus = _zone_bus(source_index)
            records.append(ProfileRef(
                dataset_id=BDG1,
                source_device_id=str(column),
                day_id=day_id,
                device_id=f"{column}_day_{day_id:03d}",
                zone_id=zone,
                bus_id=bus,
                source_path=relative,
                source_row_index=-1,
                source_locator=f"{column}|{day}",
            ))
    if len({record.source_device_id for record in records}) < 500:
        raise ValueError("BDG1 可用独立源少于 500")
    return records


def partition_profiles(
    records: Iterable[ProfileRef], seed: int
) -> ProfilePartitions:
    grouped: dict[str, list[ProfileRef]] = {}
    for record in records:
        grouped.setdefault(record.source_device_id, []).append(record)
    splits: dict[str, list[ProfileRef]] = {
        "train": [], "validation": [], "test": [],
    }
    for source_id, source_records in sorted(grouped.items()):
        ordered = sorted(source_records, key=lambda row: (row.day_id, row.device_id))
        rng = np.random.default_rng(seed + stable_seed(source_id))
        ordered = [ordered[index] for index in rng.permutation(len(ordered))]
        count = len(ordered)
        if count == 1:
            for key in splits:
                splits[key].append(ordered[0])
            continue
        if count == 2:
            splits["train"].append(ordered[0])
            splits["validation"].append(ordered[1])
            splits["test"].append(ordered[1])
            continue
        train_count = max(1, int(np.floor(0.70 * count)))
        validation_count = max(1, int(np.floor(0.15 * count)))
        if train_count + validation_count >= count:
            train_count = count - 2
            validation_count = 1
        splits["train"].extend(ordered[:train_count])
        splits["validation"].extend(
            ordered[train_count:train_count + validation_count]
        )
        splits["test"].extend(ordered[train_count + validation_count:])
    return ProfilePartitions(
        train=ProfilePartition(tuple(splits["train"])),
        validation=ProfilePartition(tuple(splits["validation"])),
        test=ProfilePartition(tuple(splits["test"])),
    )


class ProfileCatalog:
    def __init__(self, data_root: Path, partition_seed: int):
        self.data_root = data_root.resolve()
        self.partition_seed = int(partition_seed)
        self._cache: dict[str, ProfilePartitions] = {}

    def partitions(self, dataset_id: str) -> ProfilePartitions:
        if dataset_id in self._cache:
            return self._cache[dataset_id]
        spec = DATASET_SOURCES[dataset_id]
        relative = str(spec["path"])
        path = self.data_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if spec["kind"] == "canonical_npz":
            records = _load_npz_refs(dataset_id, path, relative)
        elif spec["kind"] == "bdg1_csv":
            records = _load_bdg1_refs(path, relative)
        else:
            raise ValueError(f"未知数据源类型：{spec['kind']}")
        self._cache[dataset_id] = partition_profiles(records, self.partition_seed)
        return self._cache[dataset_id]
