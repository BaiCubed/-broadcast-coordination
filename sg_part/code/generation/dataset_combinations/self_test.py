"""在不导入任何实验模块的条件下验证独立组合器。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile

from .generator import FleetGenerator, write_fleet


CORE_FIELDS = (
    "row_index", "dataset_id", "device_id", "source_device_id", "day_id",
    "zone_id", "bus_id", "capacity_kwh", "peak_power_kw", "c_rate",
    "initial_soc", "initial_soh",
)


def _assert_no_experiment_imports() -> None:
    root = Path(__file__).resolve().parent
    forbidden = ("ieee33_device_day_simulation.figures", "results/")
    for path in root.glob("*.py"):
        if path.name == "self_test.py":
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                raise RuntimeError(f"独立模块仍引用 {marker}：{path}")


def _compare_csv(generated: Path, reference: Path) -> None:
    with generated.open("r", encoding="utf-8", newline="") as handle:
        generated_rows = list(csv.DictReader(handle))
    with reference.open("r", encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    if len(generated_rows) != len(reference_rows):
        raise RuntimeError(f"行数不同：{generated} 与 {reference}")
    for row_index, (actual, expected) in enumerate(
        zip(generated_rows, reference_rows)
    ):
        for field in CORE_FIELDS:
            if actual[field] != expected[field]:
                raise RuntimeError(
                    f"{generated} 第 {row_index} 行字段 {field} 与参考数据不同"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    _assert_no_experiment_imports()
    generator = FleetGenerator(args.data_root)
    with tempfile.TemporaryDirectory(prefix="fleet_mix_self_test_") as temporary:
        output_root = Path(temporary)
        for mode in ("unique", "non_unique"):
            scenario_devices, scenario_metadata = generator.scenario_fleet(
                "S1-A", mode, "test", 0
            )
            scenario_output = output_root / mode / "ieee33/S1-A/test/seed_00"
            write_fleet(scenario_output, scenario_devices, scenario_metadata)
            pair_devices, pair_metadata = generator.pairwise_fleet("P001", mode, 0)
            pair_output = output_root / mode / "ieee33/pairwise/P001/seed_00"
            write_fleet(pair_output, pair_devices, pair_metadata)
            if mode == "non_unique" and (
                len(scenario_devices) != 5000 or len(pair_devices) != 5000
            ):
                raise RuntimeError("non_unique 冒烟 fleet 不是 5000 台")
            if mode == "unique":
                for devices in (scenario_devices, pair_devices):
                    sources = [device.source_device_id for device in devices]
                    if len(sources) != len(set(sources)):
                        raise RuntimeError("unique 冒烟 fleet 出现重复真实源")
            if args.reference_root:
                _compare_csv(
                    scenario_output / "devices.csv",
                    args.reference_root / mode / "ieee33/S1-A/test/seed_00/devices.csv",
                )
                _compare_csv(
                    pair_output / "devices.csv",
                    args.reference_root / mode / "ieee33/pairwise/P001/seed_00/devices.csv",
                )
    print("独立组合器自测通过")


if __name__ == "__main__":
    main()
