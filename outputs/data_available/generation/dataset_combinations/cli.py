"""独立 fleet 组合器的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import SCENARIOS, TEST_SEED_COUNT
from .generator import FleetGenerator, pair_specs, write_fleet


def _modes(value: str) -> tuple[str, ...]:
    return ("unique", "non_unique") if value == "both" else (value,)


def _scenario_partitions(value: str) -> tuple[str, ...]:
    return ("train", "validation", "test") if value == "all" else (value,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", choices=("scenarios", "pairwise", "all"))
    parser.add_argument("--sampling-mode", choices=("unique", "non_unique", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/dataset_combinations")
    )
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--pair-id", choices=tuple(row["pair_id"] for row in pair_specs()))
    parser.add_argument("--partition", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--seed-index", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    generator = FleetGenerator(args.data_root)
    completed = []
    modes = _modes(args.sampling_mode)
    if args.pipeline in {"scenarios", "all"}:
        scenario_ids = (args.scenario,) if args.scenario else tuple(SCENARIOS)
        for mode in modes:
            for scenario_id in scenario_ids:
                for partition in _scenario_partitions(args.partition):
                    seed_indices = (
                        (args.seed_index,)
                        if partition == "test" and args.seed_index is not None
                        else (range(TEST_SEED_COUNT) if partition == "test" else (None,))
                    )
                    for seed_index in seed_indices:
                        suffix = (
                            f"test/seed_{seed_index:02d}"
                            if partition == "test"
                            else partition
                        )
                        output = args.output_root / mode / "ieee33" / scenario_id / suffix
                        if (output / "devices.csv").is_file() and not args.force:
                            continue
                        devices, metadata = generator.scenario_fleet(
                            scenario_id, mode, partition, seed_index
                        )
                        completed.append(write_fleet(output, devices, metadata))
    if args.pipeline in {"pairwise", "all"}:
        pairs = (
            [next(row for row in pair_specs() if row["pair_id"] == args.pair_id)]
            if args.pair_id else pair_specs()
        )
        seed_indices = (
            (args.seed_index,) if args.seed_index is not None else range(TEST_SEED_COUNT)
        )
        for mode in modes:
            for pair in pairs:
                for seed_index in seed_indices:
                    output = (
                        args.output_root / mode / "ieee33" / "pairwise"
                        / pair["pair_id"] / f"seed_{seed_index:02d}"
                    )
                    if (output / "devices.csv").is_file() and not args.force:
                        continue
                    devices, metadata = generator.pairwise_fleet(
                        pair["pair_id"], mode, seed_index
                    )
                    completed.append(write_fleet(output, devices, metadata))
    manifest = {
        "protocol": "standalone_fixed_seed_fleet_mix_v1",
        "status": "completed",
        "pipeline": args.pipeline,
        "sampling_mode": args.sampling_mode,
        "data_root": str(args.data_root.resolve()),
        "generated_fleets": len(completed),
        "output_root": str(args.output_root),
        "entries": completed,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "last_generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "completed",
        "generated_fleets": len(completed),
        "output_root": str(args.output_root),
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
