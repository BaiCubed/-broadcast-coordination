"""Build the canonical 288-point device-day caches from official source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from src.extra.dataset_experiment.canonical_adapter import load_canonical_device_day_pool  # noqa: E402


def _under_data_root(value: str, data_root: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if value.startswith("data/"):
        return str(data_root / Path(value).relative_to("data"))
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    payload = json.loads((ROOT / "configuration/preprocessing_config.json").read_text(encoding="utf-8"))
    zones = {"zones": {zone: {"buses": buses} for zone, buses in payload["zone_buses"].items()}}
    datasets = args.dataset or list(payload["datasets"])
    for dataset in datasets:
        spec = dict(payload["datasets"][dataset])
        for key in ("input", "input_dir", "solar_input", "cache"):
            if key in spec:
                spec[key] = _under_data_root(spec[key], data_root)
        cache = Path(spec.get("cache", "")) if spec.get("cache") else None
        if cache and cache.exists() and not args.force:
            print(json.dumps({"dataset": dataset, "status": "cached", "cache": str(cache)}))
            continue
        config = {
            "population": {
                "canonical_adapter": {
                    **spec,
                    "fallback_capacity_kwh": payload["fallback_device_parameters"]["capacity_kwh"],
                    "fallback_peak_power_kw": payload["fallback_device_parameters"]["peak_power_kw"],
                    "fallback_soc_fraction": payload["fallback_device_parameters"]["soc_fraction"],
                }
            },
            "zones": zones,
        }
        pool = load_canonical_device_day_pool(config)
        print(json.dumps({
            "dataset": dataset,
            "status": "created",
            "records": len(pool.records),
            "source_count": pool.source_count,
            "cache": str(cache) if cache else None,
        }))


if __name__ == "__main__":
    main()
