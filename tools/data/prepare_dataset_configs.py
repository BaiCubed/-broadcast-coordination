from __future__ import annotations

import argparse
import json
import sys
import traceback

from src.extra.dataset_experiment.hardware_preflight import run_preflight
from src.extra.dataset_experiment.run import (
    DATASETS,
    _capacity_multiplier,
    _fit_experiment_to_pool,
    _mode_config,
    _prepare_config,
    dataset_results_root,
)
from src.extra.ieee33_device_day_simulation.config_loader import load_config
from src.extra.ieee33_device_day_simulation.population.device_day_loader import load_device_day_pool


def prepare(dataset: str) -> dict:
    result_root = dataset_results_root(dataset)
    result_root.mkdir(parents=True, exist_ok=True)
    config_path = _prepare_config(dataset, result_root)
    config = load_config(config_path)
    pool = load_device_day_pool(config)
    fleet = _fit_experiment_to_pool(config_path, dataset, pool)
    config = load_config(config_path)
    preflight = run_preflight(config, pool)
    if preflight["status"] != "pass":
        return {
            "dataset": dataset,
            "status": "preflight_failed",
            "preflight": preflight,
            "fleet_calibration": fleet,
        }
    capacity = _capacity_multiplier(config, pool, target_loading=0.45)
    mode_path = _mode_config(config_path, "weak_correlation", capacity)
    payload = {
        "dataset": dataset,
        "status": "ready",
        "config": str(mode_path),
        "weak_correlation_capacity_multiplier": float(capacity),
        "preflight_status": preflight["status"],
        "balanced_unique_limit": fleet.get("balanced_unique_limit"),
        "selected_main_resources": fleet.get("selected_main_resources"),
        "with_replacement_bootstrap": fleet.get("with_replacement_bootstrap"),
        "unique_sources_by_zone": fleet.get("unique_sources_by_zone"),
    }
    (result_root / "e1e4_config_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare per-dataset IEEE33 configs for the E1-E4 supplementary experiments.")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    names = sorted(DATASETS) if args.all else (args.datasets or [])
    if not names:
        parser.error("provide --dataset NAME (repeatable) or --all")
    failed = 0
    for name in names:
        try:
            payload = prepare(name)
        except Exception as exc:
            payload = {
                "dataset": name,
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        if payload["status"] != "ready":
            failed += 1
        print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
