from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import traceback
from typing import Any

from .run import (
    DATASETS,
    _capacity_multiplier,
    _compact_result_artifacts,
    _fit_experiment_to_pool,
    _mode_config,
    _prepare_config,
    _refresh_rho_scan,
    dataset_results_root,
)
from ..ieee33_device_day_simulation.config_loader import load_config
from ..ieee33_device_day_simulation.experiments.run_experiment import run_default
from ..ieee33_device_day_simulation.figures.plot_figures import plot_all
from ..ieee33_device_day_simulation.population.device_day_loader import load_device_day_pool


ROOT = Path("results")
STATUS_PATH = ROOT / "coverage_fix_full_rebuild_status.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def _run_one(dataset: str) -> tuple[str, bool, dict[str, Any]]:
    started = time.time()
    root = dataset_results_root(dataset) / "coverage_fix"
    item_status = root / "rebuild_status.json"
    try:
        _write_json(item_status, {"dataset": dataset, "phase": "configuration"})
        config_path = _prepare_config(dataset, root)
        config = load_config(config_path)
        pool = load_device_day_pool(config)
        fleet = _fit_experiment_to_pool(config_path, dataset, pool)
        config = load_config(config_path)
        multiplier = _capacity_multiplier(config, pool, target_loading=0.90)
        mode_config = _mode_config(config_path, "network_stress", multiplier)
        mode_root = root / "network_stress"

        _write_json(item_status, {"dataset": dataset, "phase": "main_experiment"})
        run_default(
            config_path=mode_config,
            mode="network_stress",
            results_root=mode_root,
            snapshot_protocol=True,
            snapshot_sampling="aligned",
        )

        _write_json(item_status, {"dataset": dataset, "phase": "figure5_rho_scan"})
        _refresh_rho_scan(mode_config, "network_stress", mode_root)
        figures = plot_all(mode_root, compact=True)
        _compact_result_artifacts(mode_root)

        estimator = json.loads(
            (mode_root / "estimation" / "estimation_validation_results.json").read_text(
                encoding="utf-8"
            )
        )
        rho = json.loads(
            (mode_root / "data" / "rho_sensitivity.json").read_text(encoding="utf-8")
        )
        details = {
            "elapsed_seconds": round(time.time() - started, 1),
            "resources": fleet["selected_main_resources"],
            "network_capacity_multiplier": multiplier,
            "r2": estimator["point_metrics"]["r2"]["value"],
            "rmse_kw": estimator["point_metrics"]["rmse_kw"],
            "picp": estimator["interval_metrics"]["picp"],
            "pinaw": estimator["interval_metrics"]["pinaw"],
            "rho_points": len(rho.get("rho_scan", {})),
            "figures": [str(path) for path in figures],
        }
        _write_json(item_status, {"dataset": dataset, "phase": "completed", **details})
        item_status.unlink(missing_ok=True)
        return dataset, True, details
    except BaseException as exc:
        details = {
            "elapsed_seconds": round(time.time() - started, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_json(item_status, {"dataset": dataset, "phase": "failed", **details})
        return dataset, False, details


def _refresh_fig5_one(dataset: str) -> tuple[str, bool, dict[str, Any]]:
    started = time.time()
    root = dataset_results_root(dataset) / "coverage_fix"
    mode_root = root / "network_stress"
    item_status = root / "rebuild_status.json"
    try:
        config_path = root / "config" / "default_network_stress.yaml"
        if not config_path.is_file() or not mode_root.is_dir():
            raise FileNotFoundError(f"existing coverage_fix result is incomplete: {root}")
        _write_json(item_status, {"dataset": dataset, "phase": "figure5_rho_scan"})
        _refresh_rho_scan(config_path, "network_stress", mode_root)
        figures = plot_all(mode_root, compact=True)
        _compact_result_artifacts(mode_root)
        rho = json.loads(
            (mode_root / "data" / "rho_sensitivity.json").read_text(encoding="utf-8")
        )
        details = {
            "elapsed_seconds": round(time.time() - started, 1),
            "rho_points": len(rho.get("rho_scan", {})),
            "iid_cv_sqrt_neff": rho["rho_scan"]["iid"]["cv_sqrt_neff"],
            "maximum_rho_cv_sqrt_neff": rho["rho_scan"]["rho~0.051"]["cv_sqrt_neff"],
            "figures": [str(path) for path in figures],
        }
        _write_json(item_status, {"dataset": dataset, "phase": "completed", **details})
        item_status.unlink(missing_ok=True)
        return dataset, True, details
    except BaseException as exc:
        details = {
            "elapsed_seconds": round(time.time() - started, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_json(item_status, {"dataset": dataset, "phase": "failed", **details})
        return dataset, False, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild coverage-fix outputs for every canonical legacy dataset.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="*", choices=list(DATASETS))
    parser.add_argument(
        "--fig5-only",
        action="store_true",
        help="overwrite Figure 5 diagnostics and figures in existing coverage_fix directories",
    )
    args = parser.parse_args()

    datasets = list(args.datasets or DATASETS)
    status: dict[str, Any] = {
        "protocol": "network_feasible_piecewise_estimator_all_datasets",
        "pending": datasets.copy(),
        "running": [],
        "completed": {},
        "failed": {},
    }
    _write_json(STATUS_PATH, status)

    worker = _refresh_fig5_one if args.fig5_only else _run_one
    status["protocol"] = (
        "original_experiments_persistent_correlation_fig5_refresh"
        if args.fig5_only
        else status["protocol"]
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, dataset): dataset for dataset in datasets}
        status["running"] = datasets[: args.workers]
        status["pending"] = datasets[args.workers :]
        _write_json(STATUS_PATH, status)
        for future in concurrent.futures.as_completed(futures):
            dataset, succeeded, details = future.result()
            destination = status["completed"] if succeeded else status["failed"]
            destination[dataset] = details
            status["running"] = [item for item in status["running"] if item != dataset]
            if status["pending"]:
                status["running"].append(status["pending"].pop(0))
            _write_json(STATUS_PATH, status)
            outcome = "DONE" if succeeded else "FAILED"
            print(f"=== {outcome} {dataset} {details['elapsed_seconds']}s ===", flush=True)

    print(json.dumps({
        "completed": list(status["completed"]),
        "failed": status["failed"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
