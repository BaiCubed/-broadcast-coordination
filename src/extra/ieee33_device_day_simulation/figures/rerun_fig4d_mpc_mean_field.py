from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
from typing import Any

from ..config_loader import load_config
from ..population.device_day_loader import load_device_day_pool
from . import fig4d_extra_baselines as legacy
from .fig4d_final_protocol import (
    _condition_config,
    _fleet_size,
    availability_probability,
    build_pure_sim_scenario,
    partition_profile_pool,
    sample_device_fleet,
)


PROTOCOL = "fig4d_mpc_mean_field_v4_dataset_keyed_fleet_coarse_mean_field_30_seeds"
STRATEGIES = ("mpc_optimal", "mean_field_control")
FLEET_MODE = "fixed5000"
AVAILABILITY_MODE = "data_driven"
NETWORK_MODES = ("aggregate", "ieee33")
SEED_COUNT = 30


def _config_path(result_root: Path) -> Path:
    return legacy._choose_config_path(result_root)


def _data_path(result_root: Path, network_mode: str) -> Path:
    stem = f"final_{FLEET_MODE}_{network_mode}_{AVAILABILITY_MODE}"
    return result_root / "data" / f"curtailment_baselines_{stem}.json"


def _is_complete(result_root: Path, network_mode: str) -> bool:
    path = _data_path(result_root, network_mode)
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    rerun = payload.get("partial_rerun", {})
    return (
        rerun.get("protocol") == PROTOCOL
        and rerun.get("paired_seed_count") == SEED_COUNT
        and all(
            len(payload.get("results", {}).get(strategy, {}).get("seed_results", []))
            == SEED_COUNT
            for strategy in STRATEGIES
        )
    )


def _completed_row(result_root: Path, network_mode: str) -> dict[str, Any]:
    path = _data_path(result_root, network_mode)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "dataset": result_root.parents[1].name.removesuffix("_ieee33_real_load"),
        "network_mode": network_mode,
        "seed_count": SEED_COUNT,
        "mpc_mean_pct": payload["results"]["mpc_optimal"]["mean_reduction_pct"],
        "mean_field_mean_pct": payload["results"]["mean_field_control"]["mean_reduction_pct"],
        "data": str(path),
    }


def rerun_condition(result_root: Path, network_mode: str) -> dict[str, Any]:
    path = _data_path(result_root, network_mode)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    original_config = load_config(_config_path(result_root))
    config = _condition_config(original_config, network_mode, AVAILABILITY_MODE)
    pool = load_device_day_pool(config)
    dataset_id = str(
        config["population"].get("canonical_adapter", {}).get("dataset", "unknown")
    )
    partitions = partition_profile_pool(pool, int(config["simulation"]["random_seed"]))
    fleet_size = _fleet_size(pool, FLEET_MODE)
    rows = {strategy: [] for strategy in STRATEGIES}
    for seed_index in range(SEED_COUNT):
        seed = int(config["simulation"]["random_seed"]) + seed_index
        records, fleet_metadata = sample_device_fleet(
            partitions.test,
            FLEET_MODE,
            fleet_size,
            seed,
            dataset_id=dataset_id,
        )
        scenario = build_pure_sim_scenario(records, config)
        availability = availability_probability(records, config, AVAILABILITY_MODE)
        for strategy in STRATEGIES:
            rows[strategy].append(legacy._run_seed(
                strategy,
                records,
                config,
                scenario,
                availability,
                seed + 100000 * (legacy.FIG4D_ORDER.index(strategy) + 1),
                seed + 1910000,
            ))
    for strategy in STRATEGIES:
        payload["results"][strategy] = legacy._summarize(rows[strategy])
        payload["strategy_definitions"][strategy] = legacy.STRATEGY_DEFINITIONS[strategy]
    payload.setdefault("baseline_design", {}).update({
        "mpc_optimal": {
            "horizon_steps": legacy.MPC_HORIZON_STEPS,
            "horizon_minutes": legacy.MPC_HORIZON_STEPS * 5,
            "energy_budget": "grid-side remaining capacity = stored-energy headroom / charge efficiency",
            "availability_forecast": "data-driven per-device probability over the prediction horizon",
            "allocation": "low-SOC weighted water filling",
        },
        "mean_field_control": {
            "soc_bins": legacy.MEAN_FIELD_SOC_BINS,
            "state": "predicted count, aggregate capacity, aggregate power, and aggregate energy per SOC bin",
            "control_interval_minutes": legacy.MEAN_FIELD_CONTROL_STEPS * 5,
            "availability": "expected fleet availability; no real-time device availability aggregation",
            "control": "one held participation probability per SOC bin",
            "allocation": "independent local random response; no post-response target normalization",
        },
    })
    payload["partial_rerun"] = {
        "protocol": PROTOCOL,
        "strategies": list(STRATEGIES),
        "paired_seed_count": SEED_COUNT,
        "seed_rule": "simulation.random_seed + seed_index, seed_index in [0, 29]",
        "dataset_id": dataset_id,
        "device_parameter_protocol": fleet_metadata["device_parameter_protocol"],
        "device_parameter_seed_rule": "simulation seed + 1000003 + stable_hash(dataset_id)",
        "last_seed_fleet_summary": {
            key: fleet_metadata[key]
            for key in (
                "mean_capacity_kwh",
                "total_capacity_kwh",
                "mean_peak_power_kw",
                "total_peak_power_kw",
            )
        },
        "other_strategy_results_preserved": True,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    stem = path.stem.removeprefix("curtailment_baselines_")
    figure = result_root / "Figs" / f"fig4-2-{stem.replace('_', '-')}.png"
    legacy.plot_fig4d_comparison(payload, figure)
    paper = result_root / "paper_figures" / figure.name
    if paper.parent.is_dir():
        shutil.copy2(figure, paper)
    return {
        "dataset": result_root.parents[1].name.removesuffix("_ieee33_real_load"),
        "network_mode": network_mode,
        "seed_count": SEED_COUNT,
        "mpc_mean_pct": payload["results"]["mpc_optimal"]["mean_reduction_pct"],
        "mean_field_mean_pct": payload["results"]["mean_field_control"]["mean_reduction_pct"],
        "data": str(path),
        "figure": str(figure),
    }


def _worker(task: tuple[str, str]) -> dict[str, Any]:
    result_root, network_mode = task
    try:
        return {"status": "completed", **rerun_condition(Path(result_root), network_mode)}
    except Exception as exc:
        return {
            "status": "failed",
            "result_root": result_root,
            "network_mode": network_mode,
            "error": repr(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for rerun fig4d mpc mean field.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--network-mode", choices=NETWORK_MODES)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()
    if args.result_root:
        modes = (args.network_mode,) if args.network_mode else NETWORK_MODES
        output = [rerun_condition(args.result_root, mode) for mode in modes]
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    roots = sorted(
        path for path in args.results_root.glob(
            "*_ieee33_real_load/coverage_fix/network_constrained_new"
        ) if all(_data_path(path, mode).is_file() for mode in NETWORK_MODES)
    )
    all_conditions = [(root, mode) for root in roots for mode in NETWORK_MODES]
    tasks = [
        (str(root), mode)
        for root, mode in all_conditions
        if not _is_complete(root, mode)
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    completed_rows = [
        _completed_row(root, mode)
        for root, mode in all_conditions
        if _is_complete(root, mode)
    ]
    failed_rows = [row for row in rows if row["status"] == "failed"]
    manifest = {
        "protocol": PROTOCOL,
        "workers": args.workers,
        "condition_count": len(all_conditions),
        "executed_this_run_count": len(tasks),
        "completed_count": len(completed_rows),
        "failed_count": len(failed_rows),
        "mixed_seed_note": "MPC and Mean-field use 30 seeds; other preserved strategies retain their original 10-seed results.",
        "results": completed_rows + failed_rows,
    }
    manifest_path = args.results_root / "fig4d_mpc_mean_field_30seed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)
    if manifest["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
