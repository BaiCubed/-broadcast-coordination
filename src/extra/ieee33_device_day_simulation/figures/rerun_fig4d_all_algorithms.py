from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as protocol


PROTOCOL = "fig4d_all_algorithms_v5_dataset_keyed_fleet_30_seeds"
FLEET_MODE = "fixed5000"
AVAILABILITY_MODE = "data_driven"
NETWORK_MODES = ("aggregate", "ieee33")
SEED_COUNT = 30


def _run_condition(result_root: Path, network_mode: str) -> dict[str, Any]:
    legacy.SEEDS = SEED_COUNT
    protocol.PROTOCOL = PROTOCOL
    return protocol.update_result_root(
        result_root,
        FLEET_MODE,
        network_mode,
        AVAILABILITY_MODE,
        force=True,
    )


def _worker(task: tuple[str, str]) -> dict[str, Any]:
    result_root, network_mode = task
    try:
        row = _run_condition(Path(result_root), network_mode)
        return {**row, "status": "completed", "seed_count": SEED_COUNT}
    except Exception as exc:
        return {
            "result_root": result_root,
            "network_mode": network_mode,
            "status": "failed",
            "error": repr(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for rerun fig4d all algorithms.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(3, (os.cpu_count() or 2) // 2)),
    )
    args = parser.parse_args()
    roots = protocol.discover_result_roots(args.results_root)
    tasks = [(str(root), mode) for root in roots for mode in NETWORK_MODES]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    manifest = {
        "protocol": PROTOCOL,
        "fleet_mode": FLEET_MODE,
        "availability_mode": AVAILABILITY_MODE,
        "seed_count": SEED_COUNT,
        "workers": args.workers,
        "condition_count": len(tasks),
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "strategies": list(legacy.FIG4D_ORDER),
        "results": rows,
    }
    manifest_path = args.results_root / "fig4d_all_algorithms_30seed_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(manifest_path)
    if manifest["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
