from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..run_experiment import run_default, run_resource_response_diagnostics


ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = ROOT / "results" / "ieee33_real_snapshot_simulation" / "network_weak_paired"
CONFIG_ROOT = ROOT / "src" / "extra" / "ieee33_device_day_simulation" / "experiments" / "snapshot" / "configs"

GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "time_aligned_network_weak",
        "config": CONFIG_ROOT / "default_network_weak.yaml",
        "capacity_multiplier": 5.0,
        "description": "Shared real time, fixed randomized device-day node assignment, active IEEE33 with five-fold capacity headroom.",
    },
    {
        "name": "time_aligned_network_stress_paired",
        "config": CONFIG_ROOT / "default_network_stress_paired.yaml",
        "capacity_multiplier": 1.0,
        "description": "The same shared real time and device-day node assignment, with the standard IEEE33 capacity limits.",
    },
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)


def run_suite(results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    index: dict[str, Any] = {
        "protocol": "paired_real_snapshot_active_network_weak_vs_stress",
        "plots_generated": False,
        "paired_control": "only network_capacity_multiplier changes",
        "weak_correlation_threshold": {
            "metric": "absolute mean conditional residual pairwise rho",
            "threshold": 0.01,
            "scaling_strict_threshold": "N_eff/N >= 0.80 at N=3000",
        },
        "groups": {},
    }
    for group in GROUPS:
        output = run_default(
            group["config"],
            mode="network_stress",
            results_root=results_root / group["name"],
            snapshot_protocol=True,
            snapshot_sampling="aligned",
            real_snapshot_only=True,
        )
        diagnostic = run_resource_response_diagnostics(
            group["config"],
            mode="network_stress",
            result_root=output,
            snapshot_sampling="aligned",
            replications=8,
            weak_rho_threshold=0.01,
        )
        index["groups"][group["name"]] = {
            "name": group["name"],
            "config": str(group["config"]),
            "capacity_multiplier": group["capacity_multiplier"],
            "description": group["description"],
            "result_dir": str(output),
            "data_dir": str(output / "data"),
            "estimation_dir": str(output / "estimation"),
            "resource_response_diagnostics": str(diagnostic),
        }
    _write_json(results_root / "run_index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paired real-snapshot network weak/stress data suite.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    args = parser.parse_args()
    run_suite(Path(args.results_root))


if __name__ == "__main__":
    main()
