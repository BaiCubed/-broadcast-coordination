from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..run_experiment import run_default, run_resource_response_diagnostics


GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "independent_real_snapshot_no_network",
        "mode": "weak_correlation",
        "sampling": "independent",
        "network_feedback": False,
        "description": "Each resource samples an independent real within-day profile point; IEEE33 does not alter dispatch.",
    },
    {
        "name": "time_aligned_real_snapshot_no_network",
        "mode": "weak_correlation",
        "sampling": "aligned",
        "network_feedback": False,
        "description": "All resources use the same real profile time index; IEEE33 is diagnostic only.",
    },
    {
        "name": "time_aligned_real_snapshot_network",
        "mode": "network_stress",
        "sampling": "aligned",
        "network_feedback": True,
        "description": "All resources use the same real profile time index and IEEE33 dispatch limits are active.",
    },
    {
        "name": "time_aligned_network_weak",
        "mode": "network_stress",
        "sampling": "aligned",
        "network_feedback": True,
        "config": "src/extra/ieee33_device_day_simulation/experiments/snapshot/configs/default_network_weak.yaml",
        "description": "Shared real time index, randomized balanced device-day node assignment, active IEEE33 with five-fold network headroom.",
    },
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)


def run_suite(
    *,
    results_root: Path,
    config: str | Path | None = None,
    selected_group: str | None = None,
) -> dict[str, Any]:

    groups = [group for group in GROUPS if selected_group in {None, group["name"]}]
    if not groups:
        raise ValueError(f"unknown snapshot group: {selected_group}")
    index: dict[str, Any] = {
        "protocol": "real_snapshot_network_ablation",
        "plots_generated": False,
        "groups": {},
        "shared_settings": {
            "real_device_day_input": True,
            "synthetic_zone_correlation_injected": False,
            "device_model": "original_eps_simulator",
            "network_model": "existing_ieee33_outer_layer_only",
            "n_threshold_replications": 8,
            "n_scaling_runs_from_config": None,
        },
    }
    for group in groups:
        result_dir = results_root / group["name"]
        group_config = group.get("config", config)
        output = run_default(
            group_config,
            mode=group["mode"],
            results_root=result_dir,
            snapshot_protocol=True,
            snapshot_sampling=group["sampling"],
            real_snapshot_only=True,
        )
        diagnostic = run_resource_response_diagnostics(
            group_config,
            mode=group["mode"],
            result_root=output,
            snapshot_sampling=group["sampling"],
            replications=8,
        )
        index["groups"][group["name"]] = {
            **group,
            "result_dir": str(output),
            "data_dir": str(output / "data"),
            "estimation_dir": str(output / "estimation"),
            "resource_response_diagnostics": str(diagnostic),
        }
        metadata_path = output / "data" / "protocol_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        index["shared_settings"]["n_scaling_runs_from_config"] = metadata["shared_settings"]["n_scaling_runs"]
    _write_json(results_root / "run_index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-snapshot ablation data suite without generating figures.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-root", default="results/ieee33_real_snapshot_simulation/snapshot")
    parser.add_argument("--group", choices=[group["name"] for group in GROUPS], default=None)
    args = parser.parse_args()
    run_suite(
        results_root=Path(args.results_root),
        config=args.config,
        selected_group=args.group,
    )


if __name__ == "__main__":
    main()
