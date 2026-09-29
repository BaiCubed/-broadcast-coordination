from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import ieee69_network_implementation as implementation
from . import ieee69_trained_vs_transferred as direct
from . import ieee123_safety_audit as audit


ROOT = Path(__file__).resolve().parents[4]
TOPOLOGY = "ieee123"
OUTPUT = ROOT / "results/ieee123_safety_audit/trained_eps_ieee123_direct"
PROTOCOL = "ieee123_trained_model_v1"


def _configure_runtime() -> dict[str, Any]:
    case = implementation.load_network_case(audit.NETWORK_FILE)
    layout = implementation._radial_layout(case)
    buses = sorted(int(bus) for bus in layout["buses"])
    ordered = sorted(buses, key=lambda bus: (layout["distance"].get(bus, 0.0), bus))
    zones = tuple(tuple(int(bus) for bus in chunk) for chunk in np.array_split(ordered, 6))
    midpoint = max(1, len(ordered) // 2)
    implementation.TOPOLOGIES = (TOPOLOGY,)
    implementation.NETWORK_FILES = {TOPOLOGY: audit.NETWORK_FILE}
    implementation.DISTAL_GROUPS[TOPOLOGY] = (tuple(ordered[:midpoint]), tuple(ordered[midpoint:]))
    implementation.ABSORPTION_ZONES[TOPOLOGY] = zones
    branch_keys = [f"{int(row[0])}-{int(row[1])}" for row in case["branches"]]
    implementation.DERATINGS[TOPOLOGY] = {
        key: value for key, value in zip(branch_keys, (0.65, 0.55, 0.60))
    }

    def network_cases() -> dict[str, dict[str, Any]]:
        return {TOPOLOGY: case}

    implementation._network_cases = network_cases
    direct.TRAINING_TOPOLOGY = TOPOLOGY
    direct.PROTOCOL = PROTOCOL
    direct.OUTPUT_ROOT = OUTPUT
    for definition in direct.NATIVE_DEFINITIONS.values():
        definition["label"] = definition["label"].replace("IEEE-69", "IEEE-123")
        definition["description"] = definition["description"].replace("IEEE-69", "IEEE-123")
    direct.NATIVE_DEFINITION["label"] = "EPS IEEE-123 fused"
    direct.NATIVE_DEFINITION["description"] = "Each dataset is trained directly on IEEE-123 M0-M6; testing keeps O(1) broadcast control."
    for algorithm, definition in direct.NATIVE_DEFINITIONS.items():
        implementation.mixed.ALGORITHM_DEFINITIONS[algorithm] = definition
    return case


def _run_dataset_job(
    dataset: str,
    seed_count: int,
    stress_modes: tuple[str, ...],
    force: bool,
    retrain_all: bool,
    training_samples: int | None,
    validation_samples: int | None,
) -> dict[str, Any]:
    _configure_runtime()
    return direct._run_dataset_job(
        dataset,
        str(OUTPUT),
        seed_count,
        (TOPOLOGY,),
        stress_modes,
        force,
        retrain_all,
        training_samples,
        validation_samples,
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    implementation._write_rows(path, rows)


def _plot_training_r2(metadata: list[dict[str, Any]]) -> None:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [implementation.DATASET_LABELS[row["dataset"]] for row in metadata]
    values = [float(row.get("held_out_r2", float("nan"))) for row in metadata]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(np.arange(len(labels)), values, color="#1f5aa6")
    axis.axhline(0.95, color="#d1495b", linestyle="--", linewidth=1.2, label="R²=0.95")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_ylim(min(-0.1, float(np.nanmin(values)) - 0.05), 1.05)
    axis.set_ylabel("Held-out R²")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "ieee123_direct_training_r2.png", dpi=220)
    figure.savefig(figure_dir / "ieee123_direct_training_r2.pdf")
    plt.close(figure)


def _write_manifest(metadata: list[dict[str, Any]], seed_count: int, stress_modes: tuple[str, ...]) -> None:
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "network": "IEEE-123",
        "network_representation": "single_phase_structural_equivalent_for_training",
        "direct_ieee123_training": True,
        "migration_model": False,
        "training_stress_modes": list(stress_modes),
        "seed_count": seed_count,
        "datasets": [row["dataset"] for row in metadata],
        "model_directory": "models",
        "multiphase_validation_entrypoint": "src/extra/ieee33_device_day_simulation/network_experiments/ieee123_opendss_check.py",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct EPS training on the IEEE-123 target network.")
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-datasets", type=int, default=len(implementation.NETWORK_DATASETS))
    parser.add_argument("--max-stress-modes", type=int, default=len(implementation.STRESS_MODES))
    parser.add_argument("--training-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-all", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_count <= 30:
        raise ValueError("seed-count must be between 1 and 30")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if args.training_samples is not None and args.training_samples < 256:
        raise ValueError("training-samples must be at least 256 so that both the charge and discharge branches have training samples")
    if args.validation_samples is not None and args.validation_samples < 64:
        raise ValueError("validation-samples must be at least 64")
    _configure_runtime()
    datasets = implementation.NETWORK_DATASETS[: max(1, min(args.max_datasets, len(implementation.NETWORK_DATASETS)))]
    stress_modes = tuple(implementation.STRESS_MODES)[: max(1, min(args.max_stress_modes, len(implementation.STRESS_MODES)))]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata_by_dataset: dict[str, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(datasets)), mp_context=context) as executor:
        jobs = {
            executor.submit(
                _run_dataset_job,
                dataset,
                args.seed_count,
                stress_modes,
                args.force,
                args.retrain_all,
                args.training_samples,
                args.validation_samples,
            ): dataset
            for dataset in datasets
        }
        for future in as_completed(jobs):
            dataset = jobs[future]
            result = future.result()
            metadata_by_dataset[dataset] = result["metadata"]
            implementation._write_json(OUTPUT / "data/training_metadata.json", [metadata_by_dataset[name] for name in datasets if name in metadata_by_dataset])
            print(json.dumps({"stage": "dataset_complete", "dataset": dataset, "completed": len(metadata_by_dataset), "total": len(datasets)}, ensure_ascii=False), flush=True)

    metadata = [metadata_by_dataset[dataset] for dataset in datasets]
    payloads = [json.loads((OUTPUT / "raw" / f"{dataset}.json").read_text(encoding="utf-8")) for dataset in datasets]
    rows = [row for payload in payloads for row in payload["seed_results"]]
    summaries = implementation._summaries(rows, 2000)
    _write_rows(OUTPUT / "data/trained_eps_by_seed.csv", rows)
    _write_rows(OUTPUT / "data/trained_eps_summary.csv", summaries)
    _plot_training_r2(metadata)
    _write_manifest(metadata, args.seed_count, stress_modes)
    implementation._write_json(OUTPUT / "checkpoint.json", {"protocol": PROTOCOL, "status": "completed", "dataset_count": len(datasets), "seed_count": args.seed_count, "row_count": len(rows)})
    print(json.dumps({"status": "complete", "datasets": len(datasets), "rows": len(rows), "summary_rows": len(summaries)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
