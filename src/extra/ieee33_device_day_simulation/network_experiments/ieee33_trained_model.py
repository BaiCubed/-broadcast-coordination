from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import ieee69_network_implementation as implementation
from . import ieee69_trained_vs_transferred as direct


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/ieee69_network_implementation/trained_eps_ieee33_direct"
PROTOCOL = "ieee33_trained_model_v1"
DEFAULT_DATASETS = (implementation.NEXTGEN, implementation.DATA2)
EVALUATION_STRESS_MODES = ("M0", "M4", "M5", "M6")


def _configure_runtime() -> None:
    direct.TRAINING_TOPOLOGY = "ieee33"
    direct.PROTOCOL = PROTOCOL
    direct.OUTPUT_ROOT = OUTPUT


def _run_dataset(dataset: str, seed_count: int) -> dict[str, Any]:
    _configure_runtime()
    existing_model = (
        ROOT
        / f"results/{dataset}_ieee33_real_load/network_baseline/network_constrained/data"
        / "eps_estimator_final_fixed5000_ieee33_data_driven.pt"
    )
    if not existing_model.is_file():
        return direct._run_dataset_job(
            dataset,
            str(OUTPUT),
            seed_count,
            ("ieee33",),
            EVALUATION_STRESS_MODES,
            False,
            False,
            None,
            None,
        )
    digest = hashlib.sha256(existing_model.read_bytes()).hexdigest()
    direct._run_dataset(
        dataset,
        OUTPUT,
        existing_model,
        digest,
        implementation._network_cases(),
        seed_count,
        ("ieee33",),
        EVALUATION_STRESS_MODES,
        False,
    )
    return {
        "dataset": dataset,
        "metadata": {
            "dataset": dataset,
            "model_path": str(existing_model),
            "model_sha256": digest,
            "model_reused": True,
        },
    }


def _merge_results(datasets: tuple[str, ...]) -> int:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        path = OUTPUT / "raw" / f"{dataset}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(
            row
            for row in payload["seed_results"]
            if row["algorithm"] == "eps_ieee69_fused"
        )
    implementation._write_rows(OUTPUT / "data/trained_eps_by_seed.csv", rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete and evaluate the per-dataset IEEE-33 directly trained EPS models.")
    parser.add_argument("--dataset", action="append", choices=implementation.NETWORK_DATASETS)
    parser.add_argument("--seed-count", type=int, default=30)
    args = parser.parse_args()
    datasets = tuple(args.dataset) if args.dataset else DEFAULT_DATASETS
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = []
    for dataset in datasets:
        result = _run_dataset(dataset, args.seed_count)
        completed.append(result["metadata"])
        implementation._write_json(
            OUTPUT / "checkpoint.json",
            {
                "protocol": PROTOCOL,
                "status": "running",
                "datasets": list(datasets),
                "completed": [row["dataset"] for row in completed],
            },
        )
    available_datasets = tuple(
        dataset
        for dataset in implementation.NETWORK_DATASETS
        if (OUTPUT / "raw" / f"{dataset}.json").is_file()
    )
    row_count = _merge_results(available_datasets)
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "datasets": list(available_datasets),
        "models": [row["model_path"] for row in completed],
        "evaluation_stress_modes": list(EVALUATION_STRESS_MODES),
        "row_count": row_count,
    }
    implementation._write_json(OUTPUT / "manifest.json", manifest)
    implementation._write_json(OUTPUT / "checkpoint.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
