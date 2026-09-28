"""补齐并统一评估 IEEE-33 逐数据集直接训练 EPS 模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import run_e22_ieee69_complexity as e22
from . import run_e22_trained_eps_comparison as direct


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/E22/trained_eps_ieee33_direct"
PROTOCOL = "E22_ieee33_missing_dataset_direct_training_v1"
DEFAULT_DATASETS = (e22.NEXTGEN, e22.DATA2)
EVALUATION_STRESS_MODES = ("M0", "M4", "M5", "M6")


def _configure_runtime() -> None:
    """将现有逐数据集训练器限定到 IEEE-33。"""
    direct.TRAINING_TOPOLOGY = "ieee33"
    direct.PROTOCOL = PROTOCOL
    direct.OUTPUT_ROOT = OUTPUT


def _run_dataset(dataset: str, seed_count: int) -> dict[str, Any]:
    _configure_runtime()
    existing_model = (
        ROOT
        / f"results/{dataset}_ieee33_real_load/coverage_fix/network_constrained_new/data"
        / "fig4d_eps_estimator_final_fixed5000_ieee33_data_driven.pt"
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
        e22._network_cases(),
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
    e22._write_rows(OUTPUT / "data/trained_eps_by_seed.csv", rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=e22.E22_DATASETS)
    parser.add_argument("--seed-count", type=int, default=30)
    args = parser.parse_args()
    datasets = tuple(args.dataset) if args.dataset else DEFAULT_DATASETS
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = []
    for dataset in datasets:
        result = _run_dataset(dataset, args.seed_count)
        completed.append(result["metadata"])
        e22._write_json(
            OUTPUT / "checkpoint.json",
            {
                "protocol": PROTOCOL,
                "status": "running",
                "datasets": list(datasets),
                "completed": [row["dataset"] for row in completed],
            },
        )
    # raw目录可能包含前一轮已经完成的15个数据集；汇总时统一读取全部17个数据集。
    available_datasets = tuple(
        dataset
        for dataset in e22.E22_DATASETS
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
    e22._write_json(OUTPUT / "manifest.json", manifest)
    e22._write_json(OUTPUT / "checkpoint.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
