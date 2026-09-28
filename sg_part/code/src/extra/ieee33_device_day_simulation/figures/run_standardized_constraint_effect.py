"""Run and plot a topology-standardized transformer constraint audit.

This experiment is intentionally separate from the original E22/E24 outputs.
It calibrates the IEEE-69 branch scale to the IEEE-33/123 reference median,
calibrates the IEEE-123 transformer scale to the IEEE-33/69 reference median,
and uses a fixed absolute transformer loading limit during dispatch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..network.local_radial_distflow import LocalRadialDistFlow, load_network_case
from . import fig4d_final_protocol as training
from . import run_e22_ieee69_complexity as e22
from . import run_e24_ieee123_audit as e24


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/E22/standardized_constraint_effect"
FIGURE_DIR = ROOT / "outputs/figs/subpanel"
TOPOLOGIES = ("ieee33", "ieee69", "ieee123")
CONSTRAINT_VALUES = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
FLEET_SIZE = 5000
DEFAULT_SEED_COUNT = 30
PROTOCOL = "E22_standardized_topology_constraint_effect_v1"

# Targets are the medians from the completed original reference audit.
TARGET_BRANCH_LOADING = 2.6073972667484906
TARGET_TRANSFORMER_LOADING = 1.1376747699233505
BRANCH_SCALE = {
    "ieee33": 1.0,
    "ieee69": 3.7164823801799014,
    "ieee123": 1.0,
}
TRANSFORMER_SCALE = {
    "ieee33": 1.0,
    "ieee69": 1.0,
    "ieee123": 1.089155734800626,
}
COLORS = {"ieee33": "#365A7C", "ieee69": "#55A6B5", "ieee123": "#E5633E"}
LABELS = {"ieee33": "IEEE-33", "ieee69": "IEEE-69", "ieee123": "IEEE-123"}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case(topology: str) -> dict[str, Any]:
    if topology == "ieee123":
        case = load_network_case(e24.NETWORK_FILE)
        layout = e22._radial_layout(case)
        buses = sorted(int(bus) for bus in layout["buses"])
        ordered = sorted(buses, key=lambda bus: (layout["distance"].get(bus, 0.0), bus))
        e22.ABSORPTION_ZONES["ieee123"] = tuple(
            tuple(int(bus) for bus in chunk) for chunk in np.array_split(ordered, 6)
        )
        midpoint = max(1, len(ordered) // 2)
        e22.DISTAL_GROUPS["ieee123"] = (
            tuple(ordered[:midpoint]),
            tuple(ordered[midpoint:]),
        )
        return case
    return e22._network_cases()[topology]


def _model_path(dataset: str, topology: str) -> Path:
    if topology == "ieee69":
        path = ROOT / "results/E22/trained_eps_ieee69_direct/models" / f"{dataset}.pt"
    elif topology == "ieee123":
        path = ROOT / "results/E24/trained_eps_ieee123_direct/models" / f"{dataset}.pt"
    elif dataset in {e22.NEXTGEN, e22.DATA2}:
        path = ROOT / "results/E22/trained_eps_ieee33_direct/models" / f"{dataset}.pt"
    else:
        path = (
            ROOT
            / f"results/{dataset}_ieee33_real_load/coverage_fix/network_constrained_new/data"
            / "fig4d_eps_estimator_final_fixed5000_ieee33_data_driven.pt"
        )
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {topology} 直接训练模型：{path}")
    return path


def _network(
    case: dict[str, Any],
    topology: str,
    constraint: str,
    value: float,
) -> LocalRadialDistFlow:
    transformer_multiplier = TRANSFORMER_SCALE[topology]
    transformer_limit = None
    if constraint == "Transformer":
        transformer_multiplier *= value
        transformer_limit = 1.0
    return LocalRadialDistFlow(
        case,
        capacity_multiplier=BRANCH_SCALE[topology],
        transformer_multiplier=transformer_multiplier,
        enforce_line_limit=constraint == "Line",
        enforce_transformer_limit=constraint == "Transformer",
        enforce_minimum_voltage=constraint == "Minimum voltage",
        enforce_maximum_voltage=constraint == "Maximum voltage",
        absolute_transformer_loading_limit=transformer_limit,
    )


def _scenario(
    dataset: str,
    topology: str,
    seed_index: int,
    case: dict[str, Any],
) -> tuple[list[Any], dict[str, Any], np.ndarray, dict[str, Any]]:
    base_seed = e22._base_seed(dataset, seed_index)
    records, config, _ = e22._sample_dataset(dataset, base_seed)
    reference_records, _ = e22._remap_records(
        records, case, base_seed + 31_000, "load_weighted"
    )
    records, placement = e22._remap_records(
        records, case, base_seed + 32_000, "uniform"
    )
    scenario = e22._build_scenario(
        records,
        reference_records,
        case,
        topology,
        "M4",
        base_seed + 33_000,
        placement,
    )
    availability = training.availability_probability(
        records, config, e22.mixed.AVAILABILITY_MODE
    )
    return records, config, availability, scenario


def _run_dataset_topology(
    dataset: str,
    topology: str,
    seed_count: int,
) -> dict[str, Any]:
    case = _case(topology)
    model_path = _model_path(dataset, topology)
    _, optimizer, _ = training.load_frozen_eps_controller(model_path)
    output_dir = OUTPUT / "data/raw" / topology / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = 1 + len(CONSTRAINT_VALUES)
    completed = 0
    for seed_index in range(seed_count):
        path = output_dir / f"seed_{seed_index:02d}.csv"
        if path.is_file():
            existing = pd.read_csv(path)
            if len(existing) == expected_rows and set(existing["protocol"]) == {PROTOCOL}:
                completed += 1
                continue
        records, config, availability, scenario = _scenario(
            dataset, topology, seed_index, case
        )
        base_seed = e22._base_seed(dataset, seed_index)
        rows: list[dict[str, Any]] = []
        reference_network = _network(case, topology, "Reference", 1.0)
        result, _ = e22._run_seed(
            "eps_ieee69_fused",
            records,
            config,
            scenario,
            availability,
            reference_network,
            base_seed + 410_000,
            base_seed + 420_000,
            optimizer,
            False,
        )
        rows.append(
            {
                **result,
                "protocol": PROTOCOL,
                "dataset": dataset,
                "dataset_label": e22.DATASET_LABELS[dataset],
                "topology": topology,
                "seed_index": seed_index,
                "constraint": "Reference",
                "constraint_value": 1.0,
                "branch_capacity_scale": BRANCH_SCALE[topology],
                "transformer_capacity_scale": TRANSFORMER_SCALE[topology],
                "absolute_transformer_loading_limit": "",
                "target_branch_loading": TARGET_BRANCH_LOADING,
                "target_transformer_loading": TARGET_TRANSFORMER_LOADING,
                "fixed_transformer_limit": False,
            }
        )
        for value in CONSTRAINT_VALUES:
            network = _network(case, topology, "Transformer", value)
            result, _ = e22._run_seed(
                "eps_ieee69_fused",
                records,
                config,
                scenario,
                availability,
                network,
                base_seed + 410_000,
                base_seed + 420_000,
                optimizer,
                False,
            )
            rows.append(
                {
                    **result,
                    "protocol": PROTOCOL,
                    "dataset": dataset,
                    "dataset_label": e22.DATASET_LABELS[dataset],
                    "topology": topology,
                    "seed_index": seed_index,
                    "constraint": "Transformer",
                    "constraint_value": value,
                    "branch_capacity_scale": BRANCH_SCALE[topology],
                    "transformer_capacity_scale": TRANSFORMER_SCALE[topology],
                    "absolute_transformer_loading_limit": 1.0,
                    "target_branch_loading": TARGET_BRANCH_LOADING,
                    "target_transformer_loading": TARGET_TRANSFORMER_LOADING,
                    "fixed_transformer_limit": True,
                }
            )
        _write_rows(path, rows)
        completed += 1
    return {
        "dataset": dataset,
        "topology": topology,
        "completed_seeds": completed,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
    }


def _summarize() -> pd.DataFrame:
    paths = sorted((OUTPUT / "data/raw").glob("*/*/seed_*.csv"))
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    metrics = [
        "mean_reduction_pct",
        "network_acceptance_ratio",
        "mean_local_clip_fraction",
        "maximum_branch_loading",
        "maximum_transformer_loading",
        "accepted_absorption_mwh",
        "remaining_curtailment_mwh",
    ]
    group_cols = ["topology", "constraint", "constraint_value"]
    summary = frame.groupby(group_cols, as_index=False)[metrics].agg(
        ["median", "mean"]
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary["seed_count"] = frame.groupby(group_cols).size().to_numpy()
    _write_rows(
        OUTPUT / "data/standardized_constraint_effect_summary.csv",
        summary.to_dict("records"),
    )
    return summary


def _plot(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    metrics = [
        ("mean_reduction_pct_median", "Curtailment absorption (%)"),
        ("network_acceptance_ratio_median", "Network acceptance ratio"),
        ("maximum_transformer_loading_median", "Maximum transformer loading"),
    ]
    for axis, (metric, ylabel) in zip(axes, metrics):
        for topology in TOPOLOGIES:
            selected = summary[
                (summary["topology"] == topology)
                & (summary["constraint"] == "Transformer")
            ].sort_values("constraint_value", ascending=False)
            x = selected["constraint_value"].to_numpy(float)
            y = selected[metric].to_numpy(float)
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=COLORS[topology],
                label=LABELS[topology],
            )
        axis.set_xlabel("Transformer capacity fraction")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0.68, 1.02)
        axis.set_xticks(CONSTRAINT_VALUES)
        axis.invert_xaxis()
        if metric == "network_acceptance_ratio_median":
            axis.set_ylim(0.0, 1.05)
        if metric == "maximum_transformer_loading_median":
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
            axis.set_ylim(0.0, 1.08)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Standardized topology comparison with fixed absolute transformer limit",
        fontsize=13,
    )
    fig.savefig(FIGURE_DIR / "08_standardized_constraint_audit.png", dpi=220)
    fig.savefig(FIGURE_DIR / "08_standardized_constraint_audit.pdf")
    plt.close(fig)


def _finalize(jobs: list[dict[str, Any]], seed_count: int) -> None:
    summary = _summarize()
    _plot(summary)
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "seed_count": seed_count,
        "topologies": TOPOLOGIES,
        "constraint_values": CONSTRAINT_VALUES,
        "target_branch_loading": TARGET_BRANCH_LOADING,
        "target_transformer_loading": TARGET_TRANSFORMER_LOADING,
        "branch_capacity_scale": BRANCH_SCALE,
        "transformer_capacity_scale": TRANSFORMER_SCALE,
        "transformer_rule": "fixed absolute loading limit: current transformer loading <= 1.0",
        "jobs": jobs,
        "figure": str(FIGURE_DIR / "08_standardized_constraint_audit.png"),
    }
    _write_json(OUTPUT / "manifest.json", manifest)
    _write_json(OUTPUT / "checkpoint.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dataset", action="append", choices=e22.E22_DATASETS)
    parser.add_argument("--topology", action="append", choices=TOPOLOGIES)
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="run jobs only and skip summary/figure generation",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="skip job execution and only rebuild summary/figure from existing raw data",
    )
    args = parser.parse_args()
    if not 1 <= args.seed_count <= DEFAULT_SEED_COUNT:
        raise ValueError("seed-count must be between 1 and 30")
    if not 1 <= args.workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        summary = _summarize()
        _plot(summary)
        jobs: list[dict[str, Any]] = []
        manifest = {
            "protocol": PROTOCOL,
            "status": "completed",
            "seed_count": args.seed_count,
            "topologies": TOPOLOGIES,
            "constraint_values": CONSTRAINT_VALUES,
            "target_branch_loading": TARGET_BRANCH_LOADING,
            "target_transformer_loading": TARGET_TRANSFORMER_LOADING,
            "branch_capacity_scale": BRANCH_SCALE,
            "transformer_capacity_scale": TRANSFORMER_SCALE,
            "transformer_rule": "fixed absolute loading limit: current transformer loading <= 1.0",
            "jobs": jobs,
            "figure": str(FIGURE_DIR / "08_standardized_constraint_audit.png"),
        }
        _write_json(OUTPUT / "manifest.json", manifest)
        _write_json(OUTPUT / "checkpoint.json", manifest)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(OUTPUT),
                    "figure": str(FIGURE_DIR / "08_standardized_constraint_audit.png"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    datasets = tuple(args.dataset) if args.dataset else tuple(e22.E22_DATASETS)
    topologies = tuple(args.topology) if args.topology else TOPOLOGIES
    jobs = [(dataset, topology) for dataset in datasets for topology in topologies]
    checkpoint_path = (
        OUTPUT / f"checkpoint_{'_'.join(topologies)}.json"
        if args.no_finalize
        else OUTPUT / "checkpoint.json"
    )
    completed: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(jobs)), mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                _run_dataset_topology, dataset, topology, args.seed_count
            ): (dataset, topology)
            for dataset, topology in jobs
        }
        for future in as_completed(futures):
            completed.append(future.result())
            _write_json(
                checkpoint_path,
                {
                    "protocol": PROTOCOL,
                    "status": "running",
                    "completed_jobs": len(completed),
                    "job_count": len(jobs),
                    "topologies": topologies,
                    "jobs": completed,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "dataset_topology_complete",
                        "completed_jobs": len(completed),
                        "job_count": len(jobs),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if args.no_finalize:
        _write_json(
            checkpoint_path,
            {
                "protocol": PROTOCOL,
                "status": "raw_complete",
                "completed_jobs": len(completed),
                "job_count": len(jobs),
                "topologies": topologies,
                "jobs": completed,
            },
        )
        print(
            json.dumps(
                {
                    "status": "raw_complete",
                    "output": str(OUTPUT),
                    "topologies": topologies,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if not args.no_finalize:
        _finalize(completed, args.seed_count)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(OUTPUT),
                    "figure": str(FIGURE_DIR / "08_standardized_constraint_audit.png"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
