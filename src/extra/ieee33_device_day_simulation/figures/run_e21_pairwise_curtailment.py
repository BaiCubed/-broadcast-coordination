from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from src.signal import SignalOptimizer

from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as protocol
from . import run_e20_transfer as transfer
from . import run_e21_mixed_scenarios as mixed


PROTOCOL = "E21_pairwise_equal_mix_curtailment_v1"
UNIQUE_PROTOCOL = "E21_pairwise_equal_mix_curtailment_source_unique_v1"
UNIQUE_DIAGONAL_PROTOCOL = "E21_single_dataset_curtailment_source_unique_v1"
DEFAULT_OUTPUT = Path("results/E21/pairwise_curtailment")
DEFAULT_MODEL = Path("results/E21/models/e21_mixed_aggregate.pt")
DEFAULT_UNIQUE_MODEL = Path("results/E21/unique/models/e21_unique_mixed_aggregate.pt")
DEFAULT_FLEET_SIZE = 5000
DEFAULT_SEED_COUNT = 30
DEFAULT_BOOTSTRAP_DRAWS = 4000
EPS = 1e-12

SHORT_NAMES = {
    mixed.BDG1: "BDG1",
    mixed.BDG2: "BDG2",
    mixed.LCL: "LCL",
    mixed.CAMSL: "CAMSL",
    mixed.IRISH: "Irish",
    mixed.GOIENER: "GoiEner",
    mixed.SGSC: "SGSC",
    mixed.DANISH: "Danish",
    mixed.HEAPO: "HEAPO",
    mixed.EU_RURAL: "EU-Rural",
    mixed.EU_35297: "EU-35k",
    mixed.EU_8087: "EU-8k",
    mixed.NORWAY: "Norway",
    mixed.CEC: "CEC",
    mixed.OPSD: "OPSD",
}

PLOT_ORDER = (
    mixed.BDG1,
    mixed.BDG2,
    mixed.LCL,
    mixed.CAMSL,
    mixed.IRISH,
    mixed.GOIENER,
    mixed.SGSC,
    mixed.DANISH,
    mixed.HEAPO,
    mixed.EU_RURAL,
    mixed.EU_35297,
    mixed.EU_8087,
    mixed.NORWAY,
    mixed.CEC,
    mixed.OPSD,
)

_WORKER_MODEL_PATH: str | None = None
_WORKER_OPTIMIZER: SignalOptimizer | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pair_specs() -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for first_index, dataset_a in enumerate(PLOT_ORDER):
        for dataset_b in PLOT_ORDER[first_index + 1 :]:
            pairs.append(
                {
                    "pair_index": len(pairs),
                    "pair_id": f"P{len(pairs) + 1:03d}",
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "pair_label": f"{SHORT_NAMES[dataset_a]} + {SHORT_NAMES[dataset_b]}",
                }
            )
    return pairs


def _worker_optimizer(model_path: str) -> SignalOptimizer:
    global _WORKER_MODEL_PATH, _WORKER_OPTIMIZER
    if _WORKER_OPTIMIZER is None or _WORKER_MODEL_PATH != model_path:
        estimator, _, _ = protocol.load_frozen_eps_controller(Path(model_path))
        _WORKER_OPTIMIZER = SignalOptimizer(
            transfer.ScaledEstimator(estimator, mixed.RESPONSE_SCALE_KW, 0.0)
        )
        _WORKER_MODEL_PATH = model_path
    return _WORKER_OPTIMIZER


def _raw_matches(
    payload: dict[str, Any],
    task: dict[str, Any],
) -> bool:
    return bool(
        payload.get("protocol") == task["protocol"]
        and payload.get("pair_id") == task["pair_id"]
        and int(payload.get("fleet_size", -1)) == int(task["fleet_size"])
        and int(payload.get("seed_count", -1)) == int(task["seed_count"])
        and payload.get("model_sha256") == task["model_sha256"]
    )


def _run_pair(task: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(task["raw_path"])
    payload: dict[str, Any] = {}
    if raw_path.is_file() and not task["force"]:
        candidate = json.loads(raw_path.read_text(encoding="utf-8"))
        if _raw_matches(candidate, task):
            payload = candidate
    rows = list(payload.get("seed_results", []))
    completed_seeds = {int(row["seed_index"]) for row in rows}
    optimizer = _worker_optimizer(task["model_path"])
    weights = {task["dataset_a"]: 0.5, task["dataset_b"]: 0.5}

    for seed_index in range(int(task["seed_count"])):
        if seed_index in completed_seeds:
            continue
        base_seed = (
            int(task["base_seed"]) + int(task["pair_index"]) * 100_000 + seed_index
        )
        if task["source_unique"]:
            from . import run_e21_unique_scenarios as unique

            records, config, fleet_audit = unique._sample_unique_fleet(
                "S4-A",
                "aggregate",
                "test",
                base_seed,
                weights,
                maximum_fleet_size=int(task["fleet_size"]),
            )
        else:
            records, config, fleet_audit = mixed._sample_mixed_fleet(
                "S4-A",
                "aggregate",
                "test",
                base_seed,
                weights,
                fleet_size=int(task["fleet_size"]),
            )
        config["control"] = dict(config["control"])
        config["control"]["network_feedback"] = False
        scenario = protocol.build_pure_sim_scenario(records, config)
        availability = protocol.availability_probability(
            records,
            config,
            mixed.AVAILABILITY_MODE,
        )
        result = legacy._run_seed(
            "eps_broadcast",
            records,
            config,
            scenario,
            availability,
            base_seed + 800_000,
            base_seed + 1_910_000,
            eps_optimizer=optimizer,
        )
        audit_a = fleet_audit["datasets"][task["dataset_a"]]
        audit_b = fleet_audit["datasets"][task["dataset_b"]]
        rows.append(
            {
                "pair_id": task["pair_id"],
                "pair_index": int(task["pair_index"]),
                "pair_label": task["pair_label"],
                "dataset_a": task["dataset_a"],
                "dataset_b": task["dataset_b"],
                "seed_index": seed_index,
                "seed": base_seed,
                "fleet_size": int(fleet_audit["fleet_size"]),
                "devices_a": int(fleet_audit["counts"][task["dataset_a"]]),
                "devices_b": int(fleet_audit["counts"][task["dataset_b"]]),
                "baseline_curtailment_mwh": float(result["baseline_curtailment_mwh"]),
                "remaining_curtailment_mwh": float(result["remaining_curtailment_mwh"]),
                "accepted_absorption_mwh": float(result["accepted_absorption_mwh"]),
                "curtailment_reduction_pct": float(result["mean_reduction_pct"]),
                "availability_fraction": float(result["availability_fraction"]),
                "mean_network_scale": float(result["mean_network_scale"]),
                "network_violation_steps": int(result["network_violation_steps"]),
                "max_soc_violation": float(result["max_soc_violation"]),
                "profile_bootstrap_a": bool(audit_a["profile_bootstrap"]),
                "profile_bootstrap_b": bool(audit_b["profile_bootstrap"]),
                "mean_profile_reuse_a": float(audit_a["mean_profile_reuse"]),
                "mean_profile_reuse_b": float(audit_b["mean_profile_reuse"]),
                "sampling_without_replacement": bool(
                    fleet_audit.get("sampling_without_replacement", False)
                ),
                "duplicate_source_count": int(
                    fleet_audit.get("duplicate_source_count", 0)
                ),
                "max_source_reuse": int(fleet_audit.get("max_source_reuse", 1)),
            }
        )
        rows.sort(key=lambda row: int(row["seed_index"]))
        payload = {
            "protocol": task["protocol"],
            "pair_id": task["pair_id"],
            "pair_index": int(task["pair_index"]),
            "pair_label": task["pair_label"],
            "dataset_a": task["dataset_a"],
            "dataset_b": task["dataset_b"],
            "weights": weights,
            "fleet_size": int(fleet_audit["fleet_size"]),
            "maximum_requested_fleet_size": int(task["fleet_size"]),
            "fleet_mode": ("source_unique" if task["source_unique"] else "fixed5000"),
            "seed_count": int(task["seed_count"]),
            "model_path": task["model_path"],
            "model_sha256": task["model_sha256"],
            "seed_results": rows,
            "updated_at": _utc_now(),
        }
        _write_json(raw_path, payload)
        if (seed_index + 1) % 5 == 0 or seed_index + 1 == int(task["seed_count"]):
            print(
                json.dumps(
                    {
                        "stage": "pair_seed_checkpoint",
                        "pair": task["pair_id"],
                        "completed_seeds": seed_index + 1,
                        "seed_count": int(task["seed_count"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return payload


def _run_source_unique_diagonal(task: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(task["raw_path"])
    payload: dict[str, Any] = {}
    if raw_path.is_file() and not task["force"]:
        candidate = json.loads(raw_path.read_text(encoding="utf-8"))
        if (
            candidate.get("protocol") == UNIQUE_DIAGONAL_PROTOCOL
            and candidate.get("dataset") == task["dataset"]
            and int(candidate.get("fleet_size", -1)) == int(task["fleet_size"])
            and int(candidate.get("seed_count", -1)) == int(task["seed_count"])
            and candidate.get("model_sha256") == task["model_sha256"]
        ):
            payload = candidate
    rows = list(payload.get("seed_results", []))
    completed_seeds = {int(row["seed_index"]) for row in rows}
    optimizer = _worker_optimizer(task["model_path"])
    weights = {task["dataset"]: 1.0}

    from . import run_e21_unique_scenarios as unique

    for seed_index in range(int(task["seed_count"])):
        if seed_index in completed_seeds:
            continue
        base_seed = (
            int(task["base_seed"])
            + 20_000_000
            + int(task["dataset_index"]) * 100_000
            + seed_index
        )
        records, config, fleet_audit = unique._sample_unique_fleet(
            "S4-A",
            "aggregate",
            "test",
            base_seed,
            weights,
            maximum_fleet_size=int(task["fleet_size"]),
        )
        config["control"] = dict(config["control"])
        config["control"]["network_feedback"] = False
        scenario = protocol.build_pure_sim_scenario(records, config)
        availability = protocol.availability_probability(
            records,
            config,
            mixed.AVAILABILITY_MODE,
        )
        result = legacy._run_seed(
            "eps_broadcast",
            records,
            config,
            scenario,
            availability,
            base_seed + 800_000,
            base_seed + 1_910_000,
            eps_optimizer=optimizer,
        )
        rows.append(
            {
                "dataset": task["dataset"],
                "seed_index": seed_index,
                "seed": base_seed,
                "fleet_size": int(fleet_audit["fleet_size"]),
                "baseline_curtailment_mwh": float(result["baseline_curtailment_mwh"]),
                "remaining_curtailment_mwh": float(result["remaining_curtailment_mwh"]),
                "accepted_absorption_mwh": float(result["accepted_absorption_mwh"]),
                "curtailment_reduction_pct": float(result["mean_reduction_pct"]),
                "availability_fraction": float(result["availability_fraction"]),
                "sampling_without_replacement": True,
                "duplicate_source_count": 0,
                "max_source_reuse": 1,
            }
        )
        rows.sort(key=lambda row: int(row["seed_index"]))
        payload = {
            "protocol": UNIQUE_DIAGONAL_PROTOCOL,
            "dataset": task["dataset"],
            "dataset_label": task["dataset_label"],
            "fleet_mode": "source_unique",
            "fleet_size": int(fleet_audit["fleet_size"]),
            "seed_count": int(task["seed_count"]),
            "model_path": task["model_path"],
            "model_sha256": task["model_sha256"],
            "seed_results": rows,
            "updated_at": _utc_now(),
        }
        _write_json(raw_path, payload)
    return payload


def _summarize_source_unique_diagonal(
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        seed_results = payload["seed_results"]
        baseline = float(sum(row["baseline_curtailment_mwh"] for row in seed_results))
        remaining = float(sum(row["remaining_curtailment_mwh"] for row in seed_results))
        reduction = (
            100.0 * (1.0 - remaining / baseline) if baseline > EPS else float("nan")
        )
        rows.append(
            {
                "dataset": payload["dataset"],
                "dataset_label": payload["dataset_label"],
                "curtailment_reduction_pct": reduction,
                "seed_count": len(seed_results),
                "fleet_size": int(payload["fleet_size"]),
                "network_mode": "aggregate",
                "availability_mode": mixed.AVAILABILITY_MODE,
                "algorithm": "eps_e21_unique_mixed",
                "protocol": UNIQUE_DIAGONAL_PROTOCOL,
                "source_file": payload["raw_path"],
            }
        )
    return sorted(rows, key=lambda row: PLOT_ORDER.index(row["dataset"]))


def _bootstrap_ratio(
    rows: list[dict[str, Any]],
    draws: int,
    seed: int,
) -> tuple[float, float]:
    baseline = np.asarray(
        [row["baseline_curtailment_mwh"] for row in rows], dtype=float
    )
    remaining = np.asarray(
        [row["remaining_curtailment_mwh"] for row in rows], dtype=float
    )
    if float(np.sum(baseline)) <= EPS:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, len(rows), size=len(rows))
        denominator = float(np.sum(baseline[selected]))
        values[draw] = 100.0 * (
            1.0 - float(np.sum(remaining[selected])) / max(denominator, EPS)
        )
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _summarize_pairs(
    payloads: list[dict[str, Any]],
    bootstrap_draws: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows = [row for payload in payloads for row in payload["seed_results"]]
    summaries: list[dict[str, Any]] = []
    for payload in payloads:
        rows = payload["seed_results"]
        baseline = float(sum(row["baseline_curtailment_mwh"] for row in rows))
        remaining = float(sum(row["remaining_curtailment_mwh"] for row in rows))
        saved = baseline - remaining
        reduction = 100.0 * saved / baseline if baseline > EPS else float("nan")
        lower, upper = _bootstrap_ratio(
            rows,
            bootstrap_draws,
            base_seed + int(payload["pair_index"]) * 7919,
        )
        reductions = np.asarray(
            [row["curtailment_reduction_pct"] for row in rows], dtype=float
        )
        summaries.append(
            {
                "pair_id": payload["pair_id"],
                "pair_index": int(payload["pair_index"]),
                "pair_label": payload["pair_label"],
                "dataset_a": payload["dataset_a"],
                "dataset_b": payload["dataset_b"],
                "fleet_size": int(payload["fleet_size"]),
                "seed_count": len(rows),
                "baseline_curtailment_mwh_total": baseline,
                "remaining_curtailment_mwh_total": remaining,
                "saved_curtailment_mwh_total": saved,
                "baseline_curtailment_mwh_mean": baseline / len(rows),
                "remaining_curtailment_mwh_mean": remaining / len(rows),
                "saved_curtailment_mwh_mean": saved / len(rows),
                "curtailment_reduction_pct": reduction,
                "curtailment_reduction_ci_lower": lower,
                "curtailment_reduction_ci_upper": upper,
                "seed_mean_reduction_pct": float(np.mean(reductions)),
                "seed_std_reduction_pct": (
                    float(np.std(reductions, ddof=1)) if len(reductions) > 1 else 0.0
                ),
                "availability_fraction_mean": float(
                    np.mean([row["availability_fraction"] for row in rows])
                ),
                "network_violation_steps": int(
                    sum(row["network_violation_steps"] for row in rows)
                ),
                "max_soc_violation": float(
                    max(row["max_soc_violation"] for row in rows)
                ),
                "status": "completed"
                if baseline > EPS
                else "undefined_no_baseline_curtailment",
            }
        )
    summaries.sort(key=lambda row: int(row["pair_index"]))
    seed_rows.sort(key=lambda row: (int(row["pair_index"]), int(row["seed_index"])))
    return seed_rows, summaries


def _dataset_summary(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in PLOT_ORDER:
        selected = [
            row
            for row in pair_rows
            if dataset in (row["dataset_a"], row["dataset_b"])
            and math.isfinite(float(row["curtailment_reduction_pct"]))
        ]
        values = np.asarray(
            [row["curtailment_reduction_pct"] for row in selected], dtype=float
        )
        if not len(values):
            continue
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": SHORT_NAMES[dataset],
                "pair_count": len(selected),
                "mean_reduction_pct": float(np.mean(values)),
                "median_reduction_pct": float(np.median(values)),
                "q25_reduction_pct": float(np.percentile(values, 25)),
                "q75_reduction_pct": float(np.percentile(values, 75)),
                "min_reduction_pct": float(np.min(values)),
                "max_reduction_pct": float(np.max(values)),
            }
        )
    return rows


def _load_single_dataset_diagonal(source_unique: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in PLOT_ORDER:
        source = (
            Path("results")
            / f"{dataset}_ieee33_real_load"
            / "coverage_fix"
            / "network_constrained_new"
            / "data"
            / (
                "curtailment_baselines_final_source_unique_aggregate_data_driven.json"
                if source_unique
                else "curtailment_baselines_final_fixed5000_aggregate_data_driven.json"
            )
        )
        if not source.is_file():
            if source_unique:
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": SHORT_NAMES[dataset],
                        "curtailment_reduction_pct": float("nan"),
                        "seed_count": 0,
                        "fleet_size": 0,
                        "network_mode": "aggregate",
                        "availability_mode": "data_driven",
                        "algorithm": "eps_broadcast",
                        "protocol": "missing_source_unique_diagonal",
                        "source_file": str(source),
                    }
                )
                continue
            raise FileNotFoundError(f"missing single-dataset curtailment result: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        condition = payload["condition"]
        expected_mode = "source_unique" if source_unique else "fixed5000"
        if (
            condition.get("fleet_mode") != expected_mode
            or condition.get("network_mode") != "aggregate"
            or condition.get("availability_mode") != "data_driven"
        ):
            raise RuntimeError(f"single-dataset result does not match the condition: {source}")
        eps_result = payload["results"]["eps_broadcast"]
        seed_results = eps_result.get("seed_results", [])
        if not seed_results:
            raise RuntimeError(f"single-dataset result has no per-seed records: {source}")
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": SHORT_NAMES[dataset],
                "curtailment_reduction_pct": float(eps_result["mean_reduction_pct"]),
                "seed_count": len(seed_results),
                "fleet_size": int(condition["fleet_size"]),
                "network_mode": condition["network_mode"],
                "availability_mode": condition["availability_mode"],
                "algorithm": "eps_broadcast",
                "protocol": payload["protocol"],
                "source_file": str(source),
            }
        )
    return rows


def _plot_matrix(
    pair_rows: list[dict[str, Any]],
    diagonal_rows: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    index = {dataset: position for position, dataset in enumerate(PLOT_ORDER)}
    values = np.full((len(PLOT_ORDER), len(PLOT_ORDER)), np.nan)
    for row in pair_rows:
        first = index[row["dataset_a"]]
        second = index[row["dataset_b"]]
        value = float(row["curtailment_reduction_pct"])
        values[first, second] = value
        values[second, first] = value
    mixed_only = values.copy()
    fleet_sizes = [int(row["fleet_size"]) for row in pair_rows]
    seed_counts = {int(row["seed_count"]) for row in pair_rows}
    fleet_label = (
        f"N={fleet_sizes[0]}"
        if len(set(fleet_sizes)) == 1
        else f"source-unique N={min(fleet_sizes)}-{max(fleet_sizes)}"
    )
    seed_label = str(next(iter(seed_counts))) if len(seed_counts) == 1 else "variable"
    diagonal_model_label = (
        "same E21 unique mixed EPS on single-dataset fleets"
        if diagonal_rows
        and all(row["protocol"] == UNIQUE_DIAGONAL_PROTOCOL for row in diagonal_rows)
        else "single-dataset EPS"
    )

    def save_matrix(
        matrix: np.ndarray,
        path: Path,
        subtitle: str,
        diagonal_label: str | None,
    ) -> list[str]:
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#d9d9d9")
        fig, axis = plt.subplots(figsize=(15, 13), constrained_layout=True)
        image = axis.imshow(matrix, cmap=cmap, vmin=0.0, vmax=100.0, aspect="equal")
        labels = [SHORT_NAMES[dataset] for dataset in PLOT_ORDER]
        axis.set_xticks(np.arange(len(labels)), labels, rotation=48, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xlabel("Dataset A")
        axis.set_ylabel("Dataset B")
        axis.set_title(
            "Pairwise curtailment reduction across equal-weight mixed fleets\n"
            f"{subtitle}"
        )
        for row_index in range(len(PLOT_ORDER)):
            for column_index in range(len(PLOT_ORDER)):
                value = matrix[row_index, column_index]
                label = (
                    diagonal_label
                    if row_index == column_index and diagonal_label is not None
                    else f"{value:.1f}%"
                    if math.isfinite(value)
                    else "--"
                )
                color = (
                    "white"
                    if math.isfinite(value) and (value < 16 or value > 72)
                    else "#222222"
                )
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=color,
                )
        axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.7)
        axis.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(
            image,
            ax=axis,
            label="Curtailment reduction rate (%)",
            shrink=0.82,
        )
        fig.savefig(path, dpi=240)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        return [str(path), str(path.with_suffix(".pdf"))]

    generated = save_matrix(
        mixed_only,
        output / "pairwise_curtailment_reduction_mixed_only.png",
        f"{fleet_label}, 50/50 composition, {seed_label} paired seeds; diagonal excluded",
        "self",
    )
    for row in diagonal_rows:
        position = index[row["dataset"]]
        values[position, position] = float(row["curtailment_reduction_pct"])
    generated.extend(
        save_matrix(
            values,
            output / "pairwise_curtailment_reduction.png",
            f"{fleet_label}, 50/50 composition, {seed_label} paired seeds; diagonal uses {diagonal_model_label}",
            None,
        )
    )
    return generated


def _plot_dataset_distribution(
    pair_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    ordered = sorted(dataset_rows, key=lambda row: row["mean_reduction_pct"])
    fig, axis = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    for position, summary in enumerate(ordered):
        dataset = summary["dataset"]
        values = [
            float(row["curtailment_reduction_pct"])
            for row in pair_rows
            if dataset in (row["dataset_a"], row["dataset_b"])
        ]
        jitter = np.linspace(-0.16, 0.16, len(values))
        axis.scatter(values, position + jitter, s=27, color="#4c78a8", alpha=0.62)
        axis.hlines(
            position,
            summary["min_reduction_pct"],
            summary["max_reduction_pct"],
            color="#777777",
            linewidth=1.0,
        )
        axis.hlines(
            position,
            summary["q25_reduction_pct"],
            summary["q75_reduction_pct"],
            color="#222222",
            linewidth=5.0,
        )
        axis.scatter(
            summary["mean_reduction_pct"],
            position,
            marker="D",
            s=52,
            color="#d1495b",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
    axis.axvline(0.0, color="#222222", linestyle=":", linewidth=1.1)
    axis.set_yticks(np.arange(len(ordered)), [row["dataset_label"] for row in ordered])
    axis.set_xlabel("Curtailment reduction rate (%)")
    axis.set_ylabel("Dataset participating in 14 pairwise mixtures")
    axis.set_title(
        "Dataset-level distribution of pairwise curtailment reduction\n"
        "dots=pairs, thick line=IQR, diamond=mean"
    )
    axis.grid(axis="x", alpha=0.22)
    path = output / "dataset_curtailment_reduction_distribution.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [str(path), str(path.with_suffix(".pdf"))]


def _plot_opportunity(pair_rows: list[dict[str, Any]], output: Path) -> list[str]:
    x = np.asarray(
        [row["baseline_curtailment_mwh_mean"] for row in pair_rows], dtype=float
    )
    y = np.asarray([row["curtailment_reduction_pct"] for row in pair_rows], dtype=float)
    saved = np.asarray(
        [row["saved_curtailment_mwh_mean"] for row in pair_rows], dtype=float
    )
    size = 34.0 + 210.0 * saved / max(float(np.max(saved)), EPS)
    fig, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    scatter = axis.scatter(
        x,
        y,
        s=size,
        c=y,
        cmap="RdYlGn",
        norm=Normalize(0.0, 100.0),
        alpha=0.78,
        edgecolor="white",
        linewidth=0.7,
    )
    selected = set(np.argsort(saved)[-6:].tolist()) | set(np.argsort(y)[:3].tolist())
    for row_index in sorted(selected):
        axis.annotate(
            pair_rows[row_index]["pair_label"],
            (x[row_index], y[row_index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axhline(0.0, color="#222222", linestyle=":", linewidth=1.1)
    axis.set_xlabel("Baseline curtailed energy per seed (MWh)")
    axis.set_ylabel("Curtailment reduction rate (%)")
    axis.set_title(
        "Curtailment opportunity versus EPS reduction\n"
        "bubble area represents saved curtailed energy"
    )
    axis.grid(alpha=0.22)
    fig.colorbar(scatter, ax=axis, label="Curtailment reduction rate (%)")
    path = output / "curtailment_opportunity_vs_reduction.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [str(path), str(path.with_suffix(".pdf"))]


def _plot_ranked_dumbbell(pair_rows: list[dict[str, Any]], output: Path) -> list[str]:
    ordered = sorted(
        pair_rows,
        key=lambda row: float(row["curtailment_reduction_pct"]),
        reverse=True,
    )
    chunks = [ordered[index : index + 35] for index in range(0, len(ordered), 35)]
    maximum = max(float(row["baseline_curtailment_mwh_mean"]) for row in ordered)
    fig, axes = plt.subplots(
        1, 3, figsize=(19, 15), constrained_layout=True, sharex=True
    )
    for chunk_index, (axis, chunk) in enumerate(zip(axes, chunks)):
        y = np.arange(len(chunk))
        baseline = np.asarray([row["baseline_curtailment_mwh_mean"] for row in chunk])
        remaining = np.asarray([row["remaining_curtailment_mwh_mean"] for row in chunk])
        for position, before, after in zip(y, baseline, remaining):
            axis.plot(
                [after, before], [position, position], color="#4c956c", linewidth=1.5
            )
        axis.scatter(
            baseline,
            y,
            facecolors="white",
            edgecolors="#666666",
            s=35,
            label="Before EPS",
        )
        axis.scatter(remaining, y, color="#2166ac", s=35, label="After EPS")
        axis.set_yticks(y, [row["pair_label"] for row in chunk], fontsize=7)
        axis.invert_yaxis()
        axis.set_xlim(-0.02 * maximum, 1.04 * maximum)
        axis.grid(axis="x", alpha=0.22)
        axis.set_title(f"Rank {chunk_index * 35 + 1}-{chunk_index * 35 + len(chunk)}")
    for axis in axes[len(chunks) :]:
        axis.axis("off")
    axes[0].legend(frameon=False, loc="lower right")
    fig.supxlabel("Mean curtailed energy per seed (MWh)")
    fig.suptitle(
        "Pairwise curtailed energy before and after EPS\n"
        "pairs ranked by curtailment reduction rate"
    )
    path = output / "pairwise_curtailment_before_after_ranked.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [str(path), str(path.with_suffix(".pdf"))]


def _save_figures(
    pair_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    diagonal_rows: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    generated.extend(_plot_matrix(pair_rows, diagonal_rows, output))
    generated.extend(_plot_dataset_distribution(pair_rows, dataset_rows, output))
    generated.extend(_plot_opportunity(pair_rows, output))
    generated.extend(_plot_ranked_dumbbell(pair_rows, output))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for run e21 pairwise curtailment.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fleet-size", type=int, default=DEFAULT_FLEET_SIZE)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(3, (os.cpu_count() or 2) // 4))
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-unique", action="store_true")
    args = parser.parse_args()
    if args.source_unique and args.model == DEFAULT_MODEL:
        args.model = DEFAULT_UNIQUE_MODEL
    if not args.model.is_file():
        raise FileNotFoundError(f"E21 mixed EPS model not found: {args.model}")
    if args.fleet_size < 2 or args.fleet_size % 2:
        raise ValueError("fleet-size must be an even number of at least 2, so that the composition can be 50/50")
    pairs = _pair_specs()
    if args.max_pairs is not None:
        pairs = pairs[: max(0, int(args.max_pairs))]
    output = args.output
    raw_dir = output / "raw" / "pairs"
    diagonal_raw_dir = output / "raw" / "diagonal"
    data_dir = output / "data"
    figure_dir = output / "figures"
    for directory in (raw_dir, diagonal_raw_dir, data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    model_hash = _sha256(args.model)
    tasks = []
    for pair in pairs:
        fleet_size = int(args.fleet_size)
        if args.source_unique:
            from . import run_e21_unique_scenarios as unique

            weights = {pair["dataset_a"]: 0.5, pair["dataset_b"]: 0.5}
            capacities = unique._partition_capacities(weights, "aggregate", "test")
            devices_per_dataset = min(
                int(args.fleet_size) // 2,
                *(int(capacity) for capacity in capacities.values()),
            )
            fleet_size = 2 * devices_per_dataset
        tasks.append(
            {
                **pair,
                "fleet_size": fleet_size,
                "seed_count": int(args.seed_count),
                "base_seed": int(args.base_seed),
                "model_path": str(args.model),
                "model_sha256": model_hash,
                "raw_path": str(raw_dir / f"{pair['pair_id']}.json"),
                "force": bool(args.force),
                "source_unique": bool(args.source_unique),
                "protocol": UNIQUE_PROTOCOL if args.source_unique else PROTOCOL,
            }
        )
    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {executor.submit(_run_pair, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                payload = future.result()
            except Exception as exc:
                failure = {
                    "pair_id": task["pair_id"],
                    "pair_label": task["pair_label"],
                    "error": repr(exc),
                }
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
                continue
            payloads.append(payload)
            print(
                json.dumps(
                    {
                        "stage": "pair_completed",
                        "pair_id": task["pair_id"],
                        "completed_pairs": len(payloads),
                        "pair_count": len(tasks),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if failures:
        _write_json(output / "failures.json", failures)
        raise SystemExit(1)
    payloads.sort(key=lambda payload: int(payload["pair_index"]))
    seed_rows, pair_rows = _summarize_pairs(
        payloads,
        int(args.bootstrap_draws),
        int(args.base_seed),
    )
    dataset_rows = _dataset_summary(pair_rows)
    if args.source_unique:
        diagonal_tasks = []
        from . import run_e21_unique_scenarios as unique

        for dataset_index, dataset in enumerate(PLOT_ORDER):
            capacities = unique._partition_capacities(
                {dataset: 1.0}, "aggregate", "test"
            )
            fleet_size = min(int(args.fleet_size), int(capacities[dataset]))
            diagonal_tasks.append(
                {
                    "dataset": dataset,
                    "dataset_label": SHORT_NAMES[dataset],
                    "dataset_index": dataset_index,
                    "fleet_size": fleet_size,
                    "seed_count": int(args.seed_count),
                    "base_seed": int(args.base_seed),
                    "model_path": str(args.model),
                    "model_sha256": model_hash,
                    "raw_path": str(diagonal_raw_dir / f"{dataset}.json"),
                    "force": bool(args.force),
                }
            )
        diagonal_payloads = []
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(_run_source_unique_diagonal, task): task
                for task in diagonal_tasks
            }
            for future in as_completed(futures):
                payload = future.result()
                payload["raw_path"] = str(
                    diagonal_raw_dir / f"{payload['dataset']}.json"
                )
                diagonal_payloads.append(payload)
        diagonal_rows = _summarize_source_unique_diagonal(diagonal_payloads)
    else:
        diagonal_rows = _load_single_dataset_diagonal(False)
    _write_rows(data_dir / "pairwise_curtailment_by_seed.csv", seed_rows)
    _write_rows(data_dir / "pairwise_curtailment_summary.csv", pair_rows)
    _write_rows(data_dir / "dataset_level_summary.csv", dataset_rows)
    _write_rows(data_dir / "single_dataset_diagonal.csv", diagonal_rows)
    figures = _save_figures(pair_rows, dataset_rows, diagonal_rows, figure_dir)
    manifest = {
        "protocol": UNIQUE_PROTOCOL if args.source_unique else PROTOCOL,
        "status": "completed",
        "completed_at": _utc_now(),
        "dataset_count": len(PLOT_ORDER),
        "pair_count": len(pair_rows),
        "seed_count": int(args.seed_count),
        "simulation_count": len(seed_rows)
        + sum(int(row["seed_count"]) for row in diagonal_rows),
        "pairwise_simulation_count": len(seed_rows),
        "diagonal_simulation_count": sum(
            int(row["seed_count"]) for row in diagonal_rows
        ),
        "diagonal_dataset_count": len(diagonal_rows),
        "diagonal_protocol": (
            UNIQUE_DIAGONAL_PROTOCOL
            if args.source_unique
            else "existing_single_dataset"
        ),
        "maximum_fleet_size": int(args.fleet_size),
        "fleet_mode": "source_unique" if args.source_unique else "fixed5000",
        "fleet_size_range": [
            min(int(row["fleet_size"]) for row in seed_rows),
            max(int(row["fleet_size"]) for row in seed_rows),
        ],
        "composition": {"dataset_a": 0.5, "dataset_b": 0.5},
        "network_mode": "aggregate",
        "availability_mode": mixed.AVAILABILITY_MODE,
        "model_path": str(args.model),
        "model_sha256": model_hash,
        "figures": figures,
        "generated_files": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["generated_files"].append(
                {
                    "path": str(path.relative_to(output)),
                    "sha256": _sha256(path),
                }
            )
    _write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "pairs": len(pair_rows),
                "pairwise_simulations": len(seed_rows),
                "diagonal_simulations": sum(
                    int(row["seed_count"]) for row in diagonal_rows
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
