"""运行 E21 数据集两两混合的弃电降低实验并生成审计图。"""

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
from tools.ncstyle import SEQ

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
            raise FileNotFoundError(f"缺少单数据集弃电结果: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        condition = payload["condition"]
        expected_mode = "source_unique" if source_unique else "fixed5000"
        if (
            condition.get("fleet_mode") != expected_mode
            or condition.get("network_mode") != "aggregate"
            or condition.get("availability_mode") != "data_driven"
        ):
            raise RuntimeError(f"单数据集结果条件不匹配: {source}")
        eps_result = payload["results"]["eps_broadcast"]
        seed_results = eps_result.get("seed_results", [])
        if not seed_results:
            raise RuntimeError(f"单数据集结果缺少逐 seed 记录: {source}")
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
        cmap = SEQ.copy()
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
        cmap=SEQ,
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


def _write_design(output: Path, args: argparse.Namespace, pair_count: int) -> None:
    fleet_description = (
        f"每个组合在 N<={args.fleet_size} 内取真实源容量允许的最大偶数规模"
        if args.source_unique
        else f"所有组合固定 N={args.fleet_size}"
    )
    composition_description = (
        "每个数据集严格占 50%，每个真实源在同一 fleet 中最多使用一次"
        if args.source_unique
        else f"每个数据集严格占 50%，即各 {args.fleet_size // 2} 台设备"
    )
    document = f"""# E21 数据集两两混合弃电实验设计

## 实验目标

该实验把 15 个数据集按无序两两组合构造物理混合 fleet，检验固定 E21 mixed EPS 模型在不同数据组成下吸收外部能源富余、降低弃电的效果。每个组合只运行一次，A+B 与 B+A 不重复计算。

## 实验规模

- 数据集：15 个。
- 无序组合：{pair_count} 个。
- 设备规模：{fleet_description}。
- 组成：{composition_description}。
- 随机种子：每个组合 {args.seed_count} 个。
- 完整日步数：288 个五分钟步。
- 完整仿真次数：{pair_count * args.seed_count}。
- 并行进程：{args.workers}。
- EPS 模型：`{args.model}`，沿用已完成的 E21 mixed-scenario aggregate 模型，不重新训练。

## 设备和数据口径

- 两个数据集都从自己的 test partition 抽取设备记录。
- 抽样模式：{"source-unique 不放回" if args.source_unique else "fixed5000；真实源不足时允许 profile bootstrap"}。
- 负荷时间形态和数据驱动可用率来自对应数据集。
- 容量、功率、初始 SOC、SOH 和执行方式沿用 E21 的纯仿真设备层设置。
- 外部能源输入沿用 E21 纯仿真场景，不应表述为 15 个数据集都提供了实测 PV。
- 网络模式为 aggregate，IEEE-33 仅保留诊断，不截断 EPS 响应。
- EPS 在线输入不包含数据集 ID、组合 ID或逐设备状态。

## 弃电口径

每个五分钟步先计算外部输入超过负荷的正富余。无控制时，正富余全部计为基线弃电；EPS 后只扣除同一步实际接受的正向充电响应，并且吸收量不超过当前富余。

组合汇总降低率采用能量加权比值：

`100 * (1 - 所有seed剩余弃电总量 / 所有seed基线弃电总量)`。

该口径不会把低弃电seed和高弃电seed等权平均。95%置信区间通过对 {args.seed_count} 个成对seed进行 {args.bootstrap_draws} 次bootstrap获得。若基线弃电为零，降低率记为 NA，不强制写成0或100%。

## 输出图

1. 两两组合弃电降低率二维图，同时保留mixed-only原版和单数据集对角线补全版。
2. 每个数据集参与14个组合时的效果分布。
3. 基线弃电机会、降低率和实际节省能量散点图。
4. 105个组合控制前后弃电能量排序哑铃图。
"""
    (output / "EXPERIMENT_DESIGN.md").write_text(document, encoding="utf-8")


def _write_documents(
    output: Path,
    pair_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    diagonal_rows: list[dict[str, Any]],
) -> None:
    valid = [row for row in pair_rows if row["status"] == "completed"]
    ordered = sorted(
        valid, key=lambda row: row["curtailment_reduction_pct"], reverse=True
    )
    best = ordered[0]
    worst = ordered[-1]
    mean_reduction = float(np.mean([row["curtailment_reduction_pct"] for row in valid]))
    mean_saved = float(np.mean([row["saved_curtailment_mwh_mean"] for row in valid]))
    fleet_sizes = [int(row["fleet_size"]) for row in valid]
    fleet_description = (
        f"固定 N={fleet_sizes[0]}"
        if len(set(fleet_sizes)) == 1
        else f"source-unique N 范围为 {min(fleet_sizes)} 至 {max(fleet_sizes)}"
    )
    diagonal_description = (
        "对角线使用与非对角线相同的 E21 source-unique mixed EPS 模型"
        if diagonal_rows
        and all(row["protocol"] == UNIQUE_DIAGONAL_PROTOCOL for row in diagonal_rows)
        else "对角线使用各数据集自己的 EPS 模型"
    )
    best_dataset = max(dataset_rows, key=lambda row: row["mean_reduction_pct"])
    worst_dataset = min(dataset_rows, key=lambda row: row["mean_reduction_pct"])
    dataset_table = "\n".join(
        "|{dataset_label}|{mean_reduction_pct:.2f}%|{median_reduction_pct:.2f}%|{min_reduction_pct:.2f}%|{max_reduction_pct:.2f}%|".format(
            **row
        )
        for row in sorted(
            dataset_rows, key=lambda row: row["mean_reduction_pct"], reverse=True
        )
    )
    diagonal_table = "\n".join(
        f"|{row['dataset_label']}|{row['curtailment_reduction_pct']:.2f}%|{row['seed_count']}|`{row['source_file']}`|"
        for row in diagonal_rows
    )
    document = f"""# 1. pairwise_curtailment_reduction

![两两组合弃电降低率](figures/pairwise_curtailment_reduction.png)

文件：

- `figures/pairwise_curtailment_reduction.png`
- `figures/pairwise_curtailment_reduction.pdf`
- 数据：`data/pairwise_curtailment_summary.csv`
- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/run_e21_pairwise_curtailment.py`

### 实验内容

该图覆盖15个数据集的105个无序两两组合。每个非对角组合中两个数据集各占50%，{fleet_description}，并使用30个成对seed运行完整288步EPS设备层仿真。矩阵对称位置展示同一结果，不代表重复实验。{diagonal_description}，同样采用真实源不放回抽样和30个seed。

### 横纵坐标

- 横坐标：组合中的数据集A。
- 纵坐标：组合中的数据集B。
- 单元格颜色和文字：能量加权弃电降低率。
- 对角线：数据集自身的现成EPS弃电降低率。
- 色标固定为0%至100%。

### 结果

- 有效组合：{len(valid)}/{len(pair_rows)}。
- 105组合平均降低率：{mean_reduction:.2f}%。
- 最佳组合：{best['pair_label']}，降低率 {best['curtailment_reduction_pct']:.2f}%。
- 最低组合：{worst['pair_label']}，降低率 {worst['curtailment_reduction_pct']:.2f}%。
- {diagonal_description}；对角线用于单数据集参照，不代表额外的两两组合。

## 对角线来源

|数据集|单数据集降低率|Seeds|来源|
|-|-:|-:|-|
{diagonal_table}

# 2. pairwise_curtailment_reduction_mixed_only

![Mixed组合弃电降低率](figures/pairwise_curtailment_reduction_mixed_only.png)

文件：

- `figures/pairwise_curtailment_reduction_mixed_only.png`
- `figures/pairwise_curtailment_reduction_mixed_only.pdf`
- 数据：`data/pairwise_curtailment_summary.csv`
- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/run_e21_pairwise_curtailment.py`

### 实验内容

该图恢复最初的mixed-only版本，只展示105个50/50两数据集混合fleet的E21 mixed EPS弃电降低率。{fleet_description}。矩阵对称位置是同一无序组合，对角线不引入单数据集模型结果。

### 横纵坐标

- 横坐标：组合中的数据集A。
- 纵坐标：组合中的数据集B。
- 非对角单元格：E21 mixed EPS的能量加权弃电降低率。
- 灰色对角线：单数据集结果不属于该mixed-only图，标记为self。

### 结果

- 有效mixed组合：{len(valid)}个。
- mixed组合平均降低率：{mean_reduction:.2f}%。
- 该图与对角线补全版共存，后续重绘不会再互相覆盖。

# 3. dataset_curtailment_reduction_distribution

![数据集配对效果分布](figures/dataset_curtailment_reduction_distribution.png)

文件：

- `figures/dataset_curtailment_reduction_distribution.png`
- `figures/dataset_curtailment_reduction_distribution.pdf`
- 数据：`data/dataset_level_summary.csv`、`data/pairwise_curtailment_summary.csv`
- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/run_e21_pairwise_curtailment.py`

### 实验内容

对每个数据集汇总其与其余14个数据集组成混合fleet时的弃电降低率，不把105个组合压缩成一个总体均值。

### 横纵坐标

- 横坐标：弃电降低率。
- 纵坐标：参与组合的数据集。
- 蓝色点：该数据集参与的14个组合。
- 黑色粗线：四分位区间。
- 红色菱形：14个组合的算术平均值。

### 结果

- 配对平均降低率最高的数据集：{best_dataset['dataset_label']}，{best_dataset['mean_reduction_pct']:.2f}%。
- 配对平均降低率最低的数据集：{worst_dataset['dataset_label']}，{worst_dataset['mean_reduction_pct']:.2f}%。
- 点的分散程度表示同一数据集对不同混合对象的敏感性。

# 4. curtailment_opportunity_vs_reduction

![弃电机会与降低率](figures/curtailment_opportunity_vs_reduction.png)

文件：

- `figures/curtailment_opportunity_vs_reduction.png`
- `figures/curtailment_opportunity_vs_reduction.pdf`
- 数据：`data/pairwise_curtailment_summary.csv`
- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/run_e21_pairwise_curtailment.py`

### 实验内容

该图同时检查百分比效果和绝对能源量，避免把基线弃电很小但比例很高的组合误判为最重要结果。

### 横纵坐标

- 横坐标：每个seed的平均基线弃电量，单位MWh。
- 纵坐标：能量加权弃电降低率。
- 点面积：每个seed平均减少的弃电量。
- 点颜色：弃电降低率。

### 结果

- 所有组合每seed平均节省弃电量的均值：{mean_saved:.4f} MWh。
- 标签标出绝对节省量最高和降低率最低的组合。
- 本图中的外部输入来自统一纯仿真场景，不代表所有源数据集都具有实测可再生能源字段。

# 5. pairwise_curtailment_before_after_ranked

![组合控制前后弃电能量](figures/pairwise_curtailment_before_after_ranked.png)

文件：

- `figures/pairwise_curtailment_before_after_ranked.png`
- `figures/pairwise_curtailment_before_after_ranked.pdf`
- 数据：`data/pairwise_curtailment_summary.csv`
- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/run_e21_pairwise_curtailment.py`

### 实验内容

105个组合按弃电降低率排序并分为三栏。每行直接比较同一组合在无控制和EPS控制后的弃电能量。

### 横纵坐标

- 横坐标：每个seed的平均弃电能量，单位MWh。
- 纵坐标：数据集组合，按降低率从高到低排序。
- 灰色空心点：无控制基线弃电。
- 蓝色实心点：EPS后剩余弃电。
- 绿色连线：同一组合减少的弃电能量。

### 结果

- 最佳组合从 {best['baseline_curtailment_mwh_mean']:.4f} MWh 降至 {best['remaining_curtailment_mwh_mean']:.4f} MWh。
- 最低组合从 {worst['baseline_curtailment_mwh_mean']:.4f} MWh 降至 {worst['remaining_curtailment_mwh_mean']:.4f} MWh。
- 该图使用能量而不是控制前弃电率，因为无协调基线将全部正富余计为弃电，其基线弃电率按定义恒为100%。

## 数据集汇总

|数据集|平均降低率|中位数|最小值|最大值|
|-|-:|-:|-:|-:|
{dataset_table}
"""
    (output / "FIGURE_DESCRIPTIONS.md").write_text(document, encoding="utf-8")
    readme = f"""# E21 Pairwise Curtailment

该目录保存15个数据集两两等权混合后的弃电降低实验，共105个组合、每组30个seed，{fleet_description}。

## 主要结论

- 组合平均弃电降低率：{mean_reduction:.2f}%。
- 最佳组合：{best['pair_label']}，{best['curtailment_reduction_pct']:.2f}%。
- 最低组合：{worst['pair_label']}，{worst['curtailment_reduction_pct']:.2f}%。
- 详细实验口径见 `EXPERIMENT_DESIGN.md`。
- 每张图的读取方式和具体数值见 `FIGURE_DESCRIPTIONS.md`。

## 数据文件

- `data/pairwise_curtailment_by_seed.csv`：3150条逐seed结果。
- `data/pairwise_curtailment_summary.csv`：105个组合汇总。
- `data/dataset_level_summary.csv`：15个数据集的配对分布汇总。

## 解释边界

外部能源输入和设备层参数沿用E21纯仿真设置。数据集提供负荷时间形态与数据驱动可用率，因此结果用于比较不同数据混合组成下的控制效果，不能表述为15个数据集都提供了实测PV弃电记录。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        raise ValueError("fleet-size 必须是至少为2的偶数，以保证50/50组成")
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
    _write_design(output, args, len(pairs))
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
    _write_documents(output, pair_rows, dataset_rows, diagonal_rows)
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
