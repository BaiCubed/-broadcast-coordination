"""运行 E21 全场景与两两组合的完整弃电 baseline 补充实验。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.ncstyle import SEQ

from src.signal import SignalOptimizer

from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as protocol
from . import run_e20_transfer as transfer
from . import run_e21_mixed_scenarios as mixed
from . import run_e21_pairwise_curtailment as pairwise


PROTOCOL = "E21_curtailment_baseline_supplement_v1"
CENTRALIZED_UB_VERSION = "compared_feasible_dispatch_envelope_v1"
DEFAULT_OUTPUT = Path("results/E21/curtailment_baseline_supplement")
NETWORK_MODES = ("aggregate", "ieee33")
ALGORITHM_ORDER = mixed.ALGORITHM_ORDER
BASELINE_ORDER = (*mixed.BASELINE_ORDER, "centralized_optimal")
SEED_COUNT = 30
FLEET_SIZE = 5000
BOOTSTRAP_DRAWS = 4000
PAIR_SEED_BASE = 20260808
EPS = 1e-12

_WORKER_OPTIMIZERS: dict[str, SignalOptimizer] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _optimizer(model_path: str) -> SignalOptimizer:
    if model_path not in _WORKER_OPTIMIZERS:
        estimator, _, _ = protocol.load_frozen_eps_controller(Path(model_path))
        _WORKER_OPTIMIZERS[model_path] = SignalOptimizer(
            transfer.ScaledEstimator(estimator, mixed.RESPONSE_SCALE_KW, 0.0)
        )
    return _WORKER_OPTIMIZERS[model_path]


def _model_paths(results_root: Path) -> dict[str, dict[str, Path]]:
    return {
        network_mode: {
            "eps_e20_pooled": results_root / "E20/models" / f"pooled_{network_mode}.pt",
            "eps_e21_mixed": results_root / "E21/models" / f"e21_mixed_{network_mode}.pt",
        }
        for network_mode in NETWORK_MODES
    }


def _task_models_match(payload: dict[str, Any], task: dict[str, Any]) -> bool:
    return (
        payload.get("protocol") == PROTOCOL
        and payload.get("composition_kind") == task["composition_kind"]
        and payload.get("composition_id") == task["composition_id"]
        and payload.get("network_mode") == task["network_mode"]
        and int(payload.get("seed_count", -1)) == int(task["seed_count"])
        and payload.get("model_sha256") == task["model_sha256"]
    )


def _base_seed(task: dict[str, Any], seed_index: int) -> int:
    if task["composition_kind"] == "scenario":
        return 4_100_000 + protocol._stable_seed(task["composition_id"]) % 100_000 + seed_index
    return PAIR_SEED_BASE + int(task["pair_index"]) * 100_000 + seed_index


def _sample_task_fleet(
    task: dict[str, Any], seed_index: int
) -> tuple[list[Any], dict[str, Any], dict[str, Any], str, int]:
    base_seed = _base_seed(task, seed_index)
    if task["composition_kind"] == "scenario":
        weights, regime = mixed._shift_weights(
            mixed.SCENARIOS[task["composition_id"]]["weights"], seed_index
        )
        scenario_id = task["composition_id"]
    else:
        weights = {task["dataset_a"]: 0.5, task["dataset_b"]: 0.5}
        regime = "equal_50_50"
        scenario_id = "S4-A"
    records, config, audit = mixed._sample_mixed_fleet(
        scenario_id,
        task["network_mode"],
        "test",
        base_seed,
        weights,
        fleet_size=FLEET_SIZE,
    )
    config = copy.deepcopy(config)
    config["control"] = dict(config["control"])
    config["control"]["network_feedback"] = task["network_mode"] == "ieee33"
    audit["composition_regime"] = regime
    audit["seed"] = base_seed
    return records, config, audit, regime, base_seed


def _apply_centralized_upper_bound(rows: list[dict[str, Any]]) -> int:
    """使集中式结果成为同一 seed 下所有可行调度的严格比较上界。"""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed_index"])].append(row)

    adjustment_count = 0
    for seed_index, seed_rows in grouped.items():
        central_rows = [
            row for row in seed_rows if row["algorithm"] == "centralized_optimal"
        ]
        if len(central_rows) != 1:
            raise RuntimeError(
                f"seed {seed_index} 的 Centralized greedy UB 行数为 {len(central_rows)}"
            )
        central = central_rows[0]
        baseline_values = [float(row["baseline_curtailment_mwh"]) for row in seed_rows]
        if max(baseline_values) - min(baseline_values) > 1e-8:
            raise RuntimeError(f"seed {seed_index} 的算法间无控制弃电不一致")

        central.setdefault(
            "native_greedy_reduction_pct", float(central["mean_reduction_pct"])
        )
        central.setdefault(
            "native_greedy_absorption_mwh", float(central["accepted_absorption_mwh"])
        )
        best = max(
            (row for row in seed_rows if row["algorithm"] != "centralized_optimal"),
            key=lambda row: float(row["accepted_absorption_mwh"]),
        )
        native_absorption = float(central["native_greedy_absorption_mwh"])
        if float(best["accepted_absorption_mwh"]) > native_absorption + 1e-12:
            for field in (
                "mean_reduction_pct",
                "remaining_curtailment_mwh",
                "accepted_absorption_mwh",
                "total_charging_mwh",
                "excess_grid_charging_mwh",
                "total_discharge_mwh",
                "mean_network_scale",
                "network_violation_steps",
                "max_soc_violation",
            ):
                if field in best:
                    central[field] = best[field]
            central["upper_bound_adjusted"] = True
            central["upper_bound_source_algorithm"] = best["algorithm"]
            adjustment_count += 1
        else:
            central["upper_bound_adjusted"] = False
            central["upper_bound_source_algorithm"] = "centralized_optimal"
        central["upper_bound_version"] = CENTRALIZED_UB_VERSION
        central["upper_bound_definition"] = (
            "同一 composition、network 和 seed 下原生集中式贪心与全部已比较可行调度的最大实际弃电消纳量；"
            "拥有完整状态和调度信息的集中式控制器至少可以复现该调度。"
        )

        central_absorption = float(central["accepted_absorption_mwh"])
        best_absorption = max(float(row["accepted_absorption_mwh"]) for row in seed_rows)
        if central_absorption + 1e-9 < best_absorption:
            raise RuntimeError(f"seed {seed_index} 的 Centralized greedy UB 上界不成立")
    return adjustment_count


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(task["raw_path"])
    payload: dict[str, Any] = {}
    if raw_path.is_file() and not task["force"]:
        candidate = json.loads(raw_path.read_text(encoding="utf-8"))
        if _task_models_match(candidate, task):
            payload = candidate
    rows = list(payload.get("seed_results", []))
    audits = {
        int(row["seed_index"]): row for row in payload.get("fleet_audit", [])
    }
    completed = {
        (int(row["seed_index"]), str(row["algorithm"])) for row in rows
    }
    for seed_index in range(int(task["seed_count"])):
        missing = [
            algorithm
            for algorithm in ALGORITHM_ORDER
            if (seed_index, algorithm) not in completed
        ]
        if not missing:
            continue
        records, config, audit, regime, base_seed = _sample_task_fleet(task, seed_index)
        scenario = protocol.build_pure_sim_scenario(records, config)
        availability = protocol.availability_probability(
            records, config, mixed.AVAILABILITY_MODE
        )
        audits[seed_index] = {"seed_index": seed_index, **audit}
        for algorithm in missing:
            started = time.perf_counter()
            if algorithm == "no_coordination":
                result = mixed._no_coordination(
                    scenario, availability, config, base_seed
                )
            else:
                strategy = "eps_broadcast" if algorithm.startswith("eps_") else algorithm
                optimizer = (
                    _optimizer(task["model_paths"][algorithm])
                    if algorithm.startswith("eps_")
                    else None
                )
                algorithm_index = ALGORITHM_ORDER.index(algorithm)
                result = legacy._run_seed(
                    strategy,
                    records,
                    config,
                    scenario,
                    availability,
                    base_seed + 100_000 * (algorithm_index + 1),
                    base_seed + 1_910_000,
                    eps_optimizer=optimizer,
                )
            runtime = time.perf_counter() - started
            row = {
                "composition_kind": task["composition_kind"],
                "composition_id": task["composition_id"],
                "composition_label": task["composition_label"],
                "network_mode": task["network_mode"],
                "algorithm": algorithm,
                "algorithm_label": mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                "complexity": mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"],
                "seed_index": seed_index,
                "seed": base_seed,
                "composition_regime": regime,
                "fleet_size": len(records),
                "runtime_seconds": runtime,
                **result,
            }
            if task["composition_kind"] == "pairwise":
                row.update({
                    "pair_index": int(task["pair_index"]),
                    "dataset_a": task["dataset_a"],
                    "dataset_b": task["dataset_b"],
                })
            rows.append(row)
            completed.add((seed_index, algorithm))
            payload = {
                "protocol": PROTOCOL,
                "composition_kind": task["composition_kind"],
                "composition_id": task["composition_id"],
                "composition_label": task["composition_label"],
                "network_mode": task["network_mode"],
                "seed_count": int(task["seed_count"]),
                "fleet_size": FLEET_SIZE,
                "model_paths": task["model_paths"],
                "model_sha256": task["model_sha256"],
                "seed_results": sorted(
                    rows,
                    key=lambda item: (
                        int(item["seed_index"]),
                        ALGORITHM_ORDER.index(item["algorithm"]),
                    ),
                ),
                "fleet_audit": [audits[index] for index in sorted(audits)],
                "updated_at": _utc_now(),
            }
            if task["composition_kind"] == "pairwise":
                payload.update({
                    "pair_index": int(task["pair_index"]),
                    "dataset_a": task["dataset_a"],
                    "dataset_b": task["dataset_b"],
                    "weights": {task["dataset_a"]: 0.5, task["dataset_b"]: 0.5},
                })
            _write_json(raw_path, payload)
        seed_rows = [row for row in rows if int(row["seed_index"]) == seed_index]
        baseline_values = [float(row["baseline_curtailment_mwh"]) for row in seed_rows]
        if max(baseline_values) - min(baseline_values) > 1e-8:
            raise RuntimeError(
                f"{task['composition_id']} {task['network_mode']} seed {seed_index} "
                "的无控制弃电在算法间不一致"
            )
        print(json.dumps({
            "stage": "seed_checkpoint",
            "kind": task["composition_kind"],
            "composition": task["composition_id"],
            "network": task["network_mode"],
            "completed_seeds": seed_index + 1,
            "seed_count": int(task["seed_count"]),
        }, ensure_ascii=False), flush=True)
    adjustment_count = _apply_centralized_upper_bound(rows)
    payload["seed_results"] = sorted(
        rows,
        key=lambda item: (
            int(item["seed_index"]), ALGORITHM_ORDER.index(item["algorithm"])
        ),
    )
    payload["centralized_upper_bound"] = {
        "version": CENTRALIZED_UB_VERSION,
        "adjusted_seed_count": adjustment_count,
        "seed_count": int(task["seed_count"]),
        "definition": (
            "原生集中式贪心与同 seed 全部已比较可行调度的实际弃电消纳量包络"
        ),
    }
    payload["updated_at"] = _utc_now()
    _write_json(raw_path, payload)
    return payload


def _bootstrap_indices(
    rows: list[dict[str, Any]], draws: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["composition_regime"])].append(index)
    sampled = []
    for _ in range(draws):
        parts = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in groups.values()
        ]
        sampled.append(np.concatenate(parts))
    return sampled


def _summarize_group(
    rows: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    baseline = np.asarray([row["baseline_curtailment_mwh"] for row in rows], dtype=float)
    remaining = np.asarray([row["remaining_curtailment_mwh"] for row in rows], dtype=float)
    reductions = np.asarray([row["mean_reduction_pct"] for row in rows], dtype=float)
    denominator = float(np.sum(baseline))
    weighted = (
        100.0 * (denominator - float(np.sum(remaining))) / denominator
        if denominator > EPS else float("nan")
    )
    bootstrap = []
    if denominator > EPS and draws > 0:
        for indices in _bootstrap_indices(rows, draws, seed):
            selected_baseline = float(np.sum(baseline[indices]))
            if selected_baseline > EPS:
                bootstrap.append(
                    100.0
                    * (selected_baseline - float(np.sum(remaining[indices])))
                    / selected_baseline
                )
    lower, upper = (
        (float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975)))
        if bootstrap else (float("nan"), float("nan"))
    )
    return {
        "seed_count": len(rows),
        "curtailment_reduction_pct": weighted,
        "curtailment_reduction_ci_lower": lower,
        "curtailment_reduction_ci_upper": upper,
        "arithmetic_mean_reduction_pct": float(np.mean(reductions)),
        "std_reduction_pct": float(np.std(reductions, ddof=1)) if len(rows) > 1 else 0.0,
        "baseline_curtailment_mwh_total": denominator,
        "remaining_curtailment_mwh_total": float(np.sum(remaining)),
        "saved_curtailment_mwh_total": float(np.sum(baseline - remaining)),
        "baseline_curtailment_mwh_mean": float(np.mean(baseline)),
        "remaining_curtailment_mwh_mean": float(np.mean(remaining)),
        "saved_curtailment_mwh_mean": float(np.mean(baseline - remaining)),
        "accepted_absorption_mwh_mean": float(np.mean([
            row["accepted_absorption_mwh"] for row in rows
        ])),
        "mean_network_scale": float(np.mean([row["mean_network_scale"] for row in rows])),
        "network_violation_steps": int(sum(row["network_violation_steps"] for row in rows)),
        "max_soc_violation": float(max(row["max_soc_violation"] for row in rows)),
        "runtime_seconds_mean": float(np.mean([row["runtime_seconds"] for row in rows])),
        "runtime_seconds_p95": float(np.quantile([row["runtime_seconds"] for row in rows], 0.95)),
    }


def _summaries(
    rows: list[dict[str, Any]], draws: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["composition_kind"], row["composition_id"],
            row["network_mode"], row["algorithm"],
        )].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        first = group[0]
        summary = {
            "composition_kind": key[0],
            "composition_id": key[1],
            "composition_label": first["composition_label"],
            "network_mode": key[2],
            "algorithm": key[3],
            "algorithm_label": first["algorithm_label"],
            "complexity": first["complexity"],
            **_summarize_group(
                group,
                draws,
                protocol._stable_seed("|".join(key)) % (2**32),
            ),
        }
        for field in ("pair_index", "dataset_a", "dataset_b"):
            if field in first:
                summary[field] = first[field]
        output.append(summary)
    return output


def _paired_comparisons(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]], draws: int
) -> list[dict[str, Any]]:
    by_key = {
        (row["composition_kind"], row["composition_id"], row["network_mode"], row["algorithm"]): row
        for row in summaries
    }
    seed_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed_groups[(
            row["composition_kind"], row["composition_id"],
            row["network_mode"], row["algorithm"],
        )].append(row)
    output = []
    compositions = sorted({(row["composition_kind"], row["composition_id"], row["network_mode"]) for row in rows})
    for kind, composition_id, network_mode in compositions:
        eps_summary = by_key[(kind, composition_id, network_mode, "eps_e21_mixed")]
        eps_rows = sorted(
            seed_groups[(kind, composition_id, network_mode, "eps_e21_mixed")],
            key=lambda row: int(row["seed_index"]),
        )
        for baseline in BASELINE_ORDER:
            baseline_summary = by_key[(kind, composition_id, network_mode, baseline)]
            baseline_rows = sorted(
                seed_groups[(kind, composition_id, network_mode, baseline)],
                key=lambda row: int(row["seed_index"]),
            )
            differences = np.asarray([
                float(eps_row["mean_reduction_pct"])
                - float(baseline_row["mean_reduction_pct"])
                for eps_row, baseline_row in zip(eps_rows, baseline_rows)
            ])
            bootstrap = []
            for indices in _bootstrap_indices(
                eps_rows,
                draws,
                protocol._stable_seed(f"comparison|{kind}|{composition_id}|{network_mode}|{baseline}") % (2**32),
            ):
                bootstrap.append(float(np.mean(differences[indices])))
            lower, upper = (
                (float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975)))
                if bootstrap else (float("nan"), float("nan"))
            )
            output.append({
                "composition_kind": kind,
                "composition_id": composition_id,
                "composition_label": eps_summary["composition_label"],
                "network_mode": network_mode,
                "baseline_algorithm": baseline,
                "baseline_label": mixed.ALGORITHM_DEFINITIONS[baseline]["label"],
                "baseline_complexity": mixed.ALGORITHM_DEFINITIONS[baseline]["complexity"],
                "eps_reduction_pct": eps_summary["curtailment_reduction_pct"],
                "baseline_reduction_pct": baseline_summary["curtailment_reduction_pct"],
                "weighted_delta_pct_points": (
                    eps_summary["curtailment_reduction_pct"]
                    - baseline_summary["curtailment_reduction_pct"]
                ),
                "paired_seed_delta_mean": float(np.mean(differences)),
                "paired_seed_delta_median": float(np.median(differences)),
                "paired_delta_ci_lower": lower,
                "paired_delta_ci_upper": upper,
                "eps_seed_win_fraction": float(np.mean(differences > 1e-9)),
            })
    return output


def _rank_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["composition_kind"], row["composition_id"], row["network_mode"])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row["curtailment_reduction_pct"], reverse=True)
        for rank, row in enumerate(ordered, start=1):
            output.append({
                "composition_kind": key[0],
                "composition_id": key[1],
                "composition_label": row["composition_label"],
                "network_mode": key[2],
                "algorithm": row["algorithm"],
                "algorithm_label": row["algorithm_label"],
                "complexity": row["complexity"],
                "rank": rank,
                "curtailment_reduction_pct": row["curtailment_reduction_pct"],
            })
    return output


def _summary_lookup(summaries: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (row["composition_kind"], row["composition_id"], row["network_mode"], row["algorithm"]): row
        for row in summaries
    }


def _save_figure(fig: plt.Figure, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(path), str(pdf)]


def _annotated_heatmap(
    axis: plt.Axes, values: np.ndarray, xlabels: list[str], ylabels: list[str],
    title: str, vmin: float = 0.0, vmax: float = 100.0, cmap=SEQ,
    annotate: bool = True,
) -> Any:
    image = axis.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    axis.set_xticks(np.arange(len(xlabels)), xlabels, rotation=35, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=7)
    axis.set_title(title)
    if annotate:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if np.isfinite(value):
                    axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=5,
                              color="white" if abs(value) > 0.55 * max(abs(vmin), abs(vmax), 1.0) else "#202020")
    return image


def _plot_scenario_atlas(summaries: list[dict[str, Any]], output: Path) -> list[str]:
    lookup = _summary_lookup(summaries)
    labels = [f"{mixed.ALGORITHM_DEFINITIONS[a]['label']}\n{mixed.ALGORITHM_DEFINITIONS[a]['complexity']}" for a in ALGORITHM_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), constrained_layout=True)
    for axis, network_mode in zip(axes, NETWORK_MODES):
        values = np.asarray([
            [lookup[("scenario", scenario, network_mode, algorithm)]["curtailment_reduction_pct"] for algorithm in ALGORITHM_ORDER]
            for scenario in mixed.SCENARIOS
        ])
        image = _annotated_heatmap(axis, values, labels, list(mixed.SCENARIOS), network_mode)
        fig.colorbar(image, ax=axis, label="Curtailment reduction (%)")
    fig.suptitle("E21 baseline atlas across 14 predefined mixed scenarios")
    return _save_figure(fig, output / "scenario_baseline_atlas.png")


def _plot_scenario_profiles(summaries: list[dict[str, Any]], output: Path) -> list[str]:
    lookup = _summary_lookup(summaries)
    paths = []
    labels = [mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHM_ORDER]
    y = np.arange(len(ALGORITHM_ORDER))
    for scenario in mixed.SCENARIOS:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True, sharey=True)
        for axis, network_mode in zip(axes, NETWORK_MODES):
            rows = [lookup[("scenario", scenario, network_mode, algorithm)] for algorithm in ALGORITHM_ORDER]
            saved = np.asarray([row["curtailment_reduction_pct"] for row in rows])
            remaining = 100.0 - saved
            axis.barh(y, saved, color="#2a9d8f", label="Avoided curtailment")
            axis.barh(y, remaining, left=saved, color="#d9d9d9", label="Remaining curtailment")
            axis.errorbar(
                saved, y,
                xerr=np.asarray([
                    [max(row["curtailment_reduction_pct"] - row["curtailment_reduction_ci_lower"], 0.0) for row in rows],
                    [max(row["curtailment_reduction_ci_upper"] - row["curtailment_reduction_pct"], 0.0) for row in rows],
                ]),
                fmt="none", ecolor="#202020", capsize=2, linewidth=0.8,
            )
            axis.set_xlim(0, 100)
            axis.set_title(network_mode)
            axis.set_xlabel("Share of no-control curtailment (%)")
            axis.grid(axis="x", alpha=0.2)
        axes[0].set_yticks(y, labels, fontsize=8)
        axes[0].invert_yaxis()
        axes[1].legend(frameon=False, loc="lower right")
        fig.suptitle(f"{scenario}: avoided and remaining curtailment")
        paths.extend(_save_figure(fig, output / "scenario_profiles" / f"{scenario}.png"))
    return paths


def _plot_pair_overview(summaries: list[dict[str, Any]], output: Path) -> list[str]:
    lookup = _summary_lookup(summaries)
    pair_specs = pairwise._pair_specs()
    paths = []
    for network_mode in NETWORK_MODES:
        ordered = sorted(
            pair_specs,
            key=lambda row: lookup[("pairwise", row["pair_id"], network_mode, "eps_e21_mixed")]["curtailment_reduction_pct"],
            reverse=True,
        )
        values = np.asarray([
            [lookup[("pairwise", row["pair_id"], network_mode, algorithm)]["curtailment_reduction_pct"] for row in ordered]
            for algorithm in ALGORITHM_ORDER
        ])
        fig, axis = plt.subplots(figsize=(24, 7), constrained_layout=True)
        image = axis.imshow(values, vmin=0, vmax=100, cmap=SEQ, aspect="auto")
        axis.set_yticks(np.arange(len(ALGORITHM_ORDER)), [mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHM_ORDER], fontsize=8)
        positions = np.arange(0, len(ordered), 5)
        axis.set_xticks(positions, [ordered[index]["pair_id"] for index in positions], rotation=60, ha="right", fontsize=6)
        axis.set_xlabel("105 pairs, sorted by E21 mixed EPS performance")
        axis.set_title(f"Pairwise baseline overview: {network_mode}")
        fig.colorbar(image, ax=axis, label="Curtailment reduction (%)")
        paths.extend(_save_figure(fig, output / f"pairwise_baseline_overview_{network_mode}.png"))
    return paths


def _pair_matrix(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    network_mode: str, algorithm: str,
) -> np.ndarray:
    order = list(pairwise.PLOT_ORDER)
    indices = {dataset: index for index, dataset in enumerate(order)}
    matrix = np.full((len(order), len(order)), np.nan)
    for spec in pairwise._pair_specs():
        value = lookup[("pairwise", spec["pair_id"], network_mode, algorithm)]["curtailment_reduction_pct"]
        first, second = indices[spec["dataset_a"]], indices[spec["dataset_b"]]
        matrix[first, second] = value
        matrix[second, first] = value
    return matrix


def _plot_pair_matrices(summaries: list[dict[str, Any]], output: Path) -> list[str]:
    lookup = _summary_lookup(summaries)
    labels = [pairwise.SHORT_NAMES[dataset] for dataset in pairwise.PLOT_ORDER]
    paths = []
    for network_mode in NETWORK_MODES:
        fig, axes = plt.subplots(2, 5, figsize=(26, 12), constrained_layout=True)
        last_image = None
        for axis, algorithm in zip(axes.flat, ALGORITHM_ORDER):
            matrix = _pair_matrix(lookup, network_mode, algorithm)
            last_image = axis.imshow(matrix, vmin=0, vmax=100, cmap=SEQ)
            axis.set_title(mixed.ALGORITHM_DEFINITIONS[algorithm]["label"], fontsize=9)
            axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=5)
            axis.set_yticks(np.arange(len(labels)), labels, fontsize=5)
        if last_image is not None:
            fig.colorbar(last_image, ax=axes.ravel().tolist(), label="Curtailment reduction (%)", shrink=0.75)
        fig.suptitle(f"All 105 pairwise combinations by algorithm: {network_mode}")
        paths.extend(_save_figure(fig, output / f"pairwise_algorithm_matrices_{network_mode}.png"))
    return paths


def _plot_eps_advantage(comparisons: list[dict[str, Any]], output: Path) -> list[str]:
    lookup = {
        (row["composition_kind"], row["composition_id"], row["network_mode"], row["baseline_algorithm"]): row
        for row in comparisons
    }
    labels = [pairwise.SHORT_NAMES[dataset] for dataset in pairwise.PLOT_ORDER]
    indices = {dataset: index for index, dataset in enumerate(pairwise.PLOT_ORDER)}
    paths = []
    for network_mode in NETWORK_MODES:
        matrices = []
        for baseline in BASELINE_ORDER:
            matrix = np.full((len(labels), len(labels)), np.nan)
            for spec in pairwise._pair_specs():
                value = lookup[("pairwise", spec["pair_id"], network_mode, baseline)]["weighted_delta_pct_points"]
                first, second = indices[spec["dataset_a"]], indices[spec["dataset_b"]]
                matrix[first, second] = value
                matrix[second, first] = value
            matrices.append(matrix)
        limit = max(5.0, max(float(np.nanmax(np.abs(matrix))) for matrix in matrices))
        fig, axes = plt.subplots(2, 4, figsize=(22, 12), constrained_layout=True)
        last_image = None
        for axis, baseline, matrix in zip(axes.flat, BASELINE_ORDER, matrices):
            last_image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap=SEQ)
            axis.set_title(f"vs {mixed.ALGORITHM_DEFINITIONS[baseline]['label']}", fontsize=9)
            axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=5)
            axis.set_yticks(np.arange(len(labels)), labels, fontsize=5)
        if last_image is not None:
            fig.colorbar(last_image, ax=axes.ravel().tolist(), label="E21 EPS advantage (percentage points)", shrink=0.75)
        fig.suptitle(f"E21 mixed EPS paired advantage across 105 pairs: {network_mode}")
        paths.extend(_save_figure(fig, output / f"pairwise_eps_advantage_{network_mode}.png"))
    return paths


def _plot_distributions(summaries: list[dict[str, Any]], output: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True, sharey=True)
    labels = [mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHM_ORDER]
    for axis, network_mode in zip(axes, NETWORK_MODES):
        data = [
            [row["curtailment_reduction_pct"] for row in summaries if row["composition_kind"] == "pairwise" and row["network_mode"] == network_mode and row["algorithm"] == algorithm]
            for algorithm in ALGORITHM_ORDER
        ]
        axis.boxplot(data, vert=False, labels=labels, showfliers=False)
        for index, values in enumerate(data, start=1):
            axis.scatter(values, np.full(len(values), index), s=7, alpha=0.25, color=legacy.FIG4D_COLORS.get(ALGORITHM_ORDER[index - 1], "#4e79a7"))
        axis.set_xlim(0, 105)
        axis.set_title(network_mode)
        axis.set_xlabel("Curtailment reduction across 105 pairs (%)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Pairwise performance distributions")
    return _save_figure(fig, output / "pairwise_algorithm_distribution.png")


def _plot_network_penalty(summaries: list[dict[str, Any]], output: Path) -> list[str]:
    lookup = _summary_lookup(summaries)
    algorithm_labels = [mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHM_ORDER]
    scenario_values = np.asarray([
        [
            lookup[("scenario", scenario, "aggregate", algorithm)]["curtailment_reduction_pct"]
            - lookup[("scenario", scenario, "ieee33", algorithm)]["curtailment_reduction_pct"]
            for algorithm in ALGORITHM_ORDER
        ]
        for scenario in mixed.SCENARIOS
    ])
    limit = max(5.0, float(np.nanmax(np.abs(scenario_values))))
    fig, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    image = _annotated_heatmap(
        axis, scenario_values, algorithm_labels, list(mixed.SCENARIOS),
        "Aggregate minus IEEE-33 performance", -limit, limit, "RdBu_r",
    )
    fig.colorbar(image, ax=axis, label="Network penalty (percentage points)")
    paths = _save_figure(fig, output / "scenario_network_penalty.png")
    aggregate = _pair_matrix(lookup, "aggregate", "eps_e21_mixed")
    constrained = _pair_matrix(lookup, "ieee33", "eps_e21_mixed")
    matrix = aggregate - constrained
    labels = [pairwise.SHORT_NAMES[dataset] for dataset in pairwise.PLOT_ORDER]
    limit = max(5.0, float(np.nanmax(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap=SEQ)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axis.set_title("E21 mixed EPS network penalty across 105 pairs")
    fig.colorbar(image, ax=axis, label="Aggregate minus IEEE-33 (percentage points)")
    paths.extend(_save_figure(fig, output / "pairwise_network_penalty.png"))
    return paths


def _plot_overall(summaries: list[dict[str, Any]], output: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True, sharex=True)
    y = np.arange(len(ALGORITHM_ORDER))
    labels = [f"{mixed.ALGORITHM_DEFINITIONS[a]['label']}\n{mixed.ALGORITHM_DEFINITIONS[a]['complexity']}" for a in ALGORITHM_ORDER]
    for row_index, kind in enumerate(("scenario", "pairwise")):
        for column_index, network_mode in enumerate(NETWORK_MODES):
            axis = axes[row_index, column_index]
            groups = [
                np.asarray([row["curtailment_reduction_pct"] for row in summaries if row["composition_kind"] == kind and row["network_mode"] == network_mode and row["algorithm"] == algorithm])
                for algorithm in ALGORITHM_ORDER
            ]
            means = np.asarray([float(np.mean(values)) for values in groups])
            errors = np.asarray([float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in groups])
            axis.errorbar(means, y, xerr=errors, fmt="o", color="#202020", capsize=3)
            for index, values in enumerate(groups):
                axis.scatter(values, np.full(len(values), index), s=9, alpha=0.2, color=legacy.FIG4D_COLORS.get(ALGORITHM_ORDER[index], "#4e79a7"))
            axis.set_xlim(0, 105)
            axis.set_title(f"{kind}: {network_mode}")
            axis.grid(axis="x", alpha=0.2)
            if column_index == 0:
                axis.set_yticks(y, labels, fontsize=7)
            else:
                axis.set_yticks(y, [])
            axis.invert_yaxis()
    fig.supxlabel("Curtailment reduction (%)")
    fig.suptitle("E21 performance across 14 predefined scenarios and 105 pairwise mixes")
    return _save_figure(fig, output / "e21_all_compositions_overall.png")


def _write_design(output: Path, args: argparse.Namespace, task_count: int) -> None:
    design = f"""# E21 弃电 Baseline 补充实验设计

## 实验范围

- 14 个预定义混合场景和 105 个无序 50/50 两两组合。
- aggregate 与 IEEE-33 两种网络。
- 每个组合和网络 {args.seed_count} 个配对 seed、5000 台逻辑设备、288 个五分钟步。
- 数据驱动可用率；纯仿真设备参数和外部可再生能源输入。
- 共 {task_count} 个 composition-network 任务，最多 {task_count * args.seed_count * len(ALGORITHM_ORDER)} 条逐算法结果。

## 对比算法

|算法|角色|在线复杂度|
|-|-|-|
"""
    roles = {algorithm: "Baseline" for algorithm in BASELINE_ORDER}
    roles["eps_e20_pooled"] = "训练消融"
    roles["eps_e21_mixed"] = "主算法"
    for algorithm in ALGORITHM_ORDER:
        definition = mixed.ALGORITHM_DEFINITIONS[algorithm]
        design += f"|{definition['label']}|{roles[algorithm]}|{definition['complexity']}|\n"
    design += """

## 弃电口径

每个 seed 的无控制弃电为逐时正富余能量。算法只能通过同一步实际接受的正向充电响应减少弃电。30 个 seed 的主结果使用总能量比值：

`100 * (sum(baseline) - sum(remaining)) / sum(baseline)`。

置信区间采用配对、按组成扰动分层的 bootstrap。两两组合始终保持 50/50，不引入方向性组成扰动。

## Centralized greedy UB 口径

- 先计算原生逐时集中式贪心调度。
- 再对同一 composition、网络和 seed 的全部已比较可行调度取实际弃电消纳量包络。
- 集中式控制器拥有完整设备状态和调度信息，理论上至少可以复现该包络对应的可行调度。
- 因此该列是相对于本实验算法集合的严格逐 seed upper bound，不再把单步低 SOC 贪心误称为完整时间域数学最优解。
- raw JSON 保留 `native_greedy_*`、`upper_bound_adjusted` 和 `upper_bound_source_algorithm`，可以审计原生贪心与最终 UB 的差别。

## 公平性

- 同一 composition、网络和 seed 的所有算法共享设备、负荷、可再生能源、可用率随机数和网络条件。
- aggregate 与 IEEE-33 使用相同 base seed，便于计算网络约束损失。
- E20 pooled 与 E21 mixed 使用已冻结模型，不在 pairwise 测试上重新训练。
- Centralized greedy UB 对所有比较算法逐 seed 执行上界断言。

## 断点

每完成一个算法即原子写入 raw JSON；后台任务中断后使用相同命令可继续。
"""
    (output / "EXPERIMENT_DESIGN.md").write_text(design, encoding="utf-8")


def _value(row: dict[str, Any], field: str) -> float:
    return float(row[field])


def _mean_sd_range(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        float(np.min(array)),
        float(np.max(array)),
    )


def _algorithm_result_table(
    summaries: list[dict[str, Any]], composition_kind: str
) -> str:
    lines = [
        "|算法|复杂度|aggregate 均值 +/- 标准差|IEEE-33 均值 +/- 标准差|",
        "|-|-|-|-|",
    ]
    for algorithm in ALGORITHM_ORDER:
        definition = mixed.ALGORITHM_DEFINITIONS[algorithm]
        cells = []
        for network_mode in NETWORK_MODES:
            values = [
                _value(row, "curtailment_reduction_pct")
                for row in summaries
                if row["composition_kind"] == composition_kind
                and row["network_mode"] == network_mode
                and row["algorithm"] == algorithm
            ]
            mean, standard_deviation, _, _ = _mean_sd_range(values)
            cells.append(f"{mean:.2f}% +/- {standard_deviation:.2f}")
        lines.append(
            f"|{definition['label']}|{definition['complexity']}|{cells[0]}|{cells[1]}|"
        )
    return "\n".join(lines)


def _scenario_profile_section(
    index: int,
    scenario_id: str,
    summaries: list[dict[str, Any]],
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> str:
    scenario = mixed.SCENARIOS[scenario_id]
    composition = "、".join(
        f"{pairwise.SHORT_NAMES.get(dataset, dataset)} {weight * 100:.2f}%"
        for dataset, weight in scenario["weights"].items()
    )
    table = [
        "|算法|复杂度|aggregate 弃电降低率（95% CI）|IEEE-33 弃电降低率（95% CI）|",
        "|-|-|-|-|",
    ]
    for algorithm in ALGORITHM_ORDER:
        cells = []
        for network_mode in NETWORK_MODES:
            row = lookup[("scenario", scenario_id, network_mode, algorithm)]
            cells.append(
                f"{_value(row, 'curtailment_reduction_pct'):.2f}% "
                f"[{_value(row, 'curtailment_reduction_ci_lower'):.2f}%, "
                f"{_value(row, 'curtailment_reduction_ci_upper'):.2f}%]"
            )
        definition = mixed.ALGORITHM_DEFINITIONS[algorithm]
        table.append(
            f"|{definition['label']}|{definition['complexity']}|{cells[0]}|{cells[1]}|"
        )
    eps_aggregate = lookup[("scenario", scenario_id, "aggregate", "eps_e21_mixed")]
    eps_ieee33 = lookup[("scenario", scenario_id, "ieee33", "eps_e21_mixed")]
    aggregate_rows = [
        row for row in summaries
        if row["composition_kind"] == "scenario"
        and row["composition_id"] == scenario_id
        and row["network_mode"] == "aggregate"
    ]
    ieee33_rows = [
        row for row in summaries
        if row["composition_kind"] == "scenario"
        and row["composition_id"] == scenario_id
        and row["network_mode"] == "ieee33"
    ]
    best_aggregate = max(aggregate_rows, key=lambda row: _value(row, "curtailment_reduction_pct"))
    best_ieee33 = max(ieee33_rows, key=lambda row: _value(row, "curtailment_reduction_pct"))
    eps_penalty = (
        _value(eps_aggregate, "curtailment_reduction_pct")
        - _value(eps_ieee33, "curtailment_reduction_pct")
    )
    table_text = "\n".join(table)
    return f"""# {index}. scenario_profiles/{scenario_id}

![{scenario_id} 场景算法对比](figures/scenario_profiles/{scenario_id}.png)

文件：

- figures/scenario_profiles/{scenario_id}.png
- figures/scenario_profiles/{scenario_id}.pdf
- 数据：data/baseline_summary.csv、data/baseline_by_seed.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

该图展示预定义混合场景 {scenario_id}（{scenario['label']}）。5000 台逻辑设备按以下来源比例组成：{composition}。左右面板复用相同 seed、设备样本、外部能源输入和可用率随机数，分别测试无配电网拓扑的 aggregate 条件与启用 IEEE-33 网络反馈的条件。

### 横纵坐标

- 横坐标：相对于 No coordination 初始弃电的比例，固定为 0% 至 100%。
- 纵坐标：10 种算法；算法定义及复杂度见 EXPERIMENT_DESIGN.md。
- 绿色：算法避免的弃电比例，即弃电降低率。
- 灰色：控制后仍然存在的弃电比例，等于 100% 减去弃电降低率。
- 黑色误差线：30 个配对 seed、4000 次分层 bootstrap 得到的 95% 置信区间。

### 结果

{table_text}

E21 mixed EPS 在 aggregate 下达到 {_value(eps_aggregate, 'curtailment_reduction_pct'):.2f}%，在 IEEE-33 下达到 {_value(eps_ieee33, 'curtailment_reduction_pct'):.2f}%；两者相差 {eps_penalty:.2f} 个百分点。aggregate 下最高值为 {best_aggregate['algorithm_label']} 的 {_value(best_aggregate, 'curtailment_reduction_pct'):.2f}%，IEEE-33 下最高值为 {best_ieee33['algorithm_label']} 的 {_value(best_ieee33, 'curtailment_reduction_pct'):.2f}%。该差值只表示本场景中加入网络反馈后的净变化，不能单独解释为算法本身的统计泛化能力变化。
"""


def _build_figure_descriptions(
    summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> str:
    lookup = _summary_lookup(summaries)
    scenario_algorithm_table = _algorithm_result_table(summaries, "scenario")
    pair_algorithm_table = _algorithm_result_table(summaries, "pairwise")
    scenario_eps_means = {
        network_mode: float(np.mean([
            _value(row, "curtailment_reduction_pct")
            for row in summaries
            if row["composition_kind"] == "scenario"
            and row["network_mode"] == network_mode
            and row["algorithm"] == "eps_e21_mixed"
        ]))
        for network_mode in NETWORK_MODES
    }
    eps_pair_rows = {
        network_mode: [
            row for row in summaries
            if row["composition_kind"] == "pairwise"
            and row["network_mode"] == network_mode
            and row["algorithm"] == "eps_e21_mixed"
        ]
        for network_mode in NETWORK_MODES
    }

    sections = [f"""# E21 弃电 baseline 补充实验图片说明

本文件逐一说明 `figures/` 下的 25 张独立结果图。每张 PNG 均有同名 PDF。所有结果来自 14 个预定义混合场景、105 个无序 50/50 两两组合、aggregate 与 IEEE-33 两种网络条件、每个条件 30 个配对 seed。每个 seed 使用 5000 台逻辑设备和 288 个五分钟时间步。

弃电降低率统一定义为：

`100 * (所有 seed 的无控制弃电总量 - 所有 seed 的控制后剩余弃电总量) / 所有 seed 的无控制弃电总量`。

因此该指标是能量加权总量比，不是逐 seed 百分比的简单平均。`aggregate` 不施加显式配电网拓扑约束；`ieee33` 启用 IEEE-33 网络反馈和响应缩放。除专门标注外，误差区间均使用 4000 次分层 bootstrap。

# 1. scenario_baseline_atlas

![14 个混合场景的算法图谱](figures/scenario_baseline_atlas.png)

文件：

- figures/scenario_baseline_atlas.png
- figures/scenario_baseline_atlas.pdf
- 数据：data/baseline_summary.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

该图在同一坐标系中比较 14 个预定义物理混合场景和 10 种算法。左图为 aggregate，右图为 IEEE-33。它回答不同混合来源、行业构成和空间部署下，各算法能够减少多少无控制弃电，以及显式网络反馈是否改变算法排序。

### 横纵坐标

- 横坐标：算法名称及其在线复杂度。
- 纵坐标：S1-A 至 S6-C 共 14 个预定义混合场景。
- 单元格颜色和数字：30 个 seed 的能量加权弃电降低率，范围 0% 至 100%；数值越高表示避免的弃电越多。
- 左右面板：aggregate 与 IEEE-33，不是重复实验的置信区间上下界。

### 结果

{scenario_algorithm_table}

E21 mixed EPS 的跨场景平均值由 aggregate 的 {scenario_eps_means['aggregate']:.2f}% 降至 IEEE-33 的 {scenario_eps_means['ieee33']:.2f}%。场景间标准差较大，说明数据构成和空间约束对结果有实质影响，不能只报告一个总体均值；具体算法均值和标准差见上表。
"""]

    for offset, scenario_id in enumerate(mixed.SCENARIOS, start=2):
        sections.append(
            _scenario_profile_section(offset, scenario_id, summaries, lookup)
        )

    next_index = 2 + len(mixed.SCENARIOS)
    for network_mode in NETWORK_MODES:
        rows = eps_pair_rows[network_mode]
        values = [_value(row, "curtailment_reduction_pct") for row in rows]
        minimum = min(rows, key=lambda row: _value(row, "curtailment_reduction_pct"))
        maximum = max(rows, key=lambda row: _value(row, "curtailment_reduction_pct"))
        mean, standard_deviation, _, _ = _mean_sd_range(values)
        sections.append(f"""# {next_index}. pairwise_baseline_overview_{network_mode}

![{network_mode} 两两组合算法总览](figures/pairwise_baseline_overview_{network_mode}.png)

文件：

- figures/pairwise_baseline_overview_{network_mode}.png
- figures/pairwise_baseline_overview_{network_mode}.pdf
- 数据：data/baseline_summary.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

15 个真实数据源构成 105 个无序 50/50 两两混合，每个组合均在 {network_mode} 条件下运行 10 种算法和 30 个配对 seed。该图用于观察算法在全部 pair 上的稳定性，而不是只选择效果最好的组合。

### 横纵坐标

- 横坐标：105 个 pair，按 E21 mixed EPS 在当前网络条件下的弃电降低率从高到低排序；每隔 5 个组合显示一个 pair ID。
- 纵坐标：10 种算法。
- 颜色：弃电降低率，统一使用 0% 至 100% 色标。
- 同一列：完全相同的两数据集组合，可横向比较不同算法。

### 结果

{pair_algorithm_table}

在 {network_mode} 下，E21 mixed EPS 的 105 组合平均值为 {mean:.2f}% +/- {standard_deviation:.2f}。最低组合是 {minimum['composition_id']}（{minimum['composition_label']}），为 {_value(minimum, 'curtailment_reduction_pct'):.2f}%；最高组合是 {maximum['composition_id']}（{maximum['composition_label']}），为 {_value(maximum, 'curtailment_reduction_pct'):.2f}%。横向排序仅为提高可读性，不参与训练或指标计算。
""")
        next_index += 1

    for network_mode in NETWORK_MODES:
        eps_pair_mean = float(np.mean([
            _value(row, "curtailment_reduction_pct")
            for row in eps_pair_rows[network_mode]
        ]))
        sections.append(f"""# {next_index}. pairwise_algorithm_matrices_{network_mode}

![{network_mode} 各算法两两组合矩阵](figures/pairwise_algorithm_matrices_{network_mode}.png)

文件：

- figures/pairwise_algorithm_matrices_{network_mode}.png
- figures/pairwise_algorithm_matrices_{network_mode}.pdf
- 数据：data/baseline_summary.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

该组图把 105 个 pair 按数据集身份恢复成 15 x 15 对称矩阵，并为每种算法单独给出一个面板。它用于判断低效果是否集中在某些数据集参与的组合中，以及不同算法是否共享相同的数据依赖模式。

### 横纵坐标

- 横坐标和纵坐标：相同顺序的 15 个源数据集。
- 非对角单元格：横纵两个数据集各占 50% 的混合结果。
- 对角线：未运行“数据集与自身”的 pair，故保持空白，不代表 0%。
- 颜色：{network_mode} 下的弃电降低率，所有算法共享 0% 至 100% 色标。
- 10 个子图：分别对应 10 种算法。

### 结果

{pair_algorithm_table}

矩阵显示的是组合结构，而不是 105 个样本的时间顺序。E21 mixed EPS 在 {network_mode} 下的平均弃电降低率为 {eps_pair_mean:.2f}%。对角线没有数据，因此不能用该图推断单一数据集自身的性能。
""")
        next_index += 1

    for network_mode in NETWORK_MODES:
        advantage_lines = [
            "|对比 baseline|复杂度|E21 平均优势|优势范围|E21 胜出的组合比例|",
            "|-|-|-|-|-|",
        ]
        for baseline in BASELINE_ORDER:
            rows = [
                row for row in comparisons
                if row["composition_kind"] == "pairwise"
                and row["network_mode"] == network_mode
                and row["baseline_algorithm"] == baseline
            ]
            values = [_value(row, "weighted_delta_pct_points") for row in rows]
            positive_fraction = 100.0 * float(np.mean(np.asarray(values) > 0.0))
            definition = mixed.ALGORITHM_DEFINITIONS[baseline]
            advantage_lines.append(
                f"|{definition['label']}|{definition['complexity']}|"
                f"{np.mean(values):.2f} 个百分点|{min(values):.2f} 至 {max(values):.2f}|"
                f"{positive_fraction:.1f}%|"
            )
        advantage_table = "\n".join(advantage_lines)
        sections.append(f"""# {next_index}. pairwise_eps_advantage_{network_mode}

![E21 mixed EPS 在 {network_mode} 下的配对优势](figures/pairwise_eps_advantage_{network_mode}.png)

文件：

- figures/pairwise_eps_advantage_{network_mode}.png
- figures/pairwise_eps_advantage_{network_mode}.pdf
- 数据：data/eps_vs_baseline.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

该图逐 pair 计算 E21 mixed EPS 与 8 个 baseline 的弃电降低率差值。比较使用相同设备、能源输入、网络条件和 seed，因此颜色反映配对差异，不混入不同随机样本造成的偏差。

### 横纵坐标

- 每个子图横纵坐标：15 个源数据集；非对角单元格对应 50/50 pair。
- 子图标题：当前比较的 baseline。
- 颜色：`E21 mixed EPS 弃电降低率 - baseline 弃电降低率`，单位为百分点。
- 正值：E21 mixed EPS 更高；负值：baseline 更高；0：两者相同。
- 对角线为空白，因为没有数据集与自身的 pair 实验。

### 结果

{advantage_table}

该图比较的是效果差值，不表示在线复杂度相同。尤其 MPC 和 Centralized greedy UB 使用更多全局信息；E21 mixed EPS 的重点是 O(1) 在线广播控制下与这些高信息基线之间的效果差距。
""")
        next_index += 1

    distribution_lines = [
        "|算法|aggregate 中位数 [Q1, Q3]|IEEE-33 中位数 [Q1, Q3]|",
        "|-|-|-|",
    ]
    for algorithm in ALGORITHM_ORDER:
        cells = []
        for network_mode in NETWORK_MODES:
            values = np.asarray([
                _value(row, "curtailment_reduction_pct")
                for row in summaries
                if row["composition_kind"] == "pairwise"
                and row["network_mode"] == network_mode
                and row["algorithm"] == algorithm
            ])
            cells.append(
                f"{np.median(values):.2f}% "
                f"[{np.quantile(values, 0.25):.2f}%, {np.quantile(values, 0.75):.2f}%]"
            )
        definition = mixed.ALGORITHM_DEFINITIONS[algorithm]
        distribution_lines.append(f"|{definition['label']}|{cells[0]}|{cells[1]}|")
    distribution_table = "\n".join(distribution_lines)
    sections.append(f"""# {next_index}. pairwise_algorithm_distribution

![两两组合的算法效果分布](figures/pairwise_algorithm_distribution.png)

文件：

- figures/pairwise_algorithm_distribution.png
- figures/pairwise_algorithm_distribution.pdf
- 数据：data/baseline_summary.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

该图把每种算法在 105 个 pair 上的结果作为一个分布展示，用于区分“平均值较高”和“多数数据组合均稳定较高”。左图为 aggregate，右图为 IEEE-33。

### 横纵坐标

- 横坐标：每个 pair 的能量加权弃电降低率。
- 纵坐标：10 种算法。
- 半透明散点：单个 pair 的结果，共 105 点/算法/网络。
- 箱体：第 25 至第 75 百分位；箱内线为中位数；须线为常规 1.5 IQR 范围；离群点本身不由箱线图重复绘制，但仍可在散点中看到。

### 结果

{distribution_table}

IEEE-33 下多个算法的下四分位显著降低，说明网络约束影响具有组合依赖性。该分布不把 105 个 pair 当作统计独立的真实电网样本，只用于描述本实验组合空间。
""")
    next_index += 1

    penalty_lookup = {
        (row["composition_kind"], row["composition_id"], row["algorithm"], row["network_mode"]):
        _value(row, "curtailment_reduction_pct")
        for row in summaries
    }
    scenario_penalty_lines = [
        "|算法|14 场景平均网络损失|最小值|最大值|",
        "|-|-|-|-|",
    ]
    for algorithm in ALGORITHM_ORDER:
        values = [
            penalty_lookup[("scenario", scenario_id, algorithm, "aggregate")]
            - penalty_lookup[("scenario", scenario_id, algorithm, "ieee33")]
            for scenario_id in mixed.SCENARIOS
        ]
        scenario_penalty_lines.append(
            f"|{mixed.ALGORITHM_DEFINITIONS[algorithm]['label']}|{np.mean(values):.2f} 个百分点|"
            f"{min(values):.2f}|{max(values):.2f}|"
        )
    scenario_penalty_table = "\n".join(scenario_penalty_lines)
    eps_scenario_penalties = [
        penalty_lookup[("scenario", scenario_id, "eps_e21_mixed", "aggregate")]
        - penalty_lookup[("scenario", scenario_id, "eps_e21_mixed", "ieee33")]
        for scenario_id in mixed.SCENARIOS
    ]
    eps_scenario_penalty_ids = list(mixed.SCENARIOS)
    eps_scenario_penalty_minimum = int(np.argmin(eps_scenario_penalties))
    eps_scenario_penalty_maximum = int(np.argmax(eps_scenario_penalties))
    sections.append(f"""# {next_index}. scenario_network_penalty

![预定义场景中的网络约束损失](figures/scenario_network_penalty.png)

文件：

- figures/scenario_network_penalty.png
- figures/scenario_network_penalty.pdf
- 数据：data/baseline_summary.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

该图对每个预定义场景和算法计算 aggregate 与 IEEE-33 的弃电降低率差，用于量化加入馈线网络反馈后损失了多少原本可实现的聚合吸收效果。

### 横纵坐标

- 横坐标：10 种算法。
- 纵坐标：14 个预定义混合场景。
- 颜色和数字：`aggregate 弃电降低率 - IEEE-33 弃电降低率`，单位为百分点。
- 正值：IEEE-33 条件降低了弃电吸收效果；负值：该配对 seed 与反馈策略下 IEEE-33 结果反而更高，不能解释成网络创造了额外物理能量。

### 结果

{scenario_penalty_table}

E21 mixed EPS 的场景平均网络损失为 {np.mean(eps_scenario_penalties):.2f} 个百分点。最小值出现在 {eps_scenario_penalty_ids[eps_scenario_penalty_minimum]}，为 {eps_scenario_penalties[eps_scenario_penalty_minimum]:.2f} 个百分点；最大值出现在 {eps_scenario_penalty_ids[eps_scenario_penalty_maximum]}，为 {eps_scenario_penalties[eps_scenario_penalty_maximum]:.2f} 个百分点。结果表明平均值不能替代逐场景网络审计。
""")
    next_index += 1

    pair_ids = [spec["pair_id"] for spec in pairwise._pair_specs()]
    eps_penalties = [
        penalty_lookup[("pairwise", pair_id, "eps_e21_mixed", "aggregate")]
        - penalty_lookup[("pairwise", pair_id, "eps_e21_mixed", "ieee33")]
        for pair_id in pair_ids
    ]
    minimum_index = int(np.argmin(eps_penalties))
    maximum_index = int(np.argmax(eps_penalties))
    pair_labels = {
        row["composition_id"]: row["composition_label"]
        for row in summaries if row["composition_kind"] == "pairwise"
    }
    sections.append(f"""# {next_index}. pairwise_network_penalty

![两两组合中的 E21 mixed EPS 网络约束损失](figures/pairwise_network_penalty.png)

文件：

- figures/pairwise_network_penalty.png
- figures/pairwise_network_penalty.pdf
- 数据：data/baseline_summary.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

该图只针对 E21 mixed EPS，把 105 个 pair 的 aggregate 与 IEEE-33 结果逐一相减并恢复成 15 x 15 数据集矩阵，用于定位哪些数据源组合对馈线约束最敏感。

### 横纵坐标

- 横坐标和纵坐标：15 个源数据集。
- 非对角单元格：对应 50/50 pair 的网络损失。
- 颜色：`aggregate - IEEE-33`，单位为百分点；正值表示加入网络反馈后效果降低。
- 对角线为空白，不是 0%。

### 结果

E21 mixed EPS 在 105 个 pair 上的平均网络损失为 {np.mean(eps_penalties):.2f} 个百分点。最小值为 {eps_penalties[minimum_index]:.2f} 个百分点，来自 {pair_ids[minimum_index]}（{pair_labels[pair_ids[minimum_index]]}）；最大值为 {eps_penalties[maximum_index]:.2f} 个百分点，来自 {pair_ids[maximum_index]}（{pair_labels[pair_ids[maximum_index]]}）。这说明网络可实施性不能由 aggregate 效果直接外推。
""")
    next_index += 1

    sections.append(f"""# {next_index}. e21_all_compositions_overall

![E21 全部场景与两两组合总体结果](figures/e21_all_compositions_overall.png)

文件：

- figures/e21_all_compositions_overall.png
- figures/e21_all_compositions_overall.pdf
- 数据：data/baseline_summary.csv
- 生成脚本：src/extra/ieee33_device_day_simulation/figures/run_e21_curtailment_baseline_supplement.py

### 实验内容

该总图同时展示预定义场景与两两组合在 aggregate 和 IEEE-33 下的算法分布。四个面板使用同一横轴，便于区分场景设计、pair 设计和网络条件造成的变化。

### 横纵坐标

- 横坐标：能量加权弃电降低率，0% 至 105%。
- 纵坐标：算法名称及在线复杂度。
- 半透明散点：一个预定义场景或一个 pair 的结果。
- 黑色圆点：跨组合算术均值。
- 黑色误差线：跨组合标准差，不是 bootstrap 置信区间。
- 左上/右上：14 个预定义场景的 aggregate/IEEE-33；左下/右下：105 个 pair 的 aggregate/IEEE-33。

### 结果

预定义场景：

{scenario_algorithm_table}

两两组合：

{pair_algorithm_table}

E21 mixed EPS 在预定义场景中的均值为 {scenario_eps_means['aggregate']:.2f}%（aggregate）和 {scenario_eps_means['ieee33']:.2f}%（IEEE-33），在两两组合中为 {np.mean([_value(row, 'curtailment_reduction_pct') for row in eps_pair_rows['aggregate']]):.2f}% 和 {np.mean([_value(row, 'curtailment_reduction_pct') for row in eps_pair_rows['ieee33']]):.2f}%。该图中的误差线描述组合间异质性，不能作为模型参数不确定性的置信区间。
""")
    return "\n\n".join(sections).rstrip() + "\n"


def _write_documents(
    output: Path,
    figures: list[str],
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    scenario_rows = [row for row in summaries if row["composition_kind"] == "scenario"]
    pair_rows = [row for row in summaries if row["composition_kind"] == "pairwise"]
    text = f"""# E21 弃电 Baseline 补充实验

该目录完整比较 14 个预定义混合场景和 105 个两两组合中的全部 baseline、E20 pooled EPS 和 E21 mixed EPS。

## 数据

- `data/baseline_by_seed.csv`：{len(scenario_rows) * 0 + sum(row['seed_count'] for row in summaries)} 条逐 seed 算法结果。
- `data/baseline_summary.csv`：{len(summaries)} 条能量加权汇总。
- `data/eps_vs_baseline.csv`：E21 mixed EPS 与8个baseline的配对差值。
- `data/algorithm_ranks.csv`：每个组合中的算法排名。

## 图

- `figures/scenario_baseline_atlas`：14场景完整算法热图。
- `figures/scenario_profiles/`：每个场景控制前后弃电构成。
- `figures/pairwise_baseline_overview_*`：105个pair的算法总览。
- `figures/pairwise_algorithm_matrices_*`：每个算法的15×15组合矩阵。
- `figures/pairwise_eps_advantage_*`：E21相对每个baseline的差值矩阵。
- `figures/*network_penalty*`：IEEE-33网络约束损失。
- `figures/e21_all_compositions_overall`：场景与pairwise分组总图。

生成图文件数：{len(figures)}。预定义场景汇总行数：{len(scenario_rows)}；pairwise汇总行数：{len(pair_rows)}。
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    descriptions = _build_figure_descriptions(summaries, comparisons)
    (output / "FIGURE_DESCRIPTIONS.md").write_text(descriptions, encoding="utf-8")


def _tasks(args: argparse.Namespace, model_paths: dict[str, dict[str, Path]]) -> list[dict[str, Any]]:
    tasks = []
    if args.kind in {"all", "scenario"}:
        scenario_ids = list(mixed.SCENARIOS)[: args.max_scenarios]
        for network_mode in NETWORK_MODES:
            for scenario_id in scenario_ids:
                tasks.append({
                    "composition_kind": "scenario",
                    "composition_id": scenario_id,
                    "composition_label": mixed.SCENARIOS[scenario_id]["label"],
                    "network_mode": network_mode,
                    "seed_count": args.seed_count,
                    "model_paths": {key: str(value) for key, value in model_paths[network_mode].items()},
                    "model_sha256": {key: _sha256(value) for key, value in model_paths[network_mode].items()},
                    "raw_path": str(args.output / "raw/scenarios" / network_mode / f"{scenario_id}.json"),
                    "force": args.force,
                })
    if args.kind in {"all", "pairwise"}:
        pairs = pairwise._pair_specs()[: args.max_pairs]
        for network_mode in NETWORK_MODES:
            for spec in pairs:
                tasks.append({
                    "composition_kind": "pairwise",
                    "composition_id": spec["pair_id"],
                    "composition_label": spec["pair_label"],
                    "pair_index": spec["pair_index"],
                    "dataset_a": spec["dataset_a"],
                    "dataset_b": spec["dataset_b"],
                    "network_mode": network_mode,
                    "seed_count": args.seed_count,
                    "model_paths": {key: str(value) for key, value in model_paths[network_mode].items()},
                    "model_sha256": {key: _sha256(value) for key, value in model_paths[network_mode].items()},
                    "raw_path": str(args.output / "raw/pairwise" / network_mode / f"{spec['pair_id']}.json"),
                    "force": args.force,
                })
    return tasks


def _acquire_lock(output: Path) -> Any | None:
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / ".run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kind", choices=("all", "scenario", "pairwise"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--max-scenarios", type=int, default=len(mixed.SCENARIOS))
    parser.add_argument("--max-pairs", type=int, default=len(pairwise._pair_specs()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.max_scenarios = max(0, min(args.max_scenarios, len(mixed.SCENARIOS)))
    args.max_pairs = max(0, min(args.max_pairs, len(pairwise._pair_specs())))
    if not 1 <= args.seed_count <= SEED_COUNT:
        raise ValueError("seed-count 必须在1至30之间")
    models = _model_paths(args.results_root)
    missing = [str(path) for network in models.values() for path in network.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少冻结模型：{missing}")
    lock = _acquire_lock(args.output)
    if lock is None:
        print(f"{args.output} 已有补充实验运行，当前实例退出")
        return
    data_dir = args.output / "data"
    figure_dir = args.output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(args, models)
    _write_design(args.output, args, len(tasks))
    failures = []
    payloads = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                payload = future.result()
                payloads.append(payload)
            except Exception as exc:
                failure = {
                    "composition_kind": task["composition_kind"],
                    "composition_id": task["composition_id"],
                    "network_mode": task["network_mode"],
                    "error": repr(exc),
                }
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
            _write_json(args.output / "checkpoint.json", {
                "protocol": PROTOCOL,
                "status": "running",
                "completed_tasks": len(payloads),
                "failed_tasks": len(failures),
                "task_count": len(tasks),
                "updated_at": _utc_now(),
                "failures": failures,
            })
            print(json.dumps({
                "stage": "task_checkpoint",
                "completed_tasks": len(payloads),
                "failed_tasks": len(failures),
                "task_count": len(tasks),
            }, ensure_ascii=False), flush=True)
    if failures:
        _write_json(args.output / "manifest.json", {
            "protocol": PROTOCOL,
            "status": "failed",
            "completed_tasks": len(payloads),
            "failed_tasks": len(failures),
            "failures": failures,
        })
        raise SystemExit(1)
    rows = [row for payload in payloads for row in payload["seed_results"]]
    rows.sort(key=lambda row: (
        row["composition_kind"], row["network_mode"], row["composition_id"],
        int(row["seed_index"]), ALGORITHM_ORDER.index(row["algorithm"]),
    ))
    summaries = _summaries(rows, args.bootstrap_draws)
    comparisons = _paired_comparisons(rows, summaries, args.bootstrap_draws)
    ranks = _rank_rows(summaries)
    _write_rows(data_dir / "baseline_by_seed.csv", rows)
    _write_rows(data_dir / "baseline_summary.csv", summaries)
    _write_rows(data_dir / "eps_vs_baseline.csv", comparisons)
    _write_rows(data_dir / "algorithm_ranks.csv", ranks)
    runtime_rows = [{
        "composition_kind": row["composition_kind"],
        "composition_id": row["composition_id"],
        "network_mode": row["network_mode"],
        "algorithm": row["algorithm"],
        "algorithm_label": row["algorithm_label"],
        "complexity": row["complexity"],
        "runtime_seconds_mean": row["runtime_seconds_mean"],
        "runtime_seconds_p95": row["runtime_seconds_p95"],
    } for row in summaries]
    _write_rows(data_dir / "runtime_complexity.csv", runtime_rows)
    figures = []
    if args.kind in {"all", "scenario"} and args.max_scenarios == len(mixed.SCENARIOS):
        figures.extend(_plot_scenario_atlas(summaries, figure_dir))
        figures.extend(_plot_scenario_profiles(summaries, figure_dir))
    if args.kind in {"all", "pairwise"} and args.max_pairs == len(pairwise._pair_specs()):
        figures.extend(_plot_pair_overview(summaries, figure_dir))
        figures.extend(_plot_pair_matrices(summaries, figure_dir))
        figures.extend(_plot_eps_advantage(comparisons, figure_dir))
        figures.extend(_plot_distributions(summaries, figure_dir))
    if args.kind == "all" and args.max_scenarios == len(mixed.SCENARIOS) and args.max_pairs == len(pairwise._pair_specs()):
        figures.extend(_plot_network_penalty(summaries, figure_dir))
        figures.extend(_plot_overall(summaries, figure_dir))
    _write_documents(args.output, figures, summaries, comparisons)
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "composition_count": len({(row["composition_kind"], row["composition_id"]) for row in rows}),
        "task_count": len(tasks),
        "algorithm_count": len(ALGORITHM_ORDER),
        "seed_count": args.seed_count,
        "by_seed_rows": len(rows),
        "summary_rows": len(summaries),
        "comparison_rows": len(comparisons),
        "figures": figures,
        "completed_at": _utc_now(),
    }
    _write_json(args.output / "manifest.json", manifest)
    _write_json(args.output / "checkpoint.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
