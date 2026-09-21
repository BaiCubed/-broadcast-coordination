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
                f"seed {seed_index}: the centralized greedy upper bound has {len(central_rows)} rows"
            )
        central = central_rows[0]
        baseline_values = [float(row["baseline_curtailment_mwh"]) for row in seed_rows]
        if max(baseline_values) - min(baseline_values) > 1e-8:
            raise RuntimeError(f"seed {seed_index}: the uncontrolled curtailment differs between algorithms")

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
            "the largest realised curtailment absorption over the native centralized greedy schedule and every compared feasible schedule of the same composition, network and seed; "
            "a centralized controller with full state and schedule information can reproduce at least that schedule."
        )

        central_absorption = float(central["accepted_absorption_mwh"])
        best_absorption = max(float(row["accepted_absorption_mwh"]) for row in seed_rows)
        if central_absorption + 1e-9 < best_absorption:
            raise RuntimeError(f"seed {seed_index}: the centralized greedy upper bound does not hold")
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
                ": the uncontrolled curtailment differs between algorithms"
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
            "envelope of the realised curtailment absorption over the native greedy schedule and every compared feasible schedule of the same seed"
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
    title: str, vmin: float = 0.0, vmax: float = 100.0, cmap: str = "YlGnBu",
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
        image = axis.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
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
            last_image = axis.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu")
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
            last_image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdBu")
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
    image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdBu_r")
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
    parser = argparse.ArgumentParser(description="Compare every baseline on the predefined and pairwise mixtures.")
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
        raise ValueError("seed-count must be between 1 and 30")
    models = _model_paths(args.results_root)
    missing = [str(path) for network in models.values() for path in network.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen model: {missing}")
    lock = _acquire_lock(args.output)
    if lock is None:
        print(f"{args.output} already holds a running experiment; this instance exits")
        return
    data_dir = args.output / "data"
    figure_dir = args.output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(args, models)
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
