from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from src.extra.nc_excel_experiments.coupling import (
    availability_at_steps,
    diurnal_residual,
    rank_availability,
)
from src.extra.nc_excel_experiments.run import _profile_and_schedule, _run_snapshots

from . import run_e21_mixed_gamma_boundary as mixed_gamma
from . import run_e21_mixed_scenarios as mixed
from . import run_e21_pairwise_curtailment as pairwise


PROTOCOL = "E21_pairwise_equal_mix_r2_v1"
DEFAULT_OUTPUT = Path("results/E21/pairwise_r2")
BASE_N_VALUES = (2, 10, 30, 50, 80, 100, 150, 250)
EXTENSION_N_VALUES = (400, 650, 1000, 1600, 2500, 5000)
R2_THRESHOLD = 0.95
EPS = 1e-12


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_seed(base_seed: int, pair_index: int, n: int) -> int:
    return int(base_seed + pair_index * 1_000_000 + n * 1009)


def _run_cell(task: dict[str, Any]) -> dict[str, Any]:
    pair = task["pair"]
    n = int(task["N"])
    seed = _cell_seed(int(task["seed"]), int(pair["pair_index"]), n)
    weights = {pair["dataset_a"]: 0.5, pair["dataset_b"]: 0.5}
    records, config, audit = mixed._sample_mixed_fleet(
        "S4-A",
        "aggregate",
        "test",
        seed,
        weights,
        fleet_size=n,
    )
    config["control"] = dict(config["control"])
    config["control"]["network_feedback"] = False
    profile_steps, schedule = _profile_and_schedule(
        records,
        config,
        int(task["conditions"]),
        int(task["seed"]) + int(pair["pair_index"]) * 10_000,
    )
    steps_per_day = int(config["simulation"]["steps_per_day"])
    availability = availability_at_steps(
        rank_availability(diurnal_residual(records, steps_per_day)),
        profile_steps,
    )
    train_replications = int(task["train_replications"])
    test_replications = int(task["test_replications"])
    aggregate, _, _, _ = _run_snapshots(
        records,
        config,
        profile_steps,
        schedule,
        replications=train_replications + test_replications,
        seed=seed,
        availability_factory=lambda _replication, table=availability: table,
    )
    normalized = aggregate / float(n)
    train = normalized[:train_replications]
    test = normalized[train_replications:]
    prediction = np.mean(train, axis=0)
    r2, lower, upper = mixed_gamma._bootstrap_r2(
        test,
        prediction,
        int(task["bootstrap_draws"]),
        seed + 700_000,
    )
    response_path = Path(task["response_path"])
    response_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        response_path,
        aggregate_response_kw=aggregate,
        frozen_prediction_kw=prediction * float(n),
        profile_steps=np.asarray(profile_steps, dtype=int),
    )
    dataset_audit_a = audit["datasets"][pair["dataset_a"]]
    dataset_audit_b = audit["datasets"][pair["dataset_b"]]
    return {
        "protocol": PROTOCOL,
        "pair_id": pair["pair_id"],
        "pair_index": int(pair["pair_index"]),
        "pair_label": pair["pair_label"],
        "dataset_a": pair["dataset_a"],
        "dataset_a_label": pairwise.SHORT_NAMES[pair["dataset_a"]],
        "dataset_b": pair["dataset_b"],
        "dataset_b_label": pairwise.SHORT_NAMES[pair["dataset_b"]],
        "weight_a": 0.5,
        "weight_b": 0.5,
        "N": n,
        "devices_a": int(audit["counts"][pair["dataset_a"]]),
        "devices_b": int(audit["counts"][pair["dataset_b"]]),
        "R2": r2,
        "R2_ci_lower": lower,
        "R2_ci_upper": upper,
        "mean_response_kw_per_device": float(np.mean(test)),
        "std_response_kw_per_device": float(np.std(test)),
        "train_replications": train_replications,
        "test_replications": test_replications,
        "conditions": int(task["conditions"]),
        "bootstrap_draws": int(task["bootstrap_draws"]),
        "seed": seed,
        "network_mode": "aggregate",
        "network_feedback": False,
        "availability_mode": "data_coupled",
        "profile_bootstrap_a": bool(dataset_audit_a["profile_bootstrap"]),
        "profile_bootstrap_b": bool(dataset_audit_b["profile_bootstrap"]),
        "mean_profile_reuse_a": float(dataset_audit_a["mean_profile_reuse"]),
        "mean_profile_reuse_b": float(dataset_audit_b["mean_profile_reuse"]),
        "response_archive": str(response_path),
    }


def _log_crossing(rows: list[dict[str, Any]]) -> float | None:
    ordered = sorted(rows, key=lambda row: int(row["N"]))
    crossing = next(
        (
            index
            for index, row in enumerate(ordered)
            if float(row["R2"]) >= R2_THRESHOLD
        ),
        None,
    )
    if crossing is None:
        return None
    if crossing == 0:
        return float(ordered[0]["N"])
    left, right = ordered[crossing - 1], ordered[crossing]
    return mixed_gamma._log_interpolate(
        float(left["N"]),
        float(left["R2"]),
        float(right["N"]),
        float(right["R2"]),
        R2_THRESHOLD,
    )


def _first_sampled_crossing(rows: list[dict[str, Any]]) -> int | None:
    ordered = sorted(rows, key=lambda row: int(row["N"]))
    crossing = next(
        (int(row["N"]) for row in ordered if float(row["R2"]) >= R2_THRESHOLD),
        None,
    )
    return crossing


def _pair_rows(rows: list[dict[str, Any]], pair_id: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["pair_id"] == pair_id],
        key=lambda row: int(row["N"]),
    )


def _run_batch(
    tasks: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    checkpoint_path: Path,
    checkpoint_settings: dict[str, Any],
    workers: int,
    total_possible: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return completed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_cell, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                _write_json(
                    checkpoint_path.parent.parent / "failure.json",
                    {
                        "pair_id": task["pair"]["pair_id"],
                        "N": task["N"],
                        "error": repr(exc),
                    },
                )
                raise
            completed.append(row)
            completed.sort(key=lambda item: (int(item["pair_index"]), int(item["N"])))
            _write_json(
                checkpoint_path,
                {
                    "protocol": PROTOCOL,
                    "settings": checkpoint_settings,
                    "cells": completed,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "pairwise_r2_cell",
                        "pair_id": row["pair_id"],
                        "N": row["N"],
                        "R2": row["R2"],
                        "completed": len(completed),
                        "maximum_possible": total_possible,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return completed


def _summaries(
    pairs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for pair in pairs:
        block = _pair_rows(rows, pair["pair_id"])
        n95 = _log_crossing(block)
        sampled = _first_sampled_crossing(block)
        maximum = max(block, key=lambda row: int(row["N"]))
        summaries.append(
            {
                "pair_id": pair["pair_id"],
                "pair_index": pair["pair_index"],
                "pair_label": pair["pair_label"],
                "dataset_a": pair["dataset_a"],
                "dataset_a_label": pairwise.SHORT_NAMES[pair["dataset_a"]],
                "dataset_b": pair["dataset_b"],
                "dataset_b_label": pairwise.SHORT_NAMES[pair["dataset_b"]],
                "N95_interpolated": n95,
                "N95_first_sampled": sampled,
                "boundary_status": "reached" if n95 is not None else "right_censored",
                "max_tested_N": int(maximum["N"]),
                "R2_at_max_tested_N": float(maximum["R2"]),
                "tested_N_values": ";".join(str(row["N"]) for row in block),
            }
        )
    return summaries


def _plot_curve(
    axis: Any,
    rows: list[dict[str, Any]],
    n95: float | None,
    *,
    compact: bool,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["N"]))
    n_values = np.asarray([int(row["N"]) for row in ordered], dtype=float)
    r2 = np.asarray([float(row["R2"]) for row in ordered])
    lower = np.asarray([float(row["R2_ci_lower"]) for row in ordered])
    upper = np.asarray([float(row["R2_ci_upper"]) for row in ordered])
    axis.fill_between(n_values, lower, upper, color="#2166ac", alpha=0.14, linewidth=0)
    axis.plot(
        n_values,
        r2,
        color="#2166ac",
        marker="o",
        linewidth=1.5 if compact else 2.1,
        markersize=3.0 if compact else 5.0,
    )
    axis.axhline(R2_THRESHOLD, color="#8e1b1b", linestyle=":", linewidth=1.2)
    if n95 is not None:
        axis.axvline(n95, color="#1b7837", linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlim(1.7, max(300.0, float(np.max(n_values)) * 1.25))
    lower_limit = max(-1.0, min(-0.05, float(np.min(lower)) - 0.04))
    axis.set_ylim(lower_limit, 1.01)
    axis.grid(alpha=0.22)


def _save_curves(
    pairs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    figure_dir: Path,
) -> list[str]:
    summary_by_pair = {row["pair_id"]: row for row in summaries}
    individual_dir = figure_dir / "r2_curves"
    page_dir = figure_dir / "r2_curve_pages"
    individual_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        block = _pair_rows(rows, pair_id)
        summary = summary_by_pair[pair_id]
        fig, axis = plt.subplots(figsize=(8.3, 6.0), constrained_layout=True)
        _plot_curve(axis, block, summary["N95_interpolated"], compact=False)
        axis.set_xlabel(r"Physical fleet size $N$")
        axis.set_ylabel(r"Independent test $R^2$")
        boundary = (
            rf"interpolated $N_{{95}}={summary['N95_interpolated']:.1f}$"
            if summary["N95_interpolated"] is not None
            else rf"$R^2<0.95$ through $N={summary['max_tested_N']}$"
        )
        axis.set_title(
            f"{pair_id}: {pair['pair_label']}\n50/50 mixed fleet; {boundary}"
        )
        path = individual_dir / f"{pair_id}_r2_curve.png"
        fig.savefig(path, dpi=240)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        generated.extend([str(path), str(path.with_suffix(".pdf"))])

    page_size = 12
    for page_index, start in enumerate(range(0, len(pairs), page_size), start=1):
        page_pairs = pairs[start : start + page_size]
        fig, axes = plt.subplots(
            3,
            4,
            figsize=(16.5, 11.5),
            constrained_layout=True,
        )
        for axis, pair in zip(axes.flat, page_pairs):
            summary = summary_by_pair[pair["pair_id"]]
            _plot_curve(
                axis,
                _pair_rows(rows, pair["pair_id"]),
                summary["N95_interpolated"],
                compact=True,
            )
            n95_text = (
                f"N95={summary['N95_interpolated']:.1f}"
                if summary["N95_interpolated"] is not None
                else f">{summary['max_tested_N']}"
            )
            axis.set_title(
                f"{pair['pair_id']}  {pair['pair_label']}\n{n95_text}", fontsize=8.5
            )
        for axis in axes.flat[len(page_pairs) :]:
            axis.axis("off")
        fig.supxlabel(r"Physical fleet size $N$")
        fig.supylabel(r"Independent test $R^2$")
        fig.suptitle(
            f"E21 pairwise 50/50 R² curves, page {page_index}",
            fontsize=14,
        )
        path = page_dir / f"pairwise_r2_curves_page_{page_index:02d}.png"
        fig.savefig(path, dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        generated.extend([str(path), str(path.with_suffix(".pdf"))])
    return generated


def _annotated_matrix(
    axis: Any,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    maximum_n: int,
) -> Any:
    finite = matrix[np.isfinite(matrix)]
    image = axis.imshow(
        matrix,
        cmap="viridis",
        norm=LogNorm(vmin=max(1.0, float(np.min(finite))), vmax=float(np.max(finite))),
    )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    axis.set_title(title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            if row == column:
                axis.text(
                    column,
                    row,
                    "--",
                    ha="center",
                    va="center",
                    color="#555",
                    fontsize=7,
                )
                continue
            value = matrix[row, column]
            text = f"{value:.0f}" if np.isfinite(value) else f">{maximum_n}"
            color = (
                "white"
                if np.isfinite(value) and value >= np.median(finite)
                else "black"
            )
            if not np.isfinite(value):
                color = "#8e1b1b"
            axis.text(
                column, row, text, ha="center", va="center", color=color, fontsize=5.8
            )
    return image


def _save_n95_overall(
    summaries: list[dict[str, Any]],
    figure_dir: Path,
) -> list[str]:
    datasets = list(pairwise.PLOT_ORDER)
    labels = [pairwise.SHORT_NAMES[dataset] for dataset in datasets]
    index = {dataset: position for position, dataset in enumerate(datasets)}
    interpolated = np.full((len(datasets), len(datasets)), np.nan, dtype=float)
    sampled = np.full_like(interpolated, np.nan)
    maximum_n = max(int(row["max_tested_N"]) for row in summaries)
    for row in summaries:
        first = index[row["dataset_a"]]
        second = index[row["dataset_b"]]
        if row["N95_interpolated"] is not None:
            interpolated[first, second] = interpolated[second, first] = float(
                row["N95_interpolated"]
            )
        if row["N95_first_sampled"] is not None:
            sampled[first, second] = sampled[second, first] = float(
                row["N95_first_sampled"]
            )
    fig, axes = plt.subplots(1, 2, figsize=(22, 10.5), constrained_layout=True)
    image_a = _annotated_matrix(
        axes[0],
        interpolated,
        labels,
        r"a  Log-interpolated physical $N_{95}$",
        maximum_n,
    )
    image_b = _annotated_matrix(
        axes[1],
        sampled,
        labels,
        r"b  First sampled $N$ with $R^2\geq0.95$",
        maximum_n,
    )
    fig.colorbar(image_a, ax=axes[0], fraction=0.046, pad=0.04, label=r"$N_{95}$")
    fig.colorbar(image_b, ax=axes[1], fraction=0.046, pad=0.04, label=r"Sampled $N$")
    fig.suptitle(
        "E21 pairwise 50/50 mixed-fleet predictability thresholds", fontsize=16
    )
    path = figure_dir / "pairwise_r2_n95_overall.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [str(path), str(path.with_suffix(".pdf"))]


def _tasks_for_values(
    pairs: list[dict[str, Any]],
    n_values: list[int],
    completed_keys: set[tuple[str, int]],
    args: argparse.Namespace,
    raw_dir: Path,
    pair_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for pair in pairs:
        if pair_filter is not None and pair["pair_id"] not in pair_filter:
            continue
        for n in n_values:
            if (pair["pair_id"], n) in completed_keys:
                continue
            tasks.append(
                {
                    "pair": pair,
                    "N": n,
                    "train_replications": args.train_replications,
                    "test_replications": args.test_replications,
                    "conditions": args.conditions,
                    "bootstrap_draws": args.bootstrap_draws,
                    "seed": args.seed,
                    "response_path": str(
                        raw_dir / pair["pair_id"] / f"responses_N{n}.npz"
                    ),
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the independent test R^2 curve of the 105 pairwise dataset mixtures.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-n-values", nargs="+", type=int, default=list(BASE_N_VALUES)
    )
    parser.add_argument(
        "--extension-n-values",
        nargs="+",
        type=int,
        default=list(EXTENSION_N_VALUES),
    )
    parser.add_argument("--train-replications", type=int, default=10)
    parser.add_argument("--test-replications", type=int, default=30)
    parser.add_argument("--conditions", type=int, default=48)
    parser.add_argument("--bootstrap-draws", type=int, default=400)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(2, (os.cpu_count() or 2) // 8))
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.base_n_values = sorted(
        {int(value) for value in args.base_n_values if int(value) >= 2}
    )
    args.extension_n_values = sorted(
        {
            int(value)
            for value in args.extension_n_values
            if int(value) > max(args.base_n_values)
        }
    )
    if any(value % 2 for value in (*args.base_n_values, *args.extension_n_values)):
        raise ValueError("a 50/50 combination accepts only an even N, so that the integer device split stays equal")

    output = args.output
    raw_dir = output / "raw" / "responses"
    data_dir = output / "data"
    figure_dir = output / "figures"
    pair_data_dir = data_dir / "pairs"
    checkpoint_path = data_dir / "checkpoint.json"
    for directory in (raw_dir, data_dir, figure_dir, pair_data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pairs = pairwise._pair_specs()
    checkpoint_settings = {
        "base_n_values": args.base_n_values,
        "extension_n_values": args.extension_n_values,
        "train_replications": args.train_replications,
        "test_replications": args.test_replications,
        "conditions": args.conditions,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }
    completed: list[dict[str, Any]] = []
    if checkpoint_path.is_file() and not args.force:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("protocol") != PROTOCOL
            or checkpoint.get("settings") != checkpoint_settings
        ):
            raise ValueError(
                "the existing checkpoint does not match the current settings; re-run with --force"
            )
        completed = checkpoint["cells"]
    if args.plot_only and not completed:
        raise FileNotFoundError("plot-only needs an existing checkpoint.json")

    maximum_possible = len(pairs) * (
        len(args.base_n_values) + len(args.extension_n_values)
    )
    if not args.plot_only:
        completed_keys = {(row["pair_id"], int(row["N"])) for row in completed}
        base_tasks = _tasks_for_values(
            pairs,
            args.base_n_values,
            completed_keys,
            args,
            raw_dir,
        )
        completed = _run_batch(
            base_tasks,
            completed,
            checkpoint_path,
            checkpoint_settings,
            args.workers,
            maximum_possible,
        )
        for n in args.extension_n_values:
            unreached = {
                pair["pair_id"]
                for pair in pairs
                if _log_crossing(_pair_rows(completed, pair["pair_id"])) is None
            }
            if not unreached:
                break
            completed_keys = {(row["pair_id"], int(row["N"])) for row in completed}
            extension_tasks = _tasks_for_values(
                pairs,
                [n],
                completed_keys,
                args,
                raw_dir,
                pair_filter=unreached,
            )
            completed = _run_batch(
                extension_tasks,
                completed,
                checkpoint_path,
                checkpoint_settings,
                args.workers,
                maximum_possible,
            )

    summaries = _summaries(pairs, completed)
    _write_rows(data_dir / "pairwise_r2_points.csv", completed)
    _write_rows(data_dir / "pairwise_r2_n95.csv", summaries)
    for pair in pairs:
        _write_rows(
            pair_data_dir / f"{pair['pair_id']}.csv",
            _pair_rows(completed, pair["pair_id"]),
        )
    figures = _save_curves(pairs, completed, summaries, figure_dir)
    figures.extend(_save_n95_overall(summaries, figure_dir))

    reached = [row for row in summaries if row["N95_interpolated"] is not None]
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "pair_count": len(pairs),
        "reached_count": len(reached),
        "right_censored_count": len(pairs) - len(reached),
        "cell_count": len(completed),
        "base_n_values": args.base_n_values,
        "extension_n_values": args.extension_n_values,
        "train_replications": args.train_replications,
        "test_replications": args.test_replications,
        "conditions": args.conditions,
        "bootstrap_draws": args.bootstrap_draws,
        "r2_threshold": R2_THRESHOLD,
        "figures": figures,
        "generated_files": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["generated_files"].append(
                {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
            )
    _write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "pairs": len(pairs),
                "reached": len(reached),
                "cells": len(completed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
