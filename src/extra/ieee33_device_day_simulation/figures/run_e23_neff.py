from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import run_e22_ieee69_complexity as e22
from . import run_e23_ieee69_relative_boundary as e23

for _candidate in ("Noto Sans CJK SC", "SimSun", "WenQuanYi Zen Hei", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(_candidate, fallback_to_default=False)
    except Exception:
        continue
    matplotlib.rcParams["font.family"] = _candidate
    break
matplotlib.rcParams["axes.unicode_minus"] = False

PROTOCOL = "E23_ieee69_continuous_effective_fleet_v1"
DEFAULT_OUTPUT = Path("results/E23_neff")
KNOTS = (0.25, 0.50, 0.75)


def broadcast_basis(requested_surplus: np.ndarray) -> np.ndarray:
    signal = np.asarray(requested_surplus, dtype=float)
    peak = float(np.max(np.abs(signal)))
    intensity = signal / peak if peak > 1e-12 else np.zeros_like(signal)
    columns = [np.ones_like(intensity), intensity, intensity ** 2, intensity ** 3]
    columns.extend(np.maximum(intensity - knot, 0.0) for knot in KNOTS)
    return np.column_stack(columns)


def mean_pairwise_rho(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    deviation = np.std(values, axis=0)
    values = values[:, deviation > 1e-12]
    device_count = values.shape[1]
    if device_count < 2:
        return float("nan")
    standardized = (values - np.mean(values, axis=0)) / np.std(values, axis=0)
    row_sums = np.sum(standardized, axis=1)
    numerator = float(np.mean(row_sums * row_sums) - device_count)
    return numerator / (device_count * (device_count - 1))


def effective_n(n: float, rho: float) -> float:
    if not np.isfinite(rho) or n < 1:
        return float("nan")
    return float(n / (1.0 + max(n - 1.0, 0.0) * max(float(rho), 0.0)))


def effective_fleet(sink: dict[str, list], fleet_size: int) -> dict[str, Any]:
    response = np.asarray(sink["accepted"], dtype=float)
    design = broadcast_basis(np.asarray(sink["requested_surplus"], dtype=float))
    active = np.std(response, axis=0) > 1e-12
    n_active = int(active.sum())
    if n_active < 2:
        return {
            "n_active": n_active,
            "active_fraction": n_active / max(fleet_size, 1),
            "rho_broadcast_residual": "NA",
            "N_eff": "NA",
            "N_eff_full_fleet": "NA",
            "N_eff_over_N": "NA",
        }
    values = response[:, active]
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    rho = mean_pairwise_rho(values - design @ coefficients)
    n_eff = effective_n(n_active, rho)
    return {
        "n_active": n_active,
        "active_fraction": n_active / max(fleet_size, 1),
        "rho_broadcast_residual": rho,
        "N_eff": n_eff,
        "N_eff_full_fleet": effective_n(fleet_size, rho),
        "N_eff_over_N": n_eff / max(fleet_size, 1),
    }


def _algorithms(with_eps: bool) -> tuple[str, ...]:
    if with_eps:
        return e23.ALGORITHMS
    return tuple(a for a in e23.ALGORITHMS if not a.startswith("eps_"))


def run_seed(
    dataset: str, seed_index: int, output: Path, with_eps: bool
) -> dict[str, Any]:
    e23._base_stress_settings()
    cases = e22._network_cases()
    algorithms = _algorithms(with_eps)
    optimizer = None
    if with_eps:
        _, optimizer, _ = e23.direct.training.load_frozen_eps_controller(
            e23._model_path(dataset)
        )
    path = output / "raw" / dataset / f"seed_{seed_index:02d}.csv"
    expected = sum(len(v["values"]) for v in e23.AXES.values()) * len(algorithms)
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if len(existing) == expected and all(r.get("protocol") == PROTOCOL for r in existing):
            return {"dataset": dataset, "seed_index": seed_index, "status": "cached"}
    started = time.time()
    base_seed = e22._base_seed(dataset, seed_index)
    rows: list[dict[str, Any]] = []
    for axis, settings in e23.AXES.items():
        for pressure_index, pressure_value in enumerate(settings["values"]):
            records, config, availability, network, scenario = e23._context(
                dataset, seed_index, axis, pressure_value, cases
            )
            for algorithm_index, algorithm in enumerate(e23.ALGORITHMS):
                if algorithm not in algorithms:
                    continue
                sink = {"accepted": [], "requested_surplus": []}
                result, _ = e22._run_seed(
                    algorithm,
                    records,
                    e23._algorithm_config(config, algorithm),
                    scenario,
                    availability,
                    network,
                    base_seed + 1_800_000 + algorithm_index * 10_000,
                    base_seed + 1_910_000,
                    optimizer if algorithm.startswith("eps_") else None,
                    False,
                    probe_sink=sink,
                )
                rows.append(
                    {
                        "protocol": PROTOCOL,
                        "dataset": dataset,
                        "dataset_label": e22.DATASET_LABELS[dataset],
                        "seed_index": seed_index,
                        "seed": base_seed,
                        "axis": axis,
                        "pressure_index": pressure_index,
                        "pressure_value": pressure_value,
                        "topology": e23.TOPOLOGY,
                        "fleet_size": len(records),
                        "algorithm": algorithm,
                        "algorithm_label": e22.mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                        "complexity": e22.mixed.ALGORITHM_DEFINITIONS[algorithm]["complexity"],
                        "mean_reduction_pct": result["mean_reduction_pct"],
                        "network_acceptance_ratio": result["network_acceptance_ratio"],
                        **effective_fleet(sink, len(records)),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return {
        "dataset": dataset, "seed_index": seed_index, "status": "completed",
        "rows": len(rows), "seconds": round(time.time() - started, 1),
    }


def _bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(output: Path, draws: int) -> list[dict[str, Any]]:
    rows: list[dict[str, str]] = []
    for path in sorted((output / "raw").glob("*/seed_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no data under {output}/raw")
    algorithms = sorted({row["algorithm"] for row in rows}, key=e23.ALGORITHMS.index)
    summary: list[dict[str, Any]] = []
    for axis in e23.AXES:
        for pressure_value in e23.AXES[axis]["values"]:
            for algorithm in algorithms:
                selected = [
                    row for row in rows
                    if row["axis"] == axis
                    and abs(float(row["pressure_value"]) - pressure_value) < 1e-9
                    and row["algorithm"] == algorithm
                ]
                if not selected:
                    continue
                entry: dict[str, Any] = {
                    "axis": axis,
                    "pressure_value": pressure_value,
                    "algorithm": algorithm,
                    "algorithm_label": selected[0]["algorithm_label"],
                    "complexity": selected[0]["complexity"],
                    "dataset_seed_count": len(selected),
                    "na_count": sum(row["N_eff"] == "NA" for row in selected),
                }
                for field in ("N_eff", "N_eff_over_N", "rho_broadcast_residual",
                              "n_active", "mean_reduction_pct"):
                    values = np.asarray([
                        float(row[field]) for row in selected if row[field] != "NA"
                    ])
                    mean, low, high = _bootstrap(
                        values, e23._stable_seed(f"{field}|{axis}|{pressure_value}|{algorithm}"), draws
                    )
                    entry[f"{field}_mean"] = mean
                    entry[f"{field}_ci_low"] = low
                    entry[f"{field}_ci_high"] = high
                summary.append(entry)
    path = output / "e23_neff_summary.csv"
    fields = sorted({key for row in summary for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def _panel(axis_object, summary, axis_name, field, algorithms):
    for algorithm in algorithms:
        rows = sorted(
            [r for r in summary if r["axis"] == axis_name and r["algorithm"] == algorithm],
            key=lambda r: float(r["pressure_value"]),
        )
        rows = [r for r in rows if np.isfinite(r[f"{field}_mean"])]
        if not rows:
            continue
        x = [float(r["pressure_value"]) for r in rows]
        y = [r[f"{field}_mean"] for r in rows]
        axis_object.plot(x, y, marker="o", markersize=3.5, linewidth=1.7,
                         color=e23.COLORS[algorithm],
                         label=f"{rows[0]['algorithm_label']} ({rows[0]['complexity']})")
        axis_object.fill_between(x, [r[f"{field}_ci_low"] for r in rows],
                                 [r[f"{field}_ci_high"] for r in rows],
                                 color=e23.COLORS[algorithm], alpha=0.12, linewidth=0)


def plot(summary: list[dict[str, Any]], output: Path, fleet_size: int) -> None:
    algorithms = sorted({r["algorithm"] for r in summary}, key=e23.ALGORITHMS.index)
    specs = (
        ("N_eff", "Effective fleet size $N_{eff}$ (devices)", True,
         "Effective fleet size $N_{eff}$ under continuous IEEE-69 stress", "e23_neff_curves"),
        ("N_eff_over_N", "$N_{eff}/N$ (N=%d)" % fleet_size, False,
         "Effective fraction $N_{eff}/N$ under continuous IEEE-69 stress", "e23_neff_ratio_curves"),
        ("rho_broadcast_residual", r"Mean pairwise correlation of the broadcast residual $\rho$", False,
         r"Device co-movement $\rho$ under continuous IEEE-69 stress", "e23_rho_curves"),
    )
    for field, ylabel, log_scale, title, name in specs:
        figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.6), sharey=True)
        for axis_object, axis_name in zip(axes, e23.AXES):
            _panel(axis_object, summary, axis_name, field, algorithms)
            axis_object.set_title(e23.AXES[axis_name]["label"])
            axis_object.set_xlabel(e23.AXES[axis_name]["unit"])
            axis_object.grid(axis="y", alpha=0.22)
            if log_scale:
                axis_object.set_yscale("log")
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        figure.suptitle(title, y=1.02, fontsize=16)
        figure.tight_layout()
        target = output / "figures" / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=220, bbox_inches="tight")
        figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def run_chunk(
    dataset: str, seed_indices: list[int], output: Path, with_eps: bool
) -> list[dict[str, Any]]:
    return [run_seed(dataset, index, output, with_eps) for index in seed_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for run e23 neff.")
    parser.add_argument("--datasets", nargs="*", default=list(e22.E22_DATASETS))
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=min(64, (os.cpu_count() or 8)))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds-per-job", type=int, default=5)
    parser.add_argument("--with-eps", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        chunk = max(1, args.seeds_per_job)
        jobs = [
            (d, list(range(start, min(start + chunk, args.seed_count))))
            for d in args.datasets
            for start in range(0, args.seed_count, chunk)
        ]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_chunk, d, seeds, args.output, args.with_eps)
                for d, seeds in jobs
            ]
            for future in as_completed(futures):
                for result in future.result():
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                done += 1
                print(json.dumps({"chunks_done": f"{done}/{len(jobs)}"}), flush=True)
    summary = aggregate(args.output, args.bootstrap_draws)
    plot(summary, args.output, e22.FLEET_SIZE)
    print(json.dumps({
        "status": "completed", "protocol": PROTOCOL,
        "datasets": len(args.datasets), "seed_count": args.seed_count,
        "with_eps": bool(args.with_eps), "summary_rows": len(summary),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
