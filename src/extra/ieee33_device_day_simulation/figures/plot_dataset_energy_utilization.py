from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DATASET_LABELS = {
    "bdg1_building_data_genome": "BDG1",
    "bdg2_building_data_genome": "BDG2",
    "low_carbon_london": "LCL",
    "danish_smart_heat_meters": "Danish",
    "smart_grid_smart_city": "SGSC",
    "heapo_heat_pumps": "HEAPO",
    "goiener_smart_meters": "GoiEner",
    "european_lv_urban_8087": "EU LV 8087",
    "european_lv_rural_2731": "EU Rural 2731",
    "european_lv_urban_35297": "EU Urban 35297",
    "norway_ami_energy_distribution": "Norway AMI",
    "camsl_japan_smart_meters": "CAMSL",
    "irish_domestic_smart_meters": "Irish",
    "opsd_household_data": "OPSD",
    "complete_energy_community": "CEC",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_root(results_root: Path, dataset: str) -> Path | None:
    candidates = [
        results_root / f"{dataset}_ieee33_real_load" / "network_stress" / "data" / "complete_results.json",
        results_root / f"{dataset}_ieee33_real_load" / "coverage_fix" / "network_stress" / "data" / "complete_results.json",
        results_root / f"{dataset}_ieee33_real_load" / "coverage_fix" / "network_constrained_new" / "data" / "complete_results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((results_root / f"{dataset}_ieee33_real_load").glob("**/complete_results.json"))
    return matches[0] if matches else None


def _scenario_rate(trace: dict[str, Any]) -> tuple[float, float, float]:
    energy_input = np.asarray(trace.get("energy_input_kw", []), dtype=float)
    baseline = np.asarray(trace.get("self_consumption_baseline_kwh", []), dtype=float)
    eps = np.asarray(trace.get("self_consumption_eps_kwh", []), dtype=float)
    if not energy_input.size or energy_input.size != baseline.size or baseline.size != eps.size:
        return float("nan"), float("nan"), float("nan")
    hours_per_step = 24.0 / float(energy_input.size)
    available_input_kwh = float(np.sum(energy_input) * hours_per_step)
    if available_input_kwh <= 1e-12:
        return 0.0, 0.0, 0.0
    baseline_rate = float(np.sum(baseline) / available_input_kwh)
    eps_rate = float(np.sum(eps) / available_input_kwh)
    return baseline_rate, eps_rate, available_input_kwh


def _plot(results_root: Path, output_path: Path) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    dataset_order = [key for key in DATASET_LABELS if _dataset_root(results_root, key)]
    if not dataset_order:
        raise FileNotFoundError(f"no dataset results found under {results_root}")

    rows: list[dict[str, Any]] = []
    for dataset in dataset_order:
        path = _dataset_root(results_root, dataset)
        if path is None:
            continue
        content = _load_json(path)
        scenarios = content.get("scenarios", {})
        baseline_rates: list[float] = []
        eps_rates: list[float] = []
        available_inputs: list[float] = []
        for scenario_name, trace in scenarios.items():
            baseline_rate, eps_rate, available_input_kwh = _scenario_rate(trace)
            if np.isnan(baseline_rate) or np.isnan(eps_rate):
                continue
            baseline_rates.append(baseline_rate)
            eps_rates.append(eps_rate)
            available_inputs.append(available_input_kwh)
        if not baseline_rates:
            continue
        rows.append(
            {
                "dataset": dataset,
                "label": DATASET_LABELS[dataset],
                "scenario_count": len(baseline_rates),
                "baseline_rate_pct": 100.0 * float(np.mean(baseline_rates)),
                "eps_rate_pct": 100.0 * float(np.mean(eps_rates)),
                "delta_pp": 100.0 * (float(np.mean(eps_rates)) - float(np.mean(baseline_rates))),
                "available_input_kwh_mean": float(np.mean(available_inputs)),
                "zero_input": float(np.mean(available_inputs)) <= 1e-12,
                "source_file": str(path.relative_to(results_root)),
            }
        )

    if not rows:
        raise RuntimeError("no usable utilization rows were found")

    x = np.arange(len(rows), dtype=float)
    width = 0.38
    baseline_values = np.asarray([row["baseline_rate_pct"] for row in rows], dtype=float)
    eps_values = np.asarray([row["eps_rate_pct"] for row in rows], dtype=float)
    max_value = float(max(np.max(baseline_values), np.max(eps_values), 1.0))

    fig, ax = plt.subplots(figsize=(16, 6.5))
    baseline_bars = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        color="#9a9a9a",
        edgecolor="#666666",
        linewidth=0.8,
        label="Before EPS",
    )
    eps_bars = ax.bar(
        x + width / 2,
        eps_values,
        width,
        color="#069c8f",
        edgecolor="#04746b",
        linewidth=0.8,
        label="After EPS",
    )

    for row_index, row in enumerate(rows):
        pair_top = max(baseline_values[row_index], eps_values[row_index])
        if row["zero_input"]:
            ax.text(
                x[row_index],
                1.2,
                "no input",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#a33a3a",
            )
        else:
            ax.text(
                x[row_index] + width / 2,
                pair_top + max_value * 0.018,
                f"{row['delta_pp']:+.1f} pp",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
        if row["zero_input"]:
            for bar in (baseline_bars[row_index], eps_bars[row_index]):
                bar.set_hatch("//")
                bar.set_facecolor("#d9d9d9" if bar is baseline_bars[row_index] else "#bfeae5")

    ax.set_ylim(0, max_value * 1.18)
    ax.set_ylabel("Energy utilization rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['label']}*" if row["zero_input"] else row["label"] for row in rows],
        rotation=30,
        ha="right",
    )
    ax.set_title("Dataset-level energy utilization before and after EPS")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.text(
        0.99,
        0.015,
        "Metric = self-consumption / available external input. *No external input trace; plotted as 0.",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dataset-level utilization comparison")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/overall_energy_utilization_comparison.png"),
    )
    args = parser.parse_args()
    path = _plot(args.results_root, args.output)
    print(path)


if __name__ == "__main__":
    main()
