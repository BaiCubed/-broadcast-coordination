from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import reference_controllers as controllers
from . import network_dispatch_protocol as training
from . import ieee69_network_implementation as implementation
from . import ieee69_trained_vs_transferred as direct


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/ieee123_safety_audit"
NETWORK_FILE = ROOT / "src/extra/ieee33_device_day_simulation/configs/network_ieee123.yaml"
MODEL_SOURCE = ROOT / "results/ieee69_network_implementation/trained_eps_ieee69_direct/models"
PROTOCOL = "ieee123_safety_audit_v2"
MODEL_REGIME = "IEEE-69 frozen model (supplementary experiment)"
EPS_LABEL = "EPS IEEE-69 fused"
TOPOLOGY = "ieee123"
DATASETS = tuple(implementation.NETWORK_DATASETS)
SCENARIOS = {
    "S0_uniform": {"label": "uniform deployment", "placement": "uniform", "capacity": 1.0},
    "S1_feeder_50": {"label": "50% on a distal feeder", "placement": "feeder_50", "capacity": 1.0},
    "S2_node_80": {"label": "80% on a distal bus", "placement": "node_80", "capacity": 1.0},
    "S3_feeder_50_derated": {"label": "50% concentration with capacity at 70%", "placement": "feeder_50", "capacity": 0.7},
}
ALGORITHMS = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_ieee69_fused",
    "centralized_optimal",
)
BASELINES = ALGORITHMS[1:7]
FLEET_SIZE = 5000
SEED_COUNT = 30
STEPS = implementation.STEPS
BOOTSTRAP_DRAWS = 1000
COLORS = {
    "no_coordination": "#9c9c9c",
    "local_rules": "#ed7d31",
    "mpc_optimal": "#7057ff",
    "mean_field_control": "#4e79a7",
    "virtual_battery": "#59a14f",
    "packetized_energy_management": "#af7aa1",
    "transactive_control": "#d55e00",
    "eps_ieee69_fused": "#1f5aa6",
    "centralized_optimal": "#3a9d66",
}
DATASET_LABELS = {dataset: implementation.DATASET_LABELS[dataset] for dataset in DATASETS}
ALGORITHM_COMPLEXITY = {
    "no_coordination": "O(0)",
    "local_rules": "O(1)/device",
    "mpc_optimal": "O(HN)",
    "mean_field_control": "O(N)",
    "virtual_battery": "O(N)",
    "packetized_energy_management": "O(N log N)",
    "transactive_control": "O(N log N)",
    "eps_ieee69_fused": "O(1)",
    "centralized_optimal": "O(N)",
}

matplotlib.rcParams["font.family"] = "SimSun"
matplotlib.rcParams["axes.unicode_minus"] = False


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _configure_network() -> dict[str, Any]:
    case = implementation.load_network_case(NETWORK_FILE)
    layout = implementation._radial_layout(case)
    buses = sorted({int(row[1]) for row in case["branches"]})
    ordered = sorted(buses, key=lambda bus: (layout["distance"].get(bus, 0.0), bus))
    zone_count = 6
    zones = tuple(tuple(chunk) for chunk in np.array_split(ordered, zone_count))
    implementation.ABSORPTION_ZONES[TOPOLOGY] = zones
    middle = max(1, len(ordered) // 2)
    implementation.DISTAL_GROUPS[TOPOLOGY] = (tuple(ordered[:middle]), tuple(ordered[middle:]))
    return case


def _base_settings() -> None:
    implementation.STRESS_MODES["ieee123_audit_base"] = {
        "label": "IEEE-123 audit base",
        "plain_label": "audit baseline",
        "load_scale": 0.45,
        "solar_ratio": 0.75,
        "wind_ratio": 0.30,
        "input_mode": "proportional",
        "capacity_multiplier": 1.0,
        "forecast_error": 0.0,
        "placement_mode": "load_weighted",
    }


def _model_path(dataset: str) -> Path:
    path = MODEL_SOURCE / f"{dataset}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing the {MODEL_REGIME}: {path}")
    return path


def _context(
    dataset: str,
    seed_index: int,
    scenario_name: str,
    case: dict[str, Any],
) -> tuple[list[Any], dict[str, Any], np.ndarray, Any, dict[str, Any]]:
    seed = implementation._base_seed(dataset, seed_index)
    records, config, _ = implementation._sample_dataset(dataset, seed)
    placement_config = SCENARIOS[scenario_name]
    placement_seed = seed + 24_000 + _stable_seed(scenario_name) % 10_000
    reference_records, _ = implementation._remap_records(records, case, placement_seed, "load_weighted")
    placed_records, placement = implementation._remap_records(
        records, case, placement_seed, placement_config["placement"]
    )
    scenario = implementation._build_scenario(
        placed_records,
        reference_records,
        case,
        TOPOLOGY,
        "ieee123_audit_base",
        seed + _stable_seed(scenario_name),
        placement,
    )
    network = implementation.LocalRadialDistFlow(
        case,
        capacity_multiplier=float(placement_config["capacity"]),
        transformer_multiplier=float(placement_config["capacity"]),
    )
    availability = training.availability_probability(
        placed_records, config, implementation.mixed.AVAILABILITY_MODE
    )
    return placed_records, config, availability, network, scenario


def _algorithm_config(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["control"] = dict(value["control"])
    if algorithm == "eps_ieee69_fused":
        value["control"]["eps_control_mode"] = "fused"
    return value


def _run_dataset(dataset: str, seed_count: int, worker_label: str) -> Path:
    _base_settings()
    case = _configure_network()
    _, optimizer, _ = direct.training.load_frozen_eps_controller(_model_path(dataset))
    output_dir = OUTPUT / "data/raw" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = len(SCENARIOS) * len(ALGORITHMS)
    for seed_index in range(seed_count):
        output_path = output_dir / f"seed_{seed_index:02d}.csv"
        if output_path.is_file():
            existing = _read_rows(output_path)
            if len(existing) == expected_rows and all(row.get("protocol") == PROTOCOL for row in existing):
                continue
            output_path.unlink()
        rows: list[dict[str, Any]] = []
        base_seed = implementation._base_seed(dataset, seed_index)
        for scenario_index, scenario_name in enumerate(SCENARIOS):
            records, config, availability, network, scenario = _context(
                dataset, seed_index, scenario_name, case
            )
            for algorithm_index, algorithm in enumerate(ALGORITHMS):
                result, _ = implementation._run_seed(
                    algorithm,
                    records,
                    _algorithm_config(config, algorithm),
                    scenario,
                    availability,
                    network,
                    base_seed + 2_400_000 + scenario_index * 20_000 + algorithm_index,
                    base_seed + 2_500_000 + scenario_index,
                    optimizer if algorithm == "eps_ieee69_fused" else None,
                    False,
                )
                rows.append({
                    "protocol": PROTOCOL,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "seed_index": seed_index,
                    "seed": base_seed,
                    "scenario": scenario_name,
                    "scenario_label": SCENARIOS[scenario_name]["label"],
                    "topology": TOPOLOGY,
                    "network_source": "official_ieee123_opendss_single_phase_structural_equivalent",
                    "fleet_size": len(records),
                    "algorithm": algorithm,
                    "algorithm_label": EPS_LABEL if algorithm == "eps_ieee69_fused" else implementation.mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                    "complexity": ALGORITHM_COMPLEXITY[algorithm],
                    "configured_capacity_multiplier": SCENARIOS[scenario_name]["capacity"],
                    "placement_mode": scenario["placement"]["placement_mode"],
                    "configured_concentration_fraction": scenario["placement"]["configured_concentration_fraction"],
                    "actual_target_feeder_fraction": scenario["placement"]["actual_target_feeder_fraction"],
                    "actual_target_bus_fraction": scenario["placement"]["actual_target_bus_fraction"],
                    **result,
                })
        _write_rows(output_path, rows)
        print(json.dumps({"dataset": dataset, "completed_seeds": seed_index + 1, "seed_count": seed_count, "worker": worker_label}, ensure_ascii=False), flush=True)
    return output_dir


def _merge_raw(seed_count: int) -> list[dict[str, str]]:
    paths = sorted((OUTPUT / "data/raw").glob("*/seed_*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_rows(path))
    expected = len(DATASETS) * seed_count * len(SCENARIOS) * len(ALGORITHMS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete per-seed rows: {len(rows)}, expected {expected}")
    _write_rows(OUTPUT / "data/by_seed.csv", rows)
    return rows


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _aggregate(rows: list[dict[str, str]], seed_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], int(row["seed_index"]), row["scenario"]), {})[row["algorithm"]] = row
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        selected = [(key, values) for key, values in groups.items() if key[2] == scenario]
        for algorithm in ALGORITHMS:
            values = np.asarray([float(value[algorithm]["mean_reduction_pct"]) for _, value in selected])
            effect, low, high = _bootstrap(values, _stable_seed(f"effect|{scenario}|{algorithm}"))
            violation = np.asarray([float(value[algorithm]["requested_added_violation_steps"]) / STEPS for _, value in selected])
            vmean, vlow, vhigh = _bootstrap(violation, _stable_seed(f"violation|{scenario}|{algorithm}"))
            acceptance = np.asarray([float(value[algorithm]["network_acceptance_ratio"]) for _, value in selected if algorithm != "no_coordination"])
            amean, alow, ahigh = _bootstrap(acceptance, _stable_seed(f"acceptance|{scenario}|{algorithm}"))
            first = selected[0][1][algorithm]
            summary.append({
                "dataset_scope": "three_representative_datasets",
                "scenario": scenario,
                "scenario_label": SCENARIOS[scenario]["label"],
                "algorithm": algorithm,
                "algorithm_label": first["algorithm_label"],
                "complexity": first["complexity"],
                "effect_mean_pct": effect,
                "effect_ci_low_pct": low,
                "effect_ci_high_pct": high,
                "network_acceptance_mean_pct": "NA" if not np.isfinite(amean) else 100.0 * amean,
                "network_acceptance_ci_low_pct": "NA" if not np.isfinite(alow) else 100.0 * alow,
                "network_acceptance_ci_high_pct": "NA" if not np.isfinite(ahigh) else 100.0 * ahigh,
                "requested_added_violation_mean_pct": 100.0 * vmean,
                "requested_added_violation_ci_low_pct": 100.0 * vlow,
                "requested_added_violation_ci_high_pct": 100.0 * vhigh,
                "paired_observations": len(selected),
                "seed_count": seed_count,
            })
        for key, value in selected:
            eps = float(value["eps_ieee69_fused"]["mean_reduction_pct"])
            best_name = max(BASELINES, key=lambda name: float(value[name]["mean_reduction_pct"]))
            paired.append({
                "dataset": key[0],
                "seed_index": key[1],
                "scenario": scenario,
                "eps_effect_pct": eps,
                "best_baseline": best_name,
                "best_baseline_effect_pct": float(value[best_name]["mean_reduction_pct"]),
                "eps_minus_best_baseline_pp": eps - float(value[best_name]["mean_reduction_pct"]),
                "eps_requested_added_violation_pct": 100.0 * float(value["eps_ieee69_fused"]["requested_added_violation_steps"]) / STEPS,
            })
    _write_rows(OUTPUT / "data/summary.csv", summary)
    _write_rows(OUTPUT / "data/paired.csv", paired)
    return summary, paired


def _save_figure(figure: Any, name: str) -> None:
    path = OUTPUT / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_effect(summary: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, len(SCENARIOS), figsize=(18, 5.8), sharey=True)
    labels = {row["algorithm"]: row["algorithm_label"] for row in summary}
    for axis, scenario in zip(axes, SCENARIOS):
        selected = [row for row in summary if row["scenario"] == scenario]
        values = {row["algorithm"]: float(row["effect_mean_pct"]) for row in selected}
        bars = axis.bar(np.arange(len(ALGORITHMS)), [values[a] for a in ALGORITHMS], color=[COLORS[a] for a in ALGORITHMS])
        axis.set_title(SCENARIOS[scenario]["label"])
        axis.set_xticks(np.arange(len(ALGORITHMS)), [f"{labels[a]}\n{ALGORITHM_COMPLEXITY[a]}" for a in ALGORITHMS], rotation=55, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.22)
        for bar, algorithm in zip(bars, ALGORITHMS):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{values[algorithm]:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    axes[0].set_ylabel("Deliverable curtailment reduction (%)")
    figure.tight_layout()
    _save_figure(figure, "effect_comparison")


def _plot_safety(summary: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, len(SCENARIOS), figsize=(18, 5.8), sharey=True)
    for axis, scenario in zip(axes, SCENARIOS):
        selected = [row for row in summary if row["scenario"] == scenario]
        x = np.arange(len(ALGORITHMS))
        width = 0.38
        violation = [float(row["requested_added_violation_mean_pct"]) for row in selected]
        acceptance = [
            np.nan if row["network_acceptance_mean_pct"] == "NA"
            else float(row["network_acceptance_mean_pct"])
            for row in selected
        ]
        axis.bar(x - width / 2, violation, width, label="Added violating requests (%)", color="#d1495b")
        axis.bar(x + width / 2, acceptance, width, label="Network acceptance (%)", color="#4e79a7")
        axis.axhline(5, color="#d1495b", linestyle="--", linewidth=1, label="5% risk line")
        axis.set_title(SCENARIOS[scenario]["label"])
        axis.set_xticks(x, [EPS_LABEL if a == "eps_ieee69_fused" else implementation.mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHMS], rotation=55, ha="right", fontsize=7)
        axis.set_ylim(0, 110)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Share (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    figure.tight_layout()
    _save_figure(figure, "safety_audit")


def _write_manifest(seed_count: int) -> None:
    (OUTPUT / "manifest.json").write_text(json.dumps({"protocol": PROTOCOL, "network": "IEEE-123", "network_representation": "single_phase_structural_equivalent", "datasets": list(DATASETS), "seed_count": seed_count, "scenario_count": len(SCENARIOS), "algorithm_count": len(ALGORITHMS), "model_regime": MODEL_REGIME, "migration_model": MODEL_REGIME != "IEEE-123 target-domain direct-trained model", "direct_ieee123_training": MODEL_REGIME == "IEEE-123 target-domain direct-trained model", "multiphase_validation_entrypoint": "src/extra/ieee33_device_day_simulation/network_experiments/ieee123_opendss_check.py"}, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    global MODEL_SOURCE, MODEL_REGIME, EPS_LABEL
    global BOOTSTRAP_DRAWS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--model-source", type=Path, default=None, help="EPS model directory; the main experiment uses the IEEE-123 direct-training model directory")
    args = parser.parse_args()
    BOOTSTRAP_DRAWS = max(100, args.bootstrap_draws)
    if args.model_source is not None:
        MODEL_SOURCE = args.model_source.resolve()
        MODEL_REGIME = "IEEE-123 target-domain direct-trained model"
        EPS_LABEL = "EPS IEEE-123 direct fused"
    if args.fresh and OUTPUT.exists(): shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _base_settings()
    _configure_network()
    completed: list[Path] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(_run_dataset, dataset, args.seed_count, f"{i + 1}/{len(DATASETS)}"): dataset for i, dataset in enumerate(DATASETS)}
        for future in as_completed(jobs):
            completed.append(future.result())
            print(json.dumps({"stage": "dataset_complete", "dataset": jobs[future], "completed": len(completed), "total": len(jobs)}, ensure_ascii=False), flush=True)
    rows = _merge_raw(args.seed_count)
    summary, paired = _aggregate(rows, args.seed_count)
    _plot_effect(summary)
    _plot_safety(summary)
    _write_manifest(args.seed_count)
    shutil.rmtree(OUTPUT / "data/raw", ignore_errors=True)
    print(json.dumps({"stage": "complete", "rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired), "figures": 2}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
