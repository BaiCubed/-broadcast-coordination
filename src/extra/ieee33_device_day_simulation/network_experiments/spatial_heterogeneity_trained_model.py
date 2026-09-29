from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from src.estimation import EPSEstimator, EstimatorConfig
from src.signal import SignalOptimizer

from ..original_adapter.eps_adapter import OriginalEPSAdapter
from . import reference_controllers as controllers
from . import network_dispatch_protocol as protocol
from . import ieee69_network_implementation as implementation
from . import spatial_heterogeneity as spatial
from . import ieee69_trained_vs_transferred as direct
from . import train_pooled_model as transfer


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/network_stress_boundary/direct_training"
PUBLISH_OUTPUT = ROOT / "outputs/figs/direct_train"
SOURCE = ROOT / "results/network_stress_boundary"
PROTOCOL_NAME = "spatial_heterogeneity_trained_model_v1"
TRAINING_TOPOLOGY = "ieee69"
FLEET_SIZE = spatial.FLEET_SIZE
TRAIN_FLEET_SEEDS = tuple(range(0, 24))
VALIDATION_FLEET_SEEDS = tuple(range(24, 30))
TRAIN_SAMPLES_PER_FLEET = 500
VALIDATION_SAMPLES_PER_FLEET = 200
RESPONSE_SCALE_KW = 5000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no data to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _network_context() -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    case = spatial._network_cases()[TRAINING_TOPOLOGY]
    base_weights = spatial._region_weights(case)
    headroom = spatial._headroom_by_region(
        case, implementation._network(case, TRAINING_TOPOLOGY, "M0")
    )
    global_counts = spatial._allocate_counts(
        np.ones(len(spatial.DATASETS)), FLEET_SIZE
    )
    return case, base_weights, headroom, {
        "global_counts": global_counts,
    }


def _mapped_fleet(
    fleet_index: int,
    condition: str,
    case: dict[str, Any],
    base_weights: np.ndarray,
    headroom: np.ndarray,
    global_counts: np.ndarray,
) -> tuple[list[Any], dict[str, Any], np.ndarray, np.ndarray]:
    records, config, _ = spatial._sample_global_fleet(900000 + fleet_index)
    matrix, region_counts = spatial._condition_matrix(
        condition, global_counts, base_weights, headroom
    )
    mapped = spatial._assign_records(
        records,
        matrix,
        region_counts,
        case,
        900000 + fleet_index + spatial._stable_seed(condition),
    )
    return mapped, config, matrix, region_counts


def _response_samples(
    records: list[Any],
    config: dict[str, Any],
    sample_count: int,
    seed: int,
    case: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float]]:
    scenario = implementation._build_scenario(
        records,
        records,
        case,
        TRAINING_TOPOLOGY,
        "M0",
        seed,
        {"placement_mode": "spatial_condition_direct_training"},
    )
    network = implementation._network(case, TRAINING_TOPOLOGY, "M0")
    availability = protocol.availability_probability(
        records, config, spatial.mixed.AVAILABILITY_MODE
    )
    adapter = OriginalEPSAdapter(records, config, seed + 1)
    rng = np.random.default_rng(seed + 2)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    zone_count = len(config["zones"]["zones"])
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    for index in range(sample_count):
        supply_demand, intensity, hour = protocol._signal_parameters(
            index, sample_count, rng
        )
        profile_step = int(hour * 12 + rng.integers(0, 12))
        adapter.reset_to_initial_state()
        energy_kwh = 0.0
        for offset in range(3):
            step = (profile_step + offset) % controllers.STEPS
            generated = [
                adapter.simulator._signal_generator.generate_signal(
                    zone, supply_demand, intensity, priority=10
                )
                for zone in range(zone_count)
            ]
            batch = adapter.step(generated, step, availability[step])
            load_map, input_map = implementation._maps(scenario, step)
            dispatch = network.dispatch(
                load_map,
                input_map,
                np.asarray(batch.desired_kw, dtype=float),
                scenario["device_buses"],
            )
            actual = adapter.apply_dispatch(dispatch.accepted_kw)
            energy_kwh += float(np.sum(actual)) * dt
        signals.append(
            {
                "supply_demand": supply_demand,
                "intensity": intensity,
                "price": 0.0,
                "hour": hour,
                "day_of_week": 0,
                "direction": 1 if supply_demand <= 7 else -1,
            }
        )
        responses.append(
            energy_kwh / max(3.0 * dt, 1e-12) / RESPONSE_SCALE_KW
        )
    return signals, responses


def _train_condition(condition: str, output: str, force: bool) -> dict[str, Any]:
    output_root = Path(output)
    model_path = output_root / "models" / f"{condition}.pt"
    metadata_path = output_root / "models" / f"{condition}.json"
    if model_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol") == PROTOCOL_NAME:
            return metadata

    case, base_weights, headroom, context = _network_context()
    train_signals: list[dict[str, Any]] = []
    train_responses: list[float] = []
    validation_signals: list[dict[str, Any]] = []
    validation_responses: list[float] = []
    for fleet_index in TRAIN_FLEET_SEEDS:
        records, config, _, _ = _mapped_fleet(
            fleet_index,
            condition,
            case,
            base_weights,
            headroom,
            context["global_counts"],
        )
        signals, responses = _response_samples(
            records,
            config,
            TRAIN_SAMPLES_PER_FLEET,
            700000 + fleet_index + implementation._stable_seed(condition),
            case,
        )
        train_signals.extend(signals)
        train_responses.extend(responses)
        print(
            json.dumps(
                {
                    "stage": "direct_training_samples",
                    "condition": condition,
                    "fleet": fleet_index + 1,
                    "fleet_count": len(TRAIN_FLEET_SEEDS),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for fleet_index in VALIDATION_FLEET_SEEDS:
        records, config, _, _ = _mapped_fleet(
            fleet_index,
            condition,
            case,
            base_weights,
            headroom,
            context["global_counts"],
        )
        signals, responses = _response_samples(
            records,
            config,
            VALIDATION_SAMPLES_PER_FLEET,
            800000 + fleet_index + implementation._stable_seed(condition),
            case,
        )
        validation_signals.extend(signals)
        validation_responses.extend(responses)

    try:
        import torch

        torch.manual_seed(3100000 + implementation._stable_seed(condition) % 100000)
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = EPSEstimator(
        EstimatorConfig(
            target_coverage=0.9,
            enable_conformal=True,
            use_cqr=True,
            use_pytorch=True,
            pytorch_epochs=controllers.EPS_TRAINING_EPOCHS,
            pytorch_batch_size=64,
            pytorch_learning_rate=0.001,
        )
    )
    estimator.fit(train_signals, train_responses)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    controllers._save_eps_model(estimator, model_path)
    direct._ensure_bidirectional_artifact(model_path)
    predicted = np.asarray(
        [estimator.estimate(signal).response_kw for signal in validation_signals],
        dtype=float,
    )
    actual = np.asarray(validation_responses, dtype=float)
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    parameters = estimator._learned_params or {}
    metadata = {
        "protocol": PROTOCOL_NAME,
        "condition": condition,
        "condition_label": spatial.CONDITIONS[condition]["label"],
        "training_topology": TRAINING_TOPOLOGY,
        "training_scope": "IEEE-69 spatial condition direct training",
        "training_fleet_seeds": list(TRAIN_FLEET_SEEDS),
        "validation_fleet_seeds": list(VALIDATION_FLEET_SEEDS),
        "training_samples": len(train_signals),
        "validation_samples": len(validation_signals),
        "epochs": controllers.EPS_TRAINING_EPOCHS,
        "model_type": str(parameters.get("model_type", "unknown")),
        "internal_training_r2": float(parameters.get("r2", float("nan"))),
        "held_out_r2": float(
            1.0 - np.sum(residual**2) / max(denominator, 1e-12)
        ),
        "held_out_rmse_normalized": float(np.sqrt(np.mean(residual**2))),
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_reused": False,
        "input_protocol": "one broadcast signal and aggregate scalar feedback",
        "communication_complexity": "O(1) with respect to N",
    }
    _write_json(metadata_path, metadata)
    return metadata


def _train_job(condition: str, output: str, force: bool) -> dict[str, Any]:
    spatial.CACHE_ROOT = SOURCE / "data/.fleet_cache"
    return _train_condition(condition, output, force)


def _evaluate_direct(
    condition: str,
    seed_index: int,
    model_path: str,
) -> dict[str, Any]:
    spatial.CACHE_ROOT = SOURCE / "data/.fleet_cache"
    case, base_weights, headroom, context = _network_context()
    records, config, matrix, region_counts = _mapped_fleet(
        seed_index,
        condition,
        case,
        base_weights,
        headroom,
        context["global_counts"],
    )
    spatial.RUN_ALGORITHMS = ("eps_global_mixed",)
    rows, metric, regions = spatial._run_condition(
        records,
        config,
        condition,
        matrix,
        region_counts,
        case,
        base_weights,
        headroom,
        SignalOptimizer(
            transfer.ScaledEstimator(
                protocol.load_frozen_eps_controller(Path(model_path))[0],
                RESPONSE_SCALE_KW,
                0.0,
            )
        ),
        900000 + seed_index,
    )
    return {"rows": rows, "metric": metric, "regions": regions}


def _publish_figures(figures: list[str]) -> list[str]:
    PUBLISH_OUTPUT.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for figure in figures:
        source = Path(figure)
        for candidate in (source, source.with_suffix(".pdf")):
            if not candidate.is_file():
                continue
            target = PUBLISH_OUTPUT / candidate.name
            shutil.copy2(candidate, target)
            published.append(str(target))
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the per-spatial-condition direct-training EPS comparison.")
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed_count != 30:
        raise ValueError("this version uses exactly 30 seeds")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    output = OUTPUT
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    spatial.CACHE_ROOT = SOURCE / "data/.fleet_cache"
    conditions = tuple(spatial.CONDITIONS)
    metadata_by_condition: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(conditions))) as executor:
        futures = {
            executor.submit(_train_job, condition, str(output), args.force): condition
            for condition in conditions
        }
        for future in as_completed(futures):
            condition = futures[future]
            metadata_by_condition[condition] = future.result()
            _write_json(
                output / "data/training_metadata.json",
                [metadata_by_condition[key] for key in conditions if key in metadata_by_condition],
            )
            print(
                json.dumps(
                    {
                        "stage": "model_complete",
                        "condition": condition,
                        "completed": len(metadata_by_condition),
                        "total": len(conditions),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    original_rows = _read_rows(SOURCE / "data/spatial_heterogeneity_by_seed.csv")
    original_metrics = _read_rows(SOURCE / "data/imbalance_metrics.csv")
    original_regions = _read_rows(SOURCE / "data/region_assignments.csv")
    baseline_rows = [row for row in original_rows if row["algorithm"] != "eps_global_mixed"]
    direct_rows: list[dict[str, Any]] = []
    spatial.RUN_ALGORITHMS = ("eps_global_mixed",)
    for seed_index in range(args.seed_count):
        for condition in conditions:
            result = _evaluate_direct(
                condition,
                seed_index,
                str(output / "models" / f"{condition}.pt"),
            )
            direct_rows.extend(result["rows"])
            print(
                json.dumps(
                    {
                        "stage": "evaluation",
                        "seed_index": seed_index,
                        "condition": condition,
                        "rows": len(direct_rows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    rows = baseline_rows + direct_rows
    rows.sort(key=lambda row: (int(row["seed_index"]), conditions.index(row["condition"]), row["algorithm"]))
    _write_rows(output / "data/spatial_heterogeneity_by_seed.csv", rows)
    _write_rows(output / "data/imbalance_metrics.csv", original_metrics)
    _write_rows(output / "data/region_assignments.csv", original_regions)
    spatial.OUTPUT = output
    spatial.RUN_ALGORITHMS = spatial.ALGORITHMS
    figures = spatial._plot_outputs(rows, original_metrics, original_regions)
    published_figures = _publish_figures(figures)
    summary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for algorithm in spatial.ALGORITHMS:
            selected = [
                row for row in rows
                if row["condition"] == condition and row["algorithm"] == algorithm
            ]
            if not selected:
                continue
            values = np.asarray([float(row["mean_reduction_pct"]) for row in selected])
            summary_rows.append(
                {
                    "condition": condition,
                    "condition_label": spatial.CONDITIONS[condition]["label"],
                    "algorithm": algorithm,
                    "algorithm_label": spatial.ALGORITHM_LABELS[algorithm],
                    "complexity": spatial.ALGORITHM_COMPLEXITY[algorithm],
                    "seed_count": len(selected),
                    "mean_reduction_pct": float(np.mean(values)),
                    "std_reduction_pct": float(np.std(values, ddof=1)),
                    "network_acceptance_mean_pct": float(np.mean([float(row["network_acceptance_ratio"]) for row in selected]) * 100.0),
                }
            )
    _write_rows(output / "data/spatial_heterogeneity_summary.csv", summary_rows)
    _write_json(
        output / "manifest.json",
        {
            "protocol": PROTOCOL_NAME,
            "status": "completed",
            "topology": TRAINING_TOPOLOGY,
            "dataset_count": len(spatial.DATASETS),
            "fleet_size": FLEET_SIZE,
            "seed_count": args.seed_count,
            "conditions": list(conditions),
            "direct_training_models": [str(output / "models" / f"{condition}.pt") for condition in conditions],
            "baseline_source": str(SOURCE / "data/spatial_heterogeneity_by_seed.csv"),
            "figures": figures,
            "published_figures": published_figures,
        },
    )
    print(json.dumps({"status": "complete", "output": str(output), "figures": figures}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
