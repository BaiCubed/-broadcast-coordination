from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PAIRWISE_SOURCE = Path("results/E21/pairwise_curtailment")
MIXED_SOURCE = Path("results/E21/data/e21_results.json")
DEFAULT_OUTPUT = Path("data/dataset_combination_summaries")

DATASET_LABELS = {
    "bdg1_building_data_genome": "BDG1",
    "bdg2_building_data_genome": "BDG2",
    "low_carbon_london": "LCL",
    "camsl_japan_smart_meters": "CAMSL",
    "irish_domestic_smart_meters": "Irish",
    "goiener_smart_meters": "GoiEner",
    "smart_grid_smart_city": "SGSC",
    "danish_smart_heat_meters": "Danish",
    "heapo_heat_pumps": "HEAPO",
    "european_lv_rural_2731": "EU-Rural",
    "european_lv_urban_35297": "EU-35297",
    "european_lv_urban_8087": "EU-8087",
    "norway_ami_energy_distribution": "Norway",
    "complete_energy_community": "CEC",
    "opsd_household_data": "OPSD",
}

SCENARIO_PURPOSES = {
    "S1-A": "similar residential half-hourly meters",
    "S1-B": "high-contrast mix of buildings, heat pumps and households with controlled load",
    "S2-A": "similar large-scale distribution and AMI sources",
    "S2-B": "buildings, households, heat demand, heat pumps and an integrated community",
    "S3-A": "ten electrical datasets with a small heat-pump perturbation",
    "S3-B": "ten datasets across several sectors",
    "S4-A": "equal dataset weights, an extreme heterogeneity stress test",
    "S4-B": "equal weights per resource type, the main full mixture",
    "S5-A": "residential-dominated long tail",
    "S5-B": "building and thermal dominated long tail",
    "S5-C": "network and DER dominated long tail",
    "S6-A": "similar network loads grouped upstream, midstream and at the feeder end",
    "S6-B": "buildings, households and thermal loads layered into congestion",
    "S6-C": "integrated DER and bidirectional subgroups on separate branches",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [
        key for key in preferred if any(key in row for row in rows)
    ] + remaining
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _json_text(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _export_pairwise(source: Path, output: Path) -> list[dict[str, Any]]:
    seed_rows = _read_csv(source / "data/pairwise_curtailment_by_seed.csv")
    summary_rows = {
        row["pair_id"]: row
        for row in _read_csv(source / "data/pairwise_curtailment_summary.csv")
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(row["pair_id"], []).append(row)

    mappings: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        summary = summary_rows[pair_id]
        dataset_a = summary["dataset_a"]
        dataset_b = summary["dataset_b"]
        exported_rows = []
        for row in sorted(grouped[pair_id], key=lambda item: int(item["seed_index"])):
            exported_rows.append(
                {
                    "combination_type": "pairwise_equal_mix",
                    "pair_id": pair_id,
                    "pair_label": summary["pair_label"],
                    "dataset_a": dataset_a,
                    "dataset_a_label": DATASET_LABELS[dataset_a],
                    "weight_a": 0.5,
                    "dataset_b": dataset_b,
                    "dataset_b_label": DATASET_LABELS[dataset_b],
                    "weight_b": 0.5,
                    "network_mode": "aggregate",
                    "availability_mode": "data_driven",
                    "protocol": "dataset_combinations_pairwise_summary_v1",
                    **row,
                }
            )
        file_name = f"{pair_id}.csv"
        _write_csv(
            output / file_name,
            exported_rows,
            [
                "combination_type",
                "pair_id",
                "pair_label",
                "dataset_a",
                "dataset_a_label",
                "weight_a",
                "devices_a",
                "dataset_b",
                "dataset_b_label",
                "weight_b",
                "devices_b",
                "fleet_size",
                "seed",
                "seed_index",
                "network_mode",
                "availability_mode",
                "baseline_curtailment_mwh",
                "remaining_curtailment_mwh",
                "accepted_absorption_mwh",
                "curtailment_reduction_pct",
                "availability_fraction",
                "mean_network_scale",
                "network_violation_steps",
                "max_soc_violation",
                "profile_bootstrap_a",
                "mean_profile_reuse_a",
                "profile_bootstrap_b",
                "mean_profile_reuse_b",
                "protocol",
            ],
        )
        mappings.append(
            {
                "file": f"pairwise/{file_name}",
                "pair_id": pair_id,
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "label_a": DATASET_LABELS[dataset_a],
                "label_b": DATASET_LABELS[dataset_b],
                "row_count": len(exported_rows),
            }
        )
    return mappings


def _export_mixed(source: Path, output: Path) -> list[dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    definitions = payload["scenario_definitions"]
    algorithm_definitions = payload["algorithm_definitions"]
    grouped: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in definitions
    }

    for condition in payload["conditions"]:
        scenario = condition["scenario"]
        audits = condition["fleet_audit"]
        for algorithm, result in condition["results"].items():
            algorithm_definition = algorithm_definitions[algorithm]
            for seed_index, seed_result in enumerate(result["seed_results"]):
                audit = audits[seed_index]
                if seed_result["composition_regime"] != audit["composition_regime"]:
                    raise ValueError(
                        f"{scenario}/{condition['network_mode']}/{algorithm} "
                        f": composition state {seed_index} cannot be aligned with the fleet audit"
                    )
                seed_metrics = {
                    key: value
                    for key, value in seed_result.items()
                    if key not in {"seed", "composition_regime"}
                }
                grouped[scenario].append(
                    {
                        "combination_type": "predefined_mixed_scenario",
                        "scenario": scenario,
                        "scenario_label": condition["scenario_label"],
                        "scenario_purpose": SCENARIO_PURPOSES[scenario],
                        "network_mode": condition["network_mode"],
                        "algorithm": algorithm,
                        "algorithm_label": algorithm_definition["label"],
                        "complexity": algorithm_definition["complexity"],
                        "seed_index": seed_index,
                        "fleet_seed": int(audit["seed"]),
                        "algorithm_seed": int(seed_result["seed"]),
                        "composition_regime": seed_result["composition_regime"],
                        "fleet_size": audit["fleet_size"],
                        "nominal_weights_json": condition["nominal_weights"],
                        "actual_weights_json": audit["weights"],
                        "device_counts_json": audit["counts"],
                        "dataset_audit_json": audit["datasets"],
                        "spatial_clustering": audit["spatial_clustering"],
                        "partition": audit["partition"],
                        "protocol": "dataset_combinations_mixed_summary_v1",
                        **seed_metrics,
                    }
                )

    mappings: list[dict[str, Any]] = []
    for scenario in definitions:
        rows = sorted(
            grouped[scenario],
            key=lambda item: (
                item["network_mode"],
                item["algorithm"],
                int(item["seed_index"]),
            ),
        )
        file_name = f"{scenario}.csv"
        _write_csv(
            output / file_name,
            rows,
            [
                "combination_type",
                "scenario",
                "scenario_label",
                "scenario_purpose",
                "network_mode",
                "algorithm",
                "algorithm_label",
                "complexity",
                "seed_index",
                "fleet_seed",
                "algorithm_seed",
                "composition_regime",
                "fleet_size",
                "nominal_weights_json",
                "actual_weights_json",
                "device_counts_json",
                "spatial_clustering",
                "partition",
                "baseline_curtailment_mwh",
                "remaining_curtailment_mwh",
                "accepted_absorption_mwh",
                "mean_reduction_pct",
                "availability_fraction",
                "mean_network_scale",
                "network_violation_steps",
                "max_soc_violation",
                "total_charging_mwh",
                "excess_grid_charging_mwh",
                "total_discharge_mwh",
                "dataset_audit_json",
                "protocol",
            ],
        )
        mappings.append(
            {
                "file": f"mixed_scenarios/{file_name}",
                "scenario": scenario,
                "label": definitions[scenario]["label"],
                "purpose": SCENARIO_PURPOSES[scenario],
                "weights": definitions[scenario]["weights"],
                "row_count": len(rows),
            }
        )
    return mappings


def _validate(
    pairwise_mappings: list[dict[str, Any]],
    mixed_mappings: list[dict[str, Any]],
) -> None:
    if len(pairwise_mappings) != 105:
        raise ValueError(f"wrong number of pairwise combinations: {len(pairwise_mappings)}, expected 105")
    if any(row["row_count"] != 30 for row in pairwise_mappings):
        raise ValueError("every pairwise file must hold 30 seeds")
    if len(mixed_mappings) != 14:
        raise ValueError(f"wrong number of mixtures: {len(mixed_mappings)}, expected 14")
    if any(row["row_count"] != 600 for row in mixed_mappings):
        raise ValueError("every mixture file must hold 600 rows")
    for row in mixed_mappings:
        if abs(sum(row["weights"].values()) - 1.0) > 1e-9:
            raise ValueError(f"{row['scenario']}: the nominal weights do not sum to 1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the dataset combination experiments as one CSV per combination.")
    parser.add_argument("--pairwise-source", type=Path, default=PAIRWISE_SOURCE)
    parser.add_argument("--mixed-source", type=Path, default=MIXED_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pairwise_output = args.output / "pairwise"
    mixed_output = args.output / "mixed_scenarios"
    pairwise_output.mkdir(parents=True, exist_ok=True)
    mixed_output.mkdir(parents=True, exist_ok=True)

    pairwise_mappings = _export_pairwise(args.pairwise_source, pairwise_output)
    mixed_mappings = _export_mixed(args.mixed_source, mixed_output)
    _validate(pairwise_mappings, mixed_mappings)


    generated_files = sorted(args.output.rglob("*.csv"))
    manifest = {
        "protocol": "dataset_combinations_summary_export_v1",
        "pairwise_combination_count": len(pairwise_mappings),
        "mixed_scenario_count": len(mixed_mappings),
        "pairwise_seed_row_count": sum(row["row_count"] for row in pairwise_mappings),
        "mixed_result_row_count": sum(row["row_count"] for row in mixed_mappings),
        "sources": [
            {
                "path": str(
                    args.pairwise_source / "data/pairwise_curtailment_by_seed.csv"
                ),
                "sha256": _sha256(
                    args.pairwise_source / "data/pairwise_curtailment_by_seed.csv"
                ),
            },
            {
                "path": str(args.mixed_source),
                "sha256": _sha256(args.mixed_source),
            },
        ],
        "generated_files": [
            {
                "path": str(path.relative_to(args.output)),
                "sha256": _sha256(path),
            }
            for path in generated_files
        ],
    }
    _write_json(args.output / "manifest.json", manifest)

    print(f"exported {len(pairwise_mappings)} pairwise CSV files")
    print(f"exported {len(mixed_mappings)} mixture CSV files")
    print(f"output directory: {args.output}")


if __name__ == "__main__":
    main()
